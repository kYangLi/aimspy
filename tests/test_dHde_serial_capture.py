#!/usr/bin/env python
"""Integration test: DFPT dH/de capture in SERIAL mode (self-contained).

Serial mode (``electric_field_serial .true.``) runs three separate CPSCF
cycles — one per Cartesian direction (j_coord = 1/2/3 = x/y/z) — instead
of a single full-memory CPSCF over all three directions at once.

This test is SELF-CONTAINED: it first runs a full-memory capture (the
reference) in ``MoS2_DFPT/`` then a serial capture in ``MoS2_DFPT_serial/``,
and cross-validates that the two physically-equivalent dH/de tensors agree.

Verifies:
  1. Serial capture produces a complete [x, y, z] first_order_hamiltonian
     (three non-None AimspyMatrix, each with n_pairs == H n_pairs).
  2. Serial dH/de == full-memory dH/de per direction per block, to within
     the CPSCF convergence noise.  The two modes run independent CPSCF
     cycles converging to ``dfpt_sc_accuracy_dm`` (~1e-3), so their dH/de
     agree only to ~1e-5 relative (not machine precision); we assert a
     global relative difference < 1e-4 (normalized by the global dH/de
     magnitude).  A genuine bug (wrong direction mapping / buffer layout)
     would produce O(1) relative errors, far above this threshold.
  3. Serial CPSCF converges for all three directions.
  4. Serial export to electric_response.h5 round-trips correctly (this
     round-trip IS exact, atol=1e-10, since it is pure I/O).

The serial deeph product is saved to
``MoS2_DFPT_serial/deeph_dHde_serial_out/`` for use by
``test_dHde_serial_inject.py``.

Usage:
    source /path/to/intel/setvars.sh
    ulimit -s unlimited
    export AIMSPY_TEST_AIMS_LIBPATH=/path/to/libaims.so
    mpiexec -np 8 python tests/test_dHde_serial_capture.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from mpi4py import MPI

from aimspy import Calculator, CalculatorConfig
from aimspy import DeepHData
from tests.mpi_utils import synchronized_python_exceptions

HERE = Path(__file__).resolve().parent
FULL_DIR = HERE / "data" / "MoS2_DFPT"  # full-memory reference
SERIAL_DIR = HERE / "data" / "MoS2_DFPT_serial"  # serial run
SERIAL_DEEPH_DIR = SERIAL_DIR / "deeph_dHde_serial_out"

FULL_LOG = "aims_dHde_serial_ref_full.out"
SERIAL_LOG = "aims_dHde_serial.out"

comm = MPI.COMM_WORLD
rank = comm.rank

_lib_env = os.environ.get("AIMSPY_TEST_AIMS_LIBPATH")
if not _lib_env:
    if rank == 0:
        print(
            "ERROR: AIMSPY_TEST_AIMS_LIBPATH environment variable not set.\n"
            "  Export the path to your patched libaims.so before running:\n"
            "    export AIMSPY_TEST_AIMS_LIBPATH=/path/to/libaims.so",
            file=sys.stderr,
        )
    comm.Abort(1)
LIB_PATH = Path(_lib_env)

for d, name in ((FULL_DIR, "full-memory"), (SERIAL_DIR, "serial")):
    if not (d / "control.in").is_file():
        if rank == 0:
            print(f"ERROR: {d / 'control.in'} not found ({name}).", file=sys.stderr)
        sys.exit(1)


def _info(msg):
    if rank == 0:
        print(msg, flush=True)


def _ok(name, condition, detail=""):
    tag = "OK " if condition else "FAIL"
    _info(f"  {tag}  {name}" + (f" — {detail}" if detail and not condition else ""))
    return condition


def _cpscf_converged(work_dir: Path, logfile: str) -> bool:
    p = work_dir / logfile
    if not p.is_file():
        return False
    with open(p, "r") as f:
        return "CP-self-consistency cycle converged" in f.read()


def _run_capture(work_dir: Path, logfile: str):
    """Run SCF+CPSCF with capture; return (fo_list, H_aimspy, structure)."""
    config = CalculatorConfig(
        lib_path=LIB_PATH,
        logfile=Path(logfile),
        log_level="WARNING",
        capture_first_order_hamiltonian=True,
    )
    calc = Calculator(config)
    result = (None, None, None)
    try:
        calc.do(comm=comm, work_dir=work_dir)
        with synchronized_python_exceptions(comm):
            if rank == 0:
                result = (
                    calc.first_order_hamiltonian,
                    calc.hamiltonian,
                    calc.structure,
                )
        return result
    finally:
        calc.close()
        comm.Barrier()


all_ok = True

# =============================================================================
# Step 1: Full-memory capture (reference)
# =============================================================================
_info("=" * 60)
_info("Step 1: Full-memory capture (reference) — MoS2_DFPT")
_info("=" * 60)

fo_full, H_full, struct_full = _run_capture(FULL_DIR, FULL_LOG)
with synchronized_python_exceptions(comm):
    if rank == 0:
        all_ok &= _ok(
            "full-memory capture returns list of 3",
            isinstance(fo_full, list) and len(fo_full) == 3,
            f"got {type(fo_full)}",
        )
        all_ok &= _ok(
            "full-memory CPSCF converged",
            _cpscf_converged(FULL_DIR, FULL_LOG),
        )

# =============================================================================
# Step 2: Serial capture
# =============================================================================
_info("")
_info("=" * 60)
_info("Step 2: Serial capture — MoS2_DFPT_serial")
_info("=" * 60)

fo_serial, H_serial, struct_serial = _run_capture(SERIAL_DIR, SERIAL_LOG)

with synchronized_python_exceptions(comm):
    if rank == 0:
        # -- 2a. completeness: full [x,y,z], no None entries --
        _info("-- serial capture completeness --")
        all_ok &= _ok(
            "serial first_order_hamiltonian is list of 3",
            isinstance(fo_serial, list) and len(fo_serial) == 3,
            f"got {type(fo_serial)} len={len(fo_serial) if isinstance(fo_serial, list) else 'n/a'}",
        )
        if isinstance(fo_serial, list) and len(fo_serial) == 3:
            all_ok &= _ok(
                "serial: no None direction (all 3 captured)",
                all(mx is not None for mx in fo_serial),
                f"None at {[i for i, mx in enumerate(fo_serial) if mx is None]}",
            )
            all_ok &= _ok(
                "serial: each mx n_pairs == H n_pairs",
                all(
                    mx is not None and mx.n_pairs == H_serial.n_pairs
                    for mx in fo_serial
                ),
            )
        all_ok &= _ok(
            "serial CPSCF converged", _cpscf_converged(SERIAL_DIR, SERIAL_LOG)
        )

        # -- 2b. serial vs full-memory numerical consistency --
        # Serial and full-memory run *independent* CPSCF cycles that converge to
        # the dfpt_sc_accuracy_dm threshold (~1e-3).  Their dH/de therefore agree
        # only up to the CPSCF convergence noise — NOT to machine precision.
        # Observed here: max|diff| ~3e-4 on a global dH/de magnitude of ~8
        # (Hartree), i.e. a global relative difference of ~4e-5.  We assert
        # rel < 1e-4 (~2.5x margin).  A genuine bug (wrong direction mapping /
        # buffer layout / off-by-one) would produce O(1) relative errors, far
        # above this threshold.
        _info("")
        _info("-- serial vs full-memory dH/de consistency (rel tol=1e-4) --")
        if (
            isinstance(fo_full, list)
            and len(fo_full) == 3
            and isinstance(fo_serial, list)
            and len(fo_serial) == 3
            and all(mx is not None for mx in fo_serial)
        ):
            # Normalize by the GLOBAL dH/de magnitude (max over all blocks and
            # directions), not per-block — per-block normalization is overly
            # sensitive to near-zero blocks.  Global ref ~2.2e2 for this system.
            global_ref = 0.0
            for cart in range(3):
                for blk in fo_full[cart].blocks.values():
                    global_ref = max(global_ref, float(np.max(np.abs(blk))))
            max_abs = 0.0
            dir_names = ["x", "y", "z"]
            for cart in range(3):
                full_blocks = fo_full[cart].blocks
                ser_blocks = fo_serial[cart].blocks
                if set(full_blocks.keys()) != set(ser_blocks.keys()):
                    all_ok &= _ok(
                        f"dir {dir_names[cart]}: block keys match",
                        False,
                        f"full {len(full_blocks)} vs serial {len(ser_blocks)} keys",
                    )
                    continue
                abs_max = 0.0
                for key in full_blocks:
                    d = np.max(np.abs(ser_blocks[key] - full_blocks[key]))
                    abs_max = max(abs_max, d)
                max_abs = max(max_abs, abs_max)
                _info(
                    f"    dir {dir_names[cart]}: max|diff|={abs_max:.3e} "
                    f"(rel to global max {abs_max / global_ref:.3e})"
                )
            rel = max_abs / global_ref if global_ref > 0 else max_abs
            # See the note above: rel < 1e-4 reflects CPSCF convergence noise;
            # a genuine bug would give O(1) relative errors.  Observed: ~4e-5.
            all_ok &= _ok(
                "serial == full-memory (global relative diff < 1e-4)",
                rel < 1e-4,
                f"max|diff|={max_abs:.2e}, global_ref={global_ref:.2e}, rel={rel:.2e}",
            )
        else:
            all_ok &= _ok(
                "serial vs full comparison runnable",
                False,
                "missing fo_full or fo_serial",
            )

        # -- 2c. export serial product to deeph_dHde_serial_out/ --
        _info("")
        _info("-- export serial product --")
        if (
            isinstance(fo_serial, list)
            and len(fo_serial) == 3
            and all(mx is not None for mx in fo_serial)
        ):
            dd = DeepHData.from_aimspy(
                struct_serial,
                hamiltonian=H_serial,
                first_order_hamiltonian=fo_serial,
            )
            _info(f"  DeepHData: {dd}")
            SERIAL_DEEPH_DIR.mkdir(parents=True, exist_ok=True)
            dd.save(SERIAL_DEEPH_DIR)
            all_ok &= _ok(
                "serial electric_response.h5 written",
                (SERIAL_DEEPH_DIR / "electric_response.h5").is_file(),
            )

            # round-trip via from_directory
            dd2 = DeepHData.from_directory(SERIAL_DEEPH_DIR)
            all_ok &= _ok(
                "serial from_directory reads fo_entries",
                dd2.first_order_hamiltonian_entries is not None,
            )
            if dd2.first_order_hamiltonian_entries is not None:
                dir_diff = np.max(
                    np.abs(
                        dd2.first_order_hamiltonian_entries
                        - dd.first_order_hamiltonian_entries
                    )
                )
                all_ok &= _ok(
                    "serial from_directory fo_entries match",
                    dir_diff < 1e-10,
                    f"max|diff|={dir_diff:.2e}",
                )

# =============================================================================
# Summary
# =============================================================================
if rank == 0:
    _info("")
    _info("=" * 60)
    _info(
        "DHDE SERIAL CAPTURE TEST PASSED"
        if all_ok
        else "DHDE SERIAL CAPTURE TEST FAILED"
    )
    _info("=" * 60)

all_ok = comm.bcast(all_ok if rank == 0 else None, root=0)
if not all_ok:
    raise AssertionError("DFPT serial/full-memory consistency test failed")
