#!/usr/bin/env python
"""Integration test: DFPT dH/de capture + export.

Runs SCF + CPSCF with capture_first_order_hamiltonian=True, captures
calc.first_order_hamiltonian = [mx_x, mx_y, mx_z], and exports to
deeph_dHde_out/electric_response.h5 via DeepHData.from_aimspy.

Also cross-validates: dHde roundtrip + HDF5 format + unit conversion.

Prerequisites:
    tests/data/MoS2_DFFT/control.in with electric_field_response DFPT

Usage:
    source /path/to/intel/setvars.sh
    ulimit -s unlimited
    export AIMSPY_TEST_AIMS_LIBPATH=/path/to/libaims.so
    mpiexec -np 8 python tests/test_dHde_capture.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
from mpi4py import MPI

from aimspy import Calculator, CalculatorConfig
from aimspy import DeepHData

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data" / "MoS2_DFFT"
DEEPH_DIR = DATA_DIR / "deeph_dHde_out"

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

if not (DATA_DIR / "control.in").is_file():
    if rank == 0:
        print(f"ERROR: {DATA_DIR / 'control.in'} not found.", file=sys.stderr)
    sys.exit(1)


def _info(msg):
    if rank == 0:
        print(msg)


def _ok(name, condition, detail=""):
    tag = "OK " if condition else "FAIL"
    _info(f"  {tag}  {name}" + (f" — {detail}" if detail and not condition else ""))
    return condition


# =============================================================================
# Step 1: Forward SCF + CPSCF (capture dHde)
# =============================================================================
_info("=" * 60)
_info("Step 1: Forward SCF + CPSCF (capture_first_order_hamiltonian=True)")
_info("=" * 60)

config = CalculatorConfig(
    lib_path=LIB_PATH,
    logfile=Path("aims_dHde_capture.out"),
    log_level="INFO",
    capture_first_order_hamiltonian=True,
)
calc = Calculator(config)
all_ok = True

try:
    calc.do(comm=comm, work_dir=DATA_DIR)

    if rank == 0:
        fo_list = calc.first_order_hamiltonian
        H_aimspy = calc.hamiltonian
        structure = calc.structure
        csr = calc.csr_descr

        if fo_list is None:
            _info("FAIL: calc.first_order_hamiltonian is None")
            all_ok = False
        else:
            _info(f"  first_order_hamiltonian: list of {len(fo_list)} AimspyMatrix")
            for i, mx in enumerate(fo_list):
                _info(f"    [{i}]: {mx}")
            all_ok &= _ok(
                "fo_list length == 3",
                len(fo_list) == 3,
                f"got {len(fo_list)}",
            )
            all_ok &= _ok(
                "each mx n_pairs > 0",
                all(mx.n_pairs > 0 for mx in fo_list),
            )
            all_ok &= _ok(
                "each mx n_pairs == H n_pairs",
                all(mx.n_pairs == H_aimspy.n_pairs for mx in fo_list),
            )

        # Check CPSCF converged
        out_path = DATA_DIR / "aims_dHde_capture.out"
        if out_path.is_file():
            with open(out_path, "r") as f:
                content = f.read()
            n_iters = content.count("End CPSCF iteration")
            converged = "CP-self-consistency cycle converged" in content
            _info(f"  CPSCF iterations: {n_iters}")
            all_ok &= _ok("CPSCF converged (capture)", converged)

        # =================================================================
        # Step 2: Export to deeph_dHde_out/
        # =================================================================
        if fo_list is not None:
            _info("")
            _info("=" * 60)
            _info(f"Step 2: Export to {DEEPH_DIR}")
            _info("=" * 60)

            dd = DeepHData.from_aimspy(
                structure,
                hamiltonian=H_aimspy,
                first_order_hamiltonian=fo_list,
            )
            _info(f"  DeepHData: {dd}")

            DEEPH_DIR.mkdir(parents=True, exist_ok=True)
            dd.save(DEEPH_DIR)
            _info("  Saved.")

            # =================================================================
            # Step 3: Cross-validation
            # =================================================================
            _info("")
            _info("=" * 60)
            _info("Step 3: Cross-validation")
            _info("=" * 60)

            # -- 3a. dHde roundtrip --
            _info("")
            _info("-- dHde roundtrip (aimspy -> DeepH -> aimspy) --")
            fo_back = dd.to_first_order_aimspy(structure)
            all_ok &= _ok(
                "roundtrip returns list of 3",
                isinstance(fo_back, list) and len(fo_back) == 3,
            )

            if isinstance(fo_back, list) and len(fo_back) == 3:
                max_diff = 0.0
                for cart in range(3):
                    orig_blocks = fo_list[cart].blocks
                    recv_blocks = fo_back[cart].blocks
                    if set(orig_blocks.keys()) != set(recv_blocks.keys()):
                        all_ok &= _ok(
                            f"direction {cart} keys match",
                            False,
                            f"orig {len(orig_blocks)} vs recv {len(recv_blocks)} keys",
                        )
                        continue
                    for key in orig_blocks:
                        diff = np.max(np.abs(recv_blocks[key] - orig_blocks[key]))
                        max_diff = max(max_diff, diff)
                all_ok &= _ok(
                    "dHde roundtrip max|diff| < 1e-10",
                    max_diff < 1e-10,
                    f"max|diff|={max_diff:.2e}",
                )

            # -- 3b. electric_response.h5 format --
            _info("")
            _info("-- electric_response.h5 format --")
            fo_h5_path = DEEPH_DIR / "electric_response.h5"
            all_ok &= _ok(
                "electric_response.h5 exists", fo_h5_path.is_file(), str(fo_h5_path)
            )

            if fo_h5_path.is_file():
                with h5py.File(fo_h5_path, "r") as f:
                    all_ok &= _ok("has atom_pairs", "atom_pairs" in f)
                    all_ok &= _ok("has chunk_boundaries", "chunk_boundaries" in f)
                    all_ok &= _ok("has chunk_shapes", "chunk_shapes" in f)
                    all_ok &= _ok("has entries", "entries" in f)

                    if "chunk_shapes" in f and "entries" in f:
                        fo_cs = f["chunk_shapes"][:]
                        fo_entries = f["entries"][:]
                        h_cs = dd.chunk_shapes
                        h_entries = dd.entries

                        if h_cs is not None:
                            fo_rows = fo_cs[:, 0]
                            h_rows = h_cs[:, 0]
                            rows_ok = np.all(fo_rows == 3 * h_rows)
                            all_ok &= _ok(
                                "fo_chunk_shape[:,0] == 3 * H chunk_shapes[:,0]",
                                rows_ok,
                                f"fo={fo_rows[:3]} vs 3*h={3 * h_rows[:3]}",
                            )
                            cols_ok = np.all(fo_cs[:, 1] == h_cs[:, 1])
                            all_ok &= _ok(
                                "fo_chunk_shape[:,1] == H chunk_shapes[:,1]",
                                cols_ok,
                            )

                        if h_entries is not None:
                            len_ok = len(fo_entries) == 3 * len(h_entries)
                            all_ok &= _ok(
                                "fo entries length == 3 * H entries",
                                len_ok,
                                f"fo={len(fo_entries)} vs 3*h={3 * len(h_entries)}",
                            )

            # -- 3c. from_directory round-trip --
            _info("")
            _info("-- from_directory round-trip --")
            dd2 = DeepHData.from_directory(DEEPH_DIR)
            all_ok &= _ok(
                "from_directory reads fo_entries",
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
                    "from_directory fo_entries match",
                    dir_diff < 1e-10,
                    f"max|diff|={dir_diff:.2e}",
                )

finally:
    calc.close()
    comm.Barrier()


# =============================================================================
# Summary
# =============================================================================
if rank == 0:
    _info("")
    _info("=" * 60)
    if all_ok:
        _info("DHDE CAPTURE TEST PASSED")
    else:
        _info("DHDE CAPTURE TEST FAILED")
    _info("=" * 60)
    if not all_ok:
        sys.exit(1)
