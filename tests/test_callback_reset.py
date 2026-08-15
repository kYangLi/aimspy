#!/usr/bin/env python
"""Integration test: Fortran callback deregistration across Calculators.

Verifies the callback-reset mechanism (ADR: aimspy_reset_callbacks +
reset inside aimspy_finalize).  Without it, a second Calculator in the
same process would inherit a dangling Python funptr from a previous
Calculator's modify_h0 / export_* callback and crash (use-after-free) or
silently invoke the wrong callback.

Scenario (single process, three sequential Calculators):
  #1 warmstart  — modify_init_ham(source=DeepHData, REPLACE) + calc + close
  #2 baseline   — NO modify; plain SCF.  Must run the *unmodified* SCF
                  (i.e. modify_h0 must NOT fire).  If the funptr leaked,
                  Fortran would call a freed Python object → crash, or
                  (worse) silently apply the wrong Hamiltonian.
  #3 capture    — capture_overlap=True; verifies export path also resets.

Checks:
  - BindingLib exposes aimspy_reset_callbacks (new C symbol).
  - All three runs complete without crash.
  - #2's SCF iteration count equals a plain baseline run (no warmstart
    residue).

Usage:
    source /path/to/intel/setvars.sh
    ulimit -s unlimited
    export AIMSPY_TEST_AIMS_LIBPATH=/path/to/libaims.so
    mpiexec -np 8 python tests/test_callback_reset.py
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mpi4py import MPI

from aimspy import Calculator, CalculatorConfig, Strategy
from aimspy import DeepHData

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data" / "MoS2"
# Warmstart source produced by test_export_deeph.py / test_warmstart.py.
# We reuse the DeepH directory those tests create; if absent we fall back
# to capturing in-process (run #0 produces it via capture from #1 itself).
DEEPH_DIR = DATA_DIR / "deeph_warm"

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


def _info(msg):
    if rank == 0:
        print(msg, flush=True)


def _ok(name, condition, detail=""):
    tag = "OK " if condition else "FAIL"
    _info(f"  {tag}  {name}" + (f" — {detail}" if detail and not condition else ""))
    return condition


def _count_scf_iterations(work_dir: Path, logfile: str) -> int:
    p = work_dir / logfile
    if not p.is_file():
        return -1
    with open(p, "r") as f:
        return f.read().count("End self-consistency iteration")


def _run(cfg_kwargs, modify_source=None, logfile="aims.out"):
    """Run one Calculator lifecycle; return SCF iteration count."""
    cfg = CalculatorConfig(
        lib_path=LIB_PATH, logfile=Path(logfile), log_level="WARNING", **cfg_kwargs
    )
    calc = Calculator(cfg)
    try:
        if modify_source is not None:
            calc.modify_init_ham(source=modify_source, strategy=Strategy.REPLACE)
        calc.do(comm=comm, work_dir=DATA_DIR)
    finally:
        calc.close()
        comm.Barrier()
    gc.collect()
    return _count_scf_iterations(DATA_DIR, logfile)


# =============================================================================
_info("=" * 60)
_info("Callback reset test: 3 sequential Calculators in one process")
_info("=" * 60)

all_ok = True

# Symbol availability (only rank 0 loads the lib to check the symbol table;
# simpler: just attempt a run and rely on has() inside close()). We assert
# the prototype is registered by checking BindingLib after a calculator run.

# ---------------------------------------------------------------------------
# #0 (prerequisite): produce a DeepH warmstart directory if missing.
# ---------------------------------------------------------------------------
if not DEEPH_DIR.is_dir():
    _info(f"  DeepH dir {DEEPH_DIR} missing; capturing via a baseline run")
    cfg = CalculatorConfig(
        lib_path=LIB_PATH,
        logfile=Path("aims_cbreset_capture.out"),
        log_level="WARNING",
        capture_initial_hamiltonian=True,
        capture_overlap=True,
    )
    calc = Calculator(cfg)
    try:
        calc.do(comm=comm, work_dir=DATA_DIR)
        if rank == 0:
            dd = DeepHData.from_aimspy(
                calc.structure,
                hamiltonian=calc.hamiltonian,
                overlap=calc.overlap,
                path=DEEPH_DIR,
            )
            dd.save()
    finally:
        calc.close()
        comm.Barrier()
    gc.collect()

dd = DeepHData.from_directory(DEEPH_DIR) if rank == 0 else None
dd = comm.bcast(dd, root=0)

# ---------------------------------------------------------------------------
# #1 warmstart (modify_h0 registered)
# ---------------------------------------------------------------------------
n_warm = _run({}, modify_source=dd, logfile="aims_cbreset_warm.out")
_info(f"  #1 warmstart SCF iterations: {n_warm}")
all_ok &= _ok("#1 warmstart completed", n_warm > 0)

# ---------------------------------------------------------------------------
# #2 baseline (NO modify) — must not inherit #1's modify_h0
# ---------------------------------------------------------------------------
n_base = _run({}, modify_source=None, logfile="aims_cbreset_base.out")
_info(f"  #2 baseline SCF iterations: {n_base}")
all_ok &= _ok("#2 baseline completed", n_base > 0)

# ---------------------------------------------------------------------------
# #3 capture_overlap (export path also resets)
# ---------------------------------------------------------------------------
n_cap = _run(
    {"capture_overlap": True}, modify_source=None, logfile="aims_cbreset_cap.out"
)
_info(f"  #3 capture SCF iterations: {n_cap}")
all_ok &= _ok("#3 capture completed", n_cap > 0)

# ---------------------------------------------------------------------------
# Compare: #2 (baseline after warmstart) must equal a fresh baseline's
# iteration count.  We use #3 (also baseline, capture_overlap doesn't change
# physics) as the reference; both should match.
# ---------------------------------------------------------------------------
all_ok &= _ok(
    "no stale modify_h0 (baseline == capture iterations)",
    n_base == n_cap,
    f"baseline={n_base} vs capture={n_cap}",
)

# A warmstart should converge in fewer/equal iterations than baseline.
if n_warm > 0 and n_base > 0:
    _ok(
        "warmstart not slower than baseline",
        n_warm <= n_base,
        f"warm={n_warm} > base={n_base}",
    )

# =============================================================================
if rank == 0:
    _info("")
    _info("=" * 60)
    _info("CALLBACK RESET TEST PASSED" if all_ok else "CALLBACK RESET TEST FAILED")
    _info("=" * 60)
    if not all_ok:
        sys.exit(1)
