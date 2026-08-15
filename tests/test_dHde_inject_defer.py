#!/usr/bin/env python
"""Integration test: DFPT dH/de warmstart (deferred mode).

Uses the @modify_init_first_order_ham decorator to generate the source
at runtime (during the modify_dHde callback), then runs SCF+CPSCF.

Verifies CPSCF converges with fewer iterations than capture.

Prerequisites:
    Run test_dHde_capture.py first (produces deeph_dHde_out/).

Usage:
    source /path/to/intel/setvars.sh
    ulimit -s unlimited
    export AIMSPY_TEST_AIMS_LIBPATH=/path/to/libaims.so
    mpiexec -np 8 python tests/test_dHde_inject_defer.py
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
DATA_DIR = HERE / "data" / "MoS2_DFPT"
DEEPH_DIR = DATA_DIR / "deeph_dHde_out"
CAPTURE_LOG = "aims_dHde_capture.out"
INJECT_LOG = "aims_dHde_inject_defer.out"

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

if not DEEPH_DIR.is_dir():
    if rank == 0:
        print(
            f"ERROR: {DEEPH_DIR} not found.\n" "  Run 'make test-dHde-capture' first.",
            file=sys.stderr,
        )
    sys.exit(1)


def _info(msg):
    if rank == 0:
        print(msg)


def _ok(name, condition, detail=""):
    tag = "OK " if condition else "FAIL"
    _info(f"  {tag}  {name}" + (f" — {detail}" if detail and not condition else ""))
    return condition


def _rss_kb() -> int:
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    return int(line.split()[1])
    except Exception:
        pass
    return 0


def _rss_snapshot(label: str) -> int:
    gc.collect()
    comm.Barrier()
    rss = _rss_kb()
    if rank == 0:
        print(f"[mem] {label}: RSS = {rss} KB ({rss / 1024:.1f} MB)")
    return rss


def _count_cpscf_iterations(work_dir: Path, logfile: str) -> int:
    out_path = work_dir / logfile
    if not out_path.is_file():
        return -1
    with open(out_path, "r") as f:
        content = f.read()
    return content.count("End CPSCF iteration")


def _check_cpscf_converged(work_dir: Path, logfile: str) -> bool:
    out_path = work_dir / logfile
    if not out_path.is_file():
        return False
    with open(out_path, "r") as f:
        content = f.read()
    return "CP-self-consistency cycle converged" in content


# =============================================================================
# RSS baseline
# =============================================================================
rss_baseline = _rss_snapshot("baseline")


# =============================================================================
# Run SCF + CPSCF with deferred dH/de injection
# =============================================================================
_info("=" * 60)
_info("Deferred mode warmstart: @modify_init_first_order_ham decorator")
_info("=" * 60)

config = CalculatorConfig(
    lib_path=LIB_PATH,
    logfile=Path(INJECT_LOG),
    log_level="INFO",
)
calc = Calculator(config)


@calc.modify_init_first_order_ham(
    strategy=Strategy.REPLACE, option={"deeph_path": str(DEEPH_DIR)}
)
def gen_fo_source(view, option):
    """Lazy source: read DeepH data at runtime (during modify_dHde callback)."""
    return DeepHData.from_directory(option["deeph_path"])


all_ok = True
try:
    calc.do(comm=comm, work_dir=DATA_DIR)

    if rank == 0:
        converged = _check_cpscf_converged(DATA_DIR, INJECT_LOG)
        n_inject_iters = _count_cpscf_iterations(DATA_DIR, INJECT_LOG)
        _info(f"  CPSCF iterations (deferred inject): {n_inject_iters}")
        all_ok &= _ok("CPSCF converged (deferred inject)", converged)

        # Compare with capture iterations
        n_capture_iters = _count_cpscf_iterations(DATA_DIR, CAPTURE_LOG)
        if n_capture_iters > 0 and n_inject_iters > 0:
            _info(f"  CPSCF iterations (capture):        {n_capture_iters}")
            reduction = n_capture_iters - n_inject_iters
            pct = 100 * reduction / n_capture_iters if n_capture_iters > 0 else 0
            _info(f"  Reduction: {reduction} iterations ({pct:.0f}%)")
            all_ok &= _ok(
                "deferred inject reduces CPSCF iterations",
                n_inject_iters < n_capture_iters,
                f"{n_inject_iters} >= {n_capture_iters}",
            )

finally:
    calc.close()
    comm.Barrier()

rss_after = _rss_snapshot("after deferred warmstart")


# =============================================================================
# Memory check
# =============================================================================
MEM_DELTA_THRESHOLD_KB = 200 * 1024
rss_delta = rss_after - rss_baseline
mem_ok = rss_delta < MEM_DELTA_THRESHOLD_KB

if rank == 0:
    _info("")
    _info("=" * 60)
    _info("Memory Check")
    _info("=" * 60)
    _info(f"  RSS baseline:  {rss_baseline:>10d} KB ({rss_baseline / 1024:.1f} MB)")
    _info(f"  RSS after:     {rss_after:>10d} KB ({rss_after / 1024:.1f} MB)")
    _info(f"  Delta:         {rss_delta:>10d} KB ({rss_delta / 1024:.1f} MB)")
    _info(f"  Threshold:     {MEM_DELTA_THRESHOLD_KB:>10d} KB")
    if mem_ok:
        _info("  MEM CHECK PASSED")
    else:
        _info("  MEM CHECK FAILED")


# =============================================================================
# Summary
# =============================================================================
if rank == 0:
    _info("")
    _info("=" * 60)
    if all_ok and mem_ok:
        _info("DHDE INJECT DEFER TEST PASSED")
    elif all_ok:
        _info("DHDE INJECT DEFER TEST PASSED (memory check failed)")
    else:
        _info("DHDE INJECT DEFER TEST FAILED")
    _info("=" * 60)
    if not all_ok:
        sys.exit(1)
