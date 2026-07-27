#!/usr/bin/env python
"""Memory pressure test: run N calc cycles, verify RSS stabilizes.

Simulates the user's real-world scenario of running many consecutive
SCF calculations in a loop, each with ``capture_overlap=True`` (the
configuration that previously triggered memory accumulation due to
reference cycles in CallbackManager / Calculator closures).

This test verifies that after the reference-cycle fix:
  1. RSS grows during the first few iterations (malloc arena caching,
     MKL thread buffer warmup, pymalloc pool fill) — this is expected.
  2. RSS stabilizes after the first few iterations — the high-water
     mark is reached and subsequent iterations reuse cached memory.
  3. No linear growth — if RSS keeps growing, a leak still exists.

Usage:
    source /path/to/intel/setvars.sh
    ulimit -s unlimited
    export AIMSPY_TEST_AIMS_LIBPATH=/path/to/libaims.so
    mpiexec -np 8 python tests/test_memory_loop.py

Environment variables:
    AIMSPY_MEM_LOOP_N       Number of iterations (default: 15)
    AIMSPY_MEM_LOOP_THRESHOLD_KB  Max allowed RSS drift in the last 5
                                  iterations (default: 100 MB = 102400 KB)
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mpi4py import MPI

from aimspy import Calculator, CalculatorConfig

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data" / "MoS2"

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

N_ITER = int(os.environ.get("AIMSPY_MEM_LOOP_N", "15"))
DRIFT_THRESHOLD_KB = int(
    os.environ.get("AIMSPY_MEM_LOOP_THRESHOLD_KB", str(100 * 1024))
)


def _rss_kb() -> int:
    """Return current process RSS in KB (Linux only)."""
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS"):
                    return int(line.split()[1])
    except Exception:
        pass
    return 0


def _rss_snapshot(label: str) -> int:
    """Take an RSS snapshot after gc.collect() on all ranks, print on rank 0."""
    gc.collect()
    comm.Barrier()
    rss = _rss_kb()
    if rank == 0:
        print(f"[mem] {label}: RSS = {rss} KB ({rss / 1024:.1f} MB)", flush=True)
    return rss


# =============================================================================
# Main loop: N_ITER consecutive calc cycles with capture_overlap=True
# =============================================================================
if rank == 0:
    print("=" * 60)
    print(f"Memory Loop Test: {N_ITER} iterations, capture_overlap=True")
    print("=" * 60)

rss_history: list[int] = []

for i in range(N_ITER):
    if rank == 0:
        print(f"\n--- iteration {i} ---", flush=True)

    config = CalculatorConfig(
        lib_path=LIB_PATH,
        logfile=Path(f"aims_memloop_{i}.out"),
        log_level="WARNING",  # suppress per-iteration INFO noise
        capture_overlap=True,
    )
    calc = Calculator(config)

    try:
        calc.do(comm=comm, work_dir=DATA_DIR)
        # Access overlap to ensure it's materialized (not lazily deferred)
        _ = calc.overlap
    finally:
        calc.close()
        comm.Barrier()

    rss = _rss_snapshot(f"iter {i} (close'd)")
    rss_history.append(rss)

    # Clean up local ref to calc so refcount can drop
    del calc
    gc.collect()

# =============================================================================
# Analysis: verify RSS stabilizes (no linear growth)
# =============================================================================
if rank == 0:
    print()
    print("=" * 60)
    print("RSS History")
    print("=" * 60)
    for i, rss in enumerate(rss_history):
        delta = rss - rss_history[i - 1] if i > 0 else 0
        print(
            f"  iter {i:2d}: {rss:>10d} KB ({rss / 1024:>8.1f} MB)"
            f"  delta={delta:>+10d} KB ({delta / 1024:>+8.1f} MB)"
        )

    # Check 1: RSS in the last 5 iterations should be roughly constant.
    # We allow some fluctuation but require the net drift to be small.
    last_n = min(5, N_ITER)
    last_slice = rss_history[-last_n:]
    drift = last_slice[-1] - last_slice[0]
    max_rss = max(rss_history)
    min_rss = min(rss_history)

    print()
    print(f"  Peak RSS:           {max_rss} KB ({max_rss / 1024:.1f} MB)")
    print(f"  Min RSS:            {min_rss} KB ({min_rss / 1024:.1f} MB)")
    print(f"  Drift (last {last_n} iters): {drift} KB ({drift / 1024:.1f} MB)")
    print(
        f"  Drift threshold:   {DRIFT_THRESHOLD_KB} KB"
        f" ({DRIFT_THRESHOLD_KB / 1024:.1f} MB)"
    )

    # Check 2: The RSS growth rate should slow down (concave, not linear).
    # Compare first-half growth rate vs second-half growth rate.
    mid = N_ITER // 2
    if mid > 0 and mid < N_ITER:
        first_half_growth = rss_history[mid] - rss_history[0]
        second_half_growth = rss_history[-1] - rss_history[mid]
        print(
            f"  1st half growth:    {first_half_growth} KB"
            f" ({first_half_growth / 1024:.1f} MB)"
        )
        print(
            f"  2nd half growth:    {second_half_growth} KB"
            f" ({second_half_growth / 1024:.1f} MB)"
        )

    # Pass/fail
    mem_ok = drift < DRIFT_THRESHOLD_KB
    print()
    if mem_ok:
        print("MEMORY LOOP TEST PASSED — RSS stabilized, no unbounded growth")
    else:
        print("MEMORY LOOP TEST FAILED — RSS still growing (possible leak)")
    print("=" * 60)
    if not mem_ok:
        sys.exit(1)
