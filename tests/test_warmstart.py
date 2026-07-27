#!/usr/bin/env python
"""Integration test: Calculator warmstart via ``calc.modify_init_ham()``.

Tests both direct source and deferred source modes.

Also verifies that RSS memory does not accumulate across two consecutive
Calculator cycles (reference-cycle regression test).

Usage:
    source /path/to/intel/setvars.sh
    ulimit -s unlimited
    export AIMSPY_TEST_AIMS_LIBPATH=/path/to/libaims.so
    mpiexec -np 8 python tests/test_warmstart.py
"""

from __future__ import annotations

import gc
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from mpi4py import MPI

from aimspy import Calculator, CalculatorConfig, Strategy
from aimspy import DeepHData

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data" / "MoS2"
DEEPH_DIR = DATA_DIR / "deeph_out"

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
            f"ERROR: {DEEPH_DIR} not found.\n"
            "  Run 'make test-export-deeph' first to generate DeepH data.",
            file=sys.stderr,
        )
    sys.exit(1)


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
        print(f"[mem] {label}: RSS = {rss} KB ({rss / 1024:.1f} MB)")
    return rss


def check_result(label, H, ref_H):
    ok = np.allclose(H, ref_H, atol=1e-6)
    if rank == 0:
        print(f"[{label}] H shape: {H.shape}")
        print(f"[{label}] max|H|        = {np.max(np.abs(H)):.6e} Hartree")
        print(f"[{label}] H[0,0]        = {H[0,0]:.6e} Hartree")
        print(f"[{label}] ref_H[0,0]    = {ref_H[0,0]:.6e} Hartree")
        print(f"[{label}] close to ref  = {ok}")
    return ok


ref_H = np.loadtxt(DATA_DIR / "rs_hamiltonian.out", dtype=np.float64)
ref_H = ref_H.reshape(1, -1) if ref_H.ndim == 1 else ref_H

# =============================================================================
# RSS baseline (before any Calculator runs)
# =============================================================================
rss_baseline = _rss_snapshot("baseline (before any calc)")

# =============================================================================
# Test 1: Direct source (pre-built DeepHData)
# =============================================================================
if rank == 0:
    print("=" * 60)
    print("Test 1: Direct source (calc.modify_init_ham(source=data))")
    print("=" * 60)

deeph_data = DeepHData.from_directory(DEEPH_DIR)
if rank == 0:
    print(
        f"[direct] loaded DeepHData: {deeph_data.n_atoms} atoms, "
        f"{deeph_data.n_pairs} pairs, {deeph_data.entries.shape[0]} entries"
    )

config = CalculatorConfig(
    lib_path=LIB_PATH,
    logfile=Path("aims_warmstart_direct.out"),
    log_level="INFO",
)
calc = Calculator(config)
calc.modify_init_ham(source=deeph_data, strategy=Strategy.REPLACE)

try:
    calc.do(comm=comm, work_dir=DATA_DIR)
    if rank == 0:
        H = calc.rs_hamiltonian
        ok1 = check_result("direct", H, ref_H)
        if ok1:
            print("DIRECT SOURCE TEST PASSED")
finally:
    calc.close()
    comm.Barrier()

# RSS after 1st calc cycle (direct mode, no capture_overlap)
rss_after_1 = _rss_snapshot("after 1st calc (direct, close'd)")

# =============================================================================
# Test 2: Deferred source (decorator, source generated at runtime)
# =============================================================================
if rank == 0:
    print()
    print("=" * 60)
    print("Test 2: Deferred source (@calc.modify_init_ham decorator)")
    print("=" * 60)

config2 = CalculatorConfig(
    lib_path=LIB_PATH,
    logfile=Path("aims_warmstart_defer.out"),
    log_level="INFO",
    capture_initial_hamiltonian=True,
)
calc2 = Calculator(config2)


@calc2.modify_init_ham(strategy=Strategy.REPLACE, option={"deeph_path": str(DEEPH_DIR)})
def gen_source(calculator, option):
    """Lazy source: read DeepH data at runtime (during python_func).

    At this point, calculator.initial_hamiltonian and calculator.overlap
    are available if capture_* was enabled.
    """
    return DeepHData.from_directory(option["deeph_path"])


try:
    calc2.do(comm=comm, work_dir=DATA_DIR)
    if rank == 0:
        H2 = calc2.rs_hamiltonian
        ok2 = check_result("defer", H2, ref_H)
        if ok2:
            print("DEFERRED SOURCE TEST PASSED")
finally:
    calc2.close()
    comm.Barrier()

# RSS after 2nd calc cycle (deferred mode, capture_initial_hamiltonian)
rss_after_2 = _rss_snapshot("after 2nd calc (deferred, close'd)")

# =============================================================================
# Memory regression check: RSS should not grow unboundedly across cycles.
# A small delta is expected (malloc arena caching, MKL internal buffers),
# but a large delta indicates a leak (e.g. reference cycle not broken).
# =============================================================================
# MoS2 test system is small (~90 basis, ~91k hamiltonian entries).
# Allow up to 200 MB growth for malloc/arena caching on this small system.
# Larger systems would need a higher threshold.
MEM_DELTA_THRESHOLD_KB = 200 * 1024  # 200 MB
rss_delta = rss_after_2 - rss_after_1
mem_ok = rss_delta < MEM_DELTA_THRESHOLD_KB

if rank == 0:
    print()
    print("=" * 60)
    print("Memory Regression Check")
    print("=" * 60)
    print(f"  RSS baseline:     {rss_baseline:>10d} KB ({rss_baseline / 1024:.1f} MB)")
    print(f"  RSS after 1st:    {rss_after_1:>10d} KB ({rss_after_1 / 1024:.1f} MB)")
    print(f"  RSS after 2nd:    {rss_after_2:>10d} KB ({rss_after_2 / 1024:.1f} MB)")
    print(f"  Delta (2nd - 1st): {rss_delta:>10d} KB ({rss_delta / 1024:.1f} MB)")
    print(
        f"  Threshold:         {MEM_DELTA_THRESHOLD_KB:>10d} KB ({MEM_DELTA_THRESHOLD_KB / 1024:.1f} MB)"
    )
    if mem_ok:
        print("  MEM CHECK PASSED — no unbounded growth")
    else:
        print("  MEM CHECK FAILED — RSS grew beyond threshold (possible leak)")

# =============================================================================
# Summary
# =============================================================================
if rank == 0:
    print()
    print("=" * 60)
    if ok1 and ok2 and mem_ok:
        print("ALL WARMSTART TESTS PASSED (incl. memory check)")
    elif ok1 and ok2:
        print("WARMSTART TESTS PASSED, but memory check FAILED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)
    if not (ok1 and ok2 and mem_ok):
        sys.exit(1)
