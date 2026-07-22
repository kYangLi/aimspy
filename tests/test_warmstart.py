#!/usr/bin/env python
"""Integration test: Calculator warmstart via ``calc.modify_init_ham()``.

Tests both direct source and deferred source modes.

Usage:
    source /path/to/intel/setvars.sh
    ulimit -s unlimited
    export AIMSPY_TEST_AIMS_LIBPATH=/path/to/libaims.so
    mpiexec -np 8 python tests/test_warmstart.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from mpi4py import MPI

from aimspy import Calculator, CalculatorConfig, Strategy
from aimspy.interface.deeph import DeepHData

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

# =============================================================================
# Summary
# =============================================================================
if rank == 0:
    print()
    print("=" * 60)
    if ok1 and ok2:
        print("ALL WARMSTART TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 60)
    if not (ok1 and ok2):
        sys.exit(1)
