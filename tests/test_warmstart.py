#!/usr/bin/env python
"""Integration test: Calculator warmstart via ``calc.modify()``.

Tests both direct source and deferred source modes.

Usage:
    source /home/deeph/software/env/IntelOneAPI/install/setvars.sh
    ulimit -s unlimited
    cd /home/deeph/software/calc/aimspy/pyapi
    mpirun -np 8 python tests/test_warmstart.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from mpi4py import MPI

from aimspy import Calculator, CalculatorConfig, Strategy
from aimspy.interface.deeph import DeepHData

HERE = Path(__file__).resolve().parent
LIB_PATH = Path(
    "/home/deeph/software/calc/aimspy/"
    "FHI-aims-deeph/build/libaims.250822_1.scalapack.mpi.so"
)
DATA_DIR = HERE / "data" / "MoS2"
DEEPH_DIR = DATA_DIR / "deeph_warm"

comm = MPI.COMM_WORLD
rank = comm.rank

if not DEEPH_DIR.is_dir():
    if rank == 0:
        print(f"ERROR: deeph_warm dir not found at {DEEPH_DIR}", file=sys.stderr)
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
    print("Test 1: Direct source (calc.modify(source=data))")
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
calc.modify(source=deeph_data, strategy=Strategy.REPLACE)

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
    print("Test 2: Deferred source (@calc.modify decorator)")
    print("=" * 60)

config2 = CalculatorConfig(
    lib_path=LIB_PATH,
    logfile=Path("aims_warmstart_defer.out"),
    log_level="INFO",
    capture_initial_hamiltonian=True,
)
calc2 = Calculator(config2)


@calc2.modify(strategy=Strategy.REPLACE, aux={"deeph_path": str(DEEPH_DIR)})
def gen_source(calculator, aux):
    """Lazy source: read DeepH data at runtime (during python_func).

    At this point, calculator.initial_hamiltonian and calculator.overlap
    are available if capture_* was enabled.
    """
    return DeepHData.from_directory(aux["deeph_path"])


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
