#!/usr/bin/env python
"""Baseline test: Calculator without modify (standard free-atom SCF).

Uses the two-step aimspy API (construct config, then run with comm/work_dir).

Usage:
    source /path/to/intel/setvars.sh
    ulimit -s unlimited
    export AIMSPY_TEST_AIMS_LIBPATH=/path/to/libaims.so
    mpiexec -np 8 python tests/test_baseline.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
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

config = CalculatorConfig(
    lib_path=LIB_PATH,
    logfile=Path("aims_baseline.out"),
    log_level="INFO",
)
calc = Calculator(config)

try:
    calc.do(comm=comm, work_dir=DATA_DIR)
    if rank == 0:
        print(
            f"[baseline] info: n_atoms={calc.info.n_atoms}, "
            f"n_basis={calc.info.n_basis}, "
            f"n_cells={calc.info.n_cells}, "
            f"n_ham_size={calc.info.n_ham_size}"
        )
        H = calc.rs_hamiltonian
        ref_H = np.loadtxt(DATA_DIR / "rs_hamiltonian.out", dtype=np.float64)
        ref_H = ref_H.reshape(1, -1) if ref_H.ndim == 1 else ref_H
        print(f"[baseline] H shape: {H.shape}")
        print(f"[baseline] max|H|      = {np.max(np.abs(H)):.6e} Hartree")
        print(f"[baseline] max|H_ref|   = {np.max(np.abs(ref_H)):.6e} Hartree")
        print(f"[baseline] H[0, 0]     = {H[0, 0]:.6e} Hartree")
        print(f"[baseline] ref_H[0, 0] = {ref_H[0, 0]:.6e} Hartree")
        print(f"[baseline] close to ref = {np.allclose(H, ref_H, atol=1e-6)}")
        print(f"[baseline] energy = {calc.energy:.6f} Hartree")
        print("BASELINE TEST PASSED")
except Exception:
    import traceback

    if rank == 0:
        traceback.print_exc()
    raise
finally:
    calc.close()
    comm.Barrier()
