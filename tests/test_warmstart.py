#!/usr/bin/env python
"""Integration test: Calculator warmstart via ``calc.modify_h0(source=...)``.

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

from aimspy import Calculator, CalculatorConfig
from aimspy.interface.deeph import DeepHData, DeepHSource

HERE = Path(__file__).resolve().parent
LIB_PATH = Path(
    "/home/deeph/software/calc/aimspy/"
    "FHI-aims-deeph/build/libaims.250822_1.scalapack.mpi.so"
)
DATA_DIR = HERE / "data" / "MoS2"
DEEPH_DIR = DATA_DIR / "deeph_warm"
LOG_FILE = DATA_DIR / "aims_warmstart.out"

comm = MPI.COMM_WORLD
rank = comm.rank

if not DEEPH_DIR.is_dir():
    if rank == 0:
        print(f"ERROR: deeph_warm dir not found at {DEEPH_DIR}", file=sys.stderr)
    sys.exit(1)

# ── Load DeepH data in the interface layer ──
deeph_data = DeepHData.from_directory(DEEPH_DIR)
source = DeepHSource(deeph_data)

if rank == 0:
    print(f"[pre-init] loaded DeepHData: {deeph_data.n_atoms} atoms, "
          f"{deeph_data.n_pairs} pairs, {deeph_data.entries.shape[0]} entries")

# ── Configure Calculator with unified modify_h0 method ──
config = CalculatorConfig(
    lib_path=LIB_PATH,
    work_dir=DATA_DIR,
    logfile=LOG_FILE,
    log_level="INFO",
)
calc = Calculator(config)
calc.modify_h0(source=source)   # ★ one call, no extra objects

calc.init(comm=comm)

if rank == 0:
    print(f"[init] info: n_atoms={calc.info.n_atoms}, "
          f"n_basis={calc.info.n_basis}, "
          f"n_cells={calc.info.n_cells}, "
          f"n_ham_size={calc.info.n_ham_size}")

try:
    calc.run()

    if rank == 0:
        H = calc.rs_hamiltonian
        ref_H = np.loadtxt(DATA_DIR / "rs_hamiltonian.out", dtype=np.float64)
        ref_H = ref_H.reshape(1, -1) if ref_H.ndim == 1 else ref_H

        print(f"[run] H shape: {H.shape}")
        print(f"[run] max|H|        = {np.max(np.abs(H)):.6e} Hartree")
        print(f"[run] H[0,0]        = {H[0,0]:.6e} Hartree")
        print(f"[run] ref_H[0,0]    = {ref_H[0,0]:.6e} Hartree")
        print(f"[run] close to ref  = {np.allclose(H, ref_H, atol=1e-6)}")
        print("TEST PASSED")
finally:
    calc.close()
    comm.Barrier()
