"""
Example: baseline SCF + full DeepH export pipeline.

Runs a standard (non-warmstart) SCF on H2O in a 30 Å periodic box
(vacuum approximation of an isolated molecule), then extracts
Hamiltonian (H), overlap (S), initial/free-atom Hamiltonian (H_init),
total energy, and forces, and exports to DeepH format.

Usage:
    source /path/to/intel/setvars.sh
    ulimit -s unlimited
    export AIMSPY_TEST_AIMS_LIBPATH=/path/to/libaims.so
    cd /path/to/examples/from_scratch
    mpiexec -np 4 python run.py
"""

import os
import sys
from pathlib import Path

from aimspy import Calculator, CalculatorConfig
from mpi4py import MPI
from aimspy.interface.deeph import DeepHData

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
FHI_AIMS_LIB_PATH = Path(_lib_env)

config = CalculatorConfig(
    lib_path=FHI_AIMS_LIB_PATH,
    logfile="aims.out",
    log_level="INFO",
    capture_initial_hamiltonian=True,  # enables calc.initial_hamiltonian
    capture_overlap=True,  # enables all-rank live calc.overlap
)

with Calculator(config) as calc:
    calc.do(comm=comm, work_dir=".")

    if rank == 0:
        R = calc.structure
        H_init = calc.initial_hamiltonian
        H = calc.hamiltonian
        S = calc.overlap

        E = calc.energy
        print(f"[info] Total energy {E} Ha.")

        F = calc.forces
        print(f"[info] Forces:\n{F}")

# Export H + S + H_init to DeepH format (rank 0 only).
if rank == 0:
    data_mgr = DeepHData.from_aimspy(
        structure=R, hamiltonian=H, overlap=S, initial_hamiltonian=H_init
    )
    data_mgr.save("./deeph_data")
    print("[info] DeepH data saved to ./deeph_data")
