"""
Example: warmstart SCF from a pre-trained DeepH Hamiltonian.

Loads the DeepH data produced by ``from_scratch/run.py`` and uses it as the
initial Hamiltonian guess (``Strategy.REPLACE``).  SCF should converge in
1 iteration, demonstrating the warmstart workflow that is aimspy's core value
proposition for DeepX/DeepH integration.

Prerequisites:
    Run ``make run-from-scratch`` first to generate ``from_scratch/deeph_data/``.

Usage:
    source /path/to/intel/setvars.sh
    ulimit -s unlimited
    export AIMSPY_TEST_AIMS_LIBPATH=/path/to/libaims.so
    mpiexec -np 4 python run.py
"""

import os
import sys
from pathlib import Path

from aimspy import Calculator, CalculatorConfig, Strategy
from aimspy.interface.deeph import DeepHData
from mpi4py import MPI

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

# DeepH data produced by from_scratch/run.py (sibling directory).
DEEPH_DIR = Path(__file__).resolve().parent.parent / "from_scratch" / "deeph_data"

if not DEEPH_DIR.is_dir():
    if rank == 0:
        print(
            f"ERROR: {DEEPH_DIR} not found.\n"
            "  Run 'make run-from-scratch' first to generate DeepH data.",
            file=sys.stderr,
        )
    sys.exit(1)

# Load the external Hamiltonian as the warmstart source.
data = DeepHData.from_directory(DEEPH_DIR)
if rank == 0:
    print(f"[info] Loaded DeepH data: {data.n_atoms} atoms, {data.n_pairs} pairs")

config = CalculatorConfig(
    lib_path=LIB_PATH,
    logfile="aims_warmstart.out",
    log_level="INFO",
)
calc = Calculator(config)
calc.modify_init_ham(source=data, strategy=Strategy.REPLACE)

with calc:
    calc.do(comm=comm, work_dir=Path(__file__).resolve().parent)
    if rank == 0:
        print(f"[warmstart] energy = {calc.energy:.6f} Hartree")
        print("[warmstart] SCF should have converged in 1 iteration")
