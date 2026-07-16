"""
Example: baseline SCF + full DeepH export pipeline.

Runs a standard (non-warmstart) SCF on H2O in a 30 Å periodic box
(vacuum approximation of an isolated molecule), then extracts
Hamiltonian (H), overlap (S), initial/free-atom Hamiltonian (H_init),
total energy, and forces, and exports to DeepH format.

Usage:
    source /home/deeph/software/env/IntelOneAPI/install/setvars.sh
    ulimit -s unlimited
    cd /path/to/examples/from_scratch
    mpirun -np 4 python run.py
"""

from aimspy import Calculator, CalculatorConfig
from mpi4py import MPI
from aimspy.interface.deeph import DeepHData

comm = MPI.COMM_WORLD
rank = comm.rank

# Path to the patched FHI-aims shared library.
FHI_AIMS_LIB_PATH = "/home/deeph/software/calc/aimspy/FHI-aims-deeph/build/libaims.250822_1.scalapack.mpi.so"

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
