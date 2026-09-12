#!/usr/bin/env python
"""Spinful warmstart integration test: modify_init_ham on a spin-collinear
system.

Requires the spinful DeepH export from ``make test-spin-export``
(``tests/data/MoS2_spin/deeph_spin_out/``) and the spinless export from
``make test-export-deeph`` (``tests/data/MoS2/deeph_out/``, used by the
cross-spin guard test).

Tests:
1. REPLACE direct source  — final two-channel H matches the baseline.
2. REPLACE deferred source — same, via the decorator mode.
3. ADD                    — live + external relaxes back to the ground state.
4. Cross-spin guard       — a spinless source on the spin-polarized system
                            raises ``AimspyCallbackError`` (no silent
                            zero-fill).

Usage:
    source /path/to/intel/setvars.sh
    ulimit -s unlimited
    export AIMSPY_TEST_AIMS_LIBPATH=/path/to/libaims.so
    mpiexec -np 8 python tests/test_spin_warmstart.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from mpi4py import MPI

from aimspy import Calculator, CalculatorConfig, Strategy
from aimspy import DeepHData
from aimspy._exceptions import AimspyCallbackError

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data" / "MoS2_spin"
DEEPH_DIR = DATA_DIR / "deeph_spin_out"
SPINLESS_DIR = HERE / "data" / "MoS2" / "deeph_out"

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
            "  Run 'make test-spin-export' first to generate spinful DeepH data.",
            file=sys.stderr,
        )
    sys.exit(1)

if not SPINLESS_DIR.is_dir():
    if rank == 0:
        print(
            f"ERROR: {SPINLESS_DIR} not found.\n"
            "  Run 'make test-export-deeph' first to generate spinless DeepH data.",
            file=sys.stderr,
        )
    sys.exit(1)


def _info(msg):
    if rank == 0:
        print(msg)


def check_result(label, H, ref_H):
    """Compare the two-channel H against the baseline reference."""
    ok = np.allclose(H, ref_H, atol=1e-6)
    if rank == 0:
        print(f"[{label}] H shape: {H.shape}")
        print(f"[{label}] max|H|       = {np.max(np.abs(H)):.6e} Hartree")
        print(f"[{label}] max|H - ref| = " f"{np.max(np.abs(H - ref_H)):.2e} Hartree")
        print(f"[{label}] close to ref = {ok}")
    return ok


# Two-channel reference: FHI-aims per-channel outputs from the export run.
ref_up_path = DATA_DIR / "rs_hamiltonian_up.out"
ref_dn_path = DATA_DIR / "rs_hamiltonian_dn.out"
if not (ref_up_path.is_file() and ref_dn_path.is_file()):
    if rank == 0:
        print(
            f"ERROR: {ref_up_path} / {ref_dn_path} not found.\n"
            "  Run 'make test-spin-export' first — the SCF run writes the "
            "per-channel reference files.",
            file=sys.stderr,
        )
    sys.exit(1)
ref_up = np.loadtxt(ref_up_path, dtype=np.float64)
ref_dn = np.loadtxt(ref_dn_path, dtype=np.float64)
ref_up = ref_up.reshape(1, -1) if ref_up.ndim == 1 else ref_up
ref_dn = ref_dn.reshape(1, -1) if ref_dn.ndim == 1 else ref_dn
ref_H = np.vstack([ref_up, ref_dn])  # (2, n_ham)

deeph_data = DeepHData.from_directory(DEEPH_DIR)
if rank == 0:
    print(
        f"[setup] loaded spinful DeepHData: {deeph_data.n_atoms} atoms, "
        f"{deeph_data.n_pairs} pairs, {deeph_data.entries.shape[0]} entries"
    )

# =============================================================================
# Test 1: REPLACE direct source
# =============================================================================
if rank == 0:
    print("=" * 60)
    print("Test 1: REPLACE direct source (spinful)")
    print("=" * 60)

config = CalculatorConfig(
    lib_path=LIB_PATH,
    logfile=Path("aims_spin_warm_direct.out"),
    log_level="INFO",
)
calc = Calculator(config)
calc.modify_init_ham(source=deeph_data, strategy=Strategy.REPLACE)

ok1 = False
try:
    calc.do(comm=comm, work_dir=DATA_DIR)
    if rank == 0:
        H = calc.rs_hamiltonian
        ok1 = check_result("direct", H, ref_H)
        if ok1:
            print("SPIN REPLACE DIRECT TEST PASSED")
finally:
    calc.close()
    comm.Barrier()

# =============================================================================
# Test 2: REPLACE deferred source (decorator, source generated at runtime)
# =============================================================================
if rank == 0:
    print()
    print("=" * 60)
    print("Test 2: REPLACE deferred source (spinful)")
    print("=" * 60)

config2 = CalculatorConfig(
    lib_path=LIB_PATH,
    logfile=Path("aims_spin_warm_defer.out"),
    log_level="INFO",
    capture_initial_hamiltonian=True,
)
calc2 = Calculator(config2)


@calc2.modify_init_ham(strategy=Strategy.REPLACE, option={"deeph_path": str(DEEPH_DIR)})
def gen_source(calculator, option):
    """Lazy spinful source: read DeepH data during python_func."""
    return DeepHData.from_directory(option["deeph_path"])


ok2 = False
try:
    calc2.do(comm=comm, work_dir=DATA_DIR)
    if rank == 0:
        H2 = calc2.rs_hamiltonian
        ok2 = check_result("defer", H2, ref_H)
        if ok2:
            print("SPIN REPLACE DEFERRED TEST PASSED")
finally:
    calc2.close()
    comm.Barrier()

# =============================================================================
# Test 3: ADD (live + external relaxes back to the ground state)
# =============================================================================
if rank == 0:
    print()
    print("=" * 60)
    print("Test 3: ADD (spinful)")
    print("=" * 60)

config3 = CalculatorConfig(
    lib_path=LIB_PATH,
    logfile=Path("aims_spin_warm_add.out"),
    log_level="INFO",
)
calc3 = Calculator(config3)
calc3.modify_init_ham(source=deeph_data, strategy=Strategy.ADD)

ok3 = False
try:
    calc3.do(comm=comm, work_dir=DATA_DIR)
    if rank == 0:
        H3 = calc3.rs_hamiltonian
        ok3 = check_result("add", H3, ref_H)
        if ok3:
            print("SPIN ADD TEST PASSED")
finally:
    calc3.close()
    comm.Barrier()

# =============================================================================
# Test 4: Cross-spin guard — spinless source on the spin-polarized system
# must raise AimspyCallbackError (no silent zero-fill of the beta channel).
# =============================================================================
if rank == 0:
    print()
    print("=" * 60)
    print("Test 4: Cross-spin guard (spinless source → spinful system)")
    print("=" * 60)

deeph_spinless = DeepHData.from_directory(SPINLESS_DIR)
config4 = CalculatorConfig(
    lib_path=LIB_PATH,
    logfile=Path("aims_spin_guard.out"),
    log_level="INFO",
)
calc4 = Calculator(config4)
calc4.modify_init_ham(source=deeph_spinless, strategy=Strategy.REPLACE)

ok4 = False
try:
    calc4.do(comm=comm, work_dir=DATA_DIR)
    if rank == 0:
        print("GUARD TEST FAILED — no exception was raised")
except AimspyCallbackError as exc:
    ok4 = True
    if rank == 0:
        cb_err = exc.callback_errors[-1] if exc.callback_errors else None
        detail = cb_err[1] if cb_err else exc
        print(f"Raised AimspyCallbackError as expected: {detail}")
        print("CROSS-SPIN GUARD TEST PASSED")
finally:
    calc4.force_close()
    comm.Barrier()

# =============================================================================
# Summary
# =============================================================================
if rank == 0:
    print()
    print("=" * 60)
    if ok1 and ok2 and ok3 and ok4:
        print("ALL SPIN WARMSTART TESTS PASSED")
    else:
        print(
            "SOME SPIN WARMSTART TESTS FAILED: "
            f"direct={ok1} defer={ok2} add={ok3} guard={ok4}"
        )
    print("=" * 60)
    if not (ok1 and ok2 and ok3 and ok4):
        sys.exit(1)
