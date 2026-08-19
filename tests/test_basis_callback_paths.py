#!/usr/bin/env python
"""Integration test: export_basis_data registration-path semantics.

The export_basis_data callback is the only callback that fires *inside*
``aimspy_init`` (during ``prepare_scf``), which gives it unique
registration-path semantics that this script exercises on real
libaims (init-only — no SCF — so each stage is fast):

  Stage 1 — user pre-init registration:
      register_callback('export_basis_data', fn) called BEFORE init()
      fires during init; calc.basis_data stays None (the built-in
      capture is not enabled); the process survives close().

  Stage 2 — failing callback surfaces at init():
      a raising user callback must propagate as AimspyCallbackError
      from init() itself (not be deferred to calc()); the process
      survives the failed init + cleanup and can continue.

  Stage 3 — user registration takes precedence:
      capture_basis_data=True AND a user pre-init registration →
      the user's callback fires, the built-in capture is replaced,
      calc.basis_data is None.

Prerequisites:
    tests/data/MoS2_LDA/control.in  (xc pw-lda, SCF only)

Usage:
    source /path/to/intel/setvars.sh
    ulimit -s unlimited
    export AIMSPY_TEST_AIMS_LIBPATH=/path/to/libaims.so
    mpiexec -np 8 python tests/test_basis_callback_paths.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mpi4py import MPI

from aimspy import Calculator, CalculatorConfig
from aimspy._exceptions import AimspyCallbackError

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data" / "MoS2_LDA"

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

if not (DATA_DIR / "control.in").is_file():
    if rank == 0:
        print(f"ERROR: {DATA_DIR / 'control.in'} not found.", file=sys.stderr)
    sys.exit(1)


all_ok = True


def _info(msg):
    if rank == 0:
        print(msg, flush=True)


def check(name, condition, detail=""):
    global all_ok
    if rank == 0:
        tag = "OK  " if condition else "FAIL"
        line = f"  {tag}  {name}"
        if detail:
            line += f"  — {detail}"
        print(line, flush=True)
    if not condition:
        all_ok = False


def _fresh_config(**kwargs):
    return CalculatorConfig(
        lib_path=LIB_PATH,
        control_path=DATA_DIR / "control.in",
        geometry_path=DATA_DIR / "geometry.in",
        logfile=DATA_DIR / "aims.out",
        **kwargs,
    )


# ==========================================================================
# Stage 1: user pre-init registration fires during init()
# ==========================================================================
_info("=== Stage 1: user pre-init registration ===")
fired = []
calc = Calculator(_fresh_config())
calc.register_callback("export_basis_data", lambda ax, bd: fired.append(bd.n_basis_fns))
calc.init()
comm.Barrier()
check(
    "user callback fired during init",
    len(fired) == 1 and fired[0] > 0,
    f"fired = {fired}",
)
check("calc.basis_data is None (no built-in capture)", calc.basis_data is None)
calc.close()
comm.Barrier()
check("process survives close() after init-only run", True)

# ==========================================================================
# Stage 2: a raising callback surfaces as AimspyCallbackError from init()
# ==========================================================================
_info("=== Stage 2: failing callback -> AimspyCallbackError from init() ===")
calc = Calculator(_fresh_config())


def _boom(ax, bd):
    raise RuntimeError("deliberate failure inside export_basis_data")


calc.register_callback("export_basis_data", _boom)
init_err = None
try:
    calc.init()
except AimspyCallbackError as exc:
    init_err = exc
except BaseException as exc:  # noqa: BLE001 — record anything unexpected
    init_err = exc
comm.Barrier()
check(
    "init() raised AimspyCallbackError",
    isinstance(init_err, AimspyCallbackError),
    f"got {type(init_err).__name__ if init_err else None}: "
    f"{str(init_err)[:60] if init_err else ''}",
)
check(
    "error message mentions init()",
    init_err is not None and "during init()" in str(init_err),
)
# The Calculator is now FAILED; force_close must clean up without dying.
calc.force_close() if hasattr(calc, "force_close") else calc.close()
comm.Barrier()
check("process survives failed-init cleanup", True)

# ==========================================================================
# Stage 3: user registration takes precedence over built-in capture
# ==========================================================================
_info("=== Stage 3: user registration precedence ===")
user_fired = []
calc = Calculator(_fresh_config(capture_basis_data=True))


def _user_handler(ax, bd):
    user_fired.append(bd.n_basis_fns)
    ax.setdefault("store", []).append(bd.n_basis_fns)


calc.register_callback("export_basis_data", _user_handler)
calc.init()
comm.Barrier()
check(
    "user callback fired (precedence over built-in capture)",
    len(user_fired) == 1 and user_fired[0] > 0,
    f"fired = {user_fired}",
)
check(
    "calc.basis_data is None (built-in capture replaced)",
    calc.basis_data is None,
)
calc.close()
comm.Barrier()

# ==========================================================================
# Summary
# ==========================================================================
if rank == 0:
    print(flush=True)
    if all_ok:
        print("BASIS CALLBACK PATHS TEST PASSED", flush=True)
    else:
        print("BASIS CALLBACK PATHS TEST FAILED", flush=True)
        sys.exit(1)
