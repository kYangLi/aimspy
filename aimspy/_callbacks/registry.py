"""Private — callback registry: the authoritative list of all callback types.

This is **the** central catalogue.  Adding a new callback = adding one
``CallbackSpec`` entry here, plus changes in 3 other wired places
(Fortran patch, ``_binding/callback_types.py``, ``_binding/prototypes.py``,
``_callbacks/base.py`` wrapper branch).
"""

from __future__ import annotations

from enum import Enum

from .._binding.callback_types import (
    GetDescrCb,
    ExportOvlpCb,
    ExportH0Cb,
    ModifyH0Cb,
    ReconstructMxCb,
)
from .base import CallbackSpec


class CallbackName(Enum):
    """Callback type identifiers (accepts ``str`` or ``CallbackName``).

    Used by :meth:`aimspy.Calculator.register_callback` and
    :meth:`aimspy.Calculator.callback_registered`.
    """

    GET_DESCR = "get_descr"
    EXPORT_OVLP = "export_ovlp"
    EXPORT_H0 = "export_h0"
    MODIFY_H0 = "modify_h0"
    PYTHON_FUNC = "python_func"


# =========================================================================
# Authoritative list — one entry per callback type
# =========================================================================
CALLBACK_SPECS: list[CallbackSpec] = [
    CallbackSpec(
        name="get_descr",
        ctypes_type=GetDescrCb,
        register_symbol="aimspy_register_get_descr_callback",
        register_arg_count=2,
        trigger_stage="pre_scf",
        fortran_module="initialize_scf.f90:912",
    ),
    CallbackSpec(
        name="export_ovlp",
        ctypes_type=ExportOvlpCb,
        register_symbol="aimspy_register_export_ovlp_callback",
        register_arg_count=2,
        trigger_stage="pre_scf",
        fortran_module="initialize_scf.f90:913",
    ),
    CallbackSpec(
        name="export_h0",
        ctypes_type=ExportH0Cb,
        register_symbol="aimspy_register_export_h0_callback",
        register_arg_count=2,
        trigger_stage="pre_scf",
        fortran_module="initialize_scf.f90:914",
    ),
    CallbackSpec(
        name="modify_h0",
        ctypes_type=ModifyH0Cb,
        register_symbol="aimspy_register_modify_h0_callback",
        register_arg_count=3,
        trigger_stage="pre_scf",
        fortran_module="initialize_scf.f90:924",
    ),
    CallbackSpec(
        name="python_func",
        ctypes_type=ReconstructMxCb,
        register_symbol="aimspy_register_python_callback",
        register_arg_count=2,
        trigger_stage="pre_scf",
        fortran_module="initialize_scf.f90:915",
    ),
]

# Build a quick name->spec lookup
SPECS_BY_NAME: dict[str, CallbackSpec] = {s.name: s for s in CALLBACK_SPECS}


def get_spec(name: str) -> CallbackSpec:
    """Look up a ``CallbackSpec`` by ``name``."""
    spec = SPECS_BY_NAME.get(name)
    if spec is None:
        available = list(SPECS_BY_NAME.keys())
        from .._exceptions import AimspyCallbackError

        raise AimspyCallbackError(
            f"unknown callback spec {name!r}; available: {available}"
        )
    return spec
