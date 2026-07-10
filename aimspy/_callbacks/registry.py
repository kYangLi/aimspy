"""Private — callback registry: the authoritative list of all callback types.

This is **the** central catalogue.  Adding a new callback = adding one
``CallbackSpec`` entry here, plus changes in the 4 other wired places.
"""
from __future__ import annotations

from .._binding.callback_types import (
    GetDescrCb, ExportH0Cb, ModifyH0Cb, ReconstructMxCb,
)
from .base import CallbackSpec, DefaultCallback
from .defaults import (
    GetDescrDefault, ExportH0Default, ModifyH0Default, PythonFuncDefault,
)

# =========================================================================
# Authoritative list — one entry per callback type
# =========================================================================
CALLBACK_SPECS: list[CallbackSpec] = [
    CallbackSpec(
        name='get_descr',
        ctypes_type=GetDescrCb,
        register_symbol='aimspy_register_get_descr_callback',
        register_arg_count=2,
        default_impl=GetDescrDefault,
        property_name=None,  # implicit: needed for deeph warmstart
        trigger_stage='pre_scf',
        fortran_module='initialize_scf.f90:912',
    ),

    CallbackSpec(
        name='export_h0',
        ctypes_type=ExportH0Cb,
        register_symbol='aimspy_register_export_h0_callback',
        register_arg_count=2,
        default_impl=ExportH0Default,
        property_name=None,
        property_doc=None,
        trigger_stage='pre_scf',
        fortran_module='initialize_scf.f90:913',
    ),

    CallbackSpec(
        name='modify_h0',
        ctypes_type=ModifyH0Cb,
        register_symbol='aimspy_register_modify_h0_callback',
        register_arg_count=3,
        default_impl=ModifyH0Default,
        property_name=None,
        property_doc=None,
        trigger_stage='pre_scf',
        fortran_module='initialize_scf.f90:923',
    ),

    CallbackSpec(
        name='python_func',
        ctypes_type=ReconstructMxCb,
        register_symbol='aimspy_register_python_callback',
        register_arg_count=2,
        default_impl=PythonFuncDefault,
        property_name=None,
        property_doc=None,
        trigger_stage='pre_scf',
        fortran_module='initialize_scf.f90:914',
    ),
]

# Build a quick name‑>spec lookup
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
