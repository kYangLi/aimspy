"""Private — argtypes/restype prototypes + BindingLib wrapper around CDLL.

Adding a new C function = adding one entry to `_PROTOTYPES` dict.
"""

from __future__ import annotations

import logging
from ctypes import CDLL, c_int, c_double, c_void_p, POINTER

from .ctypes_types import AimspyInfoC
from .callback_types import (
    GetDescrCb,
    ExportOvlpCb,
    ExportH0Cb,
    ModifyH0Cb,
    ReconstructMxCb,
)

_log = logging.getLogger(__name__)

# =========================================================================
# C-function prototype registry (extend here for new C bindings)
# =========================================================================
_PROTOTYPES: dict[str, tuple[list, object]] = {
    # ---- lifecycle ----
    "aimspy_init": ([c_int, c_void_p], None),
    "aimspy_run": ([], None),
    "aimspy_finalize": ([], None),
    # ---- info snapshot ----
    "aimspy_get_info": ([], POINTER(AimspyInfoC)),
    # ---- real-space matrix pointers ----
    "c_rs_hamiltonian": ([], POINTER(c_double)),
    "c_rs_overlap": ([], POINTER(c_double)),
    # ---- energy ----
    "aimspy_energy": ([], c_double),
    # ---- forces pointer ----
    "aimspy_forces": ([], POINTER(c_double)),
    # ---- callback registration ----
    "aimspy_register_get_descr_callback": ([GetDescrCb, c_void_p], None),
    "aimspy_register_export_ovlp_callback": ([ExportOvlpCb, c_void_p], None),
    "aimspy_register_export_h0_callback": ([ExportH0Cb, c_void_p], None),
    "aimspy_register_modify_h0_callback": ([ModifyH0Cb, c_void_p, c_void_p], None),
    "aimspy_register_python_callback": ([ReconstructMxCb, c_void_p], None),
}


def setup_prototypes(lib: CDLL) -> None:
    """Declare argtypes/restype on all known C functions in *lib*.

    Unknown symbols (missing in older libaims builds) are silently skipped.
    """
    for name, (argtypes, restype) in _PROTOTYPES.items():
        if not hasattr(lib, name):
            _log.debug("C function %r not found in lib — skipping prototype", name)
            continue
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = restype


class BindingLib:
    """Type-checked wrapper around a loaded CDLL.

    Provides:
      - `has(name)` to probe optional C functions
      - `__getattr__` with clear error messages for missing symbols
      - Remembers which symbols were detected during `setup_prototypes`

    All aimspy modules above the `_binding` layer only touch `BindingLib`,
    never the raw `CDLL`.

    Examples
    --------
    >>> lib = BindingLib(cdll)
    >>> lib.aimspy_init(...)
    >>> lib.has("c_rs_overlap")
    True
    """

    __slots__ = ("_cdll", "_available")

    def __init__(self, cdll: CDLL):
        self._cdll = cdll
        self._available: set[str] = set()

        setup_prototypes(cdll)
        for name in _PROTOTYPES:
            if hasattr(cdll, name):
                self._available.add(name)

    def has(self, name: str) -> bool:
        """Return whether the C function *name* is available."""
        return name in self._available

    def __getattr__(self, name: str):
        if name in self._available:
            return getattr(self._cdll, name)
        # Allow `has`, `_cdll`, `_available` to work (defined in __init__/class)
        if name in ("has", "_cdll", "_available"):
            return object.__getattribute__(self, name)
        from .._exceptions import AimspyBindingError

        raise AimspyBindingError(
            f"C function {name!r} not available in loaded libaims "
            f"(available: {len(self._available)} funcs)"
        )
