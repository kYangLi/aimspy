"""Private — core callback types: CallbackSpec + CallbackManager.

Adding a new callback to the system requires touching the following
well-defined places:

  1.  Fortran side (via patch): type + register + trigger subroutine
  2.  ``_binding/callback_types.py``:  one CFUNCTYPE declaration
  3.  ``_binding/prototypes.py``:  one _PROTOTYPES entry
  4.  ``_callbacks/registry.py``:  one CallbackSpec entry
  5.  ``_callbacks/base.py``:  one branch in ``_build_ctypes_wrapper``

The ``Calculator._wire_callbacks`` method registers its own inline
closures for each needed callback, bypassing any "default
implementation" registry. Users may also register custom callables
via :meth:`aimspy.Calculator.register_callback`.
"""

from __future__ import annotations

import logging
from ctypes import CFUNCTYPE, c_void_p, cast, py_object
from dataclasses import dataclass
from typing import Any, Callable, Optional, Tuple, Type

import numpy as np

_log = logging.getLogger(__name__)


# =========================================================================
# CallbackSpec — one entry per callback type
# =========================================================================
@dataclass(frozen=True)
class CallbackSpec:
    """Complete description of one callback type.

    Adding a callback = adding one CallbackSpec to ``CALLBACK_SPECS``
    in ``registry.py`` and ensuring the other 3 places are wired.

    Parameters
    ----------
    name : str
        Internal id, e.g. ``'modify_h0'``.
    ctypes_type : CFUNCTYPE subclass
        The ctypes function prototype.
    register_symbol : str
        Name of the C function that registers this callback (e.g.
        ``'aimspy_register_modify_h0_callback'``).
    register_arg_count : int
        Number of args the C register function takes *after* the CFUNCTYPE
        wrapper.  2 = (cb, aux); 3 = (cb, aux, extra_c_ptr).
    trigger_stage : str
        Human-readable description of when the callback fires (for docs).
    fortran_module : str
        Source location of the trigger point (for docs / debugging).
    """

    name: str
    ctypes_type: Type[CFUNCTYPE]
    register_symbol: str
    register_arg_count: int = 2
    trigger_stage: str = ""
    fortran_module: str = ""


# =========================================================================
# CallbackManager — per-Calculator callback lifecycle
# =========================================================================
class CallbackManager:
    """Per-``Calculator`` instance: holds CFUNCTYPE wrappers + aux objects,
    prevents garbage collection, and exposes register/unregister.

    All ``register`` methods accept **plain Python callables** — the
    manager transparently generates a ctypes wrapper that unpacks the
    ``aux`` pointer and converts pointer args to numpy views.
    """

    def __init__(self, binding: Any) -> None:  # BindingLib
        self._binding = binding
        # spec.name -> tuple[CFUNCTYPE_wrapper, original_py_callable]
        self._wrapped: dict[str, Tuple[Any, Any]] = {}
        # spec.name -> aux object (the Python object whose id we pass)
        self._auxs: dict[str, Any] = {}
        # Records callback failures: list of (spec_name, exception, traceback_str)
        self._errors: list[tuple[str, Exception, str]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def register(
        self,
        spec: CallbackSpec,
        fn: Callable,
        aux: Any = None,
        extra_ptr: Optional[int] = None,
    ) -> None:
        """Register *fn* for *spec*.

        *fn* can be any Python callable — it will be auto-wrapped into
        the appropriate CFUNCTYPE.  The original *fn* and *aux* are both
        kept alive as long as this ``CallbackManager`` exists (and GC-safe).

        Parameters
        ----------
        spec : CallbackSpec
        fn : callable
            Python-side callback function.  The framework auto-detects
            *spec.name* and converts pointer args to numpy views where
            applicable.
        aux : any
            Arbitrary Python object passed through to the callback's
            ``aux`` parameter.  Can be ``None``.
        extra_ptr : int or None
            Extra c-pointer value for 3-arg register functions
            (e.g. the ``input_mx_ptr`` of ``modify_h0``).
        """
        ctypes_fn = self._build_ctypes_wrapper(spec, fn, aux)
        wrapped = spec.ctypes_type(ctypes_fn)
        self._wrapped[spec.name] = (wrapped, fn)  # hold both for GC
        self._auxs[spec.name] = aux

        register_fn = getattr(self._binding, spec.register_symbol)
        if aux is not None:
            # Use id(aux) as the pointer value (documented CPython idiom).
            # _auxs already holds a strong reference to aux, keeping it alive
            # as long as this CallbackManager exists.
            aux_ptr = c_void_p(id(aux))
        else:
            aux_ptr = c_void_p(None)

        if spec.register_arg_count == 2:
            register_fn(wrapped, aux_ptr)
        elif spec.register_arg_count == 3:
            extra = c_void_p(extra_ptr) if extra_ptr is not None else c_void_p(None)
            register_fn(wrapped, aux_ptr, extra)
        else:
            raise ValueError(
                f"unsupported register_arg_count={spec.register_arg_count}"
            )

    def is_registered(self, spec_name: str) -> bool:
        return spec_name in self._wrapped

    def clear(self) -> None:
        """Release all held references: CFUNCTYPE wrappers, aux objects,
        and error records.

        Must be called before dropping the CallbackManager reference to
        ensure immediate refcount-based reclamation of all callback
        state (including user aux dicts that may hold large matrices).
        After clear(), the CallbackManager is empty and must not be used
        to fire callbacks.
        """
        self._wrapped.clear()
        self._auxs.clear()
        self._errors.clear()

    # ------------------------------------------------------------------
    # Internal: auto-wrap user Python fn → ctypes wrapper
    # ------------------------------------------------------------------
    def _build_ctypes_wrapper(
        self,
        spec: CallbackSpec,
        fn: Callable,
        aux: Any,
    ) -> Callable:
        """Return a ctypes-friendly wrapper that:
        1.  unpacks the ``aux`` c_void_p back to the Python *aux* object
        2.  converts pointer args to numpy ndarray views where applicable
        3.  calls the original *fn* with Python-friendly arguments
        """
        # Capture only the error list (a leaf object), NOT the manager itself.
        # Capturing `self` would create a reference cycle
        # (CallbackManager._wrapped -> wrapper_closure -> mgr -> CallbackManager)
        # that prevents refcount-based reclamation and forces reliance on the
        # cyclic garbage collector. Capturing the list directly breaks the
        # cycle while preserving the error-recording capability.
        errors = self._errors

        if spec.name == "get_descr":

            def wrapper(aux_ptr: int, descr_ptr: int) -> None:
                _aux = _unpack_aux(aux_ptr, aux) if aux is not None else {}
                from ..data import CsrMatrixDescriptor
                from .._binding.ctypes_types import CsrMxDescrC
                from ctypes import cast, c_void_p, POINTER

                try:
                    ptr = cast(c_void_p(descr_ptr), POINTER(CsrMxDescrC))
                    _aux["descr"] = CsrMatrixDescriptor._from_c_struct(ptr.contents)
                    fn(_aux)
                except Exception as exc:
                    _record_callback_error(errors, spec.name, exc)

        elif spec.name == "export_ovlp":

            def wrapper(aux_ptr: int, ovlp_ptr: int, n_ham: int, n_spin: int) -> None:
                _aux = _unpack_aux(aux_ptr, aux) if aux is not None else {}
                try:
                    ovlp = _ptr_to_view(ovlp_ptr, (int(n_spin), int(n_ham)))
                    ovlp.flags.writeable = False  # Fortran intent(in)
                    fn(_aux, ovlp, int(n_ham), int(n_spin))
                except Exception as exc:
                    _record_callback_error(errors, spec.name, exc)

        elif spec.name == "export_h0":

            def wrapper(aux_ptr: int, h0_ptr: int, n_ham: int, n_spin: int) -> None:
                _aux = _unpack_aux(aux_ptr, aux) if aux is not None else {}
                try:
                    h0 = _ptr_to_view(h0_ptr, (int(n_spin), int(n_ham)))
                    h0.flags.writeable = False  # Fortran intent(in)
                    fn(_aux, h0, int(n_ham), int(n_spin))
                except Exception as exc:
                    _record_callback_error(errors, spec.name, exc)

        elif spec.name == "modify_h0":

            def wrapper(
                aux_ptr: int, input_mx_ptr: int, h0_ptr: int, n_ham: int, n_spin: int
            ) -> None:
                _aux = _unpack_aux(aux_ptr, aux) if aux is not None else {}
                try:
                    h0 = _ptr_to_view(h0_ptr, (int(n_spin), int(n_ham)))
                    fn(_aux, h0, int(n_ham), int(n_spin))
                except Exception as exc:
                    _record_callback_error(errors, spec.name, exc)

        elif spec.name == "python_func":

            def wrapper(aux_ptr: int) -> None:
                _aux = _unpack_aux(aux_ptr, aux) if aux is not None else {}
                try:
                    fn(_aux)
                except Exception as exc:
                    _record_callback_error(errors, spec.name, exc)

        elif spec.name == "export_dHde":

            def wrapper(
                aux_ptr: int,
                mx_ptr: int,
                n_ham: int,
                n_dir: int,
                n_spin: int,
                j_coord: int,
            ) -> None:
                _aux = _unpack_aux(aux_ptr, aux) if aux is not None else {}
                try:
                    # Fortran (n_dir, 1, n_ham, n_spin) column-major flat
                    # → C-order (n_spin, n_ham, n_dir) — copy for safety
                    # (DFPT_first_order_H_sparse may be reallocated by Fortran
                    # after the callback returns; a view would dangle)
                    size = int(n_ham) * int(n_dir) * int(n_spin)
                    flat = np.ctypeslib.as_array(mx_ptr, shape=(size,)).copy()
                    dHde = flat.reshape((int(n_spin), int(n_ham), int(n_dir)))
                    fn(_aux, dHde, int(n_ham), int(n_dir), int(n_spin), int(j_coord))
                except Exception as exc:
                    _record_callback_error(errors, spec.name, exc)

        elif spec.name == "modify_dHde":

            def wrapper(
                aux_ptr: int,
                input_mx_ptr: int,
                mx_ptr: int,
                n_ham: int,
                n_dir: int,
                n_spin: int,
                j_coord: int,
            ) -> None:
                _aux = _unpack_aux(aux_ptr, aux) if aux is not None else {}
                try:
                    # mx_ptr is a POINTER(c_double) into Fortran intent(inout)
                    # array DFPT_first_order_H_sparse.  We pass it as an integer
                    # address so the callback can memmove into it directly.
                    from ctypes import addressof

                    addr = addressof(mx_ptr.contents) if mx_ptr else 0
                    fn(_aux, addr, int(n_ham), int(n_dir), int(n_spin), int(j_coord))
                except Exception as exc:
                    _record_callback_error(errors, spec.name, exc)

        elif spec.name == "export_grid_data":

            def wrapper(
                aux_ptr: int,
                descr_ptr: int,
                rho_ptr,
                vks_ptr,
                vks0_ptr,
                vh_ptr,
                vh0_ptr,
                rho0_ptr,
            ) -> None:
                _aux = _unpack_aux(aux_ptr, aux) if aux is not None else {}
                try:
                    from ..grid_data import GridData

                    # GridData._from_c copies every array out of the Fortran
                    # buffers (which may be reused / freed after return).
                    gd = GridData._from_c(
                        descr_ptr,
                        rho_ptr,
                        vks_ptr,
                        vks0_ptr,
                        vh_ptr,
                        vh0_ptr,
                        rho0_ptr,
                    )
                    # Fill structure fields from the live runtime structure
                    # (in-memory AimspyInfo via AimspyStructure — NOT from
                    # input files), keeping the dataset self-consistent with
                    # the in-memory aims state.
                    structure = _aux.get("structure")
                    if structure is not None:
                        gd.atom_coords = structure.atom_coords
                        gd.atom_symbols = structure.atom_symbols
                        gd.lattice = structure.lattice
                    _aux["grid_data"] = gd
                    fn(_aux, gd)
                except Exception as exc:
                    _record_callback_error(errors, spec.name, exc)

        elif spec.name == "export_basis_data":

            def wrapper(
                aux_ptr: int,
                descr_ptr: int,
                wave_spl_ptr,
                kinetic_spl_ptr,
                deriv_spl_ptr,
            ) -> None:
                _aux = _unpack_aux(aux_ptr, aux) if aux is not None else {}
                try:
                    from ..basis_data import BasisData

                    # BasisData._from_c copies every array out of the Fortran
                    # buffers (which are freed by aimspy_finalize after the
                    # Calculator is closed).
                    bd = BasisData._from_c(
                        descr_ptr,
                        wave_spl_ptr,
                        kinetic_spl_ptr,
                        deriv_spl_ptr,
                    )
                    _aux["basis_data"] = bd
                    fn(_aux, bd)
                except Exception as exc:
                    _record_callback_error(errors, spec.name, exc)

        else:
            # Generic fallback: pass unpacked aux only
            def wrapper(aux_ptr: int, *rest) -> None:
                _aux = _unpack_aux(aux_ptr, aux) if aux is not None else {}
                try:
                    fn(_aux, *rest)
                except Exception as exc:
                    _record_callback_error(errors, spec.name, exc)

        return wrapper


# =========================================================================
# Internal helpers
# =========================================================================
def _unpack_aux(aux_ptr: int, default: Any = None) -> Any:
    """Convert c_void_p *aux_ptr* back to its Python object.

    If the pointer is NULL, returns *default*.
    """
    if aux_ptr is None or aux_ptr == 0:
        return default
    try:
        return cast(c_void_p(aux_ptr), py_object).value
    except Exception:
        return default


def _ptr_to_view(ptr, shape: Tuple[int, ...]) -> np.ndarray:
    """Create a read-write numpy view of a C array at *ptr* with given *shape*.

    *ptr* can be a raw integer address, c_void_p, or ctypes POINTER.
    Returns a mutable VIEW (not a copy).
    """
    from ctypes import cast, c_void_p, POINTER, c_double

    n = 1
    for d in shape:
        n *= d
    # Try direct as_array first (works for ctypes pointers)
    try:
        return np.ctypeslib.as_array(ptr, shape=(n,)).reshape(shape)
    except Exception:
        pass
    # Fallback: cast from integer address
    try:
        return np.ctypeslib.as_array(
            cast(c_void_p(int(ptr)), POINTER(c_double)),
            shape=(n,),
        ).reshape(shape)
    except Exception:
        pass
    raise TypeError(f"Cannot create ndarray view from ptr type={type(ptr)}")


def _record_callback_error(errors: list, name: str, exc: Exception) -> None:
    """Report a callback failure to stderr, logger, and the error list.

    Accepts the error list directly (rather than the CallbackManager) so
    that wrapper closures can capture a leaf object instead of the
    manager itself, avoiding a reference cycle.
    """
    import sys
    import traceback

    tb_str = traceback.format_exc()
    print(f"[aimspy] {name} callback FAILED: {exc!r}", file=sys.stderr, flush=True)
    traceback.print_exc(file=sys.stderr)
    _log.error("%s callback raised %s\n%s", name, exc, tb_str)
    if errors is not None:
        errors.append((name, exc, tb_str))
