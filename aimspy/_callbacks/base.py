"""Private — core callback types: CallbackSpec, CallbackManager, DefaultCallback ABC.

This module is the extensibility hub of aimspy.  Adding a new callback
to the system requires touching 5 well-defined places (listed in the
project README):

  1.  Fortran side (via patch): type + register + trigger subroutine
  2.  ``_binding/callback_types.py``:  one CFUNCTYPE declaration
  3.  ``_binding/prototypes.py``:  one _PROTOTYPES entry
  4.  ``_callbacks/registry.py``:  one CallbackSpec entry
  5.  ``_callbacks/defaults.py``:  one DefaultCallback subclass

The ``Calculator`` class does NOT need to be touched — properties are
auto-generated from the spec list.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from ctypes import CFUNCTYPE, c_void_p, cast, py_object
from dataclasses import dataclass, field, fields
from typing import (
    Any, Callable, ClassVar, Literal, Optional, Tuple, Type, Union,
)

import numpy as np

_log = logging.getLogger(__name__)


# =========================================================================
# CallbackSpec — one entry per callback type
# =========================================================================
@dataclass(frozen=True)
class CallbackSpec:
    """Complete description of one callback type.

    Adding a callback = adding one CallbackSpec to ``CALLBACK_SPECS``
    in ``registry.py`` and ensuring the other 4 places are wired.

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
    default_impl : callable or None
        Default callback implementation.  If set the ``Calculator`` auto‑generated
        property will use it.
    property_name : str or None
        If set, a ``@property`` is auto‑attached to ``Calculator`` with this name.
        The setter registers the callback via ``register_default``.
    property_doc : str or None
        Docstring for the auto‑generated property.
    trigger_stage : str
        Human-readable description of when the callback fires (for docs).
    fortran_module : str
        Source location of the trigger point (for docs / debugging).
    """

    name: str
    ctypes_type: Type[CFUNCTYPE]
    register_symbol: str
    register_arg_count: int = 2
    default_impl: Optional[Callable] = None
    property_name: Optional[str] = None
    property_doc: Optional[str] = None
    trigger_stage: str = ""
    fortran_module: str = ""
    raw_value_key: Optional[str] = None  # if set, auto-setter wraps value as {key: value}


# =========================================================================
# DefaultCallback ABC — base class for default implementations
# =========================================================================
class DefaultCallback(ABC):
    """Base class for default callback implementations.

    Each subclass corresponds to one ``CallbackSpec``.  The subclass
    implements ``__call__`` with a Python-friendly signature (no ctypes
    types exposed to subclasses — the auto‑wrapper handles translation).
    """

    # Override in subclass:
    spec: ClassVar[CallbackSpec]

    @classmethod
    def spec_name(cls) -> str:
        return cls.spec.name

    @abstractmethod
    def __call__(self, aux: dict, *args: Any) -> None:
        """Invoke the callback.

        Parameters
        ----------
        aux : dict
            The aux dict passed through from the registration call.
        *args
            Python objects.  The first arg is always the *aux* dict after
            unpacking.  Additional args depend on the specific callback
            (the framework auto‑converts ctypes pointers to numpy views).
        """
        ...

    @classmethod
    def make_aux(cls, **kwargs: Any) -> dict:
        """Build a default *aux* dict for this callback type."""
        return dict(kwargs)


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
        # spec.name -> py_object wrapper — must survive to prevent GC
        # of the buffer used by c_void_p.from_buffer(py_object(aux))
        self._pyobjs: dict[str, Any] = {}
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

        *fn* can be any Python callable — it will be auto‑wrapped into
        the appropriate CFUNCTYPE.  The original *fn* and *aux* are both
        kept alive as long as this ``CallbackManager`` exists (and GC‑safe).

        Parameters
        ----------
        spec : CallbackSpec
        fn : callable
            Python‑side callback function.  The framework auto‑detects
            whether *fn* is a subclass of ``DefaultCallback`` or a raw
            Python function and wraps accordingly.
        aux : any
            Arbitrary Python object passed through to the callback's
            ``aux`` parameter.  Can be ``None``.
        extra_ptr : int or None
            Extra c‑pointer value for 3‑arg register functions
            (e.g. the ``input_mx_ptr`` of ``modify_h0``).
        """
        ctypes_fn = self._build_ctypes_wrapper(spec, fn, aux)
        wrapped = spec.ctypes_type(ctypes_fn)
        self._wrapped[spec.name] = (wrapped, fn)  # hold both for GC
        self._auxs[spec.name] = aux

        register_fn = getattr(self._binding, spec.register_symbol)
        if aux is not None:
            po = py_object(aux)
            self._pyobjs[spec.name] = po  # keep alive to prevent GC of buffer
            aux_ptr = c_void_p.from_buffer(po)
        else:
            aux_ptr = c_void_p(None)

        if spec.register_arg_count == 2:
            register_fn(wrapped, aux_ptr)
        elif spec.register_arg_count == 3:
            extra = c_void_p(extra_ptr) if extra_ptr is not None else c_void_p(None)
            register_fn(wrapped, aux_ptr, extra)
        else:
            raise ValueError(f"unsupported register_arg_count={spec.register_arg_count}")

    def register_default(self, spec: CallbackSpec,
                         aux: Any = None,
                         extra_ptr: Optional[int] = None) -> None:
        """Same as ``register`` but uses ``spec.default_impl`` as the fn.

        If ``spec.default_impl`` is a ``DefaultCallback`` subclass,
        the per‑instance aux is merged with ``DefaultCallback.make_aux()``.
        """
        if spec.default_impl is None:
            from .._exceptions import AimspyCallbackError
            raise AimspyCallbackError(
                f"callback {spec.name!r} has no default implementation"
            )
        impl = spec.default_impl
        if isinstance(impl, type) and issubclass(impl, DefaultCallback):
            # Instantiate the DefaultCallback subclass
            impl_instance = impl()
            merged_aux = impl.make_aux(**(aux if isinstance(aux, dict) else {}))
            self.register(spec, impl_instance, merged_aux, extra_ptr)
        else:
            self.register(spec, impl, aux, extra_ptr)

    def is_registered(self, spec_name: str) -> bool:
        return spec_name in self._wrapped

    # ------------------------------------------------------------------
    # Internal: auto‑wrap user Python fn → ctypes wrapper
    # ------------------------------------------------------------------
    def _build_ctypes_wrapper(
        self,
        spec: CallbackSpec,
        fn: Callable,
        aux: Any,
    ) -> Callable:
        """Return a ctypes‑friendly wrapper that:
        1.  unpacks the ``aux`` c_void_p back to the Python *aux* object
        2.  converts pointer args to numpy ndarray views where applicable
        3.  calls the original *fn* with Python‑friendly arguments
        """
        mgr = self  # keep the closure binding to the manager for error recording

        if spec.name == 'get_descr':
            def wrapper(aux_ptr: int, descr_ptr: int) -> None:
                _aux = _unpack_aux(aux_ptr, aux) if aux is not None else {}
                from ..data import CsrMatrixDescriptor
                from .._binding.ctypes_types import CsrMxDescrC
                from ctypes import cast, c_void_p, POINTER
                try:
                    ptr = cast(c_void_p(descr_ptr), POINTER(CsrMxDescrC))
                    _aux['descr'] = CsrMatrixDescriptor._from_c_struct(ptr.contents)
                    fn(_aux)
                except Exception as exc:
                    _record_callback_error(mgr, spec.name, exc)

        elif spec.name == 'export_h0':
            def wrapper(aux_ptr: int, h0_ptr: int, n_ham: int, n_spin: int) -> None:
                _aux = _unpack_aux(aux_ptr, aux) if aux is not None else {}
                try:
                    h0 = _ptr_to_view(h0_ptr, (int(n_spin), int(n_ham)))
                    fn(_aux, h0, int(n_ham), int(n_spin))
                except Exception as exc:
                    _record_callback_error(mgr, spec.name, exc)

        elif spec.name == 'modify_h0':
            def wrapper(aux_ptr: int, input_mx_ptr: int,
                        h0_ptr: int, n_ham: int, n_spin: int) -> None:
                _aux = _unpack_aux(aux_ptr, aux) if aux is not None else {}
                try:
                    h0 = _ptr_to_view(h0_ptr, (int(n_spin), int(n_ham)))
                    fn(_aux, h0, int(n_ham), int(n_spin))
                except Exception as exc:
                    _record_callback_error(mgr, spec.name, exc)

        elif spec.name == 'python_func':
            def wrapper(aux_ptr: int) -> None:
                _aux = _unpack_aux(aux_ptr, aux) if aux is not None else {}
                try:
                    fn(_aux)
                except Exception as exc:
                    _record_callback_error(mgr, spec.name, exc)

        else:
            # Generic fallback: pass unpacked aux only
            def wrapper(aux_ptr: int, *rest) -> None:
                _aux = _unpack_aux(aux_ptr, aux) if aux is not None else {}
                try:
                    fn(_aux, *rest)
                except Exception as exc:
                    _record_callback_error(mgr, spec.name, exc)

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
    """Create a read‑write numpy view of a C array at *ptr* with given *shape*.

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


def _record_callback_error(mgr: "CallbackManager", name: str, exc: Exception) -> None:
    """Report a callback failure to stderr, logger, and the manager's error list."""
    import sys, traceback
    tb_str = traceback.format_exc()
    print(f"[aimspy] {name} callback FAILED: {exc!r}", file=sys.stderr, flush=True)
    traceback.print_exc(file=sys.stderr)
    _log.error("%s callback raised %s\n%s", name, exc, tb_str)
    if hasattr(mgr, '_errors') and mgr._errors is not None:
        mgr._errors.append((name, exc, tb_str))
