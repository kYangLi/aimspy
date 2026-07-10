"""Public — ``Calculator``, the primary user-facing class.

Usage (context manager, recommended)::

    from aimspy import Calculator, CalculatorConfig

    with Calculator(CalculatorConfig(
        lib_path="/path/to/libaims.so",
        work_dir="./MoS2",
    )) as calc:
        calc.run()
        H = calc.hamiltonian     # AimspyMatrix
        E = calc.energy

Warmstart with DeepH data::

    from aimspy.interface.deeph import DeepHData, DeepHSource
    source = DeepHSource(DeepHData.from_directory("deeph_warm/"))
    calc.modify_h0(source=source)
    calc.init(comm)
    calc.run()
"""
from __future__ import annotations

import logging
from ctypes import memmove, sizeof, c_double
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import numpy as np

from ._system import chdir_cm
from ._exceptions import (
    AimspyConfigError, AimspyStateError,
)
from ._binding.libloader import load_aims_lib
from ._binding.prototypes import BindingLib
from ._callbacks.base import CallbackManager
from ._callbacks.registry import SPECS_BY_NAME
from .data import AimspyInfo, CsrMatrixDescriptor
from .structure import AimspyStructure
from .info import load_info
from .matrix import (
    get_rs_hamiltonian, get_rs_overlap, AimspyMatrix,
)

_log = logging.getLogger(__name__)


# =============================================================================
# Modify definition  (private — used internally by Calculator._set_modify)
# =============================================================================
class _ModifyDef:
    """Internal holder for modify_h0 strategy + source."""
    __slots__ = ("strategy", "external_source", "factor", "custom_fn")

    def __init__(self, strategy="replace", external_source=None,
                 factor=1.0, custom_fn=None):
        self.strategy = strategy
        self.external_source = external_source
        self.factor = factor
        self.custom_fn = custom_fn


# =============================================================================
# CalculatorConfig
# =============================================================================
@dataclass
class CalculatorConfig:
    lib_path: Path
    work_dir: Path = Path("./aimspy_run")
    logfile: Path = Path("aims.out")
    control_path: Optional[Path] = None
    geometry_path: Optional[Path] = None
    initializer: Optional[Callable[["Calculator"], None]] = None
    log_level: str = "INFO"


class CalcState(Enum):
    UNINIT = "uninit"
    INITED = "inited"
    RUNNING = "running"
    DONE = "done"
    FINALIZED = "finalized"


# =============================================================================
# Calculator
# =============================================================================
class Calculator:
    """In-memory interface to FHI-aims via ctypes."""

    def __init__(
        self,
        config: Optional[CalculatorConfig] = None,
        /,
        **kwargs,
    ) -> None:
        if config is None:
            config = CalculatorConfig(**kwargs)
        elif kwargs:
            raise AimspyConfigError(
                "Provide either CalculatorConfig or keyword args, not both"
            )
        self._cfg = config
        self._state = CalcState.UNINIT
        self._log = logging.getLogger("aimspy")

        self._binding: Optional[BindingLib] = None
        self._cb_mgr: Optional[CallbackManager] = None
        self._info: Optional[AimspyInfo] = None
        self._structure: Optional[AimspyStructure] = None
        self._csr_descr: Optional[CsrMatrixDescriptor] = None

        self._comm: Any = None
        self._modify_def: Optional[_ModifyDef] = None
        self._capture_h0: bool = False
        self._modify_aux: Optional[dict] = None

    # ==================================================================
    # Context manager
    # ==================================================================
    def __enter__(self) -> "Calculator":
        return self.init()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ==================================================================
    # Lifecycle
    # ==================================================================
    def init(self, comm: Any = None) -> "Calculator":
        if self._state != CalcState.UNINIT:
            raise AimspyStateError(
                f"Cannot init in state {self._state.value}; expected uninit"
            )
        if comm is None:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
        self._comm = comm
        rank = comm.rank

        logging.basicConfig(level=logging.WARNING)
        self._log.setLevel(getattr(logging, self._cfg.log_level.upper(),
                                   logging.INFO))
        self._log.info("rank %d: preparing work_dir %s", rank, self._cfg.work_dir)

        work = self._cfg.work_dir
        work.mkdir(parents=True, exist_ok=True)

        with chdir_cm(work):
            self._copy_inputs()
            if rank == 0 and self._cfg.initializer:
                self._cfg.initializer(self)
            comm.Barrier()

            self._log.info("rank %d: loading lib %s", rank, self._cfg.lib_path)
            cdll = load_aims_lib(self._cfg.lib_path)
            self._binding = BindingLib(cdll)
            self._cb_mgr = CallbackManager(self._binding)

            self._log.info("rank %d: aimspy_init", rank)
            self._binding.aimspy_init(
                _py2f(comm),
                str(self._cfg.logfile).encode('UTF-8'),
            )
            comm.Barrier()

            self._info = load_info(self._binding)
            self._structure = AimspyStructure.from_info(self._info)

            # Wire callbacks now that cb_mgr exists
            self._ensure_callbacks_wired()

        self._state = CalcState.INITED
        self._log.info("rank %d: init done. n_basis=%d",
                       rank, self._info.n_basis)
        return self

    def run(self) -> None:
        self._state_guard(CalcState.INITED, "run")
        self._state = CalcState.RUNNING
        self._log.info("rank %d: aimspy_run ...", self._comm.rank)
        with chdir_cm(self._cfg.work_dir):
            self._binding.aimspy_run()
        self._state = CalcState.DONE
        self._log.info("rank %d: run done", self._comm.rank)
        self._check_callback_errors()

    def close(self) -> None:
        if self._state in (CalcState.UNINIT, CalcState.FINALIZED):
            return
        if self._state == CalcState.RUNNING:
            raise AimspyStateError(
                "Cannot close() in RUNNING state; wait for SCF to complete "
                "or kill the process."
            )
        self._log.info("rank %d: aimspy_finalize", self._comm.rank)
        with chdir_cm(self._cfg.work_dir):
            self._binding.aimspy_finalize()
        self._state = CalcState.FINALIZED
        self._binding = None
        self._cb_mgr = None

    # ==================================================================
    # Properties: query
    # ==================================================================
    @property
    def info(self) -> AimspyInfo:
        if self._info is None:
            self._state_guard(CalcState.INITED, "access info")
            self._info = load_info(self._binding)
        return self._info

    @property
    def structure(self) -> Optional[AimspyStructure]:
        if self._structure is None and self._info is not None:
            self._structure = AimspyStructure.from_info(self._info)
        return self._structure

    @property
    def energy(self) -> float:
        self._state_guard(CalcState.DONE, "read energy",
                          allowed={CalcState.RUNNING, CalcState.DONE})
        return self._binding.aimspy_energy()

    @property
    def rs_hamiltonian(self) -> np.ndarray:
        self._state_guard(CalcState.DONE, "read rs_hamiltonian")
        return get_rs_hamiltonian(self._binding,
                                  self.info.n_spin, self.info.n_ham_size)

    @property
    def rs_overlap(self) -> np.ndarray:
        self._state_guard(CalcState.INITED, "read rs_overlap",
                          allowed={CalcState.INITED, CalcState.RUNNING, CalcState.DONE})
        return get_rs_overlap(self._binding, self.info.n_ham_size)

    @property
    def csr_descr(self) -> Optional[CsrMatrixDescriptor]:
        if self._csr_descr is not None:
            return self._csr_descr
        if self._modify_aux is not None:
            return self._modify_aux.get('csr_descr')
        return None

    @property
    def hamiltonian(self) -> AimspyMatrix:
        self._state_guard(CalcState.DONE, "read hamiltonian")
        csr = self.csr_descr
        if csr is None:
            raise AimspyStateError(
                "csr_descr not available; call modify_h0() "
                "or set capture_h0=True before run()."
            )
        return AimspyMatrix.from_aims_csr(self.rs_hamiltonian, csr,
                                          self.structure)

    @property
    def overlap(self) -> AimspyMatrix:
        self._state_guard(CalcState.DONE, "read overlap")
        csr = self.csr_descr
        if csr is None:
            raise AimspyStateError(
                "csr_descr not available; call modify_h0() "
                "or set capture_h0=True before run()."
            )
        ovlp = self.rs_overlap.reshape(1, -1)
        return AimspyMatrix.from_aims_csr(ovlp, csr, self.structure)

    # ==================================================================
    # capture_h0  (export initial H0 as AimspyMatrix)
    # ==================================================================
    @property
    def capture_h0(self) -> bool:
        return self._capture_h0

    @capture_h0.setter
    def capture_h0(self, value: bool) -> None:
        self._capture_h0 = bool(value)
        if self._capture_h0:
            self._ensure_callbacks_wired()

    @property
    def initial_hamiltonian(self) -> Optional[AimspyMatrix]:
        if self._modify_aux is not None:
            return self._modify_aux.get('initial_hamiltonian')
        return None

    # ==================================================================
    # modify_h0  (unified modify entry — a METHOD, not a property)
    # ==================================================================
    def modify_h0(self, *,
                  source: Any = None,
                  strategy: str = "replace",
                  factor: float = 1.0,
                  custom_fn: Optional[Callable] = None) -> None:
        """Configure H0 modification.  Call before ``run()`` (can be before init).

        Parameters
        ----------
        source : ExternalMatrixSource or None
            External matrix source (e.g. ``DeepHSource``).  Required for
            ``"replace"`` and ``"add"`` strategies.
        strategy : str
            ``"replace"`` | ``"add"`` | ``"scale"`` | ``"custom"``
        factor : float
            Scale factor for ``"scale"`` strategy.
        custom_fn : callable or None
            ``fn(live, external, structure, aux) -> None`` for
            ``"custom"`` strategy.  Modifies *live* in-place.

        Examples
        --------
        >>> from aimspy.interface.deeph import DeepHData, DeepHSource
        >>> src = DeepHSource(DeepHData.from_directory("deeph_warm/"))
        >>> calc.modify_h0(source=src)                     # replace
        >>> calc.modify_h0(source=src, strategy="add")     # add
        >>> calc.modify_h0(factor=0.9, strategy="scale")   # scale
        >>> calc.modify_h0(custom_fn=my_fn, source=src)    # custom
        """
        self._set_modify(_ModifyDef(
            strategy=strategy, external_source=source,
            factor=factor, custom_fn=custom_fn))

    # ==================================================================
    # Advanced: register_callback
    # ==================================================================
    def register_callback(self, name: str, fn: Callable,
                          aux: Any = None,
                          extra_ptr: Optional[int] = None) -> None:
        self._require_cb_mgr()
        from ._callbacks.registry import get_spec
        spec = get_spec(name)
        self._cb_mgr.register(spec, fn, aux, extra_ptr)

    def callback_registered(self, name: str) -> bool:
        if self._cb_mgr is None:
            return False
        return self._cb_mgr.is_registered(name)

    # ==================================================================
    # Internal helpers
    # ==================================================================
    def _state_guard(self, expected, action, *, allowed=None):
        ok = allowed.union({expected}) if allowed else {expected}
        if self._state not in ok:
            raise AimspyStateError(
                f"Cannot {action} in state {self._state.value}; "
                f"expected {expected.value}"
            )

    def _require_cb_mgr(self):
        if self._cb_mgr is None:
            raise AimspyStateError("Callbacks can only be registered after init()")

    def _copy_inputs(self):
        if self._comm.rank != 0:
            return
        import shutil
        for label, path in (
            ("control", self._cfg.control_path),
            ("geometry", self._cfg.geometry_path),
        ):
            if path is None:
                continue
            src = Path(path).resolve()
            dst = self._cfg.work_dir.resolve() / f"{label}.in"
            if src == dst:
                continue
            if not src.is_file():
                raise AimspyConfigError(f"{label} file not found: {src}")
            shutil.copy2(src, dst)

    def _check_callback_errors(self):
        if self._cb_mgr is None:
            return
        errs = getattr(self._cb_mgr, '_errors', None)
        if not errs:
            return
        _log.warning("rank %d: %d callback error(s) detected in run()",
                      self._comm.rank, len(errs))
        for name, exc, tb_str in errs:
            _log.warning("[%s] %s", name, exc)
        errs.clear()

    # ----------------------------------------------------------------
    # Modify config
    # ----------------------------------------------------------------
    def _set_modify(self, mdef: _ModifyDef) -> None:
        """Store modify definition and wire callbacks."""
        self._modify_def = mdef
        if self._modify_aux is not None:
            self._modify_aux['modify_def'] = mdef
        self._ensure_callbacks_wired()

    # ----------------------------------------------------------------
    # Callback wiring
    # ----------------------------------------------------------------
    def _ensure_callbacks_wired(self) -> None:
        """Wire get_descr + (export_h0) + (python_func) + modify_h0.

        All callbacks share a single ``aux`` dict.
        Idempotent — call as many times as needed.
        """
        if self._cb_mgr is None:
            return  # will be called again from init()

        mdef = self._modify_def
        cap  = self._capture_h0
        if not cap and mdef is None:
            return

        aux = self._modify_aux
        if aux is None:
            aux = {
                'structure': self.structure,
                'csr_descr': None,
                'initial_hamiltonian': None,
                'external_aimspy': None,
                'modify_def': mdef,
            }
            self._modify_aux = aux

        # 1. get_descr  (idempotent: only registers once)
        if not self._cb_mgr.is_registered('get_descr'):
            def _gd(ax):
                d = ax.get('descr')
                if d is not None:
                    ax['csr_descr'] = d
                    self._csr_descr = d
            self._cb_mgr.register(SPECS_BY_NAME['get_descr'], _gd, aux)

        # 2. export_h0 — capture free-atom H0 as AimspyMatrix
        if cap and not self._cb_mgr.is_registered('export_h0'):
            def _eh0(ax, h0, n_ham, n_spin):
                csr = aux.get('csr_descr')
                if csr is not None:
                    aux['initial_hamiltonian'] = AimspyMatrix.from_aims_csr(
                        h0, csr, self.structure)
            self._cb_mgr.register(SPECS_BY_NAME['export_h0'], _eh0, aux)

        # 3. python_func — convert external → aimspy
        if mdef is not None and mdef.external_source is not None \
           and not self._cb_mgr.is_registered('python_func'):
            def _pf(ax):
                md = aux.get('modify_def')
                ext = aux['external_aimspy']
                if ext is None and md is not None \
                   and md.external_source is not None:
                    aux['external_aimspy'] = md.external_source.to_aimspy(
                        self.structure)
            self._cb_mgr.register(SPECS_BY_NAME['python_func'], _pf, aux)

        # 4. modify_h0
        if mdef is not None and not self._cb_mgr.is_registered('modify_h0'):
            def _mh0(ax, h0, n_ham, n_spin):
                md = aux.get('modify_def')  # read dynamically (not captured)
                csr = aux.get('csr_descr')
                if md is None or csr is None:
                    return
                ext = aux.get('external_aimspy')
                live = AimspyMatrix.from_aims_csr(h0, csr, self.structure)
                _apply_strategy(md, live, ext, self.structure, aux)
                new_h = live.to_aims_csr(csr, self.structure)
                n_bytes = int(n_ham) * int(n_spin) * sizeof(c_double)
                memmove(h0.ctypes.data, new_h.ctypes.data, n_bytes)
            self._cb_mgr.register(SPECS_BY_NAME['modify_h0'], _mh0, aux)


# =============================================================================
# Strategy dispatch  (pure function)
# =============================================================================
def _apply_strategy(
    mdef: _ModifyDef,
    live: AimspyMatrix,
    external: Optional[AimspyMatrix],
    structure: AimspyStructure,
    aux: dict,
) -> None:
    """Apply the modify strategy in-place on *live*."""
    s = mdef.strategy

    if s == "replace":
        if external is not None:
            live.blocks.clear()
            # deep-copy each block to avoid aliasing
            live.blocks.update(
                {k: v.copy() for k, v in external.blocks.items()})
        else:
            _log.warning("modify_h0 replace: no external source provided — "
                         "H0 unchanged")

    elif s == "add":
        if external is not None:
            for key, block in external.blocks.items():
                blk = live.blocks.get(key)
                if blk is not None:
                    blk += block
                else:
                    live.blocks[key] = block.copy()
        else:
            _log.warning("modify_h0 add: no external source provided — "
                         "H0 unchanged")

    elif s == "scale":
        for key in list(live.blocks.keys()):
            live.blocks[key] *= mdef.factor

    elif s == "custom" and mdef.custom_fn is not None:
        mdef.custom_fn(live, external, structure, aux)

    else:
        _log.warning("modify_h0: unknown strategy %r — H0 unchanged", s)


# =============================================================================
# MPI helper
# =============================================================================
def _py2f(comm) -> int:
    from ctypes import c_int
    return c_int(comm.py2f())
