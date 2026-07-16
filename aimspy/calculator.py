"""Public — ``Calculator``, the primary user-facing class.

Usage (one-shot, common case)::

    from mpi4py import MPI
    from aimspy import Calculator, CalculatorConfig

    config = CalculatorConfig(lib_path="/path/to/libaims.so")
    with Calculator(config) as calc:
        calc.do(comm=MPI.COMM_WORLD, work_dir="./MoS2")
        H = calc.hamiltonian     # AimspyMatrix
        E = calc.energy

Two-step (advanced; e.g. to register callbacks between init and calc)::

    with Calculator(config) as calc:
        calc.init(comm=MPI.COMM_WORLD, work_dir="./MoS2")
        calc.register_callback('export_h0', my_fn, aux={})
        calc.calc()
        H = calc.hamiltonian

Warmstart with DeepH data (direct source)::

    from aimspy import Calculator, CalculatorConfig, Strategy
    from aimspy.interface.deeph import DeepHData

    data = DeepHData.from_directory("deeph_warm/")
    config = CalculatorConfig(lib_path="/path/to/libaims.so")
    calc = Calculator(config)
    calc.modify(source=data, strategy=Strategy.REPLACE)
    calc.do(comm=MPI.COMM_WORLD, work_dir="./MoS2")

Warmstart with DeepH data (deferred source — source generated at runtime
during ``python_func`` callback, after H0/overlap are available)::

    config = CalculatorConfig(
        lib_path=..., capture_initial_hamiltonian=True,
    )
    calc = Calculator(config)

    @calc.modify(strategy=Strategy.REPLACE, aux={"deeph_path": "deeph_warm/"})
    def gen_source(calculator, aux):
        # calculator.initial_hamiltonian / .overlap are available here
        return DeepHData.from_directory(aux["deeph_path"])

    calc.do(comm=MPI.COMM_WORLD, work_dir="./MoS2")

Capture free-atom initial Hamiltonian (optional)::

    config = CalculatorConfig(
        lib_path=..., capture_initial_hamiltonian=True,
    )
    with Calculator(config) as calc:
        calc.do(comm=MPI.COMM_WORLD, work_dir="./MoS2")
        h_init = calc.initial_hamiltonian
"""

from __future__ import annotations

import logging
from ctypes import memmove, sizeof, c_double
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple, Union

import numpy as np

if TYPE_CHECKING:
    from .interface import ExternalMatrixSource

from ._system import chdir_cm
from ._exceptions import (
    AimspyConfigError,
    AimspyStateError,
    AimspyCallbackError,
)
from ._binding.libloader import load_aims_lib
from ._binding.prototypes import BindingLib
from ._callbacks.base import CallbackManager
from ._callbacks.registry import SPECS_BY_NAME, get_spec, CallbackName
from .data import AimspyInfo, CsrMatrixDescriptor
from .structure import AimspyStructure
from .info import load_info
from .matrix import (
    get_rs_hamiltonian,
    get_rs_overlap,
    get_forces,
    AimspyMatrix,
)

_log = logging.getLogger("aimspy")
logging.basicConfig(level=logging.WARNING)


# =============================================================================
# Strategy — initial Hamiltonian modification strategies
# =============================================================================
class Strategy(Enum):
    """Initial Hamiltonian modification strategy names.

    Used by :meth:`Calculator.modify`.
    """

    REPLACE = "replace"
    ADD = "add"
    SCALE = "scale"
    CUSTOM = "custom"


# =============================================================================
# CalculatorConfig
# =============================================================================
@dataclass
class CalculatorConfig:
    """Configuration for :class:`Calculator`.

    All fields are construction-time declarations; ``work_dir`` and ``comm``
    are passed to :meth:`Calculator.init` / :meth:`Calculator.do` at
    execution time.

    Parameters
    ----------
    lib_path : Path
        Path to the patched ``libaims.so``.
    control_path, geometry_path : Path or None
        Optional input files copied into ``work_dir`` at run time.
    initializer : callable or None
        ``fn(Calculator) -> None`` invoked on rank 0 after inputs are copied
        but before ``aimspy_init``.
    log_level : str
        Python logging level name (default ``"INFO"``).
    logfile : Path
        aims log file name (relative to ``work_dir`` after chdir).
    capture_initial_hamiltonian : bool
        If True, register the ``export_h0`` callback so that
        :attr:`Calculator.initial_hamiltonian` is available after
        ``calc()``. Default False (free-atom initial Hamiltonian capture
        is opt-in).
    capture_overlap : bool
        If True, register the ``export_ovlp`` callback so that
        :attr:`Calculator.overlap` returns the live overlap matrix
        (available from ``INITED`` state onward, all MPI ranks) instead
        of the ``c_overlap`` copy (rank 0 only). Default False.
    """

    lib_path: Path | str
    control_path: Optional[Path | str] = None
    geometry_path: Optional[Path | str] = None
    initializer: Optional[Callable[["Calculator"], None]] = None
    log_level: str = "INFO"
    logfile: Path | str = Path("aims.out")
    capture_initial_hamiltonian: bool = False
    capture_overlap: bool = False

    def __post_init__(self):
        self.lib_path = Path(self.lib_path)
        if self.control_path is not None:
            self.control_path = Path(self.control_path)
        if self.geometry_path is not None:
            self.geometry_path = Path(self.geometry_path)
        self.logfile = Path(self.logfile)


class CalcState(Enum):
    UNINIT = "uninit"
    INITED = "inited"
    RUNNING = "running"  # internal transient inside calc()
    DONE = "done"
    FAILED = "failed"  # operation aborted; Fortran runtime in unknown state
    FINALIZED = "finalized"


# =============================================================================
# Calculator
# =============================================================================
class Calculator:
    """In-memory interface to FHI-aims via ctypes.

    Three lifecycle entry points:

    - :meth:`do` — one-shot (init + calc). Common case.
    - :meth:`init` + :meth:`calc` — two-step, for advanced use cases
      that need to register callbacks between init and calc.
    - :meth:`close` — finalize (also called by ``__exit__``).
      Use :meth:`force_close` after SCF failure.

    H0 modification is configured via :meth:`modify` (direct or deferred
    source), which must be called before :meth:`do` / :meth:`init`.

    .. note::

        **Thread safety**: NOT thread-safe. FHI-aims uses ``chdir(2)``
        (process-global) internally, so concurrent Calculator operations
        in the same process will race. Use one Calculator per process
        (typical MPI usage: one rank = one process).

    .. note::

        **Rank-0-only properties**: The following properties read from
        Fortran rank-0-only buffers and return ``None`` or raise on
        non-root ranks: ``rs_hamiltonian``, ``rs_overlap``, ``hamiltonian``,
        ``overlap`` (without ``capture_overlap=True``).
        Properties available on all ranks: ``info``, ``structure``,
        ``csr_descr``, ``energy``, ``forces``, ``initial_hamiltonian``
        (with ``capture_initial_hamiltonian=True``), ``overlap`` (with
        ``capture_overlap=True``).
    """

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
        self._rank: int = 0

        self._binding: Optional[BindingLib] = None
        self._cb_mgr: Optional[CallbackManager] = None
        self._info: Optional[AimspyInfo] = None
        self._structure: Optional[AimspyStructure] = None

        self._comm: Any = None
        self._work_dir: Optional[Path] = None

        # Single source of truth for runtime callback scratch state.
        # Populated in init(); keys: structure, cfg, modify, csr_descr,
        # overlap, initial_hamiltonian, external_aimspy.
        self._runtime_aux: Optional[dict] = None

        # Deferred callback registrations made before init(). Each entry:
        # (spec_name, fn, aux, extra_ptr). Applied inside init() after
        # _cb_mgr is created, before _wire_callbacks().
        self._pending_callbacks: List[Tuple[str, Callable, Any, Optional[int]]] = []

        # H0 modification config (set by modify() before do()/init()).
        # SimpleNamespace with fields: source, strategy, factor, custom_fn.
        # None = no modification (baseline SCF).
        self._modify: Optional[SimpleNamespace] = None

        self._forces: Optional[np.ndarray] = None

    # ==================================================================
    # Logging helpers — INFO/WARNING on rank 0 only; ERROR on all ranks
    # ==================================================================
    def _log_info(self, msg, *args):
        if self._rank == 0:
            self._log.info(msg, *args)

    def _log_warning(self, msg, *args):
        if self._rank == 0:
            self._log.warning(msg, *args)

    def _log_error(self, msg, *args):
        self._log.error("rank %d: " + msg, self._rank, *args)

    # ==================================================================
    # Context manager
    # ==================================================================
    def __enter__(self) -> "Calculator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            # Body raised — use force_close so cleanup never masks the
            # original exception with a state error.
            self.force_close()
        else:
            self.close()

    # ==================================================================
    # Lifecycle: init / calc / do
    # ==================================================================
    def init(
        self,
        comm: Any = None,
        work_dir: Union[Path, str] = Path("."),
    ) -> "Calculator":
        """Initialize FHI-aims runtime.

        Loads libaims, calls ``aimspy_init``, loads runtime info, and
        wires default callbacks based on :attr:`_cfg` and :attr:`_modify`.
        After init(), users can register additional callbacks via
        :meth:`register_callback` before calling :meth:`calc`.

        Parameters
        ----------
        comm : mpi4py.MPI.Comm or None
            MPI communicator. Defaults to ``MPI.COMM_WORLD``.
        work_dir : Path or str
            Working directory (created if missing). aims runs with this
            as cwd; ``logfile`` and copied input files land here.
            Default: current directory.

        Raises
        ------
        AimspyStateError
            If called from any state other than UNINIT.
        """
        if self._state != CalcState.UNINIT:
            raise AimspyStateError(
                f"Cannot init() in state {self._state.value}; "
                f"expected {CalcState.UNINIT.value}"
            )
        if comm is None:
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
        self._comm = comm
        rank = comm.rank
        self._rank = rank
        self._work_dir = Path(work_dir)

        self._log.setLevel(getattr(logging, self._cfg.log_level.upper(), logging.INFO))
        self._log_info("preparing work_dir %s", self._work_dir)

        self._work_dir.mkdir(parents=True, exist_ok=True)

        try:
            with chdir_cm(self._work_dir):
                self._copy_inputs()
                if rank == 0 and self._cfg.initializer:
                    self._cfg.initializer(self)
                comm.Barrier()

                self._log_info("loading lib %s", self._cfg.lib_path)
                cdll = load_aims_lib(self._cfg.lib_path)
                self._binding = BindingLib(cdll)
                self._cb_mgr = CallbackManager(self._binding)

                self._log_info("aimspy_init")
                self._binding.aimspy_init(
                    _py2f(comm),
                    str(self._cfg.logfile).encode("UTF-8"),
                )
                comm.Barrier()

                self._info = load_info(self._binding)
                self._structure = AimspyStructure.from_info(self._info)

                # Apply deferred user-registered callbacks first.
                for name, fn, aux, extra_ptr in self._pending_callbacks:
                    spec = get_spec(name)
                    self._cb_mgr.register(spec, fn, aux, extra_ptr)
                self._pending_callbacks.clear()

                # Wire default callbacks based on config + modify.
                self._wire_callbacks()

            self._state = CalcState.INITED
            self._log_info("init done. n_basis=%d", self._info.n_basis)
        except Exception:
            self._state = CalcState.FAILED
            self._log_error("init() failed; attempting cleanup")
            self._defensive_finalize()
            self._clear_all_state()
            raise
        return self

    def calc(self) -> "Calculator":
        """Execute SCF calculation.

        Must be called after :meth:`init`. Raises
        :class:`AimspyCallbackError` if any registered callback raised
        during SCF.
        """
        self._state_guard(CalcState.INITED, "calc")
        self._state = CalcState.RUNNING
        self._log_info("aimspy_run ...")
        try:
            with chdir_cm(self._work_dir):
                self._binding.aimspy_run()
        except Exception:
            self._state = CalcState.FAILED
            self._log_error("aimspy_run() failed")
            raise
        self._state = CalcState.DONE
        self._log_info("calc done")
        # Eager capture of forces BEFORE checking callback errors, so that
        # forces are available even if a callback raised (forces are computed
        # by Fortran independent of Python callbacks).
        try:
            self._forces = get_forces(self._binding, self.info.n_atoms)
        except Exception as e:
            self._log_warning("forces capture failed: %r", e)
        self._check_callback_errors()
        return self

    def do(
        self,
        comm: Any = None,
        work_dir: Union[Path, str] = Path("."),
    ) -> "Calculator":
        """One-shot: :meth:`init` + :meth:`calc`.

        Convenience entry point for the common case where no callbacks
        need to be registered between init and calc. Parameters are
        forwarded to :meth:`init`.

        Raises
        ------
        AimspyStateError
            If called from any state other than UNINIT.
        AimspyCallbackError
            If any registered callback raised during SCF.
        """
        self.init(comm, work_dir)
        self.calc()
        return self

    def close(self) -> None:
        """Finalize FHI-aims.

        Behavior by state:

        - UNINIT / FINALIZED: silent no-op.
        - RUNNING: raises ``AimspyStateError`` (use :meth:`force_close`
          if the SCF has aborted).
        - INITED / DONE: normal finalize (errors propagate).
        - FAILED: defensive finalize (errors swallowed and logged).

        Also invoked by ``__exit__``.
        """
        if self._state in (CalcState.UNINIT, CalcState.FINALIZED):
            return
        if self._state == CalcState.RUNNING:
            raise AimspyStateError(
                "Cannot close() in RUNNING state; use force_close() "
                "if the SCF has aborted"
            )

        if self._state == CalcState.FAILED:
            self._defensive_finalize()
        else:
            # INITED or DONE: normal finalize
            self._log_info("aimspy_finalize")
            with chdir_cm(self._work_dir or Path(".")):
                if self._binding is not None:
                    self._binding.aimspy_finalize()
        self._clear_all_state()

    def force_close(self) -> None:
        """Force-finalize regardless of state. Swallows Fortran errors.

        Use after SCF failure or partial init when :meth:`close` refuses
        (RUNNING state) or when the Fortran runtime is in an unknown
        condition (FAILED state). Always clears all retained state.
        """
        if self._state == CalcState.FINALIZED:
            return
        self._defensive_finalize()
        self._clear_all_state()

    # ==================================================================
    # Properties: query
    # ==================================================================
    @property
    def info(self) -> AimspyInfo:
        if self._info is None:
            if self._state == CalcState.UNINIT:
                hint = "call init() first"
            elif self._state == CalcState.FAILED:
                hint = "init failed; create a new Calculator instance"
            else:
                hint = "Calculator is finalized; create a new instance"
            raise AimspyStateError(
                f"Cannot access info in state {self._state.value}; {hint}"
            )
        return self._info

    @property
    def structure(self) -> AimspyStructure:
        if self._structure is None:
            if self._state == CalcState.UNINIT:
                hint = "call init() first"
            elif self._state == CalcState.FAILED:
                hint = "init failed; create a new Calculator instance"
            else:
                hint = "Calculator is finalized; create a new instance"
            raise AimspyStateError(
                f"Cannot access structure in state {self._state.value}; {hint}"
            )
        return self._structure

    @property
    def work_dir(self) -> Optional[Path]:
        return self._work_dir

    @property
    def comm(self) -> Any:
        return self._comm

    @property
    def energy(self) -> float:
        """SCF total energy (Hartree).

        Available in ``DONE`` state. Also readable in ``RUNNING`` (inside
        a callback during pre-SCF), but the value may be uninitialized
        before the first SCF iteration completes.
        """
        self._state_guard(
            CalcState.DONE, "read energy", allowed={CalcState.RUNNING, CalcState.DONE}
        )
        return self._binding.aimspy_energy()

    @property
    def forces(self) -> Optional[np.ndarray]:
        """Total atomic forces, shape (n_atoms, 3), units eV/Å.

        Eagerly captured at the end of :meth:`calc` and cached. Available
        in ``DONE`` (and ``RUNNING`` if accessed from a callback after SCF
        completion).

        Returns ``None`` when forces are not available:

        - Before :meth:`calc` is called
        - ``compute_forces .true.`` not set in ``control.in``
        - :meth:`calc` raised before forces were captured (FAILED state)
        - After :meth:`close` (state FINALIZED)
        """
        self._state_guard(
            CalcState.DONE, "read forces", allowed={CalcState.RUNNING, CalcState.DONE}
        )
        return self._forces

    @property
    def rs_hamiltonian(self) -> np.ndarray:
        self._state_guard(CalcState.DONE, "read rs_hamiltonian")
        return get_rs_hamiltonian(self._binding, self.info.n_spin, self.info.n_ham_size)

    @property
    def rs_overlap(self) -> np.ndarray:
        self._state_guard(
            CalcState.INITED,
            "read rs_overlap",
            allowed={CalcState.INITED, CalcState.RUNNING, CalcState.DONE},
        )
        return get_rs_overlap(self._binding, self.info.n_ham_size)

    @property
    def csr_descr(self) -> Optional[CsrMatrixDescriptor]:
        if self._runtime_aux is None:
            return None
        return self._runtime_aux.get("csr_descr")

    @property
    def hamiltonian(self) -> AimspyMatrix:
        self._state_guard(CalcState.DONE, "read hamiltonian")
        csr = self.csr_descr
        if csr is None:
            raise AimspyStateError(
                "csr_descr not available; calc() must complete first."
            )
        return AimspyMatrix.from_aims_csr(self.rs_hamiltonian, csr, self.structure)

    @property
    def overlap(self) -> AimspyMatrix:
        """Overlap matrix as :class:`AimspyMatrix`.

        If ``capture_overlap=True`` was set in :class:`CalculatorConfig`,
        returns the live overlap captured by the ``export_ovlp`` callback
        (available from ``INITED`` state onward, all MPI ranks).

        Otherwise, falls back to reading from ``c_overlap`` (the aims
        internal copy, rank 0 only, requires ``DONE`` state).
        """
        self._state_guard(
            CalcState.INITED,
            "read overlap",
            allowed={CalcState.INITED, CalcState.RUNNING, CalcState.DONE},
        )
        if self._runtime_aux is not None:
            ovlp_mx = self._runtime_aux.get("overlap")
            if ovlp_mx is not None:
                return ovlp_mx
        self._state_guard(CalcState.DONE, "read overlap (fallback)")
        csr = self.csr_descr
        if csr is None:
            raise AimspyStateError(
                "csr_descr not available; calc() must complete first."
            )
        ovlp = self.rs_overlap.reshape(1, -1)
        return AimspyMatrix.from_aims_csr(ovlp, csr, self.structure)

    @property
    def initial_hamiltonian(self) -> Optional[AimspyMatrix]:
        """Free-atom initial Hamiltonian (H_init) as :class:`AimspyMatrix`.

        Returns ``None`` unless
        ``CalculatorConfig.capture_initial_hamiltonian=True`` was set.
        """
        if self._runtime_aux is None:
            return None
        return self._runtime_aux.get("initial_hamiltonian")

    # ==================================================================
    # H0 modification (unified API: direct + deferred)
    # ==================================================================
    def modify(
        self,
        source: Optional["ExternalMatrixSource"] = None,
        *,
        strategy: Union[Strategy, str] = Strategy.REPLACE,
        factor: float = 1.0,
        custom_fn: Optional[Callable] = None,
        aux: Optional[dict] = None,
    ) -> Optional[Callable]:
        """Configure H0 modification — unified API for both direct and
        deferred source.

        Must be called before :meth:`do` / :meth:`init`.

        **Direct mode** (pre-built source or source-less strategy)::

            calc.modify(source=data, strategy=Strategy.REPLACE)
            calc.modify(source=data, strategy=Strategy.ADD)
            calc.modify(strategy=Strategy.SCALE, factor=0.9)
            calc.modify(strategy=Strategy.CUSTOM, custom_fn=my_fn, source=data)

        **Deferred mode** (source generated at runtime via decorator;
        only for REPLACE / ADD strategies without a pre-built source)::

            @calc.modify(strategy=Strategy.REPLACE, aux={"deeph_path": "..."})
            def gen_source(calculator, aux):
                # calculator.initial_hamiltonian / .overlap are available
                # here if capture_* was enabled in CalculatorConfig.
                return DeepHData.from_directory(aux["deeph_path"])

        In deferred mode, the decorated function ``fn(calculator, aux)``
        is called during the ``python_func`` callback (between
        ``export_h0`` and ``modify_h0`` in ``initialize_scf.f90``),
        with access to the live :class:`Calculator` object. The returned
        source must have a ``to_aimspy(structure) -> AimspyMatrix``
        method (e.g. :class:`~aimspy.interface.deeph.DeepHData`).

        Parameters
        ----------
        source : object or None
            External matrix source with ``to_aimspy(structure)`` method
            (e.g. ``DeepHData``). Required for direct REPLACE/ADD.
        strategy : Strategy or str
            Modification strategy (default REPLACE). Accepts string.
        factor : float
            Scale factor for SCALE strategy.
        custom_fn : callable or None
            ``fn(live, external, structure, aux) -> None`` for CUSTOM.
        aux : dict or None
            User-specified data passed to the deferred source function
            as second argument.

        Returns
        -------
        callable or None
            In direct mode: ``None``.
            In deferred mode: a decorator (which returns the original
            function unchanged).

        Raises
        ------
        AimspyConfigError
            If CUSTOM strategy is used without ``custom_fn``, or if
            *strategy* is not a valid :class:`Strategy` value.
        """
        if isinstance(strategy, str):
            try:
                strategy = Strategy(strategy.lower())
            except ValueError:
                raise AimspyConfigError(
                    f"modify: invalid strategy {strategy!r}; "
                    f"valid: {[s.value for s in Strategy]}"
                )

        if strategy == Strategy.CUSTOM and custom_fn is None:
            raise AimspyConfigError("modify: CUSTOM strategy requires 'custom_fn'")

        # Determine mode: direct if source is given, strategy doesn't
        # need source (SCALE), or custom_fn is given (CUSTOM).
        is_direct = (
            source is not None or strategy == Strategy.SCALE or custom_fn is not None
        )

        if is_direct:
            # ── Direct mode: store config immediately ──
            self._modify = SimpleNamespace(
                source=source,
                strategy=strategy,
                factor=factor,
                custom_fn=custom_fn,
            )
            return None

        # ── Deferred mode: return a decorator ──
        def decorator(fn: Callable) -> Callable:
            user_aux = aux if aux is not None else {}
            calc = self  # capture for closure

            def to_aimspy(structure: AimspyStructure) -> AimspyMatrix:
                # Called from _on_python_func during python_func callback.
                # At this point calc.initial_hamiltonian / .overlap etc.
                # are available (if capture_* was enabled).
                src = fn(calc, user_aux)
                if src is None:
                    raise AimspyConfigError(
                        "deferred modify source function returned None; "
                        "it must return an object with a "
                        "to_aimspy(structure) method"
                    )
                return src.to_aimspy(structure)

            wrapper = SimpleNamespace(to_aimspy=to_aimspy)
            self._modify = SimpleNamespace(
                source=wrapper,
                strategy=strategy,
                factor=factor,
                custom_fn=custom_fn,
            )
            return fn

        return decorator

    # ==================================================================
    # Advanced: register_callback (deferred; applied in init())
    # ==================================================================
    def register_callback(
        self,
        name: Union[str, "CallbackName"],
        fn: Callable,
        aux: Any = None,
        extra_ptr: Optional[int] = None,
    ) -> None:
        """Register a custom callback (advanced API).

        Can be called at two points:

        - **Pre-init** (state UNINIT): the registration is deferred and
          applied inside :meth:`init` after the callback manager is
          created, before :meth:`_wire_callbacks`.
        - **Post-init, pre-calc** (state INITED): the registration is
          applied immediately to the live callback manager.

        Calling from DONE state is allowed (registers on the live manager)
        but the callback will never fire (SCF already completed).
        Calling from FAILED/FINALIZED raises :class:`AimspyStateError`.

        If the user registers a callback with the same ``name`` as one
        that :meth:`_wire_callbacks` would register by default, the
        user's registration takes precedence (the default is skipped).
        Post-init registration overrides any previously-registered
        callback with the same name.

        Parameters
        ----------
        name : str or CallbackName
            Callback spec name (e.g. ``'export_h0'``) or enum member.
        fn : callable
            Python-side callback function.
        aux : any
            Arbitrary Python object passed through to the callback.
        extra_ptr : int or None
            Extra c-pointer for 3-arg register functions.
        """
        if isinstance(name, CallbackName):
            name = name.value
        if self._state == CalcState.UNINIT:
            self._pending_callbacks.append((name, fn, aux, extra_ptr))
            return
        if self._cb_mgr is None:
            raise AimspyStateError(
                "register_callback: callback manager unavailable "
                f"in state {self._state.value}"
            )
        if self._state == CalcState.DONE:
            import warnings

            warnings.warn(
                "register_callback from DONE state: SCF already completed, "
                "callback will never fire",
                UserWarning,
                stacklevel=2,
            )
        spec = get_spec(name)
        self._cb_mgr.register(spec, fn, aux, extra_ptr)

    def callback_registered(self, name: Union[str, "CallbackName"]) -> bool:
        if isinstance(name, CallbackName):
            name = name.value
        if self._cb_mgr is None:
            return any(n == name for n, _, _, _ in self._pending_callbacks)
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
            dst = self._work_dir.resolve() / f"{label}.in"
            if src == dst:
                continue
            if not src.is_file():
                raise AimspyConfigError(f"{label} file not found: {src}")
            shutil.copy2(src, dst)

    def _check_callback_errors(self):
        if self._cb_mgr is None:
            return
        errs = getattr(self._cb_mgr, "_errors", None)
        if not errs:
            return
        self._log_warning("%d callback error(s) detected in calc()", len(errs))
        captured = list(errs)
        # Don't clear — preserve for user inspection via _cb_mgr._errors
        for name, exc, tb_str in captured:
            # ERROR on all ranks for debugging
            self._log.error("rank %d: [%s] %s\n%s", self._rank, name, exc, tb_str)
        exc = AimspyCallbackError(
            f"{len(captured)} callback error(s) during calc(): "
            + ", ".join(n for n, _, _ in captured)
        )
        exc.callback_errors = captured
        raise exc from captured[0][1]

    def _defensive_finalize(self) -> None:
        """Attempt aimspy_finalize(), swallowing all errors.

        Used by :meth:`close` (FAILED state) and :meth:`force_close` when
        the Fortran runtime may be in an unknown condition.
        """
        try:
            if self._binding is not None:
                with chdir_cm(self._work_dir or Path(".")):
                    self._binding.aimspy_finalize()
        except Exception as e:
            self._log_warning("defensive aimspy_finalize raised: %r", e)

    def _clear_all_state(self) -> None:
        """Clear all retained state fields. Idempotent."""
        self._state = CalcState.FINALIZED
        self._binding = None
        self._cb_mgr = None
        self._runtime_aux = None
        self._info = None
        self._structure = None
        self._forces = None
        self._modify = None
        self._pending_callbacks.clear()

    # ----------------------------------------------------------------
    # Callback wiring (called once inside init())
    # ----------------------------------------------------------------
    def _wire_callbacks(self) -> None:
        """Wire default callbacks based on self._cfg and self._modify.

        Always wires ``get_descr`` (so ``hamiltonian``/``overlap`` are
        available by default). Conditionally wires ``export_ovlp`` (if
        ``capture_overlap``), ``export_h0`` (if
        ``capture_initial_hamiltonian``), and ``modify_h0``/``python_func``
        (if ``self._modify`` is set).

        Non-idempotent: intended to be called exactly once inside init().
        Pre-registered user callbacks (via :meth:`register_callback`) take
        precedence — checked via ``is_registered``.

        Note: callback spec names ``export_h0`` / ``modify_h0`` are kept
        as-is because they mirror the Fortran binding symbols
        (``aimspy_register_export_h0_callback`` etc.); the ``h0`` here
        denotes the initial Hamiltonian (H_init) in the physics sense.
        """
        if self._cb_mgr is None:
            return

        mspec = self._modify

        aux = self._runtime_aux = {
            "structure": self.structure,
            "cfg": self._cfg,
            "modify": mspec,
            "csr_descr": None,
            "overlap": None,
            "initial_hamiltonian": None,
            "external_aimspy": None,
            "rank": self._rank,
        }

        # 1. get_descr — always (enables hamiltonian / overlap by default)
        if not self._cb_mgr.is_registered("get_descr"):

            def _on_get_descr(ax):
                d = ax.get("descr")
                if d is not None:
                    ax["csr_descr"] = d

            self._cb_mgr.register(SPECS_BY_NAME["get_descr"], _on_get_descr, aux)

        # 2. export_ovlp — only if capture_overlap=True
        if self._cfg.capture_overlap and not self._cb_mgr.is_registered("export_ovlp"):

            def _on_export_overlap(ax, ovlp, n_ham, n_spin):
                csr = ax.get("csr_descr")
                if csr is not None:
                    ax["overlap"] = AimspyMatrix.from_aims_csr(
                        ovlp, csr, self.structure
                    )

            self._cb_mgr.register(SPECS_BY_NAME["export_ovlp"], _on_export_overlap, aux)

        # 3. export_h0 — only if capture_initial_hamiltonian=True
        if self._cfg.capture_initial_hamiltonian and not self._cb_mgr.is_registered(
            "export_h0"
        ):

            def _on_export_initial_hamiltonian(ax, h_init, n_ham, n_spin):
                csr = ax.get("csr_descr")
                if csr is not None:
                    ax["initial_hamiltonian"] = AimspyMatrix.from_aims_csr(
                        h_init, csr, self.structure
                    )

            self._cb_mgr.register(
                SPECS_BY_NAME["export_h0"],
                _on_export_initial_hamiltonian,
                aux,
            )

        # 4. python_func — convert external → aimspy (if modify.source)
        if (
            mspec is not None
            and mspec.source is not None
            and not self._cb_mgr.is_registered("python_func")
        ):

            def _on_python_func(ax):
                md = ax["modify"]
                if md is None or md.source is None:
                    return
                if ax.get("external_aimspy") is None:
                    ax["external_aimspy"] = md.source.to_aimspy(self.structure)

            self._cb_mgr.register(SPECS_BY_NAME["python_func"], _on_python_func, aux)

        # 5. modify_h0 — if modify is set
        if mspec is not None and not self._cb_mgr.is_registered("modify_h0"):

            def _on_modify_initial_hamiltonian(ax, h_init, n_ham, n_spin):
                md = ax["modify"]
                csr = ax.get("csr_descr")
                if md is None or csr is None:
                    return
                ext = ax.get("external_aimspy")
                live = AimspyMatrix.from_aims_csr(h_init, csr, self.structure)
                _apply_strategy(md, live, ext, self.structure, ax)
                new_h = live.to_aims_csr(csr, self.structure)
                n_bytes = int(n_ham) * int(n_spin) * sizeof(c_double)
                memmove(h_init.ctypes.data, new_h.ctypes.data, n_bytes)

            self._cb_mgr.register(
                SPECS_BY_NAME["modify_h0"],
                _on_modify_initial_hamiltonian,
                aux,
            )


# =============================================================================
# Strategy dispatch (pure function; reads from SimpleNamespace via duck typing)
# =============================================================================
def _apply_strategy(
    mspec: Any,
    live: AimspyMatrix,
    external: Optional[AimspyMatrix],
    structure: AimspyStructure,
    aux: dict,
) -> None:
    """Apply the modify strategy in-place on *live*.

    The ``modify_h0`` callback name (mirrored from the Fortran binding)
    refers to the initial Hamiltonian; ``live`` here is the live initial
    Hamiltonian as an :class:`AimspyMatrix`.
    """
    s = mspec.strategy
    rank = aux.get("rank", 0)

    if s == Strategy.REPLACE:
        if external is not None:
            live.blocks.clear()
            live.blocks.update({k: v.copy() for k, v in external.blocks.items()})
        elif rank == 0:
            _log.warning(
                "modify_h0 replace: no external source — "
                "initial Hamiltonian unchanged"
            )

    elif s == Strategy.ADD:
        if external is not None:
            for key, block in external.blocks.items():
                blk = live.blocks.get(key)
                if blk is not None:
                    blk += block
                else:
                    live.blocks[key] = block.copy()
        elif rank == 0:
            _log.warning(
                "modify_h0 add: no external source — initial Hamiltonian unchanged"
            )

    elif s == Strategy.SCALE:
        for key in list(live.blocks.keys()):
            live.blocks[key] *= mspec.factor

    elif s == Strategy.CUSTOM:
        if mspec.custom_fn is not None:
            mspec.custom_fn(live, external, structure, aux)

    elif rank == 0:
        _log.warning(
            "modify_h0: unknown strategy %r — initial Hamiltonian unchanged",
            s,
        )


# =============================================================================
# MPI helper
# =============================================================================
def _py2f(comm) -> int:
    from ctypes import c_int

    return c_int(comm.py2f())
