"""AimsPy — Pure-Python in-memory interface to FHI-aims via ctypes.

For seamless integration with DeepX / DeepH-pack.

Primary entry points:

  - :class:`Calculator`        — the main user-facing class
                                 (init / calc / do / close / force_close lifecycle)
  - :class:`CalculatorConfig`  — configuration dataclass
  - :class:`Strategy`          — initial Hamiltonian modification strategy enum
  - :class:`CallbackName`      — callback type identifiers (enum, for register_callback)
  - :class:`ExternalMatrixSource` — Protocol for external matrix sources

  H0 modification (warmstart, scale, custom) is configured via
  :meth:`Calculator.modify_init_ham` (direct or deferred source), called before
  :meth:`Calculator.do`.
  ``CalculatorConfig.capture_initial_hamiltonian=True`` opts in to
  capturing the free-atom initial Hamiltonian (exposed via
  :attr:`Calculator.initial_hamiltonian`).
  ``CalculatorConfig.capture_overlap=True`` opts in to capturing the live
  overlap matrix (exposed via :attr:`Calculator.overlap`).

  Logging: INFO and WARNING messages are emitted on rank 0 only; ERROR
  messages are emitted on all ranks for debugging.

  Note: callback spec names ``export_h0`` / ``modify_h0`` are kept as-is
  because they mirror the Fortran binding symbols; the ``h0`` there
  denotes the initial Hamiltonian (H_init) in the physics sense.

Interface layer (external format adapters):

  - :class:`DeepHData` (also at :mod:`aimspy.interface.deeph`) — DeepH format reader + converter
"""

from __future__ import annotations

from ._version import __version__
from .calculator import (
    Calculator,
    CalculatorConfig,
    CalcState,
    Strategy,
)
from ._callbacks.registry import CallbackName
from .interface import ExternalMatrixSource
from .interface.deeph import DeepHData
from .data import (
    AimspyInfo,
    CsrMatrixDescriptor,
    HARTREE_TO_EV,
    EV_TO_HARTREE,
    BOHR_TO_ANG,
)
from .structure import AimspyStructure
from .matrix import (
    AimspyMatrix,
    get_rs_hamiltonian,
    get_rs_overlap,
    get_forces,
)
from .info import load_info
from ._exceptions import (
    AimspyError,
    AimspyConfigError,
    AimspyBindingError,
    AimspyCallbackError,
    AimspyStateError,
)

__all__ = [
    "__version__",
    "AimspyBindingError",
    "AimspyCallbackError",
    "AimspyConfigError",
    "AimspyError",
    "AimspyInfo",
    "AimspyMatrix",
    "AimspyStateError",
    "AimspyStructure",
    "BOHR_TO_ANG",
    "CalcState",
    "CallbackName",
    "Calculator",
    "CalculatorConfig",
    "CsrMatrixDescriptor",
    "DeepHData",
    "EV_TO_HARTREE",
    "ExternalMatrixSource",
    "get_rs_hamiltonian",
    "get_rs_overlap",
    "get_forces",
    "HARTREE_TO_EV",
    "load_info",
    "Strategy",
]
