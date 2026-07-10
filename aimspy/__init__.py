"""AimSpy — Pure-Python in-memory interface to FHI-aims via ctypes.

For seamless integration with DeepX / DeepH-pack.

Primary entry points:

  - :class:`Calculator`      — the main user-facing class
  - :class:`CalculatorConfig` — configuration dataclass
  - :class:`AimspyInfo`       — snapshot of aims runtime state
  - :class:`AimspyStructure`  — structure + orbital descriptor (reusable)
  - :class:`AimspyMatrix`     — block-sparse matrix in aimspy standard format
  - :class:`CsrMatrixDescriptor` — aims CSR sparse layout

  Configure modification via ``calc.modify_h0(...)`` — no extra objects needed.

Interface layer (external format adapters):

  - :class:`ExternalMatrixSource` — pluggable external source ABC
  - :mod:`aimspy.interface.deeph` — DeepH format reader + converter
"""
from __future__ import annotations

from ._version import __version__
from .calculator import Calculator, CalculatorConfig, CalcState
from .data import (
    AimspyInfo, CsrMatrixDescriptor,
    HARTREE_TO_EV, EV_TO_HARTREE, BOHR_TO_ANG,
)
from .structure import AimspyStructure
from .matrix import (
    AimspyMatrix,
    get_rs_hamiltonian, get_rs_overlap,
)
from .info import load_info
from ._callbacks.base import CallbackSpec, DefaultCallback
from ._exceptions import (
    AimspyError, AimspyConfigError, AimspyBindingError,
    AimspyCallbackError, AimspyStateError,
)
from .interface import ExternalMatrixSource

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
    "Calculator",
    "CalculatorConfig",
    "CallbackSpec",
    "CsrMatrixDescriptor",
    "DefaultCallback",
    "EV_TO_HARTREE",
    "ExternalMatrixSource",
    "get_rs_hamiltonian",
    "get_rs_overlap",
    "HARTREE_TO_EV",
    "load_info",
]
