"""Public — external-format interface layer.

External format data classes (e.g.
:class:`aimspy.DeepHData`) provide a
``to_aimspy(structure) -> AimspyMatrix`` method for use with
:meth:`aimspy.Calculator.modify_init_ham` (via ``source=``).

To add support for a new external format, create a subpackage under
``aimspy/interface/<format>/`` containing a data class that implements
the :class:`ExternalMatrixSource` protocol.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..matrix import AimspyMatrix
    from ..structure import AimspyStructure


@runtime_checkable
class ExternalMatrixSource(Protocol):
    """Protocol for external matrix sources accepted by
    :meth:`aimspy.Calculator.modify_init_ham`.

    Any object with a ``to_aimspy(structure) -> AimspyMatrix`` method
    satisfies this protocol (structural typing / duck typing).

    Implementations:
      - :class:`aimspy.DeepHData`
    """

    def to_aimspy(self, structure: "AimspyStructure") -> "AimspyMatrix": ...


@runtime_checkable
class ExternalFirstOrderMatrixSource(Protocol):
    """Protocol for electric-response (DFPT) first-order Hamiltonian
    sources accepted by :meth:`aimspy.Calculator.modify_init_first_order_ham`.

    Any object with a ``to_first_order_aimspy(structure) -> list[AimspyMatrix]``
    method satisfies this protocol (structural typing / duck typing). The
    returned list must contain exactly 3 ``AimspyMatrix`` instances in
    Cartesian order ``[x, y, z]`` (Hartree units).

    Implementations:
      - :class:`aimspy.DeepHData`
    """

    def to_first_order_aimspy(
        self, structure: "AimspyStructure"
    ) -> list["AimspyMatrix"]: ...


__all__ = ["ExternalMatrixSource", "ExternalFirstOrderMatrixSource"]
