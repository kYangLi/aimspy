"""Public — pluggable external-matrix-source interface.

To add support for a new external format (DFTB+, Wannier, etc.):
  1. Subclass ``ExternalMatrixSource``.
  2. Implement ``to_aimspy(structure) -> AimspyMatrix``.
  3. Pass the instance to ``Calculator.modify_h0``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..matrix import AimspyMatrix
from ..structure import AimspyStructure


class ExternalMatrixSource(ABC):
    """Abstract base: an external H source convertible to aimspy format.

    Concrete sources (e.g. ``DeepHSource``) parse external data and
    provide a ``to_aimspy`` method that the Calculator's ``python_func``
    callback invokes at runtime.

    The conversion receives the live ``AimspyStructure`` (built from
    ``aimspy_init``'s ``AimspyInfo``), allowing atom reordering.
    """

    @abstractmethod
    def to_aimspy(self, structure: AimspyStructure) -> AimspyMatrix:
        """Convert this source to aimspy standard format.

        Called from the ``python_func`` callback during ``run()``, after
        both ``AimspyInfo`` and ``CsrMatrixDescriptor`` are available.
        """
        ...
