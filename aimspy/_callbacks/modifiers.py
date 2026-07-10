"""Private — Hamiltonian modification strategies.

Each strategy is a Python callable with the fixed signature::

    (live_h_view: np.ndarray, n_ham: int, n_spin: int, aux: dict) -> None

Where ``live_h_view`` is a **mutable view** onto the live Fortran
Hamiltonian array — in‑place modifications directly affect the Fortran
memory through the ``modify_h0`` callback.
"""
from __future__ import annotations

from typing import Callable

import numpy as np


class HamiltonianModifier:
    """Collection of strategies for the ``modify_h0`` callback.

    All strategies can be used as ``aux['strategy']``; the
    ``ModifyH0Default`` callback dispatches to the stored strategy.
    """

    # ------------------------------------------------------------------
    @staticmethod
    def replace(live_h_view: np.ndarray, n_ham: int, n_spin: int,
                aux: dict) -> None:
        """Replace the entire Hamiltonian with ``aux['hamiltonian']`` or
        ``aux['hamiltonian_aims']`` (whichever is present).

        The array must be C‑contiguous with shape ``(n_spin, n_ham_size)``.
        The ``[:]`` assignment preserves the Fortran memory binding.
        """
        new_h = aux.get('hamiltonian') or aux.get('hamiltonian_aims')
        if new_h is None:
            raise ValueError(
                "replace strategy: aux must contain 'hamiltonian' or "
                "'hamiltonian_aims'"
            )
        live_h_view[:] = new_h

    # ------------------------------------------------------------------
    @staticmethod
    def add(live_h_view: np.ndarray, n_ham: int, n_spin: int,
            aux: dict) -> None:
        """Add ``aux['delta_hamiltonian']`` (in‑place) to the live H."""
        live_h_view += aux['delta_hamiltonian']

    # ------------------------------------------------------------------
    @staticmethod
    def scale(live_h_view: np.ndarray, n_ham: int, n_spin: int,
              aux: dict) -> None:
        """Multiply the live H by ``aux['factor']`` (in‑place)."""
        live_h_view *= aux['factor']

    # ------------------------------------------------------------------
    @staticmethod
    def custom(
        fn: Callable[[np.ndarray, int, int, dict], None],
    ) -> Callable[[np.ndarray, int, int, dict], None]:
        """Wrap a user‑provided function into a strategy.

        The wrapped function has the identical signature to the strategy
        itself.  Use this when you need a one‑off modifier that doesn't
        fit the ``replace`` / ``add`` / ``scale`` templates.

        Examples
        --------
        >>> def my_mod(live_h, n_ham, n_spin, aux):
        ...     live_h[0, :n_ham//2] = aux['external'][0, :n_ham//2]
        >>> calc.modify_hamiltonian_fn = HamiltonianModifier.custom(my_mod)
        """
        def wrapper(live_h_view: np.ndarray, n_ham: int, n_spin: int,
                    aux: dict) -> None:
            fn(live_h_view, n_ham, n_spin, aux)
        return wrapper
