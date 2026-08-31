"""Helpers for scalar and small-array observables exported by FHI-aims."""

from __future__ import annotations

import numpy as np

from ._exceptions import AimspyConfigError


def get_free_atom_reference_energies(binding, n_species: int) -> np.ndarray:
    """Return an owned per-species radial-atom energy array in Hartree.

    The Fortran pointer is valid only while the initialized FHI-aims runtime
    owns its buffer, so this function always returns an independent copy.
    """
    if n_species <= 0:
        raise AimspyConfigError(
            f"free-atom reference energy requires n_species > 0, got {n_species}"
        )

    ptr = binding.aimspy_free_atom_reference_energies()
    if not ptr:
        raise AimspyConfigError(
            "free-atom reference energies are unavailable for the selected "
            "atomic-solver/XC path"
        )

    values = np.ctypeslib.as_array(ptr, shape=(n_species,))
    result = np.ascontiguousarray(values, dtype=np.float64).copy()
    if not np.all(np.isfinite(result)):
        raise AimspyConfigError("free-atom reference energies must all be finite")
    return result
