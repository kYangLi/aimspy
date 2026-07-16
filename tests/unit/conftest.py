"""Shared fixtures for unit tests (no MPI, no libaims).

Provides:
- ``mock_structure``: a minimal AimspyStructure for a 3-atom MoS2-like system.
- ``mock_csr_descr``: a tiny CsrMatrixDescriptor matching the structure.
- ``mock_matrix``: an AimspyMatrix built from the CSR descriptor.
"""

from __future__ import annotations

import numpy as np
import pytest

from aimspy.structure import AimspyStructure


@pytest.fixture
def mock_structure():
    """3-atom MoS2-like structure: 1 Mo + 2 S, 5 basis functions."""
    return AimspyStructure(
        n_atoms=3,
        n_basis=5,
        n_spin=1,
        n_periodic=3,
        lattice=np.eye(3) * 10.0,
        atom_symbols=["Mo", "S", "S"],
        atom_coords=np.array([[0.0, 0.0, 0.0], [1.5, 1.5, 0.0], [1.5, 1.5, 3.0]]),
        basis_atom=np.array([0, 0, 0, 1, 2], dtype=np.int32),
        basis_l=np.array([0, 0, 1, 0, 0], dtype=np.int32),
        basis_m=np.array([0, 0, -1, 0, 0], dtype=np.int32),
    )


@pytest.fixture
def mock_structure_same_species():
    """3-atom structure all same species (identical permutation expected)."""
    return AimspyStructure(
        n_atoms=3,
        n_basis=3,
        n_spin=1,
        n_periodic=3,
        lattice=np.eye(3) * 10.0,
        atom_symbols=["O", "O", "O"],
        atom_coords=np.zeros((3, 3)),
        basis_atom=np.array([0, 1, 2], dtype=np.int32),
        basis_l=np.array([0, 0, 0], dtype=np.int32),
        basis_m=np.array([0, 0, 0], dtype=np.int32),
    )


@pytest.fixture
def mock_structure_single_atom():
    """Single-atom structure (edge case)."""
    return AimspyStructure(
        n_atoms=1,
        n_basis=1,
        n_spin=1,
        n_periodic=3,
        lattice=np.eye(3) * 10.0,
        atom_symbols=["H"],
        atom_coords=np.zeros((1, 3)),
        basis_atom=np.array([0], dtype=np.int32),
        basis_l=np.array([0], dtype=np.int32),
        basis_m=np.array([0], dtype=np.int32),
    )
