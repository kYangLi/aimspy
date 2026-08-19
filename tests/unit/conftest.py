"""Shared fixtures for unit tests (no MPI, no libaims).

Provides:
- ``mock_structure``: a minimal AimspyStructure for a 3-atom MoS2-like system.
- ``mock_structure_same_species``: a 3-atom structure all same species.
- ``mock_structure_single_atom``: a single-atom structure (edge case).
- ``make_grid``: factory producing deterministic mock GridData objects
  (lifted from the former test_viz.py local helper; shared by the
  GridData-viz and CLI tests).
- ``mock_grid``: a default ``make_grid()`` instance.
"""

from __future__ import annotations

import numpy as np
import pytest

from aimspy.grid_data import GridData
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


def make_grid(n=60, n_atoms=2, seed=1, with_structure=False):
    """Deterministic mock GridData (bohr coords; rho>0; signed delta fields).

    Pure helper (no pytest dependency) so both fixtures and modules that
    prefer plain imports can use it.
    """
    rng = np.random.default_rng(seed)
    coords = (rng.random((3, n)) - 0.5) * 8.0  # bohr, centred
    rho = (rng.random((1, n)) + 0.05) * 10.0
    vh = -(rng.random(n) + 0.5)
    vh0 = -(rng.random(n) + 0.5)
    vxc = -np.abs(rng.random((1, n))) - 0.01
    vxc0 = -np.abs(rng.random((1, n))) - 0.01
    kwargs = {}
    if with_structure:
        kwargs = dict(
            atom_coords=rng.random((n_atoms, 3)) * 5.0,
            atom_symbols=[f"El{a}" for a in range(n_atoms)],
            lattice=np.eye(3) * 10.0,
        )
    return GridData(
        n_full_points=n,
        n_spin=1,
        n_atoms=n_atoms,
        coords=coords,
        partition_tab=np.full(n, 0.1),
        index_atom=np.arange(n, dtype=np.int32) % n_atoms,
        index_radial=np.arange(n, dtype=np.int32),
        index_angular=np.arange(n, dtype=np.int32),
        rho=rho,
        vks=vh[np.newaxis, :] + vxc,
        vks0=vh0[np.newaxis, :] + vxc0,
        vh=vh,
        vh0=vh0,
        rho0=(rng.random(n) + 0.05) * 4.0 * np.pi,
        **kwargs,
    )


@pytest.fixture
def make_grid_factory():
    """Factory fixture wrapping :func:`make_grid` (customizable per test)."""
    return make_grid


@pytest.fixture
def mock_grid():
    """A default deterministic mock GridData (see :func:`make_grid`)."""
    return make_grid()


@pytest.fixture(autouse=True)
def _close_matplotlib_figures():
    """Close all matplotlib figures after each test (prevents the
    cumulative open-figure warning / memory growth across the session).

    Only acts when pyplot was already imported by the test itself.
    """
    import sys

    yield
    if "matplotlib.pyplot" in sys.modules:
        import matplotlib.pyplot as plt

        plt.close("all")
