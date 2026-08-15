"""Unit tests for GridData (no MPI, no libaims).

Covers derived quantities, unit conversions, npz round-trip, 0-based
indexing, and gather concatenation logic (mock comm).
"""

from __future__ import annotations

import numpy as np
import pytest

from aimspy.grid_data import GridData
from aimspy.data import HARTREE_TO_EV, BOHR_TO_ANG


def _make_grid(n=5, n_spin=1, n_atoms=2, seed=0, with_structure=False):
    """Deterministic small GridData for tests (rho0 == rho_free, no 4*pi)."""
    rng = np.random.default_rng(seed)
    coords = rng.random((3, n))
    ptab = np.full(n, 0.5)
    rho = rng.random((n_spin, n)) + 0.1
    vh = -rng.random(n) - 1.0
    vh0 = -rng.random(n) - 1.0
    # v_xc = vks - vh  <= 0 by construction (LDA exchange sign)
    vxc = -np.abs(rng.random((n_spin, n))) - 0.05
    vxc0 = -np.abs(rng.random((n_spin, n))) - 0.05
    vks = vh[np.newaxis, :] + vxc
    vks0 = vh0[np.newaxis, :] + vxc0
    rho0 = rng.random(n) + 0.1  # rho0 IS rho_free (4*pi already removed)
    kwargs = {}
    if with_structure:
        kwargs = dict(
            atom_coords=rng.random((n_atoms, 3)) * 5.0,
            atom_symbols=[f"El{a}" for a in range(n_atoms)],
            lattice=np.eye(3) * 10.0,
        )
    return GridData(
        n_full_points=n,
        n_spin=n_spin,
        n_atoms=n_atoms,
        coords=coords,
        partition_tab=ptab,
        index_atom=np.arange(n, dtype=np.int32) % n_atoms,
        index_radial=np.arange(n, dtype=np.int32),
        index_angular=np.arange(n, dtype=np.int32),
        rho=rho,
        vks=vks,
        vks0=vks0,
        vh=vh,
        vh0=vh0,
        rho0=rho0,
        **kwargs,
    )


class TestDerivedQuantities:
    def test_rho_free_equals_rho0(self):
        g = _make_grid()
        # rho0 IS rho_free (4*pi removed at import time)
        np.testing.assert_array_equal(g.rho_free, g.rho0)

    def test_delta_rho_spin1(self):
        g = _make_grid(n_spin=1)
        np.testing.assert_allclose(g.delta_rho, g.rho - g.rho_free[np.newaxis, :])

    def test_delta_vks(self):
        g = _make_grid()
        np.testing.assert_allclose(g.delta_vks, g.vks - g.vks0)

    def test_delta_vh(self):
        g = _make_grid()
        np.testing.assert_allclose(g.delta_vh, g.vh - g.vh0)

    def test_vxc_equals_vks_minus_vh(self):
        g = _make_grid()
        np.testing.assert_allclose(g.vxc, g.vks - g.vh[np.newaxis, :])
        # LDA exchange potential must be <= 0
        assert np.all(g.vxc <= 0.0)

    def test_vxc0_equals_vks0_minus_vh0(self):
        g = _make_grid()
        np.testing.assert_allclose(g.vxc0, g.vks0 - g.vh0[np.newaxis, :])


class TestUnitConversion:
    def test_coords_ang(self):
        g = _make_grid()
        np.testing.assert_allclose(g.coords_ang, g.coords * BOHR_TO_ANG)

    def test_potentials_ev(self):
        g = _make_grid()
        np.testing.assert_allclose(g.vks_ev, g.vks * HARTREE_TO_EV)
        np.testing.assert_allclose(g.vks0_ev, g.vks0 * HARTREE_TO_EV)
        np.testing.assert_allclose(g.vh_ev, g.vh * HARTREE_TO_EV)
        np.testing.assert_allclose(g.vh0_ev, g.vh0 * HARTREE_TO_EV)


class TestIndexing:
    def test_index_atom_zero_based(self):
        g = _make_grid(n=6, n_atoms=3)
        assert g.index_atom.min() == 0
        assert g.index_atom.max() == 2

    def test_integrated_electrons(self):
        g = _make_grid(n=4, n_spin=1)
        expected = float(np.sum(g.partition_tab * g.rho[0]))
        assert g.integrated_electrons() == pytest.approx(expected)


class TestNpzRoundTrip:
    def test_save_load_roundtrip(self, tmp_path):
        g = _make_grid(n=7, n_spin=1)
        p = g.save_npz(tmp_path / "grid.npz")
        g2 = GridData.load_npz(p)
        assert g2.n_full_points == g.n_full_points
        assert g2.n_spin == g.n_spin
        for key in GridData._NPZ_KEYS:
            np.testing.assert_array_equal(getattr(g2, key), getattr(g, key))

    def test_save_creates_file(self, tmp_path):
        g = _make_grid()
        p = g.save_npz(tmp_path / "out.npz")
        assert p.is_file()

    def test_structure_roundtrip(self, tmp_path):
        g = _make_grid(n=6, n_atoms=3, with_structure=True)
        p = g.save_npz(tmp_path / "grid_struct.npz")
        g2 = GridData.load_npz(p)
        np.testing.assert_array_equal(g2.atom_coords, g.atom_coords)
        np.testing.assert_array_equal(g2.lattice, g.lattice)
        assert list(g2.atom_symbols) == list(g.atom_symbols)

    def test_structure_optional(self, tmp_path):
        g = _make_grid(n=5)  # no structure
        p = g.save_npz(tmp_path / "grid_nostruct.npz")
        g2 = GridData.load_npz(p)
        assert g2.atom_coords is None
        assert g2.atom_symbols is None
        assert g2.lattice is None


class _FakeCommGatherv:
    """Mock mpi4py communicator that implements allgather + Gatherv.

    Simulates a multi-rank gather by tracking Gatherv call order and
    filling the receive buffer with the pre-concatenated global array.
    """

    # Gatherv call order in GridData.gather
    _CALL_ORDER = [
        "coords",
        "partition_tab",
        "index_atom",
        "index_radial",
        "index_angular",
        "rho",
        "vks",
        "vks0",
        "vh",
        "vh0",
        "rho0",
    ]

    def __init__(self, grids):
        """grids: list of GridData, one per simulated rank."""
        self._grids = grids
        self.rank = 0  # simulate root
        self._counts = [g.n_full_points for g in grids]
        self._n_total = sum(self._counts)
        self._call_idx = 0

    def allgather(self, value):
        # For n (grid points): return each rank's actual count
        # For n_spin/n_atoms: return the same value (consistent)
        if isinstance(value, int) and value == self._grids[0].n_full_points:
            return self._counts
        return [value] * len(self._grids)

    def Gatherv(self, sendbuf, recvbuf_spec, root=0):
        recvbuf, counts, displs, mpi_type = recvbuf_spec
        if recvbuf is None:
            return

        # Identify which array by call order
        attr = self._CALL_ORDER[self._call_idx]
        self._call_idx += 1

        # Concatenate along the point axis
        arrays = [getattr(g, attr) for g in self._grids]
        if arrays[0].ndim == 2:
            global_arr = np.concatenate(arrays, axis=1)
            # Flatten to (n_total * leading_dim,) in C-order
            recvbuf[:] = np.ascontiguousarray(global_arr.T).ravel()
        else:
            global_arr = np.concatenate(arrays)
            recvbuf[:] = global_arr


class TestGather:
    def test_gather_single_rank(self):
        g = _make_grid(n=5)
        out = GridData.gather(g, _FakeCommGatherv([g]), root=0)
        assert out is not None
        assert out.n_full_points == 5
        np.testing.assert_array_equal(out.coords, g.coords)
        np.testing.assert_array_equal(out.rho, g.rho)

    def test_gather_concatenates(self):
        g1 = _make_grid(n=3, seed=1)
        g2 = _make_grid(n=4, seed=2)
        out = GridData.gather(g1, _FakeCommGatherv([g1, g2]), root=0)
        assert out.n_full_points == 7
        assert out.coords.shape == (3, 7)
        assert out.rho.shape == (1, 7)
        # first 3 columns from g1, next 4 from g2
        np.testing.assert_array_equal(out.coords[:, :3], g1.coords)
        np.testing.assert_array_equal(out.coords[:, 3:], g2.coords)

    def test_gather_inconsistent_spin_raises(self):
        g1 = _make_grid(n=3, n_spin=1)

        class InconsistentComm:
            rank = 0

            def allgather(self, value):
                # Return inconsistent n_spin
                return [1, 2]

            def Gatherv(self, sendbuf, recvbuf_spec, root=0):
                pass

        with pytest.raises(ValueError, match="inconsistent"):
            GridData.gather(g1, InconsistentComm(), root=0)

    def test_gather_keeps_structure(self):
        g = _make_grid(n=4, n_atoms=2, with_structure=True)
        out = GridData.gather(g, _FakeCommGatherv([g]), root=0)
        np.testing.assert_array_equal(out.atom_coords, g.atom_coords)
        assert list(out.atom_symbols) == list(g.atom_symbols)
        np.testing.assert_array_equal(out.lattice, g.lattice)


class TestRepr:
    def test_repr(self):
        g = _make_grid(n=10, n_atoms=3)
        s = repr(g)
        assert "n=10" in s and "n_atoms=3" in s
