"""Unit tests for AimspyStructure derived properties."""

from __future__ import annotations

import numpy as np

from aimspy.structure import AimspyStructure


class TestPhaseFactor:
    def test_phase_factor_values(self, mock_structure):
        """phase_factor: -1 if m>0 and m odd, else +1."""
        pf = mock_structure.phase_factor
        # basis_m = [0, 0, -1, 0, 0]
        # Expected: [1, 1, 1, 1, 1] (no positive odd m)
        assert pf.tolist() == [1, 1, 1, 1, 1]

    def test_phase_factor_with_positive_odd_m(self):
        s = AimspyStructure(
            n_atoms=1,
            n_basis=4,
            n_spin=1,
            lattice=np.eye(3),
            atom_symbols=["H"],
            atom_coords=np.zeros((1, 3)),
            basis_atom=np.array([0, 0, 0, 0], dtype=np.int32),
            basis_l=np.array([0, 1, 1, 1], dtype=np.int32),
            basis_m=np.array([0, -1, 0, 1], dtype=np.int32),
        )
        pf = s.phase_factor
        # m=0→1, m=-1→1, m=0→1, m=1→(1>0 and 1%2==1)→-1
        assert pf.tolist() == [1, 1, 1, -1]

    def test_phase_factor_dtype(self, mock_structure):
        assert mock_structure.phase_factor.dtype == np.int32

    def test_phase_factor_cached(self, mock_structure):
        """cached_property: same object returned on second access."""
        pf1 = mock_structure.phase_factor
        pf2 = mock_structure.phase_factor
        assert pf1 is pf2


class TestBasisSubidx:
    def test_basis_subidx_values(self, mock_structure):
        """basis_subidx[i] = position within its atom's basis functions."""
        subidx = mock_structure.basis_subidx
        # mock has basis_atom = [0, 0, 0, 1, 2] in aims order
        assert subidx.tolist() == [0, 1, 2, 0, 0]

    def test_basis_subidx_dtype(self, mock_structure):
        assert mock_structure.basis_subidx.dtype == np.int32

    def test_basis_subidx_cached(self, mock_structure):
        s1 = mock_structure.basis_subidx
        s2 = mock_structure.basis_subidx
        assert s1 is s2

    def test_basis_subidx_vs_naive(self):
        """Vectorized version must match a naive for-loop implementation."""
        basis_atom = np.array([2, 0, 2, 1, 2, 0, 1, 0], dtype=np.int32)
        n_atoms = 3
        n_basis = len(basis_atom)
        s = AimspyStructure(
            n_atoms=n_atoms,
            n_basis=n_basis,
            n_spin=1,
            lattice=np.eye(3),
            atom_symbols=["A", "B", "C"],
            atom_coords=np.zeros((n_atoms, 3)),
            basis_atom=basis_atom,
            basis_l=np.zeros(n_basis, dtype=np.int32),
            basis_m=np.zeros(n_basis, dtype=np.int32),
        )
        # Naive implementation
        expected = np.zeros(n_basis, dtype=np.int32)
        counter = np.zeros(n_atoms, dtype=np.int32)
        for i in range(n_basis):
            a = int(basis_atom[i])
            expected[i] = counter[a]
            counter[a] += 1
        np.testing.assert_array_equal(s.basis_subidx, expected)


class TestOrbitPerAtom:
    def test_orbit_per_atom_values(self, mock_structure):
        opa = mock_structure.orbit_per_atom
        # basis_atom = [0,0,0,1,2] → counts = [3, 1, 1]
        assert opa.tolist() == [3, 1, 1]

    def test_orbit_per_atom_sum(self, mock_structure):
        assert mock_structure.orbit_per_atom.sum() == mock_structure.n_basis

    def test_orbit_per_atom_dtype(self, mock_structure):
        assert mock_structure.orbit_per_atom.dtype == np.int32

    def test_orbit_per_atom_cached(self, mock_structure):
        o1 = mock_structure.orbit_per_atom
        o2 = mock_structure.orbit_per_atom
        assert o1 is o2


class TestAtomPermutation:
    def test_permutation_is_bijection(self, mock_structure):
        """old2new must be a valid permutation: all indices 0..n-1 appear."""
        old2new, new2old = mock_structure.atom_permutation
        assert sorted(old2new.tolist()) == list(range(mock_structure.n_atoms))
        # old2new[new2old[i]] == i
        for i in range(mock_structure.n_atoms):
            assert old2new[new2old[i]] == i

    def test_permutation_order(self, mock_structure):
        """POSCAR groups by element: Mo(0), S(1), S(2) → sorted stable = Mo,S,S."""
        old2new, new2old = mock_structure.atom_permutation
        # aims = [Mo, S, S] → POSCAR = [Mo, S, S] (already sorted)
        assert old2new.tolist() == [0, 1, 2]

    def test_permutation_reorders(self):
        """If aims order is [S, Mo, S], POSCAR should be [Mo, S, S]."""
        s = AimspyStructure(
            n_atoms=3,
            n_basis=3,
            n_spin=1,
            lattice=np.eye(3),
            atom_symbols=["S", "Mo", "S"],
            atom_coords=np.zeros((3, 3)),
            basis_atom=np.array([0, 1, 2], dtype=np.int32),
            basis_l=np.zeros(3, dtype=np.int32),
            basis_m=np.zeros(3, dtype=np.int32),
        )
        old2new, new2old = s.atom_permutation
        # aims[0]=S → POSCAR[1] or [2]; aims[1]=Mo → POSCAR[0]; aims[2]=S
        # Stable sort: Mo(idx1)→0, S(idx0)→1, S(idx2)→2
        assert old2new.tolist() == [1, 0, 2]
        assert new2old[0] == 1  # POSCAR[0] is aims[1] (Mo)
        assert new2old[1] == 0  # POSCAR[1] is aims[0] (S)
        assert new2old[2] == 2  # POSCAR[2] is aims[2] (S)

    def test_permutation_identity_same_species(self, mock_structure_same_species):
        old2new, new2old = mock_structure_same_species.atom_permutation
        assert old2new.tolist() == [0, 1, 2]

    def test_permutation_single_atom(self, mock_structure_single_atom):
        old2new, new2old = mock_structure_single_atom.atom_permutation
        assert old2new.tolist() == [0]
        assert new2old.tolist() == [0]

    def test_permutation_cached(self, mock_structure):
        p1 = mock_structure.atom_permutation
        p2 = mock_structure.atom_permutation
        # cached_property returns same object
        assert p1 is p2

    def test_build_atom_permutation_wrapper(self, mock_structure):
        """build_atom_permutation returns same as atom_permutation property."""
        bp = mock_structure.build_atom_permutation()
        ap = mock_structure.atom_permutation
        np.testing.assert_array_equal(bp[0], ap[0])
        np.testing.assert_array_equal(bp[1], ap[1])


class TestRepr:
    def test_repr(self, mock_structure):
        r = repr(mock_structure)
        assert "AimspyStructure" in r
        assert "n_atoms=3" in r
        assert "Mo" in r
