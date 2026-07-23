"""Unit tests for DeepHData construction, I/O, and conversion."""

from __future__ import annotations

import json
import numpy as np
import pytest

from aimspy import DeepHData
from aimspy.interface.deeph.data import (
    _build_elements_orbital_map,
    _compute_n_basis,
    _compute_occupation,
)
from aimspy.structure import AimspyStructure
from aimspy.matrix import AimspyMatrix


# =============================================================================
# Helpers
# =============================================================================
def _make_mock_structure():
    """3-atom MoS2-like structure for conversion tests."""
    return AimspyStructure(
        n_atoms=3,
        n_basis=5,
        n_spin=1,
        n_periodic=3,
        lattice=np.eye(3) * 10.0,
        atom_symbols=["Mo", "S", "S"],
        atom_coords=np.array([[0, 0, 0], [1.5, 1.5, 0], [1.5, 1.5, 3.0]]),
        basis_atom=np.array([0, 0, 0, 1, 2], dtype=np.int32),
        basis_l=np.array([0, 0, 1, 0, 0], dtype=np.int32),
        basis_m=np.array([0, 0, -1, 0, 0], dtype=np.int32),
    )


def _make_simple_blocks():
    """Minimal block dict: 2 pairs, 1x1 each."""
    return {
        (0, 0, 0, 0, 0): np.array([[1.0]]),  # Mo-Mo R=0
        (0, 0, 0, 0, 1): np.array([[0.5]]),  # Mo-S R=0
    }


# =============================================================================
# Tests: from_memory
# =============================================================================
class TestFromMemory:
    def test_from_memory_hamiltonian_only(self):
        blocks = _make_simple_blocks()
        dd = DeepHData.from_memory(
            lattice=np.eye(3) * 10.0,
            atom_symbols=["Mo", "S", "S"],
            atom_coords=np.zeros((3, 3)),
            elements_orbital_map={"Mo": [0, 0, 1], "S": [0, 0]},
            hamiltonian_blocks=blocks,
        )
        assert dd.n_atoms == 3
        assert dd.n_pairs == 2
        assert dd.entries is not None
        assert dd.overlap_entries is None
        assert dd.initial_hamiltonian_entries is None
        # Hamiltonian converted Hartree→eV
        assert dd.entries[0] == pytest.approx(27.2113845)

    def test_from_memory_all_matrices(self):
        h_blocks = {(0, 0, 0, 0, 0): np.array([[1.0]])}
        s_blocks = {(0, 0, 0, 0, 0): np.array([[0.5]])}
        h0_blocks = {(0, 0, 0, 0, 0): np.array([[0.1]])}
        dd = DeepHData.from_memory(
            lattice=np.eye(3) * 10.0,
            atom_symbols=["Mo", "S", "S"],
            atom_coords=np.zeros((3, 3)),
            elements_orbital_map={"Mo": [0, 0, 1], "S": [0, 0]},
            hamiltonian_blocks=h_blocks,
            overlap_blocks=s_blocks,
            initial_hamiltonian_blocks=h0_blocks,
        )
        assert dd.entries is not None
        assert dd.overlap_entries is not None
        assert dd.initial_hamiltonian_entries is not None
        # Overlap is dimensionless (no unit conversion)
        assert dd.overlap_entries[0] == pytest.approx(0.5)

    def test_from_memory_no_blocks_raises(self):
        from aimspy import AimspyConfigError

        with pytest.raises(AimspyConfigError, match="At least one"):
            DeepHData.from_memory(
                lattice=np.eye(3),
                atom_symbols=["H"],
                atom_coords=np.zeros((1, 3)),
                elements_orbital_map={"H": [0]},
            )

    def test_from_memory_n_basis_auto(self):
        """n_basis=0 → computed from elements_orbital_map."""
        blocks = {(0, 0, 0, 0, 0): np.array([[1.0]])}
        dd = DeepHData.from_memory(
            lattice=np.eye(3),
            atom_symbols=["Mo"],
            atom_coords=np.zeros((1, 3)),
            elements_orbital_map={"Mo": [0, 0, 1, 2, 3]},
            hamiltonian_blocks=blocks,
            n_basis=0,
        )
        # Mo has 5 shells: (2l+1) for each: 1+1+3+5+7 = 17
        assert dd.n_basis == 17

    def test_from_memory_empty_hamiltonian_dict_overlap_only(self):
        """Empty hamiltonian_blocks={} should NOT store overlap as Hamiltonian."""
        s_blocks = {(0, 0, 0, 0, 0): np.array([[0.5]])}
        dd = DeepHData.from_memory(
            lattice=np.eye(3),
            atom_symbols=["H"],
            atom_coords=np.zeros((1, 3)),
            elements_orbital_map={"H": [0]},
            hamiltonian_blocks={},
            overlap_blocks=s_blocks,
        )
        assert dd.entries is None
        assert dd.overlap_entries is not None
        assert dd.overlap_entries[0] == pytest.approx(0.5)

    def test_from_memory_empty_overlap_dict(self):
        """Empty overlap_blocks={} should produce overlap_entries=None."""
        h_blocks = {(0, 0, 0, 0, 0): np.array([[1.0]])}
        dd = DeepHData.from_memory(
            lattice=np.eye(3),
            atom_symbols=["H"],
            atom_coords=np.zeros((1, 3)),
            elements_orbital_map={"H": [0]},
            hamiltonian_blocks=h_blocks,
            overlap_blocks={},
        )
        assert dd.entries is not None
        assert dd.overlap_entries is None

    def test_from_memory_empty_init_ham_dict(self):
        """Empty initial_hamiltonian_blocks={} should produce initial_hamiltonian_entries=None."""
        h_blocks = {(0, 0, 0, 0, 0): np.array([[1.0]])}
        dd = DeepHData.from_memory(
            lattice=np.eye(3),
            atom_symbols=["H"],
            atom_coords=np.zeros((1, 3)),
            elements_orbital_map={"H": [0]},
            hamiltonian_blocks=h_blocks,
            initial_hamiltonian_blocks={},
        )
        assert dd.entries is not None
        assert dd.initial_hamiltonian_entries is None


# =============================================================================
# Tests: set_* methods
# =============================================================================
class TestSetMethods:
    def test_set_hamiltonian(self):
        """set_hamiltonian converts and stores entries."""
        blocks = _make_simple_blocks()
        dd = DeepHData.from_memory(
            lattice=np.eye(3) * 10.0,
            atom_symbols=["Mo", "S", "S"],
            atom_coords=np.zeros((3, 3)),
            elements_orbital_map={"Mo": [0, 0, 1], "S": [0, 0]},
            hamiltonian_blocks=blocks,
        )
        struct = _make_mock_structure()
        mx = AimspyMatrix(blocks=blocks)
        dd.set_hamiltonian(mx, struct)
        assert dd.entries is not None
        # Should match original (Hartree→eV)
        assert dd.entries[0] == pytest.approx(27.2113845)

    def test_set_overlap(self):
        """set_overlap stores dimensionless overlap."""
        blocks = {(0, 0, 0, 0, 0): np.array([[0.7]])}
        dd = DeepHData.from_memory(
            lattice=np.eye(3) * 10.0,
            atom_symbols=["Mo", "S", "S"],
            atom_coords=np.zeros((3, 3)),
            elements_orbital_map={"Mo": [0, 0, 1], "S": [0, 0]},
            overlap_blocks=blocks,
        )
        struct = _make_mock_structure()
        mx = AimspyMatrix(blocks=blocks)
        dd.set_overlap(mx, struct)
        assert dd.overlap_entries is not None
        assert dd.overlap_entries[0] == pytest.approx(0.7)


# =============================================================================
# Tests: save / load roundtrip
# =============================================================================
class TestSaveLoad:
    def test_save_load_roundtrip(self, tmp_path):
        blocks = _make_simple_blocks()
        dd = DeepHData.from_memory(
            lattice=np.eye(3) * 10.0,
            atom_symbols=["Mo", "S", "S"],
            atom_coords=np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
            elements_orbital_map={"Mo": [0, 0, 1], "S": [0, 0]},
            hamiltonian_blocks=blocks,
            path=tmp_path,
        )
        dd.save()
        assert (tmp_path / "POSCAR").exists()
        assert (tmp_path / "info.json").exists()
        assert (tmp_path / "hamiltonian.h5").exists()

        dd2 = DeepHData.from_directory(tmp_path)
        assert dd2.n_atoms == 3
        assert dd2.atom_symbols == ["Mo", "S", "S"]
        np.testing.assert_allclose(dd2.entries, dd.entries)

    def test_save_metadata_only(self, tmp_path):
        """save_metadata writes POSCAR + info.json but not h5."""
        dd = DeepHData.from_memory(
            lattice=np.eye(3) * 10.0,
            atom_symbols=["Mo", "S", "S"],
            atom_coords=np.zeros((3, 3)),
            elements_orbital_map={"Mo": [0, 0, 1], "S": [0, 0]},
            hamiltonian_blocks=_make_simple_blocks(),
        )
        dd.save_metadata(tmp_path)
        assert (tmp_path / "POSCAR").exists()
        assert (tmp_path / "info.json").exists()
        assert not (tmp_path / "hamiltonian.h5").exists()

    def test_info_json_content(self, tmp_path):
        dd = DeepHData.from_memory(
            lattice=np.eye(3) * 10.0,
            atom_symbols=["Mo", "S", "S"],
            atom_coords=np.zeros((3, 3)),
            elements_orbital_map={"Mo": [0, 0, 1], "S": [0, 0]},
            hamiltonian_blocks=_make_simple_blocks(),
        )
        dd.save_metadata(tmp_path)
        with open(tmp_path / "info.json") as f:
            info = json.load(f)
        assert info["atoms_quantity"] == 3
        assert info["orbits_quantity"] > 0
        assert info["occupation"] > 0  # Mo(42) + S(16)*2 = 74
        assert "Mo" in info["elements_orbital_map"]
        assert "S" in info["elements_orbital_map"]


# =============================================================================
# Tests: to_aimspy conversion
# =============================================================================
class TestToAimspy:
    def test_to_aimspy_basic(self):
        blocks = _make_simple_blocks()
        dd = DeepHData.from_memory(
            lattice=np.eye(3) * 10.0,
            atom_symbols=["Mo", "S", "S"],
            atom_coords=np.zeros((3, 3)),
            elements_orbital_map={"Mo": [0, 0, 1], "S": [0, 0]},
            hamiltonian_blocks=blocks,
        )
        struct = _make_mock_structure()
        mx = dd.to_aimspy(struct)
        assert mx.n_spin == 1
        assert mx.n_pairs > 0

    def test_to_aimspy_no_entries_raises(self):
        from aimspy import AimspyConfigError

        dd = DeepHData(
            lattice=np.eye(3),
            atom_symbols=["Mo", "S", "S"],
            atom_coords=np.zeros((3, 3)),
            elements_orbital_map={"Mo": [0], "S": [0]},
            n_basis=2,
            atom_pairs=np.zeros((1, 5), dtype=np.int32),
            chunk_boundaries=np.array([0, 1], dtype=np.int32),
            chunk_shapes=np.array([[1, 1]], dtype=np.int32),
            entries=None,
        )
        struct = _make_mock_structure()
        with pytest.raises(AimspyConfigError, match="No Hamiltonian"):
            dd.to_aimspy(struct)


# =============================================================================
# Tests: error cases
# =============================================================================
class TestErrors:
    def test_require_path_raises_config_error(self):
        from aimspy import AimspyConfigError

        dd = DeepHData(
            lattice=np.eye(3),
            atom_symbols=["H"],
            atom_coords=np.zeros((1, 3)),
            elements_orbital_map={"H": [0]},
            n_basis=1,
            atom_pairs=np.zeros((1, 5), dtype=np.int32),
            chunk_boundaries=np.array([0, 1], dtype=np.int32),
            chunk_shapes=np.array([[1, 1]], dtype=np.int32),
            entries=np.array([1.0]),
            path=None,
        )
        with pytest.raises(AimspyConfigError, match="No path"):
            dd._require_path()

    def test_from_directory_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            DeepHData.from_directory(tmp_path / "nonexistent")


# =============================================================================
# Tests: helper functions
# =============================================================================
class TestHelpers:
    def test_compute_occupation(self):
        assert _compute_occupation(["Mo", "S", "S"]) == 42 + 16 + 16

    def test_compute_n_basis(self):
        eom = {"Mo": [0, 0, 1], "S": [0, 0]}
        # Mo: (2×1)+(2×1)+(2×3)=2+2+6=10; S: (2×1)+(2×1)=2+2=4
        # Mo+S+S = 10+4+4 = 18
        # But wait: _compute_n_basis uses Counter(atom_symbols) and
        # computes cnt * sum(2*l+1 for l in shells).
        # Mo count=1, shells=[0,0,1] → sum(1+1+3)=5, so 1×5=5
        # S count=2, shells=[0,0] → sum(1+1)=2, so 2×2=4
        # Total = 5+4 = 9
        assert _compute_n_basis(["Mo", "S", "S"], eom) == 9

    def test_build_elements_orbital_map(self):
        struct = _make_mock_structure()
        eom = _build_elements_orbital_map(struct)
        assert "Mo" in eom
        assert "S" in eom
