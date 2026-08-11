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


# =============================================================================
# Tests: force / energy (MD-style force.h5)
# =============================================================================
def _make_mock_structure_unsorted():
    """3-atom MoS2-like structure with NON-identity permutation.

    aims order: ['S', 'Mo', 'S'] → POSCAR order: ['Mo', 'S', 'S']
    new2old = [1, 0, 2], old2new = [1, 0, 2]
    (coincidentally equal here, but this still tests non-trivial reordering)
    """
    return AimspyStructure(
        n_atoms=3,
        n_basis=5,
        n_spin=1,
        n_periodic=3,
        lattice=np.eye(3) * 10.0,
        atom_symbols=["S", "Mo", "S"],  # non-identity vs POSCAR ['Mo','S','S']
        atom_coords=np.array([[1.5, 1.5, 0], [0, 0, 0], [1.5, 1.5, 3.0]]),
        basis_atom=np.array([1, 1, 1, 0, 2], dtype=np.int32),
        basis_l=np.array([0, 0, 1, 0, 0], dtype=np.int32),
        basis_m=np.array([0, 0, -1, 0, 0], dtype=np.int32),
    )


class TestForce:
    def test_save_load_force_roundtrip(self, tmp_path):
        """save → from_directory roundtrip: force + energy preserved."""
        force = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        dd = DeepHData.from_memory(
            lattice=np.eye(3) * 10.0,
            atom_symbols=["Mo", "S", "S"],
            atom_coords=np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
            elements_orbital_map={"Mo": [0, 0, 1], "S": [0, 0]},
            hamiltonian_blocks=_make_simple_blocks(),
            force=force,
            energy_eV=-123.45,
            path=tmp_path,
        )
        dd.save()
        assert (tmp_path / "force.h5").exists()

        dd2 = DeepHData.from_directory(tmp_path)
        assert dd2.force is not None
        np.testing.assert_allclose(dd2.force, force)
        assert dd2.energy_eV == pytest.approx(-123.45)

    def test_from_aimspy_with_force_reorder(self):
        """from_aimspy reorders force from aims → POSCAR order."""
        struct = _make_mock_structure_unsorted()
        # aims order: [S, Mo, S], POSCAR order: [Mo, S, S]
        # force_aims = [[F_S1], [F_Mo], [F_S2]]
        force_aims = np.array(
            [[1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]]  # S (aims idx 0)  # Mo (aims idx 1)
        )  # S (aims idx 2)
        blocks = _make_simple_blocks()  # uses aims atom indices

        dd = DeepHData.from_aimspy(
            structure=struct,
            hamiltonian=AimspyMatrix(blocks=blocks, n_spin=1),
            force=force_aims,
            energy=-1.0,  # Hartree
        )
        # POSCAR order is [Mo, S, S], so force should be [2, 1, 3]
        assert dd.force is not None
        np.testing.assert_allclose(dd.force[:, 0], [2.0, 1.0, 3.0])
        # energy: -1.0 Hartree → eV
        assert dd.energy_eV == pytest.approx(-1.0 * 27.2113845)

    def test_set_force_with_energy(self):
        """set_force reorders + converts energy Hartree→eV."""
        struct = _make_mock_structure_unsorted()
        dd = DeepHData.from_aimspy(
            structure=struct,
            hamiltonian=AimspyMatrix(blocks=_make_simple_blocks(), n_spin=1),
        )
        assert dd.force is None
        assert dd.energy_eV is None

        force_aims = np.array(
            [[1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]]  # S (aims 0)  # Mo (aims 1)
        )  # S (aims 2)
        dd.set_force(force_aims, structure=struct, energy=-0.5)  # Hartree

        # POSCAR order [Mo, S, S] → [2, 1, 3]
        np.testing.assert_allclose(dd.force[:, 0], [2.0, 1.0, 3.0])
        assert dd.energy_eV == pytest.approx(-0.5 * 27.2113845)

    def test_set_force_without_energy(self):
        """set_force with energy=None leaves energy_eV unset."""
        struct = _make_mock_structure_unsorted()
        dd = DeepHData.from_aimspy(
            structure=struct,
            hamiltonian=AimspyMatrix(blocks=_make_simple_blocks(), n_spin=1),
        )
        dd.set_force(np.zeros((3, 3)), structure=struct)
        assert dd.force is not None
        assert dd.energy_eV is None

    def test_save_force_without_force_raises(self, tmp_path):
        """save_force raises AimspyConfigError when force is None."""
        from aimspy import AimspyConfigError

        dd = DeepHData.from_memory(
            lattice=np.eye(3) * 10.0,
            atom_symbols=["Mo", "S", "S"],
            atom_coords=np.zeros((3, 3)),
            elements_orbital_map={"Mo": [0, 0, 1], "S": [0, 0]},
            hamiltonian_blocks=_make_simple_blocks(),
            path=tmp_path,
        )
        with pytest.raises(AimspyConfigError, match="No force"):
            dd.save_force()

    def test_force_h5_format_matches_example(self, tmp_path):
        """Written force.h5 has exact dataset/attr layout matching reference."""
        import h5py

        force = np.zeros((12, 3), dtype=np.float64)
        lattice = np.eye(3) * 30.0
        atom_symbols = ["C"] * 12  # benzene-like
        dd = DeepHData.from_memory(
            lattice=lattice,
            atom_symbols=atom_symbols,
            atom_coords=np.zeros((12, 3)),
            elements_orbital_map={"C": [0, 0, 1]},
            hamiltonian_blocks={(0, 0, 0, 0, 0): np.array([[1.0]])},
            force=force,
            energy_eV=-0.073,
            path=tmp_path,
        )
        dd.save_force()

        with h5py.File(tmp_path / "force.h5", "r") as f:
            # Datasets
            assert set(f.keys()) == {"cell", "energy", "force", "stress"}
            assert f["cell"].shape == (3, 3)
            assert f["cell"].dtype == np.float64
            assert f["energy"].shape == ()
            assert f["energy"].dtype == np.float64
            assert f["force"].shape == (12, 3)
            assert f["force"].dtype == np.float64
            assert f["stress"].shape == (6,)
            assert f["stress"].dtype == np.float64
            np.testing.assert_allclose(f["stress"][:], np.zeros(6))
            # Attrs
            assert f.attrs["formula"] == b"X12"
            assert int(f.attrs["natoms"]) == 12
            # Energy value
            assert float(f["energy"][()]) == pytest.approx(-0.073)

    def test_force_h5_energy_defaults_to_zero(self, tmp_path):
        """When energy_eV is None, force.h5 energy dataset is 0.0."""
        import h5py

        dd = DeepHData.from_memory(
            lattice=np.eye(3) * 10.0,
            atom_symbols=["Mo", "S", "S"],
            atom_coords=np.zeros((3, 3)),
            elements_orbital_map={"Mo": [0, 0, 1], "S": [0, 0]},
            hamiltonian_blocks=_make_simple_blocks(),
            force=np.ones((3, 3)),
            energy_eV=None,
            path=tmp_path,
        )
        dd.save_force()
        with h5py.File(tmp_path / "force.h5", "r") as f:
            assert float(f["energy"][()]) == 0.0

    def test_from_directory_force_shape_mismatch(self, tmp_path):
        """force.h5 with wrong atom count raises AimspyConfigError."""
        import h5py

        from aimspy import AimspyConfigError

        # First save a valid DeepH dir (3 atoms) with a matrix file
        dd = DeepHData.from_memory(
            lattice=np.eye(3) * 10.0,
            atom_symbols=["Mo", "S", "S"],
            atom_coords=np.zeros((3, 3)),
            elements_orbital_map={"Mo": [0, 0, 1], "S": [0, 0]},
            hamiltonian_blocks=_make_simple_blocks(),
            path=tmp_path,
        )
        dd.save()

        # Overwrite force.h5 with wrong atom count (4 atoms instead of 3)
        with h5py.File(tmp_path / "force.h5", "w") as f:
            f.create_dataset("cell", data=np.eye(3) * 10.0)
            f.create_dataset("energy", data=0.0)
            f.create_dataset("force", data=np.zeros((4, 3)))
            f.create_dataset("stress", data=np.zeros(6))
            f.attrs["formula"] = b"X4"
            f.attrs["natoms"] = np.int64(4)

        with pytest.raises(AimspyConfigError, match="force.h5: force shape"):
            DeepHData.from_directory(tmp_path)

    def test_from_directory_force_wrong_columns(self, tmp_path):
        """force.h5 with wrong column count (2 instead of 3) raises."""
        import h5py

        from aimspy import AimspyConfigError

        dd = DeepHData.from_memory(
            lattice=np.eye(3) * 10.0,
            atom_symbols=["Mo", "S", "S"],
            atom_coords=np.zeros((3, 3)),
            elements_orbital_map={"Mo": [0, 0, 1], "S": [0, 0]},
            hamiltonian_blocks=_make_simple_blocks(),
            path=tmp_path,
        )
        dd.save()

        # Overwrite force.h5 with 3 atoms but 2 columns (wrong!)
        with h5py.File(tmp_path / "force.h5", "w") as f:
            f.create_dataset("cell", data=np.eye(3) * 10.0)
            f.create_dataset("energy", data=0.0)
            f.create_dataset("force", data=np.zeros((3, 2)))  # 2 cols, not 3
            f.create_dataset("stress", data=np.zeros(6))
            f.attrs["formula"] = b"X3"
            f.attrs["natoms"] = np.int64(3)

        with pytest.raises(AimspyConfigError, match="force.h5: force shape"):
            DeepHData.from_directory(tmp_path)

    def test_from_memory_force_shape_validation(self):
        """from_memory rejects force with wrong shape."""
        from aimspy import AimspyConfigError

        with pytest.raises(AimspyConfigError, match="force shape"):
            DeepHData.from_memory(
                lattice=np.eye(3) * 10.0,
                atom_symbols=["Mo", "S", "S"],
                atom_coords=np.zeros((3, 3)),
                elements_orbital_map={"Mo": [0, 0, 1], "S": [0, 0]},
                hamiltonian_blocks=_make_simple_blocks(),
                force=np.zeros((3, 2)),  # wrong: 2 columns
            )

    def test_set_force_shape_validation(self):
        """set_force rejects force with wrong shape."""
        from aimspy import AimspyConfigError

        struct = _make_mock_structure_unsorted()
        dd = DeepHData.from_aimspy(
            structure=struct,
            hamiltonian=AimspyMatrix(blocks=_make_simple_blocks(), n_spin=1),
        )
        with pytest.raises(AimspyConfigError, match="force shape"):
            dd.set_force(np.zeros((3, 2)), structure=struct)  # wrong: 2 cols

    def test_from_aimspy_force_list_input(self):
        """from_aimspy accepts Python list for force (not just ndarray)."""
        struct = _make_mock_structure_unsorted()
        # Pass as list, not ndarray — should work via np.asarray
        force_list = [
            [1.0, 0.0, 0.0],  # S (aims 0)
            [2.0, 0.0, 0.0],  # Mo (aims 1)
            [3.0, 0.0, 0.0],  # S (aims 2)
        ]
        dd = DeepHData.from_aimspy(
            structure=struct,
            hamiltonian=AimspyMatrix(blocks=_make_simple_blocks(), n_spin=1),
            force=force_list,
            energy=-1.0,
        )
        # POSCAR order [Mo, S, S] → [2, 1, 3]
        np.testing.assert_allclose(dd.force[:, 0], [2.0, 1.0, 3.0])

    def test_set_force_list_input(self):
        """set_force accepts Python list for force (not just ndarray)."""
        struct = _make_mock_structure_unsorted()
        dd = DeepHData.from_aimspy(
            structure=struct,
            hamiltonian=AimspyMatrix(blocks=_make_simple_blocks(), n_spin=1),
        )
        force_list = [[1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]]
        dd.set_force(force_list, structure=struct, energy=-0.5)
        # POSCAR order [Mo, S, S] → [2, 1, 3]
        np.testing.assert_allclose(dd.force[:, 0], [2.0, 1.0, 3.0])

    def test_repr_includes_force_tag(self):
        """__repr__ includes '+F' when force is set."""
        dd = DeepHData.from_memory(
            lattice=np.eye(3) * 10.0,
            atom_symbols=["Mo", "S", "S"],
            atom_coords=np.zeros((3, 3)),
            elements_orbital_map={"Mo": [0, 0, 1], "S": [0, 0]},
            hamiltonian_blocks=_make_simple_blocks(),
            force=np.ones((3, 3)),
        )
        r = repr(dd)
        assert "+F" in r
        assert "+H" in r

    def test_repr_no_force_tag_when_unset(self):
        """__repr__ does not include '+F' when force is None."""
        dd = DeepHData.from_memory(
            lattice=np.eye(3) * 10.0,
            atom_symbols=["Mo", "S", "S"],
            atom_coords=np.zeros((3, 3)),
            elements_orbital_map={"Mo": [0, 0, 1], "S": [0, 0]},
            hamiltonian_blocks=_make_simple_blocks(),
        )
        assert "+F" not in repr(dd)

    def test_save_includes_force(self, tmp_path):
        """save() writes force.h5 when force is set."""
        dd = DeepHData.from_memory(
            lattice=np.eye(3) * 10.0,
            atom_symbols=["Mo", "S", "S"],
            atom_coords=np.zeros((3, 3)),
            elements_orbital_map={"Mo": [0, 0, 1], "S": [0, 0]},
            hamiltonian_blocks=_make_simple_blocks(),
            force=np.ones((3, 3)),
            energy_eV=42.0,
            path=tmp_path,
        )
        dd.save()
        assert (tmp_path / "force.h5").exists()
        assert (tmp_path / "hamiltonian.h5").exists()


# =============================================================================
# Tests: first-order Hamiltonian (electric response)
# =============================================================================
def _make_three_first_order_blocks():
    """3 block dicts [x, y, z] with 1x1 blocks each, Hartree units.

    The values are chosen so each direction is distinguishable:
    x=1.0, y=2.0, z=3.0 (Hartree).
    """
    base_key = (0, 0, 0, 0, 0)
    return [
        {base_key: np.array([[1.0]])},  # x
        {base_key: np.array([[2.0]])},  # y
        {base_key: np.array([[3.0]])},  # z
    ]


def _make_three_aimspy_matrices():
    """3 AimspyMatrix [x, y, z] with 1 block each, Hartree units."""
    return [
        AimspyMatrix(blocks=blk, n_spin=1) for blk in _make_three_first_order_blocks()
    ]


class TestFirstOrderFromMemory:
    def test_from_memory_first_order_basic(self):
        fo_blocks = _make_three_first_order_blocks()
        dd = DeepHData.from_memory(
            lattice=np.eye(3) * 10.0,
            atom_symbols=["Mo", "S", "S"],
            atom_coords=np.zeros((3, 3)),
            elements_orbital_map={"Mo": [0, 0, 1], "S": [0, 0]},
            hamiltonian_blocks=_make_simple_blocks(),
            first_order_hamiltonian_blocks=fo_blocks,
        )
        assert dd.first_order_hamiltonian_entries is not None
        assert dd._fo_chunk_boundaries is not None
        assert dd._fo_chunk_shapes is not None
        # 2 pairs × 3×(1×1) = 6 entries
        assert dd.first_order_hamiltonian_entries.shape == (6,)
        # Hartree → eV
        HARTREE_TO_EV = 27.2113845
        # _make_simple_blocks has 2 pairs: (0,0,0,0,0) and (0,0,0,0,1)
        # fo_blocks only has key (0,0,0,0,0) → missing key (0,0,0,0,1) → zeros
        # DeepH order [y, z, x] = [2.0, 3.0, 1.0] Hartree → eV for pair 0
        # Pair 1 (missing) → zeros
        expected_pair0 = np.array([2.0, 3.0, 1.0]) * HARTREE_TO_EV
        expected_pair1 = np.zeros(3)
        expected = np.concatenate([expected_pair0, expected_pair1])
        np.testing.assert_allclose(dd.first_order_hamiltonian_entries, expected)
        # chunk_shapes: 3×1 row, 1 col (both pairs)
        assert dd._fo_chunk_shapes[0, 0] == 3
        assert dd._fo_chunk_shapes[0, 1] == 1
        # chunk_boundaries: [0, 3, 6]
        np.testing.assert_array_equal(
            dd._fo_chunk_boundaries, np.array([0, 3, 6], dtype=np.int32)
        )

    def test_from_memory_first_order_wrong_length_raises(self):
        from aimspy import AimspyConfigError

        fo_blocks = _make_three_first_order_blocks()[:2]  # only 2
        with pytest.raises(AimspyConfigError, match="list of 3"):
            DeepHData.from_memory(
                lattice=np.eye(3) * 10.0,
                atom_symbols=["Mo", "S", "S"],
                atom_coords=np.zeros((3, 3)),
                elements_orbital_map={"Mo": [0, 0, 1], "S": [0, 0]},
                hamiltonian_blocks=_make_simple_blocks(),
                first_order_hamiltonian_blocks=fo_blocks,
            )


class TestFirstOrderSet:
    def test_set_first_order_hamiltonian(self):
        struct = _make_mock_structure()
        dd = DeepHData.from_memory(
            lattice=np.eye(3) * 10.0,
            atom_symbols=["Mo", "S", "S"],
            atom_coords=np.zeros((3, 3)),
            elements_orbital_map={"Mo": [0, 0, 1], "S": [0, 0]},
            hamiltonian_blocks=_make_simple_blocks(),
        )
        mx_list = _make_three_aimspy_matrices()
        dd.set_first_order_hamiltonian(mx_list, struct)
        assert dd.first_order_hamiltonian_entries is not None
        assert dd._fo_chunk_boundaries is not None
        assert dd._fo_chunk_shapes is not None
        # 2 pairs × 3×(1×1) = 6 entries
        assert dd.first_order_hamiltonian_entries.shape == (6,)

    def test_set_first_order_wrong_length_raises(self):
        from aimspy import AimspyConfigError

        struct = _make_mock_structure()
        dd = DeepHData.from_memory(
            lattice=np.eye(3) * 10.0,
            atom_symbols=["Mo", "S", "S"],
            atom_coords=np.zeros((3, 3)),
            elements_orbital_map={"Mo": [0, 0, 1], "S": [0, 0]},
            hamiltonian_blocks=_make_simple_blocks(),
        )
        mx_list = _make_three_aimspy_matrices()[:2]
        with pytest.raises(AimspyConfigError, match="exactly 3"):
            dd.set_first_order_hamiltonian(mx_list, struct)


class TestFirstOrderFromAimspy:
    def test_from_aimspy_first_order(self):
        struct = _make_mock_structure()
        mx_list = _make_three_aimspy_matrices()
        dd = DeepHData.from_aimspy(
            structure=struct,
            hamiltonian=AimspyMatrix(blocks=_make_simple_blocks(), n_spin=1),
            first_order_hamiltonian=mx_list,
        )
        assert dd.first_order_hamiltonian_entries is not None
        assert dd._fo_chunk_boundaries is not None
        assert dd._fo_chunk_shapes is not None

    def test_from_aimspy_first_order_wrong_length_raises(self):
        from aimspy import AimspyConfigError

        struct = _make_mock_structure()
        mx_list = _make_three_aimspy_matrices()[:2]
        with pytest.raises(AimspyConfigError, match="list of 3"):
            DeepHData.from_aimspy(
                structure=struct,
                first_order_hamiltonian=mx_list,
            )


class TestFirstOrderSaveLoad:
    def test_save_first_order(self, tmp_path):
        dd = DeepHData.from_memory(
            lattice=np.eye(3) * 10.0,
            atom_symbols=["Mo", "S", "S"],
            atom_coords=np.zeros((3, 3)),
            elements_orbital_map={"Mo": [0, 0, 1], "S": [0, 0]},
            hamiltonian_blocks=_make_simple_blocks(),
            first_order_hamiltonian_blocks=_make_three_first_order_blocks(),
            path=tmp_path,
        )
        dd.save()
        assert (tmp_path / "electric_response.h5").exists()
        # Verify HDF5 contents
        import h5py

        with h5py.File(tmp_path / "electric_response.h5", "r") as f:
            assert "atom_pairs" in f
            assert "chunk_boundaries" in f
            assert "chunk_shapes" in f
            assert "entries" in f
            entries = f["entries"][:]
            # 2 pairs × 3 entries each = 6
            assert entries.shape == (6,)

    def test_from_directory_first_order(self, tmp_path):
        # First save, then reload
        dd = DeepHData.from_memory(
            lattice=np.eye(3) * 10.0,
            atom_symbols=["Mo", "S", "S"],
            atom_coords=np.zeros((3, 3)),
            elements_orbital_map={"Mo": [0, 0, 1], "S": [0, 0]},
            hamiltonian_blocks=_make_simple_blocks(),
            first_order_hamiltonian_blocks=_make_three_first_order_blocks(),
            path=tmp_path,
        )
        dd.save()
        # Reload
        dd2 = DeepHData.from_directory(tmp_path)
        assert dd2.first_order_hamiltonian_entries is not None
        assert dd2._fo_chunk_boundaries is not None
        assert dd2._fo_chunk_shapes is not None
        np.testing.assert_allclose(
            dd2.first_order_hamiltonian_entries,
            dd.first_order_hamiltonian_entries,
        )

    def test_save_first_order_without_entries_raises(self, tmp_path):
        from aimspy import AimspyConfigError

        dd = DeepHData.from_memory(
            lattice=np.eye(3) * 10.0,
            atom_symbols=["Mo", "S", "S"],
            atom_coords=np.zeros((3, 3)),
            elements_orbital_map={"Mo": [0, 0, 1], "S": [0, 0]},
            hamiltonian_blocks=_make_simple_blocks(),
            path=tmp_path,
        )
        with pytest.raises(AimspyConfigError, match="No first_order"):
            dd.save_first_order_hamiltonian()


class TestFirstOrderToAimspy:
    def test_to_first_order_aimspy_basic(self):
        struct = _make_mock_structure()
        mx_list = _make_three_aimspy_matrices()
        dd = DeepHData.from_aimspy(
            structure=struct,
            first_order_hamiltonian=mx_list,
        )
        # Convert back
        result = dd.to_first_order_aimspy(struct)
        assert isinstance(result, list)
        assert len(result) == 3
        for mx in result:
            assert isinstance(mx, AimspyMatrix)
            assert mx.n_spin == 1

    def test_to_first_order_aimspy_roundtrip(self):
        """Roundtrip: 3 AimspyMatrix → DeepHData → 3 AimspyMatrix.

        The blocks should match (modulo atom reordering, but since we use
        _make_mock_structure which has sorted atoms ["Mo","S","S"], the
        permutation is identity).
        """
        struct = _make_mock_structure()
        original = _make_three_aimspy_matrices()
        dd = DeepHData.from_aimspy(
            structure=struct,
            first_order_hamiltonian=original,
        )
        recovered = dd.to_first_order_aimspy(struct)
        # Each recovered matrix should have the same blocks as the original
        for cart in range(3):
            orig_blocks = original[cart].blocks
            recv_blocks = recovered[cart].blocks
            assert set(orig_blocks.keys()) == set(recv_blocks.keys())
            for key in orig_blocks:
                np.testing.assert_allclose(
                    recv_blocks[key], orig_blocks[key], atol=1e-10
                )

    def test_to_first_order_aimspy_without_entries_raises(self):
        from aimspy import AimspyConfigError

        dd = DeepHData.from_memory(
            lattice=np.eye(3) * 10.0,
            atom_symbols=["Mo", "S", "S"],
            atom_coords=np.zeros((3, 3)),
            elements_orbital_map={"Mo": [0, 0, 1], "S": [0, 0]},
            hamiltonian_blocks=_make_simple_blocks(),
        )
        struct = _make_mock_structure()
        with pytest.raises(AimspyConfigError, match="No first_order"):
            dd.to_first_order_aimspy(struct)

    def test_to_first_order_direction_order(self):
        """Verify DeepH order [y, z, x] ↔ Cartesian order [x, y, z].

        x=1.0, y=2.0, z=3.0 (Hartree). After conversion to DeepH entries
        and back, the Cartesian order should be preserved.
        """
        struct = _make_mock_structure()
        original = _make_three_aimspy_matrices()
        dd = DeepHData.from_aimspy(
            structure=struct,
            first_order_hamiltonian=original,
        )
        recovered = dd.to_first_order_aimspy(struct)
        key = (0, 0, 0, 0, 0)
        # x should be 1.0 Hartree, y=2.0, z=3.0
        assert recovered[0].blocks[key][0, 0] == pytest.approx(1.0, abs=1e-10)
        assert recovered[1].blocks[key][0, 0] == pytest.approx(2.0, abs=1e-10)
        assert recovered[2].blocks[key][0, 0] == pytest.approx(3.0, abs=1e-10)


class TestFirstOrderRepr:
    def test_repr_with_first_order(self):
        dd = DeepHData.from_memory(
            lattice=np.eye(3) * 10.0,
            atom_symbols=["Mo", "S", "S"],
            atom_coords=np.zeros((3, 3)),
            elements_orbital_map={"Mo": [0, 0, 1], "S": [0, 0]},
            hamiltonian_blocks=_make_simple_blocks(),
            first_order_hamiltonian_blocks=_make_three_first_order_blocks(),
        )
        r = repr(dd)
        assert "+dHde" in r


class TestExternalFirstOrderMatrixSource:
    def test_deeph_data_satisfies_protocol(self):
        from aimspy import ExternalFirstOrderMatrixSource

        struct = _make_mock_structure()
        dd = DeepHData.from_aimspy(
            structure=struct,
            first_order_hamiltonian=_make_three_aimspy_matrices(),
        )
        assert isinstance(dd, ExternalFirstOrderMatrixSource)


class TestCalculatorFirstOrderConfig:
    def test_config_capture_first_order_default(self):
        from aimspy import CalculatorConfig

        cfg = CalculatorConfig(lib_path="/tmp/x.so")
        assert cfg.capture_first_order_hamiltonian is False

    def test_config_capture_first_order_set(self):
        from aimspy import CalculatorConfig

        cfg = CalculatorConfig(
            lib_path="/tmp/x.so", capture_first_order_hamiltonian=True
        )
        assert cfg.capture_first_order_hamiltonian is True

    def test_modify_init_first_order_ham_direct(self):
        from aimspy import Calculator, CalculatorConfig, Strategy

        calc = Calculator(CalculatorConfig(lib_path="/tmp/x.so"))

        # Use a dummy source (None is not allowed for direct mode, but
        # we only test config storage, not actual callback firing).
        # Create a minimal stub object with to_first_order_aimspy method.
        class StubSource:
            def to_first_order_aimspy(self, structure):
                return []

        calc.modify_init_first_order_ham(source=StubSource(), strategy=Strategy.REPLACE)
        assert calc._modify_first_order is not None
        assert calc._modify_first_order.strategy == Strategy.REPLACE

    def test_modify_init_first_order_ham_invalid_strategy(self):
        from aimspy import Calculator, CalculatorConfig, AimspyConfigError

        calc = Calculator(CalculatorConfig(lib_path="/tmp/x.so"))
        with pytest.raises(AimspyConfigError, match="only REPLACE and ADD"):
            calc.modify_init_first_order_ham(strategy="custom")

    def test_modify_init_first_order_ham_custom_raises(self):
        from aimspy import Calculator, CalculatorConfig, AimspyConfigError

        calc = Calculator(CalculatorConfig(lib_path="/tmp/x.so"))
        with pytest.raises(AimspyConfigError, match="only REPLACE and ADD"):
            calc.modify_init_first_order_ham(
                strategy="custom", custom_fn=lambda *a: None
            )

    def test_modify_init_first_order_ham_deferred(self):
        from aimspy import Calculator, CalculatorConfig, Strategy

        calc = Calculator(CalculatorConfig(lib_path="/tmp/x.so"))

        @calc.modify_init_first_order_ham(
            strategy=Strategy.REPLACE, option={"path": "/tmp"}
        )
        def gen_source(view, option):
            return None

        assert calc._modify_first_order is not None
        assert calc._modify_first_order.deferred_fn is gen_source
        assert calc._modify_first_order.deferred_option == {"path": "/tmp"}
