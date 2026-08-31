"""Unit tests for DeepHData construction, I/O, and conversion."""

from __future__ import annotations

import json
import h5py
import numpy as np
import pytest

from aimspy import AimspyConfigError, DeepHData
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


_SIMPLE_EOM = {"Mo": [0], "S": [0]}


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
            elements_orbital_map=_SIMPLE_EOM,
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
    def test_direct_constructor_preserves_legacy_positional_metadata_order(
        self, tmp_path
    ):
        dd = DeepHData(
            np.eye(3),
            ["H"],
            np.zeros((1, 3)),
            {"H": [0]},
            1,
            np.array([[0, 0, 0, 0, 0]], dtype=np.int32),
            np.array([0, 1], dtype=np.int32),
            np.array([[1, 1]], dtype=np.int32),
            np.array([1.0]),
            None,
            None,
            None,
            None,
            None,
            None,
            -2.5,
            4.75,
            tmp_path,
        )

        assert dd.energy_eV == pytest.approx(-2.5)
        assert dd.fermi_energy_eV == pytest.approx(4.75)
        assert dd.path == tmp_path
        assert dd.stress is None

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
# Tests: energy / force / stress (MD-style force.h5)
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


def _make_force_data(path=None, **labels):
    return DeepHData.from_memory(
        lattice=np.eye(3) * 10.0,
        atom_symbols=["Mo", "S", "S"],
        atom_coords=np.zeros((3, 3)),
        elements_orbital_map=_SIMPLE_EOM,
        hamiltonian_blocks=_make_simple_blocks(),
        path=path,
        **labels,
    )


class TestForce:
    def test_save_load_force_roundtrip(self, tmp_path):
        """save → from_directory preserves all force-field labels."""
        force = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        stress = np.array([[1.0, 0.4, 0.5], [0.4, 2.0, 0.6], [0.5, 0.6, 3.0]])
        dd = DeepHData.from_memory(
            lattice=np.eye(3) * 10.0,
            atom_symbols=["Mo", "S", "S"],
            atom_coords=np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]]),
            elements_orbital_map=_SIMPLE_EOM,
            hamiltonian_blocks=_make_simple_blocks(),
            force=force,
            energy_eV=-123.45,
            path=tmp_path,
            stress=stress,
        )
        dd.save()
        assert (tmp_path / "force.h5").exists()

        dd2 = DeepHData.from_directory(tmp_path)
        assert dd2.force is not None
        np.testing.assert_allclose(dd2.force, force)
        assert dd2.energy_eV == pytest.approx(-123.45)
        np.testing.assert_allclose(dd2.stress, stress)

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
            stress=np.diag([1.0, 2.0, 3.0]),
        )
        # POSCAR order is [Mo, S, S], so force should be [2, 1, 3]
        assert dd.force is not None
        np.testing.assert_allclose(dd.force[:, 0], [2.0, 1.0, 3.0])
        # Structure quantities are already in Å and must not be rescaled.
        np.testing.assert_array_equal(dd.lattice, struct.lattice)
        np.testing.assert_array_equal(dd.atom_coords, struct.atom_coords[[1, 0, 2]])
        # energy: -1.0 Hartree → eV
        assert dd.energy_eV == pytest.approx(-1.0 * 27.2113845)
        np.testing.assert_allclose(dd.stress, np.diag([1.0, 2.0, 3.0]))

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
            elements_orbital_map=_SIMPLE_EOM,
            hamiltonian_blocks=_make_simple_blocks(),
            path=tmp_path,
        )
        with pytest.raises(AimspyConfigError, match="No energy, force, or stress"):
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
            stress=np.array([[1.0, 0.6, 0.5], [0.6, 2.0, 0.4], [0.5, 0.4, 3.0]]),
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
            np.testing.assert_allclose(f["stress"][:], [1, 2, 3, 0.4, 0.5, 0.6])
            # Attrs
            assert f.attrs["formula"] == b"X12"
            assert int(f.attrs["natoms"]) == 12
            # Energy value
            assert float(f["energy"][()]) == pytest.approx(-0.073)

    def test_force_h5_uses_zero_stress_when_unavailable(self, tmp_path):
        """Missing energy is omitted while unavailable stress is six zeros."""
        import h5py

        dd = DeepHData.from_memory(
            lattice=np.eye(3) * 10.0,
            atom_symbols=["Mo", "S", "S"],
            atom_coords=np.zeros((3, 3)),
            elements_orbital_map=_SIMPLE_EOM,
            hamiltonian_blocks=_make_simple_blocks(),
            force=np.ones((3, 3)),
            energy_eV=None,
            path=tmp_path,
        )
        dd.save_force()
        with h5py.File(tmp_path / "force.h5", "r") as f:
            assert set(f.keys()) == {"cell", "force", "stress"}
            assert f["stress"].shape == (6,)
            np.testing.assert_array_equal(f["stress"][:], np.zeros(6))

    def test_from_directory_force_shape_mismatch(self, tmp_path):
        """force.h5 with wrong atom count raises AimspyConfigError."""
        import h5py

        from aimspy import AimspyConfigError

        # First save a valid DeepH dir (3 atoms) with a matrix file
        dd = DeepHData.from_memory(
            lattice=np.eye(3) * 10.0,
            atom_symbols=["Mo", "S", "S"],
            atom_coords=np.zeros((3, 3)),
            elements_orbital_map=_SIMPLE_EOM,
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

        with pytest.raises(AimspyConfigError, match="force.h5: force: expected shape"):
            DeepHData.from_directory(tmp_path)

    def test_from_directory_force_wrong_columns(self, tmp_path):
        """force.h5 with wrong column count (2 instead of 3) raises."""
        import h5py

        from aimspy import AimspyConfigError

        dd = DeepHData.from_memory(
            lattice=np.eye(3) * 10.0,
            atom_symbols=["Mo", "S", "S"],
            atom_coords=np.zeros((3, 3)),
            elements_orbital_map=_SIMPLE_EOM,
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

        with pytest.raises(AimspyConfigError, match="force.h5: force: expected shape"):
            DeepHData.from_directory(tmp_path)

    def test_from_memory_force_shape_validation(self):
        """from_memory rejects force with wrong shape."""
        from aimspy import AimspyConfigError

        with pytest.raises(AimspyConfigError, match="force: expected shape"):
            DeepHData.from_memory(
                lattice=np.eye(3) * 10.0,
                atom_symbols=["Mo", "S", "S"],
                atom_coords=np.zeros((3, 3)),
                elements_orbital_map={"Mo": [0, 0, 1], "S": [0, 0]},
                hamiltonian_blocks=_make_simple_blocks(),
                force=np.zeros((3, 2)),  # wrong: 2 columns
            )

    @pytest.mark.parametrize(
        "force",
        [
            np.full((3, 3), np.nan),
            np.ones((3, 3), dtype=np.complex128),
            np.asarray([["x"] * 3] * 3),
        ],
    )
    def test_from_memory_force_value_validation(self, force):
        with pytest.raises(AimspyConfigError, match="force:"):
            _make_force_data(force=force)

    def test_set_force_shape_validation(self):
        """set_force rejects force with wrong shape."""
        from aimspy import AimspyConfigError

        struct = _make_mock_structure_unsorted()
        dd = DeepHData.from_aimspy(
            structure=struct,
            hamiltonian=AimspyMatrix(blocks=_make_simple_blocks(), n_spin=1),
        )
        with pytest.raises(AimspyConfigError, match="force: expected shape"):
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

    def test_stress_voigt_order_and_roundtrip(self, tmp_path):
        stress = np.array([[11.0, 16.0, 15.0], [16.0, 12.0, 14.0], [15.0, 14.0, 13.0]])
        dd = _make_force_data(tmp_path, energy_eV=-7.5, stress=stress)
        dd.save()

        with h5py.File(tmp_path / "force.h5", "r") as f:
            np.testing.assert_array_equal(
                f["stress"][:], [11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
            )
            assert "force" not in f

        loaded = DeepHData.from_directory(tmp_path)
        assert loaded.force is None
        assert loaded.energy_eV == pytest.approx(-7.5)
        np.testing.assert_array_equal(loaded.stress, stress)

    @pytest.mark.parametrize("shape", [(6,), (1, 6), (3, 3)])
    def test_from_memory_accepts_supported_stress_shapes(self, shape):
        voigt = np.arange(1.0, 7.0)
        tensor = np.array([[1.0, 6.0, 5.0], [6.0, 2.0, 4.0], [5.0, 4.0, 3.0]])
        value = tensor if shape == (3, 3) else voigt.reshape(shape)
        dd = _make_force_data(stress=value)
        np.testing.assert_array_equal(dd.stress, tensor)

    @pytest.mark.parametrize(
        ("stress", "message"),
        [
            (np.zeros((3, 2)), "expected shape"),
            (
                np.array([[1.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
                "symmetric",
            ),
            (np.array([1.0, 2.0, 3.0, 4.0, 5.0, np.nan]), "finite"),
            (np.asarray(["x"] * 6), "numeric dtype"),
        ],
    )
    def test_from_memory_rejects_invalid_stress(self, stress, message):
        with pytest.raises(AimspyConfigError, match=message):
            _make_force_data(stress=stress)

    @pytest.mark.parametrize("energy", [np.nan, np.inf, -np.inf])
    def test_from_memory_rejects_nonfinite_energy(self, energy):
        with pytest.raises(AimspyConfigError, match="energy: value must be finite"):
            _make_force_data(energy_eV=energy)

    def test_from_memory_rejects_nonscalar_energy(self):
        with pytest.raises(AimspyConfigError, match="energy: expected a scalar"):
            _make_force_data(energy_eV=np.array([-1.0]))

    def test_force_and_energy_only_files_roundtrip(self, tmp_path):
        force_dir = tmp_path / "force_only"
        force = np.arange(9.0).reshape(3, 3)
        _make_force_data(force_dir, force=force).save()
        force_only = DeepHData.from_directory(force_dir)
        np.testing.assert_array_equal(force_only.force, force)
        assert force_only.energy_eV is None
        np.testing.assert_array_equal(force_only.stress, np.zeros((3, 3)))

        energy_dir = tmp_path / "energy_only"
        _make_force_data(energy_dir, energy_eV=-2.5).save()
        energy_only = DeepHData.from_directory(energy_dir)
        assert energy_only.force is None
        assert energy_only.energy_eV == pytest.approx(-2.5)
        np.testing.assert_array_equal(energy_only.stress, np.zeros((3, 3)))

    def test_from_aimspy_none_stress_writes_zero_voigt(self, tmp_path):
        struct = AimspyStructure(
            n_atoms=3,
            n_basis=3,
            n_spin=1,
            n_periodic=3,
            lattice=np.eye(3) * 10.0,
            atom_symbols=["S", "Mo", "S"],
            atom_coords=np.zeros((3, 3)),
            basis_atom=np.arange(3, dtype=np.int32),
            basis_l=np.zeros(3, dtype=np.int32),
            basis_m=np.zeros(3, dtype=np.int32),
        )
        dd = DeepHData.from_aimspy(
            structure=struct,
            hamiltonian=AimspyMatrix(blocks=_make_simple_blocks(), n_spin=1),
            force=np.zeros((3, 3)),
            energy=-1.0,
            stress=None,
            path=tmp_path,
        )

        assert dd.stress is None
        dd.save()
        with h5py.File(tmp_path / "force.h5", "r") as f:
            np.testing.assert_array_equal(f["stress"][:], np.zeros(6))

    def test_from_directory_accepts_tensor_stress(self, tmp_path):
        stress = np.array([[1.0, 0.6, 0.5], [0.6, 2.0, 0.4], [0.5, 0.4, 3.0]])
        dd = _make_force_data(tmp_path)
        dd.save()
        with h5py.File(tmp_path / "force.h5", "w") as f:
            f.create_dataset("cell", data=dd.lattice)
            f.create_dataset("stress", data=stress)
        loaded = DeepHData.from_directory(tmp_path)
        np.testing.assert_array_equal(loaded.stress, stress)

    def test_from_directory_rejects_cell_mismatch(self, tmp_path):
        dd = _make_force_data(tmp_path)
        dd.save()
        with h5py.File(tmp_path / "force.h5", "w") as f:
            f.create_dataset("cell", data=np.eye(3) * 11.0)
            f.create_dataset("energy", data=-1.0)
        with pytest.raises(AimspyConfigError, match="cell: does not match"):
            DeepHData.from_directory(tmp_path)

    def test_from_directory_rejects_force_file_without_labels(self, tmp_path):
        dd = _make_force_data(tmp_path)
        dd.save()
        with h5py.File(tmp_path / "force.h5", "w") as f:
            f.create_dataset("cell", data=dd.lattice)
        with pytest.raises(AimspyConfigError, match="at least one of energy"):
            DeepHData.from_directory(tmp_path)

    @pytest.mark.parametrize("field", ["cell", "energy", "force", "stress"])
    def test_force_h5_names_must_refer_to_datasets(self, tmp_path, field):
        dd = _make_force_data(
            tmp_path,
            force=np.zeros((3, 3)),
            energy_eV=-1.0,
            stress=np.eye(3),
        )
        dd.save()
        with h5py.File(tmp_path / "force.h5", "a") as h5:
            del h5[field]
            h5.create_group(field)

        with pytest.raises(AimspyConfigError) as exc_info:
            DeepHData.from_directory(tmp_path)

        assert f"force.h5: {field}: expected an HDF5 dataset" in str(exc_info.value)

    def test_write_revalidates_mutated_labels(self, tmp_path):
        dd = _make_force_data(tmp_path, force=np.zeros((3, 3)), stress=np.eye(3))
        dd.force = np.zeros((2, 3))
        with pytest.raises(AimspyConfigError, match="force: expected shape"):
            dd.save_force()

        dd.force = np.zeros((3, 3))
        dd.stress = np.full((3, 3), np.nan)
        with pytest.raises(
            AimspyConfigError, match="stress: values must all be finite"
        ):
            dd.save_force()

    def test_set_force_stores_relative_energy_and_stress_units(self):
        struct = _make_mock_structure_unsorted()
        dd = DeepHData.from_aimspy(
            structure=struct,
            hamiltonian=AimspyMatrix(blocks=_make_simple_blocks(), n_spin=1),
        )
        force_eV_per_ang = np.arange(9.0).reshape(3, 3)
        stress_eV_per_ang3 = np.diag([1.5, 2.5, 3.5])
        energy_relative_Ha = -10.25
        dd.set_force(
            force_eV_per_ang,
            struct,
            energy=energy_relative_Ha,
            stress=stress_eV_per_ang3,
        )
        np.testing.assert_array_equal(dd.force, force_eV_per_ang[[1, 0, 2]])
        assert dd.energy_eV == pytest.approx(energy_relative_Ha * 27.2113845)
        np.testing.assert_array_equal(dd.stress, stress_eV_per_ang3)

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
            elements_orbital_map=_SIMPLE_EOM,
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
            elements_orbital_map=_SIMPLE_EOM,
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
            elements_orbital_map=_SIMPLE_EOM,
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


# =============================================================================
# Regression tests: cross-file atom_pairs validation / reordering (2.4)
# =============================================================================
class TestAtomPairsReorder:
    def _write_matrix(self, path, name, atom_pairs, cb, cs, entries):
        import h5py

        with h5py.File(path / f"{name}.h5", "w") as f:
            f.create_dataset("atom_pairs", data=atom_pairs, dtype="i4")
            f.create_dataset("chunk_boundaries", data=cb, dtype="i4")
            f.create_dataset("chunk_shapes", data=cs, dtype="i4")
            f.create_dataset("entries", data=entries)

    def _write_meta(self, path):
        from aimspy.interface.deeph.data import _write_poscar
        import json as _json

        _write_poscar(
            path / "POSCAR",
            np.eye(3) * 10.0,
            ["Mo", "S", "S"],
            np.zeros((3, 3)),
        )
        info = {
            "elements_orbital_map": _SIMPLE_EOM,
            "orbits_quantity": 3,
            "spinful": False,
        }
        with open(path / "info.json", "w") as f:
            _json.dump(info, f)

    def test_reorder_same_set_different_order(self, tmp_path):
        """overlap.h5 with a permuted (but equivalent) atom_pairs is reordered
        to match hamiltonian.h5's canonical order."""
        self._write_meta(tmp_path)
        # canonical: 2 pairs
        ap = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 1]], dtype=np.int32)
        cb = np.array([0, 1, 2], dtype=np.int32)
        cs = np.array([[1, 1], [1, 1]], dtype=np.int32)
        self._write_matrix(tmp_path, "hamiltonian", ap, cb, cs, np.array([10.0, 20.0]))
        # overlap: same pairs, reversed order
        ap2 = ap[::-1].copy()
        cb2 = np.array([0, 1, 2], dtype=np.int32)
        cs2 = np.array([[1, 1], [1, 1]], dtype=np.int32)
        # entries in the reversed order: pair (0,0,0,0,1) first
        self._write_matrix(tmp_path, "overlap", ap2, cb2, cs2, np.array([0.5, 0.9]))
        dd = DeepHData.from_directory(tmp_path)
        # overlap entries reordered to canonical: pair0 (0,0,0,0,0)=0.9,
        # pair1 (0,0,0,0,1)=0.5
        np.testing.assert_allclose(dd.overlap_entries, [0.9, 0.5])
        np.testing.assert_allclose(dd.entries, [10.0, 20.0])

    def test_different_set_raises(self, tmp_path):
        """overlap.h5 with a *different* pair set raises AimspyConfigError."""
        from aimspy import AimspyConfigError

        self._write_meta(tmp_path)
        ap = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 1]], dtype=np.int32)
        cb = np.array([0, 1, 2], dtype=np.int32)
        cs = np.array([[1, 1], [1, 1]], dtype=np.int32)
        self._write_matrix(tmp_path, "hamiltonian", ap, cb, cs, np.array([10.0, 20.0]))
        # overlap: different pair set
        ap_bad = np.array([[0, 0, 0, 0, 0], [1, 0, 0, 0, 1]], dtype=np.int32)
        self._write_matrix(tmp_path, "overlap", ap_bad, cb, cs, np.array([0.5, 0.9]))
        with pytest.raises(AimspyConfigError):
            DeepHData.from_directory(tmp_path)

    def test_spinful_info_raises(self, tmp_path):
        """info.json with spinful:true raises AimspyConfigError."""
        from aimspy import AimspyConfigError
        import json as _json
        from aimspy.interface.deeph.data import _write_poscar

        _write_poscar(
            tmp_path / "POSCAR",
            np.eye(3) * 10.0,
            ["Mo", "S", "S"],
            np.zeros((3, 3)),
        )
        with open(tmp_path / "info.json", "w") as f:
            _json.dump(
                {
                    "elements_orbital_map": {"Mo": [0, 0, 1], "S": [0, 0]},
                    "spinful": True,
                },
                f,
            )
        ap = np.array([[0, 0, 0, 0, 0]], dtype=np.int32)
        cb = np.array([0, 1], dtype=np.int32)
        cs = np.array([[1, 1]], dtype=np.int32)
        self._write_matrix(tmp_path, "hamiltonian", ap, cb, cs, np.array([1.0]))
        with pytest.raises(AimspyConfigError, match="spin"):
            DeepHData.from_directory(tmp_path)

    def test_electric_response_reorder_3x_blocks(self, tmp_path):
        """electric_response.h5 with a permuted atom_pairs is reordered using
        the 3x-expanded chunk boundaries (regression: dst must be fo_cb, not
        the 1x standard cb)."""
        self._write_meta(tmp_path)
        # canonical: 2 pairs, 1x1 blocks
        ap = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 1]], dtype=np.int32)
        cb = np.array([0, 1, 2], dtype=np.int32)
        cs = np.array([[1, 1], [1, 1]], dtype=np.int32)
        self._write_matrix(tmp_path, "hamiltonian", ap, cb, cs, np.array([10.0, 20.0]))

        # electric_response: SAME pairs but reversed order; each block is
        # (3*1, 1) = 3 values [y, z, x].  Canonical pair0=(0,0,0,0,0) has
        # values [1,2,3]; canonical pair1=(0,0,0,0,1) has [4,5,6].  In the
        # reversed file, pair1 comes first.
        ap_fo = ap[::-1].copy()
        fo_cb = np.array([0, 3, 6], dtype=np.int32)
        fo_cs = np.array([[3, 1], [3, 1]], dtype=np.int32)
        fo_entries = np.array([4.0, 5.0, 6.0, 1.0, 2.0, 3.0])
        self._write_matrix(
            tmp_path, "electric_response", ap_fo, fo_cb, fo_cs, fo_entries
        )

        dd = DeepHData.from_directory(tmp_path)
        # After reorder to canonical: pair0=[1,2,3], pair1=[4,5,6]
        np.testing.assert_allclose(
            dd.first_order_hamiltonian_entries, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        )
        # And the rebuilt 3x chunk layout matches the canonical pair order
        np.testing.assert_array_equal(dd._fo_chunk_boundaries, [0, 3, 6])
        np.testing.assert_array_equal(dd._fo_chunk_shapes, [[3, 1], [3, 1]])


# =============================================================================
# Strict HDF5 layout validation
# =============================================================================
class TestMatrixLayoutValidation:
    def _write_metadata(self, path):
        from aimspy.interface.deeph.data import _write_poscar

        _write_poscar(
            path / "POSCAR",
            np.eye(3) * 5.0,
            ["H"],
            np.zeros((1, 3)),
        )
        (path / "info.json").write_text(
            json.dumps(
                {
                    "elements_orbital_map": {"H": [0]},
                    "orbits_quantity": 1,
                    "spinful": False,
                }
            )
        )

    def _write_matrix(
        self,
        path,
        filename="hamiltonian.h5",
        *,
        atom_pairs=None,
        chunk_boundaries=None,
        chunk_shapes=None,
        entries=None,
    ):
        import h5py

        arrays = {
            "atom_pairs": np.array([[0, 0, 0, 0, 0]], dtype=np.int32),
            "chunk_boundaries": np.array([0, 1], dtype=np.int32),
            "chunk_shapes": np.array([[1, 1]], dtype=np.int32),
            "entries": np.array([1.0], dtype=np.float64),
        }
        replacements = {
            "atom_pairs": atom_pairs,
            "chunk_boundaries": chunk_boundaries,
            "chunk_shapes": chunk_shapes,
            "entries": entries,
        }
        for name, value in replacements.items():
            if value is not None:
                arrays[name] = value
        with h5py.File(path / filename, "w") as h5:
            for name, value in arrays.items():
                h5.create_dataset(name, data=value)
        return path / filename

    def _make_valid_directory(self, path):
        self._write_metadata(path)
        return self._write_matrix(path)

    @pytest.mark.parametrize(
        "missing",
        ["atom_pairs", "chunk_boundaries", "chunk_shapes", "entries"],
    )
    def test_missing_required_dataset_rejected(self, tmp_path, missing):
        import h5py

        matrix_path = self._make_valid_directory(tmp_path)
        with h5py.File(matrix_path, "a") as h5:
            del h5[missing]

        with pytest.raises(AimspyConfigError) as exc_info:
            DeepHData.from_directory(tmp_path)

        assert f"hamiltonian.h5: {missing}:" in str(exc_info.value)

    @pytest.mark.parametrize(
        "field", ["atom_pairs", "chunk_boundaries", "chunk_shapes", "entries"]
    )
    def test_required_names_must_refer_to_datasets(self, tmp_path, field):
        import h5py

        matrix_path = self._make_valid_directory(tmp_path)
        with h5py.File(matrix_path, "a") as h5:
            del h5[field]
            h5.create_group(field)

        with pytest.raises(AimspyConfigError) as exc_info:
            DeepHData.from_directory(tmp_path)

        assert f"hamiltonian.h5: {field}: expected an HDF5 dataset" in str(
            exc_info.value
        )

    def test_required_dataset_must_not_be_a_dangling_link(self, tmp_path):
        import h5py

        matrix_path = self._make_valid_directory(tmp_path)
        with h5py.File(matrix_path, "a") as h5:
            del h5["entries"]
            h5["entries"] = h5py.SoftLink("/missing")

        with pytest.raises(AimspyConfigError) as exc_info:
            DeepHData.from_directory(tmp_path)

        assert "hamiltonian.h5: entries: could not resolve" in str(exc_info.value)

    @pytest.mark.parametrize(
        ("field", "bad_value"),
        [
            ("atom_pairs", np.array([[0, 0, 0, 0, 0]], dtype=np.float64)),
            ("chunk_boundaries", np.array([0.0, 1.0])),
            ("chunk_shapes", np.array([[1.0, 1.0]])),
            ("entries", np.array([1.0 + 2.0j])),
        ],
    )
    def test_invalid_dataset_dtype_rejected(self, tmp_path, field, bad_value):
        import h5py

        matrix_path = self._make_valid_directory(tmp_path)
        with h5py.File(matrix_path, "a") as h5:
            del h5[field]
            h5.create_dataset(field, data=bad_value)

        with pytest.raises(AimspyConfigError) as exc_info:
            DeepHData.from_directory(tmp_path)

        assert f"hamiltonian.h5: {field}:" in str(exc_info.value)

    @pytest.mark.parametrize(
        ("field", "bad_value"),
        [
            ("atom_pairs", np.zeros((1, 4), dtype=np.int32)),
            ("chunk_boundaries", np.array([0], dtype=np.int32)),
            ("chunk_shapes", np.ones((1, 1), dtype=np.int32)),
            ("entries", np.ones((1, 1), dtype=np.float64)),
            ("entries", np.array(1.0)),
        ],
    )
    def test_invalid_dataset_shape_rejected(self, tmp_path, field, bad_value):
        import h5py

        matrix_path = self._make_valid_directory(tmp_path)
        with h5py.File(matrix_path, "a") as h5:
            del h5[field]
            h5.create_dataset(field, data=bad_value)

        with pytest.raises(AimspyConfigError) as exc_info:
            DeepHData.from_directory(tmp_path)

        assert f"hamiltonian.h5: {field}:" in str(exc_info.value)

    def test_duplicate_atom_pairs_rejected(self, tmp_path):
        self._write_metadata(tmp_path)
        self._write_matrix(
            tmp_path,
            atom_pairs=np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=np.int32),
            chunk_boundaries=np.array([0, 1, 2], dtype=np.int32),
            chunk_shapes=np.array([[1, 1], [1, 1]], dtype=np.int32),
            entries=np.array([1.0, 2.0]),
        )

        with pytest.raises(AimspyConfigError, match="duplicate pair keys"):
            DeepHData.from_directory(tmp_path)

    def test_atom_index_out_of_range_rejected(self, tmp_path):
        self._write_metadata(tmp_path)
        self._write_matrix(
            tmp_path,
            atom_pairs=np.array([[0, 0, 0, 0, 1]], dtype=np.int32),
        )

        with pytest.raises(AimspyConfigError, match="atom indices"):
            DeepHData.from_directory(tmp_path)

    @pytest.mark.parametrize(
        ("boundaries", "entries", "message"),
        [
            (np.array([1, 2], dtype=np.int32), np.ones(2), "first value"),
            (np.array([0, 2], dtype=np.int32), np.ones(2), "span is"),
            (np.array([0, 1], dtype=np.int32), np.ones(2), "final boundary"),
        ],
    )
    def test_boundary_invariants_rejected(self, tmp_path, boundaries, entries, message):
        self._write_metadata(tmp_path)
        self._write_matrix(
            tmp_path,
            chunk_boundaries=boundaries,
            entries=entries,
        )

        with pytest.raises(AimspyConfigError, match=message):
            DeepHData.from_directory(tmp_path)

    def test_non_monotonic_boundaries_rejected(self, tmp_path):
        self._write_metadata(tmp_path)
        self._write_matrix(
            tmp_path,
            atom_pairs=np.array([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=np.int32),
            chunk_boundaries=np.array([0, 2, 1], dtype=np.int32),
            chunk_shapes=np.array([[1, 1], [1, 1]], dtype=np.int32),
            entries=np.array([1.0]),
        )

        with pytest.raises(AimspyConfigError, match="non-decreasing"):
            DeepHData.from_directory(tmp_path)

    def test_block_shape_must_match_orbital_map(self, tmp_path):
        self._write_metadata(tmp_path)
        self._write_matrix(
            tmp_path,
            chunk_boundaries=np.array([0, 2], dtype=np.int32),
            chunk_shapes=np.array([[2, 1]], dtype=np.int32),
            entries=np.ones(2),
        )

        with pytest.raises(AimspyConfigError, match="expected.*1, 1"):
            DeepHData.from_directory(tmp_path)

    def test_overlap_only_can_define_canonical_layout(self, tmp_path):
        self._write_metadata(tmp_path)
        self._write_matrix(tmp_path, filename="overlap.h5", entries=np.array([0.5]))

        data = DeepHData.from_directory(tmp_path)

        assert data.entries is None
        np.testing.assert_allclose(data.overlap_entries, [0.5])

    @pytest.mark.parametrize(
        ("filename", "attribute"),
        [
            ("hamiltonian.h5", "entries"),
            ("overlap.h5", "overlap_entries"),
            ("hamiltonian_init.h5", "initial_hamiltonian_entries"),
        ],
    )
    def test_each_standard_matrix_file_can_load_alone(
        self, tmp_path, filename, attribute
    ):
        self._write_metadata(tmp_path)
        self._write_matrix(tmp_path, filename=filename, entries=np.array([0.75]))

        data = DeepHData.from_directory(tmp_path)

        np.testing.assert_allclose(getattr(data, attribute), [0.75])

    def test_all_standard_matrix_files_validate_and_load(self, tmp_path):
        self._write_metadata(tmp_path)
        self._write_matrix(tmp_path, entries=np.array([1.0]))
        self._write_matrix(tmp_path, filename="overlap.h5", entries=np.array([0.5]))
        self._write_matrix(
            tmp_path,
            filename="hamiltonian_init.h5",
            entries=np.array([0.25]),
        )

        data = DeepHData.from_directory(tmp_path)

        np.testing.assert_allclose(data.entries, [1.0])
        np.testing.assert_allclose(data.overlap_entries, [0.5])
        np.testing.assert_allclose(data.initial_hamiltonian_entries, [0.25])

    def test_second_matrix_with_wrong_shape_rejected(self, tmp_path):
        self._make_valid_directory(tmp_path)
        self._write_matrix(
            tmp_path,
            filename="overlap.h5",
            chunk_boundaries=np.array([0, 2], dtype=np.int32),
            chunk_shapes=np.array([[1, 2]], dtype=np.int32),
            entries=np.ones(2),
        )

        with pytest.raises(AimspyConfigError) as exc_info:
            DeepHData.from_directory(tmp_path)

        assert "overlap.h5: chunk_shapes:" in str(exc_info.value)

    def test_invalid_electric_response_shape_rejected(self, tmp_path):
        self._make_valid_directory(tmp_path)
        self._write_matrix(
            tmp_path,
            filename="electric_response.h5",
            chunk_boundaries=np.array([0, 2], dtype=np.int32),
            chunk_shapes=np.array([[2, 1]], dtype=np.int32),
            entries=np.ones(2),
        )

        with pytest.raises(AimspyConfigError) as exc_info:
            DeepHData.from_directory(tmp_path)

        assert "electric_response.h5: chunk_shapes:" in str(exc_info.value)

    def test_invalid_electric_response_entries_length_rejected(self, tmp_path):
        self._make_valid_directory(tmp_path)
        self._write_matrix(
            tmp_path,
            filename="electric_response.h5",
            chunk_boundaries=np.array([0, 3], dtype=np.int32),
            chunk_shapes=np.array([[3, 1]], dtype=np.int32),
            entries=np.ones(2),
        )

        with pytest.raises(AimspyConfigError) as exc_info:
            DeepHData.from_directory(tmp_path)

        assert "electric_response.h5: entries:" in str(exc_info.value)

    def test_invalid_in_memory_layout_rejected_before_write(self, tmp_path):
        data = DeepHData(
            lattice=np.eye(3),
            atom_symbols=["H"],
            atom_coords=np.zeros((1, 3)),
            elements_orbital_map={"H": [0]},
            n_basis=1,
            atom_pairs=np.array([[0, 0, 0, 0, 0]], dtype=np.int32),
            chunk_boundaries=np.array([0, 1], dtype=np.int32),
            chunk_shapes=np.array([[1, 1]], dtype=np.int32),
            entries=np.array([1.0, 2.0]),
            path=tmp_path,
        )

        with pytest.raises(AimspyConfigError) as exc_info:
            data.save_hamiltonian()

        assert "hamiltonian.h5: entries:" in str(exc_info.value)
        assert not (tmp_path / "hamiltonian.h5").exists()


# =============================================================================
# Regression tests: first_order three-direction completeness (2.6) + save
# guard (2.7)
# =============================================================================
class TestFirstOrderValidation:
    def test_from_memory_missing_direction_raises(self):
        """first_order with an empty direction dict raises."""
        from aimspy import AimspyConfigError

        fo = _make_three_first_order_blocks()
        fo[1] = {}  # y direction empty
        with pytest.raises(AimspyConfigError, match="all three directions"):
            DeepHData.from_memory(
                lattice=np.eye(3) * 10.0,
                atom_symbols=["Mo", "S", "S"],
                atom_coords=np.zeros((3, 3)),
                elements_orbital_map={"Mo": [0, 0, 1], "S": [0, 0]},
                hamiltonian_blocks=_make_simple_blocks(),
                first_order_hamiltonian_blocks=fo,
            )

    def test_save_first_order_without_layout_raises(self, tmp_path):
        """Manually setting entries without _fo_chunk_* raises on save."""
        from aimspy import AimspyConfigError

        dd = DeepHData.from_memory(
            lattice=np.eye(3) * 10.0,
            atom_symbols=["Mo", "S", "S"],
            atom_coords=np.zeros((3, 3)),
            elements_orbital_map={"Mo": [0, 0, 1], "S": [0, 0]},
            hamiltonian_blocks=_make_simple_blocks(),
            path=tmp_path,
        )
        # Manually set entries but leave _fo_chunk_* as None
        dd.first_order_hamiltonian_entries = np.zeros(6, dtype=np.float64)
        with pytest.raises(AimspyConfigError, match="chunk layout"):
            dd.save_first_order_hamiltonian()
