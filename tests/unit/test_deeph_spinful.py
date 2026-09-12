"""Unit tests for spinful (collinear, n_spin=2) DeepH data handling.

Covers the dual-layout contract: Hamiltonian-class matrices use the
doubled ``(2*n_rows, n_cols)`` chunk layout (each atom pair's flat segment
is ``[alpha block ‖ beta block]``) while the spin-independent overlap keeps
the standard ``(n_rows, n_cols)`` layout — on disk and in memory.

The mock structure is physically consistent (full shells): one S (1 s
orbital), one Mo (1 s + one full p shell = 4 orbitals), one S — with a
non-trivial aims↔POSCAR atom permutation (``["S", "Mo", "S"]`` in aims
order → ``["Mo", "S", "S"]`` in POSCAR order).
"""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest

from aimspy import AimspyConfigError, DeepHData
from aimspy.interface.deeph.data import _validate_matrix_layout
from aimspy.matrix import AimspyMatrix
from aimspy.structure import AimspyStructure

EOM = {"Mo": [0, 1], "S": [0]}  # Mo: 1+3=4 orbitals, S: 1 orbital


def _make_structure() -> AimspyStructure:
    return AimspyStructure(
        n_atoms=3,
        n_basis=6,
        n_spin=2,
        n_periodic=3,
        lattice=np.eye(3) * 10.0,
        atom_symbols=["S", "Mo", "S"],
        atom_coords=np.array([[0.0, 0.0, 0.0], [1.5, 1.5, 0.0], [1.5, 1.5, 3.0]]),
        # atom 0 = S (s); atom 1 = Mo (s + p[-1,0,+1]); atom 2 = S (s)
        basis_atom=np.array([0, 1, 1, 1, 1, 2], dtype=np.int32),
        basis_l=np.array([0, 0, 1, 1, 1, 0], dtype=np.int32),
        basis_m=np.array([0, 0, -1, 0, 1, 0], dtype=np.int32),
    )


def _make_spin_hamiltonian() -> AimspyMatrix:
    """Stacked (2*n_orb_i, n_orb_j) blocks in aims atom order, Hartree."""
    rng = np.random.default_rng(5)
    blocks = {
        (0, 0, 0, 1, 1): rng.standard_normal((8, 4)),  # Mo-Mo
        (0, 0, 0, 1, 0): rng.standard_normal((8, 1)),  # Mo-S
        (0, 0, 0, 0, 2): rng.standard_normal((2, 1)),  # S-S
        (1, 0, 0, 0, 2): rng.standard_normal((2, 1)),  # S-S at R=(1,0,0)
    }
    return AimspyMatrix(blocks=blocks, n_spin=2)


def _make_overlap() -> AimspyMatrix:
    """Standard (n_orb_i, n_orb_j) blocks in aims atom order."""
    rng = np.random.default_rng(6)
    blocks = {
        (0, 0, 0, 1, 1): rng.standard_normal((4, 4)),
        (0, 0, 0, 1, 0): rng.standard_normal((4, 1)),
        (0, 0, 0, 0, 2): rng.standard_normal((1, 1)),
        (1, 0, 0, 0, 2): rng.standard_normal((1, 1)),
    }
    return AimspyMatrix(blocks=blocks, n_spin=1)


def _make_spin_init_hamiltonian() -> AimspyMatrix:
    rng = np.random.default_rng(7)
    blocks = {
        (0, 0, 0, 1, 1): rng.standard_normal((8, 4)),
        (0, 0, 0, 1, 0): rng.standard_normal((8, 1)),
        (0, 0, 0, 0, 2): rng.standard_normal((2, 1)),
        (1, 0, 0, 0, 2): rng.standard_normal((2, 1)),
    }
    return AimspyMatrix(blocks=blocks, n_spin=2)


def _poscar_symbols() -> list[str]:
    # atoms_species_sorted: stable sort of ["S", "Mo", "S"] → ["Mo", "S", "S"]
    return ["Mo", "S", "S"]


def _to_poscar_blocks(mx: AimspyMatrix) -> dict:
    """Reorder an aims-ordered block dict to POSCAR atom order."""
    from aimspy.interface.deeph.data import _aimspy_blocks_to_poscar

    return _aimspy_blocks_to_poscar(mx, _make_structure())


class TestFromAimspySpinful:
    def test_dual_layout_flags(self):
        struct = _make_structure()
        dd = DeepHData.from_aimspy(
            structure=struct,
            hamiltonian=_make_spin_hamiltonian(),
            overlap=_make_overlap(),
            initial_hamiltonian=_make_spin_init_hamiltonian(),
        )
        assert dd._spinful is True
        assert dd.atom_symbols == _poscar_symbols()
        # Hamiltonian-class layout is doubled.
        assert np.all(dd.chunk_shapes[:, 0] == 2 * np.array([4, 4, 1, 1]))
        # Overlap layout is standard.
        assert np.all(dd._ovlp_chunk_shapes[:, 0] == np.array([4, 4, 1, 1]))
        # Entry counts: doubled for H-class, standard for the overlap.
        h_size = int(np.sum(dd.chunk_shapes[:, 0] * dd.chunk_shapes[:, 1]))
        s_size = int(np.sum(dd._ovlp_chunk_shapes[:, 0] * dd._ovlp_chunk_shapes[:, 1]))
        assert dd.entries.shape == (h_size,)
        assert dd.overlap_entries.shape == (s_size,)
        assert dd.initial_hamiltonian_entries.shape == (h_size,)

    def test_entries_interleaving(self):
        """Each pair's H segment is [alpha ‖ beta] in eV."""
        from aimspy.data import HARTREE_TO_EV

        struct = _make_structure()
        h = _make_spin_hamiltonian()
        dd = DeepHData.from_aimspy(structure=struct, hamiltonian=h)
        # POSCAR pair (0, 0, 0, 0, 0) = aims pair (0, 0, 0, 1, 1) (Mo-Mo).
        ip = 0  # sorted keys put (0,0,0,0,0) first
        nr, nc = int(dd.chunk_shapes[ip, 0]), int(dd.chunk_shapes[ip, 1])
        assert (nr, nc) == (8, 4)
        bnd = int(dd.chunk_boundaries[ip])
        seg = dd.entries[bnd : bnd + nr * nc]
        expected = h.blocks[(0, 0, 0, 1, 1)] * HARTREE_TO_EV
        np.testing.assert_allclose(seg.reshape(8, 4), expected, rtol=1e-12)

    def test_nspin_mismatch_raises(self):
        struct = _make_structure()
        with pytest.raises(AimspyConfigError, match="must share n_spin"):
            DeepHData.from_aimspy(
                structure=struct,
                hamiltonian=_make_spin_hamiltonian(),
                initial_hamiltonian=_make_overlap(),  # n_spin=1
            )

    def test_stacked_overlap_raises(self):
        struct = _make_structure()
        with pytest.raises(AimspyConfigError, match="spin-independent"):
            DeepHData.from_aimspy(
                structure=struct,
                hamiltonian=_make_spin_hamiltonian(),
                overlap=_make_spin_hamiltonian(),  # n_spin=2 overlap
            )

    def test_spinless_path_unchanged(self):
        struct = _make_structure()
        dd = DeepHData.from_aimspy(
            structure=struct,
            hamiltonian=_make_overlap(),  # any n_spin=1 matrix
        )
        assert dd._spinful is False
        assert dd._ovlp_chunk_shapes is None
        assert np.all(dd.chunk_shapes[:, 0] == np.array([4, 4, 1, 1]))

    def test_from_aimspy_stacked_first_order_rejected(self):
        """Stacked dH/de matrices on a spinless dataset are rejected at
        from_aimspy (previously: entries inconsistent with the 3x chunk
        layout, failing only at save time)."""
        struct = _make_structure()
        with pytest.raises(AimspyConfigError, match="spinless matrices only"):
            DeepHData.from_aimspy(
                structure=struct,
                hamiltonian=_make_overlap(),
                first_order_hamiltonian=[_make_spin_hamiltonian() for _ in range(3)],
            )


class TestRoundTrip:
    def test_save_load_to_aimspy(self, tmp_path):
        struct = _make_structure()
        h = _make_spin_hamiltonian()
        s = _make_overlap()
        dd = DeepHData.from_aimspy(
            structure=struct, hamiltonian=h, overlap=s, path=tmp_path
        )
        dd.save()

        # On-disk contract.
        with open(tmp_path / "info.json") as f:
            info_json = json.load(f)
        assert info_json["spinful"] is True
        assert info_json["spin_treatment"] == "collinear"
        with h5py.File(tmp_path / "hamiltonian.h5") as f:
            assert np.all(f["chunk_shapes"][:, 0] == 2 * np.array([4, 4, 1, 1]))
        with h5py.File(tmp_path / "overlap.h5") as f:
            assert np.all(f["chunk_shapes"][:, 0] == np.array([4, 4, 1, 1]))

        dd2 = DeepHData.from_directory(tmp_path)
        assert dd2._spinful is True
        np.testing.assert_allclose(dd2.entries, dd.entries, rtol=0)
        np.testing.assert_allclose(dd2.overlap_entries, dd.overlap_entries, rtol=0)
        np.testing.assert_array_equal(dd2.chunk_shapes, dd.chunk_shapes)
        np.testing.assert_array_equal(dd2._ovlp_chunk_shapes, dd._ovlp_chunk_shapes)

        # to_aimspy unpacks [alpha ‖ beta] back into stacked blocks.
        mx = dd2.to_aimspy(struct)
        assert mx.n_spin == 2
        assert set(mx.blocks) == set(h.blocks)
        for key, blk in h.blocks.items():
            np.testing.assert_allclose(mx.blocks[key], blk, rtol=1e-12)

    def test_spinless_round_trip_unchanged(self, tmp_path):
        struct = _make_structure()
        s = _make_overlap()
        dd = DeepHData.from_aimspy(structure=struct, hamiltonian=s, path=tmp_path)
        dd.save()
        dd2 = DeepHData.from_directory(tmp_path)
        assert dd2._spinful is False
        mx = dd2.to_aimspy(struct)
        assert mx.n_spin == 1
        for key, blk in s.blocks.items():
            np.testing.assert_allclose(mx.blocks[key], blk, rtol=1e-12)


class TestFromMemorySpinful:
    def _base_kwargs(self):
        return dict(
            lattice=np.eye(3) * 10.0,
            atom_symbols=_poscar_symbols(),
            atom_coords=np.zeros((3, 3)),
            elements_orbital_map=dict(EOM),
        )

    def test_overlap_only_raises(self):
        with pytest.raises(AimspyConfigError, match="spinful=True requires"):
            DeepHData.from_memory(
                **self._base_kwargs(),
                overlap_blocks=_make_overlap().blocks,
                spinful=True,
            )

    def test_first_order_rejected(self):
        with pytest.raises(AimspyConfigError, match="not yet supported"):
            DeepHData.from_memory(
                **self._base_kwargs(),
                hamiltonian_blocks=_make_spin_hamiltonian().blocks,
                first_order_hamiltonian_blocks=[
                    _make_overlap().blocks for _ in range(3)
                ],
                spinful=True,
            )

    def test_unstacked_hamiltonian_block_raises(self):
        with pytest.raises(AimspyConfigError, match="expected stacked"):
            DeepHData.from_memory(
                **self._base_kwargs(),
                hamiltonian_blocks=_make_overlap().blocks,  # standard shape
                spinful=True,
            )

    def test_stacked_overlap_block_raises(self):
        with pytest.raises(AimspyConfigError, match="spin-independent"):
            DeepHData.from_memory(
                **self._base_kwargs(),
                hamiltonian_blocks=_to_poscar_blocks(_make_spin_hamiltonian()),
                overlap_blocks=_to_poscar_blocks(_make_spin_hamiltonian()),
                spinful=True,
            )

    def test_missing_overlap_keys_zero_filled_standard(self):
        """H keys without an S counterpart zero-fill at the standard size."""
        h = _make_spin_hamiltonian()
        # POSCAR pair (0,0,0,0,0) = aims Mo-Mo; keep only its overlap block.
        partial_s = {(0, 0, 0, 0, 0): np.ones((4, 4))}
        dd = DeepHData.from_memory(
            **self._base_kwargs(),
            hamiltonian_blocks=_to_poscar_blocks(h),
            overlap_blocks=partial_s,
            spinful=True,
        )
        assert dd._spinful is True
        assert dd._ovlp_chunk_shapes is not None
        # 4 pairs: one real (4x4=16) + three zero-filled standard blocks
        # (4x1, 1x1, 1x1).
        np.testing.assert_allclose(dd.overlap_entries[:16], np.ones(16), rtol=0)
        assert not dd.overlap_entries[16:].any()
        assert dd.overlap_entries.shape == (16 + 4 + 1 + 1,)

    def test_short_key_raises_config_error(self):
        """A key shorter than 5 elements is a config error, not IndexError."""
        with pytest.raises(AimspyConfigError, match="5-tuple"):
            DeepHData.from_memory(
                **self._base_kwargs(),
                hamiltonian_blocks={(0, 0, 0): np.zeros((8, 4))},
                spinful=True,
            )

    def test_atom_index_out_of_range_raises_config_error(self):
        with pytest.raises(AimspyConfigError, match="out of range"):
            DeepHData.from_memory(
                **self._base_kwargs(),
                hamiltonian_blocks={(0, 0, 0, 0, 5): np.zeros((8, 1))},
                spinful=True,
            )

    def test_non_integer_atom_index_raises_config_error(self):
        with pytest.raises(AimspyConfigError, match="non-integer"):
            DeepHData.from_memory(
                **self._base_kwargs(),
                hamiltonian_blocks={(0, 0, 0, "a", 0): np.zeros((8, 1))},
                spinful=True,
            )


class TestFromDirectorySpinful:
    def _write_poscar_info(self, path, spinful=True, treatment="collinear"):
        from aimspy.interface.deeph.data import _write_poscar

        _write_poscar(
            path / "POSCAR",
            np.eye(3) * 10.0,
            _poscar_symbols(),
            np.zeros((3, 3)),
        )
        info = {"elements_orbital_map": EOM, "spinful": spinful}
        if spinful and treatment is not None:
            info["spin_treatment"] = treatment
        with open(path / "info.json", "w") as f:
            json.dump(info, f)

    def _write_matrix(self, path, name, ap, cb, cs, entries):
        with h5py.File(path / f"{name}.h5", "w") as f:
            f.create_dataset("atom_pairs", data=ap, dtype="i4")
            f.create_dataset("chunk_boundaries", data=cb, dtype="i4")
            f.create_dataset("chunk_shapes", data=cs, dtype="i4")
            f.create_dataset("entries", data=entries)

    def test_overlap_only_raises(self, tmp_path):
        self._write_poscar_info(tmp_path)
        ap = np.array([[0, 0, 0, 0, 0]], dtype=np.int32)
        cb = np.array([0, 1], dtype=np.int32)
        cs = np.array([[4, 4]], dtype=np.int32)
        self._write_matrix(tmp_path, "overlap", ap, cb, cs, np.zeros(16))
        with pytest.raises(AimspyConfigError, match="spinful=true requires"):
            DeepHData.from_directory(tmp_path)

    def test_electric_response_rejected(self, tmp_path):
        self._write_poscar_info(tmp_path)
        ap = np.array([[0, 0, 0, 0, 0]], dtype=np.int32)
        cb = np.array([0, 32], dtype=np.int32)
        cs = np.array([[8, 4]], dtype=np.int32)
        self._write_matrix(tmp_path, "hamiltonian", ap, cb, cs, np.zeros(32))
        # electric_response.h5 present (contents irrelevant — guard fires first)
        self._write_matrix(
            tmp_path,
            "electric_response",
            ap,
            np.array([0, 3]),
            np.array([[1, 1]]),
            np.zeros(3),
        )
        with pytest.raises(AimspyConfigError, match="not yet supported"):
            DeepHData.from_directory(tmp_path)

    def test_canonical_from_hamiltonian_class(self, tmp_path):
        """With hamiltonian.h5 absent, the doubled canonical layout comes
        from hamiltonian_init.h5 (overlap.h5 alone must not define it)."""
        self._write_poscar_info(tmp_path)
        ap = np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 1]], dtype=np.int32)
        cb_h = np.array([0, 32, 40], dtype=np.int32)
        cs_h = np.array([[8, 4], [8, 1]], dtype=np.int32)
        self._write_matrix(tmp_path, "hamiltonian_init", ap, cb_h, cs_h, np.zeros(40))
        cb_s = np.array([0, 16, 20], dtype=np.int32)
        cs_s = np.array([[4, 4], [4, 1]], dtype=np.int32)
        self._write_matrix(tmp_path, "overlap", ap, cb_s, cs_s, np.arange(20.0))
        dd = DeepHData.from_directory(tmp_path)
        assert dd._spinful is True
        np.testing.assert_array_equal(dd.chunk_shapes, cs_h)
        np.testing.assert_array_equal(dd._ovlp_chunk_shapes, cs_s)
        np.testing.assert_allclose(dd.overlap_entries, np.arange(20.0), rtol=0)

    def test_legacy_without_treatment_raises(self, tmp_path):
        """spinful without spin_treatment (legacy four-quadrant layout,
        e.g. ref/aims_to_deeph.py output) is rejected with a clear error."""
        self._write_poscar_info(tmp_path, spinful=True, treatment=None)
        ap = np.array([[0, 0, 0, 0, 0]], dtype=np.int32)
        cb = np.array([0, 32], dtype=np.int32)
        cs = np.array([[8, 4]], dtype=np.int32)
        self._write_matrix(tmp_path, "hamiltonian", ap, cb, cs, np.zeros(32))
        with pytest.raises(AimspyConfigError, match="four-quadrant"):
            DeepHData.from_directory(tmp_path)

    def test_unsupported_treatment_raises(self, tmp_path):
        self._write_poscar_info(tmp_path, spinful=True, treatment="non_collinear")
        ap = np.array([[0, 0, 0, 0, 0]], dtype=np.int32)
        cb = np.array([0, 32], dtype=np.int32)
        cs = np.array([[8, 4]], dtype=np.int32)
        self._write_matrix(tmp_path, "hamiltonian", ap, cb, cs, np.zeros(32))
        with pytest.raises(AimspyConfigError, match="only 'collinear'"):
            DeepHData.from_directory(tmp_path)

    def test_spinless_without_treatment_ok(self, tmp_path):
        """Spinless data never carries spin_treatment (backward compatible)."""
        self._write_poscar_info(tmp_path, spinful=False)
        ap = np.array([[0, 0, 0, 0, 0]], dtype=np.int32)
        cb = np.array([0, 16], dtype=np.int32)
        cs = np.array([[4, 4]], dtype=np.int32)
        self._write_matrix(tmp_path, "hamiltonian", ap, cb, cs, np.arange(16.0))
        dd = DeepHData.from_directory(tmp_path)
        assert dd._spinful is False
        np.testing.assert_allclose(dd.entries, np.arange(16.0), rtol=0)


class TestSetters:
    def _spinful_dd(self):
        struct = _make_structure()
        return DeepHData.from_aimspy(
            structure=struct, hamiltonian=_make_spin_hamiltonian()
        )

    def test_set_hamiltonian_nspin_guard(self):
        dd = self._spinful_dd()
        with pytest.raises(AimspyConfigError, match="n_spin=1"):
            dd.set_hamiltonian(_make_overlap(), _make_structure())

    def test_set_initial_hamiltonian_nspin_guard(self):
        dd = self._spinful_dd()
        with pytest.raises(AimspyConfigError, match="n_spin=1"):
            dd.set_initial_hamiltonian(_make_overlap(), _make_structure())

    def test_set_overlap_nspin_guard(self):
        dd = self._spinful_dd()
        with pytest.raises(AimspyConfigError, match="spin-independent"):
            dd.set_overlap(_make_spin_hamiltonian(), _make_structure())

    def test_set_overlap_lazy_layout_derivation(self):
        dd = self._spinful_dd()
        assert dd._ovlp_chunk_shapes is None
        s = _make_overlap()
        dd.set_overlap(s, _make_structure())
        assert dd._ovlp_chunk_shapes is not None
        assert np.all(dd._ovlp_chunk_shapes[:, 0] == np.array([4, 4, 1, 1]))
        assert dd.overlap_entries.shape == (16 + 4 + 1 + 1,)
        np.testing.assert_allclose(
            dd.overlap_entries[:16], s.blocks[(0, 0, 0, 1, 1)].ravel(), rtol=1e-15
        )

    def test_spinless_set_hamiltonian_stacked_guard(self):
        struct = _make_structure()
        dd = DeepHData.from_aimspy(structure=struct, hamiltonian=_make_overlap())
        with pytest.raises(AimspyConfigError, match="n_spin=2"):
            dd.set_hamiltonian(_make_spin_hamiltonian(), struct)

    def test_set_first_order_spinful_rejected(self):
        """The doubled chunk layout cannot be combined with the 3x
        first-order expansion — reject up front with a clear error."""
        dd = self._spinful_dd()
        with pytest.raises(AimspyConfigError, match="dH/de"):
            dd.set_first_order_hamiltonian([None, None, None], _make_structure())

    def test_set_first_order_stacked_matrices_rejected(self):
        """Stacked (n_spin=2) dH/de matrices through a spinless DeepHData
        previously built entries inconsistent with the 3x chunk layout
        (only caught at save time); now rejected up front."""
        struct = _make_structure()
        dd = DeepHData.from_aimspy(structure=struct, hamiltonian=_make_overlap())
        stacked = [_make_spin_hamiltonian() for _ in range(3)]
        with pytest.raises(AimspyConfigError, match="n_spin=1"):
            dd.set_first_order_hamiltonian(stacked, struct)


class TestLayoutValidation:
    """_validate_matrix_layout with an explicit row_multiplier of 2."""

    def _args(self):
        # POSCAR pair (0, 0, 0, 1, 1) = S-S: 1 orbital per S → doubled (2, 1).
        ap = np.array([[0, 0, 0, 1, 1]], dtype=np.int32)
        cb = np.array([0, 2], dtype=np.int32)
        cs = np.array([[2, 1]], dtype=np.int32)
        entries = np.zeros(2)
        return ap, cb, cs, entries

    def test_multiplier_2_accepts_doubled(self):
        ap, cb, cs, entries = self._args()
        layout = _validate_matrix_layout(
            "hamiltonian.h5",
            ap,
            cb,
            cs,
            entries,
            _poscar_symbols(),
            EOM,
            row_multiplier=2,
        )
        np.testing.assert_array_equal(layout.chunk_shapes, cs)

    def test_multiplier_1_rejects_doubled(self):
        ap, cb, cs, entries = self._args()
        with pytest.raises(AimspyConfigError, match="chunk_shapes"):
            _validate_matrix_layout(
                "hamiltonian.h5",
                ap,
                cb,
                cs,
                entries,
                _poscar_symbols(),
                EOM,
                row_multiplier=1,
            )


class TestEndToEndChain:
    """Full chain: CSR (2, n_ham) → stacked blocks → spinful DeepH on disk
    → stacked blocks → CSR — closes exactly."""

    N_BASIS = 6  # S(s) + Mo(s+p) + S(s)
    N_PER_CELL = N_BASIS * N_BASIS
    N_HAM = 2 * N_PER_CELL + 1  # ±x cells + trash slot

    def _make_csr(self, seed=31):
        from aimspy import CsrMatrixDescriptor

        rng = np.random.default_rng(seed)
        m_up = rng.standard_normal((self.N_BASIS, self.N_BASIS))
        m_dn = rng.standard_normal((self.N_BASIS, self.N_BASIS))
        h = np.zeros((2, self.N_HAM), dtype=np.float64)
        for ib in range(self.N_BASIS):
            for jb in range(self.N_BASIS):
                flat = ib * self.N_BASIS + jb
                off = self.N_PER_CELL + flat
                h[0, flat] = m_up[ib, jb]
                h[1, flat] = m_dn[ib, jb]
                # −x cell carries the transpose → Hermitian-consistent CSR.
                h[0, off] = m_up[jb, ib]
                h[1, off] = m_dn[jb, ib]

        n_cells = 3
        cell_idx = np.zeros((3, n_cells), dtype=np.int32)
        cell_idx[:, 0] = (1, 0, 0)
        cell_idx[:, 1] = (-1, 0, 0)
        cell_idx[:, 2] = np.iinfo(np.int32).max
        row_mx_idx = np.zeros((self.N_BASIS, n_cells, 2), dtype=np.int32)
        for ic in range(2):
            for ib in range(self.N_BASIS):
                start = ic * self.N_PER_CELL + ib * self.N_BASIS + 1
                row_mx_idx[ib, ic] = (start, start + self.N_BASIS - 1)
        col_mx_idx = np.zeros((self.N_HAM,), dtype=np.int32)
        for ic in range(2):
            for ib in range(self.N_BASIS):
                base = ic * self.N_PER_CELL + ib * self.N_BASIS
                col_mx_idx[base : base + self.N_BASIS] = np.arange(1, self.N_BASIS + 1)
        descr = CsrMatrixDescriptor(
            n_basis=self.N_BASIS,
            n_spin=2,
            n_cells=n_cells,
            n_ham_size=self.N_HAM,
            cell_idx=cell_idx,
            row_mx_idx=row_mx_idx,
            col_mx_idx=col_mx_idx,
        )
        return h, descr

    def test_csr_blocks_deeph_disk_blocks_csr(self, tmp_path):
        from aimspy import AimspyMatrix

        struct = _make_structure()
        h, descr = self._make_csr()
        mx = AimspyMatrix.from_aims_csr(h, descr, struct)
        assert mx.n_spin == 2
        ovlp = AimspyMatrix.from_aims_csr(
            h[:1], descr, struct
        )  # spin-independent data → standard blocks
        assert ovlp.n_spin == 1

        dd = DeepHData.from_aimspy(
            structure=struct, hamiltonian=mx, overlap=ovlp, path=tmp_path
        )
        dd.save()
        dd2 = DeepHData.from_directory(tmp_path)
        mx2 = dd2.to_aimspy(struct)
        assert mx2.n_spin == 2
        out = mx2.to_aims_csr(descr, struct)
        # rtol covers the eV↔Hartree double conversion through DeepH.
        np.testing.assert_allclose(out, h, rtol=1e-12)
