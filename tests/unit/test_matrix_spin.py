"""Unit tests for spin-collinear (n_spin=2) CSR ↔ block conversion.

Synthetic fixture: 2 atoms × 2 orbitals (one p orbital with m=+1 exercises
the wiki parity), two active cells ±x plus the sentinel cell, and full 4×4
sparsity per cell.  The −x cell carries the transpose of the +x cell's
values so the CSR is Hermitian-consistent (``H(ib,jb)@+R == H(jb,ib)@−R``),
exercising the Hermitian write-or-verify path of ``from_aims_csr``.
"""

from __future__ import annotations

import numpy as np
import pytest

from aimspy import AimspyMatrix, AimspyStructure, CsrMatrixDescriptor
from aimspy._exceptions import AimspyError

SENTINEL = np.iinfo(np.int32).max
N_BASIS = 4
N_ENTRIES_PER_CELL = N_BASIS * N_BASIS
N_HAM = 2 * N_ENTRIES_PER_CELL + 1  # two cells + trash slot
PHASE = np.array([1, -1, 1, 1], dtype=np.int64)  # wiki parity per basis


def _make_structure() -> AimspyStructure:
    return AimspyStructure(
        n_atoms=2,
        n_basis=N_BASIS,
        n_spin=2,
        n_periodic=3,
        lattice=np.eye(3, dtype=np.float64),
        atom_symbols=["A", "B"],
        atom_coords=np.zeros((2, 3), dtype=np.float64),
        basis_atom=np.array([0, 0, 1, 1], dtype=np.int32),
        # basis 1 is a p orbital with m=+1 → wiki parity −1
        basis_l=np.array([0, 1, 0, 1], dtype=np.int32),
        basis_m=np.array([0, 1, 0, -1], dtype=np.int32),
    )


def _make_descriptor(n_spin: int) -> CsrMatrixDescriptor:
    n_cells = 3  # +x, −x, sentinel
    cell_idx = np.zeros((3, n_cells), dtype=np.int32)
    cell_idx[:, 0] = (1, 0, 0)
    cell_idx[:, 1] = (-1, 0, 0)
    cell_idx[:, 2] = SENTINEL

    row_mx_idx = np.zeros((N_BASIS, n_cells, 2), dtype=np.int32)
    for ic in range(2):
        for ib in range(N_BASIS):
            start = ic * N_ENTRIES_PER_CELL + ib * N_BASIS + 1
            row_mx_idx[ib, ic, 0] = start
            row_mx_idx[ib, ic, 1] = start + N_BASIS - 1

    col_mx_idx = np.zeros((N_HAM,), dtype=np.int32)
    for ic in range(2):
        for ib in range(N_BASIS):
            base = ic * N_ENTRIES_PER_CELL + ib * N_BASIS
            col_mx_idx[base : base + N_BASIS] = np.arange(1, N_BASIS + 1)

    return CsrMatrixDescriptor(
        n_basis=N_BASIS,
        n_spin=n_spin,
        n_cells=n_cells,
        n_ham_size=N_HAM,
        cell_idx=cell_idx,
        row_mx_idx=row_mx_idx,
        col_mx_idx=col_mx_idx,
    )


def _make_values(seed: int = 7, hermitian: bool = True):
    """Build a (2, N_HAM) CSR value array plus the per-channel +x matrices."""
    rng = np.random.default_rng(seed)
    m_up = rng.standard_normal((N_BASIS, N_BASIS))
    m_dn = rng.standard_normal((N_BASIS, N_BASIS))
    if hermitian:
        n_up, n_dn = m_up.T, m_dn.T
    else:
        n_up = rng.standard_normal((N_BASIS, N_BASIS))
        n_dn = rng.standard_normal((N_BASIS, N_BASIS))
    h = np.zeros((2, N_HAM), dtype=np.float64)
    for ib in range(N_BASIS):
        for jb in range(N_BASIS):
            flat = ib * N_BASIS + jb
            off = N_ENTRIES_PER_CELL + flat
            h[0, flat] = m_up[ib, jb]
            h[1, flat] = m_dn[ib, jb]
            h[0, off] = n_up[ib, jb]
            h[1, off] = n_dn[ib, jb]
    return h, m_up, m_dn


def _expected_blocks(m):
    """Parity-applied (2, 2) block per atom pair, fed by the +x cell."""
    out = {}
    for ai in range(2):
        for aj in range(2):
            blk = np.zeros((2, 2), dtype=np.float64)
            for oi in range(2):
                for oj in range(2):
                    ib, jb = ai * 2 + oi, aj * 2 + oj
                    blk[oi, oj] = m[ib, jb] * PHASE[ib] * PHASE[jb]
            out[(ai, aj)] = blk
    return out


class TestFromAimsCsrSpin2:
    def test_block_shapes_and_n_pairs(self):
        h, _, _ = _make_values()
        mx = AimspyMatrix.from_aims_csr(h, _make_descriptor(2), _make_structure())
        assert mx.n_spin == 2
        # 2 R values (±x) × 4 atom pairs
        assert mx.n_pairs == 8
        for key, blk in mx.blocks.items():
            assert blk.shape == (4, 2), (key, blk.shape)

    def test_alpha_beta_values_match_channels(self):
        h, m_up, m_dn = _make_values()
        mx = AimspyMatrix.from_aims_csr(h, _make_descriptor(2), _make_structure())
        exp_up = _expected_blocks(m_up)
        exp_dn = _expected_blocks(m_dn)
        # +x cell (R_aims=(1,0,0)) feeds the (−1,0,0,·,·) blocks.
        for ai in range(2):
            for aj in range(2):
                blk = mx.blocks[(-1, 0, 0, ai, aj)]
                np.testing.assert_allclose(blk[:2, :], exp_up[(ai, aj)], atol=0)
                np.testing.assert_allclose(blk[2:, :], exp_dn[(ai, aj)], atol=0)
                # −x cell feeds the (+1,0,0,·,·) blocks with the transpose.
                blk_t = mx.blocks[(1, 0, 0, ai, aj)]
                np.testing.assert_allclose(blk_t[:2, :], exp_up[(aj, ai)].T, atol=0)
                np.testing.assert_allclose(blk_t[2:, :], exp_dn[(aj, ai)].T, atol=0)

    def test_parity_applied(self):
        h, m_up, m_dn = _make_values()
        mx = AimspyMatrix.from_aims_csr(h, _make_descriptor(2), _make_structure())
        # basis 1 (m=+1, parity −1) on the atom-0 diagonal, alpha channel
        blk = mx.blocks[(-1, 0, 0, 0, 0)]
        assert blk[0, 1] == pytest.approx(m_up[0, 1] * -1.0)
        assert blk[2 + 0, 1] == pytest.approx(m_dn[0, 1] * -1.0)

    def test_hermitian_block_consistency(self):
        h, _, _ = _make_values()
        mx = AimspyMatrix.from_aims_csr(h, _make_descriptor(2), _make_structure())
        for (r1, r2, r3, ai, aj), blk in mx.blocks.items():
            rev = mx.blocks[(-r1, -r2, -r3, aj, ai)]
            np.testing.assert_allclose(blk[:2, :], rev[:2, :].T, atol=0)
            np.testing.assert_allclose(blk[2:, :], rev[2:, :].T, atol=0)

    def test_hermitian_check_failure(self):
        h, _, _ = _make_values(seed=11, hermitian=False)
        with pytest.raises(AimspyError, match="Hermitian check failed"):
            AimspyMatrix.from_aims_csr(h, _make_descriptor(2), _make_structure())


class TestToAimsCsrSpin2:
    def test_roundtrip_exact(self):
        h, _, _ = _make_values()
        descr = _make_descriptor(2)
        struct = _make_structure()
        out = AimspyMatrix.from_aims_csr(h, descr, struct).to_aims_csr(descr, struct)
        assert out.shape == (2, N_HAM)
        np.testing.assert_array_equal(out, h)

    def test_hermitian_fallback(self):
        h, _, _ = _make_values(seed=13)
        descr = _make_descriptor(2)
        struct = _make_structure()
        mx = AimspyMatrix.from_aims_csr(h, descr, struct)
        # Removing the forward (+1,0,0,0,1) block forces the reverse
        # (−1,0,0,1,0) lookup at the transposed position — values match.
        del mx.blocks[(1, 0, 0, 0, 1)]
        out = mx.to_aims_csr(descr, struct)
        np.testing.assert_array_equal(out, h)

    def test_missing_both_keys_zero_filled(self):
        h, _, _ = _make_values(seed=17)
        descr = _make_descriptor(2)
        struct = _make_structure()
        mx = AimspyMatrix.from_aims_csr(h, descr, struct)
        del mx.blocks[(1, 0, 0, 0, 1)]
        del mx.blocks[(-1, 0, 0, 1, 0)]
        out = mx.to_aims_csr(descr, struct)
        expected = h.copy()
        for ib in range(N_BASIS):
            for jb in range(N_BASIS):
                ai, aj = ib // 2, jb // 2
                if (ai, aj) == (0, 1):
                    expected[:, N_ENTRIES_PER_CELL + ib * N_BASIS + jb] = 0.0
                if (ai, aj) == (1, 0):
                    expected[:, ib * N_BASIS + jb] = 0.0
        np.testing.assert_array_equal(out, expected)


class TestBackwardCompat:
    def test_nspin1_roundtrip_unchanged(self):
        h2, _, _ = _make_values(seed=19)
        h = h2[:1]  # (1, N_HAM)
        descr = _make_descriptor(1)
        struct = _make_structure()
        mx = AimspyMatrix.from_aims_csr(h, descr, struct)
        assert mx.n_spin == 1
        for blk in mx.blocks.values():
            assert blk.shape == (2, 2)
        out = mx.to_aims_csr(descr, struct)
        assert out.shape == (1, N_HAM)
        np.testing.assert_array_equal(out, h)

    def test_spin_independent_data_with_spinful_descriptor(self):
        """Overlap-style input: (1, n_ham) data + n_spin=2 descriptor."""
        h2, _, _ = _make_values(seed=23)
        h = h2[:1]
        descr = _make_descriptor(2)
        struct = _make_structure()
        mx = AimspyMatrix.from_aims_csr(h, descr, struct)
        assert mx.n_spin == 1
        for blk in mx.blocks.values():
            assert blk.shape == (2, 2)
        # Reading is spin-independent, but writing back through the
        # spinful descriptor is rejected (would zero the beta channel).
        with pytest.raises(AimspyError, match="does not match"):
            mx.to_aims_csr(descr, struct)


class TestGuards:
    def test_unsupported_nspin(self):
        h = np.zeros((3, N_HAM), dtype=np.float64)
        with pytest.raises(AimspyError, match="unsupported n_spin=3"):
            AimspyMatrix.from_aims_csr(h, _make_descriptor(2), _make_structure())

    def test_two_channel_data_with_spinless_descriptor(self):
        h, _, _ = _make_values()
        with pytest.raises(AimspyError, match="2-channel data"):
            AimspyMatrix.from_aims_csr(h, _make_descriptor(1), _make_structure())

    def test_stacked_matrix_with_spinless_descriptor(self):
        h, _, _ = _make_values()
        descr2 = _make_descriptor(2)
        descr1 = _make_descriptor(1)
        struct = _make_structure()
        mx = AimspyMatrix.from_aims_csr(h, descr2, struct)
        with pytest.raises(AimspyError, match="does not match"):
            mx.to_aims_csr(descr1, struct)

    def test_spinless_matrix_with_spinful_descriptor(self):
        """Reverse direction: an n_spin=1 matrix cannot be written
        through an n_spin=2 descriptor (previously: silent half-write
        that filled only channel 0)."""
        h2, _, _ = _make_values(seed=29)
        descr = _make_descriptor(2)
        struct = _make_structure()
        mx = AimspyMatrix.from_aims_csr(h2[:1], descr, struct)
        assert mx.n_spin == 1
        with pytest.raises(AimspyError, match="does not match"):
            mx.to_aims_csr(descr, struct)


class TestApplyStrategy:
    """The modify-strategy n_spin guard (regression: cross-spin REPLACE
    silently zero-filled the Hamiltonian via to_aims_csr bounds checks)."""

    @staticmethod
    def _mspec(strategy):
        from types import SimpleNamespace

        return SimpleNamespace(strategy=strategy, factor=1.0, custom_fn=None)

    @staticmethod
    def _make_pair(live_nspin, ext_nspin):
        h, _, _ = _make_values()
        struct = _make_structure()
        live = AimspyMatrix.from_aims_csr(
            h[:live_nspin], _make_descriptor(live_nspin), struct
        )
        ext = AimspyMatrix.from_aims_csr(
            h[:ext_nspin], _make_descriptor(ext_nspin), struct
        )
        # Both matrices use the same (Hermitian-consistent) data, so any
        # failure below is attributable to the guard alone.
        return live, ext, struct

    def test_replace_mismatch_raises(self):
        from aimspy import Strategy
        from aimspy.calculator import _apply_strategy
        from aimspy._exceptions import AimspyConfigError

        for live_nspin, ext_nspin in ((2, 1), (1, 2)):
            live, ext, struct = self._make_pair(live_nspin, ext_nspin)
            with pytest.raises(AimspyConfigError, match="n_spin"):
                _apply_strategy(self._mspec(Strategy.REPLACE), live, ext, struct, {})

    def test_add_mismatch_raises(self):
        from aimspy import Strategy
        from aimspy.calculator import _apply_strategy
        from aimspy._exceptions import AimspyConfigError

        live, ext, struct = self._make_pair(2, 1)
        with pytest.raises(AimspyConfigError, match="n_spin"):
            _apply_strategy(self._mspec(Strategy.ADD), live, ext, struct, {})

    def test_replace_match_ok(self):
        from aimspy import Strategy
        from aimspy.calculator import _apply_strategy

        live, ext, struct = self._make_pair(2, 2)
        before = {k: v.copy() for k, v in ext.blocks.items()}
        _apply_strategy(self._mspec(Strategy.REPLACE), live, ext, struct, {})
        assert set(live.blocks) == set(before)
        for k, v in before.items():
            np.testing.assert_array_equal(live.blocks[k], v)

    def test_add_match_ok(self):
        from aimspy import Strategy
        from aimspy.calculator import _apply_strategy

        live, ext, struct = self._make_pair(2, 2)
        before = {k: v.copy() for k, v in live.blocks.items()}
        _apply_strategy(self._mspec(Strategy.ADD), live, ext, struct, {})
        for k, v in before.items():
            np.testing.assert_allclose(live.blocks[k], v + ext.blocks[k], rtol=0)
