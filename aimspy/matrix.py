"""Public — AimspyMatrix + aims↔aimspy format conversions.

The aimspy standard matrix format is a block-sparse real-space
representation:

    blocks: dict[tuple[int, int, int, int, int], np.ndarray]
           key = (R1, R2, R3, i_atom, j_atom)

Convensions
-----------
- *R*: ``R_aimspy = -R_aims`` (same sign as DeepH).
- *Atoms*: aims native order (no reordering).
- *Orbitals*: aims native basis order (no reordering).
- *Parity*: wiki/DeepH convention (``phase_i * phase_j`` already applied).
- *Units*: Hartree.
- *Hermitian partners*: both ``(R,i,j)`` and ``(-R,j,i)`` stored.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from .data import CsrMatrixDescriptor
from .structure import AimspyStructure


# =============================================================================
# Helper: pointer to ndarray copy  (unchanged from earlier version)
# =============================================================================
def _ptr_to_ndarray(ptr, shape, dtype=np.float64) -> np.ndarray:
    from ctypes import cast, c_void_p, POINTER, c_double as _cd
    n = 1
    for d in shape:
        n *= d
    try:
        flat = np.ctypeslib.as_array(ptr, shape=(n,))
    except Exception:
        flat = np.ctypeslib.as_array(cast(c_void_p(ptr), POINTER(_cd)),
                                     shape=(n,))
    return np.ascontiguousarray(flat.reshape(shape), dtype=dtype).copy()


# =============================================================================
# Accessors — read Fortran arrays through ctypes  (unchanged)
# =============================================================================
def get_rs_hamiltonian(binding, n_spin: int, n_ham_size: int) -> np.ndarray:
    from ._exceptions import AimspyBindingError
    ptr = binding.c_rs_hamiltonian()
    if not ptr:
        raise AimspyBindingError("c_rs_hamiltonian() returned NULL")
    return _ptr_to_ndarray(ptr, (n_spin, n_ham_size))


def get_rs_overlap(binding, n_ham_size: int) -> np.ndarray:
    from ._exceptions import AimspyBindingError
    ptr = binding.c_rs_overlap()
    if not ptr:
        raise AimspyBindingError("c_rs_overlap() returned NULL")
    return _ptr_to_ndarray(ptr, (n_ham_size,))


# =============================================================================
# AimspyMatrix — canonical block-sparse matrix in aimspy standard format
# =============================================================================
@dataclass
class AimspyMatrix:
    """Block-sparse real-space matrix in aimspy standard format.

    Key = ``(R1, R2, R3, i_atom, j_atom)`` with all ints:
        - R follows ``R_aimspy = -R_aims`` (same sign as DeepH).
        - i_atom / j_atom in aims native order.
        - Orbital order within each atom is aims native.
        - Parity = wiki/DeepH (phase already applied).
        - Units = Hartree.
    """

    blocks: Dict[Tuple[int, ...], np.ndarray]  # key -> (n_orb_i, n_orb_j)
    n_spin: int = 1

    # ----------------------------------------------------------------
    # aims CSR ↔ aimspy
    # ----------------------------------------------------------------
    @classmethod
    def from_aims_csr(
        cls,
        h0: np.ndarray,               # (n_spin, n_ham_size), C-contiguous
        csr_descr: CsrMatrixDescriptor,
        structure: AimspyStructure,
    ) -> "AimspyMatrix":
        """Convert aims CSR flat array to aimspy block dict.

        Steps:
        1. Walk CSR triplanes (cell, basis‑row, k‑index).
        2. R_aimspy = -R_aims (sign flip) → lookup key matches DeepH.
        3. Apply wiki parity: ``v *= phase_i * phase_j``.
        4. Store block[orb_i, orb_j] and its Hermitian partner.
        """
        phase = structure.phase_factor
        subidx = structure.basis_subidx
        opa = structure.orbit_per_atom
        blocks: dict = {}

        n_cells_loop = csr_descr.n_cells - 1          # skip sentinel
        n_ham = csr_descr.n_ham_size

        for ic in range(n_cells_loop):
            R0 = -int(csr_descr.cell_idx[0, ic])       # R_aimspy = -R_aims
            R1 = -int(csr_descr.cell_idx[1, ic])
            R2 = -int(csr_descr.cell_idx[2, ic])

            for ib_row in range(csr_descr.n_basis):
                start = int(csr_descr.row_mx_idx[ib_row, ic, 0])
                end   = int(csr_descr.row_mx_idx[ib_row, ic, 1])
                if start < 1 or end < start:
                    continue

                atom_i = int(structure.basis_atom[ib_row])
                orb_i  = int(subidx[ib_row])
                pi     = int(phase[ib_row])

                for k in range(start - 1, end):
                    if k >= n_ham:
                        continue                                 # skip trash
                    ib_col = int(csr_descr.col_mx_idx[k]) - 1
                    atom_j = int(structure.basis_atom[ib_col])
                    orb_j  = int(subidx[ib_col])
                    pj     = int(phase[ib_col])

                    key     = (R0, R1, R2, atom_i, atom_j)
                    rev_key = (-R0, -R1, -R2, atom_j, atom_i)

                    if key not in blocks:
                        blocks[key] = np.zeros(
                            (opa[atom_i], opa[atom_j]), dtype=np.float64)
                    if rev_key not in blocks:
                        blocks[rev_key] = np.zeros(
                            (opa[atom_j], opa[atom_i]), dtype=np.float64)

                    v = h0[0, k] * pi * pj                     # apply parity
                    blocks[key][orb_i, orb_j] = v
                    blocks[rev_key][orb_j, orb_i] = v           # Hermitian

        return cls(blocks=blocks, n_spin=int(h0.shape[0]))

    def to_aims_csr(
        self,
        csr_descr: CsrMatrixDescriptor,
        structure: AimspyStructure,
    ) -> np.ndarray:
        """Convert aimspy block dict back to aims CSR flat array.

        Steps:
        1. Walk CSR triplanes (same order as ``from_aims_csr``).
        2. Look up block in ``self.blocks`` (dict, O(1)).
        3. Hermitian fallback: if forward key missing, try ``(-R, j, i)``.
        4. Undo parity: ``v *= phase_i * phase_j`` (self‑inverse).
        5. Return ``(n_spin, n_ham_size)`` C‑contiguous, ready to memmove.
        """
        phase  = structure.phase_factor
        subidx = structure.basis_subidx

        n_ham  = csr_descr.n_ham_size
        n_spin = csr_descr.n_spin
        out    = np.zeros((n_spin, n_ham), dtype=np.float64)
        n_cells_loop = csr_descr.n_cells - 1

        for ic in range(n_cells_loop):
            R0 = -int(csr_descr.cell_idx[0, ic])
            R1 = -int(csr_descr.cell_idx[1, ic])
            R2 = -int(csr_descr.cell_idx[2, ic])

            for ib_row in range(csr_descr.n_basis):
                start = int(csr_descr.row_mx_idx[ib_row, ic, 0])
                end   = int(csr_descr.row_mx_idx[ib_row, ic, 1])
                if start < 1 or end < start:
                    continue

                atom_i = int(structure.basis_atom[ib_row])
                orb_i  = int(subidx[ib_row])
                pi     = int(phase[ib_row])

                for k in range(start - 1, end):
                    if k >= n_ham:
                        continue
                    ib_col = int(csr_descr.col_mx_idx[k]) - 1
                    atom_j = int(structure.basis_atom[ib_col])
                    orb_j  = int(subidx[ib_col])
                    pj     = int(phase[ib_col])

                    key = (R0, R1, R2, atom_i, atom_j)
                    blk = self.blocks.get(key)
                    if blk is not None:
                        if orb_i < blk.shape[0] and orb_j < blk.shape[1]:
                            val = blk[orb_i, orb_j]
                        else:
                            val = 0.0
                    else:
                        rev_key = (-R0, -R1, -R2, atom_j, atom_i)
                        blk = self.blocks.get(rev_key)
                        if blk is not None \
                           and orb_j < blk.shape[0] and orb_i < blk.shape[1]:
                            val = blk[orb_j, orb_i]   # Hermitian fallback
                        else:
                            val = 0.0

                    val *= pi * pj                         # undo parity
                    out[0, k] = val

        return out

    # ----------------------------------------------------------------
    # Convenience constructors
    # ----------------------------------------------------------------
    @classmethod
    def from_pair_blocks(
        cls,
        pair_blocks: Dict[Tuple[int, ...], np.ndarray],
        n_spin: int = 1,
    ) -> "AimspyMatrix":
        """Construct from an in-memory ``{(R1,R2,R3,i,j): block}`` dict.

        The dict is NOT copied — the caller must not mutate it afterwards.
        """
        return cls(blocks=pair_blocks, n_spin=n_spin)

    def to_pair_blocks(self) -> Dict[Tuple[int, ...], np.ndarray]:
        """Export as ``{(R1,R2,R3,i,j): block}`` dict (reference, no copy)."""
        return self.blocks

    @property
    def n_pairs(self) -> int:
        return len(self.blocks)

    def __repr__(self) -> str:
        return f"AimspyMatrix(n_pairs={self.n_pairs}, n_spin={self.n_spin})"


# =============================================================================
# Pair-lookup helper (used by both aimspy and DeepH conversions)
# =============================================================================
class _PairLookup:
    """O(log N) ``searchsorted``-based lookup for (R1,R2,R3,i,j) int keys."""

    _DT = np.dtype([
        ('r1', 'i8'), ('r2', 'i8'), ('r3', 'i8'),
        ('i',  'i8'), ('j',  'i8'),
    ])

    def __init__(self, atom_pairs: np.ndarray):                     # (N, 5)
        ap = np.ascontiguousarray(atom_pairs.astype(np.int64))
        keys = ap.view(self._DT).ravel()
        self._sort_order = np.argsort(keys, kind='stable')
        self._sorted_keys = keys[self._sort_order]
        self._key_buf = np.zeros(1, dtype=self._DT)

    def lookup(self, r1, r2, r3, i, j):
        self._key_buf[0] = (int(r1), int(r2), int(r3), int(i), int(j))
        pos = np.searchsorted(self._sorted_keys, self._key_buf[0])
        if pos < len(self._sorted_keys) \
           and self._sorted_keys[pos] == self._key_buf[0]:
            return int(self._sort_order[pos])
        return -1
