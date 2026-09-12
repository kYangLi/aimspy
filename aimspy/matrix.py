"""Public — AimspyMatrix + aims↔aimspy format conversions.

The aimspy standard matrix format is a block-sparse real-space
representation:

    blocks: dict[tuple[int, int, int, int, int], np.ndarray]
           key = (R1, R2, R3, i_atom, j_atom)

Conventions
-----------
- *R*: ``R_aimspy = -R_aims`` (same sign as DeepH).
- *Atoms*: aims native order (no reordering).
- *Orbitals*: aims native basis order (no reordering).
- *Parity*: wiki/DeepH convention (``phase_i * phase_j`` already applied).
- *Units*: Hartree.
- *Hermitian partners*: both ``(R,i,j)`` and ``(-R,j,i)`` stored.
- *Spin*: ``n_spin=1`` blocks are ``(n_orb_i, n_orb_j)``.  ``n_spin=2``
  (collinear) blocks are stacked ``(2*n_orb_i, n_orb_j)``: the alpha
  (spin-up) channel occupies rows ``[0, n_orb_i)`` and the beta
  (spin-down) channel rows ``[n_orb_i, 2*n_orb_i)``.  Spin-independent
  matrices (the overlap) always use ``n_spin=1`` blocks.
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
        flat = np.ctypeslib.as_array(cast(c_void_p(ptr), POINTER(_cd)), shape=(n,))
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


def get_forces(binding, n_atoms: int) -> Optional[np.ndarray]:
    """Read total_forces (3, n_atoms) Fortran array → (n_atoms, 3) eV/Å.

    Fortran stores total_forces in Hartree/Bohr; we convert to eV/Å
    (the same convention FHI-aims uses for printed forces in aims.out).

    Returns None if use_forces=False (Fortran returns c_null_ptr when
    `compute_forces .true.` was not set in control.in).
    """
    from .data import HARTREE_TO_EV, BOHR_TO_ANG

    ptr = binding.aimspy_forces()
    if not ptr:
        return None  # use_forces=False — forces not computed
    # Fortran (3, n_atoms) column-major → (n_atoms, 3)
    raw = _ptr_to_ndarray(ptr, (n_atoms, 3))
    # Hartree/Bohr → eV/Å
    return raw * (HARTREE_TO_EV / BOHR_TO_ANG)


def get_stress(binding) -> Optional[np.ndarray]:
    """Read the final analytical stress tensor in eV/Å³.

    FHI-aims stores the tensor as a Fortran ``(3, 3)`` array in
    Hartree/Bohr³.  The sign convention is preserved exactly as reported by
    FHI-aims.  Returns ``None`` when analytical stress was not computed.
    """
    from .data import HARTREE_TO_EV, BOHR_TO_ANG

    ptr = binding.aimspy_stress()
    if not ptr:
        return None
    flat = _ptr_to_ndarray(ptr, (9,))
    raw = flat.reshape((3, 3), order="F")
    return np.ascontiguousarray(raw * (HARTREE_TO_EV / BOHR_TO_ANG**3))


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

    Block shape:
        - ``n_spin == 1``: ``(n_orb_i, n_orb_j)``.
        - ``n_spin == 2`` (collinear): ``(2*n_orb_i, n_orb_j)`` with the
          alpha (spin-up) channel in rows ``[0, n_orb_i)`` and the beta
          (spin-down) channel in rows ``[n_orb_i, 2*n_orb_i)``.
    """

    blocks: Dict[Tuple[int, ...], np.ndarray]  # key -> (n_spin*n_orb_i, n_orb_j)
    n_spin: int = 1

    # ----------------------------------------------------------------
    # aims CSR ↔ aimspy
    # ----------------------------------------------------------------
    @classmethod
    def from_aims_csr(
        cls,
        h0: np.ndarray,  # (n_spin, n_ham_size), C-contiguous
        csr_descr: CsrMatrixDescriptor,
        structure: AimspyStructure,
    ) -> "AimspyMatrix":
        """Convert aims CSR flat array to aimspy block dict.

        Steps:
        1. Walk CSR triplanes (cell, basis‑row, k‑index).
        2. R_aimspy = -R_aims (sign flip) → lookup key matches DeepH.
        3. Apply wiki parity: ``v *= phase_i * phase_j``.
        4. Store block[orb_i, orb_j] and its Hermitian partner.

        The number of spin channels is inferred from ``h0.shape[0]`` (the
        data), not from ``csr_descr.n_spin`` (the system): the spin-independent
        overlap arrives as ``(1, n_ham_size)`` even for spin-polarized
        systems and yields standard ``n_spin=1`` blocks, while the
        Hamiltonian arrives as ``(2, n_ham_size)`` and yields stacked
        ``(2*n_orb_i, n_orb_j)`` blocks (alpha rows first, beta rows second).

        Raises
        ------
        AimspyError
            If ``h0.shape[0]`` is neither 1 nor 2, or if 2-channel data
            is combined with a descriptor whose ``n_spin != 2``.
        """
        n_spin = int(h0.shape[0])
        if n_spin not in (1, 2):
            from ._exceptions import AimspyError

            raise AimspyError(
                f"from_aims_csr: unsupported n_spin={n_spin}; expected 1 or 2"
            )
        if n_spin == 2 and csr_descr.n_spin != 2:
            from ._exceptions import AimspyError

            raise AimspyError(
                f"from_aims_csr: 2-channel data (h0.shape[0]=2) but "
                f"csr_descr.n_spin={csr_descr.n_spin}"
            )
        phase = structure.phase_factor
        subidx = structure.basis_subidx
        opa = [int(v) for v in structure.orbit_per_atom]
        blocks: dict = {}

        n_cells_loop = csr_descr.n_cells - 1  # skip sentinel
        n_ham = csr_descr.n_ham_size

        for ic in range(n_cells_loop):
            R0 = -int(csr_descr.cell_idx[0, ic])  # R_aimspy = -R_aims
            R1 = -int(csr_descr.cell_idx[1, ic])
            R2 = -int(csr_descr.cell_idx[2, ic])

            for ib_row in range(csr_descr.n_basis):
                start = int(csr_descr.row_mx_idx[ib_row, ic, 0])
                end = int(csr_descr.row_mx_idx[ib_row, ic, 1])
                if start < 1 or end < start:
                    continue

                atom_i = int(structure.basis_atom[ib_row])
                orb_i = int(subidx[ib_row])
                pi = int(phase[ib_row])
                oi = opa[atom_i]

                for k in range(start - 1, end):
                    if k >= n_ham:
                        continue  # skip trash
                    ib_col = int(csr_descr.col_mx_idx[k]) - 1
                    atom_j = int(structure.basis_atom[ib_col])
                    orb_j = int(subidx[ib_col])
                    pj = int(phase[ib_col])
                    oj = opa[atom_j]

                    key = (R0, R1, R2, atom_i, atom_j)
                    rev_key = (-R0, -R1, -R2, atom_j, atom_i)

                    if key not in blocks:
                        blocks[key] = np.zeros((n_spin * oi, oj), dtype=np.float64)
                    if rev_key not in blocks:
                        blocks[rev_key] = np.zeros((n_spin * oj, oi), dtype=np.float64)

                    # Hermitian partner: write if unset, else verify consistency.
                    # CSR stores upper-triangle only, so the reverse entry
                    # (j,i) at -R should already equal (i,j) at R. If it was
                    # previously written (abs > 1e-12), check agreement within
                    # 1e-11 — well above double round-off (~1e-13 for |v|~1e3).
                    if n_spin == 1:
                        v = h0[0, k] * pi * pj  # apply parity
                        blocks[key][orb_i, orb_j] = v

                        existing_rev = blocks[rev_key][orb_j, orb_i]
                        if abs(existing_rev) <= 1e-12:
                            blocks[rev_key][orb_j, orb_i] = v
                        elif abs(existing_rev - v) > 1e-11:
                            from ._exceptions import AimspyError

                            raise AimspyError(
                                f"Hermitian check failed at R=({R0},{R1},{R2}), "
                                f"atom=({atom_i},{atom_j}), orb=({orb_i},{orb_j}): "
                                f"existing={existing_rev:.6e}, new={v:.6e}"
                            )
                    else:
                        # n_spin == 2 — stacked alpha/beta channels.  Both
                        # channels are always written together, so probing the
                        # beta row is sufficient to detect a prior write.
                        # Known weakening (same "unwritten" heuristic as the
                        # spinless path): beta-channel values that are exactly
                        # zero skip the alpha-channel consistency check; the
                        # last write still wins, so blocks stay Hermitian.
                        v_up = h0[0, k] * pi * pj
                        v_dn = h0[1, k] * pi * pj
                        blocks[key][orb_i, orb_j] = v_up
                        blocks[key][orb_i + oi, orb_j] = v_dn

                        existing_rev = blocks[rev_key][orb_j + oj, orb_i]
                        if abs(existing_rev) <= 1e-12:
                            blocks[rev_key][orb_j, orb_i] = v_up
                            blocks[rev_key][orb_j + oj, orb_i] = v_dn
                        elif abs(existing_rev - v_dn) > 1e-11:
                            from ._exceptions import AimspyError

                            raise AimspyError(
                                f"Hermitian check failed (spin channel 1) at "
                                f"R=({R0},{R1},{R2}), "
                                f"atom=({atom_i},{atom_j}), orb=({orb_i},{orb_j}): "
                                f"existing={existing_rev:.6e}, new={v_dn:.6e}"
                            )

        return cls(blocks=blocks, n_spin=n_spin)

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

        For stacked (``n_spin == 2``) blocks, the alpha row
        ``[orb_i, orb_j]`` and the beta row ``[orb_i + n_orb_i, orb_j]``
        are unpacked into ``out[0, k]`` and ``out[1, k]``.

        Note the asymmetry with :meth:`from_aims_csr`: *reading* may be
        spin-independent (``(1, n_ham_size)`` overlap data combined with
        an ``n_spin == 2`` descriptor is accepted and yields standard
        blocks), but *writing* requires an exact ``n_spin`` match — a
        non-stacked matrix written through a spinful descriptor would
        produce a partially-zero Hamiltonian (not a valid non-polarized
        guess, which would require both channels equal), so it is
        rejected.

        Raises
        ------
        AimspyError
            If the matrix ``n_spin`` does not match
            ``csr_descr.n_spin``.
        """
        n_spin = csr_descr.n_spin
        if self.n_spin != n_spin:
            from ._exceptions import AimspyError

            raise AimspyError(
                f"to_aims_csr: matrix n_spin={self.n_spin} does not match "
                f"descriptor n_spin={n_spin}"
            )
        stacked = self.n_spin == 2
        phase = structure.phase_factor
        subidx = structure.basis_subidx
        opa = [int(v) for v in structure.orbit_per_atom]

        n_ham = csr_descr.n_ham_size
        out = np.zeros((n_spin, n_ham), dtype=np.float64)
        n_cells_loop = csr_descr.n_cells - 1

        for ic in range(n_cells_loop):
            R0 = -int(csr_descr.cell_idx[0, ic])
            R1 = -int(csr_descr.cell_idx[1, ic])
            R2 = -int(csr_descr.cell_idx[2, ic])

            for ib_row in range(csr_descr.n_basis):
                start = int(csr_descr.row_mx_idx[ib_row, ic, 0])
                end = int(csr_descr.row_mx_idx[ib_row, ic, 1])
                if start < 1 or end < start:
                    continue

                atom_i = int(structure.basis_atom[ib_row])
                orb_i = int(subidx[ib_row])
                pi = int(phase[ib_row])
                oi = opa[atom_i]

                for k in range(start - 1, end):
                    if k >= n_ham:
                        continue
                    ib_col = int(csr_descr.col_mx_idx[k]) - 1
                    atom_j = int(structure.basis_atom[ib_col])
                    orb_j = int(subidx[ib_col])
                    pj = int(phase[ib_col])
                    oj = opa[atom_j]

                    key = (R0, R1, R2, atom_i, atom_j)
                    blk = self.blocks.get(key)
                    if blk is not None:
                        if stacked:
                            if orb_i + oi < blk.shape[0] and orb_j < blk.shape[1]:
                                val_up = blk[orb_i, orb_j]
                                val_dn = blk[orb_i + oi, orb_j]
                            else:
                                val_up = val_dn = 0.0
                        elif orb_i < blk.shape[0] and orb_j < blk.shape[1]:
                            val = blk[orb_i, orb_j]
                        else:
                            val = 0.0
                    else:
                        rev_key = (-R0, -R1, -R2, atom_j, atom_i)
                        blk = self.blocks.get(rev_key)
                        if stacked:
                            if (
                                blk is not None
                                and orb_j + oj < blk.shape[0]
                                and orb_i < blk.shape[1]
                            ):
                                val_up = blk[orb_j, orb_i]  # Hermitian fallback
                                val_dn = blk[orb_j + oj, orb_i]
                            else:
                                val_up = val_dn = 0.0
                        elif (
                            blk is not None
                            and orb_j < blk.shape[0]
                            and orb_i < blk.shape[1]
                        ):
                            val = blk[orb_j, orb_i]  # Hermitian fallback
                        else:
                            val = 0.0

                    pp = pi * pj  # undo parity (self-inverse)
                    if stacked:
                        out[0, k] = val_up * pp
                        out[1, k] = val_dn * pp
                    else:
                        out[0, k] = val * pp

        return out

    @property
    def n_pairs(self) -> int:
        return len(self.blocks)

    def __repr__(self) -> str:
        return f"AimspyMatrix(n_pairs={self.n_pairs}, n_spin={self.n_spin})"
