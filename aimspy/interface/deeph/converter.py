"""DeepH ↔ aimspy format converter.

The DeepH and aimspy formats share the same R-sign and parity
conventions, so the conversion involves only:
  - atom reordering (POSCAR ↔ aims)
  - unit conversion (eV ↔ Hartree)
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

from ...structure import AimspyStructure
from ...matrix import AimspyMatrix
from ...data import EV_TO_HARTREE, HARTREE_TO_EV
from .. import ExternalMatrixSource
from .data import DeepHData


class DeepHSource(ExternalMatrixSource):
    """Converts DeepH-format data to aimspy standard format.

    Wraps a ``DeepHData`` instance and provides ``to_aimspy``, which
    is called from the Calculator's ``python_func`` callback at runtime.
    """

    def __init__(self, data: DeepHData) -> None:
        self._data = data

    @property
    def data(self) -> DeepHData:
        return self._data

    def to_aimspy(self, structure: AimspyStructure) -> AimspyMatrix:
        """DeepH → aimspy standard format.

        - Atom reordering: POSCAR → aims (via stable-sort un‑permutation).
        - R: no flip (same convention: ``R_aimspy = R_deeph = -R_aims``).
        - Parity: no change (same wiki convention).
        - Units: eV → Hartree.
        """
        return deeph_to_aimspy(self._data, structure)

    def __repr__(self) -> str:
        return f"DeepHSource({self._data!r})"


# =============================================================================
# Conversion functions
# =============================================================================
def deeph_to_aimspy(
    deeph_data: DeepHData,
    structure: AimspyStructure,
) -> AimspyMatrix:
    """Convert DeepH-format data to aimspy block dict.

    Atom indices in *deeph_data* are POSCAR order (element‑grouped);
    the output blocks use aims atom order.
    """
    _, new2old = structure.build_atom_permutation()
    # new2old[POSCAR_atom] = aims_atom

    entries  = deeph_data.entries
    cb       = deeph_data.chunk_boundaries
    cs       = deeph_data.chunk_shapes
    ap       = deeph_data.atom_pairs

    blocks: dict = {}
    for ip in range(deeph_data.n_pairs):
        R1 = int(ap[ip, 0])
        R2 = int(ap[ip, 1])
        R3 = int(ap[ip, 2])
        i_deeph = int(ap[ip, 3])
        j_deeph = int(ap[ip, 4])

        i_aims = int(new2old[i_deeph])
        j_aims = int(new2old[j_deeph])

        bnd = int(cb[ip])
        nr  = int(cs[ip, 0])
        nc  = int(cs[ip, 1])

        block = entries[bnd:bnd + nr * nc].reshape(nr, nc).copy()
        block *= EV_TO_HARTREE   # eV -> Hartree

        key = (R1, R2, R3, i_aims, j_aims)
        blocks[key] = block

    return AimspyMatrix(blocks=blocks, n_spin=1)


def aimspy_to_deeph(
    matrix: AimspyMatrix,
    structure: AimspyStructure,
) -> DeepHData:
    """Convert aimspy block dict to DeepH format.

    Atom indices in the aimspy blocks are aims order; the output uses
    POSCAR order.
    """
    old2new, _ = structure.build_atom_permutation()
    # old2new[aims_atom] = POSCAR_atom

    # Collect blocks re-mapped to POSCAR order
    pair_blocks: dict[tuple, np.ndarray] = {}
    for (R1, R2, R3, i_aims, j_aims), block in matrix.blocks.items():
        i_deeph = int(old2new[i_aims])
        j_deeph = int(old2new[j_aims])
        key = (R1, R2, R3, i_deeph, j_deeph)
        if key not in pair_blocks:
            pair_blocks[key] = block.copy()
        else:
            # Should not happen for a clean dict, but be safe
            pair_blocks[key] = np.maximum(pair_blocks[key], block)

    return DeepHData.from_memory(
        lattice=structure.lattice.copy(),
        atom_symbols=list(structure.atoms_species_sorted),
        atom_coords=_reorder_coords(structure),
        elements_orbital_map=_build_elements_orbital_map(structure),
        pair_blocks=pair_blocks,
    )


# -------------------------------------------------------------------
# Internal helpers
# -------------------------------------------------------------------
def _reorder_coords(structure: AimspyStructure) -> np.ndarray:
    """Return atom coords in POSCAR (element‑grouped) order."""
    _, new2old = structure.build_atom_permutation()
    n = structure.n_atoms
    coords = np.zeros((n, 3), dtype=np.float64)
    for i_deeph in range(n):
        i_aims = int(new2old[i_deeph])
        coords[i_deeph] = structure.atom_coords[i_aims]
    return coords


def _build_elements_orbital_map(structure: AimspyStructure) -> dict:
    """Build ``{element: [l values]}`` from structure info."""
    result: dict[str, list[int]] = {}
    for idx in range(structure.n_atoms):
        elem = structure.atom_symbols[idx]
        mask = structure.basis_atom == idx
        ls = sorted(set(int(l) for l in structure.basis_l[mask]))
        result.setdefault(elem, []).extend(ls)
    return {k: sorted(set(v)) for k, v in result.items()}
