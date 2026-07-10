"""Public — AimspyStructure: shared structure+orbital descriptor for aimspy.

This descriptor is independent of any matrix data and can be shared
across multiple ``AimspyMatrix`` instances.

Constructed either from a runtime ``AimspyInfo`` snapshot or from
user-supplied data (for offline use).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .data import AimspyInfo, BOHR_TO_ANG


@dataclass
class AimspyStructure:
    """Structure + orbital info, reusable across multiple matrices.

    Contains everything needed for format conversions **except** the
    CSR sparse-storage layout (``CsrMatrixDescriptor``), which is
    aims‑specific and captured separately via the ``get_descr`` callback
    at runtime.

    Atom and orbital ordering follows the **aims native order** — no
    reordering is applied.
    """

    n_atoms: int
    n_basis: int
    n_spin: int
    n_periodic: int = 0

    lattice: np.ndarray = None            # (n_periodic, 3) or (1,3)
    atom_symbols: List[str] = None         # per-atom symbol, aims order
    atom_coords: np.ndarray = None         # (n_atoms, 3) in Angstrom
    basis_atom: np.ndarray = None          # (n_basis,) int32, 0-based
    basis_l: np.ndarray = None             # (n_basis,) int32
    basis_m: np.ndarray = None             # (n_basis,) int32

    def __post_init__(self):
        if self.lattice is None:
            self.lattice = np.empty((0, 3))
        if self.atom_symbols is None:
            self.atom_symbols = []
        if self.atom_coords is None:
            self.atom_coords = np.empty((0, 3))
        if self.basis_atom is None:
            self.basis_atom = np.array([], dtype=np.int32)
        if self.basis_l is None:
            self.basis_l = np.array([], dtype=np.int32)
        if self.basis_m is None:
            self.basis_m = np.array([], dtype=np.int32)

    # ----------------------------------------------------------------
    # Constructors
    # ----------------------------------------------------------------
    @classmethod
    def from_info(cls, info: AimspyInfo) -> "AimspyStructure":
        """Build from a runtime ``AimspyInfo`` snapshot (available after
        ``aimspy_init``).

        All arrays are independent copies — safe to hold after
        ``aimspy_finalize``.
        """
        if info.n_periodic > 0 and info.lattice is not None:
            lattice = info.lattice.copy()
        else:
            lattice = np.empty((0, 3))

        return cls(
            n_atoms=info.n_atoms,
            n_basis=info.n_basis,
            n_spin=info.n_spin,
            n_periodic=info.n_periodic,
            lattice=lattice,
            atom_symbols=list(info.atoms_species),
            atom_coords=info.coords.copy() if info.coords is not None
                         else np.empty((0, 3)),
            basis_atom=info.basis_atom.copy().astype(np.int32),
            basis_l=info.basis_l.copy().astype(np.int32),
            basis_m=info.basis_m.copy().astype(np.int32),
        )

    @classmethod
    def from_raw(
        cls,
        n_atoms: int,
        atom_symbols: List[str],
        atom_coords: np.ndarray,
        basis_l: np.ndarray,
        basis_m: np.ndarray,
        lattice: Optional[np.ndarray] = None,
        n_spin: int = 1,
    ) -> "AimspyStructure":
        """Build from user-supplied raw data (offline / testing path).

        ``basis_atom`` is inferred from the length of per-atom basis
        lists implied by ``basis_l`` and ``basis_m``.
        """
        n_basis = len(basis_l)
        # Infer basis_atom: the caller must provide one-per-atom basis info
        # via basis_l/basis_m in aims atom order.  We assign atom index by
        # counting the number of basis functions per atom given by the
        # structure info.
        basis_atom = np.zeros(n_basis, dtype=np.int32)
        atom_coords = np.asarray(atom_coords, dtype=np.float64)
        lattice = np.asarray(lattice, dtype=np.float64) if lattice is not None \
                  else np.empty((0, 3))
        return cls(
            n_atoms=n_atoms,
            n_basis=n_basis,
            n_spin=n_spin,
            n_periodic=1 if lattice.size >= 3 else 0,
            lattice=lattice,
            atom_symbols=list(atom_symbols),
            atom_coords=atom_coords,
            basis_atom=basis_atom,
            basis_l=np.asarray(basis_l, dtype=np.int32),
            basis_m=np.asarray(basis_m, dtype=np.int32),
        )

    # ----------------------------------------------------------------
    # Derived properties (computed on demand, not stored)
    # ----------------------------------------------------------------
    @property
    def phase_factor(self) -> np.ndarray:
        """Wiki/DeepH parity: -1 if m>0 and m odd, else +1.

        This is the real-spherical-harmonics phase convention used by
        both DeepH and the aimspy standard format.  It is **not** the
        aims native convention — applying it converts aims→aimspy (and
        reapplying it converts aimspy→aims, since ``phase² = 1``).
        """
        return np.where(
            (self.basis_m > 0) & (self.basis_m % 2 == 1), -1, 1
        ).astype(np.int32)

    @property
    def basis_subidx(self) -> np.ndarray:
        """Per-atom orbital sub-index in aims basis order.

        ``basis_subidx[i]`` = the 0‑based position of basis function *i*
        among its atom's basis functions, in aims traversal order.
        """
        subidx = np.zeros(self.n_basis, dtype=np.int32)
        counter = np.zeros(self.n_atoms, dtype=np.int32)
        for i in range(self.n_basis):
            a = int(self.basis_atom[i])
            subidx[i] = counter[a]
            counter[a] += 1
        return subidx

    @property
    def orbit_per_atom(self) -> np.ndarray:
        """Number of basis functions per atom."""
        counts = np.zeros(self.n_atoms, dtype=np.int32)
        for a in self.basis_atom:
            counts[int(a)] += 1
        return counts

    @property
    def atoms_species_sorted(self) -> List[str]:
        """Per-atom species in POSCAR/DeepH element-grouped order."""
        sort_idxs = np.argsort(self.atom_symbols, kind='stable')
        return [self.atom_symbols[i] for i in sort_idxs]

    def build_atom_permutation(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(old2new, new2old)`` mapping aims->POSCAR and back.

        ``old2new[aims_atom] == POSCAR_atom``
        ``new2old[POSCAR_atom] == aims_atom``
        """
        sort_idxs = np.argsort(self.atom_symbols, kind='stable')
        old2new = np.zeros(self.n_atoms, dtype=np.int32)
        old2new[sort_idxs] = np.arange(self.n_atoms, dtype=np.int32)
        return old2new, sort_idxs.astype(np.int32)

    def __repr__(self) -> str:
        return (
            f"AimspyStructure(n_atoms={self.n_atoms}, "
            f"n_basis={self.n_basis}, n_spin={self.n_spin}, "
            f"species={list(self.atom_symbols)})"
        )
