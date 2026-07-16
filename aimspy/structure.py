"""Public — AimspyStructure: shared structure+orbital descriptor for aimspy.

This descriptor is independent of any matrix data and can be shared
across multiple ``AimspyMatrix`` instances.

Constructed from a runtime ``AimspyInfo`` snapshot via
:meth:`from_info`. For offline use, construct directly with the
dataclass constructor.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import List

import numpy as np

from .data import AimspyInfo


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

    lattice: np.ndarray = None  # (n_periodic, 3) or (1,3)
    atom_symbols: List[str] = None  # per-atom symbol, aims order
    atom_coords: np.ndarray = None  # (n_atoms, 3) in Angstrom
    basis_atom: np.ndarray = None  # (n_basis,) int32, 0-based
    basis_l: np.ndarray = None  # (n_basis,) int32
    basis_m: np.ndarray = None  # (n_basis,) int32

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
            atom_coords=(
                info.coords.copy() if info.coords is not None else np.empty((0, 3))
            ),
            basis_atom=info.basis_atom.copy().astype(np.int32),
            basis_l=info.basis_l.copy().astype(np.int32),
            basis_m=info.basis_m.copy().astype(np.int32),
        )

    # ----------------------------------------------------------------
    # Derived properties (computed once, cached; structure is immutable)
    # ----------------------------------------------------------------
    @functools.cached_property
    def phase_factor(self) -> np.ndarray:
        """Wiki/DeepH parity: -1 if m>0 and m odd, else +1.

        This is the real-spherical-harmonics phase convention used by
        both DeepH and the aimspy standard format.  It is **not** the
        aims native convention — applying it converts aims→aimspy (and
        reapplying it converts aimspy→aims, since ``phase² = 1``).
        """
        return np.where((self.basis_m > 0) & (self.basis_m % 2 == 1), -1, 1).astype(
            np.int32
        )

    @functools.cached_property
    def basis_subidx(self) -> np.ndarray:
        """Per-atom orbital sub-index in aims basis order.

        ``basis_subidx[i]`` = the 0‑based position of basis function *i*
        among its atom's basis functions, in aims traversal order.
        """
        order = np.argsort(self.basis_atom, kind="stable")
        sorted_atoms = self.basis_atom[order]
        counts = np.bincount(sorted_atoms, minlength=self.n_atoms)
        starts = np.zeros(self.n_atoms, dtype=np.int32)
        if self.n_atoms > 1:
            np.cumsum(counts[:-1], out=starts[1:])
        subidx_sorted = np.arange(self.n_basis, dtype=np.int32) - starts[sorted_atoms]
        subidx = np.empty(self.n_basis, dtype=np.int32)
        subidx[order] = subidx_sorted
        return subidx

    @functools.cached_property
    def orbit_per_atom(self) -> np.ndarray:
        """Number of basis functions per atom."""
        return np.bincount(self.basis_atom, minlength=self.n_atoms).astype(np.int32)

    @functools.cached_property
    def atoms_species_sorted(self) -> List[str]:
        """Per-atom species in POSCAR/DeepH element-grouped order."""
        sort_idxs = np.argsort(self.atom_symbols, kind="stable")
        return [self.atom_symbols[i] for i in sort_idxs]

    @functools.cached_property
    def atom_permutation(self) -> tuple[np.ndarray, np.ndarray]:
        """``(old2new, new2old)`` mapping aims→POSCAR and back.

        ``old2new[aims_atom] == POSCAR_atom``
        ``new2old[POSCAR_atom] == aims_atom``

        Computed once and cached; safe because the structure is expected
        to be immutable after construction.
        """
        sort_idxs = np.argsort(self.atom_symbols, kind="stable")
        old2new = np.zeros(self.n_atoms, dtype=np.int32)
        old2new[sort_idxs] = np.arange(self.n_atoms, dtype=np.int32)
        return old2new, sort_idxs.astype(np.int32)

    def build_atom_permutation(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(old2new, new2old)`` mapping aims->POSCAR and back.

        Convenience wrapper around :attr:`atom_permutation` for backward
        compatibility.
        """
        return self.atom_permutation

    def __repr__(self) -> str:
        return (
            f"AimspyStructure(n_atoms={self.n_atoms}, "
            f"n_basis={self.n_basis}, n_spin={self.n_spin}, "
            f"species={list(self.atom_symbols)})"
        )
