"""DeepH-format data: reader/writer for POSCAR + info.json + hamiltonian.h5.

Provides ``DeepHData``, a complete in-memory representation of DeepH
data including structure info (from POSCAR / info.json) and matrix
data (from hamiltonian.h5 / overlap.h5 / hamiltonian_init.h5).
Supports file I/O, in-memory construction, conversion from
aimspy standard format via ``from_aimspy``, and conversion to
aimspy standard format via ``to_aimspy``.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import h5py
import numpy as np

from ...data import EV_TO_HARTREE, HARTREE_TO_EV
from ..._exceptions import AimspyConfigError

if TYPE_CHECKING:
    from ...matrix import AimspyMatrix
    from ...structure import AimspyStructure

# -------------------------------------------------------------------
# Atomic-number lookup (for occupation computation in info.json)
# -------------------------------------------------------------------
_ATOMIC_NUMBERS: dict[str, int] = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
    "Am": 95,
    "Cm": 96,
    "Bk": 97,
    "Cf": 98,
    "Es": 99,
    "Fm": 100,
    "Md": 101,
    "No": 102,
    "Lr": 103,
    "Rf": 104,
    "Db": 105,
    "Sg": 106,
    "Bh": 107,
    "Hs": 108,
    "Mt": 109,
    "Ds": 110,
    "Rg": 111,
    "Cn": 112,
    "Nh": 113,
    "Fl": 114,
    "Mc": 115,
    "Lv": 116,
    "Ts": 117,
    "Og": 118,
}


# -------------------------------------------------------------------
# Conversion helpers
# -------------------------------------------------------------------
def _aimspy_blocks_to_poscar(matrix, structure) -> dict[tuple, np.ndarray]:
    """Reorder aimspy blocks (aims atom order) to POSCAR atom order.

    Returns a new dict (blocks are NOT copied — caller must copy if needed).

    Raises ``RuntimeError`` if duplicate keys are encountered (should be
    impossible since ``build_atom_permutation`` is a bijection).
    """
    old2new, _ = structure.build_atom_permutation()
    pair_blocks: dict[tuple, np.ndarray] = {}
    for (R1, R2, R3, i_aims, j_aims), block in matrix.blocks.items():
        i_deeph = int(old2new[i_aims])
        j_deeph = int(old2new[j_aims])
        key = (R1, R2, R3, i_deeph, j_deeph)
        if key not in pair_blocks:
            pair_blocks[key] = block
        else:
            raise RuntimeError(
                f"_aimspy_blocks_to_poscar: duplicate key {key} "
                f"(indicates a bug in build_atom_permutation or matrix.blocks)"
            )
    return pair_blocks


def _blocks_to_flat_entries(
    blocks: dict[tuple[int, ...], np.ndarray],
    atom_pairs: np.ndarray,
    chunk_boundaries: np.ndarray,
    chunk_shapes: np.ndarray,
    factor: float = 1.0,
) -> np.ndarray:
    """Flatten a blocks dict to 1D entries following the existing CSR layout.

    Iterates over ``atom_pairs`` order, looks up each block by its key,
    flattens to 1D and concatenates. Missing blocks are zero-filled.
    """
    n_pairs = atom_pairs.shape[0]
    lst: list[np.ndarray] = []
    for ip in range(n_pairs):
        key = tuple(int(x) for x in atom_pairs[ip])
        nr = int(chunk_shapes[ip, 0])
        nc = int(chunk_shapes[ip, 1])
        blk = blocks.get(key)
        if blk is not None:
            lst.append(np.ascontiguousarray(blk, dtype=np.float64).ravel())
        else:
            lst.append(np.zeros(nr * nc, dtype=np.float64))
    entries = np.concatenate(lst) if lst else np.array([], dtype=np.float64)
    if factor != 1.0:
        entries *= factor
    return entries


def _reorder_coords(structure) -> np.ndarray:
    """Return atom coords in POSCAR (element-grouped) order."""
    _, new2old = structure.build_atom_permutation()
    n = structure.n_atoms
    coords = np.zeros((n, 3), dtype=np.float64)
    for i_deeph in range(n):
        i_aims = int(new2old[i_deeph])
        coords[i_deeph] = structure.atom_coords[i_aims]
    return coords


def _build_elements_orbital_map(structure) -> dict[str, list[int]]:
    """Build ``{element: [l per shell]}`` — one *l* per (n,l) shell.

    Matches the reference ``_parse_basis`` behaviour: record an entry
    when ``m == -l`` (first *m* of each shell), keeping duplicates for
    same-*l*-different-*n* shells (e.g. two s-shells → ``[0, 0, 1, …]``).
    """
    result: dict[str, list[int]] = {}
    for idx in range(structure.n_atoms):
        elem = structure.atom_symbols[idx]
        mask = structure.basis_atom == idx
        indices = np.where(mask)[0]
        ls_for_atom: list[int] = []
        for i in indices:
            ll = int(structure.basis_l[i])
            m = int(structure.basis_m[i])
            if m == -ll:
                ls_for_atom.append(ll)
        result[elem] = ls_for_atom  # overwrite — same element ⇒ same basis
    return result


def _compute_n_basis(
    atom_symbols: list[str],
    elements_orbital_map: dict[str, list[int]],
) -> int:
    """Total number of basis functions = Σ(count × Σ(2l+1))."""
    counts = Counter(atom_symbols)
    n = 0
    for elem, cnt in counts.items():
        shells = elements_orbital_map.get(elem, [])
        n += cnt * sum(2 * ll + 1 for ll in shells)
    return n


def _compute_occupation(atom_symbols: list[str]) -> int:
    """Total number of electrons = Σ(Z)."""
    return sum(_ATOMIC_NUMBERS.get(s, 0) for s in atom_symbols)


# =============================================================================
# DeepHData
# =============================================================================
@dataclass
class DeepHData:
    """Complete DeepH-format data: structure + one or more matrices.

    Read from a directory containing:
      - ``POSCAR``          — lattice, atom symbols, atom coords
      - ``info.json``       — ``elements_orbital_map``
      - ``hamiltonian.h5``  — *required* — atom_pairs, chunk_*, entries (eV)
      - ``overlap.h5``      — *optional* — same layout, overlap entries
      - ``hamiltonian_init.h5`` — *optional* — same layout, initial Hamiltonian
        entries (the ``0`` in the filename denotes the initial Hamiltonian,
        per DeepH on-disk convention)

    Can also be constructed in-memory via ``from_memory`` or from
    aimspy standard-format matrices via ``from_aimspy``.
    """

    # structure (POSCAR order = element-grouped)
    lattice: np.ndarray  # (3, 3) in Angstrom
    atom_symbols: list[str]  # POSCAR order
    atom_coords: np.ndarray  # (n_atoms, 3) in Angstrom
    elements_orbital_map: dict[str, list[int]]
    n_basis: int  # total number of basis functions

    # shared CSR layout (same for all matrices)
    atom_pairs: np.ndarray  # (N, 5)  [R1,R2,R3,i,j]
    chunk_boundaries: np.ndarray  # (N+1,)
    chunk_shapes: np.ndarray  # (N, 2)

    # Hamiltonian entries (eV) — optional to support overlap-only scenarios
    entries: Optional[np.ndarray] = None  # (M,) float64, eV
    # optional additional matrices
    overlap_entries: Optional[np.ndarray] = None
    initial_hamiltonian_entries: Optional[np.ndarray] = None

    # metadata (for info.json round-trip)
    fermi_energy_eV: float = 0.0

    # pre-specified save path (set by from_directory / from_aimspy / path= kwarg)
    path: Optional[Path] = None

    # ----------------------------------------------------------------
    # Construction from directory
    # ----------------------------------------------------------------
    @classmethod
    def from_directory(cls, path: Union[str, Path]) -> "DeepHData":
        """Read POSCAR + info.json + matrix .h5 files from *path*.

        Requires POSCAR + info.json + at least one matrix file
        (``hamiltonian.h5``, ``overlap.h5``, or ``hamiltonian_init.h5``).
        Sets ``self.path = path`` for subsequent ``save_*()`` calls.
        """
        path = Path(path)
        if not path.is_dir():
            raise FileNotFoundError(f"DeepH directory not found: {path}")

        poscar_path = path / "POSCAR"
        info_path = path / "info.json"

        if not poscar_path.is_file():
            raise FileNotFoundError(f"POSCAR missing in {path}")
        if not info_path.is_file():
            raise FileNotFoundError(f"info.json missing in {path}")

        # Detect available matrix files
        matrix_files: list[tuple[str, Path]] = [
            ("hamiltonian", path / "hamiltonian.h5"),
            ("overlap", path / "overlap.h5"),
            ("initial_hamiltonian", path / "hamiltonian_init.h5"),
        ]
        found = [(name, p) for name, p in matrix_files if p.is_file()]
        if not found:
            raise FileNotFoundError(
                f"No matrix .h5 file found in {path} "
                "(expected hamiltonian.h5, overlap.h5, or hamiltonian_init.h5)"
            )

        lattice, atom_symbols, atom_coords = _read_poscar(poscar_path)
        with open(info_path, "r") as f:
            info = json.load(f)
        eom = info.get("elements_orbital_map", {})
        n_basis = info.get("orbits_quantity", 0)
        if n_basis == 0:
            n_basis = _compute_n_basis(atom_symbols, eom)
        fermi_eV = info.get("fermi_energy_eV", 0.0)

        # Read CSR layout from the first found file
        first_name, first_path = found[0]
        with h5py.File(first_path, "r") as f:
            atom_pairs = f["atom_pairs"][:].astype(np.int32)
            cb = f["chunk_boundaries"][:].astype(np.int32)
            cs = f["chunk_shapes"][:].astype(np.int32)

        # Read each matrix
        entries = None
        overlap_entries = None
        init_entries = None
        for name, p in found:
            with h5py.File(p, "r") as f:
                data = f["entries"][:].astype(np.float64)
            if name == "hamiltonian":
                entries = data
            elif name == "overlap":
                overlap_entries = data
            elif name == "initial_hamiltonian":
                init_entries = data

        return cls(
            lattice=lattice,
            atom_symbols=atom_symbols,
            atom_coords=atom_coords,
            elements_orbital_map=eom,
            n_basis=n_basis,
            atom_pairs=atom_pairs,
            chunk_boundaries=cb,
            chunk_shapes=cs,
            entries=entries,
            overlap_entries=overlap_entries,
            initial_hamiltonian_entries=init_entries,
            fermi_energy_eV=fermi_eV,
            path=path,
        )

    @classmethod
    def from_memory(
        cls,
        lattice: np.ndarray,
        atom_symbols: list[str],
        atom_coords: np.ndarray,
        elements_orbital_map: dict[str, list[int]],
        hamiltonian_blocks: Optional[dict[tuple[int, ...], np.ndarray]] = None,
        overlap_blocks: Optional[dict[tuple[int, ...], np.ndarray]] = None,
        initial_hamiltonian_blocks: Optional[dict[tuple[int, ...], np.ndarray]] = None,
        n_basis: int = 0,
        fermi_energy_eV: float = 0.0,
        path: Optional[Union[str, Path]] = None,
    ) -> "DeepHData":
        """Build from in-memory pair-block dicts.

        All matrix blocks are optional — at least one must be given.
        Keys are ``(R1,R2,R3,i,j)`` with atoms in POSCAR order.
        Hamiltonian / initial_hamiltonian blocks in **Hartree**
        (converted to eV here). Overlap blocks are dimensionless.
        """
        if n_basis == 0:
            n_basis = _compute_n_basis(atom_symbols, elements_orbital_map)
        layout_blocks = (
            hamiltonian_blocks or overlap_blocks or initial_hamiltonian_blocks
        )
        if layout_blocks is None:
            raise AimspyConfigError("At least one matrix blocks dict must be provided")
        sorted_keys = sorted(layout_blocks.keys())
        n_pairs = len(sorted_keys)
        atom_pairs = np.zeros((n_pairs, 5), dtype=np.int32)
        chunk_boundaries = np.zeros((n_pairs + 1,), dtype=np.int32)
        chunk_shapes = np.zeros((n_pairs, 2), dtype=np.int32)
        entries_lst: list[np.ndarray] = []
        overlap_lst: list[np.ndarray] = []
        init_ham_lst: list[np.ndarray] = []

        for ip, key in enumerate(sorted_keys):
            atom_pairs[ip] = [int(k) for k in key]
            block = layout_blocks[key]
            nr, nc = int(block.shape[0]), int(block.shape[1])
            chunk_shapes[ip] = (nr, nc)

            entries_lst.append(np.ascontiguousarray(block, dtype=np.float64).ravel())
            chunk_boundaries[ip + 1] = chunk_boundaries[ip] + nr * nc

            if overlap_blocks:
                blk = overlap_blocks.get(key)
                if blk is not None:
                    overlap_lst.append(
                        np.ascontiguousarray(blk, dtype=np.float64).ravel()
                    )
                else:
                    overlap_lst.append(np.zeros(nr * nc, dtype=np.float64))

            if initial_hamiltonian_blocks:
                blk = initial_hamiltonian_blocks.get(key)
                if blk is not None:
                    init_ham_lst.append(
                        np.ascontiguousarray(blk, dtype=np.float64).ravel()
                    )
                else:
                    init_ham_lst.append(np.zeros(nr * nc, dtype=np.float64))

        entries = None
        if hamiltonian_blocks:
            entries = np.concatenate(entries_lst)
            entries *= HARTREE_TO_EV  # Hartree → eV

        ovlp = None
        if overlap_lst:
            ovlp = np.concatenate(overlap_lst)
            # Overlap is dimensionless — no unit conversion

        init = None
        if init_ham_lst:
            init = np.concatenate(init_ham_lst)
            init *= HARTREE_TO_EV  # Hartree → eV

        return cls(
            lattice=np.asarray(lattice, dtype=np.float64),
            atom_symbols=list(atom_symbols),
            atom_coords=np.asarray(atom_coords, dtype=np.float64),
            elements_orbital_map=dict(elements_orbital_map),
            n_basis=n_basis,
            atom_pairs=atom_pairs,
            chunk_boundaries=chunk_boundaries,
            chunk_shapes=chunk_shapes,
            entries=entries,
            overlap_entries=ovlp,
            initial_hamiltonian_entries=init,
            fermi_energy_eV=fermi_energy_eV,
            path=Path(path) if path is not None else None,
        )

    # ----------------------------------------------------------------
    # Construction from aimspy standard format
    # ----------------------------------------------------------------
    @classmethod
    def from_aimspy(
        cls,
        structure,
        hamiltonian=None,
        overlap=None,
        initial_hamiltonian=None,
        template: Optional["DeepHData"] = None,
        path: Optional[Union[str, Path]] = None,
    ) -> "DeepHData":
        """Build from aimspy standard-format matrices + structure.

        All matrices are optional — at least one must be given.

        Parameters
        ----------
        structure : AimspyStructure
            Used to build POSCAR-order layout unless *template* is given.
        hamiltonian : AimspyMatrix, optional
            Hamiltonian (Hartree, aims atom order).
        overlap : AimspyMatrix, optional
            Overlap matrix (dimensionless).
        initial_hamiltonian : AimspyMatrix, optional
            Initial / free-atom Hamiltonian (Hartree).
        template : DeepHData, optional
            If given, reuse its structure fields (lattice, atom_symbols,
            atom_coords, elements_orbital_map) instead of rebuilding
            from *structure*.  Convenient when adding matrices to an
            existing DeepH dataset.
        path : str or Path, optional
            Pre-specified save path for subsequent ``save_*()`` calls.
        """
        if hamiltonian is None and overlap is None and initial_hamiltonian is None:
            raise AimspyConfigError("At least one matrix must be provided")
        if template is not None:
            lattice = template.lattice.copy()
            atom_symbols = list(template.atom_symbols)
            atom_coords = template.atom_coords.copy()
            eom = dict(template.elements_orbital_map)
            n_basis = template.n_basis
            fermi_eV = template.fermi_energy_eV
        else:
            lattice = structure.lattice.copy()
            atom_symbols = list(structure.atoms_species_sorted)
            coords = _reorder_coords(structure)
            atom_coords = coords
            eom = _build_elements_orbital_map(structure)
            n_basis = structure.n_basis
            fermi_eV = 0.0

        hamiltonian_blocks = (
            _aimspy_blocks_to_poscar(hamiltonian, structure)
            if hamiltonian is not None
            else None
        )
        overlap_blocks = (
            _aimspy_blocks_to_poscar(overlap, structure)
            if overlap is not None
            else None
        )
        initial_hamiltonian_blocks = (
            _aimspy_blocks_to_poscar(initial_hamiltonian, structure)
            if initial_hamiltonian is not None
            else None
        )

        return cls.from_memory(
            lattice=lattice,
            atom_symbols=atom_symbols,
            atom_coords=atom_coords,
            elements_orbital_map=eom,
            hamiltonian_blocks=hamiltonian_blocks,
            overlap_blocks=overlap_blocks,
            initial_hamiltonian_blocks=initial_hamiltonian_blocks,
            n_basis=n_basis,
            fermi_energy_eV=fermi_eV,
            path=path,
        )

    # ----------------------------------------------------------------
    # Set individual matrices from AimspyMatrix
    # ----------------------------------------------------------------
    def set_hamiltonian(
        self, matrix: "AimspyMatrix", structure: "AimspyStructure"
    ) -> None:
        """Convert and store Hamiltonian entries (eV) from *matrix*."""
        blocks = _aimspy_blocks_to_poscar(matrix, structure)
        self.entries = _blocks_to_flat_entries(
            blocks,
            self.atom_pairs,
            self.chunk_boundaries,
            self.chunk_shapes,
            factor=HARTREE_TO_EV,
        )

    def set_overlap(self, matrix: "AimspyMatrix", structure: "AimspyStructure") -> None:
        """Convert and store overlap entries (dimensionless) from *matrix*."""
        blocks = _aimspy_blocks_to_poscar(matrix, structure)
        self.overlap_entries = _blocks_to_flat_entries(
            blocks,
            self.atom_pairs,
            self.chunk_boundaries,
            self.chunk_shapes,
        )

    def set_initial_hamiltonian(
        self, matrix: "AimspyMatrix", structure: "AimspyStructure"
    ) -> None:
        """Convert and store initial Hamiltonian entries (eV) from *matrix*."""
        blocks = _aimspy_blocks_to_poscar(matrix, structure)
        self.initial_hamiltonian_entries = _blocks_to_flat_entries(
            blocks,
            self.atom_pairs,
            self.chunk_boundaries,
            self.chunk_shapes,
            factor=HARTREE_TO_EV,
        )

    # ----------------------------------------------------------------
    # Save individual matrices / metadata
    # ----------------------------------------------------------------
    def _require_path(self) -> Path:
        if self.path is None:
            raise AimspyConfigError(
                "No path specified; pass path= to constructor or save_*()"
            )
        return self.path

    def _write_matrix_h5(self, file_path: Path, entries: np.ndarray) -> None:
        with h5py.File(file_path, "w") as f:
            f.create_dataset("atom_pairs", data=self.atom_pairs, dtype="i4")
            f.create_dataset("chunk_boundaries", data=self.chunk_boundaries, dtype="i4")
            f.create_dataset("chunk_shapes", data=self.chunk_shapes, dtype="i4")
            f.create_dataset("entries", data=entries)

    def save_metadata(self, path: Optional[Union[str, Path]] = None) -> None:
        """Write POSCAR + info.json to *path* (default: self.path)."""
        p = Path(path) if path is not None else self._require_path()
        p.mkdir(parents=True, exist_ok=True)
        _write_poscar(p / "POSCAR", self.lattice, self.atom_symbols, self.atom_coords)
        _write_info_json(p / "info.json", self)

    def save_hamiltonian(self, path: Optional[Union[str, Path]] = None) -> None:
        """Write hamiltonian.h5 (requires entries to be set)."""
        if self.entries is None:
            raise AimspyConfigError("No Hamiltonian entries to save")
        p = Path(path) if path is not None else self._require_path()
        p.mkdir(parents=True, exist_ok=True)
        self._write_matrix_h5(p / "hamiltonian.h5", self.entries)

    def save_overlap(self, path: Optional[Union[str, Path]] = None) -> None:
        """Write overlap.h5 (requires overlap_entries to be set)."""
        if self.overlap_entries is None:
            raise AimspyConfigError("No overlap entries to save")
        p = Path(path) if path is not None else self._require_path()
        p.mkdir(parents=True, exist_ok=True)
        self._write_matrix_h5(p / "overlap.h5", self.overlap_entries)

    def save_initial_hamiltonian(self, path: Optional[Union[str, Path]] = None) -> None:
        """Write hamiltonian_init.h5 (requires initial_hamiltonian_entries)."""
        if self.initial_hamiltonian_entries is None:
            raise AimspyConfigError("No initial Hamiltonian entries to save")
        p = Path(path) if path is not None else self._require_path()
        p.mkdir(parents=True, exist_ok=True)
        self._write_matrix_h5(
            p / "hamiltonian_init.h5", self.initial_hamiltonian_entries
        )

    def save(self, path: Optional[Union[str, Path]] = None) -> None:
        """Write all non-None content to *path* (default: self.path).

        Saves POSCAR + info.json + every matrix that has been set.
        """
        p = Path(path) if path is not None else self._require_path()
        self.save_metadata(p)
        if self.entries is not None:
            self.save_hamiltonian(p)
        if self.overlap_entries is not None:
            self.save_overlap(p)
        if self.initial_hamiltonian_entries is not None:
            self.save_initial_hamiltonian(p)

    @property
    def n_pairs(self) -> int:
        return self.atom_pairs.shape[0]

    @property
    def n_atoms(self) -> int:
        return len(self.atom_symbols)

    def __repr__(self) -> str:
        extra = []
        if self.entries is not None:
            extra.append("+H")
        if self.overlap_entries is not None:
            extra.append("+S")
        if self.initial_hamiltonian_entries is not None:
            extra.append("+H_init")
        tag = " ".join(extra)
        return (
            f"DeepHData(n_atoms={self.n_atoms}, n_pairs={self.n_pairs}"
            + (f", {tag}" if tag else "")
            + f", species={list(self.atom_symbols)})"
        )

    # ----------------------------------------------------------------
    # Conversion to aimspy standard format
    # ----------------------------------------------------------------
    def to_aimspy(self, structure: "AimspyStructure") -> "AimspyMatrix":
        """Convert this DeepH data to aimspy standard format.

        Converts the Hamiltonian entries (``self.entries``). If
        ``entries`` is None, raises :class:`aimspy.AimspyConfigError`.

        - Atom reordering: POSCAR → aims (via stable-sort un-permutation)
        - R: no flip (same convention: ``R_aimspy = R_deeph = -R_aims``)
        - Parity: no change (same wiki convention)
        - Units: eV → Hartree

        The result is suitable for passing to
        :meth:`aimspy.Calculator.modify_init_ham` via ``source=``.

        Parameters
        ----------
        structure : AimspyStructure
            Live runtime structure (built from ``AimspyInfo`` after
            ``aimspy_init``); provides the POSCAR↔aims atom permutation.
        """
        from ...matrix import AimspyMatrix

        if self.entries is None:
            raise AimspyConfigError(
                "No Hamiltonian entries to convert; set entries first"
            )
        _, new2old = structure.build_atom_permutation()
        # new2old[POSCAR_atom] = aims_atom

        entries = self.entries
        cb = self.chunk_boundaries
        cs = self.chunk_shapes
        ap = self.atom_pairs

        blocks: dict = {}
        for ip in range(self.n_pairs):
            R1 = int(ap[ip, 0])
            R2 = int(ap[ip, 1])
            R3 = int(ap[ip, 2])
            i_deeph = int(ap[ip, 3])
            j_deeph = int(ap[ip, 4])

            i_aims = int(new2old[i_deeph])
            j_aims = int(new2old[j_deeph])

            bnd = int(cb[ip])
            nr = int(cs[ip, 0])
            nc = int(cs[ip, 1])

            block = entries[bnd : bnd + nr * nc].reshape(nr, nc).copy()
            block *= EV_TO_HARTREE  # eV -> Hartree

            key = (R1, R2, R3, i_aims, j_aims)
            blocks[key] = block

        return AimspyMatrix(blocks=blocks, n_spin=1)


# -------------------------------------------------------------------
# Internal: POSCAR reader / writer
# -------------------------------------------------------------------
def _read_poscar(path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Minimal POSCAR parser (VASP4 + VASP5 element-line formats)."""
    lines = path.read_text().splitlines()
    lines = [ln.strip() for ln in lines if ln.strip()]

    scale = float(lines[1])
    lat = scale * np.array(
        [[float(x) for x in lines[i].split()] for i in range(2, 5)],
        dtype=np.float64,
    )

    # detect VASP5 (element line)
    tokens6 = lines[5].split()
    try:
        [float(x) for x in tokens6]  # purely numeric = VASP4
        have_element_line = False
    except ValueError:
        have_element_line = True

    if have_element_line:
        symbols_on_line = lines[5].split()
        counts = [int(x) for x in lines[6].split()]
        coord_start = 7  # line 7 may be coord-type or coordinate
    else:
        counts = [int(x) for x in lines[5].split()]
        symbols_on_line = lines[0].split()
        coord_start = 6  # line 6 may be coord-type or coordinate

    # Skip optional "Selective dynamics" and mandatory coordinate-type line
    while coord_start < len(lines):
        token = lines[coord_start].split()[0].lower()
        if token in ("cartesian", "direct", "selective", "kartesian", "d"):
            coord_start += 1
        else:
            break

    # expand symbols
    atom_symbols: list[str] = []
    total_atoms = sum(counts)
    n_uniq = min(len(symbols_on_line), len(counts))
    for i in range(n_uniq):
        atom_symbols.extend([symbols_on_line[i]] * counts[i])
    if len(atom_symbols) < total_atoms:
        missing = total_atoms - len(atom_symbols)
        atom_symbols.extend(
            [f"X{i + 1}" for i in range(len(atom_symbols), len(atom_symbols) + missing)]
        )

    n_atoms = len(atom_symbols)
    coords = np.zeros((n_atoms, 3), dtype=np.float64)
    for i in range(n_atoms):
        coords[i] = [float(x) for x in lines[coord_start + i].split()[:3]]

    # handle Direct coords
    coord_type = lines[coord_start - 1].split()[0].lower()
    if coord_type == "selective":
        coord_type = lines[coord_start - 2].split()[0].lower()
    if coord_type.startswith("d"):
        coords = coords @ lat

    return lat, atom_symbols, coords


def _write_poscar(
    path: Path,
    lattice: np.ndarray,
    atom_symbols: list[str],
    atom_coords: np.ndarray,
) -> None:
    # Group by symbol for counts (preserve POSCAR order)
    seen: list[str] = []
    counts: list[int] = []
    for s in atom_symbols:
        if s in seen:
            counts[seen.index(s)] += 1
        else:
            seen.append(s)
            counts.append(1)

    lines = [
        "POSCAR generated by aimspy",
        "1.0",
        f"  {lattice[0,0]:.16f}  {lattice[0,1]:.16f}  {lattice[0,2]:.16f}",
        f"  {lattice[1,0]:.16f}  {lattice[1,1]:.16f}  {lattice[1,2]:.16f}",
        f"  {lattice[2,0]:.16f}  {lattice[2,1]:.16f}  {lattice[2,2]:.16f}",
    ]
    lines.append("  ".join(seen))
    lines.append("  ".join(str(c) for c in counts))
    lines.append("Cartesian")
    for c in atom_coords:
        lines.append(f"  {c[0]:.16f}  {c[1]:.16f}  {c[2]:.16f}")

    path.write_text("\n".join(lines) + "\n")


def _write_info_json(path: Path, data: DeepHData) -> None:
    n_basis = data.n_basis
    if n_basis <= 0:
        n_basis = _compute_n_basis(data.atom_symbols, data.elements_orbital_map)
    obj: dict = {
        "atoms_quantity": data.n_atoms,
        "orbits_quantity": n_basis,
        "occupation": _compute_occupation(data.atom_symbols),
        "orthogonal_basis": False,
        "spinful": False,
        "fermi_energy_eV": data.fermi_energy_eV,
        "elements_orbital_map": data.elements_orbital_map,
    }
    path.write_text(json.dumps(obj, indent=2))
