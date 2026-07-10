"""DeepH-format data: reader/writer for POSCAR + info.json + hamiltonian.h5.

Provides ``DeepHData``, a complete in-memory representation of DeepH
data including structure info (from POSCAR / info.json) and matrix
data (from hamiltonian.h5 / overlap.h5 / hamiltonian0.h5).
Supports file I/O, in-memory construction, and conversion from
aimspy standard format via ``from_aimspy``.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import h5py
import numpy as np

from ...data import HARTREE_TO_EV


# -------------------------------------------------------------------
# Atomic-number lookup (for occupation computation in info.json)
# -------------------------------------------------------------------
_ATOMIC_NUMBERS: dict[str, int] = {
    "H": 1,   "He": 2,  "Li": 3,  "Be": 4,  "B": 5,   "C": 6,
    "N": 7,   "O": 8,   "F": 9,   "Ne": 10, "Na": 11, "Mg": 12,
    "Al": 13, "Si": 14, "P": 15,  "S": 16,  "Cl": 17, "Ar": 18,
    "K": 19,  "Ca": 20, "Sc": 21, "Ti": 22, "V": 23,  "Cr": 24,
    "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
    "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36,
    "Rb": 37, "Sr": 38, "Y": 39,  "Zr": 40, "Nb": 41, "Mo": 42,
    "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48,
    "In": 49, "Sn": 50, "Sb": 51, "Te": 52, "I": 53,  "Xe": 54,
    "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60,
    "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66,
    "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71, "Hf": 72,
    "Ta": 73, "W": 74,  "Re": 75, "Os": 76, "Ir": 77, "Pt": 78,
    "Au": 79, "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84,
    "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90,
    "Pa": 91, "U": 92,  "Np": 93, "Pu": 94, "Am": 95, "Cm": 96,
    "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100, "Md": 101, "No": 102,
    "Lr": 103, "Rf": 104, "Db": 105, "Sg": 106, "Bh": 107, "Hs": 108,
    "Mt": 109, "Ds": 110, "Rg": 111, "Cn": 112, "Nh": 113, "Fl": 114,
    "Mc": 115, "Lv": 116, "Ts": 117, "Og": 118,
}


# -------------------------------------------------------------------
# Conversion helpers (shared with converter.py)
# -------------------------------------------------------------------
def _map_to_center_cell(
    coords: np.ndarray, lattice: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    """Map periodic positions to the cell centred at origin.

    Mimics FHI-aims ``map_to_center_cell`` and DeepH-reference
    ``_map_positions_to_center_cell``.
    """
    inv = np.linalg.inv(lattice)
    frac = coords @ inv
    frac = frac - eps
    frac = frac - np.rint(frac)
    frac = frac + eps
    return frac @ lattice


def _aimspy_blocks_to_poscar(
    matrix, structure
) -> dict[tuple, np.ndarray]:
    """Reorder aimspy blocks (aims atom order) to POSCAR atom order.

    Returns a new dict (blocks are NOT copied — caller must copy if needed).
    """
    old2new, _ = structure.build_atom_permutation()
    pair_blocks: dict[tuple, np.ndarray] = {}
    for (R1, R2, R3, i_aims, j_aims), block in matrix.blocks.items():
        i_deeph = int(old2new[i_aims])
        j_deeph = int(old2new[j_aims])
        key = (R1, R2, R3, i_deeph, j_deeph)
        if key not in pair_blocks:
            pair_blocks[key] = block  # reference, no copy
        else:
            pair_blocks[key] = np.maximum(pair_blocks[key], block)
    return pair_blocks


def _reorder_coords(structure) -> np.ndarray:
    """Return atom coords in POSCAR (element‑grouped) order."""
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
    same-*l*-different-*n* shells (e.g. two s‑shells → ``[0, 0, 1, …]``).
    """
    result: dict[str, list[int]] = {}
    for idx in range(structure.n_atoms):
        elem = structure.atom_symbols[idx]
        mask = structure.basis_atom == idx
        indices = np.where(mask)[0]
        ls_for_atom: list[int] = []
        for i in indices:
            l = int(structure.basis_l[i])
            m = int(structure.basis_m[i])
            if m == -l:
                ls_for_atom.append(l)
        result[elem] = ls_for_atom   # overwrite — same element ⇒ same basis
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
        n += cnt * sum(2 * l + 1 for l in shells)
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
      - ``hamiltonian0.h5`` — *optional* — same layout, initial H0 entries

    Can also be constructed in-memory via ``from_memory`` or from
    aimspy standard-format matrices via ``from_aimspy``.
    """

    # structure (POSCAR order = element‑grouped)
    lattice: np.ndarray                      # (3, 3) in Angstrom
    atom_symbols: list[str]                  # POSCAR order
    atom_coords: np.ndarray                  # (n_atoms, 3) in Angstrom
    elements_orbital_map: dict[str, list[int]]
    n_basis: int                             # total number of basis functions

    # shared CSR layout (same for all matrices)
    atom_pairs: np.ndarray                  # (N, 5)  [R1,R2,R3,i,j]
    chunk_boundaries: np.ndarray            # (N+1,)
    chunk_shapes: np.ndarray                # (N, 2)

    # H entries (eV), *always* present
    entries: np.ndarray                     # (M,) float64, eV

    # optional additional matrices (eV)
    overlap_entries: Optional[np.ndarray] = None
    initial_hamiltonian_entries: Optional[np.ndarray] = None

    # metadata (for info.json round-trip)
    fermi_energy_eV: float = 0.0

    # ----------------------------------------------------------------
    # Construction from directory
    # ----------------------------------------------------------------
    @classmethod
    def from_directory(cls, path: Union[str, Path]) -> "DeepHData":
        """Read POSCAR + info.json + hamiltonian.h5 from *path*.

        Optionally also reads ``overlap.h5`` and ``hamiltonian0.h5``
        if they exist.
        """
        path = Path(path)
        if not path.is_dir():
            raise FileNotFoundError(f"DeepH directory not found: {path}")

        poscar_path = path / "POSCAR"
        info_path   = path / "info.json"
        h5_path     = path / "hamiltonian.h5"

        if not poscar_path.is_file():
            raise FileNotFoundError(f"POSCAR missing in {path}")
        if not info_path.is_file():
            raise FileNotFoundError(f"info.json missing in {path}")
        if not h5_path.is_file():
            raise FileNotFoundError(f"hamiltonian.h5 missing in {path}")

        lattice, atom_symbols, atom_coords = _read_poscar(poscar_path)
        with open(info_path, "r") as f:
            info = json.load(f)
        eom = info.get("elements_orbital_map", {})
        n_basis = info.get("orbits_quantity", 0)
        if n_basis == 0:
            n_basis = _compute_n_basis(atom_symbols, eom)
        fermi_eV = info.get("fermi_energy_eV", 0.0)

        with h5py.File(h5_path, "r") as f:
            atom_pairs = f["atom_pairs"][:].astype(np.int32)
            cb = f["chunk_boundaries"][:].astype(np.int32)
            cs = f["chunk_shapes"][:].astype(np.int32)
            entries = f["entries"][:].astype(np.float64)

        overlap_entries = None
        ovlp_path = path / "overlap.h5"
        if ovlp_path.is_file():
            with h5py.File(ovlp_path, "r") as f:
                overlap_entries = f["entries"][:].astype(np.float64)

        init_entries = None
        h0_path = path / "hamiltonian0.h5"
        if h0_path.is_file():
            with h5py.File(h0_path, "r") as f:
                init_entries = f["entries"][:].astype(np.float64)

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
        )

    @classmethod
    def from_memory(
        cls,
        lattice: np.ndarray,
        atom_symbols: list[str],
        atom_coords: np.ndarray,
        elements_orbital_map: dict[str, list[int]],
        pair_blocks: dict[tuple[int, ...], np.ndarray],
        overlap_blocks: Optional[dict[tuple[int, ...], np.ndarray]] = None,
        h0_blocks: Optional[dict[tuple[int, ...], np.ndarray]] = None,
        n_basis: int = 0,
        fermi_energy_eV: float = 0.0,
    ) -> "DeepHData":
        """Build from in-memory pair-block dicts.

        ``pair_blocks`` keys are ``(R1,R2,R3,i,j)`` with atoms in
        POSCAR order.  Block values are in **Hartree** (converted to eV
        here).  Overlap blocks are dimensionless — no unit conversion.
        """
        if n_basis == 0:
            n_basis = _compute_n_basis(atom_symbols, elements_orbital_map)
        sorted_keys = sorted(pair_blocks.keys())
        n_pairs = len(sorted_keys)
        atom_pairs = np.zeros((n_pairs, 5), dtype=np.int32)
        chunk_boundaries = np.zeros((n_pairs + 1,), dtype=np.int32)
        chunk_shapes = np.zeros((n_pairs, 2), dtype=np.int32)
        entries_lst: list[np.ndarray] = []
        overlap_lst: list[np.ndarray] = []
        h0_lst: list[np.ndarray] = []

        for ip, key in enumerate(sorted_keys):
            atom_pairs[ip] = [int(k) for k in key]
            block = pair_blocks[key]
            nr, nc = int(block.shape[0]), int(block.shape[1])
            chunk_shapes[ip] = (nr, nc)

            entries_lst.append(
                np.ascontiguousarray(block, dtype=np.float64).ravel()
            )
            chunk_boundaries[ip + 1] = chunk_boundaries[ip] + nr * nc

            if overlap_blocks is not None:
                blk = overlap_blocks.get(key)
                if blk is not None:
                    overlap_lst.append(
                        np.ascontiguousarray(blk, dtype=np.float64).ravel()
                    )
                else:
                    overlap_lst.append(np.zeros(nr * nc, dtype=np.float64))

            if h0_blocks is not None:
                blk = h0_blocks.get(key)
                if blk is not None:
                    h0_lst.append(
                        np.ascontiguousarray(blk, dtype=np.float64).ravel()
                    )
                else:
                    h0_lst.append(np.zeros(nr * nc, dtype=np.float64))

        entries = (
            np.concatenate(entries_lst)
            if entries_lst
            else np.array([], dtype=np.float64)
        )
        entries *= HARTREE_TO_EV  # Hartree → eV

        ovlp = None
        if overlap_lst:
            ovlp = np.concatenate(overlap_lst)
            # Overlap is dimensionless — no unit conversion

        init = None
        if h0_lst:
            init = np.concatenate(h0_lst)
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
        )

    # ----------------------------------------------------------------
    # Construction from aimspy standard format
    # ----------------------------------------------------------------
    @classmethod
    def from_aimspy(
        cls,
        structure,
        H,
        S=None,
        H0=None,
        template: Optional["DeepHData"] = None,
    ) -> "DeepHData":
        """Build from aimspy standard-format matrices + structure.

        Parameters
        ----------
        structure : AimspyStructure
            Used to build POSCAR-order layout unless *template* is given.
        H : AimspyMatrix
            Hamiltonian (Hartree, aims atom order).
        S : AimspyMatrix, optional
            Overlap matrix (dimensionless).
        H0 : AimspyMatrix, optional
            Initial / free-atom Hamiltonian (Hartree).
        template : DeepHData, optional
            If given, reuse its structure fields (lattice, atom_symbols,
            atom_coords, elements_orbital_map) instead of rebuilding
            from *structure*.  Convenient when adding matrices to an
            existing DeepH dataset.
        """
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
            if structure.n_periodic > 0 and lattice.size >= 9:
                coords = _map_to_center_cell(coords, lattice)
            atom_coords = coords
            eom = _build_elements_orbital_map(structure)
            n_basis = structure.n_basis
            fermi_eV = 0.0

        pair_blocks_H = _aimspy_blocks_to_poscar(H, structure)
        pair_blocks_S = (
            _aimspy_blocks_to_poscar(S, structure) if S is not None else None
        )
        pair_blocks_H0 = (
            _aimspy_blocks_to_poscar(H0, structure) if H0 is not None else None
        )

        return cls.from_memory(
            lattice=lattice,
            atom_symbols=atom_symbols,
            atom_coords=atom_coords,
            elements_orbital_map=eom,
            pair_blocks=pair_blocks_H,
            overlap_blocks=pair_blocks_S,
            h0_blocks=pair_blocks_H0,
            n_basis=n_basis,
            fermi_energy_eV=fermi_eV,
        )

    # ----------------------------------------------------------------
    # I/O
    # ----------------------------------------------------------------
    def save(self, path: Union[str, Path]) -> None:
        """Write POSCAR + info.json + hamiltonian.h5 to *path*.

        Also writes ``overlap.h5`` and ``hamiltonian0.h5`` when the
        corresponding optional fields are populated.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        _write_poscar(
            path / "POSCAR", self.lattice, self.atom_symbols, self.atom_coords
        )
        _write_info_json(path / "info.json", self)
        with h5py.File(path / "hamiltonian.h5", "w") as f:
            f.create_dataset("atom_pairs", data=self.atom_pairs, dtype="i4")
            f.create_dataset(
                "chunk_boundaries", data=self.chunk_boundaries, dtype="i4"
            )
            f.create_dataset(
                "chunk_shapes", data=self.chunk_shapes, dtype="i4"
            )
            f.create_dataset("entries", data=self.entries)

        if self.overlap_entries is not None:
            with h5py.File(path / "overlap.h5", "w") as f:
                f.create_dataset(
                    "atom_pairs", data=self.atom_pairs, dtype="i4"
                )
                f.create_dataset(
                    "chunk_boundaries",
                    data=self.chunk_boundaries,
                    dtype="i4",
                )
                f.create_dataset(
                    "chunk_shapes", data=self.chunk_shapes, dtype="i4"
                )
                f.create_dataset("entries", data=self.overlap_entries)

        if self.initial_hamiltonian_entries is not None:
            with h5py.File(path / "hamiltonian0.h5", "w") as f:
                f.create_dataset(
                    "atom_pairs", data=self.atom_pairs, dtype="i4"
                )
                f.create_dataset(
                    "chunk_boundaries",
                    data=self.chunk_boundaries,
                    dtype="i4",
                )
                f.create_dataset(
                    "chunk_shapes", data=self.chunk_shapes, dtype="i4"
                )
                f.create_dataset(
                    "entries", data=self.initial_hamiltonian_entries
                )

    @property
    def n_pairs(self) -> int:
        return self.atom_pairs.shape[0]

    @property
    def n_atoms(self) -> int:
        return len(self.atom_symbols)

    def __repr__(self) -> str:
        extra = []
        if self.overlap_entries is not None:
            extra.append("+S")
        if self.initial_hamiltonian_entries is not None:
            extra.append("+H0")
        tag = " ".join(extra)
        return (
            f"DeepHData(n_atoms={self.n_atoms}, n_pairs={self.n_pairs}"
            + (f", {tag}" if tag else "")
            + f", species={list(self.atom_symbols)})"
        )


# -------------------------------------------------------------------
# Internal: POSCAR reader / writer
# -------------------------------------------------------------------
def _read_poscar(path: Path) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Minimal POSCAR parser (VASP4 + VASP5 element‑line formats)."""
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
