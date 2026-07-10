"""DeepH-format data: reader for POSCAR + info.json + hamiltonian.h5.

Provides ``DeepHData``, a complete in-memory representation of DeepH
data including structure info (from POSCAR / info.json) and matrix
data (from hamiltonian.h5).  Supports both file and in-memory
construction.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np

from ...data import HARTREE_TO_EV, EV_TO_HARTREE

_MAGIC_LARGE = 1e9


@dataclass
class DeepHData:
    """Complete DeepH-format data: structure + matrix.

    Read from a directory containing:
      - ``POSCAR``         — lattice, atom symbols, atom coords
      - ``info.json``      — ``elements_orbital_map``
      - ``hamiltonian.h5`` — atom_pairs, chunk_*, entries (eV)

    Can also be constructed in-memory via ``from_memory``.
    """

    # structure (POSCAR order = element‑grouped)
    lattice: np.ndarray                      # (3, 3) in Angstrom
    atom_symbols: List[str]                  # POSCAR order
    atom_coords: np.ndarray                  # (n_atoms, 3) in Angstrom
    elements_orbital_map: Dict[str, List[int]]

    # matrix (DeepH flat block-CSR, eV)
    atom_pairs: np.ndarray                  # (N, 5)  [R1,R2,R3,i,j]
    chunk_boundaries: np.ndarray            # (N+1,)
    chunk_shapes: np.ndarray                # (N, 2)
    entries: np.ndarray                     # (M,) float64, eV

    # ----------------------------------------------------------------
    # Construction from directory
    # ----------------------------------------------------------------
    @classmethod
    def from_directory(cls, path: Union[str, Path]) -> "DeepHData":
        """Read POSCAR + info.json + hamiltonian.h5 from *path*."""
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
        with open(info_path, 'r') as f:
            info = json.load(f)
        eom = info.get('elements_orbital_map', {})

        with h5py.File(h5_path, 'r') as f:
            atom_pairs = f['atom_pairs'][:].astype(np.int32)
            cb = f['chunk_boundaries'][:].astype(np.int32)
            cs = f['chunk_shapes'][:].astype(np.int32)
            entries = f['entries'][:].astype(np.float64)

        return cls(
            lattice=lattice,
            atom_symbols=atom_symbols,
            atom_coords=atom_coords,
            elements_orbital_map=eom,
            atom_pairs=atom_pairs,
            chunk_boundaries=cb,
            chunk_shapes=cs,
            entries=entries,
        )

    @classmethod
    def from_memory(
        cls,
        lattice: np.ndarray,
        atom_symbols: List[str],
        atom_coords: np.ndarray,
        elements_orbital_map: Dict[str, List[int]],
        pair_blocks: Dict[Tuple[int, ...], np.ndarray],
    ) -> "DeepHData":
        """Build from in-memory pair-block dict.

        ``pair_blocks`` keys are ``(R1,R2,R3,i,j)`` with atoms in
        POSCAR order.
        """
        n_pairs = len(pair_blocks)
        atom_pairs = np.zeros((n_pairs, 5), dtype=np.int32)
        chunk_boundaries = np.zeros((n_pairs + 1,), dtype=np.int32)
        chunk_shapes = np.zeros((n_pairs, 2), dtype=np.int32)
        entries_lst: list[np.ndarray] = []

        for ip, (key, block) in enumerate(sorted(pair_blocks.items())):
            atom_pairs[ip] = [int(k) for k in key]
            nr, nc = int(block.shape[0]), int(block.shape[1])
            chunk_shapes[ip] = (nr, nc)
            entries_lst.append(np.ascontiguousarray(block, dtype=np.float64).ravel())
            chunk_boundaries[ip + 1] = chunk_boundaries[ip] + nr * nc

        entries = np.concatenate(entries_lst) if entries_lst \
                  else np.array([], dtype=np.float64)
        entries *= HARTREE_TO_EV  # Hartree -> eV for DeepH format

        return cls(
            lattice=np.asarray(lattice, dtype=np.float64),
            atom_symbols=list(atom_symbols),
            atom_coords=np.asarray(atom_coords, dtype=np.float64),
            elements_orbital_map=dict(elements_orbital_map),
            atom_pairs=atom_pairs,
            chunk_boundaries=chunk_boundaries,
            chunk_shapes=chunk_shapes,
            entries=entries,
        )

    # ----------------------------------------------------------------
    # I/O
    # ----------------------------------------------------------------
    def save(self, path: Union[str, Path]) -> None:
        """Write POSCAR + info.json + hamiltonian.h5 to *path*."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        _write_poscar(path / "POSCAR", self.lattice, self.atom_symbols,
                       self.atom_coords)
        _write_info_json(path / "info.json", self)
        with h5py.File(path / "hamiltonian.h5", 'w') as f:
            f.create_dataset('atom_pairs', data=self.atom_pairs, dtype='i4')
            f.create_dataset('chunk_boundaries', data=self.chunk_boundaries, dtype='i4')
            f.create_dataset('chunk_shapes', data=self.chunk_shapes, dtype='i4')
            f.create_dataset('entries', data=self.entries)

    @property
    def n_pairs(self) -> int:
        return self.atom_pairs.shape[0]

    @property
    def n_atoms(self) -> int:
        return len(self.atom_symbols)

    def __repr__(self) -> str:
        return (f"DeepHData(n_atoms={self.n_atoms}, n_pairs={self.n_pairs}, "
                f"species={list(self.atom_symbols)})")


# -------------------------------------------------------------------
# Internal: POSCAR reader / writer
# -------------------------------------------------------------------
def _read_poscar(path: Path) -> tuple[np.ndarray, List[str], np.ndarray]:
    """Minimal POSCAR parser (VASP4 + VASP5 element‑line formats)."""
    lines = path.read_text().splitlines()
    lines = [ln.strip() for ln in lines if ln.strip()]

    scale = float(lines[1])
    lat = scale * np.array([[float(x) for x in lines[i].split()]
                            for i in range(2, 5)], dtype=np.float64)

    # detect VASP5 (element line)
    tokens6 = lines[5].split()
    try:
        [float(x) for x in tokens6]            # purely numeric = VASP4
        have_element_line = False
    except ValueError:
        have_element_line = True

    if have_element_line:
        symbols_on_line = lines[5].split()
        counts = [int(x) for x in lines[6].split()]
        coord_start = 7   # line 7 may be coord-type or coordinate
    else:
        counts = [int(x) for x in lines[5].split()]
        symbols_on_line = lines[0].split()
        coord_start = 6   # line 6 may be coord-type or coordinate

    # Skip optional "Selective dynamics" and mandatory coordinate-type line
    while coord_start < len(lines):
        token = lines[coord_start].split()[0].lower()
        if token in ('cartesian', 'direct', 'selective',
                      'kartesian', 'd'):
            coord_start += 1
        else:
            break

    # expand symbols
    atom_symbols: list[str] = []
    total_atoms = sum(counts)
    n_uniq = min(len(symbols_on_line), len(counts))
    for i in range(n_uniq):
        atom_symbols.extend([symbols_on_line[i]] * counts[i])
    # if we didn't get enough symbols (VASP4 with title-only), fill with generic
    if len(atom_symbols) < total_atoms:
        missing = total_atoms - len(atom_symbols)
        atom_symbols.extend([f"X{i+1}" for i in range(len(atom_symbols),
                                                       len(atom_symbols) + missing)])

    n_atoms = len(atom_symbols)
    coords = np.zeros((n_atoms, 3), dtype=np.float64)
    for i in range(n_atoms):
        coords[i] = [float(x) for x in lines[coord_start + i].split()[:3]]

    # handle Direct coords — check the coord-type line we just skipped
    coord_type = lines[coord_start - 1].split()[0].lower()
    # if the last skipped line was "Selective dynamics", look one line earlier
    if coord_type == 'selective':
        coord_type = lines[coord_start - 2].split()[0].lower()
    if coord_type.startswith('d'):
        coords = coords @ lat

    return lat, atom_symbols, coords


def _write_poscar(path: Path, lattice: np.ndarray,
                  atom_symbols: List[str], atom_coords: np.ndarray) -> None:
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
    from json import dumps
    obj = {
        "atoms_quantity": data.n_atoms,
        "orbits_quantity": int(np.sum(
            sum(len(ls) for ls in data.elements_orbital_map.values()))),
        "orthogonal_basis": False,
        "spinful": False,
        "fermi_energy_eV": 0.0,
        "elements_orbital_map": data.elements_orbital_map,
    }
    path.write_text(dumps(obj))
