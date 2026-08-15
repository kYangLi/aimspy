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
def _reorder_flat_entries(
    entries: np.ndarray,
    src_atom_pairs: np.ndarray,
    src_cb: np.ndarray,
    dst_atom_pairs: np.ndarray,
    dst_cb: np.ndarray,
) -> np.ndarray:
    """Reorder a flat entries array from *src* pair order to *dst* pair order.

    Both orderings must describe the same set of atom-pair keys (a
    permutation); otherwise ``AimspyConfigError`` is raised.  Blocks are
    copied per atom pair according to the source chunk layout and placed
    at the destination offsets.  Per-pair block sizes are derived from the
    chunk boundaries (``src_cb``/``dst_cb``) and validated for equality.

    Works for both standard matrices (block = ``(nr, nc)``) and the
    3×-expanded ``electric_response.h5`` (block = ``(3*nr, nc)``) — the
    block sizes are implicit in each file's chunk boundaries.
    """
    src_index = {tuple(int(v) for v in row): i for i, row in enumerate(src_atom_pairs)}
    n_dst = dst_atom_pairs.shape[0]
    if len(src_index) != n_dst:
        raise AimspyConfigError(
            f"atom_pairs count mismatch: {len(src_index)} vs {n_dst}"
        )
    out = np.empty(int(dst_cb[-1]), dtype=entries.dtype)
    for k_dst in range(n_dst):
        key = tuple(int(v) for v in dst_atom_pairs[k_dst])
        k_src = src_index.get(key)
        if k_src is None:
            raise AimspyConfigError(
                f"atom_pairs are not a permutation: key {key} missing "
                f"in source file"
            )
        s0, s1 = int(src_cb[k_src]), int(src_cb[k_src + 1])
        d0, d1 = int(dst_cb[k_dst]), int(dst_cb[k_dst + 1])
        if (s1 - s0) != (d1 - d0):
            raise AimspyConfigError(
                f"chunk size mismatch for pair {key}: "
                f"src {s1 - s0} vs dst {d1 - d0}"
            )
        out[d0:d1] = entries[s0:s1]
    return out


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
    chunk_shapes: np.ndarray,
    factor: float = 1.0,
) -> np.ndarray:
    """Flatten a blocks dict to 1D entries following the existing CSR layout.

    Iterates over ``atom_pairs`` order, looks up each block by its key,
    flattens to 1D and concatenates. Missing blocks are zero-filled.
    Block shapes are validated against ``chunk_shapes``.
    """
    n_pairs = atom_pairs.shape[0]
    lst: list[np.ndarray] = []
    for ip in range(n_pairs):
        key = tuple(int(x) for x in atom_pairs[ip])
        nr = int(chunk_shapes[ip, 0])
        nc = int(chunk_shapes[ip, 1])
        blk = blocks.get(key)
        if blk is not None:
            if blk.shape != (nr, nc):
                raise AimspyConfigError(
                    f"Block {key}: shape {tuple(blk.shape)} != expected "
                    f"({nr}, {nc}) from chunk_shapes"
                )
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
    """Total number of electrons = Σ(Z).

    Raises ``AimspyConfigError`` for an unknown element symbol (a typo
    would otherwise silently contribute 0 electrons).
    """
    total = 0
    for s in atom_symbols:
        z = _ATOMIC_NUMBERS.get(s)
        if z is None:
            raise AimspyConfigError(f"Unknown element symbol: {s!r}")
        total += z
    return total


# DeepH direction order: [y, z, x] = indices [1, 2, 0] into Cartesian [x, y, z]
_DIR_DEEPH_FROM_CART = [1, 2, 0]


def _build_first_order_entries(
    blocks_list: list[dict[tuple[int, ...], np.ndarray]],
    atom_pairs: np.ndarray,
    chunk_shapes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build first-order Hamiltonian entries + chunk layout from 3 block dicts.

    *blocks_list* is a list of 3 block dicts in **Cartesian** order
    ``[x, y, z]`` (Hartree). Returns ``(entries, fo_cb, fo_cs)`` where
    entries are in eV (Hartree→eV converted) and ``fo_cs`` rows are
    3× expanded (``[3*n_rows, n_cols]`` per pair). The three directions
    are concatenated per atom pair in DeepH order ``[y, z, x]``.
    """
    n_pairs = atom_pairs.shape[0]
    # Reorder to DeepH [y, z, x]
    fo_blocks_deeph = [blocks_list[d] for d in _DIR_DEEPH_FROM_CART]
    fo_cs = np.zeros((n_pairs, 2), dtype=np.int32)
    fo_cb = np.zeros((n_pairs + 1,), dtype=np.int32)
    fo_lst: list[np.ndarray] = []
    for ip in range(n_pairs):
        nr = int(chunk_shapes[ip, 0])
        nc = int(chunk_shapes[ip, 1])
        key = tuple(int(x) for x in atom_pairs[ip])
        blocks_deeph = []
        for d in range(3):
            blk = fo_blocks_deeph[d].get(key)
            if blk is None:
                blocks_deeph.append(np.zeros((nr, nc), dtype=np.float64))
            else:
                blocks_deeph.append(np.ascontiguousarray(blk, dtype=np.float64))
        # Concatenate [Hy, Hz, Hx] along rows → (3*nr, nc)
        concat = np.concatenate(blocks_deeph, axis=0)
        fo_lst.append(concat.ravel())
        fo_cs[ip, 0] = 3 * nr
        fo_cs[ip, 1] = nc
        fo_cb[ip + 1] = fo_cb[ip] + 3 * nr * nc
    entries = np.concatenate(fo_lst)
    entries *= HARTREE_TO_EV  # Hartree → eV
    return entries, fo_cb, fo_cs


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
      - ``force.h5``        — *optional* — MD-style: cell, energy, force,
        stress datasets (energy in eV, force in eV/Å, stress as zeros)

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

    # Electric-response first-order Hamiltonian (dH/de) entries (eV).
    # DeepH ``electric_response.h5`` layout: 3 stacked Hamiltonians per
    # atom pair — one per Cartesian direction (y, z, x = m=-1, 0, +1).
    # ``_fo_chunk_*`` describe the 3×-expanded layout (kept in sync with
    # the standard ``chunk_*`` via :meth:`set_first_order_hamiltonian` /
    # :meth:`from_directory`); they are ``None`` when no first-order data
    # is present.
    first_order_hamiltonian_entries: Optional[np.ndarray] = None
    _fo_chunk_boundaries: Optional[np.ndarray] = None
    _fo_chunk_shapes: Optional[np.ndarray] = None

    # force / energy (optional, for MD-style force.h5 export)
    force: Optional[np.ndarray] = None  # (n_atoms, 3) float64, eV/Å, POSCAR order
    energy_eV: Optional[float] = None  # scalar, eV

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
        Optionally reads ``force.h5`` (MD-style format: cell/energy/force/stress)
        if present.  Sets ``self.path = path`` for subsequent ``save_*()`` calls.
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
        if info.get("spinful", False):
            raise AimspyConfigError(
                "spin-polarized (spinful) DeepH data is not yet supported "
                "by the aimspy adapter (n_spin=1 only)"
            )
        eom = info.get("elements_orbital_map", {})
        n_basis = info.get("orbits_quantity", 0)
        if n_basis == 0:
            n_basis = _compute_n_basis(atom_symbols, eom)
        fermi_eV = info.get("fermi_energy_eV", 0.0)

        # Read CSR layout from the first found file (canonical layout)
        first_name, first_path = found[0]
        with h5py.File(first_path, "r") as f:
            atom_pairs = f["atom_pairs"][:].astype(np.int32)
            cb = f["chunk_boundaries"][:].astype(np.int32)
            cs = f["chunk_shapes"][:].astype(np.int32)

        # Read each matrix; validate / reorder atom_pairs against the
        # canonical layout (entries are reordered per-pair if the file
        # uses a different but equivalent pair ordering).
        entries = None
        overlap_entries = None
        init_entries = None
        for name, p in found:
            with h5py.File(p, "r") as f:
                data = f["entries"][:].astype(np.float64)
                if p != first_path:
                    ap_check = f["atom_pairs"][:].astype(np.int32)
                    if not np.array_equal(ap_check, atom_pairs):
                        cb_check = f["chunk_boundaries"][:].astype(np.int32)
                        data = _reorder_flat_entries(
                            data,
                            ap_check,
                            cb_check,
                            atom_pairs,
                            cb,
                        )
            if name == "hamiltonian":
                entries = data
            elif name == "overlap":
                overlap_entries = data
            elif name == "initial_hamiltonian":
                init_entries = data

        # Check for optional force.h5 (different format from matrix .h5)
        force_path = path / "force.h5"
        force_arr = None
        energy_val = None
        if force_path.is_file():
            with h5py.File(force_path, "r") as f:
                force_arr = f["force"][:].astype(np.float64)
                if "energy" in f:
                    energy_val = float(f["energy"][()])
            expected_shape = (len(atom_symbols), 3)
            if force_arr.shape != expected_shape:
                raise AimspyConfigError(
                    f"force.h5: force shape {force_arr.shape} doesn't match "
                    f"expected {expected_shape}"
                )

        # Check for optional electric_response.h5 (dH/de, same atom_pairs as
        # hamiltonian.h5 but chunk_shapes/boundaries expanded 3× per pair).
        fo_path = path / "electric_response.h5"
        fo_entries = None
        fo_cb = None
        fo_cs = None
        if fo_path.is_file():
            with h5py.File(fo_path, "r") as f:
                fo_entries = f["entries"][:].astype(np.float64)
                fo_cb = f["chunk_boundaries"][:].astype(np.int32)
                fo_cs = f["chunk_shapes"][:].astype(np.int32)
                fo_ap = f["atom_pairs"][:].astype(np.int32)
            # Internal consistency + cross-file atom_pairs validation
            if np.any(fo_cs[:, 0] % 3 != 0):
                raise AimspyConfigError(
                    "electric_response.h5: chunk_shapes[:, 0] not all "
                    "divisible by 3 (corrupted file?)"
                )
            if int(fo_cb[-1]) != int(fo_entries.shape[0]):
                raise AimspyConfigError(
                    f"electric_response.h5: chunk_boundaries[-1]={int(fo_cb[-1])} "
                    f"!= len(entries)={int(fo_entries.shape[0])}"
                )
            if not np.array_equal(fo_ap, atom_pairs):
                # Reorder entries into the canonical atom_pairs order.  The
                # destination chunk layout must be the 3×-expanded one (each
                # first-order block is (3*nr, nc)), so rebuild it from the
                # canonical standard chunk_shapes first, then use it as dst.
                n_pairs = atom_pairs.shape[0]
                fo_cs_new = np.zeros((n_pairs, 2), dtype=np.int32)
                fo_cb_new = np.zeros((n_pairs + 1,), dtype=np.int32)
                for ip in range(n_pairs):
                    nr = int(cs[ip, 0])
                    nc = int(cs[ip, 1])
                    fo_cs_new[ip, 0] = 3 * nr
                    fo_cs_new[ip, 1] = nc
                    fo_cb_new[ip + 1] = fo_cb_new[ip] + 3 * nr * nc
                fo_entries = _reorder_flat_entries(
                    fo_entries,
                    fo_ap,
                    fo_cb,
                    atom_pairs,
                    fo_cb_new,
                )
                fo_cs = fo_cs_new
                fo_cb = fo_cb_new

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
            first_order_hamiltonian_entries=fo_entries,
            _fo_chunk_boundaries=fo_cb,
            _fo_chunk_shapes=fo_cs,
            force=force_arr,
            energy_eV=energy_val,
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
        force: Optional[np.ndarray] = None,
        energy_eV: Optional[float] = None,
        first_order_hamiltonian_blocks: Optional[
            list[dict[tuple[int, ...], np.ndarray]]
        ] = None,
        path: Optional[Union[str, Path]] = None,
    ) -> "DeepHData":
        """Build from in-memory pair-block dicts.

        All matrix blocks are optional — at least one must be given.
        Keys are ``(R1,R2,R3,i,j)`` with atoms in POSCAR order.
        Hamiltonian / initial_hamiltonian blocks in **Hartree**
        (converted to eV here). Overlap blocks are dimensionless.

        *force* and *energy_eV* are optional per-atom / scalar data for
        MD-style ``force.h5`` export. *force* is ``(n_atoms, 3)`` in
        eV/Å, already in **POSCAR atom order** (matching *atom_coords*).
        *energy_eV* is a scalar in eV.

        *first_order_hamiltonian_blocks* is an optional list of 3 block
        dicts ``[x, y, z]`` in **Hartree** (converted to eV here). The
        three directions are concatenated per atom pair in DeepH order
        ``[y, z, x]`` (= real spherical harmonics ``m = -1, 0, +1``) and
        stored in :attr:`first_order_hamiltonian_entries`.
        """
        if n_basis == 0:
            n_basis = _compute_n_basis(atom_symbols, elements_orbital_map)

        # The first-order Hamiltonian (dH/de) is only physically meaningful
        # when all three Cartesian directions are present; require all three
        # block dicts to be non-empty if provided at all.
        if first_order_hamiltonian_blocks is not None:
            if (
                not isinstance(first_order_hamiltonian_blocks, (list, tuple))
                or len(first_order_hamiltonian_blocks) != 3
            ):
                raise AimspyConfigError(
                    "first_order_hamiltonian_blocks must be a list of 3 dicts "
                    "[x, y, z]"
                )
            if not all(first_order_hamiltonian_blocks[d] for d in range(3)):
                raise AimspyConfigError(
                    "first_order_hamiltonian_blocks: all three directions "
                    "[x, y, z] must be non-empty; dH/de data is only "
                    "meaningful when all three directions are present"
                )

        layout_blocks = (
            hamiltonian_blocks
            or overlap_blocks
            or initial_hamiltonian_blocks
            or (
                first_order_hamiltonian_blocks[0]
                if first_order_hamiltonian_blocks is not None
                else None
            )
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

        # Validate force shape if provided
        if force is not None:
            force = np.asarray(force, dtype=np.float64)
            expected_shape = (len(atom_symbols), 3)
            if force.shape != expected_shape:
                raise AimspyConfigError(
                    f"force shape {force.shape} doesn't match "
                    f"expected {expected_shape}"
                )

        # Optional first-order Hamiltonian (dH/de) — 3 block dicts [x, y, z].
        # Stored in DeepH order [y, z, x] (= m = -1, 0, +1) per atom pair.
        fo_entries = None
        fo_cb = None
        fo_cs = None
        if first_order_hamiltonian_blocks is not None:
            # (list-of-3 + all-non-empty validation done above)
            fo_entries, fo_cb, fo_cs = _build_first_order_entries(
                first_order_hamiltonian_blocks,
                atom_pairs,
                chunk_shapes,
            )

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
            first_order_hamiltonian_entries=fo_entries,
            _fo_chunk_boundaries=fo_cb,
            _fo_chunk_shapes=fo_cs,
            force=force,
            energy_eV=energy_eV,
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
        force: Optional[np.ndarray] = None,
        energy: Optional[float] = None,
        first_order_hamiltonian: Optional[list] = None,
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
        force : np.ndarray, optional
            Forces ``(n_atoms, 3)`` in eV/Å, **aims atom order**.
            Reordered to POSCAR order inside.
        energy : float, optional
            Total energy in **Hartree** (converted to eV inside).
        first_order_hamiltonian : list[AimspyMatrix], optional
            Electric-response first-order Hamiltonian ``dH/de`` — a list
            of 3 ``AimspyMatrix`` in Cartesian order ``[x, y, z]``
            (Hartree, aims atom order). Reordered to POSCAR order and
            concatenated per atom pair in DeepH order ``[y, z, x]``.

        .. note::

            *force*, *energy* and *first_order_hamiltonian* are
            keyword-only (placed after *path*) to preserve
            backward-compatible positional ordering of *template*.
        """
        if (
            hamiltonian is None
            and overlap is None
            and initial_hamiltonian is None
            and first_order_hamiltonian is None
        ):
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

        # First-order Hamiltonian: list of 3 AimspyMatrix [x, y, z].
        first_order_blocks_list = None
        if first_order_hamiltonian is not None:
            if (
                not isinstance(first_order_hamiltonian, (list, tuple))
                or len(first_order_hamiltonian) != 3
            ):
                raise AimspyConfigError(
                    "first_order_hamiltonian must be a list of 3 AimspyMatrix "
                    "[x, y, z]"
                )
            first_order_blocks_list = [
                _aimspy_blocks_to_poscar(mx, structure)
                for mx in first_order_hamiltonian
            ]

        # Force: reorder aims → POSCAR (force is per-atom, not block dict)
        force_poscar = None
        if force is not None:
            _, new2old = structure.build_atom_permutation()
            force_arr = np.asarray(force, dtype=np.float64)
            if force_arr.shape != (structure.n_atoms, 3):
                raise AimspyConfigError(
                    f"force shape {force_arr.shape} doesn't match "
                    f"expected ({structure.n_atoms}, 3)"
                )
            force_poscar = np.ascontiguousarray(force_arr[new2old])

        # Energy: Hartree → eV
        energy_eV_val = float(energy) * HARTREE_TO_EV if energy is not None else None

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
            force=force_poscar,
            energy_eV=energy_eV_val,
            first_order_hamiltonian_blocks=first_order_blocks_list,
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
            self.chunk_shapes,
            factor=HARTREE_TO_EV,
        )

    def set_overlap(self, matrix: "AimspyMatrix", structure: "AimspyStructure") -> None:
        """Convert and store overlap entries (dimensionless) from *matrix*."""
        blocks = _aimspy_blocks_to_poscar(matrix, structure)
        self.overlap_entries = _blocks_to_flat_entries(
            blocks,
            self.atom_pairs,
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
            self.chunk_shapes,
            factor=HARTREE_TO_EV,
        )

    def set_force(
        self,
        force_aims: np.ndarray,
        structure: "AimspyStructure",
        energy: Optional[float] = None,
    ) -> None:
        """Store force (eV/Å) and optionally energy (Hartree→eV) from aims order.

        Parameters
        ----------
        force_aims : np.ndarray or list
            Forces ``(n_atoms, 3)`` in eV/Å, **aims atom order**.
            Reordered to POSCAR order inside.  Accepts list or ndarray.
        structure : AimspyStructure
            Provides the aims→POSCAR atom permutation.
        energy : float, optional
            Total energy in **Hartree** (converted to eV), or None.
        """
        force_arr = np.asarray(force_aims, dtype=np.float64)
        if force_arr.shape != (structure.n_atoms, 3):
            raise AimspyConfigError(
                f"force shape {force_arr.shape} doesn't match "
                f"expected ({structure.n_atoms}, 3)"
            )
        _, new2old = structure.build_atom_permutation()
        self.force = np.ascontiguousarray(force_arr[new2old])
        if energy is not None:
            self.energy_eV = float(energy) * HARTREE_TO_EV

    def set_first_order_hamiltonian(
        self,
        matrix_list: list,
        structure: "AimspyStructure",
    ) -> None:
        """Store electric-response first-order Hamiltonian (dH/de) entries.

        Converts 3 ``AimspyMatrix`` instances (Hartree, aims atom order)
        into DeepH ``electric_response.h5`` entries (eV, POSCAR order).
        The three Cartesian directions ``[x, y, z]`` are concatenated
        per atom pair in DeepH order ``[y, z, x]`` (= real spherical
        harmonics ``m = -1, 0, +1``), matching ``ref/aims_to_deeph.py``.

        Parameters
        ----------
        matrix_list : list[AimspyMatrix]
            Exactly 3 ``AimspyMatrix`` in Cartesian order ``[x, y, z]``.
        structure : AimspyStructure
            Provides the aims→POSCAR atom permutation.
        """
        if not isinstance(matrix_list, (list, tuple)) or len(matrix_list) != 3:
            raise AimspyConfigError(
                "matrix_list must contain exactly 3 AimspyMatrix [x, y, z]"
            )
        blocks_list = [_aimspy_blocks_to_poscar(mx, structure) for mx in matrix_list]
        if not all(blocks_list[d] for d in range(3)):
            raise AimspyConfigError(
                "first_order_hamiltonian: all three directions [x, y, z] must "
                "produce non-empty blocks; dH/de data is only meaningful when "
                "all three directions are present"
            )
        entries, fo_cb, fo_cs = _build_first_order_entries(
            blocks_list,
            self.atom_pairs,
            self.chunk_shapes,
        )
        self.first_order_hamiltonian_entries = entries
        self._fo_chunk_boundaries = fo_cb
        self._fo_chunk_shapes = fo_cs

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

    def _write_force_h5(self, file_path: Path) -> None:
        """Write force.h5 in DeepH MD convention.

        Datasets: ``cell`` (3,3), ``energy`` scalar, ``force`` (n_atoms,3),
        ``stress`` (6,) zeros placeholder.
        Root attrs: ``formula`` = ``b'X{natoms}'``, ``natoms`` = int64.

        ``energy`` is written as 0.0 if ``energy_eV`` is None.
        """
        n_atoms = self.n_atoms
        energy_val = float(self.energy_eV) if self.energy_eV is not None else 0.0
        with h5py.File(file_path, "w") as f:
            f.create_dataset("cell", data=self.lattice, dtype="f8")
            f.create_dataset("energy", data=energy_val)
            f.create_dataset("force", data=self.force, dtype="f8")
            f.create_dataset("stress", data=np.zeros(6, dtype=np.float64))
            f.attrs["formula"] = np.bytes_(f"X{n_atoms}".encode("utf-8"))
            f.attrs["natoms"] = np.int64(n_atoms)

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

    def save_force(self, path: Optional[Union[str, Path]] = None) -> None:
        """Write force.h5 (requires force to be set).

        Energy is written if ``energy_eV`` is set, else 0.0.
        Stress is always written as zeros (placeholder).
        """
        if self.force is None:
            raise AimspyConfigError("No force data to save")
        p = Path(path) if path is not None else self._require_path()
        p.mkdir(parents=True, exist_ok=True)
        self._write_force_h5(p / "force.h5")

    def save_first_order_hamiltonian(
        self, path: Optional[Union[str, Path]] = None
    ) -> None:
        """Write ``electric_response.h5`` (requires first_order entries set).

        Layout: same ``atom_pairs`` as ``hamiltonian.h5``, but
        ``chunk_shapes`` rows are 3× (one block per Cartesian direction
        ``[y, z, x]``) and ``entries`` is 3× longer.
        """
        if self.first_order_hamiltonian_entries is None:
            raise AimspyConfigError("No first_order_hamiltonian entries to save")
        if self._fo_chunk_boundaries is None or self._fo_chunk_shapes is None:
            raise AimspyConfigError(
                "first_order chunk layout (_fo_chunk_boundaries/_fo_chunk_shapes) "
                "is not set; use set_first_order_hamiltonian, from_memory, or "
                "from_directory to establish it before save_first_order_hamiltonian"
            )
        p = Path(path) if path is not None else self._require_path()
        p.mkdir(parents=True, exist_ok=True)
        with h5py.File(p / "electric_response.h5", "w") as f:
            f.create_dataset("atom_pairs", data=self.atom_pairs, dtype="i4")
            f.create_dataset(
                "chunk_boundaries", data=self._fo_chunk_boundaries, dtype="i4"
            )
            f.create_dataset("chunk_shapes", data=self._fo_chunk_shapes, dtype="i4")
            f.create_dataset("entries", data=self.first_order_hamiltonian_entries)

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
        if self.first_order_hamiltonian_entries is not None:
            self.save_first_order_hamiltonian(p)
        if self.force is not None:
            self.save_force(p)

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
        if self.first_order_hamiltonian_entries is not None:
            extra.append("+dHde")
        if self.force is not None:
            extra.append("+F")
        tag = " ".join(extra)
        # Summarize species as element -> count (a full per-atom list is
        # unreadable for large supercells).
        species_summary = dict(Counter(self.atom_symbols))
        return (
            f"DeepHData(n_atoms={self.n_atoms}, n_pairs={self.n_pairs}"
            + (f", {tag}" if tag else "")
            + f", species={species_summary})"
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

        .. note::

            Only the Hamiltonian is converted.  Force and energy (if
            loaded from ``force.h5``) are accessible directly via the
            ``self.force`` and ``self.energy_eV`` attributes — they do
            **not** participate in the warmstart injection path.

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

    def to_first_order_aimspy(self, structure: "AimspyStructure") -> list:
        """Convert this DeepH data's first-order Hamiltonian entries to
        aimspy standard format.

        Returns a list of 3 ``AimspyMatrix`` in Cartesian order
        ``[x, y, z]`` (Hartree, aims atom order), suitable for passing
        to :meth:`aimspy.Calculator.modify_init_first_order_ham` via
        ``source=``.

        - Atom reordering: POSCAR → aims (via stable-sort un-permutation)
        - R: no flip (same convention: ``R_aimspy = R_deeph = -R_aims``)
        - Parity: no change (same wiki convention)
        - Units: eV → Hartree
        - Direction order: DeepH ``[y, z, x]`` → ``[x, y, z]``

        Parameters
        ----------
        structure : AimspyStructure
            Live runtime structure (built from ``AimspyInfo`` after
            ``aimspy_init``); provides the POSCAR↔aims atom permutation.
        """
        from ...matrix import AimspyMatrix

        if self.first_order_hamiltonian_entries is None:
            raise AimspyConfigError(
                "No first_order_hamiltonian entries to convert; "
                "set first_order_hamiltonian_entries first"
            )
        if self._fo_chunk_boundaries is None or self._fo_chunk_shapes is None:
            raise AimspyConfigError(
                "first_order chunk layout (_fo_chunk_boundaries/_fo_chunk_shapes) "
                "is not set; use set_first_order_hamiltonian, from_memory, "
                "or from_directory to establish the layout"
            )
        _, new2old = structure.build_atom_permutation()
        # new2old[POSCAR_atom] = aims_atom

        entries = self.first_order_hamiltonian_entries
        fo_cb = self._fo_chunk_boundaries
        fo_cs = self._fo_chunk_shapes

        # Three blocks per atom pair, in DeepH order [y, z, x].
        # Cartesian order [x, y, z] = indices [2, 0, 1] into DeepH order.
        cart_idx = [2, 0, 1]
        blocks_list: list[dict] = [{}, {}, {}]

        for ip in range(self.n_pairs):
            R1 = int(self.atom_pairs[ip, 0])
            R2 = int(self.atom_pairs[ip, 1])
            R3 = int(self.atom_pairs[ip, 2])
            i_deeph = int(self.atom_pairs[ip, 3])
            j_deeph = int(self.atom_pairs[ip, 4])

            i_aims = int(new2old[i_deeph])
            j_aims = int(new2old[j_deeph])

            bnd = int(fo_cb[ip])
            nr3 = int(fo_cs[ip, 0])  # 3 * n_rows
            nc = int(fo_cs[ip, 1])
            if nr3 % 3 != 0:
                raise AimspyConfigError(
                    f"first_order chunk_shapes[{ip}, 0] = {nr3} is not "
                    f"divisible by 3 (corrupted electric_response.h5?)"
                )
            nr = nr3 // 3

            chunk = entries[bnd : bnd + nr3 * nc].reshape(nr3, nc).copy()
            # Split into 3 (nr, nc) blocks: [Hy, Hz, Hx]
            sub_blocks = [chunk[d * nr : (d + 1) * nr, :].copy() for d in range(3)]
            # Apply unit conversion eV → Hartree
            for d in range(3):
                sub_blocks[d] *= EV_TO_HARTREE

            key = (R1, R2, R3, i_aims, j_aims)
            for cart in range(3):
                d_deeph = cart_idx[cart]
                blocks_list[cart][key] = sub_blocks[d_deeph]

        return [AimspyMatrix(blocks=blocks_list[cart], n_spin=1) for cart in range(3)]


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
