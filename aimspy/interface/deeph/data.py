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
@dataclass(frozen=True)
class _MatrixLayout:
    """Validated on-disk matrix layout and entries."""

    atom_pairs: np.ndarray
    chunk_boundaries: np.ndarray
    chunk_shapes: np.ndarray
    entries: np.ndarray


_MATRIX_DATASETS = ("atom_pairs", "chunk_boundaries", "chunk_shapes", "entries")


def _layout_error(
    source: Union[str, Path], field: str, detail: str
) -> AimspyConfigError:
    return AimspyConfigError(f"{Path(source).name}: {field}: {detail}")


def _hdf5_dataset_value(h5, name, source, error_builder):
    """Read one HDF5 dataset after rejecting groups and other objects."""
    try:
        obj = h5[name]
    except (KeyError, OSError, RuntimeError) as exc:
        raise error_builder(
            source,
            name,
            "could not resolve the HDF5 object",
        ) from exc
    if not isinstance(obj, h5py.Dataset):
        raise error_builder(
            source,
            name,
            f"expected an HDF5 dataset, got {type(obj).__name__}",
        )
    try:
        return obj[()]
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise error_builder(source, name, "could not read the HDF5 dataset") from exc


def _orbital_counts_per_atom(
    atom_symbols: list[str],
    elements_orbital_map: dict[str, list[int]],
    source: Union[str, Path],
) -> np.ndarray:
    counts = np.zeros(len(atom_symbols), dtype=np.int64)
    for i, symbol in enumerate(atom_symbols):
        if symbol not in elements_orbital_map:
            raise _layout_error(
                source,
                "chunk_shapes",
                f"element {symbol!r} is missing from elements_orbital_map",
            )
        try:
            counts[i] = sum(
                2 * int(l_value) + 1 for l_value in elements_orbital_map[symbol]
            )
        except (TypeError, ValueError) as exc:
            raise _layout_error(
                source,
                "chunk_shapes",
                f"invalid orbital map for element {symbol!r}",
            ) from exc
        if counts[i] <= 0:
            raise _layout_error(
                source,
                "chunk_shapes",
                f"element {symbol!r} has no orbitals",
            )
    return counts


def _validate_matrix_layout(
    source: Union[str, Path],
    atom_pairs: np.ndarray,
    chunk_boundaries: np.ndarray,
    chunk_shapes: np.ndarray,
    entries: np.ndarray,
    atom_symbols: list[str],
    elements_orbital_map: dict[str, list[int]],
    *,
    row_multiplier: int = 1,
) -> _MatrixLayout:
    """Validate one standard or first-order DeepH matrix layout."""
    ap = np.asarray(atom_pairs)
    cb = np.asarray(chunk_boundaries)
    cs = np.asarray(chunk_shapes)
    data = np.asarray(entries)

    for field, arr in (
        ("atom_pairs", ap),
        ("chunk_boundaries", cb),
        ("chunk_shapes", cs),
    ):
        if not np.issubdtype(arr.dtype, np.integer):
            raise _layout_error(
                source, field, f"expected integer dtype, got {arr.dtype}"
            )
        if arr.size:
            limits = np.iinfo(np.int32)
            if int(arr.min()) < limits.min or int(arr.max()) > limits.max:
                raise _layout_error(source, field, "values exceed int32 range")

    if not (
        np.issubdtype(data.dtype, np.integer) or np.issubdtype(data.dtype, np.floating)
    ):
        raise _layout_error(
            source, "entries", f"expected real numeric dtype, got {data.dtype}"
        )
    if data.ndim != 1:
        raise _layout_error(source, "entries", f"expected shape (M,), got {data.shape}")

    if ap.ndim != 2 or ap.shape[1:] != (5,) or ap.shape[0] == 0:
        raise _layout_error(
            source,
            "atom_pairs",
            f"expected non-empty shape (N, 5), got {ap.shape}",
        )
    n_pairs = ap.shape[0]
    pair_keys = [tuple(int(value) for value in row) for row in ap]
    if len(set(pair_keys)) != n_pairs:
        raise _layout_error(source, "atom_pairs", "contains duplicate pair keys")

    atom_indices = ap[:, 3:5].astype(np.int64, copy=False)
    n_atoms = len(atom_symbols)
    if np.any(atom_indices < 0) or np.any(atom_indices >= n_atoms):
        raise _layout_error(
            source,
            "atom_pairs",
            f"atom indices must be in [0, {n_atoms}), got range "
            f"[{int(atom_indices.min())}, {int(atom_indices.max())}]",
        )

    if cs.shape != (n_pairs, 2):
        raise _layout_error(
            source,
            "chunk_shapes",
            f"expected shape ({n_pairs}, 2), got {cs.shape}",
        )
    cs64 = cs.astype(np.int64, copy=False)
    if np.any(cs64 <= 0):
        raise _layout_error(
            source, "chunk_shapes", "all block dimensions must be positive"
        )

    orbital_counts = _orbital_counts_per_atom(
        atom_symbols, elements_orbital_map, source
    )
    expected_shapes = np.column_stack(
        (
            orbital_counts[atom_indices[:, 0]] * int(row_multiplier),
            orbital_counts[atom_indices[:, 1]],
        )
    )
    mismatch = np.flatnonzero(np.any(cs64 != expected_shapes, axis=1))
    if mismatch.size:
        ip = int(mismatch[0])
        raise _layout_error(
            source,
            "chunk_shapes",
            f"pair {pair_keys[ip]} has shape {tuple(int(v) for v in cs64[ip])}; "
            f"expected {tuple(int(v) for v in expected_shapes[ip])}",
        )

    if cb.shape != (n_pairs + 1,):
        raise _layout_error(
            source,
            "chunk_boundaries",
            f"expected shape ({n_pairs + 1},), got {cb.shape}",
        )
    cb64 = cb.astype(np.int64, copy=False)
    if int(cb64[0]) != 0:
        raise _layout_error(
            source,
            "chunk_boundaries",
            f"first value must be 0, got {int(cb64[0])}",
        )
    deltas = np.diff(cb64)
    if np.any(deltas < 0):
        raise _layout_error(source, "chunk_boundaries", "values must be non-decreasing")
    expected_sizes = cs64[:, 0] * cs64[:, 1]
    mismatch = np.flatnonzero(deltas != expected_sizes)
    if mismatch.size:
        ip = int(mismatch[0])
        raise _layout_error(
            source,
            "chunk_boundaries",
            f"pair {pair_keys[ip]} span is {int(deltas[ip])}; "
            f"expected {int(expected_sizes[ip])} from chunk_shapes",
        )
    if int(cb64[-1]) != int(data.shape[0]):
        raise _layout_error(
            source,
            "entries",
            f"length {data.shape[0]} does not match final boundary {int(cb64[-1])}",
        )

    return _MatrixLayout(
        atom_pairs=np.ascontiguousarray(ap, dtype=np.int32),
        chunk_boundaries=np.ascontiguousarray(cb, dtype=np.int32),
        chunk_shapes=np.ascontiguousarray(cs, dtype=np.int32),
        entries=np.ascontiguousarray(data, dtype=np.float64),
    )


def _read_matrix_layout(
    file_path: Path,
    atom_symbols: list[str],
    elements_orbital_map: dict[str, list[int]],
    *,
    row_multiplier: int = 1,
) -> _MatrixLayout:
    with h5py.File(file_path, "r") as h5:
        missing = [name for name in _MATRIX_DATASETS if name not in h5]
        if missing:
            raise _layout_error(
                file_path,
                missing[0],
                f"required dataset is missing (missing: {', '.join(missing)})",
            )
        arrays = {
            name: _hdf5_dataset_value(h5, name, file_path, _layout_error)
            for name in _MATRIX_DATASETS
        }
    return _validate_matrix_layout(
        file_path,
        arrays["atom_pairs"],
        arrays["chunk_boundaries"],
        arrays["chunk_shapes"],
        arrays["entries"],
        atom_symbols,
        elements_orbital_map,
        row_multiplier=row_multiplier,
    )


def _align_matrix_layout(
    layout: _MatrixLayout,
    canonical: _MatrixLayout,
    source: Union[str, Path],
) -> np.ndarray:
    """Return entries in canonical pair order after strict layout checks."""
    src_index = {
        tuple(int(value) for value in row): i for i, row in enumerate(layout.atom_pairs)
    }
    dst_keys = [tuple(int(value) for value in row) for row in canonical.atom_pairs]
    dst_set = set(dst_keys)
    src_set = set(src_index)
    if src_set != dst_set:
        missing = sorted(dst_set - src_set)
        extra = sorted(src_set - dst_set)
        raise _layout_error(
            source,
            "atom_pairs",
            f"pair set differs from canonical layout; "
            f"missing={missing[:3]}, extra={extra[:3]}",
        )

    for dst_idx, key in enumerate(dst_keys):
        src_idx = src_index[key]
        src_shape = tuple(int(value) for value in layout.chunk_shapes[src_idx])
        dst_shape = tuple(int(value) for value in canonical.chunk_shapes[dst_idx])
        if src_shape != dst_shape:
            raise _layout_error(
                source,
                "chunk_shapes",
                f"pair {key} has shape {src_shape}; canonical shape is {dst_shape}",
            )

    if np.array_equal(layout.atom_pairs, canonical.atom_pairs):
        if not np.array_equal(layout.chunk_shapes, canonical.chunk_shapes):
            raise _layout_error(
                source, "chunk_shapes", "layout differs from canonical layout"
            )
        if not np.array_equal(layout.chunk_boundaries, canonical.chunk_boundaries):
            raise _layout_error(
                source, "chunk_boundaries", "layout differs from canonical layout"
            )
        return layout.entries

    return _reorder_flat_entries(
        layout.entries,
        layout.atom_pairs,
        layout.chunk_boundaries,
        canonical.atom_pairs,
        canonical.chunk_boundaries,
    )


def _first_order_canonical_layout(canonical: _MatrixLayout) -> _MatrixLayout:
    chunk_shapes = canonical.chunk_shapes.copy()
    chunk_shapes[:, 0] *= 3
    sizes = np.prod(chunk_shapes.astype(np.int64), axis=1)
    boundaries64 = np.concatenate(([0], np.cumsum(sizes, dtype=np.int64)))
    if int(boundaries64[-1]) > np.iinfo(np.int32).max:
        raise AimspyConfigError(
            "electric_response.h5: chunk_boundaries exceed int32 range"
        )
    boundaries = boundaries64.astype(np.int32)
    return _MatrixLayout(
        atom_pairs=canonical.atom_pairs.copy(),
        chunk_boundaries=boundaries,
        chunk_shapes=chunk_shapes,
        entries=np.empty(int(boundaries[-1]), dtype=np.float64),
    )


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

# DeepH force-field stress convention: [xx, yy, zz, yz, xz, xy].
_STRESS_VOIGT_INDICES = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def _force_field_error(
    source: Union[str, Path], field: str, detail: str
) -> AimspyConfigError:
    """Build a field-specific error for DeepH ``force.h5`` data."""
    return AimspyConfigError(f"{Path(source).name}: {field}: {detail}")


def _validate_force(
    force: np.ndarray,
    n_atoms: int,
    source: Union[str, Path],
) -> np.ndarray:
    arr = np.asarray(force)
    if not (
        np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating)
    ):
        raise _force_field_error(
            source, "force", f"expected real numeric dtype, got {arr.dtype}"
        )
    expected_shape = (n_atoms, 3)
    if arr.shape != expected_shape:
        raise _force_field_error(
            source,
            "force",
            f"expected shape {expected_shape}, got {arr.shape}",
        )
    result = np.ascontiguousarray(arr, dtype=np.float64)
    if not np.isfinite(result).all():
        raise _force_field_error(source, "force", "values must all be finite")
    return result


def _validate_energy_eV(
    energy_eV: float,
    source: Union[str, Path],
) -> float:
    arr = np.asarray(energy_eV)
    if arr.shape != ():
        raise _force_field_error(
            source, "energy", f"expected a scalar, got shape {arr.shape}"
        )
    if not (
        np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating)
    ):
        raise _force_field_error(
            source, "energy", f"expected real numeric dtype, got {arr.dtype}"
        )
    result = float(arr)
    if not np.isfinite(result):
        raise _force_field_error(source, "energy", "value must be finite")
    return result


def _validate_stress(
    stress: np.ndarray,
    source: Union[str, Path],
) -> np.ndarray:
    """Return a symmetric ``(3, 3)`` stress tensor in eV/Angstrom^3."""
    arr = np.asarray(stress)
    if not (
        np.issubdtype(arr.dtype, np.integer) or np.issubdtype(arr.dtype, np.floating)
    ):
        raise _force_field_error(
            source, "stress", f"expected real numeric dtype, got {arr.dtype}"
        )
    arr = np.asarray(arr, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise _force_field_error(source, "stress", "values must all be finite")
    if arr.shape == (1, 6):
        arr = arr[0]
    if arr.shape == (6,):
        tensor = np.zeros((3, 3), dtype=np.float64)
        for value, (i, j) in zip(arr, _STRESS_VOIGT_INDICES):
            tensor[i, j] = value
            tensor[j, i] = value
    elif arr.shape == (3, 3):
        tensor = np.ascontiguousarray(arr)
        if not np.allclose(tensor, tensor.T, rtol=1e-10, atol=1e-12):
            raise _force_field_error(source, "stress", "3x3 tensor must be symmetric")
    else:
        raise _force_field_error(
            source,
            "stress",
            f"expected shape (6,), (1, 6), or (3, 3), got {arr.shape}",
        )
    return np.ascontiguousarray(tensor)


def _stress_to_voigt(stress: np.ndarray) -> np.ndarray:
    tensor = _validate_stress(stress, "force.h5")
    return np.asarray(
        [tensor[i, j] for i, j in _STRESS_VOIGT_INDICES], dtype=np.float64
    )


def _read_force_h5(
    file_path: Path,
    n_atoms: int,
    lattice: np.ndarray,
) -> tuple[Optional[np.ndarray], Optional[float], Optional[np.ndarray]]:
    """Read and validate optional DeepH force-field labels."""
    with h5py.File(file_path, "r") as h5:
        if "cell" in h5:
            cell = np.asarray(
                _hdf5_dataset_value(h5, "cell", file_path, _force_field_error)
            )
            if not (
                np.issubdtype(cell.dtype, np.integer)
                or np.issubdtype(cell.dtype, np.floating)
            ):
                raise _force_field_error(
                    file_path,
                    "cell",
                    f"expected real numeric dtype, got {cell.dtype}",
                )
            cell = np.asarray(cell, dtype=np.float64)
            if cell.shape != (3, 3):
                raise _force_field_error(
                    file_path, "cell", f"expected shape (3, 3), got {cell.shape}"
                )
            if not np.isfinite(cell).all():
                raise _force_field_error(file_path, "cell", "values must all be finite")
            if not np.allclose(cell, lattice, rtol=1e-10, atol=1e-10):
                raise _force_field_error(
                    file_path, "cell", "does not match the POSCAR lattice"
                )

        force = (
            _validate_force(
                _hdf5_dataset_value(h5, "force", file_path, _force_field_error),
                n_atoms,
                file_path,
            )
            if "force" in h5
            else None
        )
        energy_eV = (
            _validate_energy_eV(
                _hdf5_dataset_value(h5, "energy", file_path, _force_field_error),
                file_path,
            )
            if "energy" in h5
            else None
        )
        stress = (
            _validate_stress(
                _hdf5_dataset_value(h5, "stress", file_path, _force_field_error),
                file_path,
            )
            if "stress" in h5
            else None
        )
    if force is None and energy_eV is None and stress is None:
        raise _force_field_error(
            file_path,
            "datasets",
            "at least one of energy, force, or stress must be present",
        )
    return force, energy_eV, stress


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
      - ``force.h5``        — *optional* — MD-style: cell plus available
        energy and force labels (eV and eV/Å), and stress in eV/Å³.  When
        analytical stress is unavailable, the on-disk stress label is six zeros.

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

    # force-field labels (optional, for MD-style force.h5 export)
    force: Optional[np.ndarray] = None  # (n_atoms, 3) float64, eV/Å, POSCAR order
    energy_eV: Optional[float] = None  # scalar, eV

    # metadata (for info.json round-trip)
    fermi_energy_eV: float = 0.0

    # pre-specified save path (set by from_directory / from_aimspy / path= kwarg)
    path: Optional[Path] = None

    # Added after the legacy positional fields to keep direct construction compatible.
    stress: Optional[np.ndarray] = None  # (3, 3) float64, eV/Å³

    # ----------------------------------------------------------------
    # Construction from directory
    # ----------------------------------------------------------------
    @classmethod
    def from_directory(cls, path: Union[str, Path]) -> "DeepHData":
        """Read POSCAR + info.json + matrix .h5 files from *path*.

        Requires POSCAR + info.json + at least one matrix file
        (``hamiltonian.h5``, ``overlap.h5``, or ``hamiltonian_init.h5``).
        Optionally reads ``force.h5`` (MD-style format: cell plus any available
        energy/force/stress labels) if present.  Stress is normalized to a
        symmetric ``(3, 3)`` tensor in eV/Å³.  Sets ``self.path = path`` for
        subsequent ``save_*()`` calls.
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

        # Validate every standard matrix independently, then align it to the
        # first available file's canonical atom-pair order.
        loaded = [
            (name, p, _read_matrix_layout(p, atom_symbols, eom)) for name, p in found
        ]
        canonical = loaded[0][2]
        atom_pairs = canonical.atom_pairs
        cb = canonical.chunk_boundaries
        cs = canonical.chunk_shapes

        entries = None
        overlap_entries = None
        init_entries = None
        for name, p, layout in loaded:
            data = _align_matrix_layout(layout, canonical, p)
            if name == "hamiltonian":
                entries = data
            elif name == "overlap":
                overlap_entries = data
            elif name == "initial_hamiltonian":
                init_entries = data

        # Check for optional force.h5 (different format from matrix .h5).
        force_path = path / "force.h5"
        force_arr = None
        energy_val = None
        stress_arr = None
        if force_path.is_file():
            force_arr, energy_val, stress_arr = _read_force_h5(
                force_path, len(atom_symbols), lattice
            )

        # Check for optional electric_response.h5 (dH/de, same atom_pairs as
        # hamiltonian.h5 but chunk_shapes/boundaries expanded 3× per pair).
        fo_path = path / "electric_response.h5"
        fo_entries = None
        fo_cb = None
        fo_cs = None
        if fo_path.is_file():
            fo_layout = _read_matrix_layout(
                fo_path,
                atom_symbols,
                eom,
                row_multiplier=3,
            )
            fo_canonical = _first_order_canonical_layout(canonical)
            fo_entries = _align_matrix_layout(fo_layout, fo_canonical, fo_path)
            fo_cb = fo_canonical.chunk_boundaries
            fo_cs = fo_canonical.chunk_shapes

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
            stress=stress_arr,
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
        stress: Optional[np.ndarray] = None,
    ) -> "DeepHData":
        """Build from in-memory pair-block dicts.

        All matrix blocks are optional — at least one must be given.
        Keys are ``(R1,R2,R3,i,j)`` with atoms in POSCAR order.
        Hamiltonian / initial_hamiltonian blocks in **Hartree**
        (converted to eV here). Overlap blocks are dimensionless.

        *force*, *energy_eV*, and *stress* are optional labels for MD-style
        ``force.h5`` export. *force* is ``(n_atoms, 3)`` in eV/Å, already in
        **POSCAR atom order** (matching *atom_coords*); *energy_eV* is a scalar
        in eV; *stress* is a symmetric ``(3, 3)`` tensor in eV/Å³.

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

        # Validate optional force-field labels without changing their units.
        if force is not None:
            force = _validate_force(force, len(atom_symbols), "from_memory")
        if energy_eV is not None:
            energy_eV = _validate_energy_eV(energy_eV, "from_memory")
        if stress is not None:
            stress = _validate_stress(stress, "from_memory")

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
            stress=stress,
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
        stress: Optional[np.ndarray] = None,
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
            DeepH energy target in **Hartree** (converted to eV inside).  DeepH
            force-field training normally subtracts free-atom reference
            energies; pass :attr:`Calculator.energy_free_relative` for the
            force-consistent reference-subtracted target.
        first_order_hamiltonian : list[AimspyMatrix], optional
            Electric-response first-order Hamiltonian ``dH/de`` — a list
            of 3 ``AimspyMatrix`` in Cartesian order ``[x, y, z]``
            (Hartree, aims atom order). Reordered to POSCAR order and
            concatenated per atom pair in DeepH order ``[y, z, x]``.
        stress : np.ndarray, optional
            Symmetric analytical stress tensor ``(3, 3)`` in eV/Å³.  The
            Cartesian frame and FHI-aims sign convention are preserved.

        .. note::

            *force*, *energy*, *first_order_hamiltonian*, and *stress* are
            placed after *path* to preserve the positional ordering of the
            pre-existing matrix and *template* arguments. Passing these
            observables by keyword is recommended.
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
            force_arr = _validate_force(force, structure.n_atoms, "from_aimspy")
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
            stress=stress,
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
        stress: Optional[np.ndarray] = None,
    ) -> None:
        """Store DeepH force-field labels from AimsPy observables.

        Parameters
        ----------
        force_aims : np.ndarray or list
            Forces ``(n_atoms, 3)`` in eV/Å, **aims atom order**.
            Reordered to POSCAR order inside.  Accepts list or ndarray.
        structure : AimspyStructure
            Provides the aims→POSCAR atom permutation.
        energy : float, optional
            DeepH energy target in **Hartree** (converted to eV), or None.
            Use :attr:`Calculator.energy_free_relative` for the standard
            force-consistent reference-subtracted target.
        stress : np.ndarray, optional
            Symmetric analytical stress ``(3, 3)`` in eV/Å³.
        """
        force_arr = _validate_force(force_aims, structure.n_atoms, "set_force")
        _, new2old = structure.build_atom_permutation()
        self.force = np.ascontiguousarray(force_arr[new2old])
        if energy is not None:
            self.energy_eV = _validate_energy_eV(
                float(energy) * HARTREE_TO_EV, "set_force"
            )
        if stress is not None:
            self.stress = _validate_stress(stress, "set_force")

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
        layout = _validate_matrix_layout(
            file_path,
            self.atom_pairs,
            self.chunk_boundaries,
            self.chunk_shapes,
            entries,
            self.atom_symbols,
            self.elements_orbital_map,
        )
        with h5py.File(file_path, "w") as f:
            f.create_dataset("atom_pairs", data=layout.atom_pairs, dtype="i4")
            f.create_dataset(
                "chunk_boundaries", data=layout.chunk_boundaries, dtype="i4"
            )
            f.create_dataset("chunk_shapes", data=layout.chunk_shapes, dtype="i4")
            f.create_dataset("entries", data=layout.entries)

    def _write_force_h5(self, file_path: Path) -> None:
        """Write force.h5 in DeepH MD convention.

        Always writes ``cell`` (3,3) and ``stress`` (6,).  Available ``energy``
        scalar and ``force`` (n_atoms,3) labels are written conditionally.
        Stress uses DeepH order ``[xx, yy, zz, yz, xz, xy]``; when
        :attr:`stress` is ``None``, six zeros are written.
        Root attrs: ``formula`` = ``b'X{natoms}'``, ``natoms`` = int64.
        """
        n_atoms = self.n_atoms
        lattice = np.asarray(self.lattice, dtype=np.float64)
        if lattice.shape != (3, 3) or not np.isfinite(lattice).all():
            raise _force_field_error(
                file_path, "cell", "expected a finite (3, 3) lattice"
            )
        force = (
            _validate_force(self.force, n_atoms, file_path)
            if self.force is not None
            else None
        )
        energy_eV = (
            _validate_energy_eV(self.energy_eV, file_path)
            if self.energy_eV is not None
            else None
        )
        stress_voigt = (
            _stress_to_voigt(self.stress)
            if self.stress is not None
            else np.zeros(6, dtype=np.float64)
        )
        with h5py.File(file_path, "w") as f:
            f.create_dataset("cell", data=lattice, dtype="f8")
            if energy_eV is not None:
                f.create_dataset("energy", data=energy_eV)
            if force is not None:
                f.create_dataset("force", data=force, dtype="f8")
            f.create_dataset("stress", data=stress_voigt, dtype="f8")
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
        """Write available energy, force, and stress labels to force.h5.

        At least one in-memory label must be present. Missing energy and force
        labels are omitted; missing stress is written as six zeros for DeepH
        compatibility.
        """
        if self.force is None and self.energy_eV is None and self.stress is None:
            raise AimspyConfigError("No energy, force, or stress data to save")
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
        file_path = p / "electric_response.h5"
        layout = _validate_matrix_layout(
            file_path,
            self.atom_pairs,
            self._fo_chunk_boundaries,
            self._fo_chunk_shapes,
            self.first_order_hamiltonian_entries,
            self.atom_symbols,
            self.elements_orbital_map,
            row_multiplier=3,
        )
        with h5py.File(file_path, "w") as f:
            f.create_dataset("atom_pairs", data=layout.atom_pairs, dtype="i4")
            f.create_dataset(
                "chunk_boundaries", data=layout.chunk_boundaries, dtype="i4"
            )
            f.create_dataset("chunk_shapes", data=layout.chunk_shapes, dtype="i4")
            f.create_dataset("entries", data=layout.entries)

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
        if (
            self.force is not None
            or self.energy_eV is not None
            or self.stress is not None
        ):
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
        if self.stress is not None:
            extra.append("+stress")
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

            Only the Hamiltonian is converted.  Force-field labels loaded from
            ``force.h5`` are accessible directly via ``self.force``,
            ``self.energy_eV``, and ``self.stress`` — they do **not**
            participate in the warmstart injection path.

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

    scale_tokens = lines[1].split()
    if len(scale_tokens) != 1:
        raise AimspyConfigError(
            f"{path.name}: scale: expected one scalar, got {len(scale_tokens)} values"
        )
    try:
        scale = float(scale_tokens[0])
    except ValueError as exc:
        raise AimspyConfigError(
            f"{path.name}: scale: invalid floating-point value {scale_tokens[0]!r}"
        ) from exc
    if not np.isfinite(scale) or scale == 0.0:
        raise AimspyConfigError(
            f"{path.name}: scale: expected a finite non-zero scalar, got {scale!r}"
        )

    raw_lattice = np.array(
        [[float(x) for x in lines[i].split()] for i in range(2, 5)],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(raw_lattice)):
        raise AimspyConfigError(f"{path.name}: lattice: contains non-finite values")
    if scale < 0.0:
        raw_volume = abs(float(np.linalg.det(raw_lattice)))
        if not np.isfinite(raw_volume) or raw_volume == 0.0:
            raise AimspyConfigError(
                f"{path.name}: scale: negative target-volume scale requires "
                "a finite non-singular lattice"
            )
        factor = (abs(scale) / raw_volume) ** (1.0 / 3.0)
    else:
        factor = scale
    if not np.isfinite(factor):
        raise AimspyConfigError(f"{path.name}: scale: computed factor is non-finite")
    lat = factor * raw_lattice

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

    # Parse optional Selective Dynamics and the mandatory coordinate mode.
    if coord_start >= len(lines):
        raise AimspyConfigError(f"{path.name}: coordinates: missing coordinate mode")
    token = lines[coord_start].split()[0].lower()
    if token.startswith("s"):
        coord_start += 1
        if coord_start >= len(lines):
            raise AimspyConfigError(
                f"{path.name}: coordinates: missing mode after Selective Dynamics"
            )
        token = lines[coord_start].split()[0].lower()

    if token in {"direct", "d"}:
        coord_mode = "direct"
    elif token in {"cartesian", "c", "kartesian", "k"}:
        coord_mode = "cartesian"
    else:
        raise AimspyConfigError(f"{path.name}: coordinates: unsupported mode {token!r}")
    coord_start += 1

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

    if coord_mode == "direct":
        coords = coords @ lat
    else:
        coords *= factor

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
