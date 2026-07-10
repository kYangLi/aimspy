"""Public data classes — AimspyInfo, CsrMatrixDescriptor.

These are the primary data carriers of the aimspy public API.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np

# =============================================================================
# Units
# =============================================================================
HARTREE_TO_EV = 27.2113845
EV_TO_HARTREE = 1.0 / HARTREE_TO_EV
BOHR_TO_ANG = 0.529177210903


# =============================================================================
# AimspyInfo — snapshot of FHI-aims runtime state
# =============================================================================
@dataclass
class AimspyInfo:
    """Snapshot of basic FHI-aims runtime info.

    Obtained by calling ``aimspy_get_info()`` after ``aimspy_init()``.
    All arrays are independent numpy copies — safe to hold after
    ``aimspy_finalize()``.
    """
    n_atoms: int
    n_species: int
    n_basis: int
    n_basis_fns: int
    n_spin: int
    n_k_points: int
    n_states: int
    n_cells: int
    n_ham_size: int
    n_periodic: int
    n_centers_basis_I: int
    n_centers_basis_T: int
    n_full_points: int
    n_full_points_total: int
    spin_degeneracy: int
    packed_matrix_format: int
    flag_rel: int
    spin_treatment: int
    myid: int
    n_tasks: int
    output_level: str
    use_scalapack: bool
    use_elpa: bool
    real_eigenvectors: bool
    use_hartree_fock: bool
    use_periodic_hf: bool
    use_hf_kspace: bool
    use_mpi: bool
    coords: np.ndarray
    frac_coords: Optional[np.ndarray]
    lattice: Optional[np.ndarray]
    recip_lattice: Optional[np.ndarray]
    species_idx: np.ndarray
    atoms_species: List[str]
    species_names: List[str] = field(default_factory=list)
    species_elements: List[str] = field(default_factory=list)
    species_z: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float64))
    k_points: Optional[np.ndarray] = None
    k_weights: Optional[np.ndarray] = None
    basis_atom: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    basis_l: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    basis_m: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    basis_fn: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    basisfn_n: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    basisfn_l: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))
    basisfn_type: List[str] = field(default_factory=list)
    basisfn_species: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32))

    @classmethod
    def from_c(cls, c_info) -> "AimspyInfo":
        """Build from a populated ``AimspyInfoC`` ctypes struct."""
        from ._binding.ctypes_types import AimspyInfoC as _C
        return _aimspy_info_from_c(c_info)

    @property
    def is_periodic(self) -> bool:
        return self.n_periodic > 0

    @property
    def is_csr_format(self) -> bool:
        return self.packed_matrix_format == 2

    @property
    def atoms_species_sorted_by_element(self) -> List[str]:
        """Per-atom species in POSCAR/DeepH element-grouped order."""
        sort_idxs = np.argsort(self.atoms_species, kind='stable')
        return [self.atoms_species[i] for i in sort_idxs]

    @property
    def basis_l_per_element(self) -> Dict[str, List[int]]:
        """{element_symbol: [l values]} for DeepH-compatible metadata."""
        result: Dict[str, List[int]] = {}
        for idx in range(self.n_atoms):
            elem = self.atoms_species[idx]
            bf_mask = self.basis_atom == idx
            ls = sorted(set(int(l) for l in self.basis_l[bf_mask]))
            result.setdefault(elem, []).extend(ls)
        return {k: sorted(set(v)) for k, v in result.items()}

    def __repr__(self) -> str:
        return (
            f"AimspyInfo(n_atoms={self.n_atoms}, n_species={self.n_species}, "
            f"n_basis={self.n_basis}, n_spin={self.n_spin}, "
            f"n_k_points={self.n_k_points}, n_cells={self.n_cells}, "
            f"n_ham_size={self.n_ham_size}, n_periodic={self.n_periodic}, "
            f"mpi={self.myid}/{self.n_tasks})"
        )


# =============================================================================
# CsrMatrixDescriptor — CSR sparse matrix layout
# =============================================================================
@dataclass
class CsrMatrixDescriptor:
    """Snapshot of the FHI-aims CSR sparse-storage layout.

    Captured once via the ``get_descr`` callback.  All arrays are
    independent numpy copies.
    """
    n_basis: int
    n_spin: int
    n_cells: int          # includes sentinel cell
    n_ham_size: int       # includes trash slot
    cell_idx: np.ndarray      # (3, n_cells) — Fortran cell_index(i_cell, i_cart)
    row_mx_idx: np.ndarray    # (n_basis, n_cells, 2) — [0]=start, [1]=end (1‑based)
    col_mx_idx: np.ndarray    # (n_ham_size,) — col basis idx (1‑based)

    @classmethod
    def _from_c_struct(cls, c_struct) -> "CsrMatrixDescriptor":
        """Build from the ``CsrMxDescrC`` struct populated by the
        ``aimspy_register_get_descr_callback`` default path."""
        return _csr_from_c_struct(c_struct)


# =============================================================================
# Internal helpers (used by AimspyInfo.from_c and CsrMatrixDescriptor._from_c_struct)
# =============================================================================
def _view_generic(ptr, shape, ctypes_dtype, numpy_order='C'):
    """Generalised helper: raw ptr -> C-contiguous ndarray copy.

    *ptr* may be a ctypes POINTER instance or an int address (from a
    c_void_p field).  *shape* is the **target** Python shape.  When
    *numpy_order* is ``'F'``, the flat buffer is reshaped with
    ``order='F'`` (Fortran column-major interpretation), which produces
    a transposed logical indexing relative to the C-order case.
    """
    from ctypes import cast, c_void_p, POINTER
    if not ptr:
        raise ValueError("NULL pointer")
    n = 1
    for d in shape:
        n *= d
    if isinstance(ptr, int):
        ptr = cast(c_void_p(ptr), POINTER(ctypes_dtype))
    flat = np.ctypeslib.as_array(ptr, shape=(n,))
    arr = flat.reshape(shape, order=numpy_order)
    return np.ascontiguousarray(arr).copy()


def _view_f(ptr, fshape):
    """PyFortran column-major array (e.g. ``coords(3,N)``) → C C-order copy.

    Returns shape *fshape*, e.g. ``_view_f(ptr, (3, N))`` gives ``(3, N)``
    with C-contiguous data where ``arr[i, j]`` ↔ Fortran ``arr(i+1, j+1)``
    (axes have the same order).  Callers typically apply ``.T`` to swap
    axes so that ``arr.T[j, i]`` ↔ Fortran ``arr(i+1, j+1)``.
    """
    from ctypes import c_double
    return _view_generic(ptr, fshape, c_double, numpy_order='F')


def _view_i(ptr, shape):
    """Raw ptr (POINTER(c_int) or int) -> C-contiguous int ndarray copy.

    Uses default C‑order reshape so ``arr[i, j, ...]`` ↔ Fortran
    ``arr(i+1, j+1, ...)`` (the axes have the same order).  This is
    the convention consumed by ``CsrMatrixDescriptor`` and
    ``deeph_to_aims_hamiltonian``.
    """
    from ctypes import c_int
    return _view_generic(ptr, shape, c_int, numpy_order='C')


def _decode_fortran_char_array(ptr, n_items, item_len):
    """Decode a Fortran ``character(LEN=N)(:)`` buffer into list[str]."""
    from ctypes import cast, c_void_p, POINTER, c_char
    if not ptr:
        return []
    flat = np.ctypeslib.as_array(
        cast(c_void_p(ptr), POINTER(c_char)),
        shape=(item_len * n_items,),
    ).copy()
    flat = flat.reshape(n_items, item_len)
    return [bytes(r).split(b'\0')[0].decode('utf-8').strip() for r in flat]


def _csr_from_c_struct(c):
    """Populate a CsrMatrixDescriptor from the C-side struct ``CsrMxDescrC``."""
    return CsrMatrixDescriptor(
        n_basis=int(c.n_basis),
        n_spin=int(c.n_spin),
        n_cells=int(c.n_cells),
        n_ham_size=int(c.n_ham_size),
        cell_idx=_view_i(c.cell_idx, (3, c.n_cells)).astype(np.int32),
        row_mx_idx=_view_i(c.row_mx_idx, (c.n_basis, c.n_cells, 2)).astype(np.int32),
        col_mx_idx=_view_i(c.col_mx_idx, (c.n_ham_size,)).astype(np.int32),
    )


def _aimspy_info_from_c(c):
    """Populate an AimspyInfo from the C-side struct ``AimspyInfoC``."""
    N = int(c.n_atoms)
    NS = int(c.n_species)
    NB = int(c.n_basis)
    NBF = int(c.n_basis_fns)
    NK = int(c.n_k_points)
    NP = int(c.n_periodic)

    coords = _view_f(c.coords_ptr, (3, N)).T.copy() * BOHR_TO_ANG

    if c.frac_coords_ptr:
        frac = _view_f(c.frac_coords_ptr, (3, N)).T.copy() * BOHR_TO_ANG
    else:
        frac = None

    if NP > 0:
        lat_full = _view_f(c.lattice_ptr, (3, 3))
        lat = lat_full[:, :NP].T.copy() * BOHR_TO_ANG
        if c.recip_lattice_ptr:
            rlat = _view_f(c.recip_lattice_ptr, (3, NP)).T.copy() / BOHR_TO_ANG
        else:
            rlat = None
    else:
        lat, rlat = None, None

    sp_idx = _view_i(c.species_idx_ptr, (N,)) - 1

    names = _decode_fortran_char_array(c.species_names_ptr, NS, 20)
    elements = _decode_fortran_char_array(c.species_elements_ptr, NS, 2)
    z_arr = _view_f(c.species_z_ptr, (NS,))
    atoms_species = [names[int(i)] for i in sp_idx]

    if NK > 0 and c.k_points_ptr:
        kp = _view_f(c.k_points_ptr, (3, NK)).T.copy() / BOHR_TO_ANG
        kw = _view_f(c.k_weights_ptr, (NK,))
    else:
        kp, kw = None, None

    basis_atom = _view_i(c.basis_atom_ptr, (NB,)) - 1
    basis_l = _view_i(c.basis_l_ptr, (NB,))
    basis_m = _view_i(c.basis_m_ptr, (NB,))
    basis_fn = _view_i(c.basis_fn_ptr, (NB,)) - 1

    basisfn_n = _view_i(c.basisfn_n_ptr, (NBF,))
    basisfn_l = _view_i(c.basisfn_l_ptr, (NBF,))
    basisfn_type = _decode_fortran_char_array(c.basisfn_type_ptr, NBF, 8)
    basisfn_species = _view_i(c.basisfn_species_ptr, (NBF,)) - 1

    ol = _decode_fortran_char_array(c.output_level_ptr, 1, 20)
    ol_str = ol[0] if ol else ""

    return AimspyInfo(
        n_atoms=N, n_species=NS, n_basis=NB, n_basis_fns=NBF,
        n_spin=int(c.n_spin), n_k_points=NK,
        n_states=int(c.n_states), n_cells=int(c.n_cells),
        n_ham_size=int(c.n_ham_size), n_periodic=NP,
        n_centers_basis_I=int(c.n_centers_basis_I),
        n_centers_basis_T=int(c.n_centers_basis_T),
        n_full_points=int(c.n_full_points),
        n_full_points_total=int(c.n_full_points_total),
        spin_degeneracy=int(c.spin_degeneracy),
        packed_matrix_format=int(c.packed_matrix_format),
        flag_rel=int(c.flag_rel), spin_treatment=int(c.spin_treatment),
        myid=int(c.myid), n_tasks=int(c.n_tasks),
        output_level=ol_str,
        use_scalapack=bool(c.use_scalapack),
        use_elpa=bool(c.use_elpa),
        real_eigenvectors=bool(c.real_eigenvectors),
        use_hartree_fock=bool(c.use_hartree_fock),
        use_periodic_hf=bool(c.use_periodic_hf),
        use_hf_kspace=bool(c.use_hf_kspace),
        use_mpi=bool(c.use_mpi),
        coords=coords, frac_coords=frac, lattice=lat, recip_lattice=rlat,
        species_idx=sp_idx, atoms_species=atoms_species,
        species_names=names, species_elements=elements, species_z=z_arr,
        k_points=kp, k_weights=kw,
        basis_atom=basis_atom, basis_l=basis_l, basis_m=basis_m,
        basis_fn=basis_fn,
        basisfn_n=basisfn_n, basisfn_l=basisfn_l,
        basisfn_type=basisfn_type, basisfn_species=basisfn_species,
    )
