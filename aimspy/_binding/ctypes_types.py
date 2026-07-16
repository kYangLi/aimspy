"""Private — ctypes Structure mirrors of FHI-aims bind(C) types.

Field order MUST match the Fortran `type, bind(C)` declarations exactly.
"""

from __future__ import annotations

from ctypes import Structure, c_int, c_bool, c_void_p, POINTER


class AimspyInfoC(Structure):
    """ctypes mirror of Fortran `TAimspyInfo` (info.f90:29).

    Field order MUST match info.f90:29-92 exactly.
    """

    _fields_ = [
        ("n_atoms", c_int),
        ("n_species", c_int),
        ("n_basis", c_int),
        ("n_basis_fns", c_int),
        ("n_spin", c_int),
        ("n_k_points", c_int),
        ("n_states", c_int),
        ("n_cells", c_int),
        ("n_ham_size", c_int),
        ("n_periodic", c_int),
        ("n_centers_basis_I", c_int),
        ("n_centers_basis_T", c_int),
        ("n_full_points", c_int),
        ("n_full_points_total", c_int),
        ("spin_degeneracy", c_int),
        ("packed_matrix_format", c_int),
        ("flag_rel", c_int),
        ("spin_treatment", c_int),
        ("myid", c_int),
        ("n_tasks", c_int),
        ("use_scalapack", c_bool),
        ("use_elpa", c_bool),
        ("real_eigenvectors", c_bool),
        ("use_hartree_fock", c_bool),
        ("use_periodic_hf", c_bool),
        ("use_hf_kspace", c_bool),
        ("use_mpi", c_bool),
        ("coords_ptr", c_void_p),
        ("frac_coords_ptr", c_void_p),
        ("lattice_ptr", c_void_p),
        ("recip_lattice_ptr", c_void_p),
        ("species_idx_ptr", c_void_p),
        ("species_names_ptr", c_void_p),
        ("species_elements_ptr", c_void_p),
        ("species_z_ptr", c_void_p),
        ("k_points_ptr", c_void_p),
        ("k_weights_ptr", c_void_p),
        ("basis_atom_ptr", c_void_p),
        ("basis_l_ptr", c_void_p),
        ("basis_m_ptr", c_void_p),
        ("basis_fn_ptr", c_void_p),
        ("basisfn_n_ptr", c_void_p),
        ("basisfn_l_ptr", c_void_p),
        ("basisfn_type_ptr", c_void_p),
        ("basisfn_species_ptr", c_void_p),
        ("output_level_ptr", c_void_p),
    ]


class CsrMxDescrC(Structure):
    """ctypes mirror of Fortran `TAimspyCsrMxDescr` (callback.f90:11).

    Field order MUST match callback.f90:11-19 exactly.
    """

    _fields_ = [
        ("n_basis", c_int),
        ("n_spin", c_int),
        ("n_cells", c_int),
        ("n_ham_size", c_int),
        ("cell_idx", POINTER(c_int)),
        ("row_mx_idx", POINTER(c_int)),
        ("col_mx_idx", POINTER(c_int)),
    ]
