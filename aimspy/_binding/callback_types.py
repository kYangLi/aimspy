"""Private — CFUNCTYPE declarations for all aimspy callback types.

Adding a new callback = adding one CFUNCTYPE here.  Each type is used
by `_callbacks/registry.py`.
"""

from __future__ import annotations

from ctypes import CFUNCTYPE, c_void_p, c_int, POINTER, c_double

# ---------------------------------------------------------------------------
# Callback type: get CSR matrix descriptor
# ---------------------------------------------------------------------------
GetDescrCb = CFUNCTYPE(None, c_void_p, c_void_p)
"""void(*)(void *aux, void *descr_ptr) — descr_ptr is C_ptr (integer address)"""

# ---------------------------------------------------------------------------
# Callback type: export overlap matrix
# ---------------------------------------------------------------------------
ExportOvlpCb = CFUNCTYPE(None, c_void_p, POINTER(c_double), c_int, c_int)
"""void(*)(void *aux, double *ovlp_ptr, int n_ham, int n_spin) — read overlap"""

# ---------------------------------------------------------------------------
# Callback type: export initial Hamiltonian H0
# ---------------------------------------------------------------------------
ExportH0Cb = CFUNCTYPE(None, c_void_p, POINTER(c_double), c_int, c_int)
"""void(*)(void *aux, double *h0_ptr, int n_ham, int n_spin) — read H0"""

# ---------------------------------------------------------------------------
# Callback type: modify / inject Hamiltonian
# ---------------------------------------------------------------------------
ModifyH0Cb = CFUNCTYPE(None, c_void_p, c_void_p, POINTER(c_double), c_int, c_int)
"""void(*)(void *aux, void *input_mx_ptr, double *h0_ptr, int n_ham, int n_spin)"""

# ---------------------------------------------------------------------------
# Callback type: export electric-response first-order Hamiltonian (dH/de)
# ---------------------------------------------------------------------------
ExportDHdeCb = CFUNCTYPE(None, c_void_p, POINTER(c_double), c_int, c_int, c_int, c_int)
"""void(*)(void *aux, double *mx_ptr, int n_ham, int n_dir, int n_spin, int j_coord)

Fires after CPSCF convergence. ``mx_ptr`` points to the flat buffer of
``DFPT_first_order_H_sparse`` (Fortran column-major ``(n_dir, 1, n_ham, n_spin)``).
``n_dir`` is 1 (serial) or 3 (full-memory); ``j_coord`` is 0 (full-memory)
or 1/2/3 (serial, x/y/z).
"""

# ---------------------------------------------------------------------------
# Callback type: modify / inject electric-response first-order Hamiltonian
# ---------------------------------------------------------------------------
ModifyDHdeCb = CFUNCTYPE(
    None, c_void_p, c_void_p, POINTER(c_double), c_int, c_int, c_int, c_int
)
"""void(*)(void *aux, void *input_mx_ptr, double *mx_ptr, int n_ham, int n_dir, int n_spin, int j_coord)

Fires before the initial U1 computation (CPSCF_LOOP prelude). ``mx_ptr`` is
writeable (Fortran ``intent(inout)``); Python injects the predicted dH/de
via ``memmove`` so that ``U1_init = f(Omega_MO, H1_predict)`` produces a
better starting point and accelerates CPSCF convergence.
"""

# ---------------------------------------------------------------------------
# Callback type: generic Python hook (no extra args beyond aux)
# ---------------------------------------------------------------------------
ReconstructMxCb = CFUNCTYPE(None, c_void_p)
"""void(*)(void *aux) — generic Python hook"""
