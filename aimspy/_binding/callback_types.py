"""Private — CFUNCTYPE declarations for all aimspy callback types.

Adding a new callback = adding one CFUNCTYPE here.  Each type is used
by `_callbacks/registry.py` and `_callbacks/defaults.py`.
"""
from __future__ import annotations

from ctypes import CFUNCTYPE, c_void_p, c_int, POINTER, c_double

from .ctypes_types import CsrMxDescrC

# ---------------------------------------------------------------------------
# Callback type: get CSR matrix descriptor
# ---------------------------------------------------------------------------
GetDescrCb = CFUNCTYPE(None, c_void_p, c_void_p)
"""void(*)(void *aux, void *descr_ptr) — descr_ptr is C_ptr (integer address)"""

# ---------------------------------------------------------------------------
# Callback type: export initial Hamiltonian H0
# ---------------------------------------------------------------------------
ExportH0Cb = CFUNCTYPE(None, c_void_p, POINTER(c_double), c_int, c_int)
"""void(*)(void *aux, double *h0_ptr, int n_ham, int n_spin) — read H0"""

# ---------------------------------------------------------------------------
# Callback type: modify / inject Hamiltonian
# ---------------------------------------------------------------------------
ModifyH0Cb = CFUNCTYPE(
    None, c_void_p, c_void_p, POINTER(c_double), c_int, c_int
)
"""void(*)(void *aux, void *input_mx_ptr, double *h0_ptr, int n_ham, int n_spin)"""

# ---------------------------------------------------------------------------
# Callback type: generic Python hook (no extra args beyond aux)
# ---------------------------------------------------------------------------
ReconstructMxCb = CFUNCTYPE(None, c_void_p)
"""void(*)(void *aux) — generic Python hook"""
