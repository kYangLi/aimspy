"""Private — CDLL loading + MPI RTLD_GLOBAL workaround.

This is the only place that directly calls `ctypes.CDLL` on the aims
shared-library. All higher-level code uses `BindingLib`.
"""

from __future__ import annotations

from ctypes import CDLL, RTLD_GLOBAL
from pathlib import Path

# Module-level anchor for the mpi4py CDLL.  Without this, the CDLL
# created in load_aims_lib would be garbage-collected (and dlclose'd)
# when the function returns, potentially removing RTLD_GLOBAL symbols
# that libaims needs for lazy MPI symbol resolution.
_mpi_cdll_anchor: CDLL | None = None


def load_aims_lib(lib_path: Path | str) -> CDLL:
    """Load the FHI-aims shared-library, applying the MPI RTLD_GLOBAL fix.

    The RTLD_GLOBAL workaround addresses known MPICH / mpi4py symbol-
    visibility bugs that break Fortran MPI calls from inside dlopen'd
    shared libraries.

    Callers MUST have already initialised MPI via `mpi4py` before calling
    this function — it internally does `CDLL(MPI.__file__, mode=RTLD_GLOBAL)`.
    """
    global _mpi_cdll_anchor
    from mpi4py import MPI

    _mpi_cdll_anchor = CDLL(MPI.__file__, mode=RTLD_GLOBAL)
    return CDLL(str(lib_path), mode=RTLD_GLOBAL)
