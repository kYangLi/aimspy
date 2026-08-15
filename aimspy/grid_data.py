"""Public — ``GridData``: real-space integration-grid data capture.

Captured via the ``export_grid_data`` callback (registered when
``CalculatorConfig.capture_grid_data=True``).  Fires **once after SCF
convergence** on every MPI rank; each rank receives its own grid-point
subset.

.. note:: LDA / scalar only

    ``vks`` is the **scalar** part of the Kohn-Sham potential,
    ``V_H + v_xc`` (exact for LDA).  The GGA non-local (vector) term
    ``4 * xc_gradient_deriv`` is **not** exported, so for GGA functionals
    ``vks`` contains only the scalar part.

Units follow aims native conventions: coords in bohr, potentials in
Hartree, densities in electrons/bohr^3, ``partition_tab`` in bohr^3.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np

from .data import HARTREE_TO_EV, BOHR_TO_ANG

__all__ = ["GridData"]


@dataclass
class GridData:
    """Per-rank real-space grid data (independent numpy copies).

    All arrays are this MPI rank's subset.  After ``calc()``, use
    :meth:`gather` to assemble the global grid on a root rank.

    Index arrays (``index_atom`` etc.) are **0-based** (converted from the
    Fortran 1-based convention).
    """

    # ---- scalars ----
    n_full_points: int
    n_spin: int
    n_atoms: int

    # ---- structure (in-memory, from AimspyInfo via AimspyStructure) ----
    # Optional: filled by the Calculator callback wrapper from the live
    # runtime structure (NOT from control.in / geometry.in), so the dataset
    # stays self-consistent with the in-memory aims state.  ``None`` when a
    # GridData is constructed offline without a structure.
    atom_coords: Optional[np.ndarray] = None  # (n_atoms, 3) Angstrom
    atom_symbols: Optional[list] = None  # per-atom element symbol, aims order
    lattice: Optional[np.ndarray] = None  # (n_periodic, 3) Angstrom

    # ---- structure / indexing / weights (per grid point) ----
    coords: np.ndarray = None  # (3, n) bohr
    partition_tab: np.ndarray = None  # (n,) bohr^3
    index_atom: np.ndarray = None  # (n,) int32, 0-based
    index_radial: np.ndarray = None  # (n,) int32, 0-based
    index_angular: np.ndarray = None  # (n,) int32, 0-based

    # ---- physical quantities ----
    rho: np.ndarray = None  # (n_spin, n) self-consistent spin density
    vks: np.ndarray = None  # (n_spin, n) V_KS = V_H + v_xc[rho]
    vks0: np.ndarray = None  # (n_spin, n) V_KS_0 = V_H_0 + v_xc[rho_0]
    vh: np.ndarray = None  # (n,) self-consistent electrostatic (V_H + V_nuc)
    vh0: np.ndarray = None  # (n,) free-atom electrostatic
    rho0: np.ndarray = None  # (n,) free-atom density == rho_free (4*pi removed)

    # ==================================================================
    # Construction from the C callback (internal)
    # ==================================================================
    @classmethod
    def _from_c(
        cls,
        descr_ptr,
        rho_ptr,
        vks_ptr,
        vks0_ptr,
        vh_ptr,
        vh0_ptr,
        rho0_ptr,
    ) -> "GridData":
        """Build from the raw C pointers handed to the export callback.

        All arrays are copied out of the Fortran buffers (which may be
        reused / freed after the callback returns).  Fortran arrays are
        column-major; physical 2-D arrays are reshaped with ``order='F'``
        to ``(n_spin, n)``.  Index arrays are converted to 0-based.
        """
        from ctypes import cast, c_void_p, POINTER, c_double, c_int

        from ._binding.ctypes_types import GridDescrC

        d = cast(c_void_p(descr_ptr), POINTER(GridDescrC)).contents
        n = int(d.n_full_points)
        nspin = int(d.n_spin)

        def _as_array(ptr, ctype, count):
            # np.ctypeslib.as_array accepts a ctypes POINTER instance directly
            # (the CFUNCTYPE trampoline passes POINTER(c_double)); for a raw
            # integer address or c_void_p we first cast to the typed pointer.
            if isinstance(ptr, int):
                ptr = cast(c_void_p(ptr), POINTER(ctype))
            elif isinstance(ptr, c_void_p):
                ptr = cast(ptr, POINTER(ctype))
            return np.ctypeslib.as_array(ptr, shape=(count,))

        def f64(ptr, count):
            return np.ascontiguousarray(_as_array(ptr, c_double, count)).copy()

        def i32(ptr, count):
            return np.ascontiguousarray(_as_array(ptr, c_int, count)).copy()

        coords = f64(d.coords_ptr, 3 * n).reshape(3, n, order="F")
        ptab = f64(d.partition_tab_ptr, n)
        # 1-based Fortran -> 0-based Python
        iatom = i32(d.index_atom_ptr, n) - 1
        irad = i32(d.index_radial_ptr, n) - 1
        iang = i32(d.index_angular_ptr, n) - 1

        rho = f64(rho_ptr, nspin * n).reshape(nspin, n, order="F")
        vks = f64(vks_ptr, nspin * n).reshape(nspin, n, order="F")
        vks0 = f64(vks0_ptr, nspin * n).reshape(nspin, n, order="F")
        vh = f64(vh_ptr, n)
        vh0 = f64(vh0_ptr, n)
        # Fortran exports free_rho_superpos which carries a 4*pi factor;
        # normalise here so that the stored ``rho0`` IS the free-atom density
        # (``rho_free``), keeping the dataset free of the 4*pi convention.
        rho0 = f64(rho0_ptr, n) / (4.0 * np.pi)

        # Structure fields (atom_coords / atom_symbols / lattice) are filled
        # by the Calculator callback wrapper from the live runtime structure
        # (aux["structure"]), keeping the dataset in-memory self-consistent.
        return cls(
            n_full_points=n,
            n_spin=nspin,
            n_atoms=int(d.n_atoms),
            coords=coords,
            partition_tab=ptab,
            index_atom=iatom,
            index_radial=irad,
            index_angular=iang,
            rho=rho,
            vks=vks,
            vks0=vks0,
            vh=vh,
            vh0=vh0,
            rho0=rho0,
        )

    # ==================================================================
    # Derived quantities (lazy, computed on access)
    # ==================================================================
    @property
    def rho_free(self) -> np.ndarray:
        """Free-atom superposition density, shape ``(n,)`` (spin-independent).

        Identical to ``rho0`` — the 4*pi factor is already removed at import
        time, so ``rho0`` and ``rho_free`` are the same physical density.
        """
        return self.rho0

    @property
    def delta_rho(self) -> np.ndarray:
        """Density difference ``rho - rho_free`` (broadcast to n_spin)."""
        if self.n_spin == 1:
            return self.rho - self.rho_free[np.newaxis, :]
        # spin channels share the same free-atom reference
        return self.rho - (0.5 * self.rho_free)[np.newaxis, :]

    @property
    def delta_vks(self) -> np.ndarray:
        """``vks - vks0`` (analogous to dH = H - H0)."""
        return self.vks - self.vks0

    @property
    def delta_vh(self) -> np.ndarray:
        """``vh - vh0`` (self-consistent minus free-atom electrostatic)."""
        return self.vh - self.vh0

    @property
    def vxc(self) -> np.ndarray:
        """XC potential ``vks - vh`` (scalar part; exact for LDA)."""
        return self.vks - self.vh[np.newaxis, :]

    @property
    def vxc0(self) -> np.ndarray:
        """XC potential of the free-atom density ``vks0 - vh0``."""
        return self.vks0 - self.vh0[np.newaxis, :]

    # ==================================================================
    # Unit-converted views
    # ==================================================================
    @property
    def coords_ang(self) -> np.ndarray:
        """Grid coordinates in Angstrom."""
        return self.coords * BOHR_TO_ANG

    @property
    def vks_ev(self) -> np.ndarray:
        return self.vks * HARTREE_TO_EV

    @property
    def vks0_ev(self) -> np.ndarray:
        return self.vks0 * HARTREE_TO_EV

    @property
    def vh_ev(self) -> np.ndarray:
        return self.vh * HARTREE_TO_EV

    @property
    def vh0_ev(self) -> np.ndarray:
        return self.vh0 * HARTREE_TO_EV

    # ==================================================================
    # MPI gather
    # ==================================================================
    @classmethod
    def gather(cls, local: "GridData", comm, root: int = 0) -> Optional["GridData"]:
        """Gather per-rank subsets to *root* and concatenate along the
        grid-point axis.

        Uses ``mpi4py.MPI.Comm.Gatherv`` for memory-efficient, zero-pickle
        transfer of numpy arrays.  Root peak memory is ~1x the total dataset
        (the receive buffer only), compared to ~3x for the default
        ``comm.gather`` on a Python dict (pickle + deserialize + concat).

        Parameters
        ----------
        local : GridData
            This rank's subset.
        comm : mpi4py.MPI.Comm
            MPI communicator.
        root : int
            Destination rank (default 0).

        Returns
        -------
        GridData or None
            On *root*: the global ``GridData`` (``n_full_points`` = global
            total).  On non-root ranks: ``None``.

        Notes
        -----
        The global point order is "concatenated by rank" and is not
        guaranteed to match the point order of a single-rank run; all
        integral / mapped quantities are unaffected (verified np=1/4/8).
        The structure fields (atom_coords / atom_symbols / lattice) are
        identical on every rank, so the root's copy is kept.
        """
        from mpi4py import MPI

        rank = comm.rank
        n_local = local.n_full_points

        # ---- consistency check (defensive: 2 allgather of a single int) ----
        all_nspin = comm.allgather(local.n_spin)
        all_natoms = comm.allgather(local.n_atoms)
        if len(set(all_nspin)) > 1 or len(set(all_natoms)) > 1:
            raise ValueError(
                f"inconsistent n_spin / n_atoms across ranks: "
                f"n_spin={all_nspin}, n_atoms={all_natoms}"
            )
        n_spin = local.n_spin
        n_atoms = local.n_atoms

        # ---- counts / displacements for Gatherv ----
        counts = comm.allgather(n_local)
        n_total = sum(counts)
        displs = [0]
        for c in counts[:-1]:
            displs.append(displs[-1] + c)

        def _gatherv(send_arr, leading_dim, np_dtype, mpi_type):
            """Gather a (leading_dim, n) or (n,) array along the point axis.

            For 2-D arrays the point axis is the *second* dimension; we
            transpose to (n, leading_dim) so the flattened C-order buffer
            walks points fastest, then scale counts/displacements by
            leading_dim.  For 1-D arrays the buffer is used directly.
            """
            if send_arr.ndim == 2:
                # (leading_dim, n) -> (n, leading_dim) -> flat (n*leading_dim,)
                sendflat = np.ascontiguousarray(send_arr.T).ravel()
                cnts = [c * leading_dim for c in counts]
                dspl = [d * leading_dim for d in displs]
            else:
                sendflat = np.ascontiguousarray(send_arr)
                cnts = counts
                dspl = displs

            if rank == root:
                recvbuf = np.empty(sum(cnts), dtype=np_dtype)
            else:
                recvbuf = None

            comm.Gatherv(sendflat, (recvbuf, cnts, dspl, mpi_type), root=root)

            if rank != root:
                return None
            if send_arr.ndim == 2:
                return recvbuf.reshape(n_total, leading_dim).T
            return recvbuf

        # ---- Gatherv each array ----
        coords_g = _gatherv(local.coords, 3, np.float64, MPI.DOUBLE)
        ptab_g = _gatherv(local.partition_tab, 1, np.float64, MPI.DOUBLE)
        iatom_g = _gatherv(local.index_atom, 1, np.int32, MPI.INT32_T)
        iradial_g = _gatherv(local.index_radial, 1, np.int32, MPI.INT32_T)
        iangular_g = _gatherv(local.index_angular, 1, np.int32, MPI.INT32_T)
        rho_g = _gatherv(local.rho, n_spin, np.float64, MPI.DOUBLE)
        vks_g = _gatherv(local.vks, n_spin, np.float64, MPI.DOUBLE)
        vks0_g = _gatherv(local.vks0, n_spin, np.float64, MPI.DOUBLE)
        vh_g = _gatherv(local.vh, 1, np.float64, MPI.DOUBLE)
        vh0_g = _gatherv(local.vh0, 1, np.float64, MPI.DOUBLE)
        rho0_g = _gatherv(local.rho0, 1, np.float64, MPI.DOUBLE)

        if rank != root:
            return None

        return cls(
            n_full_points=n_total,
            n_spin=n_spin,
            n_atoms=n_atoms,
            atom_coords=local.atom_coords,
            atom_symbols=local.atom_symbols,
            lattice=local.lattice,
            coords=coords_g,
            partition_tab=ptab_g,
            index_atom=iatom_g,
            index_radial=iradial_g,
            index_angular=iangular_g,
            rho=rho_g,
            vks=vks_g,
            vks0=vks0_g,
            vh=vh_g,
            vh0=vh0_g,
            rho0=rho0_g,
        )

    # ==================================================================
    # npz serialization (format-agnostic; no DeepH binding)
    # ==================================================================
    _NPZ_KEYS = (
        "coords",
        "partition_tab",
        "index_atom",
        "index_radial",
        "index_angular",
        "rho",
        "vks",
        "vks0",
        "vh",
        "vh0",
        "rho0",
    )

    def save_npz(self, path: Union[str, Path]) -> Path:
        """Save this (per-rank or gathered) dataset to a ``.npz`` file.

        Structure fields (atom_coords / atom_symbols / lattice) are stored
        when present, making the file self-describing.  All come from the
        in-memory aims runtime structure (never re-read from input files).
        """
        path = Path(path)
        arrays = {k: getattr(self, k) for k in self._NPZ_KEYS}
        meta = {
            "n_full_points": self.n_full_points,
            "n_spin": self.n_spin,
            "n_atoms": self.n_atoms,
        }
        if self.atom_coords is not None:
            meta["atom_coords"] = self.atom_coords
        if self.atom_symbols is not None:
            meta["atom_symbols"] = np.asarray(self.atom_symbols, dtype=str)
        if self.lattice is not None:
            meta["lattice"] = self.lattice
        np.savez(path, **meta, **arrays)
        return path

    @classmethod
    def load_npz(cls, path: Union[str, Path]) -> "GridData":
        """Load a dataset saved via :meth:`save_npz`."""
        path = Path(path)
        with np.load(path) as z:
            keys = set(z.files)
            return cls(
                n_full_points=int(z["n_full_points"]),
                n_spin=int(z["n_spin"]),
                n_atoms=int(z["n_atoms"]),
                atom_coords=z["atom_coords"] if "atom_coords" in keys else None,
                atom_symbols=(
                    [str(s) for s in z["atom_symbols"]]
                    if "atom_symbols" in keys
                    else None
                ),
                lattice=z["lattice"] if "lattice" in keys else None,
                coords=z["coords"],
                partition_tab=z["partition_tab"],
                index_atom=z["index_atom"],
                index_radial=z["index_radial"],
                index_angular=z["index_angular"],
                rho=z["rho"],
                vks=z["vks"],
                vks0=z["vks0"],
                vh=z["vh"],
                vh0=z["vh0"],
                rho0=z["rho0"],
            )

    # ==================================================================
    # Convenience
    # ==================================================================
    def integrated_electrons(self) -> float:
        """``sum(partition_tab * rho)`` summed over spin channels."""
        return float(
            sum(np.sum(self.partition_tab * self.rho[s]) for s in range(self.n_spin))
        )

    def __repr__(self) -> str:
        return (
            f"GridData(n={self.n_full_points}, "
            f"n_spin={self.n_spin}, n_atoms={self.n_atoms})"
        )
