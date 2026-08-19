"""Public — ``BasisData``: NAO radial basis function capture.

Captured via the ``export_basis_data`` callback (registered when
``CalculatorConfig.capture_basis_data=True``).  Fires **once** after
``shrink_fixed_basis_phi_thresh`` completes (pre-SCF), so the basis is
fully determined before any SCF iteration.

The exported data contains the complete spline representation of all
radial basis functions u(r), their kinetic terms (e−v)·u(r), and their
radial derivatives du/dr, together with the per-species logarithmic grid
parameters needed to evaluate them at arbitrary distances.

Units: lengths in bohr, energies in Hartree.  u(r) is normalized such
that ∫ u(r)² dr = 1, giving units of bohr^(−1/2).
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

__all__ = ["BasisData"]


@dataclass
class BasisData:
    """NAO radial basis function data (independent numpy copies).

    Contains the full cubic-spline representation of all radial basis
    functions, plus per-species logarithmic grid parameters.  Identity
    metadata (``n``, ``l``, ``type``, ``species``) is obtained from
    :class:`AimspyInfo` (already exported via ``aimspy_get_info``).
    """

    # ---- scalars ----
    n_species: int
    n_basis_fns: int  # total radial functions across all species
    n_max_grid: int  # max grid points across species
    n_max_spline: int  # always 4 (cubic)

    # ---- per-species grid parameters ----
    r_grid_min: np.ndarray = None  # (n_species,) bohr
    r_grid_inc: np.ndarray = None  # (n_species,) growth factor
    n_grid: np.ndarray = None  # (n_species,) int32

    # ---- full logarithmic grids (concatenated) ----
    r_grid: np.ndarray = None  # (total_grid_points,) bohr

    # ---- per-radial-function ----
    outer_radius: np.ndarray = None  # (n_basis_fns,) bohr

    # ---- spline coefficients (n_basis_fns, 4, n_max_grid) ----
    # Fortran original layout: (4, n_max_grid, n_basis_fns), i.e.
    # (coeff, grid, fn).  We reshape to (fn, 4, grid) with order='F'
    # so that spline_wave[i_fn] gives a contiguous (4, n_grid) array.
    spline_wave: np.ndarray = None  # u(r) spline coefficients
    spline_kinetic: np.ndarray = None  # (e−v)·u(r) spline coefficients
    spline_deriv: np.ndarray = None  # du/dr spline coefficients (0 if not available)

    # ---- convenience: per-function species map (0-based) ----
    # Attached by Calculator.init() after capture; lets the evaluate_*
    # methods be called without an explicit species_of_fn argument.
    species_of_fn: Optional[np.ndarray] = None  # (n_basis_fns,) int

    # ==================================================================
    # Construction from the C callback (internal)
    # ==================================================================
    @classmethod
    def _from_c(
        cls,
        descr_ptr,
        wave_spl_ptr,
        kinetic_spl_ptr,
        deriv_spl_ptr,
    ) -> "BasisData":
        """Build from the raw C pointers handed to the export callback.

        All arrays are copied out of the Fortran buffers.  The spline
        coefficient arrays arrive as flat Fortran column-major buffers
        of shape ``(n_max_spline, n_max_grid, n_basis_fns)``; they are
        reshaped to ``(n_basis_fns, n_max_spline, n_max_grid)`` with
        ``order='F'`` so that each function's ``(4, n_grid)`` block is
        contiguous.
        """
        from ctypes import POINTER, c_double, c_int, c_void_p, cast

        from ._binding.ctypes_types import BasisDescrC

        d = cast(c_void_p(descr_ptr), POINTER(BasisDescrC)).contents
        ns = int(d.n_species)
        nbf = int(d.n_basis_fns)
        nmg = int(d.n_max_grid)
        nsp = int(d.n_max_spline)
        tgp = int(d.total_grid_points)

        def _as_array(ptr, ctype, count):
            if isinstance(ptr, int):
                ptr = cast(c_void_p(ptr), POINTER(ctype))
            elif isinstance(ptr, c_void_p):
                ptr = cast(ptr, POINTER(ctype))
            return np.ctypeslib.as_array(ptr, shape=(count,))

        def f64(ptr, count):
            return np.ascontiguousarray(_as_array(ptr, c_double, count)).copy()

        def i32(ptr, count):
            return np.ascontiguousarray(_as_array(ptr, c_int, count)).copy()

        r_grid_min = f64(d.r_grid_min_ptr, ns)
        r_grid_inc = f64(d.r_grid_inc_ptr, ns)
        n_grid = i32(d.n_grid_ptr, ns)
        r_grid = f64(d.r_grid_ptr, tgp)
        outer_radius = f64(d.outer_radius_ptr, nbf)

        # Spline coefficients: flat (4 * n_max_grid * n_basis_fns,) in
        # Fortran column-major order of (n_max_spline, n_max_grid, n_basis_fns)
        # i.e. coeff varies fastest, fn varies slowest.
        # First reshape to (nsp, nmg, nbf) with order='F' (natural Fortran
        # layout), then transpose to (nbf, nsp, nmg) for Python access.
        spl_size = nsp * nmg * nbf
        spline_wave = np.ascontiguousarray(
            f64(wave_spl_ptr, spl_size)
            .reshape(nsp, nmg, nbf, order="F")
            .transpose(2, 0, 1)
        )
        spline_kinetic = np.ascontiguousarray(
            f64(kinetic_spl_ptr, spl_size)
            .reshape(nsp, nmg, nbf, order="F")
            .transpose(2, 0, 1)
        )
        spline_deriv = np.ascontiguousarray(
            f64(deriv_spl_ptr, spl_size)
            .reshape(nsp, nmg, nbf, order="F")
            .transpose(2, 0, 1)
        )

        return cls(
            n_species=ns,
            n_basis_fns=nbf,
            n_max_grid=nmg,
            n_max_spline=nsp,
            r_grid_min=r_grid_min,
            r_grid_inc=r_grid_inc,
            n_grid=n_grid,
            r_grid=r_grid,
            outer_radius=outer_radius,
            spline_wave=spline_wave,
            spline_kinetic=spline_kinetic,
            spline_deriv=spline_deriv,
        )

    # ==================================================================
    # Grid access
    # ==================================================================
    def species_r_grid(self, sp: int) -> np.ndarray:
        """Return the logarithmic radial grid for species *sp* (0-based).

        Uses the pre-computed ``r_grid`` array (extracted from the
        concatenated buffer by species offset).
        """
        offset = int(np.sum(self.n_grid[:sp]))
        n = int(self.n_grid[sp])
        return self.r_grid[offset : offset + n]

    def species_r_grid_rebuild(self, sp: int) -> np.ndarray:
        """Rebuild the logarithmic grid from the 3 scalar parameters.

        Provided for verification: ``species_r_grid_rebuild(sp)`` should
        match ``species_r_grid(sp)`` to machine precision.
        """
        return self.r_grid_min[sp] * self.r_grid_inc[sp] ** np.arange(self.n_grid[sp])

    # ==================================================================
    # Spline evaluation
    # ==================================================================
    def _resolve_species_of_fn(self, species_of_fn: Optional[np.ndarray]) -> np.ndarray:
        sof = species_of_fn if species_of_fn is not None else self.species_of_fn
        if sof is None:
            raise ValueError(
                "species_of_fn not provided and not attached to this BasisData "
                "(attached automatically when captured via "
                "CalculatorConfig.capture_basis_data=True; otherwise pass "
                "info.basisfn_species explicitly)."
            )
        return sof

    def _evaluate_spline_fn(
        self,
        spline: np.ndarray,
        i_fn: int,
        r: np.ndarray,
        species_of_fn: Optional[np.ndarray],
    ) -> np.ndarray:
        """Shared kernel: evaluate one spline channel of function *i_fn*.

        FHI-aims builds its cubic splines on the *integer grid index*
        (not on r):  S(i_r) = c1 + c2·t + c3·t² + c4·t³ with t = i_r − i
        on interval [i, i+1], where the fractional index of a distance
        on the logarithmic grid r(i) = r_min·inc^(i−1) is

            i_r = 1 + ln(r / r_grid_min) / ln(r_grid_inc)

        Evaluation therefore inverse-maps r → i_r, snaps near-integers
        (log/exp roundoff at exact grid points), and Horner-evaluates
        the local cubic.  Values outside ``[r_grid_min, outer_radius]``
        are zero — evaluating below the first grid point would
        extrapolate the first cubic segment backwards and blow up.

        Parameters
        ----------
        spline : np.ndarray
            ``(n_basis_fns, 4, n_max_grid)`` coefficient channel
            (``spline_wave`` / ``spline_kinetic`` / ``spline_deriv``).
        i_fn : int
            Global 0-based radial function index.
        r : np.ndarray
            Distances in bohr.
        species_of_fn : np.ndarray or None
            Per-function 0-based species map (explicit override);
            falls back to the map attached at capture time.
        """
        sp = int(self._resolve_species_of_fn(species_of_fn)[i_fn])
        r = np.asarray(r, dtype=np.float64)
        result = np.zeros_like(r)

        mask = (r >= self.r_grid_min[sp]) & (r <= self.outer_radius[i_fn])
        if not np.any(mask):
            return result
        r_m = r[mask]

        # Inverse logarithmic mapping: r → fractional grid index (1-based)
        log_inc = np.log(self.r_grid_inc[sp])
        i_r = 1.0 + np.log(r_m / self.r_grid_min[sp]) / log_inc

        # Snap to nearest integer when within floating-point epsilon.
        # This handles the case where r is exactly a grid point but
        # log/exp roundoff makes i_r slightly off (e.g. 10.99999999999997
        # instead of 11.0).
        i_r_rounded = np.round(i_r)
        snap_mask = np.abs(i_r - i_r_rounded) < 1e-10
        i_r = np.where(snap_mask, i_r_rounded, i_r)

        # Interval location + local parameter (1-based Fortran convention)
        n_g = int(self.n_grid[sp])
        i_spl_1based = np.clip(i_r.astype(np.int64), 1, n_g - 1)
        t = i_r - i_spl_1based

        # Convert to 0-based Python array index
        i_spl = i_spl_1based - 1

        # Horner evaluation: S = c1 + t*(c2 + t*(c3 + t*c4))
        c = spline[i_fn]  # (4, n_max_grid)
        c1, c2, c3, c4 = c[0, i_spl], c[1, i_spl], c[2, i_spl], c[3, i_spl]
        result[mask] = c1 + t * (c2 + t * (c3 + t * c4))
        return result

    def evaluate_u(
        self,
        i_fn: int,
        r: np.ndarray,
        species_of_fn: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Evaluate u(r) for radial function *i_fn* at distances *r*.

        Parameters
        ----------
        i_fn : int
            Global 0-based radial function index.
        r : np.ndarray
            Distances in bohr.
        species_of_fn : np.ndarray, optional
            ``(n_basis_fns,)`` int array mapping each radial function to
            its 0-based species index (from ``AimspyInfo.basisfn_species``).
            Defaults to the map attached to this object at capture time.

        Returns
        -------
        np.ndarray
            u(r) values, same shape as *r*.  Zero outside
            ``[r_grid_min, outer_radius]``.
        """
        return self._evaluate_spline_fn(self.spline_wave, i_fn, r, species_of_fn)

    def evaluate_kinetic(
        self,
        i_fn: int,
        r: np.ndarray,
        species_of_fn: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Evaluate the kinetic term (e−v)·u(r) of radial function *i_fn*.

        This is the (eigenvalue − potential)·u product tabulated during
        basis generation and exported as ``spline_kinetic``; units
        Hartree·bohr^(−1/2).  Same domain mask as :meth:`evaluate_u`.
        """
        return self._evaluate_spline_fn(self.spline_kinetic, i_fn, r, species_of_fn)

    def evaluate_deriv(
        self,
        i_fn: int,
        r: np.ndarray,
        species_of_fn: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Evaluate the aims-native du/dr spline of radial function *i_fn*.

        NOTE: FHI-aims builds ``spline_deriv`` only when
        ``use_basis_gradients`` is on (or x2c/q4c relativity); otherwise
        the exported array is **all zeros**.  For a derivative that is
        always available use :meth:`evaluate_du_dr`, which differentiates
        ``spline_wave`` analytically — where ``spline_deriv`` is non-zero
        the two agree to spline accuracy.
        """
        return self._evaluate_spline_fn(self.spline_deriv, i_fn, r, species_of_fn)

    def evaluate_phi(
        self,
        i_fn: int,
        r: np.ndarray,
        species_of_fn: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Evaluate φ(r) = u(r)/r (the actual radial wavefunction)."""
        u = self.evaluate_u(i_fn, r, species_of_fn)
        r = np.asarray(r, dtype=np.float64)
        r_safe = np.maximum(r, 1e-30)
        return np.where(r > 0, u / r_safe, 0.0)

    def evaluate_du_dr(
        self,
        i_fn: int,
        r: np.ndarray,
        species_of_fn: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Evaluate du/dr (physical radial derivative).

        The *analytic* derivative of the stored ``spline_wave`` cubic
        spline is used, via the chain rule
        du/dr = (du/di) / (α·r), where du/di is the spline derivative
        with respect to the grid index (α = ln(r_grid_inc)).  This is
        the same quantity FHI-aims tabulates separately in
        ``spline_deriv`` (available — when built — via
        :meth:`evaluate_deriv`); the two agree to spline accuracy, but
        are not bit-identical since ``spline_deriv`` is itself a cubic
        splined *tabulated* derivative while this one is the analytic
        derivative of the wave spline.
        """
        sp = int(self._resolve_species_of_fn(species_of_fn)[i_fn])
        r = np.asarray(r, dtype=np.float64)
        result = np.zeros_like(r)

        # Same domain guard as evaluate_u (no backward extrapolation).
        mask = (r >= self.r_grid_min[sp]) & (r <= self.outer_radius[i_fn])
        if not np.any(mask):
            return result
        r_m = r[mask]

        log_inc = np.log(self.r_grid_inc[sp])
        i_r = 1.0 + np.log(r_m / self.r_grid_min[sp]) / log_inc

        # Snap to nearest integer when within floating-point epsilon
        i_r_rounded = np.round(i_r)
        snap_mask = np.abs(i_r - i_r_rounded) < 1e-10
        i_r = np.where(snap_mask, i_r_rounded, i_r)

        n_g = int(self.n_grid[sp])
        i_spl_1based = np.clip(i_r.astype(np.int64), 1, n_g - 1)
        t = i_r - i_spl_1based

        # Convert to 0-based Python array index
        i_spl = i_spl_1based - 1

        c = self.spline_wave[i_fn]
        # S'(t) = c2 + 2*c3*t + 3*c4*t²  (derivative w.r.t. grid index)
        du_di = c[1, i_spl] + 2.0 * c[2, i_spl] * t + 3.0 * c[3, i_spl] * t**2
        # Chain rule: du/dr = (du/di) / (α·r)
        result[mask] = du_di / (log_inc * r_m)
        return result

    # ==================================================================
    # H5 export
    # ==================================================================
    def save_h5(
        self,
        path: Union[str, Path],
        info,  # AimspyInfo
    ) -> Dict[str, bool]:
        """Save basis data to an HDF5 file, one group per element.

        The file is created if it does not exist; existing elements are
        **not** overwritten (incremental add only).  The file can be
        shared across multiple calculations to build a basis library.

        Parameters
        ----------
        path : str or Path
            Path to the H5 file.
        info : AimspyInfo
            The info snapshot (for species metadata and identity arrays).

        Returns
        -------
        dict[str, bool]
            Mapping from element symbol to whether it was newly added
            (``True``) or skipped because it already existed (``False``).
        """
        import h5py

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Compute per-species function ranges.
        # NOTE: this assumes each species' radial functions occupy a
        # contiguous index range in basisfn_* — guaranteed by aims'
        # shrink_fixed_basis_phi_thresh loop ordering (species outer loop).
        # The contiguity is validated below so a future reordering on
        # either side fails loudly instead of silently exporting another
        # species' functions under the wrong element.
        species_fn_ranges: List[Tuple[int, int]] = []  # (start, end) per species
        for sp in range(self.n_species):
            mask = info.basisfn_species == sp
            indices = np.where(mask)[0]
            if len(indices) > 0:
                if not np.array_equal(indices, np.arange(indices[0], indices[-1] + 1)):
                    raise ValueError(
                        f"species {sp} ({info.species_elements[sp]}): radial basis "
                        f"functions are not contiguous in basisfn_species "
                        f"(indices {indices[:5].tolist()}...); save_h5 requires "
                        "species-major ordering"
                    )
                species_fn_ranges.append((int(indices[0]), int(indices[-1]) + 1))
            else:
                species_fn_ranges.append((0, 0))

        results: Dict[str, bool] = {}

        with h5py.File(str(path), "a") as f:
            # Initialize root attrs if new file.  The empty species_list is
            # written with an explicit string dtype (a bare [] would be
            # stored as an empty float64 attribute).
            if "format_version" not in f.attrs:
                f.attrs["format_version"] = "1.0"
                f.attrs["generator"] = "aimspy"
                f.attrs["date"] = datetime.date.today().isoformat()
                f.attrs["units"] = "atomic"
                f.attrs["species_list"] = np.array([], dtype=h5py.string_dtype())
                f.attrs["n_species"] = 0

            for sp in range(self.n_species):
                element = info.species_elements[sp]
                if element in f:
                    results[element] = False
                    continue

                start, end = species_fn_ranges[sp]
                n_fns = end - start
                if n_fns == 0:
                    continue

                # Per-species metadata
                fn_n = info.basisfn_n[start:end].astype(np.int32)
                fn_l = info.basisfn_l[start:end].astype(np.int32)
                fn_type_raw = info.basisfn_type[start:end]
                fn_type = np.array([t.encode("utf-8") for t in fn_type_raw], dtype="S8")
                fn_outer = self.outer_radius[start:end]

                # Zeta numbering: count occurrences of (n, l) pairs
                zeta = np.zeros(n_fns, dtype=np.int32)
                nl_count: Dict[Tuple[int, int], int] = {}
                for i in range(n_fns):
                    key = (int(fn_n[i]), int(fn_l[i]))
                    zeta[i] = nl_count.get(key, 0)
                    nl_count[key] = nl_count.get(key, 0) + 1

                # Element-level summary
                n_orbitals = int(np.sum(2 * fn_l + 1))
                l_max = int(np.max(fn_l))

                # Grid
                sp_r_grid = self.species_r_grid(sp)

                # Spline data (already in (fn, 4, n_grid) layout)
                sp_wave = self.spline_wave[start:end, :, : int(self.n_grid[sp])]
                sp_kinetic = self.spline_kinetic[start:end, :, : int(self.n_grid[sp])]
                sp_deriv = self.spline_deriv[start:end, :, : int(self.n_grid[sp])]

                # Write group
                grp = f.create_group(element)
                grp.attrs["element"] = element
                grp.attrs["z"] = float(info.species_z[sp])
                grp.attrs["r_grid_min"] = float(self.r_grid_min[sp])
                grp.attrs["r_grid_inc"] = float(self.r_grid_inc[sp])
                grp.attrs["n_grid"] = int(self.n_grid[sp])
                grp.attrs["n_basis_rad"] = n_fns
                grp.attrs["n_orbitals"] = n_orbitals
                grp.attrs["l_max"] = l_max

                grp.create_dataset("r_grid", data=sp_r_grid, dtype="f8")
                grp.create_dataset("n", data=fn_n, dtype="i4")
                grp.create_dataset("l", data=fn_l, dtype="i4")
                grp.create_dataset("zeta", data=zeta, dtype="i4")
                grp.create_dataset("type", data=fn_type, dtype="S8")
                grp.create_dataset("outer_radius", data=fn_outer, dtype="f8")
                grp.create_dataset("spline_wave", data=sp_wave, dtype="f8")
                grp.create_dataset("spline_kinetic", data=sp_kinetic, dtype="f8")
                grp.create_dataset("spline_deriv", data=sp_deriv, dtype="f8")

                results[element] = True

            # Update root species list.  Decode entries defensively:
            # h5py 3.0.x returned vlen-string attrs as bytes, which would
            # otherwise make the membership test below always True and
            # accumulate duplicates across saves.
            existing = [
                s.decode("utf-8") if isinstance(s, bytes) else str(s)
                for s in f.attrs.get("species_list", [])
            ]
            for elem, added in results.items():
                if added and elem not in existing:
                    existing.append(elem)
            f.attrs["species_list"] = existing
            f.attrs["n_species"] = len(existing)

        return results
