"""Public — visualization of NAO radial basis functions from ``basis.h5``.

Reads the element-grouped HDF5 file written by
:meth:`aimspy.BasisData.save_h5` and plots the radial basis functions
*u(r)* (or φ(r) = u(r)/r) of one element per figure.

Design notes
------------
* **File-driven, runtime-free** — everything is read from the H5 file, so
  plots can be made on machines without libaims / MPI.

* Curves are evaluated through the stored **cubic-spline coefficients**
  on a uniform display grid (``n_plot`` points), which is typically denser
  than the logarithmic grid in the physically relevant region and much
  smoother-looking than plotting raw grid values.

* Each radial function is labelled ``nl-z``, e.g. ``1s-0``, ``2p-1``
  (``z`` = zeta index — the repetition count of the same (n, l) pair in
  aims' internal collection order), optionally suffixed with its type
  (``atomic`` / ``hydro`` / ...).

* Colour encodes *l*, line style encodes the function *type*.

* The default ``show_grid=True`` draws a **rug plot** — short grey ticks
  along the bottom of the axes marking the logarithmic grid points within
  the displayed range.  (The log grid puts ~2/3 of its points below
  0.1 Å, so markers on the curves themselves would be unreadable.)
  With ``logx=True`` the x axis is logarithmic, which spreads the
  log-grid sample evenly across the plot — the natural view of these
  basis functions.

matplotlib is imported lazily so that importing :mod:`aimspy` stays cheap.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple, Union

import numpy as np

from .data import BOHR_TO_ANG

__all__ = ["plot_radial_basis", "list_elements"]

_L_TO_STR = {0: "s", 1: "p", 2: "d", 3: "f", 4: "g", 5: "h", 6: "i"}
_L_COLORS = {0: "C0", 1: "C1", 2: "C2", 3: "C3", 4: "C4", 5: "C5", 6: "C6"}
_TYPE_LINESTYLES = {
    "atomic": "-",
    "confined": "--",
    "ionic": "-.",
    "hydro": ":",
    "gaussian": "--",
    "sto": ":",
}


def _decode_sym(sym) -> str:
    """Decode an element symbol from an H5 attr (bytes-safe).

    h5py >= 3 returns ``str`` for variable-length string attributes,
    but older versions (and some future-proofing paths) return ``bytes``.
    """
    if isinstance(sym, bytes):
        return sym.decode("utf-8")
    return str(sym)


def _import_mpl():
    import matplotlib.pyplot as plt  # lazy

    return plt


# ============================================================================
#  Loading
# ============================================================================
def _load_element(h5_path: Union[str, Path], element: str) -> SimpleNamespace:
    """Read one element group of a ``basis.h5`` file into a namespace."""
    import h5py

    with h5py.File(str(h5_path), "r") as f:
        if element not in f:
            available = [_decode_sym(e) for e in f.attrs.get("species_list", [])]
            raise ValueError(
                f"element {element!r} not found in {h5_path}; "
                f"available: {available}"
            )
        grp = f[element]
        return SimpleNamespace(
            element=element,
            z=float(grp.attrs["z"]),
            r_grid_min=float(grp.attrs["r_grid_min"]),
            r_grid_inc=float(grp.attrs["r_grid_inc"]),
            n_grid=int(grp.attrs["n_grid"]),
            n_basis_rad=int(grp.attrs["n_basis_rad"]),
            n_orbitals=int(grp.attrs["n_orbitals"]),
            l_max=int(grp.attrs["l_max"]),
            n=grp["n"][:].astype(int),
            l=grp["l"][:].astype(int),
            zeta=grp["zeta"][:].astype(int),
            type=[t.decode() for t in grp["type"][:]],
            outer_radius=grp["outer_radius"][:],
            r_grid=grp["r_grid"][:],
            spline_wave=grp["spline_wave"][:],  # (n_basis_rad, 4, n_grid)
        )


def list_elements(h5_path: Union[str, Path]) -> list:
    """Return the element symbols available in a ``basis.h5`` file."""
    import h5py

    with h5py.File(str(h5_path), "r") as f:
        return [_decode_sym(e) for e in f.attrs.get("species_list", [])]


# ============================================================================
#  Spline evaluation (from H5-stored coefficients)
# ============================================================================
def _evaluate_spline(
    coeffs: np.ndarray,
    r_min: float,
    r_inc: float,
    n_grid: int,
    outer_radius: float,
    r_bohr: np.ndarray,
) -> np.ndarray:
    """Evaluate u(r) from one function's spline coefficients ``(4, n_grid)``.

    Inverse logarithmic map + integer snapping + Horner — identical to
    ``BasisData.evaluate_u`` but operating on plain arrays read from H5.
    Points outside ``[r_min, outer_radius]`` evaluate to 0 (the spline is
    not defined below the first grid point; extrapolating the first cubic
    segment backwards would blow up).
    """
    r_bohr = np.asarray(r_bohr, dtype=np.float64)
    result = np.zeros_like(r_bohr)

    mask = (r_bohr >= r_min) & (r_bohr <= outer_radius)
    if not np.any(mask):
        return result
    r_m = r_bohr[mask]

    log_inc = np.log(r_inc)
    i_r = 1.0 + np.log(r_m / r_min) / log_inc

    # Snap near-integers (log/exp roundoff at exact grid points).
    i_r_rounded = np.round(i_r)
    snap = np.abs(i_r - i_r_rounded) < 1e-10
    i_r = np.where(snap, i_r_rounded, i_r)

    i_spl_1based = np.clip(i_r.astype(np.int64), 1, n_grid - 1)
    t = i_r - i_spl_1based
    i_spl = i_spl_1based - 1  # 0-based array index

    c1 = coeffs[0, i_spl]
    c2 = coeffs[1, i_spl]
    c3 = coeffs[2, i_spl]
    c4 = coeffs[3, i_spl]
    result[mask] = c1 + t * (c2 + t * (c3 + t * c4))
    return result


def _label_fn(n: int, ell: int, zeta: int, type_str: str, show_type: bool) -> str:
    label = f"{n}{_L_TO_STR.get(ell, f'l{ell}')}-{zeta}"
    if show_type:
        label += f" ({type_str})"
    return label


# ============================================================================
#  Main plot
# ============================================================================
def plot_radial_basis(
    h5_path: Union[str, Path],
    element: str,
    kind: str = "u",
    angstrom: bool = True,
    n_plot: int = 500,
    r_max: Optional[float] = None,
    split_l: bool = False,
    logx: bool = False,
    show_type: bool = True,
    show_grid: bool = True,
    figsize: Optional[Tuple[float, float]] = None,
    save: Optional[Union[str, Path]] = None,
):
    """Plot the radial basis functions of one element from ``basis.h5``.

    Parameters
    ----------
    h5_path : str or Path
        Path to the basis H5 file (element-per-group format).
    element : str
        Element symbol (e.g. ``'Mo'``); must exist in the file.
    kind : {'u', 'phi'}
        Plot the reduced radial function *u(r)* (default, the raw splined
        quantity, normalized so ∫u²dr = 1) or the true radial function
        φ(r) = u(r)/r.
    angstrom : bool
        x axis in Angstrom (default) or bohr.
    n_plot : int
        Number of uniformly spaced evaluation points per curve.
    r_max : float or None
        x axis upper limit in plot units.  Default: the element's largest
        ``outer_radius`` (its global cutoff).
    split_l : bool
        One panel per angular momentum *l* instead of a single overlay.
    logx : bool
        Log-scale the x axis (default False — the natural view for the
        logarithmic grid: the log-grid sample points appear evenly
        spread instead of being squashed into the leftmost decade).
        A log y axis is intentionally not offered: radial basis
        functions carry sign (nodes), and plotting them on a log y
        axis silently hides the negative lobes.
    show_type : bool
        Append the function type (atomic/hydro/...) to labels.
    show_grid : bool
        Draw grey rug ticks at the bottom marking the logarithmic grid
        points within the displayed range (default True).
    figsize, save : figure size / optional save path (matplotlib infers
        the format from the suffix).

    Returns
    -------
    matplotlib.figure.Figure
    """
    plt = _import_mpl()
    basis = _load_element(h5_path, element)

    if kind not in ("u", "phi"):
        raise ValueError(f"kind must be 'u' or 'phi'; got {kind!r}")

    # bohr -> display-unit factor
    factor = BOHR_TO_ANG if angstrom else 1.0
    unit = "Å" if angstrom else "bohr"

    if r_max is None:
        r_display_max = float(np.max(basis.outer_radius)) * factor
    else:
        r_display_max = float(r_max)
        if r_display_max <= 0.0:
            raise ValueError(f"r_max must be positive; got {r_max!r}")

    # Plot grid (display units).  Linear x: uniform linspace from 0.
    # Log x: uniform *linspace in log r* anchored at r_grid_min, so the
    # sampling follows the logarithmic grid density (even spread, no
    # big empty decade at small r).
    if logx:
        r_plot_display_min = float(basis.r_grid_min) * factor
        if r_display_max <= r_plot_display_min:
            raise ValueError(
                f"r_max ({r_display_max:.3g}) must exceed the first grid point "
                f"({r_plot_display_min:.3g}) when logx=True — a log axis cannot "
                "start at or below the grid origin."
            )
    else:
        r_plot_display_min = 0.0
    if logx:
        r_plot = np.geomspace(r_plot_display_min, r_display_max, n_plot)
    else:
        r_plot = np.linspace(r_plot_display_min, r_display_max, n_plot)
    r_plot_bohr = r_plot / factor

    def _eval(i_fn: int) -> np.ndarray:
        u = _evaluate_spline(
            basis.spline_wave[i_fn],
            basis.r_grid_min,
            basis.r_grid_inc,
            basis.n_grid,
            float(basis.outer_radius[i_fn]),
            r_plot_bohr,
        )
        if kind == "phi":
            return np.where(r_plot_bohr > 0.0, u / np.maximum(r_plot_bohr, 1e-30), 0.0)
        return u

    n_fns = basis.n_basis_rad
    yname = "u(r)" if kind == "u" else "φ(r) = u(r)/r"
    title = f"{element} (Z={basis.z:.0f}) — NAO radial basis: {yname}"

    # ---- figure / axes ----
    if split_l:
        n_panels = basis.l_max + 1
        fig, axes = plt.subplots(
            1,
            n_panels,
            figsize=figsize or (3.4 * n_panels, 4.0),
            sharey=True,
            squeeze=False,
        )
        panel_list = list(axes[0])
    else:
        fig, ax0 = plt.subplots(figsize=figsize or (8.0, 5.0))
        panel_list = [ax0]

    for i_fn in range(n_fns):
        # target panel: all curves, or only the matching l panel
        if split_l:
            targets = [panel_list[basis.l[i_fn]]]
        else:
            targets = [panel_list[0]]

        vals = _eval(i_fn)
        label = _label_fn(
            basis.n[i_fn],
            basis.l[i_fn],
            basis.zeta[i_fn],
            basis.type[i_fn],
            show_type,
        )
        color = _L_COLORS.get(basis.l[i_fn], "gray")
        ls = _TYPE_LINESTYLES.get(basis.type[i_fn], "-")
        for ax in targets:
            ax.plot(r_plot, vals, color=color, ls=ls, lw=1.2, label=label)

    # ---- per-panel cosmetics ----
    for i_panel, ax in enumerate(panel_list):
        if split_l:
            ax.set_title(f"l = {i_panel} ({_L_TO_STR.get(i_panel, '?')})", fontsize=10)
        if logx:
            ax.set_xscale("log")
            ax.set_xlim(r_plot_display_min, r_display_max)
        else:
            ax.set_xlim(0.0, r_display_max)
        ax.set_xlabel(f"r ({unit})")
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ncol = 1 if len(handles) <= 6 else 2
            ax.legend(fontsize=7, ncol=ncol, framealpha=0.85)
        if show_grid:
            _add_rug(ax, basis.r_grid, factor, r_display_max)

    if not split_l:
        panel_list[0].set_ylabel(yname)
        panel_list[0].set_title(title, fontsize=11)
    else:
        fig.suptitle(title, fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.94))

    if save is not None:
        fig.savefig(str(save), dpi=150, bbox_inches="tight")
    return fig


def _add_rug(ax, r_grid: np.ndarray, factor: float, r_display_max: float) -> None:
    """Grey rug ticks along the bottom of *ax* marking the log-grid points.

    Only points within the displayed x range are drawn.  The ticks live in
    axes-fraction y coordinates (0 .. 0.02) so they stay visible on both
    linear and log y axes.
    """
    from matplotlib.transforms import blended_transform_factory

    r_disp = r_grid * factor
    r_disp = r_disp[(r_disp > 0.0) & (r_disp <= r_display_max)]
    if r_disp.size == 0:
        return
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.vlines(
        r_disp,
        0.0,
        0.02,
        transform=trans,
        colors="gray",
        alpha=0.35,
        lw=0.4,
        clip_on=False,
    )
