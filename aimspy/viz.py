"""Public — visualization helpers for :class:`~aimspy.GridData`.

Lightweight plotting for the atom-centred (Delley radial x Lebedev angular)
integration grid and the scalar fields living on it (``rho``, ``vks``,
``delta_rho``, ``vxc``, ...).

Design notes
------------
* The grid is **non-uniform and non-Cartesian** (dense near nuclei, sparse in
  the far field), and ``rho`` spans many orders of magnitude.  Two families
  of plots are therefore provided:

  - **scatter** (:func:`scatter_slice`) — zero interpolation, faithful to the
    raw grid values (best for diagnosing the grid itself);
  - **interpolated contour** (:func:`slice_contour`) — interpolates a field
    onto a regular 2-D mesh for publication-quality cuts (use ``log=True``
    for density / potential magnitudes).

* ``rho`` has a huge dynamic range (~1e-30 .. 1e4 e/bohr^3).  Always use
  ``log=True`` (or pass a pre-transformed array) for meaningful density plots.

* 3-D isosurfaces (:func:`isosurface`) require the optional ``pyvista``
  package (``pip install pyvista``); it is imported lazily and raises a clear
  error if unavailable.

All functions accept either a field **name** (str, looked up on the
:class:`GridData`) or an explicit 1-D ``(n,)`` array of per-point values.
matplotlib is imported lazily so that importing :mod:`aimspy` stays cheap.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

from .grid_data import GridData
from .data import BOHR_TO_ANG

__all__ = [
    "scatter_slice",
    "slice_contour",
    "radial_profile",
    "isosurface",
]

# Field-name resolver: name -> (n,) array of per-point scalar values.
# 2-D fields (n_spin, n) take spin channel 0 by default.
_FIELD_NAMES = (
    "rho",
    "rho_free",
    "delta_rho",
    "vks",
    "vks0",
    "delta_vks",
    "vh",
    "vh0",
    "delta_vh",
    "vxc",
    "vxc0",
)


def _resolve_values(grid: GridData, value: Union[str, np.ndarray]) -> np.ndarray:
    """Return a 1-D ``(n,)`` array of per-point scalar values.

    *value* may be a field name (see ``_FIELD_NAMES``) or an explicit array.
    Spin-resolved fields use spin channel 0 (adequate for n_spin=1; pass an
    explicit array to select another channel).
    """
    if isinstance(value, str):
        name = value
        if not hasattr(grid, name):
            raise ValueError(
                f"unknown GridData field {name!r}; " f"available: {list(_FIELD_NAMES)}"
            )
        arr = np.asarray(getattr(grid, name))
    else:
        arr = np.asarray(value)

    if arr.ndim == 2:  # (n_spin, n) -> take spin channel 0
        arr = arr[0]
    if arr.ndim != 1 or arr.shape[0] != grid.n_full_points:
        raise ValueError(
            f"field must reduce to shape (n_full_points={grid.n_full_points},); "
            f"got {arr.shape}"
        )
    return arr


def _import_mpl():
    import matplotlib.pyplot as plt  # lazy

    return plt


def _resolve_norm(
    vals: np.ndarray,
    log: bool,
    symlog: bool,
    linthresh: float,
):
    """Build a matplotlib normalizer for *vals*.

    Returns ``(norm, plot_vals, scale_tag)`` where *plot_vals* is the array
    actually handed to the artist and *scale_tag* describes the mapping for
    the colorbar label.

    Precedence: ``symlog`` > ``log`` > linear.

    * ``symlog``: :class:`~matplotlib.colors.SymLogNorm` — diverging, keeps
      the sign, linear within ``±linthresh`` and logarithmic outside.  Best
      for difference fields (``delta_rho`` / ``delta_vks``) whose signal is
      concentrated near zero but spans decades.
    * ``log``: pre-transform ``log10(max(vals, tiny))`` and use a linear
      normalizer — for strictly-positive fields (``rho``, |V|).
    * linear: ``vals`` used as-is.
    """
    from matplotlib.colors import SymLogNorm

    if symlog:
        vmax = float(np.max(np.abs(vals)))
        if vmax <= 0.0:
            vmax = 1.0
        norm = SymLogNorm(linthresh=linthresh, linscale=1.0, vmin=-vmax, vmax=vmax)
        return norm, vals, "symlog"

    if log:
        tiny = np.max(vals) * 1e-30 if np.max(vals) > 0 else 1e-30
        plot_vals = np.log10(np.maximum(vals, tiny))
        return None, plot_vals, "log10"

    return None, vals, ""


# ============================================================================
#  1. Scatter slice (no interpolation — faithful to raw grid)
# ============================================================================
def scatter_slice(
    grid: GridData,
    value: Union[str, np.ndarray],
    axis: int = 2,
    center: float = 0.0,
    width: float = 1.0,
    log: bool = False,
    symlog: bool = False,
    linthresh: float = 1e-3,
    cmap: Optional[str] = None,
    angstrom: bool = True,
    s: float = 2.0,
    ax=None,
    colorbar: bool = True,
):
    """Scatter-plot a field on grid points near a plane (no interpolation).

    Selects points with ``|coords[axis] - center| <= width`` and plots them in
    the remaining two coordinates, coloured by the field value.

    Parameters
    ----------
    grid : GridData
    value : str or (n,) array
        Field to colour by (e.g. ``'rho'``, ``'vks'``, ``'delta_rho'``).
    axis : int
        Normal of the slicing plane (0=x, 1=y, 2=z).
    center : float
        Plane position along *axis* (same unit as ``angstrom`` flag).
    width : float
        Half-thickness of the slab of accepted points.
    log : bool
        Colour by ``log10(value)`` — for strictly-positive fields (``rho``).
    symlog : bool
        Diverging symmetric-log colour scale (linear within ``±linthresh``,
        logarithmic outside).  **Recommended for difference fields**
        (``delta_rho`` / ``delta_vks``): their signal concentrates near zero
        yet spans decades, so a plain linear scale makes most points look
        uniformly ~0.  Takes precedence over ``log``.
    linthresh : float
        Linear region half-width for ``symlog`` (default 1e-3).
    cmap : str or None
        Colormap.  Default: ``'RdBu_r'`` when the field has both signs
        (or ``symlog=True``), else ``'viridis'``.
    angstrom : bool
        Interpret *center*/*width* and plot axes in Angstrom (default) / bohr.
    s : float
        Marker size.
    ax, colorbar : matplotlib target / toggle

    Returns
    -------
    ax : matplotlib Axes
    """
    plt = _import_mpl()
    coords = grid.coords_ang if angstrom else grid.coords
    vals = _resolve_values(grid, value)
    label = value if isinstance(value, str) else "value"

    if cmap is None:
        cmap = "RdBu_r" if (symlog or np.any(vals < 0)) else "viridis"

    norm, plot_vals, scale_tag = _resolve_norm(vals, log, symlog, linthresh)
    if scale_tag:
        label = (
            f"{scale_tag}({label})"
            if scale_tag == "log10"
            else f"{label} ({scale_tag})"
        )

    mask = np.abs(coords[axis] - center) <= width
    others = [i for i in range(3) if i != axis]
    x, y, v = coords[others[0]][mask], coords[others[1]][mask], plot_vals[mask]

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    sc = ax.scatter(x, y, c=v, s=s, cmap=cmap, norm=norm)
    unit = "Å" if angstrom else "bohr"
    ax.set_xlabel(f"{'xyz'[others[0]]} ({unit})")
    ax.set_ylabel(f"{'xyz'[others[1]]} ({unit})")
    ax.set_aspect("equal")
    ax.set_title(f"{label}  [{mask.sum()} pts, {'xyz'[axis]}={center}±{width} {unit}]")
    if colorbar:
        plt.colorbar(sc, ax=ax, label=label)
    return ax


# ============================================================================
#  2. Interpolated contour slice (publication quality)
# ============================================================================
def slice_contour(
    grid: GridData,
    value: Union[str, np.ndarray],
    axis: int = 2,
    center: float = 0.0,
    width: float = 1.0,
    nx: int = 200,
    ny: int = 200,
    log: bool = False,
    symlog: bool = False,
    linthresh: float = 1e-3,
    angstrom: bool = True,
    levels: Union[int, Sequence[float]] = 50,
    cmap: Optional[str] = None,
    method: str = "linear",
    ax=None,
    colorbar: bool = True,
):
    """Interpolate a field onto a regular 2-D mesh and draw a filled contour.

    Points within ``|coords[axis]-center| <= width`` are interpolated with
    :func:`scipy.interpolate.griddata` onto an ``nx`` x ``ny`` mesh spanning
    the data extent in the remaining two coordinates.

    Parameters
    ----------
    grid : GridData
    value : str or (n,) array
    axis, center, width, log, symlog, linthresh, cmap, angstrom, ax, colorbar
        See :func:`scatter_slice`.  Use ``symlog=True`` for difference fields
        (``delta_rho`` / ``delta_vks``); the norm is built on the *raw* values
        before interpolation so the colour mapping stays faithful.
    nx, ny : int
        Interpolation mesh resolution.
    levels : int or sequence
        Contour levels (passed to ``contourf``).  For ``symlog`` an integer
        count is recommended (matplotlib spaces them per the norm).
    method : {'linear', 'nearest', 'cubic'}
        ``griddata`` interpolation method.  ``'linear'`` is a good default;
        ``'nearest'`` avoids overshoot in sparse regions.

    Returns
    -------
    ax : matplotlib Axes
    """
    from scipy.interpolate import griddata

    plt = _import_mpl()
    coords = grid.coords_ang if angstrom else grid.coords
    vals = _resolve_values(grid, value)
    label = value if isinstance(value, str) else "value"

    if cmap is None:
        cmap = "RdBu_r" if (symlog or np.any(vals < 0)) else "viridis"

    # Build the norm on the raw values so the colour mapping is faithful;
    # interpolate the (possibly log-transformed) plot values.
    norm, plot_vals, scale_tag = _resolve_norm(vals, log, symlog, linthresh)
    if scale_tag:
        label = (
            f"{scale_tag}({label})"
            if scale_tag == "log10"
            else f"{label} ({scale_tag})"
        )

    mask = np.abs(coords[axis] - center) <= width
    others = [i for i in range(3) if i != axis]
    xs, ys, vs = coords[others[0]][mask], coords[others[1]][mask], plot_vals[mask]

    if xs.size == 0:
        raise ValueError(
            f"no grid points within {'xyz'[axis]}={center}±{width} "
            f"({'Å' if angstrom else 'bohr'}); widen `width` or move `center`"
        )

    xi = np.linspace(xs.min(), xs.max(), nx)
    yi = np.linspace(ys.min(), ys.max(), ny)
    X, Y = np.meshgrid(xi, yi)
    Z = griddata((xs, ys), vs, (X, Y), method=method)

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))
    cf = ax.contourf(X, Y, Z, levels=levels, cmap=cmap, norm=norm)
    unit = "Å" if angstrom else "bohr"
    ax.set_xlabel(f"{'xyz'[others[0]]} ({unit})")
    ax.set_ylabel(f"{'xyz'[others[1]]} ({unit})")
    ax.set_aspect("equal")
    ax.set_title(f"{label}  [{'xyz'[axis]}={center} {unit}]")
    if colorbar:
        plt.colorbar(cf, ax=ax, label=label)
    return ax


# ============================================================================
#  3. Radial profile (quantitative shell structure / asymptotics)
# ============================================================================
def radial_profile(
    grid: GridData,
    value: Union[str, np.ndarray],
    atom_index: Optional[int] = None,
    center: Optional[Sequence[float]] = None,
    angstrom: bool = True,
    logy: bool = True,
    marker: str = ".",
    ms: float = 2.0,
    ax=None,
    label: Optional[str] = None,
):
    """Plot a field versus radial distance from an atom or a point.

    Exactly one of *atom_index* / *center* must be given.  *atom_index* uses
    the true nuclear position from ``grid.atom_coords`` (preferred, exact)
    or, as a fallback when the structure fields are absent, the centroid of
    that atom's grid points (0-based index into ``grid.index_atom``).
    *center* is an explicit ``(x, y, z)``.

    Parameters
    ----------
    grid : GridData
    value : str or (n,) array
    atom_index : int or None
        0-based atom index (matches ``grid.index_atom``).
    center : (3,) sequence or None
        Explicit centre (same unit as ``angstrom`` flag).
    angstrom : bool
        Plot radius in Angstrom (default) or bohr.
    logy : bool
        Log-scale the y axis (recommended for ``rho``).
    marker, ms : matplotlib scatter style
    ax, label : matplotlib target / legend label

    Returns
    -------
    ax : matplotlib Axes
    """
    plt = _import_mpl()
    coords = grid.coords_ang if angstrom else grid.coords
    vals = _resolve_values(grid, value)
    name = label or (value if isinstance(value, str) else "value")

    if atom_index is not None:
        idx = int(atom_index)
        if grid.atom_coords is not None:
            # Prefer the exact nuclear position (in-memory structure).
            # atom_coords is stored in Angstrom; convert to bohr when the
            # plot uses bohr units.
            ctr_ang = np.asarray(grid.atom_coords[idx], dtype=float)
            ctr = ctr_ang if angstrom else ctr_ang / BOHR_TO_ANG
        else:
            # Fallback: centroid of this atom's grid points.  This can be
            # off-centre for light atoms with asymmetric integration grids.
            sel = grid.index_atom == idx
            if not np.any(sel):
                raise ValueError(f"no grid points for atom_index={idx}")
            ctr = coords[:, sel].mean(axis=1)
    elif center is not None:
        ctr = np.asarray(center, dtype=float)
        if ctr.shape != (3,):
            raise ValueError("center must be a (3,) sequence")
    else:
        raise ValueError("provide exactly one of atom_index / center")

    r = np.linalg.norm(coords - ctr[:, None], axis=0)
    order = np.argsort(r)
    r, v = r[order], vals[order]

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    if logy:
        ax.semilogy(r, np.abs(v), marker, ms=ms, label=name)
    else:
        ax.plot(r, v, marker, ms=ms, label=name)
    unit = "Å" if angstrom else "bohr"
    ax.set_xlabel(f"r ({unit})")
    ax.set_ylabel(f"|{name}|" if logy else name)
    if label is not None or isinstance(value, str):
        ax.legend()
    return ax


# ============================================================================
#  4. 3-D isosurface (optional: requires pyvista)
# ============================================================================
def isosurface(
    grid: GridData,
    value: Union[str, np.ndarray],
    iso: Union[float, Sequence[float]],
    nx: int = 80,
    ny: int = 80,
    nz: int = 80,
    log: bool = False,
    angstrom: bool = True,
    off_screen: bool = True,
    screenshot: Optional[str] = None,
    cmap: str = "viridis",
    opacity: float = 0.9,
    show: bool = True,
):
    """Render 3-D isosurface(s) of a field (requires ``pyvista``).

    The scattered grid values are first interpolated onto a regular
    ``nx`` x ``ny`` x ``nz`` mesh (scipy ``griddata``), then wrapped as a
    :class:`pyvista.ImageData` and contoured with ``contour([iso])``.

    .. note::
        ``pyvista`` is an **optional** dependency.  Install it with
        ``pip install pyvista``.  This function raises :class:`ImportError`
        with a clear message if it is not installed.

    Parameters
    ----------
    grid : GridData
    value : str or (n,) array
    iso : float or sequence of float
        Isosurface value(s), in the (possibly log-transformed) field units.
    nx, ny, nz : int
        Interpolation mesh resolution.
    log : bool
        If True, isosurface ``log10(max(value, tiny))`` (recommended for rho).
    angstrom : bool
        Interpolate in Angstrom-scaled coordinates (default) or bohr.
    off_screen : bool
        Render off-screen (headless; default True — needed on clusters).
    screenshot : str or None
        If given, save a PNG screenshot to this path.
    cmap, opacity : appearance
    show : bool
        Call ``plotter.show()``.  Set False to only build/return the plotter.

    Returns
    -------
    plotter : pyvista.Plotter
    """
    try:
        import pyvista as pv
    except ImportError as e:
        raise ImportError(
            "isosurface() requires the optional 'pyvista' package. "
            "Install it with:  pip install pyvista"
        ) from e
    from scipy.interpolate import griddata

    coords = grid.coords_ang if angstrom else grid.coords
    vals = _resolve_values(grid, value)
    label = value if isinstance(value, str) else "value"

    if log:
        tiny = np.max(vals) * 1e-30 if np.max(vals) > 0 else 1e-30
        vals = np.log10(np.maximum(vals, tiny))
        label = f"log10({label})"

    # Interpolate scattered field onto a regular mesh.
    xs = np.linspace(coords[0].min(), coords[0].max(), nx)
    ys = np.linspace(coords[1].min(), coords[1].max(), ny)
    zs = np.linspace(coords[2].min(), coords[2].max(), nz)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    V = griddata((coords[0], coords[1], coords[2]), vals, (X, Y, Z), method="linear")

    mesh = pv.ImageData(
        dimensions=(nx, ny, nz),
        spacing=(xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0]),
        origin=(xs[0], ys[0], zs[0]),
    )
    mesh.point_data[label] = V.ravel(order="F")
    mesh = mesh.set_active_scalars(label)

    iso_list = [float(iso)] if np.isscalar(iso) else [float(v) for v in iso]
    surf = mesh.contour(iso_list)

    pl = pv.Plotter(off_screen=off_screen)
    pl.add_mesh(surf, cmap=cmap, opacity=opacity, scalars=label)
    unit = "Å" if angstrom else "bohr"
    pl.add_text(f"{label} isosurface {iso_list} ({unit})", font_size=10)
    if screenshot:
        pl.show(screenshot=screenshot, auto_close=True)
    elif show:
        pl.show()
    return pl
