"""Private — CLI commands for visualization (``viz-basis``, ``viz-grid``).

The actual plotting logic lives in :mod:`aimspy.viz_basis` (file-driven,
runtime-free) and :mod:`aimspy.viz` (GridData-driven).  This module only
wraps them as click commands; ``aimspy.cli`` imports and registers them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import click


# ---------------------------------------------------------------------------
# aimspy viz-basis
# ---------------------------------------------------------------------------
@click.command("viz-basis")
@click.argument("h5_path", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option(
    "--kind",
    type=click.Choice(["u", "phi"]),
    default="u",
    show_default=True,
    help="Plot u(r) or phi(r) = u(r)/r.",
)
@click.option(
    "-j",
    "--jobs",
    type=int,
    default=1,
    show_default=True,
    help="Parallel worker processes (only used with -o; implies Agg).",
)
@click.option(
    "--n-plot",
    type=int,
    default=500,
    show_default=True,
    help="Interpolated points per curve.",
)
@click.option("--split-l", is_flag=True, help="One panel per angular momentum l.")
@click.option(
    "--logx",
    is_flag=True,
    help="Log-scale the x axis (evenly spreads the log-grid sampling).",
)
@click.option("--no-grid", is_flag=True, help="Hide the log-grid rug markers.")
@click.option("--no-type", is_flag=True, help="Omit the type from curve labels.")
@click.option("--bohr", is_flag=True, help="Use bohr instead of Angstrom.")
@click.option("--r-max", type=float, default=None, help="X-axis limit (plot units).")
@click.option(
    "-o",
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="Save one figure per element here as ELEMENT_basis.FMT "
    "(default: show interactively).",
)
@click.option(
    "--fmt",
    type=click.Choice(["png", "pdf", "svg"]),
    default="png",
    show_default=True,
    help="Figure format (with -o).",
)
def viz_basis_cmd(
    h5_path: Path,
    kind: str,
    jobs: int,
    n_plot: int,
    split_l: bool,
    logx: bool,
    no_grid: bool,
    no_type: bool,
    bohr: bool,
    r_max: Optional[float],
    output_dir: Optional[Path],
    fmt: str,
) -> None:
    """Plot radial basis functions for ALL elements in BASIS_H5.

    Each element in the file gets its own figure.  Without -o figures are
    shown interactively; with -o they are saved (nothing is displayed).
    """
    from .viz_basis import list_elements

    elements = list_elements(h5_path)
    if not elements:
        raise click.ClickException(f"no elements found in {h5_path}")

    show = output_dir is None
    if show and jobs != 1:
        click.echo("note: -j applies only with -o; running sequentially.", err=True)
        jobs = 1
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    plot_kwargs = dict(
        kind=kind,
        angstrom=not bohr,
        n_plot=n_plot,
        r_max=r_max,
        split_l=split_l,
        logx=logx,
        show_type=not no_type,
        show_grid=not no_grid,
    )

    if show:
        # Sequential, interactive backend, show all figures at once.
        import matplotlib.pyplot as plt

        from .viz_basis import plot_radial_basis

        for element in elements:
            plot_radial_basis(h5_path, element, **plot_kwargs)
        plt.show()
        return

    tasks = [
        (
            h5_path,
            element,
            {**plot_kwargs, "save": output_dir / f"{element}_basis.{fmt}"},
        )
        for element in elements
    ]

    try:
        if jobs > 1 and len(tasks) > 1:
            from concurrent.futures import ProcessPoolExecutor

            with ProcessPoolExecutor(max_workers=jobs) as pool:
                for element, out_path in pool.map(_plot_element_worker, tasks):
                    click.echo(f"saved {out_path}")
        else:
            for task in tasks:
                element, out_path = _plot_element_worker(task)
                click.echo(f"saved {out_path}")
    except (ValueError, IndexError) as exc:
        # Library-level validation (bad r_max, unknown element, ...)
        # surfacing as a clean CLI error instead of a traceback.
        raise click.ClickException(str(exc))


def _plot_element_worker(task):
    """Render one element's figure and save it (top-level: picklable).

    Forces the Agg backend — worker processes have no display.
    """
    h5_path, element, plot_kwargs = task
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .viz_basis import plot_radial_basis

    out_path = plot_kwargs["save"]
    fig = plot_radial_basis(h5_path, element, **plot_kwargs)
    plt.close(fig)
    return element, out_path


# ---------------------------------------------------------------------------
# aimspy viz-grid
# ---------------------------------------------------------------------------
@click.command("viz-grid")
@click.argument(
    "npz_path", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument("field", default="rho")
@click.option(
    "--mode",
    type=click.Choice(["scatter", "contour", "radial"]),
    default="scatter",
    show_default=True,
    help="scatter: raw grid points on a slice; contour: interpolated "
    "filled contour; radial: |value| vs distance from an atom.",
)
@click.option(
    "--axis",
    type=int,
    default=2,
    show_default=True,
    help="Slice-plane normal (0=x, 1=y, 2=z).",
)
@click.option(
    "--center",
    type=float,
    default=0.0,
    show_default=True,
    help="Slice position along the axis (plot units).",
)
@click.option(
    "--width",
    type=float,
    default=1.0,
    show_default=True,
    help="Half-thickness of the accepted slab (plot units).",
)
@click.option(
    "--nx", type=int, default=200, show_default=True, help="Contour mesh resolution x."
)
@click.option(
    "--ny", type=int, default=200, show_default=True, help="Contour mesh resolution y."
)
@click.option(
    "-l",
    "--log",
    "log_",
    is_flag=True,
    help="Colour by log10(value) — positive fields (rho).",
)
@click.option(
    "--symlog",
    is_flag=True,
    help="Diverging sym-log scale — difference fields (delta_rho).",
)
@click.option(
    "--linthresh",
    type=float,
    default=1e-3,
    show_default=True,
    help="Linear half-width for --symlog (try 0.05 for delta_vks).",
)
@click.option(
    "--point-size",
    type=float,
    default=5.0,
    show_default=True,
    help="scatter mode: marker size (s).",
)
@click.option(
    "--levels",
    type=int,
    default=60,
    show_default=True,
    help="contour mode: number of contour levels.",
)
@click.option(
    "--atom-index",
    type=int,
    default=None,
    help="radial mode: 0-based atom index (required for --mode radial).",
)
@click.option("--bohr", is_flag=True, help="Use bohr instead of Angstrom.")
@click.option("--lin-y", is_flag=True, help="radial mode: linear y axis.")
@click.option(
    "-o",
    "--output",
    type=click.Path(path_type=Path),
    default=None,
    help="Save the figure here (format from suffix) instead of showing it.",
)
def viz_grid_cmd(
    npz_path: Path,
    field: str,
    mode: str,
    axis: int,
    center: float,
    width: float,
    nx: int,
    ny: int,
    log_: bool,
    symlog: bool,
    linthresh: float,
    point_size: float,
    levels: int,
    atom_index: Optional[int],
    bohr: bool,
    lin_y: bool,
    output: Optional[Path],
) -> None:
    """Plot real-space grid data from an NPZ file (GridData.save_npz).

    FIELD is the field name, e.g. rho, delta_rho, vks, vxc.
    """
    from .grid_data import GridData
    from . import viz

    grid = GridData.load_npz(npz_path)

    try:
        if mode == "radial":
            if atom_index is None:
                raise click.ClickException("--mode radial requires --atom-index.")
            ax = viz.radial_profile(
                grid,
                field,
                atom_index=atom_index,
                angstrom=not bohr,
                logy=not lin_y,
            )
            fig = ax.figure
        elif mode == "scatter":
            ax = viz.scatter_slice(
                grid,
                field,
                axis=axis,
                center=center,
                width=width,
                log=log_,
                symlog=symlog,
                linthresh=linthresh,
                angstrom=not bohr,
                s=point_size,
            )
            fig = ax.figure
        else:
            ax = viz.slice_contour(
                grid,
                field,
                axis=axis,
                center=center,
                width=width,
                nx=nx,
                ny=ny,
                log=log_,
                symlog=symlog,
                linthresh=linthresh,
                angstrom=not bohr,
                levels=levels,
            )
            fig = ax.figure
    except (ValueError, IndexError, KeyError) as exc:
        # Unknown field name, out-of-range atom index, bad shapes, ...
        # → clean CLI error instead of a traceback.
        raise click.ClickException(str(exc))

    if output is not None:
        fig.savefig(str(output), dpi=150, bbox_inches="tight")
        click.echo(f"saved {output}")
    else:
        import matplotlib.pyplot as plt

        plt.show()
