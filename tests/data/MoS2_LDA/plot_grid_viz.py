#!/usr/bin/env python
"""Regenerate all grid-data visualization figures for the MoS2_LDA example.

Reads the gathered grid dataset (``grid_gathered.npz``, produced by running
``tests/test_grid_data_capture.py``) and writes the figures into
``viz_examples/`` next to this script.

Prerequisite — produce the dataset first (from the pyapi root)::

    export AIMSPY_TEST_AIMS_LIBPATH=/path/to/libaims.so
    mpiexec -np 4 python tests/test_grid_data_capture.py

Then render (single process, no MPI needed)::

    python tests/data/MoS2_LDA/plot_grid_viz.py

Notes
-----
* ``delta_rho`` / ``delta_vks`` are difference fields whose signal
  concentrates near zero yet spans decades — they are plotted with a
  symmetric-log colour scale (``symlog=True``) so the weak far-field signal
  is not flattened against the strong near-nucleus signal.
* The pure scatter plots show the atom-centred grid itself (no interpolation).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering

import matplotlib.pyplot as plt
import numpy as np

# pyapi root on sys.path so `import aimspy` works when run from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from aimspy import GridData, viz  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "grid_gathered.npz"
OUT = HERE / "viz_examples"

# Upper-layer Mo grid-point peak (z, Angstrom) in the centre cell — MoS2 is a
# layered structure with grid points clustered near z ~ +-9..11.5 A and a
# vacuum gap in between (no points there, by construction of the atom-centred
# grid).
Z_MO = 11.15
SLAB = 1.2  # half-thickness of the Mo-layer slice (Angstrom)

# Atom-colouring for the pure grid scatter plots.
_ATOM_COLOR = {0: "tab:blue", 1: "tab:red", 2: "tab:green"}
_ATOM_LABEL = {0: "S(0)", 1: "Mo(1)", 2: "S(2)"}


def _atom_scatter(ax, gd, in_plane_mask, title):
    coords = gd.coords_ang
    for a in range(gd.n_atoms):
        sel = in_plane_mask & (gd.index_atom == a)
        if sel.sum() > 0:
            ax.scatter(
                coords[0, sel],
                coords[1, sel],
                s=3,
                c=_ATOM_COLOR.get(a, "k"),
                label=_ATOM_LABEL.get(a, str(a)),
                alpha=0.6,
            )
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend(markerscale=8)


def main() -> None:
    if not DATA.is_file():
        sys.exit(
            f"dataset not found: {DATA}\n"
            "run first:  mpiexec -np 4 python tests/test_grid_data_capture.py"
        )
    OUT.mkdir(exist_ok=True)
    gd = GridData.load_npz(DATA)
    coords = gd.coords_ang
    n_mo = int((np.abs(coords[2] - Z_MO) <= SLAB).sum())
    print(f"loaded {gd};  Mo-layer slice z={Z_MO}+-{SLAB} A -> {n_mo} pts")

    # ---- 1. pure grid scatter, x-z (cross-layer) --------------------------
    fig, ax = plt.subplots(figsize=(7, 7))
    for a in range(gd.n_atoms):
        sel = gd.index_atom == a
        ax.scatter(
            coords[0, sel],
            coords[2, sel],
            s=1,
            c=_ATOM_COLOR.get(a, "k"),
            label=_ATOM_LABEL.get(a, str(a)),
            alpha=0.5,
        )
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("z (Å)")
    ax.set_aspect("equal")
    ax.set_title(f"Grid points ({gd.n_full_points} pts, atom-coloured)")
    ax.legend(markerscale=10)
    fig.savefig(OUT / "grid_scatter_xz.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  grid_scatter_xz.png")

    # ---- 2. pure grid scatter, x-y (Mo layer) -----------------------------
    mask = np.abs(coords[2] - Z_MO) <= SLAB
    fig, ax = plt.subplots(figsize=(6, 6))
    _atom_scatter(
        ax, gd, mask, f"Grid points, z~{Z_MO} A (Mo layer) [{mask.sum()} pts]"
    )
    fig.savefig(OUT / "grid_scatter_z.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("  grid_scatter_z.png")

    # ---- 3. delta_rho scatter (symlog) ------------------------------------
    ax = viz.scatter_slice(
        gd,
        "delta_rho",
        axis=2,
        center=Z_MO,
        width=SLAB,
        symlog=True,
        linthresh=1e-3,
        s=5,
    )
    plt.gcf().savefig(OUT / "drho_scatter_symlog.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  drho_scatter_symlog.png")

    # ---- 4. delta_rho contour (symlog, x-y Mo layer) ----------------------
    ax = viz.slice_contour(
        gd,
        "delta_rho",
        axis=2,
        center=Z_MO,
        width=SLAB,
        nx=200,
        ny=200,
        symlog=True,
        linthresh=1e-3,
        method="linear",
        levels=60,
    )
    plt.gcf().savefig(OUT / "drho_contour_symlog.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  drho_contour_symlog.png")

    # ---- 5. delta_vks scatter (symlog) ------------------------------------
    ax = viz.scatter_slice(
        gd,
        "delta_vks",
        axis=2,
        center=Z_MO,
        width=SLAB,
        symlog=True,
        linthresh=0.05,
        s=5,
    )
    plt.gcf().savefig(OUT / "dvks_scatter_symlog.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  dvks_scatter_symlog.png")

    # ---- 6. delta_vks contour (symlog, x-y Mo layer) ----------------------
    ax = viz.slice_contour(
        gd,
        "delta_vks",
        axis=2,
        center=Z_MO,
        width=SLAB,
        nx=200,
        ny=200,
        symlog=True,
        linthresh=0.05,
        method="linear",
        levels=60,
    )
    plt.gcf().savefig(OUT / "dvks_contour_z_symlog.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  dvks_contour_z_symlog.png")

    # ---- 7. delta_vks contour (symlog, x-z cross-layer) -------------------
    ax = viz.slice_contour(
        gd,
        "delta_vks",
        axis=1,
        center=0.0,
        width=1.0,
        nx=150,
        ny=250,
        symlog=True,
        linthresh=0.05,
        method="linear",
        levels=60,
    )
    plt.gcf().savefig(OUT / "dvks_contour_xz_symlog.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  dvks_contour_xz_symlog.png")

    print(f"all figures written to {OUT}")


if __name__ == "__main__":
    main()
