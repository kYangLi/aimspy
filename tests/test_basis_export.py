#!/usr/bin/env python
"""Integration test: capture_basis_data=True (NAO radial basis export).

Runs a plain LDA SCF with ``capture_basis_data=True``, then verifies:

  1. BasisData is captured and has correct dimensions
  2. Grid parameters are consistent (r_grid rebuild matches)
  3. Spline evaluation at grid points matches coeff[0] (identity c1=f),
     for the wave / kinetic / deriv channels alike
  3c. evaluate_deriv agrees with evaluate_du_dr (spline accuracy)
  3d. evaluate_phi == evaluate_u / r
  4. Radial functions are normalized (int u^2 dr ~ 1)
  5. Same-species same-l functions are orthogonal
  6. H5 save produces valid file with correct structure
  7. Re-running skips existing elements (incremental)
  7b. H5 -> viz round trip (list_elements + plot_radial_basis logx)
  8. Post-init registration of export_basis_data warns (already fired)

Prerequisites:
    tests/data/MoS2_LDA/control.in  (xc pw-lda, SCF only)

Usage:
    source /path/to/intel/setvars.sh
    ulimit -s unlimited
    export AIMSPY_TEST_AIMS_LIBPATH=/path/to/libaims.so
    mpiexec -np 8 python tests/test_basis_export.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from mpi4py import MPI

from aimspy import Calculator, CalculatorConfig

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data" / "MoS2_LDA"

comm = MPI.COMM_WORLD
rank = comm.rank

_lib_env = os.environ.get("AIMSPY_TEST_AIMS_LIBPATH")
if not _lib_env:
    if rank == 0:
        print(
            "ERROR: AIMSPY_TEST_AIMS_LIBPATH environment variable not set.\n"
            "  Export the path to your patched libaims.so before running:\n"
            "    export AIMSPY_TEST_AIMS_LIBPATH=/path/to/libaims.so",
            file=sys.stderr,
        )
    comm.Abort(1)
LIB_PATH = Path(_lib_env)

if not (DATA_DIR / "control.in").is_file():
    if rank == 0:
        print(f"ERROR: {DATA_DIR / 'control.in'} not found.", file=sys.stderr)
    sys.exit(1)

all_ok = True


def _info(msg):
    if rank == 0:
        print(msg, flush=True)


def check(name, condition, detail=""):
    global all_ok
    if rank == 0:
        tag = "OK  " if condition else "FAIL"
        line = f"  {tag}  {name}"
        if detail:
            line += f"  — {detail}"
        print(line, flush=True)
    if not condition:
        all_ok = False


# ==========================================================================
# Run calculation with capture_basis_data=True
# ==========================================================================
_info("=== Running MoS2 LDA with capture_basis_data=True ===")

cfg = CalculatorConfig(
    lib_path=LIB_PATH,
    control_path=DATA_DIR / "control.in",
    geometry_path=DATA_DIR / "geometry.in",
    logfile=DATA_DIR / "aims.out",
    capture_basis_data=True,
)

calc = Calculator(cfg)
calc.init()
# basis_data should be available immediately after init() (no calc() needed)

# ==========================================================================
# Verify on rank 0
# ==========================================================================
if rank == 0:
    _info("\n=== Verifying BasisData ===")

    bd = calc.basis_data
    info = calc.info

    # --- 1. Existence and dimensions ---
    check("basis_data captured", bd is not None)
    if bd is None:
        _info("FATAL: basis_data is None, cannot continue")
        sys.exit(1)

    check("n_species == 2 (Mo, S)", bd.n_species == 2, f"got {bd.n_species}")
    check("n_basis_fns > 0", bd.n_basis_fns > 0, f"got {bd.n_basis_fns}")
    check("n_max_spline == 4", bd.n_max_spline == 4)
    check(
        "spline_wave shape",
        bd.spline_wave.shape == (bd.n_basis_fns, 4, bd.n_max_grid),
        f"got {bd.spline_wave.shape}",
    )
    check(
        "spline_kinetic shape",
        bd.spline_kinetic.shape == (bd.n_basis_fns, 4, bd.n_max_grid),
    )
    check(
        "spline_deriv shape",
        bd.spline_deriv.shape == (bd.n_basis_fns, 4, bd.n_max_grid),
    )

    # --- 1b. species_of_fn attached + no-arg evaluate equivalence ---
    check(
        "species_of_fn attached == info.basisfn_species",
        bd.species_of_fn is not None
        and np.array_equal(bd.species_of_fn, info.basisfn_species),
    )
    _r_probe = np.array([0.1, 0.5, 1.0, 2.0])
    _u_arg = bd.evaluate_u(0, _r_probe, info.basisfn_species)
    _u_noarg = bd.evaluate_u(0, _r_probe)
    check(
        "evaluate_u: explicit arg == attached map",
        np.allclose(_u_arg, _u_noarg, rtol=0, atol=0),
    )

    # --- 2. Grid parameters ---
    check("r_grid_min positive", np.all(bd.r_grid_min > 0))
    check("r_grid_inc > 1", np.all(bd.r_grid_inc > 1.0))
    check("n_grid positive", np.all(bd.n_grid > 0))
    check(
        "r_grid total size",
        len(bd.r_grid) == int(np.sum(bd.n_grid)),
        f"got {len(bd.r_grid)} vs {int(np.sum(bd.n_grid))}",
    )

    # Rebuild check for each species
    for sp in range(bd.n_species):
        r1 = bd.species_r_grid(sp)
        r2 = bd.species_r_grid_rebuild(sp)
        match = np.allclose(r1, r2, rtol=1e-14)
        check(f"grid rebuild species {sp}", match)

    # --- 3. Spline identity: c1 = f at grid points ---
    # For a few functions, evaluate at grid points and compare with
    # spline_wave[fn, 0, grid_idx]
    species_of_fn = info.basisfn_species
    n_test = min(5, bd.n_basis_fns)
    test_fns = np.linspace(0, bd.n_basis_fns - 1, n_test, dtype=int)
    max_rel_err = 0.0
    for i_fn in test_fns:
        sp = species_of_fn[i_fn]
        r_grid_sp = bd.species_r_grid(sp)
        n_g = int(bd.n_grid[sp])
        # Test at a few grid points within outer_radius
        test_indices = [1, n_g // 4, n_g // 2, 3 * n_g // 4]
        for gi in test_indices:
            if gi >= n_g - 1:
                continue
            r_val = r_grid_sp[gi]
            if r_val > bd.outer_radius[i_fn]:
                continue
            u_eval = bd.evaluate_u(i_fn, np.array([r_val]), species_of_fn)[0]
            u_direct = bd.spline_wave[i_fn, 0, gi]  # c1 = f at grid point
            if abs(u_direct) > 1e-30:
                rel_err = abs(u_eval - u_direct) / abs(u_direct)
                max_rel_err = max(max_rel_err, rel_err)
    check(
        "spline c1=f identity",
        max_rel_err < 1e-10,
        f"max rel err = {max_rel_err:.2e}",
    )

    # --- 3b. Same c1=f identity for kinetic / deriv channels ---
    for chan_name, eval_fn in (
        ("kinetic", bd.evaluate_kinetic),
        ("deriv", bd.evaluate_deriv),
    ):
        max_rel_err_b = 0.0
        for i_fn in test_fns:
            sp = species_of_fn[i_fn]
            r_grid_sp = bd.species_r_grid(sp)
            n_g = int(bd.n_grid[sp])
            for gi in [1, n_g // 4, n_g // 2, 3 * n_g // 4]:
                if gi >= n_g - 1:
                    continue
                r_val = r_grid_sp[gi]
                if r_val > bd.outer_radius[i_fn]:
                    continue
                got = eval_fn(i_fn, np.array([r_val]), species_of_fn)[0]
                chan = getattr(bd, f"spline_{chan_name}")
                direct = chan[i_fn, 0, gi]
                if abs(direct) > 1e-30:
                    max_rel_err_b = max(max_rel_err_b, abs(got - direct) / abs(direct))
        check(
            f"spline {chan_name}: c1=f identity",
            max_rel_err_b < 1e-10,
            f"max rel err = {max_rel_err_b:.2e}",
        )

    # --- 3c. evaluate_deriv vs evaluate_du_dr agreement ---
    # (this fixture uses relativistic atomic_zora, so spline_deriv is
    # populated; guard anyway)
    if np.any(bd.spline_deriv != 0):
        max_d = 0.0
        for i_fn in test_fns:
            sp = species_of_fn[i_fn]
            r_grid_sp = bd.species_r_grid(sp)
            n_g = int(bd.n_grid[sp])
            for gi in [2, n_g // 3, 2 * n_g // 3]:
                r_val = r_grid_sp[gi]
                if r_val > bd.outer_radius[i_fn]:
                    continue
                r_arr = np.array([r_val])
                d_spl = bd.evaluate_deriv(i_fn, r_arr, species_of_fn)[0]
                d_ana = bd.evaluate_du_dr(i_fn, r_arr, species_of_fn)[0]
                denom = max(abs(d_spl), abs(d_ana), 1e-30)
                max_d = max(max_d, abs(d_spl - d_ana) / denom)
        check(
            "deriv spline vs analytic du/dr agree",
            max_d < 5e-3,
            f"max rel diff = {max_d:.2e}",
        )
    else:
        check("deriv spline zero (no use_basis_gradients) — skip", True)

    # --- 3d. evaluate_phi == evaluate_u / r (in-domain r > 0) ---
    phi_err = 0.0
    for i_fn in test_fns[:2]:
        sp = species_of_fn[i_fn]
        r_arr = np.geomspace(bd.r_grid_min[sp] * 10, bd.outer_radius[i_fn] * 0.9, 25)
        u = bd.evaluate_u(i_fn, r_arr, species_of_fn)
        phi = bd.evaluate_phi(i_fn, r_arr, species_of_fn)
        phi_err = max(phi_err, float(np.max(np.abs(phi - u / r_arr))))
    check("evaluate_phi == u/r", phi_err < 1e-12, f"max abs err = {phi_err:.2e}")

    # --- 4. Normalization: int u^2 dr ~ 1 ---
    # Use the logarithmic grid for integration: int u^2 dr ~ alpha * sum(r*u^2)
    norm_errors = []
    for i_fn in test_fns:
        sp = species_of_fn[i_fn]
        r_grid_sp = bd.species_r_grid(sp)
        n_g = int(bd.n_grid[sp])
        alpha = np.log(bd.r_grid_inc[sp])
        # u values at grid points = spline_wave[fn, 0, :n_g]
        u_vals = bd.spline_wave[i_fn, 0, :n_g]
        norm = alpha * np.sum(r_grid_sp * u_vals**2)
        norm_errors.append(abs(norm - 1.0))
    max_norm_err = max(norm_errors) if norm_errors else 1.0
    check(
        "normalization int u^2 dr ~ 1",
        max_norm_err < 0.01,
        f"max |norm-1| = {max_norm_err:.6f}",
    )

    # --- 5. Orthogonality within same species, same l ---
    # For species 0 (Mo), find pairs of same-l functions and check overlap
    max_overlap = 0.0
    for i_fn in range(bd.n_basis_fns):
        sp_i = species_of_fn[i_fn]
        l_i = info.basisfn_l[i_fn]
        for j_fn in range(i_fn):
            sp_j = species_of_fn[j_fn]
            l_j = info.basisfn_l[j_fn]
            if sp_i != sp_j or l_i != l_j:
                continue
            # Compute overlap int u_i * u_j dr on log grid
            r_grid_sp = bd.species_r_grid(sp_i)
            n_g = int(bd.n_grid[sp_i])
            alpha = np.log(bd.r_grid_inc[sp_i])
            u_i = bd.spline_wave[i_fn, 0, :n_g]
            u_j = bd.spline_wave[j_fn, 0, :n_g]
            overlap = alpha * np.sum(r_grid_sp * u_i * u_j)
            max_overlap = max(max_overlap, abs(overlap))
    check(
        "orthogonality |<u_i|u_j>| < 1e-6",
        max_overlap < 1e-6,
        f"max overlap = {max_overlap:.2e}",
    )

    # --- 6. H5 save ---
    import h5py

    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = Path(tmpdir) / "test_basis.h5"
        results = bd.save_h5(h5_path, info)
        check("H5 save succeeded", h5_path.exists())

        # All elements should be newly added
        all_added = all(results.values())
        check("all elements newly added", all_added, f"got {results}")

        with h5py.File(str(h5_path), "r") as f:
            check("H5 has Mo", "Mo" in f)
            check("H5 has S", "S" in f)
            check(
                "H5 root species_list",
                set(f.attrs["species_list"]) == {"Mo", "S"},
            )
            if "Mo" in f:
                mo = f["Mo"]
                check(
                    "Mo n_basis_rad matches",
                    mo.attrs["n_basis_rad"] == int(np.sum(info.basisfn_species == 0)),
                )
                check(
                    "Mo spline_wave shape",
                    mo["spline_wave"].shape[0] == mo.attrs["n_basis_rad"],
                )
                check(
                    "Mo r_grid size",
                    len(mo["r_grid"][:]) == mo.attrs["n_grid"],
                )

    # --- 7. Incremental: re-save should skip ---
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = Path(tmpdir) / "test_inc.h5"
        r1 = bd.save_h5(h5_path, info)
        r2 = bd.save_h5(h5_path, info)
        check("first save adds all", all(r1.values()))
        check("second save skips all", not any(r2.values()))

    # --- 7b. H5 -> viz round trip (writer/reader end-to-end) ---
    with tempfile.TemporaryDirectory() as tmpdir:
        h5_path = Path(tmpdir) / "test_viz.h5"
        bd.save_h5(h5_path, info)
        from aimspy.viz_basis import list_elements, plot_radial_basis

        check(
            "viz list_elements round trip",
            set(list_elements(h5_path)) == {"Mo", "S"},
            f"got {list_elements(h5_path)}",
        )
        import matplotlib

        matplotlib.use("Agg")
        png = Path(tmpdir) / "Mo_basis.png"
        fig = plot_radial_basis(h5_path, "Mo", logx=True, save=png)
        import matplotlib.pyplot as plt

        plt.close(fig)
        check(
            "viz plot_radial_basis logx renders",
            png.is_file() and png.stat().st_size > 5000,
            f"size = {png.stat().st_size if png.exists() else 0} bytes",
        )

    # --- 8. post-init registration of export_basis_data warns ---
    # (the callback fired inside aimspy_init; registering from INITED
    # state can never receive data — must warn, not stay silent)
    import warnings

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        calc.register_callback("export_basis_data", lambda ax, bd_: None)
    warned = any(
        "already fired" in str(w.message) and "export_basis_data" in str(w.message)
        for w in rec
    )
    check(
        "INITED registration warns (already fired)",
        warned,
        f"warnings = {[str(w.message)[:60] for w in rec]}",
    )

    _info("")

calc.close()
comm.Barrier()

if rank == 0:
    if all_ok:
        print("ALL CHECKS PASSED", flush=True)
    else:
        print("SOME CHECKS FAILED", flush=True)
        sys.exit(1)
