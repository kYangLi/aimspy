"""Unit tests for aimspy.viz_basis (matplotlib Agg backend, no display).

The H5 fixture files are produced with the same mock BasisData used by
test_basis_data.py (spline coefficients of u(r) = r^2 * exp(-r) on a
log grid), so spline-evaluation correctness can be checked against the
analytic function.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless

import numpy as np
import pytest

from aimspy.viz_basis import (
    _evaluate_spline,
    _label_fn,
    _load_element,
    list_elements,
    plot_radial_basis,
)

from .test_basis_data import _make_basis_1species


class MockInfo:
    """Single-element mock AimspyInfo (Xx, Z=99)."""

    basisfn_species = np.array([0, 0, 0], dtype=np.int32)
    basisfn_n = np.array([1, 2, 2], dtype=np.int32)
    basisfn_l = np.array([0, 0, 1], dtype=np.int32)
    basisfn_type = ["atomic", "atomic", "hydro"]
    species_elements = ["Xx"]
    species_z = np.array([99.0])


@pytest.fixture
def h5_path(tmp_path):
    """basis.h5 with one element, 3 radial functions (1s, 2s, 2p)."""
    pytest.importorskip("h5py")
    bd = _make_basis_1species(n_grid=50, n_fns=3)
    p = tmp_path / "basis.h5"
    bd.save_h5(p, MockInfo())
    return p


@pytest.fixture
def h5_path_two(tmp_path):
    """basis.h5 with two elements (H, C) for CLI all-elements tests."""
    pytest.importorskip("h5py")

    class MockInfoH(MockInfo):
        basisfn_species = np.array([0, 0], dtype=np.int32)
        basisfn_n = np.array([1, 2], dtype=np.int32)
        basisfn_l = np.array([0, 0], dtype=np.int32)
        basisfn_type = ["atomic", "atomic"]
        species_elements = ["H"]
        species_z = np.array([1.0])

    class MockInfoC(MockInfo):
        basisfn_species = np.array([0, 0], dtype=np.int32)
        basisfn_n = np.array([2, 2], dtype=np.int32)
        basisfn_l = np.array([0, 1], dtype=np.int32)
        basisfn_type = ["atomic", "hydro"]
        species_elements = ["C"]
        species_z = np.array([6.0])

    bd = _make_basis_1species(n_grid=50, n_fns=2)
    p = tmp_path / "basis_two.h5"
    bd.save_h5(p, MockInfoH())
    bd.save_h5(p, MockInfoC())
    return p


def _exact_u(r):
    return r**2 * np.exp(-r)


class TestLoadElement:
    def test_load_returns_data(self, h5_path):
        b = _load_element(h5_path, "Xx")
        assert b.element == "Xx"
        assert b.z == pytest.approx(99.0)
        assert b.n_basis_rad == 3
        assert b.l_max == 1
        assert b.spline_wave.shape == (3, 4, 50)
        assert len(b.type) == 3
        assert b.type[2] == "hydro"

    def test_unknown_element_raises(self, h5_path):
        with pytest.raises(ValueError, match="not found"):
            _load_element(h5_path, "No")

    def test_list_elements(self, h5_path_two):
        assert set(list_elements(h5_path_two)) == {"H", "C"}


class TestEvaluateSpline:
    def test_grid_point_identity(self, h5_path):
        """At a grid point the spline value equals coefficient c1."""
        b = _load_element(h5_path, "Xx")
        for i_fn in (0, 1, 2):
            r = b.r_grid[10]
            if r > b.outer_radius[i_fn]:
                continue
            val = _evaluate_spline(
                b.spline_wave[i_fn],
                b.r_grid_min,
                b.r_grid_inc,
                b.n_grid,
                float(b.outer_radius[i_fn]),
                np.array([r]),
            )
            assert val[0] == pytest.approx(b.spline_wave[i_fn, 0, 10], rel=1e-12)

    def test_analytic_function(self, h5_path):
        """Coefficients were built from r^2 exp(-r): grid values match."""
        b = _load_element(h5_path, "Xx")
        val = _evaluate_spline(
            b.spline_wave[0],
            b.r_grid_min,
            b.r_grid_inc,
            b.n_grid,
            float(b.outer_radius[0]),
            np.array([b.r_grid[20]]),
        )
        assert val[0] == pytest.approx(_exact_u(b.r_grid[20]), rel=1e-10)

    def test_truncation_beyond_outer_radius(self, h5_path):
        b = _load_element(h5_path, "Xx")
        val = _evaluate_spline(
            b.spline_wave[0],
            b.r_grid_min,
            b.r_grid_inc,
            b.n_grid,
            float(b.outer_radius[0]),
            np.array([b.outer_radius[0] * 1.1]),
        )
        assert val[0] == 0.0

    def test_below_r_grid_min_is_zero(self, h5_path):
        """No backward extrapolation below the first grid point."""
        b = _load_element(h5_path, "Xx")
        val = _evaluate_spline(
            b.spline_wave[0],
            b.r_grid_min,
            b.r_grid_inc,
            b.n_grid,
            float(b.outer_radius[0]),
            np.array([b.r_grid_min * 1e-3]),
        )
        assert val[0] == 0.0


class TestLabelFn:
    def test_plain(self):
        assert _label_fn(1, 0, 0, "atomic", False) == "1s-0"

    def test_with_type(self):
        assert _label_fn(2, 1, 1, "hydro", True) == "2p-1 (hydro)"

    def test_high_l(self):
        assert _label_fn(5, 4, 0, "hydro", False) == "5g-0"


class TestPlotRadialBasis:
    def test_returns_figure(self, h5_path):
        fig = plot_radial_basis(h5_path, "Xx")
        assert fig is not None
        assert len(fig.axes) == 1
        ax = fig.axes[0]
        assert "r (Å)" in ax.get_xlabel()
        assert len(ax.lines) == 3  # one line per radial function
        assert "Xx" in ax.get_title()

    def test_split_l(self, h5_path):
        fig = plot_radial_basis(h5_path, "Xx", split_l=True)
        assert len(fig.axes) == 2  # l_max=1 -> two panels

    def test_kind_phi(self, h5_path):
        fig = plot_radial_basis(h5_path, "Xx", kind="phi")
        assert "φ" in fig.axes[0].get_ylabel()  # ylabel set on single panel

    def test_bohr_units(self, h5_path):
        fig = plot_radial_basis(h5_path, "Xx", angstrom=False)
        assert "bohr" in fig.axes[0].get_xlabel()

    def test_logx_scales_x_axis(self, h5_path):
        """logx=True sets a log-scaled x axis; default stays linear."""
        fig_lin = plot_radial_basis(h5_path, "Xx")
        assert fig_lin.axes[0].get_xscale() == "linear"
        fig_log = plot_radial_basis(h5_path, "Xx", logx=True)
        assert fig_log.axes[0].get_xscale() == "log"
        # log axis must not include 0 — lower bound is r_grid_min
        assert fig_log.axes[0].get_xlim()[0] > 0.0

    def test_bad_kind_raises(self, h5_path):
        with pytest.raises(ValueError, match="kind"):
            plot_radial_basis(h5_path, "Xx", kind="bogus")

    def test_rug_markers_present(self, h5_path):
        """show_grid=True adds a LineCollection of grid ticks."""
        fig = plot_radial_basis(h5_path, "Xx", show_grid=True)
        assert len(fig.axes[0].collections) >= 1

    def test_no_rug(self, h5_path):
        fig = plot_radial_basis(h5_path, "Xx", show_grid=False)
        assert len(fig.axes[0].collections) == 0

    def test_rug_respects_r_max(self, h5_path):
        """Grid points beyond the x limit are not drawn."""
        fig = plot_radial_basis(h5_path, "Xx", r_max=0.5)  # 0.5 Å ~ tiny range
        rug = fig.axes[0].collections[0]
        drawn = rug.get_segments()
        # every drawn tick must lie within [0, 0.5] (data x coords)
        assert all(seg[0][0] <= 0.5 + 1e-9 for seg in drawn if len(seg))

    def test_save_png(self, h5_path, tmp_path):
        out = tmp_path / "xx_basis.png"
        plot_radial_basis(h5_path, "Xx", save=out)
        assert out.exists() and out.stat().st_size > 0

    def test_legend_has_labels(self, h5_path):
        fig = plot_radial_basis(h5_path, "Xx", show_type=True)
        leg = fig.axes[0].get_legend()
        texts = [t.get_text() for t in leg.get_texts()]
        assert "1s-0 (atomic)" in texts
        assert "2p-0 (hydro)" in texts


class TestCLIVizBasis:
    def test_all_elements_saved(self, h5_path_two, tmp_path):
        """CLI saves one figure per element, no element argument needed."""
        from click.testing import CliRunner

        from aimspy._cli_viz import viz_basis_cmd

        out_dir = tmp_path / "figs"
        runner = CliRunner()
        result = runner.invoke(viz_basis_cmd, [str(h5_path_two), "-o", str(out_dir)])
        assert result.exit_code == 0, result.output
        assert (out_dir / "H_basis.png").exists()
        assert (out_dir / "C_basis.png").exists()

    def test_parallel_jobs(self, h5_path_two, tmp_path):
        """-j 2 produces the same outputs (process pool path)."""
        from click.testing import CliRunner

        from aimspy._cli_viz import viz_basis_cmd

        out_dir = tmp_path / "figs_j2"
        runner = CliRunner()
        result = runner.invoke(
            viz_basis_cmd, [str(h5_path_two), "-o", str(out_dir), "-j", "2"]
        )
        assert result.exit_code == 0, result.output
        assert (out_dir / "H_basis.png").exists()
        assert (out_dir / "C_basis.png").exists()

    def test_options_pass_through(self, h5_path_two, tmp_path):
        from click.testing import CliRunner

        from aimspy._cli_viz import viz_basis_cmd

        out_dir = tmp_path / "figs_opts"
        runner = CliRunner()
        result = runner.invoke(
            viz_basis_cmd,
            [
                str(h5_path_two),
                "-o",
                str(out_dir),
                "--kind",
                "phi",
                "--split-l",
                "--logx",
                "--no-grid",
                "--bohr",
                "--fmt",
                "pdf",
            ],
        )
        assert result.exit_code == 0, result.output
        assert (out_dir / "H_basis.pdf").exists()


class TestCLIVizGrid:
    def test_scatter_save(self, tmp_path):
        from click.testing import CliRunner

        from aimspy._cli_viz import viz_grid_cmd
        from .conftest import make_grid

        npz = tmp_path / "grid.npz"
        make_grid(n=200).save_npz(npz)

        out = tmp_path / "rho.png"
        runner = CliRunner()
        result = runner.invoke(
            viz_grid_cmd,
            [str(npz), "rho", "--center", "0", "--width", "4", "-o", str(out)],
        )
        assert result.exit_code == 0, result.output
        assert out.exists() and out.stat().st_size > 0

    def test_contour_mode(self, tmp_path):
        from click.testing import CliRunner

        from aimspy._cli_viz import viz_grid_cmd
        from .conftest import make_grid

        npz = tmp_path / "grid.npz"
        make_grid(n=300).save_npz(npz)

        out = tmp_path / "contour.png"
        runner = CliRunner()
        result = runner.invoke(
            viz_grid_cmd,
            [
                str(npz),
                "delta_rho",
                "--mode",
                "contour",
                "--symlog",
                "--nx",
                "40",
                "--ny",
                "40",
                "--width",
                "4",
                "-o",
                str(out),
            ],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()

    def test_radial_requires_atom_index(self, tmp_path):
        from click.testing import CliRunner

        from aimspy._cli_viz import viz_grid_cmd
        from .conftest import make_grid

        npz = tmp_path / "grid.npz"
        make_grid().save_npz(npz)

        runner = CliRunner()
        result = runner.invoke(viz_grid_cmd, [str(npz), "rho", "--mode", "radial"])
        assert result.exit_code != 0
        assert "atom-index" in result.output
