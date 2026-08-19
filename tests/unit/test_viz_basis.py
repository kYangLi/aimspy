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
        # Fixture grid spans 1e-6 .. ~9.4e-4 bohr (50 points); r_max=1e-4
        # bohr excludes the last ~17 points, so the filter really bites.
        fig = plot_radial_basis(h5_path, "Xx", r_max=1e-4, angstrom=False)
        rug = fig.axes[0].collections[0]
        drawn = rug.get_segments()
        n_expected = int(np.log(1e-4 / 1e-6) / np.log(1.15)) + 1  # 33
        assert 0 < len(drawn) < 50
        assert len(drawn) == n_expected
        assert all(seg[0][0] <= 1e-4 + 1e-12 for seg in drawn if len(seg))

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


class TestCLIErrorPaths:
    """Malformed inputs must surface as clean CLI errors, not tracebacks."""

    def _invoke(self, cmd, args):
        from click.testing import CliRunner

        result = CliRunner().invoke(cmd, args)
        assert result.exit_code == 1
        assert result.exception is not None
        # ClickException (handled) — not an unhandled crash:
        assert "Traceback" not in result.output
        return result

    # ---- viz-basis ----

    def test_viz_basis_corrupt_h5(self, tmp_path):
        from aimspy._cli_viz import viz_basis_cmd

        bad = tmp_path / "basis.h5"
        bad.write_bytes(b"this is not an HDF5 file" * 10)

        result = self._invoke(viz_basis_cmd, [str(bad), "-o", str(tmp_path / "out")])
        assert "Error:" in result.output

    def test_viz_basis_missing_dataset(self, tmp_path, h5_path):
        """A group missing its datasets (e.g. a partially-failed save)."""
        import h5py

        from aimspy._cli_viz import viz_basis_cmd

        broken = tmp_path / "broken.h5"
        with h5py.File(broken, "w") as src, h5py.File(h5_path, "r") as ref:
            src.create_group("Xx")  # empty group: no spline datasets
            src.attrs["species_list"] = list(ref.attrs.get("species_list", ["Xx"]))
            src.attrs["n_species"] = 1

        result = self._invoke(viz_basis_cmd, [str(broken), "-o", str(tmp_path / "out")])
        assert "Error:" in result.output

    def test_viz_basis_bad_r_max(self, h5_path, tmp_path):
        from aimspy._cli_viz import viz_basis_cmd

        for bad in ("-1.0", "0", "nan"):
            result = self._invoke(
                viz_basis_cmd,
                [str(h5_path), "--r-max", bad, "-o", str(tmp_path / "out")],
            )
            assert "r_max" in result.output

    def test_viz_basis_output_dir_is_file(self, h5_path, tmp_path):
        """click's own Path(file_okay=False) rejects this cleanly (exit 2)."""
        from click.testing import CliRunner

        from aimspy._cli_viz import viz_basis_cmd

        blocker = tmp_path / "blocker"
        blocker.write_text("i am a file")

        result = CliRunner().invoke(viz_basis_cmd, [str(h5_path), "-o", str(blocker)])
        assert result.exit_code == 2  # usage error, message, no traceback
        assert "Traceback" not in result.output

    # ---- viz-grid ----

    def test_viz_grid_corrupt_npz(self, tmp_path):
        from aimspy._cli_viz import viz_grid_cmd

        bad = tmp_path / "grid.npz"
        bad.write_bytes(b"PK\x03\x04 definitely not a zip archive")

        result = self._invoke(viz_grid_cmd, [str(bad), "rho"])
        assert "Error:" in result.output

    def test_viz_grid_truncated_npz(self, tmp_path):
        from aimspy._cli_viz import viz_grid_cmd

        bad = tmp_path / "grid.npz"
        bad.write_bytes(b"PK\x03\x04")  # zip signature, nothing else

        result = self._invoke(viz_grid_cmd, [str(bad), "rho"])
        assert "Error:" in result.output

    def test_viz_grid_unknown_field(self, tmp_path):
        from aimspy._cli_viz import viz_grid_cmd
        from .conftest import make_grid

        npz = tmp_path / "grid.npz"
        make_grid().save_npz(npz)

        result = self._invoke(viz_grid_cmd, [str(npz), "no_such_field"])
        assert "no_such_field" in result.output

    def test_viz_grid_atom_index_out_of_range(self, tmp_path):
        from aimspy._cli_viz import viz_grid_cmd
        from .conftest import make_grid

        npz = tmp_path / "grid.npz"
        make_grid().save_npz(npz)

        result = self._invoke(
            viz_grid_cmd, [str(npz), "rho", "--mode", "radial", "--atom-index", "99"]
        )
        assert "Error:" in result.output

    def test_viz_grid_bad_output_path(self, tmp_path):
        from aimspy._cli_viz import viz_grid_cmd
        from .conftest import make_grid

        npz = tmp_path / "grid.npz"
        make_grid().save_npz(npz)

        # nonexistent parent directory -> savefig OSError
        result = self._invoke(
            viz_grid_cmd,
            [str(npz), "rho", "-o", str(tmp_path / "no" / "dir" / "f.png")],
        )
        assert "Error:" in result.output


class TestDecodeSym:
    """_decode_sym must accept both str and bytes (h5py version safety)."""

    def test_str_passthrough(self):
        from aimspy.viz_basis import _decode_sym

        assert _decode_sym("Mo") == "Mo"

    def test_bytes_decoded(self):
        from aimspy.viz_basis import _decode_sym

        assert _decode_sym(b"Mo") == "Mo"

    def test_bytes_attr_file_read(self, h5_path):
        """list_elements on a file whose species_list attr is stored as
        fixed-length bytes (legacy style) still returns str symbols."""
        import h5py

        from aimspy.viz_basis import list_elements

        import shutil

        legacy = h5_path.parent / "legacy.h5"
        shutil.copy(h5_path, legacy)
        with h5py.File(legacy, "a") as f:
            lst = [
                s.encode() if isinstance(s, str) else s for s in f.attrs["species_list"]
            ]
            f.attrs["species_list"] = np.array(lst, dtype="S4")

        assert list_elements(legacy) == ["Xx"]
