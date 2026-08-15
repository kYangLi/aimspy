"""Unit tests for aimspy.viz (matplotlib backend only, no display).

pyvista-dependent ``isosurface`` is tested only for its ImportError guard
(pyvista is not installed in the test environment).  matplotlib figures are
built with the non-interactive Agg backend.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless

import numpy as np
import pytest

from aimspy.grid_data import GridData
from aimspy import viz


def _make_grid(n=60, n_atoms=2, seed=1):
    rng = np.random.default_rng(seed)
    coords = (rng.random((3, n)) - 0.5) * 8.0  # bohr, centred
    rho = (rng.random((1, n)) + 0.05) * 10.0
    vh = -(rng.random(n) + 0.5)
    vh0 = -(rng.random(n) + 0.5)
    vxc = -np.abs(rng.random((1, n))) - 0.01
    vxc0 = -np.abs(rng.random((1, n))) - 0.01
    return GridData(
        n_full_points=n,
        n_spin=1,
        n_atoms=n_atoms,
        coords=coords,
        partition_tab=np.full(n, 0.1),
        index_atom=np.arange(n, dtype=np.int32) % n_atoms,
        index_radial=np.arange(n, dtype=np.int32),
        index_angular=np.arange(n, dtype=np.int32),
        rho=rho,
        vks=vh[np.newaxis, :] + vxc,
        vks0=vh0[np.newaxis, :] + vxc0,
        vh=vh,
        vh0=vh0,
        rho0=(rng.random(n) + 0.05) * 4.0 * np.pi,
    )


class TestResolveValues:
    def test_name_lookup(self):
        g = _make_grid()
        out = viz._resolve_values(g, "rho")
        assert out.shape == (g.n_full_points,)

    def test_explicit_array(self):
        g = _make_grid()
        arr = np.arange(g.n_full_points, dtype=float)
        np.testing.assert_array_equal(viz._resolve_values(g, arr), arr)

    def test_unknown_name_raises(self):
        g = _make_grid()
        with pytest.raises(ValueError, match="unknown GridData field"):
            viz._resolve_values(g, "not_a_field")

    def test_wrong_length_raises(self):
        g = _make_grid()
        with pytest.raises(ValueError, match="n_full_points"):
            viz._resolve_values(g, np.zeros(3))


class TestScatterSlice:
    def test_returns_axes(self):
        g = _make_grid()
        ax = viz.scatter_slice(g, "rho", center=0.0, width=2.0)
        assert ax is not None
        assert ax.get_xlabel().startswith(("x", "y"))

    def test_log_label(self):
        g = _make_grid()
        ax = viz.scatter_slice(g, "rho", center=0.0, width=8.0, log=True)
        assert "log10(rho)" in ax.get_title()

    def test_bohr_units(self):
        g = _make_grid()
        ax = viz.scatter_slice(g, "vks", center=0.0, width=8.0, angstrom=False)
        assert "bohr" in ax.get_xlabel()


class TestSliceContour:
    def test_returns_axes(self):
        g = _make_grid(n=200)
        ax = viz.slice_contour(g, "rho", center=0.0, width=4.0, nx=30, ny=30)
        assert ax is not None

    def test_nearest_method(self):
        g = _make_grid(n=200)
        ax = viz.slice_contour(
            g, "vks", center=0.0, width=4.0, nx=25, ny=25, method="nearest"
        )
        assert ax is not None

    def test_symlog_uses_symlognorm(self):
        from matplotlib.colors import SymLogNorm

        g = _make_grid(n=200)
        ax = viz.slice_contour(
            g, "delta_rho", center=0.0, width=8.0, nx=30, ny=30, symlog=True
        )
        assert isinstance(ax.collections[0].norm, SymLogNorm)

    def test_empty_slice_raises(self):
        g = _make_grid()
        with pytest.raises(ValueError, match="no grid points"):
            viz.slice_contour(g, "rho", center=100.0, width=0.1)


class TestNormAndCmap:
    def test_scatter_symlog_norm(self):
        from matplotlib.colors import SymLogNorm

        g = _make_grid()
        ax = viz.scatter_slice(g, "delta_rho", center=0.0, width=8.0, symlog=True)
        assert isinstance(ax.collections[0].norm, SymLogNorm)

    def test_auto_diverging_cmap_for_signed_field(self):
        # delta_rho has both signs in the mock -> auto 'RdBu_r'
        g = _make_grid()
        assert np.any(g.delta_rho < 0)
        ax = viz.scatter_slice(g, "delta_rho", center=0.0, width=8.0)
        assert ax.collections[0].cmap.name == "RdBu_r"

    def test_auto_sequential_cmap_for_positive_field(self):
        g = _make_grid()
        assert np.all(g.rho > 0)
        ax = viz.scatter_slice(g, "rho", center=0.0, width=8.0)
        assert ax.collections[0].cmap.name == "viridis"

    def test_symlog_label_tag(self):
        g = _make_grid()
        ax = viz.scatter_slice(g, "delta_rho", center=0.0, width=8.0, symlog=True)
        assert "symlog" in ax.get_title()


class TestRadialProfile:
    def test_atom_index(self):
        g = _make_grid()
        ax = viz.radial_profile(g, "rho", atom_index=0)
        assert ax is not None
        assert "r (" in ax.get_xlabel()

    def test_explicit_center(self):
        g = _make_grid()
        ax = viz.radial_profile(g, "vks", center=[0.0, 0.0, 0.0], angstrom=False)
        assert "bohr" in ax.get_xlabel()

    def test_requires_center_or_atom(self):
        g = _make_grid()
        with pytest.raises(ValueError, match="atom_index"):
            viz.radial_profile(g, "rho")

    def test_bad_atom_index(self):
        g = _make_grid(n_atoms=2)
        with pytest.raises(ValueError, match="no grid points"):
            viz.radial_profile(g, "rho", atom_index=5)


class TestIsosurfaceGuard:
    def test_import_error_if_no_pyvista(self):
        pytest.importorskip  # noqa: B018 (clarity)
        g = _make_grid()
        try:
            import pyvista  # noqa: F401

            pytest.skip("pyvista is installed; guard not exercised")
        except ImportError:
            with pytest.raises(ImportError, match="pyvista"):
                viz.isosurface(g, "rho", iso=1.0, show=False)
