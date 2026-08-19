"""Unit tests for BasisData (no MPI, no libaims).

Covers grid reconstruction, spline evaluation (u, phi, du/dr), zeta
numbering, H5 round-trip, incremental add, and duplicate skip.
"""

from __future__ import annotations

import numpy as np
import pytest

from aimspy.basis_data import BasisData


def _make_basis_1species(n_grid=50, n_fns=3):
    """Deterministic single-species BasisData for tests.

    Uses a simple exponential function u(r) = r * exp(-r) (hydrogen 1s-like)
    on a logarithmic grid, with known spline coefficients computed by
    the same algorithm as Fortran's cubic_spline.
    """
    # Logarithmic grid parameters
    r_min = 1e-6
    r_inc = 1.15
    n_g = n_grid
    r_grid = r_min * r_inc ** np.arange(n_g)

    # Build spline coefficients for a known function
    # u(r) = r^2 * exp(-r)  (l=1-like, smooth, decaying)
    def exact_u(r):
        return r**2 * np.exp(-r)

    def exact_du_dr(r):
        return 2 * r * np.exp(-r) - r**2 * np.exp(-r)

    def exact_kinetic(r):
        # (e - v)u for a Coulomb potential v = -1/r, e = -0.5
        return (-0.5 + 1.0 / np.maximum(r, 1e-30)) * exact_u(r)

    n_max_grid = n_g
    n_max_spline = 4

    spline_wave = np.zeros((n_fns, n_max_spline, n_max_grid))
    spline_kinetic = np.zeros((n_fns, n_max_spline, n_max_grid))
    spline_deriv = np.zeros((n_fns, n_max_spline, n_max_grid))

    for i_fn in range(n_fns):
        # Vary the function slightly per radial function
        scale = 1.0 + 0.1 * i_fn
        u_vals = exact_u(r_grid) * scale
        k_vals = exact_kinetic(r_grid) * scale
        d_vals = exact_du_dr(r_grid) * scale

        # Natural cubic spline on integer grid (reproduce cubic_spline algorithm)
        spline_wave[i_fn] = _natural_cubic_spline(u_vals, n_max_grid)
        spline_kinetic[i_fn] = _natural_cubic_spline(k_vals, n_max_grid)
        spline_deriv[i_fn] = _natural_cubic_spline(d_vals, n_max_grid)

    outer_radius = np.array([r_grid[-3], r_grid[-5], r_grid[-7]], dtype=np.float64)

    return BasisData(
        n_species=1,
        n_basis_fns=n_fns,
        n_max_grid=n_max_grid,
        n_max_spline=n_max_spline,
        r_grid_min=np.array([r_min]),
        r_grid_inc=np.array([r_inc]),
        n_grid=np.array([n_g], dtype=np.int32),
        r_grid=r_grid,
        outer_radius=outer_radius,
        spline_wave=spline_wave,
        spline_kinetic=spline_kinetic,
        spline_deriv=spline_deriv,
    )


def _natural_cubic_spline(f_vals, n_max_grid):
    """Reproduce the Fortran cubic_spline algorithm (spline.f90:69-214).

    Returns (4, n_max_grid) coefficient array.
    """
    n = len(f_vals)
    coeffs = np.zeros((4, n_max_grid))
    if n == 0:
        return coeffs
    if n == 1:
        coeffs[0, 0] = f_vals[0]
        return coeffs

    # Solve tridiagonal system for first derivatives (natural BC)
    # d_{i-1} + 4*d_i + d_{i+1} = 3*(f_{i+1} - f_{i-1})
    # 2*d_1 + d_2 = 3*(f_2 - f_1)
    # d_{n-1} + 2*d_n = 3*(f_n - f_{n-1})
    diag = np.full(n, 4.0)
    diag[0] = 2.0
    diag[-1] = 2.0
    rhs = np.zeros(n)
    rhs[0] = 3.0 * (f_vals[1] - f_vals[0])
    rhs[-1] = 3.0 * (f_vals[-1] - f_vals[-2])
    for i in range(1, n - 1):
        rhs[i] = 3.0 * (f_vals[i + 1] - f_vals[i - 1])

    # Thomas algorithm
    lower = np.ones(n - 1)
    upper = np.ones(n - 1)
    # Forward elimination
    for i in range(1, n):
        w = lower[i - 1] / diag[i - 1]
        diag[i] -= w * upper[i - 1]
        rhs[i] -= w * rhs[i - 1]
    # Back substitution
    d = np.zeros(n)
    d[-1] = rhs[-1] / diag[-1]
    for i in range(n - 2, -1, -1):
        d[i] = (rhs[i] - upper[i] * d[i + 1]) / diag[i]

    # Compute coefficients
    for i in range(n - 1):
        coeffs[0, i] = f_vals[i]
        coeffs[1, i] = d[i]
        coeffs[2, i] = 3.0 * (f_vals[i + 1] - f_vals[i]) - 2.0 * d[i] - d[i + 1]
        coeffs[3, i] = 2.0 * (f_vals[i] - f_vals[i + 1]) + d[i] + d[i + 1]
    coeffs[0, n - 1] = f_vals[n - 1]
    coeffs[1, n - 1] = d[n - 1]

    return coeffs


class TestGridReconstruction:
    def test_species_r_grid(self):
        bd = _make_basis_1species(n_grid=50)
        r = bd.species_r_grid(0)
        assert len(r) == 50
        assert r[0] == pytest.approx(1e-6)
        assert r[-1] == pytest.approx(1e-6 * 1.15**49)

    def test_rebuild_matches(self):
        bd = _make_basis_1species(n_grid=50)
        r1 = bd.species_r_grid(0)
        r2 = bd.species_r_grid_rebuild(0)
        np.testing.assert_allclose(r1, r2, rtol=1e-14)


class TestSplineEvaluation:
    def test_grid_point_values(self):
        """Spline evaluated at grid points should match original function."""
        bd = _make_basis_1species(n_grid=50, n_fns=1)
        sp_arr = np.array([0], dtype=np.int32)
        r_grid = bd.species_r_grid(0)
        # Evaluate at grid points within outer_radius (i_r = 1, 2, 3, ...)
        for i in range(1, len(r_grid) - 1):
            if r_grid[i] > bd.outer_radius[0]:
                break
            r = np.array([r_grid[i]])
            u = bd.evaluate_u(0, r, sp_arr)
            expected = r_grid[i] ** 2 * np.exp(-r_grid[i])
            assert u[0] == pytest.approx(expected, rel=1e-10)

    def test_interpolation(self):
        """Spline interpolation between grid points should be smooth."""
        bd = _make_basis_1species(n_grid=100, n_fns=1)
        sp_arr = np.array([0], dtype=np.int32)
        r_grid = bd.species_r_grid(0)
        # Evaluate at midpoints between grid points
        for i in range(5, len(r_grid) - 5):
            r_mid = 0.5 * (r_grid[i] + r_grid[i + 1])
            u_mid = bd.evaluate_u(0, np.array([r_mid]), sp_arr)
            expected = r_mid**2 * np.exp(-r_mid)
            # Cubic spline interpolation error is O(h^4) where h ~ r*ln(inc)
            assert u_mid[0] == pytest.approx(expected, rel=1e-4)

    def test_outer_radius_truncation(self):
        """u(r) should be zero beyond outer_radius."""
        bd = _make_basis_1species(n_grid=50, n_fns=3)
        sp_arr = np.zeros(3, dtype=np.int32)  # all 3 fns belong to species 0
        for i_fn in range(3):
            r_far = np.array([bd.outer_radius[i_fn] * 1.1])
            u = bd.evaluate_u(i_fn, r_far, sp_arr)
            assert u[0] == 0.0

    def test_evaluate_phi(self):
        """phi(r) = u(r)/r."""
        bd = _make_basis_1species(n_grid=50, n_fns=1)
        sp_arr = np.array([0], dtype=np.int32)
        r = np.array([1.0, 2.0, 5.0])
        u = bd.evaluate_u(0, r, sp_arr)
        phi = bd.evaluate_phi(0, r, sp_arr)
        np.testing.assert_allclose(phi, u / r, rtol=1e-14)

    def test_evaluate_du_dr(self):
        """du/dr should match the analytic derivative."""
        bd = _make_basis_1species(n_grid=200, n_fns=1)
        sp_arr = np.array([0], dtype=np.int32)
        r_grid = bd.species_r_grid(0)
        for i in range(10, len(r_grid) - 10, 20):
            r = np.array([r_grid[i]])
            du = bd.evaluate_du_dr(0, r, sp_arr)
            expected = 2 * r[0] * np.exp(-r[0]) - r[0] ** 2 * np.exp(-r[0])
            # The spline derivative is w.r.t. grid index, chain rule gives
            # some numerical error; use loose tolerance
            assert du[0] == pytest.approx(expected, rel=5e-3)

    def test_attached_species_of_fn(self):
        """With species_of_fn attached, evaluate_* need no explicit argument."""
        bd = _make_basis_1species(n_grid=50, n_fns=1)
        bd.species_of_fn = np.array([0], dtype=np.int32)
        r = np.array([0.5, 1.0, 2.0])
        explicit = bd.evaluate_u(0, r, np.array([0], dtype=np.int32))
        implicit = bd.evaluate_u(0, r)
        np.testing.assert_array_equal(explicit, implicit)
        np.testing.assert_array_equal(implicit, bd.evaluate_phi(0, r) * r)
        du = bd.evaluate_du_dr(0, r)
        assert du.shape == r.shape

    def test_missing_species_of_fn_raises(self):
        """Without explicit arg and no attachment, a clear error is raised."""
        bd = _make_basis_1species(n_grid=50, n_fns=1)
        with pytest.raises(ValueError, match="species_of_fn"):
            bd.evaluate_u(0, np.array([1.0]))

    def test_evaluate_kinetic_grid_points(self):
        """evaluate_kinetic matches the splined (e−v)·u at grid points."""
        bd = _make_basis_1species(n_grid=50, n_fns=1)
        bd.species_of_fn = np.array([0], dtype=np.int32)
        r_grid = bd.species_r_grid(0)

        def expected(r):
            # mirror of the fixture: (e−v)·u for v=−1/r, e=−0.5, u=r²·exp(−r)
            return (-0.5 + 1.0 / r) * r**2 * np.exp(-r)

        for i in range(1, len(r_grid) - 1):
            if r_grid[i] > bd.outer_radius[0]:
                break
            k = bd.evaluate_kinetic(0, np.array([r_grid[i]]))
            assert k[0] == pytest.approx(expected(r_grid[i]), rel=1e-10)

    def test_evaluate_deriv_matches_du_dr(self):
        """evaluate_deriv (raw spline_deriv) agrees with analytic du/dr."""
        bd = _make_basis_1species(n_grid=200, n_fns=1)
        bd.species_of_fn = np.array([0], dtype=np.int32)
        r_grid = bd.species_r_grid(0)
        # fixture builds spline_deriv from the SAME exact derivative, so
        # evaluate_deriv should hit those tabulated values at grid points
        for i in range(10, len(r_grid) - 10, 20):
            r = np.array([r_grid[i]])
            expected = 2 * r[0] * np.exp(-r[0]) - r[0] ** 2 * np.exp(-r[0])
            got = bd.evaluate_deriv(0, r)
            assert got[0] == pytest.approx(expected, rel=1e-10)
            # and it should agree with the analytic du/dr of the wave
            # spline to the same spline accuracy (~1e-3; see du_dr test)
            du = bd.evaluate_du_dr(0, r)
            assert got[0] == pytest.approx(du[0], rel=5e-3)

    def test_out_of_domain_zero_for_all_evaluators(self):
        """u / kinetic / deriv / phi all return 0 outside the spline domain."""
        bd = _make_basis_1species(n_grid=50, n_fns=1)
        bd.species_of_fn = np.array([0], dtype=np.int32)
        r = np.array([bd.r_grid_min[0] * 0.5, bd.outer_radius[0] * 1.5])
        assert np.all(bd.evaluate_u(0, r) == 0.0)
        assert np.all(bd.evaluate_kinetic(0, r) == 0.0)
        assert np.all(bd.evaluate_deriv(0, r) == 0.0)
        assert np.all(bd.evaluate_phi(0, r) == 0.0)
        assert np.all(bd.evaluate_du_dr(0, r) == 0.0)


class TestZetaNumbering:
    def test_zeta_unique_nl(self):
        """All unique (n,l) pairs get zeta=0."""
        # This is tested implicitly in save_h5; here we test the logic directly
        n = np.array([1, 2, 2, 3], dtype=np.int32)
        ell = np.array([0, 0, 1, 0], dtype=np.int32)
        zeta = _compute_zeta(n, ell)
        np.testing.assert_array_equal(zeta, [0, 0, 0, 0])

    def test_zeta_duplicate_nl(self):
        """Duplicate (n,l) pairs get incrementing zeta."""
        n = np.array([2, 2, 2, 3, 2], dtype=np.int32)
        ell = np.array([1, 1, 0, 1, 1], dtype=np.int32)
        zeta = _compute_zeta(n, ell)
        np.testing.assert_array_equal(zeta, [0, 1, 0, 0, 2])


def _compute_zeta(n, ell):
    """Compute zeta values (same logic as BasisData.save_h5)."""
    zeta = np.zeros(len(n), dtype=np.int32)
    nl_count = {}
    for i in range(len(n)):
        key = (int(n[i]), int(ell[i]))
        zeta[i] = nl_count.get(key, 0)
        nl_count[key] = nl_count.get(key, 0) + 1
    return zeta


class TestH5RoundTrip:
    def test_save_and_load(self, tmp_path):
        """Save to H5 and verify all datasets."""
        h5py = pytest.importorskip("h5py")
        bd = _make_basis_1species(n_grid=50, n_fns=3)

        # Mock AimspyInfo
        class MockInfo:
            basisfn_species = np.array([0, 0, 0], dtype=np.int32)
            basisfn_n = np.array([1, 2, 2], dtype=np.int32)
            basisfn_l = np.array([0, 0, 1], dtype=np.int32)
            basisfn_type = ["atomic", "atomic", "hydro"]
            species_elements = ["Xx"]
            species_z = np.array([99.0])

        h5_path = tmp_path / "test_basis.h5"
        results = bd.save_h5(h5_path, MockInfo())

        assert results == {"Xx": True}

        with h5py.File(str(h5_path), "r") as f:
            assert "Xx" in f
            grp = f["Xx"]
            assert grp.attrs["element"] == "Xx"
            assert grp.attrs["z"] == pytest.approx(99.0)
            assert grp.attrs["n_basis_rad"] == 3
            assert grp.attrs["n_orbitals"] == 1 + 1 + 3  # 2*0+1 + 2*0+1 + 2*1+1
            assert grp.attrs["l_max"] == 1
            assert grp.attrs["n_grid"] == 50

            np.testing.assert_array_equal(grp["n"][:], [1, 2, 2])
            np.testing.assert_array_equal(grp["l"][:], [0, 0, 1])
            np.testing.assert_array_equal(grp["zeta"][:], [0, 0, 0])
            assert grp["type"][0] == b"atomic"
            assert grp["type"][2] == b"hydro"
            assert grp["spline_wave"].shape == (3, 4, 50)
            assert grp["spline_kinetic"].shape == (3, 4, 50)
            assert grp["spline_deriv"].shape == (3, 4, 50)
            assert grp["r_grid"].shape == (50,)

    def test_incremental_add(self, tmp_path):
        """Adding a second element to an existing file."""
        h5py = pytest.importorskip("h5py")
        bd = _make_basis_1species(n_grid=50, n_fns=2)

        class MockInfo1:
            basisfn_species = np.array([0, 0], dtype=np.int32)
            basisfn_n = np.array([1, 2], dtype=np.int32)
            basisfn_l = np.array([0, 0], dtype=np.int32)
            basisfn_type = ["atomic", "atomic"]
            species_elements = ["H"]
            species_z = np.array([1.0])

        class MockInfo2:
            basisfn_species = np.array([0, 0], dtype=np.int32)
            basisfn_n = np.array([1, 2], dtype=np.int32)
            basisfn_l = np.array([0, 1], dtype=np.int32)
            basisfn_type = ["atomic", "hydro"]
            species_elements = ["C"]
            species_z = np.array([6.0])

        h5_path = tmp_path / "test_inc.h5"

        # First add H
        r1 = bd.save_h5(h5_path, MockInfo1())
        assert r1 == {"H": True}

        # Then add C (same BasisData but different info)
        r2 = bd.save_h5(h5_path, MockInfo2())
        assert r2 == {"C": True}

        with h5py.File(str(h5_path), "r") as f:
            assert "H" in f
            assert "C" in f
            assert set(f.attrs["species_list"]) == {"H", "C"}
            assert f.attrs["n_species"] == 2

    def test_duplicate_skip(self, tmp_path):
        """Re-adding the same element is a no-op."""
        h5py = pytest.importorskip("h5py")
        bd = _make_basis_1species(n_grid=50, n_fns=2)

        class MockInfo:
            basisfn_species = np.array([0, 0], dtype=np.int32)
            basisfn_n = np.array([1, 2], dtype=np.int32)
            basisfn_l = np.array([0, 0], dtype=np.int32)
            basisfn_type = ["atomic", "atomic"]
            species_elements = ["H"]
            species_z = np.array([1.0])

        h5_path = tmp_path / "test_dup.h5"
        r1 = bd.save_h5(h5_path, MockInfo())
        assert r1 == {"H": True}

        # Modify data and try to re-add
        bd.spline_wave *= 2.0
        r2 = bd.save_h5(h5_path, MockInfo())
        assert r2 == {"H": False}

        # Verify original data is preserved
        with h5py.File(str(h5_path), "r") as f:
            original = _make_basis_1species(n_grid=50, n_fns=2)
            np.testing.assert_allclose(
                f["H"]["spline_wave"][:], original.spline_wave[:, :, :50], rtol=1e-14
            )
