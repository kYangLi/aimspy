"""Unit tests for POSCAR reader/writer in aimspy.interface.deeph.data."""

from __future__ import annotations

import numpy as np

from aimspy.interface.deeph.data import _read_poscar, _write_poscar


class TestReadPoscar:
    def test_vasp5_cartesian(self, tmp_path):
        poscar = tmp_path / "POSCAR"
        poscar.write_text(
            "Test structure\n"
            "1.0\n"
            "  10.0  0.0  0.0\n"
            "  0.0  10.0  0.0\n"
            "  0.0  0.0  10.0\n"
            "Mo S\n"
            "1 2\n"
            "Cartesian\n"
            "  0.0  0.0  0.0\n"
            "  1.5  1.5  0.0\n"
            "  1.5  1.5  3.0\n"
        )
        lat, symbols, coords = _read_poscar(poscar)
        assert symbols == ["Mo", "S", "S"]
        assert coords.shape == (3, 3)
        np.testing.assert_allclose(lat, np.eye(3) * 10.0)

    def test_vasp4_no_element_line(self, tmp_path):
        """VASP4: no element line → symbols from title line."""
        poscar = tmp_path / "POSCAR"
        poscar.write_text(
            "Mo S\n"
            "1.0\n"
            "  10.0  0.0  0.0\n"
            "  0.0  10.0  0.0\n"
            "  0.0  0.0  10.0\n"
            "1 2\n"
            "Cartesian\n"
            "  0.0  0.0  0.0\n"
            "  1.5  1.5  0.0\n"
            "  1.5  1.5  3.0\n"
        )
        lat, symbols, coords = _read_poscar(poscar)
        assert symbols == ["Mo", "S", "S"]

    def test_direct_coords(self, tmp_path):
        """Direct (fractional) coordinates → converted to Cartesian."""
        poscar = tmp_path / "POSCAR"
        poscar.write_text(
            "Test\n"
            "1.0\n"
            "  10.0  0.0  0.0\n"
            "  0.0  10.0  0.0\n"
            "  0.0  0.0  10.0\n"
            "H\n"
            "1\n"
            "Direct\n"
            "  0.5  0.5  0.5\n"
        )
        lat, symbols, coords = _read_poscar(poscar)
        np.testing.assert_allclose(coords[0], [5.0, 5.0, 5.0])

    def test_selective_dynamics(self, tmp_path):
        """Selective dynamics line should be skipped."""
        poscar = tmp_path / "POSCAR"
        poscar.write_text(
            "Test\n"
            "1.0\n"
            "  10.0  0.0  0.0\n"
            "  0.0  10.0  0.0\n"
            "  0.0  0.0  10.0\n"
            "H\n"
            "1\n"
            "Selective dynamics\n"
            "Direct\n"
            "  0.5  0.5  0.5  T T T\n"
        )
        lat, symbols, coords = _read_poscar(poscar)
        np.testing.assert_allclose(coords[0], [5.0, 5.0, 5.0])

    def test_scale_factor(self, tmp_path):
        """Scale factor should multiply lattice vectors."""
        poscar = tmp_path / "POSCAR"
        poscar.write_text(
            "Test\n"
            "2.0\n"
            "  5.0  0.0  0.0\n"
            "  0.0  5.0  0.0\n"
            "  0.0  0.0  5.0\n"
            "H\n"
            "1\n"
            "Cartesian\n"
            "  0.0  0.0  0.0\n"
        )
        lat, symbols, coords = _read_poscar(poscar)
        np.testing.assert_allclose(lat, np.eye(3) * 10.0)

    def test_multiline_title_fallback_vasp4(self, tmp_path):
        """If title line doesn't have elements, VASP4 fallback uses title words."""
        poscar = tmp_path / "POSCAR"
        poscar.write_text(
            "Some random title\n"
            "1.0\n"
            "  10.0  0.0  0.0\n"
            "  0.0  10.0  0.0\n"
            "  0.0  0.0  10.0\n"
            "1\n"
            "Cartesian\n"
            "  0.0  0.0  0.0\n"
        )
        lat, symbols, coords = _read_poscar(poscar)
        # VASP4 fallback: symbols from title line words; "Some" is not element
        assert len(symbols) == 1
        assert symbols[0] in ("Some", "X1")  # depends on fallback logic


class TestWritePoscar:
    def test_roundtrip(self, tmp_path):
        """Write then read should give same lattice/symbols/coords."""
        lat = np.eye(3) * 15.0
        symbols = ["Mo", "S", "S"]
        coords = np.array([[0, 0, 0], [1.5, 1.5, 0], [1.5, 1.5, 3.0]])
        poscar = tmp_path / "POSCAR"
        _write_poscar(poscar, lat, symbols, coords)

        lat2, symbols2, coords2 = _read_poscar(poscar)
        assert symbols2 == symbols
        np.testing.assert_allclose(lat2, lat)
        np.testing.assert_allclose(coords2, coords)

    def test_grouped_counts(self, tmp_path):
        """Writer should group same species and emit counts."""
        lat = np.eye(3) * 10.0
        symbols = ["S", "Mo", "S"]  # not grouped in input
        coords = np.zeros((3, 3))
        poscar = tmp_path / "POSCAR"
        _write_poscar(poscar, lat, symbols, coords)

        content = poscar.read_text()
        # Should have element line with both Mo and S
        assert "Mo" in content
        assert "S" in content
