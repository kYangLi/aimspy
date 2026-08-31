"""Unit tests for energy, force, and analytical-stress exports.

These tests use synthetic ctypes buffers and do not require MPI or libaims.
"""

from __future__ import annotations

from ctypes import CDLL, POINTER, c_double, cast
from types import SimpleNamespace

import numpy as np
import pytest

from aimspy import (
    BOHR_TO_ANG,
    HARTREE_TO_EV,
    AimspyBindingError,
    AimspyConfigError,
    AimspyStateError,
    Calculator,
    CalculatorConfig,
    CalcState,
    get_stress,
)
from aimspy._binding.prototypes import BindingLib, _PROTOTYPES
from aimspy.observables import get_free_atom_reference_energies
from tests.mpi_utils import synchronized_python_exceptions


class _StressBinding:
    def __init__(self, values=None):
        self.buffer = None
        if values is not None:
            self.buffer = (c_double * 9)(*values)

    def aimspy_stress(self):
        if self.buffer is None:
            return POINTER(c_double)()
        return cast(self.buffer, POINTER(c_double))


class _ReferenceBinding:
    def __init__(self, values=None):
        self.buffer = None
        if values is not None:
            self.buffer = (c_double * len(values))(*values)

    def aimspy_free_atom_reference_energies(self):
        if self.buffer is None:
            return POINTER(c_double)()
        return cast(self.buffer, POINTER(c_double))


class _FakeGatherComm:
    def __init__(self, remote_error=None):
        self.remote_error = remote_error
        self.local_error = None

    def allgather(self, value):
        self.local_error = value
        return [value, self.remote_error]


def _calculator() -> Calculator:
    return Calculator(CalculatorConfig(lib_path="/dummy/libaims.so"))


def test_observable_prototypes_are_registered():
    assert _PROTOTYPES["aimspy_energy_raw"][1] is c_double
    assert _PROTOTYPES["aimspy_energy_free"][1] is c_double
    assert _PROTOTYPES["aimspy_stress"][1] == POINTER(c_double)
    assert _PROTOTYPES["aimspy_free_atom_reference_energies"][1] == POINTER(c_double)


def test_free_atom_reference_reader_returns_owned_finite_copy():
    binding = _ReferenceBinding([-10.0, -2.5])

    values = get_free_atom_reference_energies(binding, 2)

    np.testing.assert_array_equal(values, [-10.0, -2.5])
    assert values.flags.c_contiguous
    binding.buffer[0] = 100.0
    assert values[0] == -10.0


def test_free_atom_reference_reader_rejects_null_pointer():
    with pytest.raises(AimspyConfigError, match="unavailable"):
        get_free_atom_reference_energies(_ReferenceBinding(), 2)


def test_free_atom_reference_reader_rejects_nonfinite_values():
    with pytest.raises(AimspyConfigError, match="finite"):
        get_free_atom_reference_energies(_ReferenceBinding([-1.0, np.nan]), 2)


def test_get_stress_returns_none_for_null_pointer():
    assert get_stress(_StressBinding()) is None


def test_get_stress_preserves_fortran_layout_units_and_ownership():
    binding = _StressBinding(range(1, 10))
    stress = get_stress(binding)

    conversion = HARTREE_TO_EV / BOHR_TO_ANG**3
    expected_au = np.array([[1.0, 4.0, 7.0], [2.0, 5.0, 8.0], [3.0, 6.0, 9.0]])
    np.testing.assert_allclose(stress, expected_au * conversion)
    assert stress.shape == (3, 3)
    assert stress.flags.c_contiguous

    binding.buffer[0] = 100.0
    assert stress[0, 0] == pytest.approx(conversion)


def test_final_energy_properties_use_dedicated_symbols():
    calc = _calculator()
    calc._state = CalcState.DONE
    calc._binding = SimpleNamespace(
        aimspy_energy_raw=lambda: -12.5,
        aimspy_energy_free=lambda: -12.75,
    )

    assert calc.energy_raw == pytest.approx(-12.5)
    assert calc.energy_free == pytest.approx(-12.75)


def test_free_atom_reference_properties_and_relative_energy():
    calc = _calculator()
    calc._state = CalcState.DONE
    binding = _ReferenceBinding([-10.0, -2.5])
    binding.aimspy_energy_free = lambda: -20.0
    calc._binding = binding
    calc._info = SimpleNamespace(
        n_species=2,
        species_idx=np.array([0, 1, 1], dtype=np.int32),
    )

    returned = calc.free_atom_reference_energies
    np.testing.assert_array_equal(returned, [-10.0, -2.5])
    assert calc.free_atom_reference_energy == pytest.approx(-15.0)
    assert calc.energy_free_relative == pytest.approx(-5.0)

    returned[0] = 999.0
    np.testing.assert_array_equal(calc.free_atom_reference_energies, [-10.0, -2.5])

    calc._info.species_idx = np.array([0, 1, 1, 0, 1, 1], dtype=np.int32)
    assert calc.free_atom_reference_energy == pytest.approx(-30.0)
    assert calc.energy_free_relative == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("energy_raw", "energy_raw"),
        ("energy_free", "energy_free"),
        ("free_atom_reference_energies", "free_atom_reference_energies"),
        ("free_atom_reference_energy", "free_atom_reference_energies"),
        ("energy_free_relative", "energy_free_relative"),
    ],
)
def test_energy_and_reference_properties_reject_uninitialized_state(name, message):
    calc = _calculator()
    with pytest.raises(AimspyStateError, match=message):
        getattr(calc, name)


@pytest.mark.parametrize("name", ["energy_raw", "energy_free"])
def test_new_energy_properties_report_old_library(name):
    calc = _calculator()
    calc._state = CalcState.DONE
    calc._binding = BindingLib(CDLL(None))

    with pytest.raises(AimspyBindingError, match=name):
        getattr(calc, name)


def test_free_atom_reference_properties_report_old_library():
    calc = _calculator()
    calc._state = CalcState.INITED
    calc._binding = BindingLib(CDLL(None))
    calc._info = SimpleNamespace(n_species=2)

    with pytest.raises(AimspyBindingError, match="free_atom_reference"):
        calc.free_atom_reference_energies


def test_calc_eagerly_captures_an_independent_stress_copy(tmp_path):
    stress_binding = _StressBinding(range(1, 10))
    stress_binding.aimspy_run = lambda: None
    stress_binding.aimspy_forces = lambda: POINTER(c_double)()

    calc = _calculator()
    calc._state = CalcState.INITED
    calc._binding = stress_binding
    calc._work_dir = tmp_path
    calc._info = SimpleNamespace(n_atoms=2)

    calc.calc()
    captured = calc.stress.copy()
    stress_binding.buffer[0] = 100.0

    np.testing.assert_array_equal(calc.stress, captured)
    assert calc._state is CalcState.DONE


def test_calc_preserves_stress_capture_failure_until_property_access(tmp_path):
    stress_binding = _StressBinding()
    stress_binding.aimspy_run = lambda: None
    stress_binding.aimspy_forces = lambda: POINTER(c_double)()

    def fail_stress_capture():
        raise AimspyBindingError("missing aimspy_stress ABI")

    stress_binding.aimspy_stress = fail_stress_capture

    calc = _calculator()
    calc._state = CalcState.INITED
    calc._binding = stress_binding
    calc._work_dir = tmp_path
    calc._info = SimpleNamespace(n_atoms=2)

    calc.calc()

    assert calc._state is CalcState.DONE
    with pytest.raises(
        AimspyBindingError,
        match="stress capture failed.*missing aimspy_stress ABI",
    ):
        _ = calc.stress


def test_stress_is_none_before_calc_and_after_force_close():
    calc = _calculator()
    assert calc.stress is None

    calc._stress = np.eye(3)
    calc._stress_capture_error = "stale failure"
    calc.force_close()
    assert calc.stress is None


def test_synchronized_python_exceptions_passes_when_all_ranks_succeed():
    comm = _FakeGatherComm()

    with synchronized_python_exceptions(comm):
        pass

    assert comm.local_error is None


def test_synchronized_python_exceptions_propagates_local_traceback():
    comm = _FakeGatherComm()

    with pytest.raises(RuntimeError) as exc_info:
        with synchronized_python_exceptions(comm):
            raise ValueError("local export failed")

    message = str(exc_info.value)
    assert "rank 0" in message
    assert "ValueError: local export failed" in message
    assert comm.local_error is not None


def test_synchronized_python_exceptions_propagates_remote_traceback():
    comm = _FakeGatherComm("RuntimeError: remote export failed")

    with pytest.raises(RuntimeError) as exc_info:
        with synchronized_python_exceptions(comm):
            pass

    message = str(exc_info.value)
    assert "rank 1" in message
    assert "RuntimeError: remote export failed" in message
