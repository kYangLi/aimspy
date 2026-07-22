"""Unit tests for force_close() and CalcState.FAILED transitions.

These tests do NOT require MPI or libaims — they exercise the state
machine and error-handling paths only.
"""

from __future__ import annotations

import pytest

from aimspy import Calculator, CalculatorConfig, CalcState


@pytest.fixture
def calc():
    """A Calculator in UNINIT state (no lib_path needed, never init'd)."""
    return Calculator(CalculatorConfig(lib_path="/dummy/libaims.so"))


class TestForceClose:
    def test_force_close_from_uninit(self, calc):
        """force_close() from UNINIT transitions to FINALIZED."""
        calc.force_close()
        assert calc._state == CalcState.FINALIZED

    def test_force_close_idempotent(self, calc):
        """Calling force_close() twice should not raise."""
        calc.force_close()
        calc.force_close()
        assert calc._state == CalcState.FINALIZED

    def test_force_close_clears_state(self, calc):
        """force_close() should clear all retained state."""
        calc._forces = [1, 2, 3]  # simulate cached forces
        calc.force_close()
        assert calc._forces is None
        assert calc._info is None
        assert calc._structure is None
        assert calc._binding is None
        assert calc._cb_mgr is None
        assert calc._runtime_aux is None

    def test_force_close_after_manual_failed_state(self, calc):
        """force_close() from a manually-set FAILED state should work."""
        calc._state = CalcState.FAILED
        calc.force_close()
        assert calc._state == CalcState.FINALIZED


class TestCloseFromStates:
    def test_close_from_uninit_is_noop(self, calc):
        """close() from UNINIT should be a silent no-op."""
        calc.close()
        assert calc._state == CalcState.UNINIT

    def test_close_from_finalized_is_noop(self, calc):
        """close() from FINALIZED should be a silent no-op."""
        calc.force_close()
        calc.close()
        assert calc._state == CalcState.FINALIZED

    def test_close_from_running_raises(self, calc):
        """close() from RUNNING should raise AimspyStateError."""
        from aimspy import AimspyStateError

        calc._state = CalcState.RUNNING
        with pytest.raises(AimspyStateError, match="force_close"):
            calc.close()


class TestForcesProperty:
    def test_forces_none_before_calc(self, calc):
        """forces should return None before calc() is called."""
        assert calc.forces is None

    def test_forces_after_force_close(self, calc):
        """forces should return None after force_close clears state."""
        calc._forces = [1, 2, 3]
        calc.force_close()
        assert calc.forces is None
