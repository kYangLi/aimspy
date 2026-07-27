"""Unit tests for force_close() and CalcState.FAILED transitions.

These tests do NOT require MPI or libaims — they exercise the state
machine and error-handling paths only.
"""

from __future__ import annotations

import gc

import pytest

from aimspy import Calculator, CalculatorConfig, CalcState, Strategy


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


# =============================================================================
# CallbackManager.clear() and reference-cycle regression tests
# =============================================================================
class TestCallbackManagerClear:
    """Verify that CallbackManager.clear() empties all internal containers
    and that Calculator._clear_all_state() invokes it."""

    def test_clear_empties_all_dicts(self):
        """clear() should empty _wrapped, _auxs, _pyobjs, _errors."""
        from aimspy._callbacks.base import CallbackManager

        mgr = CallbackManager(binding=None)
        mgr._wrapped["x"] = ("wrapper", "fn")
        mgr._auxs["x"] = {"data": 1}
        mgr._pyobjs["x"] = "fake_pyobj"
        mgr._errors.append(("x", Exception("test"), "tb"))
        mgr.clear()
        assert mgr._wrapped == {}
        assert mgr._auxs == {}
        assert mgr._pyobjs == {}
        assert mgr._errors == []

    def test_clear_is_idempotent(self):
        """Calling clear() twice should not raise."""
        from aimspy._callbacks.base import CallbackManager

        mgr = CallbackManager(binding=None)
        mgr.clear()
        mgr.clear()
        assert mgr._wrapped == {}

    def test_clear_all_state_calls_cb_mgr_clear(self, calc):
        """_clear_all_state should call _cb_mgr.clear() before dropping it."""
        from unittest.mock import MagicMock

        mock_mgr = MagicMock()
        calc._cb_mgr = mock_mgr
        calc._state = CalcState.DONE
        calc._clear_all_state()
        mock_mgr.clear.assert_called_once()

    def test_clear_all_state_clears_modify_internal(self, calc):
        """_clear_all_state should clear _modify internal fields."""
        from types import SimpleNamespace

        calc._modify = SimpleNamespace(
            source=object(),
            custom_fn=lambda: None,
            deferred_fn=lambda: None,
            deferred_option={"k": "v"},
            strategy=Strategy.REPLACE,
            factor=1.0,
        )
        calc._state = CalcState.DONE
        calc._clear_all_state()
        assert calc._modify is None

    def test_record_callback_error_accepts_list(self):
        """_record_callback_error should accept a plain list, not a manager."""
        from aimspy._callbacks.base import _record_callback_error

        errors: list = []
        _record_callback_error(errors, "test_cb", ValueError("boom"))
        assert len(errors) == 1
        name, exc, tb_str = errors[0]
        assert name == "test_cb"
        assert isinstance(exc, ValueError)


# =============================================================================
# Reference-cycle regression tests
# =============================================================================
class TestNoReferenceCycles:
    """Verify that close()/force_close() leaves no reference cycles, so
    the Calculator (and its large aux data) is reclaimable by refcount
    alone — without waiting for the cyclic garbage collector.
    """

    def test_calculator_collectable_after_force_close(self):
        """After force_close(), Calculator should be immediately
        collectable by refcount (no cyclic GC needed)."""
        import weakref

        calc = Calculator(CalculatorConfig(lib_path="/dummy"))
        calc.force_close()
        ref = weakref.ref(calc)
        del calc
        gc.collect()
        assert ref() is None, "Calculator leaked — reference cycle exists"

    def test_deferred_modify_no_cycle(self):
        """Deferred modify_init_ham decorator should not create a
        reference cycle via _modify."""
        import weakref

        calc = Calculator(CalculatorConfig(lib_path="/dummy"))

        @calc.modify_init_ham(strategy=Strategy.REPLACE, option={"path": "/tmp"})
        def gen_source(calculator, option):
            return None  # dummy — never actually called in this test

        ref = weakref.ref(calc)
        calc.force_close()
        del calc
        gc.collect()
        assert ref() is None, "Calculator leaked — deferred modify cycle exists"

    def test_callback_manager_collectable_after_clear(self):
        """CallbackManager should be collectable after clear() even if
        wrappers were registered (wrapper must not capture the manager)."""
        import weakref

        from aimspy._callbacks.base import CallbackManager

        mgr = CallbackManager(binding=None)
        # Simulate a registered callback's internal state
        mgr._wrapped["test"] = (object(), lambda ax: None)
        mgr._auxs["test"] = {}
        mgr._pyobjs["test"] = None
        ref = weakref.ref(mgr)
        mgr.clear()
        del mgr
        gc.collect()
        assert ref() is None, "CallbackManager leaked — wrapper captures manager"

    def test_aux_overlap_released_after_close(self):
        """Large aux data (simulated overlap) should be released when
        close() is called, not retained by a reference cycle."""
        import weakref

        import numpy as np

        calc = Calculator(CalculatorConfig(lib_path="/dummy"))
        # Simulate a large overlap matrix held in aux (ndarray supports weakref)
        big_data = np.zeros(100, dtype=np.float64)
        calc._runtime_aux = {
            "overlap": big_data,
            "structure": None,
            "cfg": None,
            "modify": None,
            "csr_descr": None,
            "initial_hamiltonian": None,
            "external_aimspy": None,
            "rank": 0,
        }
        data_ref = weakref.ref(big_data)
        calc.force_close()
        # After force_close, _runtime_aux is None, so big_data should be
        # collectable if nothing else references it.
        del big_data
        gc.collect()
        assert data_ref() is None, "aux overlap data leaked after close()"

    def test_modify_source_released_after_close(self):
        """_modify.source (direct mode) should be released after close()."""
        import weakref

        import numpy as np

        calc = Calculator(CalculatorConfig(lib_path="/dummy"))
        # Use ndarray (supports weakref) instead of object()
        source_obj = np.zeros(10, dtype=np.float64)
        from types import SimpleNamespace

        calc._modify = SimpleNamespace(
            source=source_obj,
            custom_fn=None,
            strategy=Strategy.REPLACE,
            factor=1.0,
        )
        src_ref = weakref.ref(source_obj)
        calc.force_close()
        del source_obj
        gc.collect()
        assert src_ref() is None, "_modify.source leaked after close()"
