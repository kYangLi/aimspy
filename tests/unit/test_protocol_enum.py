"""Unit tests for ExternalMatrixSource Protocol, CallbackName Enum, Strategy validation."""

from __future__ import annotations

import pytest

from aimspy import Strategy, CallbackName, ExternalMatrixSource, AimspyConfigError
from aimspy.interface.deeph import DeepHData


class TestStrategy:
    def test_strategy_values(self):
        assert Strategy.REPLACE.value == "replace"
        assert Strategy.ADD.value == "add"
        assert Strategy.SCALE.value == "scale"
        assert Strategy.CUSTOM.value == "custom"

    def test_strategy_from_string(self):
        assert Strategy("replace") == Strategy.REPLACE
        assert Strategy("add") == Strategy.ADD
        assert Strategy("scale") == Strategy.SCALE
        assert Strategy("custom") == Strategy.CUSTOM

    def test_strategy_from_uppercase_string(self):
        """modify() accepts and lowercases string strategies."""
        from aimspy import Calculator, CalculatorConfig
        from aimspy.interface.deeph import DeepHData

        # Need a source for direct REPLACE mode
        dd = DeepHData.__new__(DeepHData)
        calc = Calculator(CalculatorConfig(lib_path="/tmp/x.so"))
        calc.modify(source=dd, strategy="REPLACE")
        assert calc._modify.strategy == Strategy.REPLACE

    def test_strategy_invalid_string_raises_config_error(self):
        from aimspy import Calculator, CalculatorConfig

        calc = Calculator(CalculatorConfig(lib_path="/tmp/x.so"))
        with pytest.raises(AimspyConfigError, match="invalid strategy"):
            calc.modify(strategy="bogus")

    def test_strategy_custom_without_fn_raises(self):
        from aimspy import Calculator, CalculatorConfig

        calc = Calculator(CalculatorConfig(lib_path="/tmp/x.so"))
        with pytest.raises(AimspyConfigError, match="CUSTOM.*custom_fn"):
            calc.modify(strategy=Strategy.CUSTOM)


class TestCallbackName:
    def test_callback_name_values(self):
        assert CallbackName.GET_DESCR.value == "get_descr"
        assert CallbackName.EXPORT_OVLP.value == "export_ovlp"
        assert CallbackName.EXPORT_H0.value == "export_h0"
        assert CallbackName.MODIFY_H0.value == "modify_h0"
        assert CallbackName.PYTHON_FUNC.value == "python_func"

    def test_all_callback_names(self):
        names = [n.value for n in CallbackName]
        assert len(names) == 5

    def test_register_callback_accepts_enum(self):
        """register_callback should accept CallbackName as well as str."""
        from aimspy import Calculator, CalculatorConfig

        calc = Calculator(CalculatorConfig(lib_path="/tmp/x.so"))
        calc.register_callback(CallbackName.EXPORT_H0, lambda aux, h, nh, ns: None)
        assert calc.callback_registered("export_h0")
        assert calc.callback_registered(CallbackName.EXPORT_H0)

    def test_register_callback_accepts_str(self):
        from aimspy import Calculator, CalculatorConfig

        calc = Calculator(CalculatorConfig(lib_path="/tmp/x.so"))
        calc.register_callback("export_h0", lambda aux, h, nh, ns: None)
        assert calc.callback_registered("export_h0")


class TestExternalMatrixSource:
    def test_deephdata_satisfies_protocol(self):
        """DeepHData has to_aimspy method → satisfies ExternalMatrixSource."""
        dd = DeepHData.__new__(DeepHData)
        assert hasattr(dd, "to_aimspy")

    def test_protocol_runtime_check(self):
        """isinstance(obj, ExternalMatrixSource) checks for to_aimspy method."""
        dd = DeepHData.__new__(DeepHData)
        assert isinstance(dd, ExternalMatrixSource)

    def test_random_object_fails_protocol(self):
        """An object without to_aimspy should not satisfy the protocol."""
        assert not isinstance(42, ExternalMatrixSource)
        assert not isinstance("hello", ExternalMatrixSource)

    def test_namespace_with_to_aimspy_satisfies(self):
        """A SimpleNamespace with to_aimspy also satisfies (deferred mode)."""
        from types import SimpleNamespace

        wrapper = SimpleNamespace(to_aimspy=lambda s: None)
        assert isinstance(wrapper, ExternalMatrixSource)
