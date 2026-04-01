"""Tests for AI Forge spec validator."""

from __future__ import annotations

from typing import Any

import pytest

from forgequant.ai_forge.schemas import BlockSpec, StrategySpec
from forgequant.ai_forge.validator import SpecValidator, ValidationResult
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.types import BlockCategory

# Import blocks so they register
import forgequant.blocks.indicators  # noqa: F401
import forgequant.blocks.entry_rules  # noqa: F401
import forgequant.blocks.exit_rules  # noqa: F401
import forgequant.blocks.money_management  # noqa: F401
import forgequant.blocks.filters  # noqa: F401
import forgequant.blocks.price_action  # noqa: F401


def _make_spec(**overrides: Any) -> StrategySpec:
    base: dict[str, Any] = {
        "name": "test_strategy",
        "description": "A comprehensive test strategy for validation testing purposes.",
        "objective": {
            "style": "trend_following",
            "timeframe": "1h",
            "instruments": ["EURUSD"],
        },
        "indicators": [
            {"block_name": "ema", "params": {"period": 20}},
        ],
        "entry_rules": [
            {"block_name": "crossover_entry", "params": {"fast_period": 10, "slow_period": 20}},
        ],
        "exit_rules": [
            {"block_name": "fixed_tpsl", "params": {"atr_period": 14}},
        ],
        "money_management": {"block_name": "fixed_risk", "params": {"atr_period": 14}},
    }
    base.update(overrides)
    return StrategySpec(**base)


@pytest.fixture
def validator() -> SpecValidator:
    registry = BlockRegistry()
    for module in [
        forgequant.blocks.indicators,
        forgequant.blocks.entry_rules,
        forgequant.blocks.exit_rules,
        forgequant.blocks.money_management,
        forgequant.blocks.filters,
        forgequant.blocks.price_action,
    ]:
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                isinstance(attr, type)
                and hasattr(attr, "metadata")
                and attr_name not in ("BaseBlock",)
            ):
                try:
                    registry.register_class(attr)
                except Exception:
                    pass
    return SpecValidator(registry)


class TestValidationResult:
    def test_default_valid(self) -> None:
        r = ValidationResult()
        assert r.is_valid is True
        assert r.errors == []
        assert r.warnings == []

    def test_add_error(self) -> None:
        r = ValidationResult()
        r.add_error("something broke")
        assert r.is_valid is False
        assert len(r.errors) == 1

    def test_add_warning(self) -> None:
        r = ValidationResult()
        r.add_warning("watch out")
        assert r.is_valid is True
        assert len(r.warnings) == 1


class TestSpecValidator:
    def test_valid_spec_passes(self, validator: SpecValidator) -> None:
        spec = _make_spec()
        result = validator.validate(spec)
        assert result.is_valid, f"Expected valid, got errors: {result.errors}"

    def test_invalid_block_name_fails(self, validator: SpecValidator) -> None:
        spec = _make_spec(
            indicators=[
                {"block_name": "nonexistent_indicator"},
            ]
        )
        result = validator.validate(spec)
        assert not result.is_valid
        assert any("not found" in e for e in result.errors)

    def test_wrong_category_fails(self, validator: SpecValidator) -> None:
        spec = _make_spec(
            entry_rules=[
                {"block_name": "ema"},
            ]
        )
        result = validator.validate(spec)
        assert not result.is_valid
        assert any("category" in e.lower() for e in result.errors)

    def test_invalid_params_fail(self, validator: SpecValidator) -> None:
        spec = _make_spec(
            indicators=[
                {"block_name": "ema", "params": {"period": -5}},
            ]
        )
        result = validator.validate(spec)
        assert not result.is_valid
        assert any("parameter" in e.lower() or "validation" in e.lower() for e in result.errors)

    def test_unknown_params_fail(self, validator: SpecValidator) -> None:
        spec = _make_spec(
            indicators=[
                {"block_name": "ema", "params": {"bogus_param": 42}},
            ]
        )
        result = validator.validate(spec)
        assert not result.is_valid
        assert any("unknown" in e.lower() for e in result.errors)

    def test_validated_params_populated(self, validator: SpecValidator) -> None:
        spec = _make_spec()
        result = validator.validate(spec)
        assert "ema" in result.validated_params
        assert "period" in result.validated_params["ema"]
        assert "source" in result.validated_params["ema"]

    def test_no_filters_produces_warning(self, validator: SpecValidator) -> None:
        spec = _make_spec(filters=[])
        result = validator.validate(spec)
        assert any("filter" in w.lower() for w in result.warnings)

    def test_atr_period_mismatch_warning(self, validator: SpecValidator) -> None:
        spec = _make_spec(
            exit_rules=[
                {"block_name": "fixed_tpsl", "params": {"atr_period": 7}},
            ],
            money_management={"block_name": "fixed_risk", "params": {"atr_period": 14}},
        )
        result = validator.validate(spec)
        assert any("atr period" in w.lower() for w in result.warnings)

    def test_full_spec_with_all_categories(self, validator: SpecValidator) -> None:
        spec = _make_spec(
            indicators=[
                {"block_name": "ema", "params": {"period": 50}},
                {"block_name": "rsi", "params": {"period": 14}},
            ],
            price_action=[
                {"block_name": "breakout", "params": {"lookback": 20}},
            ],
            entry_rules=[
                {"block_name": "crossover_entry", "params": {"fast_period": 10, "slow_period": 50}},
            ],
            exit_rules=[
                {"block_name": "fixed_tpsl", "params": {"atr_period": 14}},
                {"block_name": "trailing_stop", "params": {"atr_period": 14}},
            ],
            filters=[
                {"block_name": "trend_filter", "params": {"period": 200}},
                {"block_name": "trading_session"},
            ],
            money_management={"block_name": "fixed_risk", "params": {"atr_period": 14}},
        )
        result = validator.validate(spec)
        assert result.is_valid, f"Expected valid, got errors: {result.errors}"
        assert len(result.validated_params) >= 8
