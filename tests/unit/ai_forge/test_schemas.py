"""Tests for AI Forge Pydantic schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from forgequant.ai_forge.schemas import (
    BlockSpec,
    StrategyConstraints,
    StrategyObjective,
    StrategySpec,
)


class TestBlockSpec:
    def test_valid_creation(self) -> None:
        spec = BlockSpec(block_name="ema", params={"period": 20})
        assert spec.block_name == "ema"
        assert spec.params == {"period": 20}

    def test_name_normalized(self) -> None:
        spec = BlockSpec(block_name="  EMA  ")
        assert spec.block_name == "ema"

    def test_empty_params_default(self) -> None:
        spec = BlockSpec(block_name="rsi")
        assert spec.params == {}

    def test_with_rationale(self) -> None:
        spec = BlockSpec(
            block_name="ema",
            params={"period": 50},
            rationale="Long-term trend identification",
        )
        assert spec.rationale == "Long-term trend identification"

    def test_empty_name_raises(self) -> None:
        with pytest.raises(ValidationError):
            BlockSpec(block_name="")


class TestStrategyObjective:
    def test_valid_creation(self) -> None:
        obj = StrategyObjective(
            style="trend_following",
            timeframe="1h",
            instruments=["EURUSD"],
        )
        assert obj.style == "trend_following"
        assert obj.timeframe == "1h"

    def test_style_normalized(self) -> None:
        obj = StrategyObjective(
            style="Trend Following",
            timeframe="4h",
        )
        assert obj.style == "trend_following"

    def test_direction_default_both(self) -> None:
        obj = StrategyObjective(style="breakout", timeframe="1d")
        assert obj.direction == "both"

    def test_invalid_direction_defaults_both(self) -> None:
        obj = StrategyObjective(
            style="breakout", timeframe="1d", direction="invalid"
        )
        assert obj.direction == "both"


class TestStrategyConstraints:
    def test_defaults(self) -> None:
        c = StrategyConstraints()
        assert c.min_trades == 150
        assert c.max_drawdown == 0.18
        assert c.min_profit_factor == 1.35
        assert c.min_sharpe == 0.80
        assert c.min_win_rate == 0.35
        assert c.max_correlation == 0.70

    def test_custom_values(self) -> None:
        c = StrategyConstraints(min_trades=200, max_drawdown=0.25)
        assert c.min_trades == 200
        assert c.max_drawdown == 0.25

    def test_invalid_drawdown_raises(self) -> None:
        with pytest.raises(ValidationError):
            StrategyConstraints(max_drawdown=0.0)

    def test_invalid_win_rate_raises(self) -> None:
        with pytest.raises(ValidationError):
            StrategyConstraints(min_win_rate=1.5)


def _make_valid_spec(**overrides: object) -> dict:
    base = {
        "name": "test_strategy",
        "description": "A test strategy for unit testing purposes with sufficient detail.",
        "objective": {
            "style": "trend_following",
            "timeframe": "1h",
            "instruments": ["EURUSD"],
        },
        "indicators": [
            {"block_name": "ema", "params": {"period": 20}},
        ],
        "entry_rules": [
            {"block_name": "crossover_entry"},
        ],
        "exit_rules": [
            {"block_name": "fixed_tpsl"},
        ],
        "money_management": {"block_name": "fixed_risk"},
    }
    base.update(overrides)
    return base


class TestStrategySpec:
    def test_valid_creation(self) -> None:
        spec = StrategySpec(**_make_valid_spec())
        assert spec.name == "test_strategy"
        assert len(spec.indicators) == 1
        assert len(spec.entry_rules) == 1

    def test_name_normalized(self) -> None:
        spec = StrategySpec(**_make_valid_spec(name="My Test Strategy"))
        assert spec.name == "my_test_strategy"

    def test_all_blocks(self) -> None:
        spec = StrategySpec(**_make_valid_spec())
        all_blocks = spec.all_blocks()
        assert len(all_blocks) == 4

    def test_block_names(self) -> None:
        spec = StrategySpec(**_make_valid_spec())
        names = spec.block_names()
        assert "ema" in names
        assert "crossover_entry" in names
        assert "fixed_tpsl" in names
        assert "fixed_risk" in names

    def test_no_indicators_raises(self) -> None:
        with pytest.raises(ValidationError, match="indicator"):
            StrategySpec(**_make_valid_spec(indicators=[]))

    def test_no_entry_rules_raises(self) -> None:
        with pytest.raises(ValidationError, match="entry"):
            StrategySpec(**_make_valid_spec(entry_rules=[]))

    def test_no_exit_rules_raises(self) -> None:
        with pytest.raises(ValidationError, match="exit"):
            StrategySpec(**_make_valid_spec(exit_rules=[]))

    def test_duplicate_indicator_raises(self) -> None:
        with pytest.raises(ValidationError, match="Duplicate"):
            StrategySpec(
                **_make_valid_spec(
                    indicators=[
                        {"block_name": "ema", "params": {"period": 20}},
                        {"block_name": "ema", "params": {"period": 50}},
                    ]
                )
            )

    def test_with_optional_blocks(self) -> None:
        data = _make_valid_spec()
        data["price_action"] = [{"block_name": "breakout"}]
        data["filters"] = [{"block_name": "trend_filter"}]
        spec = StrategySpec(**data)
        assert len(spec.price_action) == 1
        assert len(spec.filters) == 1
        assert len(spec.all_blocks()) == 6

    def test_short_description_raises(self) -> None:
        with pytest.raises(ValidationError):
            StrategySpec(**_make_valid_spec(description="Too short"))

    def test_json_roundtrip(self) -> None:
        spec = StrategySpec(**_make_valid_spec())
        json_str = spec.model_dump_json()
        restored = StrategySpec.model_validate_json(json_str)
        assert restored.name == spec.name
        assert len(restored.all_blocks()) == len(spec.all_blocks())

    def test_default_constraints(self) -> None:
        spec = StrategySpec(**_make_valid_spec())
        assert spec.constraints.min_trades == 150
