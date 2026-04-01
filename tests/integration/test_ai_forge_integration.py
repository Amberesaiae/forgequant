"""
Integration test for the AI Forge module.

Tests the complete flow from schemas through validation,
without requiring actual LLM API calls.
"""

from __future__ import annotations

import importlib
from typing import Any

import pytest

from forgequant.ai_forge.prompt import build_system_prompt, build_user_message
from forgequant.ai_forge.schemas import BlockSpec, StrategySpec
from forgequant.ai_forge.validator import SpecValidator
from forgequant.blocks.registry import BlockRegistry


def _populate_registry(registry: BlockRegistry) -> None:
    for mod_name in [
        "forgequant.blocks.indicators",
        "forgequant.blocks.entry_rules",
        "forgequant.blocks.exit_rules",
        "forgequant.blocks.money_management",
        "forgequant.blocks.filters",
        "forgequant.blocks.price_action",
    ]:
        module = importlib.import_module(mod_name)
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and hasattr(attr, "metadata") and attr_name != "BaseBlock":
                try:
                    registry.register_class(attr)
                except Exception:
                    pass


@pytest.fixture
def full_registry(clean_registry: BlockRegistry) -> BlockRegistry:
    _populate_registry(clean_registry)
    return clean_registry


class TestEndToEndSpecCreation:
    def test_trend_following_spec(self, full_registry: BlockRegistry) -> None:
        spec = StrategySpec(
            name="ema_trend_follower",
            description="Trend following strategy using EMA crossover with ATR-based exits and risk management.",
            objective={
                "style": "trend_following",
                "timeframe": "1h",
                "instruments": ["EURUSD"],
            },
            indicators=[
                BlockSpec(block_name="ema", params={"period": 50}),
                BlockSpec(block_name="atr", params={"period": 14}),
            ],
            entry_rules=[
                BlockSpec(block_name="crossover_entry", params={"fast_period": 20, "slow_period": 50}),
            ],
            exit_rules=[
                BlockSpec(block_name="fixed_tpsl", params={"tp_atr_mult": 3.0, "sl_atr_mult": 1.5}),
                BlockSpec(block_name="trailing_stop", params={"trail_atr_mult": 2.5}),
            ],
            filters=[
                BlockSpec(block_name="trend_filter", params={"period": 200}),
                BlockSpec(block_name="trading_session"),
            ],
            money_management=BlockSpec(block_name="fixed_risk", params={"risk_pct": 1.0}),
        )

        validator = SpecValidator(full_registry)
        result = validator.validate(spec)
        assert result.is_valid, f"Errors: {result.errors}"

    def test_mean_reversion_spec(self, full_registry: BlockRegistry) -> None:
        spec = StrategySpec(
            name="rsi_mean_reversion",
            description="Mean reversion strategy using RSI threshold crosses with Bollinger Band confirmation.",
            objective={
                "style": "mean_reversion",
                "timeframe": "4h",
                "instruments": ["GBPUSD"],
            },
            indicators=[
                BlockSpec(block_name="rsi", params={"period": 14}),
                BlockSpec(block_name="bollinger_bands", params={"period": 20, "num_std": 2.0}),
            ],
            entry_rules=[
                BlockSpec(
                    block_name="threshold_cross_entry",
                    params={"mode": "mean_reversion", "rsi_period": 14},
                ),
            ],
            exit_rules=[
                BlockSpec(
                    block_name="fixed_tpsl",
                    params={"tp_atr_mult": 2.0, "sl_atr_mult": 1.5},
                ),
                BlockSpec(block_name="time_based_exit", params={"max_bars": 30}),
            ],
            filters=[
                BlockSpec(block_name="spread_filter"),
            ],
            money_management=BlockSpec(
                block_name="atr_based_sizing",
                params={"risk_pct": 1.5},
            ),
        )

        validator = SpecValidator(full_registry)
        result = validator.validate(spec)
        assert result.is_valid, f"Errors: {result.errors}"

    def test_breakout_spec(self, full_registry: BlockRegistry) -> None:
        spec = StrategySpec(
            name="volatility_breakout",
            description="Breakout strategy using price action breakout detection with ADX trend filter.",
            objective={
                "style": "breakout",
                "timeframe": "1h",
            },
            indicators=[
                BlockSpec(block_name="adx", params={"period": 14}),
                BlockSpec(block_name="atr", params={"period": 14}),
            ],
            price_action=[
                BlockSpec(block_name="breakout", params={"lookback": 20, "volume_multiplier": 1.5}),
            ],
            entry_rules=[
                BlockSpec(block_name="confluence_entry"),
            ],
            exit_rules=[
                BlockSpec(block_name="trailing_stop", params={"trail_atr_mult": 3.0}),
                BlockSpec(block_name="breakeven_stop", params={"activation_atr_mult": 2.0}),
            ],
            filters=[
                BlockSpec(block_name="trend_filter", params={"period": 100}),
                BlockSpec(block_name="max_drawdown_filter", params={"max_drawdown_pct": 15.0}),
            ],
            money_management=BlockSpec(
                block_name="volatility_targeting",
                params={"target_vol": 0.15},
            ),
        )

        validator = SpecValidator(full_registry)
        result = validator.validate(spec)
        assert result.is_valid, f"Errors: {result.errors}"


class TestPromptGeneration:
    def test_full_prompt_generation(self, full_registry: BlockRegistry) -> None:
        prompt = build_system_prompt(registry=full_registry)
        assert len(prompt) > 2000
        assert "INDICATOR" in prompt
        assert "ENTRY RULE" in prompt
        assert "EXIT RULE" in prompt
        assert "MONEY MANAGEMENT" in prompt
        assert "FILTER" in prompt
        assert "ema" in prompt
        assert "crossover_entry" in prompt

    def test_user_message_for_trend_strategy(self) -> None:
        msg = build_user_message(
            idea="I want a trend following strategy that uses EMA crossovers to enter, "
                 "with a trailing stop and ATR-based position sizing",
            timeframe="1h",
            instruments=["EURUSD", "GBPUSD"],
            style="trend_following",
        )
        assert "EMA" in msg
        assert "1h" in msg
        assert "EURUSD" in msg
