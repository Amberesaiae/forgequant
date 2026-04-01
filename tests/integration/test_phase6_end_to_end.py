"""
End-to-end integration test for the complete Phase 6 pipeline:
StrategySpec → Compiler → Backtester → BacktestResult with metrics.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from forgequant.ai_forge.schemas import BlockSpec, StrategySpec
from forgequant.ai_forge.validator import SpecValidator
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.compiler.compiler import StrategyCompiler
from forgequant.core.engine.backtester import Backtester, BacktestConfig


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


def _make_data(n: int = 500) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    close = 100.0 + np.cumsum(np.random.normal(0.02, 0.5, n))
    close = np.maximum(close, 50.0)
    spread = np.random.uniform(0.1, 0.5, n)

    return pd.DataFrame(
        {
            "open": close + np.random.randn(n) * 0.1,
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": np.random.randint(100, 10000, n).astype(float),
        },
        index=dates,
    )


class TestEMACrossoverEndToEnd:
    """Full pipeline test for a classic EMA crossover strategy."""

    def test_full_pipeline(self, full_registry: BlockRegistry) -> None:
        spec = StrategySpec(
            name="ema_crossover_e2e",
            description="End to end EMA crossover trend following strategy for integration testing.",
            objective={
                "style": "trend_following",
                "timeframe": "1h",
                "instruments": ["EURUSD"],
            },
            indicators=[
                BlockSpec(block_name="ema", params={"period": 20}),
                BlockSpec(block_name="atr", params={"period": 14}),
            ],
            entry_rules=[
                BlockSpec(
                    block_name="crossover_entry",
                    params={"fast_period": 10, "slow_period": 20},
                ),
            ],
            exit_rules=[
                BlockSpec(
                    block_name="fixed_tpsl",
                    params={"tp_atr_mult": 3.0, "sl_atr_mult": 1.5, "atr_period": 14},
                ),
                BlockSpec(
                    block_name="trailing_stop",
                    params={"atr_period": 14, "trail_atr_mult": 2.5},
                ),
            ],
            filters=[
                BlockSpec(block_name="trend_filter", params={"period": 50}),
            ],
            money_management=BlockSpec(
                block_name="fixed_risk",
                params={"risk_pct": 1.0, "sl_atr_mult": 1.5, "atr_period": 14},
            ),
        )

        validator = SpecValidator(full_registry)
        val_result = validator.validate(spec)
        assert val_result.is_valid, f"Validation errors: {val_result.errors}"

        data = _make_data(500)
        compiler = StrategyCompiler(registry=full_registry, validate=False)
        compiled = compiler.compile(spec, data, val_result.validated_params)

        assert compiled.entry_long is not None
        assert compiled.entry_short is not None
        assert compiled.stop_loss_long is not None
        assert compiled.take_profit_long is not None
        assert compiled.allow_long is not None
        assert compiled.position_size_long is not None

        config = BacktestConfig(initial_equity=100_000)
        backtester = Backtester(config)
        result = backtester.run(compiled)

        assert result.strategy_name == "ema_crossover_e2e"
        assert result.initial_equity == 100_000
        assert len(result.equity_curve) == 500
        assert result.drawdown_series is not None
        assert result.returns_series is not None

        assert "total_return_pct" in result.metrics
        assert "sharpe_ratio" in result.metrics
        assert "max_drawdown_pct" in result.metrics
        assert "total_trades" in result.metrics
        assert "win_rate" in result.metrics
        assert "profit_factor" in result.metrics

        summary = result.summary()
        assert isinstance(summary, dict)
        assert "strategy_name" in summary


class TestMeanReversionEndToEnd:
    """Full pipeline test for a mean-reversion strategy."""

    def test_full_pipeline(self, full_registry: BlockRegistry) -> None:
        spec = StrategySpec(
            name="rsi_mean_reversion_e2e",
            description="End to end RSI mean reversion strategy for integration testing verification.",
            objective={
                "style": "mean_reversion",
                "timeframe": "1h",
            },
            indicators=[
                BlockSpec(block_name="rsi", params={"period": 14}),
                BlockSpec(block_name="bollinger_bands", params={"period": 20}),
            ],
            entry_rules=[
                BlockSpec(
                    block_name="threshold_cross_entry",
                    params={"mode": "mean_reversion"},
                ),
            ],
            exit_rules=[
                BlockSpec(
                    block_name="fixed_tpsl",
                    params={"tp_atr_mult": 2.0, "sl_atr_mult": 1.0},
                ),
                BlockSpec(
                    block_name="time_based_exit",
                    params={"max_bars": 30},
                ),
            ],
            money_management=BlockSpec(
                block_name="atr_based_sizing",
                params={"risk_pct": 1.0},
            ),
        )

        validator = SpecValidator(full_registry)
        val_result = validator.validate(spec)
        assert val_result.is_valid, f"Errors: {val_result.errors}"

        data = _make_data(500)
        compiler = StrategyCompiler(registry=full_registry, validate=False)
        compiled = compiler.compile(spec, data, val_result.validated_params)

        result = Backtester(BacktestConfig(initial_equity=100_000)).run(compiled)

        assert result.n_trades >= 0
        assert "total_return_pct" in result.metrics
        assert result.equity_curve.iloc[0] == 100_000


class TestBreakoutEndToEnd:
    """Full pipeline test for a breakout strategy with price action."""

    def test_full_pipeline(self, full_registry: BlockRegistry) -> None:
        spec = StrategySpec(
            name="breakout_e2e",
            description="End to end breakout strategy using price action and confluence entry for testing.",
            objective={"style": "breakout", "timeframe": "1h"},
            indicators=[
                BlockSpec(block_name="adx", params={"period": 14}),
                BlockSpec(block_name="atr", params={"period": 14}),
            ],
            price_action=[
                BlockSpec(block_name="breakout", params={"lookback": 20}),
            ],
            entry_rules=[
                BlockSpec(block_name="confluence_entry", params={"trend_period": 50}),
            ],
            exit_rules=[
                BlockSpec(block_name="trailing_stop", params={"trail_atr_mult": 3.0}),
                BlockSpec(block_name="breakeven_stop", params={"activation_atr_mult": 2.0}),
            ],
            filters=[
                BlockSpec(block_name="max_drawdown_filter", params={"max_drawdown_pct": 15.0}),
            ],
            money_management=BlockSpec(
                block_name="volatility_targeting",
                params={"target_vol": 0.15},
            ),
        )

        validator = SpecValidator(full_registry)
        val_result = validator.validate(spec)
        assert val_result.is_valid, f"Errors: {val_result.errors}"

        data = _make_data(500)
        compiler = StrategyCompiler(registry=full_registry, validate=False)
        compiled = compiler.compile(spec, data, val_result.validated_params)

        result = Backtester().run(compiled)

        assert "total_return_pct" in result.metrics
        assert "max_drawdown_pct" in result.metrics
        assert "breakout" in compiled.block_outputs
