"""Tests for the backtesting engine."""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

from forgequant.ai_forge.schemas import BlockSpec, StrategySpec
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


@pytest.fixture
def compiler(full_registry: BlockRegistry) -> StrategyCompiler:
    return StrategyCompiler(registry=full_registry, validate=True)


def _make_trending_data(n: int = 500, trend: str = "up") -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="h")

    if trend == "up":
        close = 100.0 + np.cumsum(np.random.normal(0.05, 0.3, n))
    elif trend == "down":
        close = 200.0 + np.cumsum(np.random.normal(-0.05, 0.3, n))
    else:
        close = 100.0 + np.cumsum(np.random.normal(0.0, 0.5, n))

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


class TestBacktestConfig:
    def test_defaults(self) -> None:
        cfg = BacktestConfig()
        assert cfg.initial_equity == 100_000.0
        assert cfg.commission_per_unit == 0.0
        assert cfg.slippage_pct == 0.0


class TestBacktester:
    def test_basic_backtest(
        self, compiler: StrategyCompiler
    ) -> None:
        spec = StrategySpec(
            name="basic_backtest_test",
            description="A simple EMA crossover strategy for backtesting engine integration testing.",
            objective={"style": "trend_following", "timeframe": "1h"},
            indicators=[
                BlockSpec(block_name="ema", params={"period": 20}),
                BlockSpec(block_name="atr", params={"period": 14}),
            ],
            entry_rules=[
                BlockSpec(block_name="crossover_entry", params={"fast_period": 10, "slow_period": 20}),
            ],
            exit_rules=[
                BlockSpec(block_name="fixed_tpsl", params={"atr_period": 14, "tp_atr_mult": 3.0, "sl_atr_mult": 1.5}),
            ],
            money_management=BlockSpec(block_name="fixed_risk", params={"risk_pct": 1.0}),
        )

        data = _make_trending_data(500, "up")
        compiled = compiler.compile(spec, data)

        backtester = Backtester(BacktestConfig(initial_equity=100000))
        result = backtester.run(compiled)

        assert result.strategy_name == "basic_backtest_test"
        assert result.initial_equity == 100000
        assert len(result.equity_curve) == 500
        assert result.metrics is not None
        assert "total_return_pct" in result.metrics
        assert "sharpe_ratio" in result.metrics
        assert "max_drawdown_pct" in result.metrics
        assert "total_trades" in result.metrics

    def test_produces_trades(
        self, compiler: StrategyCompiler
    ) -> None:
        spec = StrategySpec(
            name="trade_generator_test",
            description="Strategy designed to generate trades for testing trade recording behavior.",
            objective={"style": "trend_following", "timeframe": "1h"},
            indicators=[BlockSpec(block_name="ema", params={"period": 10})],
            entry_rules=[
                BlockSpec(block_name="crossover_entry", params={"fast_period": 5, "slow_period": 10}),
            ],
            exit_rules=[
                BlockSpec(block_name="fixed_tpsl", params={"atr_period": 5, "tp_atr_mult": 2.0, "sl_atr_mult": 1.0}),
            ],
            money_management=BlockSpec(block_name="fixed_risk", params={"risk_pct": 1.0, "atr_period": 5}),
        )

        data = _make_trending_data(500, "up")
        compiled = compiler.compile(spec, data)

        backtester = Backtester(BacktestConfig(initial_equity=100000))
        result = backtester.run(compiled)

        assert result.n_trades > 0
        for trade in result.trades:
            assert trade.direction in ("long", "short")
            assert trade.entry_price > 0
            assert trade.exit_price > 0
            assert trade.bars_held >= 0
            assert trade.exit_reason in ("tp", "sl", "signal", "end")

    def test_equity_curve_starts_at_initial(
        self, compiler: StrategyCompiler
    ) -> None:
        spec = StrategySpec(
            name="equity_start_test",
            description="Testing that equity curve starts at initial equity for verification.",
            objective={"style": "trend_following", "timeframe": "1h"},
            indicators=[BlockSpec(block_name="ema", params={"period": 50})],
            entry_rules=[BlockSpec(block_name="crossover_entry", params={"fast_period": 20, "slow_period": 50})],
            exit_rules=[BlockSpec(block_name="fixed_tpsl")],
            money_management=BlockSpec(block_name="fixed_risk"),
        )

        data = _make_trending_data(300)
        compiled = compiler.compile(spec, data)

        cfg = BacktestConfig(initial_equity=50000)
        result = Backtester(cfg).run(compiled)

        assert result.equity_curve.iloc[0] == 50000

    def test_with_commission(
        self, compiler: StrategyCompiler
    ) -> None:
        spec = StrategySpec(
            name="commission_test",
            description="Testing the impact of commissions on final equity in backtesting results.",
            objective={"style": "trend_following", "timeframe": "1h"},
            indicators=[BlockSpec(block_name="ema", params={"period": 10})],
            entry_rules=[
                BlockSpec(block_name="crossover_entry", params={"fast_period": 5, "slow_period": 10}),
            ],
            exit_rules=[
                BlockSpec(block_name="fixed_tpsl", params={"atr_period": 5, "tp_atr_mult": 2.0, "sl_atr_mult": 1.0}),
            ],
            money_management=BlockSpec(block_name="fixed_risk", params={"risk_pct": 1.0, "atr_period": 5}),
        )

        data = _make_trending_data(500, "up")
        compiled = compiler.compile(spec, data)

        result_no_comm = Backtester(BacktestConfig(commission_per_unit=0.0)).run(compiled)
        result_with_comm = Backtester(BacktestConfig(commission_per_unit=1.0)).run(compiled)

        if result_no_comm.n_trades > 0:
            assert result_with_comm.final_equity <= result_no_comm.final_equity

    def test_metrics_populated(
        self, compiler: StrategyCompiler
    ) -> None:
        spec = StrategySpec(
            name="metrics_test",
            description="Verifying all performance metrics are computed and populated after backtesting.",
            objective={"style": "trend_following", "timeframe": "1h"},
            indicators=[BlockSpec(block_name="ema", params={"period": 10})],
            entry_rules=[BlockSpec(block_name="crossover_entry", params={"fast_period": 5, "slow_period": 10})],
            exit_rules=[BlockSpec(block_name="fixed_tpsl", params={"atr_period": 5})],
            money_management=BlockSpec(block_name="fixed_risk", params={"atr_period": 5}),
        )

        data = _make_trending_data(500)
        compiled = compiler.compile(spec, data)
        result = Backtester().run(compiled)

        required_metrics = [
            "total_return_pct", "annualized_return_pct",
            "max_drawdown_pct", "sharpe_ratio", "sortino_ratio",
            "calmar_ratio", "total_trades", "win_rate",
            "profit_factor", "expectancy_dollar",
            "avg_bars_held", "long_trades", "short_trades",
        ]
        for metric in required_metrics:
            assert metric in result.metrics, f"Missing metric: {metric}"

    def test_result_summary(
        self, compiler: StrategyCompiler
    ) -> None:
        spec = StrategySpec(
            name="summary_test",
            description="Testing the summary output format of backtest results for completeness.",
            objective={"style": "trend_following", "timeframe": "1h"},
            indicators=[BlockSpec(block_name="ema", params={"period": 20})],
            entry_rules=[BlockSpec(block_name="crossover_entry", params={"fast_period": 10, "slow_period": 20})],
            exit_rules=[BlockSpec(block_name="fixed_tpsl")],
            money_management=BlockSpec(block_name="fixed_risk"),
        )

        data = _make_trending_data(300)
        compiled = compiler.compile(spec, data)
        result = Backtester().run(compiled)

        summary = result.summary()
        assert "strategy_name" in summary
        assert "total_return_pct" in summary
        assert "n_trades" in summary
