"""
Integration test for the full robustness suite on a
compiled and backtested strategy.
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
from forgequant.core.robustness.suite import RobustnessSuite, SuiteConfig


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


def _make_trending_data(n: int = 1000) -> pd.DataFrame:
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    close = 100.0 + np.cumsum(np.random.normal(0.05, 0.3, n))
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


class TestFullRobustnessFlow:
    """End-to-end: spec → compile → backtest → robustness suite."""

    def test_complete_pipeline(self, full_registry: BlockRegistry) -> None:
        spec = StrategySpec(
            name="robustness_e2e_test",
            description="A trend following strategy for end to end robustness suite integration testing.",
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
            money_management=BlockSpec(block_name="fixed_risk", params={"risk_pct": 1.0, "atr_period": 14}),
        )

        validator = SpecValidator(full_registry)
        val = validator.validate(spec)
        assert val.is_valid

        data = _make_trending_data(1000)
        compiler = StrategyCompiler(registry=full_registry, validate=False)
        compiled = compiler.compile(spec, data, val.validated_params)

        bt_result = Backtester(BacktestConfig(initial_equity=100_000)).run(compiled)
        assert bt_result.equity_curve is not None
        assert bt_result.returns_series is not None

        suite_config = SuiteConfig(
            wf_n_folds=4,
            wf_min_consistency=0.1,
            wf_min_oos_sharpe=-10.0,
            mc_n_simulations=200,
            mc_p_value_threshold=0.5,
            mc_random_seed=42,
            cpcv_n_groups=5,
            cpcv_n_test_groups=1,
            cpcv_max_pbo=0.9,
            stab_min_r_squared=0.1,
            stab_min_tail_ratio=0.1,
            stab_min_recovery_factor=0.01,
            stab_min_regime_consistency=0.1,
        )
        suite = RobustnessSuite(suite_config)
        verdict = suite.evaluate(bt_result)

        assert verdict.strategy_name == "robustness_e2e_test"
        assert verdict.gates_total == 4
        assert verdict.walk_forward is not None
        assert verdict.monte_carlo is not None
        assert verdict.cpcv is not None
        assert verdict.stability is not None

        assert verdict.walk_forward.n_folds > 0
        assert verdict.monte_carlo.n_simulations == 200
        assert verdict.cpcv.n_combinations_evaluated > 0
        assert verdict.stability.r_squared >= 0

        summary = verdict.summary()
        assert "strategy_name" in summary
        assert "walk_forward" in summary
        assert "monte_carlo" in summary
        assert "cpcv" in summary
        assert "stability" in summary
