"""Tests for CompiledStrategy container."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.ai_forge.schemas import BlockSpec, StrategySpec
from forgequant.core.compiler.compiled_strategy import BlockOutput, CompiledStrategy


def _make_spec() -> StrategySpec:
    return StrategySpec(
        name="test_compiled",
        description="A test strategy for CompiledStrategy container testing with detail.",
        objective={"style": "trend_following", "timeframe": "1h"},
        indicators=[BlockSpec(block_name="ema")],
        entry_rules=[BlockSpec(block_name="crossover_entry")],
        exit_rules=[BlockSpec(block_name="fixed_tpsl")],
        money_management=BlockSpec(block_name="fixed_risk"),
    )


def _make_ohlcv(n: int = 100) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    close = np.linspace(100, 110, n)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": np.ones(n) * 1000,
        },
        index=dates,
    )


class TestBlockOutput:
    def test_creation(self) -> None:
        df = pd.DataFrame({"ema_20": [1.0, 2.0, 3.0]})
        bo = BlockOutput(
            block_name="ema",
            category="indicator",
            params={"period": 20},
            result=df,
        )
        assert bo.block_name == "ema"
        assert bo.category == "indicator"
        assert len(bo.result) == 3


class TestCompiledStrategy:
    def test_basic_properties(self) -> None:
        spec = _make_spec()
        ohlcv = _make_ohlcv(50)
        compiled = CompiledStrategy(spec=spec, ohlcv=ohlcv)

        assert compiled.n_bars == 50
        assert len(compiled.close) == 50
        assert len(compiled.index) == 50

    def test_filtered_entry_no_filter(self) -> None:
        spec = _make_spec()
        ohlcv = _make_ohlcv(10)
        compiled = CompiledStrategy(spec=spec, ohlcv=ohlcv)
        compiled.entry_long = pd.Series([True] * 5 + [False] * 5, index=ohlcv.index)

        filtered = compiled.filtered_entry_long()
        assert filtered.sum() == 5

    def test_filtered_entry_with_filter(self) -> None:
        spec = _make_spec()
        ohlcv = _make_ohlcv(10)
        compiled = CompiledStrategy(spec=spec, ohlcv=ohlcv)
        compiled.entry_long = pd.Series([True] * 10, index=ohlcv.index)
        compiled.allow_long = pd.Series(
            [True] * 5 + [False] * 5, index=ohlcv.index
        )

        filtered = compiled.filtered_entry_long()
        assert filtered.sum() == 5

    def test_filtered_entry_none(self) -> None:
        spec = _make_spec()
        ohlcv = _make_ohlcv(10)
        compiled = CompiledStrategy(spec=spec, ohlcv=ohlcv)

        filtered = compiled.filtered_entry_long()
        assert filtered.sum() == 0

    def test_get_block_output(self) -> None:
        spec = _make_spec()
        ohlcv = _make_ohlcv(10)
        compiled = CompiledStrategy(spec=spec, ohlcv=ohlcv)

        bo = BlockOutput("ema", "indicator", {}, pd.DataFrame({"x": [1]}))
        compiled.block_outputs["ema"] = bo

        assert compiled.get_block_output("ema") is bo
        assert compiled.get_block_output("nonexistent") is None

    def test_summary(self) -> None:
        spec = _make_spec()
        ohlcv = _make_ohlcv(20)
        compiled = CompiledStrategy(spec=spec, ohlcv=ohlcv)
        compiled.entry_long = pd.Series([True, False] * 10, index=ohlcv.index)
        compiled.entry_short = pd.Series(False, index=ohlcv.index)
        compiled.exit_long = pd.Series(False, index=ohlcv.index)
        compiled.exit_short = pd.Series(False, index=ohlcv.index)

        s = compiled.summary()
        assert s["strategy_name"] == "test_compiled"
        assert s["n_bars"] == 20
        assert s["raw_long_entries"] == 10
