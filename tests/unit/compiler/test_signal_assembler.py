"""Tests for the signal assembler."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.ai_forge.schemas import BlockSpec, StrategySpec
from forgequant.core.compiler.compiled_strategy import BlockOutput, CompiledStrategy
from forgequant.core.compiler.signal_assembler import (
    _and_combine,
    _or_combine,
    assemble_signals,
)


def _make_index(n: int = 50) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-01", periods=n, freq="h")


def _make_spec() -> StrategySpec:
    return StrategySpec(
        name="assembler_test",
        description="A test strategy for signal assembler testing with sufficient length.",
        objective={"style": "trend_following", "timeframe": "1h"},
        indicators=[BlockSpec(block_name="ema")],
        entry_rules=[BlockSpec(block_name="crossover_entry")],
        exit_rules=[BlockSpec(block_name="fixed_tpsl")],
        money_management=BlockSpec(block_name="fixed_risk"),
    )


class TestCombineFunctions:
    def test_or_combine_empty(self) -> None:
        idx = _make_index(5)
        result = _or_combine([], idx)
        assert result.sum() == 0

    def test_or_combine_single(self) -> None:
        idx = _make_index(5)
        s = pd.Series([True, False, True, False, True], index=idx)
        result = _or_combine([s], idx)
        assert result.sum() == 3

    def test_or_combine_multiple(self) -> None:
        idx = _make_index(5)
        s1 = pd.Series([True, False, False, False, False], index=idx)
        s2 = pd.Series([False, True, False, False, False], index=idx)
        result = _or_combine([s1, s2], idx)
        assert result.sum() == 2

    def test_and_combine_empty(self) -> None:
        idx = _make_index(5)
        result = _and_combine([], idx)
        assert result.all()

    def test_and_combine_single(self) -> None:
        idx = _make_index(5)
        s = pd.Series([True, True, False, True, True], index=idx)
        result = _and_combine([s], idx)
        assert result.sum() == 4

    def test_and_combine_multiple(self) -> None:
        idx = _make_index(5)
        s1 = pd.Series([True, True, True, False, True], index=idx)
        s2 = pd.Series([True, True, False, True, True], index=idx)
        result = _and_combine([s1, s2], idx)
        assert result.sum() == 3


class TestAssembleSignals:
    def test_assembles_entry_signals(self) -> None:
        spec = _make_spec()
        n = 20
        idx = _make_index(n)
        ohlcv = pd.DataFrame(
            {
                "open": np.ones(n) * 100,
                "high": np.ones(n) * 101,
                "low": np.ones(n) * 99,
                "close": np.ones(n) * 100,
                "volume": np.ones(n) * 1000,
            },
            index=idx,
        )
        compiled = CompiledStrategy(spec=spec, ohlcv=ohlcv)

        entry_df = pd.DataFrame(
            {
                "crossover_long_entry": [False] * 5 + [True] + [False] * 14,
                "crossover_short_entry": [False] * 10 + [True] + [False] * 9,
            },
            index=idx,
        )
        compiled.block_outputs["crossover_entry"] = BlockOutput(
            "crossover_entry", "entry_rule", {}, entry_df
        )

        compiled.block_outputs["fixed_tpsl"] = BlockOutput(
            "fixed_tpsl", "exit_rule", {},
            pd.DataFrame({"tpsl_long_sl": np.ones(n) * 98}, index=idx),
        )
        compiled.block_outputs["fixed_risk"] = BlockOutput(
            "fixed_risk", "money_management", {},
            pd.DataFrame({"fr_position_size": np.ones(n) * 100}, index=idx),
        )

        result = assemble_signals(compiled)

        assert result.entry_long is not None
        assert result.entry_long.sum() == 1
        assert result.entry_short is not None
        assert result.entry_short.sum() == 1

    def test_assembles_filter_masks(self) -> None:
        spec = StrategySpec(
            name="filter_test",
            description="Testing filter assembly with trend filter and session filter blocks.",
            objective={"style": "trend_following", "timeframe": "1h"},
            indicators=[BlockSpec(block_name="ema")],
            entry_rules=[BlockSpec(block_name="crossover_entry")],
            exit_rules=[BlockSpec(block_name="fixed_tpsl")],
            filters=[BlockSpec(block_name="trend_filter")],
            money_management=BlockSpec(block_name="fixed_risk"),
        )
        n = 20
        idx = _make_index(n)
        ohlcv = pd.DataFrame(
            {
                "open": np.ones(n) * 100, "high": np.ones(n) * 101,
                "low": np.ones(n) * 99, "close": np.ones(n) * 100,
                "volume": np.ones(n) * 1000,
            },
            index=idx,
        )
        compiled = CompiledStrategy(spec=spec, ohlcv=ohlcv)

        filter_df = pd.DataFrame(
            {
                "trend_allow_long": [True] * 10 + [False] * 10,
                "trend_allow_short": [False] * 10 + [True] * 10,
            },
            index=idx,
        )
        compiled.block_outputs["trend_filter"] = BlockOutput(
            "trend_filter", "filter", {}, filter_df
        )

        result = assemble_signals(compiled)

        assert result.allow_long is not None
        assert result.allow_long.sum() == 10
        assert result.allow_short is not None
        assert result.allow_short.sum() == 10

    def test_assembles_tpsl(self) -> None:
        spec = _make_spec()
        n = 10
        idx = _make_index(n)
        ohlcv = pd.DataFrame(
            {
                "open": np.ones(n) * 100, "high": np.ones(n) * 101,
                "low": np.ones(n) * 99, "close": np.ones(n) * 100,
                "volume": np.ones(n) * 1000,
            },
            index=idx,
        )
        compiled = CompiledStrategy(spec=spec, ohlcv=ohlcv)

        tpsl_df = pd.DataFrame(
            {
                "tpsl_long_tp": np.ones(n) * 105,
                "tpsl_long_sl": np.ones(n) * 97,
                "tpsl_short_tp": np.ones(n) * 95,
                "tpsl_short_sl": np.ones(n) * 103,
            },
            index=idx,
        )
        compiled.block_outputs["fixed_tpsl"] = BlockOutput(
            "fixed_tpsl", "exit_rule", {}, tpsl_df
        )
        compiled.block_outputs["fixed_risk"] = BlockOutput(
            "fixed_risk", "money_management", {},
            pd.DataFrame({"fr_position_size": np.ones(n)}, index=idx),
        )

        result = assemble_signals(compiled)

        assert result.stop_loss_long is not None
        assert result.take_profit_long is not None
        assert result.stop_loss_long.iloc[0] == 97.0
        assert result.take_profit_long.iloc[0] == 105.0
