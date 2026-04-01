"""Tests for the Crossover entry rule block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.entry_rules.crossover import CrossoverEntry
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError, BlockValidationError
from forgequant.core.types import BlockCategory


@pytest.fixture
def crossover() -> CrossoverEntry:
    return CrossoverEntry()


class TestCrossoverMetadata:
    def test_name(self, crossover: CrossoverEntry) -> None:
        assert crossover.metadata.name == "crossover_entry"

    def test_category(self, crossover: CrossoverEntry) -> None:
        assert crossover.metadata.category == BlockCategory.ENTRY_RULE

    def test_defaults(self, crossover: CrossoverEntry) -> None:
        d = crossover.metadata.get_defaults()
        assert d["fast_period"] == 10
        assert d["slow_period"] == 20
        assert d["ma_type"] == "ema"

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "crossover_entry" in registry


class TestCrossoverCompute:
    def test_output_columns(
        self, crossover: CrossoverEntry, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = crossover.execute(sample_ohlcv)
        expected = {
            "crossover_fast_ma", "crossover_slow_ma",
            "crossover_long_entry", "crossover_short_entry",
        }
        assert expected == set(result.columns)
        assert len(result) == len(sample_ohlcv)

    def test_signal_is_state_change_only(
        self, crossover: CrossoverEntry, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = crossover.execute(sample_ohlcv)
        long_signals = result["crossover_long_entry"].sum()
        short_signals = result["crossover_short_entry"].sum()
        total_signals = long_signals + short_signals
        total_bars = len(sample_ohlcv)
        assert total_signals < total_bars

    def test_long_and_short_mutually_exclusive(
        self, crossover: CrossoverEntry, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = crossover.execute(sample_ohlcv)
        both = result["crossover_long_entry"] & result["crossover_short_entry"]
        assert both.sum() == 0

    def test_uptrend_produces_long_signal(
        self, crossover: CrossoverEntry
    ) -> None:
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.full(n, 100.0)
        close[50:] = 120.0
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": np.ones(n) * 1000.0,
            },
            index=dates,
        )
        result = crossover.execute(df, {"fast_period": 5, "slow_period": 20, "ma_type": "ema"})
        long_after_jump = result["crossover_long_entry"].iloc[50:].sum()
        assert long_after_jump >= 1

    def test_downtrend_produces_short_signal(
        self, crossover: CrossoverEntry
    ) -> None:
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        # Start with an uptrend so fast > slow, then crash
        close = np.concatenate([
            np.linspace(100, 150, 100),  # Uptrend: fast EMA > slow EMA
            np.full(100, 80.0),           # Crash: fast crosses below slow
        ])
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": np.ones(n) * 1000.0,
            },
            index=dates,
        )
        result = crossover.execute(df, {"fast_period": 5, "slow_period": 20, "ma_type": "ema"})
        short_after_drop = result["crossover_short_entry"].iloc[100:].sum()
        assert short_after_drop >= 1

    def test_sma_mode(
        self, crossover: CrossoverEntry, sample_ohlcv: pd.DataFrame
    ) -> None:
        result_ema = crossover.execute(sample_ohlcv, {"ma_type": "ema"})
        result_sma = crossover.execute(sample_ohlcv, {"ma_type": "sma"})
        assert not result_ema["crossover_fast_ma"].equals(result_sma["crossover_fast_ma"])

    def test_fast_ge_slow_raises(
        self, crossover: CrossoverEntry, sample_ohlcv: pd.DataFrame
    ) -> None:
        with pytest.raises(BlockComputeError, match="must be less than"):
            crossover.execute(sample_ohlcv, {"fast_period": 20, "slow_period": 10})

    def test_fast_eq_slow_raises(
        self, crossover: CrossoverEntry, sample_ohlcv: pd.DataFrame
    ) -> None:
        with pytest.raises(BlockComputeError, match="must be less than"):
            crossover.execute(sample_ohlcv, {"fast_period": 20, "slow_period": 20})

    def test_insufficient_data_raises(self, crossover: CrossoverEntry) -> None:
        dates = pd.date_range("2024-01-01", periods=15, freq="h")
        df = pd.DataFrame(
            {
                "open": [100.0] * 15,
                "high": [101.0] * 15,
                "low": [99.0] * 15,
                "close": [100.0] * 15,
                "volume": [1000.0] * 15,
            },
            index=dates,
        )
        with pytest.raises(BlockComputeError, match="Insufficient data"):
            crossover.execute(df, {"fast_period": 10, "slow_period": 20})
