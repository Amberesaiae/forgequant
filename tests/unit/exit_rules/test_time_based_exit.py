"""Tests for the Time-Based Exit rule block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.exit_rules.time_based_exit import TimeBasedExit
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory


@pytest.fixture
def time_exit() -> TimeBasedExit:
    return TimeBasedExit()


class TestTimeBasedExitMetadata:
    def test_name(self, time_exit: TimeBasedExit) -> None:
        assert time_exit.metadata.name == "time_based_exit"

    def test_category(self, time_exit: TimeBasedExit) -> None:
        assert time_exit.metadata.category == BlockCategory.EXIT_RULE

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "time_based_exit" in registry


class TestTimeBasedExitCompute:
    def test_output_columns(
        self, time_exit: TimeBasedExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = time_exit.execute(sample_ohlcv)
        expected = {
            "time_bar_index", "time_max_bars_exit",
            "time_avoid_day", "time_near_session_close",
        }
        assert expected == set(result.columns)

    def test_bar_index_wraps(
        self, time_exit: TimeBasedExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = time_exit.execute(sample_ohlcv, {"max_bars": 50})
        bar_idx = result["time_bar_index"]
        assert bar_idx.min() == 0
        assert bar_idx.max() == 49

    def test_max_bars_exit_frequency(
        self, time_exit: TimeBasedExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        max_bars = 10
        result = time_exit.execute(sample_ohlcv, {"max_bars": max_bars})
        exits = result["time_max_bars_exit"]
        expected_count = len(sample_ohlcv) // max_bars
        assert exits.sum() == expected_count

    def test_avoid_friday(
        self, time_exit: TimeBasedExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = time_exit.execute(sample_ohlcv, {"avoid_days": "Friday"})
        avoid = result["time_avoid_day"]
        friday_mask = sample_ohlcv.index.dayofweek == 4
        if friday_mask.any():
            assert avoid[friday_mask].all()
            assert not avoid[~friday_mask].any()

    def test_avoid_multiple_days(
        self, time_exit: TimeBasedExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = time_exit.execute(sample_ohlcv, {"avoid_days": "Saturday,Sunday"})
        avoid = result["time_avoid_day"]
        weekend_mask = sample_ohlcv.index.dayofweek.isin([5, 6])
        if weekend_mask.any():
            assert avoid[weekend_mask].all()

    def test_empty_avoid_days(
        self, time_exit: TimeBasedExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = time_exit.execute(sample_ohlcv, {"avoid_days": ""})
        assert not result["time_avoid_day"].any()

    def test_session_close_warning(self, time_exit: TimeBasedExit) -> None:
        dates = pd.date_range("2024-01-01", periods=72, freq="h")
        n = len(dates)
        df = pd.DataFrame(
            {
                "open": [100.0] * n, "high": [101.0] * n,
                "low": [99.0] * n, "close": [100.0] * n,
                "volume": [1000.0] * n,
            },
            index=dates,
        )
        result = time_exit.execute(df, {"close_warning_bars": 3})
        near_close = result["time_near_session_close"]
        assert near_close.sum() > 0

    def test_session_close_disabled(
        self, time_exit: TimeBasedExit, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = time_exit.execute(sample_ohlcv, {"close_warning_bars": 0})
        assert not result["time_near_session_close"].any()
