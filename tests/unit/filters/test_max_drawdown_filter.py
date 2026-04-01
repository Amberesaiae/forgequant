"""Tests for the Max Drawdown Filter block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.filters.max_drawdown_filter import MaxDrawdownFilter
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory


@pytest.fixture
def dd_filter() -> MaxDrawdownFilter:
    return MaxDrawdownFilter()


class TestMaxDrawdownMetadata:
    def test_name(self, dd_filter: MaxDrawdownFilter) -> None:
        assert dd_filter.metadata.name == "max_drawdown_filter"

    def test_category(self, dd_filter: MaxDrawdownFilter) -> None:
        assert dd_filter.metadata.category == BlockCategory.FILTER

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "max_drawdown_filter" in registry


class TestMaxDrawdownCompute:
    def test_output_columns(
        self, dd_filter: MaxDrawdownFilter, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = dd_filter.execute(sample_ohlcv)
        expected = {
            "dd_cumulative_return", "dd_running_max",
            "dd_drawdown", "dd_drawdown_pct", "dd_allow_trading",
        }
        assert expected == set(result.columns)

    def test_uptrend_always_allowed(self, dd_filter: MaxDrawdownFilter) -> None:
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.linspace(100, 200, n)
        df = pd.DataFrame(
            {
                "open": close, "high": close + 1,
                "low": close - 1, "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = dd_filter.execute(df)
        assert result["dd_allow_trading"].all()

    def test_crash_halts_trading(self, dd_filter: MaxDrawdownFilter) -> None:
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.concatenate([
            np.linspace(100, 120, 50),
            np.linspace(120, 95, 50),
        ])
        df = pd.DataFrame(
            {
                "open": close, "high": close + 1,
                "low": close - 1, "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = dd_filter.execute(df, {"max_drawdown_pct": 15.0, "recovery_pct": 10.0})
        assert not result["dd_allow_trading"].all()

    def test_hysteresis_recovery(self, dd_filter: MaxDrawdownFilter) -> None:
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.concatenate([
            np.linspace(100, 120, 50),
            np.linspace(120, 98, 50),
            np.linspace(98, 110, 50),
            np.linspace(110, 120, 50),
        ])
        df = pd.DataFrame(
            {
                "open": close, "high": close + 0.5,
                "low": close - 0.5, "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = dd_filter.execute(df, {"max_drawdown_pct": 15.0, "recovery_pct": 5.0})
        allow = result["dd_allow_trading"]
        halted_bars = (~allow).sum()
        assert halted_bars > 0
        assert allow.iloc[-1] == True

    def test_drawdown_non_positive(
        self, dd_filter: MaxDrawdownFilter, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = dd_filter.execute(sample_ohlcv)
        assert (result["dd_drawdown"] <= 0 + 1e-10).all()

    def test_drawdown_pct_non_negative(
        self, dd_filter: MaxDrawdownFilter, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = dd_filter.execute(sample_ohlcv)
        assert (result["dd_drawdown_pct"] >= 0 - 1e-10).all()

    def test_recovery_gt_max_dd_raises(
        self, dd_filter: MaxDrawdownFilter, sample_ohlcv: pd.DataFrame
    ) -> None:
        with pytest.raises(BlockComputeError, match="recovery_pct"):
            dd_filter.execute(
                sample_ohlcv, {"max_drawdown_pct": 10.0, "recovery_pct": 15.0}
            )

    def test_insufficient_data_raises(self, dd_filter: MaxDrawdownFilter) -> None:
        dates = pd.date_range("2024-01-01", periods=1, freq="h")
        df = pd.DataFrame(
            {
                "open": [100.0], "high": [101.0],
                "low": [99.0], "close": [100.0],
                "volume": [1000.0],
            },
            index=dates,
        )
        with pytest.raises(BlockComputeError, match="Insufficient data"):
            dd_filter.execute(df)
