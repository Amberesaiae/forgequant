"""Tests for forgequant.core.types."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.core.types import (
    BlockCategory,
    MovingAverageType,
    TimeFrame,
    TradeDirection,
    validate_ohlcv,
)


class TestBlockCategory:
    def test_all_values(self) -> None:
        assert BlockCategory.INDICATOR.value == "indicator"
        assert BlockCategory.PRICE_ACTION.value == "price_action"
        assert BlockCategory.ENTRY_RULE.value == "entry_rule"
        assert BlockCategory.EXIT_RULE.value == "exit_rule"
        assert BlockCategory.MONEY_MANAGEMENT.value == "money_management"
        assert BlockCategory.FILTER.value == "filter"

    def test_count(self) -> None:
        assert len(BlockCategory) == 6

    def test_str(self) -> None:
        assert str(BlockCategory.INDICATOR) == "indicator"


class TestTimeFrame:
    def test_all_values(self) -> None:
        expected = {"1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M"}
        actual = {tf.value for tf in TimeFrame}
        assert actual == expected

    def test_str(self) -> None:
        assert str(TimeFrame.H1) == "1h"


class TestTradeDirection:
    def test_values(self) -> None:
        assert TradeDirection.LONG.value == "long"
        assert TradeDirection.SHORT.value == "short"
        assert TradeDirection.BOTH.value == "both"


class TestMovingAverageType:
    def test_values(self) -> None:
        assert MovingAverageType.SMA.value == "sma"
        assert MovingAverageType.EMA.value == "ema"


class TestValidateOhlcv:
    """Tests for the validate_ohlcv function."""

    def _make_valid_df(self, n: int = 50) -> pd.DataFrame:
        """Create a minimal valid OHLCV DataFrame."""
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        np.random.seed(0)
        close = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
        return pd.DataFrame(
            {
                "open": close + np.random.randn(n) * 0.1,
                "high": close + abs(np.random.randn(n) * 0.5),
                "low": close - abs(np.random.randn(n) * 0.5),
                "close": close,
                "volume": np.random.randint(100, 1000, n).astype(float),
            },
            index=dates,
        )

    def test_valid_df_passes(self) -> None:
        df = self._make_valid_df()
        validate_ohlcv(df)  # Should not raise

    def test_empty_df_raises(self) -> None:
        df = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df.index = pd.DatetimeIndex([])
        with pytest.raises(ValueError, match="empty"):
            validate_ohlcv(df)

    def test_missing_column_raises(self) -> None:
        df = self._make_valid_df()
        df = df.drop(columns=["close"])
        with pytest.raises(ValueError, match="Missing required OHLCV columns"):
            validate_ohlcv(df)

    def test_wrong_index_type_raises(self) -> None:
        df = self._make_valid_df()
        df.index = range(len(df))
        with pytest.raises(ValueError, match="DatetimeIndex"):
            validate_ohlcv(df)

    def test_all_nan_column_raises(self) -> None:
        df = self._make_valid_df()
        df["close"] = np.nan
        with pytest.raises(ValueError, match="entirely NaN"):
            validate_ohlcv(df)

    def test_uppercase_columns_normalized(self) -> None:
        """Columns should be lowered automatically."""
        df = self._make_valid_df()
        df.columns = df.columns.str.upper()
        validate_ohlcv(df)  # Should not raise

    def test_block_name_in_error(self) -> None:
        df = pd.DataFrame()
        with pytest.raises(ValueError, match="my_block"):
            validate_ohlcv(df, block_name="my_block")
