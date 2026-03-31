"""
Average True Range (ATR) Indicator Block.

ATR measures market volatility by decomposing the entire range of
a price bar for a given period. It is the average of true ranges.

True Range = max of:
    1. Current High - Current Low
    2. abs(Current High - Previous Close)
    3. abs(Current Low - Previous Close)

ATR = Rolling mean of True Range over 'period' bars.

Common uses:
    - Stop loss placement (e.g., 2x ATR from entry)
    - Position sizing (risk a fixed dollar amount / ATR)
    - Volatility filtering (only trade when ATR is above threshold)

Default Parameters:
    period: 14
"""

from typing import Any, Dict

import pandas as pd

from strategies_library.base import BaseBlock, BlockMetadata
from strategies_library.registry import BlockRegistry


@BlockRegistry.register
class ATR(BaseBlock):
    """Average True Range volatility indicator."""

    metadata = BlockMetadata(
        name="ATR",
        category="indicator",
        description="Average True Range — measures market volatility",
        complexity=2,
        typical_use=["volatility", "stop_loss", "position_sizing", "trailing_stop"],
        required_columns=["high", "low", "close"],
        version="1.0.0",
        tags=["volatility", "range", "risk"],
    )

    def compute(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any] | None = None,
    ) -> pd.Series:
        """Compute ATR.

        Args:
            data: OHLCV DataFrame with 'high', 'low', 'close' columns.
            params: Optional dict with:
                - period (int): ATR lookback period. Default 14.

        Returns:
            pd.Series with ATR values. First 'period' values will be NaN.
        """
        params = params or {}
        period = int(params.get("period", 14))

        high = data["high"]
        low = data["low"]
        close_prev = data["close"].shift(1)

        # Three components of True Range
        high_low = high - low
        high_close_prev = (high - close_prev).abs()
        low_close_prev = (low - close_prev).abs()

        # True Range is the max of all three
        true_range = pd.concat(
            [high_low, high_close_prev, low_close_prev],
            axis=1,
        ).max(axis=1)

        # ATR is the rolling mean of True Range
        atr = true_range.rolling(window=period, min_periods=period).mean()

        return atr

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate ATR parameters.

        Rules:
            - period must be between 2 and 100.
        """
        period = params.get("period", 14)
        if not isinstance(period, (int, float)):
            return False
        if int(period) < 2 or int(period) > 100:
            return False
        return True
