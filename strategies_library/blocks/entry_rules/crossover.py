"""
Moving Average Crossover Entry Block.

Generates entry signals when a fast moving average crosses above
(bullish) or below (bearish) a slow moving average.

This is one of the most fundamental and widely used entry signals
in systematic trading. It captures the transition from one trend
regime to another.

The crossover is detected by comparing the current state
(fast > slow) with the previous bar's state. A new True in the
difference indicates a crossover event.

Supports both EMA and SMA calculation.

Default Parameters:
    fast_period: 9
    slow_period: 21
    ma_type: 'ema'
"""

from typing import Any, Dict

import pandas as pd

from strategies_library.base import BaseBlock, BlockMetadata
from strategies_library.registry import BlockRegistry


@BlockRegistry.register
class Crossover(BaseBlock):
    """Moving average crossover entry signal generator."""

    metadata = BlockMetadata(
        name="Crossover",
        category="entry",
        description="Entry signals when fast MA crosses above or below slow MA",
        complexity=2,
        typical_use=["trend_following", "crossover"],
        required_columns=["close"],
        version="1.0.0",
        tags=["crossover", "trend", "entry", "moving_average"],
    )

    def compute(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any] | None = None,
    ) -> Dict[str, pd.Series]:
        """Compute crossover entry signals.

        Args:
            data: OHLCV DataFrame with a 'close' column.
            params: Optional dict with:
                - fast_period (int): Fast MA period. Default 9.
                - slow_period (int): Slow MA period. Default 21.
                - ma_type (str): 'ema' or 'sma'. Default 'ema'.

        Returns:
            Dict with keys:
                - 'long_entry': Boolean Series (True on bar where fast crosses above slow)
                - 'short_entry': Boolean Series (True on bar where fast crosses below slow)
                - 'fast_ma': The fast moving average Series
                - 'slow_ma': The slow moving average Series
        """
        params = params or {}
        fast_period = int(params.get("fast_period", 9))
        slow_period = int(params.get("slow_period", 21))
        ma_type = str(params.get("ma_type", "ema"))

        close = data["close"]

        if ma_type == "ema":
            fast_ma = close.ewm(span=fast_period, adjust=False).mean()
            slow_ma = close.ewm(span=slow_period, adjust=False).mean()
        else:
            fast_ma = close.rolling(window=fast_period, min_periods=fast_period).mean()
            slow_ma = close.rolling(window=slow_period, min_periods=slow_period).mean()

        # Current and previous state
        fast_above_slow = fast_ma > slow_ma
        fast_above_slow_prev = fast_above_slow.shift(1).fillna(False)

        # Crossover detection
        long_entry = fast_above_slow & ~fast_above_slow_prev
        short_entry = ~fast_above_slow & fast_above_slow_prev

        return {
            "long_entry": long_entry,
            "short_entry": short_entry,
            "fast_ma": fast_ma,
            "slow_ma": slow_ma,
        }

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate Crossover parameters.

        Rules:
            - fast_period must be less than slow_period.
            - fast_period must be between 2 and 100.
            - slow_period must be between 5 and 500.
            - ma_type must be 'ema' or 'sma'.
        """
        fast = int(params.get("fast_period", 9))
        slow = int(params.get("slow_period", 21))
        ma_type = params.get("ma_type", "ema")

        if fast >= slow:
            return False
        if fast < 2 or fast > 100:
            return False
        if slow < 5 or slow > 500:
            return False
        if ma_type not in ["ema", "sma"]:
            return False

        return True
