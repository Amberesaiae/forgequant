"""
Price Breakout Detection Block.

Detects when price closes above the highest high (for long breakouts)
or below the lowest low (for short breakouts) of the lookback period.

A breakout signals that price has moved beyond a significant level,
often indicating the start of a new trend or continuation of momentum.

Key implementation details:
    - Uses shift(1) on the rolling max/min so that the current bar's
      high/low is NOT included in the lookback range. This prevents
      look-ahead bias where a bar would compare against itself.
    - Returns a boolean Series where True = breakout bar.

Default Parameters:
    lookback: 20
    direction: 'long'  ('long' for upward breakout, 'short' for downward)
"""

from typing import Any, Dict

import pandas as pd

from strategies_library.base import BaseBlock, BlockMetadata
from strategies_library.registry import BlockRegistry


@BlockRegistry.register
class Breakout(BaseBlock):
    """Price breakout detection above recent high or below recent low."""

    metadata = BlockMetadata(
        name="Breakout",
        category="price_action",
        description="Detects price breakout above recent high or below recent low",
        complexity=3,
        typical_use=["momentum", "trend_following", "breakout"],
        required_columns=["high", "low", "close"],
        version="1.0.0",
        tags=["breakout", "momentum", "trend"],
    )

    def compute(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any] | None = None,
    ) -> pd.Series:
        """Detect breakout conditions.

        Args:
            data: OHLCV DataFrame with 'high', 'low', 'close' columns.
            params: Optional dict with:
                - lookback (int): Number of bars to look back for the range. Default 20.
                - direction (str): 'long' for upward breakout, 'short' for downward. Default 'long'.

        Returns:
            Boolean pd.Series. True on bars where a breakout occurs.
        """
        params = params or {}
        lookback = int(params.get("lookback", 20))
        direction = str(params.get("direction", "long"))

        if direction == "long":
            # Close breaks above the highest high of the previous 'lookback' bars
            # shift(1) excludes the current bar from the range
            recent_high = data["high"].shift(1).rolling(
                window=lookback, min_periods=lookback
            ).max()
            return data["close"] > recent_high
        else:
            # Close breaks below the lowest low of the previous 'lookback' bars
            recent_low = data["low"].shift(1).rolling(
                window=lookback, min_periods=lookback
            ).min()
            return data["close"] < recent_low

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate Breakout parameters.

        Rules:
            - lookback must be between 5 and 200.
            - direction must be 'long' or 'short'.
        """
        lookback = params.get("lookback", 20)
        direction = params.get("direction", "long")

        if not isinstance(lookback, (int, float)):
            return False
        if int(lookback) < 5 or int(lookback) > 200:
            return False
        if direction not in ["long", "short"]:
            return False

        return True
