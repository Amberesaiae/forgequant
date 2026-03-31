"""
ATR-Based Trailing Stop Block.

Computes a dynamic trailing stop level based on Average True Range.
The stop trails below price (for longs) or above price (for shorts)
by a distance of (multiplier × ATR).

As price moves in the trade's favor, the trailing stop moves with it.
When price reverses, the stop stays at its highest/lowest level.

This block computes the raw trailing offset. The execution layer
is responsible for tracking the actual trailing level over time
(ratcheting it in the trade's favor).

Default Parameters:
    atr_period: 14
    multiplier: 2.5
"""

from typing import Any, Dict

import pandas as pd

from strategies_library.base import BaseBlock, BlockMetadata
from strategies_library.registry import BlockRegistry


@BlockRegistry.register
class TrailingStop(BaseBlock):
    """ATR-based dynamic trailing stop."""

    metadata = BlockMetadata(
        name="TrailingStop",
        category="exit",
        description="ATR-based trailing stop that dynamically adjusts with volatility",
        complexity=3,
        typical_use=["trend_following", "exit", "risk_management"],
        required_columns=["high", "low", "close"],
        version="1.0.0",
        tags=["trailing", "atr", "dynamic", "exit"],
    )

    def compute(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any] | None = None,
    ) -> Dict[str, pd.Series]:
        """Compute trailing stop levels.

        Args:
            data: OHLCV DataFrame with 'high', 'low', 'close' columns.
            params: Optional dict with:
                - atr_period (int): ATR calculation period. Default 14.
                - multiplier (float): ATR multiplier for stop distance. Default 2.5.

        Returns:
            Dict with keys:
                - 'trailing_stop_long': Stop level for long positions (below price)
                - 'trailing_stop_short': Stop level for short positions (above price)
                - 'atr': The computed ATR Series
                - 'trailing_offset': The distance from price to stop
        """
        params = params or {}
        atr_period = int(params.get("atr_period", 14))
        multiplier = float(params.get("multiplier", 2.5))

        # Compute ATR
        high = data["high"]
        low = data["low"]
        close_prev = data["close"].shift(1)

        tr1 = high - low
        tr2 = (high - close_prev).abs()
        tr3 = (low - close_prev).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=atr_period, min_periods=atr_period).mean()

        trailing_offset = multiplier * atr

        # For long positions: stop is below price
        trailing_stop_long = data["close"] - trailing_offset

        # For short positions: stop is above price
        trailing_stop_short = data["close"] + trailing_offset

        return {
            "trailing_stop_long": trailing_stop_long,
            "trailing_stop_short": trailing_stop_short,
            "atr": atr,
            "trailing_offset": trailing_offset,
        }

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate TrailingStop parameters.

        Rules:
            - atr_period must be between 5 and 50.
            - multiplier must be between 0.5 and 5.0.
        """
        atr_period = int(params.get("atr_period", 14))
        multiplier = float(params.get("multiplier", 2.5))

        if atr_period < 5 or atr_period > 50:
            return False
        if multiplier < 0.5 or multiplier > 5.0:
            return False

        return True
