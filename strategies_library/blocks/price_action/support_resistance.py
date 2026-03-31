"""
Dynamic Support and Resistance Level Detection Block.

Identifies key support and resistance levels using rolling min/max
of lows and highs over a specified lookback period.

Also provides proximity detection: boolean signals indicating when
price is near a support or resistance level.

Logic:
    - Support: Rolling minimum of lows over lookback period.
    - Resistance: Rolling maximum of highs over lookback period.
    - Near support: Close is within tolerance_pct of the support level.
    - Near resistance: Close is within tolerance_pct of the resistance level.

Default Parameters:
    lookback: 50
    tolerance_pct: 0.005  (0.5% proximity to trigger near-level signal)
"""

from typing import Any, Dict

import pandas as pd

from strategies_library.base import BaseBlock, BlockMetadata
from strategies_library.registry import BlockRegistry


@BlockRegistry.register
class SupportResistance(BaseBlock):
    """Dynamic support and resistance level detection."""

    metadata = BlockMetadata(
        name="SupportResistance",
        category="price_action",
        description="Dynamic support and resistance levels with proximity detection",
        complexity=4,
        typical_use=["mean_reversion", "breakout", "level_trading"],
        required_columns=["high", "low", "close"],
        version="1.0.0",
        tags=["levels", "support", "resistance", "zones"],
    )

    def compute(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any] | None = None,
    ) -> Dict[str, pd.Series]:
        """Compute support, resistance, and proximity signals.

        Args:
            data: OHLCV DataFrame with 'high', 'low', 'close' columns.
            params: Optional dict with:
                - lookback (int): Rolling window for level calculation. Default 50.
                - tolerance_pct (float): Proximity percentage for near-level signals.
                  Default 0.005 (0.5%).

        Returns:
            Dict with keys:
                - 'support': Support level Series
                - 'resistance': Resistance level Series
                - 'near_support': Boolean Series (True when close is near support)
                - 'near_resistance': Boolean Series (True when close is near resistance)
        """
        params = params or {}
        lookback = int(params.get("lookback", 50))
        tolerance_pct = float(params.get("tolerance_pct", 0.005))

        close = data["close"]
        high = data["high"]
        low = data["low"]

        # Support: rolling minimum of lows
        support = low.rolling(window=lookback, min_periods=lookback).min()

        # Resistance: rolling maximum of highs
        resistance = high.rolling(window=lookback, min_periods=lookback).max()

        # Proximity detection
        near_support = close <= support * (1.0 + tolerance_pct)
        near_resistance = close >= resistance * (1.0 - tolerance_pct)

        return {
            "support": support,
            "resistance": resistance,
            "near_support": near_support,
            "near_resistance": near_resistance,
        }

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate SupportResistance parameters.

        Rules:
            - lookback must be between 10 and 500.
            - tolerance_pct must be between 0.001 and 0.05.
        """
        lookback = params.get("lookback", 50)
        tolerance_pct = params.get("tolerance_pct", 0.005)

        if int(lookback) < 10 or int(lookback) > 500:
            return False
        if float(tolerance_pct) < 0.001 or float(tolerance_pct) > 0.05:
            return False

        return True
