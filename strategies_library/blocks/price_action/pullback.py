"""
Pullback in Trend Detection Block.

Detects when price pulls back to a support level within an existing
uptrend (or resistance in a downtrend). This is a "buy the dip"
style entry that waits for a temporary retracement before joining
the prevailing trend.

Logic:
    1. Determine if an uptrend exists (close > long-term moving average).
    2. Detect if price has pulled back near the recent low
       (within proximity_pct of the rolling minimum).
    3. Signal is True when BOTH conditions are met.

For short-side pullbacks (sell the rally in a downtrend), use
direction='short'.

Default Parameters:
    trend_period: 50
    pullback_lookback: 10
    proximity_pct: 0.008  (0.8% from recent low/high)
    direction: 'long'
"""

from typing import Any, Dict

import pandas as pd

from strategies_library.base import BaseBlock, BlockMetadata
from strategies_library.registry import BlockRegistry


@BlockRegistry.register
class Pullback(BaseBlock):
    """Pullback to support/resistance within a trend."""

    metadata = BlockMetadata(
        name="Pullback",
        category="price_action",
        description="Detects pullback to support within a trend for high-probability entries",
        complexity=4,
        typical_use=["trend_following", "dip_buying", "mean_reversion"],
        required_columns=["high", "low", "close"],
        version="1.0.0",
        tags=["pullback", "dip", "trend", "support"],
    )

    def compute(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any] | None = None,
    ) -> pd.Series:
        """Detect pullback conditions.

        Args:
            data: OHLCV DataFrame with 'high', 'low', 'close' columns.
            params: Optional dict with:
                - trend_period (int): MA period for trend confirmation. Default 50.
                - pullback_lookback (int): Bars to look back for recent low/high. Default 10.
                - proximity_pct (float): How close to recent low/high to trigger. Default 0.008.
                - direction (str): 'long' for buy-the-dip, 'short' for sell-the-rally. Default 'long'.

        Returns:
            Boolean pd.Series. True on bars where a pullback entry is detected.
        """
        params = params or {}
        trend_period = int(params.get("trend_period", 50))
        pullback_lookback = int(params.get("pullback_lookback", 10))
        proximity_pct = float(params.get("proximity_pct", 0.008))
        direction = str(params.get("direction", "long"))

        close = data["close"]
        trend_ma = close.rolling(window=trend_period, min_periods=trend_period).mean()

        if direction == "long":
            # Uptrend: close above long-term MA
            in_trend = close > trend_ma

            # Pullback: close is near the recent low
            recent_low = data["low"].rolling(
                window=pullback_lookback, min_periods=pullback_lookback
            ).min()
            near_level = close <= recent_low * (1.0 + proximity_pct)

            return in_trend & near_level
        else:
            # Downtrend: close below long-term MA
            in_trend = close < trend_ma

            # Rally: close is near the recent high
            recent_high = data["high"].rolling(
                window=pullback_lookback, min_periods=pullback_lookback
            ).max()
            near_level = close >= recent_high * (1.0 - proximity_pct)

            return in_trend & near_level

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate Pullback parameters.

        Rules:
            - trend_period must be between 20 and 200.
            - pullback_lookback must be between 3 and 50.
            - proximity_pct must be between 0.001 and 0.05.
            - direction must be 'long' or 'short'.
        """
        trend_period = params.get("trend_period", 50)
        pullback_lookback = params.get("pullback_lookback", 10)
        proximity_pct = params.get("proximity_pct", 0.008)
        direction = params.get("direction", "long")

        if int(trend_period) < 20 or int(trend_period) > 200:
            return False
        if int(pullback_lookback) < 3 or int(pullback_lookback) > 50:
            return False
        if float(proximity_pct) < 0.001 or float(proximity_pct) > 0.05:
            return False
        if direction not in ["long", "short"]:
            return False

        return True
