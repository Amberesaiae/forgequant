"""
Higher High / Lower Low Structure Detection Block.

Analyzes price structure by identifying swing highs and swing lows,
then comparing consecutive swings to determine trend structure.

A swing high is a bar whose high is the highest within a window
of (2 * swing_lookback + 1) bars centered on it.

A swing low is a bar whose low is the lowest within a window
of (2 * swing_lookback + 1) bars centered on it.

Structure:
    - Higher Highs + Higher Lows = Uptrend
    - Lower Highs + Lower Lows = Downtrend
    - Mixed = Ranging / Transition

Default Parameters:
    swing_lookback: 5  (bars on each side to confirm a swing point)
"""

from typing import Any, Dict

import pandas as pd

from strategies_library.base import BaseBlock, BlockMetadata
from strategies_library.registry import BlockRegistry


@BlockRegistry.register
class HigherHighLowerLow(BaseBlock):
    """Swing structure analysis — higher highs, lower lows detection."""

    metadata = BlockMetadata(
        name="HigherHighLowerLow",
        category="price_action",
        description="Detects higher highs and lower lows in price swing structure",
        complexity=4,
        typical_use=["trend_following", "structure_analysis", "regime_detection"],
        required_columns=["high", "low"],
        version="1.0.0",
        tags=["structure", "trend", "swing", "higher_high", "lower_low"],
    )

    def compute(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any] | None = None,
    ) -> Dict[str, pd.Series]:
        """Detect higher highs and lower lows.

        Args:
            data: OHLCV DataFrame with 'high' and 'low' columns.
            params: Optional dict with:
                - swing_lookback (int): Bars on each side for swing detection. Default 5.

        Returns:
            Dict with keys:
                - 'higher_highs': Boolean Series (True when current swing high > previous)
                - 'lower_lows': Boolean Series (True when current swing low < previous)
                - 'swing_highs': Series with swing high values (NaN elsewhere)
                - 'swing_lows': Series with swing low values (NaN elsewhere)
        """
        params = params or {}
        swing_lookback = int(params.get("swing_lookback", 5))

        high = data["high"]
        low = data["low"]
        window_size = 2 * swing_lookback + 1

        # Detect swing highs: bar high equals the rolling max centered on it
        rolling_max = high.rolling(window=window_size, center=True, min_periods=window_size).max()
        is_swing_high = high == rolling_max

        # Detect swing lows: bar low equals the rolling min centered on it
        rolling_min = low.rolling(window=window_size, center=True, min_periods=window_size).min()
        is_swing_low = low == rolling_min

        # Extract swing high and low values (NaN where not a swing point)
        swing_high_values = high.where(is_swing_high)
        swing_low_values = low.where(is_swing_low)

        # Forward-fill to compare consecutive swings
        prev_swing_high = swing_high_values.ffill()
        prev_prev_swing_high = swing_high_values.ffill().shift(1)

        prev_swing_low = swing_low_values.ffill()
        prev_prev_swing_low = swing_low_values.ffill().shift(1)

        # Compare consecutive swing points
        higher_highs = (prev_swing_high > prev_prev_swing_high).fillna(False)
        lower_lows = (prev_swing_low < prev_prev_swing_low).fillna(False)

        return {
            "higher_highs": higher_highs,
            "lower_lows": lower_lows,
            "swing_highs": swing_high_values,
            "swing_lows": swing_low_values,
        }

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate HigherHighLowerLow parameters.

        Rules:
            - swing_lookback must be between 2 and 20.
        """
        swing_lookback = params.get("swing_lookback", 5)
        if not isinstance(swing_lookback, (int, float)):
            return False
        if int(swing_lookback) < 2 or int(swing_lookback) > 20:
            return False
        return True
