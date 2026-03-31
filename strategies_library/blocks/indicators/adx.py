"""
Average Directional Index (ADX) Indicator Block.

ADX measures the strength of a trend regardless of its direction.
It is derived from the Directional Movement System:
    - +DI (Plus Directional Indicator): Measures upward movement
    - -DI (Minus Directional Indicator): Measures downward movement
    - ADX: Smoothed average of the directional index (DX)

Interpretation:
    - ADX > 25: Strong trend (good for trend-following strategies)
    - ADX < 20: Weak or no trend (good for mean-reversion strategies)
    - +DI > -DI: Uptrend
    - -DI > +DI: Downtrend

Default Parameters:
    period: 14
"""

from typing import Any, Dict

import pandas as pd

from strategies_library.base import BaseBlock, BlockMetadata
from strategies_library.registry import BlockRegistry


@BlockRegistry.register
class ADX(BaseBlock):
    """Average Directional Index trend strength indicator."""

    metadata = BlockMetadata(
        name="ADX",
        category="indicator",
        description="ADX — measures trend strength regardless of direction",
        complexity=4,
        typical_use=["trend_strength", "filter", "regime_detection"],
        required_columns=["high", "low", "close"],
        version="1.0.0",
        tags=["trend", "strength", "filter", "directional"],
    )

    def compute(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any] | None = None,
    ) -> Dict[str, pd.Series]:
        """Compute ADX, +DI, and -DI.

        Args:
            data: OHLCV DataFrame with 'high', 'low', 'close' columns.
            params: Optional dict with:
                - period (int): ADX calculation period. Default 14.

        Returns:
            Dict with keys:
                - 'adx': ADX line Series (0 to 100)
                - 'plus_di': +DI line Series
                - 'minus_di': -DI line Series
        """
        params = params or {}
        period = int(params.get("period", 14))

        high = data["high"]
        low = data["low"]
        close = data["close"]

        # Previous values
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)

        # True Range components
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = high - prev_high
        down_move = prev_low - low

        # Plus DM: up_move when it is positive and greater than down_move
        plus_dm = pd.Series(0.0, index=data.index, dtype=float)
        plus_dm_mask = (up_move > down_move) & (up_move > 0)
        plus_dm[plus_dm_mask] = up_move[plus_dm_mask]

        # Minus DM: down_move when it is positive and greater than up_move
        minus_dm = pd.Series(0.0, index=data.index, dtype=float)
        minus_dm_mask = (down_move > up_move) & (down_move > 0)
        minus_dm[minus_dm_mask] = down_move[minus_dm_mask]

        # Smoothed averages using rolling mean
        smoothed_tr = true_range.rolling(window=period, min_periods=period).mean()
        smoothed_plus_dm = plus_dm.rolling(window=period, min_periods=period).mean()
        smoothed_minus_dm = minus_dm.rolling(window=period, min_periods=period).mean()

        # Prevent division by zero
        smoothed_tr = smoothed_tr.replace(0.0, 1e-10)

        # Directional Indicators
        plus_di = 100.0 * (smoothed_plus_dm / smoothed_tr)
        minus_di = 100.0 * (smoothed_minus_dm / smoothed_tr)

        # Directional Index
        di_sum = plus_di + minus_di
        di_sum = di_sum.replace(0.0, 1e-10)
        dx = 100.0 * ((plus_di - minus_di).abs() / di_sum)

        # ADX is the smoothed DX
        adx = dx.rolling(window=period, min_periods=period).mean()

        return {
            "adx": adx,
            "plus_di": plus_di,
            "minus_di": minus_di,
        }

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate ADX parameters.

        Rules:
            - period must be between 7 and 50.
        """
        period = params.get("period", 14)
        if not isinstance(period, (int, float)):
            return False
        if int(period) < 7 or int(period) > 50:
            return False
        return True
