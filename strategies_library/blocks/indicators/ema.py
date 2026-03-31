"""
Exponential Moving Average (EMA) Indicator Block.

The EMA gives more weight to recent prices, making it more responsive
to new information compared to a Simple Moving Average (SMA).

Formula:
    EMA_today = (Price_today * k) + (EMA_yesterday * (1 - k))
    where k = 2 / (period + 1)

Common uses:
    - Trend direction (price above EMA = bullish, below = bearish)
    - Crossover signals (fast EMA crosses slow EMA)
    - Dynamic support/resistance levels

Default Parameters:
    period: 20
"""

from typing import Any, Dict, Union

import pandas as pd

from strategies_library.base import BaseBlock, BlockMetadata
from strategies_library.registry import BlockRegistry


@BlockRegistry.register
class EMA(BaseBlock):
    """Exponential Moving Average indicator."""

    metadata = BlockMetadata(
        name="EMA",
        category="indicator",
        description="Exponential Moving Average — gives more weight to recent prices",
        complexity=2,
        typical_use=["trend_following", "crossover", "dynamic_support"],
        required_columns=["close"],
        version="1.0.0",
        tags=["trend", "moving_average", "ema"],
    )

    def compute(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any] | None = None,
    ) -> pd.Series:
        """Compute EMA on the close price.

        Args:
            data: OHLCV DataFrame with a 'close' column.
            params: Optional dict with:
                - period (int): EMA lookback period. Default 20.

        Returns:
            pd.Series containing the EMA values. First (period - 1) values
            will be less accurate due to insufficient history but are still
            computed (ewm handles warmup internally).
        """
        params = params or {}
        period = int(params.get("period", 20))

        return data["close"].ewm(span=period, adjust=False).mean()

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate EMA parameters.

        Rules:
            - period must be between 2 and 500.
        """
        period = params.get("period", 20)
        if not isinstance(period, (int, float)):
            return False
        if int(period) < 2 or int(period) > 500:
            return False
        return True
