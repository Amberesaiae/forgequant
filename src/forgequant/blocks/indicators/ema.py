"""
Exponential Moving Average (EMA) indicator block.

The EMA applies exponentially decreasing weights to past prices, giving
more significance to recent data than a Simple Moving Average (SMA).

Formula:
    multiplier = 2 / (period + 1)
    EMA_t = close_t * multiplier + EMA_{t-1} * (1 - multiplier)

This implementation uses pandas ewm(span=period, adjust=False) which
produces the recursive EMA formula above. adjust=False is chosen because:
    1. It matches the standard MetaTrader / TradingView EMA definition.
    2. It avoids the expanding-window correction that adjust=True applies,
       which would make early values differ from reference platforms.

Output columns:
    - ema_{period}: The EMA values
"""

from __future__ import annotations

import pandas as pd

from forgequant.blocks.base import BaseBlock
from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory, BlockParams, BlockResult


@BlockRegistry.register
class EMAIndicator(BaseBlock):
    """Exponential Moving Average indicator."""

    metadata = BlockMetadata(
        name="ema",
        display_name="Exponential Moving Average",
        category=BlockCategory.INDICATOR,
        description=(
            "Calculates the Exponential Moving Average of the close price. "
            "The EMA weights recent prices more heavily than older prices, "
            "making it more responsive to new information than a Simple "
            "Moving Average of the same period."
        ),
        parameters=(
            ParameterSpec(
                name="period",
                param_type="int",
                default=20,
                min_value=2,
                max_value=500,
                description="Number of periods for the EMA calculation",
            ),
            ParameterSpec(
                name="source",
                param_type="str",
                default="close",
                choices=("open", "high", "low", "close"),
                description="Price column to compute the EMA on",
            ),
        ),
        tags=("trend", "moving_average", "smoothing", "lagging"),
        typical_use=(
            "Used as a trend filter (price above EMA = bullish), as a "
            "dynamic support/resistance level, or as a component in "
            "crossover systems (fast EMA crossing slow EMA)."
        ),
    )

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        period: int = params["period"]
        source: str = params["source"]

        if source not in data.columns:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Source column '{source}' not found in data. "
                       f"Available: {list(data.columns)}",
            )

        if len(data) < period:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Insufficient data: need at least {period} rows, "
                       f"got {len(data)}",
            )

        ema_series = data[source].ewm(span=period, adjust=False).mean()

        col_name = f"ema_{period}"
        result = pd.DataFrame({col_name: ema_series}, index=data.index)

        return result
