"""
Trend Filter block.

Only allows trades aligned with the broader trend direction.
Uses a long-period moving average with a buffer band to reduce whipsaws.
"""

from __future__ import annotations

import pandas as pd

from forgequant.blocks.base import BaseBlock
from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory, BlockParams, BlockResult


@BlockRegistry.register
class TrendFilter(BaseBlock):
    """Trend alignment filter with buffer band to reduce whipsaws."""

    metadata = BlockMetadata(
        name="trend_filter",
        display_name="Trend Filter",
        category=BlockCategory.FILTER,
        description=(
            "Allows trades only when price is clearly above (for longs) "
            "or below (for shorts) a long-period moving average. A buffer "
            "band around the MA creates a neutral zone where no trades "
            "are permitted, reducing whipsaw losses near the trend boundary."
        ),
        parameters=(
            ParameterSpec(
                name="period",
                param_type="int",
                default=200,
                min_value=10,
                max_value=1000,
                description="Moving average period for trend determination",
            ),
            ParameterSpec(
                name="ma_type",
                param_type="str",
                default="ema",
                choices=("ema", "sma"),
                description="Moving average type",
            ),
            ParameterSpec(
                name="buffer_pct",
                param_type="float",
                default=0.5,
                min_value=0.0,
                max_value=5.0,
                description=(
                    "Buffer zone width as percentage of MA. "
                    "Set to 0 for no buffer."
                ),
            ),
        ),
        tags=("filter", "trend", "moving_average", "direction", "whipsaw"),
        typical_use=(
            "Essential for trend-following systems. Use a 200-period EMA "
            "with a 0.5% buffer. Only take long entries when trend_allow_long "
            "is True, and short entries when trend_allow_short is True. "
            "This single filter can eliminate a large portion of losing "
            "trades in choppy markets."
        ),
    )

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        period: int = params["period"]
        ma_type: str = params["ma_type"]
        buffer_pct: float = params["buffer_pct"]

        if len(data) < period:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Insufficient data: need at least {period} rows, "
                       f"got {len(data)}",
            )

        close = data["close"]

        if ma_type == "ema":
            trend_ma = close.ewm(span=period, adjust=False).mean()
        else:
            trend_ma = close.rolling(window=period).mean()

        buffer_factor = buffer_pct / 100.0
        upper_buffer = trend_ma * (1.0 + buffer_factor)
        lower_buffer = trend_ma * (1.0 - buffer_factor)

        allow_long = close > upper_buffer
        allow_short = close < lower_buffer

        direction = pd.Series("neutral", index=data.index, dtype="object")
        direction[allow_long] = "bullish"
        direction[allow_short] = "bearish"

        result = pd.DataFrame(
            {
                "trend_ma": trend_ma,
                "trend_upper_buffer": upper_buffer,
                "trend_lower_buffer": lower_buffer,
                "trend_allow_long": allow_long.fillna(False),
                "trend_allow_short": allow_short.fillna(False),
                "trend_direction": direction,
            },
            index=data.index,
        )

        return result
