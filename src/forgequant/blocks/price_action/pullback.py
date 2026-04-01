"""
Pullback price action block.

Detects pullbacks (retracements) to a moving average in a trending market.
A pullback long is when price was above the MA, dips to touch or cross
below it, and then closes back above it. Vice versa for short pullbacks.
"""

from __future__ import annotations

import pandas as pd

from forgequant.blocks.base import BaseBlock
from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory, BlockParams, BlockResult


@BlockRegistry.register
class PullbackBlock(BaseBlock):
    """Pullback to moving average detection."""

    metadata = BlockMetadata(
        name="pullback",
        display_name="Pullback",
        category=BlockCategory.PRICE_ACTION,
        description=(
            "Detects pullbacks (retracements) to a moving average. A long "
            "pullback occurs when price is in an uptrend (above MA), dips "
            "into the MA proximity zone, and closes back above the MA. "
            "Short pullbacks are the mirror image."
        ),
        parameters=(
            ParameterSpec(
                name="ma_period",
                param_type="int",
                default=20,
                min_value=2,
                max_value=500,
                description="Period for the moving average",
            ),
            ParameterSpec(
                name="ma_type",
                param_type="str",
                default="ema",
                choices=("ema", "sma"),
                description="Type of moving average (EMA or SMA)",
            ),
            ParameterSpec(
                name="proximity_pct",
                param_type="float",
                default=0.5,
                min_value=0.01,
                max_value=5.0,
                description=(
                    "Percentage distance from MA defining the proximity zone. "
                    "Price must touch this zone for a pullback to be valid."
                ),
            ),
        ),
        tags=("pullback", "retracement", "trend", "moving_average", "mean_reversion"),
        typical_use=(
            "Used in trend-following systems to enter on dips rather than "
            "breakouts. The proximity zone prevents requiring exact MA "
            "touches, which are rare on higher timeframes. Combine with "
            "a trend filter (ADX > 25) for best results."
        ),
    )

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        ma_period: int = params["ma_period"]
        ma_type: str = params["ma_type"]
        proximity_pct: float = params["proximity_pct"]

        if len(data) < ma_period + 1:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Insufficient data: need at least {ma_period + 1} rows, "
                       f"got {len(data)}",
            )

        close = data["close"]
        high = data["high"]
        low = data["low"]

        if ma_type == "ema":
            ma = close.ewm(span=ma_period, adjust=False).mean()
        else:
            ma = close.rolling(window=ma_period).mean()

        zone_factor = proximity_pct / 100.0
        upper_zone = ma * (1.0 + zone_factor)
        lower_zone = ma * (1.0 - zone_factor)

        prev_close_above_ma = close.shift(1) > ma.shift(1)
        prev_close_below_ma = close.shift(1) < ma.shift(1)

        pullback_long = (
            prev_close_above_ma
            & (low <= upper_zone)
            & (close > ma)
        )

        pullback_short = (
            prev_close_below_ma
            & (high >= lower_zone)
            & (close < ma)
        )

        result = pd.DataFrame(
            {
                "pullback_ma": ma,
                "pullback_upper_zone": upper_zone,
                "pullback_lower_zone": lower_zone,
                "pullback_long": pullback_long.fillna(False),
                "pullback_short": pullback_short.fillna(False),
            },
            index=data.index,
        )

        return result
