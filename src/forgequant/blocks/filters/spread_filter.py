"""
Spread Filter block.

Filters out bars where the bid-ask spread (approximated from OHLCV data)
is too wide relative to ATR. Wide spreads indicate low liquidity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forgequant.blocks.base import BaseBlock
from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory, BlockParams, BlockResult, SIGNAL_COLUMNS as SC


@BlockRegistry.register
class SpreadFilter(BaseBlock):
    """Bid-ask spread quality filter."""

    metadata = BlockMetadata(
        name="spread_filter",
        display_name="Spread Filter",
        category=BlockCategory.FILTER,
        description=(
            "Filters out bars where the spread is too wide, indicating "
            "low liquidity or data quality issues. Uses an explicit "
            "'spread' column if available, otherwise approximates from "
            "the high-low range."
        ),
        parameters=(
            ParameterSpec(
                name="max_spread",
                param_type="float",
                default=50.0,
                min_value=0.1,
                max_value=10000.0,
                description=(
                    "Maximum absolute spread in pips-like units. "
                    "Bars exceeding this are filtered out."
                ),
            ),
            ParameterSpec(
                name="max_spread_ratio",
                param_type="float",
                default=3.0,
                min_value=1.0,
                max_value=20.0,
                description=(
                    "Maximum spread as a ratio of the rolling average spread. "
                    "Bars where spread > avg * ratio are filtered out."
                ),
            ),
            ParameterSpec(
                name="lookback",
                param_type="int",
                default=50,
                min_value=5,
                max_value=500,
                description="Lookback period for average spread calculation",
            ),
        ),
        tags=("filter", "spread", "liquidity", "cost", "quality"),
        typical_use=(
            "Used to avoid entering trades during illiquid periods (e.g. "
            "news events, market opens, off-hours) where wide spreads "
            "would erode profits. Essential for scalping and short-term "
            "strategies."
        ),
    )

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        max_spread: float = params["max_spread"]
        max_ratio: float = params["max_spread_ratio"]
        lookback: int = params["lookback"]

        high = data["high"]
        low = data["low"]
        close = data["close"]

        if "spread" in data.columns:
            spread = data["spread"].astype(float)
        else:
            spread = ((high - low) / close.replace(0, np.nan)) * 10000.0

        avg_spread = spread.rolling(window=lookback, min_periods=1).mean()

        below_absolute = spread <= max_spread
        below_ratio = spread <= (avg_spread * max_ratio)
        spread_ok = below_absolute & below_ratio

        result = pd.DataFrame(
            {
                "spread_value": spread,
                "spread_avg": avg_spread,
                SC.spread_ok: spread_ok.fillna(False),
            },
            index=data.index,
        )

        return result
