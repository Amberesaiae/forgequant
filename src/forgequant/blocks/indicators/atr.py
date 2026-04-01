"""
Average True Range (ATR) indicator block.

The ATR measures market volatility by decomposing the entire range
of an asset price for a given period.

Calculation:
    1. True Range:
       TR = max(high - low, |high - prev_close|, |low - prev_close|)
    2. ATR = Wilder-smoothed average of TR over the period
       (equivalent to EMA with alpha = 1/period)

Output columns:
    - atr_{period}: The ATR values (in price units)
    - atr_{period}_pct: ATR as a percentage of close price
"""

from __future__ import annotations

import pandas as pd

from forgequant.blocks.base import BaseBlock
from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory, BlockParams, BlockResult


@BlockRegistry.register
class ATRIndicator(BaseBlock):
    """Average True Range volatility indicator using Wilder's smoothing."""

    metadata = BlockMetadata(
        name="atr",
        display_name="Average True Range",
        category=BlockCategory.INDICATOR,
        description=(
            "Calculates the Average True Range, a volatility measure that "
            "accounts for gaps between bars. Higher ATR indicates higher "
            "volatility. ATR is commonly used for position sizing, stop-loss "
            "placement, and volatility filtering."
        ),
        parameters=(
            ParameterSpec(
                name="period",
                param_type="int",
                default=14,
                min_value=1,
                max_value=200,
                description="Lookback period for ATR smoothing",
            ),
        ),
        tags=("volatility", "range", "wilder", "stop_loss", "position_sizing"),
        typical_use=(
            "Used for ATR-based stop losses (e.g. 2x ATR from entry), "
            "position sizing (risk amount / ATR = position size), and as "
            "a volatility filter to avoid low-volatility chop or excessive "
            "volatility."
        ),
    )

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        period: int = params["period"]

        if len(data) < period + 1:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Insufficient data: need at least {period + 1} rows, "
                       f"got {len(data)}",
            )

        high = data["high"]
        low = data["low"]
        close = data["close"]
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        alpha = 1.0 / period
        atr = true_range.ewm(alpha=alpha, adjust=False).mean()
        atr_pct = (atr / close) * 100.0

        col = f"atr_{period}"
        result = pd.DataFrame(
            {
                col: atr,
                f"{col}_pct": atr_pct,
            },
            index=data.index,
        )

        return result
