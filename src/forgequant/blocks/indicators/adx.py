"""
Average Directional Index (ADX) indicator block.

The ADX quantifies trend strength regardless of direction. It is derived
from the Directional Movement System developed by J. Welles Wilder Jr.

Calculation:
    1. True Range (TR):
       TR = max(high - low, |high - prev_close|, |low - prev_close|)
    2. Directional Movement:
       +DM = high - prev_high  if (high - prev_high) > (prev_low - low) AND > 0
       -DM = prev_low - low    if (prev_low - low) > (high - prev_high) AND > 0
    3. Smoothed TR, +DM, -DM using Wilder's smoothing (alpha=1/period)
    4. Directional Indicators:
       +DI = 100 * smoothed(+DM) / smoothed(TR)
       -DI = 100 * smoothed(-DM) / smoothed(TR)
    5. DX = 100 * |+DI - -DI| / (+DI + -DI)
    6. ADX = Wilder-smoothed DX over the specified period

Output columns:
    - adx: The ADX line (0-100, trend strength)
    - plus_di: +DI line
    - minus_di: -DI line
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from forgequant.blocks.base import BaseBlock
from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory, BlockParams, BlockResult


@BlockRegistry.register
class ADXIndicator(BaseBlock):
    """Average Directional Index indicator using Wilder's method."""

    metadata = BlockMetadata(
        name="adx",
        display_name="Average Directional Index",
        category=BlockCategory.INDICATOR,
        description=(
            "Calculates the Average Directional Index (ADX) along with the "
            "+DI and -DI directional indicators. ADX measures trend strength "
            "on a 0-100 scale: below 20 indicates a weak/absent trend, "
            "above 25 a developing trend, and above 50 a strong trend."
        ),
        parameters=(
            ParameterSpec(
                name="period",
                param_type="int",
                default=14,
                min_value=2,
                max_value=200,
                description="Lookback period for ADX calculation",
            ),
        ),
        tags=("trend", "strength", "directional", "wilder", "volatility"),
        typical_use=(
            "Used as a trend strength filter: only take trend-following trades "
            "when ADX > 25. The +DI/-DI crossover can also serve as a "
            "directional entry signal. Useful for distinguishing trending "
            "vs ranging market regimes."
        ),
    )

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        period: int = params["period"]

        min_rows = 2 * period + 1
        if len(data) < min_rows:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Insufficient data: need at least {min_rows} rows, "
                       f"got {len(data)}",
            )

        high = data["high"]
        low = data["low"]
        close = data["close"]

        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        up_move = high - prev_high
        down_move = prev_low - low

        plus_dm = pd.Series(
            np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
            index=data.index,
        )
        minus_dm = pd.Series(
            np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
            index=data.index,
        )

        alpha = 1.0 / period
        smoothed_tr = true_range.ewm(alpha=alpha, adjust=False).mean()
        smoothed_plus_dm = plus_dm.ewm(alpha=alpha, adjust=False).mean()
        smoothed_minus_dm = minus_dm.ewm(alpha=alpha, adjust=False).mean()

        plus_di = 100.0 * smoothed_plus_dm / smoothed_tr.replace(0, np.nan)
        minus_di = 100.0 * smoothed_minus_dm / smoothed_tr.replace(0, np.nan)

        di_sum = plus_di + minus_di
        di_diff = (plus_di - minus_di).abs()
        dx = 100.0 * di_diff / di_sum.replace(0, np.nan)

        adx = dx.ewm(alpha=alpha, adjust=False).mean()

        result = pd.DataFrame(
            {
                "adx": adx,
                "plus_di": plus_di,
                "minus_di": minus_di,
            },
            index=data.index,
        )

        return result
