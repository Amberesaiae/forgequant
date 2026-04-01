"""
Trailing Stop exit rule block.

Computes a trailing stop level that follows price in the direction of
the trade, locking in profits. The stop only moves in the favorable
direction and never retreats.
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
class TrailingStopExit(BaseBlock):
    """ATR-based trailing stop exit."""

    metadata = BlockMetadata(
        name="trailing_stop",
        display_name="Trailing Stop",
        category=BlockCategory.EXIT_RULE,
        description=(
            "Computes an ATR-based trailing stop that follows price in the "
            "profitable direction and never retreats. For longs, the stop "
            "ratchets upward; for shorts, downward. Exit signals fire when "
            "price crosses the trailing stop level."
        ),
        parameters=(
            ParameterSpec(
                name="atr_period",
                param_type="int",
                default=14,
                min_value=1,
                max_value=200,
                description="ATR calculation period",
            ),
            ParameterSpec(
                name="trail_atr_mult",
                param_type="float",
                default=2.5,
                min_value=0.1,
                max_value=20.0,
                description="Trailing distance as a multiple of ATR",
            ),
        ),
        tags=("exit", "trailing", "stop_loss", "atr", "trend_following"),
        typical_use=(
            "Essential exit for trend-following systems. Lets profits run "
            "while protecting against reversals. A wider multiplier (3x ATR) "
            "gives the trade more room but risks giving back more profit. "
            "A tighter multiplier (1.5x ATR) exits sooner but may be "
            "whipsawed in volatile trends."
        ),
    )

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        atr_period: int = params["atr_period"]
        trail_mult: float = params["trail_atr_mult"]

        min_rows = atr_period + 1
        if len(data) < min_rows:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Insufficient data: need at least {min_rows} rows, "
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
        alpha = 1.0 / atr_period
        atr = true_range.ewm(alpha=alpha, adjust=False).mean()

        trail_distance = atr * trail_mult

        raw_long_stop = high - trail_distance
        raw_short_stop = low + trail_distance

        n = len(data)
        long_stop = np.full(n, np.nan)
        short_stop = np.full(n, np.nan)

        long_stop[0] = raw_long_stop.iloc[0]
        short_stop[0] = raw_short_stop.iloc[0]

        raw_long_vals = raw_long_stop.values
        raw_short_vals = raw_short_stop.values

        for i in range(1, n):
            if not np.isnan(raw_long_vals[i]):
                if not np.isnan(long_stop[i - 1]):
                    long_stop[i] = max(long_stop[i - 1], raw_long_vals[i])
                else:
                    long_stop[i] = raw_long_vals[i]
            else:
                long_stop[i] = long_stop[i - 1]

            if not np.isnan(raw_short_vals[i]):
                if not np.isnan(short_stop[i - 1]):
                    short_stop[i] = min(short_stop[i - 1], raw_short_vals[i])
                else:
                    short_stop[i] = raw_short_vals[i]
            else:
                short_stop[i] = short_stop[i - 1]

        long_stop_series = pd.Series(long_stop, index=data.index)
        short_stop_series = pd.Series(short_stop, index=data.index)

        long_exit = close < long_stop_series
        short_exit = close > short_stop_series

        result = pd.DataFrame(
            {
                "trail_atr": atr,
                "trail_long_stop": long_stop_series,
                "trail_short_stop": short_stop_series,
                "trail_long_exit": long_exit.fillna(False),
                "trail_short_exit": short_exit.fillna(False),
            },
            index=data.index,
        )

        return result
