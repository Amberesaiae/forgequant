"""
Breakout price action block.

Detects when price breaks above the highest high or below the lowest low
of the preceding lookback window. Uses shift(1) on the rolling extremes
to guarantee the current bar is NEVER compared against itself, preventing
lookahead bias.
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
class BreakoutBlock(BaseBlock):
    """Breakout detection with anti-lookahead protection."""

    metadata = BlockMetadata(
        name="breakout",
        display_name="Breakout",
        category=BlockCategory.PRICE_ACTION,
        description=(
            "Detects price breakouts above the highest high or below the "
            "lowest low of the preceding lookback window. The rolling "
            "extreme is shifted by 1 bar to prevent the current bar from "
            "being included in its own comparison, eliminating lookahead bias."
        ),
        parameters=(
            ParameterSpec(
                name="lookback",
                param_type="int",
                default=20,
                min_value=2,
                max_value=500,
                description="Number of bars for the rolling high/low window",
            ),
            ParameterSpec(
                name="volume_multiplier",
                param_type="float",
                default=1.5,
                min_value=0.0,
                max_value=10.0,
                description=(
                    "Volume must exceed rolling average * this multiplier for "
                    "volume confirmation. Set to 0 to disable volume filter."
                ),
            ),
            ParameterSpec(
                name="volume_lookback",
                param_type="int",
                default=20,
                min_value=2,
                max_value=200,
                description="Lookback period for the average volume calculation",
            ),
        ),
        tags=("breakout", "momentum", "range", "high", "low", "volume"),
        typical_use=(
            "Used to enter trades when price breaks out of a consolidation "
            "range. Often combined with a volume confirmation filter and a "
            "trend filter (e.g. only take long breakouts in an uptrend)."
        ),
    )

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        lookback: int = params["lookback"]
        vol_mult: float = params["volume_multiplier"]
        vol_lookback: int = params["volume_lookback"]

        min_rows = lookback + 1
        if len(data) < min_rows:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Insufficient data: need at least {min_rows} rows, "
                       f"got {len(data)}",
            )

        high = data["high"]
        low = data["low"]
        close = data["close"]
        volume = data["volume"]

        rolling_high = high.rolling(window=lookback).max().shift(1)
        rolling_low = low.rolling(window=lookback).min().shift(1)

        breakout_long = close > rolling_high
        breakout_short = close < rolling_low

        if vol_mult > 0:
            avg_volume = volume.rolling(window=vol_lookback).mean().shift(1)
            volume_confirm = volume > (avg_volume * vol_mult)
        else:
            volume_confirm = pd.Series(True, index=data.index)

        result = pd.DataFrame(
            {
                "breakout_resistance": rolling_high,
                "breakout_support": rolling_low,
                SC.breakout_long: breakout_long.fillna(False),
                SC.breakout_short: breakout_short.fillna(False),
                SC.breakout_volume_confirm: volume_confirm.fillna(False),
            },
            index=data.index,
        )

        return result
