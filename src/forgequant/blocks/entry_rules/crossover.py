"""
Crossover entry rule block.

Generates entry signals when a fast moving average crosses a slow moving
average. The cross is detected via state change — the signal fires on
the FIRST bar where the cross condition becomes true.
"""

from __future__ import annotations

import pandas as pd

from forgequant.blocks.base import BaseBlock
from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory, BlockParams, BlockResult, SIGNAL_COLUMNS as SC


@BlockRegistry.register
class CrossoverEntry(BaseBlock):
    """Moving average crossover entry signals."""

    metadata = BlockMetadata(
        name="crossover_entry",
        display_name="MA Crossover Entry",
        category=BlockCategory.ENTRY_RULE,
        description=(
            "Generates entry signals when a fast moving average crosses "
            "a slow moving average. The signal fires only on the exact "
            "bar of the crossover (state change), not on every bar where "
            "fast > slow."
        ),
        parameters=(
            ParameterSpec(
                name="fast_period",
                param_type="int",
                default=10,
                min_value=2,
                max_value=200,
                description="Period for the fast moving average",
            ),
            ParameterSpec(
                name="slow_period",
                param_type="int",
                default=20,
                min_value=2,
                max_value=500,
                description="Period for the slow moving average",
            ),
            ParameterSpec(
                name="ma_type",
                param_type="str",
                default="ema",
                choices=("ema", "sma"),
                description="Moving average type (EMA or SMA)",
            ),
        ),
        tags=("crossover", "moving_average", "trend", "entry", "signal"),
        typical_use=(
            "Classic trend-following entry: go long on golden cross (fast "
            "above slow), go short on death cross. Works best in trending "
            "markets. Combine with a trend strength filter (ADX) and "
            "exit rules for a complete system."
        ),
    )

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        fast_period: int = params["fast_period"]
        slow_period: int = params["slow_period"]
        ma_type: str = params["ma_type"]

        if fast_period >= slow_period:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"fast_period ({fast_period}) must be less than "
                       f"slow_period ({slow_period})",
            )

        min_rows = slow_period + 1
        if len(data) < min_rows:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Insufficient data: need at least {min_rows} rows, "
                       f"got {len(data)}",
            )

        close = data["close"]

        if ma_type == "ema":
            fast_ma = close.ewm(span=fast_period, adjust=False).mean()
            slow_ma = close.ewm(span=slow_period, adjust=False).mean()
        else:
            fast_ma = close.rolling(window=fast_period).mean()
            slow_ma = close.rolling(window=slow_period).mean()

        fast_above = (fast_ma > slow_ma).fillna(False)
        prev_fast_above = fast_above.shift(1).fillna(False)

        long_entry = fast_above & prev_fast_above.eq(False)
        short_entry = fast_above.eq(False) & prev_fast_above

        result = pd.DataFrame(
            {
                "crossover_fast_ma": fast_ma,
                "crossover_slow_ma": slow_ma,
                SC.crossover_long_entry: long_entry.fillna(False),
                SC.crossover_short_entry: short_entry.fillna(False),
            },
            index=data.index,
        )

        return result
