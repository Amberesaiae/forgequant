"""
Higher High / Lower Low (HHLL) price action block.

Identifies swing highs and swing lows using a configurable number of
left/right confirmation bars, then classifies the structure:
    - Higher High (HH): current swing high > previous swing high
    - Lower High (LH): current swing high < previous swing high
    - Higher Low (HL): current swing low > previous swing low
    - Lower Low (LL): current swing low < previous swing low
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
class HigherHighLowerLowBlock(BaseBlock):
    """Swing structure identification: HH, HL, LH, LL."""

    metadata = BlockMetadata(
        name="higher_high_lower_low",
        display_name="Higher High / Lower Low",
        category=BlockCategory.PRICE_ACTION,
        description=(
            "Identifies swing highs and swing lows using left/right bar "
            "confirmation, then classifies the structure as Higher High, "
            "Lower High, Higher Low, or Lower Low. Also derives a trend "
            "state: bullish (HH + HL), bearish (LH + LL), or neutral."
        ),
        parameters=(
            ParameterSpec(
                name="left_bars",
                param_type="int",
                default=5,
                min_value=1,
                max_value=50,
                description="Number of bars to the left that must be lower/higher",
            ),
            ParameterSpec(
                name="right_bars",
                param_type="int",
                default=5,
                min_value=1,
                max_value=50,
                description=(
                    "Number of bars to the right that must be lower/higher. "
                    "This determines the confirmation lag."
                ),
            ),
        ),
        tags=("swing", "structure", "trend", "higher_high", "lower_low", "price_action"),
        typical_use=(
            "Used to determine market structure and trend direction. A "
            "sequence of HH + HL indicates a bullish trend; LH + LL "
            "indicates bearish. A break of structure (e.g. first LL in "
            "a bullish sequence) can signal trend reversal."
        ),
    )

    @staticmethod
    def _find_swings(
        series: pd.Series,
        left_bars: int,
        right_bars: int,
        find_highs: bool,
    ) -> pd.Series:
        values = series.values
        n = len(values)
        swings = np.full(n, np.nan)

        for i in range(left_bars, n - right_bars):
            candidate = values[i]
            if np.isnan(candidate):
                continue

            is_swing = True

            if find_highs:
                for j in range(i - left_bars, i):
                    if np.isnan(values[j]) or candidate <= values[j]:
                        is_swing = False
                        break
                if is_swing:
                    for j in range(i + 1, i + right_bars + 1):
                        if np.isnan(values[j]) or candidate <= values[j]:
                            is_swing = False
                            break
            else:
                for j in range(i - left_bars, i):
                    if np.isnan(values[j]) or candidate >= values[j]:
                        is_swing = False
                        break
                if is_swing:
                    for j in range(i + 1, i + right_bars + 1):
                        if np.isnan(values[j]) or candidate >= values[j]:
                            is_swing = False
                            break

            if is_swing:
                swings[i] = candidate

        return pd.Series(swings, index=series.index)

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        left_bars: int = params["left_bars"]
        right_bars: int = params["right_bars"]

        min_rows = left_bars + right_bars + 1
        if len(data) < min_rows:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Insufficient data: need at least {min_rows} rows, "
                       f"got {len(data)}",
            )

        high = data["high"]
        low = data["low"]

        swing_highs = self._find_swings(high, left_bars, right_bars, find_highs=True)
        swing_lows = self._find_swings(low, left_bars, right_bars, find_highs=False)

        sh_values = swing_highs.dropna()
        is_hh = pd.Series(False, index=data.index)
        is_lh = pd.Series(False, index=data.index)

        prev_sh = np.nan
        for idx, val in sh_values.items():
            if not np.isnan(prev_sh):
                if val > prev_sh:
                    is_hh.at[idx] = True
                else:
                    is_lh.at[idx] = True
            prev_sh = val

        sl_values = swing_lows.dropna()
        is_hl = pd.Series(False, index=data.index)
        is_ll = pd.Series(False, index=data.index)

        prev_sl = np.nan
        for idx, val in sl_values.items():
            if not np.isnan(prev_sl):
                if val > prev_sl:
                    is_hl.at[idx] = True
                else:
                    is_ll.at[idx] = True
            prev_sl = val

        trend = pd.Series("neutral", index=data.index, dtype="object")

        for idx in data.index:
            if is_hh.at[idx] or is_hl.at[idx]:
                trend.at[idx] = "bullish"
            elif is_lh.at[idx] or is_ll.at[idx]:
                trend.at[idx] = "bearish"

        trend = trend.replace("neutral", np.nan)
        trend = trend.ffill().fillna("neutral")

        result = pd.DataFrame(
            {
                "hhll_swing_high": swing_highs,
                "hhll_swing_low": swing_lows,
                "hhll_is_hh": is_hh,
                "hhll_is_lh": is_lh,
                "hhll_is_hl": is_hl,
                "hhll_is_ll": is_ll,
                "hhll_trend": trend,
            },
            index=data.index,
        )

        return result
