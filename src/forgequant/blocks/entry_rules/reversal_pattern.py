"""
Reversal Pattern entry rule block.

Detects common candlestick reversal patterns:
    1. Engulfing (Bullish & Bearish)
    2. Pin Bar / Hammer / Shooting Star
    3. Morning Star / Evening Star (3-bar patterns)
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
class ReversalPatternEntry(BaseBlock):
    """Candlestick reversal pattern entry signals."""

    metadata = BlockMetadata(
        name="reversal_pattern_entry",
        display_name="Reversal Pattern Entry",
        category=BlockCategory.ENTRY_RULE,
        description=(
            "Detects candlestick reversal patterns including Engulfing, "
            "Pin Bar (Hammer / Shooting Star), and Star patterns "
            "(Morning Star / Evening Star). Signals fire on the "
            "completing bar of each pattern."
        ),
        parameters=(
            ParameterSpec(
                name="pin_bar_ratio",
                param_type="float",
                default=2.0,
                min_value=1.0,
                max_value=10.0,
                description=(
                    "For pin bars: the dominant shadow must be at least "
                    "this many times the body size"
                ),
            ),
            ParameterSpec(
                name="max_opposite_wick_ratio",
                param_type="float",
                default=0.5,
                min_value=0.0,
                max_value=2.0,
                description=(
                    "For pin bars: the opposite shadow must be no more "
                    "than this fraction of the dominant shadow"
                ),
            ),
            ParameterSpec(
                name="star_body_pct",
                param_type="float",
                default=30.0,
                min_value=5.0,
                max_value=80.0,
                description=(
                    "For star patterns: the middle bar's body must be no "
                    "more than this percentage of the first bar's body"
                ),
            ),
        ),
        tags=(
            "reversal", "candlestick", "engulfing", "pin_bar", "hammer",
            "shooting_star", "morning_star", "evening_star", "entry",
        ),
        typical_use=(
            "Used to time entries at potential reversal points. Best "
            "combined with support/resistance levels (enter on a bullish "
            "engulfing at support) or with an oversold/overbought filter "
            "(enter on hammer when RSI is oversold)."
        ),
    )

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        pin_bar_ratio: float = params["pin_bar_ratio"]
        max_opp_ratio: float = params["max_opposite_wick_ratio"]
        star_body_pct: float = params["star_body_pct"]

        if len(data) < 3:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Insufficient data: need at least 3 rows, "
                       f"got {len(data)}",
            )

        o = data["open"].values
        h = data["high"].values
        l = data["low"].values
        c = data["close"].values
        n = len(data)

        bull_engulf = np.zeros(n, dtype=bool)
        bear_engulf = np.zeros(n, dtype=bool)
        hammer = np.zeros(n, dtype=bool)
        shooting_star = np.zeros(n, dtype=bool)
        morning_star = np.zeros(n, dtype=bool)
        evening_star = np.zeros(n, dtype=bool)

        for i in range(1, n):
            body = abs(c[i] - o[i])
            upper_shadow = h[i] - max(o[i], c[i])
            lower_shadow = min(o[i], c[i]) - l[i]
            is_bullish = c[i] > o[i]
            is_bearish = c[i] < o[i]

            prev_body = abs(c[i - 1] - o[i - 1])
            prev_bullish = c[i - 1] > o[i - 1]
            prev_bearish = c[i - 1] < o[i - 1]

            if is_bullish and prev_bearish:
                curr_body_low = min(o[i], c[i])
                curr_body_high = max(o[i], c[i])
                prev_body_low = min(o[i - 1], c[i - 1])
                prev_body_high = max(o[i - 1], c[i - 1])
                if curr_body_low <= prev_body_low and curr_body_high >= prev_body_high:
                    if body > 0:
                        bull_engulf[i] = True

            if is_bearish and prev_bullish:
                curr_body_low = min(o[i], c[i])
                curr_body_high = max(o[i], c[i])
                prev_body_low = min(o[i - 1], c[i - 1])
                prev_body_high = max(o[i - 1], c[i - 1])
                if curr_body_low <= prev_body_low and curr_body_high >= prev_body_high:
                    if body > 0:
                        bear_engulf[i] = True

            if body > 0 and lower_shadow >= pin_bar_ratio * body:
                if upper_shadow <= max_opp_ratio * lower_shadow:
                    hammer[i] = True

            if body > 0 and upper_shadow >= pin_bar_ratio * body:
                if lower_shadow <= max_opp_ratio * upper_shadow:
                    shooting_star[i] = True

            if i >= 2:
                bar_m2_body = abs(c[i - 2] - o[i - 2])
                bar_m1_body = abs(c[i - 1] - o[i - 1])
                bar_m2_bullish = c[i - 2] > o[i - 2]
                bar_m2_bearish = c[i - 2] < o[i - 2]
                bar_m2_midpoint = (o[i - 2] + c[i - 2]) / 2.0

                small_middle = (
                    bar_m2_body > 0
                    and bar_m1_body <= (star_body_pct / 100.0) * bar_m2_body
                )

                if small_middle and bar_m2_bearish and is_bullish and c[i] > bar_m2_midpoint:
                    morning_star[i] = True

                if small_middle and bar_m2_bullish and is_bearish and c[i] < bar_m2_midpoint:
                    evening_star[i] = True

        long_entry = bull_engulf | hammer | morning_star
        short_entry = bear_engulf | shooting_star | evening_star

        result = pd.DataFrame(
            {
                "reversal_bullish_engulfing": bull_engulf,
                "reversal_bearish_engulfing": bear_engulf,
                "reversal_hammer": hammer,
                "reversal_shooting_star": shooting_star,
                "reversal_morning_star": morning_star,
                "reversal_evening_star": evening_star,
                SC.reversal_long_entry: long_entry,
                SC.reversal_short_entry: short_entry,
            },
            index=data.index,
        )

        return result
