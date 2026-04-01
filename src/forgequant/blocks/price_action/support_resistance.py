"""
Support and Resistance price action block.

Identifies horizontal support and resistance zones using a pivot-point
approach. Swing highs become resistance levels and swing lows become
support levels. Nearby levels are merged into zones using a configurable
tolerance.
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
class SupportResistanceBlock(BaseBlock):
    """Horizontal support and resistance zone identification."""

    metadata = BlockMetadata(
        name="support_resistance",
        display_name="Support & Resistance",
        category=BlockCategory.PRICE_ACTION,
        description=(
            "Identifies horizontal support and resistance zones by clustering "
            "swing highs and swing lows. Nearby swing points are merged into "
            "zones, and each zone's strength is measured by its touch count."
        ),
        parameters=(
            ParameterSpec(
                name="left_bars",
                param_type="int",
                default=5,
                min_value=1,
                max_value=50,
                description="Left confirmation bars for swing detection",
            ),
            ParameterSpec(
                name="right_bars",
                param_type="int",
                default=5,
                min_value=1,
                max_value=50,
                description="Right confirmation bars for swing detection",
            ),
            ParameterSpec(
                name="merge_pct",
                param_type="float",
                default=0.5,
                min_value=0.01,
                max_value=5.0,
                description=(
                    "Percentage tolerance for merging nearby levels into zones. "
                    "Levels within merge_pct% of each other are combined."
                ),
            ),
            ParameterSpec(
                name="max_zones",
                param_type="int",
                default=20,
                min_value=2,
                max_value=100,
                description="Maximum number of zones to retain (by strength)",
            ),
        ),
        tags=("support", "resistance", "zones", "levels", "pivot", "horizontal"),
        typical_use=(
            "Used to identify key price levels for entries, exits, and stop "
            "placement. Strong zones (high touch count) are more reliable. "
            "Combine with breakout or pullback blocks for entry timing."
        ),
    )

    @staticmethod
    def _find_swing_levels(
        high: pd.Series,
        low: pd.Series,
        left_bars: int,
        right_bars: int,
    ) -> tuple[list[float], list[float]]:
        h_vals = high.values
        l_vals = low.values
        n = len(h_vals)
        resistances: list[float] = []
        supports: list[float] = []

        for i in range(left_bars, n - right_bars):
            h_candidate = h_vals[i]
            if not np.isnan(h_candidate):
                is_sh = True
                for j in range(i - left_bars, i):
                    if np.isnan(h_vals[j]) or h_candidate <= h_vals[j]:
                        is_sh = False
                        break
                if is_sh:
                    for j in range(i + 1, i + right_bars + 1):
                        if np.isnan(h_vals[j]) or h_candidate <= h_vals[j]:
                            is_sh = False
                            break
                if is_sh:
                    resistances.append(float(h_candidate))

            l_candidate = l_vals[i]
            if not np.isnan(l_candidate):
                is_sl = True
                for j in range(i - left_bars, i):
                    if np.isnan(l_vals[j]) or l_candidate >= l_vals[j]:
                        is_sl = False
                        break
                if is_sl:
                    for j in range(i + 1, i + right_bars + 1):
                        if np.isnan(l_vals[j]) or l_candidate >= l_vals[j]:
                            is_sl = False
                            break
                if is_sl:
                    supports.append(float(l_candidate))

        return resistances, supports

    @staticmethod
    def _merge_levels(
        levels: list[float],
        merge_pct: float,
    ) -> list[tuple[float, int]]:
        if not levels:
            return []

        sorted_levels = sorted(levels)
        zones: list[tuple[float, int]] = []
        current_group: list[float] = [sorted_levels[0]]

        for i in range(1, len(sorted_levels)):
            group_mean = sum(current_group) / len(current_group)
            pct_diff = abs(sorted_levels[i] - group_mean) / group_mean * 100.0

            if pct_diff <= merge_pct:
                current_group.append(sorted_levels[i])
            else:
                zone_price = sum(current_group) / len(current_group)
                zones.append((zone_price, len(current_group)))
                current_group = [sorted_levels[i]]

        zone_price = sum(current_group) / len(current_group)
        zones.append((zone_price, len(current_group)))

        return zones

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        left_bars: int = params["left_bars"]
        right_bars: int = params["right_bars"]
        merge_pct: float = params["merge_pct"]
        max_zones: int = params["max_zones"]

        min_rows = left_bars + right_bars + 1
        if len(data) < min_rows:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Insufficient data: need at least {min_rows} rows, "
                       f"got {len(data)}",
            )

        high = data["high"]
        low = data["low"]
        close = data["close"]

        res_levels, sup_levels = self._find_swing_levels(
            high, low, left_bars, right_bars
        )

        res_zones = self._merge_levels(res_levels, merge_pct)
        sup_zones = self._merge_levels(sup_levels, merge_pct)

        res_zones = sorted(res_zones, key=lambda z: z[1], reverse=True)[:max_zones]
        sup_zones = sorted(sup_zones, key=lambda z: z[1], reverse=True)[:max_zones]

        res_zones = sorted(res_zones, key=lambda z: z[0])
        sup_zones = sorted(sup_zones, key=lambda z: z[0])

        res_prices = [z[0] for z in res_zones]
        res_strengths = {z[0]: z[1] for z in res_zones}
        sup_prices = [z[0] for z in sup_zones]
        sup_strengths = {z[0]: z[1] for z in sup_zones}

        n = len(data)
        nearest_sup = np.full(n, np.nan)
        nearest_res = np.full(n, np.nan)
        sup_str = np.full(n, np.nan)
        res_str = np.full(n, np.nan)

        close_vals = close.values

        for i in range(n):
            c = close_vals[i]
            if np.isnan(c):
                continue

            for p in reversed(sup_prices):
                if p < c:
                    nearest_sup[i] = p
                    sup_str[i] = sup_strengths[p]
                    break

            for p in res_prices:
                if p > c:
                    nearest_res[i] = p
                    res_str[i] = res_strengths[p]
                    break

        close_arr = close.values
        dist_sup = np.where(
            ~np.isnan(nearest_sup),
            (close_arr - nearest_sup) / close_arr * 100.0,
            np.nan,
        )
        dist_res = np.where(
            ~np.isnan(nearest_res),
            (nearest_res - close_arr) / close_arr * 100.0,
            np.nan,
        )

        result = pd.DataFrame(
            {
                "sr_nearest_support": nearest_sup,
                "sr_nearest_resistance": nearest_res,
                "sr_support_strength": sup_str,
                "sr_resistance_strength": res_str,
                "sr_distance_to_support_pct": dist_sup,
                "sr_distance_to_resistance_pct": dist_res,
            },
            index=data.index,
        )

        return result
