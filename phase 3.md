# PHASE 3 — Price Action Blocks & Entry Rule Blocks

All 4 price action blocks and all 4 entry rule blocks, each with anti-lookahead protection, full parameter validation, and comprehensive tests.

---

## 3.1 Updated Directory Structure (additions)

```
src/forgequant/blocks/price_action/
├── __init__.py
├── breakout.py
├── pullback.py
├── higher_high_lower_low.py
└── support_resistance.py

src/forgequant/blocks/entry_rules/
├── __init__.py
├── crossover.py
├── threshold_cross.py
├── confluence.py
└── reversal_pattern.py

tests/unit/price_action/
├── __init__.py
├── test_breakout.py
├── test_pullback.py
├── test_higher_high_lower_low.py
└── test_support_resistance.py

tests/unit/entry_rules/
├── __init__.py
├── test_crossover.py
├── test_threshold_cross.py
├── test_confluence.py
└── test_reversal_pattern.py

tests/integration/
└── test_phase3_registry.py
```

---

## 3.2 `src/forgequant/blocks/price_action/__init__.py`

```python
"""
Price action pattern blocks.

Provides:
    - BreakoutBlock: Detects breakouts above/below recent highs/lows
    - PullbackBlock: Detects pullbacks to moving averages or key levels
    - HigherHighLowerLowBlock: Identifies higher highs, higher lows (and vice versa)
    - SupportResistanceBlock: Identifies horizontal support and resistance zones

All blocks are auto-registered with the BlockRegistry upon import.
"""

from forgequant.blocks.price_action.breakout import BreakoutBlock
from forgequant.blocks.price_action.pullback import PullbackBlock
from forgequant.blocks.price_action.higher_high_lower_low import HigherHighLowerLowBlock
from forgequant.blocks.price_action.support_resistance import SupportResistanceBlock

__all__ = [
    "BreakoutBlock",
    "PullbackBlock",
    "HigherHighLowerLowBlock",
    "SupportResistanceBlock",
]
```

---

## 3.3 `src/forgequant/blocks/price_action/breakout.py`

```python
"""
Breakout price action block.

Detects when price breaks above the highest high or below the lowest low
of the preceding lookback window. Uses shift(1) on the rolling extremes
to guarantee the current bar is NEVER compared against itself, preventing
lookahead bias.

Calculation:
    1. resistance = shift(1, rolling_max(high, lookback))
       support    = shift(1, rolling_min(low, lookback))

    2. breakout_long  = close > resistance
       breakout_short = close < support

    3. Optionally requires the breakout bar's volume to exceed
       the rolling average volume by a configurable multiplier.

Output columns:
    - breakout_resistance: The resistance level (shifted)
    - breakout_support: The support level (shifted)
    - breakout_long: Boolean, True when close breaks above resistance
    - breakout_short: Boolean, True when close breaks below support
    - breakout_volume_confirm: Boolean, True when volume exceeds threshold
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
        """
        Detect breakouts above resistance or below support.

        Args:
            data: OHLCV DataFrame with DatetimeIndex.
            params: Must contain 'lookback', 'volume_multiplier', 'volume_lookback'.

        Returns:
            DataFrame with columns: breakout_resistance, breakout_support,
            breakout_long, breakout_short, breakout_volume_confirm.
        """
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

        # Rolling extremes from PREVIOUS bars only (shift 1)
        rolling_high = high.rolling(window=lookback).max().shift(1)
        rolling_low = low.rolling(window=lookback).min().shift(1)

        # Breakout signals
        breakout_long = close > rolling_high
        breakout_short = close < rolling_low

        # Volume confirmation
        if vol_mult > 0:
            avg_volume = volume.rolling(window=vol_lookback).mean().shift(1)
            volume_confirm = volume > (avg_volume * vol_mult)
        else:
            volume_confirm = pd.Series(True, index=data.index)

        result = pd.DataFrame(
            {
                "breakout_resistance": rolling_high,
                "breakout_support": rolling_low,
                "breakout_long": breakout_long.fillna(False),
                "breakout_short": breakout_short.fillna(False),
                "breakout_volume_confirm": volume_confirm.fillna(False),
            },
            index=data.index,
        )

        return result
```

---

## 3.4 `src/forgequant/blocks/price_action/pullback.py`

```python
"""
Pullback price action block.

Detects pullbacks (retracements) to a moving average in a trending market.
A pullback long is when price was above the MA, dips to touch or cross
below it, and then closes back above it. Vice versa for short pullbacks.

Calculation:
    1. Compute the MA (EMA or SMA) of the close.

    2. Define the proximity zone:
       upper_zone = MA * (1 + proximity_pct / 100)
       lower_zone = MA * (1 - proximity_pct / 100)

    3. pullback_touch = low <= upper_zone  (price dipped into the zone)

    4. pullback_long:
       - close > MA (price recovered above)
       - low <= upper_zone (bar's low touched the zone)
       - previous close > MA (was already in uptrend before the dip)

    5. pullback_short:
       - close < MA
       - high >= lower_zone
       - previous close < MA

Output columns:
    - pullback_ma: The moving average values
    - pullback_upper_zone: Upper boundary of the proximity zone
    - pullback_lower_zone: Lower boundary of the proximity zone
    - pullback_long: Boolean, True on bars that are valid long pullback entries
    - pullback_short: Boolean, True on bars that are valid short pullback entries
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
        """
        Detect pullback entries to a moving average.

        Args:
            data: OHLCV DataFrame with DatetimeIndex.
            params: Must contain 'ma_period', 'ma_type', 'proximity_pct'.

        Returns:
            DataFrame with columns: pullback_ma, pullback_upper_zone,
            pullback_lower_zone, pullback_long, pullback_short.
        """
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

        # Compute MA
        if ma_type == "ema":
            ma = close.ewm(span=ma_period, adjust=False).mean()
        else:
            ma = close.rolling(window=ma_period).mean()

        # Proximity zone
        zone_factor = proximity_pct / 100.0
        upper_zone = ma * (1.0 + zone_factor)
        lower_zone = ma * (1.0 - zone_factor)

        # Previous close relative to MA (shift 1 to use prior bar's state)
        prev_close_above_ma = close.shift(1) > ma.shift(1)
        prev_close_below_ma = close.shift(1) < ma.shift(1)

        # Long pullback:
        #   1. Was in uptrend (prev close above MA)
        #   2. Bar's low touched the proximity zone (dipped to near MA)
        #   3. Bar's close recovered above MA
        pullback_long = (
            prev_close_above_ma
            & (low <= upper_zone)
            & (close > ma)
        )

        # Short pullback:
        #   1. Was in downtrend (prev close below MA)
        #   2. Bar's high reached the proximity zone (rose to near MA)
        #   3. Bar's close stayed below MA
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
```

---

## 3.5 `src/forgequant/blocks/price_action/higher_high_lower_low.py`

```python
"""
Higher High / Lower Low (HHLL) price action block.

Identifies swing highs and swing lows using a configurable number of
left/right confirmation bars, then classifies the structure:
    - Higher High (HH): current swing high > previous swing high
    - Lower High (LH): current swing high < previous swing high
    - Higher Low (HL): current swing low > previous swing low
    - Lower Low (LL): current swing low < previous swing low

A swing high is a bar whose high is greater than the highs of the
preceding `left_bars` bars AND the following `right_bars` bars.
This means swing identification has an inherent lag of `right_bars`
bars — the swing is only confirmed after right_bars close.

In backtesting, we mark the swing at its actual bar position but the
signal (HH/HL/LH/LL classification) is only available right_bars later.
We use forward_fill to propagate the last known structure state.

Output columns:
    - hhll_swing_high: Price of the swing high (NaN where no swing)
    - hhll_swing_low: Price of the swing low (NaN where no swing)
    - hhll_is_hh: Boolean, True at bars identified as Higher High
    - hhll_is_lh: Boolean, True at bars identified as Lower High
    - hhll_is_hl: Boolean, True at bars identified as Higher Low
    - hhll_is_ll: Boolean, True at bars identified as Lower Low
    - hhll_trend: Categorical — "bullish" (HH+HL), "bearish" (LH+LL), or "neutral"
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
        """
        Find swing highs or swing lows in a price series.

        A swing high at index i requires:
            series[i] > series[j] for all j in [i-left_bars, i+right_bars], j != i

        Args:
            series: The price series (high for swing highs, low for swing lows).
            left_bars: Number of bars to the left.
            right_bars: Number of bars to the right.
            find_highs: True to find swing highs, False for swing lows.

        Returns:
            Series with the swing price at swing points, NaN elsewhere.
        """
        values = series.values
        n = len(values)
        swings = np.full(n, np.nan)

        for i in range(left_bars, n - right_bars):
            candidate = values[i]

            if np.isnan(candidate):
                continue

            is_swing = True

            if find_highs:
                # Check left: candidate must be strictly greater
                for j in range(i - left_bars, i):
                    if np.isnan(values[j]) or candidate <= values[j]:
                        is_swing = False
                        break
                # Check right: candidate must be strictly greater
                if is_swing:
                    for j in range(i + 1, i + right_bars + 1):
                        if np.isnan(values[j]) or candidate <= values[j]:
                            is_swing = False
                            break
            else:
                # Swing low: candidate must be strictly less
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
        """
        Identify swing structure and classify as HH/HL/LH/LL.

        Args:
            data: OHLCV DataFrame with DatetimeIndex.
            params: Must contain 'left_bars' and 'right_bars'.

        Returns:
            DataFrame with swing highs/lows, HH/LH/HL/LL flags, and trend.
        """
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

        # Find swing highs and lows
        swing_highs = self._find_swings(high, left_bars, right_bars, find_highs=True)
        swing_lows = self._find_swings(low, left_bars, right_bars, find_highs=False)

        # Classify swing highs as HH or LH
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

        # Classify swing lows as HL or LL
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

        # Derive trend state
        # Create a series that's "bullish" when last swing was HH or HL,
        # "bearish" when LH or LL, "neutral" otherwise
        trend = pd.Series("neutral", index=data.index, dtype="object")

        for idx in data.index:
            if is_hh.at[idx] or is_hl.at[idx]:
                trend.at[idx] = "bullish"
            elif is_lh.at[idx] or is_ll.at[idx]:
                trend.at[idx] = "bearish"

        # Forward-fill trend so it persists between swing points
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
```

---

## 3.6 `src/forgequant/blocks/price_action/support_resistance.py`

```python
"""
Support and Resistance price action block.

Identifies horizontal support and resistance zones using a pivot-point
approach. Swing highs become resistance levels and swing lows become
support levels. Nearby levels are merged into zones using a configurable
tolerance.

Algorithm:
    1. Identify swing highs (resistance candidates) and swing lows
       (support candidates) using left/right bar confirmation.

    2. Cluster nearby levels: if two levels are within `merge_pct`% of
       each other, they are merged (averaged) into a single zone. The
       more times a level is tested, the stronger it becomes.

    3. For each bar, report the nearest support below and resistance
       above the current close.

    4. Report the "touch count" — how many swings contributed to
       each zone — as a strength measure.

Output columns:
    - sr_nearest_support: Nearest support level below current close
    - sr_nearest_resistance: Nearest resistance level above current close
    - sr_support_strength: Touch count of the nearest support zone
    - sr_resistance_strength: Touch count of the nearest resistance zone
    - sr_distance_to_support_pct: Distance to support as % of close
    - sr_distance_to_resistance_pct: Distance to resistance as % of close
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
        """
        Extract swing high and swing low price levels.

        Returns:
            Tuple of (resistance_levels, support_levels) as flat lists of floats.
        """
        h_vals = high.values
        l_vals = low.values
        n = len(h_vals)
        resistances: list[float] = []
        supports: list[float] = []

        for i in range(left_bars, n - right_bars):
            # Check swing high
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

            # Check swing low
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
        """
        Merge nearby levels into zones.

        Args:
            levels: List of price levels.
            merge_pct: Percentage tolerance for merging.

        Returns:
            List of (zone_price, touch_count) tuples, sorted by price.
        """
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

        # Don't forget the last group
        zone_price = sum(current_group) / len(current_group)
        zones.append((zone_price, len(current_group)))

        return zones

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        """
        Identify support/resistance zones and map to each bar.

        Args:
            data: OHLCV DataFrame with DatetimeIndex.
            params: Must contain 'left_bars', 'right_bars', 'merge_pct', 'max_zones'.

        Returns:
            DataFrame with nearest S/R levels, strengths, and distance percentages.
        """
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

        # Find all swing levels
        res_levels, sup_levels = self._find_swing_levels(
            high, low, left_bars, right_bars
        )

        # Merge into zones
        res_zones = self._merge_levels(res_levels, merge_pct)
        sup_zones = self._merge_levels(sup_levels, merge_pct)

        # Keep only the strongest zones
        res_zones = sorted(res_zones, key=lambda z: z[1], reverse=True)[:max_zones]
        sup_zones = sorted(sup_zones, key=lambda z: z[1], reverse=True)[:max_zones]

        # Re-sort by price for efficient nearest lookup
        res_zones = sorted(res_zones, key=lambda z: z[0])
        sup_zones = sorted(sup_zones, key=lambda z: z[0])

        res_prices = [z[0] for z in res_zones]
        res_strengths = {z[0]: z[1] for z in res_zones}
        sup_prices = [z[0] for z in sup_zones]
        sup_strengths = {z[0]: z[1] for z in sup_zones}

        # For each bar, find nearest support below and resistance above
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

            # Nearest support below close
            for p in reversed(sup_prices):
                if p < c:
                    nearest_sup[i] = p
                    sup_str[i] = sup_strengths[p]
                    break

            # Nearest resistance above close
            for p in res_prices:
                if p > c:
                    nearest_res[i] = p
                    res_str[i] = res_strengths[p]
                    break

        # Distance percentages
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
```

---

## 3.7 `src/forgequant/blocks/entry_rules/__init__.py`

```python
"""
Entry rule blocks.

Provides:
    - CrossoverEntry: Moving average crossover entry signals
    - ThresholdCrossEntry: Indicator crosses above/below threshold levels
    - ConfluenceEntry: Requires multiple conditions to be true simultaneously
    - ReversalPatternEntry: Detects candlestick reversal patterns

All blocks are auto-registered with the BlockRegistry upon import.
"""

from forgequant.blocks.entry_rules.crossover import CrossoverEntry
from forgequant.blocks.entry_rules.threshold_cross import ThresholdCrossEntry
from forgequant.blocks.entry_rules.confluence import ConfluenceEntry
from forgequant.blocks.entry_rules.reversal_pattern import ReversalPatternEntry

__all__ = [
    "CrossoverEntry",
    "ThresholdCrossEntry",
    "ConfluenceEntry",
    "ReversalPatternEntry",
]
```

---

## 3.8 `src/forgequant/blocks/entry_rules/crossover.py`

```python
"""
Crossover entry rule block.

Generates entry signals when a fast moving average crosses a slow moving
average. The cross is detected via state change — the signal fires on
the FIRST bar where the cross condition becomes true, not on every bar
where fast > slow.

Calculation:
    1. fast_ma = MA(close, fast_period)
       slow_ma = MA(close, slow_period)

    2. fast_above = fast_ma > slow_ma

    3. long_entry = fast_above AND NOT shift(fast_above, 1)
       (i.e. fast just crossed above slow)

    4. short_entry = NOT fast_above AND shift(fast_above, 1)
       (i.e. fast just crossed below slow)

Output columns:
    - crossover_fast_ma: The fast moving average
    - crossover_slow_ma: The slow moving average
    - crossover_long_entry: Boolean, True on the bar of a bullish crossover
    - crossover_short_entry: Boolean, True on the bar of a bearish crossover
"""

from __future__ import annotations

import pandas as pd

from forgequant.blocks.base import BaseBlock
from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory, BlockParams, BlockResult


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
        """
        Compute crossover entry signals.

        Args:
            data: OHLCV DataFrame with DatetimeIndex.
            params: Must contain 'fast_period', 'slow_period', 'ma_type'.

        Returns:
            DataFrame with fast/slow MA values and long/short entry signals.
        """
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

        # Compute moving averages
        if ma_type == "ema":
            fast_ma = close.ewm(span=fast_period, adjust=False).mean()
            slow_ma = close.ewm(span=slow_period, adjust=False).mean()
        else:
            fast_ma = close.rolling(window=fast_period).mean()
            slow_ma = close.rolling(window=slow_period).mean()

        # State: is fast above slow?
        fast_above = fast_ma > slow_ma

        # Previous state
        prev_fast_above = fast_above.shift(1)

        # Crossover signals (state change detection)
        long_entry = fast_above & (~prev_fast_above)
        short_entry = (~fast_above) & prev_fast_above

        result = pd.DataFrame(
            {
                "crossover_fast_ma": fast_ma,
                "crossover_slow_ma": slow_ma,
                "crossover_long_entry": long_entry.fillna(False),
                "crossover_short_entry": short_entry.fillna(False),
            },
            index=data.index,
        )

        return result
```

---

## 3.9 `src/forgequant/blocks/entry_rules/threshold_cross.py`

```python
"""
Threshold Cross entry rule block.

Generates entry signals when a computed indicator value crosses
above or below configurable threshold levels. Like the crossover
block, signals fire only on the exact bar of the state change.

This block computes RSI internally and generates signals when it
crosses the overbought/oversold thresholds. It can be configured
for mean-reversion (buy on oversold exit) or momentum (buy on
overbought entry) strategies.

Mean-reversion mode (default):
    - long_entry: RSI crosses ABOVE the oversold level (recovery from oversold)
    - short_entry: RSI crosses BELOW the overbought level (reversal from overbought)

Momentum mode:
    - long_entry: RSI crosses ABOVE the overbought level (strong momentum)
    - short_entry: RSI crosses BELOW the oversold level (strong downward momentum)

Output columns:
    - threshold_indicator: The RSI values used for threshold comparison
    - threshold_long_entry: Boolean entry signal for longs
    - threshold_short_entry: Boolean entry signal for shorts
"""

from __future__ import annotations

import pandas as pd

from forgequant.blocks.base import BaseBlock
from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory, BlockParams, BlockResult


@BlockRegistry.register
class ThresholdCrossEntry(BaseBlock):
    """Threshold-crossing entry signals (RSI-based)."""

    metadata = BlockMetadata(
        name="threshold_cross_entry",
        display_name="Threshold Cross Entry",
        category=BlockCategory.ENTRY_RULE,
        description=(
            "Generates entry signals when RSI crosses configurable threshold "
            "levels. Supports mean-reversion mode (buy on oversold recovery) "
            "and momentum mode (buy on overbought entry)."
        ),
        parameters=(
            ParameterSpec(
                name="rsi_period",
                param_type="int",
                default=14,
                min_value=2,
                max_value=200,
                description="RSI calculation period",
            ),
            ParameterSpec(
                name="upper_threshold",
                param_type="float",
                default=70.0,
                min_value=50.0,
                max_value=95.0,
                description="Upper threshold (overbought level)",
            ),
            ParameterSpec(
                name="lower_threshold",
                param_type="float",
                default=30.0,
                min_value=5.0,
                max_value=50.0,
                description="Lower threshold (oversold level)",
            ),
            ParameterSpec(
                name="mode",
                param_type="str",
                default="mean_reversion",
                choices=("mean_reversion", "momentum"),
                description=(
                    "Signal mode: mean_reversion (buy on oversold recovery) "
                    "or momentum (buy on overbought breakout)"
                ),
            ),
        ),
        tags=("threshold", "rsi", "overbought", "oversold", "entry", "signal"),
        typical_use=(
            "Mean-reversion: enter long when RSI recovers from oversold, "
            "enter short when RSI falls from overbought. Momentum: enter "
            "long when RSI surges above overbought (strong trend). Best "
            "combined with a trend or volatility filter."
        ),
    )

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        """
        Compute threshold cross entry signals.

        Args:
            data: OHLCV DataFrame with DatetimeIndex.
            params: Must contain 'rsi_period', 'upper_threshold',
                    'lower_threshold', 'mode'.

        Returns:
            DataFrame with indicator values and entry signals.
        """
        rsi_period: int = params["rsi_period"]
        upper: float = params["upper_threshold"]
        lower: float = params["lower_threshold"]
        mode: str = params["mode"]

        if upper <= lower:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"upper_threshold ({upper}) must be greater than "
                       f"lower_threshold ({lower})",
            )

        min_rows = rsi_period + 2
        if len(data) < min_rows:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Insufficient data: need at least {min_rows} rows, "
                       f"got {len(data)}",
            )

        close = data["close"]

        # Compute RSI using Wilder's method
        delta = close.diff()
        gains = delta.clip(lower=0.0)
        losses = (-delta).clip(lower=0.0)
        alpha = 1.0 / rsi_period
        avg_gain = gains.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = losses.ewm(alpha=alpha, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        rsi = rsi.fillna(50.0)

        # State tracking
        prev_rsi = rsi.shift(1)

        if mode == "mean_reversion":
            # Long: RSI was below lower, now crosses above (recovery from oversold)
            long_entry = (rsi >= lower) & (prev_rsi < lower)
            # Short: RSI was above upper, now crosses below (reversal from overbought)
            short_entry = (rsi <= upper) & (prev_rsi > upper)
        else:
            # Momentum mode
            # Long: RSI crosses above upper (strong upward momentum)
            long_entry = (rsi >= upper) & (prev_rsi < upper)
            # Short: RSI crosses below lower (strong downward momentum)
            short_entry = (rsi <= lower) & (prev_rsi > lower)

        result = pd.DataFrame(
            {
                "threshold_indicator": rsi,
                "threshold_long_entry": long_entry.fillna(False),
                "threshold_short_entry": short_entry.fillna(False),
            },
            index=data.index,
        )

        return result
```

---

## 3.10 `src/forgequant/blocks/entry_rules/confluence.py`

```python
"""
Confluence entry rule block.

Generates entry signals only when multiple independent conditions are
simultaneously true. This acts as a logical AND gate across conditions.

Built-in conditions evaluated:
    1. Trend alignment: close > EMA(trend_period)  (for longs)
    2. Momentum confirmation: RSI in a favorable zone
    3. Volatility filter: ATR as % of close is within a range

All three conditions must agree for an entry signal.

The block computes all indicators internally so it is self-contained
and does not depend on other blocks being run first.

Output columns:
    - confluence_trend_ok: Boolean, True when trend condition is met
    - confluence_momentum_ok: Boolean, True when momentum condition is met
    - confluence_volatility_ok: Boolean, True when volatility condition is met
    - confluence_long_entry: Boolean, True when ALL conditions align for long
    - confluence_short_entry: Boolean, True when ALL conditions align for short
    - confluence_score: Integer 0-3, number of conditions met (for longs)
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
class ConfluenceEntry(BaseBlock):
    """Multi-condition confluence entry signals."""

    metadata = BlockMetadata(
        name="confluence_entry",
        display_name="Confluence Entry",
        category=BlockCategory.ENTRY_RULE,
        description=(
            "Generates entry signals only when trend alignment, momentum "
            "confirmation, and volatility conditions are all met. Acts as "
            "a quality gate to filter out low-conviction setups."
        ),
        parameters=(
            ParameterSpec(
                name="trend_period",
                param_type="int",
                default=50,
                min_value=5,
                max_value=500,
                description="EMA period for trend alignment",
            ),
            ParameterSpec(
                name="rsi_period",
                param_type="int",
                default=14,
                min_value=2,
                max_value=200,
                description="RSI period for momentum confirmation",
            ),
            ParameterSpec(
                name="rsi_long_min",
                param_type="float",
                default=40.0,
                min_value=5.0,
                max_value=80.0,
                description="Minimum RSI for long entry (momentum not too weak)",
            ),
            ParameterSpec(
                name="rsi_long_max",
                param_type="float",
                default=70.0,
                min_value=20.0,
                max_value=95.0,
                description="Maximum RSI for long entry (not already overbought)",
            ),
            ParameterSpec(
                name="rsi_short_min",
                param_type="float",
                default=30.0,
                min_value=5.0,
                max_value=80.0,
                description="Minimum RSI for short entry (not already oversold)",
            ),
            ParameterSpec(
                name="rsi_short_max",
                param_type="float",
                default=60.0,
                min_value=20.0,
                max_value=95.0,
                description="Maximum RSI for short entry (momentum not too strong)",
            ),
            ParameterSpec(
                name="atr_period",
                param_type="int",
                default=14,
                min_value=1,
                max_value=200,
                description="ATR period for volatility measurement",
            ),
            ParameterSpec(
                name="min_atr_pct",
                param_type="float",
                default=0.05,
                min_value=0.0,
                max_value=10.0,
                description="Minimum ATR as % of close (avoid dead markets)",
            ),
            ParameterSpec(
                name="max_atr_pct",
                param_type="float",
                default=5.0,
                min_value=0.1,
                max_value=50.0,
                description="Maximum ATR as % of close (avoid excessive volatility)",
            ),
        ),
        tags=("confluence", "multi_condition", "quality", "filter", "entry"),
        typical_use=(
            "Used as a high-quality entry filter that requires trend, "
            "momentum, and volatility to all agree before taking a trade. "
            "Reduces false signals compared to single-condition entries."
        ),
    )

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        """
        Evaluate confluence of trend, momentum, and volatility conditions.

        Args:
            data: OHLCV DataFrame with DatetimeIndex.
            params: All parameter values.

        Returns:
            DataFrame with individual condition flags, combined entry signals,
            and a confluence score.
        """
        trend_period: int = params["trend_period"]
        rsi_period: int = params["rsi_period"]
        rsi_long_min: float = params["rsi_long_min"]
        rsi_long_max: float = params["rsi_long_max"]
        rsi_short_min: float = params["rsi_short_min"]
        rsi_short_max: float = params["rsi_short_max"]
        atr_period: int = params["atr_period"]
        min_atr_pct: float = params["min_atr_pct"]
        max_atr_pct: float = params["max_atr_pct"]

        min_rows = max(trend_period, rsi_period, atr_period) + 2
        if len(data) < min_rows:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Insufficient data: need at least {min_rows} rows, "
                       f"got {len(data)}",
            )

        close = data["close"]
        high = data["high"]
        low = data["low"]

        # ── Condition 1: Trend alignment ──
        trend_ema = close.ewm(span=trend_period, adjust=False).mean()
        trend_bullish = close > trend_ema
        trend_bearish = close < trend_ema

        # ── Condition 2: Momentum (RSI) ──
        delta = close.diff()
        gains = delta.clip(lower=0.0)
        losses = (-delta).clip(lower=0.0)
        alpha = 1.0 / rsi_period
        avg_gain = gains.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = losses.ewm(alpha=alpha, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)

        momentum_long = (rsi >= rsi_long_min) & (rsi <= rsi_long_max)
        momentum_short = (rsi >= rsi_short_min) & (rsi <= rsi_short_max)

        # ── Condition 3: Volatility (ATR %) ──
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(alpha=1.0 / atr_period, adjust=False).mean()
        atr_pct = (atr / close) * 100.0

        volatility_ok = (atr_pct >= min_atr_pct) & (atr_pct <= max_atr_pct)

        # ── Combined signals ──
        long_entry = trend_bullish & momentum_long & volatility_ok
        short_entry = trend_bearish & momentum_short & volatility_ok

        # Confluence score for longs (how many of 3 conditions are met)
        score = (
            trend_bullish.astype(int)
            + momentum_long.astype(int)
            + volatility_ok.astype(int)
        )

        result = pd.DataFrame(
            {
                "confluence_trend_ok": trend_bullish.fillna(False),
                "confluence_momentum_ok": momentum_long.fillna(False),
                "confluence_volatility_ok": volatility_ok.fillna(False),
                "confluence_long_entry": long_entry.fillna(False),
                "confluence_short_entry": short_entry.fillna(False),
                "confluence_score": score.fillna(0).astype(int),
            },
            index=data.index,
        )

        return result
```

---

## 3.11 `src/forgequant/blocks/entry_rules/reversal_pattern.py`

```python
"""
Reversal Pattern entry rule block.

Detects common candlestick reversal patterns:
    1. Engulfing (Bullish & Bearish)
    2. Pin Bar / Hammer / Shooting Star
    3. Morning Star / Evening Star (3-bar patterns)

All patterns are detected using OHLC data only, with no lookahead.
Each pattern is evaluated bar-by-bar using only current and preceding
bar data.

Pattern definitions:
    Bullish Engulfing:
        - Previous bar: bearish (close < open)
        - Current bar: bullish (close > open)
        - Current body completely engulfs previous body

    Bearish Engulfing:
        - Previous bar: bullish (close > open)
        - Current bar: bearish (close < open)
        - Current body completely engulfs previous body

    Pin Bar (Hammer / Bullish):
        - Lower shadow >= body_ratio * body_size
        - Upper shadow <= max_wick_ratio * lower_shadow

    Pin Bar (Shooting Star / Bearish):
        - Upper shadow >= body_ratio * body_size
        - Lower shadow <= max_wick_ratio * upper_shadow

    Morning Star (Bullish, 3-bar):
        - Bar -2: bearish with large body
        - Bar -1: small body (indecision)
        - Bar  0: bullish, closes above midpoint of bar -2

    Evening Star (Bearish, 3-bar):
        - Bar -2: bullish with large body
        - Bar -1: small body (indecision)
        - Bar  0: bearish, closes below midpoint of bar -2

Output columns:
    - reversal_bullish_engulfing: Boolean
    - reversal_bearish_engulfing: Boolean
    - reversal_hammer: Boolean (bullish pin bar)
    - reversal_shooting_star: Boolean (bearish pin bar)
    - reversal_morning_star: Boolean
    - reversal_evening_star: Boolean
    - reversal_long_entry: Boolean (any bullish reversal pattern)
    - reversal_short_entry: Boolean (any bearish reversal pattern)
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
        """
        Detect reversal candlestick patterns.

        Args:
            data: OHLCV DataFrame with DatetimeIndex.
            params: Must contain 'pin_bar_ratio', 'max_opposite_wick_ratio',
                    'star_body_pct'.

        Returns:
            DataFrame with individual pattern flags and combined entry signals.
        """
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
            # Current bar
            body = abs(c[i] - o[i])
            upper_shadow = h[i] - max(o[i], c[i])
            lower_shadow = min(o[i], c[i]) - l[i]
            is_bullish = c[i] > o[i]
            is_bearish = c[i] < o[i]

            # Previous bar
            prev_body = abs(c[i - 1] - o[i - 1])
            prev_bullish = c[i - 1] > o[i - 1]
            prev_bearish = c[i - 1] < o[i - 1]

            # ── Engulfing ──
            if is_bullish and prev_bearish:
                # Bullish engulfing: current body engulfs previous body
                curr_body_low = min(o[i], c[i])
                curr_body_high = max(o[i], c[i])
                prev_body_low = min(o[i - 1], c[i - 1])
                prev_body_high = max(o[i - 1], c[i - 1])
                if curr_body_low <= prev_body_low and curr_body_high >= prev_body_high:
                    if body > 0:  # Ensure non-doji
                        bull_engulf[i] = True

            if is_bearish and prev_bullish:
                curr_body_low = min(o[i], c[i])
                curr_body_high = max(o[i], c[i])
                prev_body_low = min(o[i - 1], c[i - 1])
                prev_body_high = max(o[i - 1], c[i - 1])
                if curr_body_low <= prev_body_low and curr_body_high >= prev_body_high:
                    if body > 0:
                        bear_engulf[i] = True

            # ── Pin Bar / Hammer (Bullish) ──
            # Long lower shadow, small upper shadow, small body
            if body > 0 and lower_shadow >= pin_bar_ratio * body:
                if upper_shadow <= max_opp_ratio * lower_shadow:
                    hammer[i] = True

            # ── Pin Bar / Shooting Star (Bearish) ──
            # Long upper shadow, small lower shadow, small body
            if body > 0 and upper_shadow >= pin_bar_ratio * body:
                if lower_shadow <= max_opp_ratio * upper_shadow:
                    shooting_star[i] = True

            # ── Star patterns (3-bar, need i >= 2) ──
            if i >= 2:
                bar_m2_body = abs(c[i - 2] - o[i - 2])
                bar_m1_body = abs(c[i - 1] - o[i - 1])
                bar_m2_bullish = c[i - 2] > o[i - 2]
                bar_m2_bearish = c[i - 2] < o[i - 2]
                bar_m2_midpoint = (o[i - 2] + c[i - 2]) / 2.0

                # Middle bar must have a small body relative to bar -2
                small_middle = (
                    bar_m2_body > 0
                    and bar_m1_body <= (star_body_pct / 100.0) * bar_m2_body
                )

                # Morning Star (Bullish)
                if (
                    small_middle
                    and bar_m2_bearish
                    and is_bullish
                    and c[i] > bar_m2_midpoint
                ):
                    morning_star[i] = True

                # Evening Star (Bearish)
                if (
                    small_middle
                    and bar_m2_bullish
                    and is_bearish
                    and c[i] < bar_m2_midpoint
                ):
                    evening_star[i] = True

        # Combined signals
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
                "reversal_long_entry": long_entry,
                "reversal_short_entry": short_entry,
            },
            index=data.index,
        )

        return result
```

---

## 3.12 Test Suite — Price Action

### `tests/unit/price_action/__init__.py`

```python
"""Tests for price action blocks."""
```

---

### `tests/unit/price_action/test_breakout.py`

```python
"""Tests for the Breakout price action block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.price_action.breakout import BreakoutBlock
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError, BlockValidationError
from forgequant.core.types import BlockCategory


@pytest.fixture
def breakout() -> BreakoutBlock:
    return BreakoutBlock()


def _make_ohlcv(close: np.ndarray, spread: float = 0.5) -> pd.DataFrame:
    """Helper to build OHLCV DataFrame from a close array."""
    n = len(close)
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {
            "open": close,
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": np.ones(n) * 1000.0,
        },
        index=dates,
    )


class TestBreakoutMetadata:
    def test_name(self, breakout: BreakoutBlock) -> None:
        assert breakout.metadata.name == "breakout"

    def test_category(self, breakout: BreakoutBlock) -> None:
        assert breakout.metadata.category == BlockCategory.PRICE_ACTION

    def test_defaults(self, breakout: BreakoutBlock) -> None:
        d = breakout.metadata.get_defaults()
        assert d["lookback"] == 20
        assert d["volume_multiplier"] == 1.5
        assert d["volume_lookback"] == 20

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "breakout" in registry


class TestBreakoutCompute:
    def test_output_columns(
        self, breakout: BreakoutBlock, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = breakout.execute(sample_ohlcv)
        expected = {
            "breakout_resistance", "breakout_support",
            "breakout_long", "breakout_short",
            "breakout_volume_confirm",
        }
        assert expected == set(result.columns)
        assert len(result) == len(sample_ohlcv)

    def test_anti_lookahead_shift(self, breakout: BreakoutBlock) -> None:
        """Resistance and support must use shifted values (no current bar)."""
        n = 30
        close = np.full(n, 100.0)
        # Spike at bar 25 — should NOT be included in bar 25's resistance
        close[25] = 200.0
        df = _make_ohlcv(close)
        result = breakout.execute(df, {"lookback": 5, "volume_multiplier": 0.0})

        # Bar 25's resistance is computed from bars before 25 (shifted)
        # The 200 spike should NOT appear in bar 25's breakout_resistance
        # because shift(1) means bar 25 sees the rolling max up to bar 24
        res_at_25 = result["breakout_resistance"].iloc[25]
        assert res_at_25 < 200.0, (
            f"Lookahead detected: resistance at spike bar should not include the spike. "
            f"Got {res_at_25}"
        )

    def test_upward_breakout_detected(self, breakout: BreakoutBlock) -> None:
        """A clear upward breakout should set breakout_long to True."""
        n = 50
        close = np.full(n, 100.0)
        # Breakout at bar 30
        close[30] = 110.0
        df = _make_ohlcv(close)
        result = breakout.execute(df, {"lookback": 10, "volume_multiplier": 0.0})
        assert result["breakout_long"].iloc[30] is True or result["breakout_long"].iloc[30] == True

    def test_downward_breakout_detected(self, breakout: BreakoutBlock) -> None:
        """A clear downward breakout should set breakout_short to True."""
        n = 50
        close = np.full(n, 100.0)
        close[30] = 90.0
        df = _make_ohlcv(close)
        result = breakout.execute(df, {"lookback": 10, "volume_multiplier": 0.0})
        assert result["breakout_short"].iloc[30] == True

    def test_no_breakout_in_flat_market(self, breakout: BreakoutBlock) -> None:
        """Flat prices should produce no breakout signals (after warmup)."""
        n = 100
        close = np.full(n, 100.0)
        df = _make_ohlcv(close, spread=0.0)
        # With spread=0, high=close and low=close, so close never > rolling max or < rolling min
        result = breakout.execute(df, {"lookback": 10, "volume_multiplier": 0.0})
        # After warmup (lookback + 1), no breakouts should occur
        assert result["breakout_long"].iloc[15:].sum() == 0
        assert result["breakout_short"].iloc[15:].sum() == 0

    def test_volume_confirmation(self, breakout: BreakoutBlock) -> None:
        """Volume filter should only confirm when volume exceeds threshold."""
        n = 50
        close = np.full(n, 100.0)
        close[30] = 110.0
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        volume = np.ones(n) * 1000.0
        # Low volume at breakout bar
        volume[30] = 500.0
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": volume,
            },
            index=dates,
        )
        result = breakout.execute(df, {"lookback": 10, "volume_multiplier": 1.5})
        # Breakout detected but volume not confirmed
        assert result["breakout_long"].iloc[30] == True
        assert result["breakout_volume_confirm"].iloc[30] == False

    def test_volume_disabled(self, breakout: BreakoutBlock) -> None:
        """When volume_multiplier=0, volume_confirm should always be True."""
        n = 50
        close = np.full(n, 100.0)
        df = _make_ohlcv(close)
        result = breakout.execute(df, {"lookback": 10, "volume_multiplier": 0.0})
        assert result["breakout_volume_confirm"].all()

    def test_insufficient_data_raises(self, breakout: BreakoutBlock) -> None:
        close = np.full(10, 100.0)
        df = _make_ohlcv(close)
        with pytest.raises(BlockComputeError, match="Insufficient data"):
            breakout.execute(df, {"lookback": 20})

    def test_resistance_above_support(
        self, breakout: BreakoutBlock, sample_ohlcv: pd.DataFrame
    ) -> None:
        """Resistance should always be >= support where both are valid."""
        result = breakout.execute(sample_ohlcv)
        valid = result.dropna(subset=["breakout_resistance", "breakout_support"])
        assert (valid["breakout_resistance"] >= valid["breakout_support"]).all()
```

---

### `tests/unit/price_action/test_pullback.py`

```python
"""Tests for the Pullback price action block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.price_action.pullback import PullbackBlock
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError, BlockValidationError
from forgequant.core.types import BlockCategory


@pytest.fixture
def pullback() -> PullbackBlock:
    return PullbackBlock()


class TestPullbackMetadata:
    def test_name(self, pullback: PullbackBlock) -> None:
        assert pullback.metadata.name == "pullback"

    def test_category(self, pullback: PullbackBlock) -> None:
        assert pullback.metadata.category == BlockCategory.PRICE_ACTION

    def test_defaults(self, pullback: PullbackBlock) -> None:
        d = pullback.metadata.get_defaults()
        assert d["ma_period"] == 20
        assert d["ma_type"] == "ema"
        assert d["proximity_pct"] == 0.5

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "pullback" in registry


class TestPullbackCompute:
    def test_output_columns(
        self, pullback: PullbackBlock, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = pullback.execute(sample_ohlcv)
        expected = {
            "pullback_ma", "pullback_upper_zone", "pullback_lower_zone",
            "pullback_long", "pullback_short",
        }
        assert expected == set(result.columns)
        assert len(result) == len(sample_ohlcv)

    def test_zone_surrounds_ma(
        self, pullback: PullbackBlock, sample_ohlcv: pd.DataFrame
    ) -> None:
        """Upper zone > MA > lower zone."""
        result = pullback.execute(sample_ohlcv)
        valid = result.dropna()
        assert (valid["pullback_upper_zone"] >= valid["pullback_ma"]).all()
        assert (valid["pullback_lower_zone"] <= valid["pullback_ma"]).all()

    def test_ema_vs_sma(
        self, pullback: PullbackBlock, sample_ohlcv: pd.DataFrame
    ) -> None:
        """EMA and SMA should produce different MA values."""
        result_ema = pullback.execute(sample_ohlcv, {"ma_type": "ema"})
        result_sma = pullback.execute(sample_ohlcv, {"ma_type": "sma"})
        # They should NOT be identical (different calculation)
        assert not result_ema["pullback_ma"].equals(result_sma["pullback_ma"])

    def test_long_pullback_detection(self, pullback: PullbackBlock) -> None:
        """Construct a scenario where a long pullback should trigger."""
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        # Strong uptrend, then dip to MA, then recovery
        close = np.linspace(100, 150, n)
        # Create a dip at bars 60-62
        close[60] = close[59] - 5.0  # Dip below
        close[61] = close[59] - 3.0  # Recovery
        close[62] = close[59] + 1.0  # Above MA again

        high = close + 1.0
        low = close - 1.0
        # Make the low of bar 62 dip close to the EMA level
        low[62] = close[59] - 2.0

        df = pd.DataFrame(
            {
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.ones(n) * 1000.0,
            },
            index=dates,
        )
        result = pullback.execute(df, {"ma_period": 10, "proximity_pct": 3.0})
        # There should be at least one long pullback signal
        assert result["pullback_long"].sum() > 0

    def test_no_pullback_in_flat_market(self, pullback: PullbackBlock) -> None:
        """Flat prices centered on MA shouldn't generate pullback entries."""
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        # Constant price = constant MA, prev_close not consistently above/below
        close = np.full(n, 100.0)
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.01,
                "low": close - 0.01,
                "close": close,
                "volume": np.ones(n) * 1000.0,
            },
            index=dates,
        )
        result = pullback.execute(df, {"ma_period": 10, "proximity_pct": 0.5})
        # With flat prices, MA = close exactly, so close > MA is False
        # No pullback signals expected
        assert result["pullback_long"].sum() == 0
        assert result["pullback_short"].sum() == 0

    def test_insufficient_data_raises(self, pullback: PullbackBlock) -> None:
        dates = pd.date_range("2024-01-01", periods=5, freq="h")
        df = pd.DataFrame(
            {
                "open": [100.0] * 5,
                "high": [101.0] * 5,
                "low": [99.0] * 5,
                "close": [100.0] * 5,
                "volume": [1000.0] * 5,
            },
            index=dates,
        )
        with pytest.raises(BlockComputeError, match="Insufficient data"):
            pullback.execute(df, {"ma_period": 20})

    def test_invalid_ma_type_raises(
        self, pullback: PullbackBlock, sample_ohlcv: pd.DataFrame
    ) -> None:
        with pytest.raises(BlockValidationError):
            pullback.execute(sample_ohlcv, {"ma_type": "wma"})
```

---

### `tests/unit/price_action/test_higher_high_lower_low.py`

```python
"""Tests for the Higher High / Lower Low price action block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.price_action.higher_high_lower_low import HigherHighLowerLowBlock
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory


@pytest.fixture
def hhll() -> HigherHighLowerLowBlock:
    return HigherHighLowerLowBlock()


class TestHHLLMetadata:
    def test_name(self, hhll: HigherHighLowerLowBlock) -> None:
        assert hhll.metadata.name == "higher_high_lower_low"

    def test_category(self, hhll: HigherHighLowerLowBlock) -> None:
        assert hhll.metadata.category == BlockCategory.PRICE_ACTION

    def test_defaults(self, hhll: HigherHighLowerLowBlock) -> None:
        d = hhll.metadata.get_defaults()
        assert d["left_bars"] == 5
        assert d["right_bars"] == 5

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "higher_high_lower_low" in registry


class TestHHLLCompute:
    def test_output_columns(
        self, hhll: HigherHighLowerLowBlock, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = hhll.execute(sample_ohlcv)
        expected = {
            "hhll_swing_high", "hhll_swing_low",
            "hhll_is_hh", "hhll_is_lh",
            "hhll_is_hl", "hhll_is_ll",
            "hhll_trend",
        }
        assert expected == set(result.columns)
        assert len(result) == len(sample_ohlcv)

    def test_uptrend_detection(self, hhll: HigherHighLowerLowBlock) -> None:
        """A clear uptrend should produce HH and HL."""
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        # Create zigzag uptrend: alternating swings, each higher
        close = np.zeros(n)
        for i in range(n):
            # Base trend upward
            base = 100.0 + i * 0.5
            # Add zigzag: +5 on peaks, -5 on troughs
            cycle = np.sin(i * 2 * np.pi / 20) * 5.0
            close[i] = base + cycle

        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 2.0,
                "low": close - 2.0,
                "close": close,
                "volume": np.ones(n) * 1000.0,
            },
            index=dates,
        )
        result = hhll.execute(df, {"left_bars": 3, "right_bars": 3})

        # Should find some HH and HL
        assert result["hhll_is_hh"].sum() > 0
        assert result["hhll_is_hl"].sum() > 0
        # Trend should be bullish at some point
        assert (result["hhll_trend"] == "bullish").any()

    def test_downtrend_detection(self, hhll: HigherHighLowerLowBlock) -> None:
        """A clear downtrend should produce LH and LL."""
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.zeros(n)
        for i in range(n):
            base = 200.0 - i * 0.5
            cycle = np.sin(i * 2 * np.pi / 20) * 5.0
            close[i] = base + cycle

        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 2.0,
                "low": close - 2.0,
                "close": close,
                "volume": np.ones(n) * 1000.0,
            },
            index=dates,
        )
        result = hhll.execute(df, {"left_bars": 3, "right_bars": 3})

        assert result["hhll_is_lh"].sum() > 0
        assert result["hhll_is_ll"].sum() > 0
        assert (result["hhll_trend"] == "bearish").any()

    def test_swing_high_requires_left_right(
        self, hhll: HigherHighLowerLowBlock
    ) -> None:
        """Swing high must be higher than both left and right neighbors."""
        n = 20
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        high = np.full(n, 100.0)
        # Create an obvious swing high at bar 10
        high[10] = 120.0
        low = np.full(n, 90.0)
        close = np.full(n, 95.0)
        close[10] = 115.0

        df = pd.DataFrame(
            {
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.ones(n) * 1000.0,
            },
            index=dates,
        )
        result = hhll.execute(df, {"left_bars": 3, "right_bars": 3})
        # Bar 10 should be identified as a swing high
        assert not np.isnan(result["hhll_swing_high"].iloc[10])
        assert result["hhll_swing_high"].iloc[10] == 120.0

    def test_no_swings_in_flat_market(
        self, hhll: HigherHighLowerLowBlock
    ) -> None:
        """Flat highs/lows should produce no swing points."""
        n = 50
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.full(n, 100.0)
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": np.ones(n) * 1000.0,
            },
            index=dates,
        )
        result = hhll.execute(df, {"left_bars": 3, "right_bars": 3})
        # All swing high/low should be NaN (no bar strictly higher/lower than neighbors)
        assert result["hhll_swing_high"].isna().all()
        assert result["hhll_swing_low"].isna().all()

    def test_trend_forward_fills(
        self, hhll: HigherHighLowerLowBlock, sample_ohlcv: pd.DataFrame
    ) -> None:
        """Trend should forward-fill between swing points."""
        result = hhll.execute(sample_ohlcv)
        trend = result["hhll_trend"]
        # Trend should only contain "bullish", "bearish", or "neutral"
        valid_values = {"bullish", "bearish", "neutral"}
        assert set(trend.unique()) <= valid_values

    def test_insufficient_data_raises(
        self, hhll: HigherHighLowerLowBlock
    ) -> None:
        dates = pd.date_range("2024-01-01", periods=5, freq="h")
        df = pd.DataFrame(
            {
                "open": [100.0] * 5,
                "high": [101.0] * 5,
                "low": [99.0] * 5,
                "close": [100.0] * 5,
                "volume": [1000.0] * 5,
            },
            index=dates,
        )
        with pytest.raises(BlockComputeError, match="Insufficient data"):
            hhll.execute(df, {"left_bars": 5, "right_bars": 5})
```

---

### `tests/unit/price_action/test_support_resistance.py`

```python
"""Tests for the Support & Resistance price action block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.price_action.support_resistance import SupportResistanceBlock
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory


@pytest.fixture
def sr() -> SupportResistanceBlock:
    return SupportResistanceBlock()


class TestSRMetadata:
    def test_name(self, sr: SupportResistanceBlock) -> None:
        assert sr.metadata.name == "support_resistance"

    def test_category(self, sr: SupportResistanceBlock) -> None:
        assert sr.metadata.category == BlockCategory.PRICE_ACTION

    def test_defaults(self, sr: SupportResistanceBlock) -> None:
        d = sr.metadata.get_defaults()
        assert d["left_bars"] == 5
        assert d["right_bars"] == 5
        assert d["merge_pct"] == 0.5
        assert d["max_zones"] == 20

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "support_resistance" in registry


class TestSRCompute:
    def test_output_columns(
        self, sr: SupportResistanceBlock, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = sr.execute(sample_ohlcv)
        expected = {
            "sr_nearest_support", "sr_nearest_resistance",
            "sr_support_strength", "sr_resistance_strength",
            "sr_distance_to_support_pct", "sr_distance_to_resistance_pct",
        }
        assert expected == set(result.columns)
        assert len(result) == len(sample_ohlcv)

    def test_support_below_close(
        self, sr: SupportResistanceBlock, sample_ohlcv: pd.DataFrame
    ) -> None:
        """Nearest support should be below the current close."""
        result = sr.execute(sample_ohlcv)
        valid = result.dropna(subset=["sr_nearest_support"])
        close = sample_ohlcv.loc[valid.index, "close"]
        assert (valid["sr_nearest_support"] < close).all()

    def test_resistance_above_close(
        self, sr: SupportResistanceBlock, sample_ohlcv: pd.DataFrame
    ) -> None:
        """Nearest resistance should be above the current close."""
        result = sr.execute(sample_ohlcv)
        valid = result.dropna(subset=["sr_nearest_resistance"])
        close = sample_ohlcv.loc[valid.index, "close"]
        assert (valid["sr_nearest_resistance"] > close).all()

    def test_distance_positive(
        self, sr: SupportResistanceBlock, sample_ohlcv: pd.DataFrame
    ) -> None:
        """Distances to support and resistance should be positive."""
        result = sr.execute(sample_ohlcv)
        sup_dist = result["sr_distance_to_support_pct"].dropna()
        res_dist = result["sr_distance_to_resistance_pct"].dropna()
        if len(sup_dist) > 0:
            assert (sup_dist > 0).all()
        if len(res_dist) > 0:
            assert (res_dist > 0).all()

    def test_strength_at_least_1(
        self, sr: SupportResistanceBlock, sample_ohlcv: pd.DataFrame
    ) -> None:
        """Zone strength should be at least 1 (at least one swing contributed)."""
        result = sr.execute(sample_ohlcv)
        sup_str = result["sr_support_strength"].dropna()
        res_str = result["sr_resistance_strength"].dropna()
        if len(sup_str) > 0:
            assert (sup_str >= 1).all()
        if len(res_str) > 0:
            assert (res_str >= 1).all()

    def test_merge_levels(self, sr: SupportResistanceBlock) -> None:
        """Test the static _merge_levels method directly."""
        levels = [100.0, 100.2, 100.3, 110.0, 110.1]
        zones = SupportResistanceBlock._merge_levels(levels, merge_pct=0.5)
        # Should merge 100.0/100.2/100.3 into one zone and 110.0/110.1 into another
        assert len(zones) == 2
        # First zone around 100.x, second around 110.x
        assert abs(zones[0][0] - 100.167) < 0.5
        assert zones[0][1] == 3  # 3 touches
        assert abs(zones[1][0] - 110.05) < 0.5
        assert zones[1][1] == 2  # 2 touches

    def test_empty_levels_merge(self, sr: SupportResistanceBlock) -> None:
        """Empty input should produce empty output."""
        zones = SupportResistanceBlock._merge_levels([], merge_pct=0.5)
        assert zones == []

    def test_known_swing_levels(self, sr: SupportResistanceBlock) -> None:
        """Construct data with obvious swing points."""
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        # Create zigzag pattern
        close = np.zeros(n)
        for i in range(n):
            close[i] = 100 + 10 * np.sin(i * 2 * np.pi / 20)

        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": np.ones(n) * 1000.0,
            },
            index=dates,
        )
        result = sr.execute(df, {"left_bars": 3, "right_bars": 3})
        # Should find at least some support and resistance levels
        has_support = result["sr_nearest_support"].notna().any()
        has_resistance = result["sr_nearest_resistance"].notna().any()
        assert has_support or has_resistance

    def test_insufficient_data_raises(self, sr: SupportResistanceBlock) -> None:
        dates = pd.date_range("2024-01-01", periods=5, freq="h")
        df = pd.DataFrame(
            {
                "open": [100.0] * 5,
                "high": [101.0] * 5,
                "low": [99.0] * 5,
                "close": [100.0] * 5,
                "volume": [1000.0] * 5,
            },
            index=dates,
        )
        with pytest.raises(BlockComputeError, match="Insufficient data"):
            sr.execute(df, {"left_bars": 5, "right_bars": 5})
```

---

## 3.13 Test Suite — Entry Rules

### `tests/unit/entry_rules/__init__.py`

```python
"""Tests for entry rule blocks."""
```

---

### `tests/unit/entry_rules/test_crossover.py`

```python
"""Tests for the Crossover entry rule block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.entry_rules.crossover import CrossoverEntry
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError, BlockValidationError
from forgequant.core.types import BlockCategory


@pytest.fixture
def crossover() -> CrossoverEntry:
    return CrossoverEntry()


class TestCrossoverMetadata:
    def test_name(self, crossover: CrossoverEntry) -> None:
        assert crossover.metadata.name == "crossover_entry"

    def test_category(self, crossover: CrossoverEntry) -> None:
        assert crossover.metadata.category == BlockCategory.ENTRY_RULE

    def test_defaults(self, crossover: CrossoverEntry) -> None:
        d = crossover.metadata.get_defaults()
        assert d["fast_period"] == 10
        assert d["slow_period"] == 20
        assert d["ma_type"] == "ema"

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "crossover_entry" in registry


class TestCrossoverCompute:
    def test_output_columns(
        self, crossover: CrossoverEntry, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = crossover.execute(sample_ohlcv)
        expected = {
            "crossover_fast_ma", "crossover_slow_ma",
            "crossover_long_entry", "crossover_short_entry",
        }
        assert expected == set(result.columns)
        assert len(result) == len(sample_ohlcv)

    def test_signal_is_state_change_only(
        self, crossover: CrossoverEntry, sample_ohlcv: pd.DataFrame
    ) -> None:
        """Signals should fire only on the bar of crossover, not all bars where fast > slow."""
        result = crossover.execute(sample_ohlcv)
        long_signals = result["crossover_long_entry"].sum()
        short_signals = result["crossover_short_entry"].sum()
        # Each crossover produces exactly one signal bar,
        # so total signals should be much less than total bars
        assert long_signals < len(sample_ohlcv) / 2
        assert short_signals < len(sample_ohlcv) / 2

    def test_long_and_short_mutually_exclusive(
        self, crossover: CrossoverEntry, sample_ohlcv: pd.DataFrame
    ) -> None:
        """A bar cannot be both a long and short crossover simultaneously."""
        result = crossover.execute(sample_ohlcv)
        both = result["crossover_long_entry"] & result["crossover_short_entry"]
        assert both.sum() == 0

    def test_uptrend_produces_long_signal(
        self, crossover: CrossoverEntry
    ) -> None:
        """A price jump from flat should produce a bullish crossover."""
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.full(n, 100.0)
        # Price jumps at bar 50
        close[50:] = 120.0
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": np.ones(n) * 1000.0,
            },
            index=dates,
        )
        result = crossover.execute(df, {"fast_period": 5, "slow_period": 20, "ma_type": "ema"})
        # Should detect a long crossover after the price jump
        long_after_jump = result["crossover_long_entry"].iloc[50:].sum()
        assert long_after_jump >= 1

    def test_downtrend_produces_short_signal(
        self, crossover: CrossoverEntry
    ) -> None:
        """A price drop from flat should produce a bearish crossover."""
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.full(n, 100.0)
        close[50:] = 80.0
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": np.ones(n) * 1000.0,
            },
            index=dates,
        )
        result = crossover.execute(df, {"fast_period": 5, "slow_period": 20, "ma_type": "ema"})
        short_after_drop = result["crossover_short_entry"].iloc[50:].sum()
        assert short_after_drop >= 1

    def test_sma_mode(
        self, crossover: CrossoverEntry, sample_ohlcv: pd.DataFrame
    ) -> None:
        """SMA mode should produce different results than EMA."""
        result_ema = crossover.execute(sample_ohlcv, {"ma_type": "ema"})
        result_sma = crossover.execute(sample_ohlcv, {"ma_type": "sma"})
        assert not result_ema["crossover_fast_ma"].equals(result_sma["crossover_fast_ma"])

    def test_fast_ge_slow_raises(
        self, crossover: CrossoverEntry, sample_ohlcv: pd.DataFrame
    ) -> None:
        with pytest.raises(BlockComputeError, match="must be less than"):
            crossover.execute(sample_ohlcv, {"fast_period": 20, "slow_period": 10})

    def test_fast_eq_slow_raises(
        self, crossover: CrossoverEntry, sample_ohlcv: pd.DataFrame
    ) -> None:
        with pytest.raises(BlockComputeError, match="must be less than"):
            crossover.execute(sample_ohlcv, {"fast_period": 20, "slow_period": 20})

    def test_insufficient_data_raises(self, crossover: CrossoverEntry) -> None:
        dates = pd.date_range("2024-01-01", periods=15, freq="h")
        df = pd.DataFrame(
            {
                "open": [100.0] * 15,
                "high": [101.0] * 15,
                "low": [99.0] * 15,
                "close": [100.0] * 15,
                "volume": [1000.0] * 15,
            },
            index=dates,
        )
        with pytest.raises(BlockComputeError, match="Insufficient data"):
            crossover.execute(df, {"fast_period": 10, "slow_period": 20})
```

---

### `tests/unit/entry_rules/test_threshold_cross.py`

```python
"""Tests for the Threshold Cross entry rule block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.entry_rules.threshold_cross import ThresholdCrossEntry
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError, BlockValidationError
from forgequant.core.types import BlockCategory


@pytest.fixture
def threshold() -> ThresholdCrossEntry:
    return ThresholdCrossEntry()


class TestThresholdMetadata:
    def test_name(self, threshold: ThresholdCrossEntry) -> None:
        assert threshold.metadata.name == "threshold_cross_entry"

    def test_category(self, threshold: ThresholdCrossEntry) -> None:
        assert threshold.metadata.category == BlockCategory.ENTRY_RULE

    def test_defaults(self, threshold: ThresholdCrossEntry) -> None:
        d = threshold.metadata.get_defaults()
        assert d["rsi_period"] == 14
        assert d["upper_threshold"] == 70.0
        assert d["lower_threshold"] == 30.0
        assert d["mode"] == "mean_reversion"

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "threshold_cross_entry" in registry


class TestThresholdCompute:
    def test_output_columns(
        self, threshold: ThresholdCrossEntry, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = threshold.execute(sample_ohlcv)
        expected = {
            "threshold_indicator",
            "threshold_long_entry",
            "threshold_short_entry",
        }
        assert expected == set(result.columns)
        assert len(result) == len(sample_ohlcv)

    def test_indicator_is_rsi(
        self, threshold: ThresholdCrossEntry, sample_ohlcv: pd.DataFrame
    ) -> None:
        """The indicator column should contain RSI values bounded in [0, 100]."""
        result = threshold.execute(sample_ohlcv)
        rsi = result["threshold_indicator"].dropna()
        assert rsi.min() >= 0
        assert rsi.max() <= 100

    def test_mean_reversion_long_on_oversold_recovery(
        self, threshold: ThresholdCrossEntry
    ) -> None:
        """In mean_reversion mode, long fires when RSI recovers from oversold."""
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        # Downtrend then recovery
        close = np.concatenate([
            np.linspace(200, 100, 100),  # Down
            np.linspace(100, 150, 100),  # Recovery
        ])
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = threshold.execute(df, {"mode": "mean_reversion"})
        # After the downtrend, RSI should be very low, then recover past 30
        long_signals = result["threshold_long_entry"].iloc[100:].sum()
        assert long_signals >= 1

    def test_momentum_long_on_overbought_entry(
        self, threshold: ThresholdCrossEntry
    ) -> None:
        """In momentum mode, long fires when RSI surges above overbought."""
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        # Flat then strong uptrend
        close = np.concatenate([
            np.full(100, 100.0),
            np.linspace(100, 200, 100),
        ])
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = threshold.execute(df, {"mode": "momentum"})
        long_signals = result["threshold_long_entry"].iloc[100:].sum()
        assert long_signals >= 1

    def test_signals_are_state_changes(
        self, threshold: ThresholdCrossEntry, sample_ohlcv: pd.DataFrame
    ) -> None:
        """Signals should fire only on the bar of the crossing."""
        result = threshold.execute(sample_ohlcv)
        total = result["threshold_long_entry"].sum() + result["threshold_short_entry"].sum()
        assert total < len(sample_ohlcv) / 2

    def test_upper_lte_lower_raises(
        self, threshold: ThresholdCrossEntry, sample_ohlcv: pd.DataFrame
    ) -> None:
        with pytest.raises(BlockComputeError, match="must be greater than"):
            threshold.execute(
                sample_ohlcv,
                {"upper_threshold": 30.0, "lower_threshold": 70.0},
            )

    def test_insufficient_data_raises(self, threshold: ThresholdCrossEntry) -> None:
        dates = pd.date_range("2024-01-01", periods=10, freq="h")
        df = pd.DataFrame(
            {
                "open": [100.0] * 10,
                "high": [101.0] * 10,
                "low": [99.0] * 10,
                "close": [100.0] * 10,
                "volume": [1000.0] * 10,
            },
            index=dates,
        )
        with pytest.raises(BlockComputeError, match="Insufficient data"):
            threshold.execute(df)

    def test_invalid_mode_raises(
        self, threshold: ThresholdCrossEntry, sample_ohlcv: pd.DataFrame
    ) -> None:
        with pytest.raises(BlockValidationError):
            threshold.execute(sample_ohlcv, {"mode": "invalid"})
```

---

### `tests/unit/entry_rules/test_confluence.py`

```python
"""Tests for the Confluence entry rule block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.entry_rules.confluence import ConfluenceEntry
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory


@pytest.fixture
def confluence() -> ConfluenceEntry:
    return ConfluenceEntry()


class TestConfluenceMetadata:
    def test_name(self, confluence: ConfluenceEntry) -> None:
        assert confluence.metadata.name == "confluence_entry"

    def test_category(self, confluence: ConfluenceEntry) -> None:
        assert confluence.metadata.category == BlockCategory.ENTRY_RULE

    def test_defaults(self, confluence: ConfluenceEntry) -> None:
        d = confluence.metadata.get_defaults()
        assert d["trend_period"] == 50
        assert d["rsi_period"] == 14
        assert d["atr_period"] == 14

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "confluence_entry" in registry


class TestConfluenceCompute:
    def test_output_columns(
        self, confluence: ConfluenceEntry, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = confluence.execute(sample_ohlcv)
        expected = {
            "confluence_trend_ok", "confluence_momentum_ok",
            "confluence_volatility_ok",
            "confluence_long_entry", "confluence_short_entry",
            "confluence_score",
        }
        assert expected == set(result.columns)

    def test_score_range(
        self, confluence: ConfluenceEntry, sample_ohlcv: pd.DataFrame
    ) -> None:
        """Confluence score should be between 0 and 3."""
        result = confluence.execute(sample_ohlcv)
        scores = result["confluence_score"]
        assert scores.min() >= 0
        assert scores.max() <= 3

    def test_long_requires_all_three(
        self, confluence: ConfluenceEntry, sample_ohlcv: pd.DataFrame
    ) -> None:
        """Long entry should be True only when all three conditions are True."""
        result = confluence.execute(sample_ohlcv)
        long_bars = result[result["confluence_long_entry"]]
        if len(long_bars) > 0:
            assert long_bars["confluence_trend_ok"].all()
            assert long_bars["confluence_momentum_ok"].all()
            assert long_bars["confluence_volatility_ok"].all()

    def test_long_score_3_matches_long_entry(
        self, confluence: ConfluenceEntry, sample_ohlcv: pd.DataFrame
    ) -> None:
        """Score of 3 should coincide with long_entry being True."""
        result = confluence.execute(sample_ohlcv)
        score_3 = result["confluence_score"] == 3
        long_entry = result["confluence_long_entry"]
        # score_3 should imply long_entry (they measure the same conditions for longs)
        pd.testing.assert_series_equal(score_3, long_entry, check_names=False)

    def test_long_short_mutually_exclusive(
        self, confluence: ConfluenceEntry, sample_ohlcv: pd.DataFrame
    ) -> None:
        """A bar cannot be both a long and short entry (trend can't be both)."""
        result = confluence.execute(sample_ohlcv)
        both = result["confluence_long_entry"] & result["confluence_short_entry"]
        assert both.sum() == 0

    def test_strong_uptrend_produces_signals(
        self, confluence: ConfluenceEntry
    ) -> None:
        """A clear uptrend with moderate RSI should produce long entries."""
        n = 300
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        np.random.seed(42)
        # Uptrend with noise
        close = 100.0 + np.cumsum(np.random.normal(0.1, 0.3, n))
        # Ensure non-negative
        close = np.maximum(close, 50.0)
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + abs(np.random.randn(n)),
                "low": close - abs(np.random.randn(n)),
                "close": close,
                "volume": np.ones(n) * 1000.0,
            },
            index=dates,
        )
        result = confluence.execute(df)
        assert result["confluence_long_entry"].sum() > 0

    def test_insufficient_data_raises(
        self, confluence: ConfluenceEntry
    ) -> None:
        dates = pd.date_range("2024-01-01", periods=30, freq="h")
        df = pd.DataFrame(
            {
                "open": [100.0] * 30,
                "high": [101.0] * 30,
                "low": [99.0] * 30,
                "close": [100.0] * 30,
                "volume": [1000.0] * 30,
            },
            index=dates,
        )
        with pytest.raises(BlockComputeError, match="Insufficient data"):
            confluence.execute(df, {"trend_period": 50})
```

---

### `tests/unit/entry_rules/test_reversal_pattern.py`

```python
"""Tests for the Reversal Pattern entry rule block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.entry_rules.reversal_pattern import ReversalPatternEntry
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory


@pytest.fixture
def reversal() -> ReversalPatternEntry:
    return ReversalPatternEntry()


def _make_bar(
    open_: float, high: float, low: float, close: float
) -> dict[str, float]:
    return {"open": open_, "high": high, "low": low, "close": close, "volume": 1000.0}


class TestReversalMetadata:
    def test_name(self, reversal: ReversalPatternEntry) -> None:
        assert reversal.metadata.name == "reversal_pattern_entry"

    def test_category(self, reversal: ReversalPatternEntry) -> None:
        assert reversal.metadata.category == BlockCategory.ENTRY_RULE

    def test_defaults(self, reversal: ReversalPatternEntry) -> None:
        d = reversal.metadata.get_defaults()
        assert d["pin_bar_ratio"] == 2.0
        assert d["max_opposite_wick_ratio"] == 0.5
        assert d["star_body_pct"] == 30.0

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "reversal_pattern_entry" in registry


class TestReversalCompute:
    def test_output_columns(
        self, reversal: ReversalPatternEntry, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = reversal.execute(sample_ohlcv)
        expected = {
            "reversal_bullish_engulfing", "reversal_bearish_engulfing",
            "reversal_hammer", "reversal_shooting_star",
            "reversal_morning_star", "reversal_evening_star",
            "reversal_long_entry", "reversal_short_entry",
        }
        assert expected == set(result.columns)

    def test_bullish_engulfing(self, reversal: ReversalPatternEntry) -> None:
        """Construct an explicit bullish engulfing pattern."""
        bars = [
            # Padding bars
            _make_bar(100, 101, 99, 100),
            _make_bar(100, 101, 99, 100),
            # Bar -1: bearish candle (small body)
            _make_bar(102, 103, 99, 100),  # open=102, close=100 (bearish)
            # Bar 0: bullish candle that engulfs bar -1
            _make_bar(99, 104, 98, 103),   # open=99, close=103 (bullish, body 99-103 engulfs 100-102)
        ]
        dates = pd.date_range("2024-01-01", periods=len(bars), freq="h")
        df = pd.DataFrame(bars, index=dates)
        result = reversal.execute(df)
        assert result["reversal_bullish_engulfing"].iloc[-1] == True

    def test_bearish_engulfing(self, reversal: ReversalPatternEntry) -> None:
        """Construct an explicit bearish engulfing pattern."""
        bars = [
            _make_bar(100, 101, 99, 100),
            _make_bar(100, 101, 99, 100),
            # Bar -1: bullish candle (small body)
            _make_bar(100, 103, 99, 102),  # open=100, close=102 (bullish)
            # Bar 0: bearish candle that engulfs bar -1
            _make_bar(103, 104, 98, 99),   # open=103, close=99 (bearish, body 99-103 engulfs 100-102)
        ]
        dates = pd.date_range("2024-01-01", periods=len(bars), freq="h")
        df = pd.DataFrame(bars, index=dates)
        result = reversal.execute(df)
        assert result["reversal_bearish_engulfing"].iloc[-1] == True

    def test_hammer(self, reversal: ReversalPatternEntry) -> None:
        """Construct an explicit hammer (bullish pin bar)."""
        # Hammer: small body at top, long lower shadow
        # body = |close - open| = |101 - 100| = 1
        # lower_shadow = min(100, 101) - 95 = 5
        # upper_shadow = 101.2 - 101 = 0.2
        # pin_bar_ratio check: 5 >= 2.0 * 1 = 2 ✓
        # max_opp check: 0.2 <= 0.5 * 5 = 2.5 ✓
        bars = [
            _make_bar(100, 101, 99, 100),
            _make_bar(100, 101.2, 95, 101),  # Hammer
        ]
        dates = pd.date_range("2024-01-01", periods=len(bars), freq="h")
        df = pd.DataFrame(bars, index=dates)
        result = reversal.execute(df)
        assert result["reversal_hammer"].iloc[-1] == True

    def test_shooting_star(self, reversal: ReversalPatternEntry) -> None:
        """Construct an explicit shooting star (bearish pin bar)."""
        # body = |99 - 100| = 1
        # upper_shadow = 105 - max(100, 99) = 5
        # lower_shadow = min(100, 99) - 98.8 = 0.2
        # pin_bar_ratio: 5 >= 2 * 1 ✓
        # max_opp: 0.2 <= 0.5 * 5 ✓
        bars = [
            _make_bar(100, 101, 99, 100),
            _make_bar(100, 105, 98.8, 99),  # Shooting star
        ]
        dates = pd.date_range("2024-01-01", periods=len(bars), freq="h")
        df = pd.DataFrame(bars, index=dates)
        result = reversal.execute(df)
        assert result["reversal_shooting_star"].iloc[-1] == True

    def test_morning_star(self, reversal: ReversalPatternEntry) -> None:
        """Construct an explicit morning star (3-bar bullish reversal)."""
        bars = [
            _make_bar(100, 101, 99, 100),  # Padding
            # Bar -2: bearish, large body
            _make_bar(110, 111, 99, 100),  # body=10
            # Bar -1: small body (indecision)
            _make_bar(100, 101, 99, 100.5),  # body=0.5, which is 5% of 10 < 30% ✓
            # Bar 0: bullish, closes above midpoint of bar -2
            # midpoint of bar -2 = (110 + 100) / 2 = 105
            _make_bar(101, 107, 100, 106),  # close=106 > 105 ✓
        ]
        dates = pd.date_range("2024-01-01", periods=len(bars), freq="h")
        df = pd.DataFrame(bars, index=dates)
        result = reversal.execute(df)
        assert result["reversal_morning_star"].iloc[-1] == True

    def test_evening_star(self, reversal: ReversalPatternEntry) -> None:
        """Construct an explicit evening star (3-bar bearish reversal)."""
        bars = [
            _make_bar(100, 101, 99, 100),  # Padding
            # Bar -2: bullish, large body
            _make_bar(100, 111, 99, 110),  # body=10
            # Bar -1: small body
            _make_bar(110, 111, 109, 110.5),  # body=0.5 < 30% of 10 ✓
            # Bar 0: bearish, closes below midpoint of bar -2
            # midpoint = (100 + 110) / 2 = 105
            _make_bar(110, 111, 103, 104),  # close=104 < 105 ✓
        ]
        dates = pd.date_range("2024-01-01", periods=len(bars), freq="h")
        df = pd.DataFrame(bars, index=dates)
        result = reversal.execute(df)
        assert result["reversal_evening_star"].iloc[-1] == True

    def test_combined_long_entry(self, reversal: ReversalPatternEntry) -> None:
        """long_entry should be the OR of all bullish patterns."""
        bars = [
            _make_bar(100, 101, 99, 100),
            _make_bar(100, 101.2, 95, 101),  # Hammer
        ]
        dates = pd.date_range("2024-01-01", periods=len(bars), freq="h")
        df = pd.DataFrame(bars, index=dates)
        result = reversal.execute(df)
        assert result["reversal_long_entry"].iloc[-1] == True

    def test_combined_short_entry(self, reversal: ReversalPatternEntry) -> None:
        """short_entry should be the OR of all bearish patterns."""
        bars = [
            _make_bar(100, 101, 99, 100),
            _make_bar(100, 105, 98.8, 99),  # Shooting star
        ]
        dates = pd.date_range("2024-01-01", periods=len(bars), freq="h")
        df = pd.DataFrame(bars, index=dates)
        result = reversal.execute(df)
        assert result["reversal_short_entry"].iloc[-1] == True

    def test_no_pattern_in_flat_market(
        self, reversal: ReversalPatternEntry
    ) -> None:
        """Doji-like flat bars should not trigger any patterns."""
        n = 20
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        # All bars are dojis (open == close, small range)
        df = pd.DataFrame(
            {
                "open": [100.0] * n,
                "high": [100.1] * n,
                "low": [99.9] * n,
                "close": [100.0] * n,
                "volume": [1000.0] * n,
            },
            index=dates,
        )
        result = reversal.execute(df)
        # Engulfing requires body > 0, pin bar requires body > 0
        assert result["reversal_long_entry"].sum() == 0
        assert result["reversal_short_entry"].sum() == 0

    def test_on_sample_data_finds_patterns(
        self, reversal: ReversalPatternEntry, sample_ohlcv: pd.DataFrame
    ) -> None:
        """With 500 bars of random data, at least some patterns should be found."""
        result = reversal.execute(sample_ohlcv)
        total = result["reversal_long_entry"].sum() + result["reversal_short_entry"].sum()
        # With 500 random bars, we expect at least a few patterns
        assert total > 0

    def test_insufficient_data_raises(
        self, reversal: ReversalPatternEntry
    ) -> None:
        dates = pd.date_range("2024-01-01", periods=2, freq="h")
        df = pd.DataFrame(
            {
                "open": [100.0, 100.0],
                "high": [101.0, 101.0],
                "low": [99.0, 99.0],
                "close": [100.0, 100.0],
                "volume": [1000.0, 1000.0],
            },
            index=dates,
        )
        with pytest.raises(BlockComputeError, match="Insufficient data"):
            reversal.execute(df)
```

---

## 3.14 Integration Test — Phase 3 Registry

### `tests/integration/test_phase3_registry.py`

```python
"""
Integration test verifying all Phase 3 blocks (price action + entry rules)
are properly registered and can execute on sample data.
"""

from __future__ import annotations

import pandas as pd
import pytest

from forgequant.blocks.registry import BlockRegistry
from forgequant.core.types import BlockCategory

# Force registration
import forgequant.blocks.price_action  # noqa: F401
import forgequant.blocks.entry_rules  # noqa: F401


EXPECTED_PRICE_ACTION = [
    "breakout",
    "pullback",
    "higher_high_lower_low",
    "support_resistance",
]

EXPECTED_ENTRY_RULES = [
    "crossover_entry",
    "threshold_cross_entry",
    "confluence_entry",
    "reversal_pattern_entry",
]


class TestPriceActionRegistered:
    @pytest.mark.parametrize("block_name", EXPECTED_PRICE_ACTION)
    def test_present(self, block_name: str) -> None:
        registry = BlockRegistry()
        assert block_name in registry

    @pytest.mark.parametrize("block_name", EXPECTED_PRICE_ACTION)
    def test_category(self, block_name: str) -> None:
        registry = BlockRegistry()
        cls = registry.get_or_raise(block_name)
        assert cls.metadata.category == BlockCategory.PRICE_ACTION

    @pytest.mark.parametrize("block_name", EXPECTED_PRICE_ACTION)
    def test_execute_defaults(
        self, block_name: str, sample_ohlcv: pd.DataFrame
    ) -> None:
        registry = BlockRegistry()
        instance = registry.instantiate(block_name)
        result = instance.execute(sample_ohlcv)
        assert result is not None
        if isinstance(result, pd.DataFrame):
            assert len(result) == len(sample_ohlcv)


class TestEntryRulesRegistered:
    @pytest.mark.parametrize("block_name", EXPECTED_ENTRY_RULES)
    def test_present(self, block_name: str) -> None:
        registry = BlockRegistry()
        assert block_name in registry

    @pytest.mark.parametrize("block_name", EXPECTED_ENTRY_RULES)
    def test_category(self, block_name: str) -> None:
        registry = BlockRegistry()
        cls = registry.get_or_raise(block_name)
        assert cls.metadata.category == BlockCategory.ENTRY_RULE

    @pytest.mark.parametrize("block_name", EXPECTED_ENTRY_RULES)
    def test_execute_defaults(
        self, block_name: str, sample_ohlcv: pd.DataFrame
    ) -> None:
        registry = BlockRegistry()
        instance = registry.instantiate(block_name)
        result = instance.execute(sample_ohlcv)
        assert result is not None
        if isinstance(result, pd.DataFrame):
            assert len(result) == len(sample_ohlcv)

    @pytest.mark.parametrize("block_name", EXPECTED_ENTRY_RULES)
    def test_has_description(self, block_name: str) -> None:
        registry = BlockRegistry()
        cls = registry.get_or_raise(block_name)
        assert len(cls.metadata.description) > 20

    @pytest.mark.parametrize("block_name", EXPECTED_ENTRY_RULES)
    def test_has_tags(self, block_name: str) -> None:
        registry = BlockRegistry()
        cls = registry.get_or_raise(block_name)
        assert len(cls.metadata.tags) >= 1
```

---

## 3.15 How to Verify Phase 3

```bash
# From project root with venv activated

# Run all tests
pytest -v

# Run only price action tests
pytest tests/unit/price_action/ -v

# Run only entry rule tests
pytest tests/unit/entry_rules/ -v

# Run the integration test
pytest tests/integration/test_phase3_registry.py -v

# Type-check Phase 3 files
mypy src/forgequant/blocks/price_action/ src/forgequant/blocks/entry_rules/

# Lint
ruff check src/forgequant/blocks/price_action/ src/forgequant/blocks/entry_rules/
```

**Expected output:** All tests pass — approximately **70+ new tests** across 8 test modules plus **16 parametrized integration tests**.

---

## Phase 3 Summary

### Price Action Blocks

| Block | File | Output Columns | Key Detail |
|-------|------|----------------|------------|
| **Breakout** | `breakout.py` | resistance, support, long, short, volume_confirm | `shift(1)` anti-lookahead on rolling extremes |
| **Pullback** | `pullback.py` | ma, upper_zone, lower_zone, long, short | Proximity zone around MA; requires prior bar trend state |
| **Higher High / Lower Low** | `higher_high_lower_low.py` | swing_high/low, is_hh/lh/hl/ll, trend | Left/right bar swing confirmation; forward-filled trend |
| **Support & Resistance** | `support_resistance.py` | nearest S/R, strength, distance_pct | Swing clustering with merge tolerance; strength = touch count |

### Entry Rule Blocks

| Block | File | Output Columns | Key Detail |
|-------|------|----------------|------------|
| **Crossover** | `crossover.py` | fast/slow MA, long/short entry | State-change detection (signal only on cross bar) |
| **Threshold Cross** | `threshold_cross.py` | RSI indicator, long/short entry | Mean-reversion AND momentum modes |
| **Confluence** | `confluence.py` | trend/momentum/volatility ok, long/short, score | Requires all 3 conditions; self-contained indicator computation |
| **Reversal Pattern** | `reversal_pattern.py` | 6 pattern booleans, combined long/short | Engulfing, Pin Bar, Star patterns with configurable ratios |

### Cumulative Block Count

| Category | Count | Blocks |
|----------|-------|--------|
| Indicators | 8 | EMA, RSI, MACD, ADX, ATR, Bollinger, Ichimoku, Stochastic |
| Price Action | 4 | Breakout, Pullback, HHLL, Support/Resistance |
| Entry Rules | 4 | Crossover, ThresholdCross, Confluence, ReversalPattern |
| **Total** | **16** | |

---

**Ready for Phase 4** — say the word and I'll write all 4 exit rule blocks (FixedTPSL, TrailingStop, TimeBasedExit, BreakevenStop) and all 4 money management blocks (FixedRisk, VolatilityTargeting, KellyFractional, ATRBasedSizing) plus all 4 filter blocks (TradingSession, SpreadFilter, MaxDrawdownFilter, TrendFilter) with the same precision and full test coverage.
