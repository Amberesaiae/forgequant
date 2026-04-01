"""
Time-Based Exit rule block.

Provides bar-counting exit signals for maximum holding period
enforcement, along with day-of-week avoidance flags and
session-close proximity warnings.
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
class TimeBasedExit(BaseBlock):
    """Time-based exit signals and holding period management."""

    metadata = BlockMetadata(
        name="time_based_exit",
        display_name="Time-Based Exit",
        category=BlockCategory.EXIT_RULE,
        description=(
            "Provides bar-counting exit signals for maximum holding period "
            "enforcement, along with day-of-week avoidance flags and "
            "session-close proximity warnings."
        ),
        parameters=(
            ParameterSpec(
                name="max_bars",
                param_type="int",
                default=50,
                min_value=1,
                max_value=5000,
                description="Maximum number of bars to hold a position",
            ),
            ParameterSpec(
                name="avoid_days",
                param_type="str",
                default="",
                description=(
                    "Comma-separated day names to avoid (e.g. 'Friday,Sunday'). "
                    "Case-insensitive. Leave empty to disable."
                ),
            ),
            ParameterSpec(
                name="close_warning_bars",
                param_type="int",
                default=3,
                min_value=0,
                max_value=100,
                description=(
                    "Number of bars before a detected daily session change "
                    "to flag as near-session-close. Set to 0 to disable."
                ),
            ),
        ),
        tags=("exit", "time", "holding_period", "session", "day_of_week"),
        typical_use=(
            "Used to force-close trades that have been open too long "
            "(preventing stale positions), to avoid entering trades on "
            "certain days (e.g. Friday afternoon in forex), and to warn "
            "when the trading session is about to end."
        ),
    )

    _DAY_MAP: dict[str, int] = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
    }

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        max_bars: int = params["max_bars"]
        avoid_days_str: str = params["avoid_days"]
        close_warning: int = params["close_warning_bars"]

        n = len(data)

        bar_index = pd.Series(np.arange(n) % max_bars, index=data.index, dtype=int)
        max_bars_exit = bar_index == (max_bars - 1)

        avoid_day = pd.Series(False, index=data.index)

        if avoid_days_str.strip():
            avoid_names = [
                d.strip().lower() for d in avoid_days_str.split(",") if d.strip()
            ]
            avoid_ints: list[int] = []
            for name in avoid_names:
                if name in self._DAY_MAP:
                    avoid_ints.append(self._DAY_MAP[name])

            if avoid_ints:
                avoid_day = pd.Series(
                    data.index.dayofweek.isin(avoid_ints), index=data.index
                )

        near_close = pd.Series(False, index=data.index)

        if close_warning > 0 and isinstance(data.index, pd.DatetimeIndex):
            dates = data.index.date
            date_series = pd.Series(dates, index=data.index)
            date_changes = date_series != date_series.shift(-1)

            change_positions = np.where(date_changes.values)[0]
            near_close_arr = np.zeros(n, dtype=bool)

            for pos in change_positions:
                start = max(0, pos - close_warning + 1)
                near_close_arr[start : pos + 1] = True

            near_close = pd.Series(near_close_arr, index=data.index)

        result = pd.DataFrame(
            {
                "time_bar_index": bar_index,
                "time_max_bars_exit": max_bars_exit,
                "time_avoid_day": avoid_day,
                "time_near_session_close": near_close,
            },
            index=data.index,
        )

        return result
