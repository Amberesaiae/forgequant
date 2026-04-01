"""
Trading Session filter block.

Restricts trading to specific hours of the day, allowing strategies
to focus on the most liquid sessions and avoid thin/volatile off-hours.
"""

from __future__ import annotations

import pandas as pd

from forgequant.blocks.base import BaseBlock
from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory, BlockParams, BlockResult


@BlockRegistry.register
class TradingSessionFilter(BaseBlock):
    """Time-of-day trading session filter."""

    metadata = BlockMetadata(
        name="trading_session",
        display_name="Trading Session Filter",
        category=BlockCategory.FILTER,
        description=(
            "Restricts trading to specific hours of the day. Supports "
            "two session windows to cover major trading sessions "
            "(e.g. London + New York). Bars outside all sessions are "
            "flagged as inactive."
        ),
        parameters=(
            ParameterSpec(
                name="session1_start",
                param_type="int",
                default=8,
                min_value=0,
                max_value=23,
                description="Session 1 start hour (0-23, inclusive)",
            ),
            ParameterSpec(
                name="session1_end",
                param_type="int",
                default=16,
                min_value=0,
                max_value=23,
                description="Session 1 end hour (0-23, exclusive)",
            ),
            ParameterSpec(
                name="session2_start",
                param_type="int",
                default=-1,
                min_value=-1,
                max_value=23,
                description="Session 2 start hour (-1 to disable)",
            ),
            ParameterSpec(
                name="session2_end",
                param_type="int",
                default=-1,
                min_value=-1,
                max_value=23,
                description="Session 2 end hour (-1 to disable)",
            ),
        ),
        tags=("filter", "session", "time", "hours", "liquidity"),
        typical_use=(
            "Used to avoid low-liquidity periods (Asian session for EUR/USD), "
            "or to focus on the overlap between London and New York. "
            "Session hours are in the timezone of your data."
        ),
    )

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        s1_start: int = params["session1_start"]
        s1_end: int = params["session1_end"]
        s2_start: int = params["session2_start"]
        s2_end: int = params["session2_end"]

        if not isinstance(data.index, pd.DatetimeIndex):
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason="DataFrame index must be a DatetimeIndex for session filtering",
            )

        hours = data.index.hour
        n = len(data)

        if s1_start < s1_end:
            in_s1 = (hours >= s1_start) & (hours < s1_end)
        elif s1_start > s1_end:
            in_s1 = (hours >= s1_start) | (hours < s1_end)
        else:
            in_s1 = pd.Series(True, index=data.index)

        in_s1 = pd.Series(in_s1, index=data.index)

        if s2_start >= 0 and s2_end >= 0:
            if s2_start < s2_end:
                in_s2 = (hours >= s2_start) & (hours < s2_end)
            elif s2_start > s2_end:
                in_s2 = (hours >= s2_start) | (hours < s2_end)
            else:
                in_s2 = pd.Series(True, index=data.index)
            in_s2 = pd.Series(in_s2, index=data.index)
        else:
            in_s2 = pd.Series(False, index=data.index)

        active = in_s1 | in_s2

        names = pd.Series("outside", index=data.index, dtype="object")
        names[in_s1 & ~in_s2] = "session_1"
        names[in_s2 & ~in_s1] = "session_2"
        names[in_s1 & in_s2] = "overlap"

        result = pd.DataFrame(
            {
                "session_active": active,
                "session_name": names,
            },
            index=data.index,
        )

        return result
