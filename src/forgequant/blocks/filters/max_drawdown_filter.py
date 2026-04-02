"""
Max Drawdown Filter block.

Monitors the cumulative return curve and halts trading when the
current drawdown from peak exceeds a configurable threshold.
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
class MaxDrawdownFilter(BaseBlock):
    """Maximum drawdown circuit breaker filter."""

    metadata = BlockMetadata(
        name="max_drawdown_filter",
        display_name="Max Drawdown Filter",
        category=BlockCategory.FILTER,
        description=(
            "Halts trading when the current drawdown from peak exceeds a "
            "threshold. Trading resumes once the drawdown recovers to "
            "within a configurable percentage. Acts as a circuit breaker "
            "to prevent catastrophic losses."
        ),
        parameters=(
            ParameterSpec(
                name="max_drawdown_pct",
                param_type="float",
                default=15.0,
                min_value=1.0,
                max_value=50.0,
                description="Maximum drawdown percentage before halting trades",
            ),
            ParameterSpec(
                name="recovery_pct",
                param_type="float",
                default=10.0,
                min_value=0.5,
                max_value=50.0,
                description=(
                    "Drawdown must recover to this percentage or less before "
                    "trading resumes"
                ),
            ),
        ),
        tags=("filter", "drawdown", "risk", "circuit_breaker", "protection"),
        typical_use=(
            "Essential risk management filter. Set max_drawdown to 15-20% "
            "for most strategies. The recovery threshold prevents premature "
            "re-entry during a declining market. Combine with position "
            "sizing to create a multi-layered defense."
        ),
    )

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        max_dd: float = params["max_drawdown_pct"]
        recovery: float = params["recovery_pct"]

        if recovery > max_dd:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=(
                    f"recovery_pct ({recovery}) must be <= max_drawdown_pct "
                    f"({max_dd}) — you can't require recovery beyond the "
                    f"halt threshold"
                ),
            )

        close = data["close"]
        n = len(close)

        if n < 2:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Insufficient data: need at least 2 rows, got {n}",
            )

        first_close = close.iloc[0]
        if first_close == 0:
            raise BlockComputeError(  # pragma: no cover
                block_name=self.metadata.name,
                reason="First close price is 0; cannot compute returns",
            )

        cum_return = close / first_close
        running_max = cum_return.expanding().max()
        drawdown = (cum_return - running_max) / running_max
        drawdown_pct = drawdown.abs() * 100.0

        allow_arr = np.ones(n, dtype=bool)
        dd_pct_vals = drawdown_pct.values
        halted = False

        for i in range(n):
            if halted:
                if dd_pct_vals[i] <= recovery:
                    halted = False
                    allow_arr[i] = True
                else:
                    allow_arr[i] = False
            else:
                if dd_pct_vals[i] > max_dd:
                    halted = True
                    allow_arr[i] = False

        allow_trading = pd.Series(allow_arr, index=data.index)

        result = pd.DataFrame(
            {
                "dd_cumulative_return": cum_return,
                "dd_running_max": running_max,
                "dd_drawdown": drawdown,
                "dd_drawdown_pct": drawdown_pct,
                SC.dd_allow_trading: allow_trading,
            },
            index=data.index,
        )

        return result
