"""
Time-Based Exit Block.

Forces closure of a position after a specified number of bars,
regardless of profit or loss. This prevents trades from sitting
indefinitely in stagnant markets and limits exposure duration.

Commonly used in mean-reversion strategies where the edge
diminishes over time, or as a safety mechanism.

The block returns the max_bars configuration value and an exit_signal
boolean Series that is True every N bars. The execution layer
tracks how many bars each position has been open and
exits when the limit is reached.

Default Parameters:
    max_bars: 12
"""

from typing import Any, Dict

import pandas as pd

from strategies_library.base import BaseBlock, BlockMetadata
from strategies_library.registry import BlockRegistry


@BlockRegistry.register
class TimeBasedExit(BaseBlock):
    """Force exit after a specified number of bars."""

    metadata = BlockMetadata(
        name="TimeBasedExit",
        category="exit",
        description="Force position closure after a specified number of bars",
        complexity=1,
        typical_use=["mean_reversion", "risk_management"],
        required_columns=[],
        version="1.0.0",
        tags=["time", "bars", "exit", "duration"],
    )

    def compute(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """Return time-based exit configuration.

        Args:
            data: OHLCV DataFrame (not used directly, but required by interface).
            params: Optional dict with:
                - max_bars (int): Maximum bars to hold a position. Default 12.

        Returns:
            Dict with keys:
                - 'max_bars': Number of bars after which to exit
                - 'exit_signal': Boolean Series that is True every N bars
        """
        params = params or {}
        max_bars = int(params.get("max_bars", 12))

        # Create a rolling counter that resets every max_bars
        bar_count = pd.Series(range(len(data)), index=data.index)
        exit_signal = (bar_count % max_bars == 0) & (bar_count > 0)

        return {
            "max_bars": max_bars,
            "exit_signal": exit_signal,
        }

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate TimeBasedExit parameters.

        Rules:
            - max_bars must be between 1 and 500.
        """
        max_bars = int(params.get("max_bars", 12))
        if max_bars < 1 or max_bars > 500:
            return False
        return True
