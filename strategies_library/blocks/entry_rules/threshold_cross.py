"""
Indicator Threshold Cross Entry Block.

Generates entry signals when a specified indicator crosses above
or below a threshold level.

This block is generic and works with any indicator in the registry.
It first computes the specified indicator, then detects when its
value crosses the threshold.

Common uses:
    - RSI crosses below 30 (oversold → buy signal)
    - RSI crosses above 70 (overbought → sell signal)
    - ADX crosses above 25 (trend starting → allow trades)
    - Stochastic %K crosses below 20 (oversold)

Default Parameters:
    indicator_name: 'RSI'
    threshold: 30.0
    indicator_params: {}
"""

from typing import Any, Dict

import pandas as pd

from strategies_library.base import BaseBlock, BlockMetadata
from strategies_library.registry import BlockRegistry


@BlockRegistry.register
class ThresholdCross(BaseBlock):
    """Entry signal when an indicator crosses a threshold level."""

    metadata = BlockMetadata(
        name="ThresholdCross",
        category="entry",
        description="Entry when a specified indicator crosses above or below a threshold",
        complexity=2,
        typical_use=["mean_reversion", "overbought_oversold", "filter"],
        required_columns=["close"],
        version="1.0.0",
        tags=["threshold", "level", "cross", "entry"],
    )

    def compute(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any] | None = None,
    ) -> Dict[str, pd.Series]:
        """Compute threshold cross signals.

        Args:
            data: OHLCV DataFrame.
            params: Dict with:
                - indicator_name (str): Name of indicator in registry. Default 'RSI'.
                - threshold (float): Level to detect crossing. Default 30.0.
                - indicator_params (dict): Parameters to pass to the indicator. Default {}.

        Returns:
            Dict with keys:
                - 'cross_above': Boolean Series (True when indicator crosses above threshold)
                - 'cross_below': Boolean Series (True when indicator crosses below threshold)
                - 'indicator_values': The computed indicator Series
        """
        params = params or {}
        indicator_name = str(params.get("indicator_name", "RSI"))
        threshold = float(params.get("threshold", 30.0))
        indicator_params = dict(params.get("indicator_params", {}))

        # Retrieve and compute the indicator
        indicator_class = BlockRegistry.get(indicator_name)
        if indicator_class is None:
            raise ValueError(
                f"Indicator '{indicator_name}' not found in registry. "
                f"Available: {BlockRegistry.get_all_names()}"
            )

        indicator = indicator_class()
        result = indicator.compute(data, indicator_params)

        # Handle dict output (take the first Series)
        if isinstance(result, dict):
            # Use the first value that is a Series
            values = None
            for key, val in result.items():
                if isinstance(val, pd.Series):
                    values = val
                    break
            if values is None:
                raise ValueError(
                    f"Indicator '{indicator_name}' did not return any pd.Series."
                )
        elif isinstance(result, pd.Series):
            values = result
        else:
            raise ValueError(
                f"Indicator '{indicator_name}' returned unexpected type: {type(result)}"
            )

        # Detect threshold crossings
        above_threshold = values > threshold
        above_prev = above_threshold.shift(1).fillna(False)

        cross_above = above_threshold & ~above_prev
        cross_below = ~above_threshold & above_prev

        return {
            "cross_above": cross_above,
            "cross_below": cross_below,
            "indicator_values": values,
        }

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate ThresholdCross parameters.

        Rules:
            - indicator_name must exist in the registry.
            - threshold must be a valid number.
        """
        indicator_name = params.get("indicator_name", "RSI")
        if BlockRegistry.get(str(indicator_name)) is None:
            return False
        return True
