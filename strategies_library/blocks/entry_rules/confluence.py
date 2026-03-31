"""
Multi-Condition Confluence Entry Block.

Requires multiple independent conditions to be True simultaneously
before generating an entry signal. This produces higher-quality,
higher-probability entries by demanding agreement between different
types of analysis.

Each condition is specified as a block name + parameters. The
Confluence block computes each one, then counts how many are True
at each bar. If the count meets or exceeds min_conditions, a
signal is generated.

Example:
    conditions = [
        {"block_name": "Breakout", "params": {"lookback": 20}},
        {"block_name": "ThresholdCross", "params": {"indicator_name": "ADX", "threshold": 25}},
    ]
    min_conditions = 2  (both must be True)

Default Parameters:
    conditions: []  (list of dicts with 'block_name' and 'params')
    min_conditions: len(conditions)  (all conditions must be met by default)
"""

from typing import Any, Dict, List

import pandas as pd

from strategies_library.base import BaseBlock, BlockMetadata
from strategies_library.registry import BlockRegistry


@BlockRegistry.register
class Confluence(BaseBlock):
    """Multi-condition confluence entry requiring multiple confirmations."""

    metadata = BlockMetadata(
        name="Confluence",
        category="entry",
        description="Entry requires multiple conditions to be True simultaneously",
        complexity=4,
        typical_use=["high_probability_entry", "confluence"],
        required_columns=["high", "low", "close"],
        version="1.0.0",
        tags=["confluence", "multi_condition", "precision", "quality"],
    )

    def compute(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any] | None = None,
    ) -> pd.Series:
        """Compute confluence entry signal.

        Args:
            data: OHLCV DataFrame.
            params: Dict with:
                - conditions (list): List of dicts, each with:
                    - block_name (str): Name of block in registry.
                    - params (dict): Parameters for that block.
                - min_conditions (int): Minimum conditions that must be True.
                  Defaults to len(conditions) (all must be True).

        Returns:
            Boolean pd.Series. True when enough conditions are met.
        """
        params = params or {}
        conditions_config: List[Dict[str, Any]] = params.get("conditions", [])
        min_conditions = int(params.get("min_conditions", len(conditions_config)))

        if not conditions_config:
            return pd.Series(False, index=data.index)

        # Compute each condition
        condition_series: List[pd.Series] = []

        for cond in conditions_config:
            block_name = str(cond.get("block_name", ""))
            block_params = dict(cond.get("params", {}))

            block_class = BlockRegistry.get(block_name)
            if block_class is None:
                continue

            block = block_class()
            result = block.compute(data, block_params)

            # Extract a boolean Series from the result
            bool_series = self._extract_boolean_series(result, data.index)
            if bool_series is not None:
                condition_series.append(bool_series)

        if not condition_series:
            return pd.Series(False, index=data.index)

        # Count how many conditions are True at each bar
        conditions_df = pd.DataFrame(condition_series).T.fillna(False)
        conditions_met_count = conditions_df.sum(axis=1)

        return conditions_met_count >= min_conditions

    @staticmethod
    def _extract_boolean_series(
        result: Any, index: pd.Index
    ) -> pd.Series | None:
        """Extract a boolean Series from a block's compute result.

        If the result is a dict, it looks for keys containing 'entry',
        'signal', 'long', or takes the first boolean Series found.
        If the result is a numeric Series, it converts it to boolean
        (positive values = True).

        Args:
            result: Output from a block's compute() method.
            index: DataFrame index for creating default Series.

        Returns:
            Boolean pd.Series or None if extraction fails.
        """
        if isinstance(result, pd.Series):
            if result.dtype == bool:
                return result
            else:
                return result > 0

        if isinstance(result, dict):
            # Priority keys for entry signals
            priority_keys = ["long_entry", "entry", "signal", "cross_above"]
            for key in priority_keys:
                if key in result and isinstance(result[key], pd.Series):
                    if result[key].dtype == bool:
                        return result[key]

            # Fallback: first boolean Series in the dict
            for value in result.values():
                if isinstance(value, pd.Series) and value.dtype == bool:
                    return value

        return None

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate Confluence parameters.

        Rules:
            - conditions must be a list with at least 2 entries.
            - min_conditions must be between 1 and len(conditions).
        """
        conditions = params.get("conditions", [])
        min_conditions = params.get("min_conditions", len(conditions))

        if not isinstance(conditions, list):
            return False
        if len(conditions) < 2:
            return False
        if int(min_conditions) < 1 or int(min_conditions) > len(conditions):
            return False

        return True
