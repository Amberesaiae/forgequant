"""
ATR-Based position sizing block.

Position size is inversely proportional to ATR, equalizing dollar risk
across different volatility regimes.
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
class ATRBasedSizing(BaseBlock):
    """ATR-inverse position sizing for volatility equalization."""

    metadata = BlockMetadata(
        name="atr_based_sizing",
        display_name="ATR-Based Sizing",
        category=BlockCategory.MONEY_MANAGEMENT,
        description=(
            "Sizes positions inversely proportional to ATR, equalizing "
            "dollar risk across different volatility regimes. Simpler "
            "than full volatility targeting but effective for single-asset "
            "strategies."
        ),
        parameters=(
            ParameterSpec(
                name="atr_period",
                param_type="int",
                default=14,
                min_value=1,
                max_value=200,
                description="ATR calculation period",
            ),
            ParameterSpec(
                name="risk_atr_mult",
                param_type="float",
                default=1.5,
                min_value=0.1,
                max_value=20.0,
                description="Risk distance as ATR multiple",
            ),
            ParameterSpec(
                name="risk_pct",
                param_type="float",
                default=1.0,
                min_value=0.01,
                max_value=10.0,
                description="Percentage of equity to risk per trade",
            ),
            ParameterSpec(
                name="account_equity",
                param_type="float",
                default=100000.0,
                min_value=100.0,
                max_value=1e12,
                description="Account equity in base currency",
            ),
            ParameterSpec(
                name="max_position_pct",
                param_type="float",
                default=20.0,
                min_value=1.0,
                max_value=100.0,
                description="Maximum position size as percentage of equity",
            ),
        ),
        tags=("sizing", "atr", "volatility", "inverse", "equalization"),
        typical_use=(
            "Good default position sizing for any single-instrument strategy. "
            "Risk 1% with 1.5x ATR stop for conservative sizing. The "
            "max_position_pct cap prevents oversized positions when ATR "
            "is unusually low."
        ),
    )

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        atr_period: int = params["atr_period"]
        risk_mult: float = params["risk_atr_mult"]
        risk_pct: float = params["risk_pct"]
        equity: float = params["account_equity"]
        max_pct: float = params["max_position_pct"]

        min_rows = atr_period + 1
        if len(data) < min_rows:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Insufficient data: need at least {min_rows} rows, "
                       f"got {len(data)}",
            )

        high = data["high"]
        low = data["low"]
        close = data["close"]
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        alpha = 1.0 / atr_period
        atr = true_range.ewm(alpha=alpha, adjust=False).mean()

        risk_per_unit = atr * risk_mult

        risk_amount = equity * risk_pct / 100.0
        position_size = risk_amount / risk_per_unit.replace(0, np.nan)

        position_value = position_size * close
        position_pct = (position_value / equity) * 100.0

        cap_mask = position_pct > max_pct
        if cap_mask.any():
            max_position_value = equity * max_pct / 100.0
            position_size = position_size.where(
                ~cap_mask,
                max_position_value / close.replace(0, np.nan),
            )
            position_value = position_size * close
            position_pct = (position_value / equity) * 100.0

        result = pd.DataFrame(
            {
                "atrs_atr": atr,
                "atrs_risk_per_unit": risk_per_unit,
                "atrs_position_size": position_size,
                "atrs_position_value": position_value,
                "atrs_position_pct": position_pct,
            },
            index=data.index,
        )

        return result
