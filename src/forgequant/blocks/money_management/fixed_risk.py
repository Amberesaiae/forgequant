"""
Fixed Risk position sizing block.

Risk a fixed percentage of account equity per trade. Position size is
determined by risk amount divided by stop distance (ATR-based).
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
class FixedRiskSizing(BaseBlock):
    """Fixed percentage risk per trade position sizing."""

    metadata = BlockMetadata(
        name="fixed_risk",
        display_name="Fixed Risk Sizing",
        category=BlockCategory.MONEY_MANAGEMENT,
        description=(
            "Sizes positions so that a fixed percentage of account equity "
            "is risked per trade. The position size adapts to volatility "
            "through ATR-based stop distance calculation."
        ),
        parameters=(
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
                name="sl_atr_mult",
                param_type="float",
                default=1.5,
                min_value=0.1,
                max_value=20.0,
                description="Stop-loss distance as ATR multiple",
            ),
            ParameterSpec(
                name="atr_period",
                param_type="int",
                default=14,
                min_value=1,
                max_value=200,
                description="ATR calculation period",
            ),
        ),
        tags=("sizing", "risk", "fixed", "percentage", "money_management"),
        typical_use=(
            "The workhorse of position sizing. Risk 1-2% of equity per trade "
            "for conservative systems, up to 5% for aggressive. Always "
            "combined with a stop-loss exit rule whose ATR multiplier "
            "matches sl_atr_mult."
        ),
    )

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        risk_pct: float = params["risk_pct"]
        equity: float = params["account_equity"]
        sl_mult: float = params["sl_atr_mult"]
        atr_period: int = params["atr_period"]

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

        risk_amount = equity * risk_pct / 100.0
        stop_distance = atr * sl_mult

        position_size = risk_amount / stop_distance.replace(0, np.nan)
        position_pct = (position_size * close / equity) * 100.0

        result = pd.DataFrame(
            {
                "fr_atr": atr,
                "fr_stop_distance": stop_distance,
                "fr_risk_amount": risk_amount,
                "fr_position_size": position_size,
                "fr_position_pct": position_pct,
            },
            index=data.index,
        )

        return result
