"""
Fixed Take-Profit / Stop-Loss exit rule block.

Computes per-bar take-profit and stop-loss levels using ATR multiples.
"""

from __future__ import annotations

import pandas as pd

from forgequant.blocks._utils import _compute_atr
from forgequant.blocks.base import BaseBlock
from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory, BlockParams, BlockResult, SIGNAL_COLUMNS as SC


@BlockRegistry.register
class FixedTPSLExit(BaseBlock):
    """Fixed take-profit and stop-loss based on ATR multiples."""

    metadata = BlockMetadata(
        name="fixed_tpsl",
        display_name="Fixed TP/SL",
        category=BlockCategory.EXIT_RULE,
        description=(
            "Computes take-profit and stop-loss levels for each bar using "
            "ATR multiples. Levels adapt to current market volatility. "
            "Rejects configurations with a risk-reward ratio below the "
            "configured minimum."
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
                name="tp_atr_mult",
                param_type="float",
                default=3.0,
                min_value=0.1,
                max_value=20.0,
                description="Take-profit distance as multiple of ATR",
            ),
            ParameterSpec(
                name="sl_atr_mult",
                param_type="float",
                default=1.5,
                min_value=0.1,
                max_value=20.0,
                description="Stop-loss distance as multiple of ATR",
            ),
            ParameterSpec(
                name="min_rr",
                param_type="float",
                default=0.5,
                min_value=0.0,
                max_value=10.0,
                description=(
                    "Minimum acceptable risk-reward ratio (tp_mult / sl_mult). "
                    "Set to 0 to disable this check."
                ),
            ),
        ),
        tags=("exit", "stop_loss", "take_profit", "atr", "risk_reward", "fixed"),
        typical_use=(
            "Standard exit mechanism for trend-following or breakout systems. "
            "Use wider ATR multiples (3x TP, 1.5x SL) for trending markets, "
            "tighter for mean-reversion. Always pair with a trailing stop or "
            "time-based exit for additional protection."
        ),
    )

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        atr_period: int = params["atr_period"]
        tp_mult: float = params["tp_atr_mult"]
        sl_mult: float = params["sl_atr_mult"]
        min_rr: float = params["min_rr"]

        risk_reward = tp_mult / sl_mult

        if min_rr > 0 and risk_reward < min_rr:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=(
                    f"Risk-reward ratio {risk_reward:.2f} "
                    f"(tp={tp_mult}/sl={sl_mult}) is below minimum {min_rr:.2f}"
                ),
            )

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
        atr = _compute_atr(data, atr_period)

        tp_distance = atr * tp_mult
        sl_distance = atr * sl_mult

        result = pd.DataFrame(
            {
                "tpsl_atr": atr,
                SC.tpsl_long_tp: close + tp_distance,
                SC.tpsl_long_sl: close - sl_distance,
                SC.tpsl_short_tp: close - tp_distance,
                SC.tpsl_short_sl: close + sl_distance,
                "tpsl_risk_reward": risk_reward,
            },
            index=data.index,
        )

        return result
