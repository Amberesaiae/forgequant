"""
Fixed Take Profit and Stop Loss Block.

The simplest and most common exit mechanism.
Defines a fixed distance in pips for both take profit and stop loss
from the entry price.

The execution layer uses these values to set TP and SL orders
when placing trades.

Also computes the risk-reward ratio (TP / SL) as a quality metric.
Strategies with risk-reward < 1.0 are generally discouraged unless
they have very high win rates.

Default Parameters:
    tp_pips: 50.0
    sl_pips: 30.0
"""

from typing import Any, Dict

import pandas as pd

from strategies_library.base import BaseBlock, BlockMetadata
from strategies_library.registry import BlockRegistry


@BlockRegistry.register
class FixedTPSL(BaseBlock):
    """Fixed take profit and stop loss in pips."""

    metadata = BlockMetadata(
        name="FixedTPSL",
        category="exit",
        description="Fixed take profit and stop loss distances in pips",
        complexity=1,
        typical_use=["risk_management", "exit"],
        required_columns=[],
        version="1.0.0",
        tags=["tp", "sl", "fixed", "exit", "risk"],
    )

    def compute(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any] | None = None,
    ) -> Dict[str, float]:
        """Return TP and SL configuration.

        Args:
            data: OHLCV DataFrame (not used for this block, but required
                  by the interface for consistency).
            params: Optional dict with:
                - tp_pips (float): Take profit distance in pips. Default 50.0.
                - sl_pips (float): Stop loss distance in pips. Default 30.0.

        Returns:
            Dict with keys:
                - 'tp_pips': Take profit pips
                - 'sl_pips': Stop loss pips
                - 'risk_reward': TP / SL ratio
        """
        params = params or {}
        tp_pips = float(params.get("tp_pips", 50.0))
        sl_pips = float(params.get("sl_pips", 30.0))

        risk_reward = tp_pips / sl_pips if sl_pips > 0 else 0.0

        return {
            "tp_pips": tp_pips,
            "sl_pips": sl_pips,
            "risk_reward": risk_reward,
        }

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate FixedTPSL parameters.

        Rules:
            - tp_pips must be positive and <= 500.
            - sl_pips must be positive and <= 500.
            - Risk-reward ratio must be >= 0.5 (TP must be at least half of SL).
        """
        tp = float(params.get("tp_pips", 50.0))
        sl = float(params.get("sl_pips", 30.0))

        if tp <= 0 or tp > 500:
            return False
        if sl <= 0 or sl > 500:
            return False
        if tp / sl < 0.5:
            return False

        return True
