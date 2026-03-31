"""
Breakeven Stop Block.

Moves the stop loss to the entry price (breakeven) once the trade
has moved a specified number of pips into profit.

This locks in a risk-free trade once a minimum profit threshold
is reached, protecting against reversals that would turn a
winning trade into a loser.

An optional offset_pips parameter allows setting the breakeven
stop slightly above entry to cover commission costs.

The block returns configuration values. The execution layer
monitors open positions and adjusts the stop when activation
conditions are met.

Default Parameters:
    activation_pips: 20.0
    offset_pips: 2.0
"""

from typing import Any, Dict

import pandas as pd

from strategies_library.base import BaseBlock, BlockMetadata
from strategies_library.registry import BlockRegistry


@BlockRegistry.register
class BreakevenStop(BaseBlock):
    """Move stop loss to breakeven after specified profit threshold."""

    metadata = BlockMetadata(
        name="BreakevenStop",
        category="exit",
        description="Move stop loss to entry price after reaching a profit threshold",
        complexity=2,
        typical_use=["risk_management", "protection"],
        required_columns=[],
        version="1.0.0",
        tags=["breakeven", "stop_loss", "protection", "exit"],
    )

    def compute(
        self,
        data: pd.DataFrame,
        params: Dict[str, Any] | None = None,
    ) -> Dict[str, float]:
        """Return breakeven stop configuration.

        Args:
            data: OHLCV DataFrame (not used directly, but required by interface).
            params: Optional dict with:
                - activation_pips (float): Pips of profit required to activate. Default 20.0.
                - offset_pips (float): Pips above entry for the breakeven level. Default 2.0.

        Returns:
            Dict with keys:
                - 'activation_pips': Profit threshold to activate breakeven
                - 'offset_pips': Offset above/below entry for commission coverage
        """
        params = params or {}
        activation_pips = float(params.get("activation_pips", 20.0))
        offset_pips = float(params.get("offset_pips", 2.0))

        return {
            "activation_pips": activation_pips,
            "offset_pips": offset_pips,
        }

    def validate_params(self, params: Dict[str, Any]) -> bool:
        """Validate BreakevenStop parameters.

        Rules:
            - activation_pips must be positive and <= 200.
            - offset_pips must be non-negative and less than activation_pips.
        """
        activation = float(params.get("activation_pips", 20.0))
        offset = float(params.get("offset_pips", 2.0))

        if activation <= 0 or activation > 200:
            return False
        if offset < 0 or offset >= activation:
            return False

        return True
