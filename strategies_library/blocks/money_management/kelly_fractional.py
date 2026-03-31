from typing import Dict, Any
import pandas as pd
from strategies_library.base import BaseBlock, BlockMetadata
from strategies_library.registry import BlockRegistry


@BlockRegistry.register
class KellyFractional(BaseBlock):
    """Fractional Kelly Criterion Position Sizing.

    Uses the Kelly formula to determine optimal bet size:
        kelly_fraction = (win_prob * avg_win - (1 - win_prob) * avg_loss) / avg_win

    Then applies a fraction (typically 0.25 to 0.5) of the full Kelly
    for safety (full Kelly is too aggressive for most traders).

    Requires historical trade statistics to compute.

    Default Parameters:
        kelly_fraction: 0.25    (quarter Kelly — conservative)
        min_risk_pct: 0.5       (minimum 0.5% risk)
        max_risk_pct: 3.0       (maximum 3% risk)
        win_rate: 0.55          (historical win rate)
        avg_win_loss_ratio: 1.5 (average win / average loss)
    """

    metadata = BlockMetadata(
        name="KellyFractional",
        category="money_management",
        description="Fractional Kelly Criterion position sizing",
        complexity=4,
        typical_use=["professional", "optimal_sizing"],
        required_columns=[],
        tags=["kelly", "optimal", "professional"]
    )

    def compute(self, data: pd.DataFrame, params: Dict[str, Any] = None) -> dict:
        params = params or {}
        kelly_fraction = float(params.get("kelly_fraction", 0.25))
        min_risk_pct = float(params.get("min_risk_pct", 0.5))
        max_risk_pct = float(params.get("max_risk_pct", 3.0))
        win_rate = float(params.get("win_rate", 0.55))
        avg_win_loss_ratio = float(params.get("avg_win_loss_ratio", 1.5))

        # Full Kelly calculation
        # f* = (p * b - q) / b
        # where p = win probability, q = loss probability, b = win/loss ratio
        p = win_rate
        q = 1 - p
        b = avg_win_loss_ratio

        full_kelly = (p * b - q) / b if b > 0 else 0
        full_kelly = max(full_kelly, 0)

        # Apply fraction
        adjusted_kelly = full_kelly * kelly_fraction

        # Clamp to safe range
        risk_pct = max(min_risk_pct, min(adjusted_kelly * 100, max_risk_pct))

        return {
            "full_kelly_pct": full_kelly * 100,
            "adjusted_kelly_pct": adjusted_kelly * 100,
            "risk_pct": risk_pct,
        }

    def validate_params(self, params: Dict[str, Any]) -> bool:
        win_rate = params.get("win_rate", 0.55)
        kelly_fraction = params.get("kelly_fraction", 0.25)
        if win_rate <= 0 or win_rate >= 1.0:
            return False
        if kelly_fraction <= 0 or kelly_fraction > 1.0:
            return False
        return True
