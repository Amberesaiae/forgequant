from typing import Dict, Any
import pandas as pd
import numpy as np
from strategies_library.base import BaseBlock, BlockMetadata
from strategies_library.registry import BlockRegistry


@BlockRegistry.register
class VolatilityTargeting(BaseBlock):
    """Volatility-Targeted Position Sizing.

    Scales position size inversely to current realized volatility.
    When volatility is high, position size decreases.
    When volatility is low, position size increases.

    This produces more consistent risk-adjusted returns.

    vol_scalar = target_annual_vol / realized_annual_vol
    Capped between min_scalar and max_scalar to prevent extremes.

    Default Parameters:
        target_vol: 0.15       (15% annualized)
        lookback: 60           (trading days for vol estimation)
        min_scalar: 0.2
        max_scalar: 2.0
        annualization_factor: 252
    """

    metadata = BlockMetadata(
        name="VolatilityTargeting",
        category="money_management",
        description="Position sizing scaled inversely to volatility",
        complexity=4,
        typical_use=["risk_management", "professional"],
        required_columns=["close"],
        tags=["volatility", "targeting", "professional"]
    )

    def compute(self, data: pd.DataFrame, params: Dict[str, Any] = None) -> pd.Series:
        params = params or {}
        target_vol = float(params.get("target_vol", 0.15))
        lookback = int(params.get("lookback", 60))
        min_scalar = float(params.get("min_scalar", 0.2))
        max_scalar = float(params.get("max_scalar", 2.0))
        annualization = int(params.get("annualization_factor", 252))

        returns = data["close"].pct_change()
        realized_vol = returns.rolling(window=lookback).std() * np.sqrt(annualization)

        # Avoid division by zero
        realized_vol = realized_vol.replace(0, np.nan).ffill().fillna(target_vol)

        vol_scalar = target_vol / realized_vol
        vol_scalar = vol_scalar.clip(lower=min_scalar, upper=max_scalar)

        return vol_scalar

    def validate_params(self, params: Dict[str, Any]) -> bool:
        target_vol = params.get("target_vol", 0.15)
        lookback = params.get("lookback", 60)
        if target_vol <= 0 or target_vol > 0.5:
            return False
        if lookback < 10 or lookback > 252:
            return False
        return True
