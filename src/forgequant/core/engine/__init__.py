"""
Backtesting engine.

Provides vectorized trade simulation, performance metrics computation,
and structured result containers.
"""

from forgequant.core.engine.backtester import Backtester, BacktestConfig
from forgequant.core.engine.metrics import compute_metrics
from forgequant.core.engine.results import BacktestResult, TradeRecord

__all__ = [
    "Backtester",
    "BacktestConfig",
    "BacktestResult",
    "TradeRecord",
    "compute_metrics",
]
