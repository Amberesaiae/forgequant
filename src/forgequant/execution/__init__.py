"""
Live execution layer (MetaTrader 5 via aiomql).

Provides async order management, position tracking, signal-to-order
translation, and risk checks for live trading.
"""

from forgequant.execution.mt5_client import MT5Client, MT5Config
from forgequant.execution.order_manager import OrderManager, OrderResult
from forgequant.execution.position_tracker import PositionTracker, PositionRecord
from forgequant.execution.signal_translator import SignalTranslator, TradeSignal

__all__ = [
    "MT5Client",
    "MT5Config",
    "OrderManager",
    "OrderResult",
    "PositionTracker",
    "PositionRecord",
    "SignalTranslator",
    "TradeSignal",
]
