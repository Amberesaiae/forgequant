"""
ForgeQuant Indicator Building Blocks.

8 technical indicators covering:
- Trend: EMA, Ichimoku
- Momentum: RSI, MACD, Stochastic
- Volatility: ATR, BollingerBands
- Trend Strength: ADX

All indicators return either a pd.Series or a dict of pd.Series.
All accept a standard OHLCV DataFrame and optional params dict.
"""

from .ema import EMA
from .rsi import RSI
from .atr import ATR
from .bollinger_bands import BollingerBands
from .macd import MACD
from .adx import ADX
from .stochastic import Stochastic
from .ichimoku import Ichimoku

__all__ = [
    "EMA",
    "RSI",
    "ATR",
    "BollingerBands",
    "MACD",
    "ADX",
    "Stochastic",
    "Ichimoku",
]
