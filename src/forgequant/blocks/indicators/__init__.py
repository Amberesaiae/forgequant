"""
Technical indicator blocks.

Provides:
    - EMAIndicator: Exponential Moving Average
    - RSIIndicator: Relative Strength Index
    - MACDIndicator: Moving Average Convergence Divergence
    - ADXIndicator: Average Directional Index
    - ATRIndicator: Average True Range
    - BollingerBandsIndicator: Bollinger Bands
    - IchimokuIndicator: Ichimoku Kinko Hyo
    - StochasticIndicator: Stochastic Oscillator

All blocks are auto-registered with the BlockRegistry upon import.
"""

from forgequant.blocks.indicators.ema import EMAIndicator
from forgequant.blocks.indicators.rsi import RSIIndicator
from forgequant.blocks.indicators.macd import MACDIndicator
from forgequant.blocks.indicators.adx import ADXIndicator
from forgequant.blocks.indicators.atr import ATRIndicator
from forgequant.blocks.indicators.bollinger_bands import BollingerBandsIndicator
from forgequant.blocks.indicators.ichimoku import IchimokuIndicator
from forgequant.blocks.indicators.stochastic import StochasticIndicator

__all__ = [
    "EMAIndicator",
    "RSIIndicator",
    "MACDIndicator",
    "ADXIndicator",
    "ATRIndicator",
    "BollingerBandsIndicator",
    "IchimokuIndicator",
    "StochasticIndicator",
]
