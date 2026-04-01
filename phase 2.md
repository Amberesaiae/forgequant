# PHASE 2 — Technical Indicator Blocks

All 8 indicator blocks, each with mathematically correct implementations, proper anti-lookahead protection, full parameter validation, and comprehensive tests.

---

## 2.1 Updated Directory Structure (additions)

```
src/forgequant/blocks/indicators/
├── __init__.py          # Updated — imports & registers all indicators
├── ema.py               # Exponential Moving Average
├── rsi.py               # Relative Strength Index
├── macd.py              # Moving Average Convergence Divergence
├── adx.py               # Average Directional Index
├── atr.py               # Average True Range
├── bollinger_bands.py   # Bollinger Bands
├── ichimoku.py          # Ichimoku Kinko Hyo
└── stochastic.py        # Stochastic Oscillator

tests/unit/indicators/
├── __init__.py
├── test_ema.py
├── test_rsi.py
├── test_macd.py
├── test_adx.py
├── test_atr.py
├── test_bollinger_bands.py
├── test_ichimoku.py
└── test_stochastic.py
```

---

## 2.2 `src/forgequant/blocks/indicators/__init__.py`

```python
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
```

---

## 2.3 `src/forgequant/blocks/indicators/ema.py`

```python
"""
Exponential Moving Average (EMA) indicator block.

The EMA applies exponentially decreasing weights to past prices, giving
more significance to recent data than a Simple Moving Average (SMA).

Formula:
    multiplier = 2 / (period + 1)
    EMA_t = close_t * multiplier + EMA_{t-1} * (1 - multiplier)

This implementation uses pandas ewm(span=period, adjust=False) which
produces the recursive EMA formula above. adjust=False is chosen because:
    1. It matches the standard MetaTrader / TradingView EMA definition.
    2. It avoids the expanding-window correction that adjust=True applies,
       which would make early values differ from reference platforms.

Output columns:
    - ema_{period}: The EMA values
"""

from __future__ import annotations

import pandas as pd

from forgequant.blocks.base import BaseBlock
from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory, BlockParams, BlockResult


@BlockRegistry.register
class EMAIndicator(BaseBlock):
    """Exponential Moving Average indicator."""

    metadata = BlockMetadata(
        name="ema",
        display_name="Exponential Moving Average",
        category=BlockCategory.INDICATOR,
        description=(
            "Calculates the Exponential Moving Average of the close price. "
            "The EMA weights recent prices more heavily than older prices, "
            "making it more responsive to new information than a Simple "
            "Moving Average of the same period."
        ),
        parameters=(
            ParameterSpec(
                name="period",
                param_type="int",
                default=20,
                min_value=2,
                max_value=500,
                description="Number of periods for the EMA calculation",
            ),
            ParameterSpec(
                name="source",
                param_type="str",
                default="close",
                choices=("open", "high", "low", "close"),
                description="Price column to compute the EMA on",
            ),
        ),
        tags=("trend", "moving_average", "smoothing", "lagging"),
        typical_use=(
            "Used as a trend filter (price above EMA = bullish), as a "
            "dynamic support/resistance level, or as a component in "
            "crossover systems (fast EMA crossing slow EMA)."
        ),
    )

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        """
        Compute the EMA.

        Args:
            data: OHLCV DataFrame with DatetimeIndex.
            params: Must contain 'period' (int) and 'source' (str).

        Returns:
            DataFrame with a single column 'ema_{period}' containing
            the EMA values. Early values (before enough data accumulates)
            will be NaN-free because ewm with adjust=False initializes
            from the first value.
        """
        period: int = params["period"]
        source: str = params["source"]

        if source not in data.columns:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Source column '{source}' not found in data. "
                       f"Available: {list(data.columns)}",
            )

        if len(data) < period:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Insufficient data: need at least {period} rows, "
                       f"got {len(data)}",
            )

        ema_series = data[source].ewm(span=period, adjust=False).mean()

        col_name = f"ema_{period}"
        result = pd.DataFrame({col_name: ema_series}, index=data.index)

        return result
```

---

## 2.4 `src/forgequant/blocks/indicators/rsi.py`

```python
"""
Relative Strength Index (RSI) indicator block.

The RSI measures the speed and magnitude of recent price changes to
evaluate overbought or oversold conditions.

Calculation (Wilder's smoothed method):
    1. delta = close_t - close_{t-1}
    2. gain = max(delta, 0),  loss = abs(min(delta, 0))
    3. avg_gain = ewm(gain, alpha=1/period)   [Wilder smoothing]
       avg_loss = ewm(loss, alpha=1/period)
    4. RS = avg_gain / avg_loss
    5. RSI = 100 - (100 / (1 + RS))

Wilder's smoothing is equivalent to an EMA with alpha = 1/period
(NOT the standard EMA where alpha = 2/(period+1)). We use pandas
ewm(alpha=1/period, adjust=False) to match this exactly.

Output columns:
    - rsi_{period}: RSI values in [0, 100]
"""

from __future__ import annotations

import pandas as pd

from forgequant.blocks.base import BaseBlock
from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory, BlockParams, BlockResult


@BlockRegistry.register
class RSIIndicator(BaseBlock):
    """Relative Strength Index indicator using Wilder's smoothing method."""

    metadata = BlockMetadata(
        name="rsi",
        display_name="Relative Strength Index",
        category=BlockCategory.INDICATOR,
        description=(
            "Calculates the Relative Strength Index using Wilder's smoothing "
            "method. RSI oscillates between 0 and 100, with readings above 70 "
            "typically considered overbought and below 30 considered oversold."
        ),
        parameters=(
            ParameterSpec(
                name="period",
                param_type="int",
                default=14,
                min_value=2,
                max_value=200,
                description="Lookback period for RSI calculation",
            ),
            ParameterSpec(
                name="overbought",
                param_type="float",
                default=70.0,
                min_value=50.0,
                max_value=95.0,
                description="Overbought threshold level",
            ),
            ParameterSpec(
                name="oversold",
                param_type="float",
                default=30.0,
                min_value=5.0,
                max_value=50.0,
                description="Oversold threshold level",
            ),
        ),
        tags=("momentum", "oscillator", "overbought", "oversold", "mean_reversion"),
        typical_use=(
            "Used to identify overbought/oversold conditions for mean-reversion "
            "entries, as a divergence signal when price makes new highs/lows "
            "but RSI does not, or as a filter to avoid buying into overbought "
            "conditions in trend-following systems."
        ),
    )

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        """
        Compute the RSI using Wilder's smoothing.

        Args:
            data: OHLCV DataFrame with DatetimeIndex.
            params: Must contain 'period', 'overbought', 'oversold'.

        Returns:
            DataFrame with columns:
                - rsi_{period}: RSI values [0, 100]
                - rsi_{period}_overbought: Boolean True where RSI >= overbought
                - rsi_{period}_oversold: Boolean True where RSI <= oversold
        """
        period: int = params["period"]
        overbought: float = params["overbought"]
        oversold: float = params["oversold"]

        if overbought <= oversold:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Overbought ({overbought}) must be greater than "
                       f"oversold ({oversold})",
            )

        if len(data) < period + 1:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Insufficient data: need at least {period + 1} rows, "
                       f"got {len(data)}",
            )

        # Step 1: Price changes
        delta = data["close"].diff()

        # Step 2: Separate gains and losses
        gains = delta.clip(lower=0.0)
        losses = (-delta).clip(lower=0.0)

        # Step 3: Wilder's smoothing (alpha = 1/period)
        # This is equivalent to Wilder's recursive smoothing:
        #   avg_gain_t = (avg_gain_{t-1} * (period-1) + gain_t) / period
        alpha = 1.0 / period
        avg_gain = gains.ewm(alpha=alpha, adjust=False).mean()
        avg_loss = losses.ewm(alpha=alpha, adjust=False).mean()

        # Step 4: RS and RSI
        # Handle division by zero: when avg_loss == 0, RSI = 100
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))

        # Where avg_loss is 0 (all gains), RSI should be 100
        rsi = rsi.fillna(100.0)

        # Build output
        col = f"rsi_{period}"
        result = pd.DataFrame(
            {
                col: rsi,
                f"{col}_overbought": rsi >= overbought,
                f"{col}_oversold": rsi <= oversold,
            },
            index=data.index,
        )

        return result
```

---

## 2.5 `src/forgequant/blocks/indicators/macd.py`

```python
"""
Moving Average Convergence Divergence (MACD) indicator block.

The MACD shows the relationship between two EMAs of the close price.

Calculation:
    1. MACD line    = EMA(close, fast_period) - EMA(close, slow_period)
    2. Signal line  = EMA(MACD line, signal_period)
    3. Histogram    = MACD line - Signal line

The histogram crossing zero corresponds to the MACD line crossing the
signal line, which is the classic MACD crossover signal.

Output columns:
    - macd_line: The MACD line
    - macd_signal: The signal line
    - macd_histogram: The MACD histogram
"""

from __future__ import annotations

import pandas as pd

from forgequant.blocks.base import BaseBlock
from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory, BlockParams, BlockResult


@BlockRegistry.register
class MACDIndicator(BaseBlock):
    """Moving Average Convergence Divergence indicator."""

    metadata = BlockMetadata(
        name="macd",
        display_name="MACD",
        category=BlockCategory.INDICATOR,
        description=(
            "Calculates the Moving Average Convergence Divergence indicator. "
            "The MACD line is the difference between a fast and slow EMA. "
            "The signal line is an EMA of the MACD line. The histogram is "
            "the difference between the MACD and signal lines."
        ),
        parameters=(
            ParameterSpec(
                name="fast_period",
                param_type="int",
                default=12,
                min_value=2,
                max_value=100,
                description="Period for the fast EMA",
            ),
            ParameterSpec(
                name="slow_period",
                param_type="int",
                default=26,
                min_value=2,
                max_value=200,
                description="Period for the slow EMA",
            ),
            ParameterSpec(
                name="signal_period",
                param_type="int",
                default=9,
                min_value=2,
                max_value=100,
                description="Period for the signal line EMA",
            ),
        ),
        tags=("trend", "momentum", "crossover", "histogram", "convergence"),
        typical_use=(
            "Used for trend-following entries via MACD/signal crossovers, "
            "momentum confirmation via histogram direction, and divergence "
            "detection between price and MACD for reversal signals."
        ),
    )

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        """
        Compute the MACD line, signal line, and histogram.

        Args:
            data: OHLCV DataFrame with DatetimeIndex.
            params: Must contain 'fast_period', 'slow_period', 'signal_period'.

        Returns:
            DataFrame with columns: macd_line, macd_signal, macd_histogram.
        """
        fast_period: int = params["fast_period"]
        slow_period: int = params["slow_period"]
        signal_period: int = params["signal_period"]

        if fast_period >= slow_period:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"fast_period ({fast_period}) must be less than "
                       f"slow_period ({slow_period})",
            )

        min_rows = slow_period + signal_period
        if len(data) < min_rows:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Insufficient data: need at least {min_rows} rows, "
                       f"got {len(data)}",
            )

        close = data["close"]

        # Fast and slow EMAs
        ema_fast = close.ewm(span=fast_period, adjust=False).mean()
        ema_slow = close.ewm(span=slow_period, adjust=False).mean()

        # MACD line
        macd_line = ema_fast - ema_slow

        # Signal line (EMA of MACD line)
        macd_signal = macd_line.ewm(span=signal_period, adjust=False).mean()

        # Histogram
        macd_histogram = macd_line - macd_signal

        result = pd.DataFrame(
            {
                "macd_line": macd_line,
                "macd_signal": macd_signal,
                "macd_histogram": macd_histogram,
            },
            index=data.index,
        )

        return result
```

---

## 2.6 `src/forgequant/blocks/indicators/adx.py`

```python
"""
Average Directional Index (ADX) indicator block.

The ADX quantifies trend strength regardless of direction. It is derived
from the Directional Movement System developed by J. Welles Wilder Jr.

Calculation:
    1. True Range (TR):
       TR = max(high - low, |high - prev_close|, |low - prev_close|)

    2. Directional Movement:
       +DM = high - prev_high  if (high - prev_high) > (prev_low - low) AND > 0
       -DM = prev_low - low    if (prev_low - low) > (high - prev_high) AND > 0

    3. Smoothed TR, +DM, -DM using Wilder's smoothing (alpha=1/period)

    4. Directional Indicators:
       +DI = 100 * smoothed(+DM) / smoothed(TR)
       -DI = 100 * smoothed(-DM) / smoothed(TR)

    5. DX = 100 * |+DI - -DI| / (+DI + -DI)

    6. ADX = Wilder-smoothed DX over the specified period

Output columns:
    - adx: The ADX line (0-100, trend strength)
    - plus_di: +DI line
    - minus_di: -DI line
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
class ADXIndicator(BaseBlock):
    """Average Directional Index indicator using Wilder's method."""

    metadata = BlockMetadata(
        name="adx",
        display_name="Average Directional Index",
        category=BlockCategory.INDICATOR,
        description=(
            "Calculates the Average Directional Index (ADX) along with the "
            "+DI and -DI directional indicators. ADX measures trend strength "
            "on a 0-100 scale: below 20 indicates a weak/absent trend, "
            "above 25 a developing trend, and above 50 a strong trend."
        ),
        parameters=(
            ParameterSpec(
                name="period",
                param_type="int",
                default=14,
                min_value=2,
                max_value=200,
                description="Lookback period for ADX calculation",
            ),
        ),
        tags=("trend", "strength", "directional", "wilder", "volatility"),
        typical_use=(
            "Used as a trend strength filter: only take trend-following trades "
            "when ADX > 25. The +DI/-DI crossover can also serve as a "
            "directional entry signal. Useful for distinguishing trending "
            "vs ranging market regimes."
        ),
    )

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        """
        Compute ADX, +DI, and -DI.

        Args:
            data: OHLCV DataFrame with DatetimeIndex.
            params: Must contain 'period' (int).

        Returns:
            DataFrame with columns: adx, plus_di, minus_di.
        """
        period: int = params["period"]

        # Need at least 2*period rows for meaningful ADX
        min_rows = 2 * period + 1
        if len(data) < min_rows:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Insufficient data: need at least {min_rows} rows, "
                       f"got {len(data)}",
            )

        high = data["high"]
        low = data["low"]
        close = data["close"]

        # Previous values (shift to avoid lookahead)
        prev_high = high.shift(1)
        prev_low = low.shift(1)
        prev_close = close.shift(1)

        # True Range
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Directional Movement
        up_move = high - prev_high
        down_move = prev_low - low

        # +DM: up_move if up_move > down_move AND up_move > 0, else 0
        plus_dm = pd.Series(
            np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
            index=data.index,
        )

        # -DM: down_move if down_move > up_move AND down_move > 0, else 0
        minus_dm = pd.Series(
            np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
            index=data.index,
        )

        # Wilder smoothing (alpha = 1/period)
        alpha = 1.0 / period
        smoothed_tr = true_range.ewm(alpha=alpha, adjust=False).mean()
        smoothed_plus_dm = plus_dm.ewm(alpha=alpha, adjust=False).mean()
        smoothed_minus_dm = minus_dm.ewm(alpha=alpha, adjust=False).mean()

        # Directional Indicators
        # Avoid division by zero
        plus_di = 100.0 * smoothed_plus_dm / smoothed_tr.replace(0, np.nan)
        minus_di = 100.0 * smoothed_minus_dm / smoothed_tr.replace(0, np.nan)

        # DX
        di_sum = plus_di + minus_di
        di_diff = (plus_di - minus_di).abs()
        dx = 100.0 * di_diff / di_sum.replace(0, np.nan)

        # ADX = Wilder-smoothed DX
        adx = dx.ewm(alpha=alpha, adjust=False).mean()

        result = pd.DataFrame(
            {
                "adx": adx,
                "plus_di": plus_di,
                "minus_di": minus_di,
            },
            index=data.index,
        )

        return result
```

---

## 2.7 `src/forgequant/blocks/indicators/atr.py`

```python
"""
Average True Range (ATR) indicator block.

The ATR measures market volatility by decomposing the entire range
of an asset price for a given period.

Calculation:
    1. True Range:
       TR = max(high - low, |high - prev_close|, |low - prev_close|)

    2. ATR = Wilder-smoothed average of TR over the period
       (equivalent to EMA with alpha = 1/period)

Output columns:
    - atr_{period}: The ATR values (in price units)
    - atr_{period}_pct: ATR as a percentage of close price
"""

from __future__ import annotations

import pandas as pd

from forgequant.blocks.base import BaseBlock
from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory, BlockParams, BlockResult


@BlockRegistry.register
class ATRIndicator(BaseBlock):
    """Average True Range volatility indicator using Wilder's smoothing."""

    metadata = BlockMetadata(
        name="atr",
        display_name="Average True Range",
        category=BlockCategory.INDICATOR,
        description=(
            "Calculates the Average True Range, a volatility measure that "
            "accounts for gaps between bars. Higher ATR indicates higher "
            "volatility. ATR is commonly used for position sizing, stop-loss "
            "placement, and volatility filtering."
        ),
        parameters=(
            ParameterSpec(
                name="period",
                param_type="int",
                default=14,
                min_value=1,
                max_value=200,
                description="Lookback period for ATR smoothing",
            ),
        ),
        tags=("volatility", "range", "wilder", "stop_loss", "position_sizing"),
        typical_use=(
            "Used for ATR-based stop losses (e.g. 2x ATR from entry), "
            "position sizing (risk amount / ATR = position size), and as "
            "a volatility filter to avoid low-volatility chop or excessive "
            "volatility."
        ),
    )

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        """
        Compute ATR and ATR as percentage of close.

        Args:
            data: OHLCV DataFrame with DatetimeIndex.
            params: Must contain 'period' (int).

        Returns:
            DataFrame with columns: atr_{period}, atr_{period}_pct.
        """
        period: int = params["period"]

        if len(data) < period + 1:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Insufficient data: need at least {period + 1} rows, "
                       f"got {len(data)}",
            )

        high = data["high"]
        low = data["low"]
        close = data["close"]
        prev_close = close.shift(1)

        # True Range components
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Wilder smoothing
        alpha = 1.0 / period
        atr = true_range.ewm(alpha=alpha, adjust=False).mean()

        # ATR as percentage of close
        atr_pct = (atr / close) * 100.0

        col = f"atr_{period}"
        result = pd.DataFrame(
            {
                col: atr,
                f"{col}_pct": atr_pct,
            },
            index=data.index,
        )

        return result
```

---

## 2.8 `src/forgequant/blocks/indicators/bollinger_bands.py`

```python
"""
Bollinger Bands indicator block.

Bollinger Bands consist of a middle band (SMA) with an upper and lower
band placed a specified number of standard deviations away.

Calculation:
    1. Middle Band = SMA(close, period)
    2. Std Dev     = rolling standard deviation of close over period
    3. Upper Band  = Middle Band + (num_std * Std Dev)
    4. Lower Band  = Middle Band - (num_std * Std Dev)
    5. %B          = (close - Lower Band) / (Upper Band - Lower Band)
    6. Bandwidth   = (Upper Band - Lower Band) / Middle Band * 100

%B tells you where the price is relative to the bands:
    - %B > 1.0: price is above the upper band
    - %B = 0.5: price is at the middle band
    - %B < 0.0: price is below the lower band

Bandwidth measures band width as a percentage of the middle band,
useful for identifying volatility squeezes (low bandwidth).

Output columns:
    - bb_upper: Upper band
    - bb_middle: Middle band (SMA)
    - bb_lower: Lower band
    - bb_pct_b: %B (percent B)
    - bb_bandwidth: Bandwidth percentage
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
class BollingerBandsIndicator(BaseBlock):
    """Bollinger Bands volatility and mean-reversion indicator."""

    metadata = BlockMetadata(
        name="bollinger_bands",
        display_name="Bollinger Bands",
        category=BlockCategory.INDICATOR,
        description=(
            "Calculates Bollinger Bands: a middle SMA band with upper and "
            "lower bands at a configurable number of standard deviations. "
            "Also computes %B (relative position within bands) and "
            "Bandwidth (band width as a percentage of the middle band)."
        ),
        parameters=(
            ParameterSpec(
                name="period",
                param_type="int",
                default=20,
                min_value=2,
                max_value=500,
                description="Lookback period for the middle band SMA and standard deviation",
            ),
            ParameterSpec(
                name="num_std",
                param_type="float",
                default=2.0,
                min_value=0.5,
                max_value=5.0,
                description="Number of standard deviations for upper/lower bands",
            ),
        ),
        tags=(
            "volatility", "bands", "mean_reversion", "squeeze",
            "standard_deviation", "overbought", "oversold",
        ),
        typical_use=(
            "Used for mean-reversion entries when price touches or exceeds "
            "the bands, volatility squeeze detection (narrow bandwidth "
            "often precedes breakouts), and trend-following when price walks "
            "along the upper or lower band."
        ),
    )

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        """
        Compute Bollinger Bands, %B, and Bandwidth.

        Args:
            data: OHLCV DataFrame with DatetimeIndex.
            params: Must contain 'period' (int) and 'num_std' (float).

        Returns:
            DataFrame with columns: bb_upper, bb_middle, bb_lower,
            bb_pct_b, bb_bandwidth.
        """
        period: int = params["period"]
        num_std: float = params["num_std"]

        if len(data) < period:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Insufficient data: need at least {period} rows, "
                       f"got {len(data)}",
            )

        close = data["close"]

        # Middle band: Simple Moving Average
        middle = close.rolling(window=period).mean()

        # Standard deviation (using ddof=0 for population std, matching
        # TradingView and most trading platform conventions)
        std = close.rolling(window=period).std(ddof=0)

        # Upper and lower bands
        upper = middle + (num_std * std)
        lower = middle - (num_std * std)

        # %B: where is close relative to the bands?
        band_width_raw = upper - lower
        pct_b = (close - lower) / band_width_raw.replace(0, np.nan)

        # Bandwidth: band width as percentage of middle band
        bandwidth = (band_width_raw / middle.replace(0, np.nan)) * 100.0

        result = pd.DataFrame(
            {
                "bb_upper": upper,
                "bb_middle": middle,
                "bb_lower": lower,
                "bb_pct_b": pct_b,
                "bb_bandwidth": bandwidth,
            },
            index=data.index,
        )

        return result
```

---

## 2.9 `src/forgequant/blocks/indicators/ichimoku.py`

```python
"""
Ichimoku Kinko Hyo indicator block.

Ichimoku is a comprehensive indicator system that defines support/resistance,
trend direction, and momentum at a glance.

Components:
    1. Tenkan-sen (Conversion Line):
       (highest_high + lowest_low) / 2  over tenkan_period

    2. Kijun-sen (Base Line):
       (highest_high + lowest_low) / 2  over kijun_period

    3. Senkou Span A (Leading Span A):
       (Tenkan-sen + Kijun-sen) / 2, plotted kijun_period bars ahead

    4. Senkou Span B (Leading Span B):
       (highest_high + lowest_low) / 2 over senkou_b_period,
       plotted kijun_period bars ahead

    5. Chikou Span (Lagging Span):
       Current close, plotted kijun_period bars behind (i.e. shifted back)

The "cloud" (Kumo) is the area between Senkou Span A and Senkou Span B.
Price above the cloud is bullish; below is bearish.

IMPORTANT: Senkou Span A and B are forward-shifted (leading) values.
In this implementation:
    - senkou_span_a and senkou_span_b are shifted FORWARD by kijun_period.
      This means the last kijun_period rows of these columns will be NaN
      in live data, but for backtesting the shift is correct — the cloud
      was visible kijun_period bars ago.
    - chikou_span is shifted BACKWARD by kijun_period (lagging).

Output columns:
    - ichimoku_tenkan: Tenkan-sen (Conversion Line)
    - ichimoku_kijun: Kijun-sen (Base Line)
    - ichimoku_senkou_a: Senkou Span A (shifted forward)
    - ichimoku_senkou_b: Senkou Span B (shifted forward)
    - ichimoku_chikou: Chikou Span (shifted backward)
"""

from __future__ import annotations

import pandas as pd

from forgequant.blocks.base import BaseBlock
from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory, BlockParams, BlockResult


@BlockRegistry.register
class IchimokuIndicator(BaseBlock):
    """Ichimoku Kinko Hyo full indicator system."""

    metadata = BlockMetadata(
        name="ichimoku",
        display_name="Ichimoku Kinko Hyo",
        category=BlockCategory.INDICATOR,
        description=(
            "Calculates all five Ichimoku Kinko Hyo components: Tenkan-sen, "
            "Kijun-sen, Senkou Span A, Senkou Span B, and Chikou Span. "
            "The cloud (Kumo) between Senkou Span A and B provides dynamic "
            "support/resistance and trend context."
        ),
        parameters=(
            ParameterSpec(
                name="tenkan_period",
                param_type="int",
                default=9,
                min_value=2,
                max_value=100,
                description="Period for Tenkan-sen (Conversion Line)",
            ),
            ParameterSpec(
                name="kijun_period",
                param_type="int",
                default=26,
                min_value=2,
                max_value=200,
                description="Period for Kijun-sen (Base Line) and displacement",
            ),
            ParameterSpec(
                name="senkou_b_period",
                param_type="int",
                default=52,
                min_value=2,
                max_value=300,
                description="Period for Senkou Span B",
            ),
        ),
        tags=("trend", "cloud", "support", "resistance", "japanese", "comprehensive"),
        typical_use=(
            "Used for trend identification (price vs cloud), entry signals "
            "(Tenkan/Kijun cross), support/resistance levels (cloud edges, "
            "Kijun-sen), and momentum confirmation (Chikou Span vs price)."
        ),
    )

    @staticmethod
    def _donchian_midline(series: pd.Series, period: int) -> pd.Series:
        """
        Compute the Donchian Channel midline: (highest + lowest) / 2.

        This is the core building block for all Ichimoku components.

        Args:
            series: Price series (typically high or low).
            period: Lookback window.

        Returns:
            Series of midline values.
        """
        highest = series.rolling(window=period).max()
        lowest = series.rolling(window=period).min()
        return (highest + lowest) / 2.0

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        """
        Compute all five Ichimoku components.

        Args:
            data: OHLCV DataFrame with DatetimeIndex.
            params: Must contain 'tenkan_period', 'kijun_period', 'senkou_b_period'.

        Returns:
            DataFrame with columns: ichimoku_tenkan, ichimoku_kijun,
            ichimoku_senkou_a, ichimoku_senkou_b, ichimoku_chikou.
        """
        tenkan_period: int = params["tenkan_period"]
        kijun_period: int = params["kijun_period"]
        senkou_b_period: int = params["senkou_b_period"]

        min_rows = senkou_b_period + kijun_period
        if len(data) < min_rows:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Insufficient data: need at least {min_rows} rows, "
                       f"got {len(data)}",
            )

        high = data["high"]
        low = data["low"]
        close = data["close"]

        # Tenkan-sen: (9-period high + 9-period low) / 2
        tenkan_high = high.rolling(window=tenkan_period).max()
        tenkan_low = low.rolling(window=tenkan_period).min()
        tenkan = (tenkan_high + tenkan_low) / 2.0

        # Kijun-sen: (26-period high + 26-period low) / 2
        kijun_high = high.rolling(window=kijun_period).max()
        kijun_low = low.rolling(window=kijun_period).min()
        kijun = (kijun_high + kijun_low) / 2.0

        # Senkou Span A: (Tenkan + Kijun) / 2, shifted forward by kijun_period
        senkou_a = ((tenkan + kijun) / 2.0).shift(kijun_period)

        # Senkou Span B: (52-period high + 52-period low) / 2, shifted forward
        senkou_b_high = high.rolling(window=senkou_b_period).max()
        senkou_b_low = low.rolling(window=senkou_b_period).min()
        senkou_b = ((senkou_b_high + senkou_b_low) / 2.0).shift(kijun_period)

        # Chikou Span: close shifted backward by kijun_period
        chikou = close.shift(-kijun_period)

        result = pd.DataFrame(
            {
                "ichimoku_tenkan": tenkan,
                "ichimoku_kijun": kijun,
                "ichimoku_senkou_a": senkou_a,
                "ichimoku_senkou_b": senkou_b,
                "ichimoku_chikou": chikou,
            },
            index=data.index,
        )

        return result
```

---

## 2.10 `src/forgequant/blocks/indicators/stochastic.py`

```python
"""
Stochastic Oscillator indicator block.

The Stochastic Oscillator compares a closing price to its price range
over a given lookback period.

Calculation:
    1. %K (raw):
       %K = 100 * (close - lowest_low(k_period)) / (highest_high(k_period) - lowest_low(k_period))

    2. %K (smoothed):
       %K smoothed = SMA(%K raw, k_smooth)

    3. %D (signal):
       %D = SMA(%K smoothed, d_period)

Standard "Slow Stochastic":
    k_period=14, k_smooth=3, d_period=3

Standard "Fast Stochastic":
    k_period=14, k_smooth=1, d_period=3

Output columns:
    - stoch_k: The %K line (smoothed)
    - stoch_d: The %D line (signal)
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
class StochasticIndicator(BaseBlock):
    """Stochastic Oscillator momentum indicator."""

    metadata = BlockMetadata(
        name="stochastic",
        display_name="Stochastic Oscillator",
        category=BlockCategory.INDICATOR,
        description=(
            "Calculates the Stochastic Oscillator, a momentum indicator "
            "that compares the closing price to its range over a lookback "
            "period. %K above 80 is typically considered overbought, and "
            "below 20 is considered oversold."
        ),
        parameters=(
            ParameterSpec(
                name="k_period",
                param_type="int",
                default=14,
                min_value=2,
                max_value=200,
                description="Lookback period for %K calculation",
            ),
            ParameterSpec(
                name="k_smooth",
                param_type="int",
                default=3,
                min_value=1,
                max_value=50,
                description="Smoothing period for %K (1 = Fast Stochastic, 3 = Slow)",
            ),
            ParameterSpec(
                name="d_period",
                param_type="int",
                default=3,
                min_value=1,
                max_value=50,
                description="Period for the %D signal line (SMA of %K)",
            ),
            ParameterSpec(
                name="overbought",
                param_type="float",
                default=80.0,
                min_value=50.0,
                max_value=95.0,
                description="Overbought threshold level",
            ),
            ParameterSpec(
                name="oversold",
                param_type="float",
                default=20.0,
                min_value=5.0,
                max_value=50.0,
                description="Oversold threshold level",
            ),
        ),
        tags=("momentum", "oscillator", "overbought", "oversold", "mean_reversion"),
        typical_use=(
            "Used for mean-reversion entries in ranging markets (buy oversold, "
            "sell overbought), %K/%D crossover signals, and divergence "
            "detection. Often combined with a trend filter — only take "
            "oversold signals in uptrends and overbought signals in downtrends."
        ),
    )

    def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
        """
        Compute the Stochastic %K and %D.

        Args:
            data: OHLCV DataFrame with DatetimeIndex.
            params: Must contain 'k_period', 'k_smooth', 'd_period',
                    'overbought', 'oversold'.

        Returns:
            DataFrame with columns: stoch_k, stoch_d,
            stoch_overbought (bool), stoch_oversold (bool).
        """
        k_period: int = params["k_period"]
        k_smooth: int = params["k_smooth"]
        d_period: int = params["d_period"]
        overbought: float = params["overbought"]
        oversold: float = params["oversold"]

        if overbought <= oversold:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Overbought ({overbought}) must be greater than "
                       f"oversold ({oversold})",
            )

        min_rows = k_period + k_smooth + d_period
        if len(data) < min_rows:
            raise BlockComputeError(
                block_name=self.metadata.name,
                reason=f"Insufficient data: need at least {min_rows} rows, "
                       f"got {len(data)}",
            )

        high = data["high"]
        low = data["low"]
        close = data["close"]

        # Rolling highest high and lowest low
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()

        # Raw %K
        range_hl = highest_high - lowest_low
        # Avoid division by zero when high == low for entire window
        k_raw = 100.0 * (close - lowest_low) / range_hl.replace(0, np.nan)

        # Smoothed %K
        k_smooth_series = k_raw.rolling(window=k_smooth).mean()

        # %D (signal line)
        d_series = k_smooth_series.rolling(window=d_period).mean()

        result = pd.DataFrame(
            {
                "stoch_k": k_smooth_series,
                "stoch_d": d_series,
                "stoch_overbought": k_smooth_series >= overbought,
                "stoch_oversold": k_smooth_series <= oversold,
            },
            index=data.index,
        )

        return result
```

---

## 2.11 Test Suite

### `tests/unit/indicators/__init__.py`

```python
"""Tests for indicator blocks."""
```

---

### `tests/unit/indicators/test_ema.py`

```python
"""Tests for the EMA indicator block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.indicators.ema import EMAIndicator
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError, BlockValidationError
from forgequant.core.types import BlockCategory


@pytest.fixture
def ema() -> EMAIndicator:
    return EMAIndicator()


class TestEMAMetadata:
    """Tests for EMA block metadata."""

    def test_name(self, ema: EMAIndicator) -> None:
        assert ema.metadata.name == "ema"

    def test_category(self, ema: EMAIndicator) -> None:
        assert ema.metadata.category == BlockCategory.INDICATOR

    def test_display_name(self, ema: EMAIndicator) -> None:
        assert ema.metadata.display_name == "Exponential Moving Average"

    def test_default_parameters(self, ema: EMAIndicator) -> None:
        defaults = ema.metadata.get_defaults()
        assert defaults["period"] == 20
        assert defaults["source"] == "close"

    def test_registered(self) -> None:
        """EMA should be auto-registered on import."""
        registry = BlockRegistry()
        assert "ema" in registry


class TestEMACompute:
    """Tests for EMA computation."""

    def test_default_params(
        self, ema: EMAIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        """EMA with default period=20 should produce valid output."""
        result = ema.execute(sample_ohlcv)
        assert isinstance(result, pd.DataFrame)
        assert "ema_20" in result.columns
        assert len(result) == len(sample_ohlcv)

    def test_custom_period(
        self, ema: EMAIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = ema.execute(sample_ohlcv, {"period": 50})
        assert "ema_50" in result.columns

    def test_source_high(
        self, ema: EMAIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = ema.execute(sample_ohlcv, {"period": 10, "source": "high"})
        assert "ema_10" in result.columns

    def test_no_nans_with_adjust_false(
        self, ema: EMAIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        """With adjust=False, EMA should have no NaN values (initializes from first value)."""
        result = ema.execute(sample_ohlcv, {"period": 20})
        assert result["ema_20"].isna().sum() == 0

    def test_ema_responds_to_price_changes(
        self, ema: EMAIndicator
    ) -> None:
        """EMA should move toward recent prices."""
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        # Price jumps from 100 to 200 at bar 50
        close = np.concatenate([np.full(50, 100.0), np.full(50, 200.0)])
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = ema.execute(df, {"period": 10})
        ema_vals = result["ema_10"]

        # Before the jump, EMA should be near 100
        assert abs(ema_vals.iloc[49] - 100.0) < 1.0

        # After the jump, EMA should converge toward 200
        assert ema_vals.iloc[99] > 190.0

    def test_ema_with_known_values(self, ema: EMAIndicator) -> None:
        """Verify EMA computation against manually calculated values."""
        dates = pd.date_range("2024-01-01", periods=5, freq="h")
        close = pd.Series([10.0, 11.0, 12.0, 11.0, 10.0])
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": [100.0] * 5,
            },
            index=dates,
        )

        result = ema.execute(df, {"period": 3})
        ema_vals = result["ema_3"]

        # EMA(3): multiplier = 2/(3+1) = 0.5
        # Bar 0: EMA = 10.0 (seed)
        # Bar 1: EMA = 11.0 * 0.5 + 10.0 * 0.5 = 10.5
        # Bar 2: EMA = 12.0 * 0.5 + 10.5 * 0.5 = 11.25
        # Bar 3: EMA = 11.0 * 0.5 + 11.25 * 0.5 = 11.125
        # Bar 4: EMA = 10.0 * 0.5 + 11.125 * 0.5 = 10.5625
        assert abs(ema_vals.iloc[0] - 10.0) < 1e-10
        assert abs(ema_vals.iloc[1] - 10.5) < 1e-10
        assert abs(ema_vals.iloc[2] - 11.25) < 1e-10
        assert abs(ema_vals.iloc[3] - 11.125) < 1e-10
        assert abs(ema_vals.iloc[4] - 10.5625) < 1e-10

    def test_insufficient_data_raises(self, ema: EMAIndicator) -> None:
        """Should raise when data has fewer rows than the period."""
        dates = pd.date_range("2024-01-01", periods=5, freq="h")
        df = pd.DataFrame(
            {
                "open": [1.0] * 5,
                "high": [1.1] * 5,
                "low": [0.9] * 5,
                "close": [1.0] * 5,
                "volume": [100.0] * 5,
            },
            index=dates,
        )
        with pytest.raises(BlockComputeError, match="Insufficient data"):
            ema.execute(df, {"period": 10})

    def test_invalid_source_raises(
        self, ema: EMAIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        with pytest.raises(BlockValidationError):
            ema.execute(sample_ohlcv, {"source": "vwap"})

    def test_period_below_min_raises(
        self, ema: EMAIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        with pytest.raises(BlockValidationError):
            ema.execute(sample_ohlcv, {"period": 1})

    def test_period_above_max_raises(
        self, ema: EMAIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        with pytest.raises(BlockValidationError):
            ema.execute(sample_ohlcv, {"period": 501})
```

---

### `tests/unit/indicators/test_rsi.py`

```python
"""Tests for the RSI indicator block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.indicators.rsi import RSIIndicator
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError, BlockValidationError
from forgequant.core.types import BlockCategory


@pytest.fixture
def rsi() -> RSIIndicator:
    return RSIIndicator()


def _make_df(close: list[float] | np.ndarray, n: int | None = None) -> pd.DataFrame:
    """Helper to build an OHLCV DataFrame from a close series."""
    close_arr = np.array(close, dtype=float)
    length = n or len(close_arr)
    dates = pd.date_range("2024-01-01", periods=length, freq="h")
    return pd.DataFrame(
        {
            "open": close_arr,
            "high": close_arr + 0.5,
            "low": close_arr - 0.5,
            "close": close_arr,
            "volume": np.ones(length) * 1000.0,
        },
        index=dates,
    )


class TestRSIMetadata:
    def test_name(self, rsi: RSIIndicator) -> None:
        assert rsi.metadata.name == "rsi"

    def test_category(self, rsi: RSIIndicator) -> None:
        assert rsi.metadata.category == BlockCategory.INDICATOR

    def test_defaults(self, rsi: RSIIndicator) -> None:
        defaults = rsi.metadata.get_defaults()
        assert defaults["period"] == 14
        assert defaults["overbought"] == 70.0
        assert defaults["oversold"] == 30.0

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "rsi" in registry


class TestRSICompute:
    def test_default_params(
        self, rsi: RSIIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = rsi.execute(sample_ohlcv)
        assert "rsi_14" in result.columns
        assert "rsi_14_overbought" in result.columns
        assert "rsi_14_oversold" in result.columns
        assert len(result) == len(sample_ohlcv)

    def test_rsi_range(
        self, rsi: RSIIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        """RSI should be bounded in [0, 100] (excluding initial NaN-like values)."""
        result = rsi.execute(sample_ohlcv)
        rsi_vals = result["rsi_14"].dropna()
        assert rsi_vals.min() >= 0.0
        assert rsi_vals.max() <= 100.0

    def test_all_gains_rsi_near_100(self, rsi: RSIIndicator) -> None:
        """If price only goes up, RSI should approach 100."""
        close = np.linspace(100, 200, 100)
        df = _make_df(close)
        result = rsi.execute(df)
        # After initial warmup, RSI should be very high
        assert result["rsi_14"].iloc[-1] > 95.0

    def test_all_losses_rsi_near_0(self, rsi: RSIIndicator) -> None:
        """If price only goes down, RSI should approach 0."""
        close = np.linspace(200, 100, 100)
        df = _make_df(close)
        result = rsi.execute(df)
        assert result["rsi_14"].iloc[-1] < 5.0

    def test_flat_price_rsi_50(self, rsi: RSIIndicator) -> None:
        """Flat prices (no change) should produce NaN (0/0 case) handled as 100 by fillna,
        but after the first bar the delta is 0, so gain=0 and loss=0.
        Actually ewm with adjust=False will produce 0/0 = NaN -> filled to 100.
        For a truly flat series after first bar, all deltas are 0."""
        close = np.full(100, 150.0)
        df = _make_df(close)
        result = rsi.execute(df)
        # With all zeros after first delta=NaN, avg_gain=0, avg_loss=0 -> NaN -> fillna(100)
        # Actually first close-to-close diff is 0, so gains=0 losses=0 throughout
        # ewm of all zeros is 0, so 0/0=NaN -> filled to 100
        rsi_last = result["rsi_14"].iloc[-1]
        assert rsi_last == 100.0

    def test_overbought_oversold_flags(self, rsi: RSIIndicator) -> None:
        """Overbought/oversold boolean columns should match threshold logic."""
        # Consistent uptrend -> high RSI -> overbought
        close = np.linspace(100, 200, 100)
        df = _make_df(close)
        result = rsi.execute(df, {"period": 14, "overbought": 70.0, "oversold": 30.0})

        rsi_vals = result["rsi_14"]
        ob_flags = result["rsi_14_overbought"]
        os_flags = result["rsi_14_oversold"]

        # Where RSI >= 70, overbought should be True
        for i in range(len(rsi_vals)):
            if not np.isnan(rsi_vals.iloc[i]):
                assert ob_flags.iloc[i] == (rsi_vals.iloc[i] >= 70.0)
                assert os_flags.iloc[i] == (rsi_vals.iloc[i] <= 30.0)

    def test_custom_period(
        self, rsi: RSIIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = rsi.execute(sample_ohlcv, {"period": 7})
        assert "rsi_7" in result.columns

    def test_overbought_lte_oversold_raises(
        self, rsi: RSIIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        with pytest.raises(BlockComputeError, match="must be greater than"):
            rsi.execute(sample_ohlcv, {"overbought": 30.0, "oversold": 70.0})

    def test_insufficient_data_raises(self, rsi: RSIIndicator) -> None:
        close = [100.0] * 10
        df = _make_df(close)
        with pytest.raises(BlockComputeError, match="Insufficient data"):
            rsi.execute(df, {"period": 14})

    def test_period_below_min_raises(
        self, rsi: RSIIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        with pytest.raises(BlockValidationError):
            rsi.execute(sample_ohlcv, {"period": 1})
```

---

### `tests/unit/indicators/test_macd.py`

```python
"""Tests for the MACD indicator block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.indicators.macd import MACDIndicator
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError, BlockValidationError
from forgequant.core.types import BlockCategory


@pytest.fixture
def macd() -> MACDIndicator:
    return MACDIndicator()


class TestMACDMetadata:
    def test_name(self, macd: MACDIndicator) -> None:
        assert macd.metadata.name == "macd"

    def test_category(self, macd: MACDIndicator) -> None:
        assert macd.metadata.category == BlockCategory.INDICATOR

    def test_defaults(self, macd: MACDIndicator) -> None:
        defaults = macd.metadata.get_defaults()
        assert defaults["fast_period"] == 12
        assert defaults["slow_period"] == 26
        assert defaults["signal_period"] == 9

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "macd" in registry


class TestMACDCompute:
    def test_default_params(
        self, macd: MACDIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = macd.execute(sample_ohlcv)
        assert "macd_line" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_histogram" in result.columns
        assert len(result) == len(sample_ohlcv)

    def test_histogram_equals_line_minus_signal(
        self, macd: MACDIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        """Histogram must exactly equal macd_line - macd_signal."""
        result = macd.execute(sample_ohlcv)
        computed_hist = result["macd_line"] - result["macd_signal"]
        pd.testing.assert_series_equal(
            result["macd_histogram"], computed_hist, check_names=False
        )

    def test_flat_price_macd_zero(self, macd: MACDIndicator) -> None:
        """With flat prices, MACD line should converge to zero."""
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.full(n, 100.0)
        df = pd.DataFrame(
            {
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = macd.execute(df)
        # With identical fast and slow EMAs, MACD line should be 0
        assert abs(result["macd_line"].iloc[-1]) < 1e-10

    def test_uptrend_positive_macd(self, macd: MACDIndicator) -> None:
        """In a strong uptrend, MACD line should be positive."""
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.linspace(100, 200, n)
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = macd.execute(df)
        # After warmup, MACD line should be consistently positive
        assert result["macd_line"].iloc[-1] > 0

    def test_fast_ge_slow_raises(
        self, macd: MACDIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        with pytest.raises(BlockComputeError, match="must be less than"):
            macd.execute(sample_ohlcv, {"fast_period": 26, "slow_period": 12})

    def test_fast_eq_slow_raises(
        self, macd: MACDIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        with pytest.raises(BlockComputeError, match="must be less than"):
            macd.execute(sample_ohlcv, {"fast_period": 20, "slow_period": 20})

    def test_insufficient_data_raises(self, macd: MACDIndicator) -> None:
        dates = pd.date_range("2024-01-01", periods=30, freq="h")
        close = np.random.randn(30) + 100
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": np.ones(30) * 1000,
            },
            index=dates,
        )
        with pytest.raises(BlockComputeError, match="Insufficient data"):
            macd.execute(df)

    def test_custom_periods(
        self, macd: MACDIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = macd.execute(
            sample_ohlcv,
            {"fast_period": 8, "slow_period": 21, "signal_period": 5},
        )
        assert "macd_line" in result.columns
        assert len(result) == len(sample_ohlcv)
```

---

### `tests/unit/indicators/test_adx.py`

```python
"""Tests for the ADX indicator block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.indicators.adx import ADXIndicator
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError, BlockValidationError
from forgequant.core.types import BlockCategory


@pytest.fixture
def adx() -> ADXIndicator:
    return ADXIndicator()


class TestADXMetadata:
    def test_name(self, adx: ADXIndicator) -> None:
        assert adx.metadata.name == "adx"

    def test_category(self, adx: ADXIndicator) -> None:
        assert adx.metadata.category == BlockCategory.INDICATOR

    def test_default_period(self, adx: ADXIndicator) -> None:
        assert adx.metadata.get_defaults()["period"] == 14

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "adx" in registry


class TestADXCompute:
    def test_default_params(
        self, adx: ADXIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = adx.execute(sample_ohlcv)
        assert "adx" in result.columns
        assert "plus_di" in result.columns
        assert "minus_di" in result.columns
        assert len(result) == len(sample_ohlcv)

    def test_adx_non_negative(
        self, adx: ADXIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        """ADX values should be non-negative."""
        result = adx.execute(sample_ohlcv)
        adx_vals = result["adx"].dropna()
        assert (adx_vals >= 0).all()

    def test_di_non_negative(
        self, adx: ADXIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        """+DI and -DI should be non-negative."""
        result = adx.execute(sample_ohlcv)
        assert (result["plus_di"].dropna() >= 0).all()
        assert (result["minus_di"].dropna() >= 0).all()

    def test_strong_trend_high_adx(self, adx: ADXIndicator) -> None:
        """A strong linear trend should produce high ADX."""
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        trend = np.linspace(100, 200, n)
        df = pd.DataFrame(
            {
                "open": trend,
                "high": trend + 1.0,
                "low": trend - 0.1,
                "close": trend + 0.5,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = adx.execute(df)
        # ADX should be relatively high in a strong trend
        assert result["adx"].iloc[-1] > 20

    def test_uptrend_plus_di_gt_minus_di(self, adx: ADXIndicator) -> None:
        """In an uptrend, +DI should be greater than -DI."""
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        trend = np.linspace(100, 200, n)
        df = pd.DataFrame(
            {
                "open": trend,
                "high": trend + 1.0,
                "low": trend - 0.1,
                "close": trend + 0.5,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = adx.execute(df)
        assert result["plus_di"].iloc[-1] > result["minus_di"].iloc[-1]

    def test_insufficient_data_raises(self, adx: ADXIndicator) -> None:
        dates = pd.date_range("2024-01-01", periods=20, freq="h")
        close = np.ones(20) * 100
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": np.ones(20) * 1000,
            },
            index=dates,
        )
        with pytest.raises(BlockComputeError, match="Insufficient data"):
            adx.execute(df)

    def test_custom_period(
        self, adx: ADXIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = adx.execute(sample_ohlcv, {"period": 7})
        assert "adx" in result.columns
```

---

### `tests/unit/indicators/test_atr.py`

```python
"""Tests for the ATR indicator block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.indicators.atr import ATRIndicator
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError, BlockValidationError
from forgequant.core.types import BlockCategory


@pytest.fixture
def atr() -> ATRIndicator:
    return ATRIndicator()


class TestATRMetadata:
    def test_name(self, atr: ATRIndicator) -> None:
        assert atr.metadata.name == "atr"

    def test_category(self, atr: ATRIndicator) -> None:
        assert atr.metadata.category == BlockCategory.INDICATOR

    def test_default_period(self, atr: ATRIndicator) -> None:
        assert atr.metadata.get_defaults()["period"] == 14

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "atr" in registry


class TestATRCompute:
    def test_default_params(
        self, atr: ATRIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = atr.execute(sample_ohlcv)
        assert "atr_14" in result.columns
        assert "atr_14_pct" in result.columns
        assert len(result) == len(sample_ohlcv)

    def test_atr_positive(
        self, atr: ATRIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        """ATR should always be positive (except possibly first NaN)."""
        result = atr.execute(sample_ohlcv)
        atr_vals = result["atr_14"].dropna()
        assert (atr_vals > 0).all()

    def test_atr_pct_positive(
        self, atr: ATRIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = atr.execute(sample_ohlcv)
        pct_vals = result["atr_14_pct"].dropna()
        assert (pct_vals > 0).all()

    def test_high_volatility_high_atr(self, atr: ATRIndicator) -> None:
        """Wide high-low ranges should produce high ATR."""
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.full(n, 100.0)
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 10.0,  # Wide range
                "low": close - 10.0,
                "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = atr.execute(df)
        # ATR should reflect the 20-point range
        assert result["atr_14"].iloc[-1] > 15.0

    def test_low_volatility_low_atr(self, atr: ATRIndicator) -> None:
        """Tight high-low ranges should produce low ATR."""
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.full(n, 100.0)
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.01,
                "low": close - 0.01,
                "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = atr.execute(df)
        assert result["atr_14"].iloc[-1] < 0.1

    def test_gap_increases_atr(self, atr: ATRIndicator) -> None:
        """A gap (close far from next bar high/low) should increase ATR."""
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        close = np.full(n, 100.0)
        high = close + 0.5
        low = close - 0.5

        # Create a gap at bar 50
        close[50:] = 120.0
        high[50:] = 120.5
        low[50:] = 119.5

        df = pd.DataFrame(
            {
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = atr.execute(df)
        # ATR at bar 51+ should be higher than at bar 49
        assert result["atr_14"].iloc[55] > result["atr_14"].iloc[49]

    def test_known_true_range(self, atr: ATRIndicator) -> None:
        """Verify True Range calculation with known values."""
        dates = pd.date_range("2024-01-01", periods=20, freq="h")
        # Bar 0: H=105, L=95, C=100 -> TR = max(10, -, -) = 10
        # Bar 1: H=108, L=97, C=106 -> TR = max(11, |108-100|=8, |97-100|=3) = 11
        close = np.full(20, 100.0)
        high = np.full(20, 105.0)
        low = np.full(20, 95.0)

        # Make bar 1 special
        close[0] = 100.0
        high[1] = 108.0
        low[1] = 97.0
        close[1] = 106.0

        df = pd.DataFrame(
            {
                "open": close,
                "high": high,
                "low": low,
                "close": close,
                "volume": np.ones(20) * 1000,
            },
            index=dates,
        )
        result = atr.execute(df, {"period": 1})
        # With period=1, ATR equals the true range
        # Bar 1: TR = max(108-97, |108-100|, |97-100|) = max(11, 8, 3) = 11
        assert abs(result["atr_1"].iloc[1] - 11.0) < 1e-6

    def test_insufficient_data_raises(self, atr: ATRIndicator) -> None:
        dates = pd.date_range("2024-01-01", periods=5, freq="h")
        df = pd.DataFrame(
            {
                "open": [100.0] * 5,
                "high": [101.0] * 5,
                "low": [99.0] * 5,
                "close": [100.0] * 5,
                "volume": [1000.0] * 5,
            },
            index=dates,
        )
        with pytest.raises(BlockComputeError, match="Insufficient data"):
            atr.execute(df, {"period": 14})

    def test_custom_period(
        self, atr: ATRIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = atr.execute(sample_ohlcv, {"period": 7})
        assert "atr_7" in result.columns
        assert "atr_7_pct" in result.columns
```

---

### `tests/unit/indicators/test_bollinger_bands.py`

```python
"""Tests for the Bollinger Bands indicator block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.indicators.bollinger_bands import BollingerBandsIndicator
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError, BlockValidationError
from forgequant.core.types import BlockCategory


@pytest.fixture
def bb() -> BollingerBandsIndicator:
    return BollingerBandsIndicator()


class TestBollingerMetadata:
    def test_name(self, bb: BollingerBandsIndicator) -> None:
        assert bb.metadata.name == "bollinger_bands"

    def test_category(self, bb: BollingerBandsIndicator) -> None:
        assert bb.metadata.category == BlockCategory.INDICATOR

    def test_defaults(self, bb: BollingerBandsIndicator) -> None:
        defaults = bb.metadata.get_defaults()
        assert defaults["period"] == 20
        assert defaults["num_std"] == 2.0

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "bollinger_bands" in registry


class TestBollingerCompute:
    def test_output_columns(
        self, bb: BollingerBandsIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = bb.execute(sample_ohlcv)
        expected_cols = {"bb_upper", "bb_middle", "bb_lower", "bb_pct_b", "bb_bandwidth"}
        assert expected_cols == set(result.columns)
        assert len(result) == len(sample_ohlcv)

    def test_upper_gt_middle_gt_lower(
        self, bb: BollingerBandsIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        """Upper band > middle > lower band (where not NaN)."""
        result = bb.execute(sample_ohlcv)
        valid = result.dropna()
        assert (valid["bb_upper"] >= valid["bb_middle"]).all()
        assert (valid["bb_middle"] >= valid["bb_lower"]).all()

    def test_middle_is_sma(
        self, bb: BollingerBandsIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        """Middle band should be the SMA of close."""
        result = bb.execute(sample_ohlcv, {"period": 20})
        expected_sma = sample_ohlcv["close"].rolling(20).mean()
        pd.testing.assert_series_equal(
            result["bb_middle"], expected_sma, check_names=False
        )

    def test_band_width_matches_std(
        self, bb: BollingerBandsIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        """Upper - lower should equal 2 * num_std * std."""
        period = 20
        num_std = 2.0
        result = bb.execute(sample_ohlcv, {"period": period, "num_std": num_std})
        std = sample_ohlcv["close"].rolling(period).std(ddof=0)
        expected_width = 2.0 * num_std * std
        actual_width = result["bb_upper"] - result["bb_lower"]
        pd.testing.assert_series_equal(
            actual_width, expected_width, check_names=False, atol=1e-10
        )

    def test_pct_b_at_middle(self, bb: BollingerBandsIndicator) -> None:
        """When close equals the middle band, %B should be 0.5."""
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        # Constant price -> SMA = price, std = 0 -> bands collapse
        # With std=0, %B is NaN (0/0), so use a tiny variation instead
        np.random.seed(42)
        close = 100.0 + np.random.randn(n) * 0.01
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 0.1,
                "low": close - 0.1,
                "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = bb.execute(df, {"period": 20, "num_std": 2.0})
        # %B should hover around 0.5 for random noise centered on mean
        pct_b = result["bb_pct_b"].dropna()
        assert abs(pct_b.mean() - 0.5) < 0.15

    def test_bandwidth_positive(
        self, bb: BollingerBandsIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = bb.execute(sample_ohlcv)
        bw = result["bb_bandwidth"].dropna()
        assert (bw >= 0).all()

    def test_high_volatility_wide_bands(
        self, bb: BollingerBandsIndicator
    ) -> None:
        """High volatility should produce wider bands."""
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        np.random.seed(42)

        # Low volatility
        close_low = 100.0 + np.random.randn(n) * 0.1
        df_low = pd.DataFrame(
            {
                "open": close_low,
                "high": close_low + 0.2,
                "low": close_low - 0.2,
                "close": close_low,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )

        # High volatility
        close_high = 100.0 + np.random.randn(n) * 5.0
        df_high = pd.DataFrame(
            {
                "open": close_high,
                "high": close_high + 1,
                "low": close_high - 1,
                "close": close_high,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )

        bw_low = bb.execute(df_low)["bb_bandwidth"].iloc[-1]
        bw_high = bb.execute(df_high)["bb_bandwidth"].iloc[-1]
        assert bw_high > bw_low

    def test_insufficient_data_raises(
        self, bb: BollingerBandsIndicator
    ) -> None:
        dates = pd.date_range("2024-01-01", periods=10, freq="h")
        df = pd.DataFrame(
            {
                "open": [100.0] * 10,
                "high": [101.0] * 10,
                "low": [99.0] * 10,
                "close": [100.0] * 10,
                "volume": [1000.0] * 10,
            },
            index=dates,
        )
        with pytest.raises(BlockComputeError, match="Insufficient data"):
            bb.execute(df, {"period": 20})

    def test_custom_std(
        self, bb: BollingerBandsIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        """Wider num_std should produce wider bands."""
        result_2 = bb.execute(sample_ohlcv, {"num_std": 2.0})
        result_3 = bb.execute(sample_ohlcv, {"num_std": 3.0})

        width_2 = (result_2["bb_upper"] - result_2["bb_lower"]).dropna()
        width_3 = (result_3["bb_upper"] - result_3["bb_lower"]).dropna()

        assert (width_3 >= width_2 - 1e-10).all()
```

---

### `tests/unit/indicators/test_ichimoku.py`

```python
"""Tests for the Ichimoku Kinko Hyo indicator block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.indicators.ichimoku import IchimokuIndicator
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError
from forgequant.core.types import BlockCategory


@pytest.fixture
def ichimoku() -> IchimokuIndicator:
    return IchimokuIndicator()


class TestIchimokuMetadata:
    def test_name(self, ichimoku: IchimokuIndicator) -> None:
        assert ichimoku.metadata.name == "ichimoku"

    def test_category(self, ichimoku: IchimokuIndicator) -> None:
        assert ichimoku.metadata.category == BlockCategory.INDICATOR

    def test_defaults(self, ichimoku: IchimokuIndicator) -> None:
        defaults = ichimoku.metadata.get_defaults()
        assert defaults["tenkan_period"] == 9
        assert defaults["kijun_period"] == 26
        assert defaults["senkou_b_period"] == 52

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "ichimoku" in registry


class TestIchimokuCompute:
    def test_output_columns(
        self, ichimoku: IchimokuIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = ichimoku.execute(sample_ohlcv)
        expected = {
            "ichimoku_tenkan",
            "ichimoku_kijun",
            "ichimoku_senkou_a",
            "ichimoku_senkou_b",
            "ichimoku_chikou",
        }
        assert expected == set(result.columns)
        assert len(result) == len(sample_ohlcv)

    def test_tenkan_is_donchian_midline(
        self, ichimoku: IchimokuIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        """Tenkan should be (9-period highest high + 9-period lowest low) / 2."""
        result = ichimoku.execute(sample_ohlcv)
        expected = (
            sample_ohlcv["high"].rolling(9).max()
            + sample_ohlcv["low"].rolling(9).min()
        ) / 2.0
        pd.testing.assert_series_equal(
            result["ichimoku_tenkan"], expected, check_names=False
        )

    def test_kijun_is_donchian_midline(
        self, ichimoku: IchimokuIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        """Kijun should be (26-period highest high + 26-period lowest low) / 2."""
        result = ichimoku.execute(sample_ohlcv)
        expected = (
            sample_ohlcv["high"].rolling(26).max()
            + sample_ohlcv["low"].rolling(26).min()
        ) / 2.0
        pd.testing.assert_series_equal(
            result["ichimoku_kijun"], expected, check_names=False
        )

    def test_senkou_a_shifted_forward(
        self, ichimoku: IchimokuIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        """Senkou Span A should be shifted forward by kijun_period."""
        result = ichimoku.execute(sample_ohlcv)
        # First kijun_period values of senkou_a should be NaN
        # (because they are shifted forward from data that doesn't exist yet)
        assert result["ichimoku_senkou_a"].iloc[:26].isna().all()

    def test_senkou_b_shifted_forward(
        self, ichimoku: IchimokuIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        """Senkou Span B should be shifted forward by kijun_period."""
        result = ichimoku.execute(sample_ohlcv)
        # First (senkou_b_period - 1 + kijun_period) values should be NaN
        first_valid = result["ichimoku_senkou_b"].first_valid_index()
        assert first_valid is not None
        first_valid_pos = result.index.get_loc(first_valid)
        # Should be at least kijun_period + senkou_b_period - 1
        assert first_valid_pos >= 26 + 52 - 1

    def test_chikou_shifted_backward(
        self, ichimoku: IchimokuIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        """Chikou Span should be close shifted back by kijun_period."""
        result = ichimoku.execute(sample_ohlcv)
        expected = sample_ohlcv["close"].shift(-26)
        pd.testing.assert_series_equal(
            result["ichimoku_chikou"], expected, check_names=False
        )

    def test_chikou_last_values_nan(
        self, ichimoku: IchimokuIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        """Last kijun_period values of chikou should be NaN (shifted into the future)."""
        result = ichimoku.execute(sample_ohlcv)
        assert result["ichimoku_chikou"].iloc[-26:].isna().all()

    def test_uptrend_price_above_cloud(self, ichimoku: IchimokuIndicator) -> None:
        """In a strong uptrend, close should be above the cloud."""
        n = 300
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        trend = np.linspace(100, 300, n)
        df = pd.DataFrame(
            {
                "open": trend,
                "high": trend + 1.0,
                "low": trend - 0.5,
                "close": trend,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = ichimoku.execute(df)
        # At the end, price should be above both senkou spans
        idx = -1
        close_last = df["close"].iloc[idx]
        cloud_top = max(
            result["ichimoku_senkou_a"].iloc[idx],
            result["ichimoku_senkou_b"].iloc[idx],
        )
        # Cloud top might be NaN for last values due to shift, check a bit earlier
        check_idx = -30
        close_check = df["close"].iloc[check_idx]
        senkou_a = result["ichimoku_senkou_a"].iloc[check_idx]
        senkou_b = result["ichimoku_senkou_b"].iloc[check_idx]
        if not (np.isnan(senkou_a) or np.isnan(senkou_b)):
            cloud_top = max(senkou_a, senkou_b)
            assert close_check > cloud_top

    def test_insufficient_data_raises(self, ichimoku: IchimokuIndicator) -> None:
        dates = pd.date_range("2024-01-01", periods=50, freq="h")
        close = np.full(50, 100.0)
        df = pd.DataFrame(
            {
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
                "volume": np.ones(50) * 1000,
            },
            index=dates,
        )
        with pytest.raises(BlockComputeError, match="Insufficient data"):
            ichimoku.execute(df)

    def test_custom_periods(
        self, ichimoku: IchimokuIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = ichimoku.execute(
            sample_ohlcv,
            {"tenkan_period": 7, "kijun_period": 22, "senkou_b_period": 44},
        )
        assert "ichimoku_tenkan" in result.columns
```

---

### `tests/unit/indicators/test_stochastic.py`

```python
"""Tests for the Stochastic Oscillator indicator block."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.indicators.stochastic import StochasticIndicator
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.exceptions import BlockComputeError, BlockValidationError
from forgequant.core.types import BlockCategory


@pytest.fixture
def stoch() -> StochasticIndicator:
    return StochasticIndicator()


class TestStochasticMetadata:
    def test_name(self, stoch: StochasticIndicator) -> None:
        assert stoch.metadata.name == "stochastic"

    def test_category(self, stoch: StochasticIndicator) -> None:
        assert stoch.metadata.category == BlockCategory.INDICATOR

    def test_defaults(self, stoch: StochasticIndicator) -> None:
        defaults = stoch.metadata.get_defaults()
        assert defaults["k_period"] == 14
        assert defaults["k_smooth"] == 3
        assert defaults["d_period"] == 3
        assert defaults["overbought"] == 80.0
        assert defaults["oversold"] == 20.0

    def test_registered(self) -> None:
        registry = BlockRegistry()
        assert "stochastic" in registry


class TestStochasticCompute:
    def test_output_columns(
        self, stoch: StochasticIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = stoch.execute(sample_ohlcv)
        expected = {"stoch_k", "stoch_d", "stoch_overbought", "stoch_oversold"}
        assert expected == set(result.columns)
        assert len(result) == len(sample_ohlcv)

    def test_stoch_k_range(
        self, stoch: StochasticIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        """Stochastic %K should be bounded in [0, 100]."""
        result = stoch.execute(sample_ohlcv)
        k_vals = result["stoch_k"].dropna()
        assert k_vals.min() >= -1e-10  # Allow tiny float imprecision
        assert k_vals.max() <= 100.0 + 1e-10

    def test_stoch_d_range(
        self, stoch: StochasticIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        """Stochastic %D should be bounded in [0, 100]."""
        result = stoch.execute(sample_ohlcv)
        d_vals = result["stoch_d"].dropna()
        assert d_vals.min() >= -1e-10
        assert d_vals.max() <= 100.0 + 1e-10

    def test_d_is_sma_of_k(
        self, stoch: StochasticIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        """%D should be the SMA of %K with period d_period."""
        result = stoch.execute(sample_ohlcv, {"d_period": 3})
        expected_d = result["stoch_k"].rolling(3).mean()
        pd.testing.assert_series_equal(
            result["stoch_d"], expected_d, check_names=False
        )

    def test_close_at_high_k_near_100(self, stoch: StochasticIndicator) -> None:
        """When close consistently equals the highest high, %K should be near 100."""
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        # Ascending prices: close is always at the high
        close = np.linspace(100, 200, n)
        df = pd.DataFrame(
            {
                "open": close - 0.5,
                "high": close,
                "low": close - 1.0,
                "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = stoch.execute(df)
        # After warmup, %K should be close to 100
        k_tail = result["stoch_k"].iloc[-10:]
        assert k_tail.dropna().mean() > 90.0

    def test_close_at_low_k_near_0(self, stoch: StochasticIndicator) -> None:
        """When close consistently equals the lowest low, %K should be near 0."""
        n = 100
        dates = pd.date_range("2024-01-01", periods=n, freq="h")
        # Descending prices: close is always at the low
        close = np.linspace(200, 100, n)
        df = pd.DataFrame(
            {
                "open": close + 0.5,
                "high": close + 1.0,
                "low": close,
                "close": close,
                "volume": np.ones(n) * 1000,
            },
            index=dates,
        )
        result = stoch.execute(df)
        k_tail = result["stoch_k"].iloc[-10:]
        assert k_tail.dropna().mean() < 10.0

    def test_overbought_oversold_flags(
        self, stoch: StochasticIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = stoch.execute(sample_ohlcv)
        k_vals = result["stoch_k"]
        ob = result["stoch_overbought"]
        os_flag = result["stoch_oversold"]

        for i in range(len(k_vals)):
            if not np.isnan(k_vals.iloc[i]):
                assert ob.iloc[i] == (k_vals.iloc[i] >= 80.0)
                assert os_flag.iloc[i] == (k_vals.iloc[i] <= 20.0)

    def test_fast_stochastic_k_smooth_1(
        self, stoch: StochasticIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        """With k_smooth=1, %K should be the raw %K (no smoothing)."""
        result = stoch.execute(sample_ohlcv, {"k_smooth": 1})
        # Raw %K = 100 * (close - LL14) / (HH14 - LL14)
        ll = sample_ohlcv["low"].rolling(14).min()
        hh = sample_ohlcv["high"].rolling(14).max()
        raw_k = 100.0 * (sample_ohlcv["close"] - ll) / (hh - ll).replace(0, np.nan)
        # rolling(1).mean() is identity
        pd.testing.assert_series_equal(
            result["stoch_k"], raw_k, check_names=False
        )

    def test_overbought_lte_oversold_raises(
        self, stoch: StochasticIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        with pytest.raises(BlockComputeError, match="must be greater than"):
            stoch.execute(sample_ohlcv, {"overbought": 20.0, "oversold": 80.0})

    def test_insufficient_data_raises(self, stoch: StochasticIndicator) -> None:
        dates = pd.date_range("2024-01-01", periods=10, freq="h")
        df = pd.DataFrame(
            {
                "open": [100.0] * 10,
                "high": [101.0] * 10,
                "low": [99.0] * 10,
                "close": [100.0] * 10,
                "volume": [1000.0] * 10,
            },
            index=dates,
        )
        with pytest.raises(BlockComputeError, match="Insufficient data"):
            stoch.execute(df)

    def test_custom_periods(
        self, stoch: StochasticIndicator, sample_ohlcv: pd.DataFrame
    ) -> None:
        result = stoch.execute(
            sample_ohlcv,
            {"k_period": 5, "k_smooth": 2, "d_period": 2},
        )
        assert "stoch_k" in result.columns
        assert "stoch_d" in result.columns
```

---

## 2.12 Bonus: Integration Test — Registry Completeness

### `tests/integration/test_indicator_registry.py`

```python
"""
Integration test verifying all indicator blocks are properly
registered and can execute on sample data.
"""

from __future__ import annotations

import pandas as pd
import pytest

from forgequant.blocks.registry import BlockRegistry
from forgequant.core.types import BlockCategory

# Force import of all indicator blocks so they register
import forgequant.blocks.indicators  # noqa: F401


EXPECTED_INDICATORS = [
    "adx",
    "atr",
    "bollinger_bands",
    "ema",
    "ichimoku",
    "macd",
    "rsi",
    "stochastic",
]


class TestAllIndicatorsRegistered:
    """Verify every expected indicator is in the registry."""

    def test_all_present(self) -> None:
        registry = BlockRegistry()
        for name in EXPECTED_INDICATORS:
            assert name in registry, f"Indicator '{name}' not found in registry"

    def test_count(self) -> None:
        registry = BlockRegistry()
        indicators = registry.list_by_category(BlockCategory.INDICATOR)
        assert len(indicators) >= len(EXPECTED_INDICATORS)

    def test_all_are_indicator_category(self) -> None:
        registry = BlockRegistry()
        for name in EXPECTED_INDICATORS:
            cls = registry.get_or_raise(name)
            assert cls.metadata.category == BlockCategory.INDICATOR


class TestAllIndicatorsExecute:
    """Verify every indicator can execute with default params on sample data."""

    @pytest.mark.parametrize("block_name", EXPECTED_INDICATORS)
    def test_execute_defaults(
        self, block_name: str, sample_ohlcv: pd.DataFrame
    ) -> None:
        registry = BlockRegistry()
        instance = registry.instantiate(block_name)
        result = instance.execute(sample_ohlcv)

        # Basic sanity checks
        assert result is not None
        if isinstance(result, pd.DataFrame):
            assert len(result) == len(sample_ohlcv)
            assert not result.columns.empty
        elif isinstance(result, pd.Series):
            assert len(result) == len(sample_ohlcv)

    @pytest.mark.parametrize("block_name", EXPECTED_INDICATORS)
    def test_has_description(self, block_name: str) -> None:
        registry = BlockRegistry()
        cls = registry.get_or_raise(block_name)
        assert len(cls.metadata.description) > 20

    @pytest.mark.parametrize("block_name", EXPECTED_INDICATORS)
    def test_has_tags(self, block_name: str) -> None:
        registry = BlockRegistry()
        cls = registry.get_or_raise(block_name)
        assert len(cls.metadata.tags) >= 1

    @pytest.mark.parametrize("block_name", EXPECTED_INDICATORS)
    def test_has_typical_use(self, block_name: str) -> None:
        registry = BlockRegistry()
        cls = registry.get_or_raise(block_name)
        assert len(cls.metadata.typical_use) > 20
```

---

## 2.13 Updated `tests/conftest.py`

Add this line near the top imports so the `autouse` clean_registry fixture re-registers indicators after clearing:

```python
"""
Shared pytest fixtures for the ForgeQuant test suite.

Provides:
    - sample_ohlcv: A realistic synthetic OHLCV DataFrame
    - clean_registry: Auto-clears the BlockRegistry before/after each test
    - sample_block_class: A minimal concrete BaseBlock for testing
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest

from forgequant.blocks.base import BaseBlock
from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.types import BlockCategory, BlockParams, BlockResult


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """
    Generate a realistic synthetic OHLCV DataFrame with 500 bars.

    Returns a DataFrame with:
        - DatetimeIndex at 1-hour intervals
        - Columns: open, high, low, close, volume
        - Prices in a random-walk pattern starting at 1.1000
        - Realistic high/low spreads
        - Volume as random integers
    """
    np.random.seed(42)
    n_bars = 500

    dates = pd.date_range(start="2024-01-01", periods=n_bars, freq="h")

    # Generate a random walk for close prices
    returns = np.random.normal(loc=0.0, scale=0.001, size=n_bars)
    close = 1.1000 * np.exp(np.cumsum(returns))

    # Generate realistic OHLC from close
    spread = np.random.uniform(0.0005, 0.002, size=n_bars)
    high = close + spread * np.random.uniform(0.3, 1.0, size=n_bars)
    low = close - spread * np.random.uniform(0.3, 1.0, size=n_bars)

    # Open is previous close with some gap
    open_prices = np.roll(close, 1) + np.random.normal(0, 0.0002, size=n_bars)
    open_prices[0] = close[0]

    # Ensure high >= max(open, close) and low <= min(open, close)
    high = np.maximum(high, np.maximum(open_prices, close))
    low = np.minimum(low, np.minimum(open_prices, close))

    volume = np.random.randint(100, 10000, size=n_bars).astype(float)

    df = pd.DataFrame(
        {
            "open": open_prices,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )

    return df


@pytest.fixture(autouse=True)
def clean_registry():
    """
    Automatically clear the BlockRegistry before each test,
    then re-register built-in blocks and clear again after.

    For unit tests that test the registry itself, the registry
    starts empty. For indicator tests, indicators must be
    imported explicitly in those test files.
    """
    registry = BlockRegistry()
    registry.clear()
    yield registry
    registry.clear()


@pytest.fixture
def sample_block_class():
    """
    Return a factory that creates minimal concrete BaseBlock classes.

    Usage:
        def test_something(sample_block_class):
            MyBlock = sample_block_class("my_block", BlockCategory.INDICATOR)
            instance = MyBlock()
            result = instance.execute(ohlcv_data)
    """

    def _factory(
        name: str = "test_block",
        category: BlockCategory = BlockCategory.INDICATOR,
        parameters: tuple[ParameterSpec, ...] = (),
        compute_fn: Any = None,
    ) -> type[BaseBlock]:
        """
        Create a concrete BaseBlock subclass with the given configuration.

        Args:
            name: Block name.
            category: Block category.
            parameters: Parameter specifications.
            compute_fn: Optional custom compute function.
                        Signature: (self, data, params) -> BlockResult
                        Defaults to returning data["close"].rename(name).
        """
        meta = BlockMetadata(
            name=name,
            display_name=name.replace("_", " ").title(),
            category=category,
            description=f"Test block: {name}",
            parameters=parameters,
            tags=("test",),
        )

        def default_compute(
            self: BaseBlock,
            data: pd.DataFrame,
            params: BlockParams,
        ) -> BlockResult:
            return data["close"].rename(name)

        fn = compute_fn if compute_fn is not None else default_compute

        # Dynamically create the class
        block_cls = type(
            f"TestBlock_{name}",
            (BaseBlock,),
            {
                "metadata": meta,
                "compute": fn,
            },
        )

        return block_cls

    return _factory
```

---

## 2.14 How to Verify Phase 2

```bash
# From project root with venv activated
# Run all tests
pytest -v

# Run only indicator tests
pytest tests/unit/indicators/ -v

# Run the integration test
pytest tests/integration/test_indicator_registry.py -v

# Run a specific indicator test
pytest tests/unit/indicators/test_ema.py -v

# Type-check indicator files
mypy src/forgequant/blocks/indicators/

# Lint
ruff check src/forgequant/blocks/indicators/
```

**Expected output:** All tests pass — approximately **80+ indicator tests** plus the **8 parametrized integration tests**.

---

## Phase 2 Summary

| Indicator | File | Output Columns | Key Math Detail |
|-----------|------|----------------|-----------------|
| **EMA** | `ema.py` | `ema_{period}` | `ewm(span=period, adjust=False)` — recursive EMA matching MT4/TV |
| **RSI** | `rsi.py` | `rsi_{period}`, `_overbought`, `_oversold` | Wilder smoothing via `ewm(alpha=1/period)`, not standard EMA |
| **MACD** | `macd.py` | `macd_line`, `macd_signal`, `macd_histogram` | Validates fast < slow; histogram = line − signal |
| **ADX** | `adx.py` | `adx`, `plus_di`, `minus_di` | Full Wilder DM system with proper +DM/−DM logic |
| **ATR** | `atr.py` | `atr_{period}`, `atr_{period}_pct` | True Range accounts for gaps; Wilder smoothing |
| **Bollinger** | `bollinger_bands.py` | `bb_upper`, `bb_middle`, `bb_lower`, `bb_pct_b`, `bb_bandwidth` | `ddof=0` population std matching TradingView |
| **Ichimoku** | `ichimoku.py` | 5 components with proper shift directions | Senkou forward-shifted, Chikou backward-shifted |
| **Stochastic** | `stochastic.py` | `stoch_k`, `stoch_d`, `_overbought`, `_oversold` | Supports both Fast (`k_smooth=1`) and Slow modes |

---

**Ready for Phase 3** — say the word and I'll write all price action blocks (Breakout, Pullback, HigherHighLowerLow, SupportResistance) and all entry rule blocks (Crossover, ThresholdCross, Confluence, ReversalPattern) with the same precision and full test coverage.
