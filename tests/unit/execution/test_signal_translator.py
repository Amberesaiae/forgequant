"""Tests for signal translator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from forgequant.execution.signal_translator import (
    SignalTranslator,
    SignalType,
    TradeSignal,
)


class TestTradeSignal:
    def test_creation(self) -> None:
        sig = TradeSignal(
            signal_type=SignalType.ENTRY_LONG,
            symbol="EURUSD",
            timestamp=pd.Timestamp("2024-01-01"),
            price=1.1000,
            volume=0.1,
            sl=1.0950,
            tp=1.1100,
        )
        assert sig.signal_type == SignalType.ENTRY_LONG
        assert sig.price == 1.1000
        assert sig.metadata == {}


class TestSignalTranslator:
    def test_translate_entry_long(self) -> None:
        translator = SignalTranslator(symbol="EURUSD", default_volume=0.1)

        idx = pd.date_range("2024-01-01", periods=10, freq="h")
        entry_long = pd.Series([False] * 5 + [True] + [False] * 4, index=idx)
        close_prices = pd.Series(np.ones(10) * 1.1000, index=idx)

        signals = translator.translate(
            entry_long=entry_long,
            close_prices=close_prices,
        )

        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.ENTRY_LONG
        assert signals[0].symbol == "EURUSD"
        assert signals[0].volume == 0.1

    def test_translate_entry_short(self) -> None:
        translator = SignalTranslator(symbol="EURUSD", default_volume=0.1)

        idx = pd.date_range("2024-01-01", periods=10, freq="h")
        entry_short = pd.Series([False] * 3 + [True] + [False] * 6, index=idx)
        close_prices = pd.Series(np.ones(10) * 1.1000, index=idx)

        signals = translator.translate(
            entry_short=entry_short,
            close_prices=close_prices,
        )

        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.ENTRY_SHORT

    def test_translate_exit_signals(self) -> None:
        translator = SignalTranslator(symbol="EURUSD")

        idx = pd.date_range("2024-01-01", periods=10, freq="h")
        exit_long = pd.Series([False] * 7 + [True] + [False] * 2, index=idx)
        close_prices = pd.Series(np.ones(10) * 1.1000, index=idx)

        signals = translator.translate(
            exit_long=exit_long,
            close_prices=close_prices,
        )

        assert len(signals) == 1
        assert signals[0].signal_type == SignalType.EXIT_LONG

    def test_translate_with_tp_sl(self) -> None:
        translator = SignalTranslator(symbol="EURUSD")

        idx = pd.date_range("2024-01-01", periods=5, freq="h")
        entry_long = pd.Series([True, False, False, False, False], index=idx)
        close_prices = pd.Series([1.1000, 1.1001, 1.1002, 1.1003, 1.1004], index=idx)
        sl = pd.Series([1.0950] * 5, index=idx)
        tp = pd.Series([1.1100] * 5, index=idx)

        signals = translator.translate(
            entry_long=entry_long,
            stop_loss_long=sl,
            take_profit_long=tp,
            close_prices=close_prices,
        )

        assert len(signals) == 1
        assert signals[0].sl == 1.0950
        assert signals[0].tp == 1.1100

    def test_translate_no_signals(self) -> None:
        translator = SignalTranslator()

        idx = pd.date_range("2024-01-01", periods=5, freq="h")
        entry_long = pd.Series(False, index=idx)
        close_prices = pd.Series(np.ones(5) * 1.1000, index=idx)

        signals = translator.translate(
            entry_long=entry_long,
            close_prices=close_prices,
        )

        assert len(signals) == 0

    def test_translate_multiple_signals(self) -> None:
        translator = SignalTranslator(symbol="EURUSD")

        idx = pd.date_range("2024-01-01", periods=10, freq="h")
        entry_long = pd.Series([True, False, False, False, False, False, False, False, False, False], index=idx)
        entry_short = pd.Series([False, False, False, False, False, True, False, False, False, False], index=idx)
        close_prices = pd.Series(np.ones(10) * 1.1000, index=idx)

        signals = translator.translate(
            entry_long=entry_long,
            entry_short=entry_short,
            close_prices=close_prices,
        )

        assert len(signals) == 2
        assert signals[0].signal_type == SignalType.ENTRY_LONG
        assert signals[1].signal_type == SignalType.ENTRY_SHORT
