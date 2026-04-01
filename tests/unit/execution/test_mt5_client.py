"""Tests for MT5 client."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from forgequant.execution.mt5_client import MT5Client, MT5Config


class TestMT5Config:
    def test_defaults(self) -> None:
        cfg = MT5Config()
        assert cfg.login == 0
        assert cfg.password == ""
        assert cfg.server == ""
        assert cfg.timeout == 60000


class TestMT5Client:
    def test_init_defaults(self) -> None:
        client = MT5Client()
        assert client.is_initialized is False

    def test_init_with_config(self) -> None:
        cfg = MT5Config(login=12345, server="Broker-Live")
        client = MT5Client(cfg)
        assert client._config.login == 12345
        assert client._config.server == "Broker-Live"

    @pytest.mark.asyncio
    async def test_initialize_import_error(self) -> None:
        """If MetaTrader5 is not installed, should return False."""
        client = MT5Client()
        # The initialize method tries to import MetaTrader5.
        # Since it's not installed in the test environment, it should return False.
        # We just verify the method returns False and doesn't crash.
        result = await client.initialize()
        assert result is False
        assert client.is_initialized is False

    @pytest.mark.asyncio
    async def test_get_tick_not_initialized(self) -> None:
        client = MT5Client()
        tick = await client.get_tick("EURUSD")
        assert tick is None

    @pytest.mark.asyncio
    async def test_get_rates_not_initialized(self) -> None:
        client = MT5Client()
        rates = await client.get_rates("EURUSD", 0)
        assert rates is None

    @pytest.mark.asyncio
    async def test_get_account_balance_not_initialized(self) -> None:
        client = MT5Client()
        balance = await client.get_account_balance()
        assert balance == 0.0

    @pytest.mark.asyncio
    async def test_get_positions_not_initialized(self) -> None:
        client = MT5Client()
        positions = await client.get_positions()
        assert positions == []

    @pytest.mark.asyncio
    async def test_shutdown_not_initialized(self) -> None:
        client = MT5Client()
        await client.shutdown()
        assert client.is_initialized is False
