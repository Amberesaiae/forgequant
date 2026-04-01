"""Tests for order manager."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from forgequant.execution.order_manager import (
    OrderAction,
    OrderManager,
    OrderRequest,
    OrderResponse,
    OrderResult,
)


class TestOrderRequest:
    def test_creation(self) -> None:
        req = OrderRequest(
            symbol="EURUSD",
            action=OrderAction.BUY,
            volume=0.1,
            sl=1.0800,
            tp=1.1000,
        )
        assert req.symbol == "EURUSD"
        assert req.action == OrderAction.BUY
        assert req.volume == 0.1


class TestOrderResponse:
    def test_success(self) -> None:
        resp = OrderResponse(
            result=OrderResult.SUCCESS,
            ticket=12345,
            price=1.0900,
            volume=0.1,
        )
        assert resp.result == OrderResult.SUCCESS
        assert resp.ticket == 12345


class TestOrderManager:
    def test_init(self) -> None:
        mgr = OrderManager()
        assert mgr._mt5 is None

    @pytest.mark.asyncio
    async def test_place_market_order_no_mt5(self) -> None:
        mgr = OrderManager()
        result = await mgr.place_market_order("EURUSD", OrderAction.BUY, 0.1)
        assert result.result == OrderResult.NO_CONNECTION

    @pytest.mark.asyncio
    async def test_close_position_no_mt5(self) -> None:
        mgr = OrderManager()
        result = await mgr.close_position(12345)
        assert result.result == OrderResult.NO_CONNECTION

    @pytest.mark.asyncio
    async def test_modify_position_no_mt5(self) -> None:
        mgr = OrderManager()
        result = await mgr.modify_position(12345, sl=1.0800)
        assert result.result == OrderResult.NO_CONNECTION
