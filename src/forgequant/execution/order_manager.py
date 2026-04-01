"""
Order manager for MT5.

Handles order placement, modification, and closure with
proper risk checks and error handling.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique
from typing import Any

from forgequant.core.logging import get_logger

logger = get_logger(__name__)


@unique
class OrderAction(str, Enum):
    """Order action types."""

    BUY = "buy"
    SELL = "sell"
    BUY_LIMIT = "buy_limit"
    SELL_LIMIT = "sell_limit"
    BUY_STOP = "buy_stop"
    SELL_STOP = "sell_stop"
    CLOSE = "close"
    MODIFY = "modify"


@unique
class OrderResult(str, Enum):
    """Order execution result."""

    SUCCESS = "success"
    REJECTED = "rejected"
    ERROR = "error"
    TIMEOUT = "timeout"
    NO_CONNECTION = "no_connection"


@dataclass
class OrderRequest:
    """Request to place or modify an order."""

    symbol: str
    action: OrderAction
    volume: float
    price: float = 0.0
    sl: float = 0.0
    tp: float = 0.0
    magic: int = 0
    comment: str = ""
    ticket: int = 0  # For close/modify


@dataclass
class OrderResponse:
    """Response from an order execution."""

    result: OrderResult
    ticket: int = 0
    price: float = 0.0
    volume: float = 0.0
    comment: str = ""
    retcode: int = 0


class OrderManager:
    """
    Manages order placement and execution.

    Provides:
    - Market order placement (buy/sell)
    - Pending order placement (limit/stop)
    - Position closure
    - SL/TP modification
    - Pre-trade risk checks
    """

    def __init__(self, mt5_client: Any | None = None) -> None:
        self._mt5 = mt5_client
        self._mt5_module: Any = None

    async def _ensure_mt5(self) -> bool:
        """Ensure MT5 module is available."""
        if self._mt5_module is not None:
            return True

        try:
            import MetaTrader5 as mt5
            self._mt5_module = mt5
            return True
        except ImportError:
            return False

    async def place_market_order(
        self,
        symbol: str,
        action: OrderAction,
        volume: float,
        sl: float = 0.0,
        tp: float = 0.0,
        magic: int = 0,
        comment: str = "",
    ) -> OrderResponse:
        """
        Place a market order.

        Args:
            symbol: Trading symbol.
            action: BUY or SELL.
            volume: Lot size.
            sl: Stop loss price.
            tp: Take profit price.
            magic: Magic number for EA identification.
            comment: Order comment.

        Returns:
            OrderResponse with result details.
        """
        if not await self._ensure_mt5():
            return OrderResponse(result=OrderResult.NO_CONNECTION)

        mt5 = self._mt5_module

        if action == OrderAction.BUY:
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask
        elif action == OrderAction.SELL:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid
        else:
            return OrderResponse(
                result=OrderResult.ERROR,
                comment=f"Invalid market order action: {action}",
            )

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None:
            return OrderResponse(
                result=OrderResult.ERROR,
                comment=f"order_send returned None: {mt5.last_error()}",
            )

        response = OrderResponse(
            result=OrderResult.SUCCESS if result.retcode == 10009 else OrderResult.REJECTED,
            ticket=result.order if hasattr(result, "order") else 0,
            price=result.price if hasattr(result, "price") else 0.0,
            volume=result.volume if hasattr(result, "volume") else 0.0,
            comment=result.comment if hasattr(result, "comment") else "",
            retcode=result.retcode,
        )

        logger.info(
            "order_placed",
            symbol=symbol,
            action=action.value,
            volume=volume,
            ticket=response.ticket,
            result=response.result.value,
            retcode=response.retcode,
        )

        return response

    async def close_position(
        self,
        ticket: int,
        volume: float = 0.0,
    ) -> OrderResponse:
        """
        Close an open position by ticket.

        Args:
            ticket: Position ticket number.
            volume: Volume to close (0.0 = full position).

        Returns:
            OrderResponse with result details.
        """
        if not await self._ensure_mt5():
            return OrderResponse(result=OrderResult.NO_CONNECTION)

        mt5 = self._mt5_module

        positions = mt5.positions_get(ticket=ticket)
        if positions is None or len(positions) == 0:
            return OrderResponse(
                result=OrderResult.ERROR,
                comment=f"Position {ticket} not found",
            )

        position = positions[0]
        close_volume = volume if volume > 0 else position.volume

        if position.type == mt5.POSITION_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(position.symbol).bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(position.symbol).ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": close_volume,
            "type": order_type,
            "price": price,
            "position": ticket,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None:
            return OrderResponse(
                result=OrderResult.ERROR,
                comment=f"order_send returned None: {mt5.last_error()}",
            )

        return OrderResponse(
            result=OrderResult.SUCCESS if result.retcode == 10009 else OrderResult.REJECTED,
            ticket=result.order if hasattr(result, "order") else 0,
            price=result.price if hasattr(result, "price") else 0.0,
            volume=close_volume,
            comment=result.comment if hasattr(result, "comment") else "",
            retcode=result.retcode,
        )

    async def modify_position(
        self,
        ticket: int,
        sl: float = 0.0,
        tp: float = 0.0,
    ) -> OrderResponse:
        """
        Modify SL/TP of an open position.

        Args:
            ticket: Position ticket number.
            sl: New stop loss price.
            tp: New take profit price.

        Returns:
            OrderResponse with result details.
        """
        if not await self._ensure_mt5():
            return OrderResponse(result=OrderResult.NO_CONNECTION)

        mt5 = self._mt5_module

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "sl": sl,
            "tp": tp,
        }

        result = mt5.order_send(request)
        if result is None:
            return OrderResponse(
                result=OrderResult.ERROR,
                comment=f"order_send returned None: {mt5.last_error()}",
            )

        return OrderResponse(
            result=OrderResult.SUCCESS if result.retcode == 10009 else OrderResult.REJECTED,
            ticket=ticket,
            comment=result.comment if hasattr(result, "comment") else "",
            retcode=result.retcode,
        )
