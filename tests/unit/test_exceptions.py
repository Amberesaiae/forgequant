"""Tests for forgequant.core.exceptions."""

from __future__ import annotations

import pytest

from forgequant.core.exceptions import (
    BlockComputeError,
    BlockNotFoundError,
    BlockRegistrationError,
    BlockValidationError,
    ConfigurationError,
    ForgeQuantError,
    RobustnessError,
    StrategyCompileError,
)


class TestForgeQuantError:
    """Tests for the base exception."""

    def test_message(self) -> None:
        err = ForgeQuantError("something broke")
        assert str(err) == "something broke"
        assert err.message == "something broke"

    def test_details_default_empty(self) -> None:
        err = ForgeQuantError("oops")
        assert err.details == {}

    def test_details_provided(self) -> None:
        err = ForgeQuantError("oops", details={"foo": "bar"})
        assert err.details == {"foo": "bar"}

    def test_repr_without_details(self) -> None:
        err = ForgeQuantError("msg")
        assert "ForgeQuantError" in repr(err)
        assert "msg" in repr(err)

    def test_repr_with_details(self) -> None:
        err = ForgeQuantError("msg", details={"k": "v"})
        r = repr(err)
        assert "details" in r
        assert "k" in r

    def test_is_exception(self) -> None:
        err = ForgeQuantError("x")
        assert isinstance(err, Exception)

    def test_catch_as_base(self) -> None:
        """All custom exceptions should be catchable as ForgeQuantError."""
        with pytest.raises(ForgeQuantError):
            raise BlockNotFoundError("test")


class TestBlockNotFoundError:
    def test_attributes(self) -> None:
        err = BlockNotFoundError("ema")
        assert err.block_name == "ema"
        assert "ema" in err.message
        assert err.details["block_name"] == "ema"

    def test_inherits_base(self) -> None:
        assert issubclass(BlockNotFoundError, ForgeQuantError)


class TestBlockRegistrationError:
    def test_attributes(self) -> None:
        err = BlockRegistrationError("rsi", "duplicate name")
        assert err.block_name == "rsi"
        assert err.reason == "duplicate name"
        assert "rsi" in err.message
        assert "duplicate" in err.message


class TestBlockComputeError:
    def test_attributes(self) -> None:
        err = BlockComputeError("macd", "division by zero")
        assert err.block_name == "macd"
        assert err.reason == "division by zero"


class TestBlockValidationError:
    def test_attributes(self) -> None:
        err = BlockValidationError(
            block_name="ema",
            param_name="period",
            value=-5,
            constraint="must be >= 2",
        )
        assert err.block_name == "ema"
        assert err.param_name == "period"
        assert err.value == -5
        assert err.constraint == "must be >= 2"
        assert "ema" in err.message
        assert "period" in err.message


class TestConfigurationError:
    def test_message(self) -> None:
        err = ConfigurationError("missing API key")
        assert err.message == "missing API key"


class TestStrategyCompileError:
    def test_attributes(self) -> None:
        err = StrategyCompileError("trend_follow", "missing exit block")
        assert err.strategy_name == "trend_follow"
        assert err.reason == "missing exit block"


class TestRobustnessError:
    def test_attributes(self) -> None:
        err = RobustnessError("trend_follow", "walk_forward", "degraded sharpe")
        assert err.strategy_name == "trend_follow"
        assert err.gate_name == "walk_forward"
        assert err.reason == "degraded sharpe"
