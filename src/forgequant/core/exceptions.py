"""
Structured exception hierarchy for ForgeQuant.

All custom exceptions inherit from ForgeQuantError so callers can catch
the entire tree with a single except clause when appropriate.
"""

from __future__ import annotations

from typing import Any


class ForgeQuantError(Exception):
    """
    Base exception for all ForgeQuant errors.

    Attributes:
        message: Human-readable error description.
        details: Optional dictionary of structured context for logging.
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __repr__(self) -> str:
        cls_name = type(self).__name__
        if self.details:
            return f"{cls_name}(message={self.message!r}, details={self.details!r})"
        return f"{cls_name}(message={self.message!r})"


# ── Block Errors ─────────────────────────────────────────────────────────────


class BlockNotFoundError(ForgeQuantError):
    """Raised when a requested block is not in the registry."""

    def __init__(self, block_name: str) -> None:
        super().__init__(
            message=f"Block '{block_name}' not found in registry",
            details={"block_name": block_name},
        )
        self.block_name = block_name


class BlockRegistrationError(ForgeQuantError):
    """Raised when a block fails to register (e.g., duplicate name, invalid class)."""

    def __init__(self, block_name: str, reason: str) -> None:
        super().__init__(
            message=f"Failed to register block '{block_name}': {reason}",
            details={"block_name": block_name, "reason": reason},
        )
        self.block_name = block_name
        self.reason = reason


class BlockComputeError(ForgeQuantError):
    """Raised when a block's compute() method encounters a runtime error."""

    def __init__(self, block_name: str, reason: str) -> None:
        super().__init__(
            message=f"Block '{block_name}' compute failed: {reason}",
            details={"block_name": block_name, "reason": reason},
        )
        self.block_name = block_name
        self.reason = reason


class BlockValidationError(ForgeQuantError):
    """Raised when block parameters fail validation."""

    def __init__(
        self,
        block_name: str,
        param_name: str,
        value: Any,
        constraint: str,
    ) -> None:
        super().__init__(
            message=(
                f"Block '{block_name}' parameter '{param_name}' value {value!r} "
                f"violates constraint: {constraint}"
            ),
            details={
                "block_name": block_name,
                "param_name": param_name,
                "value": value,
                "constraint": constraint,
            },
        )
        self.block_name = block_name
        self.param_name = param_name
        self.value = value
        self.constraint = constraint


# ── Configuration Errors ─────────────────────────────────────────────────────


class ConfigurationError(ForgeQuantError):
    """Raised when the application configuration is invalid or incomplete."""

    def __init__(self, message: str) -> None:
        super().__init__(message=message)


# ── Strategy Errors ──────────────────────────────────────────────────────────


class StrategyCompileError(ForgeQuantError):
    """Raised when a strategy specification cannot be compiled into runnable code."""

    def __init__(self, strategy_name: str, reason: str) -> None:
        super().__init__(
            message=f"Strategy '{strategy_name}' compilation failed: {reason}",
            details={"strategy_name": strategy_name, "reason": reason},
        )
        self.strategy_name = strategy_name
        self.reason = reason


class RobustnessError(ForgeQuantError):
    """Raised when a strategy fails a robustness gate."""

    def __init__(self, strategy_name: str, gate_name: str, reason: str) -> None:
        super().__init__(
            message=(
                f"Strategy '{strategy_name}' failed robustness gate "
                f"'{gate_name}': {reason}"
            ),
            details={
                "strategy_name": strategy_name,
                "gate_name": gate_name,
                "reason": reason,
            },
        )
        self.strategy_name = strategy_name
        self.gate_name = gate_name
        self.reason = reason
