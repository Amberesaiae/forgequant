"""
AI Forge specific exceptions.

All inherit from AIForgeError which itself inherits from ForgeQuantError,
keeping the exception hierarchy unified.
"""

from __future__ import annotations

from typing import Any

from forgequant.core.exceptions import ForgeQuantError


class AIForgeError(ForgeQuantError):
    """Base exception for all AI Forge errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message=message, details=details)


class PromptBuildError(AIForgeError):
    """Raised when the system prompt cannot be constructed."""

    def __init__(self, reason: str) -> None:
        super().__init__(
            message=f"Failed to build system prompt: {reason}",
            details={"reason": reason},
        )
        self.reason = reason


class LLMCallError(AIForgeError):
    """Raised when the LLM API call fails."""

    def __init__(self, provider: str, reason: str) -> None:
        super().__init__(
            message=f"LLM call to '{provider}' failed: {reason}",
            details={"provider": provider, "reason": reason},
        )
        self.provider = provider
        self.reason = reason


class SpecValidationError(AIForgeError):
    """Raised when a StrategySpec fails validation against the registry."""

    def __init__(
        self,
        strategy_name: str,
        errors: list[str],
    ) -> None:
        super().__init__(
            message=(
                f"Strategy '{strategy_name}' spec validation failed with "
                f"{len(errors)} error(s): {'; '.join(errors[:5])}"
            ),
            details={"strategy_name": strategy_name, "errors": errors},
        )
        self.strategy_name = strategy_name
        self.errors = errors


class GroundingError(AIForgeError):
    """Raised when RAG grounding operations fail."""

    def __init__(self, operation: str, reason: str) -> None:
        super().__init__(
            message=f"Grounding {operation} failed: {reason}",
            details={"operation": operation, "reason": reason},
        )
        self.operation = operation
        self.reason = reason
