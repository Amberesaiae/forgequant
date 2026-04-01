"""
AI Forge: LLM-driven strategy specification and assembly.

Provides:
    - schemas: Pydantic models for StrategySpec and BlockSpec
    - prompt: System prompt builder with block catalog grounding
    - grounding: ChromaDB-based RAG for trading knowledge retrieval
    - pipeline: End-to-end LLM call + validation + retry pipeline
    - validator: Validates specs against the live BlockRegistry
    - providers: Abstraction over OpenAI, Anthropic, and Groq
"""

from forgequant.ai_forge.schemas import (
    BlockSpec,
    StrategySpec,
    StrategyConstraints,
    StrategyObjective,
)
from forgequant.ai_forge.validator import SpecValidator, ValidationResult
from forgequant.ai_forge.exceptions import (
    AIForgeError,
    PromptBuildError,
    LLMCallError,
    SpecValidationError,
    GroundingError,
)

__all__ = [
    "BlockSpec",
    "StrategySpec",
    "StrategyConstraints",
    "StrategyObjective",
    "SpecValidator",
    "ValidationResult",
    "AIForgeError",
    "PromptBuildError",
    "LLMCallError",
    "SpecValidationError",
    "GroundingError",
]
