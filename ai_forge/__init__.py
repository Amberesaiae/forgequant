"""
ForgeQuant AI Forge.

Natural language strategy generation with structured output and RAG grounding.
"""

from .prompt import StrategyPrompt
from .grounding import load_knowledge_base

__all__ = [
    "StrategyPrompt",
    "load_knowledge_base",
]
