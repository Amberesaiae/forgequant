"""
AI Forge Prompt Templates.

Structured prompts for converting natural language trading ideas
into StrategySpec objects.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class BlockSpec(BaseModel):
    """Specification for a single building block."""
    block_name: str = Field(..., description="Name of the block in the registry")
    params: Dict[str, Any] = Field(default_factory=dict, description="Block parameters")


class StrategySpec(BaseModel):
    """Complete strategy specification generated from natural language."""
    name: str = Field(..., description="Strategy name")
    description: str = Field(..., description="Strategy description")
    entry: BlockSpec = Field(..., description="Entry rule block")
    exit: BlockSpec = Field(..., description="Exit rule block")
    money_management: BlockSpec = Field(..., description="Money management block")
    filters: List[BlockSpec] = Field(default_factory=list, description="Filter blocks")
    recommended_timeframes: List[str] = Field(default_factory=list, description="Recommended timeframes")
    recommended_instruments: List[str] = Field(default_factory=list, description="Recommended instruments")
    reasoning: str = Field(..., description="Explanation of why these blocks were chosen")


class StrategyPrompt:
    """Prompt builder for AI Forge strategy generation."""

    SYSTEM_PROMPT = """You are an expert systematic trading strategy designer.
Given a natural language description of a trading idea, convert it into a
structured strategy specification using the available building blocks.

Available blocks by category:
- Indicators: EMA, RSI, ATR, BollingerBands, MACD, ADX, Stochastic, Ichimoku
- Price Action: Breakout, Pullback, HigherHighLowerLow, SupportResistance
- Entry Rules: Crossover, ThresholdCross, Confluence, ReversalPattern
- Exit Rules: FixedTPSL, TrailingStop, TimeBasedExit, BreakevenStop
- Money Management: FixedRisk, VolatilityTargeting, KellyFractional, ATRBasedSizing
- Filters: TradingSessionFilter, SpreadFilter, MaxDrawdownFilter, TrendFilter

Rules:
1. Every strategy MUST have an entry, exit, and money_management block.
2. Filters are optional but recommended.
3. Use realistic parameter values based on common trading practice.
4. Provide clear reasoning for your block choices.
5. Suggest appropriate timeframes and instruments.
"""

    @classmethod
    def build(cls, user_idea: str, context: Optional[str] = None) -> str:
        """Build the full prompt for strategy generation.

        Args:
            user_idea: Natural language description of the trading idea.
            context: Optional RAG-retrieved context from knowledge base.

        Returns:
            Complete prompt string for the LLM.
        """
        prompt = f"Trading Idea: {user_idea}\n\n"

        if context:
            prompt += f"Relevant Context:\n{context}\n\n"

        prompt += "Convert this idea into a structured strategy specification."

        return prompt
