"""
Pydantic models for strategy specifications.

These models define the structured output that the LLM must produce.
The StrategySpec is the top-level object representing a complete
trading strategy assembled from building blocks.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class BlockSpec(BaseModel):
    """
    Specification for a single building block within a strategy.
    """

    block_name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description=(
            "Name of the block as registered in the BlockRegistry. "
            "Must be lowercase with underscores (e.g. 'ema', 'crossover_entry')."
        ),
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Parameter overrides for this block. Keys must match the block's "
            "declared parameter names. Missing keys use the block's defaults."
        ),
    )
    rationale: str = Field(
        default="",
        max_length=500,
        description=(
            "Brief explanation of why this block was chosen and how its "
            "parameters were selected. Helps with strategy documentation."
        ),
    )

    @field_validator("block_name", mode="before")
    @classmethod
    def normalize_block_name(cls, v: str) -> str:
        if isinstance(v, str):
            return v.strip().lower()
        return v


class StrategyObjective(BaseModel):
    """Defines the strategic objective and trading style."""

    style: str = Field(
        ...,
        description=(
            "Trading style: 'trend_following', 'mean_reversion', 'breakout', "
            "'momentum', 'scalping', or 'hybrid'."
        ),
    )
    timeframe: str = Field(
        ...,
        description="Primary timeframe (e.g. '1h', '4h', '1d').",
    )
    instruments: list[str] = Field(
        default_factory=list,
        description="Target instruments (e.g. ['EURUSD', 'GBPUSD']).",
    )
    direction: str = Field(
        default="both",
        description="Trade direction: 'long', 'short', or 'both'.",
    )
    description: str = Field(
        default="",
        max_length=1000,
        description="Natural language description of the strategy idea.",
    )

    @field_validator("style", mode="before")
    @classmethod
    def normalize_style(cls, v: str) -> str:
        if isinstance(v, str):
            return v.strip().lower().replace(" ", "_").replace("-", "_")
        return v

    @field_validator("direction", mode="before")
    @classmethod
    def normalize_direction(cls, v: str) -> str:
        if isinstance(v, str):
            v = v.strip().lower()
            if v not in ("long", "short", "both"):
                return "both"
        return v


class StrategyConstraints(BaseModel):
    """Constraints and quality gates for the strategy."""

    min_trades: int = Field(
        default=150,
        ge=1,
        description="Minimum number of trades required for validation.",
    )
    max_drawdown: float = Field(
        default=0.18,
        gt=0.0,
        le=1.0,
        description="Maximum acceptable drawdown as a decimal fraction.",
    )
    min_profit_factor: float = Field(
        default=1.35,
        gt=0.0,
        description="Minimum acceptable profit factor.",
    )
    min_sharpe: float = Field(
        default=0.80,
        description="Minimum acceptable annualized Sharpe ratio.",
    )
    min_win_rate: float = Field(
        default=0.35,
        gt=0.0,
        le=1.0,
        description="Minimum acceptable win rate.",
    )
    max_correlation: float = Field(
        default=0.70,
        gt=0.0,
        le=1.0,
        description="Maximum equity curve correlation with existing strategies.",
    )


class StrategySpec(BaseModel):
    """
    Complete specification for a trading strategy.

    Invariants enforced:
        - At least one indicator
        - At least one entry rule
        - At least one exit rule
        - Exactly one money management block
    """

    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Unique strategy name (lowercase, underscores).",
    )
    description: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Detailed description of the strategy logic.",
    )
    objective: StrategyObjective = Field(
        ...,
        description="Strategic objective and trading style.",
    )
    constraints: StrategyConstraints = Field(
        default_factory=StrategyConstraints,
        description="Quality constraints for strategy evaluation.",
    )
    indicators: list[BlockSpec] = Field(
        ...,
        min_length=1,
        description="Technical indicator blocks used by this strategy.",
    )
    price_action: list[BlockSpec] = Field(
        default_factory=list,
        description="Optional price action pattern blocks.",
    )
    entry_rules: list[BlockSpec] = Field(
        ...,
        min_length=1,
        description="Entry rule blocks (at least one required).",
    )
    exit_rules: list[BlockSpec] = Field(
        ...,
        min_length=1,
        description="Exit rule blocks (at least one required).",
    )
    filters: list[BlockSpec] = Field(
        default_factory=list,
        description="Optional filter blocks for trade quality.",
    )
    money_management: BlockSpec = Field(
        ...,
        description="Position sizing block (exactly one required).",
    )

    @field_validator("name", mode="before")
    @classmethod
    def normalize_name(cls, v: str) -> str:
        if isinstance(v, str):
            return v.strip().lower().replace(" ", "_").replace("-", "_")
        return v

    @model_validator(mode="after")
    def validate_structure(self) -> StrategySpec:
        errors: list[str] = []

        if len(self.indicators) == 0:
            errors.append("At least one indicator block is required")

        if len(self.entry_rules) == 0:
            errors.append("At least one entry rule block is required")

        if len(self.exit_rules) == 0:
            errors.append("At least one exit rule block is required")

        for category_name, blocks in [
            ("indicators", self.indicators),
            ("entry_rules", self.entry_rules),
            ("exit_rules", self.exit_rules),
            ("filters", self.filters),
            ("price_action", self.price_action),
        ]:
            names = [b.block_name for b in blocks]
            seen: set[str] = set()
            for n in names:
                if n in seen:
                    errors.append(
                        f"Duplicate block '{n}' in {category_name}. "
                        f"Use different parameters instead of duplicating."
                    )
                seen.add(n)

        if errors:
            raise ValueError(
                f"Strategy spec structural validation failed: {'; '.join(errors)}"
            )

        return self

    def all_blocks(self) -> list[BlockSpec]:
        blocks: list[BlockSpec] = []
        blocks.extend(self.indicators)
        blocks.extend(self.price_action)
        blocks.extend(self.entry_rules)
        blocks.extend(self.exit_rules)
        blocks.extend(self.filters)
        blocks.append(self.money_management)
        return blocks

    def block_names(self) -> list[str]:
        return [b.block_name for b in self.all_blocks()]
