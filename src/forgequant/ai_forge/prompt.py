"""
System prompt builder for LLM-driven strategy generation.
"""

from __future__ import annotations

import json
from typing import Any

from forgequant.ai_forge.exceptions import PromptBuildError
from forgequant.ai_forge.schemas import StrategySpec
from forgequant.blocks.registry import BlockRegistry
from forgequant.core.logging import get_logger
from forgequant.core.types import BlockCategory

logger = get_logger(__name__)


def _format_block_catalog(registry: BlockRegistry) -> str:
    sections: list[str] = []

    category_order = [
        BlockCategory.INDICATOR,
        BlockCategory.PRICE_ACTION,
        BlockCategory.ENTRY_RULE,
        BlockCategory.EXIT_RULE,
        BlockCategory.MONEY_MANAGEMENT,
        BlockCategory.FILTER,
    ]

    for category in category_order:
        blocks = registry.list_by_category(category)
        if not blocks:
            continue

        lines: list[str] = [f"\n### {category.value.upper().replace('_', ' ')} BLOCKS\n"]

        for cls in blocks:
            meta = cls.metadata
            lines.append(f"**{meta.name}** — {meta.display_name}")
            lines.append(f"  Description: {meta.description}")

            if meta.parameters:
                lines.append("  Parameters:")
                for p in meta.parameters:
                    default_str = repr(p.default)
                    range_str = ""
                    if p.min_value is not None or p.max_value is not None:
                        parts = []
                        if p.min_value is not None:
                            parts.append(f"min={p.min_value}")
                        if p.max_value is not None:
                            parts.append(f"max={p.max_value}")
                        range_str = f" ({', '.join(parts)})"
                    choices_str = ""
                    if p.choices:
                        choices_str = f" choices={list(p.choices)}"
                    lines.append(
                        f"    - {p.name}: {p.param_type}, "
                        f"default={default_str}{range_str}{choices_str}"
                        f" — {p.description}"
                    )

            if meta.typical_use:
                lines.append(f"  Typical use: {meta.typical_use}")

            lines.append("")

        sections.append("\n".join(lines))

    return "\n".join(sections)


def _get_output_schema_description() -> str:
    schema = StrategySpec.model_json_schema()
    return json.dumps(schema, indent=2)


SYSTEM_PROMPT_TEMPLATE = """You are ForgeQuant Strategy Architect, an expert system for \
designing systematic trading strategies using composable building blocks.

## YOUR ROLE

You design trading strategies by selecting and parameterizing building blocks \
from the ForgeQuant library. Each strategy is assembled from:

1. **Indicators**: Technical calculations on OHLCV data (EMA, RSI, MACD, etc.)
2. **Price Action** (optional): Pattern detection (breakouts, pullbacks, S/R, etc.)
3. **Entry Rules**: Signal generators that determine when to enter trades
4. **Exit Rules**: Signal generators that determine when to exit trades
5. **Money Management**: Position sizing (exactly one required)
6. **Filters** (optional but recommended): Quality gates that restrict trading conditions

## CRITICAL RULES

1. Every block_name MUST exactly match a name from the Available Blocks catalog below.
2. Every strategy MUST have at least one indicator, one entry rule, and one exit rule.
3. Every strategy MUST have exactly one money management block.
4. Parameters must be within the declared min/max ranges.
5. Do NOT invent block names that don't exist in the catalog.
6. For trend-following strategies, always include a trend filter.
7. For mean-reversion strategies, always include an overbought/oversold indicator.
8. Always include at least one exit rule with a stop-loss mechanism.
9. The entry rule's indicators should reference indicators you've included.
10. ATR periods should be consistent between exit rules and money management.

## STRATEGY DESIGN PRINCIPLES

- **Simplicity**: Prefer fewer, well-chosen blocks over complex configurations.
- **Robustness**: Choose parameters that work across multiple market conditions.
- **Risk management**: Always size positions based on volatility (ATR).
- **Anti-overfitting**: Avoid extreme parameter values at the edges of ranges.
- **Consistency**: ATR periods and SL multipliers should match across blocks.

## AVAILABLE BLOCKS

{block_catalog}

## OUTPUT FORMAT

You MUST respond with a valid JSON object conforming to the StrategySpec schema.
Do NOT include any text before or after the JSON. The JSON schema is:

{output_schema}

## ADDITIONAL CONTEXT

{rag_context}

## USER REQUEST

Design a strategy based on the following request:
"""


def build_system_prompt(
    registry: BlockRegistry | None = None,
    rag_context: str = "",
) -> str:
    reg = registry or BlockRegistry()

    if reg.count() == 0:
        raise PromptBuildError(
            "No blocks registered. Import block modules before building the prompt."
        )

    try:
        block_catalog = _format_block_catalog(reg)
        output_schema = _get_output_schema_description()
    except Exception as e:
        raise PromptBuildError(f"Error generating prompt components: {e}") from e

    rag_section = rag_context if rag_context else "No additional context provided."

    prompt = SYSTEM_PROMPT_TEMPLATE.format(
        block_catalog=block_catalog,
        output_schema=output_schema,
        rag_context=rag_section,
    )

    logger.debug(
        "system_prompt_built",
        prompt_length=len(prompt),
        block_count=reg.count(),
        has_rag_context=bool(rag_context),
    )

    return prompt


def build_user_message(
    idea: str,
    timeframe: str = "1h",
    instruments: list[str] | None = None,
    style: str | None = None,
    additional_requirements: str = "",
) -> str:
    parts = [idea]
    parts.append(f"\nTimeframe: {timeframe}")

    if instruments:
        parts.append(f"Instruments: {', '.join(instruments)}")

    if style:
        parts.append(f"Style: {style}")

    if additional_requirements:
        parts.append(f"\nAdditional requirements:\n{additional_requirements}")

    return "\n".join(parts)
