"""
forgequant.frontend.pages.forge — AI Forge strategy generation page.
"""

from __future__ import annotations

import reflex as rx

from forgequant.frontend.components.layout import page_layout
from forgequant.frontend.state.forge_state import ForgeState
from forgequant.frontend.styles import CARD_STYLE, CATEGORY_COLORS, COLORS, INPUT_STYLE


def _input_section() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.text("Describe your strategy idea", font_size="16px", font_weight="600", color=COLORS["text_primary"]),
            rx.text_area(
                placeholder="e.g., RSI mean-reversion strategy on EURUSD 1H...",
                value=ForgeState.idea,
                on_change=ForgeState.set_idea,
                min_height="120px",
                width="100%",
                style=INPUT_STYLE,
            ),
            rx.hstack(
                rx.vstack(
                    rx.text("Provider", font_size="12px", color=COLORS["text_muted"]),
                    rx.select(ForgeState.provider_options, value=ForgeState.provider, on_change=ForgeState.set_provider),
                    spacing="1",
                ),
                rx.vstack(
                    rx.text("Model", font_size="12px", color=COLORS["text_muted"]),
                    rx.select(ForgeState.current_model_options, value=ForgeState.model, on_change=ForgeState.set_model),
                    spacing="1",
                ),
                spacing="4",
                flex_wrap="wrap",
                align="end",
            ),
            rx.button(
                rx.cond(ForgeState.is_generating, rx.text("Generating..."), rx.text("Generate Strategy")),
                on_click=ForgeState.generate_strategy,
                disabled=~ForgeState.can_generate,
                color_scheme="blue",
                width="100%",
                size="3",
            ),
            spacing="4",
            width="100%",
        ),
        **CARD_STYLE,
    )


def _block_list(label: str, blocks_var, category: str) -> rx.Component:
    color = CATEGORY_COLORS.get(category, COLORS["accent_blue"])
    return rx.cond(
        blocks_var.length() > 0,
        rx.vstack(
            rx.text(label, font_size="13px", font_weight="600", color=COLORS["text_muted"]),
            rx.hstack(
                rx.foreach(blocks_var, lambda b: rx.box(
                    rx.text(b["name"], font_size="13px", font_weight="600", color="white"),
                    background=color,
                    padding="4px 12px",
                    border_radius="16px",
                )),
                spacing="2",
                flex_wrap="wrap",
            ),
            spacing="2",
            width="100%",
        ),
        rx.fragment(),
    )


def _result_section() -> rx.Component:
    return rx.cond(
        ForgeState.generation_attempted,
        rx.box(
            rx.vstack(
                rx.hstack(
                    rx.text("Generation Result", font_size="16px", font_weight="600", color=COLORS["text_primary"]),
                    rx.spacer(),
                    rx.text(
                        rx.cond(ForgeState.generation_success, "Valid", "Failed"),
                        font_size="14px",
                        font_weight="600",
                        color=rx.cond(ForgeState.generation_success, COLORS["accent_green"], COLORS["accent_red"]),
                    ),
                    width="100%",
                    align="center",
                ),
                rx.cond(
                    ForgeState.has_result,
                    rx.vstack(
                        rx.heading(ForgeState.spec_name, size="4", color=COLORS["text_primary"]),
                        rx.text(ForgeState.spec_description, color=COLORS["text_secondary"], font_size="14px"),
                        rx.separator(color=COLORS["border"]),
                        _block_list("Indicators", ForgeState.spec_indicators, "indicator"),
                        _block_list("Price Action", ForgeState.spec_price_action, "price_action"),
                        _block_list("Entry Rules", ForgeState.spec_entry_rules, "entry_rule"),
                        _block_list("Exit Rules", ForgeState.spec_exit_rules, "exit_rule"),
                        _block_list("Filters", ForgeState.spec_filters, "filter"),
                        _block_list("Money Management", ForgeState.spec_money_management, "money_management"),
                        spacing="3",
                        width="100%",
                    ),
                    rx.fragment(),
                ),
                rx.cond(
                    ForgeState.validation_warnings.length() > 0,
                    rx.vstack(
                        rx.foreach(ForgeState.validation_warnings, lambda w: rx.text(f"⚠️ {w}", font_size="13px", color=COLORS["accent_amber"])),
                        spacing="1",
                        width="100%",
                    ),
                    rx.fragment(),
                ),
                rx.cond(
                    ForgeState.validation_errors.length() > 0,
                    rx.vstack(
                        rx.foreach(ForgeState.validation_errors, lambda e: rx.text(f"❌ {e}", font_size="13px", color=COLORS["accent_red"])),
                        spacing="1",
                        width="100%",
                    ),
                    rx.fragment(),
                ),
                rx.cond(
                    ForgeState.has_result,
                    rx.hstack(
                        rx.button("Save Strategy", on_click=ForgeState.save_current_spec, color_scheme="green"),
                        rx.button("Copy JSON", on_click=rx.set_clipboard(ForgeState.spec_json), variant="outline"),
                        rx.link(rx.button("Backtest", variant="outline", color_scheme="blue"), href="/backtest"),
                        spacing="3",
                    ),
                    rx.fragment(),
                ),
                spacing="4",
                width="100%",
            ),
            **CARD_STYLE,
        ),
        rx.fragment(),
    )


@rx.page(route="/forge", title="ForgeQuant — AI Forge")
def forge_page() -> rx.Component:
    return page_layout("AI Forge", _input_section(), _result_section())
