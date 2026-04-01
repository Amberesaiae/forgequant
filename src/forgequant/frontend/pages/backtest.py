"""
forgequant.frontend.pages.backtest — Backtesting page.
"""

from __future__ import annotations

import reflex as rx

from forgequant.frontend.components.layout import page_layout
from forgequant.frontend.components.stat_card import stat_card
from forgequant.frontend.state.backtest_state import BacktestState
from forgequant.frontend.styles import CARD_STYLE, COLORS, INPUT_STYLE


def _config_section() -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.text("Backtest Configuration", font_size="16px", font_weight="600", color=COLORS["text_primary"]),
            rx.hstack(
                rx.vstack(
                    rx.text("Strategy", font_size="12px", color=COLORS["text_muted"]),
                    rx.select(BacktestState.available_strategies, value=BacktestState.selected_strategy_name, on_change=BacktestState.set_strategy, placeholder="Select a strategy..."),
                    spacing="1",
                    flex="1",
                ),
                rx.vstack(rx.text("Symbol", font_size="12px", color=COLORS["text_muted"]), rx.input(value=BacktestState.symbol, on_change=BacktestState.set_symbol, style=INPUT_STYLE), spacing="1", width="120px"),
                rx.vstack(rx.text("Timeframe", font_size="12px", color=COLORS["text_muted"]), rx.select(["1M", "5M", "15M", "1H", "4H", "1D"], value=BacktestState.timeframe, on_change=BacktestState.set_timeframe), spacing="1", width="100px"),
                spacing="4",
                flex_wrap="wrap",
                width="100%",
            ),
            rx.button(
                rx.cond(BacktestState.is_running, rx.text("Running..."), rx.text("Run Backtest")),
                on_click=BacktestState.run_backtest,
                disabled=BacktestState.is_running,
                color_scheme="blue",
                width="100%",
                size="3",
            ),
            spacing="4",
            width="100%",
        ),
        **CARD_STYLE,
    )


def _results_section() -> rx.Component:
    return rx.cond(
        BacktestState.has_result,
        rx.vstack(
            rx.hstack(
                stat_card("Sharpe Ratio", BacktestState.sharpe_ratio, COLORS["accent_blue"]),
                stat_card("Max Drawdown", BacktestState.max_drawdown, COLORS["accent_red"]),
                stat_card("Profit Factor", BacktestState.profit_factor, COLORS["accent_green"]),
                stat_card("Total Trades", BacktestState.total_trades.to(str), COLORS["accent_purple"]),
                stat_card("Win Rate", BacktestState.win_rate, COLORS["accent_amber"]),
                stat_card("Net Profit", BacktestState.net_profit, COLORS["accent_green"]),
                spacing="4",
                flex_wrap="wrap",
                width="100%",
            ),
            rx.box(
                rx.vstack(
                    rx.text("Trade Log", font_size="16px", font_weight="600", color=COLORS["text_primary"]),
                    rx.table.root(
                        rx.table.header(rx.table.row(
                            rx.table.column_header_cell("Time"), rx.table.column_header_cell("Side"),
                            rx.table.column_header_cell("Size"), rx.table.column_header_cell("Entry"),
                            rx.table.column_header_cell("Exit"), rx.table.column_header_cell("P&L"),
                        )),
                        rx.table.body(rx.foreach(BacktestState.trades, lambda t: rx.table.row(
                            rx.table.cell(t["time"]),
                            rx.table.cell(rx.text(t["side"], color=rx.cond(t["side"] == "LONG", COLORS["accent_green"], COLORS["accent_red"]), font_weight="600")),
                            rx.table.cell(str(t["size"])),
                            rx.table.cell(f"{t['entry']:.4f}"),
                            rx.table.cell(f"{t['exit']:.4f}"),
                            rx.table.cell(rx.text(f"${t['pnl']:.2f}", color=rx.cond(t["pnl"] >= 0, COLORS["accent_green"], COLORS["accent_red"]), font_weight="600")),
                        ))),
                        width="100%",
                    ),
                    spacing="3",
                    width="100%",
                ),
                **CARD_STYLE,
            ),
            rx.link(rx.button("Run Robustness Suite", size="3", color_scheme="purple"), href="/robustness"),
            spacing="5",
            width="100%",
        ),
        rx.cond(BacktestState.error_message != "", rx.callout(BacktestState.error_message, icon="alert_triangle", color_scheme="red"), rx.fragment()),
    )


@rx.page(route="/backtest", title="ForgeQuant — Backtest")
def backtest_page() -> rx.Component:
    return page_layout("Backtest", _config_section(), _results_section())
