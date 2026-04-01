"""
forgequant.frontend.app — Reflex application entry point.
"""

from __future__ import annotations

import reflex as rx

from forgequant.frontend.pages.backtest import backtest_page
from forgequant.frontend.pages.blocks import blocks_page
from forgequant.frontend.pages.dashboard import dashboard_page
from forgequant.frontend.pages.execution import execution_page
from forgequant.frontend.pages.forge import forge_page
from forgequant.frontend.pages.robustness import robustness_page
from forgequant.frontend.pages.settings import settings_page
from forgequant.frontend.pages.strategies import strategies_page

app = rx.App(
    theme=rx.theme(
        appearance="dark",
        accent_color="blue",
        radius="medium",
    ),
)
