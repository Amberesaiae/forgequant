"""
forgequant.frontend.state.forge_state — State for the AI Forge page.
"""

from __future__ import annotations

from typing import Any

import reflex as rx

from forgequant.ai_forge.providers import (
    MODEL_CATALOG,
    PROVIDER_DISPLAY_NAMES,
    PROVIDER_MODEL_IDS,
    PROVIDER_ORDER,
)


class ForgeState(rx.State):
    """
    Manages the AI Forge page's state: idea input, generation, and results.

    The model string is a LiteLLM identifier (e.g. "zai/glm-5-turbo")
    which encodes both provider and model in one field.
    """

    # ── Input fields ────────────────────────────────────────────────────
    idea: str = ""
    selected_provider: str = "glm"
    selected_model: str = "zai/glm-5-turbo"
    temperature: float = 0.7
    use_rag: bool = False

    # ── Generation state ────────────────────────────────────────────────
    is_generating: bool = False
    generation_success: bool = False
    generation_attempted: bool = False
    generation_attempts: int = 0

    # ── Result ──────────────────────────────────────────────────────────
    spec_dict: dict[str, Any] = {}
    spec_json: str = ""
    validation_errors: list[str] = []
    validation_warnings: list[str] = []

    # ── Error ───────────────────────────────────────────────────────────
    error_message: str = ""

    # ── Static data (derived from ai_forge/providers.py catalog) ─────────
    provider_options: list[str] = list(PROVIDER_ORDER)

    provider_display_names: dict[str, str] = dict(PROVIDER_DISPLAY_NAMES)

    model_options: dict[str, list[str]] = {k: list(v) for k, v in PROVIDER_MODEL_IDS.items()}

    model_descriptions: dict[str, str] = {
        mid: (
            f"{info.get('tier', '').title()} | "
            f"{info.get('context_window', 0) // 1000}K ctx | "
            f"{info.get('pricing', '')} — "
            f"{info.get('description', '')[:60]}"
        )
        for mid, info in MODEL_CATALOG.items()
    }

    @rx.var
    def current_model_options(self) -> list[str]:
        return self.model_options.get(self.selected_provider, [])

    @rx.var
    def current_model_description(self) -> str:
        return self.model_descriptions.get(self.selected_model, "")

    @rx.var
    def current_provider_display(self) -> str:
        return self.provider_display_names.get(self.selected_provider, self.selected_provider)

    @rx.var
    def can_generate(self) -> bool:
        return len(self.idea.strip()) > 10 and not self.is_generating

    @rx.var
    def has_result(self) -> bool:
        return self.generation_attempted and len(self.spec_dict) > 0

    @rx.var
    def spec_name(self) -> str:
        return self.spec_dict.get("name", "")

    @rx.var
    def spec_description(self) -> str:
        return self.spec_dict.get("description", "")

    @rx.var
    def spec_timeframe(self) -> str:
        return self.spec_dict.get("timeframe", "")

    @rx.var
    def spec_symbols(self) -> list[str]:
        return self.spec_dict.get("symbols", [])

    @rx.var
    def spec_indicators(self) -> list[dict]:
        return self.spec_dict.get("indicators", [])

    @rx.var
    def spec_price_action(self) -> list[dict]:
        return self.spec_dict.get("price_action", [])

    @rx.var
    def spec_entry_rules(self) -> list[dict]:
        return self.spec_dict.get("entry_rules", [])

    @rx.var
    def spec_exit_rules(self) -> list[dict]:
        return self.spec_dict.get("exit_rules", [])

    @rx.var
    def spec_filters(self) -> list[dict]:
        return self.spec_dict.get("filters", [])

    @rx.var
    def spec_money_management(self) -> list[dict]:
        return self.spec_dict.get("money_management", [])

    @rx.event
    def set_idea(self, value: str) -> None:
        self.idea = value

    @rx.event
    def set_provider(self, value: str) -> None:
        self.selected_provider = value
        models = self.model_options.get(value, [])
        if models:
            self.selected_model = models[0]
        else:
            self.selected_model = "zai/glm-5-turbo"

    @rx.event
    def set_model(self, value: str) -> None:
        self.selected_model = value

    @rx.event
    def set_temperature(self, value: str) -> None:
        try:
            self.temperature = float(value)
        except ValueError:
            pass

    @rx.event
    def toggle_rag(self, value: bool) -> None:
        self.use_rag = value

    @rx.event(background=True)
    async def generate_strategy(self) -> None:
        """Run the AI Forge pipeline."""
        if not self.can_generate:
            return

        async with self:
            self.is_generating = True
            self.generation_attempted = False
            self.error_message = ""
            self.spec_dict = {}
            self.spec_json = ""
            self.validation_errors = []
            self.validation_warnings = []

        try:
            import forgequant.blocks  # noqa: F401 — eager registration

            from forgequant.ai_forge.pipeline import ForgeQuantPipeline, PipelineConfig
            from forgequant.blocks.registry import BlockRegistry

            config = PipelineConfig(
                model=self.selected_model,
                temperature=self.temperature,
                use_rag=self.use_rag,
            )

            pipeline = ForgeQuantPipeline(registry=BlockRegistry(), config=config)
            result = pipeline.generate(self.idea)

            async with self:
                self.generation_attempted = True
                self.generation_attempts = result.attempts

                if result.success and result.spec is not None:
                    self.generation_success = True
                    self.spec_dict = result.spec.model_dump()
                    self.spec_json = result.spec.model_dump_json(indent=2)
                    if result.validation:
                        self.validation_warnings = result.validation.warnings
                else:
                    self.generation_success = False
                    self.validation_errors = result.errors

        except ImportError:
            async with self:
                self.error_message = (
                    "AI Forge dependencies not installed. "
                    "Run: pip install -e '.[ai]'"
                )
        except Exception as exc:
            async with self:
                self.error_message = f"Pipeline error: {exc}"
        finally:
            async with self:
                self.is_generating = False

    @rx.event
    def save_current_spec(self) -> None:
        if self.spec_dict:
            from forgequant.frontend.state.app_state import AppState
            yield AppState.save_strategy(self.spec_dict)

    @rx.event
    def clear_results(self) -> None:
        self.generation_attempted = False
        self.generation_success = False
        self.spec_dict = {}
        self.spec_json = ""
        self.validation_errors = []
        self.validation_warnings = []
        self.error_message = ""
