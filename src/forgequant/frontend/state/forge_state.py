"""
forgequant.frontend.state.forge_state — State for the AI Forge page.
"""

from __future__ import annotations

from typing import Any

import reflex as rx


class ForgeState(rx.State):
    """Manages the AI Forge page's state."""

    idea: str = ""
    provider: str = "openai"
    model: str = "gpt-4o"
    temperature: float = 0.7
    use_rag: bool = False

    is_generating: bool = False
    generation_success: bool = False
    generation_attempted: bool = False
    generation_attempts: int = 0

    spec_dict: dict[str, Any] = {}
    spec_json: str = ""
    validation_errors: list[str] = []
    validation_warnings: list[str] = []
    error_message: str = ""

    provider_options: list[str] = ["openai", "anthropic", "groq"]
    model_options: dict[str, list[str]] = {
        "openai": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"],
        "anthropic": ["claude-sonnet-4-20250514", "claude-3-5-haiku-20241022"],
        "groq": ["llama-3.1-70b-versatile", "mixtral-8x7b-32768"],
    }

    @rx.var
    def current_model_options(self) -> list[str]:
        return self.model_options.get(self.provider, ["gpt-4o"])

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
        self.provider = value
        models = self.model_options.get(value, ["gpt-4o"])
        self.model = models[0] if models else "gpt-4o"

    @rx.event
    def set_model(self, value: str) -> None:
        self.model = value

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
            import importlib
            for mod in [
                "forgequant.blocks.indicators",
                "forgequant.blocks.price_action",
                "forgequant.blocks.entry_rules",
                "forgequant.blocks.exit_rules",
                "forgequant.blocks.filters",
                "forgequant.blocks.money_management",
            ]:
                importlib.import_module(mod)

            from forgequant.ai_forge.pipeline import ForgeQuantPipeline, PipelineConfig
            from forgequant.blocks.registry import BlockRegistry

            config = PipelineConfig(
                provider=self.provider,
                model=self.model,
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
                self.error_message = "AI Forge dependencies not installed. Run: pip install -e '.[ai]'"
        except Exception as exc:
            async with self:
                self.error_message = f"Pipeline error: {exc}"
        finally:
            async with self:
                self.is_generating = False

    @rx.event
    def save_current_spec(self) -> None:
        if self.spec_dict:
            parent = self.get_parent_state()
            if parent is not None:
                parent.save_strategy(self.spec_dict)

    @rx.event
    def clear_results(self) -> None:
        self.generation_attempted = False
        self.generation_success = False
        self.spec_dict = {}
        self.spec_json = ""
        self.validation_errors = []
        self.validation_warnings = []
        self.error_message = ""
