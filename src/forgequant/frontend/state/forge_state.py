"""
forgequant.frontend.state.forge_state — State for the AI Forge page.
"""

from __future__ import annotations

from typing import Any

import reflex as rx


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

    # ── Static data (provider/model options from catalog) ───────────────
    provider_options: list[str] = ["glm", "openai", "anthropic", "groq"]

    provider_display_names: dict[str, str] = {
        "glm": "GLM (Z.ai)",
        "openai": "OpenAI",
        "anthropic": "Anthropic",
        "groq": "Groq",
    }

    # LiteLLM model IDs grouped by provider
    model_options: dict[str, list[str]] = {
        "glm": [
            "zai/glm-5-turbo",
            "zai/glm-5.1",
            "zai/glm-5",
            "zai/glm-4.7",
            "zai/glm-4.7-flash",
            "zai/glm-4.6",
            "zai/glm-4.5",
            "zai/glm-4.5-flash",
            "zai/glm-4.5-air",
        ],
        "openai": [
            "openai/gpt-4o",
            "openai/gpt-4o-mini",
            "openai/gpt-4-turbo",
        ],
        "anthropic": [
            "anthropic/claude-sonnet-4-20250514",
            "anthropic/claude-3-5-haiku-20241022",
        ],
        "groq": [
            "groq/llama-3.1-70b-versatile",
            "groq/mixtral-8x7b-32768",
        ],
    }

    # Human-readable descriptions for model selector
    model_descriptions: dict[str, str] = {
        "zai/glm-5-turbo": "Turbo · 48 tok/s · 203K ctx · $0.96/M in — Best for iteration speed",
        "zai/glm-5.1": "Flagship (post-training) · 200K ctx · 94.6% of Claude Opus coding",
        "zai/glm-5": "Flagship (open-source, MIT) · 205K ctx · Complex systems engineering",
        "zai/glm-4.7": "Production workhorse · 200K ctx · Rivals Claude Sonnet 4",
        "zai/glm-4.7-flash": "Free tier · Lightweight · Zero cost",
        "zai/glm-4.6": "Previous flagship · 200K ctx · Strong coding",
        "zai/glm-4.5": "Previous gen · 128K ctx · 355B MoE",
        "zai/glm-4.5-flash": "Free tier · Previous gen · Zero cost",
        "zai/glm-4.5-air": "Lightweight · Agent-centric · Compact MoE",
        "openai/gpt-4o": "OpenAI flagship · 128K ctx · Multimodal",
        "openai/gpt-4o-mini": "OpenAI compact · 128K ctx · Cost-effective",
        "openai/gpt-4-turbo": "OpenAI previous gen · 128K ctx",
        "anthropic/claude-sonnet-4-20250514": "Anthropic balanced · 200K ctx",
        "anthropic/claude-3-5-haiku-20241022": "Anthropic fast · 200K ctx",
        "groq/llama-3.1-70b-versatile": "Llama 3.1 70B on Groq · Ultra-fast",
        "groq/mixtral-8x7b-32768": "Mixtral MoE on Groq · Budget",
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
