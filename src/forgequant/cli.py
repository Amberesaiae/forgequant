"""
forgequant.cli — Command-line interface for ForgeQuant.

Usage:
    forgequant --help
    forgequant info
    forgequant blocks
    forgequant models
    forgequant generate "RSI mean-reversion on EURUSD 1H with ATR stops"
    forgequant generate "Breakout strategy" --model zai/glm-4.7-flash
"""

from __future__ import annotations

import argparse
import json
import sys


def _ensure_blocks_registered() -> None:
    """Import all block sub-packages to trigger @BlockRegistry.register decorators."""
    import forgequant.blocks  # noqa: F401 — eager registration


def _cmd_info() -> None:
    """Print platform info."""
    from forgequant import __version__
    from forgequant.ai_forge.providers import (
        PROVIDER_DISPLAY_NAMES,
        PROVIDER_ORDER,
        is_provider_configured,
    )
    from forgequant.core.config import get_settings

    settings = get_settings()
    print(f"ForgeQuant v{__version__}")
    print(f"  Environment : {settings.forgequant_env.value}")
    print(f"  Log level   : {settings.forgequant_log_level}")
    print(f"  Log format  : {settings.forgequant_log_format.value}")
    print()
    print("  LLM Providers (via LiteLLM):")
    for prov in PROVIDER_ORDER:
        status = "configured" if is_provider_configured(prov) else "not set"
        display = PROVIDER_DISPLAY_NAMES.get(prov, prov)
        marker = " <- PRIMARY" if prov == "glm" else ""
        print(f"    {display:<20s} {status}{marker}")
    print()
    print(f"  MT5 path    : {settings.mt5_terminal_path or 'not set'}")


def _cmd_blocks() -> None:
    """List all registered blocks."""
    from forgequant.blocks.registry import BlockRegistry

    _ensure_blocks_registered()
    registry = BlockRegistry()

    print(f"Registered blocks ({registry.count()}):\n")
    for category in registry.count_by_category():
        names = [cls.metadata.name for cls in registry.list_by_category(category)]
        print(f"  [{category.value}]")
        for name in names:
            cls = registry.get(name)
            if cls is None:  # pragma: no cover
                continue
            desc = cls.metadata.description[:80] if cls.metadata.description else ""
            print(f"    {name:<30s} {desc}")
        print()


def _cmd_catalog() -> None:
    """Print the full block catalog as JSON."""
    from forgequant.blocks.registry import BlockRegistry

    _ensure_blocks_registered()
    catalog = BlockRegistry().to_catalog_dict()
    print(json.dumps(catalog, indent=2))


def _cmd_models() -> None:
    """Print all available LLM models grouped by provider."""
    from forgequant.ai_forge.providers import (
        MODEL_CATALOG,
        PROVIDER_DISPLAY_NAMES,
        PROVIDER_MODEL_IDS,
        PROVIDER_ORDER,
        is_provider_configured,
    )

    for prov in PROVIDER_ORDER:
        status = "configured" if is_provider_configured(prov) else "not set"
        display = PROVIDER_DISPLAY_NAMES.get(prov, prov)
        print(f"\n  [{status}] {display}")
        print(f"  {'-' * 60}")

        model_ids = PROVIDER_MODEL_IDS.get(prov, [])
        for mid in model_ids:
            info = MODEL_CATALOG[mid]
            tier = f"[{info['tier']}]"
            ctx = f"{info['context_window'] // 1000}K ctx"
            price = info.get("pricing", "")
            print(f"    {mid:<40s} {tier:<14s} {ctx:<10s} {price}")
            print(f"    {'':40s} {info['description'][:70]}")
        print()


def _cmd_generate(idea: str, model: str) -> None:
    """Run the AI Forge pipeline for a given strategy idea."""
    try:
        from forgequant.ai_forge.pipeline import ForgeQuantPipeline, PipelineConfig
    except ImportError:
        print(
            "ERROR: AI Forge dependencies not installed.\n"
            "Run:  pip install -e '.[ai]'",
            file=sys.stderr,
        )
        sys.exit(1)

    from forgequant.blocks.registry import BlockRegistry
    from forgequant.core.logging import configure_logging

    configure_logging()
    _ensure_blocks_registered()

    config = PipelineConfig(model=model)
    pipeline = ForgeQuantPipeline(registry=BlockRegistry(), config=config)

    print(f"Generating strategy for: {idea!r}")
    print(f"Model: {model}\n")
    result = pipeline.generate(idea)

    if result.success:
        print("Strategy spec generated successfully!\n")
        print(result.spec.model_dump_json(indent=2))
    else:
        print("Generation failed:\n", file=sys.stderr)
        for error in result.errors:
            print(f"  - {error}", file=sys.stderr)
        sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="forgequant",
        description="ForgeQuant — systematic strategy generation platform.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("info", help="Show platform configuration and provider status")
    subparsers.add_parser("blocks", help="List all registered blocks")
    subparsers.add_parser("catalog", help="Print full block catalog as JSON")
    subparsers.add_parser("models", help="List all available LLM models")

    gen_parser = subparsers.add_parser("generate", help="Generate a strategy via AI Forge")
    gen_parser.add_argument("idea", type=str, help="Strategy idea in plain English")
    gen_parser.add_argument(
        "--model", "-m",
        type=str,
        default="zai/glm-5-turbo",
        help="LiteLLM model string (default: zai/glm-5-turbo)",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "info": _cmd_info,
        "blocks": _cmd_blocks,
        "catalog": _cmd_catalog,
        "models": _cmd_models,
    }

    if args.command in commands:
        commands[args.command]()
    elif args.command == "generate":
        _cmd_generate(args.idea, args.model)
    else:  # pragma: no cover
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
