"""
forgequant.cli — Command-line interface for ForgeQuant.

Usage:
    forgequant --help
    forgequant info
    forgequant blocks
    forgequant generate "RSI mean-reversion on EURUSD 1H with ATR stops"
"""

from __future__ import annotations

import argparse
import json
import sys


def _ensure_blocks_registered() -> None:
    """Import all block sub-packages to trigger @BlockRegistry.register decorators."""
    import forgequant.blocks.indicators  # noqa: F401
    import forgequant.blocks.price_action  # noqa: F401
    import forgequant.blocks.entry_rules  # noqa: F401
    import forgequant.blocks.exit_rules  # noqa: F401
    import forgequant.blocks.filters  # noqa: F401
    import forgequant.blocks.money_management  # noqa: F401


def _cmd_info() -> None:
    """Print platform info."""
    from forgequant import __version__
    from forgequant.core.config import get_settings

    settings = get_settings()
    print(f"ForgeQuant v{__version__}")
    print(f"  Environment : {settings.forgequant_env.value}")
    print(f"  Log level   : {settings.forgequant_log_level}")
    print(f"  Log format  : {settings.forgequant_log_format.value}")
    print(f"  OpenAI key  : {'set' if settings.openai_api_key else 'not set'}")
    print(f"  Anthropic   : {'set' if settings.anthropic_api_key else 'not set'}")
    print(f"  Groq key    : {'set' if settings.groq_api_key else 'not set'}")
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
            if cls is None:
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


def _cmd_generate(idea: str) -> None:
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

    config = PipelineConfig(provider="openai", model="gpt-4o")
    pipeline = ForgeQuantPipeline(registry=BlockRegistry(), config=config)

    print(f"Generating strategy for: {idea!r}\n")
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

    subparsers.add_parser("info", help="Show platform configuration")
    subparsers.add_parser("blocks", help="List all registered blocks")
    subparsers.add_parser("catalog", help="Print full block catalog as JSON")

    gen_parser = subparsers.add_parser("generate", help="Generate a strategy via AI Forge")
    gen_parser.add_argument("idea", type=str, help="Strategy idea in plain English")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "info":
        _cmd_info()
    elif args.command == "blocks":
        _cmd_blocks()
    elif args.command == "catalog":
        _cmd_catalog()
    elif args.command == "generate":
        _cmd_generate(args.idea)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
