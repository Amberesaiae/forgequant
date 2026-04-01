# ForgeQuant

**Open-source StrategyQuant X–style systematic strategy generation platform.**

ForgeQuant provides a modular, AI-assisted framework for generating, evaluating, and
deploying systematic trading strategies. It uses composable building blocks — indicators,
price action patterns, entry/exit rules, money management, and filters — that can be
assembled by humans or by LLM-driven pipelines.

## Status

🚧 **Pre-Alpha** — under active development. Not suitable for live trading yet.

## Quick Start

```bash
# Clone
git clone https://github.com/Amberesaiae/forgequant.git
cd forgequant

# Create virtual environment (Python 3.12+)
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Install in development mode
pip install -e ".[dev]"

# Copy and configure environment
cp .env.example .env
# Edit .env with your API keys and preferences

# Run tests
pytest
```

## Architecture

```
forgequant/
├── core/         # Config, logging, exceptions, type definitions
├── blocks/       # Composable strategy building blocks
│   ├── indicators/        # Technical indicators (EMA, RSI, MACD, etc.)
│   ├── price_action/      # Price action patterns
│   ├── entry_rules/       # Entry signal generators
│   ├── exit_rules/        # Exit signal generators
│   ├── money_management/  # Position sizing
│   └── filters/           # Trade filters
├── ai_forge/     # LLM-driven strategy specification
├── execution/    # Live execution (MetaTrader 5)
└── frontend/     # Reflex-based dashboard
```

## License

MIT
