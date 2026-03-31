# ForgeQuant

**Open-source, high-precision systematic strategy generation platform.**

Our archetype of StrategyQuant X — built entirely in Python.

## What Is ForgeQuant?

ForgeQuant combines:

- **AI Forge**: Describe a trading idea in natural language → get a validated, production-ready strategy.
- **Building Blocks System**: 22 modular, composable blocks (indicators, entries, exits, money management, filters).
- **Genetic Evolution Engine**: Evolve and breed strategies across multiple islands.
- **VectorBT Fast Evaluation**: Ultra-fast vectorized backtesting and parameter optimization.
- **Robustness Testing Suite**: Walk-Forward, Combinatorial Purged CV, Monte Carlo (4 types), parameter stability, regime testing.
- **Strict Quality Gates**: Only strategies that pass all tests proceed to compilation.
- **aiomql Execution**: Seamless live trading on MetaTrader 5.
- **Reflex Dashboard**: Modern, TradingView-inspired UI for monitoring and control.

## Tech Stack

| Layer               | Technology                     |
|---------------------|--------------------------------|
| Language            | Python 3.12+                   |
| Package Manager     | uv                             |
| Configuration       | Pydantic v2 + YAML             |
| Backtesting         | VectorBT                       |
| AI/LLM              | OpenAI/Anthropic via Instructor|
| RAG                 | ChromaDB                       |
| Live Execution      | aiomql (MetaTrader 5)          |
| UI Dashboard        | Reflex                         |
| Logging             | structlog                      |
| Linting             | Ruff                           |
| Type Checking       | mypy (strict)                  |
| Testing             | pytest + pytest-asyncio        |

## Quick Start

### 1. Clone and enter project directory
```bash
git clone https://github.com/yourusername/forgequant.git
cd forgequant
```

### 2. Create environment and install dependencies
```bash
uv sync
```

### 3. Copy and configure environment variables
```bash
cp .env.example .env
# Edit .env with your actual credentials
```

### 4. Verify installation
```bash
uv run python -c "from core.config import settings; print('ForgeQuant ready.')"
```

## Project Structure

```
forgequant/
├── core/                        # Engine: generator, robustness, compiler, workflow
├── ai_forge/                    # AI natural language strategy generation
├── execution/                   # aiomql live trading integration
├── frontend/                    # Reflex UI dashboard
├── strategies_library/          # Building blocks, templates, safety rules
├── data/                        # Price data storage
├── knowledge_base/              # RAG embeddings
├── notebooks/                   # Research and exploration
├── scripts/                     # CLI entry points
└── tests/                       # Unit and integration tests
```

## Development Phases

- [x] Phase 1: Project Foundation
- [ ] Phase 2: Strategies Library (Building Blocks + Safety)
- [ ] Phase 3: Templates + AI Forge
- [ ] Phase 4: Genetic Evolution Engine
- [ ] Phase 5: Robustness Testing Suite
- [ ] Phase 6: Compiler + Execution Layer
- [ ] Phase 7: Reflex UI Dashboard
- [ ] Phase 8: Orchestrator + Deployment

## License

MIT
