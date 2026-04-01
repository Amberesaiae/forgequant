"""
Strategy compiler.

Transforms a validated StrategySpec into a runnable CompiledStrategy
by instantiating blocks, executing them on OHLCV data, and assembling
the outputs into unified signal matrices.
"""

from forgequant.core.compiler.compiled_strategy import BlockOutput, CompiledStrategy
from forgequant.core.compiler.compiler import StrategyCompiler
from forgequant.core.compiler.signal_assembler import assemble_signals

__all__ = [
    "BlockOutput",
    "CompiledStrategy",
    "StrategyCompiler",
    "assemble_signals",
]
