"""Tests to close remaining coverage gaps."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from forgequant.core.engine.metrics import compute_metrics
from forgequant.core.engine.results import BacktestResult, TradeRecord
from forgequant.core.compiler.signal_assembler import assemble_signals, _find_column
from forgequant.core.compiler.compiled_strategy import BlockOutput, CompiledStrategy
from forgequant.ai_forge.schemas import BlockSpec, StrategySpec


class TestMetricsLongWinRate:
    """Cover metrics.py line 141 - long_win_rate when no long trades."""

    def _make_result(self, equity_values, trades):
        n = len(equity_values)
        dates = pd.date_range("2024-01-01", periods=n, freq="D")
        equity = pd.Series(equity_values, index=dates, name="equity")
        return BacktestResult(
            strategy_name="test",
            start_date=dates[0].to_pydatetime(),
            end_date=dates[-1].to_pydatetime(),
            initial_equity=equity_values[0],
            final_equity=equity_values[-1],
            equity_curve=equity,
            trades=trades,
        )

    def _make_short_trade(self, pnl_dollar):
        return TradeRecord(
            trade_id=0, direction="short",
            entry_time=datetime(2024, 1, 1), entry_price=100,
            exit_time=datetime(2024, 1, 2), exit_price=100 - pnl_dollar,
            exit_reason="signal", position_size=1.0,
            pnl=pnl_dollar, pnl_pct=pnl_dollar,
            pnl_dollar=pnl_dollar, bars_held=5,
        )

    def test_only_short_trades(self):
        trades = [self._make_short_trade(100), self._make_short_trade(-50)]
        equity = [100000 + i * 50 for i in range(50)]
        result = compute_metrics(self._make_result(equity, trades))
        assert result.metrics["long_trades"] == 0.0
        assert result.metrics["long_win_rate"] == 0.0
        assert result.metrics["short_trades"] == 2.0


class TestSignalAssemblerPullback:
    """Cover signal_assembler.py lines 150, 154 - pullback_long/short columns."""

    def test_pullback_columns(self):
        spec = StrategySpec(
            name="pullback_test",
            description="Strategy with pullback entry for testing signal assembler.",
            objective={"style": "trend_following", "timeframe": "1h"},
            indicators=[BlockSpec(block_name="ema")],
            entry_rules=[BlockSpec(block_name="crossover_entry")],
            exit_rules=[BlockSpec(block_name="fixed_tpsl")],
            money_management=BlockSpec(block_name="fixed_risk"),
        )
        n = 20
        idx = pd.date_range("2024-01-01", periods=n, freq="h")
        ohlcv = pd.DataFrame(
            {
                "open": np.ones(n) * 100, "high": np.ones(n) * 101,
                "low": np.ones(n) * 99, "close": np.ones(n) * 100,
                "volume": np.ones(n) * 1000,
            },
            index=idx,
        )
        compiled = CompiledStrategy(spec=spec, ohlcv=ohlcv)

        pullback_df = pd.DataFrame(
            {
                "pullback_long": [False] * 10 + [True] + [False] * 9,
                "pullback_short": [False] * 15 + [True] + [False] * 4,
            },
            index=idx,
        )
        compiled.block_outputs["pullback_entry"] = BlockOutput(
            "pullback_entry", "price_action", {}, pullback_df,
        )

        compiled.block_outputs["fixed_tpsl"] = BlockOutput(
            "fixed_tpsl", "exit_rule", {},
            pd.DataFrame({"tpsl_long_sl": np.ones(n) * 98}, index=idx),
        )
        compiled.block_outputs["fixed_risk"] = BlockOutput(
            "fixed_risk", "money_management", {},
            pd.DataFrame({"fr_position_size": np.ones(n) * 100}, index=idx),
        )

        result = assemble_signals(compiled)
        assert result.entry_long is not None
        assert result.entry_long.sum() == 1
        assert result.entry_short is not None
        assert result.entry_short.sum() == 1


class TestFindColumnAstypeFailure:
    """Cover signal_assembler.py lines 66-68 - astype(bool) failure."""

    def test_astype_bool_fails(self):
        from forgequant.core.compiler.signal_assembler import _find_column
        from forgequant.core.types import SIGNAL_COLUMNS as SC
        from forgequant.core.compiler.compiled_strategy import BlockOutput

        output = BlockOutput(
            block_name="test",
            category="entry_rule",
            params={},
            result=pd.DataFrame({"crossover_long_entry": [1, 2, 3]}),
        )

        series_mock = MagicMock()
        series_mock.dtype = "int64"
        series_mock.astype.side_effect = ValueError("cannot convert to bool")

        with patch.object(pd.DataFrame, "__getitem__", return_value=series_mock):
            result = _find_column(output, SC.entry_long_patterns)
        assert result is None


class TestCompilerBlockException:
    """Cover compiler.py line 122 - block execution exception."""

    def test_block_execution_exception(self):
        from forgequant.blocks.registry import BlockRegistry
        from forgequant.blocks.base import BaseBlock
        from forgequant.blocks.metadata import BlockMetadata, ParameterSpec
        from forgequant.core.types import BlockCategory, BlockParams, BlockResult
        from forgequant.core.compiler.compiler import StrategyCompiler
        from forgequant.core.exceptions import StrategyCompileError
        from forgequant.ai_forge.schemas import BlockSpec, StrategySpec

        class FailingBlock(BaseBlock):
            metadata = BlockMetadata(
                name="failing_block",
                display_name="Failing Block",
                category=BlockCategory.INDICATOR,
                description="A block that always fails.",
                parameters=(),
                tags=("test",),
            )

            def compute(self, data: pd.DataFrame, params: BlockParams) -> BlockResult:
                raise RuntimeError("Intentional failure")

        registry = BlockRegistry()
        registry.register_class(FailingBlock)

        compiler = StrategyCompiler(registry=registry, validate=False)

        spec = StrategySpec(
            name="fail_test",
            description="Strategy to test compiler exception handling path.",
            objective={"style": "trend_following", "timeframe": "1h"},
            indicators=[BlockSpec(block_name="failing_block")],
            entry_rules=[BlockSpec(block_name="crossover_entry")],
            exit_rules=[BlockSpec(block_name="fixed_tpsl")],
            money_management=BlockSpec(block_name="fixed_risk"),
        )

        n = 50
        idx = pd.date_range("2024-01-01", periods=n, freq="h")
        ohlcv = pd.DataFrame(
            {
                "open": np.ones(n) * 100, "high": np.ones(n) * 101,
                "low": np.ones(n) * 99, "close": np.ones(n) * 100,
                "volume": np.ones(n) * 1000,
            },
            index=idx,
        )

        with pytest.raises(StrategyCompileError, match="execution failed"):
            compiler.compile(spec, ohlcv)


class TestTradingSessionOvernight:
    """Cover trading_session.py line 102-103 - session2 overnight."""

    def test_session2_overnight(self):
        from forgequant.blocks.filters.trading_session import TradingSessionFilter

        session = TradingSessionFilter()
        dates = pd.date_range("2024-01-15", periods=24, freq="h")
        n = len(dates)
        df = pd.DataFrame(
            {
                "open": [100.0] * n, "high": [101.0] * n,
                "low": [99.0] * n, "close": [100.0] * n,
                "volume": [1000.0] * n,
            },
            index=dates,
        )
        result = session.execute(
            df,
            {
                "session1_start": 8, "session1_end": 16,
                "session2_start": 22, "session2_end": 6,
            },
        )
        active = result["session_active"]
        names = result["session_name"]
        for i, dt in enumerate(dates):
            h = dt.hour
            expected = (8 <= h < 16) or (h >= 22) or (h < 6)
            assert active.iloc[i] == expected, f"Hour {h}"
            if 22 <= h or h < 6:
                assert "session_2" in names.iloc[i] or names.iloc[i] == "overlap"


class TestCLIGenerateViaMain:
    """Cover cli.py lines 134-135 - generate command through main()."""

    def test_generate_via_main(self, capsys):
        from forgequant.cli import main

        mock_spec = MagicMock()
        mock_spec.model_dump_json.return_value = '{"name": "test"}'
        mock_result = MagicMock()
        mock_result.success = True
        mock_result.spec = mock_spec
        mock_result.validation = None

        mock_pipeline_cls = MagicMock()
        mock_pipeline_cls.return_value.generate.return_value = mock_result

        with patch("forgequant.ai_forge.pipeline.ForgeQuantPipeline", mock_pipeline_cls):
            with patch("forgequant.ai_forge.pipeline.PipelineConfig"):
                main(["generate", "test idea"])

        out = capsys.readouterr().out
        assert "successfully" in out
