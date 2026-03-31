"""
Trend Follower Template.

A classic trend-following strategy that enters on moving average
crossovers and exits with an ATR-based trailing stop.

Components:
    - Entry: Crossover (EMA 9 / EMA 21)
    - Exit: TrailingStop (ATR 14 × 2.5)
    - Money Management: FixedRisk (1% per trade)
    - Filter: TrendFilter (EMA 200)
"""

TREND_FOLLOWER = {
    "name": "TrendFollower",
    "description": "Classic trend-following strategy using EMA crossover with trailing stop",
    "version": "1.0.0",
    "blocks": {
        "entry": {
            "block_name": "Crossover",
            "params": {
                "fast_period": 9,
                "slow_period": 21,
                "ma_type": "ema",
            },
        },
        "exit": {
            "block_name": "TrailingStop",
            "params": {
                "atr_period": 14,
                "multiplier": 2.5,
            },
        },
        "money_management": {
            "block_name": "FixedRisk",
            "params": {
                "risk_percent": 1.0,
                "min_volume": 0.01,
                "max_volume": 10.0,
            },
        },
        "filters": [
            {
                "block_name": "TrendFilter",
                "params": {
                    "period": 200,
                    "ma_type": "ema",
                    "buffer_pct": 0.001,
                },
            },
        ],
    },
    "recommended_timeframes": ["H1", "H4", "D1"],
    "recommended_instruments": ["EURUSD", "GBPUSD", "XAUUSD"],
}
