"""
Mean Reversion Template.

A mean-reversion strategy that enters when RSI crosses below 30
(oversold) and exits at a fixed take profit.

Components:
    - Entry: ThresholdCross (RSI < 30)
    - Exit: FixedTPSL (TP: 40 pips, SL: 25 pips)
    - Money Management: FixedRisk (0.5% per trade)
    - Filter: MaxDrawdownFilter (8% max drawdown)
"""

MEAN_REVERSION = {
    "name": "MeanReversion",
    "description": "Mean reversion strategy using RSI oversold entry with fixed TP/SL",
    "version": "1.0.0",
    "blocks": {
        "entry": {
            "block_name": "ThresholdCross",
            "params": {
                "indicator_name": "RSI",
                "threshold": 30.0,
                "indicator_params": {"period": 14},
            },
        },
        "exit": {
            "block_name": "FixedTPSL",
            "params": {
                "tp_pips": 40.0,
                "sl_pips": 25.0,
            },
        },
        "money_management": {
            "block_name": "FixedRisk",
            "params": {
                "risk_percent": 0.5,
                "min_volume": 0.01,
                "max_volume": 10.0,
            },
        },
        "filters": [
            {
                "block_name": "MaxDrawdownFilter",
                "params": {
                    "max_drawdown_pct": 0.08,
                    "lookback": 252,
                },
            },
        ],
    },
    "recommended_timeframes": ["M15", "M30", "H1"],
    "recommended_instruments": ["EURUSD", "USDJPY", "AUDUSD"],
}
