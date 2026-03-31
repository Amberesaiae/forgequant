"""
Scalper Template.

A short-term scalping strategy using Stochastic oscillator for
entry signals with tight fixed exits and session filtering.

Components:
    - Entry: ThresholdCross (Stochastic %K < 20)
    - Exit: FixedTPSL (TP: 15 pips, SL: 10 pips)
    - Money Management: FixedRisk (0.25% per trade)
    - Filter: TradingSessionFilter (London/NY overlap only)
"""

SCALPER = {
    "name": "Scalper",
    "description": "Short-term scalping strategy using Stochastic with session filtering",
    "version": "1.0.0",
    "blocks": {
        "entry": {
            "block_name": "ThresholdCross",
            "params": {
                "indicator_name": "Stochastic",
                "threshold": 20.0,
                "indicator_params": {
                    "k_period": 14,
                    "d_period": 3,
                    "smooth_k": 3,
                },
            },
        },
        "exit": {
            "block_name": "FixedTPSL",
            "params": {
                "tp_pips": 15.0,
                "sl_pips": 10.0,
            },
        },
        "money_management": {
            "block_name": "FixedRisk",
            "params": {
                "risk_percent": 0.25,
                "min_volume": 0.01,
                "max_volume": 10.0,
            },
        },
        "filters": [
            {
                "block_name": "TradingSessionFilter",
                "params": {
                    "session": "london_ny_overlap",
                },
            },
            {
                "block_name": "SpreadFilter",
                "params": {
                    "max_spread_pips": 2.0,
                },
            },
        ],
    },
    "recommended_timeframes": ["M1", "M5"],
    "recommended_instruments": ["EURUSD", "USDJPY"],
}
