"""
Breakout Momentum Template.

A momentum strategy that enters on price breakouts above recent
highs, confirmed by ADX trend strength.

Components:
    - Entry: Breakout (20-bar high)
    - Exit: BreakevenStop (activate at 20 pips profit)
    - Money Management: ATRBasedSizing (ATR 14 × 2.0)
    - Filter: ADX threshold (ADX > 25 for strong trend)
"""

BREAKOUT_MOMENTUM = {
    "name": "BreakoutMomentum",
    "description": "Momentum breakout strategy with ADX trend strength confirmation",
    "version": "1.0.0",
    "blocks": {
        "entry": {
            "block_name": "Breakout",
            "params": {
                "lookback": 20,
                "direction": "long",
            },
        },
        "exit": {
            "block_name": "BreakevenStop",
            "params": {
                "activation_pips": 20.0,
                "offset_pips": 2.0,
            },
        },
        "money_management": {
            "block_name": "ATRBasedSizing",
            "params": {
                "atr_period": 14,
                "atr_multiplier": 2.0,
                "risk_percent": 1.0,
            },
        },
        "filters": [
            {
                "block_name": "ThresholdCross",
                "params": {
                    "indicator_name": "ADX",
                    "threshold": 25.0,
                    "indicator_params": {"period": 14},
                },
            },
        ],
    },
    "recommended_timeframes": ["H1", "H4"],
    "recommended_instruments": ["GBPUSD", "GBPJPY", "XAUUSD"],
}
