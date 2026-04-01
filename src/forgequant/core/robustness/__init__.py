"""
Robustness testing suite.

Provides multiple independent robustness tests that a strategy must
pass before being considered viable.
"""

from forgequant.core.robustness.walk_forward import WalkForwardAnalysis, WalkForwardResult
from forgequant.core.robustness.monte_carlo import MonteCarloAnalysis, MonteCarloResult
from forgequant.core.robustness.cpcv import CPCVAnalysis, CPCVResult
from forgequant.core.robustness.parameter_sensitivity import (
    ParameterSensitivity,
    SensitivityResult,
)
from forgequant.core.robustness.stability import EquityStability, StabilityResult
from forgequant.core.robustness.suite import RobustnessSuite, RobustnessVerdict

__all__ = [
    "WalkForwardAnalysis",
    "WalkForwardResult",
    "MonteCarloAnalysis",
    "MonteCarloResult",
    "CPCVAnalysis",
    "CPCVResult",
    "ParameterSensitivity",
    "SensitivityResult",
    "EquityStability",
    "StabilityResult",
    "RobustnessSuite",
    "RobustnessVerdict",
]
