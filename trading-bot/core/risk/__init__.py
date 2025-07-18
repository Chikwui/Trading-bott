"""
Core risk management system for the trading bot.

This package implements advanced risk management techniques including:
- Position sizing (Kelly Criterion, Volatility-based)
- Risk metrics (VaR, CVaR, Maximum Drawdown)
- Circuit breakers and risk limits
- Correlation and exposure management
"""

from .position_sizing import PositionSizer
from .risk_metrics import RiskMetrics
from .circuit_breakers import CircuitBreaker
from .exposure_manager import ExposureManager

__all__ = ['PositionSizer', 'RiskMetrics', 'CircuitBreaker', 'ExposureManager']
