"""
Backtesting Module

This module provides tools for backtesting trading strategies, including:
- Backtester: Main backtesting engine
- BacktestResult: Container for backtest results and metrics
- Data handlers for various data sources
- Performance analysis and visualization tools
"""

from .backtester import Backtester
from .result import BacktestResult

# Define public API
__all__ = [
    'Backtester',
    'BacktestResult',
]
