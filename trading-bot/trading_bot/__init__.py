"""
Trading Bot - An AI-powered algorithmic trading system.

This package provides a comprehensive framework for algorithmic trading across
multiple asset classes including Forex, Cryptocurrencies, Commodities, and Indices.
"""

__version__ = "0.1.0"
__author__ = "Your Name <your.email@example.com>"
__license__ = "MIT"

# Import core components
from .app import TradingBot
from .config import settings
from .services import MarketDataService, SignalService
from .core.market import MarketDataHandler, ExecutionHandler, Portfolio, RiskManager
from .core.calendar import CalendarFactory

# Define public API
__all__ = [
    'TradingBot',
    'MarketDataService',
    'SignalService',
    'MarketDataHandler',
    'ExecutionHandler',
    'Portfolio',
    'RiskManager',
    'CalendarFactory',
    'settings'
]
