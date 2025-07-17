"""
Trading Strategies Module

This module contains various trading strategies that can be used with the backtesting framework.
Each strategy implements a common interface for signal generation and position management.
"""
from .base_strategy import BaseStrategy
from .multi_timeframe_strategy import MultiTimeframeStrategy
from .ml_based_strategy import MLBasedStrategy
from .market_regime_strategy import MarketRegimeStrategy
from .order_flow_strategy import OrderFlowStrategy
from .mean_reversion_strategy import MeanReversionStrategy
from .breakout_strategy import BreakoutStrategy
from .smart_money_strategy import SmartMoneyStrategy
from .adaptive_trend_strategy import AdaptiveTrendStrategy
from .pairs_trading_strategy import PairsTradingStrategy
from .rl_trading_strategy import RLTradingStrategy
from .sentiment_strategy import SentimentStrategy

__all__ = [
    'BaseStrategy',
    'MultiTimeframeStrategy',
    'MLBasedStrategy',
    'MarketRegimeStrategy',
    'OrderFlowStrategy',
    'MeanReversionStrategy',
    'BreakoutStrategy',
    'SmartMoneyStrategy',
    'AdaptiveTrendStrategy',
    'PairsTradingStrategy',
    'RLTradingStrategy',
    'SentimentStrategy'
]
