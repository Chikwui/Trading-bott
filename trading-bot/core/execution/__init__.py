"""
Advanced order execution system with ML integration.

This module provides a comprehensive framework for executing trades with advanced algorithms,
risk management, and ML-powered optimization.
"""

from .base import ExecutionClient
from .base_broker import (
    BaseBroker,
    Order,
    OrderType,
    OrderSide,
    OrderStatus,
    TimeInForce,
    Position,
    AccountInfo,
)
from .models import ExecutionParameters, SlippageModel, MarketImpactModel
from .clients import (
    TWAPExecutionClient,
    VWAPExecutionClient,
    IcebergExecutionClient,
    SniperExecutionClient,
    SmartRouter
)
from .monitoring import ExecutionMonitor
from .risk import ExecutionRiskManager

__all__ = [
    'ExecutionClient',
    'ExecutionParameters',
    'SlippageModel',
    'MarketImpactModel',
    'TWAPExecutionClient',
    'VWAPExecutionClient',
    'IcebergExecutionClient',
    'SniperExecutionClient',
    'SmartRouter',
    'ExecutionMonitor',
    'ExecutionRiskManager'
]
