"""
Execution algorithms for the trading system.

This package contains various execution algorithms that can be used
by the OrderExecutionService to execute orders in different ways.
"""

# Export algorithm implementations
from .base import ExecutionAlgorithm
from .twap import TWAPExecutor
from .vwap import VWAPExecutor
from .iceberg import IcebergExecutor
from .sniper import SniperExecutor

__all__ = [
    'ExecutionAlgorithm',
    'TWAPExecutor',
    'VWAPExecutor',
    'IcebergExecutor',
    'SniperExecutor',
]
