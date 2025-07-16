"""
Core modules for the trading bot.
"""
from .instruments import (
    InstrumentMetadata,
    InstrumentManager,
    InstrumentFactory,
    InstrumentRegistry,
    AssetClass,
    InstrumentType,
    TradingHours
)

__all__ = [
    'InstrumentMetadata',
    'InstrumentManager',
    'InstrumentFactory',
    'InstrumentRegistry',
    'AssetClass',
    'InstrumentType',
    'TradingHours'
]
