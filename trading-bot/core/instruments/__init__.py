"""
Instrument management module for the trading bot.
"""
from .metadata import (
    InstrumentMetadata,
    InstrumentManager,
    AssetClass,
    InstrumentType,
    TradingHours
)
from .factory import InstrumentFactory
from .registry import InstrumentRegistry

__all__ = [
    'InstrumentMetadata',
    'InstrumentManager',
    'AssetClass',
    'InstrumentType',
    'TradingHours',
    'InstrumentFactory',
    'InstrumentRegistry'
]
