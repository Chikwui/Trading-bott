"""
Market calendar and session management for the trading bot.

This package provides functionality to determine trading sessions, market hours,
and holidays for different asset classes and instruments.
"""

from .market_calendar import (
    MarketCalendar,
    ForexCalendar,
    CryptoCalendar,
    EquityCalendar,
    CommodityCalendar,
    CalendarFactory,
    TradingSession,
    MarketSessionType
)
from .exceptions import (
    CalendarError,
    InvalidTradingSessionError,
    HolidayError,
    NonTradingDayError,
    SessionClosedError,
    InvalidMarketHoursError,
    TimezoneConversionError
)
from . import utils

__all__ = [
    'MarketCalendar',
    'ForexCalendar',
    'CryptoCalendar',
    'EquityCalendar',
    'CommodityCalendar',
    'CalendarFactory',
    'TradingSession',
    'MarketSessionType',
    'CalendarError',
    'InvalidTradingSessionError',
    'HolidayError',
    'NonTradingDayError',
    'SessionClosedError',
    'InvalidMarketHoursError',
    'TimezoneConversionError',
    'utils'
]
