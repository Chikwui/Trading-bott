"""
Custom exceptions for the market calendar system.
"""


class CalendarError(Exception):
    """Base exception for all calendar-related errors."""
    pass


class InvalidTradingSessionError(CalendarError):
    """Raised when an invalid trading session is encountered."""
    pass


class HolidayError(CalendarError):
    """Raised when a date is a holiday but was expected to be a trading day."""
    pass


class NonTradingDayError(CalendarError):
    """Raised when an operation is attempted on a non-trading day."""
    pass


class SessionClosedError(CalendarError):
    """Raised when an operation is attempted outside of trading hours."""
    pass


class InvalidMarketHoursError(CalendarError):
    """Raised when invalid market hours are provided."""
    pass


class TimezoneConversionError(CalendarError):
    """Raised when there's an error converting between timezones."""
    pass
