"""
Utility functions for the market calendar system.
"""
from __future__ import annotations

import datetime
from typing import Optional, Tuple, Union

import pytz
from dateutil import rrule
from dateutil.relativedelta import MO, TU, WE, TH, FR, SA, SU


def is_weekend(dt: datetime.date) -> bool:
    """Check if the given date is a weekend (Saturday or Sunday)."""
    return dt.weekday() >= 5  # 5 = Saturday, 6 = Sunday


def next_weekday(dt: datetime.date, weekday: int) -> datetime.date:
    """Get the next occurrence of a specific weekday (0=Monday, 6=Sunday)."""
    days_ahead = (weekday - dt.weekday()) % 7
    if days_ahead <= 0:  # Target day already happened this week
        days_ahead += 7
    return dt + datetime.timedelta(days=days_ahead)


def previous_weekday(dt: datetime.date, weekday: int) -> datetime.date:
    """Get the previous occurrence of a specific weekday (0=Monday, 6=Sunday)."""
    days_ago = (dt.weekday() - weekday) % 7
    if days_ago == 0:  # Today is the target day
        days_ago = 7
    return dt - datetime.timedelta(days=days_ago)


def nth_weekday_in_month(
    year: int,
    month: int,
    n: int,
    weekday: int
) -> datetime.date:
    """Get the nth occurrence of a specific weekday in a month.
    
    Args:
        year: The year
        month: The month (1-12)
        n: The occurrence (1=first, 2=second, etc., negative counts from the end)
        weekday: The day of the week (0=Monday, 6=Sunday)
        
    Returns:
        The date of the nth weekday in the specified month and year
    """
    if n > 0:
        # 1st, 2nd, 3rd, etc.
        days = [
            dt for dt in rrule.rrule(
                rrule.MONTHLY,
                byweekday=weekday,
                bysetpos=n,
                dtstart=datetime.datetime(year, month, 1),
                count=1
            )
        ]
    else:
        # -1 = last, -2 = second last, etc.
        days = [
            dt for dt in rrule.rrule(
                rrule.MONTHLY,
                byweekday=weekday,
                bysetpos=n if n < 0 else 1,
                dtstart=datetime.datetime(year, month, 1),
                count=abs(n)
            )
        ][-1:]
    
    return days[0].date()


def last_weekday_in_month(year: int, month: int, weekday: int) -> datetime.date:
    """Get the last occurrence of a specific weekday in a month."""
    return nth_weekday_in_month(year, month, -1, weekday)


def get_market_timezone(asset_class: str) -> str:
    """Get the default timezone for a given asset class.
    
    Args:
        asset_class: The asset class (e.g., 'forex', 'equity', 'crypto')
        
    Returns:
        The default timezone for the asset class
    """
    asset_class = asset_class.lower()
    if asset_class in ['forex', 'fx']:
        return 'UTC'
    elif asset_class in ['equity', 'stock', 'stocks']:
        return 'America/New_York'
    elif asset_class in ['futures', 'commodity', 'commodities']:
        return 'America/Chicago'  # CME, CBOT, etc.
    elif asset_class in ['crypto', 'cryptocurrency']:
        return 'UTC'
    else:
        return 'UTC'  # Default to UTC


def localize_datetime(
    dt: datetime.datetime,
    timezone: Union[str, pytz.BaseTzInfo],
    is_dst: Optional[bool] = None
) -> datetime.datetime:
    """Localize a naive datetime to the specified timezone.
    
    Args:
        dt: The datetime to localize
        timezone: The target timezone (string or tzinfo object)
        is_dst: Whether the datetime is in daylight saving time
        
    Returns:
        A timezone-aware datetime in the specified timezone
    """
    if dt.tzinfo is not None:
        return dt.astimezone(pytz.timezone(timezone) if isinstance(timezone, str) else timezone)
    
    if isinstance(timezone, str):
        timezone = pytz.timezone(timezone)
    
    try:
        return timezone.localize(dt, is_dst=is_dst)
    except Exception as e:
        # Fallback to UTC if localization fails
        return pytz.utc.localize(dt)


def normalize_time(
    dt: datetime.datetime,
    timeframe_minutes: int,
    method: str = 'floor'
) -> datetime.datetime:
    """Normalize a datetime to the nearest timeframe boundary.
    
    Args:
        dt: The datetime to normalize
        timeframe_minutes: The timeframe in minutes
        method: How to normalize ('floor', 'ceil', or 'round')
        
    Returns:
        The normalized datetime
    """
    if timeframe_minutes <= 0:
        return dt
    
    # Convert to minutes since epoch
    total_seconds = int(dt.timestamp())
    total_minutes = total_seconds // 60
    
    # Calculate the number of complete timeframes
    if method == 'floor':
        normalized_minutes = (total_minutes // timeframe_minutes) * timeframe_minutes
    elif method == 'ceil':
        normalized_minutes = ((total_minutes + timeframe_minutes - 1) // timeframe_minutes) * timeframe_minutes
    elif method == 'round':
        normalized_minutes = ((total_minutes + timeframe_minutes // 2) // timeframe_minutes) * timeframe_minutes
    else:
        raise ValueError(f"Invalid method: {method}. Use 'floor', 'ceil', or 'round'.")
    
    # Convert back to datetime
    return datetime.datetime.fromtimestamp(normalized_minutes * 60, tz=dt.tzinfo)


def is_market_hours(
    dt: datetime.datetime,
    exchange: str = 'NYSE',
    extended_hours: bool = False
) -> bool:
    """Check if the given datetime is within market hours.
    
    Args:
        dt: The datetime to check
        exchange: The exchange to check market hours for
        extended_hours: Whether to include extended hours
        
    Returns:
        True if the market is open, False otherwise
    """
    # This is a simplified implementation. In a real application, you would
    # use the MarketCalendar class with the appropriate exchange calendar.
    if dt.weekday() >= 5:  # Saturday or Sunday
        return False
    
    time = dt.time()
    
    if exchange.upper() in ['NYSE', 'NASDAQ']:
        # Regular trading hours: 9:30 AM - 4:00 PM ET
        if extended_hours:
            # Extended hours: 4:00 AM - 8:00 PM ET
            return datetime.time(4, 0) <= time < datetime.time(20, 0)
        else:
            return datetime.time(9, 30) <= time < datetime.time(16, 0)
    elif exchange.upper() in ['FOREX', 'FX']:
        # Forex is open 24/5 (closed weekends)
        return True
    elif exchange.upper() in ['CRYPTO', 'CRYPTOCURRENCY']:
        # Crypto is open 24/7
        return True
    else:
        # Default to NYSE hours
        return datetime.time(9, 30) <= time < datetime.time(16, 0)
