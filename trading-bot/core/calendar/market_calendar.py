"""
Market calendar implementation for the trading bot.

This module provides functionality to determine trading sessions, market hours,
and holidays for different asset classes and instruments.
"""
from __future__ import annotations

import abc
import datetime
import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Union

import pandas as pd
import pytz
from dateutil import rrule
from dateutil.relativedelta import MO, TU, WE, TH, FR, SA, SU

from core.instruments import AssetClass, InstrumentMetadata
from .exceptions import (
    CalendarError, InvalidTradingSessionError, HolidayError, NonTradingDayError,
    SessionClosedError, InvalidMarketHoursError, TimezoneConversionError
)

logger = logging.getLogger(__name__)


class MarketSessionType(Enum):
    """Types of market sessions."""
    PRE_MARKET = auto()
    REGULAR = auto()
    POST_MARKET = auto()
    EXTENDED_HOURS = auto()
    OVERNIGHT = auto()
    WEEKEND = auto()


@dataclass
class TradingSession:
    """Represents a trading session with open and close times."""
    name: str
    open_time: datetime.time
    close_time: datetime.time
    timezone: str
    session_type: MarketSessionType = MarketSessionType.REGULAR
    weekdays: Tuple[Union[int, Any], ...] = (MO, TU, WE, TH, FR)  # Monday to Friday by default
    
    def __init__(
        self,
        name: str,
        open_time: datetime.time,
        close_time: datetime.time,
        timezone: str,
        session_type: MarketSessionType = MarketSessionType.REGULAR,
        weekdays: Tuple[Union[int, Any], ...] = (MO, TU, WE, TH, FR)  # Monday to Friday by default
    ):
        self.name = name
        self.open_time = open_time
        self.close_time = close_time
        self.timezone = timezone
        self.session_type = session_type
        
        # Convert integer weekdays to dateutil.relativedelta constants if needed
        weekday_mapping = {
            0: MO, 1: TU, 2: WE, 3: TH, 4: FR, 5: SA, 6: SU,
            'MO': MO, 'TU': TU, 'WE': WE, 'TH': TH, 'FR': FR, 'SA': SA, 'SU': SU,
            'monday': MO, 'tuesday': TU, 'wednesday': WE, 'thursday': TH,
            'friday': FR, 'saturday': SA, 'sunday': SU
        }
        
        self.weekdays = tuple(
            weekday_mapping.get(day, day) if not hasattr(day, 'weekday') else day 
            for day in weekdays
        )
    
    def is_open_at(self, dt: datetime.datetime) -> bool:
        """Check if the session is open at the given datetime."""
        try:
            # Get the timezone for this session
            tz = pytz.timezone(self.timezone)
            
            # Log the input datetime and its timezone
            logger.debug(f"[is_open_at] Input datetime: {dt} (tz: {dt.tzinfo})")
            
            # Convert input datetime to the session's timezone
            if dt.tzinfo is None:
                dt = pytz.utc.localize(dt)
            dt_local = dt.astimezone(tz)
            
            logger.debug(f"[is_open_at] Session '{self.name}' checking at local time: {dt_local}")
            logger.debug(f"[is_open_at] Session timezone: {self.timezone}")
            logger.debug(f"[is_open_at] Session open: {self.open_time}, close: {self.close_time}")
            
            # Check if it's the correct day of the week
            current_weekday = dt_local.weekday()  # 0=Monday, 6=Sunday
            
            # Convert weekdays to integers for comparison
            weekday_mapping = {
                MO: 0, TU: 1, WE: 2, TH: 3, FR: 4, SA: 5, SU: 6,
                'MO': 0, 'TU': 1, 'WE': 2, 'TH': 3, 'FR': 4, 'SA': 5, 'SU': 6,
                'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
                'friday': 4, 'saturday': 5, 'sunday': 6
            }
            
            # Convert each weekday in self.weekdays to its integer representation
            session_weekdays = []
            for day in self.weekdays:
                if hasattr(day, 'weekday'):
                    # It's a dateutil.relativedelta constant
                    session_weekdays.append(day.weekday)
                elif day in weekday_mapping:
                    # It's a string or other mappable value
                    session_weekdays.append(weekday_mapping[day])
                else:
                    # Assume it's already an integer
                    session_weekdays.append(day)
            
            logger.debug(f"[is_open_at] Current weekday: {current_weekday} (0=Monday, 6=Sunday)")
            logger.debug(f"[is_open_at] Valid weekdays for session: {session_weekdays}")
            
            if current_weekday not in session_weekdays:
                logger.debug(f"[is_open_at] Session '{self.name}' is not open on weekday {current_weekday} (valid weekdays: {session_weekdays})")
                return False
            
            # Get the current time in the session's timezone
            current_time = dt_local.time()
            
            # Log the current time and session times for debugging
            logger.debug(f"[is_open_at] Current time: {current_time}")
            logger.debug(f"[is_open_at] Session open time: {self.open_time}, close time: {self.close_time}")
            
            # Check if the current time is within the session hours
            if self.close_time > self.open_time:
                # Normal case: session doesn't cross midnight (e.g., 9:30 AM - 4:00 PM)
                is_open = self.open_time <= current_time < self.close_time
                logger.debug(f"[is_open_at] Normal session check: {self.open_time} <= {current_time} < {self.close_time} = {is_open}")
                return is_open
            
            # For sessions that cross midnight (e.g., 22:00-04:00 or 18:00-17:00)
            is_open = current_time >= self.open_time or current_time < self.close_time
            logger.debug(f"[is_open_at] Overnight session check: {current_time} >= {self.open_time} OR {current_time} < {self.close_time} = {is_open}")
            
            # Special debug for the failing test case
            if hasattr(self, 'name') and self.name == "Overnight":
                logger.debug(f"[DEBUG] Overnight session check details:")
                logger.debug(f"- Current time: {current_time}")
                logger.debug(f"- Open time: {self.open_time}")
                logger.debug(f"- Close time: {self.close_time}")
                logger.debug(f"- Open <= current: {self.open_time <= current_time}")
                logger.debug(f"- Current < close: {current_time < self.close_time}")
                logger.debug(f"- Result: {is_open}")
            return is_open
            
        except Exception as e:
            logger.error(f"Error checking if session is open: {e}", exc_info=True)
            return False


class MarketCalendar(abc.ABC):
    """Abstract base class for market calendars."""
    
    def __init__(self, timezone: str = 'UTC', exchange: Optional[str] = None):
        """Initialize the market calendar.
        
        Args:
            timezone: The timezone for the market (e.g., 'America/New_York' for NYSE)
            exchange: The exchange code (e.g., 'NYSE', 'NASDAQ', 'CME')
        """
        self.timezone = pytz.timezone(timezone)
        self.exchange = exchange
        self.sessions: List[TradingSession] = []
        self.holidays: Set[datetime.date] = set()
        self._setup_calendar()
    
    @abc.abstractmethod
    def _setup_calendar(self) -> None:
        """Set up the trading sessions and holidays for this calendar."""
        pass
    
    def is_trading_day(self, dt: datetime.date) -> bool:
        """Check if the given date is a trading day."""
        try:
            dt = self._ensure_date(dt)
            
            # Check if it's a weekend (default implementation)
            if dt.weekday() >= 5:  # Saturday or Sunday
                return False
                
            # Check if it's a holiday
            if dt in self.holidays:
                return False
                
            return True
        except Exception as e:
            logger.error(f"Error checking if trading day: {e}", exc_info=True)
            return False
    
    def is_session_open(self, dt: Optional[datetime.datetime] = None) -> bool:
        """Check if the market is currently open.
        
        For equity markets, this only returns True during regular trading hours (9:30 AM - 4:00 PM ET).
        For commodity markets, it checks if the current time is within any defined session.
        For other markets, it returns True if any session is open.
        """
        try:
            dt = dt or datetime.datetime.now(datetime.timezone.utc)
            
            # Convert to the calendar's timezone
            if dt.tzinfo is None:
                dt = pytz.utc.localize(dt)
            dt = dt.astimezone(self.timezone)
            
            logger.debug(f"[is_session_open] Checking if session is open at {dt} (timezone: {self.timezone})")
            
            # Check if it's a trading day
            if not self.is_trading_day(dt.date()):
                logger.debug(f"[is_session_open] Not a trading day: {dt.date()}")
                return False
            
            # For commodity markets, check if any session is open
            if isinstance(self, CommodityCalendar):
                for session in self.sessions:
                    if session.is_open_at(dt):
                        logger.debug(f"[is_session_open] Session '{session.name}' is open at {dt.time()}")
                        return True
                logger.debug(f"[is_session_open] No sessions open at {dt.time()}")
                return False
            
            # For equity markets, only return True during regular trading hours
            if isinstance(self, EquityCalendar):
                current_time = dt.time()
                regular_market_open = datetime.time(9, 30)  # 9:30 AM ET
                regular_market_close = datetime.time(16, 0)  # 4:00 PM ET
                
                is_open = regular_market_open <= current_time < regular_market_close
                logger.debug(f"[is_session_open] Equity market check: {regular_market_open} <= {current_time} < {regular_market_close} = {is_open}")
                return is_open
            
            # For other markets, check if any session is open
            for session in self.sessions:
                if session.is_open_at(dt):
                    logger.debug(f"[is_session_open] Session '{session.name}' is open at {dt.time()}")
                    return True
            logger.debug(f"[is_session_open] No sessions open at {dt.time()}")
            return False
        except Exception as e:
            logger.error(f"Error checking if session is open: {e}", exc_info=True)
            return False
    
    def next_trading_day(self, dt: Optional[datetime.date] = None) -> datetime.date:
        """Get the next trading day."""
        dt = self._ensure_date(dt or datetime.datetime.now(datetime.timezone.utc).date())
        
        next_day = dt + datetime.timedelta(days=1)
        while not self.is_trading_day(next_day):
            next_day += datetime.timedelta(days=1)
            
        return next_day
    
    def previous_trading_day(self, dt: Optional[datetime.date] = None) -> datetime.date:
        """Get the previous trading day."""
        dt = self._ensure_date(dt or datetime.datetime.now(datetime.timezone.utc).date())
        
        prev_day = dt - datetime.timedelta(days=1)
        while not self.is_trading_day(prev_day):
            prev_day -= datetime.timedelta(days=1)
            
        return prev_day
    
    def get_current_session(self, dt: Optional[datetime.datetime] = None) -> Optional[TradingSession]:
        """Get the current trading session, if any."""
        try:
            dt = dt or datetime.datetime.now(datetime.timezone.utc)
            dt = dt.astimezone(self.timezone)
            
            for session in self.sessions:
                if session.is_open_at(dt):
                    return session
                    
            return None
        except Exception as e:
            logger.error(f"Error getting current session: {e}", exc_info=True)
            return None
    
    def get_market_hours(self, dt: datetime.date) -> List[Tuple[datetime.datetime, datetime.datetime]]:
        """Get the market hours for the given date."""
        try:
            if not self.is_trading_day(dt):
                return []
                
            result = []
            for session in self.sessions:
                open_dt = datetime.datetime.combine(dt, session.open_time, tzinfo=self.timezone)
                close_dt = datetime.datetime.combine(dt, session.close_time, tzinfo=self.timezone)
                
                # Handle sessions that cross midnight
                if session.close_time < session.open_time:
                    close_dt += datetime.timedelta(days=1)
                    
                result.append((open_dt, close_dt))
                
            return result
        except Exception as e:
            logger.error(f"Error getting market hours: {e}", exc_info=True)
            return []
    
    def _ensure_date(self, dt: Union[datetime.date, datetime.datetime]) -> datetime.date:
        """Ensure the input is a date object."""
        if isinstance(dt, datetime.datetime):
            return dt.astimezone(self.timezone).date()
        return dt


class ForexCalendar(MarketCalendar):
    """Forex market calendar (24/5)."""
    
    def _setup_calendar(self) -> None:
        # Forex is open 24/5 (Monday to Friday) - use UTC as the base timezone
        self.sessions = [
            TradingSession(
                name="Forex",
                open_time=datetime.time(0, 0),  # Start of day in UTC
                close_time=datetime.time(23, 59, 59, 999999),  # End of day in UTC
                timezone='UTC',  # Always use UTC for Forex
                session_type=MarketSessionType.REGULAR,
                weekdays=(MO, TU, WE, TH, FR)
            )
        ]
        
        # Add major forex market holidays (minimal for forex)
        self._add_forex_holidays()
        
    def is_session_open(self, dt: Optional[datetime.datetime] = None) -> bool:
        """Check if the market is currently open."""
        if dt is None:
            dt = datetime.datetime.now(datetime.timezone.utc)
            
        # Convert to UTC if it has timezone info
        if dt.tzinfo is not None:
            dt = dt.astimezone(datetime.timezone.utc)
            
        # Check if it's a trading day (Mon-Fri)
        if not self.is_trading_day(dt.date()):
            return False
            
        # Check if it's a holiday
        if dt.date() in self.holidays:
            return False
            
        # Forex is open 24 hours on trading days
        return True
    
    def _add_forex_holidays(self) -> None:
        """Add major forex market holidays."""
        # Forex is closed on major holidays (varies by broker)
        current_year = datetime.datetime.now().year
        
        # New Year's Day
        self.holidays.add(datetime.date(current_year, 1, 1))
        
        # Christmas Day
        self.holidays.add(datetime.date(current_year, 12, 25))


class CryptoCalendar(MarketCalendar):
    """Cryptocurrency market calendar (24/7)."""
    
    def _setup_calendar(self) -> None:
        # Crypto is open 24/7
        self.sessions = [
            TradingSession(
                name="Crypto",
                open_time=datetime.time(0, 0),
                close_time=datetime.time(23, 59, 59, 999999),
                timezone='UTC',  # Crypto markets are typically UTC-based
                session_type=MarketSessionType.REGULAR,
                weekdays=(MO, TU, WE, TH, FR, SA, SU)  # All days of the week
            )
        ]
        
    def is_trading_day(self, dt: datetime.date) -> bool:
        """Check if the given date is a trading day.
        
        For crypto markets, every day is a trading day.
        """
        try:
            dt = self._ensure_date(dt)
            return True  # Crypto markets are always open
        except Exception as e:
            logger.error(f"Error checking if trading day: {e}", exc_info=True)
            return False
            
    def is_session_open(self, dt: Optional[datetime.datetime] = None) -> bool:
        """Check if the market is currently open.
        
        For crypto markets, the market is always open.
        """
        try:
            dt = dt or datetime.datetime.now(datetime.timezone.utc)
            dt = dt.astimezone(self.timezone)
            return True  # Crypto markets are always open
        except Exception as e:
            logger.error(f"Error checking if session is open: {e}", exc_info=True)
            return False


class EquityCalendar(MarketCalendar):
    """Equity market calendar (e.g., NYSE, NASDAQ)."""
    
    def __init__(self, timezone: str = 'America/New_York', exchange: str = 'NYSE'):
        """Initialize the equity market calendar.
        
        Args:
            timezone: The timezone for the market (e.g., 'America/New_York' for NYSE)
            exchange: The exchange code (e.g., 'NYSE', 'NASDAQ')
        """
        self.exchange = exchange
        super().__init__(timezone=timezone, exchange=exchange)
    
    def _setup_calendar(self) -> None:
        # Regular trading hours (e.g., 9:30 AM - 4:00 PM ET for NYSE)
        self.sessions = [
            # Pre-market
            TradingSession(
                name="Pre-Market",
                open_time=datetime.time(4, 0),  # 4:00 AM ET
                close_time=datetime.time(9, 30),  # 9:30 AM ET
                timezone=self.timezone.zone,
                session_type=MarketSessionType.PRE_MARKET,
                weekdays=(MO, TU, WE, TH, FR)
            ),
            # Regular session
            TradingSession(
                name="Regular Trading",
                open_time=datetime.time(9, 30),  # 9:30 AM ET
                close_time=datetime.time(16, 0),  # 4:00 PM ET
                timezone=self.timezone.zone,
                session_type=MarketSessionType.REGULAR,
                weekdays=(MO, TU, WE, TH, FR)
            ),
            # After-hours
            TradingSession(
                name="After-Hours",
                open_time=datetime.time(16, 0),  # 4:00 PM ET
                close_time=datetime.time(20, 0),  # 8:00 PM ET
                timezone=self.timezone.zone,
                session_type=MarketSessionType.POST_MARKET,
                weekdays=(MO, TU, WE, TH, FR)
            )
        ]
        
        # Add equity market holidays
        self._add_equity_holidays()
    
    def _add_equity_holidays(self) -> None:
        """Add US equity market holidays."""
        current_year = datetime.datetime.now().year
        
        # Add holidays for a range of years to cover testing
        # This includes past years to handle test cases
        for year in range(current_year - 5, current_year + 10):
            # New Year's Day (January 1, or Monday if on weekend)
            new_years_day = self._get_observed_holiday(
                datetime.date(year, 1, 1),
                roll_forward=False
            )
            self.holidays.add(new_years_day)
            
            # Martin Luther King Jr. Day (third Monday in January)
            mlk_day = self._nth_weekday_in_month(year, 1, 3, 0)  # 3rd Monday in January
            self.holidays.add(mlk_day)
            
            # Washington's Birthday / Presidents Day (third Monday in February)
            presidents_day = self._nth_weekday_in_month(year, 2, 3, 0)  # 3rd Monday in February
            self.holidays.add(presidents_day)
            
            # Good Friday (varies)
            good_friday = self._calculate_good_friday(year)
            self.holidays.add(good_friday)
            
            # Memorial Day (last Monday in May)
            memorial_day = self._last_weekday_in_month(year, 5, 0)  # Last Monday in May
            self.holidays.add(memorial_day)
            
            # Juneteenth (June 19, or Friday before if on weekend)
            juneteenth = self._get_observed_holiday(
                datetime.date(year, 6, 19),
                roll_forward=False
            )
            self.holidays.add(juneteenth)
            
            # Independence Day (July 4, or Friday before/Monday after if on weekend)
            independence_day = self._get_observed_holiday(
                datetime.date(year, 7, 4),
                roll_forward=True
            )
            self.holidays.add(independence_day)
            
            # Labor Day (first Monday in September)
            labor_day = self._nth_weekday_in_month(year, 9, 1, 0)  # 1st Monday in September
            self.holidays.add(labor_day)
            
            # Thanksgiving Day (fourth Thursday in November)
            thanksgiving = self._nth_weekday_in_month(year, 11, 4, 3)  # 4th Thursday in November
            self.holidays.add(thanksgiving)
            
            # Day after Thanksgiving (fourth Friday in November)
            day_after_thanksgiving = self._nth_weekday_in_month(year, 11, 4, 4)  # 4th Friday in November
            self.holidays.add(day_after_thanksgiving)
            
            # Christmas Day (December 25, or Friday before/Monday after if on weekend)
            christmas = self._get_observed_holiday(
                datetime.date(year, 12, 25),
                roll_forward=True
            )
            self.holidays.add(christmas)
    
    def _nth_weekday_in_month(self, year: int, month: int, n: int, weekday: int) -> datetime.date:
        """Get the nth occurrence of a specific weekday in a month."""
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
    
    def _last_weekday_in_month(self, year: int, month: int, weekday: int) -> datetime.date:
        """Get the last specific weekday in a month."""
        return self._nth_weekday_in_month(year, month, -1, weekday)
    
    def _calculate_good_friday(self, year: int) -> datetime.date:
        """Calculate Good Friday (Friday before Easter Sunday)."""
        # Using the Meeus/Jones/Butcher algorithm to calculate Easter Sunday
        a = year % 19
        b = year // 100
        c = year % 100
        d = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i = c // 4
        k = c % 4
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) // 451
        month = (h + l - 7 * m + 114) // 31
        day = ((h + l - 7 * m + 114) % 31) + 1
        
        # Easter Sunday
        easter_sunday = datetime.date(year, month, day)
        # Good Friday is 2 days before Easter Sunday
        good_friday = easter_sunday - datetime.timedelta(days=2)
        
        return good_friday
    
    def _get_observed_holiday(self, holiday: datetime.date, roll_forward: bool = True) -> datetime.date:
        """Get the observed date for a holiday that falls on a weekend.
        
        Args:
            holiday: The actual holiday date
            roll_forward: If True, observe the holiday on the next business day if it falls on a weekend.
                         If False, observe on the previous business day.
        """
        # For New Year's Day, always observe on the next business day if it falls on a weekend
        if holiday.month == 1 and holiday.day == 1:
            if holiday.weekday() == 5:  # Saturday
                return holiday + datetime.timedelta(days=2)  # Following Monday
            elif holiday.weekday() == 6:  # Sunday
                return holiday + datetime.timedelta(days=1)  # Following Monday
            return holiday
            
        # For other holidays, use the roll_forward parameter
        if holiday.weekday() == 5:  # Saturday
            if roll_forward:
                # For Saturday, observe on the following Monday
                return holiday + datetime.timedelta(days=2)
            else:
                # Or previous Friday if rolling backward
                return holiday - datetime.timedelta(days=1)
        elif holiday.weekday() == 6:  # Sunday
            if roll_forward:
                # For Sunday, observe on the following Monday
                return holiday + datetime.timedelta(days=1)
            else:
                # Or previous Friday if rolling backward
                return holiday - datetime.timedelta(days=2)
        return holiday


class CommodityCalendar(EquityCalendar):
    """Commodity market calendar (e.g., CME, COMEX)."""
    
    def _setup_calendar(self) -> None:
        # Set the timezone to 'America/New_York' for commodity markets (CME, COMEX, etc.)
        self.timezone = pytz.timezone('America/New_York')
        
        # Regular trading hours for commodities (e.g., 6:00 PM - 5:00 PM ET next day for CME)
        self.sessions = [
            # Overnight session (6:00 PM - 5:00 PM ET next day)
            TradingSession(
                name="Overnight",
                open_time=datetime.time(18, 0),  # 6:00 PM ET
                close_time=datetime.time(17, 0),  # 5:00 PM ET next day
                timezone='America/New_York',  # Explicitly set to ET
                session_type=MarketSessionType.OVERNIGHT,
                weekdays=(SU, MO, TU, WE, TH, FR, SA)  # All days of the week
            )
        ]
        
        # Add commodity market holidays (similar to equities but may vary by exchange)
        self._add_equity_holidays()


class CalendarFactory:
    """Factory for creating market calendars."""
    
    @classmethod
    def get_calendar(
        cls,
        asset_class: AssetClass,
        timezone: str = 'UTC',
        exchange: Optional[str] = None
    ) -> MarketCalendar:
        """Get a market calendar for the given asset class.
        
        Args:
            asset_class: The asset class (e.g., FOREX, CRYPTO, STOCK, COMMODITY)
            timezone: The timezone for the market (e.g., 'America/New_York')
            exchange: Optional exchange name for more specific calendars

        Returns:
            A MarketCalendar instance for the specified asset class
        """
        if asset_class == AssetClass.FOREX:
            return ForexCalendar(timezone=timezone)
        elif asset_class == AssetClass.CRYPTO:
            return CryptoCalendar(timezone=timezone)
        elif asset_class == AssetClass.STOCK:
            return EquityCalendar(timezone=timezone, exchange=exchange)
        elif asset_class == AssetClass.COMMODITY:
            return CommodityCalendar(timezone=timezone, exchange=exchange)
        else:
            logger.warning(f"No specific calendar for asset class {asset_class}, using default")
            return MarketCalendar(timezone=timezone)

    @classmethod
    def get_calendar_for_instrument(
        cls,
        instrument: InstrumentMetadata,
        timezone: Optional[str] = None
    ) -> MarketCalendar:
        """Get a market calendar for the given instrument.
        
        Args:
            instrument: The instrument metadata
            timezone: Optional timezone override
            
        Returns:
            A MarketCalendar instance for the instrument
        """
        timezone = timezone or getattr(instrument, 'timezone', None) or 'UTC'
        return cls.get_calendar(
            asset_class=instrument.asset_class,
            timezone=timezone,
            exchange=getattr(instrument, 'exchange', None)
        )
