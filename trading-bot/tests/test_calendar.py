"""
Tests for the market calendar system.
"""
import datetime
import unittest
import pytz
from unittest.mock import patch, MagicMock

from core.calendar import (
    MarketCalendar, ForexCalendar, CryptoCalendar, EquityCalendar, CommodityCalendar,
    CalendarFactory, TradingSession, MarketSessionType, CalendarError
)
from core.instruments import AssetClass, InstrumentMetadata


class TestTradingSession(unittest.TestCase):
    """Test the TradingSession class."""
    
    def setUp(self):
        self.session = TradingSession(
            name="Test Session",
            open_time=datetime.time(9, 30),
            close_time=datetime.time(16, 0),
            timezone="America/New_York",
            session_type=MarketSessionType.REGULAR,
            weekdays=(0, 1, 2, 3, 4)  # Monday to Friday
        )
    
    def test_is_open_at_weekday(self):
        """Test is_open_at on a weekday during session hours."""
        # Monday, 10:00 AM ET
        et = pytz.timezone('America/New_York')
        dt = et.localize(datetime.datetime(2023, 1, 2, 10, 0))  # 10:00 AM ET
        self.assertTrue(self.session.is_open_at(dt))
    
    def test_is_open_at_weekend(self):
        """Test is_open_at on a weekend."""
        # Saturday, 10:00 AM ET
        et = pytz.timezone('America/New_York')
        dt = et.localize(datetime.datetime(2023, 1, 7, 10, 0))  # 10:00 AM ET (Saturday)
        self.assertFalse(self.session.is_open_at(dt))
    
    def test_is_open_at_outside_hours(self):
        """Test is_open_at outside session hours."""
        # Monday, 8:00 AM ET (before open)
        et = pytz.timezone('America/New_York')
        dt = et.localize(datetime.datetime(2023, 1, 2, 8, 0))  # 8:00 AM ET
        self.assertFalse(self.session.is_open_at(dt))


class TestForexCalendar(unittest.TestCase):
    """Test the ForexCalendar class."""
    
    def setUp(self):
        self.calendar = ForexCalendar()
    
    def test_is_trading_day_weekday(self):
        """Test is_trading_day on a weekday."""
        # Monday
        dt = datetime.date(2023, 1, 2)
        self.assertTrue(self.calendar.is_trading_day(dt))
    
    def test_is_trading_day_weekend(self):
        """Test is_trading_day on a weekend."""
        # Saturday
        dt = datetime.date(2023, 1, 7)
        self.assertFalse(self.calendar.is_trading_day(dt))
    
    def test_is_session_open_weekday(self):
        """Test is_session_open on a weekday during market hours."""
        # Monday, 10:00 AM ET
        dt = datetime.datetime(2023, 1, 2, 10, 0, tzinfo=datetime.timezone.utc)
        self.assertTrue(self.calendar.is_session_open(dt))


class TestCryptoCalendar(unittest.TestCase):
    """Test the CryptoCalendar class."""
    
    def setUp(self):
        self.calendar = CryptoCalendar()
    
    def test_is_trading_day_weekday(self):
        """Test is_trading_day on a weekday."""
        # Monday
        dt = datetime.date(2023, 1, 2)
        self.assertTrue(self.calendar.is_trading_day(dt))
    
    def test_is_trading_day_weekend(self):
        """Test is_trading_day on a weekend (crypto markets are 24/7)."""
        # Saturday
        dt = datetime.date(2023, 1, 7)
        self.assertTrue(self.calendar.is_trading_day(dt))
    
    def test_is_session_open_weekend(self):
        """Test is_session_open on a weekend (crypto markets are 24/7)."""
        # Saturday, 10:00 AM UTC
        dt = datetime.datetime(2023, 1, 7, 10, 0, tzinfo=datetime.timezone.utc)
        self.assertTrue(self.calendar.is_session_open(dt))


class TestEquityCalendar(unittest.TestCase):
    """Test the EquityCalendar class."""
    
    def setUp(self):
        self.calendar = EquityCalendar(timezone='America/New_York')
    
    def test_is_trading_day_weekday(self):
        """Test is_trading_day on a regular weekday."""
        # Wednesday
        dt = datetime.date(2023, 1, 4)
        self.assertTrue(self.calendar.is_trading_day(dt))
    
    def test_is_trading_day_weekend(self):
        """Test is_trading_day on a weekend."""
        # Saturday
        dt = datetime.date(2023, 1, 7)
        self.assertFalse(self.calendar.is_trading_day(dt))
    
    def test_is_trading_day_holiday(self):
        """Test is_trading_day on a market holiday."""
        # New Year's Day (observed on Monday, January 2, 2023)
        dt = datetime.date(2023, 1, 2)
        self.assertFalse(self.calendar.is_trading_day(dt))
    
    def test_is_session_open_regular_hours(self):
        """Test is_session_open during regular trading hours."""
        # Set up debug logging
        import logging
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger(__name__)
        
        # Wednesday, 10:00 AM ET
        et = pytz.timezone('America/New_York')
        dt = et.localize(datetime.datetime(2023, 1, 4, 10, 0))  # 10:00 AM ET
        
        logger.debug(f"\n{'='*80}")
        logger.debug(f"Testing is_session_open at {dt} (ET)")
        logger.debug(f"Calendar timezone: {self.calendar.timezone}")
        logger.debug(f"Is trading day: {self.calendar.is_trading_day(dt.date())}")
        
        # Check each session
        logger.debug("\nChecking all sessions:")
        for i, session in enumerate(self.calendar.sessions, 1):
            logger.debug(f"\nSession {i} - {session.name}:")
            logger.debug(f"  Time: {session.open_time} - {session.close_time} {session.timezone}")
            logger.debug(f"  Weekdays: {session.weekdays}")
            logger.debug(f"  Is open: {session.is_open_at(dt)}")
        
        # The actual test
        result = self.calendar.is_session_open(dt)
        logger.debug(f"\nFinal result - is_session_open: {result}")
        logger.debug(f"{'='*80}\n")
        
        self.assertTrue(result)
    
    def test_is_session_open_pre_market(self):
        """Test is_session_open during pre-market hours."""
        # Wednesday, 7:00 AM ET
        et = pytz.timezone('America/New_York')
        dt = et.localize(datetime.datetime(2023, 1, 4, 7, 0))  # 7:00 AM ET
        self.assertFalse(self.calendar.is_session_open(dt))


class TestCommodityCalendar(unittest.TestCase):
    """Test the CommodityCalendar class."""
    
    def setUp(self):
        self.calendar = CommodityCalendar()
    
    def test_is_trading_day_weekday(self):
        """Test is_trading_day on a regular weekday."""
        # Wednesday
        dt = datetime.date(2023, 1, 4)
        self.assertTrue(self.calendar.is_trading_day(dt))
    
    def test_is_session_open_overnight(self):
        """Test is_session_open during overnight session."""
        # Wednesday, 11:00 PM ET (previous day's overnight session)
        dt = datetime.datetime(2023, 1, 5, 4, 0, tzinfo=datetime.timezone.utc)  # 11:00 PM ET
        self.assertTrue(self.calendar.is_session_open(dt))


class TestCalendarFactory(unittest.TestCase):
    """Test the CalendarFactory class."""
    
    def test_get_calendar_forex(self):
        """Test getting a Forex calendar."""
        calendar = CalendarFactory.get_calendar(AssetClass.FOREX)
        self.assertIsInstance(calendar, ForexCalendar)
    
    def test_get_calendar_crypto(self):
        """Test getting a Crypto calendar."""
        calendar = CalendarFactory.get_calendar(AssetClass.CRYPTO)
        self.assertIsInstance(calendar, CryptoCalendar)
    
    def test_get_calendar_equity(self):
        """Test getting an Equity calendar."""
        calendar = CalendarFactory.get_calendar(AssetClass.STOCK)
        self.assertIsInstance(calendar, EquityCalendar)
    
    def test_get_calendar_commodity(self):
        """Test getting a Commodity calendar."""
        calendar = CalendarFactory.get_calendar(AssetClass.COMMODITY)
        self.assertIsInstance(calendar, CommodityCalendar)
    
    def test_get_calendar_for_instrument(self):
        """Test getting a calendar for an instrument."""
        instrument = InstrumentMetadata(
            symbol="AAPL",
            name="Apple Inc.",
asset_class=AssetClass.STOCK,
            exchange="NASDAQ"
        )
        calendar = CalendarFactory.get_calendar_for_instrument(instrument)
        self.assertIsInstance(calendar, EquityCalendar)


if __name__ == "__main__":
    unittest.main()
