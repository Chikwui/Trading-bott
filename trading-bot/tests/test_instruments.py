"""
Tests for the instrument management system.
"""
import unittest
from datetime import datetime, time
from pathlib import Path
import tempfile
import shutil
import json

from core.instruments import (
    InstrumentMetadata,
    InstrumentManager,
    InstrumentFactory,
    InstrumentRegistry,
    AssetClass,
    InstrumentType,
    TradingHours
)


class TestTradingHours(unittest.TestCase):
    """Test the TradingHours class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.regular_hours = TradingHours(
            open_time=time(9, 30),  # 9:30 AM
            close_time=time(16, 0),  # 4:00 PM
            timezone="America/New_York",
            days=[0, 1, 2, 3, 4],  # Monday to Friday
            is_24h=False
        )
        
        self.crypto_hours = TradingHours(
            open_time=time(0, 0),
            close_time=time(23, 59, 59),
            timezone="UTC",
            days=list(range(7)),  # 7 days a week
            is_24h=True
        )
    
    def test_is_market_open_regular(self):
        """Test regular market hours."""
        # Monday 10:00 AM
        dt = datetime(2023, 1, 2, 10, 0)  # Monday
        self.assertTrue(self.regular_hours.is_market_open(dt))
        
        # Friday 4:00 PM (market close)
        dt = datetime(2023, 1, 6, 16, 0)  # Friday
        self.assertTrue(self.regular_hours.is_market_open(dt))
        
        # Saturday 10:00 AM (market closed)
        dt = datetime(2023, 1, 7, 10, 0)  # Saturday
        self.assertFalse(self.regular_hours.is_market_open(dt))
        
        # Monday 9:29 AM (before open)
        dt = datetime(2023, 1, 2, 9, 29)  # Monday
        self.assertFalse(self.regular_hours.is_market_open(dt))
    
    def test_is_market_open_crypto(self):
        """Test 24/7 crypto market hours."""
        # Weekday
        dt = datetime(2023, 1, 2, 10, 0)  # Monday
        self.assertTrue(self.crypto_hours.is_market_open(dt))
        
        # Weekend
        dt = datetime(2023, 1, 7, 10, 0)  # Saturday
        self.assertTrue(self.crypto_hours.is_market_open(dt))
        
        # Midnight
        dt = datetime(2023, 1, 2, 0, 0)  # Monday midnight
        self.assertTrue(self.crypto_hours.is_market_open(dt))
    
    def test_serialization(self):
        """Test serialization to/from dict."""
        data = self.regular_hours.to_dict()
        new_hours = TradingHours.from_dict(data)
        
        self.assertEqual(self.regular_hours.open_time, new_hours.open_time)
        self.assertEqual(self.regular_hours.close_time, new_hours.close_time)
        self.assertEqual(self.regular_hours.timezone, new_hours.timezone)
        self.assertEqual(self.regular_hours.days, new_hours.days)
        self.assertEqual(self.regular_hours.is_24h, new_hours.is_24h)


class TestInstrumentMetadata(unittest.TestCase):
    """Test the InstrumentMetadata class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.trading_hours = TradingHours(
            open_time=time(0, 0),
            close_time=time(23, 59, 59),
            timezone="UTC",
            days=list(range(7)),
            is_24h=True
        )
        
        self.instrument = InstrumentMetadata(
            symbol="BTCUSDT",
            name="Bitcoin / Tether",
            asset_class=AssetClass.CRYPTO,
            instrument_type=InstrumentType.COIN,
            base_currency="BTC",
            quote_currency="USDT",
            exchange="Binance",
            lot_size=1.0,
            min_lot_size=0.0001,
            max_lot_size=1000.0,
            lot_step=0.0001,
            tick_size=0.01,
            tick_value=1.0,
            margin_required=0.1,
            leverage=10.0,
            trading_hours=self.trading_hours,
            tags={"crypto", "digital_assets", "top_market_cap"},
            is_active=True
        )
    
    def test_initialization(self):
        """Test instrument initialization."""
        self.assertEqual(self.instrument.symbol, "BTCUSDT")
        self.assertEqual(self.instrument.asset_class, AssetClass.CRYPTO)
        self.assertEqual(self.instrument.instrument_type, InstrumentType.COIN)
        self.assertEqual(self.instrument.base_currency, "BTC")
        self.assertEqual(self.instrument.quote_currency, "USDT")
        self.assertEqual(self.instrument.exchange, "Binance")
        self.assertEqual(self.instrument.leverage, 10.0)
        self.assertIn("crypto", self.instrument.tags)
        self.assertTrue(self.instrument.is_active)
    
    def test_tag_management(self):
        """Test tag management methods."""
        # Test adding a tag
        self.instrument.add_tag("defi")
        self.assertIn("defi", self.instrument.tags)
        
        # Test removing a tag
        self.instrument.remove_tag("defi")
        self.assertNotIn("defi", self.instrument.tags)
        
        # Test has_tag
        self.assertTrue(self.instrument.has_tag("crypto"))
        self.assertFalse(self.instrument.has_tag("nonexistent"))
        
        # Test has_any_tag
        self.assertTrue(self.instrument.has_any_tag(["crypto", "stocks"]))
        self.assertFalse(self.instrument.has_any_tag(["stocks", "bonds"]))
        
        # Test has_all_tags
        self.assertTrue(self.instrument.has_all_tags(["crypto", "digital_assets"]))
        self.assertFalse(self.instrument.has_all_tags(["crypto", "stocks"]))
    
    def test_serialization(self):
        """Test serialization to/from dict."""
        # Convert to dict and back
        data = self.instrument.to_dict()
        new_instrument = InstrumentMetadata.from_dict(data)
        
        # Verify all attributes match
        self.assertEqual(self.instrument.symbol, new_instrument.symbol)
        self.assertEqual(self.instrument.asset_class, new_instrument.asset_class)
        self.assertEqual(self.instrument.instrument_type, new_instrument.instrument_type)
        self.assertEqual(self.instrument.base_currency, new_instrument.base_currency)
        self.assertEqual(self.instrument.quote_currency, new_instrument.quote_currency)
        self.assertEqual(self.instrument.exchange, new_instrument.exchange)
        self.assertEqual(self.instrument.leverage, new_instrument.leverage)
        self.assertEqual(self.instrument.tags, new_instrument.tags)
        self.assertEqual(self.instrument.is_active, new_instrument.is_active)
        
        # Verify trading hours
        self.assertEqual(
            self.instrument.trading_hours.to_dict(),
            new_instrument.trading_hours.to_dict()
        )
    
    def test_file_io(self):
        """Test saving and loading instruments to/from file."""
        # Create a temporary directory
        temp_dir = Path(tempfile.mkdtemp())
        try:
            # Save to file
            file_path = temp_dir / "test_instruments.json"
            InstrumentMetadata.save_to_file(
                {"BTCUSDT": self.instrument},
                file_path,
                format="json"
            )
            
            # Load from file
            instruments = InstrumentMetadata.load_from_file(file_path)
            
            # Verify the instrument was loaded correctly
            self.assertIn("BTCUSDT", instruments)
            loaded_instrument = instruments["BTCUSDT"]
            self.assertEqual(loaded_instrument.symbol, "BTCUSDT")
            self.assertEqual(loaded_instrument.asset_class, AssetClass.CRYPTO)
            self.assertEqual(loaded_instrument.instrument_type, InstrumentType.COIN)
            
        finally:
            # Clean up
            shutil.rmtree(temp_dir)


class TestInstrumentFactory(unittest.TestCase):
    """Test the InstrumentFactory class."""
    
    def test_create_forex_pair(self):
        """Test creating a forex pair."""
        eurusd = InstrumentFactory.create_forex_pair("EUR", "USD")
        
        self.assertEqual(eurusd.symbol, "EURUSD")
        self.assertEqual(eurusd.asset_class, AssetClass.FOREX)
        self.assertEqual(eurusd.instrument_type, InstrumentType.MAJOR)
        self.assertEqual(eurusd.base_currency, "EUR")
        self.assertEqual(eurusd.quote_currency, "USD")
        self.assertEqual(eurusd.leverage, 30.0)  # Default leverage for forex
        self.assertTrue(eurusd.has_tag("forex"))
        self.assertTrue(eurusd.has_tag("fx_major"))
    
    def test_create_crypto_pair(self):
        """Test creating a crypto pair."""
        btcusdt = InstrumentFactory.create_crypto_pair("BTC", "USDT")
        
        self.assertEqual(btcusdt.symbol, "BTCUSDT")
        self.assertEqual(btcusdt.asset_class, AssetClass.CRYPTO)
        self.assertEqual(btcusdt.instrument_type, InstrumentType.STABLECOIN)
        self.assertEqual(btcusdt.base_currency, "BTC")
        self.assertEqual(btcusdt.quote_currency, "USDT")
        self.assertEqual(btcusdt.leverage, 10.0)  # Default leverage for crypto
        self.assertTrue(btcusdt.has_tag("crypto"))
        self.assertTrue(btcusdt.has_tag("crypto_stablecoin"))
    
    def test_create_commodity(self):
        """Test creating a commodity instrument."""
        gold = InstrumentFactory.create_commodity(
            "XAUUSD",
            "Gold / US Dollar",
            InstrumentType.METAL
        )
        
        self.assertEqual(gold.symbol, "XAUUSD")
        self.assertEqual(gold.asset_class, AssetClass.COMMODITY)
        self.assertEqual(gold.instrument_type, InstrumentType.METAL)
        self.assertEqual(gold.leverage, 20.0)  # Default leverage for commodities
        self.assertTrue(gold.has_tag("commodity"))
        self.assertTrue(gold.has_tag("commodity_metal"))
    
    def test_create_stock(self):
        """Test creating a stock instrument."""
        aapl = InstrumentFactory.create_stock(
            "AAPL",
            "Apple Inc.",
            "NASDAQ"
        )
        
        self.assertEqual(aapl.symbol, "AAPL")
        self.assertEqual(aapl.asset_class, AssetClass.STOCK)
        self.assertEqual(aapl.instrument_type, InstrumentType.STOCK)
        self.assertEqual(aapl.exchange, "NASDAQ")
        self.assertEqual(aapl.leverage, 4.0)  # Default leverage for stocks
        self.assertTrue(aapl.has_tag("stock"))
        self.assertTrue(aapl.has_tag("equity"))


class TestInstrumentRegistry(unittest.TestCase):
    """Test the InstrumentRegistry class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.registry = InstrumentRegistry()
        
        # Add some test instruments
        self.eurusd = InstrumentFactory.create_forex_pair("EUR", "USD")
        self.btcusdt = InstrumentFactory.create_crypto_pair("BTC", "USDT")
        self.xauusd = InstrumentFactory.create_commodity(
            "XAUUSD", "Gold / US Dollar", InstrumentType.METAL
        )
        self.aapl = InstrumentFactory.create_stock("AAPL", "Apple Inc.", "NASDAQ")
        
        self.registry.add_instrument(self.eurusd)
        self.registry.add_instrument(self.btcusdt)
        self.registry.add_instrument(self.xauusd)
        self.registry.add_instrument(self.aapl)
    
    def test_add_and_get_instrument(self):
        """Test adding and retrieving instruments."""
        # Test getting by symbol
        self.assertEqual(self.registry.get_instrument("EURUSD"), self.eurusd)
        self.assertEqual(self.registry.get_instrument("BTCUSDT"), self.btcusdt)
        
        # Test case insensitivity
        self.assertEqual(self.registry.get_instrument("eurusd"), self.eurusd)
        
        # Test non-existent instrument
        with self.assertRaises(KeyError):
            self.registry.get_instrument("NONEXISTENT")
    
    def test_get_instruments_by_asset_class(self):
        """Test getting instruments by asset class."""
        # Get all forex instruments
        forex_instruments = self.registry.get_instruments_by_asset_class(AssetClass.FOREX)
        self.assertEqual(len(forex_instruments), 1)
        self.assertIn(self.eurusd, forex_instruments)
        
        # Get all crypto instruments
        crypto_instruments = self.registry.get_instruments_by_asset_class(AssetClass.CRYPTO)
        self.assertEqual(len(crypto_instruments), 1)
        self.assertIn(self.btcusdt, crypto_instruments)
    
    def test_get_instruments_by_tag(self):
        """Test getting instruments by tag."""
        # Get all forex instruments using tag
        forex_instruments = self.registry.get_instruments_by_tag("forex")
        self.assertEqual(len(forex_instruments), 1)
        self.assertIn(self.eurusd, forex_instruments)
        
        # Get all crypto instruments using tag
        crypto_instruments = self.registry.get_instruments_by_tag("crypto")
        self.assertEqual(len(crypto_instruments), 1)
        self.assertIn(self.btcusdt, crypto_instruments)
    
    def test_search_instruments(self):
        """Test searching for instruments."""
        # Search by symbol
        results = self.registry.search_instruments(symbol="EUR")
        self.assertEqual(len(results), 1)
        self.assertIn(self.eurusd, results)
        
        # Search by asset class
        results = self.registry.search_instruments(asset_class=AssetClass.CRYPTO)
        self.assertEqual(len(results), 1)
        self.assertIn(self.btcusdt, results)
        
        # Search by tag
        results = self.registry.search_instruments(tags=["forex"])
        self.assertEqual(len(results), 1)
        self.assertIn(self.eurusd, results)
        
        # Search by multiple criteria
        results = self.registry.search_instruments(
            asset_class=AssetClass.STOCK,
            exchange="NASDAQ"
        )
        self.assertEqual(len(results), 1)
        self.assertIn(self.aapl, results)
    
    def test_remove_instrument(self):
        """Test removing an instrument."""
        # Remove an instrument
        self.assertTrue(self.registry.remove_instrument("EURUSD"))
        self.assertEqual(len(self.registry), 3)
        
        # Try to remove non-existent instrument
        self.assertFalse(self.registry.remove_instrument("NONEXISTENT"))
        self.assertEqual(len(self.registry), 3)
    
    def test_file_io(self):
        """Test saving and loading the registry to/from file."""
        # Create a temporary directory
        temp_dir = Path(tempfile.mkdtemp())
        try:
            # Save to file
            file_path = temp_dir / "test_registry.json"
            self.registry.save_to_file(file_path, format="json")
            
            # Create a new registry and load from file
            new_registry = InstrumentRegistry()
            count = new_registry.load_from_file(file_path)
            
            # Verify the number of loaded instruments
            self.assertEqual(count, 4)
            self.assertEqual(len(new_registry), 4)
            
            # Verify one of the instruments
            self.assertIn("EURUSD", new_registry)
            eurusd = new_registry["EURUSD"]
            self.assertEqual(eurusd.asset_class, AssetClass.FOREX)
            self.assertEqual(eurusd.instrument_type, InstrumentType.MAJOR)
            
        finally:
            # Clean up
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    unittest.main()
