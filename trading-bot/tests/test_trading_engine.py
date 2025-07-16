"""
Tests for the trading engine and its components.
"""
import unittest
import pandas as pd
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

from core.instruments import InstrumentMetadata, AssetClass, InstrumentType, TradingHours
from core.calendar import MarketCalendar, TradingSession, MarketSessionType
from core.market import TradingEngine, MarketDataHandler, ExecutionHandler, Portfolio, RiskManager, RiskParameters


class TestTradingEngine(unittest.TestCase):
    """Test the TradingEngine class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock market calendar
        self.calendar = MagicMock(spec=MarketCalendar)
        self.calendar.is_session_open.return_value = True
        self.calendar.next_trading_day.return_value = datetime.now(timezone.utc).date() + timedelta(days=1)
        
        # Create a test instrument
        self.instrument = InstrumentMetadata(
            symbol="TEST",
            name="Test Instrument",
            asset_class=AssetClass.FOREX,
            instrument_type=InstrumentType.SPOT,
            base_currency="USD",
            quote_currency="JPY",
            trading_hours=TradingHours.for_asset_class("forex")
        )
        
        # Initialize the trading engine with mock components
        self.engine = TradingEngine(calendar=self.calendar)
        self.engine.data_handler = MagicMock()
        self.engine.execution_handler = MagicMock()
        self.engine.portfolio = MagicMock()
        self.engine.risk_manager = MagicMock()
    
    def test_add_instrument(self):
        """Test adding an instrument to the trading engine."""
        self.engine.add_instrument(self.instrument)
        self.assertIn(self.instrument.symbol, self.engine.instruments)
        self.assertEqual(self.engine.instruments[self.instrument.symbol], self.instrument)
    
    def test_remove_instrument(self):
        """Test removing an instrument from the trading engine."""
        self.engine.add_instrument(self.instrument)
        self.engine.remove_instrument(self.instrument.symbol)
        self.assertNotIn(self.instrument.symbol, self.engine.instruments)
    
    def test_is_market_open(self):
        """Test checking if market is open for an instrument."""
        self.engine.add_instrument(self.instrument)
        
        # Test when market is open
        self.calendar.is_session_open.return_value = True
        self.assertTrue(self.engine.is_market_open(self.instrument.symbol))
        
        # Test when market is closed
        self.calendar.is_session_open.return_value = False
        self.assertFalse(self.engine.is_market_open(self.instrument.symbol))
    
    def test_submit_order_market_closed(self):
        """Test order submission when market is closed."""
        self.engine.add_instrument(self.instrument)
        self.calendar.is_session_open.return_value = False
        
        order = {
            'symbol': self.instrument.symbol,
            'order_type': 'market',
            'side': 'buy',
            'quantity': 100,
            'price': 100.0,
            'account_id': 'test_account',
            'order_id': 'test_order_1'
        }
        
        result = self.engine.submit_order(order)
        self.assertFalse(result)
        self.engine.execution_handler.execute_order.assert_not_called()


class TestMarketDataHandler(unittest.TestCase):
    """Test the MarketDataHandler class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.instrument = InstrumentMetadata(
            symbol="TEST",
            name="Test Instrument",
            asset_class=AssetClass.FOREX
        )
        self.handler = MarketDataHandler([self.instrument])
    
    def test_subscribe(self):
        """Test subscribing to market data."""
        self.assertTrue(self.handler.subscribe("TEST", "1m"))
        self.assertIn("1m", self.handler.bars["TEST"])
    
    def test_on_tick(self):
        """Test processing a new tick."""
        tick = {
            'bid': 100.0,
            'ask': 100.1,
            'last': 100.05,
            'volume': 1000
        }
        
        self.handler.on_tick("TEST", tick)
        self.assertEqual(self.handler.latest_ticks["TEST"]['bid'], 100.0)
    
    def test_on_bar(self):
        """Test processing a new bar."""
        # First subscribe to the timeframe
        self.handler.subscribe("TEST", "1m")
        
        bar = {
            'open': 100.0,
            'high': 100.5,
            'low': 99.5,
            'close': 100.2,
            'volume': 5000
        }
        
        self.handler.on_bar("TEST", "1m", bar)
        self.assertFalse(self.handler.bars["TEST"]["1m"].empty())


class TestPortfolio(unittest.TestCase):
    """Test the Portfolio class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.portfolio = Portfolio(initial_cash=100000.0)
    
    def test_initialization(self):
        """Test portfolio initialization."""
        self.assertEqual(self.portfolio.cash, 100000.0)
        self.assertEqual(self.portfolio.initial_cash, 100000.0)
        self.assertEqual(len(self.portfolio.positions), 0)
    
    def test_update_position_long(self):
        """Test updating a long position."""
        # Open a long position
        pnl, new_qty = self.portfolio.update_position(
            symbol="TEST",
            quantity=100,
            price=50.0
        )
        
        self.assertEqual(pnl, 0.0)  # No PnL on opening
        self.assertEqual(new_qty, 100)
        self.assertEqual(self.portfolio.cash, 100000.0 - (100 * 50.0))
        
        # Close the position
        pnl, new_qty = self.portfolio.update_position(
            symbol="TEST",
            quantity=-100,
            price=60.0
        )
        
        self.assertAlmostEqual(pnl, 1000.0)  # 100 * (60 - 50)
        self.assertEqual(new_qty, 0)
        self.assertAlmostEqual(self.portfolio.cash, 100000.0 + 1000.0)
    
    def test_update_position_short(self):
        """Test updating a short position."""
        # Open a short position
        pnl, new_qty = self.portfolio.update_position(
            symbol="TEST",
            quantity=-100,
            price=50.0
        )
        
        self.assertEqual(pnl, 0.0)  # No PnL on opening
        self.assertEqual(new_qty, -100)
        self.assertEqual(self.portfolio.cash, 100000.0 + (100 * 50.0))
        
        # Close the position
        pnl, new_qty = self.portfolio.update_position(
            symbol="TEST",
            quantity=100,
            price=40.0
        )
        
        self.assertAlmostEqual(pnl, 1000.0)  # 100 * (50 - 40)
        self.assertEqual(new_qty, 0)
        self.assertAlmostEqual(self.portfolio.cash, 100000.0 + 1000.0)


class TestRiskManager(unittest.TestCase):
    """Test the RiskManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.params = RiskParameters(
            max_position_size_pct=10.0,
            max_portfolio_risk_pct=20.0,
            max_leverage=3.0,
            max_daily_drawdown_pct=5.0,
            max_drawdown_pct=15.0,
            max_positions=5,
            max_position_concentration_pct=30.0,
            max_volatility_pct=5.0,
            risk_per_trade_pct=1.0,
            max_trades_per_day=10
        )
        
        self.risk_manager = RiskManager(parameters=self.params)
    
    def test_check_order_risk(self):
        """Test order risk validation."""
        # Test a valid order
        order = {
            'symbol': 'TEST',
            'quantity': 100,
            'price': 50.0,
            'side': 'buy',
            'order_type': 'market',
            'account_id': 'test',
            'order_id': '1'
        }
        
        portfolio_value = 100000.0
        is_allowed, reason = self.risk_manager.check_order_risk(order, portfolio_value)
        self.assertTrue(is_allowed, reason)
        
        # Test exceeding max position size
        order['quantity'] = 10000  # 10000 * 50 = 500,000 > 10% of 1,000,000
        is_allowed, reason = self.risk_manager.check_order_risk(order, portfolio_value)
        self.assertFalse(is_allowed)
        self.assertIn("exceeds max", reason)
    
    def test_update_drawdown(self):
        """Test drawdown tracking and circuit breakers."""
        # Initial equity high water mark
        is_ok, msg = self.risk_manager.update_drawdown(100000.0)
        self.assertTrue(is_ok)
        self.assertEqual(self.risk_manager.equity_high_water_mark, 100000.0)
        
        # Small drawdown within limits
        is_ok, msg = self.risk_manager.update_drawdown(98000.0)  # 2% drawdown
        self.assertTrue(is_ok)
        
        # Trigger daily drawdown limit (5%)
        is_ok, msg = self.risk_manager.update_drawdown(94000.0)  # 6% drawdown
        self.assertFalse(is_ok)
        self.assertIn("Daily drawdown", msg)
        
        # Reset and test circuit breaker (10%)
        self.risk_manager = RiskManager(parameters=self.params)
        is_ok, msg = self.risk_manager.update_drawdown(100000.0)
        is_ok, msg = self.risk_manager.update_drawdown(89000.0)  # 11% drawdown
        self.assertFalse(is_ok)
        self.assertIn("Circuit breaker", msg)
        self.assertTrue(self.risk_manager.circuit_breaker_triggered)


if __name__ == "__main__":
    unittest.main()
