"""
Additional tests for market components and their integration.
"""
import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch, call
import pandas as pd

from core.instruments import InstrumentMetadata, AssetClass, InstrumentType
from core.calendar import MarketCalendar, TradingSession, MarketSessionType
from core.market import (
    TradingEngine, MarketDataHandler, ExecutionHandler, 
    Portfolio, RiskManager, RiskParameters, Position, Trade
)

class TestMarketIntegration(unittest.TestCase):
    """Test integration between market components."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create test instruments
        self.forex_pair = InstrumentMetadata(
            symbol="EURUSD",
            name="Euro/US Dollar",
            asset_class=AssetClass.FOREX,
            instrument_type=InstrumentType.SPOT,
            base_currency="EUR",
            quote_currency="USD"
        )
        
        self.equity = InstrumentMetadata(
            symbol="AAPL",
            name="Apple Inc.",
            asset_class=AssetClass.EQUITY,
            instrument_type=InstrumentType.STOCK,
            exchange="NASDAQ"
        )
        
        # Create a test market calendar
        self.calendar = MarketCalendar("America/New_York")
        
        # Initialize components
        self.data_handler = MarketDataHandler(
            [self.forex_pair, self.equity],
            calendar=self.calendar
        )
        
        self.execution_handler = ExecutionHandler(
            calendar=self.calendar
        )
        
        self.portfolio = Portfolio(
            initial_cash=100000.0,
            currency="USD",
            calendar=self.calendar
        )
        
        self.risk_manager = RiskManager(
            parameters=RiskParameters(),
            calendar=self.calendar
        )
        
        # Initialize trading engine
        self.engine = TradingEngine(
            data_handler=self.data_handler,
            execution_handler=self.execution_handler,
            portfolio=self.portfolio,
            risk_manager=self.risk_manager,
            calendar=self.calendar
        )
        
        # Add instruments to the engine
        self.engine.add_instrument(self.forex_pair)
        self.engine.add_instrument(self.equity)
    
    def test_market_data_flow(self):
        """Test the flow of market data through the system."""
        # Subscribe to market data
        self.data_handler.subscribe("EURUSD", "1m")
        self.data_handler.subscribe("AAPL", "1m")
        
        # Process ticks
        eur_usd_tick = {
            'bid': 1.1000,
            'ask': 1.1001,
            'last': 1.10005,
            'volume': 1000,
            'timestamp': datetime.now(timezone.utc)
        }
        
        aapl_tick = {
            'bid': 150.0,
            'ask': 150.1,
            'last': 150.05,
            'volume': 100,
            'timestamp': datetime.now(timezone.utc)
        }
        
        self.data_handler.on_tick("EURUSD", eur_usd_tick)
        self.data_handler.on_tick("AAPL", aapl_tick)
        
        # Verify ticks were processed
        self.assertEqual(self.data_handler.latest_ticks["EURUSD"]['bid'], 1.1000)
        self.assertEqual(self.data_handler.latest_ticks["AAPL"]['bid'], 150.0)
    
    def test_order_execution_flow(self):
        """Test the order execution flow."""
        # Mock the execution handler's execute_order method
        self.execution_handler.execute_order = MagicMock(return_value=True)
        
        # Submit a market order
        order = {
            'symbol': 'EURUSD',
            'order_type': 'market',
            'side': 'buy',
            'quantity': 100000,  # 1 standard lot
            'price': 1.1000,
            'account_id': 'test_account',
            'order_id': 'test_order_1'
        }
        
        # Process the order
        result = self.engine.submit_order(order)
        
        # Verify the order was processed
        self.assertTrue(result)
        self.execution_handler.execute_order.assert_called_once()
        
        # Verify the order details
        executed_order = self.execution_handler.execute_order.call_args[0][0]
        self.assertEqual(executed_order['symbol'], 'EURUSD')
        self.assertEqual(executed_order['side'], 'buy')
        self.assertEqual(executed_order['quantity'], 100000)
    
    def test_portfolio_tracking(self):
        """Test portfolio position tracking."""
        # Execute a trade
        self.portfolio.update_position(
            symbol='AAPL',
            quantity=100,
            price=150.0,
            commission=10.0
        )
        
        # Verify position was updated
        position = self.portfolio.get_position('AAPL')
        self.assertIsNotNone(position)
        self.assertEqual(position.quantity, 100)
        self.assertEqual(position.avg_price, 150.0)
        
        # Update market price
        self.portfolio.update_market_value('AAPL', 155.0)
        position = self.portfolio.get_position('AAPL')
        self.assertEqual(position.unrealized_pnl, 500.0)  # 100 * (155 - 150)
        
        # Close position
        self.portfolio.update_position(
            symbol='AAPL',
            quantity=-100,
            price=160.0,
            commission=10.0
        )
        
        # Verify position is closed and PnL is recorded
        position = self.portfolio.get_position('AAPL')
        self.assertIsNone(position)
        self.assertEqual(len(self.portfolio.trades), 1)
        self.assertEqual(self.portfolio.trades[0].pnl, 1000.0)  # 100 * (160 - 150)
    
    def test_risk_management(self):
        """Test risk management controls."""
        # Configure risk parameters
        self.risk_manager.parameters.max_position_size_pct = 10.0  # 10% of portfolio
        self.risk_manager.parameters.max_leverage = 2.0
        
        # Test a valid order (within limits)
        valid_order = {
            'symbol': 'AAPL',
            'order_type': 'market',
            'side': 'buy',
            'quantity': 100,
            'price': 100.0,
            'account_id': 'test',
            'order_id': '1'
        }
        
        is_allowed, reason = self.risk_manager.check_order_risk(valid_order, 100000.0)
        self.assertTrue(is_allowed, reason)
        
        # Test an order that exceeds position size limit
        large_order = valid_order.copy()
        large_order['quantity'] = 10000  # Would be $1M position (1000% of portfolio)
        large_order['order_id'] = '2'
        
        is_allowed, reason = self.risk_manager.check_order_risk(large_order, 100000.0)
        self.assertFalse(is_allowed)
        self.assertIn("exceeds max position size", reason.lower())
    
    def test_end_to_end_trade(self):
        """Test a complete trade from signal to execution."""
        # Mock the execution handler
        self.execution_handler.execute_order = MagicMock(return_value=True)
        
        # Generate a market data event
        tick = {
            'bid': 1.1000,
            'ask': 1.1001,
            'last': 1.10005,
            'volume': 1000,
            'timestamp': datetime.now(timezone.utc)
        }
        
        # Process the tick
        self.data_handler.on_tick("EURUSD", tick)
        
        # Create a signal (in a real system, this would come from a strategy)
        signal = {
            'symbol': 'EURUSD',
            'signal': 'buy',
            'strength': 0.8,
            'timestamp': datetime.now(timezone.utc)
        }
        
        # Generate an order from the signal
        order = {
            'symbol': signal['symbol'],
            'order_type': 'market',
            'side': signal['signal'],
            'quantity': 100000,  # 1 standard lot
            'price': tick['ask'],  # Buy at ask
            'account_id': 'test_account',
            'order_id': 'test_order_2',
            'signal': signal
        }
        
        # Submit the order
        result = self.engine.submit_order(order)
        self.assertTrue(result)
        
        # Verify the order was executed
        self.execution_handler.execute_order.assert_called_once()
        
        # Verify the position was updated in the portfolio
        position = self.portfolio.get_position('EURUSD')
        self.assertIsNotNone(position)
        self.assertEqual(position.quantity, 100000)
        
        # Update market price and check PnL
        new_tick = tick.copy()
        new_tick['bid'] = 1.1050
        new_tick['ask'] = 1.1051
        new_tick['last'] = 1.10505
        new_tick['timestamp'] = datetime.now(timezone.utc)
        
        self.data_handler.on_tick("EURUSD", new_tick)
        self.portfolio.update_market_value("EURUSD", new_tick['bid'])
        
        position = self.portfolio.get_position('EURUSD')
        self.assertAlmostEqual(position.unrealized_pnl, 500.0)  # 100000 * (1.1050 - 1.1000)


class TestPositionAndTrade(unittest.TestCase):
    """Test the Position and Trade classes."""
    
    def test_position_calculations(self):
        """Test position calculations."""
        # Create a position
        position = Position(symbol="AAPL")
        
        # Update with initial purchase
        position.quantity = 100
        position.avg_price = 150.0
        position.current_price = 150.0
        
        self.assertEqual(position.market_value, 15000.0)  # 100 * 150.0
        self.assertEqual(position.cost_basis, 15000.0)    # 100 * 150.0
        self.assertEqual(position.unrealized_pnl, 0.0)    # No PnL yet
        
        # Update price
        position.update_price(155.0)
        self.assertEqual(position.current_price, 155.0)
        self.assertEqual(position.market_value, 15500.0)  # 100 * 155.0
        self.assertEqual(position.unrealized_pnl, 500.0)  # 100 * (155 - 150)
        self.assertAlmostEqual(position.unrealized_pnl_pct, (5.0 / 150.0) * 100)
    
    def test_trade_calculations(self):
        """Test trade calculations."""
        # Create a trade
        entry_time = datetime(2023, 1, 1, 9, 30, tzinfo=timezone.utc)
        exit_time = datetime(2023, 1, 1, 15, 30, tzinfo=timezone.utc)
        
        trade = Trade(
            symbol="AAPL",
            quantity=100,
            entry_price=150.0,
            exit_price=160.0,
            entry_time=entry_time,
            exit_time=exit_time,
            side="long",
            pnl=1000.0,  # 100 * (160 - 150)
            pnl_pct=(10.0 / 150.0) * 100,
            commission=10.0
        )
        
        self.assertEqual(trade.duration, 6 * 3600)  # 6 hours in seconds
        self.assertEqual(trade.pnl, 1000.0)
        self.assertAlmostEqual(trade.pnl_pct, (10.0 / 150.0) * 100)


if __name__ == "__main__":
    unittest.main()
