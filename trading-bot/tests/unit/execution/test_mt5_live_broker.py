"""
Unit tests for the MT5 Live Broker implementation.

These tests verify the core functionality of the MT5 Live Broker,
including connection management, order execution, and position tracking.
"""

import asyncio
import os
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime

import MetaTrader5 as mt5
import pytest
from freezegun import freeze_time

from core.execution.live.mt5_live_broker import MT5LiveBroker
from core.models import Order, OrderType, OrderSide, OrderStatus, TimeInForce
from core.exceptions import ConnectionError, OrderError, MarketDataError

# Test configuration
TEST_SYMBOL = "EURUSD"
TEST_VOLUME = 0.1
TEST_PRICE = 1.1000
TEST_STOP_LOSS = 1.0950
TEST_TAKE_PROFIT = 1.1050
TEST_ORDER_ID = "12345"
TEST_POSITION_ID = "54321"


class TestMT5LiveBroker(unittest.IsolatedAsyncioTestCase):
    """Test suite for MT5LiveBroker."""

    def setUp(self):
        """Set up test fixtures."""
        self.broker = MT5LiveBroker(
            server="TestServer",
            login=12345,
            password="testpass",
            timeout=1000,
            polling_interval=0.1,
            max_reconnect_attempts=3
        )
        
        # Mock MT5 module
        self.mt5_patcher = patch('core.execution.live.mt5_live_broker.mt5')
        self.mock_mt5 = self.mt5_patcher.start()
        
        # Set up mock return values
        self.mock_mt5.initialize.return_value = True
        self.mock_mt5.login.return_value = True
        self.mock_mt5.terminal_info.return_value = MagicMock(trade_allowed=True)
        self.mock_mt5.last_error.return_value = (0, "No error")
        
        # Mock account info
        self.mock_account = MagicMock()
        self.mock_account.balance = 10000.0
        self.mock_account.equity = 10050.0
        self.mock_account.margin = 1000.0
        self.mock_account.margin_free = 9000.0
        self.mock_account.margin_level = 1000.0
        self.mock_account.leverage = 100
        self.mock_account.currency = "USD"
        self.mock_account.name = "Test Account"
        self.mock_account.server = "TestServer"
        self.mock_mt5.account_info.return_value = self.mock_account
        
        # Mock symbol info
        self.mock_symbol = MagicMock()
        self.mock_symbol.name = TEST_SYMBOL
        self.mock_symbol.digits = 5
        self.mock_symbol.spread = 10
        self.mock_symbol.volume_min = 0.01
        self.mock_symbol.volume_max = 100.0
        self.mock_symbol.volume_step = 0.01
        self.mock_symbol.trade_tick_size = 0.00001
        self.mock_symbol.trade_tick_value = 1.0
        self.mock_symbol.currency_base = "EUR"
        self.mock_symbol.currency_profit = "USD"
        self.mock_symbol.currency_margin = "USD"
        self.mock_mt5.symbol_info.return_value = self.mock_symbol
        
        # Mock order result
        self.mock_order_result = MagicMock()
        self.mock_order_result.retcode = mt5.TRADE_RETCODE_DONE
        self.mock_order_result.order = int(TEST_ORDER_ID)
        self.mock_order_result.comment = "Test order"
        
        # Mock position
        self.mock_position = MagicMock()
        self.mock_position.ticket = int(TEST_POSITION_ID)
        self.mock_position.symbol = TEST_SYMBOL
        self.mock_position.type = mt5.POSITION_TYPE_BUY
        self.mock_position.volume = TEST_VOLUME
        self.mock_position.price_open = TEST_PRICE
        self.mock_position.price_current = TEST_PRICE + 0.0010
        self.mock_position.sl = TEST_STOP_LOSS
        self.mock_position.tp = TEST_TAKE_PROFIT
        self.mock_position.swap = 0.0
        self.mock_position.profit = 1.0
        self.mock_position.comment = "Test position"
        self.mock_position.magic = 123
        self.mock_position.time_msc = int(datetime.now().timestamp() * 1000)
        
        # Mock order
        self.mock_order = MagicMock()
        self.mock_order.ticket = int(TEST_ORDER_ID)
        self.mock_order.symbol = TEST_SYMBOL
        self.mock_order.type = mt5.ORDER_TYPE_BUY
        self.mock_order.volume_current = TEST_VOLUME
        self.mock_order.price_open = TEST_PRICE
        self.mock_order.sl = TEST_STOP_LOSS
        self.mock_order.tp = TEST_TAKE_PROFIT
        self.mock_order.comment = "Test order"
        self.mock_order.magic = 123
        self.mock_order.time_setup = int(datetime.now().timestamp())
        self.mock_order.state = mt5.ORDER_STATE_PLACED

    def tearDown(self):
        """Clean up after each test."""
        self.mt5_patcher.stop()
        
    async def test_connect_success(self):
        """Test successful connection to MT5."""
        # Test
        result = await self.broker.connect()
        
        # Assert
        self.assertTrue(result)
        self.assertTrue(self.broker.connected)
        self.mock_mt5.initialize.assert_called_once()
        self.mock_mt5.login.assert_called_once_with(
            login=12345,
            password="testpass",
            server="TestServer",
            timeout=1000
        )
        
    async def test_connect_failure(self):
        """Test connection failure to MT5."""
        # Setup
        self.mock_mt5.initialize.return_value = False
        self.mock_mt5.last_error.return_value = (1, "Connection failed")
        
        # Test & Assert
        with self.assertRaises(ConnectionError):
            await self.broker.connect()
            
    async def test_place_market_order_success(self):
        """Test placing a market order successfully."""
        # Setup
        await self.broker.connect()
        self.mock_mt5.symbol_info.return_value.ask = TEST_PRICE + 0.0005
        self.mock_mt5.symbol_info.return_value.bid = TEST_PRICE - 0.0005
        self.mock_mt5.order_send.return_value = self.mock_order_result
        
        # Create test order
        order = Order(
            symbol=TEST_SYMBOL,
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=TEST_VOLUME,
            stop_loss=TEST_STOP_LOSS,
            take_profit=TEST_TAKE_PROFIT,
            comment="Test order"
        )
        
        # Test
        order_id = await self.broker.place_order(order)
        
        # Assert
        self.assertEqual(order_id, TEST_ORDER_ID)
        self.mock_mt5.order_send.assert_called_once()
        
    async def test_place_limit_order_success(self):
        """Test placing a limit order successfully."""
        # Setup
        await self.broker.connect()
        self.mock_mt5.order_send.return_value = self.mock_order_result
        
        # Create test order
        order = Order(
            symbol=TEST_SYMBOL,
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=TEST_VOLUME,
            price=TEST_PRICE - 0.0050,  # Below market
            stop_loss=TEST_STOP_LOSS,
            take_profit=TEST_TAKE_PROFIT,
            time_in_force=TimeInForce.GTC,
            comment="Test limit order"
        )
        
        # Test
        order_id = await self.broker.place_order(order)
        
        # Assert
        self.assertEqual(order_id, TEST_ORDER_ID)
        self.mock_mt5.order_send.assert_called_once()
        
    async def test_cancel_order_success(self):
        """Test canceling an order successfully."""
        # Setup
        await self.broker.connect()
        self.mock_mt5.order_send.return_value = self.mock_order_result
        
        # Test
        result = await self.broker.cancel_order(TEST_ORDER_ID)
        
        # Assert
        self.assertTrue(result)
        self.mock_mt5.order_send.assert_called_once()
        
    async def test_modify_order_success(self):
        """Test modifying an order successfully."""
        # Setup
        await self.broker.connect()
        self.mock_mt5.orders_get.return_value = [self.mock_order]
        self.mock_mt5.order_send.return_value = self.mock_order_result
        
        # Test
        result = await self.broker.modify_order(
            order_id=TEST_ORDER_ID,
            price=TEST_PRICE + 0.0010,
            stop_loss=TEST_STOP_LOSS + 0.0010,
            take_profit=TEST_TAKE_PROFIT + 0.0010
        )
        
        # Assert
        self.assertTrue(result)
        self.mock_mt5.order_send.assert_called_once()
        
    async def test_close_position_success(self):
        """Test closing a position successfully."""
        # Setup
        await self.broker.connect()
        self.mock_mt5.positions_get.return_value = [self.mock_position]
        self.mock_mt5.symbol_info.return_value.ask = TEST_PRICE + 0.0005
        self.mock_mt5.symbol_info.return_value.bid = TEST_PRICE - 0.0005
        self.mock_mt5.order_send.return_value = self.mock_order_result
        
        # Test
        result = await self.broker.close_position(TEST_POSITION_ID)
        
        # Assert
        self.assertTrue(result)
        self.mock_mt5.order_send.assert_called_once()
        
    async def test_get_positions(self):
        """Test getting open positions."""
        # Setup
        await self.broker.connect()
        self.mock_mt5.positions_get.return_value = [self.mock_position]
        
        # Test
        positions = await self.broker.get_positions()
        
        # Assert
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0]['position_id'], TEST_POSITION_ID)
        self.assertEqual(positions[0]['symbol'], TEST_SYMBOL)
        self.assertEqual(positions[0]['volume'], TEST_VOLUME)
        
    async def test_get_orders(self):
        """Test getting open orders."""
        # Setup
        await self.broker.connect()
        self.mock_mt5.orders_get.return_value = [self.mock_order]
        
        # Test
        orders = await self.broker.get_orders()
        
        # Assert
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0]['order_id'], TEST_ORDER_ID)
        self.assertEqual(orders[0]['symbol'], TEST_SYMBOL)
        self.assertEqual(orders[0]['volume'], TEST_VOLUME)
        
    async def test_get_account_info(self):
        """Test getting account information."""
        # Setup
        await self.broker.connect()
        
        # Test
        account_info = await self.broker.get_account_info()
        
        # Assert
        self.assertEqual(account_info['balance'], 10000.0)
        self.assertEqual(account_info['equity'], 10050.0)
        self.assertEqual(account_info['margin'], 1000.0)
        self.assertEqual(account_info['margin_free'], 9000.0)
        self.assertEqual(account_info['leverage'], 100)
        
    async def test_get_symbol_info(self):
        """Test getting symbol information."""
        # Setup
        await self.broker.connect()
        
        # Test
        symbol_info = await self.broker.get_symbol_info(TEST_SYMBOL)
        
        # Assert
        self.assertEqual(symbol_info['symbol'], TEST_SYMBOL)
        self.assertEqual(symbol_info['digits'], 5)
        self.assertEqual(symbol_info['volume_min'], 0.01)
        self.assertEqual(symbol_info['volume_max'], 100.0)
        
    async def test_connection_monitoring(self):
        """Test the connection monitoring background task."""
        # Setup
        await self.broker.connect()
        
        # Simulate connection loss
        self.mock_mt5.terminal_info.return_value = None
        
        # Wait for monitor to detect the issue
        await asyncio.sleep(0.2)
        
        # Verify reconnection was attempted
        self.assertGreaterEqual(self.mock_mt5.initialize.call_count, 1)
        
    @freeze_time("2023-01-01 12:00:00")
    async def test_parse_position(self):
        """Test parsing an MT5 position to our internal format."""
        # Setup
        position = self.mock_position
        
        # Test
        parsed = self.broker._parse_position(position)
        
        # Assert
        self.assertEqual(parsed['position_id'], TEST_POSITION_ID)
        self.assertEqual(parsed['symbol'], TEST_SYMBOL)
        self.assertEqual(parsed['type'], 'long')
        self.assertEqual(parsed['volume'], TEST_VOLUME)
        self.assertEqual(parsed['entry_price'], TEST_PRICE)
        self.assertEqual(parsed['current_price'], TEST_PRICE + 0.0010)
        self.assertEqual(parsed['sl'], TEST_STOP_LOSS)
        self.assertEqual(parsed['tp'], TEST_TAKE_PROFIT)
        self.assertEqual(parsed['profit'], 1.0)
        self.assertEqual(parsed['comment'], "Test position")
        self.assertEqual(parsed['magic'], 123)
        
    @freeze_time("2023-01-01 12:00:00")
    async def test_parse_order(self):
        """Test parsing an MT5 order to our internal format."""
        # Setup
        order = self.mock_order
        
        # Test
        parsed = self.broker._parse_order(order)
        
        # Assert
        self.assertEqual(parsed['order_id'], TEST_ORDER_ID)
        self.assertEqual(parsed['symbol'], TEST_SYMBOL)
        self.assertEqual(parsed['type'], 'market')
        self.assertEqual(parsed['side'], 'buy')
        self.assertEqual(parsed['volume'], TEST_VOLUME)
        self.assertEqual(parsed['price'], TEST_PRICE)
        self.assertEqual(parsed['stop_loss'], TEST_STOP_LOSS)
        self.assertEqual(parsed['take_profit'], TEST_TAKE_PROFIT)
        self.assertEqual(parsed['status'], 'pending')
        self.assertEqual(parsed['comment'], "Test order")
        self.assertEqual(parsed['magic'], 123)


if __name__ == '__main__':
    unittest.main()
