"""
Integration tests for the MT5 Live Broker with a demo account.

These tests verify the broker's functionality against a live MT5 demo account.
WARNING: These tests will place real orders on the demo account.
"""

import asyncio
import os
import time
import unittest
from datetime import datetime
from decimal import Decimal

import MetaTrader5 as mt5
import pytest
from dotenv import load_dotenv

from core.execution.live.mt5_live_broker import MT5LiveBroker
from core.models import Order, OrderType, OrderSide, TimeInForce

# Load environment variables
load_dotenv()

# Test configuration
TEST_SYMBOL = os.getenv('MT5_TEST_SYMBOL', 'EURUSD')
TEST_VOLUME = float(os.getenv('MT5_TEST_VOLUME', '0.01'))  # Minimum volume

# Skip tests if we don't have MT5 credentials
pytestmark = pytest.mark.skipif(
    not all([
        os.getenv('MT5_SERVER'),
        os.getenv('MT5_LOGIN'),
        os.getenv('MT5_PASSWORD')
    ]),
    reason='MT5 credentials not provided in environment variables'
)


class TestMT5LiveBrokerIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for MT5LiveBroker with a demo account."""

    @classmethod
    def setUpClass(cls):
        """Set up test class."""
        cls.broker = MT5LiveBroker(
            server=os.getenv('MT5_SERVER'),
            login=int(os.getenv('MT5_LOGIN')),
            password=os.getenv('MT5_PASSWORD'),
            timeout=10000,  # 10 seconds
            polling_interval=0.1,
            max_reconnect_attempts=3
        )

    async def asyncSetUp(self):
        """Set up before each test."""
        # Connect to MT5
        connected = await self.broker.connect()
        if not connected:
            self.fail("Failed to connect to MT5")
        
        # Cancel all open orders
        orders = await self.broker.get_orders()
        for order in orders:
            await self.broker.cancel_order(order['order_id'])
        
        # Close all open positions
        positions = await self.broker.get_positions()
        for position in positions:
            await self.broker.close_position(position['position_id'])
        
        # Small delay to ensure all operations complete
        await asyncio.sleep(1)

    async def test_connection(self):
        """Test connection to MT5."""
        self.assertTrue(await self.broker.connect())
        
        # Verify we can get account info
        account_info = await self.broker.get_account_info()
        self.assertIsNotNone(account_info)
        self.assertIn('balance', account_info)
        self.assertIn('equity', account_info)
        self.assertIn('margin', account_info)
        
        print(f"\nConnected to account: {account_info['name']}")
        print(f"Balance: {account_info['balance']} {account_info['currency']}")
        print(f"Equity: {account_info['equity']} {account_info['currency']}")
        print(f"Margin: {account_info['margin']} {account_info['currency']}")

    async def test_market_order_round_trip(self):
        """Test placing and closing a market order."""
        # Get current price
        symbol_info = await self.broker.get_symbol_info(TEST_SYMBOL)
        self.assertIsNotNone(symbol_info)
        
        # Place a market buy order
        order = Order(
            symbol=TEST_SYMBOL,
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=TEST_VOLUME,
            comment="Test market order"
        )
        
        order_id = await self.broker.place_order(order)
        self.assertIsNotNone(order_id)
        print(f"\nPlaced market order: {order_id}")
        
        # Verify the order was executed
        await asyncio.sleep(2)  # Give it time to fill
        
        # Check open positions
        positions = await self.broker.get_positions(symbol=TEST_SYMBOL)
        self.assertEqual(len(positions), 1)
        position = positions[0]
        
        print(f"Position opened: {position}")
        
        # Close the position
        result = await self.broker.close_position(position['position_id'])
        self.assertTrue(result)
        
        # Verify position is closed
        await asyncio.sleep(1)  # Give it time to close
        positions = await self.broker.get_positions(symbol=TEST_SYMBOL)
        self.assertEqual(len(positions), 0)
        
        print("Position closed successfully")

    async def test_limit_order(self):
        """Test placing a limit order."""
        # Get current price
        symbol_info = await self.broker.get_symbol_info(TEST_SYMBOL)
        self.assertIsNotNone(symbol_info)
        
        # Calculate a limit price far from current price to avoid execution
        current_price = (symbol_info['ask'] + symbol_info['bid']) / 2
        limit_price = round(current_price * 0.95, 5)  # 5% below current price
        
        # Place a limit buy order
        order = Order(
            symbol=TEST_SYMBOL,
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=TEST_VOLUME,
            price=limit_price,
            time_in_force=TimeInForce.GTC,
            comment="Test limit order"
        )
        
        order_id = await self.broker.place_order(order)
        self.assertIsNotNone(order_id)
        print(f"\nPlaced limit order: {order_id} at {limit_price}")
        
        # Verify the order is in the order book
        await asyncio.sleep(1)
        orders = await self.broker.get_orders(symbol=TEST_SYMBOL)
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0]['order_id'], order_id)
        
        # Cancel the order
        result = await self.broker.cancel_order(order_id)
        self.assertTrue(result)
        
        # Verify order is cancelled
        await asyncio.sleep(1)
        orders = await self.broker.get_orders(symbol=TEST_SYMBOL)
        self.assertEqual(len(orders), 0)
        
        print("Limit order cancelled successfully")

    async def test_stop_order(self):
        """Test placing a stop order."""
        # Get current price
        symbol_info = await self.broker.get_symbol_info(TEST_SYMBOL)
        self.assertIsNotNone(symbol_info)
        
        # Calculate a stop price far from current price to avoid execution
        current_price = (symbol_info['ask'] + symbol_info['bid']) / 2
        stop_price = round(current_price * 1.05, 5)  # 5% above current price
        
        # Place a stop buy order
        order = Order(
            symbol=TEST_SYMBOL,
            order_type=OrderType.STOP,
            side=OrderSide.BUY,
            quantity=TEST_VOLUME,
            price=stop_price,
            time_in_force=TimeInForce.GTC,
            comment="Test stop order"
        )
        
        order_id = await self.broker.place_order(order)
        self.assertIsNotNone(order_id)
        print(f"\nPlaced stop order: {order_id} at {stop_price}")
        
        # Verify the order is in the order book
        await asyncio.sleep(1)
        orders = await self.broker.get_orders(symbol=TEST_SYMBOL)
        self.assertEqual(len(orders), 1)
        self.assertEqual(orders[0]['order_id'], order_id)
        
        # Cancel the order
        result = await self.broker.cancel_order(order_id)
        self.assertTrue(result)
        
        # Verify order is cancelled
        await asyncio.sleep(1)
        orders = await self.broker.get_orders(symbol=TEST_SYMBOL)
        self.assertEqual(len(orders), 0)
        
        print("Stop order cancelled successfully")

    async def test_modify_position(self):
        """Test modifying a position's stop loss and take profit."""
        # Place a market buy order
        order = Order(
            symbol=TEST_SYMBOL,
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=TEST_VOLUME,
            comment="Test modify position"
        )
        
        order_id = await self.broker.place_order(order)
        self.assertIsNotNone(order_id)
        
        # Wait for the position to open
        await asyncio.sleep(2)
        
        # Get the position
        positions = await self.broker.get_positions(symbol=TEST_SYMBOL)
        self.assertEqual(len(positions), 1)
        position = positions[0]
        
        # Modify the position
        new_sl = position['entry_price'] * 0.99  # 1% stop loss
        new_tp = position['entry_price'] * 1.02  # 2% take profit
        
        # In MT5, we need to close and reopen the position to modify SL/TP
        # So we'll test the broker's modify_position method
        result = await self.broker.modify_order(
            order_id=position['position_id'],
            stop_loss=new_sl,
            take_profit=new_tp
        )
        
        self.assertTrue(result)
        print(f"\nModified position {position['position_id']} - SL: {new_sl}, TP: {new_tp}")
        
        # Verify the modification
        await asyncio.sleep(1)
        positions = await self.broker.get_positions(symbol=TEST_SYMBOL)
        updated_position = positions[0]
        
        self.assertAlmostEqual(float(updated_position['sl']), new_sl, places=5)
        self.assertAlmostEqual(float(updated_position['tp']), new_tp, places=5)
        
        # Close the position
        await self.broker.close_position(updated_position['position_id'])
        
        print("Position modified and closed successfully")

    async def test_get_historical_data(self):
        """Test getting historical price data."""
        # Note: This is a basic test - the actual implementation would use MT5's copy_rates_from
        # For now, we'll just test that the symbol info is accessible
        symbol_info = await self.broker.get_symbol_info(TEST_SYMBOL)
        self.assertIsNotNone(symbol_info)
        
        print(f"\nSymbol info for {TEST_SYMBOL}:")
        print(f"Min volume: {symbol_info['volume_min']}")
        print(f"Max volume: {symbol_info['volume_max']}")
        print(f"Volume step: {symbol_info['volume_step']}")
        print(f"Digits: {symbol_info['digits']}")
        print(f"Spread: {symbol_info['spread']}")

    async def test_account_info(self):
        """Test getting account information."""
        account_info = await self.broker.get_account_info()
        self.assertIsNotNone(account_info)
        
        print("\nAccount Information:")
        for key, value in account_info.items():
            print(f"{key}: {value}")

    async def test_symbol_info(self):
        """Test getting symbol information."""
        symbol_info = await self.broker.get_symbol_info(TEST_SYMBOL)
        self.assertIsNotNone(symbol_info)
        
        print(f"\nSymbol Information for {TEST_SYMBOL}:")
        for key, value in symbol_info.items():
            if key != 'timestamp':  # Skip timestamp as it changes
                print(f"{key}: {value}")


if __name__ == '__main__':
    unittest.main()
