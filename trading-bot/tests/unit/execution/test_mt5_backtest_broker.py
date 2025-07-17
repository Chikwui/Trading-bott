"""
Unit tests for the MT5BacktestBroker implementation.

These tests verify the functionality of the MT5 backtest broker.
"""
import pytest
import asyncio
import random
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch, call

import pandas as pd
import numpy as np

from core.execution.backtest.mt5_backtest_broker import MT5BacktestBroker, create_mt5_backtest_broker
from core.execution.broker import BrokerType, BrokerError, OrderError, MarketDataError
from core.models import Order, OrderType, OrderSide, OrderStatus, TimeInForce, Position
from core.data.providers.mt5_provider import MT5DataProvider

# Test data
TEST_SYMBOL = "EURUSD"
TEST_PRICE = 1.1000
TEST_QUANTITY = 1000
TEST_ORDER_ID = "test_order_123"
TEST_ACCOUNT_INFO = {
    "balance": 10000.0,
    "equity": 10000.0,
    "margin_used": 0.0,
    "margin_free": 10000.0,
    "leverage": 1.0,
    "currency": "USD"
}

# Fixtures
@pytest.fixture
def mock_mt5_provider():
    """Create a mock MT5 data provider."""
    mock = AsyncMock(spec=MT5DataProvider)
    
    # Mock last tick data
    mock.get_last_tick.return_value = {
        'bid': TEST_PRICE - 0.0001,
        'ask': TEST_PRICE + 0.0001,
        'last': TEST_PRICE,
        'volume': 1000,
        'time': datetime.utcnow()
    }
    
    # Mock historical data
    mock.get_historical_data.return_value = [
        {
            'timestamp': datetime.utcnow() - timedelta(minutes=i),
            'open': TEST_PRICE - 0.0005,
            'high': TEST_PRICE + 0.0005,
            'low': TEST_PRICE - 0.001,
            'close': TEST_PRICE,
            'volume': 1000,
            'spread': 2
        }
        for i in range(100, 0, -1)
    ]
    
    # Mock order book
    mock.get_order_book.return_value = {
        'bids': [[TEST_PRICE - 0.0001, 1000]],
        'asks': [[TEST_PRICE + 0.0001, 1000]],
        'timestamp': datetime.utcnow().isoformat()
    }
    
    # Mock available symbols
    mock.available_symbols.return_value = [TEST_SYMBOL, "GBPUSD", "USDJPY"]
    
    return mock

@pytest.fixture
def mt5_backtest_broker(mock_mt5_provider):
    """Create an MT5BacktestBroker instance for testing."""
    broker = MT5BacktestBroker(
        data_provider=mock_mt5_provider,
        initial_balance=10000.0,
        leverage=1.0,
        commission=0.0005,
        spread=0.0002,
        slippage=0.0001
    )
    return broker

@pytest.fixture
async def connected_broker(mt5_backtest_broker):
    """Create and connect an MT5BacktestBroker instance."""
    await mt5_backtest_broker.connect()
    return mt5_backtest_broker

# Tests
class TestMT5BacktestBroker:
    """Test suite for MT5BacktestBroker."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, mt5_backtest_broker):
        """Test broker initialization."""
        assert mt5_backtest_broker is not None
        assert mt5_backtest_broker.broker_type == BrokerType.BACKTEST
        assert not mt5_backtest_broker.connected
        assert mt5_backtest_broker.initial_balance == Decimal('10000.0')
        assert mt5_backtest_broker.leverage == Decimal('1.0')
        assert mt5_backtest_broker.commission_rate == Decimal('0.0005')
        assert mt5_backtest_broker.spread == Decimal('0.0002')
        assert mt5_backtest_broker.slippage == Decimal('0.0001')
    
    @pytest.mark.asyncio
    async def test_connect_disconnect(self, mt5_backtest_broker, mock_mt5_provider):
        """Test connecting to and disconnecting from the broker."""
        # Test connect
        result = await mt5_backtest_broker.connect()
        assert result is True
        assert mt5_backtest_broker.connected
        assert mt5_backtest_broker.current_time is not None
        
        # Verify data provider was called
        mock_mt5_provider.connect.assert_awaited_once()
        
        # Test disconnect
        await mt5_backtest_broker.disconnect()
        assert not mt5_backtest_broker.connected
    
    @pytest.mark.asyncio
    async def test_get_account_info(self, connected_broker):
        """Test getting account information."""
        account_info = await connected_broker.get_account_info()
        
        assert isinstance(account_info, dict)
        assert "balance" in account_info
        assert "equity" in account_info
        assert "margin_used" in account_info
        assert "margin_free" in account_info
        assert "leverage" in account_info
        assert "currency" in account_info
        assert "timestamp" in account_info
        
        # Initial values
        assert account_info["balance"] == 10000.0
        assert account_info["equity"] == 10000.0
        assert account_info["margin_used"] == 0.0
        assert account_info["margin_free"] == 10000.0
        assert account_info["leverage"] == 1.0
        assert account_info["currency"] == "USD"
    
    @pytest.mark.asyncio
    async def test_place_market_order(self, connected_broker, mock_mt5_provider):
        """Test placing a market order."""
        # Create a market order
        order = Order(
            symbol=TEST_SYMBOL,
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=TEST_QUANTITY,
            time_in_force=TimeInForce.DAY
        )
        
        # Place the order
        order_id = await connected_broker.place_order(order)
        
        # Verify the order was processed
        assert order_id is not None
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == TEST_QUANTITY
        assert order.avg_fill_price > 0
        assert order.commission > 0
        
        # Verify the position was created
        positions = await connected_broker.get_positions()
        assert len(positions) == 1
        assert positions[0]["symbol"] == TEST_SYMBOL
        assert positions[0]["quantity"] == TEST_QUANTITY
        
        # Verify commission was applied
        assert connected_broker.commissions_paid > 0
    
    @pytest.mark.asyncio
    async def test_place_limit_order(self, connected_broker, mock_mt5_provider):
        """Test placing a limit order."""
        # Create a limit order
        limit_price = TEST_PRICE - 0.0010  # Below current price
        order = Order(
            symbol=TEST_SYMBOL,
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=TEST_QUANTITY,
            limit_price=limit_price,
            time_in_force=TimeInForce.GTC
        )
        
        # Place the order
        order_id = await connected_broker.place_order(order)
        
        # Verify the order was placed but not filled yet
        assert order_id is not None
        assert order.status == OrderStatus.NEW
        assert order.filled_quantity == 0
        
        # Verify the order is in the orders list
        assert order_id in connected_broker.orders
        
        # Simulate price movement to trigger the limit order
        mock_mt5_provider.get_last_tick.return_value = {
            'bid': limit_price - 0.0001,
            'ask': limit_price + 0.0001,
            'last': limit_price,
            'volume': 1000,
            'time': datetime.utcnow()
        }
        
        # Update market data to trigger order processing
        new_time = datetime.utcnow()
        await connected_broker.update_market_data(new_time)
        
        # Verify the order was filled
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == TEST_QUANTITY
        assert abs(order.avg_fill_price - limit_price) < 0.0001
    
    @pytest.mark.asyncio
    async def test_place_stop_order(self, connected_broker, mock_mt5_provider):
        """Test placing a stop order."""
        # Create a stop order
        stop_price = TEST_PRICE + 0.0010  # Above current price
        order = Order(
            symbol=TEST_SYMBOL,
            order_type=OrderType.STOP,
            side=OrderSide.BUY,
            quantity=TEST_QUANTITY,
            stop_price=stop_price,
            time_in_force=TimeInForce.GTC
        )
        
        # Place the order
        order_id = await connected_broker.place_order(order)
        
        # Verify the order was placed but not filled yet
        assert order_id is not None
        assert order.status == OrderStatus.NEW
        assert order.filled_quantity == 0
        
        # Simulate price movement to trigger the stop order
        mock_mt5_provider.get_last_tick.return_value = {
            'bid': stop_price - 0.0001,
            'ask': stop_price + 0.0001,
            'last': stop_price,
            'volume': 1000,
            'time': datetime.utcnow()
        }
        
        # Update market data to trigger order processing
        new_time = datetime.utcnow()
        await connected_broker.update_market_data(new_time)
        
        # Verify the order was filled as a market order
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == TEST_QUANTITY
        assert order.avg_fill_price > stop_price  # Should be filled above the stop price
    
    @pytest.mark.asyncio
    async def test_cancel_order(self, connected_broker):
        """Test canceling an order."""
        # Create and place a limit order
        order = Order(
            symbol=TEST_SYMBOL,
            order_type=OrderType.LIMIT,
            side=OrderSide.BUY,
            quantity=TEST_QUANTITY,
            limit_price=TEST_PRICE - 0.0010,
            time_in_force=TimeInForce.GTC
        )
        
        # Place the order
        order_id = await connected_broker.place_order(order)
        
        # Verify the order exists
        assert order_id in connected_broker.orders
        
        # Cancel the order
        result = await connected_broker.cancel_order(order_id)
        assert result is True
        
        # Verify the order was canceled
        assert order.status == OrderStatus.CANCELED
        
        # Try to cancel a non-existent order
        with pytest.raises(OrderError):
            await connected_broker.cancel_order("nonexistent_order")
    
    @pytest.mark.asyncio
    async def test_get_order_status(self, connected_broker):
        """Test getting order status."""
        # Create and place an order
        order = Order(
            symbol=TEST_SYMBOL,
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=TEST_QUANTITY,
            time_in_force=TimeInForce.DAY
        )
        
        # Place the order
        order_id = await connected_broker.place_order(order)
        
        # Get order status
        status = await connected_broker.get_order_status(order_id)
        
        # Verify the status
        assert status is not None
        assert "order_id" in status
        assert "status" in status
        assert status["status"] == "FILLED"
        
        # Test with non-existent order
        with pytest.raises(OrderError):
            await connected_broker.get_order_status("nonexistent_order")
    
    @pytest.mark.asyncio
    async def test_get_positions(self, connected_broker):
        """Test getting positions."""
        # Initially no positions
        positions = await connected_broker.get_positions()
        assert len(positions) == 0
        
        # Open a position
        order = Order(
            symbol=TEST_SYMBOL,
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=TEST_QUANTITY,
            time_in_force=TimeInForce.DAY
        )
        await connected_broker.place_order(order)
        
        # Get all positions
        positions = await connected_broker.get_positions()
        assert len(positions) == 1
        assert positions[0]["symbol"] == TEST_SYMBOL
        assert positions[0]["quantity"] == TEST_QUANTITY
        
        # Get position for specific symbol
        positions = await connected_broker.get_positions(TEST_SYMBOL)
        assert len(positions) == 1
        
        # Get position for non-existent symbol
        positions = await connected_broker.get_positions("NONEXISTENT")
        assert len(positions) == 0
    
    @pytest.mark.asyncio
    async def test_get_historical_data(self, connected_broker, mock_mt5_provider):
        """Test getting historical market data."""
        # Test with required parameters only
        data = await connected_broker.get_historical_data(TEST_SYMBOL, "1h")
        assert isinstance(data, list)
        assert len(data) > 0
        
        # Verify required fields
        for item in data:
            assert "timestamp" in item
            assert "open" in item
            assert "high" in item
            assert "low" in item
            assert "close" in item
            assert "volume" in item
        
        # Test with all parameters
        end = datetime.utcnow()
        start = end - timedelta(days=7)
        data = await connected_broker.get_historical_data(
            symbol=TEST_SYMBOL,
            timeframe="1d",
            start=start,
            end=end,
            limit=10
        )
        assert isinstance(data, list)
        assert len(data) <= 10
    
    @pytest.mark.asyncio
    async def test_get_current_price(self, connected_broker, mock_mt5_provider):
        """Test getting the current market price."""
        price = await connected_broker.get_current_price(TEST_SYMBOL)
        assert isinstance(price, float)
        assert price > 0
        
        # Test with non-existent symbol
        with pytest.raises(MarketDataError):
            await connected_broker.get_current_price("NONEXISTENT")
    
    @pytest.mark.asyncio
    async def test_get_order_book(self, connected_broker, mock_mt5_provider):
        """Test getting the order book."""
        order_book = await connected_broker.get_order_book(TEST_SYMBOL)
        
        assert isinstance(order_book, dict)
        assert "bids" in order_book
        assert "asks" in order_book
        assert "timestamp" in order_book
        assert isinstance(order_book["bids"], list)
        assert isinstance(order_book["asks"], list)
        
        # Test with non-existent symbol
        with pytest.raises(MarketDataError):
            await connected_broker.get_order_book("NONEXISTENT")
    
    @pytest.mark.asyncio
    async def test_update_market_data(self, connected_broker, mock_mt5_provider):
        """Test updating market data."""
        # Open a position to test updates
        order = Order(
            symbol=TEST_SYMBOL,
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=TEST_QUANTITY,
            time_in_force=TimeInForce.DAY
        )
        await connected_broker.place_order(order)
        
        # Get initial position
        positions = await connected_broker.get_positions()
        initial_pnl = positions[0]["unrealized_pnl"]
        
        # Update market data with new price
        new_price = TEST_PRICE + 0.0010
        mock_mt5_provider.get_last_tick.return_value = {
            'bid': new_price - 0.0001,
            'ask': new_price + 0.0001,
            'last': new_price,
            'volume': 1000,
            'time': datetime.utcnow()
        }
        
        # Update market data
        new_time = datetime.utcnow()
        await connected_broker.update_market_data(new_time)
        
        # Verify position was updated
        positions = await connected_broker.get_positions()
        updated_pnl = positions[0]["unrealized_pnl"]
        
        # P&L should have increased (long position with price increase)
        assert updated_pnl > initial_pnl
    
    @pytest.mark.asyncio
    async def test_position_management(self, connected_broker):
        """Test position management (open, update, close)."""
        # Open a position
        buy_order = Order(
            symbol=TEST_SYMBOL,
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=TEST_QUANTITY,
            time_in_force=TimeInForce.DAY
        )
        await connected_broker.place_order(buy_order)
        
        # Verify position was opened
        positions = await connected_broker.get_positions()
        assert len(positions) == 1
        assert positions[0]["quantity"] == TEST_QUANTITY
        
        # Partially close the position
        sell_order = Order(
            symbol=TEST_SYMBOL,
            order_type=OrderType.MARKET,
            side=OrderSide.SELL,
            quantity=TEST_QUANTITY // 2,
            time_in_force=TimeInForce.DAY
        )
        await connected_broker.place_order(sell_order)
        
        # Verify position was partially closed
        positions = await connected_broker.get_positions()
        assert len(positions) == 1
        assert positions[0]["quantity"] == TEST_QUANTITY // 2
        
        # Close the remaining position
        sell_order = Order(
            symbol=TEST_SYMBOL,
            order_type=OrderType.MARKET,
            side=OrderSide.SELL,
            quantity=TEST_QUANTITY // 2,
            time_in_force=TimeInForce.DAY
        )
        await connected_broker.place_order(sell_order)
        
        # Verify position was closed
        positions = await connected_broker.get_positions()
        assert len(positions) == 0
    
    @pytest.mark.asyncio
    async def test_leverage_and_margin(self, connected_broker):
        """Test leverage and margin calculations."""
        # Set leverage to 10:1
        connected_broker.leverage = Decimal('10.0')
        
        # Get initial account info
        initial_info = await connected_broker.get_account_info()
        
        # Open a position
        order = Order(
            symbol=TEST_SYMBOL,
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=TEST_QUANTITY * 5,  # Larger position
            time_in_force=TimeInForce.DAY
        )
        await connected_broker.place_order(order)
        
        # Get updated account info
        updated_info = await connected_broker.get_account_info()
        
        # Verify margin used increased
        assert updated_info["margin_used"] > initial_info["margin_used"]
        
        # Verify margin used is approximately position value / leverage
        positions = await connected_broker.get_positions()
        position_value = positions[0]["quantity"] * positions[0]["entry_price"]
        expected_margin = position_value / 10.0  # 10:1 leverage
        assert abs(updated_info["margin_used"] - expected_margin) < 0.01
    
    @pytest.mark.asyncio
    async def test_insufficient_funds(self, connected_broker):
        """Test order placement with insufficient funds."""
        # Try to place an order that's too large
        large_quantity = 1000000  # This should exceed the account balance
        
        order = Order(
            symbol=TEST_SYMBOL,
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=large_quantity,
            time_in_force=TimeInForce.DAY
        )
        
        # Should raise OrderError due to insufficient funds
        with pytest.raises(OrderError) as excinfo:
            await connected_broker.place_order(order)
        
        assert "insufficient funds" in str(excinfo.value).lower()
    
    @pytest.mark.asyncio
    async def test_factory_function(self, mock_mt5_provider):
        """Test the create_mt5_backtest_broker factory function."""
        broker = create_mt5_backtest_broker(
            data_provider=mock_mt5_provider,
            initial_balance=50000.0,
            leverage=5.0,
            commission=0.001,
            spread=0.0003,
            slippage=0.0002
        )
        
        assert broker is not None
        assert broker.initial_balance == Decimal('50000.0')
        assert broker.leverage == Decimal('5.0')
        assert broker.commission_rate == Decimal('0.001')
        assert broker.spread == Decimal('0.0003')
        assert broker.slippage == Decimal('0.0002')
        
        # Test default values
        broker = create_mt5_backtest_broker(mock_mt5_provider)
        assert broker.initial_balance == Decimal('10000.0')
        assert broker.leverage == Decimal('1.0')
        assert broker.commission_rate == Decimal('0.0005')
        assert broker.spread == Decimal('0.0002')
        assert broker.slippage == Decimal('0.0001')
