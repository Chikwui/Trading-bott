"""
Integration tests for the execution flow with MT5BacktestBroker.

These tests verify the interaction between the execution handler and the backtest broker.
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import numpy as np

from core.execution.execution_handler import ExecutionHandler
from core.execution.backtest.mt5_backtest_broker import MT5BacktestBroker, create_mt5_backtest_broker
from core.execution.broker import BrokerType, BrokerError, OrderError, MarketDataError
from core.models import Order, OrderType, OrderSide, OrderStatus, TimeInForce, Position, TradeSignal, SignalType
from core.data.providers.mt5_provider import MT5DataProvider

# Test data
TEST_SYMBOL = "EURUSD"
TEST_PRICE = 1.1000
TEST_QUANTITY = 1000
TEST_ORDER_ID = "test_order_123"

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
    
    return mock

@pytest.fixture
def execution_handler(mock_mt5_provider):
    """Create an execution handler with a backtest broker."""
    broker = create_mt5_backtest_broker(
        data_provider=mock_mt5_provider,
        initial_balance=10000.0,
        leverage=1.0,
        commission=0.0005,
        spread=0.0002,
        slippage=0.0001
    )
    
    handler = ExecutionHandler(broker=broker)
    return handler

@pytest.fixture
async def connected_execution_handler(execution_handler):
    """Create and connect an execution handler."""
    await execution_handler.initialize()
    return execution_handler

# Tests
class TestExecutionFlow:
    """Test execution flow with MT5BacktestBroker."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, execution_handler):
        """Test execution handler initialization."""
        assert execution_handler is not None
        assert execution_handler.broker is not None
        assert not execution_handler.initialized
        
        # Initialize
        await execution_handler.initialize()
        assert execution_handler.initialized
    
    @pytest.mark.asyncio
    async def test_place_market_order(self, connected_execution_handler, mock_mt5_provider):
        """Test placing a market order through the execution handler."""
        # Create a trade signal
        signal = TradeSignal(
            symbol=TEST_SYMBOL,
            signal_type=SignalType.LONG,
            price=TEST_PRICE,
            quantity=TEST_QUANTITY,
            timestamp=datetime.utcnow(),
            stop_loss=TEST_PRICE - 0.0010,
            take_profit=TEST_PRICE + 0.0020
        )
        
        # Place the order
        order = await connected_execution_handler.place_order(signal)
        
        # Verify the order was placed
        assert order is not None
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == TEST_QUANTITY
        assert order.avg_fill_price > 0
        
        # Verify the position was created
        positions = await connected_execution_handler.get_positions()
        assert len(positions) == 1
        assert positions[0]["symbol"] == TEST_SYMBOL
        assert positions[0]["quantity"] == TEST_QUANTITY
    
    @pytest.mark.asyncio
    async def test_place_limit_order(self, connected_execution_handler, mock_mt5_provider):
        """Test placing a limit order through the execution handler."""
        # Create a trade signal with limit order
        limit_price = TEST_PRICE - 0.0010  # Below current price
        signal = TradeSignal(
            symbol=TEST_SYMBOL,
            signal_type=SignalType.LONG_LIMIT,
            price=limit_price,
            quantity=TEST_QUANTITY,
            timestamp=datetime.utcnow(),
            stop_loss=limit_price - 0.0010,
            take_profit=limit_price + 0.0020
        )
        
        # Place the order
        order = await connected_execution_handler.place_order(signal)
        
        # Verify the order was placed but not filled yet
        assert order is not None
        assert order.status == OrderStatus.NEW
        assert order.filled_quantity == 0
        
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
        await connected_execution_handler.broker.update_market_data(new_time)
        
        # Verify the order was filled
        order_status = await connected_execution_handler.get_order_status(order.order_id)
        assert order_status["status"] == OrderStatus.FILLED
        
        # Verify the position was created
        positions = await connected_execution_handler.get_positions()
        assert len(positions) == 1
        assert positions[0]["symbol"] == TEST_SYMBOL
        assert positions[0]["quantity"] == TEST_QUANTITY
    
    @pytest.mark.asyncio
    async def test_place_stop_order(self, connected_execution_handler, mock_mt5_provider):
        """Test placing a stop order through the execution handler."""
        # Create a trade signal with stop order
        stop_price = TEST_PRICE + 0.0010  # Above current price
        signal = TradeSignal(
            symbol=TEST_SYMBOL,
            signal_type=SignalType.LONG_STOP,
            price=stop_price,
            quantity=TEST_QUANTITY,
            timestamp=datetime.utcnow(),
            stop_loss=stop_price - 0.0010,
            take_profit=stop_price + 0.0020
        )
        
        # Place the order
        order = await connected_execution_handler.place_order(signal)
        
        # Verify the order was placed but not filled yet
        assert order is not None
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
        await connected_execution_handler.broker.update_market_data(new_time)
        
        # Verify the order was filled as a market order
        order_status = await connected_execution_handler.get_order_status(order.order_id)
        assert order_status["status"] == OrderStatus.FILLED
        
        # Verify the position was created
        positions = await connected_execution_handler.get_positions()
        assert len(positions) == 1
        assert positions[0]["symbol"] == TEST_SYMBOL
        assert positions[0]["quantity"] == TEST_QUANTITY
    
    @pytest.mark.asyncio
    async def test_modify_order(self, connected_execution_handler):
        """Test modifying an existing order through the execution handler."""
        # Create and place a limit order
        limit_price = TEST_PRICE - 0.0010
        signal = TradeSignal(
            symbol=TEST_SYMBOL,
            signal_type=SignalType.LONG_LIMIT,
            price=limit_price,
            quantity=TEST_QUANTITY,
            timestamp=datetime.utcnow(),
            stop_loss=limit_price - 0.0010,
            take_profit=limit_price + 0.0020
        )
        
        # Place the order
        order = await connected_execution_handler.place_order(signal)
        
        # Modify the order (change price and quantity)
        new_price = limit_price - 0.0005
        new_quantity = TEST_QUANTITY * 2
        
        modified_order = await connected_execution_handler.modify_order(
            order_id=order.order_id,
            price=new_price,
            quantity=new_quantity
        )
        
        # Verify the order was modified
        assert modified_order is not None
        assert modified_order.order_id == order.order_id
        assert modified_order.limit_price == new_price
        assert modified_order.quantity == new_quantity
        
        # Verify the order is still active
        order_status = await connected_execution_handler.get_order_status(order.order_id)
        assert order_status["status"] == OrderStatus.NEW
    
    @pytest.mark.asyncio
    async def test_cancel_order(self, connected_execution_handler):
        """Test canceling an order through the execution handler."""
        # Create and place a limit order
        limit_price = TEST_PRICE - 0.0010
        signal = TradeSignal(
            symbol=TEST_SYMBOL,
            signal_type=SignalType.LONG_LIMIT,
            price=limit_price,
            quantity=TEST_QUANTITY,
            timestamp=datetime.utcnow()
        )
        
        # Place the order
        order = await connected_execution_handler.place_order(signal)
        
        # Cancel the order
        result = await connected_execution_handler.cancel_order(order.order_id)
        assert result is True
        
        # Verify the order was canceled
        order_status = await connected_execution_handler.get_order_status(order.order_id)
        assert order_status["status"] == OrderStatus.CANCELED
        
        # Try to cancel a non-existent order
        with pytest.raises(OrderError):
            await connected_execution_handler.cancel_order("nonexistent_order")
    
    @pytest.mark.asyncio
    async def test_position_management(self, connected_execution_handler):
        """Test position management through the execution handler."""
        # Open a position
        signal = TradeSignal(
            symbol=TEST_SYMBOL,
            signal_type=SignalType.LONG,
            price=TEST_PRICE,
            quantity=TEST_QUANTITY,
            timestamp=datetime.utcnow(),
            stop_loss=TEST_PRICE - 0.0010,
            take_profit=TEST_PRICE + 0.0020
        )
        
        # Place the order
        order = await connected_execution_handler.place_order(signal)
        
        # Verify the position was opened
        positions = await connected_execution_handler.get_positions()
        assert len(positions) == 1
        assert positions[0]["symbol"] == TEST_SYMBOL
        assert positions[0]["quantity"] == TEST_QUANTITY
        
        # Close the position
        close_signal = TradeSignal(
            symbol=TEST_SYMBOL,
            signal_type=SignalType.CLOSE_LONG,
            price=TEST_PRICE + 0.0005,
            quantity=TEST_QUANTITY,
            timestamp=datetime.utcnow()
        )
        
        # Place the close order
        close_order = await connected_execution_handler.place_order(close_signal)
        
        # Verify the position was closed
        positions = await connected_execution_handler.get_positions()
        assert len(positions) == 0
    
    @pytest.mark.asyncio
    async def test_get_account_info(self, connected_execution_handler):
        """Test getting account information through the execution handler."""
        account_info = await connected_execution_handler.get_account_info()
        
        assert isinstance(account_info, dict)
        assert "balance" in account_info
        assert "equity" in account_info
        assert "margin_used" in account_info
        assert "margin_free" in account_info
        assert "leverage" in account_info
        assert "currency" in account_info
        assert "timestamp" in account_info
    
    @pytest.mark.asyncio
    async def test_get_order_status(self, connected_execution_handler):
        """Test getting order status through the execution handler."""
        # Create and place an order
        signal = TradeSignal(
            symbol=TEST_SYMBOL,
            signal_type=SignalType.LONG,
            price=TEST_PRICE,
            quantity=TEST_QUANTITY,
            timestamp=datetime.utcnow()
        )
        
        # Place the order
        order = await connected_execution_handler.place_order(signal)
        
        # Get order status
        status = await connected_execution_handler.get_order_status(order.order_id)
        
        # Verify the status
        assert status is not None
        assert "order_id" in status
        assert "status" in status
        assert status["status"] == "FILLED"
        
        # Test with non-existent order
        with pytest.raises(OrderError):
            await connected_execution_handler.get_order_status("nonexistent_order")
    
    @pytest.mark.asyncio
    async def test_get_historical_data(self, connected_execution_handler):
        """Test getting historical market data through the execution handler."""
        # Test with required parameters only
        data = await connected_execution_handler.get_historical_data(TEST_SYMBOL, "1h")
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
        data = await connected_execution_handler.get_historical_data(
            symbol=TEST_SYMBOL,
            timeframe="1d",
            start=start,
            end=end,
            limit=10
        )
        assert isinstance(data, list)
        assert len(data) <= 10
    
    @pytest.mark.asyncio
    async def test_get_current_price(self, connected_execution_handler):
        """Test getting the current market price through the execution handler."""
        price = await connected_execution_handler.get_current_price(TEST_SYMBOL)
        assert isinstance(price, float)
        assert price > 0
        
        # Test with non-existent symbol
        with pytest.raises(MarketDataError):
            await connected_execution_handler.get_current_price("NONEXISTENT")
    
    @pytest.mark.asyncio
    async def test_get_order_book(self, connected_execution_handler):
        """Test getting the order book through the execution handler."""
        order_book = await connected_execution_handler.get_order_book(TEST_SYMBOL)
        
        assert isinstance(order_book, dict)
        assert "bids" in order_book
        assert "asks" in order_book
        assert "timestamp" in order_book
        assert isinstance(order_book["bids"], list)
        assert isinstance(order_book["asks"], list)
        
        # Test with non-existent symbol
        with pytest.raises(MarketDataError):
            await connected_execution_handler.get_order_book("NONEXISTENT")
