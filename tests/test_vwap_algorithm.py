"""
Tests for the VWAP (Volume-Weighted Average Price) execution algorithm.
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from decimal import Decimal
from datetime import datetime, timedelta, timezone
import numpy as np

from trading_bot.core.trading.algorithms.vwap import VWAPExecutor
from trading_bot.core.order_types import Order, OrderSide, OrderType, OrderStatus, TimeInForce

# Test configuration
TEST_SYMBOL = "BTC/USDT"
TEST_QUANTITY = Decimal('10.0')
TEST_PRICE = Decimal('50000.0')

@pytest.fixture
def mock_exchange_adapter():
    """Create a mock exchange adapter with basic functionality."""
    mock = AsyncMock()
    
    # Mock market data
    mock.get_market_data.return_value = {
        'best_bid': Decimal('49990'),
        'best_ask': Decimal('50010'),
        'last_price': Decimal('50000'),
        'volume_24h': Decimal('1000'),
        'spread': Decimal('0.0004')
    }
    
    # Mock historical volume profile
    mock.get_historical_volume_profile.return_value = [
        {
            'start_time': datetime.utcnow() - timedelta(minutes=5),
            'end_time': datetime.utcnow() - timedelta(minutes=4),
            'volume': Decimal('100'),
            'vwap': Decimal('49950'),
            'trades': 50
        },
        {
            'start_time': datetime.utcnow() - timedelta(minutes=4),
            'end_time': datetime.utcnow() - timedelta(minutes=3),
            'volume': Decimal('150'),
            'vwap': Decimal('50020'),
            'trades': 75
        },
        {
            'start_time': datetime.utcnow() - timedelta(minutes=3),
            'end_time': datetime.utcnow() - timedelta(minutes=2),
            'volume': Decimal('200'),
            'vwap': Decimal('50050'),
            'trades': 100
        },
        {
            'start_time': datetime.utcnow() - timedelta(minutes=2),
            'end_time': datetime.utcnow() - timedelta(minutes=1),
            'volume': Decimal('180'),
            'vwap': Decimal('50030'),
            'trades': 90
        },
        {
            'start_time': datetime.utcnow() - timedelta(minutes=1),
            'end_time': datetime.utcnow(),
            'volume': Decimal('120'),
            'vwap': Decimal('50010'),
            'trades': 60
        }
    ]
    
    # Mock order submission
    async def submit_order_side_effect(*args, **kwargs):
        return {
            'order_id': kwargs.get('client_order_id', 'mock_order_id'),
            'status': 'FILLED',
            'filled_quantity': kwargs['quantity'],
            'remaining_quantity': Decimal('0'),
            'filled_price': kwargs.get('price', Decimal('50000')),
            'timestamp': datetime.utcnow()
        }
    
    mock.submit_order.side_effect = submit_order_side_effect
    return mock

@pytest.fixture
def mock_position_manager():
    """Create a mock position manager."""
    mock = AsyncMock()
    return mock

@pytest.fixture
def sample_order():
    """Create a sample order for testing."""
    return Order(
        order_id='test_order_123',
        client_order_id='client_123',
        symbol=TEST_SYMBOL,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=TEST_QUANTITY,
        status=OrderStatus.NEW,
        timestamp=datetime.now(timezone.utc)
    )

@pytest.mark.asyncio
async def test_vwap_basic_execution(mock_exchange_adapter, mock_position_manager, sample_order):
    """Test basic VWAP execution with market orders."""
    # Configure VWAP executor
    config = {
        'duration': '5m',  # 5 minutes total
        'time_slice': '1m',  # 1 minute slices
        'allow_market_orders': True
    }
    
    vwap = VWAPExecutor(mock_exchange_adapter, mock_position_manager, config)
    
    # Execute VWAP
    start_time = datetime.utcnow()
    result = await vwap.execute(sample_order)
    end_time = datetime.utcnow()
    
    # Verify results
    assert result.status == OrderStatus.FILLED
    assert result.filled_quantity == sample_order.quantity
    assert result.remaining_quantity == Decimal('0')
    
    # Verify order was sliced according to volume profile
    # Should be 5 slices based on the mock data
    assert mock_exchange_adapter.submit_order.call_count == 5
    
    # Verify timing (should be approximately 4 minutes for 5 slices with 1m intervals)
    duration = (end_time - start_time).total_seconds()
    assert 3 <= duration <= 5  # Allow some flexibility

@pytest.mark.asyncio
async def test_vwap_limit_orders(mock_exchange_adapter, mock_position_manager, sample_order):
    """Test VWAP execution with limit orders."""
    # Configure VWAP executor to use limit orders
    config = {
        'duration': '5m',
        'time_slice': '1m',
        'allow_market_orders': False,
        'limit_order_spread': 0.0005  # 5 bps inside mid
    }
    
    vwap = VWAPExecutor(mock_exchange_adapter, mock_position_manager, config)
    
    # Execute VWAP
    sample_order.order_type = OrderType.LIMIT
    result = await vwap.execute(sample_order)
    
    # Verify results
    assert result.status == OrderStatus.FILLED
    assert result.filled_quantity == sample_order.quantity
    
    # Verify limit orders were used
    for call in mock_exchange_adapter.submit_order.call_args_list:
        _, kwargs = call
        assert kwargs['order_type'] == OrderType.LIMIT
        assert kwargs['time_in_force'] == TimeInForce.GTC
        
        # Verify limit price is within expected range
        if sample_order.side == OrderSide.BUY:
            assert kwargs['price'] < Decimal('50010')  # Below ask
        else:
            assert kwargs['price'] > Decimal('49990')  # Above bid

@pytest.mark.asyncio
async def test_vwap_volume_constraints(mock_exchange_adapter, mock_position_manager, sample_order):
    """Test VWAP execution with volume constraints."""
    # Configure VWAP with aggressive volume constraints
    config = {
        'duration': '5m',
        'time_slice': '1m',
        'max_participation': 0.1,  # Max 10% of volume per slice
    }
    
    vwap = VWAPExecutor(mock_exchange_adapter, mock_position_manager, config)
    
    # Execute VWAP with a large order
    sample_order.quantity = Decimal('100')  # 100 BTC total
    result = await vwap.execute(sample_order)
    
    # Verify results - should be filled based on volume profile
    assert result.status == OrderStatus.FILLED
    assert result.filled_quantity == sample_order.quantity
    
    # Verify each slice respected volume constraints
    # Based on mock data, total volume is 750, so max 75 per slice (10% of total)
    for call in mock_exchange_adapter.submit_order.call_args_list:
        _, kwargs = call
        assert kwargs['quantity'] <= Decimal('75')

@pytest.mark.asyncio
async def test_vwap_cancellation(mock_exchange_adapter, mock_position_manager, sample_order):
    """Test VWAP execution cancellation."""
    # Configure VWAP with longer duration
    config = {
        'duration': '10m',  # 10 minutes total
        'time_slice': '1m',  # 1 minute slices
        'allow_market_orders': True
    }
    
    # Add delay to mock exchange to simulate execution time
    async def delayed_submit_order(*args, **kwargs):
        await asyncio.sleep(0.1)  # 100ms delay
        return {
            'order_id': kwargs.get('client_order_id', 'mock_order_id'),
            'status': 'FILLED',
            'filled_quantity': kwargs['quantity'],
            'remaining_quantity': Decimal('0'),
            'filled_price': kwargs.get('price', Decimal('50000')),
            'timestamp': datetime.utcnow()
        }
    
    mock_exchange_adapter.submit_order.side_effect = delayed_submit_order
    
    vwap = VWAPExecutor(mock_exchange_adapter, mock_position_manager, config)
    
    # Start execution in background
    task = asyncio.create_task(vwap.execute(sample_order))
    
    # Wait for first slice to start executing
    await asyncio.sleep(0.05)
    
    # Cancel execution
    await vwap.cancel()
    
    # Wait for task to complete
    result = await task
    
    # Verify results - should be partially filled
    assert result.status in [OrderStatus.PARTIALLY_FILLED, OrderStatus.CANCELED]
    assert result.filled_quantity > 0
    assert result.filled_quantity < sample_order.quantity
    
    # Verify only a few slices were executed
    assert 1 <= mock_exchange_adapter.submit_order.call_count < 5

@pytest.mark.asyncio
async def test_vwap_error_handling(mock_exchange_adapter, mock_position_manager, sample_order):
    """Test VWAP execution with order submission errors."""
    # Make every other order submission fail
    call_count = 0
    
    async def mock_submit_order(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        
        if call_count % 2 == 0:
            raise Exception("Exchange error")
            
        return {
            'order_id': kwargs.get('client_order_id', 'mock_order_id'),
            'status': 'FILLED',
            'filled_quantity': kwargs['quantity'],
            'remaining_quantity': Decimal('0'),
            'filled_price': kwargs.get('price', Decimal('50000')),
            'timestamp': datetime.utcnow()
        }
    
    mock_exchange_adapter.submit_order.side_effect = mock_submit_order
    
    # Configure VWAP
    config = {
        'duration': '5m',
        'time_slice': '1m',
        'allow_market_orders': True,
        'max_retries': 1
    }
    
    vwap = VWAPExecutor(mock_exchange_adapter, mock_position_manager, config)
    
    # Execute VWAP
    result = await vwap.execute(sample_order)
    
    # Verify results - should be partially filled
    assert result.status == OrderStatus.PARTIALLY_FILLED
    assert result.filled_quantity > 0
    assert result.filled_quantity < sample_order.quantity
    
    # Verify we attempted all slices
    assert mock_exchange_adapter.submit_order.call_count == 5

@pytest.mark.asyncio
async def test_vwap_default_profile(mock_exchange_adapter, mock_position_manager, sample_order):
    """Test VWAP execution when exchange data is unavailable."""
    # Configure mock to return no volume profile
    mock_exchange_adapter.get_historical_volume_profile.return_value = []
    
    # Configure VWAP
    config = {
        'duration': '5m',
        'time_slice': '1m',
        'allow_market_orders': True
    }
    
    vwap = VWAPExecutor(mock_exchange_adapter, mock_position_manager, config)
    
    # Execute VWAP
    result = await vwap.execute(sample_order)
    
    # Verify results - should still complete with default profile
    assert result.status == OrderStatus.FILLED
    assert result.filled_quantity == sample_order.quantity
    
    # Should have created a default profile with 5 slices (5m / 1m)
    assert mock_exchange_adapter.submit_order.call_count == 5

def test_parse_duration():
    """Test duration string parsing."""
    vwap = VWAPExecutor(None, None, {})
    
    # Test various duration formats
    assert vwap._parse_duration('1d') == timedelta(days=1)
    assert vwap._parse_duration('4h') == timedelta(hours=4)
    assert vwap._parse_duration('30m') == timedelta(minutes=30)
    
    # Test invalid format
    assert vwap._parse_duration('invalid') == timedelta(hours=1)  # Default 1h
    assert vwap._parse_duration('') == timedelta(hours=1)  # Default 1h

@pytest.mark.asyncio
async def test_vwap_small_order(mock_exchange_adapter, mock_position_manager, sample_order):
    """Test VWAP execution with a very small order."""
    # Configure VWAP with minimum order size
    config = {
        'duration': '5m',
        'time_slice': '1m',
        'min_order_size': 0.1,
        'allow_market_orders': True
    }
    
    vwap = VWAPExecutor(mock_exchange_adapter, mock_position_manager, config)
    
    # Execute VWAP with a very small order
    sample_order.quantity = Decimal('0.05')  # Below minimum order size
    result = await vwap.execute(sample_order)
    
    # Verify results - should be rejected due to size
    assert result.status == OrderStatus.REJECTED
    assert "below minimum order size" in str(result.error_message).lower()
    
    # Execute with size just above minimum
    sample_order.quantity = Decimal('0.15')
    sample_order.status = OrderStatus.NEW  # Reset status
    result = await vwap.execute(sample_order)
    
    # Should execute normally
    assert result.status == OrderStatus.FILLED
    assert result.filled_quantity == sample_order.quantity
