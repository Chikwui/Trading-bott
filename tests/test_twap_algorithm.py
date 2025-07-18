""
Tests for the TWAP (Time-Weighted Average Price) execution algorithm.
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from decimal import Decimal
from datetime import datetime, timedelta

from trading_bot.core.trading.algorithms.twap import TWAPExecutor
from trading_bot.core.order_types import (
    Order, OrderSide, OrderType, OrderStatus, TimeInForce
)

@pytest.fixture
def mock_exchange_adapter():
    """Create a mock exchange adapter."""
    mock = AsyncMock()
    
    # Mock market data
    mock.get_market_data.return_value = {
        'best_bid': Decimal('49990'),
        'best_ask': Decimal('50010'),
        'last_price': Decimal('50000'),
        'volume_24h': Decimal('1000'),
        'spread': Decimal('0.0004')
    }
    
    # Mock order submission
    async def submit_order_side_effect(*args, **kwargs):
        return {
            'order_id': kwargs.get('client_order_id', 'mock_order_id'),
            'status': 'FILLED',
            'filled_quantity': kwargs['quantity'],
            'remaining_quantity': Decimal('0'),
            'filled_price': Decimal('50000'),
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
        symbol='BTC/USDT',
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal('1.0'),
        status=OrderStatus.NEW,
        timestamp=datetime.utcnow()
    )

@pytest.mark.asyncio
async def test_twap_basic_execution(mock_exchange_adapter, mock_position_manager, sample_order):
    """Test basic TWAP execution with market orders."""
    # Configure TWAP executor
    config = {
        'duration_seconds': 10,  # 10 seconds total
        'slices': 5,  # 5 slices = 1 slice every 2 seconds
        'allow_market_orders': True
    }
    
    twap = TWAPExecutor(mock_exchange_adapter, mock_position_manager, config)
    
    # Execute TWAP
    start_time = datetime.utcnow()
    result = await twap.execute(sample_order)
    end_time = datetime.utcnow()
    
    # Verify results
    assert result.status == OrderStatus.FILLED
    assert result.filled_quantity == sample_order.quantity
    assert result.remaining_quantity == Decimal('0')
    
    # Verify order was sliced correctly
    assert mock_exchange_adapter.submit_order.call_count == 5
    
    # Verify timing (approximately 8-12 seconds for 5 slices with 2s intervals)
    duration = (end_time - start_time).total_seconds()
    assert 8 <= duration <= 12

@pytest.mark.asyncio
async def test_twap_limit_orders(mock_exchange_adapter, mock_position_manager, sample_order):
    """Test TWAP execution with limit orders."""
    # Configure TWAP executor to use limit orders
    config = {
        'duration_seconds': 10,
        'slices': 3,
        'allow_market_orders': False,
        'limit_order_spread': 0.0005  # 5 bps inside mid
    }
    
    twap = TWAPExecutor(mock_exchange_adapter, mock_position_manager, config)
    
    # Execute TWAP
    sample_order.order_type = OrderType.LIMIT
    result = await twap.execute(sample_order)
    
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
async def test_twap_volume_constraints(mock_exchange_adapter, mock_position_manager, sample_order):
    """Test TWAP execution with volume constraints."""
    # Configure TWAP with aggressive volume constraints
    config = {
        'duration_seconds': 10,
        'slices': 5,
        'max_slice_pct_volume': 0.001,  # Max 0.1% of daily volume per slice
        'volume_24h': 1000  # 1000 BTC daily volume
    }
    
    twap = TWAPExecutor(mock_exchange_adapter, mock_position_manager, config)
    
    # Execute TWAP with a large order
    sample_order.quantity = Decimal('10')  # 10 BTC total
    result = await twap.execute(sample_order)
    
    # Verify results - should be partially filled due to volume constraints
    assert result.status == OrderStatus.PARTIALLY_FILLED
    assert result.filled_quantity > 0
    assert result.filled_quantity < sample_order.quantity
    
    # Verify each slice was within volume constraints (0.1% of 1000 = 1 BTC max per slice)
    for call in mock_exchange_adapter.submit_order.call_args_list:
        _, kwargs = call
        assert kwargs['quantity'] <= Decimal('1.0')

@pytest.mark.asyncio
async def test_twap_cancellation(mock_exchange_adapter, mock_position_manager, sample_order):
    """Test TWAP execution cancellation."""
    # Configure TWAP with longer duration
    config = {
        'duration_seconds': 60,  # 60 seconds total
        'slices': 10,  # 10 slices = 1 every 6 seconds
        'allow_market_orders': True
    }
    
    twap = TWAPExecutor(mock_exchange_adapter, mock_position_manager, config)
    
    # Start execution in background
    task = asyncio.create_task(twap.execute(sample_order))
    
    # Wait for first slice to execute
    await asyncio.sleep(1)
    
    # Cancel execution
    await twap.cancel()
    
    # Wait for task to complete
    result = await task
    
    # Verify results - should be partially filled
    assert result.status in [OrderStatus.PARTIALLY_FILLED, OrderStatus.CANCELED]
    assert result.filled_quantity > 0
    assert result.filled_quantity < sample_order.quantity
    
    # Verify only a few slices were executed
    assert 0 < mock_exchange_adapter.submit_order.call_count < 5

@pytest.mark.asyncio
async def test_twap_error_handling(mock_exchange_adapter, mock_position_manager, sample_order):
    """Test TWAP execution with order submission errors."""
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
            'filled_price': Decimal('50000'),
            'timestamp': datetime.utcnow()
        }
    
    mock_exchange_adapter.submit_order.side_effect = mock_submit_order
    
    # Configure TWAP
    config = {
        'duration_seconds': 10,
        'slices': 4,
        'allow_market_orders': True,
        'max_retries': 1
    }
    
    twap = TWAPExecutor(mock_exchange_adapter, mock_position_manager, config)
    
    # Execute TWAP
    result = await twap.execute(sample_order)
    
    # Verify results - should be partially filled
    assert result.status == OrderStatus.PARTIALLY_FILLED
    assert result.filled_quantity > 0
    assert result.filled_quantity < sample_order.quantity
    
    # Verify we attempted all slices
    assert mock_exchange_adapter.submit_order.call_count == 4

@pytest.mark.asyncio
async def test_twap_slice_calculation():
    """Test TWAP slice size and interval calculations."""
    from trading_bot.core.trading.algorithms.twap import TWAPExecutor
    
    # Mock dependencies
    mock_exchange = AsyncMock()
    mock_position_manager = AsyncMock()
    
    # Test with even distribution
    twap = TWAPExecutor(mock_exchange, mock_position_manager, {
        'slices': 5,
        'duration_seconds': 60
    })
    
    # Test slice size calculation
    assert twap._calculate_slice_size(Decimal('10'), 5) == Decimal('2')
    assert twap._calculate_slice_size(Decimal('1'), 3) == Decimal('0.33333333')
    
    # Test with min/max constraints
    assert twap._calculate_slice_size(
        Decimal('10'), 5, 
        min_size=Decimal('3')
    ) == Decimal('3')
    
    assert twap._calculate_slice_size(
        Decimal('10'), 2,
        max_size=Decimal('4')
    ) == Decimal('4')
    
    # Test interval calculation
    start = datetime(2023, 1, 1)
    end = start + timedelta(seconds=60)
    
    interval = twap._calculate_slice_interval(start, end, 5)
    assert 14 <= interval <= 16  # Should be ~15 seconds between slices
    
    # Test with randomization
    intervals = [
        twap._calculate_slice_interval(start, end, 5, randomize=True)
        for _ in range(10)
    ]
    
    # Should have some variation
    assert len(set(round(i, 2) for i in intervals)) > 1
    
    # But still centered around 15s
    avg_interval = sum(intervals) / len(intervals)
    assert 14 <= avg_interval <= 16
