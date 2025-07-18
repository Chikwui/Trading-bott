"""
Test suite for the Sniper execution algorithm.
"""
import asyncio
import pytest
from decimal import Decimal
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from core.trading.order_types import (
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce
)
from core.trading.algorithms.sniper import SniperExecutor

@pytest.fixture
def mock_exchange_adapter():
    """Create a mock exchange adapter with basic functionality."""
    adapter = MagicMock()
    adapter.get_market_data.return_value = {
        'best_bid': '100.0',
        'best_ask': '100.1',
        'volume_24h': '1000000',
        'lot_size': '0.00000001'
    }
    adapter.get_order_book.return_value = {
        'bids': [(Decimal('99.9'), Decimal('1000')), (Decimal('99.8'), Decimal('2000'))],
        'asks': [(Decimal('100.1'), Decimal('1000')), (Decimal('100.2'), Decimal('2000'))]
    }
    adapter.get_recent_trades.return_value = [
        {'price': '100.05', 'quantity': '10', 'timestamp': datetime.now(timezone.utc).timestamp()},
        {'price': '100.10', 'quantity': '5', 'timestamp': datetime.now(timezone.utc).timestamp() - 1}
    ]
    adapter.get_historical_volume.return_value = [
        {'timestamp': '2023-01-01', 'volume': '1000000'},
        {'timestamp': '2023-01-02', 'volume': '1100000'},
        {'timestamp': '2023-01-03', 'volume': '900000'}
    ]
    return adapter

@pytest.fixture
def mock_position_manager():
    """Create a mock position manager."""
    return MagicMock()

@pytest.fixture
def sample_order():
    """Create a sample order for testing."""
    return Order(
        order_id="test_order_123",
        client_order_id="client_123",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("1.0"),
        price=Decimal("100.0"),
        time_in_force=TimeInForce.GTC,
        status=OrderStatus.NEW,
        timestamp=datetime.now(timezone.utc)
    )

@pytest.fixture
def sniper_executor(mock_exchange_adapter, mock_position_manager):
    """Create a SniperExecutor instance for testing."""
    return SniperExecutor(
        exchange_adapter=mock_exchange_adapter,
        position_manager=mock_position_manager,
        config={
            'max_slippage': '10',  # 10 bps
            'urgency': '0.8',
            'max_participation': '0.3',
            'min_slice_size': '0.1',
            'max_slice_size': '10.0',
            'refresh_interval': '0.1',
            'max_retries': '3',
            'dark_pool_enabled': 'True',
            'anti_gaming': 'True',
            'volatility_adaptive': 'True'
        }
    )

@pytest.mark.asyncio
async def test_initialization(sniper_executor, mock_exchange_adapter, mock_position_manager):
    """Test that the SniperExecutor initializes correctly."""
    assert sniper_executor.exchange_adapter == mock_exchange_adapter
    assert sniper_executor.position_manager == mock_position_manager
    assert sniper_executor.urgency == 0.8
    assert sniper_executor.max_slippage == Decimal('0.001')  # 10 bps
    assert sniper_executor.max_participation == Decimal('0.3')
    assert sniper_executor.min_slice_size == Decimal('0.1')
    assert sniper_executor.max_slice_size == Decimal('10.0')
    assert sniper_executor.refresh_interval == 0.1
    assert sniper_executor.max_retries == 3
    assert sniper_executor.dark_pool_enabled is True
    assert sniper_executor.anti_gaming is True
    assert sniper_executor.volatility_adaptive is True

@pytest.mark.asyncio
async def test_order_validation(sniper_executor, sample_order):
    """Test that order validation works correctly."""
    # Valid order should not raise
    sniper_executor._validate_order(sample_order)
    
    # Invalid order type should raise
    invalid_order = sample_order.copy()
    invalid_order.order_type = OrderType.STOP_LIMIT
    with pytest.raises(ValueError, match="Sniper algorithm supports MARKET, LIMIT, IOC, and FOK order types"):
        sniper_executor._validate_order(invalid_order)
    
    # Invalid quantity should raise
    invalid_order = sample_order.copy()
    invalid_order.quantity = Decimal('0')
    with pytest.raises(ValueError, match="Order quantity must be positive"):
        sniper_executor._validate_order(invalid_order)

@pytest.mark.asyncio
async def test_execute_basic(sniper_executor, sample_order, mock_exchange_adapter):
    """Test basic order execution flow."""
    # Mock order submission to return a filled order
    mock_exchange_adapter.submit_order.return_value = {
        'order_id': 'test_order_123_0',
        'status': 'FILLED',
        'filled_quantity': '1.0',
        'filled_price': '100.05',
        'remaining_quantity': '0.0'
    }
    
    # Execute the order
    result = await sniper_executor.execute(sample_order)
    
    # Verify the order was executed
    assert result is not None
    assert result.quantity_executed == Decimal('1.0')
    assert result.avg_execution_price == Decimal('100.05')
    assert result.implementation_shortfall_bps is not None
    
    # Verify the exchange adapter was called correctly
    mock_exchange_adapter.submit_order.assert_called_once()
    args, kwargs = mock_exchange_adapter.submit_order.call_args
    assert kwargs['symbol'] == 'BTC/USDT'
    assert kwargs['side'] == OrderSide.BUY
    assert kwargs['quantity'] == Decimal('1.0')
    assert kwargs['price'] > Decimal('100.0')  # Should be more aggressive than the limit price

@pytest.mark.asyncio
async def test_partial_fill(sniper_executor, sample_order, mock_exchange_adapter):
    """Test handling of partial fills."""
    # First fill is partial
    mock_exchange_adapter.submit_order.side_effect = [
        {
            'order_id': 'test_order_123_0',
            'status': 'PARTIALLY_FILLED',
            'filled_quantity': '0.5',
            'filled_price': '100.05',
            'remaining_quantity': '0.5'
        },
        {
            'order_id': 'test_order_123_1',
            'status': 'FILLED',
            'filled_quantity': '0.5',
            'filled_price': '100.06',
            'remaining_quantity': '0.0'
        }
    ]
    
    # Execute the order
    result = await sniper_executor.execute(sample_order)
    
    # Verify the order was fully executed
    assert result is not None
    assert result.quantity_executed == Decimal('1.0')
    assert result.avg_execution_price == Decimal('100.055')  # (100.05 + 100.06) / 2
    
    # Should have submitted two orders
    assert mock_exchange_adapter.submit_order.call_count == 2

@pytest.mark.asyncio
async def test_dark_pool_execution(sniper_executor, sample_order, mock_exchange_adapter):
    """Test dark pool order execution."""
    # Configure for dark pool testing
    sniper_executor.dark_pool_fill_probability = 1.0  # Ensure fill
    mock_exchange_adapter.submit_order.side_effect = Exception("Should not call regular submit for dark pool")
    
    # Execute with dark pool enabled
    result = await sniper_executor.execute(sample_order)
    
    # Verify dark pool execution
    assert result is not None
    assert result.quantity_executed > 0
    assert result.avg_execution_price > Decimal('0')
    
    # Verify dark pool was used
    assert any(fill.get('venue') == 'DARK_POOL' for fill in result.fills)

@pytest.mark.asyncio
async def test_volatility_adaptation(sniper_executor, sample_order, mock_exchange_adapter):
    """Test that the algorithm adapts to changing volatility."""
    # Initial execution with normal volatility
    mock_exchange_adapter.submit_order.return_value = {
        'order_id': 'test_order_123_0',
        'status': 'FILLED',
        'filled_quantity': '1.0',
        'filled_price': '100.05',
        'remaining_quantity': '0.0'
    }
    
    # Simulate high volatility by modifying recent trades
    high_vol_trades = [
        {'price': str(100 + i % 10), 'quantity': '10', 
         'timestamp': datetime.now(timezone.utc).timestamp() - (10 - i)}
        for i in range(10)
    ]
    mock_exchange_adapter.get_recent_trades.return_value = high_vol_trades
    
    # Execute with high volatility
    await sniper_executor._update_market_state()
    assert sniper_executor.volatility > Decimal('0.5')  # Should detect high volatility
    
    # Verify slice size is reduced due to high volatility
    original_slice_size = sniper_executor.max_slice_size
    optimal_size = sniper_executor._calculate_optimal_slice_size()
    assert optimal_size < original_slice_size

@pytest.mark.asyncio
async def test_anti_gaming_mechanism(sniper_executor, sample_order, mock_exchange_adapter):
    """Test the anti-gaming mechanism."""
    # Configure order book updates to simulate adverse moves
    def mock_get_order_book(*args, **kwargs):
        return {
            'bids': [(Decimal('99.8'), Decimal('1000')), (Decimal('99.7'), Decimal('2000'))],
            'asks': [(Decimal('100.2'), Decimal('1000')), (Decimal('100.3'), Decimal('2000'))]
        }
    
    mock_exchange_adapter.get_order_book.side_effect = mock_get_order_book
    mock_exchange_adapter.submit_order.return_value = {
        'order_id': 'test_order_123_0',
        'status': 'FILLED',
        'filled_quantity': '1.0',
        'filled_price': '100.05',
        'remaining_quantity': '0.0'
    }
    
    # Execute with anti-gaming detection
    await sniper_executor.execute(sample_order)
    
    # Verify anti-gaming metrics were updated
    assert sniper_executor.predation_score > 0
    
    # If predation score is high, urgency should be reduced
    original_urgency = sniper_executor.urgency
    sniper_executor.predation_score = 0.7  # Above threshold
    await sniper_executor._monitor_execution()
    assert sniper_executor.urgency < original_urgency

@pytest.mark.asyncio
async def test_order_cancellation(sniper_executor, sample_order, mock_exchange_adapter):
    """Test order cancellation functionality."""
    # Mock order submission to return a working order
    mock_exchange_adapter.submit_order.return_value = {
        'order_id': 'test_order_123_0',
        'status': 'NEW',
        'filled_quantity': '0.0',
        'remaining_quantity': '1.0'
    }
    
    # Start execution in background
    task = asyncio.create_task(sniper_executor.execute(sample_order))
    
    # Wait for order to be submitted
    await asyncio.sleep(0.2)
    
    # Cancel the execution
    await sniper_executor.cancel()
    
    # Wait for task to complete
    result = await task
    
    # Verify cancellation
    assert result is not None
    assert result.quantity_executed < sample_order.quantity
    mock_exchange_adapter.cancel_order.assert_called_once_with('test_order_123_0')

@pytest.mark.asyncio
async def test_market_order_execution(sniper_executor, sample_order, mock_exchange_adapter):
    """Test execution with market orders."""
    # Convert to market order
    market_order = sample_order.copy()
    market_order.order_type = OrderType.MARKET
    market_order.price = None  # Market orders don't have a price
    
    # Mock market order execution
    mock_exchange_adapter.submit_order.return_value = {
        'order_id': 'test_order_123_0',
        'status': 'FILLED',
        'filled_quantity': '1.0',
        'filled_price': '100.10',  # Should be near best ask
        'remaining_quantity': '0.0'
    }
    
    # Execute market order
    result = await sniper_executor.execute(market_order)
    
    # Verify execution
    assert result is not None
    assert result.quantity_executed == Decimal('1.0')
    assert result.avg_execution_price == Decimal('100.10')
    
    # Verify order type was preserved
    args, kwargs = mock_exchange_adapter.submit_order.call_args
    assert kwargs['order_type'] == OrderType.MARKET

@pytest.mark.asyncio
async def test_error_handling(sniper_executor, sample_order, mock_exchange_adapter):
    """Test error handling and retry logic."""
    # Mock transient error followed by success
    mock_exchange_adapter.submit_order.side_effect = [
        Exception("Temporary error"),
        {
            'order_id': 'test_order_123_0',
            'status': 'FILLED',
            'filled_quantity': '1.0',
            'filled_price': '100.05',
            'remaining_quantity': '0.0'
        }
    ]
    
    # Execute with error handling
    result = await sniper_executor.execute(sample_order)
    
    # Verify recovery from error
    assert result is not None
    assert result.quantity_executed == Decimal('1.0')
    assert mock_exchange_adapter.submit_order.call_count == 2
    
    # Test max retries exceeded
    mock_exchange_adapter.submit_order.side_effect = Exception("Persistent error")
    sniper_executor.max_retries = 2
    
    with pytest.raises(Exception, match="Max retries.*exceeded"):
        await sniper_executor.execute(sample_order)
    
    assert mock_exchange_adapter.submit_order.call_count == 4  # 2 retries = 3 total attempts

@pytest.mark.asyncio
async def test_slice_size_calculation(sniper_executor, sample_order):
    """Test dynamic slice size calculation."""
    # Test with normal conditions
    normal_slice = sniper_executor._calculate_optimal_slice_size()
    assert normal_slice >= sniper_executor.min_slice_size
    assert normal_slice <= sniper_executor.max_slice_size
    
    # Test with high urgency
    original_urgency = sniper_executor.urgency
    sniper_executor.urgency = 1.0
    high_urgency_slice = sniper_executor._calculate_optimal_slice_size()
    assert high_urgency_slice >= normal_slice  # Should be more aggressive
    
    # Test with low remaining quantity
    sniper_executor.remaining_quantity = Decimal('0.5')
    small_qty_slice = sniper_executor._calculate_optimal_slice_size()
    assert small_qty_slice <= sniper_executor.remaining_quantity
    
    # Test with high volatility
    sniper_executor.volatility = Decimal('1.0')  # Very high
    sniper_executor.volatility_regime = 'HIGH'
    high_vol_slice = sniper_executor._calculate_optimal_slice_size()
    assert high_vol_slice < normal_slice  # Should be more conservative
    
    # Cleanup
    sniper_executor.urgency = original_urgency
    sniper_executor.remaining_quantity = sample_order.quantity

@pytest.mark.asyncio
async def test_market_impact_analysis(sniper_executor, sample_order, mock_exchange_adapter):
    """Test market impact analysis and order slicing."""
    # Configure deep order book
    mock_exchange_adapter.get_order_book.return_value = {
        'bids': [(Decimal(f'{100 - i*0.1}'), Decimal('1000')) for i in range(10)],
        'asks': [(Decimal(f'{100.1 + i*0.1}'), Decimal('1000')) for i in range(10)]
    }
    
    # Large order that would cause significant market impact
    large_order = sample_order.copy()
    large_order.quantity = Decimal('10000')  # Very large order
    
    # Mock order submission
    mock_exchange_adapter.submit_order.return_value = {
        'order_id': 'test_order_123_0',
        'status': 'FILLED',
        'filled_quantity': str(large_order.quantity),
        'filled_price': '100.2',
        'remaining_quantity': '0.0'
    }
    
    # Execute large order
    result = await sniper_executor.execute(large_order)
    
    # Verify order was sliced to minimize market impact
    assert result is not None
    assert result.quantity_executed == large_order.quantity
    
    # Should have used multiple slices to minimize impact
    assert len(result.fills) > 1
    
    # Verify average execution price is reasonable
    assert Decimal('100.0') < result.avg_execution_price < Decimal('101.0')
    
    # Verify participation rate is controlled
    assert result.metadata.get('participation_rate', 1.0) <= 0.5  # Should be below max participation
