"""
Test suite for the TWAP (Time-Weighted Average Price) execution strategy.
"""
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import patch, MagicMock, AsyncMock
import pytest

from core.execution.strategies.twap import TWAPExecution, TWAPConfig, OrderRequest, OrderSide, OrderStatus

@pytest.fixture
def twap_config():
    """Default TWAP configuration for testing."""
    return {
        'interval_seconds': 60,  # 1 minute between slices
        'slice_ratio': 0.2,     # 20% per slice
        'max_slices': 5,        # Max 5 slices
        'price_improvement': 0.001,  # 0.1% price improvement
        'min_order_size': 0.001,
        'time_in_force': 'DAY'
    }

@pytest.fixture
def order_request():
    """Sample order request for testing."""
    return OrderRequest(
        symbol='BTC/USD',
        side=OrderSide.BUY,
        quantity=Decimal('1.0'),
        order_type='LIMIT',
        price=Decimal('50000.0'),
        time_in_force='DAY'
    )

@pytest.mark.asyncio
async def test_twap_initialization(twap_config):
    """Test TWAP initialization with default config."""
    twap = TWAPExecution(twap_config)
    assert twap is not None
    assert twap.config.interval_seconds == 60
    assert twap.config.slice_ratio == 0.2
    assert twap.config.max_slices == 5

@pytest.mark.asyncio
async def test_twap_execution_basic(twap_config, order_request):
    """Test basic TWAP execution flow."""
    twap = TWAPExecution(twap_config)
    
    # Mock the execution methods
    with patch.object(twap, '_execute_slice') as mock_execute:
        # Start execution
        response = await twap.execute(order_request)
        
        # Check initial response
        assert response.status == OrderStatus.NEW
        assert 'twap_' in response.order_id
        
        # Let the event loop run for a bit
        await asyncio.sleep(0.1)
        
        # Check that slices were scheduled
        assert mock_execute.call_count > 0
        
        # Check status
        status = twap.get_status()
        assert status['is_running'] is True
        assert status['total_quantity'] == 1.0
        assert status['total_slices'] == 5
        assert status['current_slice'] > 0

@pytest.mark.asyncio
async def test_twap_cancel(twap_config, order_request):
    """Test canceling TWAP execution."""
    twap = TWAPExecution(twap_config)
    
    # Start execution
    await twap.execute(order_request)
    
    # Cancel after a short delay
    await asyncio.sleep(0.1)
    result = await twap.cancel()
    
    # Check cancellation
    assert result is True
    status = twap.get_status()
    assert status['is_running'] is False

@pytest.mark.asyncio
async def test_twap_end_time(twap_config, order_request):
    """Test TWAP with explicit end time."""
    config = twap_config.copy()
    config['start_time'] = datetime.utcnow()
    config['end_time'] = config['start_time'] + timedelta(minutes=10)  # 10-minute window
    config['interval_seconds'] = 60  # 1-minute intervals
    
    twap = TWAPExecution(config)
    
    # Mock the execution methods
    with patch.object(twap, '_execute_slice') as mock_execute:
        # Start execution
        await twap.execute(order_request)
        
        # Let the event loop run for a bit
        await asyncio.sleep(0.1)
        
        # Check that slices were scheduled
        assert mock_execute.call_count > 0
        
        # Check that the number of slices matches the time window
        status = twap.get_status()
        assert status['total_slices'] == 10  # 10 minutes / 1 minute interval

@pytest.mark.asyncio
async def test_twap_min_order_size(twap_config, order_request):
    """Test TWAP with minimum order size constraint."""
    config = twap_config.copy()
    config['min_order_size'] = 0.3  # Minimum 0.3 per slice
    
    # Small order that would require slices smaller than min_order_size
    small_order = order_request.copy()
    small_order.quantity = Decimal('1.0')
    
    twap = TWAPExecution(config)
    
    # Start execution
    await twap.execute(small_order)
    await asyncio.sleep(0.1)
    
    # Check that the number of slices was adjusted for min_order_size
    status = twap.get_status()
    assert status['total_slices'] == 4  # 1.0 / 0.3 = 3.33 -> 4 slices

@pytest.mark.asyncio
async def test_twap_price_improvement(twap_config, order_request):
    """Test TWAP price improvement calculation."""
    twap = TWAPExecution(twap_config)
    
    # Set up test data
    twap._symbol = 'BTC/USD'
    twap._side = OrderSide.BUY
    
    # Mock the market price
    with patch.object(twap, '_get_current_market_price', return_value=Decimal('50000.0')):
        target_price = twap._calculate_target_price()
        expected_price = Decimal('50000.0') * (1 - Decimal('0.001'))  # 0.1% improvement
        assert abs(float(target_price - expected_price)) < 0.000001  # Allow for floating point errors

@pytest.mark.asyncio
async def test_twap_concurrent_executions(twap_config, order_request):
    """Test that multiple TWAP executions don't interfere with each other."""
    twap1 = TWAPExecution(twap_config)
    twap2 = TWAPExecution(twap_config)
    
    # Mock the execution methods
    with patch.object(twap1, '_execute_slice') as mock_execute1, \
         patch.object(twap2, '_execute_slice') as mock_execute2:
        
        # Start both executions
        await twap1.execute(order_request)
        await twap2.execute(order_request)
        
        # Let the event loop run for a bit
        await asyncio.sleep(0.1)
        
        # Check that both executions are running
        assert twap1.get_status()['is_running'] is True
        assert twap2.get_status()['is_running'] is True
        
        # Check that both executions made progress
        assert mock_execute1.call_count > 0
        assert mock_execute2.call_count > 0

@pytest.mark.asyncio
async def test_twap_error_handling(twap_config, order_request):
    """Test TWAP error handling during execution."""
    twap = TWAPExecution(twap_config)
    
    # Simulate an error in _execute_slice
    with patch.object(twap, '_execute_slice', side_effect=Exception("Test error")) as mock_execute:
        # Start execution
        await twap.execute(order_request)
        
        # Let the event loop run for a bit
        await asyncio.sleep(0.1)
        
        # Check that the error was handled and execution continues
        assert mock_execute.call_count > 0
        status = twap.get_status()
        assert status['is_running'] is True

if __name__ == "__main__":
    pytest.main(["-v", "test_twap_strategy.py"])
