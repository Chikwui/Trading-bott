"""
Integration tests for Sniper execution algorithm with real exchange adapters.

Note: These tests require valid API credentials and may execute real trades on connected exchanges.
Set environment variables for API keys and use testnet/sandbox environments when possible.
"""
import os
import asyncio
import pytest
import time
from decimal import Decimal
from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock

import pandas as pd
import numpy as np

from core.trading.order_types import (
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce
)
from core.trading.algorithms.sniper import SniperExecutor
from core.adapters.exchanges import ExchangeFactory  # Assuming this exists

# Mark all tests in this module as integration tests
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not os.getenv("RUN_INTEGRATION_TESTS"),
        reason="Skipping integration tests (set RUN_INTEGRATION_TESTS=1 to run)"
    )
]

# Exchange-specific test configurations
EXCHANGE_CONFIGS = {
    'binance': {
        'symbol': 'BTC/USDT',
        'test_symbol': 'BTC/USDT',
        'min_order_qty': 0.001,
        'price_tick': 0.1,
        'test_mode': True  # Use testnet/sandbox when available
    },
    'ftx': {
        'symbol': 'BTC-PERP',
        'test_symbol': 'BTC-PERP',
        'min_order_qty': 0.001,
        'price_tick': 0.1,
        'test_mode': True
    },
    # Add other exchanges as needed
}

@pytest.fixture(scope="module")
def event_loop():
    """Create an instance of the default event loop for the test module."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(params=["binance", "ftx"])
async def exchange_adapter(request):
    """Create and initialize exchange adapter with test configuration."""
    exchange_id = request.param
    config = EXCHANGE_CONFIGS[exchange_id].copy()
    
    # Initialize exchange adapter
    exchange = ExchangeFactory.create(
        exchange_id=exchange_id,
        api_key=os.getenv(f"{exchange_id.upper()}_API_KEY"),
        api_secret=os.getenv(f"{exchange_id.upper()}_API_SECRET"),
        testnet=config.pop('test_mode', True)
    )
    
    # Initialize connection
    await exchange.initialize()
    
    # Verify connectivity
    try:
        await exchange.get_market_data(config['test_symbol'])
    except Exception as e:
        pytest.skip(f"Failed to connect to {exchange_id}: {str(e)}")
    
    yield exchange, config
    
    # Cleanup
    await exchange.close()

@pytest.fixture
async def sniper_executor(exchange_adapter):
    """Create a SniperExecutor instance with real exchange adapter."""
    exchange, config = exchange_adapter
    
    executor = SniperExecutor(
        exchange_adapter=exchange,
        position_manager=MagicMock(),  # Mock position manager for now
        config={
            'max_slippage': '10',  # 10 bps
            'urgency': '0.7',
            'max_participation': '0.2',
            'min_slice_size': '0.001',
            'max_slice_size': '1.0',
            'refresh_interval': '0.5',
            'max_retries': '3',
            'dark_pool_enabled': 'False',  # Disable dark pool for testing
            'anti_gaming': 'True',
            'volatility_adaptive': 'True'
        }
    )
    
    return executor, config

@pytest.mark.asyncio
async def test_sniper_market_order_execution(sniper_executor):
    """Test executing a small market order using the Sniper algorithm."""
    executor, config = sniper_executor
    symbol = config['test_symbol']
    min_order_qty = config['min_order_qty']
    
    # Create a small market order (0.1x minimum to minimize impact)
    test_qty = min_order_qty * 0.1
    order = Order(
        order_id=f"test_mkt_{int(time.time())}",
        client_order_id=f"client_mkt_{int(time.time())}",
        symbol=symbol,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal(str(test_qty)),
        time_in_force=TimeInForce.IOC,
        status=OrderStatus.NEW,
        timestamp=datetime.now(timezone.utc)
    )
    
    # Execute order
    result = await executor.execute(order)
    
    # Verify execution
    assert result is not None
    assert result.quantity_executed > 0
    assert result.avg_execution_price > 0
    assert result.status == OrderStatus.FILLED
    assert result.fills and len(result.fills) > 0
    
    # Verify execution metrics
    assert 0 <= result.implementation_shortfall_bps < 100  # Should be reasonable
    assert result.execution_end_time > result.execution_start_time
    assert result.metadata.get('participation_rate') <= 0.2  # Within max participation

@pytest.mark.asyncio
async def test_sniper_limit_order_execution(sniper_executor):
    """Test executing a limit order using the Sniper algorithm."""
    executor, config = sniper_executor
    symbol = config['test_symbol']
    min_order_qty = config['min_order_qty']
    price_tick = config['price_tick']
    
    # Get current market data
    market_data = await executor.exchange_adapter.get_market_data(symbol)
    best_bid = Decimal(str(market_data['best_bid']))
    
    # Create a limit order below current price (shouldn't execute)
    test_qty = min_order_qty * 0.1
    limit_price = best_bid * Decimal('0.9')  # 10% below current price
    limit_price = (limit_price // Decimal(str(price_tick))) * Decimal(str(price_tick))  # Round to tick size
    
    order = Order(
        order_id=f"test_lmt_{int(time.time())}",
        client_order_id=f"client_lmt_{int(time.time())}",
        symbol=symbol,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal(str(test_qty)),
        price=limit_price,
        time_in_force=TimeInForce.GTC,
        status=OrderStatus.NEW,
        timestamp=datetime.now(timezone.utc)
    )
    
    # Execute with timeout (should not fill)
    with patch('core.trading.algorithms.sniper.SniperExecutor._monitor_execution') as mock_monitor:
        mock_monitor.side_effect = asyncio.TimeoutError("Test timeout")
        with pytest.raises(asyncio.TimeoutError):
            await executor.execute(order, timeout=2.0)
    
    # Verify order was cancelled
    assert executor.cancelled is True
    
    # Check order status (should be CANCELLED or PARTIALLY_FILLED)
    assert order.status in [OrderStatus.CANCELLED, OrderStatus.PARTIALLY_FILLED]
    
    # If partially filled, verify metrics
    if order.status == OrderStatus.PARTIALLY_FILLED:
        assert order.quantity_executed > 0
        assert order.avg_execution_price >= limit_price  # Should get price >= limit price for buy

@pytest.mark.asyncio
async def test_sniper_volatility_adaptation(sniper_executor):
    """Test that the Sniper algorithm adapts to changing volatility."""
    executor, config = sniper_executor
    symbol = config['test_symbol']
    
    # Get historical data to simulate volatility
    end_time = datetime.now(timezone.utc)
    start_time = end_time - timedelta(hours=24)
    
    # Get historical trades
    trades = await executor.exchange_adapter.get_historical_trades(
        symbol=symbol,
        start_time=start_time,
        end_time=end_time,
        limit=1000
    )
    
    if not trades:
        pytest.skip("Insufficient historical data for volatility test")
    
    # Calculate realized volatility
    df = pd.DataFrame(trades)
    df['returns'] = df['price'].pct_change()
    realized_vol = df['returns'].std() * np.sqrt(365 * 24)  # Annualized
    
    # Update market state with historical data
    executor.volatility = Decimal(str(realized_vol))
    
    # Check if volatility regime is detected correctly
    if realized_vol > 0.5:  # High volatility threshold
        assert executor.volatility_regime == 'HIGH'
        assert executor.urgency < 0.7  # Should be reduced from default 0.7
    else:
        assert executor.volatility_regime in ['LOW', 'NORMAL']
    
    # Test slice size adaptation
    original_slice = executor._calculate_optimal_slice_size()
    
    # Increase volatility and verify slice size decreases
    executor.volatility = Decimal('1.0')  # Very high volatility
    executor.volatility_regime = 'HIGH'
    new_slice = executor._calculate_optimal_slice_size()
    assert new_slice < original_slice

@pytest.mark.asyncio
async def test_sniper_large_order_execution(sniper_executor):
    """Test executing a large order that should be sliced into multiple child orders."""
    executor, config = sniper_executor
    symbol = config['test_symbol']
    min_order_qty = config['min_order_qty']
    
    # Get current market data
    market_data = await executor.exchange_adapter.get_market_data(symbol)
    best_ask = Decimal(str(market_data['best_ask']))
    
    # Create a large order (10x minimum to trigger slicing)
    test_qty = min_order_qty * 10
    
    # Calculate order value and ensure it's within test account limits
    order_value = test_qty * best_ask
    if order_value > 1000:  # $1000 max for testing
        pytest.skip("Order value too large for test account")
    
    order = Order(
        order_id=f"test_lrg_{int(time.time())}",
        client_order_id=f"client_lrg_{int(time.time())}",
        symbol=symbol,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal(str(test_qty)),
        time_in_force=TimeInForce.IOC,
        status=OrderStatus.NEW,
        timestamp=datetime.now(timezone.utc)
    )
    
    # Execute order with monitoring
    start_time = time.time()
    result = await executor.execute(order, timeout=30.0)
    execution_time = time.time() - start_time
    
    # Verify execution
    assert result is not None
    assert result.quantity_executed > 0
    assert result.avg_execution_price > 0
    
    # Verify order was sliced (multiple fills)
    assert len(result.fills) > 1
    
    # Verify execution metrics
    assert execution_time < 30.0  # Should complete within timeout
    assert result.implementation_shortfall_bps < 50  # Should be reasonable for market order
    
    # Verify participation rate was controlled
    participation_rate = result.metadata.get('participation_rate', 0)
    assert 0 < participation_rate <= 0.2  # Within max participation
    
    # Verify slice sizes were appropriate
    slice_sizes = [float(fill['quantity']) for fill in result.fills]
    assert all(slice_sizes[i] <= slice_sizes[i+1] * 1.5 for i in range(len(slice_sizes)-1))  # Sizes should be similar

@pytest.mark.asyncio
async def test_sniper_error_handling(sniper_executor):
    """Test error handling and recovery during order execution."""
    executor, config = sniper_executor
    
    # Create an invalid order (missing required fields)
    invalid_order = Order(
        order_id="",
        client_order_id="",
        symbol="",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("0"),
        time_in_force=TimeInForce.IOC,
        status=OrderStatus.NEW,
        timestamp=datetime.now(timezone.utc)
    )
    
    # Should raise validation error
    with pytest.raises(ValueError):
        await executor.execute(invalid_order)
    
    # Test network error recovery
    symbol = config['test_symbol']
    test_qty = config['min_order_qty'] * 0.1
    
    order = Order(
        order_id=f"test_err_{int(time.time())}",
        client_order_id=f"client_err_{int(time.time())}",
        symbol=symbol,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal(str(test_qty)),
        time_in_force=TimeInForce.IOC,
        status=OrderStatus.NEW,
        timestamp=datetime.now(timezone.utc)
    )
    
    # Simulate temporary network error
    original_submit = executor.exchange_adapter.submit_order
    
    async def mock_submit_with_error(*args, **kwargs):
        # First call fails, subsequent calls succeed
        if not hasattr(mock_submit_with_error, 'retry_count'):
            mock_submit_with_error.retry_count = 1
            raise ConnectionError("Temporary network error")
        return await original_submit(*args, **kwargs)
    
    with patch.object(executor.exchange_adapter, 'submit_order', new=mock_submit_with_error):
        result = await executor.execute(order)
        assert result is not None
        assert result.quantity_executed > 0
        assert mock_submit_with_error.retry_count == 1  # Verify retry happened
