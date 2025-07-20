"""
Comprehensive tests for OrderManager's OCO order functionality.

Tests cover:
- OCO order submission and management
- Order state synchronization
- Error handling and edge cases
- Performance characteristics
- Integration with position management
"""
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

import pytest
from prometheus_client import REGISTRY

# Make datadog import optional
try:
    from datadog import statsd
    HAS_DATADOG = True
except ImportError:
    HAS_DATADOG = False
    # Create a mock statsd module
    class MockStatsD:
        """Mock datadog statsd client."""
        def __init__(self, *args, **kwargs):
            pass
        
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    
    statsd = MockStatsD()

from core.config import settings
from core.exceptions import OrderError, ValidationError
from core.trading.order import (
    Order, OrderSide, OrderStatus, OrderType, TimeInForce, OrderUpdate
)
from core.trading.oco_order import OCOOrderConfig, OCOOrderStatus
from core.trading.order_manager import OrderManager, OrderGroup

# Reset metrics before each test
@pytest.fixture(autouse=True)
def reset_metrics():
    """Reset Prometheus metrics before each test."""
    for collector in list(REGISTRY._collector_to_names):
        REGISTRY.unregister(collector)
    
    # Re-register default collectors
    from prometheus_client import PROCESS_COLLECTOR, PLATFORM_COLLECTOR, GC_COLLECTOR
    REGISTRY.register(PROCESS_COLLECTOR)
    REGISTRY.register(PLATFORM_COLLECTOR)
    REGISTRY.register(GC_COLLECTOR)


@pytest.fixture
def mock_exchange():
    """Create a mock exchange adapter."""
    mock = AsyncMock()
    mock.submit_order.return_value = MagicMock(
        status=OrderStatus.NEW,
        exchange_order_id="EX123",
        timestamp=datetime.utcnow()
    )
    mock.cancel_order.return_value = True
    mock.get_order_updates.return_value = []
    return mock


@pytest.fixture
def mock_risk_manager():
    """Create a mock risk manager."""
    mock = AsyncMock()
    mock.validate_order.return_value = True
    return mock


@pytest.fixture
def mock_position_manager():
    """Create a mock position manager."""
    mock = AsyncMock()
    return mock


@pytest.fixture
async def order_manager(mock_exchange, mock_risk_manager, mock_position_manager):
    """Create an OrderManager instance for testing."""
    manager = OrderManager(
        exchange_adapter=mock_exchange,
        risk_manager=mock_risk_manager,
        position_manager=mock_position_manager
    )
    await manager.start()
    yield manager
    await manager.stop()


@pytest.fixture
def oco_config():
    """Create a sample OCO order configuration."""
    return OCOOrderConfig(
        symbol="BTC/USDT",
        quantity=Decimal("1.0"),
        limit_price=Decimal("52000.00"),
        stop_price=Decimal("48000.00"),
        stop_limit_price=Decimal("47900.00"),
        metadata={"strategy": "mean_reversion"}
    )


@pytest.mark.asyncio
async def test_submit_oco_order_success(order_manager, mock_exchange, oco_config):
    """Test successful OCO order submission."""
    # Setup
    mock_exchange.submit_order.side_effect = [
        MagicMock(status=OrderStatus.NEW, exchange_order_id="EX123"),
        MagicMock(status=OrderStatus.NEW, exchange_order_id="EX124")
    ]
    
    # Execute
    oco_order = await order_manager.submit_oco_order(oco_config)
    
    # Verify
    assert oco_order is not None
    assert oco_order.status == OCOOrderStatus.NEW
    assert len(oco_order.orders) == 2
    assert mock_exchange.submit_order.call_count == 2
    
    # Verify metrics
    metrics = order_manager.get_metrics()
    assert metrics['active_oco_orders'] == 1


@pytest.mark.asyncio
async def test_oco_order_fill_limit(order_manager, mock_exchange, oco_config):
    """Test OCO order behavior when limit order is filled."""
    # Submit OCO order
    oco_order = await order_manager.submit_oco_order(oco_config)
    
    # Get the limit order
    limit_order = next(
        o for o in oco_order.orders.values() 
        if o.metadata.get('oco_type') == 'LIMIT'
    )
    
    # Simulate limit order fill
    update = OrderUpdate(
        order_id=limit_order.id,
        status=OrderStatus.FILLED,
        filled_quantity=limit_order.quantity,
        remaining_quantity=Decimal("0"),
        avg_fill_price=limit_order.price,
        timestamp=datetime.utcnow()
    )
    
    # Process update
    await order_manager._handle_order_update(update)
    
    # Verify the stop order was canceled
    stop_order = next(
        o for o in oco_order.orders.values() 
        if o.metadata.get('oco_type') == 'STOP'
    )
    
    assert oco_order.status == OCOOrderStatus.FILLED
    assert stop_order.status == OrderStatus.CANCELED
    
    # Verify metrics
    metrics = order_manager.get_metrics()
    assert metrics['active_oco_orders'] == 0


@pytest.mark.asyncio
async def test_oco_order_fill_stop(order_manager, mock_exchange, oco_config):
    """Test OCO order behavior when stop order is filled."""
    # Submit OCO order
    oco_order = await order_manager.submit_oco_order(oco_config)
    
    # Get the stop order
    stop_order = next(
        o for o in oco_order.orders.values() 
        if o.metadata.get('oco_type') == 'STOP'
    )
    
    # Simulate stop order fill
    update = OrderUpdate(
        order_id=stop_order.id,
        status=OrderStatus.FILLED,
        filled_quantity=stop_order.quantity,
        remaining_quantity=Decimal("0"),
        avg_fill_price=stop_order.price,
        timestamp=datetime.utcnow()
    )
    
    # Process update
    await order_manager._handle_order_update(update)
    
    # Verify the limit order was canceled
    limit_order = next(
        o for o in oco_order.orders.values() 
        if o.metadata.get('oco_type') == 'LIMIT'
    )
    
    assert oco_order.status == OCOOrderStatus.FILLED
    assert limit_order.status == OrderStatus.CANCELED


@pytest.mark.asyncio
async def test_cancel_oco_order(order_manager, mock_exchange, oco_config):
    """Test OCO order cancellation."""
    # Submit OCO order
    oco_order = await order_manager.submit_oco_order(oco_config)
    
    # Cancel the OCO order
    result = await order_manager.cancel_oco_order(oco_order.client_order_id)
    
    # Verify
    assert result is True
    assert oco_order.status == OCOOrderStatus.CANCELED
    
    # Verify both child orders were canceled
    for order in oco_order.orders.values():
        assert order.status in (OrderStatus.CANCELED, OrderStatus.REJECTED)
    
    # Verify metrics
    metrics = order_manager.get_metrics()
    assert metrics['active_oco_orders'] == 0


@pytest.mark.asyncio
async def test_oco_order_error_handling(order_manager, mock_exchange, oco_config):
    """Test OCO order error handling."""
    # Make the second order submission fail
    mock_exchange.submit_order.side_effect = [
        MagicMock(status=OrderStatus.NEW, exchange_order_id="EX123"),
        Exception("Exchange error")
    ]
    
    # Submit OCO order (should raise)
    with pytest.raises(OrderError, match="Failed to submit OCO order"):
        await order_manager.submit_oco_order(oco_config)
    
    # Verify the first order was canceled
    assert mock_exchange.cancel_order.called
    assert mock_exchange.cancel_order.call_args[0][0] == "EX123"
    
    # Verify metrics
    metrics = order_manager.get_metrics()
    assert metrics['active_oco_orders'] == 0
    assert metrics['errors'] > 0


@pytest.mark.asyncio
async def test_oco_order_concurrent_updates(order_manager, mock_exchange, oco_config):
    """Test OCO order behavior with concurrent updates."""
    # Submit OCO order
    oco_order = await order_manager.submit_oco_order(oco_config)
    
    # Get the orders
    limit_order = next(o for o in oco_order.orders.values() 
                      if o.metadata.get('oco_type') == 'LIMIT')
    stop_order = next(o for o in oco_order.orders.values() 
                     if o.metadata.get('oco_type') == 'STOP')
    
    # Simulate concurrent fills (should never happen in reality)
    limit_update = OrderUpdate(
        order_id=limit_order.id,
        status=OrderStatus.FILLED,
        filled_quantity=limit_order.quantity,
        remaining_quantity=Decimal("0"),
        avg_fill_price=limit_order.price,
        timestamp=datetime.utcnow()
    )
    
    stop_update = OrderUpdate(
        order_id=stop_order.id,
        status=OrderStatus.FILLED,
        filled_quantity=stop_order.quantity,
        remaining_quantity=Decimal("0"),
        avg_fill_price=stop_order.price,
        timestamp=datetime.utcnow()
    )
    
    # Process updates concurrently
    await asyncio.gather(
        order_manager._handle_order_update(limit_update),
        order_manager._handle_order_update(stop_update)
    )
    
    # Verify only one order was filled
    filled_orders = [o for o in oco_order.orders.values() 
                    if o.status == OrderStatus.FILLED]
    assert len(filled_orders) == 1
    assert oco_order.status == OCOOrderStatus.FILLED


@pytest.mark.asyncio
async def test_oco_order_metrics(order_manager, mock_exchange, oco_config):
    """Test OCO order metrics collection."""
    # Submit OCO order
    oco_order = await order_manager.submit_oco_order(oco_config)
    
    # Verify metrics
    metrics = order_manager.get_metrics()
    assert metrics['active_oco_orders'] == 1
    
    # Cancel the order
    await order_manager.cancel_oco_order(oco_order.client_order_id)
    
    # Verify metrics updated
    metrics = order_manager.get_metrics()
    assert metrics['active_oco_orders'] == 0
    assert metrics['orders_processed'] > 0


@pytest.mark.asyncio
async def test_oco_order_position_updates(order_manager, mock_exchange, mock_position_manager, oco_config):
    """Test OCO order position updates."""
    # Submit OCO order
    oco_order = await order_manager.submit_oco_order(oco_config)
    
    # Get the limit order
    limit_order = next(o for o in oco_order.orders.values() 
                      if o.metadata.get('oco_type') == 'LIMIT')
    
    # Simulate fill
    update = OrderUpdate(
        order_id=limit_order.id,
        status=OrderStatus.FILLED,
        filled_quantity=limit_order.quantity,
        remaining_quantity=Decimal("0"),
        avg_fill_price=limit_order.price,
        timestamp=datetime.utcnow()
    )
    
    # Process update
    await order_manager._handle_order_update(update)
    
    # Verify position was updated
    mock_position_manager.update_position.assert_called_once()
    args, kwargs = mock_position_manager.update_position.call_args
    assert kwargs['symbol'] == oco_config.symbol
    assert kwargs['quantity'] == oco_config.quantity
    assert kwargs['price'] == oco_config.limit_price


@pytest.mark.asyncio
async def test_oco_order_listing(order_manager, oco_config):
    """Test listing OCO orders with filters."""
    # Submit multiple OCO orders
    oco_orders = []
    for i in range(3):
        config = oco_config.copy()
        config.symbol = f"BTC/USDT:{i}"
        oco_order = await order_manager.submit_oco_order(config)
        oco_orders.append(oco_order)
    
    # List all OCO orders
    all_orders = await order_manager.list_oco_orders()
    assert len(all_orders) == 3
    
    # Filter by symbol
    btc_orders = await order_manager.list_oco_orders(symbol="BTC/USDT:1")
    assert len(btc_orders) == 1
    assert btc_orders[0].config.symbol == "BTC/USDT:1"
    
    # Filter by status
    active_orders = await order_manager.list_oco_orders(status=OCOOrderStatus.NEW)
    assert len(active_orders) == 3
    
    # Cancel an order and verify filtering
    await order_manager.cancel_oco_order(oco_orders[0].client_order_id)
    
    active_orders = await order_manager.list_oco_orders(status=OCOOrderStatus.NEW)
    assert len(active_orders) == 2
    
    canceled_orders = await order_manager.list_oco_orders(status=OCOOrderStatus.CANCELED)
    assert len(canceled_orders) == 1


class TestOrderManagerOCO(IsolatedAsyncioTestCase):
    """Test suite for OCO order functionality in OrderManager."""

    async def asyncSetUp(self):
        """Set up test fixtures."""
        self.exchange = MockExchangeAdapter()
        self.manager = await create_test_order_manager(self.exchange)
        
    async def asyncTearDown(self):
        """Clean up test fixtures."""
        await self.manager.stop()
        
    async def test_submit_oco_order_success(self):
        """Test successful OCO order submission."""
        config = create_test_oco_config()
        oco_order = await self.manager.submit_oco_order(config)
        
        self.assertEqual(oco_order.status, OCOOrderStatus.ACTIVE)
        self.assertIn(oco_order.oco_id, self.manager.oco_orders)
        self.assertEqual(len(oco_order.orders), 2)
        
        # Verify both orders were submitted
        self.exchange.submit_order.assert_awaited()
        self.assertEqual(self.exchange.submit_order.await_count, 2)
        
    async def test_oco_order_completion_on_fill(self):
        """Test OCO order completion when one order is filled."""
        config = create_test_oco_config()
        oco_order = await self.manager.submit_oco_order(config)
        
        # Simulate one order being filled
        filled_order = oco_order.orders[0]
        filled_order.status = OrderStatus.FILLED
        
        # Process the order update
        await self.manager.on_order_update({
            'order_id': filled_order.order_id,
            'status': OrderStatus.FILLED,
            'filled_quantity': str(filled_order.quantity),
            'avg_fill_price': str(filled_order.price)
        })
        
        # The other order should be canceled
        self.exchange.cancel_order.assert_awaited_once()
        self.assertEqual(oco_order.status, OCOOrderStatus.COMPLETED)
        
    async def test_oco_order_cancellation(self):
        """Test OCO order cancellation."""
        config = create_test_oco_config()
        oco_order = await self.manager.submit_oco_order(config)
        
        # Cancel the OCO order
        result = await self.manager.cancel_oco_order(oco_order.oco_id)
        
        self.assertTrue(result)
        self.assertEqual(oco_order.status, OCOOrderStatus.CANCELED)
        # Both orders should be canceled
        self.assertEqual(self.exchange.cancel_order.await_count, 2)
        
    async def test_oco_order_status_updates(self):
        """Test OCO order status updates."""
        config = create_test_oco_config()
        oco_order = await self.manager.submit_oco_order(config)
        
        # Verify initial status
        self.assertEqual(oco_order.status, OCOOrderStatus.ACTIVE)
        
        # Simulate partial fill of one order
        order = oco_order.orders[0]
        await self.manager.on_order_update({
            'order_id': order.order_id,
            'status': OrderStatus.PARTIALLY_FILLED,
            'filled_quantity': str(order.quantity / 2),
            'avg_fill_price': str(order.price)
        })
        
        # Status should still be active
        self.assertEqual(oco_order.status, OCOOrderStatus.ACTIVE)
        
    async def test_oco_order_error_handling(self):
        """Test error handling in OCO order processing."""
        # Test with invalid configuration
        with self.assertRaises(ValueError):
            config = create_test_oco_config(quantity=Decimal("0"))  # Invalid quantity
            await self.manager.submit_oco_order(config)
            
        # Test canceling non-existent OCO order
        result = await self.manager.cancel_oco_order("nonexistent_oco")
        self.assertFalse(result)
        
    async def test_oco_order_with_trailing_stop(self):
        """Test OCO order with trailing stop functionality."""
        config = create_test_oco_config()
        config.trailing_stop = True
        config.trailing_distance = Decimal("100.00")  # $100 trailing distance
        
        oco_order = await self.manager.submit_oco_order(config)
        
        # Verify trailing stop order was created
        self.assertEqual(len(oco_order.orders), 3)  # Entry, stop loss, take profit
        self.assertTrue(any(o.order_type == OrderType.TRAILING_STOP for o in oco_order.orders))
        
        # Simulate price movement and verify trailing stop updates
        await self.manager.on_market_data({
            'symbol': config.symbol,
            'bid': Decimal("50100.00"),
            'ask': Decimal("50100.00")
        })
        
        # Verify trailing stop was updated
        trailing_stop = next(o for o in oco_order.orders 
                           if o.order_type == OrderType.TRAILING_STOP)
        self.assertEqual(trailing_stop.stop_price, Decimal("50000.00"))  # 50100 - 100
