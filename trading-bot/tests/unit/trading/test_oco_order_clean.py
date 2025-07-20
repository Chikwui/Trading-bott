"""
Comprehensive unit tests for OCO (One-Cancels-Other) order implementation.
"""
import asyncio
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Dict, Optional
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from core.trading.oco_order import (
    OCOOrder,
    OCOOrderConfig,
    OCOOrderStatus,
    OrderError,
    OrderValidationError,
    RiskCheckFailed
)
from core.trading.order import Order, OrderStatus, OrderType, OrderSide, TimeInForce


class MockPosition:
    """Mock position for testing."""
    
    def __init__(self, symbol: str, size: Decimal = Decimal('0')):
        self.symbol = symbol
        self.size = size


class MockOrderManager:
    """Mock order manager for testing OCO orders with enhanced functionality."""
    
    def __init__(self):
        self.submit_order = AsyncMock()
        self.cancel_order = AsyncMock()
        self.orders: Dict[str, Order] = {}
        self.positions: Dict[str, MockPosition] = {}
        self.market_prices: Dict[str, Decimal] = {}
        
        # Setup default mock behaviors
        self.submit_order.side_effect = self._submit_order_impl
        self.cancel_order.side_effect = self._cancel_order_impl
        self.get_position = self._get_position_impl
        self.get_market_price = self._get_market_price_impl
        self.get_max_position_size = MagicMock(return_value=Decimal('10.0'))
    
    async def _submit_order_impl(self, order: Order) -> Order:
        """Mock order submission implementation."""
        order.status = OrderStatus.NEW
        self.orders[order.client_order_id] = order
        return order
    
    async def _cancel_order_impl(self, order_id: str) -> bool:
        """Mock order cancellation implementation."""
        if order_id in self.orders:
            self.orders[order_id].status = OrderStatus.CANCELED
            return True
        return False
    
    def _get_position_impl(self, symbol: str) -> Optional[MockPosition]:
        """Mock get position implementation."""
        return self.positions.get(symbol)
    
    def _get_market_price_impl(self, symbol: str) -> Optional[Decimal]:
        """Mock get market price implementation."""
        return self.market_prices.get(symbol)
    
    def set_position(self, symbol: str, size: Decimal) -> None:
        """Set a mock position."""
        self.positions[symbol] = MockPosition(symbol, size)
    
    def set_market_price(self, symbol: str, price: Decimal) -> None:
        """Set a mock market price."""
        self.market_prices[symbol] = price


@pytest.fixture
def order_manager():
    """Create a mock order manager for testing."""
    return MockOrderManager()


@pytest.fixture
def oco_config():
    """Create a default OCO order configuration."""
    return OCOOrderConfig(
        symbol="BTC/USDT",
        quantity=Decimal("1.0"),
        limit_price=Decimal("52000.00"),
        stop_price=Decimal("48000.00"),
        stop_limit_price=Decimal("47900.00"),
        time_in_force=TimeInForce.GTC,
        expire_time=datetime.now(timezone.utc) + timedelta(days=30)
    )


@pytest.fixture
def oco_order(order_manager, oco_config):
    """Create an OCO order for testing."""
    return OCOOrder(order_manager, oco_config)


@pytest.mark.asyncio
async def test_oco_order_initialization(oco_order, oco_config):
    """Test OCO order initialization."""
    assert oco_order.status == OCOOrderStatus.NEW
    assert oco_order.client_order_id.startswith("oco_")
    assert oco_order.config == oco_config


@pytest.mark.asyncio
async def test_oco_order_submission(oco_order, order_manager):
    """Test OCO order submission."""
    await oco_order.submit()
    
    # Verify both orders were submitted
    assert len(order_manager.orders) == 2
    assert any("_LIMIT" in oid for oid in order_manager.orders)
    assert any("_STOP" in oid for oid in order_manager.orders)
    
    # Verify order properties
    limit_order = next(o for o in order_manager.orders.values() if o.order_type == OrderType.LIMIT)
    stop_order = next(o for o in order_manager.orders.values() if o.order_type in (OrderType.STOP, OrderType.STOP_LIMIT))
    
    assert limit_order.symbol == "BTC/USDT"
    assert stop_order.symbol == "BTC/USDT"
    assert limit_order.quantity == Decimal("1.0")
    assert stop_order.quantity == Decimal("1.0")
    assert limit_order.price == Decimal("52000.00")
    assert stop_order.stop_price == Decimal("48000.00")
    assert stop_order.price == Decimal("47900.00")  # stop_limit_price


@pytest.mark.asyncio
async def test_oco_order_fill_limit(oco_order, order_manager):
    """Test OCO order behavior when limit order is filled."""
    await oco_order.submit()
    
    # Get the limit order
    limit_order = next(o for o in order_manager.orders.values() if o.order_type == OrderType.LIMIT)
    
    # Simulate limit order fill
    limit_order.status = OrderStatus.FILLED
    await oco_order.handle_order_update(limit_order)
    
    # Verify the stop order was canceled
    stop_order = next(o for o in order_manager.orders.values() 
                     if o.order_type in (OrderType.STOP, OrderType.STOP_LIMIT))
    assert stop_order.status == OrderStatus.CANCELED
    assert oco_order.status == OCOOrderStatus.FILLED


@pytest.mark.asyncio
async def test_oco_order_fill_stop(oco_order, order_manager):
    """Test OCO order behavior when stop order is filled."""
    await oco_order.submit()
    
    # Get the stop order
    stop_order = next(o for o in order_manager.orders.values() 
                     if o.order_type in (OrderType.STOP, OrderType.STOP_LIMIT))
    
    # Simulate stop order fill
    stop_order.status = OrderStatus.FILLED
    await oco_order.handle_order_update(stop_order)
    
    # Verify the limit order was canceled
    limit_order = next(o for o in order_manager.orders.values() 
                      if o.order_type == OrderType.LIMIT)
    assert limit_order.status == OrderStatus.CANCELED
    assert oco_order.status == OCOOrderStatus.FILLED


@pytest.mark.asyncio
async def test_oco_order_cancel(oco_order, order_manager):
    """Test OCO order cancellation."""
    await oco_order.submit()
    await oco_order.cancel()
    
    # Verify both orders were canceled
    assert all(o.status == OrderStatus.CANCELED for o in order_manager.orders.values())
    assert oco_order.status == OCOOrderStatus.CANCELED


@pytest.mark.asyncio
async def test_oco_order_rejection(oco_order, order_manager):
    """Test OCO order rejection handling."""
    # Make submit_order raise an error
    order_manager.submit_order.side_effect = Exception("Connection error")
    
    with pytest.raises(Exception, match="Failed to submit OCO order"):
        await oco_order.submit()
    
    assert oco_order.status == OCOOrderStatus.REJECTED


@pytest.mark.asyncio
async def test_oco_order_partial_fill(oco_order, order_manager):
    """Test OCO order with partial fills."""
    await oco_order.submit()
    
    # Get the limit order
    limit_order = next(o for o in order_manager.orders.values() 
                      if o.order_type == OrderType.LIMIT)
    
    # Simulate partial fill
    limit_order.filled_quantity = Decimal("0.5")
    limit_order.status = OrderStatus.PARTIALLY_FILLED
    await oco_order.handle_order_update(limit_order)
    
    # Verify the stop order quantity was updated
    stop_order = next(o for o in order_manager.orders.values() 
                     if o.order_type in (OrderType.STOP, OrderType.STOP_LIMIT))
    assert stop_order.quantity == Decimal("0.5")


@pytest.mark.asyncio
async def test_oco_order_callbacks(oco_order, order_manager):
    """Test OCO order callbacks."""
    # Setup callbacks
    on_filled = AsyncMock()
    on_canceled = AsyncMock()
    on_rejected = AsyncMock()
    
    oco_order.on_filled = on_filled
    oco_order.on_canceled = on_canceled
    oco_order.on_rejected = on_rejected
    
    # Test filled callback
    await oco_order.submit()
    limit_order = next(o for o in order_manager.orders.values() 
                      if o.order_type == OrderType.LIMIT)
    limit_order.status = OrderStatus.FILLED
    await oco_order.handle_order_update(limit_order)
    
    on_filled.assert_called_once()
    assert not on_canceled.called
    assert not on_rejected.called
    
    # Reset mocks
    on_filled.reset_mock()
    
    # Test canceled callback
    oco_order.status = OCOOrderStatus.ACTIVE
    await oco_order.cancel()
    on_canceled.assert_called_once()
    
    # Test rejected callback
    oco_order.status = OCOOrderStatus.NEW
    order_manager.submit_order.side_effect = Exception("Error")
    with pytest.raises(Exception):
        await oco_order.submit()
    on_rejected.assert_called_once()


@pytest.mark.asyncio
async def test_oco_order_concurrent_updates(oco_order, order_manager):
    """Test OCO order behavior with concurrent updates."""
    await oco_order.submit()
    
    # Get both orders
    limit_order = next(o for o in order_manager.orders.values() 
                      if o.order_type == OrderType.LIMIT)
    stop_order = next(o for o in order_manager.orders.values() 
                     if o.order_type in (OrderType.STOP, OrderType.STOP_LIMIT))
    
    # Simulate concurrent fills (should only process one)
    limit_order.status = OrderStatus.FILLED
    stop_order.status = OrderStatus.FILLED
    
    # Process updates concurrently
    await asyncio.gather(
        oco_order.handle_order_update(limit_order),
        oco_order.handle_order_update(stop_order)
    )
    
    # Verify only one order was processed
    assert oco_order.status == OCOOrderStatus.FILLED
    canceled_count = sum(1 for o in order_manager.orders.values() 
                        if o.status == OrderStatus.CANCELED)
    assert canceled_count == 1


@pytest.mark.asyncio
async def test_oco_order_modification(oco_order, order_manager, oco_config):
    """Test OCO order modification."""
    await oco_order.submit()
    
    # Modify the order
    new_config = OCOOrderConfig(
        symbol="BTC/USDT",
        quantity=Decimal("2.0"),
        limit_price=Decimal("53000.00"),
        stop_price=Decimal("47000.00"),
        stop_limit_price=Decimal("46900.00"),
        time_in_force=TimeInForce.GTC,
        expire_time=datetime.now(timezone.utc) + timedelta(days=30)
    )
    
    await oco_order.modify(new_config)
    
    # Verify orders were canceled and new ones created
    assert len(order_manager.orders) == 4  # 2 original + 2 new
    
    # Verify new orders were submitted with updated parameters
    new_limit_order = next(o for o in order_manager.orders.values() 
                          if o.order_type == OrderType.LIMIT and o.quantity == Decimal("2.0"))
    new_stop_order = next(o for o in order_manager.orders.values() 
                         if o.order_type in (OrderType.STOP, OrderType.STOP_LIMIT) 
                         and o.quantity == Decimal("2.0"))
    
    assert new_limit_order.price == Decimal("53000.00")
    assert new_stop_order.stop_price == Decimal("47000.00")
    assert new_stop_order.price == Decimal("46900.00")


@pytest.mark.asyncio
async def test_oco_order_modification_validation(oco_order, order_manager):
    """Test OCO order modification validation."""
    await oco_order.submit()
    
    # Test invalid price
    with pytest.raises(ValueError):
        invalid_config = oco_order.config.copy()
        invalid_config.limit_price = Decimal("-100.00")
        await oco_order.modify(invalid_config)
    
    # Test invalid quantity
    with pytest.raises(ValueError):
        invalid_config = oco_order.config.copy()
        invalid_config.quantity = Decimal("-1.0")
        await oco_order.modify(invalid_config)
    
    # Test invalid symbol
    with pytest.raises(ValueError):
        invalid_config = oco_order.config.copy()
        invalid_config.symbol = ""
        await oco_order.modify(invalid_config)


@pytest.mark.asyncio
async def test_oco_order_risk_checks(oco_order, order_manager):
    """Test OCO order risk checks."""
    # Setup market price
    order_manager.set_market_price("BTC/USDT", Decimal("50000.00"))
    
    # Test position size limit
    order_manager.get_max_position_size.return_value = Decimal("0.5")
    
    with pytest.raises(RiskCheckFailed):
        await oco_order.submit()
    
    # Reset and test price deviation
    order_manager.get_max_position_size.return_value = Decimal("10.0")
    
    # Test price too far from market
    oco_order.config.limit_price = Decimal("100000.00")  # 100% above market
    with pytest.raises(RiskCheckFailed):
        await oco_order.submit()
    
    # Test stop price too close to market
    oco_order.config.limit_price = Decimal("51000.00")
    oco_order.config.stop_price = Decimal("49500.00")  # 1% below market
    with pytest.raises(RiskCheckFailed):
        await oco_order.submit()


@pytest.mark.asyncio
async def test_oco_order_metrics(oco_order, order_manager):
    """Test OCO order metrics collection."""
    # Import metrics module to reset counters
    from core.trading import metrics
    
    # Reset metrics
    metrics.ORDERS_SUBMITTED.labels("OCO")._value.set(0)
    metrics.ORDERS_FILLED.labels("OCO")._value.set(0)
    metrics.ORDER_LATENCY.labels("OCO")._value._sum = 0
    metrics.ORDER_LATENCY.labels("OCO")._value._count = 0
    
    # Submit and fill order
    await oco_order.submit()
    limit_order = next(o for o in order_manager.orders.values() 
                      if o.order_type == OrderType.LIMIT)
    limit_order.status = OrderStatus.FILLED
    await oco_order.handle_order_update(limit_order)
    
    # Verify metrics
    assert metrics.ORDERS_SUBMITTED.labels("OCO")._value.get() == 1
    assert metrics.ORDERS_FILLED.labels("OCO")._value.get() == 1
    assert metrics.ORDER_LATENCY.labels("OCO")._value._count == 1


@pytest.mark.asyncio
async def test_oco_order_edge_cases(oco_order, order_manager):
    """Test OCO order edge cases."""
    # Test duplicate order submission
    await oco_order.submit()
    with pytest.raises(OrderError, match="Cannot submit OCO order in"):
        await oco_order.submit()
    
    # Test cancel already canceled order
    await oco_order.cancel()
    await oco_order.cancel()  # Should not raise
    
    # Test invalid order update
    invalid_order = Order(
        client_order_id="invalid",
        symbol="BTC/USDT",
        order_type=OrderType.LIMIT,
        side=OrderSide.BUY,
        quantity=Decimal("1.0"),
        price=Decimal("50000.00")
    )
    await oco_order.handle_order_update(invalid_order)  # Should log warning but not fail
    
    # Test modification with no changes
    with patch('core.trading.oco_order.logger') as mock_logger:
        await oco_order.modify()  # No changes
        mock_logger.warning.assert_not_called()
