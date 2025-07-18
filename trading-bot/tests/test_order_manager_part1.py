"""
Part 1 of comprehensive test suite for OrderManager.

This file covers:
- Basic order management functionality
- Order creation and retrieval
- Order cancellation
- Basic error handling
"""
import asyncio
import pytest
import random
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
from hypothesis import given, strategies as st, settings, HealthCheck

from core.trading.order import (
    Order, OrderStatus, OrderType, OrderSide, TimeInForce
)
from core.trading.order_manager import (
    OrderManager, OrderManagerConfig, SlippageConfig, SlippageModelType,
    OrderManagerError, OrderNotFoundError, OrderAlreadyExistsError
)

# Test configuration
TEST_SYMBOL = "BTC/USD"
TEST_QUANTITY = Decimal("1.0")
TEST_PRICE = Decimal("50000.00")

# Hypothesis strategies
order_side = st.sampled_from([OrderSide.BUY, OrderSide.SELL])
order_type = st.sampled_from([OrderType.MARKET, OrderType.LIMIT, OrderType.STOP, OrderType.STOP_LIMIT])
tif = st.sampled_from([TimeInForce.DAY, TimeInForce.GTC, TimeInForce.IOC, TimeInForce.FOK])

# Fixtures
@pytest.fixture
def mock_exchange():
    """Mock exchange adapter."""
    exchange = AsyncMock()
    exchange.submit_order = AsyncMock(return_value={"order_id": "mock_exchange_id"})
    exchange.cancel_order = AsyncMock(return_value=True)
    return exchange

@pytest.fixture
def mock_order_book():
    """Mock order book."""
    book = MagicMock()
    book.get_snapshot.return_value = {
        'bids': [(Decimal('49999'), Decimal('10')), (Decimal('49998'), Decimal('5'))],
        'asks': [(Decimal('50001'), Decimal('8')), (Decimal('50002'), Decimal('12'))],
        'timestamp': datetime.utcnow()
    }
    return book

@pytest.fixture
def order_manager(mock_exchange, mock_order_book):
    """Order manager instance for testing."""
    config = OrderManagerConfig(
        default_slippage_model=SlippageConfig(
            model_type=SlippageModelType.CONSTANT,
            constant=Decimal('0.0005')
        )
    )
    return OrderManager(
        exchange_adapter=mock_exchange,
        order_book=mock_order_book,
        config=config
    )

@pytest.fixture
def sample_order():
    """Sample order for testing."""
    return Order(
        symbol=TEST_SYMBOL,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=TEST_QUANTITY,
        price=TEST_PRICE,
        time_in_force=TimeInForce.GTC
    )

class TestOrderManagerBasics:
    """Basic functionality tests for OrderManager."""
    
    @pytest.mark.asyncio
    async def test_submit_order(self, order_manager, sample_order):
        """Test order submission."""
        await order_manager.start()
        try:
            result = await order_manager.submit_order(sample_order)
            
            assert result.status == OrderStatus.NEW
            assert result.id is not None
            assert order_manager.get_order(result.id) is not None
        finally:
            await order_manager.stop()
    
    @pytest.mark.asyncio
    async def test_duplicate_order_id(self, order_manager, sample_order):
        """Test duplicate order ID rejection."""
        await order_manager.start()
        try:
            await order_manager.submit_order(sample_order)
            
            with pytest.raises(OrderAlreadyExistsError):
                await order_manager.submit_order(sample_order)
        finally:
            await order_manager.stop()
    
    @pytest.mark.asyncio
    async def test_cancel_order(self, order_manager, sample_order):
        """Test order cancellation."""
        await order_manager.start()
        try:
            order = await order_manager.submit_order(sample_order)
            
            result = await order_manager.cancel_order(order.id)
            assert result is True
            
            canceled_order = order_manager.get_order(order.id)
            assert canceled_order.status == OrderStatus.CANCELLED
        finally:
            await order_manager.stop()
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_order(self, order_manager):
        """Test retrieving non-existent order."""
        await order_manager.start()
        try:
            with pytest.raises(OrderNotFoundError):
                order_manager.get_order("nonexistent_order_id")
        finally:
            await order_manager.stop()

class TestOrderPropertyBased:
    """Property-based tests for OrderManager."""
    
    @given(
        side=order_side,
        order_type=order_type,
        price=st.decimals(min_value='0.00000001', max_value='1000000', places=8),
        quantity=st.decimals(min_value='0.00000001', max_value='1000000', places=8),
        time_in_force=tif
    )
    @settings(suppress_health_check=[HealthCheck.too_slow], max_examples=100)
    @pytest.mark.asyncio
    async def test_order_creation_variations(
        self, order_manager, side, order_type, price, quantity, time_in_force
    ):
        """Test order creation with various parameter combinations."""
        await order_manager.start()
        try:
            order = Order(
                symbol=TEST_SYMBOL,
                side=side,
                order_type=order_type,
                price=price,
                quantity=quantity,
                time_in_force=time_in_force
            )
            
            result = await order_manager.submit_order(order)
            assert result is not None
            assert result.status == OrderStatus.NEW
            
            # Verify order can be retrieved
            stored_order = order_manager.get_order(result.id)
            assert stored_order is not None
            assert stored_order.id == result.id
            
            # Cleanup
            await order_manager.cancel_order(result.id)
        finally:
            await order_manager.stop()

class TestErrorHandling:
    """Error handling tests for OrderManager."""
    
    @pytest.mark.asyncio
    async def test_nonexistent_order_cancellation(self, order_manager):
        """Test cancellation of non-existent order."""
        await order_manager.start()
        try:
            with pytest.raises(OrderNotFoundError):
                await order_manager.cancel_order("nonexistent_order_id")
        finally:
            await order_manager.stop()
    
    @pytest.mark.asyncio
    async def test_invalid_order_submission(self, order_manager):
        """Test submission of invalid orders."""
        await order_manager.start()
        try:
            # Missing required fields
            with pytest.raises(ValueError):
                invalid_order = Order(
                    symbol=None,  # Missing required field
                    side=OrderSide.BUY,
                    order_type=OrderType.MARKET,
                    quantity=Decimal('1.0')
                )
                await order_manager.submit_order(invalid_order)
        finally:
            await order_manager.stop()

class TestOrderLifecycle:
    """Tests for order lifecycle management."""
    
    @pytest.mark.asyncio
    async def test_order_status_transitions(self, order_manager, sample_order):
        """Test valid order status transitions."""
        await order_manager.start()
        try:
            # Submit new order
            order = await order_manager.submit_order(sample_order)
            assert order.status == OrderStatus.NEW
            
            # Partially fill order
            await order_manager._on_order_event(
                order,
                {
                    'event_type': 'FILL',
                    'filled_quantity': order.quantity / 2,
                    'price': order.price,
                    'timestamp': datetime.utcnow()
                }
            )
            
            updated_order = order_manager.get_order(order.id)
            assert updated_order.status == OrderStatus.PARTIALLY_FILLED
            
            # Complete fill
            await order_manager._on_order_event(
                order,
                {
                    'event_type': 'FILL',
                    'filled_quantity': order.quantity,
                    'price': order.price,
                    'timestamp': datetime.utcnow()
                }
            )
            
            completed_order = order_manager.get_order(order.id)
            assert completed_order.status == OrderStatus.FILLED
            
        finally:
            await order_manager.stop()
    
    @pytest.mark.asyncio
    async def test_order_rejection(self, order_manager, sample_order):
        """Test order rejection handling."""
        await order_manager.start()
        try:
            order = await order_manager.submit_order(sample_order)
            
            # Simulate rejection
            await order_manager._on_order_event(
                order,
                {
                    'event_type': 'REJECT',
                    'reason': 'INSUFFICIENT_FUNDS',
                    'timestamp': datetime.utcnow()
                }
            )
            
            rejected_order = order_manager.get_order(order.id)
            assert rejected_order.status == OrderStatus.REJECTED
            
        finally:
            await order_manager.stop()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
