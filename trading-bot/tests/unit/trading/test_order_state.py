"""
Unit tests for order state management.

Tests the Order class and its state machine integration.
"""
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.trading.order_state import Order, OrderStatus, OrderSide, OrderType, TimeInForce
from core.trading.state_machine import StateTransitionError

class TestOrderState:
    """Test order state management."""
    
    @pytest.fixture
    def order(self):
        """Create a test order."""
        return Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000.00"),
            time_in_force=TimeInForce.GTC
        )
    
    @pytest.fixture
    def mock_redis(self):
        """Create a mock Redis client."""
        mock = AsyncMock()
        mock.eval.return_value = 1  # Simulate successful lock acquisition
        return mock
    
    @pytest.mark.asyncio
    async def test_initial_state(self, order):
        """Test initial order state."""
        assert order.status == OrderStatus.PENDING_NEW
        assert order.remaining_quantity == order.quantity
        assert order.is_active is False  # PENDING_NEW is not considered active
    
    @pytest.mark.asyncio
    async def test_activate_order(self, order):
        """Test order activation."""
        result = await order.update_status(OrderStatus.NEW)
        assert result is True
        assert order.status == OrderStatus.NEW
        assert order.is_active is True
    
    @pytest.mark.asyncio
    async def test_invalid_transition(self, order):
        """Test invalid state transition."""
        # Can't go directly from PENDING_NEW to FILLED
        with pytest.raises(StateTransitionError):
            await order.update_status(OrderStatus.FILLED)
    
    @pytest.mark.asyncio
    async def test_order_fill_flow(self, order):
        """Test order fill workflow."""
        # Activate order
        await order.update_status(OrderStatus.NEW)
        
        # Partially fill
        await order.fill(Decimal("0.5"), Decimal("50000.00"))
        assert order.status == OrderStatus.PARTIALLY_FILLED
        assert order.filled_quantity == Decimal("0.5")
        assert order.remaining_quantity == Decimal("0.5")
        
        # Complete fill
        await order.fill(Decimal("0.5"), Decimal("50000.00"))
        assert order.status == OrderStatus.FILLED
        assert order.filled_quantity == order.quantity
        assert order.remaining_quantity == Decimal("0")
    
    @pytest.mark.asyncio
    async def test_order_cancel(self, order):
        """Test order cancellation."""
        # Activate order
        await order.update_status(OrderStatus.NEW)
        
        # Cancel
        result = await order.cancel()
        assert result is True
        assert order.status == OrderStatus.CANCELED
        assert order.is_active is False
    
    @pytest.mark.asyncio
    async def test_order_reject(self, order):
        """Test order rejection."""
        # Reject pending order
        result = await order.reject("Insufficient funds")
        assert result is True
        assert order.status == OrderStatus.REJECTED
        assert order.is_active is False
    
    @pytest.mark.asyncio
    async def test_distributed_lock_integration(self, order, mock_redis):
        """Test distributed lock integration."""
        # Set up Redis mock
        Order.set_redis(mock_redis)
        
        # This should now use the distributed lock
        result = await order.update_status(OrderStatus.NEW)
        assert result is True
        
        # Verify Redis was called
        mock_redis.eval.assert_called()
    
    @pytest.mark.asyncio
    async def test_concurrent_updates(self, order, mock_redis):
        """Test handling of concurrent updates."""
        Order.set_redis(mock_redis)
        
        # Simulate lock contention
        mock_redis.eval.side_effect = [0, 1]  # First attempt fails, second succeeds
        
        result = await order.update_status(OrderStatus.NEW)
        assert result is True  # Should eventually succeed
        assert mock_redis.eval.call_count == 2
    
    @pytest.mark.asyncio
    async def test_state_hooks(self, order):
        """Test state transition hooks."""
        # Mock the hook
        mock_hook = AsyncMock()
        order.on_filled = mock_hook
        
        # Go through fill flow
        await order.update_status(OrderStatus.NEW)
        await order.fill(Decimal("1.0"), Decimal("50000.00"))
        
        # Verify hook was called
        mock_hook.assert_awaited_once()
    
    @pytest.mark.parametrize("from_state,to_state,should_succeed", [
        (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED, True),
        (OrderStatus.NEW, OrderStatus.FILLED, False),  # Must be partially filled first
        (OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED, True),
        (OrderStatus.FILLED, OrderStatus.CANCELED, False),  # Can't cancel filled order
        (OrderStatus.CANCELED, OrderStatus.NEW, False),  # Can't uncancel
    ])
    @pytest.mark.asyncio
    async def test_state_transitions(self, order, from_state, to_state, should_succeed):
        """Test various state transitions."""
        # Set initial state
        order.status = from_state
        
        if should_succeed:
            result = await order.update_status(to_state)
            assert result is True
            assert order.status == to_state
        else:
            with pytest.raises(StateTransitionError):
                await order.update_status(to_state)
    
    @pytest.mark.asyncio
    async def test_order_validation(self):
        """Test order parameter validation."""
        # Invalid quantity
        with pytest.raises(ValueError):
            Order(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("-1.0"),
                price=Decimal("50000.00")
            )
        
        # Missing price for limit order
        with pytest.raises(ValueError):
            Order(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("1.0")
                # No price provided
            )
        
        # Valid order
        order = Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1.0")
        )
        assert order is not None
