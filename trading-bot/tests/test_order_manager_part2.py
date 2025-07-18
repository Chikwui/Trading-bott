"""
Part 2 of comprehensive test suite for OrderManager.

This file covers:
- Advanced order types (OCO, Bracket, etc.)
- Concurrency and race conditions
- Order modification
- Performance testing
"""
import asyncio
import random
import time
from datetime import datetime, timedelta
from decimal import Decimal
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.trading.order import (
    Order, OrderStatus, OrderType, OrderSide, TimeInForce
)
from core.trading.advanced_orders import OCOOrder, BracketOrder
from core.trading.order_manager import OrderManager, OrderManagerConfig

# Reuse fixtures from part1
from tests.test_order_manager_part1 import (
    mock_exchange, mock_order_book, order_manager, sample_order, TEST_SYMBOL
)

class TestAdvancedOrders:
    """Tests for advanced order types (OCO, Bracket, etc.)."""
    
    @pytest.mark.asyncio
    async def test_oco_order_creation(self, order_manager):
        """Test OCO (One-Cancels-Other) order creation."""
        await order_manager.start()
        try:
            # Create two orders that form an OCO pair
            order1 = Order(
                symbol=TEST_SYMBOL,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=Decimal('49000'),
                quantity=Decimal('1.0'),
                time_in_force=TimeInForce.GTC
            )
            
            order2 = Order(
                symbol=TEST_SYMBOL,
                side=OrderSide.BUY,
                order_type=OrderType.STOP,
                price=Decimal('51000'),
                stop_price=Decimal('51000'),
                quantity=Decimal('1.0'),
                time_in_force=TimeInForce.GTC
            )
            
            # Create OCO order
            oco_order = await order_manager.create_oco_order(order1, order2)
            
            # Verify both orders were created and linked
            assert oco_order.order1.id is not None
            assert oco_order.order2.id is not None
            assert oco_order.group_id is not None
            
            # Verify orders are in the correct state
            assert order_manager.get_order(oco_order.order1.id) is not None
            assert order_manager.get_order(oco_order.order2.id) is not None
            
            # Cancel one order and verify the other is also canceled
            await order_manager.cancel_order(oco_order.order1.id)
            
            assert order_manager.get_order(oco_order.order1.id).status == OrderStatus.CANCELLED
            assert order_manager.get_order(oco_order.order2.id).status == OrderStatus.CANCELLED
            
        finally:
            await order_manager.stop()
    
    @pytest.mark.asyncio
    async def test_bracket_order_creation(self, order_manager):
        """Test bracket order creation."""
        await order_manager.start()
        
        try:
            # Create entry order
            entry = Order(
                symbol=TEST_SYMBOL,
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal('1.0'),
                time_in_force=TimeInForce.DAY
            )
            
            # Create take profit order
            take_profit = Order(
                symbol=TEST_SYMBOL,
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                price=Decimal('52000'),  # 4% profit
                quantity=Decimal('1.0'),
                time_in_force=TimeInForce.GTC
            )
            
            # Create stop loss order
            stop_loss = Order(
                symbol=TEST_SYMBOL,
                side=OrderSide.SELL,
                order_type=OrderType.STOP,
                price=Decimal('48000'),  # 4% loss
                stop_price=Decimal('48000'),
                quantity=Decimal('1.0'),
                time_in_force=TimeInForce.GTC
            )
            
            # Create bracket order
            bracket_order = await order_manager.create_bracket_order(
                entry=entry,
                take_profit=take_profit,
                stop_loss=stop_loss
            )
            
            # Verify all orders were created and linked
            assert bracket_order.entry_order.id is not None
            assert bracket_order.take_profit_order.id is not None
            assert bracket_order.stop_loss_order.id is not None
            assert bracket_order.group_id is not None
            
            # Verify orders are in the correct state
            assert order_manager.get_order(bracket_order.entry_order.id) is not None
            assert order_manager.get_order(bracket_order.take_profit_order.id) is not None
            assert order_manager.get_order(bracket_order.stop_loss_order.id) is not None
            
        finally:
            await order_manager.stop()

class TestConcurrency:
    """Concurrency and race condition tests."""
    
    @pytest.mark.asyncio
    async def test_concurrent_order_submission(self, order_manager):
        """Test concurrent order submission."""
        await order_manager.start()
        
        try:
            count = 100
            orders = [
                Order(
                    symbol=TEST_SYMBOL,
                    side=random.choice([OrderSide.BUY, OrderSide.SELL]),
                    order_type=OrderType.LIMIT,
                    price=Decimal(str(round(random.uniform(100, 100000), 2))),
                    quantity=Decimal('1.0'),
                    time_in_force=TimeInForce.GTC
                )
                for _ in range(count)
            ]
            
            # Submit all orders concurrently
            tasks = [order_manager.submit_order(order) for order in orders]
            results = await asyncio.gather(*tasks)
            
            # Verify all orders were created
            assert len(results) == count
            for order in results:
                assert order.status == OrderStatus.NEW
                assert order_manager.get_order(order.id) is not None
            
            # Cancel all orders
            cancel_tasks = [order_manager.cancel_order(order.id) for order in results]
            await asyncio.gather(*cancel_tasks)
            
        finally:
            await order_manager.stop()
    
    @pytest.mark.asyncio
    async def test_concurrent_order_modification(self, order_manager, sample_order):
        """Test concurrent order modification."""
        await order_manager.start()
        
        try:
            # Submit initial order
            order = await order_manager.submit_order(sample_order)
            
            # Define modification function
            async def modify_order():
                # Random delay to ensure concurrency
                await asyncio.sleep(random.uniform(0, 0.1))
                
                # Get current order
                current_order = order_manager.get_order(order.id)
                
                # Modify order
                new_price = current_order.price * Decimal('1.01')  # 1% increase
                modified_order = await order_manager.modify_order(
                    order_id=order.id,
                    price=new_price
                )
                
                return modified_order
            
            # Run multiple modifications concurrently
            tasks = [modify_order() for _ in range(10)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Verify all modifications were applied
            for result in results:
                if isinstance(result, Exception):
                    raise result
                assert result is not None
                assert result.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]
            
            # Cleanup
            await order_manager.cancel_order(order.id)
            
        finally:
            await order_manager.stop()

class TestOrderModification:
    """Tests for order modification."""
    
    @pytest.mark.asyncio
    async def test_modify_order_price(self, order_manager, sample_order):
        """Test modifying order price."""
        await order_manager.start()
        
        try:
            # Submit initial order
            order = await order_manager.submit_order(sample_order)
            
            # Modify order price
            new_price = order.price * Decimal('1.01')  # 1% increase
            modified_order = await order_manager.modify_order(
                order_id=order.id,
                price=new_price
            )
            
            # Verify modification
            assert modified_order.price == new_price
            assert modified_order.updated_at > order.updated_at
            
            # Cleanup
            await order_manager.cancel_order(order.id)
            
        finally:
            await order_manager.stop()
    
    @pytest.mark.asyncio
    async def test_modify_order_quantity(self, order_manager, sample_order):
        """Test modifying order quantity."""
        await order_manager.start()
        
        try:
            # Submit initial order
            order = await order_manager.submit_order(sample_order)
            
            # Modify order quantity
            new_quantity = order.quantity * Decimal('2')  # Double the quantity
            modified_order = await order_manager.modify_order(
                order_id=order.id,
                quantity=new_quantity
            )
            
            # Verify modification
            assert modified_order.quantity == new_quantity
            assert modified_order.updated_at > order.updated_at
            
            # Cleanup
            await order_manager.cancel_order(order.id)
            
        finally:
            await order_manager.stop()

class TestPerformance:
    """Performance tests for OrderManager."""
    
    @pytest.mark.benchmark(group="order_throughput")
    @pytest.mark.asyncio
    async def test_order_throughput(self, benchmark, order_manager):
        """Test order submission throughput."""
        await order_manager.start()
        
        try:
            count = 1000
            orders = [
                Order(
                    symbol=TEST_SYMBOL,
                    side=random.choice([OrderSide.BUY, OrderSide.SELL]),
                    order_type=OrderType.LIMIT,
                    price=Decimal(str(round(random.uniform(100, 100000), 2))),
                    quantity=Decimal(str(round(random.uniform(0.001, 100), 8))),
                    time_in_force=TimeInForce.GTC
                )
                for _ in range(count)
            ]
            
            async def submit_orders():
                return await asyncio.gather(
                    *(order_manager.submit_order(order) for order in orders)
                )
            
            # Warmup
            await submit_orders()
            
            # Benchmark
            result = await benchmark(submit_orders)
            
            assert len(result) == count
            
            # Cleanup
            await asyncio.gather(
                *(order_manager.cancel_order(order.id) for order in result)
            )
            
        finally:
            await order_manager.stop()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
