"""
Test suite for execution strategies (VWAP, Iceberg).

This file covers:
- VWAP execution strategy
- Iceberg execution strategy
- Integration with OrderManager
- Performance benchmarks
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
from core.trading.execution.vwap import VWAPExecution
from core.trading.execution.iceberg import IcebergExecution
from core.trading.order_manager import OrderManager

# Test configuration
TEST_SYMBOL = "BTC/USD"
TEST_QUANTITY = Decimal("10.0")  # Larger quantity for execution testing

@pytest.fixture
def mock_market_data():
    """Mock market data feed."""
    class MockMarketData:
        def __init__(self):
            self.callbacks = []
            self.running = False
            
        def subscribe_trades(self, symbol, callback):
            self.callbacks.append(callback)
            return True
            
        def start(self):
            self.running = True
            
        def stop(self):
            self.running = False
            
        def generate_trade(self, price, quantity):
            """Generate a mock trade."""
            trade = {
                'symbol': TEST_SYMBOL,
                'price': price,
                'quantity': quantity,
                'timestamp': datetime.utcnow(),
                'is_buyer_maker': random.choice([True, False]),
                'trade_id': str(random.randint(1, 1000000))
            }
            for callback in self.callbacks:
                callback(trade)
    
    return MockMarketData()

class TestVWAPExecution:
    """Tests for VWAP (Volume-Weighted Average Price) execution strategy."""
    
    @pytest.fixture
    def vwap_strategy(self):
        """Create a VWAP execution strategy instance."""
        return VWAPExecution(
            order_manager=MagicMock(),
            market_data=MagicMock(),
            symbol=TEST_SYMBOL,
            side=OrderSide.BUY,
            quantity=TEST_QUANTITY,
            time_in_force=TimeInForce.DAY,
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow() + timedelta(minutes=5),
            interval=timedelta(seconds=30)
        )
    
    @pytest.mark.asyncio
    async def test_vwap_basic_execution(self, vwap_strategy, mock_market_data):
        """Test basic VWAP execution."""
        # Test setup
        symbol = "BTC/USDT"
        quantity = Decimal("1.0")
        side = OrderSide.BUY
        
        # Execute order
        result = await vwap_strategy.execute_order(symbol, quantity, side)
        
        # Verify results
        assert result.status == OrderStatus.FILLED
        assert result.filled_quantity == quantity
        assert result.avg_price > 0
        assert result.execution_time > 0
        assert result.slippage is not None
    
    @pytest.mark.asyncio
    async def test_vwap_large_order_slicing(self, vwap_strategy, mock_market_data):
        """Test VWAP execution with large order slicing."""
        symbol = "ETH/USDT"
        quantity = Decimal("1000.0")  # Large order that should be sliced
        side = OrderSide.SELL
        
        # Execute order
        result = await vwap_strategy.execute_order(
            symbol=symbol,
            quantity=quantity,
            side=side,
            params={
                'max_slice_size': Decimal('100.0'),
                'participation_rate': 0.1
            }
        )
        
        # Verify results
        assert result.status == OrderStatus.FILLED
        assert result.filled_quantity == quantity
        assert result.child_orders and len(result.child_orders) > 1
        assert all(o.status == OrderStatus.FILLED for o in result.child_orders)
    
    @pytest.mark.asyncio
    async def test_vwap_market_conditions(self, vwap_strategy, mock_market_data):
        """Test VWAP adaptation to different market conditions."""
        symbol = "BTC/USDT"
        test_cases = [
            # (quantity, volatility, expected_slices, expected_participation)
            (Decimal("10.0"), 0.2, 1, 0.2),  # Low volatility, small order
            (Decimal("100.0"), 0.5, 3, 0.15),  # Medium volatility, medium order
            (Decimal("1000.0"), 1.0, 10, 0.1),  # High volatility, large order
        ]
        
        for qty, vol, exp_slices, exp_part in test_cases:
            # Mock market conditions
            mock_market_data.volatility = vol
            
            # Execute order
            result = await vwap_strategy.execute_order(
                symbol=symbol,
                quantity=qty,
                side=OrderSide.BUY,
                params={'max_volatility': 1.0}
            )
            
            # Verify adaptation to market conditions
            assert result.status == OrderStatus.FILLED
            assert len(result.child_orders) >= exp_slices
            
            # Verify participation rate adjustment
            part_rates = [o.participation_rate for o in result.child_orders]
            avg_part = sum(part_rates) / len(part_rates)
            assert abs(avg_part - exp_part) < 0.1  # Within 10% of expected
    
    @pytest.mark.asyncio
    async def test_vwap_basic_execution(self, mock_market_data):
        """Test basic VWAP execution."""
        # Create mock order manager
        order_manager = MagicMock()
        order_manager.submit_order = AsyncMock()
        
        # Create VWAP execution
        vwap = VWAPExecution(
            order_manager=order_manager,
            market_data=mock_market_data,
            symbol=TEST_SYMBOL,
            side=OrderSide.BUY,
            quantity=TEST_QUANTITY,
            time_in_force=TimeInForce.DAY,
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow() + timedelta(minutes=5),
            interval=timedelta(seconds=30)
        )
        
        # Start execution
        await vwap.start()
        
        # Simulate some market data
        mock_market_data.generate_trade(Decimal('50000'), Decimal('2.0'))
        mock_market_data.generate_trade(Decimal('50050'), Decimal('3.0'))
        
        # Wait for orders to be placed
        await asyncio.sleep(0.1)
        
        # Verify orders were placed
        assert order_manager.submit_order.called
        
        # Stop execution
        await vwap.stop()
    
    @pytest.mark.asyncio
    async def test_vwap_slippage_control(self, mock_market_data):
        """Test VWAP execution with slippage control."""
        # Create mock order manager
        order_manager = MagicMock()
        order_manager.submit_order = AsyncMock()
        
        # Create VWAP execution with tight slippage control
        vwap = VWAPExecution(
            order_manager=order_manager,
            market_data=mock_market_data,
            symbol=TEST_SYMBOL,
            side=OrderSide.BUY,
            quantity=TEST_QUANTITY,
            time_in_force=TimeInForce.DAY,
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow() + timedelta(minutes=5),
            interval=timedelta(seconds=30),
            max_slippage=Decimal('0.001')  # 0.1% max slippage
        )
        
        # Start execution
        await vwap.start()
        
        # Simulate market data with high volatility
        mock_market_data.generate_trade(Decimal('50000'), Decimal('1.0'))
        mock_market_data.generate_trade(Decimal('50500'), Decimal('1.0'))  # 1% move - should be outside slippage
        
        # Wait for orders to be placed
        await asyncio.sleep(0.1)
        
        # Verify no orders were placed due to slippage
        assert not order_manager.submit_order.called
        
        # Stop execution
        await vwap.stop()

class TestIcebergExecution:
    """Tests for Iceberg order execution strategy."""
    
    @pytest.mark.asyncio
    async def test_iceberg_basic_execution(self, mock_market_data):
        """Test basic Iceberg order execution."""
        # Create mock order manager
        order_manager = MagicMock()
        order_manager.submit_order = AsyncMock()
        
        # Create Iceberg execution
        iceberg = IcebergExecution(
            order_manager=order_manager,
            market_data=mock_market_data,
            symbol=TEST_SYMBOL,
            side=OrderSide.BUY,
            total_quantity=TEST_QUANTITY,
            display_quantity=Decimal('1.0'),  # Show only 1.0 at a time
            price=Decimal('50000'),
            time_in_force=TimeInForce.DAY
        )
        
        # Start execution
        await iceberg.start()
        
        # Simulate market data
        mock_market_data.generate_trade(Decimal('50000'), Decimal('0.5'))
        
        # Wait for orders to be placed
        await asyncio.sleep(0.1)
        
        # Verify initial order was placed
        order_manager.submit_order.assert_called_once()
        args, kwargs = order_manager.submit_order.call_args
        order = args[0]
        
        assert order.quantity == Decimal('1.0')  # Display quantity
        assert order.price == Decimal('50000')
        
        # Simulate fill
        order_manager.submit_order.reset_mock()
        await iceberg._on_order_fill({
            'order_id': order.id,
            'filled_quantity': Decimal('1.0'),
            'remaining_quantity': Decimal('0'),
            'price': Decimal('50000'),
            'timestamp': datetime.utcnow()
        })
        
        # Verify new order was placed
        order_manager.submit_order.assert_called_once()
        
        # Stop execution
        await iceberg.stop()
    
    @pytest.mark.asyncio
    async def test_iceberg_slice_refill(self, mock_market_data):
        """Test Iceberg order slice refill logic."""
        # Create mock order manager
        order_manager = MagicMock()
        order_manager.submit_order = AsyncMock()
        
        # Create Iceberg execution with small display quantity
        iceberg = IcebergExecution(
            order_manager=order_manager,
            market_data=mock_market_data,
            symbol=TEST_SYMBOL,
            side=OrderSide.BUY,
            total_quantity=Decimal('5.0'),
            display_quantity=Decimal('1.0'),
            price=Decimal('50000'),
            time_in_force=TimeInForce.DAY
        )
        
        # Start execution
        await iceberg.start()
        
        # Get the first order
        order_manager.submit_order.assert_called_once()
        args, kwargs = order_manager.submit_order.call_args
        order = args[0]
        
        # Simulate partial fill
        order_manager.submit_order.reset_mock()
        await iceberg._on_order_fill({
            'order_id': order.id,
            'filled_quantity': Decimal('0.5'),
            'remaining_quantity': Decimal('0.5'),
            'price': Decimal('50000'),
            'timestamp': datetime.utcnow()
        })
        
        # Verify no new order yet (still have remaining quantity)
        order_manager.submit_order.assert_not_called()
        
        # Simulate complete fill of first slice
        await iceberg._on_order_fill({
            'order_id': order.id,
            'filled_quantity': Decimal('1.0'),
            'remaining_quantity': Decimal('0'),
            'price': Decimal('50000'),
            'timestamp': datetime.utcnow()
        })
        
        # Verify new order was placed
        order_manager.submit_order.assert_called_once()
        
        # Stop execution
        await iceberg.stop()

class TestExecutionIntegration:
    """Integration tests for execution strategies with OrderManager."""
    
    @pytest.mark.asyncio
    async def test_vwap_integration(self, order_manager, mock_market_data):
        """Test VWAP execution integrated with OrderManager."""
        # Replace market data with our mock
        order_manager.market_data = mock_market_data
        
        # Start order manager
        await order_manager.start()
        
        try:
            # Create VWAP execution
            vwap = VWAPExecution(
                order_manager=order_manager,
                market_data=mock_market_data,
                symbol=TEST_SYMBOL,
                side=OrderSide.BUY,
                quantity=TEST_QUANTITY,
                time_in_force=TimeInForce.DAY,
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow() + timedelta(minutes=5),
                interval=timedelta(seconds=10)
            )
            
            # Start execution
            await vwap.start()
            
            # Simulate some market data
            for _ in range(5):
                mock_market_data.generate_trade(
                    Decimal(str(50000 + random.uniform(-100, 100))),
                    Decimal(str(random.uniform(0.1, 5.0)))
                )
                await asyncio.sleep(0.01)
            
            # Wait for orders to be placed
            await asyncio.sleep(0.5)
            
            # Verify orders were placed
            orders = order_manager.get_orders()
            assert len(orders) > 0
            
            # Stop execution
            await vwap.stop()
            
        finally:
            await order_manager.stop()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
