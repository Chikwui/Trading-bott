"""
Integration tests for order monitoring system.

These tests verify the integration between OrderManager and OrderMonitor,
ensuring that all order lifecycle events are properly tracked and monitored.
"""
import asyncio
import pytest
from datetime import datetime, timezone
from decimal import Decimal
from unittest.mock import AsyncMock, patch, MagicMock

from core.monitoring.order_monitor import OrderMonitor, StateTransitionEvent
from core.trading.order import Order, OrderSide, OrderType, TimeInForce
from core.trading.order_manager import OrderManager
from core.trading.oco_order import OCOOrder
from core.trading.order_monitoring import OrderMonitorIntegration
from core.trading.types import OrderStatus


@pytest.fixture
def order_manager():
    """Fixture providing a configured OrderManager instance."""
    return OrderManager()


@pytest.fixture
def order_monitor():
    """Fixture providing a configured OrderMonitor instance."""
    return OrderMonitor(metrics_port=0)  # Use port 0 to avoid port conflicts


@pytest.fixture
async def monitor_integration(order_manager, order_monitor):
    """Fixture providing a configured OrderMonitorIntegration instance."""
    integration = OrderMonitorIntegration(
        order_manager=order_manager,
        order_monitor=order_monitor,
        enable_metrics=True,
        enable_alerting=True
    )
    await integration.start()
    return integration


class TestOrderMonitoringIntegration:
    """Test suite for order monitoring integration."""

    @pytest.mark.asyncio
    async def test_order_creation_metrics(self, monitor_integration, order_manager, order_monitor):
        """Test that order creation is properly tracked with metrics."""
        # Create a test order
        order = Order(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000.00"),
            time_in_force=TimeInForce.GTC
        )

        # Submit the order
        await order_manager.submit_order(order)

        # Verify metrics were recorded
        metrics = await order_monitor.get_metrics()
        assert 'order_events_total' in metrics
        
        # Check that the order creation event was recorded
        counter = metrics['order_events_total']._metrics.get(
            ('state_transition', 'CREATED', 'BTC/USD', 'LIMIT')
        )
        assert counter._value.get() == 1

    @pytest.mark.asyncio
    async def test_order_fill_metrics(self, monitor_integration, order_manager, order_monitor):
        """Test that order fills are properly tracked with metrics."""
        # Create and submit a test order
        order = Order(
            symbol="ETH/USD",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Decimal("10.0"),
            price=Decimal("3000.00"),
            time_in_force=TimeInForce.GTC
        )
        await order_manager.submit_order(order)

        # Simulate partial fill
        await order_manager.update_order_status(order, OrderStatus.PARTIALLY_FILLED)
        await order_manager.process_fill(order, Decimal("5.0"), Decimal("3000.00"))

        # Simulate complete fill
        await order_manager.update_order_status(order, OrderStatus.FILLED)
        await order_manager.process_fill(order, Decimal("5.0"), Decimal("3000.00"))

        # Verify metrics
        metrics = order_monitor.metrics._metrics
        
        # Check partial fill event
        partial_counter = metrics['order_events_total']._metrics.get(
            ('state_transition', 'PARTIALLY_FILLED', 'ETH/USD', 'LIMIT')
        )
        assert partial_counter._value.get() == 1
        
        # Check complete fill event
        fill_counter = metrics['order_events_total']._metrics.get(
            ('state_transition', 'FILLED', 'ETH/USD', 'LIMIT')
        )
        assert fill_counter._value.get() == 1
        
        # Check order size metric
        assert 'order_size' in metrics
        size_hist = next(
            (h for h in metrics['order_size']._metrics.values() 
             if h._labelvalues == ('ETH/USD', 'LIMIT')),
            None
        )
        assert size_hist is not None
        assert size_hist._sum.get() == 10.0  # Total quantity

    @pytest.mark.asyncio
    async def test_order_cancellation_metrics(self, monitor_integration, order_manager, order_monitor):
        """Test that order cancellations are properly tracked with metrics."""
        # Create and submit a test order
        order = Order(
            symbol="XRP/USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("100.0"),
            price=Decimal("0.50"),
            time_in_force=TimeInForce.GTC
        )
        await order_manager.submit_order(order)

        # Cancel the order
        await order_manager.cancel_order(order, reason="User requested cancellation")

        # Verify metrics
        metrics = order_monitor.metrics._metrics
        
        # Check cancellation event
        cancel_counter = metrics['order_events_total']._metrics.get(
            ('state_transition', 'CANCELED', 'XRP/USD', 'LIMIT')
        )
        assert cancel_counter._value.get() == 1

    @pytest.mark.asyncio
    async def test_order_rejection_metrics(self, monitor_integration, order_manager, order_monitor):
        """Test that order rejections are properly tracked with metrics."""
        # Create a test order
        order = Order(
            symbol="INVALID/SYMBOL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1000.0")
        )

        # Simulate rejection by the exchange
        await order_manager.reject_order(order, reason="Invalid symbol")

        # Verify metrics
        metrics = order_monitor.metrics._metrics
        
        # Check rejection event
        reject_counter = metrics['order_events_total']._metrics.get(
            ('state_transition', 'REJECTED', 'INVALID/SYMBOL', 'MARKET')
        )
        assert reject_counter._value.get() == 1
        
        # Check error counter
        error_counter = metrics['order_errors_total']._metrics.get(
            ('rejection', 'INVALID/SYMBOL', 'MARKET')
        )
        assert error_counter._value.get() == 1

    @pytest.mark.asyncio
    async def test_oco_order_metrics(self, monitor_integration, order_manager, order_monitor):
        """Test that OCO orders are properly tracked with metrics."""
        # Create an OCO order
        oco_order = OCOOrder(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            limit_price=Decimal("49000.00"),
            stop_price=Decimal("49500.00"),
            stop_limit_price=Decimal("49600.00")
        )
        
        # Submit the OCO order
        await order_manager.submit_oco_order(oco_order)
        
        # Verify OCO creation metrics
        metrics = order_monitor.metrics._metrics
        oco_counter = metrics['order_events_total']._metrics.get(
            ('state_transition', 'OCO_CREATED', 'BTC/USD', 'OCO')
        )
        assert oco_counter._value.get() == 1
        
        # Simulate OCO order completion
        await order_manager.complete_oco_order(oco_order, "Take profit triggered")
        
        # Verify OCO completion metrics
        complete_counter = metrics['order_events_total']._metrics.get(
            ('state_transition', 'OCO_COMPLETED', 'BTC/USD', 'OCO')
        )
        assert complete_counter._value.get() == 1

    @pytest.mark.asyncio
    async def test_alert_rules_triggering(self, monitor_integration, order_manager, order_monitor):
        """Test that alert rules are properly triggered."""
        # Mock the alert handler to track alerts
        alerts = []
        
        async def mock_alert_handler(rule, alert_data):
            alerts.append((rule.name, alert_data))
        
        # Replace the default log handler with our mock
        order_monitor._alert_handlers['test'] = mock_alert_handler
        
        # Create and submit an order that should trigger the rejection alert
        order = Order(
            symbol="INVALID/PAIR",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("1000.0")
        )
        
        # Simulate rejection
        await order_manager.reject_order(order, reason="Invalid trading pair")
        
        # Verify the alert was triggered
        assert len(alerts) == 1
        alert_name, alert_data = alerts[0]
        assert alert_name == "Order Rejection"
        assert alert_data['context']['event'].to_state == 'REJECTED'

    @pytest.mark.asyncio
    async def test_latency_metrics(self, monitor_integration, order_manager, order_monitor):
        """Test that order processing latency is properly tracked."""
        # Create and submit a test order
        order = Order(
            symbol="LTC/USD",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("5.0"),
            price=Decimal("150.00"),
            time_in_force=TimeInForce.GTC
        )
        
        # Submit the order and simulate processing time
        start_time = datetime.now(timezone.utc)
        await order_manager.submit_order(order)
        
        # Simulate fill with some latency
        await asyncio.sleep(0.1)  # Simulate processing time
        await order_manager.update_order_status(order, OrderStatus.FILLED)
        
        # Verify latency metrics were recorded
        metrics = order_monitor.metrics._metrics
        latency_hist = next(
            (h for h in metrics['order_latency']._metrics.values() 
             if 'SUBMITTED_to_FILLED' in str(h._labelvalues)),
            None
        )
        assert latency_hist is not None
        assert latency_hist._sum.get() > 0

    @pytest.mark.asyncio
    async def test_position_metrics(self, monitor_integration, order_manager, order_monitor):
        """Test that position metrics are properly updated."""
        # Submit a BUY order
        buy_order = Order(
            symbol="SOL/USD",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("10.0")
        )
        await order_manager.submit_order(buy_order)
        await order_manager.update_order_status(buy_order, OrderStatus.FILLED)
        
        # Verify position metrics
        metrics = order_monitor.metrics._metrics
        position_gauge = next(
            (g for g in metrics['position_size']._metrics.values() 
             if g._labelvalues == ('SOL/USD', 'default')),
            None
        )
        assert position_gauge is not None
        assert position_gauge._value.get() == 10.0  # Position size should be 10.0
        
        # Submit a SELL order
        sell_order = Order(
            symbol="SOL/USD",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("4.0")
        )
        await order_manager.submit_order(sell_order)
        await order_manager.update_order_status(sell_order, OrderStatus.FILLED)
        
        # Verify position metrics after selling
        assert position_gauge._value.get() == 6.0  # Position size should be 6.0 after selling 4.0
