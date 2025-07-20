"""
Unit tests for OrderMonitor and related monitoring components.

These tests focus on individual components in isolation, using mocks where appropriate.
"""
import asyncio
import pytest
import time
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch, call, ANY

import numpy as np
from prometheus_client import Counter, Gauge, Histogram

from core.monitoring.order_monitor import (
    OrderMonitor,
    OrderMetrics,
    AlertRule,
    MetricType,
    MetricDefinition,
    StateTransitionEvent
)
from core.trading.order import Order, OrderSide, OrderType, TimeInForce


class TestStateTransitionEvent:
    """Tests for the StateTransitionEvent class."""
    
    def test_creation(self):
        """Test basic event creation with all fields."""
        timestamp = datetime.now(timezone.utc)
        event = StateTransitionEvent(
            from_state="NEW",
            to_state="CREATED",
            timestamp=timestamp,
            reason="Order created",
            metadata={"order_id": "123", "symbol": "BTC/USD"}
        )
        
        assert event.from_state == "NEW"
        assert event.to_state == "CREATED"
        assert event.timestamp == timestamp
        assert event.reason == "Order created"
        assert event.metadata == {"order_id": "123", "symbol": "BTC/USD"}
    
    def test_default_timestamp(self):
        """Test that timestamp defaults to current time if not provided."""
        before = datetime.now(timezone.utc)
        event = StateTransitionEvent(
            from_state="NEW",
            to_state="CREATED",
            reason="Test"
        )
        after = datetime.now(timezone.utc)
        
        assert before <= event.timestamp <= after
    
    def test_str_representation(self):
        """Test string representation of the event."""
        event = StateTransitionEvent(
            from_state="NEW",
            to_state="CREATED",
            timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
            reason="Test"
        )
        
        assert "NEW -> CREATED" in str(event)
        assert "Test" in str(event)
        assert "2023-01-01" in str(event)


class TestAlertRule:
    """Tests for the AlertRule class."""
    
    def test_creation(self):
        """Test alert rule creation with all fields."""
        rule = AlertRule(
            name="Test Rule",
            condition="event.to_state == 'REJECTED'",
            severity="critical",
            message="Order was rejected",
            action="print('Alert!')",
            cooldown_seconds=300,
            active=True
        )
        
        assert rule.name == "Test Rule"
        assert rule.condition == "event.to_state == 'REJECTED'"
        assert rule.severity == "critical"
        assert rule.message == "Order was rejected"
        assert rule.action == "print('Alert!')"
        assert rule.cooldown_seconds == 300
        assert rule.active is True
        assert isinstance(rule.id, str) and len(rule.id) > 0
    
    def test_default_values(self):
        """Test alert rule with default values."""
        rule = AlertRule(
            name="Test Rule",
            condition="True",
            severity="info",
            message="Test"
        )
        
        assert rule.action is None
        assert rule.cooldown_seconds == 300
        assert rule.active is True
    
    def test_validate_severity(self):
        """Test severity validation."""
        with pytest.raises(ValueError):
            AlertRule(
                name="Invalid Severity",
                condition="True",
                severity="invalid",
                message="Test"
            )
    
    def test_json_serialization(self):
        """Test that the rule can be serialized to JSON."""
        rule = AlertRule(
            name="Test Rule",
            condition="event.to_state == 'REJECTED'",
            severity="critical",
            message="Order was rejected"
        )
        
        # Just verify it doesn't raise an exception
        rule.json()


class TestOrderMetrics:
    """Tests for the OrderMetrics class."""
    
    @pytest.fixture
    def metrics(self):
        """Fixture providing a clean OrderMetrics instance."""
        return OrderMetrics(namespace="test")
    
    def test_initialization(self, metrics):
        """Test that metrics are properly initialized."""
        assert hasattr(metrics, '_metrics')
        assert isinstance(metrics._metrics, dict)
        
        # Check that all expected metrics are registered
        expected_metrics = [
            'order_events_total',
            'order_execution_time',
            'order_size',
            'order_latency',
            'order_errors_total',
            'position_size',
            'realized_pnl',
            'unrealized_pnl'
        ]
        
        for metric in expected_metrics:
            assert metric in metrics._metrics, f"Metric {metric} not found in metrics"
    
    def test_record_order_event(self, metrics):
        """Test recording order events."""
        # Test with quantity and price
        metrics.record_order_event(
            event_type="state_transition",
            status="CREATED",
            symbol="BTC/USD",
            order_type="LIMIT",
            quantity=1.0,
            price=50000.0
        )
        
        # Check counter was incremented
        counter = metrics._metrics['order_events_total']._metrics[
            ('state_transition', 'CREATED', 'BTC/USD', 'LIMIT')
        ]
        assert counter._value.get() == 1
        
        # Check order size histogram was updated
        size_hist = next(
            h for h in metrics._metrics['order_size']._metrics.values()
            if h._labelvalues == ('BTC/USD', 'LIMIT')
        )
        assert size_hist._sum.get() == 1.0
    
    def test_record_latency(self, metrics):
        """Test recording order processing latency."""
        metrics.record_latency(
            stage="execution",
            symbol="ETH/USD",
            order_type="MARKET",
            latency_seconds=0.123
        )
        
        # Find the histogram for this combination of labels
        hist = next(
            h for h in metrics._metrics['order_latency']._metrics.values()
            if h._labelvalues == ('execution', 'ETH/USD', 'MARKET')
        )
        
        assert hist._sum.get() == pytest.approx(0.123)
        assert hist._count.get() == 1
    
    def test_record_error(self, metrics):
        """Test recording order processing errors."""
        metrics.record_error(
            error_type="timeout",
            symbol="XRP/USD",
            order_type="LIMIT"
        )
        
        counter = metrics._metrics['order_errors_total']._metrics[
            ('timeout', 'XRP/USD', 'LIMIT')
        ]
        assert counter._value.get() == 1
    
    def test_update_position(self, metrics):
        """Test updating position metrics."""
        # Initial position
        metrics.update_position(
            symbol="BTC/USD",
            account_id="account_123",
            size=1.5,
            realized_pnl=100.0,
            unrealized_pnl=50.0
        )
        
        # Check position size
        position_gauge = metrics._metrics['position_size']._metrics[
            ('BTC/USD', 'account_123')
        ]
        assert position_gauge._value.get() == 1.5
        
        # Check PnL metrics
        realized_gauge = metrics._metrics['realized_pnl']._metrics[
            ('BTC/USD', 'account_123')
        ]
        assert realized_gauge._value.get() == 100.0
        
        unrealized_gauge = metrics._metrics['unrealized_pnl']._metrics[
            ('BTC/USD', 'account_123')
        ]
        assert unrealized_gauge._value.get() == 50.0
        
        # Update position
        metrics.update_position(
            symbol="BTC/USD",
            account_id="account_123",
            size=2.0,
            realized_pnl=150.0,
            unrealized_pnl=75.0
        )
        
        assert position_gauge._value.get() == 2.0
        assert realized_gauge._value.get() == 150.0
        assert unrealized_gauge._value.get() == 75.0


class TestOrderMonitor:
    """Tests for the OrderMonitor class."""
    
    @pytest.fixture
    async def monitor(self):
        """Fixture providing an OrderMonitor instance with metrics server disabled."""
        # Use port 0 to avoid port conflicts
        monitor = OrderMonitor(metrics_port=0, start_http_server=False)
        yield monitor
        # Cleanup if needed
        
    @pytest.fixture
    def mock_alert_handler(self):
        """Fixture providing a mock alert handler."""
        async def handler(rule, alert_data):
            handler.calls.append((rule, alert_data))
        
        handler.calls = []
        return handler
    
    @pytest.mark.asyncio
    async def test_start_server(self, monitor):
        """Test starting the metrics server."""
        # This is a bit tricky to test without actually binding to a port
        # So we'll just verify the method exists and can be called
        with patch('prometheus_client.start_http_server') as mock_start:
            await monitor.start()
            mock_start.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_load_alert_rules(self, monitor):
        """Test loading alert rules from configuration."""
        rules = [
            {
                'name': 'Test Rule',
                'condition': "event.to_state == 'REJECTED'",
                'severity': 'critical',
                'message': 'Order was rejected',
                'cooldown_seconds': 0
            }
        ]
        
        monitor.load_alert_rules(rules)
        
        assert len(monitor.alert_rules) == 1
        rule = list(monitor.alert_rules.values())[0]
        assert rule.name == 'Test Rule'
        assert rule.condition == "event.to_state == 'REJECTED'"
    
    @pytest.mark.asyncio
    async def test_register_alert_handler(self, monitor, mock_alert_handler):
        """Test registering alert handlers."""
        monitor.register_alert_handler('test_handler', mock_alert_handler)
        assert 'test_handler' in monitor._alert_handlers
        assert monitor._alert_handlers['test_handler'] == mock_alert_handler
    
    @pytest.mark.asyncio
    async def test_process_order_event(self, monitor, mock_alert_handler):
        """Test processing an order event with metrics and alerts."""
        # Register a mock alert handler
        monitor.register_alert_handler('test', mock_alert_handler)
        
        # Add an alert rule that will trigger on this event
        monitor.load_alert_rules([
            {
                'name': 'Test Alert',
                'condition': "event.to_state == 'FILLED'",
                'severity': 'info',
                'message': 'Order filled',
                'cooldown_seconds': 0
            }
        ])
        
        # Create a test order and event
        order = MagicMock()
        order.symbol = 'BTC/USD'
        order.order_type = 'LIMIT'
        
        event = StateTransitionEvent(
            from_state='PARTIALLY_FILLED',
            to_state='FILLED',
            timestamp=datetime.now(timezone.utc),
            reason='Order filled',
            metadata={'quantity': 1.0, 'price': 50000.0}
        )
        
        # Process the event
        await monitor.process_order_event(event, order)
        
        # Verify metrics were updated
        metrics = monitor.metrics._metrics
        counter = metrics['order_events_total']._metrics[
            ('state_transition', 'FILLED', 'BTC/USD', 'LIMIT')
        ]
        assert counter._value.get() == 1
        
        # Verify alert was triggered
        assert len(mock_alert_handler.calls) == 1
        rule, alert_data = mock_alert_handler.calls[0]
        assert rule.name == 'Test Alert'
        assert alert_data['context']['event'] == event
    
    @pytest.mark.asyncio
    async def test_alert_cooldown(self, monitor, mock_alert_handler):
        """Test that alert cooldowns are respected."""
        # Register a mock alert handler
        monitor.register_alert_handler('test', mock_alert_handler)
        
        # Add an alert rule with cooldown
        monitor.load_alert_rules([
            {
                'name': 'Cooldown Test',
                'condition': "True",
                'severity': 'info',
                'message': 'Test',
                'cooldown_seconds': 60
            }
        ])
        
        rule_id = next(iter(monitor.alert_rules.keys()))
        
        # First call - should trigger
        event = StateTransitionEvent(
            from_state='NEW',
            to_state='CREATED',
            timestamp=datetime.now(timezone.utc),
            reason='Test'
        )
        
        await monitor.process_order_event(event, MagicMock())
        assert len(mock_alert_handler.calls) == 1
        
        # Reset mock calls
        mock_alert_handler.calls.clear()
        
        # Second call immediately after - should be skipped due to cooldown
        await monitor.process_order_event(event, MagicMock())
        assert len(mock_alert_handler.calls) == 0
        
        # Move time forward past cooldown
        with patch('time.time', return_value=time.time() + 61):
            await monitor.process_order_event(event, MagicMock())
            assert len(mock_alert_handler.calls) == 1
    
    @pytest.mark.asyncio
    async def test_alert_handling_error(self, monitor, caplog):
        """Test that errors in alert handlers don't break processing."""
        # Create a failing alert handler
        async def failing_handler(rule, alert_data):
            raise ValueError("Alert handler failed")
        
        monitor.register_alert_handler('failing', failing_handler)
        
        # Add a rule that will trigger
        monitor.load_alert_rules([
            {
                'name': 'Failing Alert',
                'condition': "True",
                'severity': 'info',
                'message': 'This will fail',
                'cooldown_seconds': 0
            }
        ])
        
        # Process an event that would trigger the alert
        event = StateTransitionEvent(
            from_state='NEW',
            to_state='CREATED',
            timestamp=datetime.now(timezone.utc),
            reason='Test'
        )
        
        # Should not raise
        await monitor.process_order_event(event, MagicMock())
        
        # Should log the error
        assert "Error in alert handler" in caplog.text
        assert "Alert handler failed" in caplog.text


class TestOrderMonitorIntegration:
    """Tests for the OrderMonitorIntegration class."""
    
    @pytest.fixture
    async def integration(self, order_manager):
        """Fixture providing an OrderMonitorIntegration instance."""
        integration = OrderMonitorIntegration(
            order_manager=order_manager,
            enable_metrics=True,
            enable_alerting=True,
            default_alert_rules=False
        )
        # Don't actually start the HTTP server
        integration.order_monitor.start = AsyncMock()
        await integration.start()
        return integration
    
    @pytest.mark.asyncio
    async def test_initialization(self, integration, order_manager):
        """Test that integration is properly initialized."""
        assert integration.order_manager == order_manager
        assert isinstance(integration.order_monitor, OrderMonitor)
        assert integration.enable_metrics is True
        assert integration.enable_alerting is True
    
    @pytest.mark.asyncio
    async def test_order_creation_event(self, integration, order_manager):
        """Test that order creation triggers the right events."""
        # Mock the process_order_event method to verify calls
        with patch.object(integration.order_monitor, 'process_order_event') as mock_process:
            # Create and submit an order
            order = Order(
                symbol="BTC/USD",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("1.0"),
                price=Decimal("50000.00"),
                time_in_force=TimeInForce.GTC
            )
            
            await order_manager.submit_order(order)
            
            # Verify process_order_event was called with the right arguments
            assert mock_process.called
            args, _ = mock_process.call_args
            event, order_arg = args
            
            assert isinstance(event, StateTransitionEvent)
            assert event.from_state == 'NEW'
            assert event.to_state == 'CREATED'
            assert order_arg == order
    
    @pytest.mark.asyncio
    async def test_order_fill_events(self, integration, order_manager):
        """Test that order fill events are properly handled."""
        # Create and submit an order
        order = Order(
            symbol="ETH/USD",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Decimal("10.0"),
            price=Decimal("3000.00"),
            time_in_force=TimeInForce.GTC
        )
        
        await order_manager.submit_order(order)
        
        # Mock the process_order_event method for the update
        with patch.object(integration.order_monitor, 'process_order_event') as mock_process:
            # Simulate a fill
            await order_manager.update_order_status(order, OrderStatus.PARTIALLY_FILLED)
            await order_manager.process_fill(order, Decimal("5.0"), Decimal("3000.00"))
            
            # Verify the partial fill event
            assert mock_process.called
            args, _ = mock_process.call_args
            event, _ = args
            
            assert event.to_state == 'PARTIALLY_FILLED'
            assert event.metadata['fill_quantity'] == 5.0
            assert event.metadata['fill_price'] == 3000.00
            assert event.metadata['is_complete'] is False
    
    @pytest.mark.asyncio
    async def test_alert_rule_loading(self, integration):
        """Test that default alert rules are loaded when requested."""
        # Create a new integration with default rules
        order_manager = MagicMock()
        integration_with_rules = OrderMonitorIntegration(
            order_manager=order_manager,
            enable_alerting=True,
            default_alert_rules=True
        )
        
        # Should have loaded some default rules
        assert len(integration_with_rules.order_monitor.alert_rules) > 0
    
    @pytest.mark.asyncio
    async def test_oco_order_handling(self, integration, order_manager):
        """Test that OCO orders are properly handled."""
        # Create an OCO order
        oco_order = OCOOrder(
            symbol="BTC/USD",
            side=OrderSide.BUY,
            quantity=Decimal("0.1"),
            limit_price=Decimal("49000.00"),
            stop_price=Decimal("49500.00")
        )
        
        # Mock the process_order_event method
        with patch.object(integration.order_monitor, 'process_order_event') as mock_process:
            # Submit the OCO order
            await order_manager.submit_oco_order(oco_order)
            
            # Should have called process_order_event for OCO creation
            assert mock_process.called
            args, _ = mock_process.call_args
            event, _ = args
            
            assert event.to_state == 'OCO_CREATED'
            assert event.metadata['symbol'] == 'BTC/USD'
