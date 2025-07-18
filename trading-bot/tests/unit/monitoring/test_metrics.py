"""
Unit tests for the monitoring.metrics module.
"""
import time
import pytest
from unittest.mock import MagicMock, patch
from prometheus_client import REGISTRY, CollectorRegistry

from core.monitoring.metrics import (
    CircuitBreakerMetrics,
    EventBusMetrics,
    MetricsNamespace,
    start_metrics_server,
    get_metrics_registry
)

class TestCircuitBreakerMetrics:
    """Test cases for CircuitBreakerMetrics class."""
    
    @pytest.fixture
def circuit_breaker_metrics(self):
    """Create a fresh instance of CircuitBreakerMetrics for each test."""
    # Clear any existing metrics to avoid test interference
    for collector in list(REGISTRY._collector_to_names):
        REGISTRY.unregister(collector)
    return CircuitBreakerMetrics()

    def test_record_state_change(self, circuit_breaker_metrics):
        """Test recording circuit breaker state changes."""
        cb_metrics = circuit_breaker_metrics
        
        # Record state changes
        cb_metrics.record_state_change("test_cb", "CLOSED", "OPEN")
        cb_metrics.record_state_change("test_cb", "OPEN", "HALF_OPEN")
        cb_metrics.record_state_change("test_cb", "HALF_OPEN", "CLOSED")
        
        # Verify state gauge
        samples = list(cb_metrics._metrics['state']._samples())
        assert any(s[1]["name"] == "test_cb" and s[2] == 2 for s in samples)  # CLOSED state
        
        # Verify transitions counter
        samples = list(cb_metrics._metrics['transitions_total']._samples())
        assert any(
            s[1]["name"] == "test_cb" and 
            s[1]["from_state"] == "HALF_OPEN" and 
            s[1]["to_state"] == "CLOSED"
            for s in samples
        )
    
    def test_record_failure(self, circuit_breaker_metrics):
        """Test recording circuit breaker failures."""
        cb_metrics = circuit_breaker_metrics
        
        # Record failures
        for _ in range(3):
            cb_metrics.record_failure("test_cb")
        
        # Verify failure count
        samples = list(cb_metrics._metrics['failures_total']._samples())
        assert any(s[1]["name"] == "test_cb" and s[2] == 3 for s in samples)
    
    def test_record_success(self, circuit_breaker_metrics):
        """Test recording circuit breaker successes."""
        cb_metrics = circuit_breaker_metrics
        
        # Record successes
        for _ in range(5):
            cb_metrics.record_success("test_cb")
        
        # Verify success count
        samples = list(cb_metrics._metrics['successes_total']._samples())
        assert any(s[1]["name"] == "test_cb" and s[2] == 5 for s in samples)
    
    def test_time_execution(self, circuit_breaker_metrics):
        """Test timing execution through the circuit breaker."""
        cb_metrics = circuit_breaker_metrics
        
        # Time an operation
        with cb_metrics.time_execution("test_cb"):
            time.sleep(0.1)
        
        # Verify latency was recorded
        samples = list(cb_metrics._metrics['latency_seconds']._samples())
        assert any(s[1]["name"] == "test_cb" for s in samples)


class TestEventBusMetrics:
    """Test cases for EventBusMetrics class."""
    
    @pytest.fixture
def event_bus_metrics(self):
    """Create a fresh instance of EventBusMetrics for each test."""
    # Clear any existing metrics to avoid test interference
    for collector in list(REGISTRY._collector_to_names):
        REGISTRY.unregister(collector)
    return EventBusMetrics()
    
    def test_record_event_published(self, event_bus_metrics):
        """Test recording published events."""
        eb_metrics = event_bus_metrics
        
        # Record published events
        for _ in range(3):
            eb_metrics.record_event_published("TestEvent")
        
        # Verify published count
        samples = list(eb_metrics._metrics['events_published_total']._samples())
        assert any(s[1]["event_type"] == "TestEvent" and s[2] == 3 for s in samples)
    
    def test_record_event_processed(self, event_bus_metrics):
        """Test recording processed events."""
        eb_metrics = event_bus_metrics
        
        # Record processed events with success and failure
        eb_metrics.record_event_processed("TestEvent", success=True)
        eb_metrics.record_event_processed("TestEvent", success=True)
        eb_metrics.record_event_processed("TestEvent", success=False)
        
        # Verify processed counts
        samples = list(eb_metrics._metrics['events_processed_total']._samples())
        success_count = next(
            s[2] for s in samples 
            if s[1]["event_type"] == "TestEvent" and s[1]["status"] == "success"
        )
        error_count = next(
            s[2] for s in samples 
            if s[1]["event_type"] == "TestEvent" and s[1]["status"] == "error"
        )
        assert success_count == 2
        assert error_count == 1
    
    def test_time_processing(self, event_bus_metrics):
        """Test timing event processing."""
        eb_metrics = event_bus_metrics
        
        # Time event processing
        with eb_metrics.time_processing("TestEvent"):
            time.sleep(0.1)
        
        # Verify processing time was recorded
        samples = list(eb_metrics._metrics['event_processing_duration_seconds']._samples())
        assert any(s[1]["event_type"] == "TestEvent" for s in samples)
    
    def test_update_subscribers_count(self, event_bus_metrics):
        """Test updating subscriber count."""
        eb_metrics = event_bus_metrics
        
        # Update subscriber count
        eb_metrics.update_subscribers_count("TestEvent", 5)
        
        # Verify subscriber count
        samples = list(eb_metrics._metrics['subscribers_gauge']._samples())
        assert any(s[1]["event_type"] == "TestEvent" and s[2] == 5 for s in samples)
    
    def test_update_queue_size(self, event_bus_metrics):
        """Test updating queue size."""
        eb_metrics = event_bus_metrics
        
        # Update queue size
        eb_metrics.update_queue_size(10)
        
        # Verify queue size
        samples = list(eb_metrics._metrics['queue_size_gauge']._samples())
        assert samples[0][2] == 10


def test_start_metrics_server():
    """Test starting the metrics server."""
    with patch('core.monitoring.metrics.start_http_server') as mock_start_server:
        start_metrics_server(8000)
        mock_start_server.assert_called_once_with(8000)


def test_get_metrics_registry():
    """Test getting the metrics registry."""
    # Create a test registry
    test_registry = CollectorRegistry()
    
    with patch('core.monitoring.metrics.REGISTRY', test_registry):
        assert get_metrics_registry() is test_registry


class TestMetricsNamespace:
    """Test cases for MetricsNamespace class."""
    
    def test_namespace_constants(self):
        """Test that namespace constants are correctly defined."""
        assert MetricsNamespace.CIRCUIT_BREAKER == "circuit_breaker"
        assert MetricsNamespace.EVENT_BUS == "event_bus"
        assert MetricsNamespace.ORDER_EXECUTION == "order_execution"
