"""
Integration tests for metrics with CircuitBreaker and EventBus.
"""
import asyncio
import time
from unittest.mock import AsyncMock, patch
import pytest
from prometheus_client import REGISTRY

from core.utils.event_bus import EventBus, CircuitBreaker
from core.monitoring.metrics import (
    CircuitBreakerMetrics,
    EventBusMetrics,
    MetricsNamespace
)

@pytest.fixture
def clean_metrics():
    """Clean up metrics before and after each test."""
    # Clear any existing metrics before test
    for collector in list(REGISTRY._collector_to_names):
        REGISTRY.unregister(collector)
    
    yield  # Test runs here
    
    # Clean up after test
    for collector in list(REGISTRY._collector_to_names):
        REGISTRY.unregister(collector)

@pytest.mark.asyncio
async def test_circuit_breaker_metrics_integration(clean_metrics):
    """Test that CircuitBreaker properly records metrics."""
    # Create a circuit breaker with metrics enabled
    cb = CircuitBreaker(
        name="test_cb",
        failure_threshold=2,
        recovery_timeout=0.1,
        half_open_success_threshold=1,
        metrics_enabled=True
    )
    
    # Create a mock function that will fail twice then succeed
    mock_func = AsyncMock(side_effect=[Exception("Fail"), Exception("Fail"), "Success"])
    
    # First call - should fail and increment failure count
    with pytest.raises(Exception):
        await cb.execute(mock_func)
    
    # Second call - should trip the circuit to OPEN
    with pytest.raises(Exception):
        await cb.execute(mock_func)
    
    # Wait for recovery
    await asyncio.sleep(0.15)
    
    # Third call - should be in HALF_OPEN state and succeed
    result = await cb.execute(mock_func)
    assert result == "Success"
    
    # Verify metrics were recorded
    cb_metrics = CircuitBreakerMetrics()
    
    # Check failure count
    samples = list(cb_metrics._metrics['failures_total']._samples())
    assert any(s[1]["name"] == "test_cb" and s[2] == 2 for s in samples)
    
    # Check success count
    samples = list(cb_metrics._metrics['successes_total']._samples())
    assert any(s[1]["name"] == "test_cb" and s[2] == 1 for s in samples)
    
    # Check state transitions
    samples = list(cb_metrics._metrics['transitions_total']._samples())
    assert any(
        s[1]["name"] == "test_cb" and 
        s[1]["from_state"] == "CLOSED" and 
        s[1]["to_state"] == "OPEN"
        for s in samples
    )
    assert any(
        s[1]["name"] == "test_cb" and 
        s[1]["from_state"] == "OPEN" and 
        s[1]["to_state"] == "HALF_OPEN"
        for s in samples
    )
    assert any(
        s[1]["name"] == "test_cb" and 
        s[1]["from_state"] == "HALF_OPEN" and 
        s[1]["to_state"] == "CLOSED"
        for s in samples
    )

@pytest.mark.asyncio
async def test_event_bus_metrics_integration(clean_metrics):
    """Test that EventBus properly records metrics."""
    # Create an event bus with metrics enabled
    event_bus = EventBus(max_queue_size=10, metrics_enabled=True)
    
    # Create a test event
    @dataclass
    class TestEvent:
        data: str
    
    # Create a mock handler
    mock_handler = AsyncMock()
    
    # Subscribe to the test event
    unsubscribe = event_bus.subscribe(TestEvent, mock_handler)
    
    # Start the event bus
    await event_bus.start()
    
    try:
        # Publish some events
        for i in range(3):
            await event_bus.publish(TestEvent(f"test_{i}"))
        
        # Give the event bus time to process
        await asyncio.sleep(0.1)
        
        # Verify the handler was called for each event
        assert mock_handler.await_count == 3
        
        # Get the metrics instance
        eb_metrics = EventBusMetrics()
        
        # Check published events count
        samples = list(eb_metrics._metrics['events_published_total']._samples())
        assert any(
            s[1]["event_type"] == "TestEvent" and s[2] == 3 
            for s in samples
        )
        
        # Check processed events count (success)
        samples = list(eb_metrics._metrics['events_processed_total']._samples())
        assert any(
            s[1]["event_type"] == "TestEvent" and 
            s[1]["status"] == "success" and 
            s[2] == 3
            for s in samples
        )
        
        # Check queue size (should be 0 after processing)
        samples = list(eb_metrics._metrics['queue_size_gauge']._samples())
        assert samples[0][2] == 0
        
        # Check subscriber count
        samples = list(eb_metrics._metrics['subscribers_gauge']._samples())
        assert any(
            s[1]["event_type"] == "TestEvent" and s[2] == 1 
            for s in samples
        )
        
    finally:
        # Clean up
        await event_bus.stop()

@pytest.mark.asyncio
async def test_event_bus_error_metrics(clean_metrics):
    """Test that EventBus records error metrics when handlers fail."""
    # Create an event bus with metrics enabled
    event_bus = EventBus(max_queue_size=10, metrics_enabled=True)
    
    # Create a test event
    @dataclass
    class TestEvent:
        data: str
    
    # Create a mock handler that fails
    async def failing_handler(event):
        raise ValueError("Handler failed")
    
    # Subscribe to the test event
    unsubscribe = event_bus.subscribe(TestEvent, failing_handler)
    
    # Start the event bus
    await event_bus.start()
    
    try:
        # Publish an event
        await event_bus.publish(TestEvent("test"))
        
        # Give the event bus time to process
        await asyncio.sleep(0.1)
        
        # Get the metrics instance
        eb_metrics = EventBusMetrics()
        
        # Check error count
        samples = list(eb_metrics._metrics['events_processed_total']._samples())
        assert any(
            s[1]["event_type"] == "TestEvent" and 
            s[1]["status"] == "error" and 
            s[2] == 1
            for s in samples
        )
        
    finally:
        # Clean up
        await event_bus.stop()

@pytest.mark.asyncio
async def test_metrics_disabled(clean_metrics):
    """Test that no metrics are recorded when metrics are disabled."""
    # Create a circuit breaker with metrics disabled
    cb = CircuitBreaker(
        name="test_cb_disabled",
        failure_threshold=1,
        metrics_enabled=False
    )
    
    # Create a mock function that will fail
    mock_func = AsyncMock(side_effect=Exception("Fail"))
    
    # Call the circuit breaker
    with pytest.raises(Exception):
        await cb.execute(mock_func)
    
    # Verify no metrics were recorded
    for metric in [
        'circuit_breaker_failures_total',
        'circuit_breaker_successes_total',
        'circuit_breaker_transitions_total'
    ]:
        assert f'{MetricsNamespace.CIRCUIT_BREAKER}_{metric}' not in str(REGISTRY._names_to_collectors)
