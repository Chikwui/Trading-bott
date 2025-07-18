"""
Metrics collection for the trading system using Prometheus.

This module provides metrics collection for circuit breakers, event bus,
and other critical components of the trading system.
"""
from typing import Dict, Optional, Type, TypeVar
from enum import Enum
import time
from prometheus_client import (
    Counter, Gauge, Histogram, Summary, start_http_server, REGISTRY
)
from prometheus_client.metrics import MetricWrapperBase

# Type variable for generic metric types
M = TypeVar('M', bound=MetricWrapperBase)

class MetricsNamespace:
    """Namespace for Prometheus metrics."""
    CIRCUIT_BREAKER = "circuit_breaker"
    EVENT_BUS = "event_bus"
    ORDER_EXECUTION = "order_execution"

class CircuitBreakerMetrics:
    """Metrics for circuit breaker pattern."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._metrics = self._initialize_metrics()
            self._initialized = True
    
    def _initialize_metrics(self) -> Dict[str, MetricWrapperBase]:
        """Initialize all circuit breaker metrics."""
        return {
            'state': Gauge(
                f'{MetricsNamespace.CIRCUIT_BREAKER}_state',
                'Current state of the circuit breaker',
                ['name'],
                states=['closed', 'open', 'half_open']
            ),
            'transitions_total': Counter(
                f'{MetricsNamespace.CIRCUIT_BREAKER}_transitions_total',
                'Total number of circuit breaker state transitions',
                ['name', 'from_state', 'to_state']
            ),
            'failures_total': Counter(
                f'{MetricsNamespace.CIRCUIT_BREAKER}_failures_total',
                'Total number of failures that triggered the circuit breaker',
                ['name']
            ),
            'successes_total': Counter(
                f'{MetricsNamespace.CIRCUIT_BREAKER}_successes_total',
                'Total number of successful executions through the circuit breaker',
                ['name']
            ),
            'latency_seconds': Histogram(
                f'{MetricsNamespace.CIRCUIT_BREAKER}_latency_seconds',
                'Execution latency through the circuit breaker',
                ['name'],
                buckets=(.005, .01, .025, .05, .075, .1, .25, .5, .75, 1.0, 2.5, 5.0, 7.5, 10.0, float('inf'))
            )
        }
    
    def record_state_change(self, name: str, from_state: str, to_state: str) -> None:
        """Record a circuit breaker state change."""
        # Update state gauge
        self._metrics['state'].labels(name=name).state(to_state.lower())
        
        # Increment transition counter
        self._metrics['transitions_total'].labels(
            name=name,
            from_state=from_state.lower(),
            to_state=to_state.lower()
        ).inc()
    
    def record_failure(self, name: str) -> None:
        """Record a failure through the circuit breaker."""
        self._metrics['failures_total'].labels(name=name).inc()
    
    def record_success(self, name: str) -> None:
        """Record a success through the circuit breaker."""
        self._metrics['successes_total'].labels(name=name).inc()
    
    def time_execution(self, name: str) -> 'Timer':
        """Time an execution through the circuit breaker."""
        return self._metrics['latency_seconds'].labels(name=name).time()


class EventBusMetrics:
    """Metrics for event bus operations."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._metrics = self._initialize_metrics()
            self._initialized = True
    
    def _initialize_metrics(self) -> Dict[str, MetricWrapperBase]:
        """Initialize all event bus metrics."""
        return {
            'events_published_total': Counter(
                f'{MetricsNamespace.EVENT_BUS}_events_published_total',
                'Total number of events published',
                ['event_type']
            ),
            'events_processed_total': Counter(
                f'{MetricsNamespace.EVENT_BUS}_events_processed_total',
                'Total number of events processed',
                ['event_type', 'status']
            ),
            'event_processing_duration_seconds': Histogram(
                f'{MetricsNamespace.EVENT_BUS}_processing_duration_seconds',
                'Time taken to process events',
                ['event_type'],
                buckets=(.001, .0025, .005, .01, .025, .05, .1, .25, .5, 1.0, 2.5, 5.0, float('inf'))
            ),
            'subscribers_gauge': Gauge(
                f'{MetricsNamespace.EVENT_BUS}_subscribers',
                'Current number of subscribers',
                ['event_type']
            ),
            'queue_size_gauge': Gauge(
                f'{MetricsNamespace.EVENT_BUS}_queue_size',
                'Current number of events in the processing queue'
            )
        }
    
    def record_event_published(self, event_type: str) -> None:
        """Record that an event was published."""
        self._metrics['events_published_total'].labels(event_type=event_type).inc()
    
    def record_event_processed(self, event_type: str, success: bool = True) -> None:
        """Record that an event was processed."""
        status = 'success' if success else 'error'
        self._metrics['events_processed_total'].labels(
            event_type=event_type,
            status=status
        ).inc()
    
    def time_processing(self, event_type: str) -> 'Histogram.Timer':
        """Time how long it takes to process an event."""
        return self._metrics['event_processing_duration_seconds']\
            .labels(event_type=event_type).time()
    
    def update_subscribers_count(self, event_type: str, count: int) -> None:
        """Update the count of subscribers for an event type."""
        self._metrics['subscribers_gauge'].labels(event_type=event_type).set(count)
    
    def update_queue_size(self, size: int) -> None:
        """Update the current queue size."""
        self._metrics['queue_size_gauge'].set(size)


def start_metrics_server(port: int = 8000) -> None:
    """Start the Prometheus metrics server."""
    try:
        start_http_server(port)
        print(f"Metrics server started on port {port}")
    except Exception as e:
        print(f"Failed to start metrics server: {e}")
        raise


def get_metrics_registry():
    """Get the Prometheus metrics registry."""
    return REGISTRY
