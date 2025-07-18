"""
Optimized metrics collection with batching and sampling support.

This module provides optimized implementations of metrics collectors that reduce
overhead in high-throughput scenarios through batching and sampling.
"""
import time
import random
import threading
from typing import Dict, List, Optional, Any, Callable, TypeVar, Generic, Union
from dataclasses import dataclass
from queue import Queue, Empty
from prometheus_client import (
    Counter, Gauge, Histogram, Summary, 
    start_http_server as prom_start_http_server,
    REGISTRY
)
from ..utils.singleton import Singleton

# Type variable for generic metrics
T = TypeVar('T')

class BatchedCollector(Generic[T]):
    """Base class for batched metrics collection."""
    
    def __init__(self, batch_size: int = 100, max_delay: float = 1.0):
        """
        Initialize the batched collector.
        
        Args:
            batch_size: Maximum number of items to collect before flushing
            max_delay: Maximum time in seconds to wait before flushing
        """
        self.batch_size = batch_size
        self.max_delay = max_delay
        self.batch: List[T] = []
        self.last_flush = time.time()
        self.lock = threading.RLock()
    
    def add(self, item: T) -> None:
        """Add an item to the current batch."""
        with self.lock:
            self.batch.append(item)
            current_time = time.time()
            
            # Flush if we've reached batch size or max delay
            if (len(self.batch) >= self.batch_size or 
                (current_time - self.last_flush) >= self.max_delay):
                self.flush()
    
    def flush(self) -> None:
        """Process the current batch of items."""
        with self.lock:
            if not self.batch:
                return
                
            self._process_batch(self.batch)
            self.batch = []
            self.last_flush = time.time()
    
    def _process_batch(self, batch: List[T]) -> None:
        """Process a batch of items. Subclasses must implement this method."""
        raise NotImplementedError


class SampledCollector(Generic[T]):
    """Wrapper for sampling metrics collection."""
    
    def __init__(self, collector: Any, sample_rate: float = 1.0):
        """
        Initialize the sampled collector.
        
        Args:
            collector: The underlying metrics collector
            sample_rate: Sampling rate between 0.0 and 1.0 (1.0 = 100%)
        """
        self.collector = collector
        self.sample_rate = max(0.0, min(1.0, sample_rate))
    
    def __getattr__(self, name: str) -> Any:
        """Delegate attribute access to the underlying collector."""
        return getattr(self.collector, name)
    
    def should_sample(self) -> bool:
        """Determine if the current operation should be sampled."""
        return random.random() < self.sample_rate


class OptimizedCircuitBreakerMetrics(metaclass=Singleton):
    """Optimized metrics collector for circuit breakers."""
    
    def __init__(self, batch_size: int = 100, sample_rate: float = 1.0):
        """
        Initialize the optimized circuit breaker metrics collector.
        
        Args:
            batch_size: Batch size for state transitions
            sample_rate: Sampling rate for latency metrics (0.0 to 1.0)
        """
        # Initialize Prometheus metrics
        self._state = Gauge(
            'circuit_breaker_state',
            'Current state of the circuit breaker',
            ['name']
        )
        
        self._transitions = Counter(
            'circuit_breaker_transitions_total',
            'Total number of state transitions',
            ['name', 'from_state', 'to_state']
        )
        
        self._failures = Counter(
            'circuit_breaker_failures_total',
            'Total number of failures',
            ['name']
        )
        
        self._successes = Counter(
            'circuit_breaker_successes_total',
            'Total number of successes',
            ['name']
        )
        
        self._latency = Histogram(
            'circuit_breaker_latency_seconds',
            'Execution latency in seconds',
            ['name'],
            buckets=(.001, .0025, .005, .01, .025, .05, .1, .25, .5, 1.0, 2.5, 5.0, 10.0)
        )
        
        # Initialize batched collectors
        self._batch_size = batch_size
        self._sample_rate = sample_rate
        
        # Use sampling for latency metrics
        self._sampled_latency = SampledCollector(self._latency, sample_rate)
    
    def record_state_change(self, name: str, from_state: str, to_state: str) -> None:
        """Record a state transition."""
        self._state.labels(name=name).set(
            0 if to_state == 'CLOSED' else 
            1 if to_state == 'OPEN' else 
            2  # HALF_OPEN
        )
        self._transitions.labels(
            name=name, 
            from_state=from_state, 
            to_state=to_state
        ).inc()
    
    def record_failure(self, name: str) -> None:
        """Record a failure."""
        self._failures.labels(name=name).inc()
    
    def record_success(self, name: str) -> None:
        """Record a success."""
        self._successes.labels(name=name).inc()
    
    def time_execution(self, name: str):
        """Time the execution of a code block."""
        if not self._sampled_latency.should_sample():
            return DummyTimer()
        return self._latency.labels(name=name).time()


class OptimizedEventBusMetrics(metaclass=Singleton):
    """Optimized metrics collector for the event bus."""
    
    def __init__(self, batch_size: int = 100, sample_rate: float = 1.0):
        """
        Initialize the optimized event bus metrics collector.
        
        Args:
            batch_size: Batch size for event processing
            sample_rate: Sampling rate for processing time metrics (0.0 to 1.0)
        """
        # Initialize Prometheus metrics
        self._events_published = Counter(
            'event_bus_events_published_total',
            'Total number of events published',
            ['event_type']
        )
        
        self._events_processed = Counter(
            'event_bus_events_processed_total',
            'Total number of events processed',
            ['event_type', 'status']
        )
        
        self._queue_size = Gauge(
            'event_bus_queue_size',
            'Current size of the event queue'
        )
        
        self._processing_time = Histogram(
            'event_bus_processing_duration_seconds',
            'Time spent processing events',
            ['event_type'],
            buckets=(.001, .0025, .005, .01, .025, .05, .1, .25, .5, 1.0, 2.5)
        )
        
        self._subscribers = Gauge(
            'event_bus_subscribers_gauge',
            'Current number of subscribers',
            ['event_type']
        )
        
        # Initialize batched collectors
        self._batch_size = batch_size
        self._sample_rate = sample_rate
        
        # Use sampling for processing time metrics
        self._sampled_processing = SampledCollector(self._processing_time, sample_rate)
    
    def record_event_published(self, event_type: str) -> None:
        """Record that an event was published."""
        self._events_published.labels(event_type=event_type).inc()
    
    def record_event_processed(self, event_type: str, success: bool = True) -> None:
        """Record that an event was processed."""
        status = 'success' if success else 'error'
        self._events_processed.labels(
            event_type=event_type, 
            status=status
        ).inc()
    
    def time_processing(self, event_type: str):
        """Time the processing of an event."""
        if not self._sampled_processing.should_sample():
            return DummyTimer()
        return self._processing_time.labels(event_type=event_type).time()
    
    def update_subscribers_count(self, event_type: str, count: int) -> None:
        """Update the count of subscribers for an event type."""
        self._subscribers.labels(event_type=event_type).set(count)
    
    def update_queue_size(self, size: int) -> None:
        """Update the current event queue size."""
        self._queue_size.set(size)


class DummyTimer:
    """Dummy context manager that does nothing, used when sampling is disabled."""
    def __enter__(self):
        pass
    
    def __exit__(self, *args):
        pass


def start_metrics_server(port: int = 8000, addr: str = '0.0.0.0') -> None:
    """Start the metrics HTTP server."""
    prom_start_http_server(port, addr)


def get_metrics_registry():
    """Get the Prometheus metrics registry."""
    return REGISTRY
