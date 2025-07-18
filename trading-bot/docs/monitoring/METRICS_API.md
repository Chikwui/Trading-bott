# Metrics API Documentation

## Overview
The metrics system provides real-time monitoring capabilities for the trading bot, focusing on circuit breakers and event bus operations. It uses Prometheus for metrics collection and exposure.

## Table of Contents
1. [Core Concepts](#core-concepts)
2. [CircuitBreakerMetrics](#circuitbreakermetrics)
3. [EventBusMetrics](#eventbusmetrics)
4. [Configuration](#configuration)
5. [Performance Considerations](#performance-considerations)
6. [Best Practices](#best-practices)

## Core Concepts

### Metrics Types
- **Counters**: Monotonically increasing counters (e.g., number of events processed)
- **Gauges**: Point-in-time measurements (e.g., current queue size)
- **Histograms**: Sample observations with configurable buckets (e.g., request latencies)
- **Summaries**: Similar to histograms but with client-side quantiles

### Labeling
Metrics support labels for multi-dimensional data. Common labels include:
- `name`: Identifier for the circuit breaker or event type
- `event_type`: Type of event being processed
- `status`: Success/error status of operations

## CircuitBreakerMetrics

### Initialization
```python
from core.monitoring.metrics import CircuitBreakerMetrics

# Get the singleton instance
cb_metrics = CircuitBreakerMetrics()
```

### Methods

#### `record_state_change(name: str, from_state: str, to_state: str) -> None`
Record a state transition for a circuit breaker.

**Parameters:**
- `name`: Circuit breaker identifier
- `from_state`: Previous state ("CLOSED", "OPEN", "HALF_OPEN")
- `to_state`: New state ("CLOSED", "OPEN", "HALF_OPEN")

**Example:**
```python
cb_metrics.record_state_change("order_execution", "CLOSED", "OPEN")
```

#### `record_failure(name: str) -> None`
Record a failure through a circuit breaker.

**Parameters:**
- `name`: Circuit breaker identifier

#### `record_success(name: str) -> None`
Record a successful operation through a circuit breaker.

**Parameters:**
- `name`: Circuit breaker identifier

#### `time_execution(name: str) -> ContextManager`
Time the execution of a code block.

**Parameters:**
- `name`: Circuit breaker identifier

**Example:**
```python
with cb_metrics.time_execution("order_execution"):
    # Your code here
    pass
```

## EventBusMetrics

### Initialization
```python
from core.monitoring.metrics import EventBusMetrics

# Get the singleton instance
eb_metrics = EventBusMetrics()
```

### Methods

#### `record_event_published(event_type: str) -> None`
Record that an event was published.

**Parameters:**
- `event_type`: Type of the published event

#### `record_event_processed(event_type: str, success: bool = True) -> None`
Record that an event was processed.

**Parameters:**
- `event_type`: Type of the processed event
- `success`: Whether processing was successful

#### `time_processing(event_type: str) -> ContextManager`
Time the processing of an event.

**Example:**
```python
with eb_metrics.time_processing("OrderCreated"):
    # Process the event
    pass
```

#### `update_subscribers_count(event_type: str, count: int) -> None`
Update the count of subscribers for an event type.

**Parameters:**
- `event_type`: Type of the event
- `count`: Current number of subscribers

#### `update_queue_size(size: int) -> None`
Update the current event queue size.

**Parameters:**
- `size`: Current number of events in the queue

## Configuration

### Enabling/Disabling Metrics
Metrics collection can be disabled for performance-critical sections:

```python
# When creating a circuit breaker
cb = CircuitBreaker(name="example", metrics_enabled=False)

# When creating an event bus
event_bus = EventBus(metrics_enabled=True)
```

### Prometheus Endpoint
Metrics are exposed on `/metrics` endpoint. To start the metrics server:

```python
from core.monitoring.metrics import start_metrics_server

# Start server on port 8000
start_metrics_server(port=8000)
```

## Performance Considerations

### High-Volume Scenarios
For high-volume event processing, consider:
1. Increasing the Prometheus scrape interval
2. Using sampling for high-frequency metrics
3. Disabling non-critical metrics in hot code paths

### Memory Usage
Each unique combination of metric name and label values creates a new time series. Be cautious with:
- High-cardinality labels (e.g., user IDs)
- Unbounded label values
- Frequent creation of new label combinations

## Best Practices

### Naming Conventions
- Use `_total` suffix for counters
- Use `_seconds` for time-based metrics
- Use `_bytes` for byte sizes
- Use `_ratio` for ratios between 0 and 1

### Label Cardinality
- Keep label cardinality low (< 100 values per label)
- Avoid user IDs, email addresses, etc. as label values
- Use hashing or bucketing for high-cardinality data

### Metric Design
- Focus on metrics that drive business decisions
- Prefer fewer, more meaningful metrics over many noisy ones
- Document the meaning and interpretation of each metric

### Alerting
- Set up alerts for critical metrics
- Use meaningful alert thresholds
- Include runbook links in alert annotations

## Example Dashboard Queries

### Circuit Breaker State
```promql
# Current state of all circuit breakers
circuit_breaker_state

# Rate of failures per circuit breaker
rate(circuit_breaker_failures_total[5m])
```

### Event Bus Metrics
```promql
# Events published per second by type
rate(event_bus_events_published_total[1m])

# Current queue size
event_bus_queue_size

# Processing time percentiles
histogram_quantile(0.99, sum(rate(event_bus_processing_duration_seconds_bucket[1m])) by (le, event_type))
```

## Troubleshooting

### Common Issues
1. **Missing Metrics**: Ensure metrics are enabled and the Prometheus server is scraping the endpoint
2. **High Cardinality**: Check for unbounded label values if memory usage is high
3. **Performance Impact**: If metrics collection is slow, consider disabling non-critical metrics

### Debugging
1. Check the `/metrics` endpoint for expected metrics
2. Verify Prometheus is properly configured to scrape the endpoint
3. Check application logs for metric-related errors
