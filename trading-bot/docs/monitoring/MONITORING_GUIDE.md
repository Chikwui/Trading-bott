# Monitoring Guide

## Table of Contents
1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Quick Start](#quick-start)
4. [Detailed Setup](#detailed-setup)
   - [Prometheus](#prometheus-setup)
   - [Grafana](#grafana-setup)
   - [Alerting](#alerting-setup)
5. [Metrics Reference](#metrics-reference)
6. [Troubleshooting](#troubleshooting)
7. [Performance Tuning](#performance-tuning)

## Introduction

This guide explains how to set up and use the monitoring system for the trading bot. The system uses Prometheus for metrics collection and Grafana for visualization.

## Prerequisites

- Python 3.8+
- Docker and Docker Compose (for containerized deployment)
- Basic understanding of Prometheus and Grafana

## Quick Start

1. **Start the metrics server** in your application:
   ```python
   from core.monitoring.metrics import start_metrics_server
   
   # Start the metrics server on port 8000
   start_metrics_server(port=8000)
   ```

2. **Run the application** with metrics enabled:
   ```bash
   # For circuit breakers
   cb = CircuitBreaker(name="example", metrics_enabled=True)
   
   # For event bus
   event_bus = EventBus(metrics_enabled=True)
   ```

3. **Access metrics** at `http://localhost:8000/metrics`

## Detailed Setup

### Prometheus Setup

1. **Install Prometheus** (Docker example):
   ```yaml
   # docker-compose.yml
   version: '3'
   services:
     prometheus:
       image: prom/prometheus:latest
       ports:
         - "9090:9090"
       volumes:
         - ./prometheus.yml:/etc/prometheus/prometheus.yml
       command:
         - '--config.file=/etc/prometheus/prometheus.yml'
   ```

2. **Configure Prometheus** (`prometheus.yml`):
   ```yaml
   global:
     scrape_interval: 15s
     evaluation_interval: 15s
   
   scrape_configs:
     - job_name: 'trading-bot'
       static_configs:
         - targets: ['host.docker.internal:8000']  # Use your application's host
   ```

3. **Start Prometheus**:
   ```bash
   docker-compose up -d prometheus
   ```

### Grafana Setup

1. **Install Grafana** (Docker example):
   ```yaml
   # docker-compose.yml
   grafana:
     image: grafana/grafana:latest
     ports:
       - "3000:3000"
     volumes:
       - grafana-storage:/var/lib/grafana
     depends_on:
       - prometheus
   
   volumes:
     grafana-storage:
   ```

2. **Start Grafana**:
   ```bash
   docker-compose up -d grafana
   ```

3. **Configure Data Source**:
   - Access Grafana at `http://localhost:3000`
   - Add Prometheus as a data source (URL: `http://prometheus:9090`)
   - Import the provided dashboard (see next section)

### Importing Dashboards

1. **Download the dashboard JSON** from `deploy/grafana/dashboards/trading_system.json`
2. In Grafana:
   - Navigate to Dashboards > Import
   - Upload the JSON file
   - Select the Prometheus data source
   - Click "Import"

## Metrics Reference

### Circuit Breaker Metrics

| Metric Name | Type | Description |
|-------------|------|-------------|
| `circuit_breaker_state` | Gauge | Current state (0=CLOSED, 1=OPEN, 2=HALF_OPEN) |
| `circuit_breaker_transitions_total` | Counter | Total state transitions |
| `circuit_breaker_failures_total` | Counter | Total failures |
| `circuit_breaker_successes_total` | Counter | Total successes |
| `circuit_breaker_latency_seconds` | Histogram | Execution latency |

### Event Bus Metrics

| Metric Name | Type | Description |
|-------------|------|-------------|
| `event_bus_events_published_total` | Counter | Total events published |
| `event_bus_events_processed_total` | Counter | Total events processed |
| `event_bus_queue_size` | Gauge | Current queue size |
| `event_bus_processing_duration_seconds` | Histogram | Event processing time |
| `event_bus_subscribers_gauge` | Gauge | Current subscriber count |

## Alerting Setup

### Alert Rules

1. **Circuit Breaker Tripped**:
   ```yaml
   - alert: CircuitBreakerTripped
     expr: circuit_breaker_state == 1  # OPEN state
     for: 5m
     labels:
       severity: critical
     annotations:
       summary: "Circuit breaker {{ $labels.name }} is OPEN"
       description: "Circuit breaker {{ $labels.name }} has tripped to OPEN state"
   ```

2. **High Event Processing Latency**:
   ```yaml
   - alert: HighEventProcessingLatency
     expr: histogram_quantile(0.99, sum(rate(event_bus_processing_duration_seconds_bucket[5m])) by (le)) > 1
     for: 10m
     labels:
       severity: warning
     annotations:
       summary: "High event processing latency detected"
       description: "99th percentile event processing latency is {{ $value }}s"
   ```

### Notification Channels

1. **Email**
   - Configure in Grafana UI: Alerting > Notification channels > New channel
   - Type: Email
   - Add recipient emails

2. **Slack**
   - Create a Slack webhook
   - In Grafana: Add notification channel > Type: Slack
   - Add webhook URL and channel

## Troubleshooting

### Metrics Not Appearing
1. Verify the metrics server is running
2. Check if metrics are enabled in your components
3. Check Prometheus targets page: `http://<prometheus>:9090/targets`
4. Check application logs for errors

### High Resource Usage
1. Check for high-cardinality labels
2. Reduce scrape frequency if needed
3. Disable non-essential metrics

## Performance Tuning

### Reducing Overhead
1. **Sampling**: For high-frequency metrics, implement sampling:
   ```python
   # Example: Sample 10% of events
   import random
   
   if random.random() < 0.1:  # 10% sampling
       eb_metrics.record_event_processed("HighVolumeEvent")
   ```

2. **Batching**: For high-throughput scenarios, batch updates:
   ```python
   # Example: Batch updates
   class BatchedMetrics:
       def __init__(self, metrics, batch_size=100):
           self.metrics = metrics
           self.batch_size = batch_size
           self.batch = []
       
       def record_event(self, event_type):
           self.batch.append(event_type)
           if len(self.batch) >= self.batch_size:
               self._flush()
       
       def _flush(self):
           # Process batch
           for event_type in self.batch:
               self.metrics.record_event_processed(event_type)
           self.batch = []
   ```

3. **Label Optimization**:
   - Avoid high-cardinality labels
   - Use enums or bounded sets for label values
   - Consider hashing or bucketing for high-cardinality data

### Resource Limits

1. **Prometheus Configuration**:
   ```yaml
   # prometheus.yml
   storage:
     tsdb:
       retention: 15d  # Reduce retention period
   ```

2. **Grafana Configuration**:
   ```bash
   # grafana.ini
   [dashboards]
   min_refresh_interval = 5s  # Increase minimum refresh interval
   ```

## Maintenance

### Backup and Restore

1. **Backup Grafana Dashboards**:
   ```bash
   # Export all dashboards
   curl -s http://admin:admin@localhost:3000/api/search?query=& | jq -r '.[] | .uri' | xargs -I{} curl -s http://admin:admin@localhost:3000/api/dashboards/{} | jq > all_dashboards.json
   ```

2. **Restore Dashboards**:
   ```bash
   # Import dashboard
   curl -X POST -H "Content-Type: application/json" -d @dashboard.json http://admin:admin@localhost:3000/api/dashboards/db
   ```

### Upgrading

1. Backup all configurations and dashboards
2. Check release notes for breaking changes
3. Test in staging environment first
4. Follow rolling update strategy for production
