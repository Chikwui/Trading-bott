# Lightweight Monitoring for Trading Bot

This directory contains a lightweight monitoring solution that doesn't require Docker or virtualization. It's designed to work on any system with Python 3.7+.

## Features

- **Metrics Collection**: Collects system and application metrics using Prometheus client
- **Health Checks**: Provides health check endpoints for monitoring
- **Grafana Integration**: Compatible with Grafana Cloud or self-hosted Grafana
- **Redis Monitoring**: Optional Redis monitoring if Redis is available
- **Low Resource Usage**: Minimal impact on system performance

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements-monitoring.txt
   ```

2. **Start the Monitoring Server**:
   ```bash
   python lightweight_monitor.py
   ```

3. **Access the Web Interface**:
   - Metrics: http://localhost:8000/metrics
   - Health Check: http://localhost:8000/health
   - System Info: http://localhost:8000/system

## Grafana Setup

1. **Set up a free Grafana Cloud account** at [grafana.com](https://grafana.com/)

2. **Add Prometheus as a data source** in Grafana:
   - URL: `http://your-server-ip:8000` (or `http://localhost:8000` if running locally)
   - Set "Scrape interval" to 15s

3. **Import the Dashboard**:
   - In Grafana, go to Dashboards > New > Import
   - Upload or paste the contents of `grafana-dashboard.json`
   - Select your Prometheus data source

## Available Metrics

### System Metrics
- `system_cpu_usage_percent`: System-wide CPU usage percentage
- `system_memory_usage_percent`: System memory usage percentage
- `process_cpu_usage_percent`: Trading bot process CPU usage
- `process_memory_mb`: Trading bot process memory usage in MB

### Application Metrics
- `orders_processed_total`: Total number of orders processed
- `order_processing_seconds`: Time taken to process orders
- `active_strategies`: Number of active trading strategies
- `redis_up`: Redis connection status (1=up, 0=down)

## API Endpoints

- `GET /`: Basic information and available endpoints
- `GET /metrics`: Prometheus metrics
- `GET /health`: Health check endpoint
- `GET /system`: System and process information

## Configuration

Create a `.env` file in the monitoring directory to customize settings:

```ini
# Monitoring server settings
HOST=0.0.0.0
PORT=8000
METRICS_PORT=9100

# Redis settings (optional)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Logging
LOG_LEVEL=INFO
```

## Alerting

Set up alerts in Grafana using the following example queries:

- **High CPU Usage**: `system_cpu_usage_percent > 80`
- **High Memory Usage**: `system_memory_usage_percent > 85`
- **Redis Down**: `redis_up == 0`
- **No Orders Processed**: `rate(orders_processed_total[5m]) == 0`

## Troubleshooting

1. **Port already in use**:
   - Change the `PORT` or `METRICS_PORT` in the `.env` file

2. **Can't connect to Redis**:
   - Make sure Redis is running and accessible
   - Check the Redis connection settings in `.env`

3. **Metrics not showing up in Grafana**:
   - Verify the Prometheus data source URL is correct
   - Check if the monitoring server is running and accessible
  
## Security Considerations

- By default, the monitoring server is accessible from any network interface
- For production use, consider:
  - Running behind a reverse proxy with authentication
  - Using HTTPS
  - Restricting access with firewall rules

## License

This monitoring solution is part of the Trading Bot project and is available under the same license.
