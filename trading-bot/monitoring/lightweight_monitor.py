"""
Lightweight Monitoring Solution for Trading Bot
----------------------------------------------
This script provides a monitoring solution without requiring Docker or virtualization.
It exposes Prometheus metrics and health checks via a FastAPI web server.
"""

import os
import time
import psutil
from fastapi import FastAPI, Response, status
from fastapi.responses import JSONResponse
from prometheus_client import (
    generate_latest, 
    CONTENT_TYPE_LATEST,
    Gauge, 
    Counter,
    start_http_server
)
import uvicorn
import redis
from typing import Dict, Any, Optional

# Initialize FastAPI app
app = FastAPI(title="Trading Bot Monitor")

# Initialize Redis connection (if available)
redis_client = None
try:
    redis_client = redis.Redis(host='localhost', port=6379, db=0, socket_connect_timeout=1)
    redis_client.ping()
except (redis.ConnectionError, redis.TimeoutError):
    redis_client = None

# Prometheus Metrics
SYSTEM_CPU_USAGE = Gauge('system_cpu_usage_percent', 'System CPU usage percent')
SYSTEM_MEMORY_USAGE = Gauge('system_memory_usage_percent', 'System memory usage percent')
PROCESS_CPU_USAGE = Gauge('process_cpu_usage_percent', 'Process CPU usage percent')
PROCESS_MEMORY_MB = Gauge('process_memory_mb', 'Process memory usage in MB')
REDIS_UP = Gauge('redis_up', 'Redis connection status (1=up, 0=down)')
TRADING_BOT_REQUESTS = Counter('trading_bot_requests_total', 'Total number of requests to the trading bot')

# Custom metrics for trading bot
ORDERS_PROCESSED = Counter('orders_processed_total', 'Total number of orders processed')
ORDER_PROCESSING_TIME = Gauge('order_processing_seconds', 'Time taken to process an order')
ACTIVE_STRATEGIES = Gauge('active_strategies', 'Number of active trading strategies')

# Track last update time
last_update = time.time()

@app.get('/')
async def root():
    """Root endpoint with basic information."""
    return {
        "service": "Trading Bot Monitor",
        "status": "running",
        "endpoints": {
            "metrics": "/metrics",
            "health": "/health",
            "system": "/system"
        }
    }

@app.get('/metrics')
async def metrics():
    """Expose Prometheus metrics."""
    update_metrics()
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

@app.get('/health')
async def health():
    """Health check endpoint."""
    try:
        # Check Redis connection
        redis_status = bool(redis_client and redis_client.ping())
        
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "status": "healthy",
                "redis": "connected" if redis_status else "disconnected",
                "timestamp": time.time()
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "unhealthy", "error": str(e)}
        )

@app.get('/system')
async def system_info():
    """Get system information."""
    process = psutil.Process()
    with process.oneshot():
        return {
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent
            },
            "process": {
                "cpu_percent": process.cpu_percent(),
                "memory_mb": process.memory_info().rss / 1024 / 1024,
                "threads": process.num_threads()
            },
            "redis": {
                "connected": bool(redis_client and redis_client.ping())
            }
        }

def update_metrics():
    """Update all metrics."""
    # System metrics
    SYSTEM_CPU_USAGE.set(psutil.cpu_percent())
    SYSTEM_MEMORY_USAGE.set(psutil.virtual_memory().percent)
    
    # Process metrics
    process = psutil.Process()
    with process.oneshot():
        PROCESS_CPU_USAGE.set(process.cpu_percent())
        PROCESS_MEMORY_MB.set(process.memory_info().rss / 1024 / 1024)
    
    # Redis status
    try:
        if redis_client and redis_client.ping():
            REDIS_UP.set(1)
        else:
            REDIS_UP.set(0)
    except:
        REDIS_UP.set(0)

def start_monitoring(port: int = 8000):
    """Start the monitoring server."""
    # Start Prometheus metrics server on a different port
    start_http_server(9100)
    
    # Start FastAPI server
    config = uvicorn.Config(
        app=app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
    server = uvicorn.Server(config)
    
    print(f"Monitoring server starting on http://localhost:{port}")
    print(f"Metrics available at http://localhost:9100/metrics")
    
    try:
        server.run()
    except KeyboardInterrupt:
        print("\nShutting down monitoring server...")

if __name__ == "__main__":
    start_monitoring()
