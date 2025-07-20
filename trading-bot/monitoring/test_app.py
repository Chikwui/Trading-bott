"""
Enhanced monitoring test application with Prometheus metrics and additional features.
"""
from fastapi import FastAPI, Request, Response, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import logging
import psutil
import platform
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from prometheus_client import (
    generate_latest, 
    CONTENT_TYPE_LATEST,
    Counter, 
    Gauge, 
    Histogram,
    Summary,
    CollectorRegistry,
    ProcessCollector
)
import os
import socket

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('monitoring.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Trading Bot Monitoring System",
    description="Real-time monitoring and metrics for the AI Trading Bot",
    version="1.0.0"
)

# Prometheus metrics setup
registry = CollectorRegistry()
process_collector = ProcessCollector()
registry.register(process_collector)

# Define metrics
REQUEST_COUNTER = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code'],
    registry=registry
)
REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency in seconds',
    ['method', 'endpoint'],
    registry=registry
)
REQUEST_IN_PROGRESS = Gauge(
    'http_requests_in_progress',
    'Number of HTTP requests in progress',
    ['method', 'endpoint'],
    registry=registry
)
SYSTEM_CPU_USAGE = Gauge(
    'system_cpu_usage_percent',
    'System CPU usage percentage',
    registry=registry
)
SYSTEM_MEMORY_USAGE = Gauge(
    'system_memory_usage_percent',
    'System memory usage percentage',
    registry=registry
)
SYSTEM_DISK_USAGE = Gauge(
    'system_disk_usage_percent',
    'System disk usage percentage',
    registry=registry
)

# Track uptime
START_TIME = datetime.utcnow()

# Middleware for request monitoring
@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    """Middleware to monitor and log all incoming requests."""
    method = request.method
    endpoint = request.url.path
    
    # Track request start time
    start_time = datetime.utcnow()
    
    # Increment in-progress counter
    REQUEST_IN_PROGRESS.labels(method=method, endpoint=endpoint).inc()
    
    try:
        # Process the request
        response = await call_next(request)
        status_code = response.status_code
        
        # Record metrics
        REQUEST_COUNTER.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code
        ).inc()
        
        return response
        
    except Exception as e:
        # Handle exceptions and record error metrics
        status_code = 500
        REQUEST_COUNTER.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code
        ).inc()
        logger.error(f"Request failed: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=status_code,
            content={"status": "error", "message": str(e)}
        )
        
    finally:
        # Record request duration and decrement in-progress counter
        duration = (datetime.utcnow() - start_time).total_seconds()
        REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)
        REQUEST_IN_PROGRESS.labels(method=method, endpoint=endpoint).dec()

# System information endpoint
@app.get("/system/info", response_model=Dict[str, Any])
async def system_info():
    """Get system information and resource usage."""
    # Update system metrics
    update_system_metrics()
    
    return {
        "status": "ok",
        "system": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "uptime_seconds": (datetime.utcnow() - START_TIME).total_seconds(),
            "cpu_usage_percent": psutil.cpu_percent(),
            "memory_usage_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage('/').percent,
            "process": {
                "pid": os.getpid(),
                "create_time": psutil.Process().create_time(),
                "memory_info": dict(psutil.Process().memory_info()._asdict()),
                "cpu_percent": psutil.Process().cpu_percent()
            }
        },
        "timestamp": datetime.utcnow().isoformat()
    }

def update_system_metrics():
    """Update system metrics."""
    try:
        # Update CPU usage
        cpu_percent = psutil.cpu_percent()
        SYSTEM_CPU_USAGE.set(cpu_percent)
        
        # Update memory usage
        memory = psutil.virtual_memory()
        SYSTEM_MEMORY_USAGE.set(memory.percent)
        
        # Update disk usage
        disk = psutil.disk_usage('/')
        SYSTEM_DISK_USAGE.set(disk.percent)
        
    except Exception as e:
        logger.error(f"Failed to update system metrics: {str(e)}")

# Metrics endpoint for Prometheus
@app.get("/metrics")
async def metrics():
    """Expose Prometheus metrics."""
    update_system_metrics()
    return Response(
        content=generate_latest(registry),
        media_type=CONTENT_TYPE_LATEST
    )

# Root endpoint
@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with system information."""
    update_system_metrics()
    return {
        "status": "ok",
        "service": "Trading Bot Monitoring System",
        "version": "1.0.0",
        "uptime_seconds": (datetime.utcnow() - START_TIME).total_seconds(),
        "endpoints": {
            "health": "/health",
            "metrics": "/metrics",
            "system_info": "/system/info"
        },
        "timestamp": datetime.utcnow().isoformat()
    }

# Health check endpoint
@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Health check endpoint with system status."""
    update_system_metrics()
    
    # Basic health checks
    checks = {
        "database_connected": True,  # Placeholder for actual DB check
        "disk_space_ok": psutil.disk_usage('/').percent < 90,
        "memory_ok": psutil.virtual_memory().percent < 90,
        "cpu_ok": psutil.cpu_percent() < 90
    }
    
    # Determine overall status
    status_code = 200 if all(checks.values()) else 503
    
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "healthy" if all(checks.values()) else "degraded",
            "checks": checks,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: Exception):
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={"status": "error", "message": "Resource not found"}
    )

@app.exception_handler(500)
async def server_error_handler(request: Request, exc: Exception):
    """Handle 500 errors."""
    logger.error(f"Server error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "Internal server error"}
    )

if __name__ == "__main__":
    logger.info("Starting enhanced monitoring server...")
    
    # Initial system metrics update
    update_system_metrics()
    
    # Start the server
    uvicorn.run(
        "test_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        workers=1
    )
