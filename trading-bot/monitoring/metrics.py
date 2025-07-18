"""
Metrics Collection Module

This module provides a centralized way to collect and expose metrics
for the trading system, including circuit breakers, event processing,
and system performance metrics.
"""
from typing import Dict, Optional, List, Any, Callable
from prometheus_client import (
    Counter, Gauge, Histogram, start_http_server, generate_latest, REGISTRY,
    CollectorRegistry, multiprocess, make_wsgi_app
)
from prometheus_client.core import GaugeMetricFamily, CounterMetricFamily, HistogramMetricFamily
import time
import threading
import logging
from pathlib import Path

from .config import config

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Centralized metrics collection for the trading system."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(MetricsCollector, cls).__new__(cls)
                cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, registry: Optional[CollectorRegistry] = None):
        if self._initialized:
            return
            
        self.registry = registry or REGISTRY
        self._metrics: Dict[str, Any] = {}
        self._initialized = True
        
        # Initialize core metrics
        self._init_circuit_breaker_metrics()
        self._init_event_metrics()
        self._init_system_metrics()
    
    def _init_circuit_breaker_metrics(self) -> None:
        """Initialize metrics for circuit breakers."""
        prefix = config.METRICS_PREFIX + "circuit_breaker_"
        
        self._metrics.update({
            "state": Gauge(
                f"{prefix}state",
                "State of circuit breakers (0=closed, 1=open, 2=half-open)",
                ["name"],
                registry=self.registry
            ),
            "transitions": Counter(
                f"{prefix}transitions_total",
                "Total number of circuit breaker state transitions",
                ["name", "from_state", "to_state"],
                registry=self.registry
            ),
            "latency_seconds": Histogram(
                f"{prefix}latency_seconds",
                "Circuit breaker operation latency",
                ["name", "operation"],
                buckets=config.DEFAULT_BUCKETS,
                registry=self.registry
            ),
            "failures": Counter(
                f"{prefix}failures_total",
                "Total number of circuit breaker failures",
                ["name", "error_type"],
                registry=self.registry
            ),
        })
    
    def _init_event_metrics(self) -> None:
        """Initialize metrics for event processing."""
        prefix = config.METRICS_PREFIX + "event_"
        
        self._metrics.update({
            "processed_total": Counter(
                f"{prefix}processed_total",
                "Total number of processed events",
                ["event_type", "status"],
                registry=self.registry
            ),
            "processing_duration_seconds": Histogram(
                f"{prefix}processing_duration_seconds",
                "Time spent processing events",
                ["event_type"],
                buckets=config.DEFAULT_BUCKETS,
                registry=self.registry
            ),
            "queue_size": Gauge(
                f"{prefix}queue_size",
                "Current size of the event queue",
                registry=self.registry
            ),
            "queue_duration_seconds": Histogram(
                f"{prefix}queue_duration_seconds",
                "Time events spend in queue",
                ["event_type"],
                buckets=config.DEFAULT_BUCKETS,
                registry=self.registry
            ),
        })
    
    def _init_system_metrics(self) -> None:
        """Initialize system-level metrics."""
        prefix = config.METRICS_PREFIX + "system_"
        
        self._metrics.update({
            "start_time": Gauge(
                f"{prefix}start_time_seconds",
                "Start time of the application",
                registry=self.registry
            ),
            "uptime_seconds": Gauge(
                f"{prefix}uptime_seconds",
                "Uptime of the application in seconds",
                registry=self.registry
            ),
            "memory_usage_bytes": Gauge(
                f"{prefix}memory_usage_bytes",
                "Memory usage of the application",
                ["type"],  # type: heap, rss, vms, etc.
                registry=self.registry
            ),
            "cpu_usage_percent": Gauge(
                f"{prefix}cpu_usage_percent",
                "CPU usage percentage",
                registry=self.registry
            ),
        })
        
        # Set initial values
        self._metrics["start_time"].set_to_current_time()
    
    def get_metric(self, name: str):
        """Get a metric by name."""
        if name not in self._metrics:
            raise ValueError(f"Unknown metric: {name}")
        return self._metrics[name]
    
    def update_circuit_breaker_state(self, name: str, state: int) -> None:
        """Update circuit breaker state metric."""
        self._metrics["state"].labels(name=name).set(state)
    
    def record_circuit_breaker_transition(
        self, name: str, from_state: str, to_state: str
    ) -> None:
        """Record a circuit breaker state transition."""
        self._metrics["transitions"].labels(
            name=name, from_state=from_state, to_state=to_state
        ).inc()
    
    def record_event_processing(
        self, 
        event_type: str, 
        status: str, 
        duration_seconds: float
    ) -> None:
        """Record event processing metrics."""
        self._metrics["processed_total"].labels(
            event_type=event_type, status=status
        ).inc()
        
        if status == "success":
            self._metrics["processing_duration_seconds"].labels(
                event_type=event_type
            ).observe(duration_seconds)
    
    def update_queue_size(self, size: int) -> None:
        """Update the current queue size metric."""
        self._metrics["queue_size"].set(size)
    
    def record_queue_duration(
        self, event_type: str, duration_seconds: float
    ) -> None:
        """Record time spent in queue for an event."""
        self._metrics["queue_duration_seconds"].labels(
            event_type=event_type
        ).observe(duration_seconds)
    
    def update_system_metrics(self) -> None:
        """Update system-level metrics."""
        try:
            import psutil
            process = psutil.Process()
            
            # Update memory metrics
            mem_info = process.memory_info()
            self._metrics["memory_usage_bytes"].labels(type="rss").set(mem_info.rss)
            self._metrics["memory_usage_bytes"].labels(type="vms").set(mem_info.vms)
            
            # Update CPU usage
            self._metrics["cpu_usage_percent"].set(process.cpu_percent())
            
            # Update uptime
            self._metrics["uptime_seconds"].set(time.time() - process.create_time())
            
        except ImportError:
            logger.warning("psutil not available, system metrics will not be collected")
    
    def start_metrics_server(
        self, 
        port: int = 8001, 
        addr: str = "0.0.0.0"
    ) -> threading.Thread:
        """Start a background thread with the metrics HTTP server."""
        def run_server():
            import http.server
            from prometheus_client.exposition import _bake_output, _SilentHandler
            
            class MetricsHandler(_SilentHandler):
                def do_GET(self):
                    if self.path == "/metrics":
                        output = _bake_output(self.registry, self.accept_header, None)
                        self.send_response(200)
                        self.send_header("Content-Type", output.content_type)
                        self.end_headers()
                        self.wfile.write(output.data)
                    else:
                        self.send_response(404)
                        self.end_headers()
            
            httpd = http.server.HTTPServer((addr, port), MetricsHandler)
            httpd.RequestHandlerClass.registry = self.registry
            httpd.serve_forever()
        
        thread = threading.Thread(
            target=run_server, 
            name="metrics-server",
            daemon=True
        )
        thread.start()
        logger.info(f"Started metrics server on http://{addr}:{port}/metrics")
        return thread


# Create a singleton instance
metrics = MetricsCollector()

# Export common metric names for easier access
CIRCUIT_BREAKER_STATE = "circuit_breaker_state"
CIRCUIT_BREAKER_TRANSITIONS = "circuit_breaker_transitions"
EVENT_PROCESSED_TOTAL = "event_processed_total"
EVENT_PROCESSING_DURATION = "event_processing_duration_seconds"
EVENT_QUEUE_SIZE = "event_queue_size"
EVENT_QUEUE_DURATION = "event_queue_duration_seconds"
