"""
Metrics Collection and Monitoring Module

This module provides a comprehensive metrics collection system for the trading platform,
including trading metrics, system performance, and integration with Prometheus.
"""
from __future__ import annotations

import asyncio
import logging
import sys
import time
import threading
from collections import defaultdict, deque
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import (
    Any, AsyncGenerator, Callable, Dict, Iterator, List, Optional, 
    Set, Tuple, Type, TypeVar, Union, cast, overload
)

import numpy as np
from prometheus_client import (
    Counter, Gauge, Histogram, REGISTRY, CollectorRegistry, 
    generate_latest, make_wsgi_app, start_http_server
)
from prometheus_client.core import (
    CounterMetricFamily, GaugeMetricFamily, HistogramMetricFamily, Metric
)

from .config import config
from .circuit_breaker import CircuitBreaker, CircuitBreakerConfig, CircuitState

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('monitoring.log')
    ]
)
logger = logging.getLogger(__name__)

# Type aliases
T = TypeVar('T')
MetricValue = Union[int, float, str, bool]

class MetricError(Exception):
    """Base exception for metrics-related errors."""
    def __init__(
        self, 
        message: str, 
        metric_name: Optional[str] = None, 
        labels: Optional[Dict[str, str]] = None
    ):
        self.message = message
        self.metric_name = metric_name
        self.labels = labels or {}
        super().__init__(message)

    def __str__(self) -> str:
        if self.metric_name:
            return f"{self.message} (metric: {self.metric_name}, labels: {self.labels})"
        return self.message

class MetricsCollector:
    """Centralized metrics collection for the trading system.
    
    This class implements the Singleton pattern to ensure only one instance exists.
    It provides thread-safe metrics collection and exposure via Prometheus.
    """
    _instance: Optional['MetricsCollector'] = None
    _lock = threading.Lock()
    _initialized: bool = False
    
    def __init__(self, registry: Optional[CollectorRegistry] = None) -> None:
        """Initialize the metrics collector.
        
        Args:
            registry: Optional custom Prometheus registry to use.
        """
        if self._initialized:
            return
            
        try:
            self.registry = registry or REGISTRY
            self._metrics = {}
            
            # Track initialization status
            self._start_time = time.time()
            self._initialized = True
            
            # Initialize core metrics
            self._init_circuit_breaker_metrics()
            self._init_event_metrics()
            self._init_system_metrics()
            self._init_error_metrics()
            self._init_trading_metrics()
            
            logger.info("Metrics collector initialized successfully")
            
        except Exception as e:
            self._initialized = False
            logger.critical("Failed to initialize metrics collector: %s", str(e), exc_info=True)
            raise MetricError("Failed to initialize metrics collector") from e
    
    def __new__(cls):
        """Create or return the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(MetricsCollector, cls).__new__(cls)
        return cls._instance
    
    def _init_error_metrics(self) -> None:
        """Initialize error tracking metrics."""
        self.error_counter = Counter(
            f"{config.METRICS_PREFIX}errors_total",
            "Total number of errors by type",
            ["error_type", "severity"]
        )
        
        self.exception_counter = Counter(
            f"{config.METRICS_PREFIX}exceptions_total",
            "Total number of exceptions by type",
            ["exception_type", "module", "function"]
        )
        
        self.error_rate = Gauge(
            f"{config.METRICS_PREFIX}error_rate",
            "Error rate (errors per second)"
        )
    
    def _init_circuit_breaker_metrics(self) -> None:
        """Initialize metrics for circuit breakers."""
        self.circuit_breaker_state = Gauge(
            f"{config.METRICS_PREFIX}circuit_breaker_state",
            "Current state of circuit breakers (0=CLOSED, 1=OPEN, 2=HALF_OPEN)",
            ["circuit"]
        )
        
        self.circuit_breaker_failures = Counter(
            f"{config.METRICS_PREFIX}circuit_breaker_failures_total",
            "Total number of failures for circuit breakers",
            ["circuit"]
        )
        
        self.circuit_breaker_successes = Counter(
            f"{config.METRICS_PREFIX}circuit_breaker_successes_total",
            "Total number of successful operations for circuit breakers",
            ["circuit"]
        )
        
        self.circuit_breaker_state_changes = Counter(
            f"{config.METRICS_PREFIX}circuit_breaker_state_changes_total",
            "Total number of state changes for circuit breakers",
            ["circuit", "from_state", "to_state"]
        )
    
    def _init_event_metrics(self) -> None:
        """Initialize metrics for event processing with comprehensive tracking."""
        self.event_processing_time = Histogram(
            f"{config.METRICS_PREFIX}event_processing_seconds",
            "Time spent processing events",
            ["event_type", "status"],
            buckets=config.DEFAULT_BUCKETS
        )
        
        self.event_queue_size = Gauge(
            f"{config.METRICS_PREFIX}event_queue_size",
            "Current size of event queues",
            ["queue"]
        )
        
        self.event_queue_duration = Histogram(
            f"{config.METRICS_PREFIX}event_queue_duration_seconds",
            "Time events spend in queues",
            ["event_type"],
            buckets=config.DEFAULT_BUCKETS
        )
    
    def _init_system_metrics(self) -> None:
        """Initialize system-level metrics."""
        self.system_cpu_usage = Gauge(
            f"{config.METRICS_PREFIX}system_cpu_usage_percent",
            "Current system CPU usage percentage"
        )
        
        self.system_memory_usage = Gauge(
            f"{config.METRICS_PREFIX}system_memory_usage_bytes",
            "Current system memory usage in bytes"
        )
        
        self.process_cpu_usage = Gauge(
            f"{config.METRICS_PREFIX}process_cpu_usage_percent",
            "Current process CPU usage percentage"
        )
        
        self.process_memory_usage = Gauge(
            f"{config.METRICS_PREFIX}process_memory_usage_bytes",
            "Current process memory usage in bytes"
        )
        
        self.uptime = Gauge(
            f"{config.METRICS_PREFIX}uptime_seconds",
            "Process uptime in seconds"
        )
    
    def _init_trading_metrics(self) -> None:
        """Initialize trading-specific metrics."""
        # Trade metrics
        self.trade_pnl = Gauge(
            f"{config.METRICS_PREFIX}trade_pnl",
            "Profit and loss for trades",
            ["symbol", "direction"]
        )
        
        self.trade_pnl_percentage = Gauge(
            f"{config.METRICS_PREFIX}trade_pnl_percent",
            "Profit and loss percentage for trades",
            ["symbol", "direction"]
        )
        
        self.trade_duration = Histogram(
            f"{config.METRICS_PREFIX}trade_duration_seconds",
            "Duration of trades",
            ["symbol", "direction"],
            buckets=config.DEFAULT_BUCKETS
        )
        
        self.trade_slippage = Histogram(
            f"{config.METRICS_PREFIX}trade_slippage",
            "Slippage for trades",
            ["symbol", "direction"],
            buckets=config.DEFAULT_BUCKETS
        )
        
        # Order metrics
        self.order_fill_rate = Gauge(
            f"{config.METRICS_PREFIX}order_fill_rate",
            "Fill rate for orders",
            ["symbol", "order_type"]
        )
        
        self.order_latency = Histogram(
            f"{config.METRICS_PREFIX}order_latency_seconds",
            "Latency for order processing",
            ["symbol", "order_type"],
            buckets=config.DEFAULT_BUCKETS
        )
        
        # Risk metrics
        self.drawdown = Gauge(
            f"{config.METRICS_PREFIX}drawdown",
            "Current drawdown percentage"
        )
        
        self.max_drawdown = Gauge(
            f"{config.METRICS_PREFIX}max_drawdown",
            "Maximum drawdown percentage"
        )
        
        self.sharpe_ratio = Gauge(
            f"{config.METRICS_PREFIX}sharpe_ratio",
            "Sharpe ratio"
        )
        
        self.sortino_ratio = Gauge(
            f"{config.METRICS_PREFIX}sortino_ratio",
            "Sortino ratio"
        )
        
        self.win_rate = Gauge(
            f"{config.METRICS_PREFIX}win_rate",
            "Win rate for trades",
            ["symbol", "timeframe"]
        )
    
    def record_trade_pnl(
        self, 
        symbol: str, 
        pnl: float, 
        pnl_pct: float, 
        direction: str,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record trade PnL metrics.
        
        Args:
            symbol: Trading symbol
            pnl: Profit/loss amount
            pnl_pct: Profit/loss percentage
            direction: Trade direction (buy/sell)
            tags: Additional tags for the metric
        """
        try:
            self.trade_pnl.labels(symbol=symbol, direction=direction).set(pnl)
            self.trade_pnl_percentage.labels(symbol=symbol, direction=direction).set(pnl_pct)
            
            if tags:
                # Add any additional tagged metrics here
                pass
                
        except Exception as e:
            logger.error("Failed to record trade PnL metrics: %s", str(e), exc_info=True)
            self.record_error("trade_metrics_error", "Failed to record trade PnL")
    
    def record_drawdown(self, current_drawdown: float, max_drawdown: float) -> None:
        """Record drawdown metrics.
        
        Args:
            current_drawdown: Current drawdown percentage
            max_drawdown: Maximum drawdown percentage
        """
        try:
            self.drawdown.set(current_drawdown)
            self.max_drawdown.set(max_drawdown)
        except Exception as e:
            logger.error("Failed to record drawdown metrics: %s", str(e), exc_info=True)
            self.record_error("drawdown_metrics_error", "Failed to record drawdown")
    
    def record_trade_duration(
        self, 
        symbol: str, 
        direction: str, 
        duration_seconds: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record trade duration metrics.
        
        Args:
            symbol: Trading symbol
            direction: Trade direction (buy/sell)
            duration_seconds: Duration of the trade in seconds
            tags: Additional tags for the metric
        """
        try:
            self.trade_duration.labels(
                symbol=symbol, 
                direction=direction
            ).observe(duration_seconds)
            
            if tags:
                # Add any additional tagged metrics here
                pass
                
        except Exception as e:
            logger.error("Failed to record trade duration: %s", str(e), exc_info=True)
            self.record_error("trade_duration_error", "Failed to record trade duration")
    
    def record_slippage(
        self, 
        symbol: str, 
        direction: str, 
        slippage: float,
        price: Optional[float] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record slippage metrics.
        
        Args:
            symbol: Trading symbol
            direction: Trade direction (buy/sell)
            slippage: Slippage amount
            price: Optional price for percentage calculation
            tags: Additional tags for the metric
        """
        try:
            self.trade_slippage.labels(
                symbol=symbol, 
                direction=direction
            ).observe(slippage)
            
            if price is not None and price != 0:
                slippage_pct = (slippage / price) * 100
                # Could add a separate metric for percentage slippage if needed
                
            if tags:
                # Add any additional tagged metrics here
                pass
                
        except Exception as e:
            logger.error("Failed to record slippage: %s", str(e), exc_info=True)
            self.record_error("slippage_metrics_error", "Failed to record slippage")
    
    def record_win_rate(
        self, 
        symbol: str, 
        timeframe: str, 
        win_rate: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record win rate metrics.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe for the win rate calculation
            win_rate: Win rate (0-1)
            tags: Additional tags for the metric
        """
        try:
            self.win_rate.labels(
                symbol=symbol, 
                timeframe=timeframe
            ).set(win_rate * 100)  # Store as percentage
            
            if tags:
                # Add any additional tagged metrics here
                pass
                
        except Exception as e:
            logger.error("Failed to record win rate: %s", str(e), exc_info=True)
            self.record_error("win_rate_metrics_error", "Failed to record win rate")
    
    def record_sharpe_ratio(self, ratio: float) -> None:
        """Record Sharpe ratio.
        
        Args:
            ratio: Sharpe ratio value
        """
        try:
            self.sharpe_ratio.set(ratio)
        except Exception as e:
            logger.error("Failed to record Sharpe ratio: %s", str(e), exc_info=True)
            self.record_error("sharpe_ratio_error", "Failed to record Sharpe ratio")
    
    def record_sortino_ratio(self, ratio: float) -> None:
        """Record Sortino ratio.
        
        Args:
            ratio: Sortino ratio value
        """
        try:
            self.sortino_ratio.set(ratio)
        except Exception as e:
            logger.error("Failed to record Sortino ratio: %s", str(e), exc_info=True)
            self.record_error("sortino_ratio_error", "Failed to record Sortino ratio")
    
    def record_fill_rate(
        self, 
        symbol: str, 
        order_type: str, 
        fill_rate: float,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Record order fill rate metrics.
        
        Args:
            symbol: Trading symbol
            order_type: Type of order (market, limit, etc.)
            fill_rate: Fill rate (0-1)
            tags: Additional tags for the metric
        """
        try:
            self.order_fill_rate.labels(
                symbol=symbol, 
                order_type=order_type
            ).set(fill_rate * 100)  # Store as percentage
            
            if tags:
                # Add any additional tagged metrics here
                pass
                
        except Exception as e:
            logger.error("Failed to record fill rate: %s", str(e), exc_info=True)
            self.record_error("fill_rate_error", "Failed to record fill rate")
    
    def get_metric(self, name: str):
        """Get a metric by name.
        
        Args:
            name: Name of the metric to retrieve
            
        Returns:
            The requested metric or None if not found
        """
        return getattr(self, name, None)
    
    def update_circuit_breaker_state(self, name: str, state: int) -> None:
        """Update circuit breaker state metric.
        
        Args:
            name: Name of the circuit breaker
            state: New state (0=CLOSED, 1=OPEN, 2=HALF_OPEN)
        """
        try:
            self.circuit_breaker_state.labels(circuit=name).set(state)
        except Exception as e:
            logger.error("Failed to update circuit breaker state: %s", str(e), exc_info=True)
            self.record_error("circuit_breaker_error", "Failed to update circuit breaker state")
    
    def record_circuit_breaker_transition(
        self, name: str, from_state: str, to_state: str
    ) -> None:
        """Record a circuit breaker state transition.
        
        Args:
            name: Name of the circuit breaker
            from_state: Previous state
            to_state: New state
        """
        try:
            self.circuit_breaker_state_changes.labels(
                circuit=name,
                from_state=from_state,
                to_state=to_state
            ).inc()
        except Exception as e:
            logger.error("Failed to record circuit breaker transition: %s", str(e), exc_info=True)
            self.record_error("circuit_breaker_error", "Failed to record circuit breaker transition")
    
    def record_event_processing(
        self, 
        event_type: str, 
        status: str, 
        duration_seconds: float
    ) -> None:
        """Record event processing metrics.
        
        Args:
            event_type: Type of event
            status: Status of processing (success, error, etc.)
            duration_seconds: Time taken to process the event
        """
        try:
            self.event_processing_time.labels(
                event_type=event_type,
                status=status
            ).observe(duration_seconds)
        except Exception as e:
            logger.error("Failed to record event processing: %s", str(e), exc_info=True)
            self.record_error("event_processing_error", "Failed to record event processing")
    
    def update_queue_size(self, size: int) -> None:
        """Update the current queue size metric.
        
        Args:
            size: Current size of the queue
        """
        try:
            self.event_queue_size.labels(queue="default").set(size)
        except Exception as e:
            logger.error("Failed to update queue size: %s", str(e), exc_info=True)
            self.record_error("queue_metrics_error", "Failed to update queue size")
    
    def record_queue_duration(
        self, event_type: str, duration_seconds: float
    ) -> None:
        """Record time spent in queue for an event.
        
        Args:
            event_type: Type of event
            duration_seconds: Time spent in queue
        """
        try:
            self.event_queue_duration.labels(
                event_type=event_type
            ).observe(duration_seconds)
        except Exception as e:
            logger.error("Failed to record queue duration: %s", str(e), exc_info=True)
            self.record_error("queue_metrics_error", "Failed to record queue duration")
    
    def update_system_metrics(self) -> None:
        """Update system-level metrics."""
        try:
            import psutil
            import os
            import time
            
            # Update process metrics
            process = psutil.Process(os.getpid())
            
            # CPU usage
            self.process_cpu_usage.set(process.cpu_percent())
            self.system_cpu_usage.set(psutil.cpu_percent())
            
            # Memory usage
            process_memory = process.memory_info().rss
            self.process_memory_usage.set(process_memory)
            
            system_memory = psutil.virtual_memory()
            self.system_memory_usage.set(system_memory.used)
            
            # Uptime
            self.uptime.set(time.time() - self._start_time)
            
        except Exception as e:
            logger.error("Failed to update system metrics: %s", str(e), exc_info=True)
            self.record_error("system_metrics_error", "Failed to update system metrics")
    
    def start_metrics_server(
        self, 
        port: int = 8001, 
        addr: str = "0.0.0.0"
    ) -> None:
        """Start a background thread with the metrics HTTP server.
        
        Args:
            port: Port to expose metrics on
            addr: Address to bind to
        """
        from http.server import HTTPServer
        from prometheus_client.exposition import ThreadingWSGIServer, _SilentHandler
        from wsgiref.simple_server import make_server, WSGIRequestHandler
        
        class _ThreadingSimpleServer(ThreadingWSGIServer):
            """Thread per request HTTP server."""
            daemon_threads = True
        
        class MetricsHandler(_SilentHandler):
            """Handler for the metrics endpoint."""
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.registry = self.server.registry
            
            def do_GET(self):
                """Handle GET requests."""
                if self.path == '/metrics':
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/plain; version=0.0.4')
                    self.end_headers()
                    self.wfile.write(generate_latest(self.registry))
                else:
                    self.send_response(404)
                    self.end_headers()
        
        def run_server():
            """Run the metrics server."""
            try:
                server = HTTPServer((addr, port), MetricsHandler)
                server.registry = self.registry
                logger.info("Starting metrics server on %s:%s", addr, port)
                server.serve_forever()
            except Exception as e:
                logger.error("Metrics server error: %s", str(e), exc_info=True)
                raise
        
        # Start the server in a daemon thread
        import threading
        thread = threading.Thread(target=run_server, daemon=True)
        thread.start()
        logger.info("Metrics server started on http://%s:%s/metrics", addr, port)
    
    def record_error(
        self, 
        error_type: str, 
        message: str, 
        severity: str = "error",
        exc_info: Optional[Exception] = None
    ) -> None:
        """Record an error in metrics and logs.
        
        Args:
            error_type: Type of error
            message: Error message
            severity: Error severity (error, warning, critical)
            exc_info: Optional exception for logging
        """
        try:
            self.error_counter.labels(
                error_type=error_type,
                severity=severity
            ).inc()
            
            log_method = getattr(logger, severity.lower(), logger.error)
            log_method(message, exc_info=exc_info)
            
        except Exception as e:
            logger.critical(
                "Failed to record error: %s (original error: %s)", 
                str(e), 
                message,
                exc_info=True
            )

# Create singleton instance
metrics = MetricsCollector()

def get_metrics() -> MetricsCollector:
    """Get the metrics collector instance.
    
    Returns:
        The metrics collector instance
    """
    return metrics

def start_metrics_server(port: int = 8001, addr: str = "0.0.0.0") -> None:
    """Start the metrics HTTP server.
    
    Args:
        port: Port to expose metrics on
        addr: Address to bind to
    """
    metrics.start_metrics_server(port=port, addr=addr)

# For backward compatibility
def record_error(*args, **kwargs):
    """Record an error in metrics and logs.
    
    This is a convenience wrapper around metrics.record_error().
    """
    return metrics.record_error(*args, **kwargs)
