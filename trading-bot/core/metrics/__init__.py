"""
Centralized metrics registry for the trading system.

This module provides a singleton registry for all Prometheus metrics
used throughout the application, ensuring consistent naming and preventing
duplicate registration issues.
"""
from typing import Dict, Optional

from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry


class MetricsRegistry:
    """Centralized registry for all application metrics.
    
    This class provides a single place to define and access all metrics
    used throughout the application, preventing duplicate registration
    and ensuring consistent naming.
    """
    
    _instance = None
    
    def __new__(cls, registry: Optional[CollectorRegistry] = None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._registry = registry
            cls._instance._metrics = {}
            cls._instance._initialize_metrics()
        return cls._instance
    
    def _initialize_metrics(self):
        """Initialize all metrics used in the application."""
        # Order metrics
        self.orders_total = self.counter(
            'trading_orders_total',
            'Total number of orders processed',
            ['type', 'status', 'symbol']
        )
        
        self.order_latency = self.histogram(
            'trading_order_latency_seconds',
            'Order processing latency',
            ['operation']
        )
        
        self.active_orders = self.gauge(
            'trading_active_orders',
            'Number of active orders',
            ['symbol', 'side']
        )
        
        # Object pool metrics
        self.object_pool_size = self.gauge(
            'trading_object_pool_size',
            'Number of objects in the pool',
            ['type']
        )
        
        self.object_allocations = self.counter(
            'trading_object_allocations_total',
            'Total number of object allocations',
            ['type', 'source']
        )
        
        # Rate limiting metrics
        self.rate_limit_hits = self.counter(
            'trading_rate_limit_hits_total',
            'Number of times rate limiting was applied',
            ['resource']
        )
        
        # Performance metrics
        self.operation_latency = self.histogram(
            'trading_operation_latency_seconds',
            'Latency of trading operations',
            ['operation']
        )
        
        self.operation_count = self.counter(
            'trading_operation_count',
            'Number of operations performed',
            ['operation']
        )
        
        # Order book metrics
        self.order_book_depth = self.gauge(
            'trading_order_book_depth',
            'Number of orders at each price level',
            ['symbol', 'side', 'price_level']
        )
        
        self.order_book_orders = self.gauge(
            'trading_order_book_orders',
            'Number of orders in the order book',
            ['symbol']
        )
    
    def counter(self, name: str, description: str, labels: list) -> Counter:
        """Get or create a counter metric."""
        if name not in self._metrics:
            self._metrics[name] = Counter(
                name,
                description,
                labels,
                registry=self._registry
            )
        return self._metrics[name]
    
    def gauge(self, name: str, description: str, labels: list) -> Gauge:
        """Get or create a gauge metric."""
        if name not in self._metrics:
            self._metrics[name] = Gauge(
                name,
                description,
                labels,
                registry=self._registry
            )
        return self._metrics[name]
    
    def histogram(self, name: str, description: str, labels: list) -> Histogram:
        """Get or create a histogram metric."""
        if name not in self._metrics:
            self._metrics[name] = Histogram(
                name,
                description,
                labels,
                registry=self._registry
            )
        return self._metrics[name]


# Default instance using the default Prometheus registry
metrics = MetricsRegistry()
