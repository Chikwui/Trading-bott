"""
Performance-critical components for high-frequency order management.

This module provides optimized data structures and utilities for the OrderManager
to handle high-frequency order operations with minimal latency and maximum throughput.
"""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

from prometheus_client import CollectorRegistry
from ..metrics import metrics

# Type variables for generic operations
T = TypeVar('T')

class ObjectPool:
    """Thread-safe object pool for frequently allocated objects."""
    
    def __init__(self, factory: Type[T], max_size: int = 1000):
        self._factory = factory
        self._max_size = max_size
        self._pool: List[T] = []
        self._lock = asyncio.Lock()
        
        # Get metrics from the central registry
        self.pool_size = metrics.object_pool_size
        self.allocations = metrics.object_allocations
        
    async def acquire(self, *args, **kwargs) -> T:
        """Acquire an object from the pool or create a new one."""
        async with self._lock:
            if self._pool:
                obj = self._pool.pop()
                self.pool_size.labels(type=self._factory.__name__).dec()
                self.allocations.labels(
                    type=self._factory.__name__,
                    source='pool'
                ).inc()
                return obj
                
        # Create new object if pool is empty
        self.allocations.labels(
            type=self._factory.__name__,
            source='new'
        ).inc()
        return self._factory(*args, **kwargs)
    
    async def release(self, obj: T) -> None:
        """Return an object to the pool."""
        async with self._lock:
            if len(self._pool) < self._max_size:
                # Reset object state if possible
                if hasattr(obj, 'reset'):
                    obj.reset()
                self._pool.append(obj)
                self.pool_size.labels(type=type(obj).__name__).inc()


@dataclass
class OrderBookEntry:
    """Optimized order book entry for high-frequency operations."""
    order_id: str
    price: Decimal
    quantity: Decimal
    timestamp: float = field(default_factory=time.monotonic)
    
    def __post_init__(self):
        # Convert to Decimal if needed
        if not isinstance(self.price, Decimal):
            self.price = Decimal(str(self.price))
        if not isinstance(self.quantity, Decimal):
            self.quantity = Decimal(str(self.quantity))


class LockFreeOrderBook:
    """Lock-free order book implementation for high-frequency trading.
    
    This implementation uses separate locks for different price levels to
    minimize contention during concurrent operations.
    """
    
    def __init__(self):
        self._bids: Dict[Decimal, List[OrderBookEntry]] = defaultdict(list)
        self._asks: Dict[Decimal, List[OrderBookEntry]] = defaultdict(list)
        self._locks: Dict[Decimal, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._price_levels: Set[Decimal] = set()
        
        # Get metrics from the central registry
        self.depth = metrics.order_book_depth
        self.orders = metrics.order_book_orders
    
    async def add_order(self, order: OrderBookEntry, is_bid: bool, symbol: str = '') -> None:
        """Add an order to the order book.
        
        Args:
            order: The order to add
            is_bid: Whether this is a bid (True) or ask (False)
            symbol: The trading pair symbol (e.g., 'BTC/USDT')
        """
        price = order.price
        book = self._bids if is_bid else self._asks
        
        async with self._get_lock(price):
            if price not in book:
                self._price_levels.add(price)
                self.depth.labels(
                    symbol=symbol,
                    side='bid' if is_bid else 'ask',
                    price_level=str(price)
                ).inc()
                
            book[price].append(order)
            self.orders.labels(symbol=symbol).inc()
    
    async def remove_order(self, order_id: str, price: Decimal, is_bid: bool, symbol: str = '') -> bool:
        """Remove an order from the order book."""
        book = self._bids if is_bid else self._asks
        
        async with self._get_lock(price):
            if price not in book:
                return False
                
            orders = book[price]
            for i, order in enumerate(orders):
                if order.order_id == order_id:
                    orders.pop(i)
                    self.orders.labels(symbol=symbol).dec()
                    
                    # Remove price level if empty
                    if not orders:
                        del book[price]
                        self._price_levels.discard(price)
                        self.depth.labels(
                            symbol=symbol,
                            side='bid' if is_bid else 'ask',
                            price_level=str(price)
                        ).dec()
                        
                    return True
                    
            return False
    
    async def get_best_bid_ask(self) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """Get the best bid and ask prices."""
        best_bid = max(self._bids.keys()) if self._bids else None
        best_ask = min(self._asks.keys()) if self._asks else None
        return best_bid, best_ask
    
    def _get_lock(self, price: Decimal) -> asyncio.Lock:
        """Get or create a lock for the given price level."""
        return self._locks[price]


class RateLimiter:
    """Token bucket rate limiter for API calls and order submissions."""
    
    def __init__(self, rate: float, capacity: int):
        """Initialize the rate limiter.
        
        Args:
            rate: Number of tokens added per second
            capacity: Maximum number of tokens in the bucket
        """
        self._rate = rate
        self._capacity = capacity
        self._tokens = capacity
        self._last_update = time.monotonic()
        self._lock = asyncio.Lock()
        
        # Get metrics from the central registry
        self.rate_limit_hits = metrics.rate_limit_hits
    
    async def acquire(self, tokens: int = 1, resource: str = 'default') -> bool:
        """Acquire tokens from the bucket.
        
        Returns:
            bool: True if tokens were acquired, False if rate limited
        """
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_update
            
            # Add tokens based on elapsed time
            self._tokens = min(
                self._capacity,
                self._tokens + elapsed * self._rate
            )
            self._last_update = now
            
            # Check if we have enough tokens
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
                
            # Rate limit hit
            self.rate_limit_hits.labels(resource=resource).inc()
            return False


class PerformanceMonitor:
    """Performance monitoring and profiling for the trading system."""
    
    def __init__(self):
        self._latencies: Dict[str, List[float]] = defaultdict(list)
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = defaultdict(float)
        self._lock = asyncio.Lock()
        
        # Get metrics from the central registry
        self.latency_histogram = metrics.operation_latency
        self.operation_counter = metrics.operation_count
    
    async def record_latency(self, operation: str, latency: float) -> None:
        """Record the latency of an operation."""
        async with self._lock:
            self._latencies[operation].append(latency)
            self.latency_histogram.labels(operation=operation).observe(latency)
    
    async def increment_counter(self, operation: str, value: int = 1) -> None:
        """Increment a counter."""
        async with self._lock:
            self._counters[operation] += value
            self.operation_counter.labels(operation=operation).inc(value)
    
    async def set_gauge(self, name: str, value: float) -> None:
        """Set a gauge value."""
        async with self._lock:
            self._gauges[name] = value
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get a summary of performance metrics."""
        return {
            'latencies': {
                op: {
                    'count': len(times),
                    'avg': sum(times) / len(times) if times else 0,
                    'p95': sorted(times)[int(len(times) * 0.95)] if times else 0,
                    'max': max(times) if times else 0,
                }
                for op, times in self._latencies.items()
            },
            'counters': dict(self._counters),
            'gauges': dict(self._gauges),
        }
