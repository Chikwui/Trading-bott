"""
Connection Pool for managing and reusing network connections.

This module provides a thread-safe connection pool for efficiently managing
network connections to market data providers, exchanges, and other services.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, AsyncContextManager, Awaitable, Callable, Deque, Dict, Generic, Optional, TypeVar, Union
from typing_extensions import Protocol

from pydantic import BaseModel, Field

# Type variables
T = TypeVar('T')

class Connection(Protocol):
    """Protocol for connection objects that can be managed by the pool."""
    async def close(self) -> None:
        ...

    async def ping(self) -> bool:
        ...

@dataclass
class PoolStats:
    """Statistics about the connection pool."""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    connection_attempts: int = 0
    connection_failures: int = 0
    wait_time_avg: float = 0.0
    wait_time_total: float = 0.0
    wait_count: int = 0

class ConnectionPool(Generic[T]):
    """Thread-safe connection pool for managing reusable connections.
    
    This implementation supports:
    - Connection reuse to reduce overhead
    - Connection validation
    - Timeout for connection acquisition
    - Maximum pool size limits
    - Automatic reconnection
    - Statistics tracking
    """
    
    def __init__(
        self,
        factory: Callable[[], Awaitable[T]],
        max_size: int = 10,
        max_usage: int = 1000,
        max_idle_time: float = 300.0,  # 5 minutes
        connect_timeout: float = 10.0,
        pool_timeout: float = 30.0,
        **kwargs
    ):
        """Initialize the connection pool.
        
        Args:
            factory: Async callable that creates a new connection
            max_size: Maximum number of connections in the pool
            max_usage: Maximum number of times a connection can be used before being replaced
            max_idle_time: Maximum time in seconds a connection can be idle before being closed
            connect_timeout: Timeout for establishing new connections
            pool_timeout: Maximum time to wait for a connection from the pool
        """
        self._factory = factory
        self.max_size = max_size
        self.max_usage = max_usage
        self.max_idle_time = max_idle_time
        self.connect_timeout = connect_timeout
        self.pool_timeout = pool_timeout
        
        # Connection tracking
        self._pool: Deque[T] = deque()
        self._in_use: Dict[int, T] = {}
        self._usage_count: Dict[int, int] = {}
        self._last_used: Dict[int, float] = {}
        
        # Statistics
        self.stats = PoolStats()
        
        # Synchronization
        self._lock = asyncio.Lock()
        self._not_empty = asyncio.Condition(self._lock)
        
        # Background task for connection maintenance
        self._maintenance_task: Optional[asyncio.Task] = None
        self._closed = False
    
    async def __aenter__(self) -> 'ConnectionPool[T]':
        """Context manager entry."""
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.close()
    
    async def start(self) -> None:
        """Start the connection pool and maintenance tasks."""
        if self._maintenance_task is None:
            self._closed = False
            self._maintenance_task = asyncio.create_task(self._maintenance_loop())
    
    async def close(self) -> None:
        """Close the connection pool and all connections."""
        if self._closed:
            return
            
        self._closed = True
        
        # Cancel maintenance task
        if self._maintenance_task is not None:
            self._maintenance_task.cancel()
            try:
                await self._maintenance_task
            except asyncio.CancelledError:
                pass
            self._maintenance_task = None
        
        # Close all connections
        async with self._lock:
            # Close idle connections
            while self._pool:
                conn = self._pool.popleft()
                if hasattr(conn, 'close'):
                    await conn.close()
            
            # Close in-use connections (they'll be removed when returned)
            for conn in list(self._in_use.values()):
                if hasattr(conn, 'close'):
                    await conn.close()
            
            self._in_use.clear()
            self._usage_count.clear()
            self._last_used.clear()
    
    async def _create_connection(self) -> T:
        """Create a new connection."""
        self.stats.connection_attempts += 1
        try:
            conn = await asyncio.wait_for(self._factory(), timeout=self.connect_timeout)
            self.stats.total_connections += 1
            return conn
        except Exception as e:
            self.stats.connection_failures += 1
            raise ConnectionError(f"Failed to create connection: {e}") from e
    
    async def _validate_connection(self, conn: T) -> bool:
        """Validate that a connection is still usable."""
        try:
            if hasattr(conn, 'ping'):
                return await asyncio.wait_for(conn.ping(), timeout=1.0)
            return True
        except Exception:
            return False
    
    async def acquire(self, timeout: Optional[float] = None) -> 'PooledConnection[T]':
        """Acquire a connection from the pool.
        
        Args:
            timeout: Maximum time to wait for a connection (defaults to pool_timeout)
            
        Returns:
            A context manager that returns a connection
            
        Raises:
            TimeoutError: If no connection is available within the timeout
            PoolClosedError: If the pool is closed
        """
        if self._closed:
            raise RuntimeError("Connection pool is closed")
            
        timeout = timeout if timeout is not None else self.pool_timeout
        start_time = time.monotonic()
        
        async with self._lock:
            while True:
                # Try to get an idle connection
                while self._pool:
                    conn = self._pool.popleft()
                    conn_id = id(conn)
                    
                    # Check if connection is still valid
                    if (time.monotonic() - self._last_used.get(conn_id, 0) > self.max_idle_time or
                            self._usage_count.get(conn_id, 0) >= self.max_usage or
                            not await self._validate_connection(conn)):
                        # Connection is stale or overused, close it
                        if hasattr(conn, 'close'):
                            await conn.close()
                        self.stats.total_connections -= 1
                        continue
                    
                    # Found a valid connection
                    self._in_use[conn_id] = conn
                    self._usage_count[conn_id] = self._usage_count.get(conn_id, 0) + 1
                    self.stats.active_connections += 1
                    
                    return PooledConnection(self, conn)
                
                # No idle connections, try to create a new one
                if len(self._in_use) < self.max_size:
                    try:
                        conn = await self._create_connection()
                        conn_id = id(conn)
                        self._in_use[conn_id] = conn
                        self._usage_count[conn_id] = 1
                        self.stats.active_connections += 1
                        
                        return PooledConnection(self, conn)
                    except Exception as e:
                        logging.warning(f"Failed to create new connection: {e}")
                
                # No connections available, wait for one to be returned
                wait_time = time.monotonic() - start_time
                if wait_time >= timeout:
                    raise TimeoutError("Timed out waiting for connection")
                
                # Wait for a connection to be returned
                wait_start = time.monotonic()
                try:
                    await asyncio.wait_for(
                        self._not_empty.wait(),
                        timeout=timeout - wait_time
                    )
                except asyncio.TimeoutError:
                    raise TimeoutError("Timed out waiting for connection") from None
                finally:
                    wait_elapsed = time.monotonic() - wait_start
                    self.stats.wait_time_total += wait_elapsed
                    self.stats.wait_count += 1
                    self.stats.wait_time_avg = (
                        self.stats.wait_time_total / self.stats.wait_count
                    )
    
    async def _release_connection(self, conn: T) -> None:
        """Release a connection back to the pool."""
        conn_id = id(conn)
        
        if conn_id not in self._in_use:
            # Connection wasn't from this pool or was already released
            return
        
        # Remove from in-use tracking
        self._in_use.pop(conn_id, None)
        self.stats.active_connections -= 1
        
        # Check if connection should be closed
        if (self._closed or 
                self._usage_count.get(conn_id, 0) >= self.max_usage or
                not await self._validate_connection(conn)):
            if hasattr(conn, 'close'):
                await conn.close()
            self.stats.total_connections -= 1
            self._usage_count.pop(conn_id, None)
            self._last_used.pop(conn_id, None)
        else:
            # Return to pool
            self._pool.append(conn)
            self._last_used[conn_id] = time.monotonic()
            self.stats.idle_connections = len(self._pool)
    
    async def _maintenance_loop(self) -> None:
        """Background task for connection maintenance."""
        while not self._closed:
            try:
                await asyncio.sleep(60)  # Run every minute
                await self._cleanup_idle_connections()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.error(f"Error in maintenance loop: {e}")
                await asyncio.sleep(5)  # Prevent tight loop on errors
    
    async def _cleanup_idle_connections(self) -> None:
        """Clean up idle connections that have exceeded max_idle_time."""
        async with self._lock:
            now = time.monotonic()
            stale_conns = []
            
            # Find stale connections
            for conn in list(self._pool):
                conn_id = id(conn)
                last_used = self._last_used.get(conn_id, 0)
                if now - last_used > self.max_idle_time:
                    stale_conns.append(conn)
            
            # Remove and close stale connections
            for conn in stale_conns:
                self._pool.remove(conn)
                if hasattr(conn, 'close'):
                    await conn.close()
                conn_id = id(conn)
                self._usage_count.pop(conn_id, None)
                self._last_used.pop(conn_id, None)
                self.stats.total_connections -= 1
            
            self.stats.idle_connections = len(self._pool)


class PooledConnection(Generic[T]):
    """Context manager for pooled connections."""
    
    def __init__(self, pool: ConnectionPool[T], conn: T):
        self.pool = pool
        self.conn = conn
        self._released = False
    
    async def __aenter__(self) -> T:
        """Enter the context and return the connection."""
        if self._released:
            raise RuntimeError("Connection already released")
        return self.conn
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and release the connection back to the pool."""
        await self.release()
    
    async def release(self) -> None:
        """Explicitly release the connection back to the pool."""
        if not self._released:
            self._released = True
            await self.pool._release_connection(self.conn)
