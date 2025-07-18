"""
Test suite for the connection pool implementation.
"""
import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from core.utils.connection_pool import ConnectionPool, PoolStats

class MockConnection:
    """Mock connection for testing."""
    
    def __init__(self, conn_id: int, *, fail_ping: bool = False, close_error: bool = False):
        self.conn_id = conn_id
        self.fail_ping = fail_ping
        self.close_error = close_error
        self.closed = False
    
    async def ping(self) -> bool:
        """Mock ping method."""
        if self.fail_ping:
            raise ConnectionError("Ping failed")
        return True
    
    async def close(self) -> None:
        """Mock close method."""
        if self.close_error:
            raise RuntimeError("Close error")
        self.closed = True

@pytest.fixture
def mock_connection_factory():
    """Factory for creating mock connections."""
    counter = 0
    
    async def factory():
        nonlocal counter
        counter += 1
        return MockConnection(counter)
    
    return factory

@pytest.mark.asyncio
async def test_connection_pool_acquire_release(mock_connection_factory):
    """Test basic acquire and release of connections."""
    pool = ConnectionPool(mock_connection_factory, max_size=2)
    
    # Acquire first connection
    async with pool.acquire() as conn1:
        assert isinstance(conn1, MockConnection)
        assert not conn1.closed
        assert pool.stats.active_connections == 1
        assert pool.stats.idle_connections == 0
        
        # Acquire second connection
        async with pool.acquire() as conn2:
            assert isinstance(conn2, MockConnection)
            assert conn1.conn_id != conn2.conn_id
            assert pool.stats.active_connections == 2
            assert pool.stats.idle_connections == 0
            
            # Pool is full, next acquire should wait
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(pool.acquire(), timeout=0.1)
    
    # Connections should be returned to the pool
    assert pool.stats.active_connections == 0
    assert pool.stats.idle_connections == 2

@pytest.mark.asyncio
async def test_connection_pool_max_usage(mock_connection_factory):
    """Test connection replacement after max usage."""
    pool = ConnectionPool(mock_connection_factory, max_size=2, max_usage=2)
    conn_ids = set()
    
    # Use the same connection twice
    for _ in range(2):
        async with pool.acquire() as conn:
            conn_ids.add(conn.conn_id)
    
    # Should have used the same connection twice
    assert len(conn_ids) == 1
    
    # Third use should create a new connection
    async with pool.acquire() as conn:
        conn_ids.add(conn.conn_id)
    
    # Should have created a new connection
    assert len(conn_ids) == 2
    assert pool.stats.total_connections == 2
    assert pool.stats.connection_attempts == 2

@pytest.mark.asyncio
async def test_connection_pool_validation(mock_connection_factory):
    """Test connection validation."""
    # Create a connection that will fail ping
    async def failing_factory():
        return MockConnection(1, fail_ping=True)
    
    pool = ConnectionPool(failing_factory, max_size=2)
    
    # Should create a new connection since the first one fails validation
    with pytest.raises(ConnectionError):
        async with pool.acquire():
            pass
    
    # Should have tried to create a connection
    assert pool.stats.connection_attempts == 1
    assert pool.stats.connection_failures == 1

@pytest.mark.asyncio
async def test_connection_pool_close(mock_connection_factory):
    """Test closing the connection pool."""
    pool = ConnectionPool(mock_connection_factory, max_size=2)
    
    # Acquire and release a connection
    async with pool.acquire() as conn:
        pass
    
    # Close the pool
    await pool.close()
    
    # Should not be able to acquire after close
    with pytest.raises(RuntimeError):
        await pool.acquire()
    
    # Should have closed all connections
    assert pool.stats.total_connections == 0
    assert pool.stats.idle_connections == 0
    assert pool.stats.active_connections == 0

@pytest.mark.asyncio
async def test_connection_pool_timeout(mock_connection_factory):
    """Test connection acquisition timeout."""
    pool = ConnectionPool(mock_connection_factory, max_size=1, pool_timeout=0.1)
    
    # Acquire the only available connection
    async with pool.acquire():
        # Try to acquire another connection (should time out)
        with pytest.raises(TimeoutError):
            await pool.acquire()
    
    # Should be able to acquire after release
    async with pool.acquire():
        pass

@pytest.mark.asyncio
async def test_connection_pool_maintenance(mock_connection_factory):
    """Test connection maintenance (idle timeout)."""
    pool = ConnectionPool(
        mock_connection_factory,
        max_size=2,
        max_idle_time=0.1,  # Very short idle timeout for testing
        pool_timeout=0.1
    )
    
    # Start the maintenance task
    await pool.start()
    
    try:
        # Acquire and release a connection
        async with pool.acquire() as conn:
            conn_id = conn.conn_id
        
        # Connection should be in the pool
        assert pool.stats.idle_connections == 1
        
        # Wait for the connection to become stale
        await asyncio.sleep(0.2)
        
        # Run maintenance manually (normally done by the background task)
        await pool._cleanup_idle_connections()
        
        # Connection should have been removed
        assert pool.stats.idle_connections == 0
        assert pool.stats.total_connections == 0
    finally:
        await pool.close()

@pytest.mark.asyncio
async def test_connection_pool_context_manager(mock_connection_factory):
    """Test using the connection pool as a context manager."""
    async with ConnectionPool(mock_connection_factory, max_size=2) as pool:
        async with pool.acquire() as conn:
            assert isinstance(conn, MockConnection)
        
        # Should still be able to acquire after exiting the context
        async with pool.acquire() as conn:
            assert isinstance(conn, MockConnection)
    
    # Pool should be closed after exiting the context
    with pytest.raises(RuntimeError):
        await pool.acquire()

@pytest.mark.asyncio
async def test_connection_pool_stats(mock_connection_factory):
    """Test connection pool statistics."""
    pool = ConnectionPool(mock_connection_factory, max_size=2, pool_timeout=0.1)
    
    # Initial stats
    stats = pool.stats
    assert stats.total_connections == 0
    assert stats.active_connections == 0
    assert stats.idle_connections == 0
    assert stats.connection_attempts == 0
    assert stats.connection_failures == 0
    
    # Acquire and release a connection
    async with pool.acquire():
        stats = pool.stats
        assert stats.total_connections == 1
        assert stats.active_connections == 1
        assert stats.idle_connections == 0
        assert stats.connection_attempts == 1
        assert stats.connection_failures == 0
    
    # After release
    stats = pool.stats
    assert stats.total_connections == 1
    assert stats.active_connections == 0
    assert stats.idle_connections == 1
    
    # Test wait time stats
    async with pool.acquire():
        pass
    
    assert pool.stats.wait_count > 0
    assert pool.stats.wait_time_total > 0
    assert pool.stats.wait_time_avg > 0

if __name__ == "__main__":
    pytest.main(["-v", "test_connection_pool.py"])
