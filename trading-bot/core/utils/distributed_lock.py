"""
Distributed locking mechanism for order state management.

Implements a Redis-based distributed lock with auto-expiration and
automatic renewal to prevent race conditions in order state transitions.
"""
import asyncio
import time
import uuid
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
import aioredis
from loguru import logger

class DistributedLockError(Exception):
    """Base exception for distributed lock errors."""
    pass

class LockAcquisitionError(DistributedLockError):
    """Raised when a lock cannot be acquired."""
    pass

class LockReleaseError(DistributedLockError):
    """Raised when a lock cannot be released."""
    pass

class DistributedLock:
    """
    Distributed lock using Redis with auto-expiration and renewal.
    
    Features:
    - Auto-expiring locks to prevent deadlocks
    - Automatic lock renewal for long-running operations
    - Token-based ownership verification
    - Context manager support
    """
    
    def __init__(
        self,
        redis: aioredis.Redis,
        lock_key: str,
        timeout: float = 30.0,
        blocking: bool = True,
        block_timeout: Optional[float] = 30.0
    ):
        """
        Initialize the distributed lock.
        
        Args:
            redis: Redis client instance
            lock_key: Key to use for the lock
            timeout: Lock timeout in seconds
            blocking: Whether to block until lock is acquired
            block_timeout: Maximum time to wait for lock (None = forever)
        """
        self.redis = redis
        self.lock_key = f"lock:{lock_key}"
        self.timeout = timeout
        self.blocking = blocking
        self.block_timeout = block_timeout
        self.token = str(uuid.uuid4())
        self._lock_renewal_task = None
        self._locked = False
    
    async def acquire(self) -> bool:
        """Acquire the lock."""
        start_time = time.monotonic()
        
        while True:
            # Try to acquire the lock
            acquired = await self._acquire_once()
            if acquired:
                return True
                
            if not self.blocking:
                return False
                
            # Check if we've exceeded the block timeout
            if self.block_timeout is not None:
                elapsed = time.monotonic() - start_time
                if elapsed >= self.block_timeout:
                    return False
                    
            # Wait before retrying
            await asyncio.sleep(0.1)
    
    async def _acquire_once(self) -> bool:
        """Try to acquire the lock once."""
        # Use SET with NX and PX options for atomic lock acquisition
        result = await self.redis.set(
            self.lock_key,
            self.token,
            px=int(self.timeout * 1000),  # Convert to milliseconds
            nx=True
        )
        
        if result:
            self._locked = True
            self._start_lock_renewal()
            return True
            
        return False
    
    def _start_lock_renewal(self) -> None:
        """Start background task to renew the lock."""
        if self.timeout <= 0:
            return
            
        loop = asyncio.get_event_loop()
        self._lock_renewal_task = loop.create_task(self._renew_lock_periodically())
    
    async def _renew_lock_periodically(self) -> None:
        """Periodically renew the lock to prevent expiration."""
        try:
            renew_interval = self.timeout * 0.7  # Renew at 70% of timeout
            
            while self._locked:
                await asyncio.sleep(renew_interval)
                
                if not self._locked:
                    break
                    
                # Use Lua script for atomic renewal
                script = """
                if redis.call("get", KEYS[1]) == ARGV[1] then
                    return redis.call("pexpire", KEYS[1], ARGV[2])
                else
                    return 0
                """
                
                renewed = await self.redis.eval(
                    script,
                    keys=[self.lock_key],
                    args=[self.token, int(self.timeout * 1000)]
                )
                
                if not renewed:
                    logger.warning(f"Failed to renew lock {self.lock_key}")
                    self._locked = False
                    break
                    
        except asyncio.CancelledError:
            # Task was cancelled
            pass
        except Exception as e:
            logger.error(f"Error in lock renewal: {e}")
            self._locked = False
    
    async def release(self) -> None:
        """Release the lock."""
        if not self._locked:
            return
            
        # Stop the renewal task
        if self._lock_renewal_task and not self._lock_renewal_task.done():
            self._lock_renewal_task.cancel()
            try:
                await self._lock_renewal_task
            except asyncio.CancelledError:
                pass
                
        # Use Lua script for atomic release
        script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """
        
        try:
            released = await self.redis.eval(
                script,
                keys=[self.lock_key],
                args=[self.token]
            )
            
            if not released:
                raise LockReleaseError("Lock was lost or already released")
                
        except Exception as e:
            raise LockReleaseError(f"Failed to release lock: {e}") from e
        finally:
            self._locked = False
    
    async def __aenter__(self):
        """Context manager entry."""
        if not await self.acquire():
            raise LockAcquisitionError(f"Could not acquire lock {self.lock_key}")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        await self.release()

@asynccontextmanager
async def order_lock(
    redis: aioredis.Redis,
    order_id: str,
    timeout: float = 30.0,
    blocking: bool = True,
    block_timeout: Optional[float] = 30.0
):
    """Context manager for acquiring an order lock."""
    lock = DistributedLock(
        redis=redis,
        lock_key=f"order:{order_id}",
        timeout=timeout,
        blocking=blocking,
        block_timeout=block_timeout
    )
    
    try:
        await lock.acquire()
        yield lock
    finally:
        await lock.release()
