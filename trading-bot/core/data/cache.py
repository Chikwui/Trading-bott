"""Data caching module for the trading bot."""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Type, TypeVar, Union
from datetime import datetime, timedelta
import asyncio
import json
import logging
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import hashlib
import time

logger = logging.getLogger(__name__)
T = TypeVar('T')

class CacheEntry:
    """A single cache entry with value and expiration time."""
    
    def __init__(self, value: Any, ttl: Optional[float] = None):
        """Initialize a cache entry.
        
        Args:
            value: The value to cache
            ttl: Time to live in seconds, or None for no expiration
        """
        self.value = value
        self.expires_at = time.time() + ttl if ttl is not None else None
    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired.
        
        Returns:
            bool: True if expired, False otherwise
        """
        if self.expires_at is None:
            return False
        return time.time() >= self.expires_at

class CacheBackend(ABC):
    """Abstract base class for cache backends."""
    
    @abstractmethod
    async def get(self, key: str) -> Any:
        """Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            The cached value, or None if not found or expired
        """
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds, or None for no expiration
        """
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> None:
        """Delete a value from the cache.
        
        Args:
            key: Cache key
        """
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """Clear all values from the cache."""
        pass
    
    @abstractmethod
    async def has(self, key: str) -> bool:
        """Check if a key exists in the cache.
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if the key exists and is not expired
        """
        pass

class MemoryCacheBackend(CacheBackend):
    """In-memory cache backend."""
    
    def __init__(self):
        """Initialize the in-memory cache."""
        self._store: Dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Any:
        """Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            The cached value, or None if not found or expired
        """
        async with self._lock:
            if key not in self._store:
                return None
            
            entry = self._store[key]
            if entry.is_expired():
                del self._store[key]
                return None
            
            return entry.value
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds, or None for no expiration
        """
        async with self._lock:
            self._store[key] = CacheEntry(value, ttl)
    
    async def delete(self, key: str) -> None:
        """Delete a value from the cache.
        
        Args:
            key: Cache key
        """
        async with self._lock:
            if key in self._store:
                del self._store[key]
    
    async def clear(self) -> None:
        """Clear all values from the cache."""
        async with self._lock:
            self._store.clear()
    
    async def has(self, key: str) -> bool:
        """Check if a key exists in the cache.
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if the key exists and is not expired
        """
        async with self._lock:
            if key not in self._store:
                return False
            
            entry = self._store[key]
            if entry.is_expired():
                del self._store[key]
                return False
            
            return True

class RedisCacheBackend(CacheBackend):
    """Redis cache backend."""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, db: int = 0, 
                 password: Optional[str] = None, prefix: str = 'trading_bot:'):
        """Initialize the Redis cache backend.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            password: Redis password
            prefix: Prefix for all cache keys
        """
        self._host = host
        self._port = port
        self._db = db
        self._password = password
        self._prefix = prefix
        self._redis = None
        self._lock = asyncio.Lock()
    
    async def _get_redis(self):
        """Get a Redis connection."""
        if self._redis is None:
            import redis.asyncio as redis
            self._redis = redis.Redis(
                host=self._host,
                port=self._port,
                db=self._db,
                password=self._password,
                decode_responses=False
            )
        return self._redis
    
    def _get_key(self, key: str) -> str:
        """Get the full cache key with prefix."""
        return f"{self._prefix}{key}"
    
    async def get(self, key: str) -> Any:
        """Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            The cached value, or None if not found or expired
        """
        try:
            redis = await self._get_redis()
            data = await redis.get(self._get_key(key))
            if data is None:
                return None
            return pickle.loads(data)
        except Exception as e:
            logger.error(f"Error getting from Redis cache: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds, or None for no expiration
        """
        try:
            redis = await self._get_redis()
            data = pickle.dumps(value)
            if ttl is not None:
                await redis.setex(self._get_key(key), int(ttl), data)
            else:
                await redis.set(self._get_key(key), data)
        except Exception as e:
            logger.error(f"Error setting value in Redis cache: {e}")
    
    async def delete(self, key: str) -> None:
        """Delete a value from the cache.
        
        Args:
            key: Cache key
        """
        try:
            redis = await self._get_redis()
            await redis.delete(self._get_key(key))
        except Exception as e:
            logger.error(f"Error deleting from Redis cache: {e}")
    
    async def clear(self) -> None:
        """Clear all values from the cache."""
        try:
            redis = await self._get_redis()
            keys = await redis.keys(f"{self._prefix}*")
            if keys:
                await redis.delete(*keys)
        except Exception as e:
            logger.error(f"Error clearing Redis cache: {e}")
    
    async def has(self, key: str) -> bool:
        """Check if a key exists in the cache.
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if the key exists and is not expired
        """
        try:
            redis = await self._get_redis()
            return await redis.exists(self._get_key(key)) == 1
        except Exception as e:
            logger.error(f"Error checking key in Redis cache: {e}")
            return False

class DiskCacheBackend(CacheBackend):
    """Disk-based cache backend."""
    
    def __init__(self, path: str, max_size: int = 104857600):  # 100MB default max size
        """Initialize the disk cache backend.
        
        Args:
            path: Path to the cache directory
            max_size: Maximum cache size in bytes
        """
        import os
        from pathlib import Path
        
        self._path = Path(path)
        self._max_size = max_size
        self._lock = asyncio.Lock()
        
        # Create cache directory if it doesn't exist
        self._path.mkdir(parents=True, exist_ok=True)
    
    def _get_path(self, key: str):
        """Get the filesystem path for a cache key."""
        # Create a hash of the key to use as the filename
        key_hash = hashlib.md5(key.encode('utf-8')).hexdigest()
        return self._path / f"{key_hash}.cache"
    
    async def get(self, key: str) -> Any:
        """Get a value from the cache.
        
        Args:
            key: Cache key
            
        Returns:
            The cached value, or None if not found or expired
        """
        path = self._get_path(key)
        if not path.exists():
            return None
        
        try:
            async with self._lock:
                with open(path, 'rb') as f:
                    entry = pickle.load(f)
                
                # Check if entry is expired
                if entry.is_expired():
                    path.unlink()
                    return None
                
                return entry.value
        except Exception as e:
            logger.error(f"Error reading from disk cache: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds, or None for no expiration
        """
        path = self._get_path(key)
        entry = CacheEntry(value, ttl)
        
        try:
            async with self._lock:
                # Check cache size and clean up if needed
                await self._enforce_size_limit()
                
                with open(path, 'wb') as f:
                    pickle.dump(entry, f)
        except Exception as e:
            logger.error(f"Error writing to disk cache: {e}")
    
    async def _enforce_size_limit(self) -> None:
        """Enforce the maximum cache size by removing oldest entries."""
        import os
        from pathlib import Path
        
        # Get all cache files with their modification times
        files = []
        total_size = 0
        
        for file_path in self._path.glob('*.cache'):
            stat = file_path.stat()
            files.append((file_path, stat.st_mtime, stat.st_size))
            total_size += stat.st_size
        
        # Sort by modification time (oldest first)
        files.sort(key=lambda x: x[1])
        
        # Remove oldest files until under size limit
        while files and total_size > self._max_size:
            file_path, _, size = files.pop(0)
            try:
                file_path.unlink()
                total_size -= size
            except Exception as e:
                logger.error(f"Error cleaning up cache file {file_path}: {e}")
    
    async def delete(self, key: str) -> None:
        """Delete a value from the cache.
        
        Args:
            key: Cache key
        """
        path = self._get_path(key)
        try:
            async with self._lock:
                if path.exists():
                    path.unlink()
        except Exception as e:
            logger.error(f"Error deleting from disk cache: {e}")
    
    async def clear(self) -> None:
        """Clear all values from the cache."""
        try:
            async with self._lock:
                for file_path in self._path.glob('*.cache'):
                    try:
                        file_path.unlink()
                    except Exception as e:
                        logger.error(f"Error clearing cache file {file_path}: {e}")
        except Exception as e:
            logger.error(f"Error clearing disk cache: {e}")
    
    async def has(self, key: str) -> bool:
        """Check if a key exists in the cache.
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if the key exists and is not expired
        """
        path = self._get_path(key)
        if not path.exists():
            return False
        
        try:
            async with self._lock:
                with open(path, 'rb') as f:
                    entry = pickle.load(f)
                
                if entry.is_expired():
                    path.unlink()
                    return False
                
                return True
        except Exception:
            return False

class DataCache:
    """High-level cache interface with multiple backend support."""
    
    def __init__(self, config: 'DataCacheConfig'):
        """Initialize the data cache.
        
        Args:
            config: Cache configuration
        """
        self._config = config
        self._backend = self._create_backend()
    
    def _create_backend(self) -> CacheBackend:
        """Create the appropriate cache backend based on config."""
        if self._config.backend == 'memory':
            return MemoryCacheBackend()
        elif self._config.backend == 'redis':
            return RedisCacheBackend(
                host=self._config.host or 'localhost',
                port=self._config.port or 6379,
                db=self._config.db or 0,
                password=self._config.password,
                prefix=self._config.prefix
            )
        elif self._config.backend == 'disk':
            if not self._config.path:
                raise ValueError("Path must be specified for disk cache")
            return DiskCacheBackend(
                path=self._config.path,
                max_size=self._config.max_size or 104857600  # 100MB default
            )
        else:
            raise ValueError(f"Unsupported cache backend: {self._config.backend}")
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the cache.
        
        Args:
            key: Cache key
            default: Default value to return if key not found
            
        Returns:
            The cached value, or the default if not found
        """
        if not self._config.enabled:
            return default
            
        value = await self._backend.get(key)
        if value is None:
            return default
        return value
    
    async def set(self, key: str, value: Any, ttl: Optional[float] = None) -> None:
        """Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds, or None to use default
        """
        if not self._config.enabled:
            return
            
        ttl = ttl if ttl is not None else self._config.ttl
        await self._backend.set(key, value, ttl)
    
    async def delete(self, key: str) -> None:
        """Delete a value from the cache.
        
        Args:
            key: Cache key
        """
        if not self._config.enabled:
            return
            
        await self._backend.delete(key)
    
    async def clear(self) -> None:
        """Clear all values from the cache."""
        if not self._config.enabled:
            return
            
        await self._backend.clear()
    
    async def has(self, key: str) -> bool:
        """Check if a key exists in the cache.
        
        Args:
            key: Cache key
            
        Returns:
            bool: True if the key exists and is not expired
        """
        if not self._config.enabled:
            return False
            
        return await self._backend.has(key)
    
    async def get_or_set(
        self,
        key: str,
        default: Any = None,
        ttl: Optional[float] = None,
        setter: callable = None,
        *args, **kwargs
    ) -> Any:
        """Get a value from the cache, or set it if not found.
        
        Args:
            key: Cache key
            default: Default value to set if key not found
            ttl: Time to live in seconds, or None to use default
            setter: Optional callable to generate the value if not found
            *args, **kwargs: Arguments to pass to the setter
            
        Returns:
            The cached or newly set value
        """
        if not self._config.enabled:
            return default if setter is None else setter(*args, **kwargs)
        
        value = await self.get(key)
        if value is not None:
            return value
        
        if setter is not None:
            value = setter(*args, **kwargs)
            if asyncio.iscoroutinefunction(setter):
                value = await value
        else:
            value = default
        
        if value is not None:
            ttl = ttl if ttl is not None else self._config.ttl
            await self.set(key, value, ttl)
        
        return value
    
    async def memoize(
        self,
        ttl: Optional[float] = None,
        key_func: callable = None
    ) -> callable:
        """Decorator to cache function results.
        
        Args:
            ttl: Time to live in seconds, or None to use default
            key_func: Function to generate cache key from function arguments
            
        Returns:
            Decorated function with caching
        """
        def decorator(func):
            async def wrapper(*args, **kwargs):
                if not self._config.enabled:
                    return await func(*args, **kwargs)
                
                # Generate cache key
                if key_func is not None:
                    cache_key = key_func(*args, **kwargs)
                else:
                    # Default key generation
                    cache_key = f"{func.__module__}:{func.__name__}:{args}:{kwargs}"
                
                # Try to get from cache
                cached = await self.get(cache_key)
                if cached is not None:
                    return cached
                
                # Call the function and cache the result
                result = await func(*args, **kwargs)
                await self.set(cache_key, result, ttl)
                
                return result
            
            return wrapper
        
        return decorator
