"""
Persistence layer for monitoring data storage.

This module provides interfaces and implementations for storing metrics and alerts
in various backends (in-memory, file-based, database, etc.).
"""
import asyncio
import json
import os
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar, Generic, Type

import aiofiles
import aiofiles.os
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)

class StorageBackend(ABC, Generic[T]):
    """Abstract base class for storage backends."""
    
    @abstractmethod
    async def save(self, key: str, value: T) -> None:
        """Save a value with the given key."""
        pass
    
    @abstractmethod
    async def load(self, key: str) -> Optional[T]:
        """Load a value by key."""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete a value by key."""
        pass
    
    @abstractmethod
    async def list_keys(self, prefix: str = "") -> List[str]:
        """List all keys with optional prefix filtering."""
        pass
    
    @abstractmethod
    async def cleanup(self, older_than: Optional[datetime] = None) -> int:
        """Clean up old entries. Returns number of deleted items."""
        pass


class FileStorageBackend(StorageBackend[T]):
    """File-based storage backend using JSON files."""
    
    def __init__(self, base_dir: str, model_type: Type[T]):
        """Initialize file storage backend.
        
        Args:
            base_dir: Base directory for storage
            model_type: Pydantic model type for deserialization
        """
        self.base_dir = Path(base_dir)
        self.model_type = model_type
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_path(self, key: str) -> Path:
        """Get filesystem path for a key."""
        # Sanitize key to prevent directory traversal
        safe_key = "".join(c for c in key if c.isalnum() or c in '_-./')
        return self.base_dir / f"{safe_key}.json"
    
    async def save(self, key: str, value: T) -> None:
        """Save a value to a JSON file."""
        path = self._get_path(key)
        async with aiofiles.open(path, 'w') as f:
            await f.write(value.json())
    
    async def load(self, key: str) -> Optional[T]:
        """Load a value from a JSON file."""
        path = self._get_path(key)
        try:
            async with aiofiles.open(path, 'r') as f:
                data = await f.read()
                return self.model_type.parse_raw(data)
        except (FileNotFoundError, json.JSONDecodeError):
            return None
    
    async def delete(self, key: str) -> bool:
        """Delete a value by key."""
        path = self._get_path(key)
        try:
            await aiofiles.os.remove(path)
            return True
        except FileNotFoundError:
            return False
    
    async def list_keys(self, prefix: str = "") -> List[str]:
        """List all keys with optional prefix."""
        keys = []
        if not self.base_dir.exists():
            return keys
            
        for file_path in self.base_dir.glob("*.json"):
            key = file_path.stem
            if key.startswith(prefix):
                keys.append(key)
        return keys
    
    async def cleanup(self, older_than: Optional[datetime] = None) -> int:
        """Clean up old files."""
        if not older_than:
            older_than = datetime.now() - timedelta(days=7)  # Default 7-day retention
            
        deleted = 0
        for file_path in self.base_dir.glob("*.json"):
            try:
                stat = await aiofiles.os.stat(file_path)
                mtime = datetime.fromtimestamp(stat.st_mtime)
                if mtime < older_than:
                    await aiofiles.os.remove(file_path)
                    deleted += 1
            except (OSError, FileNotFoundError):
                continue
                
        return deleted


class InMemoryStorageBackend(StorageBackend[T]):
    """In-memory storage backend for testing and development."""
    
    def __init__(self):
        self._store: Dict[str, T] = {}
        self._timestamps: Dict[str, float] = {}
    
    async def save(self, key: str, value: T) -> None:
        """Save a value in memory."""
        self._store[key] = value
        self._timestamps[key] = time.time()
    
    async def load(self, key: str) -> Optional[T]:
        """Load a value from memory."""
        return self._store.get(key)
    
    async def delete(self, key: str) -> bool:
        """Delete a value from memory."""
        if key in self._store:
            del self._store[key]
            del self._timestamps[key]
            return True
        return False
    
    async def list_keys(self, prefix: str = "") -> List[str]:
        """List all keys with optional prefix."""
        return [k for k in self._store.keys() if k.startswith(prefix)]
    
    async def cleanup(self, older_than: Optional[datetime] = None) -> int:
        """Clean up old entries."""
        if not older_than:
            older_than = datetime.now() - timedelta(days=7)  # Default 7-day retention
            
        cutoff = older_than.timestamp()
        to_delete = [k for k, ts in self._timestamps.items() if ts < cutoff]
        
        for key in to_delete:
            del self._store[key]
            del self._timestamps[key]
            
        return len(to_delete)
