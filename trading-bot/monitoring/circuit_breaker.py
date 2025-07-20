"""
Circuit Breaker Pattern Implementation

This module provides a thread-safe implementation of the circuit breaker pattern
for handling failures in distributed systems and microservices.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, Optional, TypeVar, Union, cast

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Type aliases
T = TypeVar('T')

class CircuitState(Enum):
    """Possible states of a circuit breaker."""
    CLOSED = auto()
    OPEN = auto()
    HALF_OPEN = auto()

class CircuitBreakerError(Exception):
    """Raised when the circuit breaker is open."""
    def __init__(self, circuit_name: str, state: CircuitState, retry_after: Optional[float] = None):
        self.circuit_name = circuit_name
        self.state = state
        self.retry_after = retry_after
        super().__init__(f"Circuit '{circuit_name}' is {state.name}" + 
                        (f", retry after {retry_after:.1f}s" if retry_after else ""))

class CircuitBreakerConfig(BaseModel):
    """Configuration for a circuit breaker."""
    failure_threshold: int = Field(
        default=3,
        description="Number of failures before opening the circuit"
    )
    recovery_timeout: float = Field(
        default=30.0,
        description="Time in seconds before attempting to close the circuit"
    )
    success_threshold: int = Field(
        default=3,
        description="Number of consecutive successes required to close the circuit"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retries in half-open state"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "failure_threshold": 3,
                "recovery_timeout": 30.0,
                "success_threshold": 3,
                "max_retries": 3
            }
        }

@dataclass
class CircuitBreakerStats:
    """Statistics for a circuit breaker."""
    failures: int = 0
    successes: int = 0
    state_changes: int = 0
    last_failure: Optional[datetime] = None
    last_success: Optional[datetime] = None
    last_state_change: Optional[datetime] = None
    
    def reset(self) -> None:
        """Reset all statistics."""
        self.failures = 0
        self.successes = 0
        self.state_changes = 0
        self.last_failure = None
        self.last_success = None
        self.last_state_change = None

class CircuitBreaker:
    """Thread-safe implementation of the circuit breaker pattern."""
    
    def __init__(self, name: str, config: Optional[CircuitBreakerConfig] = None):
        """Initialize the circuit breaker.
        
        Args:
            name: Name of the circuit breaker
            config: Configuration for the circuit breaker
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._stats = CircuitBreakerStats()
        self._lock = None  # Will be set by _ensure_lock()
        self._half_open_retries = 0
        self._state_handlers = {
            CircuitState.CLOSED: self._handle_closed_state,
            CircuitState.OPEN: self._handle_open_state,
            CircuitState.HALF_OPEN: self._handle_half_open_state
        }
        self._ensure_lock()
    
    def _ensure_lock(self) -> None:
        """Ensure the lock is initialized (for thread safety)."""
        if self._lock is None:
            import threading
            self._lock = threading.RLock()
    
    @property
    def state(self) -> CircuitState:
        """Get the current state of the circuit breaker."""
        return self._state
    
    @state.setter
    def state(self, new_state: CircuitState) -> None:
        """Set the state of the circuit breaker with thread safety."""
        with self._lock:
            if self._state != new_state:
                old_state = self._state
                self._state = new_state
                self._stats.state_changes += 1
                self._stats.last_state_change = datetime.utcnow()
                
                # Reset appropriate counters on state change
                if new_state == CircuitState.CLOSED:
                    self._stats.failures = 0
                    self._half_open_retries = 0
                elif new_state == CircuitState.OPEN:
                    self._half_open_retries = 0
                
                logger.info(
                    "Circuit '%s' state changed: %s -> %s",
                    self.name, old_state.name, new_state.name
                )
    
    def record_success(self) -> None:
        """Record a successful operation."""
        with self._lock:
            self._stats.successes += 1
            self._stats.last_success = datetime.utcnow()
            
            if self.state == CircuitState.HALF_OPEN:
                self._half_open_retries += 1
                if self._half_open_retries >= self.config.success_threshold:
                    self.state = CircuitState.CLOSED
    
    def record_failure(self) -> None:
        """Record a failed operation."""
        with self._lock:
            self._stats.failures += 1
            self._stats.last_failure = datetime.utcnow()
            
            if self.state == CircuitState.CLOSED:
                if self._stats.failures >= self.config.failure_threshold:
                    self.state = CircuitState.OPEN
            elif self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
    
    def _handle_closed_state(self, *args: Any, **kwargs: Any) -> None:
        """Handle operations in CLOSED state."""
        if self._stats.failures >= self.config.failure_threshold:
            self.state = CircuitState.OPEN
            raise CircuitBreakerError(self.name, self.state)
    
    def _handle_open_state(self, *args: Any, **kwargs: Any) -> None:
        """Handle operations in OPEN state."""
        last_failure = self._stats.last_failure
        if last_failure is None:
            raise CircuitBreakerError(self.name, self.state)
            
        elapsed = (datetime.utcnow() - last_failure).total_seconds()
        if elapsed >= self.config.recovery_timeout:
            self.state = CircuitState.HALF_OPEN
            self._half_open_retries = 0
        else:
            retry_after = self.config.recovery_timeout - elapsed
            raise CircuitBreakerError(
                self.name, self.state, retry_after=retry_after
            )
    
    def _handle_half_open_state(self, *args: Any, **kwargs: Any) -> None:
        """Handle operations in HALF_OPEN state."""
        if self._half_open_retries >= self.config.max_retries:
            self.state = CircuitState.OPEN
            raise CircuitBreakerError(self.name, self.state)
    
    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Decorator to wrap functions with circuit breaker logic."""
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Check circuit state before execution
            with self._lock:
                handler = self._state_handlers[self.state]
                handler(*args, **kwargs)
            
            # Execute the wrapped function
            try:
                result = func(*args, **kwargs)
                self.record_success()
                return result
            except Exception as e:
                self.record_failure()
                if isinstance(e, CircuitBreakerError):
                    raise
                # Wrap other exceptions in CircuitBreakerError
                raise CircuitBreakerError(
                    self.name, self.state
                ) from e
        
        return wrapper
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics for the circuit breaker."""
        with self._lock:
            return {
                "name": self.name,
                "state": self.state.name,
                "failures": self._stats.failures,
                "successes": self._stats.successes,
                "state_changes": self._stats.state_changes,
                "last_failure": self._stats.last_failure.isoformat() if self._stats.last_failure else None,
                "last_success": self._stats.last_success.isoformat() if self._stats.last_success else None,
                "last_state_change": self._stats.last_state_change.isoformat() if self._stats.last_state_change else None,
                "half_open_retries": self._half_open_retries,
                "config": self.config.dict()
            }

# Global circuit breaker registry
_global_circuit_breakers: Dict[str, CircuitBreaker] = {}

def get_circuit_breaker(
    name: str, 
    config: Optional[CircuitBreakerConfig] = None
) -> CircuitBreaker:
    """Get or create a named circuit breaker with thread safety."""
    if name not in _global_circuit_breakers:
        _global_circuit_breakers[name] = CircuitBreaker(name, config)
    return _global_circuit_breakers[name]

def get_all_circuit_breakers() -> Dict[str, Dict[str, Any]]:
    """Get statistics for all circuit breakers."""
    return {name: cb.get_stats() for name, cb in _global_circuit_breakers.items()}

def reset_all_circuit_breakers() -> None:
    """Reset all circuit breakers to CLOSED state."""
    for cb in _global_circuit_breakers.values():
        cb.state = CircuitState.CLOSED
