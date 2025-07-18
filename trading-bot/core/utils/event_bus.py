"""
Event Bus implementation for decoupled component communication.

This module provides a publish-subscribe mechanism for system-wide events,
supporting both synchronous and asynchronous event handling.
"""
from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import (
    Any, Callable, Coroutine, DefaultDict, Dict, List, Optional, Set, Type, TypeVar, Union
)
from uuid import uuid4

import orjson
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

T = TypeVar('T', bound='Event')
EventHandler = Callable[[Any], Union[None, Coroutine[Any, Any, None]]]


class EventType(str, Enum):
    """Core event types in the system."""
    ORDER_CREATED = "order_created"
    ORDER_UPDATED = "order_updated"
    ORDER_FILLED = "order_filled"
    ORDER_CANCELED = "order_canceled"
    MARKET_DATA = "market_data"
    RISK_VIOLATION = "risk_violation"
    CIRCUIT_BROKEN = "circuit_broken"
    SYSTEM_ALERT = "system_alert"


class Event(BaseModel):
    """Base event class with common attributes and serialization."""
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    event_type: EventType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    source: str = "system"
    data: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_loads = orjson.loads
        json_dumps = orjson.dumps
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }

    @classmethod
    def parse_raw(cls: Type[T], b: Union[bytes, str], **kwargs) -> T:
        """Parse event from raw JSON bytes or string."""
        if isinstance(b, str):
            b = b.encode('utf-8')
        return super().parse_raw(b, **kwargs)


class EventBus(ABC):
    """Abstract base class for event bus implementations."""
    
    @abstractmethod
    async def publish(self, event: Event) -> None:
        """Publish an event to the bus."""
        raise NotImplementedError
    
    @abstractmethod
    def subscribe(self, event_type: EventType, handler: EventHandler) -> str:
        """Subscribe to events of a specific type."""
        raise NotImplementedError
    
    @abstractmethod
    def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe a handler using its subscription ID."""
        raise NotImplementedError


class LocalEventBus(EventBus):
    """In-memory implementation of the event bus."""
    
    def __init__(self) -> None:
        self._subscriptions: Dict[EventType, Dict[str, EventHandler]] = defaultdict(dict)
        self._lock = asyncio.Lock()
    
    async def publish(self, event: Event) -> None:
        """Publish an event to all subscribers."""
        if not isinstance(event, Event):
            raise ValueError("Event must be an instance of Event class")
            
        handlers = []
        async with self._lock:
            # Get handlers for this specific event type and for all events
            specific_handlers = self._subscriptions.get(event.event_type, {})
            all_handlers = self._subscriptions.get(EventType.SYSTEM_ALERT, {})
            handlers = list(specific_handlers.values()) + list(all_handlers.values())
        
        # Execute handlers in parallel
        tasks = []
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    tasks.append(asyncio.create_task(handler(event)))
                else:
                    # Run synchronous handlers in thread pool
                    loop = asyncio.get_event_loop()
                    tasks.append(loop.run_in_executor(None, handler, event))
            except Exception as e:
                logger.error(f"Error in event handler: {e}", exc_info=True)
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    def subscribe(self, event_type: EventType, handler: EventHandler) -> str:
        """Subscribe to events of a specific type."""
        if not callable(handler):
            raise ValueError("Handler must be callable")
            
        subscription_id = str(uuid4())
        with self._lock:
            self._subscriptions[event_type][subscription_id] = handler
        return subscription_id
    
    def unsubscribe(self, subscription_id: str) -> None:
        """Unsubscribe a handler using its subscription ID."""
        with self._lock:
            for handlers in self._subscriptions.values():
                if subscription_id in handlers:
                    del handlers[subscription_id]
                    break


class CircuitBreaker:
    """Circuit breaker pattern implementation for fault tolerance with metrics."""
    
    class State(Enum):
        CLOSED = "CLOSED"
        OPEN = "OPEN"
        HALF_OPEN = "HALF_OPEN"
    
    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_success_threshold: int = 3,
        metrics_enabled: bool = True
    ):
        """Initialize the circuit breaker.
        
        Args:
            name: Unique name for this circuit breaker
            failure_threshold: Number of failures before opening the circuit
            recovery_timeout: Time in seconds before attempting to recover
            half_open_success_threshold: Number of successes needed to close the circuit
            metrics_enabled: Whether to collect metrics
        """
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_success_threshold = half_open_success_threshold
        self.metrics_enabled = metrics_enabled
        
        self._state = self.State.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = asyncio.Lock()
        
        # Initialize metrics
        if self.metrics_enabled:
            from core.monitoring.metrics import CircuitBreakerMetrics
            self._metrics = CircuitBreakerMetrics()
            self._metrics.record_state_change(self.name, 'init', self._state.value)
    
    @property
    def state(self) -> State:
        """Get the current state of the circuit breaker."""
        return self._state
    
    async def execute(self, func: Callable[..., Coroutine], *args, **kwargs) -> Any:
        """Execute a function with circuit breaker protection.
        
        Args:
            func: The function to execute
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function
            
        Returns:
            The result of the function call
            
        Raises:
            CircuitOpenError: If the circuit is open
            Exception: If the function call fails
        """
        # Check circuit state
        if self._state == self.State.OPEN:
            if self._should_try_recovery():
                await self._set_state(self.State.HALF_OPEN)
            else:
                if self.metrics_enabled:
                    self._metrics.record_failure(self.name)
                raise CircuitOpenError(f"Circuit {self.name} is open")
        
        # Execute the function with timing
        start_time = time.monotonic()
        try:
            result = await func(*args, **kwargs)
            execution_time = time.monotonic() - start_time
            
            # Record success
            await self._on_success()
            
            # Record metrics
            if self.metrics_enabled:
                self._metrics.record_success(self.name)
                with self._metrics.time_execution(self.name) as timer:
                    # Set the observed duration
                    timer.observe(execution_time)
            
            return result
            
        except Exception as e:
            execution_time = time.monotonic() - start_time
            
            # Record failure
            await self._on_failure()
            
            # Record metrics
            if self.metrics_enabled:
                self._metrics.record_failure(self.name)
                with self._metrics.time_execution(self.name) as timer:
                    # Set the observed duration even for failures
                    timer.observe(execution_time)
            
            # Re-raise the exception
            raise
    
    async def _set_state(self, new_state: State) -> None:
        """Safely update the circuit breaker state."""
        async with self._lock:
            old_state = self._state
            self._state = new_state
            
            # Reset counters when leaving HALF_OPEN state
            if old_state == self.State.HALF_OPEN and new_state != self.State.HALF_OPEN:
                self._success_count = 0
                self._failure_count = 0
            
            # Record state transition
            if self.metrics_enabled and old_state != new_state:
                self._metrics.record_state_change(
                    self.name,
                    old_state.value,
                    new_state.value
                )
            
            logger.info(
                f"Circuit {self.name} state changed: {old_state.value} -> {new_state.value}"
            )
    
    def _should_try_recovery(self) -> bool:
        """Check if we should attempt to recover from an open state."""
        if self._last_failure_time is None:
            return True
        return (time.monotonic() - self._last_failure_time) >= self.recovery_timeout
    
    async def _on_success(self) -> None:
        """Handle a successful execution."""
        async with self._lock:
            if self._state == self.State.HALF_OPEN:
                self._success_count += 1
                
                # If we have enough successes, close the circuit
                if self._success_count >= self.half_open_success_threshold:
                    await self._set_state(self.State.CLOSED)
            
            # Reset failure count on success in CLOSED state
            if self._state == self.State.CLOSED:
                self._failure_count = 0
    
    async def _on_failure(self) -> None:
        """Handle a failed execution."""
        async with self._lock:
            self._failure_count += 1
            self._last_failure_time = time.monotonic()
            
            if self._state == self.State.HALF_OPEN:
                # If we fail in half-open state, go back to open
                await self._set_state(self.State.OPEN)
            elif (self._state == self.State.CLOSED and 
                  self._failure_count >= self.failure_threshold):
                # Too many failures, open the circuit
                await self._set_state(self.State.OPEN)


class CircuitOpenError(Exception):
    """Raised when the circuit breaker is open."""
    pass
