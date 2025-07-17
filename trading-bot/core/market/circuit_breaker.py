"""
Circuit breaker implementation for risk management.

This module provides a circuit breaker pattern to protect the trading system
from excessive losses, extreme market conditions, and system failures.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Callable, Any, Tuple
import time
import asyncio
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class CircuitBreakerState(Enum):
    """Possible states of a circuit breaker."""
    CLOSED = auto()     # Normal operation
    OPEN = auto()       # Circuit is open, no new positions allowed
    HALF_OPEN = auto()  # Testing if conditions have improved

class BreakerType(Enum):
    """Types of circuit breakers."""
    DRAWDOWN = auto()           # Maximum drawdown limit
    LOSS = auto()               # Consecutive losing trades
    VOLATILITY = auto()         # Market volatility
    LIQUIDITY = auto()          # Market liquidity
    LATENCY = auto()            # System latency
    ERROR_RATE = auto()         # System error rate
    POSITION = auto()           # Position size limit
    LEVERAGE = auto()           # Account leverage limit
    MARGIN = auto()            # Margin level
    CUSTOM = auto()             # Custom condition

@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker."""
    breaker_type: BreakerType
    threshold: float
    lookback_period: timedelta = timedelta(minutes=5)
    cooldown_period: timedelta = timedelta(minutes=15)
    action: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)

class CircuitBreakerAction:
    """Base class for circuit breaker actions."""
    
    async def execute(self, breaker: 'CircuitBreaker', **kwargs) -> bool:
        """Execute the action when the circuit breaker is triggered."""
        raise NotImplementedError

class ClosePositionsAction(CircuitBreakerAction):
    """Action to close all open positions."""
    
    async def execute(self, breaker: 'CircuitBreaker', **kwargs) -> bool:
        """Execute the close positions action."""
        logger.warning(f"Circuit breaker {breaker.name}: Closing all positions")
        # Implementation would depend on your trading system
        # Example: await trading_system.close_all_positions()
        return True

class DisableTradingAction(CircuitBreakerAction):
    """Action to disable trading."""
    
    async def execute(self, breaker: 'CircuitBreaker', **kwargs) -> bool:
        """Execute the disable trading action."""
        logger.warning(f"Circuit breaker {breaker.name}: Disabling trading")
        # Implementation would depend on your trading system
        # Example: trading_system.disable_trading()
        return True

class ReduceLeverageAction(CircuitBreakerAction):
    """Action to reduce leverage."""
    
    async def execute(self, breaker: 'CircuitBreaker', **kwargs) -> bool:
        """Execute the reduce leverage action."""
        target_leverage = kwargs.get('target_leverage', 1.0)
        logger.warning(
            f"Circuit breaker {breaker.name}: "
            f"Reducing leverage to {target_leverage}x"
        )
        # Implementation would depend on your trading system
        # Example: await trading_system.adjust_leverage(target_leverage)
        return True

class CircuitBreaker:
    """Manages a single circuit breaker."""
    
    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig,
        check_condition: Callable[..., bool],
        action: Optional[CircuitBreakerAction] = None
    ):
        """
        Initialize the circuit breaker.
        
        Args:
            name: Name of the circuit breaker
            config: Configuration for the circuit breaker
            check_condition: Function that checks if the breaker should trigger
            action: Action to take when the breaker is triggered
        """
        self.name = name
        self.config = config
        self._check_condition = check_condition
        self.action = action
        
        self.state = CircuitBreakerState.CLOSED
        self.last_triggered: Optional[datetime] = None
        self.trigger_count = 0
        self.metrics: List[Tuple[datetime, float]] = []
        
    def update_metric(self, value: float, timestamp: Optional[datetime] = None) -> None:
        """Update the metric being monitored by this circuit breaker."""
        ts = timestamp or datetime.utcnow()
        self.metrics.append((ts, value))
        
        # Clean up old metrics
        cutoff = ts - self.config.lookback_period
        self.metrics = [(t, v) for t, v in self.metrics if t >= cutoff]
    
    def check(self) -> bool:
        """
        Check if the circuit breaker should trigger.
        
        Returns:
            bool: True if the breaker is triggered, False otherwise
        """
        now = datetime.utcnow()
        
        # Skip if in cooldown
        if self.state == CircuitBreakerState.OPEN:
            if (self.last_triggered and 
                now < self.last_triggered + self.config.cooldown_period):
                return False
            self.state = CircuitBreakerState.HALF_OPEN
        
        # Check the condition
        should_trigger = self._check_condition(self)
        
        # Update state based on condition
        if should_trigger:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.OPEN
                self.last_triggered = now
            elif self.state == CircuitBreakerState.CLOSED:
                self.state = CircuitBreakerState.OPEN
                self.last_triggered = now
                self.trigger_count += 1
            return True
        else:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.CLOSED
            return False
    
    async def trigger(self, **kwargs) -> bool:
        """
        Trigger the circuit breaker's action.
        
        Returns:
            bool: True if the action was successful, False otherwise
        """
        if self.action:
            return await self.action.execute(self, **kwargs)
        return True

class CircuitBreakerManager:
    """Manages multiple circuit breakers."""
    
    def __init__(self):
        """Initialize the circuit breaker manager."""
        self.breakers: Dict[str, CircuitBreaker] = {}
        self._enabled = True
        self._task: Optional[asyncio.Task] = None
        
    def add_breaker(self, name: str, breaker: CircuitBreaker) -> None:
        """Add a circuit breaker to the manager."""
        self.breakers[name] = breaker
    
    def remove_breaker(self, name: str) -> None:
        """Remove a circuit breaker from the manager."""
        if name in self.breakers:
            del self.breakers[name]
    
    def get_breaker(self, name: str) -> Optional[CircuitBreaker]:
        """Get a circuit breaker by name."""
        return self.breakers.get(name)
    
    def enable(self) -> None:
        """Enable all circuit breakers."""
        self._enabled = True
    
    def disable(self) -> None:
        """Disable all circuit breakers."""
        self._enabled = False
    
    async def check_all(self) -> Dict[str, bool]:
        """
        Check all circuit breakers.
        
        Returns:
            Dict mapping breaker names to whether they triggered
        """
        if not self._enabled:
            return {}
            
        results = {}
        for name, breaker in self.breakers.items():
            try:
                triggered = breaker.check()
                results[name] = triggered
                if triggered:
                    await breaker.trigger()
            except Exception as e:
                logger.error(f"Error checking circuit breaker {name}: {e}")
                results[name] = False
        
        return results
    
    async def run(self, interval: float = 1.0) -> None:
        """
        Run the circuit breaker manager in the background.
        
        Args:
            interval: Seconds between checks
        """
        self._task = asyncio.create_task(self._run_loop(interval))
    
    async def stop(self) -> None:
        """Stop the circuit breaker manager."""
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
    
    async def _run_loop(self, interval: float) -> None:
        """Background loop for checking circuit breakers."""
        while True:
            try:
                await self.check_all()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in circuit breaker loop: {e}")
                await asyncio.sleep(interval)  # Prevent tight loop on error
