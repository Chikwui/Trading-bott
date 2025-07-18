"""
Advanced Circuit Breaker service for managing trading halts and resumptions
based on real-time risk metrics and market conditions.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from enum import Enum, auto
from typing import (
    Any, Dict, List, Optional, Set, Tuple, Callable, Awaitable, Union,
    DefaultDict
)
import json
from collections import defaultdict

from ..market.market_data_service import MarketDataService
from .risk_manager import RiskManager

logger = logging.getLogger(__name__)

class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    TRADING_PAUSED = "TRADING_PAUSED"
    LIQUIDATION_ONLY = "LIQUIDATION_ONLY"
    HALTED = "HALTED"

class CircuitBreakerType(Enum):
    """Types of circuit breakers."""
    PRICE_MOVEMENT = "PRICE_MOVEMENT"
    VOLATILITY = "VOLATILITY"
    LIQUIDITY = "LIQUIDITY"
    VOLUME = "VOLUME"
    POSITION_SIZE = "POSITION_SIZE"
    DAILY_LOSS = "DAILY_LOSS"
    DRAWDOWN = "DRAWDOWN"
    LEVERAGE = "LEVERAGE"
    CONCENTRATION = "CONCENTRATION"
    NEWS_EVENT = "NEWS_EVENT"
    EXCHANGE_DISRUPTION = "EXCHANGE_DISRUPTION"
    SYSTEM = "SYSTEM"
    CUSTOM = "CUSTOM"

class CircuitBreakerAction(Enum):
    """Actions to take when a circuit breaker is triggered."""
    NOTIFY = "NOTIFY"
    PAUSE_NEW_ORDERS = "PAUSE_NEW_ORDERS"
    CANCEL_OPEN_ORDERS = "CANCEL_OPEN_ORDERS"
    REDUCE_LEVERAGE = "REDUCE_LEVERAGE"
    CLOSE_POSITIONS = "CLOSE_POSITIONS"
    LIQUIDATE = "LIQUIDATE"
    HALT_TRADING = "HALT_TRADING"

@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker rule."""
    name: str
    breaker_type: CircuitBreakerType
    threshold: float
    lookback_period: timedelta
    cooldown_period: timedelta
    actions: List[CircuitBreakerAction]
    is_active: bool = True
    auto_reset: bool = True
    notify_on_trigger: bool = True
    notify_on_reset: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CircuitBreakerStatus:
    """Current status of a circuit breaker."""
    name: str
    breaker_type: CircuitBreakerType
    state: CircuitBreakerState = CircuitBreakerState.NORMAL
    current_value: float = 0.0
    triggered_at: Optional[datetime] = None
    last_reset: Optional[datetime] = None
    trigger_count: int = 0
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'breaker_type': self.breaker_type.value,
            'state': self.state.value,
            'current_value': self.current_value,
            'triggered_at': self.triggered_at.isoformat() if self.triggered_at else None,
            'last_reset': self.last_reset.isoformat() if self.last_reset else None,
            'trigger_count': self.trigger_count,
            'is_active': self.is_active,
            'metadata': self.metadata
        }

class CircuitBreakerService:
    """
    Advanced Circuit Breaker service that monitors trading conditions and
    automatically triggers circuit breakers when risk thresholds are exceeded.
    """
    
    def __init__(
        self,
        risk_manager: RiskManager,
        market_data: MarketDataService,
        check_interval: float = 1.0,  # seconds
        default_cooldown: timedelta = timedelta(minutes=5)
    ):
        """Initialize the Circuit Breaker service."""
        self.risk_manager = risk_manager
        self.market_data = market_data
        self.check_interval = check_interval
        self.default_cooldown = default_cooldown
        
        # Circuit breaker configurations
        self.breakers: Dict[str, CircuitBreakerConfig] = {}
        self.statuses: Dict[str, CircuitBreakerStatus] = {}
        
        # State
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        self._state = CircuitBreakerState.NORMAL
        self._state_history: List[Tuple[datetime, CircuitBreakerState, str]] = []
        
        # Callbacks
        self._callbacks: Dict[str, List[Callable[[Dict[str, Any]], Awaitable[None]]]] = {
            'trigger': [],
            'reset': [],
            'state_change': []
        }
        
        # Initialize default circuit breakers
        self._initialize_default_breakers()
    
    def _initialize_default_breakers(self) -> None:
        """Initialize default circuit breakers."""
        # Daily loss breaker
        self.add_breaker(CircuitBreakerConfig(
            name="daily_loss_breaker",
            breaker_type=CircuitBreakerType.DAILY_LOSS,
            threshold=float(self.risk_manager.max_daily_loss_pct * 100),  # Convert to percentage
            lookback_period=timedelta(hours=24),
            cooldown_period=timedelta(hours=1),
            actions=[
                CircuitBreakerAction.NOTIFY,
                CircuitBreakerAction.PAUSE_NEW_ORDERS,
                CircuitBreakerAction.REDUCE_LEVERAGE
            ],
            metadata={
                'severity': 'high',
                'description': 'Triggers when daily loss exceeds threshold'
            }
        ))
        
        # Drawdown breaker
        self.add_breaker(CircuitBreakerConfig(
            name="drawdown_breaker",
            breaker_type=CircuitBreakerType.DRAWDOWN,
            threshold=float(self.risk_manager.max_drawdown_pct * 100),  # Convert to percentage
            lookback_period=timedelta(days=7),
            cooldown_period=timedelta(hours=4),
            actions=[
                CircuitBreakerAction.NOTIFY,
                CircuitBreakerAction.PAUSE_NEW_ORDERS,
                CircuitBreakerAction.CLOSE_POSITIONS
            ],
            metadata={
                'severity': 'critical',
                'description': 'Triggers when account drawdown exceeds threshold'
            }
        ))
        
        # Volatility breaker
        self.add_breaker(CircuitBreakerConfig(
            name="volatility_breaker",
            breaker_type=CircuitBreakerType.VOLATILITY,
            threshold=0.05,  # 5% daily volatility
            lookback_period=timedelta(days=5),
            cooldown_period=timedelta(hours=2),
            actions=[
                CircuitBreakerAction.NOTIFY,
                CircuitBreakerAction.REDUCE_LEVERAGE
            ],
            metadata={
                'severity': 'medium',
                'description': 'Triggers when market volatility exceeds threshold'
            }
        ))
        
        # Concentration breaker
        self.add_breaker(CircuitBreakerConfig(
            name="concentration_breaker",
            breaker_type=CircuitBreakerType.CONCENTRATION,
            threshold=float(self.risk_manager.max_concentration * 2 * 100),  # 2x max concentration
            lookback_period=timedelta(hours=1),
            cooldown_period=timedelta(minutes=30),
            actions=[
                CircuitBreakerAction.NOTIFY,
                CircuitBreakerAction.PAUSE_NEW_ORDERS
            ],
            metadata={
                'severity': 'high',
                'description': 'Triggers when position concentration exceeds threshold'
            }
        ))
    
    async def start(self) -> None:
        """Start the circuit breaker service."""
        if self._running:
            return
            
        logger.info("Starting Circuit Breaker Service...")
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Circuit Breaker Service started")
    
    async def stop(self) -> None:
        """Stop the circuit breaker service."""
        if not self._running:
            return
            
        logger.info("Stopping Circuit Breaker Service...")
        self._running = False
        
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Circuit Breaker Service stopped")
    
    def add_breaker(self, config: CircuitBreakerConfig) -> None:
        """Add a new circuit breaker configuration."""
        self.breakers[config.name] = config
        
        # Initialize status if not exists
        if config.name not in self.statuses:
            self.statuses[config.name] = CircuitBreakerStatus(
                name=config.name,
                breaker_type=config.breaker_type,
                is_active=config.is_active,
                metadata=config.metadata.copy()
            )
    
    def update_breaker(self, name: str, **updates) -> bool:
        """Update an existing circuit breaker configuration."""
        if name not in self.breakers:
            return False
            
        breaker = self.breakers[name]
        
        for key, value in updates.items():
            if hasattr(breaker, key):
                setattr(breaker, key, value)
        
        # Update status if needed
        if name in self.statuses:
            self.statuses[name].is_active = breaker.is_active
            
        return True
    
    def remove_breaker(self, name: str) -> bool:
        """Remove a circuit breaker configuration."""
        if name in self.breakers:
            del self.breakers[name]
            
            # Don't remove status, just mark as inactive
            if name in self.statuses:
                self.statuses[name].is_active = False
                
            return True
            
        return False
    
    async def trigger_breaker(
        self,
        name: str,
        value: Optional[float] = None,
        reason: Optional[str] = None
    ) -> bool:
        """Manually trigger a circuit breaker."""
        if name not in self.breakers or name not in self.statuses:
            return False
            
        config = self.breakers[name]
        status = self.statuses[name]
        
        # Skip if already triggered and not auto-reset
        if status.state != CircuitBreakerState.NORMAL and not config.auto_reset:
            return False
        
        # Update status
        status.state = CircuitBreakerState.WARNING
        status.triggered_at = datetime.now(timezone.utc)
        status.trigger_count += 1
        
        if value is not None:
            status.current_value = value
        
        # Execute actions
        await self._execute_actions(config.actions, config, status, reason)
        
        # Update global state if needed
        self._update_global_state()
        
        # Notify
        await self._notify_trigger(config, status, reason)
        
        logger.warning(f"Circuit breaker '{name}' triggered: {reason or 'No reason provided'}")
        return True
    
    async def reset_breaker(self, name: str, reason: Optional[str] = None) -> bool:
        """Reset a triggered circuit breaker."""
        if name not in self.breakers or name not in self.statuses:
            return False
            
        config = self.breakers[name]
        status = self.statuses[name]
        
        # Skip if not triggered
        if status.state == CircuitBreakerState.NORMAL:
            return False
        
        # Update status
        previous_state = status.state
        status.state = CircuitBreakerState.NORMAL
        status.last_reset = datetime.now(timezone.utc)
        
        # Update global state
        self._update_global_state()
        
        # Notify
        if config.notify_on_reset:
            await self._notify_reset(config, status, previous_state, reason)
        
        logger.info(f"Circuit breaker '{name}' reset: {reason or 'No reason provided'}")
        return True
    
    def get_status(self, name: str) -> Optional[Dict[str, Any]]:
        """Get the status of a circuit breaker."""
        if name in self.statuses:
            return self.statuses[name].to_dict()
        return None
    
    def get_all_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get the status of all circuit breakers."""
        return {name: status.to_dict() for name, status in self.statuses.items()}
    
    def get_global_state(self) -> Dict[str, Any]:
        """Get the global circuit breaker state."""
        return {
            'state': self._state.value,
            'state_since': self._state_history[-1][0].isoformat() if self._state_history else None,
            'active_breakers': [
                name for name, status in self.statuses.items()
                if status.state != CircuitBreakerState.NORMAL and status.is_active
            ]
        }
    
    def register_callback(
        self,
        event_type: str,
        callback: Callable[[Dict[str, Any]], Awaitable[None]]
    ) -> None:
        """Register a callback for circuit breaker events."""
        if event_type in self._callbacks:
            self._callbacks[event_type].append(callback)
    
    async def _monitor_loop(self) -> None:
        """Main monitoring loop for circuit breakers."""
        while self._running:
            try:
                # Check all active circuit breakers
                for name, config in self.breakers.items():
                    if not config.is_active or name not in self.statuses:
                        continue
                        
                    status = self.statuses[name]
                    
                    # Skip if in cooldown
                    if (status.triggered_at and 
                        (datetime.now(timezone.utc) - status.triggered_at) < config.cooldown_period):
                        continue
                    
                    # Get current value for this breaker
                    value = await self._get_breaker_value(config)
                    status.current_value = value
                    
                    # Check threshold
                    if value >= config.threshold and status.state == CircuitBreakerState.NORMAL:
                        await self.trigger_breaker(
                            name,
                            value=value,
                            reason=f"Threshold exceeded: {value:.4f} >= {config.threshold:.4f}"
                        )
                    # Auto-reset if below threshold and auto-reset is enabled
                    elif value < config.threshold and config.auto_reset and status.state != CircuitBreakerState.NORMAL:
                        await self.reset_breaker(
                            name,
                            reason=f"Value below threshold: {value:.4f} < {config.threshold:.4f}"
                        )
                
                # Small sleep to prevent tight loop
                await asyncio.sleep(self.check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in circuit breaker monitor: {e}", exc_info=True)
                await asyncio.sleep(5)  # Prevent tight loop on errors
    
    async def _get_breaker_value(self, config: CircuitBreakerConfig) -> float:
        """Get the current value for a circuit breaker."""
        if config.breaker_type == CircuitBreakerType.DAILY_LOSS:
            # Get daily P&L as percentage of account
            return float(abs(self.risk_manager.daily_pnl) / self.risk_manager.account_balance * 100)
            
        elif config.breaker_type == CircuitBreakerType.DRAWDOWN:
            # Get current drawdown
            if not self.risk_manager.historical_balance:
                return 0.0
                
            peak = max(bal for _, bal in self.risk_manager.historical_balance)
            current = self.risk_manager.historical_balance[-1][1]
            return float((peak - current) / peak * 100)
            
        elif config.breaker_type == CircuitBreakerType.VOLATILITY:
            # Calculate 24h rolling volatility
            # This is a simplified example - in production, use proper volatility calculation
            return 0.02  # Placeholder
            
        elif config.breaker_type == CircuitBreakerType.CONCENTRATION:
            # Calculate max position concentration
            positions = await self.risk_manager.position_manager.get_positions()
            if not positions:
                return 0.0
                
            portfolio_value = await self.risk_manager._calculate_portfolio_value()
            if portfolio_value <= 0:
                return 0.0
                
            max_concentration = max(
                (p.quantity * (p.avg_entry_price or Decimal('1'))) / portfolio_value
                for p in positions
            )
            return float(max_concentration * 100)
            
        # Add more breaker types as needed
        
        return 0.0
    
    async def _execute_actions(
        self,
        actions: List[CircuitBreakerAction],
        config: CircuitBreakerConfig,
        status: CircuitBreakerStatus,
        reason: Optional[str] = None
    ) -> None:
        """Execute actions for a triggered circuit breaker."""
        for action in actions:
            try:
                if action == CircuitBreakerAction.NOTIFY:
                    # Already handled by _notify_trigger
                    pass
                    
                elif action == CircuitBreakerAction.PAUSE_NEW_ORDERS:
                    # In a real implementation, this would pause order submission
                    logger.warning(f"PAUSING NEW ORDERS due to {config.name} trigger")
                    
                elif action == CircuitBreakerAction.CANCEL_OPEN_ORDERS:
                    # In a real implementation, this would cancel all open orders
                    logger.warning(f"CANCELLING OPEN ORDERS due to {config.name} trigger")
                    
                elif action == CircuitBreakerAction.REDUCE_LEVERAGE:
                    # In a real implementation, this would reduce leverage
                    logger.warning(f"REDUCING LEVERAGE due to {config.name} trigger")
                    
                elif action == CircuitBreakerAction.CLOSE_POSITIONS:
                    # In a real implementation, this would start closing positions
                    logger.warning(f"CLOSING POSITIONS due to {config.name} trigger")
                    
                elif action == CircuitBreakerAction.LIQUIDATE:
                    # In a real implementation, this would liquidate all positions
                    logger.warning(f"LIQUIDATING POSITIONS due to {config.name} trigger")
                    
                elif action == CircuitBreakerAction.HALT_TRADING:
                    # In a real implementation, this would halt all trading
                    logger.critical(f"HALTING ALL TRADING due to {config.name} trigger")
                    self._state = CircuitBreakerState.HALTED
                    self._record_state_change("Halted by circuit breaker")
                    
            except Exception as e:
                logger.error(f"Error executing circuit breaker action {action}: {e}", exc_info=True)
    
    def _update_global_state(self) -> None:
        """Update the global circuit breaker state based on active breakers."""
        active_breakers = [
            status for status in self.statuses.values()
            if status.state != CircuitBreakerState.NORMAL and status.is_active
        ]
        
        if not active_breakers:
            new_state = CircuitBreakerState.NORMAL
            reason = "All circuit breakers normal"
        else:
            # Find the most severe state among active breakers
            state_priority = {
                CircuitBreakerState.HALTED: 4,
                CircuitBreakerState.LIQUIDATION_ONLY: 3,
                CircuitBreakerState.TRADING_PAUSED: 2,
                CircuitBreakerState.WARNING: 1,
                CircuitBreakerState.NORMAL: 0
            }
            
            new_state = max(
                (status.state for status in active_breakers),
                key=lambda s: state_priority.get(s, 0)
            )
            
            reason = f"Active circuit breakers: {', '.join(b.name for b in active_breakers)}"
        
        # Update state if changed
        if new_state != self._state:
            old_state = self._state
            self._state = new_state
            self._record_state_change(reason)
            
            # Notify state change
            asyncio.create_task(self._notify_state_change(old_state, new_state, reason))
    
    def _record_state_change(self, reason: str) -> None:
        """Record a state change in history."""
        self._state_history.append((datetime.now(timezone.utc), self._state, reason))
        # Keep history size manageable
        if len(self._state_history) > 1000:
            self._state_history = self._state_history[-1000:]
    
    async def _notify_trigger(
        self,
        config: CircuitBreakerConfig,
        status: CircuitBreakerStatus,
        reason: Optional[str] = None
    ) -> None:
        """Notify about a circuit breaker trigger."""
        if not config.notify_on_trigger:
            return
            
        event = {
            'event': 'circuit_breaker_triggered',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'breaker': {
                'name': config.name,
                'type': config.breaker_type.value,
                'threshold': config.threshold,
                'current_value': status.current_value,
                'state': status.state.value,
                'triggered_at': status.triggered_at.isoformat() if status.triggered_at else None,
                'trigger_count': status.trigger_count,
                'reason': reason
            },
            'actions': [action.value for action in config.actions]
        }
        
        # Call registered callbacks
        for callback in self._callbacks['trigger']:
            try:
                await callback(event)
            except Exception as e:
                logger.error(f"Error in circuit breaker callback: {e}", exc_info=True)
    
    async def _notify_reset(
        self,
        config: CircuitBreakerConfig,
        status: CircuitBreakerStatus,
        previous_state: CircuitBreakerState,
        reason: Optional[str] = None
    ) -> None:
        """Notify about a circuit breaker reset."""
        event = {
            'event': 'circuit_breaker_reset',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'breaker': {
                'name': config.name,
                'type': config.breaker_type.value,
                'previous_state': previous_state.value,
                'current_state': status.state.value,
                'last_reset': status.last_reset.isoformat() if status.last_reset else None,
                'reason': reason
            }
        }
        
        # Call registered callbacks
        for callback in self._callbacks['reset']:
            try:
                await callback(event)
            except Exception as e:
                logger.error(f"Error in circuit breaker reset callback: {e}", exc_info=True)
    
    async def _notify_state_change(
        self,
        old_state: CircuitBreakerState,
        new_state: CircuitBreakerState,
        reason: str
    ) -> None:
        """Notify about a global state change."""
        event = {
            'event': 'circuit_breaker_state_changed',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'old_state': old_state.value,
            'new_state': new_state.value,
            'reason': reason
        }
        
        # Call registered callbacks
        for callback in self._callbacks['state_change']:
            try:
                await callback(event)
            except Exception as e:
                logger.error(f"Error in state change callback: {e}", exc_info=True)
