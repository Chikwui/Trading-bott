"""
Advanced Risk Management System with real-time monitoring and circuit breakers.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum, auto
from typing import (
    Any, Dict, List, Optional, Set, Tuple, Callable, Awaitable, Union,
    DefaultDict
)
import numpy as np
from scipy.stats import norm
import pandas as pd

from ..market.market_data_service import MarketDataService
from ..trading.order_types import Order, OrderSide, OrderType, OrderStatus
from ..trading.position_manager import Position, PositionManager

logger = logging.getLogger(__name__)

class RiskViolationType(Enum):
    """Types of risk violations."""
    MAX_POSITION_SIZE = "MAX_POSITION_SIZE"
    MAX_ORDER_SIZE = "MAX_ORDER_SIZE"
    MAX_DAILY_LOSS = "MAX_DAILY_LOSS"
    MAX_DRAWDOWN = "MAX_DRAWDOWN"
    MAX_LEVERAGE = "MAX_LEVERAGE"
    CONCENTRATION = "CONCENTRATION"
    LIQUIDITY = "LIQUIDITY"
    VOLATILITY = "VOLATILITY"
    CORRELATION = "CORRELATION"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"
    SESSION_RISK = "SESSION_RISK"
    NEWS_EVENT = "NEWS_EVENT"

class RiskSeverity(Enum):
    """Severity levels for risk violations."""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class RiskViolation:
    """Represents a risk violation event."""
    violation_type: RiskViolationType
    severity: RiskSeverity
    message: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    details: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True

class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    TRADING_PAUSED = "TRADING_PAUSED"
    LIQUIDATION_ONLY = "LIQUIDATION_ONLY"
    HALTED = "HALTED"

@dataclass
class CircuitBreaker:
    """Circuit breaker configuration and state."""
    name: str
    threshold: float
    lookback_period: timedelta
    cooldown_period: timedelta
    action: Callable[['RiskManager'], Awaitable[None]]
    state: CircuitBreakerState = CircuitBreakerState.NORMAL
    last_triggered: Optional[datetime] = None
    current_value: float = 0.0
    history: List[Tuple[datetime, float]] = field(default_factory=list)

    def update(self, value: float) -> bool:
        """Update circuit breaker with new value and check if triggered."""
        now = datetime.now(timezone.utc)
        self.current_value = value
        self.history.append((now, value))
        
        # Clean up old history
        cutoff = now - self.lookback_period
        self.history = [(t, v) for t, v in self.history if t >= cutoff]
        
        # Check cooldown
        if (self.last_triggered and 
            (now - self.last_triggered) < self.cooldown_period):
            return False
            
        # Check threshold
        if value >= self.threshold:
            self.last_triggered = now
            self.state = CircuitBreakerState.WARNING
            return True
            
        return False

class RiskManager:
    """
    Advanced Risk Management System with real-time monitoring and circuit breakers.
    """
    
    def __init__(
        self,
        market_data: MarketDataService,
        position_manager: PositionManager,
        account_balance: Decimal = Decimal('100000'),  # Default account balance
        max_position_size_pct: float = 5.0,  # Max position size as % of account
        max_order_size_pct: float = 2.0,     # Max order size as % of account
        max_daily_loss_pct: float = 2.0,     # Max daily loss as % of account
        max_drawdown_pct: float = 10.0,      # Max drawdown as % of account
        max_leverage: float = 10.0,          # Maximum allowed leverage
        max_concentration: float = 20.0,     # Max concentration in a single position (%)
        volatility_lookback: int = 21,       # Days for volatility calculation
        correlation_lookback: int = 63,      # Days for correlation calculation
        risk_free_rate: float = 0.05,        # Risk-free rate for risk-adjusted returns
        enable_circuit_breakers: bool = True,
    ):
        self.market_data = market_data
        self.position_manager = position_manager
        self.account_balance = account_balance
        self.starting_balance = account_balance
        self.historical_balance = [(datetime.now(timezone.utc), account_balance)]
        
        # Risk parameters
        self.max_position_size_pct = Decimal(str(max_position_size_pct)) / 100
        self.max_order_size_pct = Decimal(str(max_order_size_pct)) / 100
        self.max_daily_loss_pct = Decimal(str(max_daily_loss_pct)) / 100
        self.max_drawdown_pct = Decimal(str(max_drawdown_pct)) / 100
        self.max_leverage = Decimal(str(max_leverage))
        self.max_concentration = Decimal(str(max_concentration)) / 100
        self.volatility_lookback = volatility_lookback
        self.correlation_lookback = correlation_lookback
        self.risk_free_rate = Decimal(str(risk_free_rate))
        
        # State
        self.violations: List[RiskViolation] = []
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.trading_enabled = True
        self.last_check_time = datetime.now(timezone.utc)
        self.daily_pnl = Decimal('0')
        self.max_daily_drawdown = Decimal('0')
        self.portfolio_metrics: Dict[str, Any] = {}
        
        # Initialize circuit breakers if enabled
        if enable_circuit_breakers:
            self._initialize_circuit_breakers()
    
    def _initialize_circuit_breakers(self) -> None:
        """Initialize default circuit breakers."""
        # Daily loss circuit breaker
        self.add_circuit_breaker(
            name="daily_loss_breaker",
            threshold=float(self.max_daily_loss_pct * 100),  # Convert to percentage
            lookback_period=timedelta(hours=24),
            cooldown_period=timedelta(hours=1),
            action=self._on_daily_loss_breach
        )
        
        # Drawdown circuit breaker
        self.add_circuit_breaker(
            name="drawdown_breaker",
            threshold=float(self.max_drawdown_pct * 100),  # Convert to percentage
            lookback_period=timedelta(days=7),
            cooldown_period=timedelta(hours=4),
            action=self._on_drawdown_breach
        )
        
        # Volatility circuit breaker
        self.add_circuit_breaker(
            name="volatility_breaker",
            threshold=0.05,  # 5% daily volatility
            lookback_period=timedelta(days=5),
            cooldown_period=timedelta(hours=2),
            action=self._on_volatility_breach
        )
        
        # Concentration circuit breaker
        self.add_circuit_breaker(
            name="concentration_breaker",
            threshold=float(self.max_concentration * 2 * 100),  # 2x max concentration
            lookback_period=timedelta(hours=1),
            cooldown_period=timedelta(minutes=30),
            action=self._on_concentration_breach
        )
    
    async def add_circuit_breaker(
        self,
        name: str,
        threshold: float,
        lookback_period: timedelta,
        cooldown_period: timedelta,
        action: Callable[['RiskManager'], Awaitable[None]]
    ) -> None:
        """Add a custom circuit breaker."""
        self.circuit_breakers[name] = CircuitBreaker(
            name=name,
            threshold=threshold,
            lookback_period=lookback_period,
            cooldown_period=cooldown_period,
            action=action
        )
    
    async def update_account_balance(self, new_balance: Decimal) -> None:
        """Update the account balance and track historical values."""
        now = datetime.now(timezone.utc)
        self.account_balance = new_balance
        self.historical_balance.append((now, new_balance))
        
        # Calculate daily P&L
        if now.date() > self.last_check_time.date():
            self.daily_pnl = Decimal('0')
        else:
            self.daily_pnl += (new_balance - self.historical_balance[-2][1])
        
        self.last_check_time = now
        
        # Update drawdown metrics
        self._update_drawdown_metrics()
    
    def _update_drawdown_metrics(self) -> None:
        """Update drawdown-related metrics."""
        if not self.historical_balance:
            return
            
        peak = max(bal for _, bal in self.historical_balance)
        current_drawdown = (peak - self.account_balance) / peak * 100
        
        if current_drawdown > self.max_daily_drawdown:
            self.max_daily_drawdown = current_drawdown
            
            # Check if we've breached max drawdown
            if current_drawdown > self.max_drawdown_pct * 100:
                self._add_violation(
                    RiskViolationType.MAX_DRAWDOWN,
                    RiskSeverity.CRITICAL,
                    f"Max drawdown limit breached: {current_drawdown:.2f}%"
                )
    
    async def check_order_risk(self, order: Order) -> List[RiskViolation]:
        """Check if an order violates any risk parameters."""
        violations = []
        
        # 1. Check order size vs max order size
        order_value = order.quantity * (order.price or Decimal('0'))
        max_order_value = self.account_balance * self.max_order_size_pct
        
        if order_value > max_order_value:
            violations.append(RiskViolation(
                violation_type=RiskViolationType.MAX_ORDER_SIZE,
                severity=RiskSeverity.ERROR,
                message=f"Order size {order_value:.2f} exceeds max order size {max_order_value:.2f}",
                details={
                    'order_size': float(order_value),
                    'max_allowed': float(max_order_value),
                    'percent_used': float((order_value / max_order_value) * 100)
                }
            ))
        
        # 2. Check position size limits
        current_position = await self.position_manager.get_open_position(order.symbol)
        new_position_size = (current_position.quantity + order.quantity) if current_position else order.quantity
        max_position_size = self.account_balance * self.max_position_size_pct
        
        if new_position_size > max_position_size:
            violations.append(RiskViolation(
                violation_type=RiskViolationType.MAX_POSITION_SIZE,
                severity=RiskSeverity.ERROR,
                message=f"Position size {new_position_size:.2f} exceeds max position size {max_position_size:.2f}",
                details={
                    'current_size': float(current_position.quantity) if current_position else 0.0,
                    'new_size': float(new_position_size),
                    'max_allowed': float(max_position_size)
                }
            ))
        
        # 3. Check concentration limits
        portfolio_value = await self._calculate_portfolio_value()
        position_value = new_position_size * (order.price or Decimal('0'))
        concentration_pct = (position_value / portfolio_value * 100) if portfolio_value > 0 else 0
        
        if concentration_pct > self.max_concentration * 100:
            violations.append(RiskViolation(
                violation_type=RiskViolationType.CONCENTRATION,
                severity=RiskSeverity.WARNING,
                message=f"Position concentration {concentration_pct:.2f}% exceeds max {self.max_concentration*100:.2f}%",
                details={
                    'concentration_pct': float(concentration_pct),
                    'max_allowed_pct': float(self.max_concentration * 100)
                }
            ))
        
        # 4. Check leverage limits
        total_position_value = await self._calculate_total_position_value()
        leverage = (total_position_value + order_value) / self.account_balance
        
        if leverage > self.max_leverage:
            violations.append(RiskViolation(
                violation_type=RiskViolationType.MAX_LEVERAGE,
                severity=RiskSeverity.ERROR,
                message=f"Leverage {leverage:.2f}x exceeds max {self.max_leverage}x",
                details={
                    'current_leverage': float(leverage),
                    'max_leverage': float(self.max_leverage)
                }
            ))
        
        # 5. Check liquidity risk (implementation depends on market data)
        # 6. Check volatility risk (requires historical data)
        
        # Add violations to history
        self.violations.extend(violations)
        
        return violations
    
    async def check_portfolio_risk(self) -> List[RiskViolation]:
        """Check portfolio-wide risk metrics."""
        violations = []
        
        # 1. Calculate portfolio metrics
        portfolio_value = await self._calculate_portfolio_value()
        total_position_value = await self._calculate_total_position_value()
        
        # 2. Check daily P&L
        daily_pnl_pct = (self.daily_pnl / self.account_balance * 100) if self.account_balance > 0 else 0
        
        if daily_pnl_pct < -float(self.max_daily_loss_pct * 100):
            violations.append(RiskViolation(
                violation_type=RiskViolationType.MAX_DAILY_LOSS,
                severity=RiskSeverity.CRITICAL,
                message=f"Daily P&L {daily_pnl_pct:.2f}% below maximum loss threshold {-self.max_daily_loss_pct*100:.2f}%",
                details={
                    'daily_pnl': float(self.daily_pnl),
                    'daily_pnl_pct': float(daily_pnl_pct),
                    'max_daily_loss_pct': float(self.max_daily_loss_pct * 100)
                }
            ))
        
        # 3. Check portfolio volatility
        # 4. Check correlation risk
        # 5. Check sector/asset class concentration
        
        # Update circuit breakers
        await self._check_circuit_breakers()
        
        # Add violations to history
        self.violations.extend(violations)
        
        return violations
    
    async def _check_circuit_breakers(self) -> None:
        """Check all circuit breakers and trigger actions if needed."""
        for name, cb in self.circuit_breakers.items():
            try:
                # Get the current value for this circuit breaker
                value = await self._get_circuit_breaker_value(name)
                
                # Update the circuit breaker
                if cb.update(value):
                    logger.warning(f"Circuit breaker '{name}' triggered with value {value:.2f}")
                    await cb.action(self)
            except Exception as e:
                logger.error(f"Error in circuit breaker '{name}': {e}", exc_info=True)
    
    async def _get_circuit_breaker_value(self, name: str) -> float:
        """Get the current value for a circuit breaker."""
        if name == "daily_loss_breaker":
            return float(abs(self.daily_pnl) / self.account_balance * 100)
        
        elif name == "drawdown_breaker":
            if not self.historical_balance:
                return 0.0
            peak = max(bal for _, bal in self.historical_balance)
            current = self.historical_balance[-1][1]
            return float((peak - current) / peak * 100)
        
        elif name == "volatility_breaker":
            # Calculate 24h rolling volatility
            prices = await self.market_data.get_historical_prices(
                "PORTFOLIO",  # Or a specific symbol
                interval="1d",
                limit=30  # Last 30 days
            )
            if len(prices) < 2:
                return 0.0
                
            returns = np.diff(np.log([p['close'] for p in prices]))
            return float(np.std(returns) * np.sqrt(252))  # Annualized volatility
        
        elif name == "concentration_breaker":
            # Calculate max position concentration
            positions = await self.position_manager.get_positions()
            if not positions:
                return 0.0
                
            portfolio_value = await self._calculate_portfolio_value()
            if portfolio_value <= 0:
                return 0.0
                
            max_concentration = max(
                (p.quantity * (p.avg_entry_price or Decimal('1'))) / portfolio_value
                for p in positions
            )
            return float(max_concentration * 100)
        
        return 0.0
    
    async def _on_daily_loss_breach(self) -> None:
        """Action to take when daily loss circuit breaker is triggered."""
        # Send alert
        logger.critical("DAILY LOSS CIRCUIT BREAKER: Pausing all trading")
        
        # Close all positions (implementation depends on your trading system)
        await self._liquidate_positions()
        
        # Disable trading
        self.trading_enabled = False
        
        # Schedule re-enable after cooldown
        asyncio.create_task(self._reenable_trading_after(timedelta(hours=1)))
    
    async def _on_drawdown_breach(self) -> None:
        """Action to take when drawdown circuit breaker is triggered."""
        logger.critical("DRAWDOWN CIRCUIT BREAKER: Reducing position sizes")
        
        # Reduce position sizes (implementation depends on your trading system)
        await self._reduce_position_sizes(0.5)  # Reduce by 50%
    
    async def _on_volatility_breach(self) -> None:
        """Action to take when volatility circuit breaker is triggered."""
        logger.warning("VOLATILITY CIRCUIT BREAKER: Increasing margin requirements")
        
        # Increase margin requirements (implementation depends on your trading system)
        self.max_leverage = max(1.0, float(self.max_leverage) * 0.8)  # Reduce max leverage by 20%
    
    async def _on_concentration_breach(self) -> None:
        """Action to take when concentration circuit breaker is triggered."""
        logger.warning("CONCENTRATION CIRCUIT BREAKER: Rebalancing portfolio")
        
        # Rebalance portfolio (implementation depends on your trading system)
        await self._rebalance_portfolio()
    
    async def _liquidate_positions(self) -> None:
        """Liquidate all open positions."""
        positions = await self.position_manager.get_positions()
        for position in positions:
            try:
                # Implementation depends on your trading system
                await self.position_manager.close_position(
                    position.symbol,
                    price=await self._get_current_price(position.symbol)
                )
            except Exception as e:
                logger.error(f"Error liquidating position {position.symbol}: {e}")
    
    async def _reduce_position_sizes(self, reduction_factor: float) -> None:
        """Reduce all position sizes by a factor."""
        positions = await self.position_manager.get_positions()
        for position in positions:
            try:
                # Close a portion of the position
                close_qty = position.quantity * Decimal(str(reduction_factor))
                await self.position_manager.close_position(
                    position.symbol,
                    quantity=close_qty,
                    price=await self._get_current_price(position.symbol)
                )
            except Exception as e:
                logger.error(f"Error reducing position {position.symbol}: {e}")
    
    async def _rebalance_portfolio(self) -> None:
        """Rebalance portfolio to target weights."""
        # Implementation depends on your rebalancing strategy
        pass
    
    async def _reenable_trading_after(self, delay: timedelta) -> None:
        """Re-enable trading after a delay."""
        await asyncio.sleep(delay.total_seconds())
        self.trading_enabled = True
        logger.info("Trading has been re-enabled after circuit breaker cooldown")
    
    async def _calculate_portfolio_value(self) -> Decimal:
        """Calculate total portfolio value (cash + positions)."""
        positions = await self.position_manager.get_positions()
        position_values = []
        
        for position in positions:
            current_price = await self._get_current_price(position.symbol)
            position_value = position.quantity * current_price
            position_values.append(position_value)
        
        return self.account_balance + sum(position_values)
    
    async def _calculate_total_position_value(self) -> Decimal:
        """Calculate total value of all positions."""
        positions = await self.position_manager.get_positions()
        position_values = []
        
        for position in positions:
            current_price = await self._get_current_price(position.symbol)
            position_values.append(position.quantity * current_price)
        
        return sum(position_values)
    
    async def _get_current_price(self, symbol: str) -> Decimal:
        """Get current market price for a symbol."""
        try:
            ticker = await self.market_data.get_ticker(symbol)
            return Decimal(str(ticker.get('last_price', '0')))
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return Decimal('0')
    
    def _add_violation(
        self,
        violation_type: RiskViolationType,
        severity: RiskSeverity,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a new risk violation."""
        violation = RiskViolation(
            violation_type=violation_type,
            severity=severity,
            message=message,
            details=details or {}
        )
        self.violations.append(violation)
        
        # Log the violation
        if severity == RiskSeverity.CRITICAL:
            logger.critical(f"RISK VIOLATION: {message}")
        elif severity == RiskSeverity.ERROR:
            logger.error(f"RISK VIOLATION: {message}")
        elif severity == RiskSeverity.WARNING:
            logger.warning(f"RISK WARNING: {message}")
        else:
            logger.info(f"RISK INFO: {message}")
        
        # Trigger any registered callbacks
        self._on_risk_violation(violation)
    
    def _on_risk_violation(self, violation: RiskViolation) -> None:
        """Handle a new risk violation (can be overridden by subclasses)."""
        # This method can be overridden to implement custom handling
        # of risk violations, such as sending alerts or notifications
        pass
