"""
Advanced Risk Management System

This module provides real-time risk management for trading operations, including:
- Pre-trade risk checks
- Position tracking and exposure limits
- Circuit breakers
- Volatility-based position sizing
- Correlation-based risk controls
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Type, TypeVar, Union, Any

from pydantic import BaseModel, Field, validator

from core.trading.order import Order, OrderSide, OrderStatus, OrderType
from core.market.instrument import Instrument
from core.utils.helpers import get_logger
from core.utils.trade_logger import (
    TradeEventType,
    log_risk_check,
    log_position_update,
    log_order_error
)

logger = get_logger(__name__)


class RiskCheckResult(BaseModel):
    """Result of a risk check."""
    passed: bool
    reason: Optional[str] = None
    metadata: Dict = Field(default_factory=dict)


class RiskViolationType(str, Enum):
    """Types of risk violations."""
    POSITION_LIMIT = "POSITION_LIMIT"
    LOSS_LIMIT = "LOSS_LIMIT"
    DRAWDOWN_LIMIT = "DRAWDOWN_LIMIT"
    VOLATILITY_LIMIT = "VOLATILITY_LIMIT"
    LIQUIDITY_LIMIT = "LIQUIDITY_LIMIT"
    CONCENTRATION_LIMIT = "CONCENTRATION_LIMIT"
    LEVERAGE_LIMIT = "LEVERAGE_LIMIT"
    SESSION_LIMIT = "SESSION_LIMIT"
    CIRCUIT_BREAKER = "CIRCUIT_BREAKER"
    MARGIN_LIMIT = "MARGIN_LIMIT"


@dataclass
class Position:
    """Tracks a single position."""
    instrument: Instrument
    quantity: Decimal = Decimal("0")
    avg_price: Optional[Decimal] = None
    unrealized_pnl: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    last_price: Optional[Decimal] = None
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def update_price(self, price: Decimal) -> None:
        """Update position with new market price."""
        if self.quantity == 0:
            return
            
        self.last_price = price
        self.last_updated = datetime.utcnow()
        
        if self.avg_price is not None:
            price_diff = price - self.avg_price
            if self.quantity < 0:  # Short position
                price_diff = -price_diff
            self.unrealized_pnl = price_diff * abs(self.quantity)
    
    def add_trade(self, quantity: Decimal, price: Decimal) -> Decimal:
        """Add a trade to the position."""
        if quantity == 0:
            return Decimal("0")
            
        realized_pnl = Decimal("0")
        
        # If position is being reduced or reversed
        if (quantity * self.quantity) < 0:
            # Calculate how much of the position is being closed
            close_qty = min(abs(quantity), abs(self.quantity)) * (-1 if quantity < 0 else 1)
            
            # Calculate realized P&L for the closed portion
            if self.avg_price is not None:
                price_diff = price - self.avg_price
                if self.quantity < 0:  # Short position
                    price_diff = -price_diff
                realized_pnl = price_diff * abs(close_qty)
                self.realized_pnl += realized_pnl
            
            # Update position
            self.quantity += close_qty
            
            # If position is completely closed
            if self.quantity == 0:
                self.avg_price = None
            
            # If there's remaining quantity in the opposite direction
            remaining_qty = quantity - close_qty
            if remaining_qty != 0:
                self.quantity += remaining_qty
                self.avg_price = price
        else:
            # Calculate new average price if adding to position
            if self.quantity != 0:
                total_cost = (self.avg_price or Decimal("0")) * abs(self.quantity) + price * abs(quantity)
                self.avg_price = total_cost / (abs(self.quantity) + abs(quantity))
            else:
                self.avg_price = price
            
            self.quantity += quantity
        
        self.last_price = price
        self.last_updated = datetime.utcnow()
        
        return realized_pnl


class RiskManager:
    """Manages risk for trading operations with comprehensive logging."""
    
    def __init__(
        self,
        account_balance: Decimal = Decimal("100000"),  # Default $100,000
        max_position_size_pct: float = 0.05,  # 5% of account per position
        max_daily_loss_pct: float = 0.02,  # 2% max daily loss
        max_drawdown_pct: float = 0.1,  # 10% max drawdown
        max_leverage: float = 10.0,  # 10x leverage max
        volatility_window: int = 20,  # 20 periods for volatility calculation
        circuit_breaker_pct: float = 0.05,  # 5% move triggers circuit breaker
    ):
        self.account_balance = account_balance
        self.initial_balance = account_balance
        self.equity = account_balance
        self.max_position_size_pct = Decimal(str(max_position_size_pct))
        self.max_daily_loss_pct = Decimal(str(max_daily_loss_pct))
        self.max_drawdown_pct = Decimal(str(max_drawdown_pct))
        self.max_leverage = Decimal(str(max_leverage))
        self.volatility_window = volatility_window
        self.circuit_breaker_pct = Decimal(str(circuit_breaker_pct))
        
        # State tracking
        self.positions: Dict[str, Position] = {}
        self.open_orders: Dict[str, Order] = {}
        self.daily_pnl: Decimal = Decimal("0")
        self.max_daily_drawdown: Decimal = Decimal("0")
        self.daily_high_watermark: Decimal = account_balance
        self.daily_low_watermark: Decimal = account_balance
        self.circuit_breaker_triggered: bool = False
        self.circuit_breaker_time: Optional[datetime] = None
        
        # Volatility tracking
        self.price_history: Dict[str, List[Decimal]] = {}
        self.volatility: Dict[str, Decimal] = {}
        
        # Locks for thread safety
        self._lock = asyncio.Lock()
        
        # Log risk manager initialization
        log_risk_check(
            check_name="RISK_MANAGER_INIT",
            passed=True,
            message=f"Risk Manager initialized with balance: {account_balance}, max position: {max_position_size_pct*100}%, "
                   f"max daily loss: {max_daily_loss_pct*100}%, max drawdown: {max_drawdown_pct*100}%"
        )

    async def check_order(self, order: Order) -> RiskCheckResult:
        """
        Check if an order passes all risk checks with detailed logging.
        
        Args:
            order: The order to check
            
        Returns:
            RiskCheckResult indicating if the order passes risk checks
        """
        async with self._lock:
            # Check circuit breaker first
            if self.circuit_breaker_triggered:
                msg = f"Circuit breaker active since {self.circuit_breaker_time}"
                log_risk_check(
                    check_name="CIRCUIT_BREAKER",
                    passed=False,
                    message=msg,
                    symbol=order.symbol,
                    metadata={
                        'order_id': order.order_id,
                        'side': order.side.name,
                        'quantity': float(order.quantity),
                        'price': float(order.price) if order.price else None,
                        'order_type': order.order_type.name
                    }
                )
                return RiskCheckResult(
                    passed=False,
                    reason="Circuit breaker active",
                    metadata={"circuit_breaker_time": str(self.circuit_breaker_time)}
                )
            
            # Initialize result
            result = RiskCheckResult(passed=True)
            
            # Check position size limit
            max_position_value = self.account_balance * self.max_position_size_pct
            position_value = Decimal(order.quantity) * (order.price or Decimal("1"))
            
            if position_value > max_position_value:
                msg = f"Position size {position_value} exceeds max {max_position_value}"
                log_risk_check(
                    check_name="POSITION_SIZE_LIMIT",
                    passed=False,
                    message=msg,
                    symbol=order.symbol,
                    metadata={
                        'order_id': order.order_id,
                        'position_value': float(position_value),
                        'max_allowed': float(max_position_value),
                        'percentage': float(self.max_position_size_pct * 100)
                    }
                )
                return RiskCheckResult(
                    passed=False,
                    reason=msg,
                    metadata={
                        "position_value": float(position_value),
                        "max_allowed": float(max_position_value)
                    }
                )
            
            # Check daily loss limit
            daily_loss_pct = (self.daily_high_watermark - self.equity) / self.daily_high_watermark
            if daily_loss_pct > self.max_daily_loss_pct:
                msg = f"Daily loss {daily_loss_pct*100:.2f}% exceeds max {self.max_daily_loss_pct*100}%"
                log_risk_check(
                    check_name="DAILY_LOSS_LIMIT",
                    passed=False,
                    message=msg,
                    symbol=order.symbol,
                    metadata={
                        'order_id': order.order_id,
                        'daily_loss_pct': float(daily_loss_pct * 100),
                        'max_daily_loss_pct': float(self.max_daily_loss_pct * 100),
                        'equity': float(self.equity),
                        'high_watermark': float(self.daily_high_watermark)
                    }
                )
                return RiskCheckResult(
                    passed=False,
                    reason=msg,
                    metadata={
                        "daily_loss_pct": float(daily_loss_pct),
                        "max_daily_loss_pct": float(self.max_daily_loss_pct)
                    }
                )
            
            # If we get here, all checks passed
            log_risk_check(
                check_name="ORDER_RISK_CHECKS_PASSED",
                passed=True,
                message=f"Order {order.order_id} passed all risk checks",
                symbol=order.symbol,
                metadata={
                    'order_id': order.order_id,
                    'side': order.side.name,
                    'quantity': float(order.quantity),
                    'price': float(order.price) if order.price else None,
                    'order_type': order.order_type.name
                }
            )
            return result

    async def on_order_fill(self, order: Order, fill_quantity: Decimal, fill_price: Decimal) -> None:
        """
        Update risk state when an order is filled with detailed logging.
        
        Args:
            order: The filled order
            fill_quantity: The filled quantity
            fill_price: The fill price
        """
        async with self._lock:
            # Update position
            position_id = f"{order.symbol}_{order.side.name}"
            if position_id not in self.positions:
                self.positions[position_id] = Position(
                    symbol=order.symbol,
                    side=order.side,
                    quantity=Decimal("0"),
                    entry_price=Decimal("0"),
                    position_id=position_id
                )
            
            position = self.positions[position_id]
            old_quantity = position.quantity
            
            # Update position with fill
            position.quantity += fill_quantity
            position.entry_price = (
                (position.entry_price * old_quantity + fill_price * fill_quantity) / 
                (old_quantity + fill_quantity)
            )
            
            # Update equity and P&L
            fill_value = fill_quantity * fill_price
            if order.side == OrderSide.BUY:
                self.equity -= fill_value
            else:  # SELL
                self.equity += fill_value
            
            # Update daily high/low watermarks
            self.daily_high_watermark = max(self.daily_high_watermark, self.equity)
            self.daily_low_watermark = min(self.daily_low_watermark, self.equity)
            
            # Log position update
            log_position_update(
                position_id=position_id,
                symbol=order.symbol,
                side=order.side.name,
                size=float(position.quantity),
                entry_price=float(position.entry_price),
                unrealized_pnl=float(self.equity - self.account_balance),
                event_type=TradeEventType.POSITION_MODIFIED,
                metadata={
                    'order_id': order.order_id,
                    'fill_quantity': float(fill_quantity),
                    'fill_price': float(fill_price),
                    'new_equity': float(self.equity)
                }
            )
            
            # Check for circuit breaker
            self._check_circuit_breaker()

    def _check_circuit_breaker(self) -> None:
        """Check if circuit breaker conditions are met and trigger if necessary."""
        if self.circuit_breaker_triggered:
            return
            
        # Calculate drawdown from high watermark
        if self.daily_high_watermark > 0:
            drawdown = (self.daily_high_watermark - self.equity) / self.daily_high_watermark
            
            if drawdown >= self.circuit_breaker_pct:
                self.circuit_breaker_triggered = True
                self.circuit_breaker_time = datetime.utcnow()
                
                # Log circuit breaker trigger
                log_risk_check(
                    check_name="CIRCUIT_BREAKER_TRIGGERED",
                    passed=False,
                    message=f"Circuit breaker triggered: {drawdown*100:.2f}% drawdown exceeds {self.circuit_breaker_pct*100}% limit",
                    metadata={
                        'drawdown_pct': float(drawdown * 100),
                        'threshold_pct': float(self.circuit_breaker_pct * 100),
                        'equity': float(self.equity),
                        'high_watermark': float(self.daily_high_watermark)
                    }
                )
                
                # Log all open positions
                for pos_id, position in self.positions.items():
                    if position.quantity > 0:
                        log_position_update(
                            position_id=pos_id,
                            symbol=position.symbol,
                            side=position.side.name,
                            size=float(position.quantity),
                            entry_price=float(position.entry_price),
                            event_type=TradeEventType.RISK_LIMIT_BREACHED,
                            metadata={
                                'reason': 'circuit_breaker_triggered',
                                'drawdown_pct': float(drawdown * 100)
                            }
                        )

    async def reset_daily(self) -> None:
        """Reset daily metrics at the start of a new trading day with logging."""
        async with self._lock:
            # Log end of day summary
            daily_pnl = self.equity - self.account_balance
            log_risk_check(
                check_name="DAILY_RESET",
                passed=True,
                message=f"Resetting daily metrics. P&L: {daily_pnl:.2f}",
                metadata={
                    'start_balance': float(self.account_balance),
                    'end_balance': float(self.equity),
                    'daily_pnl': float(daily_pnl),
                    'high_watermark': float(self.daily_high_watermark),
                    'low_watermark': float(self.daily_low_watermark)
                }
            )
            
            # Reset metrics
            self.account_balance = self.equity
            self.daily_pnl = Decimal("0")
            self.daily_high_watermark = self.equity
            self.daily_low_watermark = self.equity
            self.max_daily_drawdown = Decimal("0")
            
            # Reset circuit breaker
            if self.circuit_breaker_triggered:
                log_risk_check(
                    check_name="CIRCUIT_BREAKER_RESET",
                    passed=True,
                    message=f"Resetting circuit breaker after {datetime.utcnow() - self.circuit_breaker_time}",
                    metadata={
                        'trigger_time': self.circuit_breaker_time.isoformat(),
                        'reset_time': datetime.utcnow().isoformat()
                    }
                )
                self.circuit_breaker_triggered = False
                self.circuit_breaker_time = None

    async def reset_circuit_breaker(self) -> None:
        """Manually reset the circuit breaker with logging."""
        async with self._lock:
            if self.circuit_breaker_triggered:
                log_risk_check(
                    check_name="CIRCUIT_BREAKER_MANUAL_RESET",
                    passed=True,
                    message=f"Manually resetting circuit breaker after {datetime.utcnow() - self.circuit_breaker_time}",
                    metadata={
                        'trigger_time': self.circuit_breaker_time.isoformat(),
                        'reset_time': datetime.utcnow().isoformat()
                    }
                )
                self.circuit_breaker_triggered = False
                self.circuit_breaker_time = None
            else:
                log_risk_check(
                    check_name="CIRCUIT_BREAKER_RESET_ATTEMPT",
                    passed=False,
                    message="Attempted to reset circuit breaker when it was not triggered",
                    metadata={
                        'reset_time': datetime.utcnow().isoformat()
                    }
                )

    async def update_market_data(self, symbol: str, price: Decimal) -> None:
        """Update market data and recalculate risk metrics."""
        async with self._lock:
            # Update price history for volatility calculation
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            
            self.price_history[symbol].append(price)
            
            # Keep only the last N prices for volatility calculation
            if len(self.price_history[symbol]) > self.volatility_window:
                self.price_history[symbol].pop(0)
            
            # Calculate volatility if we have enough data
            if len(self.price_history[symbol]) >= 2:
                returns = []
                prices = self.price_history[symbol]
                
                for i in range(1, len(prices)):
                    if prices[i-1] > 0:
                        returns.append((prices[i] - prices[i-1]) / prices[i-1])
                
                if returns:
                    # Simple standard deviation of returns as volatility proxy
                    mean_return = sum(returns) / len(returns)
                    variance = sum((r - mean_return) ** 2 for r in returns) / len(returns)
                    self.volatility[symbol] = Decimal(str(variance ** 0.5))
            
            # Update position P&L
            if symbol in self.positions:
                position = self.positions[symbol]
                position.update_price(price)
                
                # Update equity
                self.equity = self.account_balance + sum(
                    pos.unrealized_pnl + pos.realized_pnl 
                    for pos in self.positions.values()
                )

    def get_risk_metrics(self) -> Dict:
        """Get current risk metrics."""
        return {
            "equity": float(self.equity),
            "daily_pnl": float(self.daily_pnl),
            "daily_high_watermark": float(self.daily_high_watermark),
            "daily_low_watermark": float(self.daily_low_watermark),
            "max_daily_drawdown": float(self.max_daily_drawdown),
            "circuit_breaker_triggered": self.circuit_breaker_triggered,
            "circuit_breaker_time": self.circuit_breaker_time.isoformat() if self.circuit_breaker_time else None,
            "positions": {
                symbol: {
                    "quantity": float(pos.quantity),
                    "avg_price": float(pos.avg_price) if pos.avg_price else None,
                    "unrealized_pnl": float(pos.unrealized_pnl),
                    "realized_pnl": float(pos.realized_pnl),
                    "last_price": float(pos.last_price) if pos.last_price else None
                }
                for symbol, pos in self.positions.items()
            },
            "volatility": {
                symbol: float(vol) 
                for symbol, vol in self.volatility.items()
            }
        }
