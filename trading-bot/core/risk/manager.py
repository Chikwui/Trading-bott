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
from typing import Dict, List, Optional, Set, Tuple, Type, TypeVar, Union

from pydantic import BaseModel, Field, validator

from core.trading.order import Order, OrderSide, OrderStatus, OrderType
from core.market.instrument import Instrument
from core.utils.helpers import get_logger

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
    """Manages risk for trading operations."""
    
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
    
    async def check_order(self, order: Order) -> RiskCheckResult:
        """Check if an order passes all risk checks."""
        async with self._lock:
            # Check if circuit breaker is active
            if self.circuit_breaker_triggered:
                return RiskCheckResult(
                    passed=False,
                    reason="Circuit breaker triggered",
                    metadata={"violation": RiskViolationType.CIRCUIT_BREAKER}
                )
            
            # Basic order validation
            if order.quantity <= 0:
                return RiskCheckResult(
                    passed=False,
                    reason="Invalid order quantity",
                    metadata={"quantity": str(order.quantity)}
                )
            
            # Calculate position impact
            position_impact = order.quantity
            if order.side == OrderSide.SELL:
                position_impact = -position_impact
            
            # Get current position
            current_position = self.positions.get(order.symbol, Position(order.instrument))
            new_position_size = (current_position.quantity if current_position else Decimal("0")) + position_impact
            
            # 1. Position size limit check
            max_position_size = self.account_balance * self.max_position_size_pct
            if abs(new_position_size) > max_position_size:
                return RiskCheckResult(
                    passed=False,
                    reason=f"Position size {abs(new_position_size)} exceeds maximum {max_position_size}",
                    metadata={
                        "violation": RiskViolationType.POSITION_LIMIT,
                        "current_size": str(current_position.quantity if current_position else 0),
                        "new_size": str(new_position_size),
                        "max_allowed": str(max_position_size)
                    }
                )
            
            # 2. Leverage check
            total_exposure = sum(
                (pos.avg_price or Decimal("0")) * abs(pos.quantity) 
                for pos in self.positions.values()
            )
            
            # Add the new order's exposure
            order_exposure = (order.limit_price or order.stop_price or Decimal("0")) * order.quantity
            if order.side == OrderSide.BUY:
                total_exposure += order_exposure
            
            leverage = total_exposure / self.equity if self.equity > 0 else Decimal("0")
            if leverage > self.max_leverage:
                return RiskCheckResult(
                    passed=False,
                    reason=f"Leverage {leverage:.2f}x exceeds maximum {self.max_leverage}x",
                    metadata={
                        "violation": RiskViolationType.LEVERAGE_LIMIT,
                        "current_leverage": str(leverage),
                        "max_leverage": str(self.max_leverage)
                    }
                )
            
            # 3. Daily loss limit check
            daily_loss_limit = self.initial_balance * self.max_daily_loss_pct
            if self.daily_pnl < -daily_loss_limit:
                return RiskCheckResult(
                    passed=False,
                    reason=f"Daily PnL {self.daily_pnl} below limit {-daily_loss_limit}",
                    metadata={
                        "violation": RiskViolationType.LOSS_LIMIT,
                        "daily_pnl": str(self.daily_pnl),
                        "daily_limit": str(-daily_loss_limit)
                    }
                )
            
            # 4. Drawdown check
            drawdown = (self.equity - self.daily_high_watermark) / self.daily_high_watermark
            if drawdown < -self.max_drawdown_pct:
                return RiskCheckResult(
                    passed=False,
                    reason=f"Drawdown {drawdown*100:.2f}% exceeds maximum {self.max_drawdown_pct*100:.2f}%",
                    metadata={
                        "violation": RiskViolationType.DRAWDOWN_LIMIT,
                        "current_drawdown": str(drawdown),
                        "max_drawdown": str(-self.max_drawdown_pct)
                    }
                )
            
            # 5. Volatility check (if we have enough history)
            if order.symbol in self.volatility:
                vol = self.volatility[order.symbol]
                if vol > Decimal("0.05"):  # 5% volatility threshold
                    # Reduce position size based on volatility
                    vol_adj_size = max_position_size * (Decimal("0.05") / vol)
                    if abs(new_position_size) > vol_adj_size:
                        return RiskCheckResult(
                            passed=False,
                            reason=f"Position size {abs(new_position_size)} exceeds volatility-adjusted limit {vol_adj_size:.2f}",
                            metadata={
                                "violation": RiskViolationType.VOLATILITY_LIMIT,
                                "volatility": str(vol),
                                "vol_adj_size": str(vol_adj_size)
                            }
                        )
            
            # All checks passed
            return RiskCheckResult(passed=True)
    
    async def on_order_fill(self, order: Order, fill_quantity: Decimal, fill_price: Decimal) -> None:
        """Update risk state when an order is filled."""
        async with self._lock:
            # Update position
            if order.symbol not in self.positions:
                self.positions[order.symbol] = Position(order.instrument)
            
            position = self.positions[order.symbol]
            realized_pnl = position.add_trade(
                fill_quantity if order.side == OrderSide.BUY else -fill_quantity,
                fill_price
            )
            
            # Update P&L and equity
            self.daily_pnl += realized_pnl
            self.equity = self.account_balance + sum(
                pos.unrealized_pnl + pos.realized_pnl 
                for pos in self.positions.values()
            )
            
            # Update watermarks
            self.daily_high_watermark = max(self.daily_high_watermark, self.equity)
            self.daily_low_watermark = min(self.daily_low_watermark, self.equity)
            
            # Update max drawdown
            current_drawdown = (self.equity - self.daily_high_watermark) / self.daily_high_watermark
            self.max_daily_drawdown = min(self.max_daily_drawdown, current_drawdown)
            
            # Check for circuit breaker
            if abs(current_drawdown) >= self.circuit_breaker_pct:
                self.circuit_breaker_triggered = True
                self.circuit_breaker_time = datetime.utcnow()
                logger.warning(
                    f"Circuit breaker triggered: drawdown {current_drawdown*100:.2f}% "
                    f"exceeds {self.circuit_breaker_pct*100:.2f}%"
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
    
    async def reset_daily(self) -> None:
        """Reset daily metrics at the start of a new trading day."""
        async with self._lock:
            self.initial_balance = self.equity
            self.daily_pnl = Decimal("0")
            self.daily_high_watermark = self.equity
            self.daily_low_watermark = self.equity
            self.max_daily_drawdown = Decimal("0")
            self.circuit_breaker_triggered = False
            self.circuit_breaker_time = None
    
    async def reset_circuit_breaker(self) -> None:
        """Manually reset the circuit breaker."""
        async with self._lock:
            self.circuit_breaker_triggered = False
            self.circuit_breaker_time = None
            logger.info("Circuit breaker reset")
    
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
