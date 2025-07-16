"""
Risk management with position sizing, drawdown limits, and circuit breakers.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
import numpy as np

from core.instruments import InstrumentMetadata
from core.calendar import MarketCalendar

logger = logging.getLogger(__name__)


@dataclass
class RiskParameters:
    """Risk management parameters for a trading strategy."""
    # Position sizing
    max_position_size_pct: float = 5.0  # Max % of portfolio per trade
    max_portfolio_risk_pct: float = 20.0  # Max % of portfolio at risk
    max_leverage: float = 3.0  # Maximum allowed leverage
    
    # Drawdown limits
    max_daily_drawdown_pct: float = 5.0  # Max daily drawdown %
    max_drawdown_pct: float = 15.0  # Max total drawdown %
    
    # Position limits
    max_positions: int = 10  # Max number of concurrent positions
    max_position_concentration_pct: float = 30.0  # Max % in a single position
    
    # Volatility limits
    max_volatility_pct: float = 5.0  # Max allowed volatility (ATR/price %)
    
    # Trading hours
    trade_only_during_market_hours: bool = True
    
    # Circuit breakers
    circuit_breaker_drawdown_pct: float = 10.0  # Drawdown % to trigger circuit breaker
    circuit_breaker_cooldown_hours: int = 24  # Hours to wait after circuit breaker
    
    # Risk per trade
    risk_per_trade_pct: float = 1.0  # % of portfolio to risk per trade
    
    # Slippage and spread
    max_slippage_pct: float = 0.1  # Max allowed slippage %
    max_spread_pct: float = 0.1  # Max allowed spread %
    
    # Session-based limits
    max_trades_per_day: int = 20  # Max number of trades per day
    max_overnight_positions: int = 5  # Max number of positions to hold overnight
    
    def validate(self) -> bool:
        """Validate the risk parameters."""
        if self.max_position_size_pct <= 0 or self.max_position_size_pct > 100:
            logger.error("max_position_size_pct must be between 0 and 100")
            return False
            
        if self.max_portfolio_risk_pct <= 0 or self.max_portfolio_risk_pct > 100:
            logger.error("max_portfolio_risk_pct must be between 0 and 100")
            return False
            
        if self.max_leverage <= 0:
            logger.error("max_leverage must be positive")
            return False
            
        if self.max_daily_drawdown_pct <= 0 or self.max_daily_drawdown_pct > 100:
            logger.error("max_daily_drawdown_pct must be between 0 and 100")
            return False
            
        if self.max_drawdown_pct <= 0 or self.max_drawdown_pct > 100:
            logger.error("max_drawdown_pct must be between 0 and 100")
            return False
            
        if self.max_positions <= 0:
            logger.error("max_positions must be positive")
            return False
            
        if self.max_position_concentration_pct <= 0 or self.max_position_concentration_pct > 100:
            logger.error("max_position_concentration_pct must be between 0 and 100")
            return False
            
        if self.risk_per_trade_pct <= 0 or self.risk_per_trade_pct > 100:
            logger.error("risk_per_trade_pct must be between 0 and 100")
            return False
            
        if self.max_slippage_pct < 0:
            logger.error("max_slippage_pct must be non-negative")
            return False
            
        if self.max_spread_pct < 0:
            logger.error("max_spread_pct must be non-negative")
            return False
            
        if self.max_trades_per_day <= 0:
            logger.error("max_trades_per_day must be positive")
            return False
            
        if self.max_overnight_positions < 0:
            logger.error("max_overnight_positions must be non-negative")
            return False
            
        return True


class RiskManager:
    """Manages trading risk with position sizing, drawdown limits, and circuit breakers."""
    
    def __init__(
        self,
        parameters: Optional[RiskParameters] = None,
        calendar: Optional[MarketCalendar] = None,
        timezone: str = "UTC"
    ):
        """Initialize the risk manager.
        
        Args:
            parameters: Risk management parameters
            calendar: Market calendar for time-based checks
            timezone: Timezone for the risk manager
        """
        self.parameters = parameters or RiskParameters()
        self.calendar = calendar
        self.timezone = timezone
        
        # State tracking
        self.equity_high_water_mark: float = 0.0
        self.daily_high_water_mark: float = 0.0
        self.daily_trade_count: int = 0
        self.last_trade_day: Optional[datetime] = None
        self.circuit_breaker_triggered: bool = False
        self.circuit_breaker_time: Optional[datetime] = None
        
        # Volatility tracking
        self.volatility: Dict[str, float] = {}  # symbol -> ATR/price %
        
        # Position tracking
        self.positions: Dict[str, Dict] = {}  # symbol -> position info
        
        logger.info("Risk manager initialized")
    
    def check_order_risk(self, order: Dict[str, Any], portfolio_value: float) -> Tuple[bool, str]:
        """Check if an order meets risk management criteria.
        
        Args:
            order: Order details with keys: symbol, quantity, price, side, etc.
            portfolio_value: Current total portfolio value
            
        Returns:
            Tuple of (is_allowed, reason)
        """
        symbol = order.get('symbol')
        quantity = abs(order.get('quantity', 0))
        price = order.get('price', 0.0)
        side = order.get('side', '').lower()
        
        if not all([symbol, quantity > 0, price > 0, side in ['buy', 'sell']]):
            return False, "Invalid order parameters"
        
        # Check circuit breaker
        if self.circuit_breaker_triggered:
            if self.circuit_breaker_time:
                cooldown = self.parameters.circuit_breaker_cooldown_hours
                if (datetime.now(timezone.utc) - self.circuit_breaker_time).total_seconds() < cooldown * 3600:
                    return False, f"Circuit breaker active. Cooldown for {cooldown} hours."
                else:
                    self.circuit_breaker_triggered = False
                    self.circuit_breaker_time = None
            else:
                return False, "Circuit breaker active"
        
        # Check position size vs max position size
        position_value = quantity * price
        position_pct = (position_value / portfolio_value) * 100
        
        if position_pct > self.parameters.max_position_size_pct:
            return False, f"Position size {position_pct:.2f}% exceeds max {self.parameters.max_position_size_pct}%"
        
        # Check position count
        if len(self.positions) >= self.parameters.max_positions and symbol not in self.positions:
            return False, f"Max positions ({self.parameters.max_positions}) reached"
        
        # Check position concentration
        current_position = self.positions.get(symbol, {'value': 0.0})
        new_position_value = current_position['value'] + (quantity * price * (1 if side == 'buy' else -1))
        new_position_pct = (abs(new_position_value) / portfolio_value) * 100
        
        if new_position_pct > self.parameters.max_position_concentration_pct:
            return False, (
                f"Position concentration {new_position_pct:.2f}% exceeds max "
                f"{self.parameters.max_position_concentration_pct}%"
            )
        
        # Check daily trade count
        self._update_daily_tracking()
        if self.daily_trade_count >= self.parameters.max_trades_per_day:
            return False, f"Max trades per day ({self.parameters.max_trades_per_day}) reached"
        
        # Check volatility
        if symbol in self.volatility and self.volatility[symbol] > self.parameters.max_volatility_pct:
            return False, f"Volatility {self.volatility[symbol]:.2f}% exceeds max {self.parameters.max_volatility_pct}%"
        
        # Check leverage
        total_position_value = sum(abs(p['value']) for p in self.positions.values())
        new_total_position_value = total_position_value + position_value
        leverage = new_total_position_value / portfolio_value
        
        if leverage > self.parameters.max_leverage:
            return False, f"Leverage {leverage:.2f}x exceeds max {self.parameters.max_leverage}x"
        
        # If all checks pass, update tracking
        self.daily_trade_count += 1
        
        return True, ""
    
    def update_drawdown(self, equity: float) -> Tuple[bool, str]:
        """Update drawdown tracking and check for circuit breakers.
        
        Args:
            equity: Current portfolio equity
            
        Returns:
            Tuple of (is_ok, message)
        """
        # Update high water marks
        self.equity_high_water_mark = max(self.equity_high_water_mark, equity)
        self.daily_high_water_mark = max(self.daily_high_water_mark, equity)
        
        # Calculate drawdowns
        total_drawdown_pct = ((self.equity_high_water_mark - equity) / self.equity_high_water_mark) * 100
        daily_drawdown_pct = ((self.daily_high_water_mark - equity) / self.daily_high_water_mark) * 100
        
        # Check circuit breaker
        if total_drawdown_pct >= self.parameters.circuit_breaker_drawdown_pct and not self.circuit_breaker_triggered:
            self.circuit_breaker_triggered = True
            self.circuit_breaker_time = datetime.now(timezone.utc)
            return False, (
                f"Circuit breaker triggered: {total_drawdown_pct:.2f}% drawdown exceeds "
                f"{self.parameters.circuit_breaker_drawdown_pct}%"
            )
        
        # Check daily drawdown
        if daily_drawdown_pct > self.parameters.max_daily_drawdown_pct:
            return False, (
                f"Daily drawdown {daily_drawdown_pct:.2f}% exceeds "
                f"{self.parameters.max_daily_drawdown_pct}%"
            )
        
        # Check total drawdown
        if total_drawdown_pct > self.parameters.max_drawdown_pct:
            return False, (
                f"Total drawdown {total_drawdown_pct:.2f}% exceeds "
                f"{self.parameters.max_drawdown_pct}%"
            )
        
        return True, ""
    
    def update_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp: Optional[datetime] = None
    ) -> None:
        """Update position tracking."""
        if symbol not in self.positions:
            self.positions[symbol] = {
                'quantity': 0.0,
                'value': 0.0,
                'entry_price': price,
                'last_update': timestamp or datetime.now(timezone.utc)
            }
        
        position = self.positions[symbol]
        position['quantity'] += quantity
        position['value'] = position['quantity'] * price
        position['last_update'] = timestamp or datetime.now(timezone.utc)
        
        # Remove if position is closed
        if abs(position['quantity']) < 1e-10:  # Account for floating point precision
            del self.positions[symbol]
    
    def update_volatility(self, symbol: str, atr: float, price: float) -> None:
        """Update volatility tracking for a symbol."""
        if price > 0:
            self.volatility[symbol] = (atr / price) * 100  # ATR as % of price
    
    def _update_daily_tracking(self) -> None:
        """Update daily tracking metrics."""
        now = datetime.now(timezone.utc)
        
        # Reset daily metrics if it's a new trading day
        if self.last_trade_day is None or now.date() > self.last_trade_day.date():
            self.daily_trade_count = 0
            self.daily_high_water_mark = 0.0
            self.last_trade_day = now
    
    def reset_daily_metrics(self) -> None:
        """Reset daily metrics (call at the start of each trading day)."""
        self.daily_trade_count = 0
        self.daily_high_water_mark = 0.0
        self.last_trade_day = datetime.now(timezone.utc)
    
    def reset_circuit_breaker(self) -> None:
        """Manually reset the circuit breaker."""
        self.circuit_breaker_triggered = False
        self.circuit_breaker_time = None
