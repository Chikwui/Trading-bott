"""
Advanced circuit breakers and trading limits monitoring.

This module implements various circuit breakers to protect against:
- Excessive drawdowns
- Volatility spikes
- Position concentration
- Rapid losses
- System failures
- Market regime changes
- Liquidity crunches
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
import pandas as pd
from collections import deque
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BreakerType(Enum):
    """Types of circuit breakers."""
    DRAWDOWN = auto()
    VOLATILITY = auto()
    LOSS = auto()
    POSITION = auto()
    LIQUIDITY = auto()
    NEWS_EVENT = auto()
    SYSTEM = auto()
    CORRELATION = auto()
    LEVERAGE = auto()
    FREQUENCY = auto()


class MarketRegime(Enum):
    """Market regime classifications."""
    NORMAL = auto()
    HIGH_VOLATILITY = auto()
    LOW_VOLATILITY = auto()
    TRENDING_UP = auto()
    TRENDING_DOWN = auto()
    CRASH = auto()
    RALLY = auto()


@dataclass
class BreakerStatus:
    """Status of a circuit breaker."""
    is_triggered: bool = False
    trigger_time: Optional[datetime] = None
    trigger_value: Optional[float] = None
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    cooldown_end: Optional[datetime] = None
    trigger_count: int = 0
    last_reset: Optional[datetime] = None
    adaptive_threshold: Optional[float] = None


class CircuitBreaker:
    """
    Advanced circuit breaker system for trading risk management.
    
    Features:
    - Multiple trigger conditions with adaptive thresholds
    - Market regime awareness
    - Cooldown periods with progressive backoff
    - Performance metrics and analytics
    - Event callbacks and hooks
    - Support for both account-level and strategy-level breakers
    """
    
    def __init__(
        self,
        initial_balance: float = 1_000_000,
        max_drawdown_pct: float = 0.10,  # 10% max drawdown
        max_daily_loss_pct: float = 0.05,  # 5% max daily loss
        max_position_pct: float = 0.25,  # 25% of portfolio in single position
        max_volatility_pct: float = 0.05,  # 5% daily volatility threshold
        cooldown_period: int = 3600,  # 1 hour cooldown in seconds
        max_leverage: float = 5.0,  # Maximum allowed leverage
        max_trade_frequency: int = 100,  # Max trades per hour
        correlation_threshold: float = 0.7,  # Threshold for correlated assets
        regime_aware: bool = True,  # Enable market regime awareness
    ):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        self.max_drawdown_pct = max_drawdown_pct
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_position_pct = max_position_pct
        self.max_volatility_pct = max_volatility_pct
        self.cooldown_period = cooldown_period
        self.max_leverage = max_leverage
        self.max_trade_frequency = max_trade_frequency
        self.correlation_threshold = correlation_threshold
        self.regime_aware = regime_aware
        
        # Track daily metrics
        self.daily_start_balance = initial_balance
        self.last_reset_time = datetime.utcnow()
        
        # Breaker states
        self.breakers: Dict[BreakerType, BreakerStatus] = {
            breaker_type: BreakerStatus() 
            for breaker_type in BreakerType
        }
        
        # Historical data
        self.balance_history: List[Tuple[datetime, float]] = []
        self.trade_history: List[Dict] = []
        self.trade_timestamps: List[datetime] = []
        
        # Market regime tracking
        self.current_regime = MarketRegime.NORMAL
        self.volatility_history = deque(maxlen=100)  # Store recent volatility readings
        self.price_history = deque(maxlen=200)  # Store recent prices for trend analysis
        
        # Callbacks
        self.on_trigger_callbacks: List[Callable] = []
        self.on_reset_callbacks: List[Callable] = []
        
        # Performance metrics
        self.metrics = {
            'total_triggers': 0,
            'last_trigger_time': None,
            'total_cooldown_time': timedelta(),
            'max_drawdown': 0.0,
            'peak_balance': initial_balance,
            'trades_since_reset': 0,
            'drawdown_breaches': 0,
            'volatility_breaches': 0,
            'position_breaches': 0,
        }
    
    def register_callback(self, callback: Callable, event_type: str = 'trigger') -> None:
        """
        Register a callback function to be called on breaker events.
        
        Args:
            callback: Function to call
            event_type: Type of event ('trigger' or 'reset')
        """
        if event_type.lower() == 'trigger':
            self.on_trigger_callbacks.append(callback)
        elif event_type.lower() == 'reset':
            self.on_reset_callbacks.append(callback)
    
    def _notify_callbacks(self, event_type: str, breaker_type: BreakerType, **kwargs) -> None:
        """Notify all registered callbacks of an event."""
        callbacks = self.on_trigger_callbacks if event_type == 'trigger' else self.on_reset_callbacks
        for callback in callbacks:
            try:
                callback(
                    event_type=event_type,
                    breaker_type=breaker_type,
                    timestamp=datetime.utcnow(),
                    **kwargs
                )
            except Exception as e:
                logger.error(f"Error in {event_type} callback: {e}", exc_info=True)
    
    def update_balance(self, new_balance: float, timestamp: Optional[datetime] = None):
        """Update the current balance and check for drawdowns."""
        if timestamp is None:
            timestamp = datetime.utcnow()
            
        self.current_balance = new_balance
        self.peak_balance = max(self.peak_balance, new_balance)
        self.balance_history.append((timestamp, new_balance))
        
        # Update metrics
        self.metrics['peak_balance'] = self.peak_balance
        current_drawdown = self._calculate_drawdown()
        self.metrics['max_drawdown'] = max(self.metrics['max_drawdown'], current_drawdown)
        
        # Check for daily reset
        self._check_daily_reset(timestamp)
        
        # Update market regime
        if self.regime_aware:
            self._update_market_regime()
        
        # Update all breakers
        self._check_breakers(timestamp)
    
    def add_trade(
        self, 
        symbol: str, 
        size: float, 
        price: float, 
        side: str,
        timestamp: Optional[datetime] = None
    ) -> Dict:
        """
        Record a new trade and check position limits.
        
        Returns:
            Dict with trade info and any triggered breakers
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
            
        trade = {
            'timestamp': timestamp,
            'symbol': symbol,
            'size': size,
            'price': price,
            'side': side,
            'notional': abs(size * price),
            'leverage': self._calculate_current_leverage()
        }
        
        self.trade_history.append(trade)
        self.trade_timestamps.append(timestamp)
        self.metrics['trades_since_reset'] += 1
        
        # Update position concentration
        position_violations = self._check_position_concentration(timestamp)
        
        # Check trade frequency
        frequency_violations = self._check_trade_frequency(timestamp)
        
        # Check leverage
        leverage_violations = self._check_leverage(trade['leverage'], timestamp)
        
        # Combine all violations
        all_violations = position_violations + frequency_violations + leverage_violations
        
        return {
            'trade': trade,
            'violations': all_violations,
            'current_leverage': trade['leverage'],
            'current_regime': self.current_regime.name
        }
    
    def check_market_conditions(
        self,
        volatility: float,
        spread_pct: float,
        liquidity: float,
        timestamp: Optional[datetime] = None
    ) -> Tuple[bool, List[Dict]]:
        """
        Check current market conditions against thresholds.
        
        Returns:
            Tuple of (is_allowed, list_of_violations)
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
            
        violations = []
        
        # Store volatility for regime detection
        self.volatility_history.append(volatility)
        
        # Check volatility
        if volatility > self._get_adaptive_threshold('volatility'):
            violations.append({
                'type': BreakerType.VOLATILITY,
                'value': volatility,
                'threshold': self._get_adaptive_threshold('volatility'),
                'message': f"Volatility {volatility:.2%} exceeds threshold {self._get_adaptive_threshold('volatility'):.2%}"
            })
            self.metrics['volatility_breaches'] += 1
            
        # Check spread (adaptive based on market regime)
        max_spread = self._get_adaptive_threshold('spread')
        if spread_pct > max_spread:
            violations.append({
                'type': BreakerType.LIQUIDITY,
                'value': spread_pct,
                'threshold': max_spread,
                'message': f"Spread {spread_pct:.4%} exceeds threshold {max_spread:.4%}"
            })
            
        # Check liquidity (adaptive based on position size)
        min_liquidity = self._get_adaptive_threshold('liquidity')
        if liquidity < min_liquidity:
            violations.append({
                'type': BreakerType.LIQUIDITY,
                'value': liquidity,
                'threshold': min_liquidity,
                'message': f"Liquidity ${liquidity:,.0f} below threshold ${min_liquidity:,.0f}"
            })
        
        # Trigger breakers for any violations
        for violation in violations:
            self._trigger_breaker(
                violation['type'],
                violation['message'],
                timestamp,
                {
                    'value': violation['value'],
                    'threshold': violation['threshold'],
                    'regime': self.current_regime.name
                }
            )
            
        return len(violations) == 0, violations
    
    def is_trading_allowed(self) -> Tuple[bool, List[Dict]]:
        """
        Check if trading is currently allowed.
        
        Returns:
            Tuple of (is_allowed, list_of_violations)
        """
        now = datetime.utcnow()
        violations = []
        
        for breaker_type, status in self.breakers.items():
            if status.is_triggered:
                # Check if cooldown period has passed
                if status.cooldown_end and now >= status.cooldown_end:
                    self._reset_breaker(breaker_type)
                else:
                    time_remaining = (status.cooldown_end - now).total_seconds() if status.cooldown_end else 0
                    violations.append({
                        'type': breaker_type,
                        'message': status.message,
                        'trigger_time': status.trigger_time,
                        'trigger_value': status.trigger_value,
                        'time_remaining': max(0, time_remaining),
                        'cooldown_end': status.cooldown_end
                    })
        
        return len(violations) == 0, violations
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of all breakers and metrics."""
        status = {
            'current_balance': self.current_balance,
            'peak_balance': self.peak_balance,
            'drawdown': self._calculate_drawdown(),
            'daily_pnl': self._calculate_daily_pnl(),
            'current_regime': self.current_regime.name,
            'breakers': {},
            'metrics': self.metrics,
            'leverage': self._calculate_current_leverage(),
            'last_update': datetime.utcnow().isoformat()
        }
        
        # Add breaker statuses
        for bt, bs in self.breakers.items():
            status['breakers'][bt.name] = {
                'is_triggered': bs.is_triggered,
                'trigger_time': bs.trigger_time.isoformat() if bs.trigger_time else None,
                'trigger_value': bs.trigger_value,
                'message': bs.message,
                'cooldown_end': bs.cooldown_end.isoformat() if bs.cooldown_end else None,
                'trigger_count': bs.trigger_count,
                'adaptive_threshold': bs.adaptive_threshold
            }
        
        return status
    
    def reset_all_breakers(self) -> None:
        """Reset all circuit breakers."""
        for breaker_type in self.breakers:
            self._reset_breaker(breaker_type)
    
    def _check_breakers(self, timestamp: datetime):
        """Check all circuit breakers."""
        # Check drawdown
        drawdown = self._calculate_drawdown()
        max_dd = self._get_adaptive_threshold('drawdown')
        if drawdown > max_dd:
            self.metrics['drawdown_breaches'] += 1
            self._trigger_breaker(
                BreakerType.DRAWDOWN,
                f"Drawdown {drawdown:.2%} exceeds maximum {max_dd:.2%} (regime: {self.current_regime.name})",
                timestamp,
                {'drawdown': drawdown, 'threshold': max_dd, 'regime': self.current_regime.name}
            )
        
        # Check daily loss
        daily_pnl = self._calculate_daily_pnl()
        max_daily_loss = self.max_daily_loss_pct * self.daily_start_balance
        if daily_pnl < -max_daily_loss:
            self._trigger_breaker(
                BreakerType.LOSS,
                f"Daily loss {daily_pnl:,.2f} exceeds maximum {max_daily_loss:,.2f}",
                timestamp,
                {'daily_pnl': daily_pnl, 'threshold': -max_daily_loss}
            )
    
    def _check_position_concentration(self, timestamp: datetime) -> List[Dict]:
        """Check if any single position exceeds concentration limits."""
        if not self.trade_history:
            return []
            
        violations = []
            
        # Calculate current positions (simplified)
        positions: Dict[str, float] = {}
        for trade in self.trade_history:
            sign = 1 if trade['side'] == 'buy' else -1
            positions[trade['symbol']] = positions.get(trade['symbol'], 0) + sign * trade['size']
        
        # Check position sizes
        for symbol, size in positions.items():
            position_value = abs(size * self._get_current_price(symbol))  # Would need price feed
            position_pct = position_value / self.current_balance if self.current_balance > 0 else 0
            
            max_position = self._get_adaptive_threshold('position')
            if position_pct > max_position:
                self.metrics['position_breaches'] += 1
                violation = {
                    'type': BreakerType.POSITION,
                    'symbol': symbol,
                    'position_pct': position_pct,
                    'threshold': max_position,
                    'message': f"Position {symbol} is {position_pct:.2%} of portfolio (max {max_position:.2%})"
                }
                violations.append(violation)
                self._trigger_breaker(
                    BreakerType.POSITION,
                    violation['message'],
                    timestamp,
                    {'symbol': symbol, 'position_pct': position_pct, 'threshold': max_position}
                )
                
        return violations
    
    def _check_trade_frequency(self, timestamp: datetime) -> List[Dict]:
        """Check if trading frequency is too high."""
        # Count trades in last hour
        one_hour_ago = timestamp - timedelta(hours=1)
        recent_trades = [t for t in self.trade_timestamps if t > one_hour_ago]
        
        if len(recent_trades) > self.max_trade_frequency:
            message = f"Trade frequency {len(recent_trades)} exceeds maximum {self.max_trade_frequency} trades/hour"
            self._trigger_breaker(
                BreakerType.FREQUENCY,
                message,
                timestamp,
                {'trades': len(recent_trades), 'threshold': self.max_trade_frequency}
            )
            return [{
                'type': BreakerType.FREQUENCY,
                'trades': len(recent_trades),
                'threshold': self.max_trade_frequency,
                'message': message
            }]
        return []
    
    def _check_leverage(self, current_leverage: float, timestamp: datetime) -> List[Dict]:
        """Check if leverage exceeds limits."""
        max_leverage = self._get_adaptive_threshold('leverage')
        if current_leverage > max_leverage:
            message = f"Leverage {current_leverage:.2f}x exceeds maximum {max_leverage:.2f}x"
            self._trigger_breaker(
                BreakerType.LEVERAGE,
                message,
                timestamp,
                {'leverage': current_leverage, 'threshold': max_leverage}
            )
            return [{
                'type': BreakerType.LEVERAGE,
                'leverage': current_leverage,
                'threshold': max_leverage,
                'message': message
            }]
        return []
    
    def _get_adaptive_threshold(self, threshold_type: str) -> float:
        """Get adaptive threshold based on market regime and other factors."""
        base_thresholds = {
            'volatility': self.max_volatility_pct,
            'drawdown': self.max_drawdown_pct,
            'position': self.max_position_pct,
            'spread': 0.001,  # 0.1% default max spread
            'liquidity': 1_000_000,  # $1M default min liquidity
            'leverage': self.max_leverage
        }
        
        if not self.regime_aware:
            return base_thresholds.get(threshold_type, 0.0)
        
        # Adjust thresholds based on market regime
        regime_multipliers = {
            MarketRegime.NORMAL: 1.0,
            MarketRegime.HIGH_VOLATILITY: 0.7,  # Tighter limits in high vol
            MarketRegime.LOW_VOLATILITY: 1.3,   # Looser in low vol
            MarketRegime.TRENDING_UP: 1.1,      # Slightly higher in uptrend
            MarketRegime.TRENDING_DOWN: 0.9,    # Tighter in downtrend
            MarketRegime.CRASH: 0.5,            # Very tight in crashes
            MarketRegime.RALLY: 1.2             # Looser in strong rallies
        }
        
        multiplier = regime_multipliers.get(self.current_regime, 1.0)
        return base_thresholds.get(threshold_type, 0.0) * multiplier
    
    def _update_market_regime(self) -> None:
        """Update the current market regime based on recent price and volatility."""
        if len(self.volatility_history) < 20 or len(self.balance_history) < 20:
            return  # Not enough data
            
        # Calculate recent volatility
        recent_vol = np.mean(list(self.volatility_history)[-20:])
        
        # Calculate price trend (using balance as proxy for portfolio value)
        prices = [b[1] for b in self.balance_history[-50:]]
        if len(prices) < 10:
            return
            
        # Simple trend detection
        returns = np.diff(prices) / prices[:-1]
        avg_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        
        # Regime detection
        if recent_vol > 2 * self.max_volatility_pct:
            self.current_regime = MarketRegime.HIGH_VOLATILITY
        elif recent_vol < 0.5 * self.max_volatility_pct:
            self.current_regime = MarketRegime.LOW_VOLATILITY
        elif avg_return > 2 * std_return and len(returns) > 10 and np.all(returns[-5:] > 0):
            self.current_regime = MarketRegime.TRENDING_UP
        elif avg_return < -2 * std_return and len(returns) > 10 and np.all(returns[-5:] < 0):
            self.current_regime = MarketRegime.TRENDING_DOWN
        elif len(returns) > 5 and np.all(returns[-5:] < -3 * std_return):
            self.current_regime = MarketRegime.CRASH
        elif len(returns) > 5 and np.all(returns[-5:] > 3 * std_return):
            self.current_regime = MarketRegime.RALLY
        else:
            self.current_regime = MarketRegime.NORMAL
    
    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        if self.peak_balance == 0:
            return 0.0
        return max(0, (self.peak_balance - self.current_balance) / self.peak_balance)
    
    def _calculate_daily_pnl(self) -> float:
        """Calculate today's P&L."""
        return self.current_balance - self.daily_start_balance
    
    def _calculate_current_leverage(self) -> float:
        """Calculate current portfolio leverage."""
        if not self.trade_history or self.current_balance <= 0:
            return 0.0
            
        # Sum of absolute notional values of all positions
        total_notional = sum(abs(t['notional']) for t in self.trade_history[-100:])  # Limit lookback
        return total_notional / self.current_balance
    
    def _get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol (placeholder implementation)."""
        # In a real implementation, this would query a price feed
        # For now, return the last trade price for this symbol
        for trade in reversed(self.trade_history):
            if trade['symbol'] == symbol:
                return trade['price']
        return 1.0  # Default if no trades found
    
    def _trigger_breaker(
        self, 
        breaker_type: BreakerType, 
        message: str, 
        timestamp: datetime,
        metadata: Optional[Dict] = None
    ) -> None:
        """Trigger a circuit breaker."""
        status = self.breakers[breaker_type]
        if not status.is_triggered:
            status.is_triggered = True
            status.trigger_time = timestamp
            status.message = message
            status.metadata = metadata or {}
            status.trigger_count += 1
            
            # Progressive cooldown - increase with each trigger
            cooldown_multiplier = min(2 ** (status.trigger_count - 1), 8)  # Cap at 8x
            cooldown = self.cooldown_period * cooldown_multiplier
            status.cooldown_end = timestamp + timedelta(seconds=cooldown)
            
            # Update metrics
            self.metrics['total_triggers'] += 1
            self.metrics['last_trigger_time'] = timestamp
            
            # Log the event
            logger.warning(f"Circuit breaker triggered: {breaker_type.name} - {message}")
            
            # Notify callbacks
            self._notify_callbacks('trigger', breaker_type, **status.metadata)
    
    def _reset_breaker(self, breaker_type: BreakerType) -> None:
        """Reset a circuit breaker."""
        status = self.breakers[breaker_type]
        if status.is_triggered:
            logger.info(f"Resetting circuit breaker: {breaker_type.name}")
            
            # Calculate cooldown time for metrics
            if status.trigger_time and status.cooldown_end:
                cooldown_time = status.cooldown_end - status.trigger_time
                self.metrics['total_cooldown_time'] += cooldown_time
            
            # Reset status
            status.is_triggered = False
            status.last_reset = datetime.utcnow()
            status.cooldown_end = None
            
            # Notify callbacks
            self._notify_callbacks('reset', breaker_type, **status.metadata)
    
    def _check_daily_reset(self, timestamp: datetime):
        """Reset daily metrics if needed."""
        if timestamp.date() > self.last_reset_time.date():
            self.daily_start_balance = self.current_balance
            self.last_reset_time = timestamp
            self.metrics['trades_since_reset'] = 0


class TradingHaltException(Exception):
    """Exception raised when trading is halted by a circuit breaker."""
    def __init__(self, messages: List[Dict]):
        self.messages = messages
        super().__init__("\n".join(["Trading halted due to:"] + [m.get('message', 'Unknown reason') for m in messages]))
