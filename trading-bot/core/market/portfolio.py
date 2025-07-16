"""
Portfolio management with position tracking and performance metrics.
"""
from __future__ import annotations

import logging
import datetime
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
import pandas as pd
import numpy as np

from core.instruments import InstrumentMetadata
from core.calendar import MarketCalendar

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a trading position."""
    symbol: str
    quantity: float = 0.0
    avg_price: float = 0.0
    current_price: float = 0.0
    timestamp: datetime.datetime = field(default_factory=lambda: datetime.datetime.now(datetime.timezone.utc))
    
    @property
    def market_value(self) -> float:
        """Current market value of the position."""
        return self.quantity * self.current_price
    
    @property
    def cost_basis(self) -> float:
        """Total cost basis of the position."""
        return abs(self.quantity) * self.avg_price
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss."""
        if self.quantity == 0:
            return 0.0
        return (self.current_price - self.avg_price) * self.quantity
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized profit/loss as a percentage of cost basis."""
        if self.cost_basis == 0:
            return 0.0
        return (self.unrealized_pnl / self.cost_basis) * 100.0
    
    def update_price(self, price: float, timestamp: Optional[datetime] = None) -> None:
        """Update the current market price."""
        self.current_price = price
        if timestamp:
            self.timestamp = timestamp
        else:
            self.timestamp = datetime.datetime.now(datetime.timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'avg_price': self.avg_price,
            'current_price': self.current_price,
            'market_value': self.market_value,
            'cost_basis': self.cost_basis,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_pct': self.unrealized_pnl_pct,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class Trade:
    """Represents a completed trade."""
    symbol: str
    quantity: float
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    side: str  # 'long' or 'short'
    pnl: float
    pnl_pct: float
    commission: float = 0.0
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> float:
        """Trade duration in seconds."""
        return (self.exit_time - self.entry_time).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'entry_time': self.entry_time.isoformat(),
            'exit_time': self.exit_time.isoformat(),
            'duration_seconds': self.duration,
            'side': self.side,
            'pnl': self.pnl,
            'pnl_pct': self.pnl_pct,
            'commission': self.commission,
            'tags': list(self.tags),
            'metadata': self.metadata
        }


class Portfolio:
    """Manages a trading portfolio with position tracking and performance metrics."""
    
    def __init__(
        self,
        initial_cash: float = 100000.0,
        currency: str = "USD",
        calendar: Optional[MarketCalendar] = None,
        timezone: str = "UTC"
    ):
        """Initialize the portfolio.
        
        Args:
            initial_cash: Initial cash balance
            currency: Account currency
            calendar: Market calendar for time-based calculations
            timezone: Timezone for the portfolio
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.currency = currency
        self.calendar = calendar
        self.timezone = timezone
        
        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.trades: List[Trade] = []
        
        # Performance tracking
        self.equity_curve = [{'timestamp': datetime.datetime.now(datetime.timezone.utc), 'equity': initial_cash}]
        self.daily_returns = []
        self.metrics: Dict[str, float] = {}
        
        logger.info(f"Portfolio initialized with {initial_cash:.2f} {currency}")
    
    def update_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        timestamp: Optional[datetime] = None,
        commission: float = 0.0
    ) -> Tuple[float, float]:
        """Update a position with a new trade.
        
        Args:
            symbol: Instrument symbol
            quantity: Number of units (positive for long, negative for short)
            price: Execution price per unit
            timestamp: Trade timestamp
            commission: Trade commission
            
        Returns:
            Tuple of (realized_pnl, new_quantity)
        """
        if timestamp is None:
            timestamp = datetime.datetime.now(datetime.timezone.utc)
            
        # Update cash balance (subtract cost of trade and commission)
        cost = quantity * price
        self.cash -= cost + commission
        
        # Get or create position
        if symbol not in self.positions:
            self.positions[symbol] = Position(symbol=symbol)
            
        position = self.positions[symbol]
        old_quantity = position.quantity
        old_avg_price = position.avg_price
        
        # Calculate new position
        new_quantity = old_quantity + quantity
        
        # Calculate realized P&L
        realized_pnl = 0.0
        
        if (quantity > 0 and old_quantity >= 0) or (quantity < 0 and old_quantity <= 0):
            # Adding to position in same direction
            total_quantity = abs(old_quantity) + abs(quantity)
            if total_quantity > 0:
                position.avg_price = (
                    (abs(old_quantity) * old_avg_price) + 
                    (abs(quantity) * price)
                ) / total_quantity
        else:
            # Closing or reversing position
            if abs(quantity) >= abs(old_quantity):
                # Full close or reverse
                realized_pnl = (price - old_avg_price) * old_quantity
                position.avg_price = price
            else:
                # Partial close
                realized_pnl = (price - old_avg_price) * -quantity
                
        # Update position
        position.quantity = new_quantity
        position.current_price = price
        position.timestamp = timestamp
        
        # Remove position if quantity is zero
        if new_quantity == 0:
            self.closed_positions.append(position)
            del self.positions[symbol]
        
        # Record trade
        if quantity != 0:
            side = 'long' if quantity > 0 else 'short'
            trade = Trade(
                symbol=symbol,
                quantity=abs(quantity),
                entry_price=price,
                exit_price=price,  # Will be updated when position is closed
                entry_time=timestamp,
                exit_time=timestamp,  # Will be updated when position is closed
                side=side,
                pnl=realized_pnl,
                pnl_pct=(realized_pnl / (abs(quantity) * price)) * 100 if quantity != 0 else 0.0,
                commission=commission
            )
            self.trades.append(trade)
        
        # Update equity curve
        self._update_equity_curve(timestamp)
        
        return realized_pnl, new_quantity
    
    def update_market_value(self, symbol: str, price: float, timestamp: Optional[datetime] = None) -> None:
        """Update the market value of a position."""
        if symbol in self.positions:
            self.positions[symbol].update_price(price, timestamp)
            
            # Update equity curve if this is a new timestamp
            if timestamp and self.equity_curve[-1]['timestamp'] < timestamp:
                self._update_equity_curve(timestamp)
    
    def _update_equity_curve(self, timestamp: datetime) -> None:
        """Update the equity curve with current portfolio value."""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        equity = self.cash + positions_value
        
        self.equity_curve.append({
            'timestamp': timestamp,
            'equity': equity,
            'cash': self.cash,
            'positions_value': positions_value
        })
    
    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics for the portfolio."""
        if not self.equity_curve:
            return {}
            
        # Convert to DataFrame for easier calculations
        df = pd.DataFrame(self.equity_curve)
        df['return'] = df['equity'].pct_change()
        
        # Basic metrics
        total_return = (df['equity'].iloc[-1] / self.initial_cash - 1) * 100
        annualized_return = 0.0
        sharpe_ratio = 0.0
        max_drawdown = 0.0
        
        if len(df) > 1:
            # Calculate annualized return (assuming daily data)
            days = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).days
            if days > 0:
                annualized_return = ((1 + total_return/100) ** (365.25/days) - 1) * 100
            
            # Calculate Sharpe ratio (assuming risk-free rate = 0)
            if df['return'].std() > 0:
                sharpe_ratio = (df['return'].mean() / df['return'].std()) * np.sqrt(252)  # Annualized
            
            # Calculate maximum drawdown
            df['cummax'] = df['equity'].cummax()
            df['drawdown'] = (df['equity'] - df['cummax']) / df['cummax']
            max_drawdown = df['drawdown'].min() * 100  # As percentage
        
        # Trade statistics
        win_trades = [t for t in self.trades if t.pnl > 0]
        loss_trades = [t for t in self.trades if t.pnl <= 0]
        win_rate = len(win_trades) / len(self.trades) * 100 if self.trades else 0
        
        avg_win = np.mean([t.pnl for t in win_trades]) if win_trades else 0
        avg_loss = np.mean([t.pnl for t in loss_trades]) if loss_trades else 0
        profit_factor = -avg_win / avg_loss if avg_loss < 0 else float('inf')
        
        # Update metrics
        self.metrics = {
            'initial_cash': self.initial_cash,
            'current_equity': df['equity'].iloc[-1],
            'cash': self.cash,
            'positions_value': df['positions_value'].iloc[-1],
            'total_return_pct': total_return,
            'annualized_return_pct': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown,
            'total_trades': len(self.trades),
            'win_rate_pct': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'timestamp': df['timestamp'].iloc[-1].isoformat()
        }
        
        return self.metrics
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get the current position for a symbol."""
        return self.positions.get(symbol)
    
    def get_positions(self) -> Dict[str, Position]:
        """Get all current positions."""
        return self.positions
    
    def get_trades(self) -> List[Trade]:
        """Get all completed trades."""
        return self.trades
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get the equity curve as a DataFrame."""
        return pd.DataFrame(self.equity_curve)
    
    def get_metrics(self) -> Dict[str, float]:
        """Get the latest performance metrics."""
        if not self.metrics:
            self.calculate_metrics()
        return self.metrics
