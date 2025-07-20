"""
Trading-specific metrics collection and analysis.

This module provides metrics specific to trading operations including:
- PnL (Profit and Loss)
- Drawdown calculations
- Slippage analysis
- Fill rates and execution quality
- Trade statistics
"""
"""
Trading Metrics Module

This module provides comprehensive tracking and analysis of trading performance metrics
with robust error handling and validation.
"""
from typing import Dict, List, Optional, Tuple, Any, Union, TypeVar, Type, Callable
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
import traceback
from functools import wraps
import time
from contextlib import contextmanager

from .metrics import metrics
from .config import config

# Type variables
T = TypeVar('T')
Number = Union[int, float]

# Configure logging
logger = logging.getLogger(__name__)

# Custom exceptions
class TradingMetricsError(Exception):
    """Base exception for trading metrics errors."""
    pass

class ValidationError(TradingMetricsError):
    """Raised when input validation fails."""
    pass

class TradeExecutionError(TradingMetricsError):
    """Raised when there's an error executing a trade."""
    pass

class MetricCalculationError(TradingMetricsError):
    """Raised when there's an error calculating metrics."""
    pass

class TradeDirection(Enum):
    """Direction of a trade."""
    LONG = "LONG"
    SHORT = "SHORT"

def validate_positive_number(field_name: str) -> Callable:
    """Decorator to validate that a field value is positive."""
    def decorator(method):
        @wraps(method)
        def wrapper(self, value):
            if not isinstance(value, (int, float)) or value < 0:
                raise ValidationError(
                    f"{field_name} must be a positive number, got {value}"
                )
            return method(self, value)
        return wrapper
    return decorator

def validate_timestamp(field_name: str) -> Callable:
    """Decorator to validate that a field is a datetime and not in the future."""
    def decorator(method):
        @wraps(method)
        def wrapper(self, value):
            if not isinstance(value, datetime):
                raise ValidationError(
                    f"{field_name} must be a datetime object, got {type(value).__name__}"
                )
            if value > datetime.utcnow():
                raise ValidationError(
                    f"{field_name} cannot be in the future: {value}"
                )
            return method(self, value)
        return wrapper
    return decorator

@dataclass
class Trade:
    """
    Represents a single trade execution with comprehensive validation.
    
    Attributes:
        trade_id: Unique identifier for the trade
        symbol: Trading pair symbol (e.g., 'BTC/USD')
        direction: TradeDirection.LONG or TradeDirection.SHORT
        entry_price: Price at which the position was opened
        exit_price: Price at which the position was closed
        quantity: Size of the position
        entry_time: Timestamp when the position was opened
        exit_time: Timestamp when the position was closed
        fees: Total fees paid for the trade
        slippage: Slippage amount in quote currency
        tags: Optional metadata for the trade
    """
    trade_id: str
    symbol: str
    direction: TradeDirection
    entry_price: float
    exit_price: float
    quantity: float
    entry_time: datetime
    exit_time: datetime
    fees: float = 0.0
    slippage: float = 0.0
    tags: Dict[str, str] = field(default_factory=dict)
    _initialized: bool = field(init=False, default=False)
    _metrics: Dict[str, Any] = field(init=False, default_factory=dict)
    
    def __post_init__(self):
        """Validate and initialize the trade."""
        try:
            self.validate()
            self.calculate_metrics()
            self._initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize trade {self.trade_id}: {str(e)}")
            raise TradeExecutionError(f"Trade initialization failed: {str(e)}") from e
    
    def validate(self):
        """Validate all trade parameters."""
        if not isinstance(self.trade_id, str) or not self.trade_id.strip():
            raise ValidationError("trade_id must be a non-empty string")
            
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValidationError("symbol must be a non-empty string")
            
        if not isinstance(self.direction, TradeDirection):
            raise ValidationError(f"direction must be a TradeDirection enum, got {type(self.direction)}")
            
        if not (isinstance(self.entry_price, (int, float)) and self.entry_price > 0):
            raise ValidationError(f"entry_price must be a positive number, got {self.entry_price}")
            
        if not (isinstance(self.exit_price, (int, float)) and self.exit_price > 0):
            raise ValidationError(f"exit_price must be a positive number, got {self.exit_price}")
            
        if not (isinstance(self.quantity, (int, float)) and self.quantity > 0):
            raise ValidationError(f"quantity must be a positive number, got {self.quantity}")
            
        if not isinstance(self.entry_time, datetime):
            raise ValidationError(f"entry_time must be a datetime object, got {type(self.entry_time)}")
            
        if not isinstance(self.exit_time, datetime):
            raise ValidationError(f"exit_time must be a datetime object, got {type(self.exit_time)}")
            
        if self.exit_time < self.entry_time:
            raise ValidationError("exit_time cannot be before entry_time")
            
        if not (isinstance(self.fees, (int, float)) and self.fees >= 0):
            raise ValidationError(f"fees must be a non-negative number, got {self.fees}")
            
        if not (isinstance(self.slippage, (int, float)) and self.slippage >= 0):
            raise ValidationError(f"slippage must be a non-negative number, got {self.slippage}")
            
        if self.tags is not None and not isinstance(self.tags, dict):
            raise ValidationError(f"tags must be a dictionary or None, got {type(self.tags)}")
    
    def calculate_metrics(self):
        """Calculate and store trade metrics."""
        try:
            # Calculate PnL
            if self.direction == TradeDirection.LONG:
                raw_pnl = (self.exit_price - self.entry_price) * self.quantity
            else:  # SHORT
                raw_pnl = (self.entry_price - self.exit_price) * self.quantity
                
            net_pnl = raw_pnl - self.fees
            pnl_pct = (net_pnl / (self.entry_price * self.quantity)) * 100
            
            # Calculate duration
            duration = (self.exit_time - self.entry_time).total_seconds()
            
            # Store metrics
            self._metrics.update({
                'raw_pnl': raw_pnl,
                'net_pnl': net_pnl,
                'pnl_pct': pnl_pct,
                'duration_seconds': duration,
                'entry_value': self.entry_price * self.quantity,
                'exit_value': self.exit_price * self.quantity,
                'is_winning': net_pnl > 0,
                'slippage_pct': (self.slippage / self.entry_price) * 100 if self.entry_price > 0 else 0,
                'fees_pct': (self.fees / (self.entry_price * self.quantity)) * 100 if self.entry_price * self.quantity > 0 else 0
            })
            
        except Exception as e:
            logger.error(f"Failed to calculate metrics for trade {self.trade_id}: {str(e)}")
            raise MetricCalculationError(f"Failed to calculate metrics: {str(e)}") from e
    
    @property
    def metrics(self) -> Dict[str, Any]:
        """Get a read-only view of trade metrics."""
        if not self._initialized:
            raise TradeExecutionError("Trade not properly initialized")
        return self._metrics.copy()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to a dictionary."""
        result = asdict(self)
        # Remove internal attributes
        result.pop('_initialized', None)
        # Add metrics
        result.update(self.metrics)
        return result

class TradingMetricsCollector:
    """
    Collects and analyzes trading-specific metrics with comprehensive error handling.
    
    This class is thread-safe and provides detailed metrics about trading performance,
    including PnL, drawdown, win rate, and risk-adjusted returns.
    """
    
    def __init__(self):
        """Initialize the metrics collector with empty state."""
        self._trades: Dict[str, Trade] = {}
        self._open_trades: Dict[str, Trade] = {}
        self._equity_curve: List[Tuple[datetime, float]] = []
        self._starting_equity: float = 10000.0  # Default starting equity
        self._lock = threading.RLock()  # For thread safety
        self._initialized: bool = False
        self._last_updated: Optional[datetime] = None
        
    def initialize_equity(self, starting_equity: float) -> None:
        """
        Initialize the equity curve with starting equity.
        
        Args:
            starting_equity: Initial equity amount to start tracking from.
            
        Raises:
            ValidationError: If starting_equity is not a positive number.
        """
        if not isinstance(starting_equity, (int, float)) or starting_equity <= 0:
            raise ValidationError("starting_equity must be a positive number")
            
        with self._lock:
            self._starting_equity = float(starting_equity)
            self._equity_curve = [(datetime.utcnow(), starting_equity)]
            self._initialized = True
            self._last_updated = datetime.utcnow()
            logger.info(f"Initialized equity tracking with starting equity: {starting_equity}")
    
    @property
    def trades(self) -> Dict[str, 'Trade']:
        """Get a read-only view of all recorded trades."""
        return self._trades.copy()
    
    @property
    def open_trades(self) -> Dict[str, 'Trade']:
        """Get a read-only view of currently open trades."""
        return self._open_trades.copy()
    
    @property
    def is_initialized(self) -> bool:
        """Check if the collector has been initialized with starting equity."""
        return self._initialized
    
    @contextmanager
    def _handle_metrics_error(self, operation: str, trade_id: str = None):
        """Context manager for handling errors during metrics operations."""
        try:
            yield
        except ValidationError as ve:
            logger.warning(f"Validation error during {operation}: {str(ve)}")
            raise
        except Exception as e:
            error_msg = f"Error during {operation}"
            if trade_id:
                error_msg += f" for trade {trade_id}"
            logger.error(f"{error_msg}: {str(e)}\n{traceback.format_exc()}")
            raise MetricCalculationError(f"{error_msg}: {str(e)}") from e
    
    def record_trade(
        self,
        trade_id: str,
        symbol: str,
        direction: TradeDirection,
        entry_price: float,
        exit_price: float,
        quantity: float,
        entry_time: datetime,
        exit_time: datetime,
        fees: float = 0.0,
        slippage: float = 0.0,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """
        Record a completed trade and update all relevant metrics.
        
        Args:
            trade_id: Unique identifier for the trade.
            symbol: Trading pair symbol (e.g., 'BTC/USD').
            direction: TradeDirection.LONG or TradeDirection.SHORT.
            entry_price: Price at which the position was opened.
            exit_price: Price at which the position was closed.
            quantity: Size of the position.
            entry_time: Timestamp when the position was opened.
            exit_time: Timestamp when the position was closed.
            fees: Total fees paid for the trade (default: 0.0).
            slippage: Slippage amount in quote currency (default: 0.0).
            tags: Optional metadata for the trade (default: None).
            
        Raises:
            TradeExecutionError: If there's an error recording the trade.
            ValidationError: If any input parameters are invalid.
        """
        with self._lock, self._handle_metrics_error("trade recording", trade_id):
            if not self._initialized:
                logger.warning("Metrics collector not initialized with starting equity, using default 10000.0")
                self.initialize_equity(10000.0)
            
            if trade_id in self._trades:
                logger.warning(f"Trade with ID {trade_id} already exists, skipping duplicate")
                return
                
            try:
                # Create and validate the trade
                trade = Trade(
                    trade_id=trade_id,
                    symbol=symbol,
                    direction=direction,
                    entry_price=entry_price,
                    exit_price=exit_price,
                    quantity=quantity,
                    entry_time=entry_time,
                    exit_time=exit_time,
                    fees=fees,
                    slippage=slippage,
                    tags=tags or {}
                )
                
                # Store the trade
                self._trades[trade_id] = trade
                
                # Update metrics
                self._update_metrics(trade)
                
                # Update last updated timestamp
                self._last_updated = datetime.utcnow()
                
                logger.info(f"Recorded trade {trade_id} for {symbol} {direction.value}")
                
            except Exception as e:
                error_msg = f"Failed to record trade {trade_id}"
                logger.error(f"{error_msg}: {str(e)}\n{traceback.format_exc()}")
                raise TradeExecutionError(f"{error_msg}: {str(e)}") from e
    
    def _update_metrics(self, trade: Trade) -> None:
        """
        Update all relevant metrics based on the recorded trade.
        
        Args:
            trade: The trade to update metrics for.
            
        Raises:
            MetricCalculationError: If there's an error updating metrics.
        """
        with self._handle_metrics_error("metrics update", trade.trade_id):
            try:
                # Get trade metrics
                metrics = trade.metrics
                
                # Update Prometheus metrics
                metrics.record_trade_pnl(
                    symbol=trade.symbol,
                    pnl=metrics['net_pnl'],
                    pnl_pct=metrics['pnl_pct'],
                    direction=trade.direction.value,
                    tags=trade.tags
                )
                
                # Record trade duration
                metrics.record_trade_duration(
                    symbol=trade.symbol,
                    direction=trade.direction.value,
                    duration_seconds=metrics['duration_seconds'],
                    tags=trade.tags
                )
                
                # Record slippage
                if trade.slippage > 0:
                    metrics.record_slippage(
                        symbol=trade.symbol,
                        direction=trade.direction.value,
                        slippage=trade.slippage,
                        price=trade.entry_price,
                        tags=trade.tags
                    )
                
                # Update equity curve
                current_equity = self._equity_curve[-1][1] if self._equity_curve else self._starting_equity
                self._equity_curve.append((trade.exit_time, current_equity + metrics['net_pnl']))
                
                # Update drawdown metrics
                self._update_drawdown_metrics()
                
                # Update win rate metrics
                self._update_win_rate_metrics()
                
            except Exception as e:
                error_msg = f"Failed to update metrics for trade {trade.trade_id}"
                logger.error(f"{error_msg}: {str(e)}\n{traceback.format_exc()}")
                raise MetricCalculationError(f"{error_msg}: {str(e)}") from e
    
    def _update_drawdown_metrics(self) -> None:
        """Calculate and update drawdown metrics."""
        with self._handle_metrics_error("drawdown calculation"):
            if not self._equity_curve:
                return
                
            try:
                equity_df = pd.DataFrame(self._equity_curve, columns=['timestamp', 'equity'])
                equity_df['peak'] = equity_df['equity'].cummax()
                equity_df['drawdown'] = (equity_df['equity'] / equity_df['peak']) - 1.0
                
                max_drawdown = equity_df['drawdown'].min() * 100  # as percentage
                current_drawdown = equity_df['drawdown'].iloc[-1] * 100  # as percentage
                
                # Update metrics
                metrics.record_drawdown(
                    current_drawdown=abs(current_drawdown),
                    max_drawdown=abs(max_drawdown)
                )
                
            except Exception as e:
                logger.error(f"Error calculating drawdown: {str(e)}\n{traceback.format_exc()}")
                raise
    
    def _update_win_rate_metrics(self) -> None:
        """Calculate and update win rate metrics."""
        with self._handle_metrics_error("win rate calculation"):
            if not self._trades:
                return
                
            try:
                # Group trades by symbol and timeframe (daily for now)
                trades_df = pd.DataFrame([t.to_dict() for t in self._trades.values()])
                
                if trades_df.empty:
                    return
                
                # Calculate win rate by symbol
                for symbol, symbol_trades in trades_df.groupby('symbol'):
                    total_trades = len(symbol_trades)
                    winning_trades = len(symbol_trades[symbol_trades['net_pnl'] > 0])
                    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
                    
                    metrics.record_win_rate(
                        symbol=symbol,
                        timeframe="all",
                        win_rate=win_rate
                    )
                    
            except Exception as e:
                logger.error(f"Error calculating win rate: {str(e)}\n{traceback.format_exc()}")
                raise
    
    def get_summary_metrics(self) -> Dict[str, Any]:
        """
        Calculate and return summary trading metrics.
        
        Returns:
            Dictionary containing various trading performance metrics.
            
        Raises:
            MetricCalculationError: If there's an error calculating metrics.
        """
        with self._handle_metrics_error("summary metrics calculation"):
            if not self._trades:
                return {}
                
            try:
                trades_df = pd.DataFrame([t.to_dict() for t in self._trades.values()])
                
                if trades_df.empty:
                    return {}
                
                # Basic metrics
                total_trades = len(trades_df)
                winning_trades = len(trades_df[trades_df['net_pnl'] > 0])
                losing_trades = total_trades - winning_trades
                win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
                
                # PnL metrics
                total_pnl = trades_df['net_pnl'].sum()
                avg_win = trades_df[trades_df['net_pnl'] > 0]['net_pnl'].mean() if winning_trades > 0 else 0
                avg_loss = abs(trades_df[trades_df['net_pnl'] < 0]['net_pnl'].mean()) if losing_trades > 0 else 0
                profit_factor = (avg_win * winning_trades) / (avg_loss * losing_trades) if losing_trades > 0 else float('inf')
                
                # Risk metrics
                sharpe_ratio = self._calculate_sharpe_ratio()
                sortino_ratio = self._calculate_sortino_ratio()
                
                # Get max drawdown from metrics
                try:
                    max_drawdown = abs(metrics.get_metric('trading_max_drawdown').collect()[0].samples[0].value)
                except Exception as e:
                    logger.warning(f"Could not get max drawdown from metrics: {str(e)}")
                    max_drawdown = 0.0
                
                # Calculate average trade metrics
                avg_trade_pnl = total_pnl / total_trades if total_trades > 0 else 0
                avg_trade_duration = trades_df['duration_seconds'].mean()
                
                # Calculate risk metrics
                pnl_std = trades_df['net_pnl'].std()
                pnl_skew = trades_df['net_pnl'].skew()
                pnl_kurt = trades_df['net_pnl'].kurtosis()
                
                return {
                    'summary': {
                        'total_trades': total_trades,
                        'winning_trades': winning_trades,
                        'losing_trades': losing_trades,
                        'win_rate': win_rate,
                        'total_pnl': total_pnl,
                        'avg_trade_pnl': avg_trade_pnl,
                        'avg_win': avg_win,
                        'avg_loss': avg_loss,
                        'profit_factor': profit_factor,
                        'max_drawdown': max_drawdown,
                        'sharpe_ratio': sharpe_ratio,
                        'sortino_ratio': sortino_ratio,
                        'avg_trade_duration_seconds': avg_trade_duration,
                        'pnl_std_dev': pnl_std,
                        'pnl_skewness': pnl_skew,
                        'pnl_kurtosis': pnl_kurt,
                        'last_updated': self._last_updated.isoformat() if self._last_updated else None
                    },
                    'equity_curve': [{'timestamp': ts.isoformat(), 'equity': eq} for ts, eq in self._equity_curve[-1000:]]  # Last 1000 points
                }
                
            except Exception as e:
                error_msg = f"Error calculating summary metrics: {str(e)}"
                logger.error(f"{error_msg}\n{traceback.format_exc()}")
                raise MetricCalculationError(error_msg) from e
    
    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Calculate the annualized Sharpe ratio of the strategy.
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 0.0).
            
        Returns:
            Annualized Sharpe ratio.
        """
        with self._handle_metrics_error("Sharpe ratio calculation"):
            if not self._equity_curve or len(self._equity_curve) < 2:
                return 0.0
                
            try:
                equity_df = pd.DataFrame(self._equity_curve, columns=['timestamp', 'equity'])
                equity_df = equity_df.set_index('timestamp')
                
                # Resample to daily returns if we have enough data
                if len(equity_df) > 1:
                    # Calculate daily returns
                    returns = equity_df['equity'].pct_change().dropna()
                    
                    if len(returns) < 2:
                        return 0.0
                    
                    # Calculate annualized metrics
                    avg_daily_return = returns.mean() * 252  # 252 trading days
                    daily_volatility = returns.std() * np.sqrt(252)  # Annualized volatility
                    
                    if daily_volatility == 0:
                        return 0.0
                        
                    sharpe = (avg_daily_return - risk_free_rate) / daily_volatility
                    
                    # Update metrics
                    metrics.record_sharpe_ratio(sharpe)
                    
                    return float(sharpe)
                
                return 0.0
                
            except Exception as e:
                logger.error(f"Error calculating Sharpe ratio: {str(e)}\n{traceback.format_exc()}")
                return 0.0
    
    def _calculate_sortino_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Calculate the annualized Sortino ratio of the strategy.
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 0.0).
            
        Returns:
            Annualized Sortino ratio.
        """
        with self._handle_metrics_error("Sortino ratio calculation"):
            if not self._equity_curve or len(self._equity_curve) < 2:
                return 0.0
                
            try:
                equity_df = pd.DataFrame(self._equity_curve, columns=['timestamp', 'equity'])
                equity_df = equity_df.set_index('timestamp')
                
                # Resample to daily returns if we have enough data
                if len(equity_df) > 1:
                    # Calculate daily returns
                    returns = equity_df['equity'].pct_change().dropna()
                    
                    if len(returns) < 2:
                        return 0.0
                    
                    # Calculate annualized metrics
                    avg_daily_return = returns.mean() * 252  # 252 trading days
                    
                    # Calculate downside deviation
                    downside_returns = returns[returns < 0]
                    if len(downside_returns) == 0:
                        return float('inf')  # No downside risk
                        
                    downside_deviation = downside_returns.std() * np.sqrt(252)  # Annualized
                    
                    if downside_deviation == 0:
                        return float('inf')
                        
                    sortino = (avg_daily_return - risk_free_rate) / downside_deviation
                    
                    # Update metrics
                    metrics.record_sortino_ratio(sortino)
                    
                    return float(sortino)
                
                return 0.0
                
            except Exception as e:
                logger.error(f"Error calculating Sortino ratio: {str(e)}\n{traceback.format_exc()}")
                return 0.0

# Singleton instance
trading_metrics = TradingMetricsCollector()
