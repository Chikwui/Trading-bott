"""
Backtesting Framework for Trading Strategies

This module provides a flexible backtesting framework to test trading strategies
using historical data. It supports multiple assets, timeframes, and includes
performance metrics and visualization.
"""
import os
import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field
from enum import Enum, auto
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# Add parent directory to path to allow imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data.providers.mt5_provider import MT5DataProvider
from core.data.provider_config import DataProviderConfig
from core.indicators.ta_indicators import (
    moving_average, rsi, macd, bollinger_bands, atr, stochastic_oscillator,
    MovingAverageType, IndicatorType, IndicatorResult
)
from core.indicators.advanced_indicators import (
    ichimoku_cloud, parabolic_sar, adx, volume_profile, VolumeProfileLevels
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('backtest.log')
    ]
)
logger = logging.getLogger(__name__)

class TradeDirection(Enum):
    """Direction of a trade."""
    LONG = auto()
    SHORT = auto()
    NONE = auto()

@dataclass
class Trade:
    """Represents a single trade."""
    entry_time: datetime
    exit_time: Optional[datetime] = None
    entry_price: float = 0.0
    exit_price: float = 0.0
    direction: TradeDirection = TradeDirection.NONE
    size: float = 1.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    holding_period: timedelta = field(default_factory=timedelta)
    indicators: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class BacktestResult:
    """Results of a backtest."""
    trades: List[Trade] = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)
    returns: pd.Series = field(default_factory=pd.Series)
    metrics: Dict[str, float] = field(default_factory=dict)
    indicators: Dict[str, Any] = field(default_factory=dict)
    
    def calculate_metrics(self, risk_free_rate: float = 0.0) -> Dict[str, float]:
        """Calculate performance metrics for the backtest."""
        if not self.trades:
            return {}
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.pnl > 0])
        losing_trades = total_trades - winning_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # PnL metrics
        total_pnl = sum(t.pnl for t in self.trades)
        avg_pnl = np.mean([t.pnl for t in self.trades]) if self.trades else 0
        avg_win = np.mean([t.pnl for t in self.trades if t.pnl > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t.pnl for t in self.trades if t.pnl <= 0]) if losing_trades > 0 else 0
        profit_factor = -avg_win / avg_loss if avg_loss < 0 else float('inf')
        
        # Risk metrics
        returns = np.array([t.pnl_pct / 100.0 for t in self.trades])
        sharpe_ratio = self._calculate_sharpe_ratio(returns, risk_free_rate)
        sortino_ratio = self._calculate_sortino_ratio(returns, risk_free_rate)
        max_drawdown = self._calculate_max_drawdown()
        calmar_ratio = total_pnl / abs(max_drawdown) if max_drawdown < 0 else float('inf')
        
        # Time metrics
        holding_periods = [t.holding_period.total_seconds() / 3600 for t in self.trades if t.holding_period]
        avg_holding_hours = np.mean(holding_periods) if holding_periods else 0
        
        self.metrics = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate * 100,  # as percentage
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'avg_holding_hours': avg_holding_hours,
        }
        
        return self.metrics
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float) -> float:
        """Calculate the Sharpe ratio."""
        if len(returns) < 2:
            return 0.0
        excess_returns = returns - (risk_free_rate / 252)  # Annualized daily risk-free rate
        return np.sqrt(252) * (np.mean(excess_returns) / np.std(excess_returns, ddof=1))
    
    def _calculate_sortino_ratio(self, returns: np.ndarray, risk_free_rate: float) -> float:
        """Calculate the Sortino ratio."""
        if len(returns) < 2:
            return 0.0
        excess_returns = returns - (risk_free_rate / 252)  # Annualized daily risk-free rate
        downside_returns = returns[returns < 0]
        if len(downside_returns) == 0:
            return float('inf')
        downside_std = np.std(downside_returns, ddof=1)
        return np.sqrt(252) * (np.mean(excess_returns) / downside_std) if downside_std != 0 else float('inf')
    
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown."""
        if not self.equity_curve.empty:
            cum_returns = (1 + self.equity_curve.pct_change()).cumprod()
            peak = cum_returns.expanding(min_periods=1).max()
            drawdown = (cum_returns / peak) - 1
            return drawdown.min() * 100  # as percentage
        return 0.0

class BacktestEngine:
    """Backtesting engine for trading strategies."""
    
    def __init__(self, initial_balance: float = 10000.0):
        """Initialize the backtesting engine."""
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0.0
        self.current_trade = None
        self.trades = []
        self.equity_curve = pd.Series(dtype=float)
        self.current_time = None
        self.data_provider = None
        self.strategy = None
    
    async def run_backtest(
        self,
        symbol: str,
        timeframe: str,
        start_date: datetime,
        end_date: datetime,
        strategy: Callable,
        strategy_params: Optional[dict] = None,
        data_provider: Optional[MT5DataProvider] = None,
        commission: float = 0.0005,  # 0.05% commission per trade
        slippage: float = 0.0001,    # 0.01% slippage per trade
        position_size: float = 0.1,  # 10% of balance per trade
        max_drawdown: float = 0.2,   # 20% max drawdown
        stop_loss: float = 0.02,     # 2% stop loss
        take_profit: float = 0.04    # 4% take profit
    ) -> BacktestResult:
        """
        Run a backtest for the given symbol and timeframe.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Timeframe (e.g., 'H1', 'D1')
            start_date: Start date for backtest
            end_date: End date for backtest
            strategy: Strategy function that generates signals
            strategy_params: Parameters for the strategy
            data_provider: Data provider instance
            commission: Commission per trade (as a fraction of trade value)
            slippage: Slippage per trade (as a fraction of trade value)
            position_size: Fraction of balance to risk per trade
            max_drawdown: Maximum allowed drawdown (as a fraction of balance)
            stop_loss: Stop loss level (as a fraction of entry price)
            take_profit: Take profit level (as a fraction of entry price)
            
        Returns:
            BacktestResult with trades and performance metrics
        """
        logger.info(f"Starting backtest for {symbol} {timeframe} from {start_date} to {end_date}")
        
        # Initialize data provider if not provided
        if data_provider is None:
            mt5_config = DataProviderConfig(
                name="MT5",
                provider_type="mt5",
                enabled=True,
                extra={
                    'login': os.getenv('MT5_LOGIN'),
                    'password': os.getenv('MT5_PASSWORD'),
                    'server': os.getenv('MT5_SERVER'),
                    'path': os.getenv('MT5_PATH'),
                    'timeout': 60000
                }
            )
            self.data_provider = MT5DataProvider(mt5_config)
            await self.data_provider.initialize()
            await self.data_provider.connect()
        else:
            self.data_provider = data_provider
        
        # Load historical data
        ohlcv = await self.data_provider.get_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_date,
            end_time=end_date
        )
        
        if not ohlcv:
            raise ValueError(f"No data returned for {symbol} {timeframe} from {start_date} to {end_date}")
        
        # Convert to DataFrame
        data = []
        for candle in ohlcv:
            data.append({
                'timestamp': candle.timestamp,
                'open': candle.open,
                'high': candle.high,
                'low': candle.low,
                'close': candle.close,
                'volume': candle.volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        # Initialize strategy
        self.strategy = strategy
        self.strategy.initialize(df, **(strategy_params or {}))
        
        # Initialize backtest state
        self.balance = self.initial_balance
        self.position = 0.0
        self.current_trade = None
        self.trades = []
        self.equity_curve = pd.Series(index=df.index, dtype=float)
        
        # Run backtest
        for i in range(1, len(df)):
            current_candle = df.iloc[i]
            prev_candle = df.iloc[i-1]
            self.current_time = current_candle.name
            
            # Update equity curve
            if self.position != 0 and self.current_trade is not None:
                # Update open trade PnL
                if self.position > 0:  # Long position
                    self.current_trade.pnl = (current_candle['close'] - self.current_trade.entry_price) * self.position
                else:  # Short position
                    self.current_trade.pnl = (self.current_trade.entry_price - current_candle['close']) * abs(self.position)
                
                self.current_trade.pnl_pct = (self.current_trade.pnl / (abs(self.current_trade.entry_price * self.position))) * 100
                self.current_trade.holding_period = self.current_time - self.current_trade.entry_time
                
                # Check for exit conditions
                exit_trade = False
                exit_reason = ""
                
                # Stop loss / Take profit
                if self.position > 0:  # Long position
                    if current_candle['low'] <= self.current_trade.entry_price * (1 - stop_loss):
                        exit_trade = True
                        exit_reason = "STOP_LOSS"
                    elif current_candle['high'] >= self.current_trade.entry_price * (1 + take_profit):
                        exit_trade = True
                        exit_reason = "TAKE_PROFIT"
                else:  # Short position
                    if current_candle['high'] >= self.current_trade.entry_price * (1 + stop_loss):
                        exit_trade = True
                        exit_reason = "STOP_LOSS"
                    elif current_candle['low'] <= self.current_trade.entry_price * (1 - take_profit):
                        exit_trade = True
                        exit_reason = "TAKE_PROFIT"
                
                # Max drawdown check
                if (self.balance + self.current_trade.pnl) / self.initial_balance - 1 <= -max_drawdown:
                    exit_trade = True
                    exit_reason = "MAX_DRAWDOWN_REACHED"
                
                # Close trade if exit condition met
                if exit_trade:
                    self._close_trade(current_candle['close'], exit_reason)
            
            # Generate signals
            signals = self.strategy.calculate_signals(df.iloc[:i+1])
            
            # Execute trades based on signals
            if signals and 'signal' in signals:
                signal = signals['signal']
                
                # Close existing position if signal is opposite
                if signal != 0 and self.position != 0 and ((signal > 0 and self.position < 0) or (signal < 0 and self.position > 0)):
                    self._close_trade(current_candle['close'], "REVERSAL_SIGNAL")
                
                # Open new position if no position is open
                if signal != 0 and self.position == 0:
                    self._open_trade(
                        entry_price=current_candle['close'],
                        direction=TradeDirection.LONG if signal > 0 else TradeDirection.SHORT,
                        size=(self.balance * position_size) / current_candle['close'],
                        indicators=signals.get('indicators', {})
                    )
            
            # Update equity curve
            self.equity_curve[self.current_time] = self.balance + (
                self.current_trade.pnl if self.current_trade else 0.0
            )
        
        # Close any open position at the end
        if self.current_trade is not None:
            self._close_trade(df.iloc[-1]['close'], "END_OF_BACKTEST")
        
        # Calculate metrics
        result = BacktestResult(
            trades=self.trades,
            equity_curve=self.equity_curve,
            returns=self.equity_curve.pct_change().dropna()
        )
        
        result.calculate_metrics()
        
        logger.info("Backtest completed successfully")
        logger.info(f"Final Balance: ${self.balance:.2f} ({(self.balance/self.initial_balance-1)*100:.2f}%)")
        logger.info(f"Total Trades: {len(self.trades)}")
        logger.info(f"Win Rate: {result.metrics.get('win_rate', 0):.2f}%")
        
        return result
    
    def _open_trade(self, entry_price: float, direction: TradeDirection, size: float, indicators: dict = None):
        """Open a new trade."""
        self.current_trade = Trade(
            entry_time=self.current_time,
            entry_price=entry_price,
            direction=direction,
            size=size,
            indicators=indicators or {}
        )
        
        if direction == TradeDirection.LONG:
            self.position = size
            logger.info(f"{self.current_time} - Opened LONG position at {entry_price:.5f} (Size: {size:.2f})")
        else:
            self.position = -size
            logger.info(f"{self.current_time} - Opened SHORT position at {entry_price:.5f} (Size: {size:.2f})")
    
    def _close_trade(self, exit_price: float, reason: str = ""):
        """Close the current trade."""
        if self.current_trade is None:
            return
        
        self.current_trade.exit_time = self.current_time
        self.current_trade.exit_price = exit_price
        self.current_trade.holding_period = self.current_time - self.current_trade.entry_time
        
        # Calculate PnL
        if self.current_trade.direction == TradeDirection.LONG:
            self.current_trade.pnl = (exit_price - self.current_trade.entry_price) * self.current_trade.size
        else:
            self.current_trade.pnl = (self.current_trade.entry_price - exit_price) * self.current_trade.size
        
        # Apply commission and slippage
        trade_value = abs(self.current_trade.entry_price * self.current_trade.size)
        commission = trade_value * 0.0005  # 0.05% commission
        slippage = trade_value * 0.0001    # 0.01% slippage
        self.current_trade.pnl -= (commission + slippage)
        
        # Update balance
        self.balance += self.current_trade.pnl
        
        # Calculate PnL percentage
        self.current_trade.pnl_pct = (self.current_trade.pnl / (self.current_trade.entry_price * self.current_trade.size)) * 100
        
        # Add metadata
        self.current_trade.metadata['exit_reason'] = reason
        
        # Add to trades list
        self.trades.append(self.current_trade)
        
        logger.info(
            f"{self.current_time} - Closed {self.current_trade.direction.name} position at {exit_price:.5f} "
            f"(PnL: ${self.current_trade.pnl:.2f}, {self.current_trade.pnl_pct:.2f}%) - {reason}"
        )
        
        # Reset position and current trade
        self.position = 0.0
        self.current_trade = None

    def plot_results(self, result: BacktestResult, symbol: str):
        """Plot backtest results."""
        if result.equity_curve.empty:
            logger.warning("No equity curve data to plot")
            return
        
        plt.style.use('seaborn')
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot equity curve
        ax1.plot(result.equity_curve.index, result.equity_curve.values, 'b-', linewidth=2, label='Equity Curve')
        ax1.set_title(f'{symbol} - Backtest Results', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Equity ($)', fontsize=12)
        ax1.legend(loc='upper left')
        ax1.grid(True)
        
        # Format x-axis dates
        date_format = DateFormatter("%Y-%m-%d")
        ax1.xaxis.set_major_formatter(date_format)
        
        # Plot drawdown
        equity = result.equity_curve
        rolling_max = equity.expanding().max()
        drawdown = (equity / rolling_max - 1.0) * 100
        
        ax2.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
        ax2.plot(drawdown.index, drawdown.values, 'r-', linewidth=1, label='Drawdown')
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.legend(loc='lower left')
        ax2.grid(True)
        
        # Format x-axis dates
        ax2.xaxis.set_major_formatter(date_format)
        
        plt.tight_layout()
        
        # Save figure
        os.makedirs('backtest_results', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fig_path = f'backtest_results/{symbol}_backtest_{timestamp}.png'
        plt.savefig(fig_path)
        logger.info(f"Backtest results plot saved to {fig_path}")
        
        # Show plot
        plt.show()

class MovingAverageCrossoverStrategy:
    """Example strategy using moving average crossover."""
    
    def __init__(self):
        self.fast_window = 10
        self.slow_window = 30
        self.rsi_window = 14
        self.overbought = 70
        self.oversold = 30
        self.data = None
    
    def initialize(self, data: pd.DataFrame, **params):
        """Initialize the strategy with historical data."""
        self.data = data.copy()
        
        # Update parameters if provided
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def calculate_signals(self, data: pd.DataFrame) -> dict:
        """Calculate trading signals."""
        if len(data) < max(self.slow_window, self.rsi_window) + 1:
            return {}
        
        # Calculate indicators
        close = data['close']
        
        # Moving Averages
        fast_ma = moving_average(close, window=self.fast_window, ma_type=MovingAverageType.SMA).values
        slow_ma = moving_average(close, window=self.slow_window, ma_type=MovingAverageType.SMA).values
        
        # RSI
        rsi_values = rsi(close, window=self.rsi_window).values
        
        # Generate signals
        signal = 0
        
        # Bullish signal: Fast MA crosses above Slow MA and RSI is not overbought
        if (fast_ma[-2] <= slow_ma[-2] and 
            fast_ma[-1] > slow_ma[-1] and 
            rsi_values[-1] < self.overbought):
            signal = 1
        
        # Bearish signal: Fast MA crosses below Slow MA and RSI is not oversold
        elif (fast_ma[-2] >= slow_ma[-2] and 
              fast_ma[-1] < slow_ma[-1] and 
              rsi_values[-1] > self.oversold):
            signal = -1
        
        return {
            'signal': signal,
            'indicators': {
                'fast_ma': fast_ma[-1],
                'slow_ma': slow_ma[-1],
                'rsi': rsi_values[-1]
            }
        }

async def run_example():
    """Run an example backtest."""
    # Check if required environment variables are set
    required_vars = ['MT5_LOGIN', 'MT5_PASSWORD', 'MT5_SERVER']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("Error: The following environment variables must be set:")
        for var in missing_vars:
            print(f"- {var}")
        print("\nPlease create a .env file with these variables or set them in your environment.")
        return
    
    # Initialize backtest engine
    engine = BacktestEngine(initial_balance=10000.0)
    
    # Define strategy parameters
    strategy_params = {
        'fast_window': 10,
        'slow_window': 30,
        'rsi_window': 14,
        'overbought': 70,
        'oversold': 30
    }
    
    # Run backtest
    result = await engine.run_backtest(
        symbol='EURUSD',
        timeframe='H1',
        start_date=datetime.now() - timedelta(days=90),  # Last 90 days
        end_date=datetime.now(),
        strategy=MovingAverageCrossoverStrategy(),
        strategy_params=strategy_params,
        commission=0.0005,  # 0.05% commission
        slippage=0.0001,    # 0.01% slippage
        position_size=0.1,  # 10% of balance per trade
        max_drawdown=0.2,   # 20% max drawdown
        stop_loss=0.02,     # 2% stop loss
        take_profit=0.04    # 4% take profit
    )
    
    # Plot results
    engine.plot_results(result, 'EURUSD')
    
    # Print performance metrics
    print("\n=== Backtest Results ===")
    print(f"Initial Balance: ${engine.initial_balance:.2f}")
    print(f"Final Balance: ${engine.balance:.2f}")
    print(f"Total Return: {((engine.balance / engine.initial_balance) - 1) * 100:.2f}%")
    print(f"Total Trades: {len(result.trades)}")
    print(f"Win Rate: {result.metrics.get('win_rate', 0):.2f}%")
    print(f"Profit Factor: {result.metrics.get('profit_factor', 0):.2f}")
    print(f"Sharpe Ratio: {result.metrics.get('sharpe_ratio', 0):.2f}")
    print(f"Max Drawdown: {result.metrics.get('max_drawdown', 0):.2f}%")
    print("======================")

if __name__ == "__main__":
    # Load environment variables from .env file if it exists
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run the example
    asyncio.run(run_example())
