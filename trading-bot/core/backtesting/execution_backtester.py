"""
Execution Algorithm Backtester

This module provides a framework for backtesting execution algorithms with ML integration,
allowing for performance evaluation and optimization.
"""
import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
import json
import os
from pathlib import Path

# Import ML components
from core.ml import ModelPipeline, ModelConfig, FeatureEngineer
from core.trading.algorithms.base import ExecutionAlgorithm
from core.trading.order_types import Order, OrderSide, OrderType, OrderStatus
from core.risk_models import RiskManager, RiskAssessment

logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    """Configuration for execution algorithm backtesting."""
    # Data configuration
    data_path: str = "data/backtest/"
    symbols: List[str] = field(default_factory=lambda: ["BTC/USDT", "ETH/USDT"])
    timeframe: str = "1m"
    start_date: str = "2023-01-01"
    end_date: str = "2023-12-31"
    
    # Execution configuration
    initial_balance: float = 100000.0  # Initial portfolio balance
    max_position_size: float = 0.1  # Max position size as fraction of portfolio
    commission_rate: float = 0.0005  # 5 bps per trade
    
    # ML configuration
    use_ml: bool = True
    ml_config: Dict[str, Any] = field(default_factory=lambda: {
        'model_type': 'xgboost',
        'feature_window': 100,
        'prediction_horizon': 10,
        'train_interval': 3600
    })
    
    # Output configuration
    output_dir: str = "results/backtest/"
    save_trades: bool = True
    save_metrics: bool = True
    plot_results: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {k: v for k, v in self.__dict__.items() 
                if not k.startswith('_') and not callable(v)}

class ExecutionBacktester:
    """Backtesting framework for execution algorithms."""
    
    def __init__(self, config: Optional[BacktestConfig] = None):
        """Initialize the backtester."""
        self.config = config or BacktestConfig()
        self.data = {}
        self.results = {}
        self.metrics = {}
        self.risk_manager = RiskManager()
        self.current_time = None
        self.current_prices = {}
        self.portfolio = {
            'cash': self.config.initial_balance,
            'positions': {symbol: 0.0 for symbol in self.config.symbols},
            'value': self.config.initial_balance,
            'trades': []
        }
        
        # Create output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    async def load_data(self):
        """Load historical market data for backtesting."""
        logger.info(f"Loading market data from {self.config.data_path}")
        
        for symbol in self.config.symbols:
            try:
                # Load OHLCV data (example: CSV format)
                file_path = Path(self.config.data_path) / f"{symbol.replace('/', '-')}_{self.config.timeframe}.csv"
                if not file_path.exists():
                    logger.warning(f"Data file not found: {file_path}")
                    continue
                    
                df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
                df = df.sort_index()
                
                # Filter by date range
                mask = (df.index >= self.config.start_date) & (df.index <= self.config.end_date)
                self.data[symbol] = df[mask].copy()
                
                logger.info(f"Loaded {len(self.data[symbol])} rows for {symbol}")
                
            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {str(e)}")
                
    async def run_backtest(
        self, 
        algorithm_class: type,
        algorithm_params: Optional[Dict] = None,
        symbols: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Run backtest for the given execution algorithm."""
        logger.info("Starting backtest...")
        
        # Use provided symbols or default from config
        symbols = symbols or self.config.symbols
        algorithm_params = algorithm_params or {}
        
        # Initialize algorithm instances
        algorithms = {}
        for symbol in symbols:
            if symbol not in self.data:
                logger.warning(f"No data for symbol {symbol}, skipping")
                continue
                
            # Create algorithm instance with ML config if enabled
            params = algorithm_params.copy()
            if self.config.use_ml:
                params['ml_config'] = self.config.ml_config
                
            algorithms[symbol] = algorithm_class(
                exchange_adapter=BacktestExchangeAdapter(
                    symbol=symbol,
                    data=self.data[symbol],
                    commission_rate=self.config.commission_rate
                ),
                position_manager=BacktestPositionManager(
                    initial_balance=self.config.initial_balance
                ),
                **params
            )
        
        # Main backtest loop
        for symbol, df in self.data.items():
            if symbol not in algorithms:
                continue
                
            logger.info(f"Running backtest for {symbol}")
            
            # Initialize algorithm
            algo = algorithms[symbol]
            
            # Process each time step
            for i, (timestamp, row) in enumerate(df.iterrows()):
                self.current_time = timestamp
                self.current_prices[symbol] = row['close']
                
                # Update portfolio value
                await self._update_portfolio_value(symbol, row['close'])
                
                # Generate signals or execute orders based on strategy
                # This is where you would integrate with your strategy logic
                
                # Example: Place a market order every N periods
                if i % 100 == 0 and i > 0:
                    order = Order(
                        symbol=symbol,
                        side=OrderSide.BUY if i % 200 == 0 else OrderSide.SELL,
                        order_type=OrderType.MARKET,
                        quantity=0.1,  # Example size
                        timestamp=timestamp
                    )
                    
                    # Execute order through algorithm
                    filled_order = await algo.execute(order)
                    
                    # Record trade
                    if filled_order.status == OrderStatus.FILLED:
                        self._record_trade(filled_order, row['close'])
            
            # Calculate performance metrics
            self._calculate_metrics(symbol)
        
        # Save results
        self._save_results()
        
        logger.info("Backtest completed successfully")
        return self.metrics
    
    async def _update_portfolio_value(self, symbol: str, price: float):
        """Update portfolio value based on current prices."""
        # Update position value
        position_value = self.portfolio['positions'].get(symbol, 0) * price
        
        # Calculate total portfolio value
        total_value = self.portfolio['cash']
        for sym, qty in self.portfolio['positions'].items():
            if sym == symbol:
                total_value += position_value
            else:
                total_value += qty * self.current_prices.get(sym, 0)
        
        self.portfolio['value'] = total_value
    
    def _record_trade(self, order: Order, price: float):
        """Record a trade in the portfolio."""
        # Calculate trade value and commission
        trade_value = float(order.quantity * price)
        commission = trade_value * self.config.commission_rate
        
        # Update cash and positions
        if order.side == OrderSide.BUY:
            self.portfolio['cash'] -= (trade_value + commission)
            self.portfolio['positions'][order.symbol] = self.portfolio['positions'].get(order.symbol, 0) + order.quantity
        else:  # SELL
            self.portfolio['cash'] += (trade_value - commission)
            self.portfolio['positions'][order.symbol] = self.portfolio['positions'].get(order.symbol, 0) - order.quantity
        
        # Record trade
        trade = {
            'timestamp': order.timestamp,
            'symbol': order.symbol,
            'side': order.side.value,
            'quantity': float(order.quantity),
            'price': float(price),
            'commission': commission,
            'portfolio_value': self.portfolio['value']
        }
        self.portfolio['trades'].append(trade)
    
    def _calculate_metrics(self, symbol: str):
        """Calculate performance metrics for the backtest."""
        if not self.portfolio['trades']:
            logger.warning("No trades were executed")
            return
        
        trades = pd.DataFrame(self.portfolio['trades'])
        
        # Basic metrics
        num_trades = len(trades)
        winning_trades = len(trades[trades['pnl'] > 0])
        losing_trades = num_trades - winning_trades
        win_rate = winning_trades / num_trades if num_trades > 0 else 0
        
        # PnL metrics
        total_pnl = trades['pnl'].sum()
        avg_win = trades[trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = abs(trades[trades['pnl'] < 0]['pnl'].mean()) if losing_trades > 0 else 0
        profit_factor = (winning_trades * avg_win) / (losing_trades * avg_loss) if losing_trades > 0 else float('inf')
        
        # Risk metrics
        max_drawdown = self._calculate_max_drawdown(trades['portfolio_value'].values)
        sharpe_ratio = self._calculate_sharpe_ratio(trades['pnl'].values)
        
        self.metrics[symbol] = {
            'num_trades': num_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'final_portfolio_value': self.portfolio['value']
        }
    
    def _calculate_max_drawdown(self, values: np.ndarray) -> float:
        """Calculate maximum drawdown from a series of portfolio values."""
        peak = values[0]
        max_dd = 0.0
        
        for value in values[1:]:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
                
        return max_dd
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate annualized Sharpe ratio from a series of returns."""
        if len(returns) < 2:
            return 0.0
            
        excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
        return np.sqrt(252) * (excess_returns.mean() / (excess_returns.std() + 1e-9))
    
    def _save_results(self):
        """Save backtest results to disk."""
        # Save metrics
        if self.config.save_metrics:
            metrics_path = self.output_dir / "metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics, f, indent=2, default=str)
            logger.info(f"Saved metrics to {metrics_path}")
        
        # Save trades
        if self.config.save_trades and self.portfolio['trades']:
            trades_path = self.output_dir / "trades.csv"
            trades_df = pd.DataFrame(self.portfolio['trades'])
            trades_df.to_csv(trades_path, index=False)
            logger.info(f"Saved trades to {trades_path}")
        
        # Save config
        config_path = self.output_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        logger.info(f"Saved config to {config_path}")


class BacktestExchangeAdapter:
    """Mock exchange adapter for backtesting."""
    
    def __init__(self, symbol: str, data: pd.DataFrame, commission_rate: float = 0.0005):
        self.symbol = symbol
        self.data = data
        self.commission_rate = commission_rate
        self.current_idx = 0
        
    async def get_market_data(self) -> Dict[str, Any]:
        """Get current market data."""
        if self.current_idx >= len(self.data):
            return None
            
        row = self.data.iloc[self.current_idx]
        self.current_idx += 1
        
        return {
            'timestamp': row.name,
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
            'volume': row['volume'],
            'bid': row['close'] * 0.999,  # Simulate bid/ask spread
            'ask': row['close'] * 1.001,
            'spread': (row['close'] * 0.002) / row['close']  # 20 bps spread
        }
    
    async def execute_order(self, order: Order) -> Order:
        """Execute an order (mock implementation)."""
        # Get current market data
        market_data = await self.get_market_data()
        if not market_data:
            order.status = OrderStatus.REJECTED
            return order
        
        # Set execution price and timestamp
        if order.order_type == OrderType.MARKET:
            price = market_data['ask'] if order.side == OrderSide.BUY else market_data['bid']
        else:  # LIMIT order
            if (order.side == OrderSide.BUY and order.price >= market_data['bid']) or \
               (order.side == OrderSide.SELL and order.price <= market_data['ask']):
                price = order.price
            else:
                order.status = OrderStatus.REJECTED
                return order
        
        # Calculate commission
        commission = float(order.quantity * price * self.commission_rate)
        
        # Update order status
        order.status = OrderStatus.FILLED
        order.avg_price = price
        order.commission = commission
        order.filled_quantity = order.quantity
        order.filled_time = market_data['timestamp']
        
        return order


class BacktestPositionManager:
    """Simple position manager for backtesting."""
    
    def __init__(self, initial_balance: float = 100000.0):
        self.initial_balance = initial_balance
        self.positions = {}
        self.cash = initial_balance
        self.trades = []
    
    def update_position(self, symbol: str, quantity: float, price: float, timestamp: datetime):
        """Update position for a symbol."""
        if symbol not in self.positions:
            self.positions[symbol] = {'quantity': 0.0, 'avg_price': 0.0}
        
        position = self.positions[symbol]
        total_quantity = position['quantity'] + quantity
        
        if total_quantity == 0:
            # Closing position
            position['quantity'] = 0.0
            position['avg_price'] = 0.0
        else:
            # Update average price
            if quantity > 0:  # Adding to position
                position['avg_price'] = (
                    (position['quantity'] * position['avg_price']) + 
                    (quantity * price)
                ) / total_quantity
            position['quantity'] = total_quantity
        
        # Record trade
        self.trades.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'position': position['quantity'],
            'avg_price': position['avg_price']
        })
    
    def get_position(self, symbol: str) -> Dict[str, float]:
        """Get current position for a symbol."""
        return self.positions.get(symbol, {'quantity': 0.0, 'avg_price': 0.0})
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value."""
        total = self.cash
        for symbol, position in self.positions.items():
            if position['quantity'] != 0:
                total += position['quantity'] * current_prices.get(symbol, 0.0)
        return total
