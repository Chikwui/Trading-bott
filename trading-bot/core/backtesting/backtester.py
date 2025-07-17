"""
Backtesting Framework - Core Backtester

This module provides the core backtesting functionality for evaluating trading strategies.
"""
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import gridspec
import seaborn as sns
import pyfolio as pf
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

from ..strategies.base_strategy import BaseStrategy, SignalType, PositionState

class BacktestResult:
    """Container for backtest results and performance metrics."""
    
    def __init__(self):
        self.signals = []
        self.positions = []
        self.trades = []
        self.equity_curve = None
        self.metrics = {}
        self.trade_analysis = {}
        self.daily_returns = None
        self.monthly_returns = None
        self.yearly_returns = None
        self.drawdown = None
        self.drawdown_duration = None

class Backtester:
    """
    Backtesting engine for trading strategies.
    
    This class handles the backtesting of trading strategies on historical data.
    It supports multiple assets, various performance metrics, and visualization of results.
    """
    
    def __init__(self, initial_capital=100000.0, commission=0.001, slippage=0.0005, 
                 data_handler=None, strategy=None, risk_free_rate=0.0):
        """
        Initialize the backtester.
        
        Args:
            initial_capital: Initial capital for backtesting
            commission: Commission per trade (as a percentage of trade value)
            slippage: Slippage per trade (as a percentage of trade value)
            data_handler: Data handler for fetching historical data
            strategy: Trading strategy to backtest
            risk_free_rate: Risk-free rate for calculating risk-adjusted returns
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.data_handler = data_handler
        self.strategy = strategy
        self.risk_free_rate = risk_free_rate
        self.results = BacktestResult()
        self.current_positions = {}
        self.current_holdings = {'cash': initial_capital, 'total': initial_capital}
        self.trade_count = 0
        self.bar_count = 0
        
    def set_strategy(self, strategy):
        """Set the trading strategy."""
        self.strategy = strategy
    
    def set_data_handler(self, data_handler):
        """Set the data handler."""
        self.data_handler = data_handler
    
    def _initialize_backtest(self):
        """Initialize the backtest."""
        if self.strategy is None:
            raise ValueError("No strategy set for backtesting")
        
        if self.data_handler is None:
            raise ValueError("No data handler set for backtesting")
        
        # Reset state
        self.results = BacktestResult()
        self.current_positions = {}
        self.current_holdings = {'cash': self.initial_capital, 'total': self.initial_capital}
        self.trade_count = 0
        self.bar_count = 0
        
        # Initialize strategy
        self.strategy.initialize(self.data_handler.get_latest_bars())
    
    def _calculate_commission(self, quantity, price):
        """Calculate commission for a trade."""
        return abs(quantity) * price * self.commission
    
    def _calculate_slippage(self, quantity, price):
        """Calculate slippage for a trade."""
        return abs(quantity) * price * self.slippage
    
    def _update_positions(self, symbol, quantity, price, timestamp):
        """Update positions after a trade."""
        cost = quantity * price
        commission = self._calculate_commission(quantity, price)
        slippage = self._calculate_slippage(quantity, price)
        
        # Update cash
        self.current_holdings['cash'] -= (cost + commission + slippage)
        
        # Update positions
        if symbol not in self.current_positions:
            self.current_positions[symbol] = {
                'quantity': 0,
                'avg_price': 0.0,
                'market_value': 0.0,
                'unrealized_pnl': 0.0,
                'realized_pnl': 0.0,
                'entry_time': timestamp,
                'last_updated': timestamp
            }
        
        pos = self.current_positions[symbol]
        old_quantity = pos['quantity']
        old_avg_price = pos['avg_price']
        
        # Calculate new position
        new_quantity = old_quantity + quantity
        
        if new_quantity == 0:
            # Position closed
            pos['realized_pnl'] += (price - old_avg_price) * -quantity
            pos['quantity'] = 0
            pos['avg_price'] = 0.0
            pos['market_value'] = 0.0
            pos['unrealized_pnl'] = 0.0
            pos['exit_time'] = timestamp
        else:
            # Position updated
            if (quantity > 0 and old_quantity >= 0) or (quantity < 0 and old_quantity <= 0):
                # Adding to position in same direction
                pos['avg_price'] = ((old_quantity * old_avg_price) + (quantity * price)) / new_quantity
            else:
                # Partial close or reversing position
                if abs(quantity) < abs(old_quantity):
                    # Partial close
                    pos['realized_pnl'] += (price - old_avg_price) * -quantity
                else:
                    # Reversing position
                    pos['realized_pnl'] += (price - old_avg_price) * old_quantity
                    pos['avg_price'] = price
            
            pos['quantity'] = new_quantity
            pos['market_value'] = new_quantity * price
            pos['unrealized_pnl'] = (price - pos['avg_price']) * new_quantity
        
        pos['last_updated'] = timestamp
        
        # Update total holdings
        self._update_holdings(timestamp)
        
        # Log the trade
        self._log_trade(symbol, quantity, price, timestamp)
    
    def _update_holdings(self, timestamp):
        """Update total portfolio value."""
        total_value = self.current_holdings['cash']
        
        for symbol, pos in self.current_positions.items():
            # Get current market price
            price = self.data_handler.get_latest_bar_value(symbol, 'close')
            if price is None:
                price = pos['avg_price']  # Use entry price if no current price
            
            # Update position value
            pos['market_value'] = pos['quantity'] * price
            pos['unrealized_pnl'] = (price - pos['avg_price']) * pos['quantity']
            
            # Add to total value
            total_value += pos['market_value']
        
        # Update total holdings
        self.current_holdings['total'] = total_value
        
        # Record equity curve
        self.results.equity_curve = pd.concat([
            self.results.equity_curve,
            pd.DataFrame({
                'timestamp': [timestamp],
                'equity': [total_value],
                'cash': [self.current_holdings['cash']],
                'positions': [len([p for p in self.current_positions.values() if p['quantity'] != 0])]
            })
        ], ignore_index=True) if hasattr(self.results, 'equity_curve') and self.results.equity_curve is not None else pd.DataFrame({
            'timestamp': [timestamp],
            'equity': [total_value],
            'cash': [self.current_holdings['cash']],
            'positions': [len([p for p in self.current_positions.values() if p['quantity'] != 0])]
        })
    
    def _log_trade(self, symbol, quantity, price, timestamp):
        """Log a trade in the results."""
        if quantity == 0:
            return
            
        # Check if this is a new trade or an update to an existing position
        is_new_trade = True
        
        if symbol in self.current_positions:
            pos = self.current_positions[symbol]
            if (quantity > 0 and pos['quantity'] < 0) or (quantity < 0 and pos['quantity'] > 0):
                # This is a reversal, close the existing position first
                self._close_position(symbol, price, timestamp)
            elif (quantity > 0 and pos['quantity'] > 0) or (quantity < 0 and pos['quantity'] < 0):
                # Adding to existing position
                is_new_trade = False
        
        if is_new_trade:
            self.trade_count += 1
            
            # Create new trade entry
            trade = {
                'id': self.trade_count,
                'symbol': symbol,
                'side': 'long' if quantity > 0 else 'short',
                'entry_time': timestamp,
                'entry_price': price,
                'quantity': abs(quantity),
                'exit_time': None,
                'exit_price': None,
                'pnl': 0.0,
                'pnl_pct': 0.0,
                'commission': self._calculate_commission(quantity, price),
                'slippage': self._calculate_slippage(quantity, price),
                'status': 'open'
            }
            
            # Add to open trades
            self.results.trades.append(trade)
        else:
            # Update existing trade
            for trade in reversed(self.results.trades):
                if trade['symbol'] == symbol and trade['status'] == 'open':
                    trade['quantity'] = abs(self.current_positions[symbol]['quantity'])
                    break
    
    def _close_position(self, symbol, price, timestamp):
        """Close an existing position."""
        if symbol not in self.current_positions:
            return
            
        pos = self.current_positions[symbol]
        if pos['quantity'] == 0:
            return
            
        # Calculate P&L
        pnl = (price - pos['avg_price']) * pos['quantity']
        pnl_pct = (price / pos['avg_price'] - 1) * (1 if pos['quantity'] > 0 else -1)
        
        # Update trade history
        for trade in reversed(self.results.trades):
            if trade['symbol'] == symbol and trade['status'] == 'open':
                trade['exit_time'] = timestamp
                trade['exit_price'] = price
                trade['pnl'] = pnl
                trade['pnl_pct'] = pnl_pct
                trade['status'] = 'closed'
                trade['duration'] = (timestamp - trade['entry_time']).days
                break
        
        # Close the position
        self._update_positions(symbol, -pos['quantity'], price, timestamp)
    
    def run_backtest(self):
        """Run the backtest."""
        self._initialize_backtest()
        
        # Get the data iterator
        data_iterator = self.data_handler.get_data_iterator()
        
        # Main backtest loop
        for bars in tqdm(data_iterator, desc="Running backtest"):
            self.bar_count += 1
            
            # Update strategy with latest data
            self.strategy.update_bars(bars)
            
            # Generate signals
            signals = self.strategy.generate_signals()
            
            # Process signals
            for signal in signals:
                self._process_signal(signal)
            
            # Update portfolio
            self._update_holdings(bars[0]['timestamp'] if bars else datetime.now())
        
        # Close all open positions at the last price
        self._close_all_positions()
        
        # Calculate performance metrics
        self.results.calculate_metrics()
        
        return self.results
    
    def _process_signal(self, signal):
        """Process a trading signal."""
        if signal is None or signal['signal'] == SignalType.NONE:
            return
        
        symbol = signal.get('symbol', 'UNKNOWN')
        price = signal.get('price', 0.0)
        timestamp = signal.get('timestamp', datetime.now())
        
        # Process the signal based on type
        if signal['signal'] == SignalType.LONG_ENTRY:
            # Calculate position size based on available capital
            position_size = self._calculate_position_size(price)
            self._update_positions(symbol, position_size, price, timestamp)
            
        elif signal['signal'] == SignalType.SHORT_ENTRY:
            # Calculate position size for short
            position_size = -self._calculate_position_size(price)
            self._update_positions(symbol, position_size, price, timestamp)
            
        elif signal['signal'] == SignalType.LONG_EXIT:
            if symbol in self.current_positions and self.current_positions[symbol]['quantity'] > 0:
                self._close_position(symbol, price, timestamp)
                
        elif signal['signal'] == SignalType.SHORT_EXIT:
            if symbol in self.current_positions and self.current_positions[symbol]['quantity'] < 0:
                self._close_position(symbol, price, timestamp)
    
    def _calculate_position_size(self, price):
        """Calculate position size based on available capital and risk parameters."""
        # Simple implementation: use 1% of portfolio per trade
        risk_per_trade = 0.01
        position_value = self.current_holdings['total'] * risk_per_trade
        return int(position_value / price)
    
    def _close_all_positions(self):
        """Close all open positions at the last available price."""
        for symbol, pos in list(self.current_positions.items()):
            if pos['quantity'] != 0:
                price = self.data_handler.get_latest_bar_value(symbol, 'close')
                if price is None:
                    price = pos['avg_price']
                self._close_position(symbol, price, datetime.now())
    
    def generate_report(self, output_dir='reports'):
        """Generate a comprehensive backtest report."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Generate report using the result's method
        return self.results.generate_report(output_dir=output_dir)
