"""
Portfolio Management Module

This module provides portfolio tracking and management functionality,
including position tracking, P&L calculation, and performance metrics.
"""
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import logging

from .position import Position

logger = logging.getLogger(__name__)


class Portfolio:
    """
    Tracks the portfolio's value, positions, and performance metrics.
    
    The Portfolio class is responsible for:
    1. Tracking cash and positions
    2. Calculating P&L (realized and unrealized)
    3. Computing performance metrics
    4. Managing portfolio rebalancing
    5. Tracking transaction history
    """
    
    def __init__(
        self, 
        initial_cash: float = 100000.0,
        risk_free_rate: float = 0.0,
        max_leverage: float = 1.0
    ):
        """
        Initialize the portfolio.
        
        Args:
            initial_cash: Initial cash balance
            risk_free_rate: Annual risk-free rate for performance calculations
            max_leverage: Maximum allowed leverage (1.0 = no leverage)
        """
        # Core attributes
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.risk_free_rate = risk_free_rate
        self.max_leverage = max_leverage
        
        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.position_history: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.realized_pnl = 0.0
        self.unrealized_pnl = 0.0
        self.total_commission = 0.0
        self.total_slippage = 0.0
        self.total_fees = 0.0
        
        # Historical data
        self.historical_values = []
        self.daily_returns = []
        self.trade_history = []
        
        # Initialize with starting values
        self._record_portfolio_value()
        
        logger.info(f"Portfolio initialized with ${initial_cash:,.2f} in cash")
    
    @property
    def total_value(self) -> float:
        """Calculate the total portfolio value (cash + positions)."""
        return self.cash + sum(
            pos.quantity * (pos.current_price or 0) 
            for pos in self.positions.values()
        )
    
    @property
    def equity(self) -> float:
        """Calculate the portfolio equity (total value - borrowed funds)."""
        return self.total_value - self.borrowed
    
    @property
    def borrowed(self) -> float:
        """Calculate the amount of borrowed funds (for leverage)."""
        # In a real implementation, this would track borrowed amounts
        return 0.0  # Simplified for now
    
    @property
    def leverage(self) -> float:
        """Calculate the current leverage ratio."""
        equity = self.equity
        if equity <= 0:
            return 0.0
        return (self.total_value - equity) / equity
    
    @property
    def margin_available(self) -> float:
        """Calculate available margin for new positions."""
        if self.leverage >= self.max_leverage:
            return 0.0
        return (self.max_leverage * self.equity) - (self.total_value - self.cash)
    
    def update_positions(self, positions: List[Position], prices: Dict[str, float]):
        """
        Update portfolio positions with current market prices.
        
        Args:
            positions: List of current positions
            prices: Dictionary of symbol -> current price
        """
        # Update positions with current prices
        for position in positions:
            symbol = position.symbol
            if symbol in prices:
                position.update_pnl(prices[symbol])
        
        # Rebuild positions dictionary
        self.positions = {pos.symbol: pos for pos in positions}
        
        # Update unrealized P&L
        self.unrealized_pnl = sum(
            pos.unrealized_pnl for pos in self.positions.values()
        )
        
        # Record portfolio value for performance tracking
        self._record_portfolio_value()
    
    def update_from_buy(
        self, 
        symbol: str, 
        quantity: float, 
        price: float, 
        commission: float = 0.0,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Update portfolio from a buy order fill.
        
        Args:
            symbol: Symbol being bought
            quantity: Number of shares/contracts
            price: Price per share/contract
            commission: Commission/fees for the trade
            timestamp: Timestamp of the trade
            
        Returns:
            Dictionary with trade details
        """
        timestamp = timestamp or datetime.utcnow()
        cost_basis = quantity * price
        total_cost = cost_basis + commission
        
        # Check if we have enough cash
        if self.cash < total_cost:
            raise ValueError("Insufficient cash for purchase")
        
        # Update cash
        self.cash -= total_cost
        self.total_commission += commission
        
        # Update or create position
        if symbol in self.positions:
            position = self.positions[symbol]
            position.quantity += quantity
            position.entry_price = (
                (position.entry_price * (position.quantity - quantity) + cost_basis) / 
                position.quantity
            )
        else:
            position = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=price,
                current_price=price,
                timestamp=timestamp
            )
            self.positions[symbol] = position
        
        # Record trade
        trade = {
            'timestamp': timestamp,
            'symbol': symbol,
            'side': 'BUY',
            'quantity': quantity,
            'price': price,
            'cost': cost_basis,
            'commission': commission,
            'position_size': position.quantity,
            'avg_price': position.entry_price
        }
        self.trade_history.append(trade)
        
        logger.info(
            f"Bought {quantity} {symbol} @ {price:.2f} (cost: ${cost_basis:,.2f}, "
            f"commission: ${commission:,.2f})"
        )
        
        return trade
    
    def update_from_sell(
        self, 
        symbol: str, 
        quantity: float, 
        price: float, 
        commission: float = 0.0,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Update portfolio from a sell order fill.
        
        Args:
            symbol: Symbol being sold
            quantity: Number of shares/contracts
            price: Price per share/contract
            commission: Commission/fees for the trade
            timestamp: Timestamp of the trade
            
        Returns:
            Dictionary with trade details
            
        Raises:
            ValueError: If position doesn't exist or insufficient quantity
        """
        timestamp = timestamp or datetime.utcnow()
        
        # Check if we have the position
        if symbol not in self.positions:
            raise ValueError(f"No position found for {symbol}")
        
        position = self.positions[symbol]
        
        # Check if we have enough to sell
        if quantity > position.quantity:
            raise ValueError(
                f"Insufficient quantity to sell. Have {position.quantity}, "
                f"tried to sell {quantity}"
            )
        
        # Calculate P&L
        proceeds = quantity * price
        cost_basis = quantity * position.entry_price
        pnl = proceeds - cost_basis - commission
        pnl_pct = (pnl / cost_basis) * 100 if cost_basis > 0 else 0.0
        
        # Update cash
        self.cash += proceeds - commission
        self.realized_pnl += pnl
        self.total_commission += commission
        
        # Update position
        position.quantity -= quantity
        
        # Remove position if fully closed
        if position.quantity <= 0.000001:  # Handle floating point precision
            del self.positions[symbol]
            position_closed = True
        else:
            position_closed = False
        
        # Record trade
        trade = {
            'timestamp': timestamp,
            'symbol': symbol,
            'side': 'SELL',
            'quantity': quantity,
            'price': price,
            'proceeds': proceeds,
            'cost_basis': cost_basis,
            'commission': commission,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'position_size': max(0, position.quantity - quantity),
            'position_closed': position_closed
        }
        self.trade_history.append(trade)
        
        # Record position history if closed
        if position_closed:
            self.position_history.append({
                'symbol': symbol,
                'entry_time': position.timestamp,
                'exit_time': timestamp,
                'quantity': position.quantity + quantity,  # Original position size
                'entry_price': position.entry_price,
                'exit_price': price,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'commission': commission,
                'holding_period': (timestamp - position.timestamp).days
            })
        
        logger.info(
            f"Sold {quantity} {symbol} @ {price:.2f} (proceeds: ${proceeds:,.2f}, "
            f"P&L: ${pnl:+,.2f} ({pnl_pc:+,.2f}%), commission: ${commission:,.2f}".format(
                pnl_pc=pnl_pct
            )
        )
        
        return trade
    
    def _record_portfolio_value(self):
        """Record the current portfolio value for performance tracking."""
        now = datetime.utcnow()
        self.historical_values.append({
            'timestamp': now,
            'total_value': self.total_value,
            'cash': self.cash,
            'equity': self.equity,
            'leverage': self.leverage,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl
        })
        
        # Calculate daily return if we have previous data
        if len(self.historical_values) > 1:
            prev_value = self.historical_values[-2]['total_value']
            if prev_value > 0:
                daily_return = (self.total_value / prev_value) - 1
                self.daily_returns.append({
                    'date': now.date(),
                    'return': daily_return
                })
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics for the portfolio.
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.historical_values:
            return {}
        
        # Extract returns
        returns = pd.Series(
            [r['return'] for r in self.daily_returns],
            index=[r['date'] for r in self.daily_returns]
        )
        
        if len(returns) < 2:
            return {}
        
        # Calculate metrics
        total_return = (self.total_value / self.initial_cash) - 1
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        annualized_vol = returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - self.risk_free_rate) / (annualized_vol + 1e-10)
        
        # Calculate max drawdown
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.cummax()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        max_drawdown = drawdowns.min()
        
        # Calculate win rate and profit factor
        wins = [t for t in self.trade_history if t.get('pnl', 0) > 0]
        losses = [t for t in self.trade_history if t.get('pnl', 0) < 0]
        
        win_rate = len(wins) / len(self.trade_history) if self.trade_history else 0
        avg_win = np.mean([t.get('pnl', 0) for t in wins]) if wins else 0
        avg_loss = abs(np.mean([t.get('pnl', 0) for t in losses])) if losses else 0
        profit_factor = (len(wins) * avg_win) / (len(losses) * avg_loss + 1e-10) if losses else float('inf')
        
        return {
            'initial_value': self.initial_cash,
            'current_value': self.total_value,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_vol,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(self.trade_history),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'total_commission': self.total_commission,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'leverage': self.leverage
        }
    
    def get_position_sizes(self) -> Dict[str, float]:
        """
        Get the current position sizes as a percentage of portfolio value.
        
        Returns:
            Dictionary of symbol -> position size (%)
        """
        total = self.total_value
        if total <= 0:
            return {}
            
        return {
            symbol: (pos.quantity * (pos.current_price or 0)) / total
            for symbol, pos in self.positions.items()
        }
    
    def rebalance(self, target_weights: Dict[str, float], prices: Dict[str, float]):
        """
        Rebalance the portfolio to match target weights.
        
        Args:
            target_weights: Dictionary of symbol -> target weight (0-1)
            prices: Current prices for all symbols
            
        Returns:
            List of trades needed to rebalance
        """
        current_value = self.total_value
        current_weights = self.get_position_sizes()
        
        trades = []
        
        for symbol, target_weight in target_weights.items():
            if symbol not in prices:
                logger.warning(f"No price available for {symbol}, skipping")
                continue
                
            target_value = current_value * target_weight
            current_pos = self.positions.get(symbol)
            current_value_pos = current_pos.quantity * prices[symbol] if current_pos else 0
            
            # Calculate delta
            delta_value = target_value - current_value_pos
            
            if abs(delta_value) < 1.0:  # Ignore very small deltas
                continue
                
            # Calculate quantity to trade
            price = prices[symbol]
            quantity = delta_value / price
            
            # Round to whole shares/contracts
            quantity = int(quantity) if quantity > 1 else round(quantity, 6)
            
            if quantity == 0:
                continue
                
            # Execute trade
            try:
                if quantity > 0:
                    trade = self.update_from_buy(
                        symbol=symbol,
                        quantity=quantity,
                        price=price,
                        timestamp=datetime.utcnow()
                    )
                else:
                    trade = self.update_from_sell(
                        symbol=symbol,
                        quantity=-quantity,
                        price=price,
                        timestamp=datetime.utcnow()
                    )
                trades.append(trade)
                
            except Exception as e:
                logger.error(f"Failed to rebalance {symbol}: {str(e)}")
        
        return trades
