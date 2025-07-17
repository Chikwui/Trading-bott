"""
Base Strategy Class

This module defines the base class for all trading strategies in the system.
All strategies should inherit from this class and implement the required methods.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from enum import Enum, auto

class PositionState(Enum):
    """Represents the current position state."""
    FLAT = auto()
    LONG = auto()
    SHORT = auto()

class SignalType(Enum):
    """Type of trading signal."""
    NONE = 0
    LONG_ENTRY = 1
    SHORT_ENTRY = -1
    LONG_EXIT = 2
    SHORT_EXIT = -2
    CLOSE_ALL = 3

class SignalStrength(Enum):
    """Strength of a trading signal."""
    WEAK = 0.3
    MODERATE = 0.7
    STRONG = 1.0

class BaseStrategy(ABC):
    """
    Base class for all trading strategies.
    
    This class defines the interface that all strategies must implement.
    Concrete strategies should inherit from this class and override the required methods.
    """
    
    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize the strategy with optional parameters.
        
        Args:
            params: Dictionary of strategy parameters
        """
        self.params = params or {}
        self.position_state = PositionState.FLAT
        self.signals = []
        self.indicators = {}
        self.data = None
        self.initialized = False
        self.position_size = 0.0
        self.entry_price = 0.0
        self.stop_loss = 0.0
        self.take_profit = 0.0
        self.trailing_stop = 0.0
        self.risk_per_trade = 0.01  # Default 1% risk per trade
        self.max_drawdown = 0.2  # 20% max drawdown
        self.leverage = 1.0
        self.commission = 0.0005  # 0.05% commission per trade
        self.slippage = 0.0001    # 0.01% slippage per trade
        
        # Initialize with default parameters if not provided
        self._set_default_params()
        
        # Update with user-provided parameters
        if params:
            self._update_params(params)
    
    def _set_default_params(self):
        """Set default strategy parameters."""
        self.default_params = {
            'risk_per_trade': 0.01,  # 1% risk per trade
            'max_drawdown': 0.2,     # 20% max drawdown
            'leverage': 1.0,         # No leverage by default
            'commission': 0.0005,    # 0.05% commission
            'slippage': 0.0001,      # 0.01% slippage
            'initial_balance': 10000.0,  # $10,000 initial balance
            'position_sizing': 'fixed',  # 'fixed', 'volatility', 'kelly'
            'stop_loss_type': 'atr',     # 'fixed', 'atr', 'percent', 'none'
            'take_profit_type': 'rr',    # 'fixed', 'rr' (risk-reward), 'atr', 'none'
            'risk_reward_ratio': 2.0,    # 2:1 risk-reward ratio
            'trailing_stop': False,      # Enable trailing stop
            'trail_percent': 0.01,       # 1% trailing stop
            'max_position_size': 0.1,    # 10% of portfolio max position size
            'min_position_size': 0.01,   # 1% of portfolio min position size
            'allow_shorting': True,      # Allow short positions
            'allow_leverage': False,     # Allow using leverage
            'max_leverage': 5.0,         # Maximum allowed leverage
            'warmup_period': 50,         # Number of bars needed for indicators to warm up
        }
    
    def _update_params(self, params: Dict):
        """
        Update strategy parameters.
        
        Args:
            params: Dictionary of parameters to update
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            elif key in self.default_params:
                self.default_params[key] = value
        
        # Update default params with any new values
        self.default_params.update(params)
        
        # Ensure risk parameters are within bounds
        self.risk_per_trade = max(0.001, min(0.2, self.risk_per_trade))  # 0.1% to 20%
        self.max_drawdown = max(0.01, min(0.5, self.max_drawdown))  # 1% to 50%
        self.leverage = max(1.0, min(self.default_params.get('max_leverage', 5.0), 
                                    self.leverage if self.default_params.get('allow_leverage', False) else 1.0))
    
    @abstractmethod
    def initialize(self, data: pd.DataFrame):
        """
        Initialize the strategy with historical data.
        
        This method should be called before any other methods.
        It's used to calculate indicators and set up the strategy state.
        
        Args:
            data: DataFrame with OHLCV data
        """
        self.data = data.copy()
        self.initialized = True
    
    @abstractmethod
    def calculate_signals(self, data: pd.DataFrame) -> Dict:
        """
        Calculate trading signals based on the latest data.
        
        Args:
            data: Latest market data (OHLCV)
            
        Returns:
            Dictionary containing signals and other information
        """
        if not self.initialized:
            self.initialize(data)
        
        signals = {
            'signal': SignalType.NONE,
            'strength': SignalStrength.MODERATE,
            'price': data['close'].iloc[-1],
            'timestamp': data.index[-1],
            'indicators': {},
            'metadata': {}
        }
        
        return signals
    
    def update_position(self, position_state: PositionState, price: float, size: float):
        """
        Update the current position state.
        
        Args:
            position_state: New position state
            price: Entry/exit price
            size: Position size (absolute value)
        """
        self.position_state = position_state
        self.position_size = size if position_state != PositionState.FLAT else 0.0
        
        if position_state != PositionState.FLAT:
            self.entry_price = price
            self._update_stop_loss_take_profit(price, position_state)
    
    def _update_stop_loss_take_profit(self, price: float, position: PositionState):
        """
        Update stop loss and take profit levels based on the current position.
        
        Args:
            price: Current price
            position: Current position state
        """
        stop_loss_type = self.default_params.get('stop_loss_type', 'atr')
        take_profit_type = self.default_params.get('take_profit_type', 'rr')
        
        # Calculate stop loss
        if stop_loss_type == 'fixed':
            sl_pct = self.default_params.get('stop_loss_pct', 0.02)  # 2% default
            self.stop_loss = price * (1 - sl_pct) if position == PositionState.LONG else price * (1 + sl_pct)
        elif stop_loss_type == 'atr' and 'atr' in self.indicators:
            atr_multiplier = self.default_params.get('atr_multiplier', 2.0)
            atr_value = self.indicators['atr'].iloc[-1] if isinstance(self.indicators['atr'], pd.Series) else self.indicators['atr'][-1]
            self.stop_loss = price - (atr_multiplier * atr_value) if position == PositionState.LONG else price + (atr_multiplier * atr_value)
        else:  # 'none' or invalid type
            self.stop_loss = 0.0
        
        # Calculate take profit
        if take_profit_type == 'fixed':
            tp_pct = self.default_params.get('take_profit_pct', 0.04)  # 4% default
            self.take_profit = price * (1 + tp_pct) if position == PositionState.LONG else price * (1 - tp_pct)
        elif take_profit_type == 'rr':
            rr_ratio = self.default_params.get('risk_reward_ratio', 2.0)
            if position == PositionState.LONG and self.stop_loss < price:
                risk = price - self.stop_loss
                self.take_profit = price + (risk * rr_ratio)
            elif position == PositionState.SHORT and self.stop_loss > price:
                risk = self.stop_loss - price
                self.take_profit = max(0, price - (risk * rr_ratio))
            else:
                self.take_profit = 0.0
        elif take_profit_type == 'atr' and 'atr' in self.indicators:
            atr_multiplier = self.default_params.get('take_profit_atr_multiplier', 4.0)
            atr_value = self.indicators['atr'].iloc[-1] if isinstance(self.indicators['atr'], pd.Series) else self.indicators['atr'][-1]
            self.take_profit = price + (atr_multiplier * atr_value) if position == PositionState.LONG else price - (atr_multiplier * atr_value)
        else:  # 'none' or invalid type
            self.take_profit = 0.0
        
        # Initialize trailing stop if enabled
        if self.default_params.get('trailing_stop', False):
            trail_pct = self.default_params.get('trail_percent', 0.01)
            self.trailing_stop = price * (1 - trail_pct) if position == PositionState.LONG else price * (1 + trail_pct)
        else:
            self.trailing_stop = 0.0
    
    def calculate_position_size(self, price: float, stop_loss: float, account_balance: float) -> float:
        """
        Calculate position size based on risk parameters.
        
        Args:
            price: Entry price
            stop_loss: Stop loss price
            account_balance: Current account balance
            
        Returns:
            Position size in units of the asset
        """
        if stop_loss == 0 or price == 0:
            return 0.0
        
        position_sizing = self.default_params.get('position_sizing', 'fixed')
        risk_amount = account_balance * self.risk_per_trade
        
        if position_sizing == 'fixed':
            position_size = (risk_amount * self.leverage) / abs(price - stop_loss)
        elif position_sizing == 'volatility' and 'atr' in self.indicators:
            atr_value = self.indicators['atr'].iloc[-1] if isinstance(self.indicators['atr'], pd.Series) else self.indicators['atr'][-1]
            position_size = (risk_amount * self.leverage) / (atr_value * 2)  # 2 ATRs as risk
        elif position_sizing == 'kelly' and 'win_rate' in self.indicators:
            win_rate = self.indicators['win_rate']
            avg_win = self.indicators.get('avg_win', 1.0)
            avg_loss = self.indicators.get('avg_loss', 1.0)
            if avg_loss == 0:
                kelly_f = 0.0
            else:
                win_loss_ratio = avg_win / avg_loss
                kelly_f = win_rate - ((1 - win_rate) / win_loss_ratio)
            position_size = (risk_amount * kelly_f * self.leverage) / abs(price - stop_loss)
        else:  # Default to fixed
            position_size = (risk_amount * self.leverage) / abs(price - stop_loss)
        
        # Apply position size limits
        max_size = account_balance * self.default_params.get('max_position_size', 0.1) * self.leverage / price
        min_size = account_balance * self.default_params.get('min_position_size', 0.01) / price
        
        return max(min(position_size, max_size), min_size)
    
    def update_trailing_stop(self, current_price: float) -> bool:
        """
        Update trailing stop level based on price movement.
        
        Args:
            current_price: Current market price
            
        Returns:
            True if trailing stop was triggered, False otherwise
        """
        if not self.default_params.get('trailing_stop', False) or self.position_state == PositionState.FLAT:
            return False
        
        trail_pct = self.default_params.get('trail_percent', 0.01)
        
        if self.position_state == PositionState.LONG:
            new_trailing_stop = current_price * (1 - trail_pct)
            if new_trailing_stop > self.trailing_stop:
                self.trailing_stop = new_trailing_stop
            return current_price <= self.trailing_stop
        else:  # SHORT
            new_trailing_stop = current_price * (1 + trail_pct)
            if new_trailing_stop < self.trailing_stop or self.trailing_stop == 0:
                self.trailing_stop = new_trailing_stop
            return current_price >= self.trailing_stop
    
    def check_exit_conditions(self, current_price: float) -> SignalType:
        """
        Check if any exit conditions are met.
        
        Args:
            current_price: Current market price
            
        Returns:
            SignalType indicating the exit signal, or NONE if no exit
        """
        if self.position_state == PositionState.FLAT:
            return SignalType.NONE
        
        # Check stop loss
        if ((self.position_state == PositionState.LONG and current_price <= self.stop_loss) or
            (self.position_state == PositionState.SHORT and current_price >= self.stop_loss)):
            return SignalType.LONG_EXIT if self.position_state == PositionState.LONG else SignalType.SHORT_EXIT
        
        # Check take profit
        if ((self.position_state == PositionState.LONG and current_price >= self.take_profit) or
            (self.position_state == PositionState.SHORT and current_price <= self.take_profit)):
            return SignalType.LONG_EXIT if self.position_state == PositionState.LONG else SignalType.SHORT_EXIT
        
        # Check trailing stop
        if self.update_trailing_stop(current_price):
            return SignalType.LONG_EXIT if self.position_state == PositionState.LONG else SignalType.SHORT_EXIT
        
        return SignalType.NONE
    
    def get_indicators(self) -> Dict:
        """
        Get all calculated indicators.
        
        Returns:
            Dictionary of indicator names and values
        """
        return self.indicators
    
    def get_signals(self) -> List[Dict]:
        """
        Get all generated signals.
        
        Returns:
            List of signal dictionaries
        """
        return self.signals
    
    def get_position_state(self) -> PositionState:
        """
        Get the current position state.
        
        Returns:
            Current PositionState
        """
        return self.position_state
    
    def get_risk_metrics(self) -> Dict:
        """
        Get current risk metrics.
        
        Returns:
            Dictionary of risk metrics
        """
        return {
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'trailing_stop': self.trailing_stop,
            'position_size': self.position_size,
            'risk_per_trade': self.risk_per_trade,
            'max_drawdown': self.max_drawdown,
            'leverage': self.leverage
        }
