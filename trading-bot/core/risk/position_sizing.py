"""
Advanced position sizing strategies for risk management.
"""
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum, auto


class PositionSizingMethod(Enum):
    FIXED_FRACTIONAL = auto()
    KELLY_CRITERION = auto()
    VOLATILITY_ADJUSTED = auto()
    CORRELATION_ADJUSTED = auto()


@dataclass
class PositionSizeResult:
    size: float  # Position size in lots
    risk_percent: float  # Risk as percentage of capital
    stop_loss: float  # Stop loss price
    take_profit: float  # Take profit price
    risk_reward_ratio: float  # Risk/Reward ratio


class PositionSizer:
    """
    Advanced position sizing with multiple strategies.
    
    Implements:
    - Fixed fractional position sizing
    - Kelly Criterion
    - Volatility-adjusted sizing
    - Correlation-adjusted sizing
    """
    
    def __init__(
        self,
        account_balance: float,
        risk_per_trade: float = 0.01,  # 1% risk per trade
        max_risk_per_trade: float = 0.05,  # Max 5% risk per trade
        max_portfolio_risk: float = 0.2,  # Max 20% portfolio risk
    ):
        self.account_balance = account_balance
        self.risk_per_trade = risk_per_trade
        self.max_risk_per_trade = max_risk_per_trade
        self.max_portfolio_risk = max_portfolio_risk
        self.open_positions: Dict[str, float] = {}
        
    def calculate_position_size(
        self,
        symbol: str,
        entry_price: float,
        stop_loss: float,
        take_profit: Optional[float] = None,
        volatility: Optional[float] = None,
        correlation_matrix: Optional[Dict[str, float]] = None,
        method: PositionSizingMethod = PositionSizingMethod.VOLATILITY_ADJUSTED,
    ) -> PositionSizeResult:
        """
        Calculate position size based on the selected method.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss: Stop loss price
            take_profit: Take profit price (optional)
            volatility: Volatility measure (e.g., ATR, standard deviation)
            correlation_matrix: Correlation matrix with other positions
            method: Position sizing method to use
            
        Returns:
            PositionSizeResult with calculated position size and metrics
        """
        # Calculate risk per share
        risk_per_share = abs(entry_price - stop_loss)
        if risk_per_share == 0:
            raise ValueError("Stop loss cannot be equal to entry price")
            
        # Calculate risk/reward ratio if take profit is provided
        risk_reward = (
            abs(take_profit - entry_price) / risk_per_share
            if take_profit is not None
            else 0
        )
        
        # Base position size (fixed fractional)
        max_risk_amount = self.account_balance * self.risk_per_trade
        base_size = max_risk_amount / risk_per_share
        
        # Apply selected method
        if method == PositionSizingMethod.KELLY_CRITERION:
            size = self._apply_kelly_criterion(base_size, risk_reward)
        elif method == PositionSizingMethod.VOLATILITY_ADJUSTED:
            size = self._adjust_for_volatility(base_size, volatility)
        elif method == PositionSizingMethod.CORRELATION_ADJUSTED:
            size = self._adjust_for_correlation(base_size, symbol, correlation_matrix)
        else:
            size = base_size
            
        # Apply position limits
        size = self._apply_position_limits(size, symbol)
        
        # Calculate risk as percentage of capital
        risk_percent = (size * risk_per_share / self.account_balance) * 100
        
        return PositionSizeResult(
            size=size,
            risk_percent=risk_percent,
            stop_loss=stop_loss,
            take_profit=take_profit or 0,
            risk_reward_ratio=risk_reward
        )
    
    def _apply_kelly_criterion(self, base_size: float, win_probability: float) -> float:
        """Apply Kelly Criterion to position sizing."""
        if win_probability <= 0 or win_probability >= 1:
            return base_size
            
        # Kelly fraction: f* = (bp - q) / b
        # where b is the net odds received on the wager (b = win_amount / loss_amount)
        # p is the probability of winning
        # q is the probability of losing (1 - p)
        b = 2.0  # Assuming 2:1 reward:risk ratio
        p = win_probability
        q = 1 - p
        kelly_fraction = (b * p - q) / b
        
        # Use half-kelly for more conservative sizing
        return base_size * (kelly_fraction * 0.5)
    
    def _adjust_for_volatility(self, size: float, volatility: Optional[float]) -> float:
        """Adjust position size based on market volatility."""
        if volatility is None or volatility == 0:
            return size
            
        # Normalize volatility (assuming 1.0 is average)
        volatility_ratio = 1.0 / (1.0 + volatility)  # Inverse relationship
        return size * volatility_ratio
    
    def _adjust_for_correlation(
        self,
        size: float,
        symbol: str,
        correlation_matrix: Optional[Dict[str, float]]
    ) -> float:
        """Adjust position size based on correlation with existing positions."""
        if not correlation_matrix or not self.open_positions:
            return size
            
        # Calculate net exposure considering correlations
        net_exposure = 0.0
        for pos_symbol, pos_size in self.open_positions.items():
            correlation = correlation_matrix.get((symbol, pos_symbol), 0)
            net_exposure += pos_size * correlation
            
        # Reduce size if positively correlated with existing positions
        if net_exposure > 0:
            size *= max(0, 1 - net_exposure)
            
        return size
    
    def _apply_position_limits(self, size: float, symbol: str) -> float:
        """Apply position limits and round to valid lot size."""
        # Apply max risk per trade
        max_size = (self.account_balance * self.max_risk_per_trade) / \
                  (self.account_balance * self.risk_per_trade)
        size = min(size, max_size)
        
        # Apply portfolio risk limit
        total_risk = sum(
            pos_risk for pos_risk in self.open_positions.values()
        ) + (size * self.risk_per_trade)
        
        if total_risk > (self.account_balance * self.max_portfolio_risk):
            size = 0  # Would exceed portfolio risk limit
            
        # Round to nearest valid lot size (0.01 for most forex pairs)
        return round(size * 100) / 100
    
    def update_position(self, symbol: str, size: float):
        """Update the tracker for open positions."""
        if size == 0:
            self.open_positions.pop(symbol, None)
        else:
            self.open_positions[symbol] = size
    
    def update_balance(self, new_balance: float):
        """Update the account balance."""
        self.account_balance = new_balance
