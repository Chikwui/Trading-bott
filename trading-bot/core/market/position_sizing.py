"""
Advanced position sizing strategies for risk management.
"""
from __future__ import annotations
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from dataclasses import dataclass
from decimal import Decimal
import logging

logger = logging.getLogger(__name__)

@dataclass
class PositionSizingParameters:
    """Parameters for position sizing calculations."""
    account_balance: float
    risk_per_trade: float = 0.01  # 1% risk per trade by default
    max_position_size: float = 0.1  # 10% of account per position
    volatility_period: int = 20  # Lookback period for volatility
    kelly_fraction: float = 0.5  # Fraction of Kelly to use (0.5 = half-Kelly)
    min_win_rate: float = 0.5  # Minimum win rate for Kelly
    max_leverage: float = 3.0  # Maximum allowed leverage

def calculate_kelly_position(
    win_rate: float,
    win_loss_ratio: float,
    parameters: PositionSizingParameters
) -> float:
    """
    Calculate position size using the Kelly Criterion.
    
    Args:
        win_rate: Historical win rate (0-1)
        win_loss_ratio: Average win / average loss
        parameters: Position sizing parameters
        
    Returns:
        Position size as a fraction of account balance
    """
    if win_rate <= 0 or win_rate >= 1:
        raise ValueError("Win rate must be between 0 and 1")
    if win_loss_ratio <= 0:
        raise ValueError("Win/loss ratio must be positive")
    
    # Kelly formula: f* = (bp - q) / b
    # where: b = win/loss ratio, p = win rate, q = 1-p
    b = win_loss_ratio
    p = win_rate
    q = 1 - p
    
    # Calculate full Kelly
    kelly_f = (b * p - q) / b if b != 0 else 0
    
    # Apply fractional Kelly and risk parameters
    fraction = parameters.kelly_fraction * parameters.risk_per_trade
    position_size = kelly_f * fraction
    
    # Apply position limits
    position_size = min(position_size, parameters.max_position_size)
    position_size = max(0, position_size)  # No negative positions
    
    return position_size

def calculate_volatility_adjusted_position(
    symbol: str,
    current_price: float,
    atr: float,
    parameters: PositionSizingParameters,
    volatility_cache: Optional[Dict] = None
) -> Tuple[float, Dict]:
    """
    Calculate position size adjusted for volatility.
    
    Args:
        symbol: Trading symbol
        current_price: Current price of the asset
        atr: Average True Range
        parameters: Position sizing parameters
        volatility_cache: Optional cache for volatility data
        
    Returns:
        Tuple of (position_size, updated_volatility_cache)
    """
    if volatility_cache is None:
        volatility_cache = {}
    
    # Calculate volatility as ATR/price
    volatility = atr / current_price if current_price > 0 else 0
    
    # Update volatility cache
    if symbol not in volatility_cache:
        volatility_cache[symbol] = []
    
    volatility_cache[symbol].append(volatility)
    if len(volatility_cache[symbol]) > parameters.volatility_period:
        volatility_cache[symbol].pop(0)
    
    # Calculate average volatility
    avg_volatility = (
        np.mean(volatility_cache[symbol])
        if volatility_cache[symbol]
        else volatility
    )
    
    # Adjust position size based on volatility
    # Higher volatility = smaller position size
    if avg_volatility > 0:
        # Base position size (from Kelly or fixed fraction)
        base_size = parameters.risk_per_trade
        
        # Adjust for volatility (normalize to target volatility)
        target_volatility = 0.02  # 2% target volatility
        vol_adjustment = min(1.0, target_volatility / max(avg_volatility, 1e-6))
        
        # Apply adjustment with bounds
        position_size = base_size * vol_adjustment
        position_size = min(position_size, parameters.max_position_size)
        position_size = max(0, position_size)
    else:
        position_size = parameters.risk_per_trade
    
    return position_size, volatility_cache

def calculate_max_position_size(
    account_balance: float,
    entry_price: float,
    stop_loss: float,
    risk_percent: float = 1.0
) -> float:
    """
    Calculate maximum position size based on risk parameters.
    
    Args:
        account_balance: Total account balance
        entry_price: Entry price
        stop_loss: Stop loss price
        risk_percent: Maximum risk per trade as percentage of account
        
    Returns:
        Maximum position size in units of the asset
    """
    if entry_price <= 0 or stop_loss <= 0:
        raise ValueError("Prices must be positive")
    if risk_percent <= 0 or risk_percent > 100:
        raise ValueError("Risk percent must be between 0 and 100")
    
    # Calculate risk amount in account currency
    risk_amount = account_balance * (risk_percent / 100.0)
    
    # Calculate position size
    price_risk = abs(entry_price - stop_loss)
    if price_risk <= 0:
        return 0.0
    
    position_size = risk_amount / price_risk
    
    return position_size
