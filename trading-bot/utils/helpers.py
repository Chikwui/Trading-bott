"""
Helper functions for the trading bot.
"""
import os
import json
import yaml
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TypeVar, Type
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np

T = TypeVar('T')

def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from JSON or YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing the configuration
        
    Raises:
        ValueError: If the file extension is not .json or .yaml/.yml
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    suffix = path.suffix.lower()
    
    if suffix == '.json':
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif suffix in ('.yaml', '.yml'):
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config file format: {suffix}")

def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """Save configuration to a JSON or YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration file
        
    Raises:
        ValueError: If the file extension is not .json or .yaml/.yml
    """
    path = Path(config_path)
    suffix = path.suffix.lower()
    
    # Create parent directories if they don't exist
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if suffix == '.json':
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
    elif suffix in ('.yaml', '.yml'):
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    else:
        raise ValueError(f"Unsupported config file format: {suffix}")

def generate_id(prefix: str = '', length: int = 8) -> str:
    """Generate a unique ID.
    
    Args:
        prefix: Optional prefix for the ID
        length: Length of the random part of the ID
        
    Returns:
        Generated ID string
    """
    import secrets
    import string
    
    chars = string.ascii_letters + string.digits
    random_part = ''.join(secrets.choice(chars) for _ in range(length))
    return f"{prefix}{random_part}" if prefix else random_part

def format_timestamp(
    dt: Union[datetime, str, float, int], 
    fmt: str = "%Y-%m-%d %H:%M:%S",
    tz: Optional[timezone] = timezone.utc
) -> str:
    """Format a timestamp as a string.
    
    Args:
        dt: Datetime object, ISO format string, or Unix timestamp
        fmt: Format string (default: "%Y-%m-%d %H:%M:%S")
        tz: Timezone to convert to (default: UTC)
        
    Returns:
        Formatted datetime string
    """
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt)
        except ValueError:
            try:
                dt = datetime.utcfromtimestamp(float(dt))
            except (ValueError, TypeError):
                raise ValueError(f"Invalid datetime string: {dt}")
    elif isinstance(dt, (int, float)):
        dt = datetime.utcfromtimestamp(dt)
    
    if not isinstance(dt, datetime):
        raise TypeError(f"Expected datetime, str, or number, got {type(dt)}")
    
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    
    if tz is not None:
        dt = dt.astimezone(tz)
    
    return dt.strftime(fmt)

def parse_timestamp(
    ts: Union[str, float, int], 
    tz: Optional[timezone] = timezone.utc
) -> datetime:
    """Parse a timestamp string or number into a timezone-aware datetime.
    
    Args:
        ts: Timestamp string, Unix timestamp, or datetime object
        tz: Timezone to use if ts is timezone-naive (default: UTC)
        
    Returns:
        Timezone-aware datetime object
    """
    if isinstance(ts, datetime):
        dt = ts
    elif isinstance(ts, (int, float)):
        dt = datetime.utcfromtimestamp(ts)
    else:
        try:
            dt = datetime.fromisoformat(ts)
        except ValueError:
            try:
                dt = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                try:
                    dt = datetime.strptime(ts, "%Y-%m-%d")
                except ValueError as e:
                    raise ValueError(f"Could not parse timestamp: {ts}") from e
    
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    
    if tz is not None and dt.tzinfo != tz:
        dt = dt.astimezone(tz)
    
    return dt

def calculate_pct_change(
    current: float, 
    previous: float, 
    default: float = 0.0
) -> float:
    """Calculate percentage change between two values.
    
    Args:
        current: Current value
        previous: Previous value
        default: Value to return if previous is 0 (default: 0.0)
        
    Returns:
        Percentage change as a decimal (e.g., 0.05 for 5%)
    """
    if previous == 0:
        return default
    return (current - previous) / abs(previous)

def round_to_tick(price: float, tick_size: float) -> float:
    """Round a price to the nearest tick size.
    
    Args:
        price: Price to round
        tick_size: Minimum price movement
        
    Returns:
        Rounded price
    """
    if tick_size <= 0:
        return price
    return round(price / tick_size) * tick_size

def calculate_position_size(
    account_balance: float,
    risk_per_trade: float,
    entry_price: float,
    stop_loss: float,
    risk_reward_ratio: float = 1.0,
    leverage: float = 1.0,
    min_size: float = 0.0,
    max_size: float = float('inf'),
    lot_size: float = 1.0
) -> float:
    """Calculate position size based on risk parameters.
    
    Args:
        account_balance: Total account balance
        risk_per_trade: Percentage of account to risk per trade (0.0 to 1.0)
        entry_price: Entry price
        stop_loss: Stop loss price
        risk_reward_ratio: Risk/reward ratio (default: 1.0)
        leverage: Leverage to apply (default: 1.0)
        min_size: Minimum position size (default: 0.0)
        max_size: Maximum position size (default: no limit)
        lot_size: Lot size for the instrument (default: 1.0)
        
    Returns:
        Position size in units of the instrument
    """
    # Calculate risk amount in account currency
    risk_amount = account_balance * risk_per_trade
    
    # Calculate price difference (absolute risk per unit)
    price_diff = abs(entry_price - stop_loss)
    
    if price_diff == 0:
        return 0.0
    
    # Calculate base position size
    position_size = (risk_amount / price_diff) * leverage
    
    # Apply risk/reward ratio
    if risk_reward_ratio > 0:
        position_size *= risk_reward_ratio
    
    # Round to nearest lot size
    if lot_size > 0:
        position_size = round(position_size / lot_size) * lot_size
    
    # Apply min/max constraints
    position_size = max(min_size, min(position_size, max_size))
    
    return position_size
