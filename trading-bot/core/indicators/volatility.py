"""Volatility indicators for technical analysis."""
import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple

def atr(high: Union[pd.Series, np.ndarray],
        low: Union[pd.Series, np.ndarray],
        close: Union[pd.Series, np.ndarray],
        window: int = 14) -> np.ndarray:
    """Average True Range (ATR)
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        window: Lookback period
        
    Returns:
        numpy.ndarray: ATR values
    """
    high = np.asarray(high)
    low = np.asarray(low)
    close = np.asarray(close)
    
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Calculate ATR
    return pd.Series(tr).rolling(window=window).mean().values

def bollinger_bands(close: Union[pd.Series, np.ndarray],
                   window: int = 20,
                   num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands
    
    Args:
        close: Close prices
        window: Lookback period
        num_std: Number of standard deviations for the bands
        
    Returns:
        Tuple of (upper, middle, lower) bands
    """
    close = pd.Series(close)
    middle = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    
    return upper.values, middle.values, lower.values

def keltner_channels(high: Union[pd.Series, np.ndarray],
                    low: Union[pd.Series, np.ndarray],
                    close: Union[pd.Series, np.ndarray],
                    window: int = 20,
                    atr_window: int = 10,
                    multiplier: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keltner Channels
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        window: Lookback period for EMA
        atr_window: Lookback period for ATR
        multiplier: ATR multiplier for the bands
        
    Returns:
        Tuple of (upper, middle, lower) bands
    """
    close = pd.Series(close)
    middle = close.ewm(span=window, adjust=False).mean()
    
    # Calculate ATR
    tr = np.maximum(
        np.maximum(
            high - low,
            np.abs(high - np.roll(close, 1))
        ),
        np.abs(low - np.roll(close, 1))
    )
    atr = pd.Series(tr).rolling(window=atr_window).mean()
    
    upper = middle + (atr * multiplier)
    lower = middle - (atr * multiplier)
    
    return upper.values, middle.values, lower.values

def donchian_channels(high: Union[pd.Series, np.ndarray],
                     low: Union[pd.Series, np.ndarray],
                     window: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Donchian Channels
    
    Args:
        high: High prices
        low: Low prices
        window: Lookback period
        
    Returns:
        Tuple of (upper, middle, lower) bands
    """
    upper = pd.Series(high).rolling(window=window).max()
    lower = pd.Series(low).rolling(window=window).min()
    middle = (upper + lower) / 2
    
    return upper.values, middle.values, lower.values

def average_true_range_pct(high: Union[pd.Series, np.ndarray],
                          low: Union[pd.Series, np.ndarray],
                          close: Union[pd.Series, np.ndarray],
                          window: int = 14) -> np.ndarray:
    """Average True Range Percentage (ATRp)
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        window: Lookback period
        
    Returns:
        numpy.ndarray: ATRp values (as percentage)
    """
    atr_vals = atr(high, low, close, window)
    close = np.asarray(close)
    return (atr_vals / close) * 100

def chaikin_volatility(high: Union[pd.Series, np.ndarray],
                      low: Union[pd.Series, np.ndarray],
                      window: int = 10,
                      ema_window: int = 10) -> np.ndarray:
    """Chaikin Volatility
    
    Args:
        high: High prices
        low: Low prices
        window: Lookback period for high-low range
        ema_window: EMA smoothing period
        
    Returns:
        numpy.ndarray: Chaikin Volatility values
    """
    high = np.asarray(high)
    low = np.asarray(low)
    
    # Calculate high-low range
    hl_range = high - low
    
    # Calculate EMA of the range
    ema_range = pd.Series(hl_range).ewm(span=ema_window, adjust=False).mean()
    
    # Calculate percentage change over the window
    return (ema_range - ema_range.shift(window)) / ema_range.shift(window) * 100

def historical_volatility(close: Union[pd.Series, np.ndarray],
                         window: int = 20,
                         annualize: bool = True) -> np.ndarray:
    """Historical Volatility
    
    Args:
        close: Close prices
        window: Lookback period in days
        annualize: Whether to annualize the volatility
        
    Returns:
        numpy.ndarray: Historical volatility values
    """
    close = pd.Series(close)
    returns = np.log(close / close.shift(1))
    vol = returns.rolling(window=window).std() * np.sqrt(252 if annualize else 1) * 100
    return vol.values

def ulcer_index(close: Union[pd.Series, np.ndarray],
                window: int = 14) -> np.ndarray:
    """Ulcer Index
    
    Measures downside volatility over a lookback period.
    
    Args:
        close: Close prices
        window: Lookback period
        
    Returns:
        numpy.ndarray: Ulcer Index values
    """
    close = pd.Series(close)
    max_close = close.rolling(window=window, min_periods=1).max()
    drawdown = 100 * (close - max_close) / max_close
    return np.sqrt((drawdown ** 2).rolling(window=window).mean()).values
