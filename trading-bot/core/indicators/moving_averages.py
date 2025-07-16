"""Moving average indicators for technical analysis."""
import numpy as np
import pandas as pd
from typing import Optional, Union, List, Tuple

def sma(series: Union[pd.Series, np.ndarray], window: int) -> np.ndarray:
    """Simple Moving Average (SMA)
    
    Args:
        series: Input data series
        window: Number of periods for the moving average
        
    Returns:
        numpy.ndarray: SMA values
    """
    if isinstance(series, pd.Series):
        return series.rolling(window=window, min_periods=1).mean().values
    return np.convolve(series, np.ones(window)/window, 'same')

def ema(series: Union[pd.Series, np.ndarray], window: int, 
       adjust: bool = False) -> np.ndarray:
    """Exponential Moving Average (EMA)
    
    Args:
        series: Input data series
        window: Number of periods for the moving average
        adjust: Whether to use adjust the weights
        
    Returns:
        numpy.ndarray: EMA values
    """
    if isinstance(series, pd.Series):
        return series.ewm(span=window, adjust=adjust).mean().values
    return pd.Series(series).ewm(span=window, adjust=adjust).mean().values

def dema(series: Union[pd.Series, np.ndarray], window: int) -> np.ndarray:
    """Double Exponential Moving Average (DEMA)
    
    Args:
        series: Input data series
        window: Number of periods for the moving average
        
    Returns:
        numpy.ndarray: DEMA values
    """
    ema1 = ema(series, window)
    ema2 = ema(ema1, window)
    return 2 * ema1 - ema2

def tema(series: Union[pd.Series, np.ndarray], window: int) -> np.ndarray:
    """Triple Exponential Moving Average (TEMA)
    
    Args:
        series: Input data series
        window: Number of periods for the moving average
        
    Returns:
        numpy.ndarray: TEMA values
    """
    ema1 = ema(series, window)
    ema2 = ema(ema1, window)
    ema3 = ema(ema2, window)
    return 3 * (ema1 - ema2) + ema3

def wma(series: Union[pd.Series, np.ndarray], window: int) -> np.ndarray:
    """Weighted Moving Average (WMA)
    
    Args:
        series: Input data series
        window: Number of periods for the moving average
        
    Returns:
        numpy.ndarray: WMA values
    """
    weights = np.arange(1, window + 1)
    if isinstance(series, pd.Series):
        return series.rolling(window=window).apply(
            lambda x: np.sum(weights * x) / weights.sum(), raw=True
        ).values
    return np.convolve(series, weights/weights.sum(), 'same')

def vwap(high: Union[pd.Series, np.ndarray],
         low: Union[pd.Series, np.ndarray],
         close: Union[pd.Series, np.ndarray],
         volume: Union[pd.Series, np.ndarray]) -> np.ndarray:
    """Volume Weighted Average Price (VWAP)
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Trading volume
        
    Returns:
        numpy.ndarray: VWAP values
    """
    typical_price = (np.array(high) + np.array(low) + np.array(close)) / 3
    return np.cumsum(typical_price * volume) / np.cumsum(volume)

def hull_moving_average(series: Union[pd.Series, np.ndarray], 
                       window: int) -> np.ndarray:
    """Hull Moving Average (HMA)
    
    Args:
        series: Input data series
        window: Number of periods for the moving average
        
    Returns:
        numpy.ndarray: HMA values
    """
    half_length = window // 2
    sqrt_length = int(np.sqrt(window))
    
    wma1 = wma(series, half_length)
    wma2 = wma(series, window)
    raw_hma = 2 * wma1 - wma2
    return wma(raw_hma, sqrt_length)

def kama(series: Union[pd.Series, np.ndarray], 
         window: int = 10, 
         fast: int = 2, 
         slow: int = 30) -> np.ndarray:
    """Kaufman's Adaptive Moving Average (KAMA)
    
    Args:
        series: Input data series
        window: Number of periods for efficiency ratio calculation
        fast: Fast EMA constant
        slow: Slow EMA constant
        
    Returns:
        numpy.ndarray: KAMA values
    """
    series = np.asarray(series)
    n = len(series)
    if n < window:
        return np.full(n, np.nan)
    
    # Calculate change and volatility
    change = np.abs(series[window:] - series[:-window])
    volatility = np.abs(np.diff(series, 1))
    
    # Calculate efficiency ratio (ER)
    volatility = np.convolve(volatility, np.ones(window-1), 'valid')
    er = change / np.maximum(volatility, 1e-10)
    er = np.concatenate((np.full(window, np.nan), er))
    
    # Calculate smoothing constant (SC)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[window-1] = series[window-1]
    
    for i in range(window, n):
        kama[i] = kama[i-1] + sc[i] * (series[i] - kama[i-1])
    
    return kama

def zlema(series: Union[pd.Series, np.ndarray], window: int) -> np.ndarray:
    """Zero Lag Exponential Moving Average (ZLEMA)
    
    Args:
        series: Input data series
        window: Number of periods for the moving average
        
    Returns:
        numpy.ndarray: ZLEMA values
    """
    lag = (window - 1) // 2
    shifted = np.roll(series, -lag)
    shifted[-lag:] = series[-1]  # Handle edge case
    return ema(2 * series - shifted, window)

def t3(series: Union[pd.Series, np.ndarray], 
      window: int = 5, 
      volume_factor: float = 0.7) -> np.ndarray:
    """T3 Moving Average
    
    Args:
        series: Input data series
        window: Number of periods for the moving average
        volume_factor: Volume factor (0 < volume_factor < 1)
        
    Returns:
        numpy.ndarray: T3 values
    """
    e1 = ema(series, window)
    e2 = ema(e1, window)
    e3 = ema(e2, window)
    e4 = ema(e3, window)
    e5 = ema(e4, window)
    e6 = ema(e5, window)
    
    c1 = -volume_factor * volume_factor * volume_factor
    c2 = 3 * volume_factor * volume_factor + 3 * volume_factor * volume_factor * volume_factor
    c3 = -6 * volume_factor * volume_factor - 3 * volume_factor - 3 * volume_factor * volume_factor * volume_factor
    c4 = 1 + 3 * volume_factor + volume_factor * volume_factor * volume_factor + 3 * volume_factor * volume_factor
    
    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3
