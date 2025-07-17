"""
Technical Analysis Indicators for Trading

This module provides a collection of commonly used technical indicators
for financial market analysis. All indicators are implemented using NumPy
for performance and can work with pandas Series/DataFrames.
"""
from typing import Union, Tuple, List, Optional
import numpy as np
import pandas as pd
from enum import Enum

class IndicatorType(Enum):
    TREND = "trend"
    MOMENTUM = "momentum"
    VOLATILITY = "volatility"
    VOLUME = "volume"
    OTHER = "other"

class MovingAverageType(Enum):
    SMA = "sma"
    EMA = "ema"
    WMA = "wma"
    HMA = "hma"
    DEMA = "dema"
    TEMA = "tema"

class IndicatorResult:
    """Container for indicator results with metadata."""
    
    def __init__(self, 
                 values: Union[np.ndarray, pd.Series], 
                 name: str, 
                 indicator_type: IndicatorType,
                 params: dict,
                 metadata: Optional[dict] = None):
        """
        Initialize IndicatorResult.
        
        Args:
            values: Indicator values
            name: Name of the indicator
            indicator_type: Type of indicator (trend, momentum, etc.)
            params: Parameters used to calculate the indicator
            metadata: Additional metadata
        """
        self.values = values
        self.name = name
        self.indicator_type = indicator_type
        self.params = params
        self.metadata = metadata or {}
    
    def to_series(self, index=None) -> pd.Series:
        """Convert to pandas Series."""
        if isinstance(self.values, pd.Series):
            return self.values
        return pd.Series(self.values, index=index)
    
    def __repr__(self) -> str:
        return f"{self.name}({', '.join(f'{k}={v}' for k, v in self.params.items())})"

def moving_average(series: Union[np.ndarray, pd.Series], 
                  window: int,
                  ma_type: MovingAverageType = MovingAverageType.SMA,
                  fillna: bool = True) -> IndicatorResult:
    """
    Calculate various types of moving averages.
    
    Args:
        series: Input data series
        window: Window size for the moving average
        ma_type: Type of moving average to calculate
        fillna: If True, fill NaN values with the first valid value
        
    Returns:
        IndicatorResult with the moving average values
    """
    if isinstance(series, pd.Series):
        values = series.values
    else:
        values = series
    
    if window <= 0:
        raise ValueError("Window size must be greater than 0")
    
    if len(values) < window:
        return IndicatorResult(
            values=np.full_like(values, np.nan),
            name=f"{ma_type.value.upper()}_{window}",
            indicator_type=IndicatorType.TREND,
            params={"window": window, "ma_type": ma_type.value}
        )
    
    if ma_type == MovingAverageType.SMA:
        result = _sma(values, window)
    elif ma_type == MovingAverageType.EMA:
        result = _ema(values, window)
    elif ma_type == MovingAverageType.WMA:
        result = _wma(values, window)
    elif ma_type == MovingAverageType.HMA:
        result = _hma(values, window)
    elif ma_type == MovingAverageType.DEMA:
        result = _dema(values, window)
    elif ma_type == MovingAverageType.TEMA:
        result = _tema(values, window)
    else:
        raise ValueError(f"Unsupported moving average type: {ma_type}")
    
    if fillna:
        result = _fillna(result)
    
    return IndicatorResult(
        values=result,
        name=f"{ma_type.value.upper()}_{window}",
        indicator_type=IndicatorType.TREND,
        params={"window": window, "ma_type": ma_type.value}
    )

def rsi(series: Union[np.ndarray, pd.Series], 
       window: int = 14, 
       fillna: bool = True) -> IndicatorResult:
    """
    Relative Strength Index (RSI)
    
    Args:
        series: Input data series
        window: Window size for RSI calculation
        fillna: If True, fill NaN values with 50 (neutral RSI)
        
    Returns:
        IndicatorResult with RSI values (0-100)
    """
    if isinstance(series, pd.Series):
        values = series.values
    else:
        values = series
    
    if len(values) < window + 1:
        return IndicatorResult(
            values=np.full_like(values, np.nan),
            name=f"RSI_{window}",
            indicator_type=IndicatorType.MOMENTUM,
            params={"window": window}
        )
    
    deltas = np.diff(values)
    seed = deltas[:window + 1]
    
    # Calculate initial average gain/loss
    up = seed[seed >= 0].sum() / window
    down = -seed[seed < 0].sum() / window
    
    rs = np.zeros_like(values)
    rs[:window] = 100. - 100. / (1. + up / down) if down != 0 else 100.
    
    # Calculate RSI for the rest of the data
    for i in range(window, len(values) - 1):
        delta = deltas[i - 1]
        
        if delta > 0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
        
        up = (up * (window - 1) + upval) / window
        down = (down * (window - 1) + downval) / window
        
        rs[i] = 100. - 100. / (1. + up / down) if down != 0 else 100.
    
    # Add NaN for the first 'window' values
    result = np.concatenate(([np.nan] * window, rs[window:]))
    
    if fillna:
        result = _fillna(result, fill_value=50.0)
    
    return IndicatorResult(
        values=result,
        name=f"RSI_{window}",
        indicator_type=IndicatorType.MOMENTUM,
        params={"window": window}
    )

def macd(series: Union[np.ndarray, pd.Series],
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        fillna: bool = True) -> dict[str, IndicatorResult]:
    """
    Moving Average Convergence Divergence (MACD)
    
    Args:
        series: Input data series
        fast_period: Period for fast EMA
        slow_period: Period for slow EMA
        signal_period: Period for signal line
        fillna: If True, fill NaN values
        
    Returns:
        Dictionary with three IndicatorResults:
        - macd: MACD line (fast_ema - slow_ema)
        - signal: Signal line (EMA of MACD)
        - histogram: MACD - Signal line
    """
    if isinstance(series, pd.Series):
        values = series.values
    else:
        values = series
    
    # Calculate EMAs
    fast_ema = _ema(values, fast_period)
    slow_ema = _ema(values, slow_period)
    
    # Calculate MACD line
    macd_line = fast_ema - slow_ema
    
    # Calculate signal line (EMA of MACD)
    signal_line = _ema(macd_line, signal_period)
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    if fillna:
        macd_line = _fillna(macd_line)
        signal_line = _fillna(signal_line)
        histogram = _fillna(histogram, fill_value=0.0)
    
    params = {
        "fast_period": fast_period,
        "slow_period": slow_period,
        "signal_period": signal_period
    }
    
    return {
        "macd": IndicatorResult(
            values=macd_line,
            name="MACD",
            indicator_type=IndicatorType.MOMENTUM,
            params=params
        ),
        "signal": IndicatorResult(
            values=signal_line,
            name="MACD_Signal",
            indicator_type=IndicatorType.MOMENTUM,
            params=params
        ),
        "histogram": IndicatorResult(
            values=histogram,
            name="MACD_Hist",
            indicator_type=IndicatorType.MOMENTUM,
            params=params
        )
    }

def bollinger_bands(series: Union[np.ndarray, pd.Series],
                   window: int = 20,
                   std_dev: float = 2.0,
                   fillna: bool = True) -> dict[str, IndicatorResult]:
    """
    Bollinger Bands
    
    Args:
        series: Input data series
        window: Window size for moving average and standard deviation
        std_dev: Number of standard deviations for the bands
        fillna: If True, fill NaN values
        
    Returns:
        Dictionary with three IndicatorResults:
        - middle: Middle band (SMA)
        - upper: Upper band (middle + std_dev * std)
        - lower: Lower band (middle - std_dev * std)
    """
    if isinstance(series, pd.Series):
        values = series.values
    else:
        values = series
    
    # Calculate middle band (SMA)
    middle_band = _sma(values, window)
    
    # Calculate standard deviation
    std = _rolling_std(values, window)
    
    # Calculate upper and lower bands
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    
    if fillna:
        middle_band = _fillna(middle_band)
        upper_band = _fillna(upper_band)
        lower_band = _fillna(lower_band)
    
    params = {"window": window, "std_dev": std_dev}
    
    return {
        "middle": IndicatorResult(
            values=middle_band,
            name="BB_Middle",
            indicator_type=IndicatorType.VOLATILITY,
            params=params
        ),
        "upper": IndicatorResult(
            values=upper_band,
            name="BB_Upper",
            indicator_type=IndicatorType.VOLATILITY,
            params=params
        ),
        "lower": IndicatorResult(
            values=lower_band,
            name="BB_Lower",
            indicator_type=IndicatorType.VOLATILITY,
            params=params
        )
    }

def atr(high: Union[np.ndarray, pd.Series],
       low: Union[np.ndarray, pd.Series],
       close: Union[np.ndarray, pd.Series],
       window: int = 14,
       fillna: bool = True) -> IndicatorResult:
    """
    Average True Range (ATR)
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        window: Window size for ATR calculation
        fillna: If True, fill NaN values
        
    Returns:
        IndicatorResult with ATR values
    """
    if isinstance(high, pd.Series):
        high = high.values
    if isinstance(low, pd.Series):
        low = low.values
    if isinstance(close, pd.Series):
        close = close.values
    
    # Calculate True Range
    prev_close = np.roll(close, 1)
    prev_close[0] = np.nan
    
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Calculate ATR
    atr_values = _sma(tr, window)
    
    if fillna:
        atr_values = _fillna(atr_values)
    
    return IndicatorResult(
        values=atr_values,
        name=f"ATR_{window}",
        indicator_type=IndicatorType.VOLATILITY,
        params={"window": window}
    )

def stochastic_oscillator(high: Union[np.ndarray, pd.Series],
                        low: Union[np.ndarray, pd.Series],
                        close: Union[np.ndarray, pd.Series],
                        k_window: int = 14,
                        d_window: int = 3,
                        fillna: bool = True) -> dict[str, IndicatorResult]:
    """
    Stochastic Oscillator
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        k_window: Window size for %K calculation
        d_window: Window size for %D calculation (signal line)
        fillna: If True, fill NaN values
        
    Returns:
        Dictionary with two IndicatorResults:
        - k: %K line
        - d: %D line (signal line)
    """
    if isinstance(high, pd.Series):
        high = high.values
    if isinstance(low, pd.Series):
        low = low.values
    if isinstance(close, pd.Series):
        close = close.values
    
    # Calculate %K
    lowest_low = _rolling_min(low, k_window)
    highest_high = _rolling_max(high, k_window)
    
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    
    # Calculate %D (signal line)
    d = _sma(k, d_window)
    
    if fillna:
        k = _fillna(k, fill_value=50.0)
        d = _fillna(d, fill_value=50.0)
    
    params = {"k_window": k_window, "d_window": d_window}
    
    return {
        "k": IndicatorResult(
            values=k,
            name="Stoch_%K",
            indicator_type=IndicatorType.MOMENTUM,
            params=params
        ),
        "d": IndicatorResult(
            values=d,
            name="Stoch_%D",
            indicator_type=IndicatorType.MOMENTUM,
            params=params
        )
    }

# ====================
# Helper functions
# ====================

def _sma(values: np.ndarray, window: int) -> np.ndarray:
    """Simple Moving Average"""
    if window <= 0 or window > len(values):
        return np.full_like(values, np.nan)
    
    cumsum = np.cumsum(np.insert(values, 0, 0))
    return (cumsum[window:] - cumsum[:-window]) / window

def _ema(values: np.ndarray, window: int) -> np.ndarray:
    """Exponential Moving Average"""
    if window <= 0 or window > len(values):
        return np.full_like(values, np.nan)
    
    alpha = 2.0 / (window + 1.0)
    ema = np.zeros_like(values)
    ema[0] = values[0]
    
    for i in range(1, len(values)):
        ema[i] = (values[i] * alpha) + (ema[i-1] * (1 - alpha))
    
    return ema

def _wma(values: np.ndarray, window: int) -> np.ndarray:
    """Weighted Moving Average"""
    if window <= 0 or window > len(values):
        return np.full_like(values, np.nan)
    
    weights = np.arange(1, window + 1)
    wma = np.convolve(values, weights, 'valid') / weights.sum()
    return np.concatenate(([np.nan] * (window - 1), wma))

def _hma(values: np.ndarray, window: int) -> np.ndarray:
    """Hull Moving Average"""
    if window <= 0 or window > len(values):
        return np.full_like(values, np.nan)
    
    half_window = window // 2
    sqrt_window = int(np.sqrt(window))
    
    wma1 = _wma(values, half_window)
    wma2 = _wma(values, window)
    
    hma = _wma(2 * wma1 - wma2, sqrt_window)
    return hma

def _dema(values: np.ndarray, window: int) -> np.ndarray:
    """Double Exponential Moving Average"""
    ema1 = _ema(values, window)
    ema2 = _ema(ema1, window)
    return 2 * ema1 - ema2

def _tema(values: np.ndarray, window: int) -> np.ndarray:
    """Triple Exponential Moving Average"""
    ema1 = _ema(values, window)
    ema2 = _ema(ema1, window)
    ema3 = _ema(ema2, window)
    return 3 * (ema1 - ema2) + ema3

def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling standard deviation"""
    if window <= 0 or window > len(values):
        return np.full_like(values, np.nan)
    
    result = np.full_like(values, np.nan)
    for i in range(window - 1, len(values)):
        result[i] = np.std(values[i - window + 1:i + 1])
    
    return result

def _rolling_min(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling minimum"""
    if window <= 0 or window > len(values):
        return np.full_like(values, np.nan)
    
    result = np.full_like(values, np.nan)
    for i in range(window - 1, len(values)):
        result[i] = np.min(values[i - window + 1:i + 1])
    
    return result

def _rolling_max(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling maximum"""
    if window <= 0 or window > len(values):
        return np.full_like(values, np.nan)
    
    result = np.full_like(values, np.nan)
    for i in range(window - 1, len(values)):
        result[i] = np.max(values[i - window + 1:i + 1])
    
    return result

def _fillna(values: np.ndarray, fill_value: Optional[float] = None) -> np.ndarray:
    """Fill NaN values with the first valid value"""
    if fill_value is None:
        # Forward fill, then backfill if needed
        mask = np.isnan(values)
        idx = np.where(~mask, np.arange(len(mask)), 0)
        np.maximum.accumulate(idx, axis=0, out=idx)
        result = values[idx]
        
        # Backfill any remaining NaNs at the beginning
        if np.isnan(result[0]):
            mask = np.isnan(result)
            first_valid = np.argmax(~mask)
            if first_valid > 0:
                result[:first_valid] = result[first_valid]
    else:
        # Fill with specified value
        result = np.where(np.isnan(values), fill_value, values)
    
    return result
