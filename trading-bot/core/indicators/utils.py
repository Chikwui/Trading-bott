"""Utility functions for technical indicators."""
import numpy as np
import pandas as pd
from typing import Optional, Union, Callable, List, Any, Tuple

def cross_above(series1: Union[pd.Series, np.ndarray],
               series2: Union[pd.Series, np.ndarray, float],
               period: int = 1) -> np.ndarray:
    """Check if series1 crosses above series2.
    
    Args:
        series1: First series
        series2: Second series or scalar value
        period: Number of periods to look back for the cross
        
    Returns:
        Boolean array where True indicates a cross above
    """
    if not isinstance(series1, (pd.Series, pd.DataFrame)):
        series1 = pd.Series(series1)
    if not isinstance(series2, (pd.Series, pd.DataFrame)) and not np.isscalar(series2):
        series2 = pd.Series(series2)
    
    if period == 1:
        return (series1 > series2) & (series1.shift(1) <= series2.shift(1))
    else:
        above = (series1 > series2)
        was_below = (series1.shift(period) <= series2.shift(period))
        return above & was_below

def cross_below(series1: Union[pd.Series, np.ndarray],
               series2: Union[pd.Series, np.ndarray, float],
               period: int = 1) -> np.ndarray:
    """Check if series1 crosses below series2.
    
    Args:
        series1: First series
        series2: Second series or scalar value
        period: Number of periods to look back for the cross
        
    Returns:
        Boolean array where True indicates a cross below
    """
    if not isinstance(series1, (pd.Series, pd.DataFrame)):
        series1 = pd.Series(series1)
    if not isinstance(series2, (pd.Series, pd.DataFrame)) and not np.isscalar(series2):
        series2 = pd.Series(series2)
    
    if period == 1:
        return (series1 < series2) & (series1.shift(1) >= series2.shift(1))
    else:
        below = (series1 < series2)
        was_above = (series1.shift(period) >= series2.shift(period))
        return below & was_above

def cross(series1: Union[pd.Series, np.ndarray],
          series2: Union[pd.Series, np.ndarray, float],
          direction: str = 'both',
          period: int = 1) -> np.ndarray:
    """Check for cross between two series.
    
    Args:
        series1: First series
        series2: Second series or scalar value
        direction: Type of cross to check ('above', 'below', or 'both')
        period: Number of periods to look back for the cross
        
    Returns:
        Boolean array where True indicates a cross in the specified direction
    """
    if direction == 'above':
        return cross_above(series1, series2, period)
    elif direction == 'below':
        return cross_below(series1, series2, period)
    else:  # 'both'
        return cross_above(series1, series2, period) | cross_below(series1, series2, period)

def highest(series: Union[pd.Series, np.ndarray],
            window: int,
            min_periods: Optional[int] = None) -> np.ndarray:
    """Rolling maximum of a series.
    
    Args:
        series: Input series
        window: Rolling window size
        min_periods: Minimum number of observations in window required to have a value
        
    Returns:
        Array with rolling maximum values
    """
    if not isinstance(series, (pd.Series, pd.DataFrame)):
        series = pd.Series(series)
    return series.rolling(window=window, min_periods=min_periods).max().values

def lowest(series: Union[pd.Series, np.ndarray],
           window: int,
           min_periods: Optional[int] = None) -> np.ndarray:
    """Rolling minimum of a series.
    
    Args:
        series: Input series
        window: Rolling window size
        min_periods: Minimum number of observations in window required to have a value
        
    Returns:
        Array with rolling minimum values
    """
    if not isinstance(series, (pd.Series, pd.DataFrame)):
        series = pd.Series(series)
    return series.rolling(window=window, min_periods=min_periods).min().values

def change(series: Union[pd.Series, np.ndarray],
           periods: int = 1) -> np.ndarray:
    """Calculate the change between the current and a prior element.
    
    Args:
        series: Input series
        periods: Number of periods to shift for calculating change
        
    Returns:
        Array with the change values
    """
    if not isinstance(series, (pd.Series, pd.DataFrame)):
        series = pd.Series(series)
    return series.diff(periods=periods).values

def percent_change(series: Union[pd.Series, np.ndarray],
                  periods: int = 1) -> np.ndarray:
    """Calculate percentage change between the current and a prior element.
    
    Args:
        series: Input series
        periods: Number of periods to shift for calculating percentage change
        
    Returns:
        Array with percentage change values (as decimal, e.g., 0.05 for 5%)
    """
    if not isinstance(series, (pd.Series, pd.DataFrame)):
        series = pd.Series(series)
    return series.pct_change(periods=periods).values

def rolling_apply(series: Union[pd.Series, np.ndarray],
                 window: int,
                 func: Callable,
                 min_periods: Optional[int] = None,
                 **kwargs) -> np.ndarray:
    """Apply a custom function over a rolling window.
    
    Args:
        series: Input series
        window: Rolling window size
        func: Function to apply to each window
        min_periods: Minimum number of observations in window required to have a value
        **kwargs: Additional arguments to pass to the function
        
    Returns:
        Array with the function applied to each rolling window
    """
    if not isinstance(series, (pd.Series, pd.DataFrame)):
        series = pd.Series(series)
    
    def wrapped_func(x):
        return func(x, **kwargs)
    
    return series.rolling(window=window, min_periods=min_periods).apply(wrapped_func).values

def zscore(series: Union[pd.Series, np.ndarray],
           window: Optional[int] = None) -> np.ndarray:
    """Calculate Z-Score of a series.
    
    If window is None, calculates the Z-Score of the entire series.
    If window is provided, calculates rolling Z-Score.
    
    Args:
        series: Input series
        window: Rolling window size (None for entire series)
        
    Returns:
        Array with Z-Score values
    """
    if not isinstance(series, (pd.Series, pd.DataFrame)):
        series = pd.Series(series)
    
    if window is None:
        return ((series - series.mean()) / series.std()).values
    else:
        mean = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        return ((series - mean) / std).values

def rolling_zscore(series: Union[pd.Series, np.ndarray],
                  window: int) -> np.ndarray:
    """Calculate rolling Z-Score of a series.
    
    Alias for zscore(series, window).
    
    Args:
        series: Input series
        window: Rolling window size
        
    Returns:
        Array with rolling Z-Score values
    """
    return zscore(series, window)

def rolling_rank(series: Union[pd.Series, np.ndarray],
                window: int,
                pct: bool = True) -> np.ndarray:
    """Calculate rolling rank of a series.
    
    Args:
        series: Input series
        window: Rolling window size
        pct: If True, return percentile rank (0-1). If False, return rank (1 to N).
        
    Returns:
        Array with rolling rank values
    """
    if not isinstance(series, (pd.Series, pd.DataFrame)):
        series = pd.Series(series)
    
    def rank_func(x):
        if pct:
            return x.rank(pct=True).iloc[-1]
        else:
            return x.rank().iloc[-1]
    
    return series.rolling(window=window).apply(rank_func).values

def rolling_quantile(series: Union[pd.Series, np.ndarray],
                    window: int,
                    q: float = 0.5) -> np.ndarray:
    """Calculate rolling quantile of a series.
    
    Args:
        series: Input series
        window: Rolling window size
        q: Quantile to compute (0 <= q <= 1)
        
    Returns:
        Array with rolling quantile values
    """
    if not isinstance(series, (pd.Series, pd.DataFrame)):
        series = pd.Series(series)
    return series.rolling(window=window).quantile(q).values

def rolling_std(series: Union[pd.Series, np.ndarray],
               window: int,
               ddof: int = 1) -> np.ndarray:
    """Calculate rolling standard deviation of a series.
    
    Args:
        series: Input series
        window: Rolling window size
        ddof: Delta degrees of freedom (1 for sample std, 0 for population std)
        
    Returns:
        Array with rolling standard deviation values
    """
    if not isinstance(series, (pd.Series, pd.DataFrame)):
        series = pd.Series(series)
    return series.rolling(window=window).std(ddof=ddof).values

def rolling_var(series: Union[pd.Series, np.ndarray],
               window: int,
               ddof: int = 1) -> np.ndarray:
    """Calculate rolling variance of a series.
    
    Args:
        series: Input series
        window: Rolling window size
        ddof: Delta degrees of freedom (1 for sample var, 0 for population var)
        
    Returns:
        Array with rolling variance values
    """
    if not isinstance(series, (pd.Series, pd.DataFrame)):
        series = pd.Series(series)
    return series.rolling(window=window).var(ddof=ddof).values

def rolling_skew(series: Union[pd.Series, np.ndarray],
                window: int) -> np.ndarray:
    """Calculate rolling skewness of a series.
    
    Args:
        series: Input series
        window: Rolling window size
        
    Returns:
        Array with rolling skewness values
    """
    if not isinstance(series, (pd.Series, pd.DataFrame)):
        series = pd.Series(series)
    return series.rolling(window=window).skew().values

def rolling_kurt(series: Union[pd.Series, np.ndarray],
                window: int) -> np.ndarray:
    """Calculate rolling kurtosis of a series.
    
    Args:
        series: Input series
        window: Rolling window size
        
    Returns:
        Array with rolling kurtosis values
    """
    if not isinstance(series, (pd.Series, pd.DataFrame)):
        series = pd.Series(series)
    return series.rolling(window=window).kurt().values
