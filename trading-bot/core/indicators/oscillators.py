"""Oscillator indicators for technical analysis."""
import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple

def rsi(series: Union[pd.Series, np.ndarray], window: int = 14) -> np.ndarray:
    """Relative Strength Index (RSI)
    
    Args:
        series: Input price series
        window: Lookback period
        
    Returns:
        numpy.ndarray: RSI values (0-100)
    """
    if isinstance(series, pd.Series):
        delta = series.diff()
    else:
        delta = np.diff(series, prepend=np.nan)
    
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=window, min_periods=1).mean()
    avg_loss = pd.Series(loss).rolling(window=window, min_periods=1).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.values

def stoch(high: Union[pd.Series, np.ndarray], 
          low: Union[pd.Series, np.ndarray], 
          close: Union[pd.Series, np.ndarray],
          k_window: int = 14,
          d_window: int = 3,
          smooth_k: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """Stochastic Oscillator
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        k_window: %K lookback period
        d_window: %D signal period
        smooth_k: Smoothing period for %K
        
    Returns:
        Tuple of (K, D) where K is the fast line and D is the slow line
    """
    high = np.asarray(high)
    low = np.asarray(low)
    close = np.asarray(close)
    
    # Calculate %K
    lowest_low = pd.Series(low).rolling(window=k_window).min().values
    highest_high = pd.Series(high).rolling(window=k_window).max().values
    k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    
    # Smooth %K if needed
    if smooth_k > 1:
        k = pd.Series(k).rolling(window=smooth_k).mean().values
    
    # Calculate %D (signal line)
    d = pd.Series(k).rolling(window=d_window).mean().values
    
    return k, d

def stoch_rsi(series: Union[pd.Series, np.ndarray], 
              window: int = 14, 
              k_window: int = 3, 
              d_window: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Stochastic RSI
    
    Args:
        series: Input price series
        window: RSI lookback period
        k_window: %K lookback period
        d_window: %D signal period
        
    Returns:
        Tuple of (K, D) where K is the fast line and D is the slow line
    """
    rsi_vals = rsi(series, window)
    return stoch(rsi_vals, rsi_vals, rsi_vals, k_window, d_window)

def williams_r(high: Union[pd.Series, np.ndarray],
               low: Union[pd.Series, np.ndarray],
               close: Union[pd.Series, np.ndarray],
               window: int = 14) -> np.ndarray:
    """Williams %R
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        window: Lookback period
        
    Returns:
        numpy.ndarray: Williams %R values (-100 to 0)
    """
    highest_high = pd.Series(high).rolling(window=window).max().values
    lowest_low = pd.Series(low).rolling(window=window).min().values
    
    return -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)

def cci(high: Union[pd.Series, np.ndarray],
        low: Union[pd.Series, np.ndarray],
        close: Union[pd.Series, np.ndarray],
        window: int = 20,
        constant: float = 0.015) -> np.ndarray:
    """Commodity Channel Index (CCI)
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        window: Lookback period
        constant: Scaling constant (typically 0.015)
        
    Returns:
        numpy.ndarray: CCI values
    """
    tp = (np.asarray(high) + np.asarray(low) + np.asarray(close)) / 3
    ma = pd.Series(tp).rolling(window=window).mean().values
    md = pd.Series(np.abs(tp - ma)).rolling(window=window).mean().values
    
    return (tp - ma) / (constant * md + 1e-10)

def awesome_oscillator(high: Union[pd.Series, np.ndarray],
                      low: Union[pd.Series, np.ndarray],
                      window1: int = 5,
                      window2: int = 34) -> np.ndarray:
    """Awesome Oscillator
    
    Args:
        high: High prices
        low: Low prices
        window1: Short period
        window2: Long period
        
    Returns:
        numpy.ndarray: Awesome Oscillator values
    """
    median_price = (np.asarray(high) + np.asarray(low)) / 2
    ao = (pd.Series(median_price).rolling(window=window1).mean() - 
          pd.Series(median_price).rolling(window=window2).mean()).values
    return ao

def klinger_oscillator(high: Union[pd.Series, np.ndarray],
                       low: Union[pd.Series, np.ndarray],
                       close: Union[pd.Series, np.ndarray],
                       volume: Union[pd.Series, np.ndarray],
                       short_period: int = 34,
                       long_period: int = 55,
                       signal_period: int = 13) -> Tuple[np.ndarray, np.ndarray]:
    """Klinger Oscillator
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Trading volume
        short_period: Short period
        long_period: Long period
        signal_period: Signal line period
        
    Returns:
        Tuple of (KO, signal) where KO is the Klinger Oscillator
    """
    high = np.asarray(high)
    low = np.asarray(low)
    close = np.asarray(close)
    volume = np.asarray(volume)
    
    # Calculate trend
    trend = np.ones_like(close)
    for i in range(1, len(close)):
        if close[i] + low[i] + high[i] > close[i-1] + low[i-1] + high[i-1]:
            trend[i] = 1
        else:
            trend[i] = -1
    
    # Calculate DM and CM
    dm = high - low
    cm = np.zeros_like(close)
    cm[0] = dm[0]
    
    for i in range(1, len(close)):
        if trend[i] == trend[i-1]:
            cm[i] = cm[i-1] + dm[i]
        else:
            cm[i] = dm[i-1] + dm[i]
    
    # Calculate VF (Volume Force)
    vf = volume * abs(2 * (dm / cm - 1)) * trend * 100
    
    # Calculate KO and signal line
    ko = pd.Series(vf).ewm(span=short_period, adjust=False).mean() - \
         pd.Series(vf).ewm(span=long_period, adjust=False).mean()
    signal = ko.rolling(window=signal_period).mean()
    
    return ko.values, signal.values

def ultimate_oscillator(high: Union[pd.Series, np.ndarray],
                        low: Union[pd.Series, np.ndarray],
                        close: Union[pd.Series, np.ndarray],
                        window1: int = 7,
                        window2: int = 14,
                        window3: int = 28,
                        weight1: float = 4.0,
                        weight2: float = 2.0,
                        weight3: float = 1.0) -> np.ndarray:
    """Ultimate Oscillator
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        window1: First period
        window2: Second period
        window3: Third period
        weight1: Weight for first period
        weight2: Weight for second period
        weight3: Weight for third period
        
    Returns:
        numpy.ndarray: Ultimate Oscillator values (0-100)
    """
    high = np.asarray(high)
    low = np.asarray(low)
    close = np.asarray(close)
    
    # Calculate buying pressure (BP)
    bp = close - np.minimum(low, np.roll(close, 1))
    
    # Calculate true range (TR)
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1))
        )
    )
    
    # Calculate average BP and TR for each period
    avg_bp1 = pd.Series(bp).rolling(window=window1).sum()
    avg_tr1 = pd.Series(tr).rolling(window=window1).sum()
    
    avg_bp2 = pd.Series(bp).rolling(window=window2).sum()
    avg_tr2 = pd.Series(tr).rolling(window=window2).sum()
    
    avg_bp3 = pd.Series(bp).rolling(window=window3).sum()
    avg_tr3 = pd.Series(tr).rolling(window=window3).sum()
    
    # Calculate raw values
    raw1 = 100 * avg_bp1 / avg_tr1
    raw2 = 100 * avg_bp2 / avg_tr2
    raw3 = 100 * avg_bp3 / avg_tr3
    
    # Calculate weighted average
    total_weight = weight1 + weight2 + weight3
    return (weight1 * raw1 + weight2 * raw2 + weight3 * raw3) / total_weight

def choppiness_index(high: Union[pd.Series, np.ndarray],
                     low: Union[pd.Series, np.ndarray],
                     close: Union[pd.Series, np.ndarray],
                     window: int = 14) -> np.ndarray:
    """Choppiness Index
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        window: Lookback period
        
    Returns:
        numpy.ndarray: Choppiness Index values (0-100)
    """
    high = np.asarray(high)
    low = np.asarray(low)
    close = np.asarray(close)
    
    # Calculate True Range (TR) and ATR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).rolling(window=1).mean()  # 1-period ATR
    
    # Calculate highest high and lowest low
    highest_high = pd.Series(high).rolling(window=window).max()
    lowest_low = pd.Series(low).rolling(window=window).min()
    
    # Calculate Choppiness Index
    sum_atr = pd.Series(atr).rolling(window=window).sum()
    ci = 100 * np.log10((sum_atr / (highest_high - lowest_low)) / window)
    
    return ci.values

def kst_oscillator(close: Union[pd.Series, np.ndarray],
                   roc1: int = 10,
                   roc2: int = 15,
                   roc3: int = 20,
                   roc4: int = 30,
                   sma1: int = 10,
                   sma2: int = 10,
                   sma3: int = 10,
                   sma4: int = 15,
                   signal: int = 9) -> Tuple[np.ndarray, np.ndarray]:
    """Know Sure Thing (KST) Oscillator
    
    Args:
        close: Close prices
        roc1: First ROC period
        roc2: Second ROC period
        roc3: Third ROC period
        roc4: Fourth ROC period
        sma1: First SMA period
        sma2: Second SMA period
        sma3: Third SMA period
        sma4: Fourth SMA period
        signal: Signal line period
        
    Returns:
        Tuple of (KST, signal) where KST is the oscillator and signal is the signal line
    """
    close = np.asarray(close)
    
    # Calculate ROC for each period
    roc1_vals = 100 * (close / np.roll(close, roc1) - 1)
    roc2_vals = 100 * (close / np.roll(close, roc2) - 1)
    roc3_vals = 100 * (close / np.roll(close, roc3) - 1)
    roc4_vals = 100 * (close / np.roll(close, roc4) - 1)
    
    # Smooth ROC with SMA
    roc1_sma = pd.Series(roc1_vals).rolling(window=sma1).mean()
    roc2_sma = pd.Series(roc2_vals).rolling(window=sma2).mean()
    roc3_sma = pd.Series(roc3_vals).rolling(window=sma3).mean()
    roc4_sma = pd.Series(roc4_vals).rolling(window=sma4).mean()
    
    # Calculate KST
    kst = roc1_sma + 2 * roc2_sma + 3 * roc3_sma + 4 * roc4_sma
    
    # Calculate signal line
    signal_line = kst.rolling(window=signal).mean()
    
    return kst.values, signal_line.values

def detrended_price_oscillator(close: Union[pd.Series, np.ndarray],
                              window: int = 20) -> np.ndarray:
    """Detrended Price Oscillator (DPO)
    
    Args:
        close: Close prices
        window: Lookback period
        
    Returns:
        numpy.ndarray: DPO values
    """
    close = np.asarray(close)
    shift = (window // 2) + 1
    ma = pd.Series(close).rolling(window=window).mean().shift(shift)
    return close - ma
