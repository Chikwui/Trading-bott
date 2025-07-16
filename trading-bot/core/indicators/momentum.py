"""Momentum indicators for technical analysis."""
import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple

def macd(close: Union[pd.Series, np.ndarray],
         fast_period: int = 12,
         slow_period: int = 26,
         signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Moving Average Convergence Divergence (MACD)
    
    Args:
        close: Close prices
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period
        
    Returns:
        Tuple of (macd, signal, hist) where:
        - macd: MACD line (fast_ema - slow_ema)
        - signal: Signal line (EMA of MACD)
        - hist: Histogram (MACD - signal)
    """
    close = pd.Series(close)
    
    # Calculate EMAs
    fast_ema = close.ewm(span=fast_period, adjust=False).mean()
    slow_ema = close.ewm(span=slow_period, adjust=False).mean()
    
    # Calculate MACD and signal line
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    # Calculate histogram
    hist = macd_line - signal_line
    
    return macd_line.values, signal_line.values, hist.values

def ppo(close: Union[pd.Series, np.ndarray],
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9,
        ma_type: str = 'ema') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Percentage Price Oscillator (PPO)
    
    Args:
        close: Close prices
        fast_period: Fast MA period
        slow_period: Slow MA period
        signal_period: Signal line period
        ma_type: Type of moving average ('ema' or 'sma')
        
    Returns:
        Tuple of (ppo, signal, hist) where:
        - ppo: PPO line (fast_ma - slow_ma) / slow_ma * 100
        - signal: Signal line (MA of PPO)
        - hist: Histogram (PPO - signal)
    """
    close = pd.Series(close)
    
    # Calculate MAs
    if ma_type.lower() == 'ema':
        fast_ma = close.ewm(span=fast_period, adjust=False).mean()
        slow_ma = close.ewm(span=slow_period, adjust=False).mean()
    else:  # SMA
        fast_ma = close.rolling(window=fast_period).mean()
        slow_ma = close.rolling(window=slow_period).mean()
    
    # Calculate PPO
    ppo_line = ((fast_ma - slow_ma) / slow_ma) * 100
    
    # Calculate signal line and histogram
    signal_line = ppo_line.ewm(span=signal_period, adjust=False).mean()
    hist = ppo_line - signal_line
    
    return ppo_line.values, signal_line.values, hist.values

def roc(close: Union[pd.Series, np.ndarray], period: int = 10) -> np.ndarray:
    """Rate of Change (ROC)
    
    Args:
        close: Close prices
        period: Lookback period
        
    Returns:
        numpy.ndarray: ROC values
    """
    close = pd.Series(close)
    return close.pct_change(periods=period).mul(100).values

def roc_pct(close: Union[pd.Series, np.ndarray], period: int = 10) -> np.ndarray:
    """Percentage Price Oscillator (ROC %)
    
    This is an alias for the ROC function for backward compatibility.
    
    Args:
        close: Close prices
        period: Lookback period
        
    Returns:
        numpy.ndarray: ROC values as percentage
    """
    return roc(close, period)

def tsi(close: Union[pd.Series, np.ndarray],
        long_period: int = 25,
        short_period: int = 13,
        signal_period: int = 13) -> Tuple[np.ndarray, np.ndarray]:
    """True Strength Index (TSI)
    
    Args:
        close: Close prices
        long_period: Long EMA period
        short_period: Short EMA period
        signal_period: Signal line period
        
    Returns:
        Tuple of (tsi, signal) where tsi is the TSI line and signal is the signal line
    """
    close = pd.Series(close)
    
    # Calculate price changes and absolute price changes
    diff = close.diff()
    abs_diff = diff.abs()
    
    # Calculate EMAs of price changes and absolute price changes
    ema1 = diff.ewm(span=long_period, adjust=False).mean()
    ema2 = ema1.ewm(span=short_period, adjust=False).mean()
    
    ema_abs1 = abs_diff.ewm(span=long_period, adjust=False).mean()
    ema_abs2 = ema_abs1.ewm(span=short_period, adjust=False).mean()
    
    # Calculate TSI
    tsi = 100 * (ema2 / ema_abs2)
    
    # Calculate signal line
    signal = tsi.ewm(span=signal_period, adjust=False).mean()
    
    return tsi.values, signal.values

def kst(close: Union[pd.Series, np.ndarray],
        roc1: int = 10, roc2: int = 15, roc3: int = 20, roc4: int = 30,
        sma1: int = 10, sma2: int = 10, sma3: int = 10, sma4: int = 15,
        signal_period: int = 9) -> Tuple[np.ndarray, np.ndarray]:
    """Know Sure Thing (KST)
    
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
        signal_period: Signal line period
        
    Returns:
        Tuple of (kst, signal) where kst is the KST line and signal is the signal line
    """
    close = pd.Series(close)
    
    # Calculate ROC for each period
    roc1_vals = close.pct_change(roc1)
    roc2_vals = close.pct_change(roc2)
    roc3_vals = close.pct_change(roc3)
    roc4_vals = close.pct_change(roc4)
    
    # Smooth ROC with SMA
    roc1_sma = roc1_vals.rolling(window=sma1).mean()
    roc2_sma = roc2_vals.rolling(window=sma2).mean()
    roc3_sma = roc3_vals.rolling(window=sma3).mean()
    roc4_sma = roc4_vals.rolling(window=sma4).mean()
    
    # Calculate KST
    kst = 100 * (roc1_sma + 2 * roc2_sma + 3 * roc3_sma + 4 * roc4_sma)
    
    # Calculate signal line
    signal = kst.rolling(window=signal_period).mean()
    
    return kst.values, signal.values

def stoch_oscillator(high: Union[pd.Series, np.ndarray],
                    low: Union[pd.Series, np.ndarray],
                    close: Union[pd.Series, np.ndarray],
                    k_period: int = 14,
                    k_slowing: int = 3,
                    d_period: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Stochastic Oscillator
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        k_period: %K period
        k_slowing: %K slowing period
        d_period: %D period (signal line)
        
    Returns:
        Tuple of (k, d) where k is the %K line and d is the %D line
    """
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    
    # Calculate %K
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low + 1e-10))
    
    # Smooth %K
    k = k.rolling(window=k_slowing).mean()
    
    # Calculate %D (signal line)
    d = k.rolling(window=d_period).mean()
    
    return k.values, d.values

def awesome_oscillator_momentum(high: Union[pd.Series, np.ndarray],
                               low: Union[pd.Series, np.ndarray],
                               fast_period: int = 5,
                               slow_period: int = 34) -> np.ndarray:
    """Awesome Oscillator Momentum
    
    This is a momentum-based version of the Awesome Oscillator that shows
    the difference between fast and slow simple moving averages of the median price.
    
    Args:
        high: High prices
        low: Low prices
        fast_period: Fast MA period
        slow_period: Slow MA period
        
    Returns:
        numpy.ndarray: Awesome Oscillator values
    """
    high = pd.Series(high)
    low = pd.Series(low)
    
    # Calculate median price
    median_price = (high + low) / 2
    
    # Calculate fast and slow SMAs
    fast_sma = median_price.rolling(window=fast_period).mean()
    slow_sma = median_price.rolling(window=slow_period).mean()
    
    # Calculate Awesome Oscillator
    ao = fast_sma - slow_sma
    
    return ao.values

def chande_momentum_oscillator(close: Union[pd.Series, np.ndarray],
                              period: int = 14) -> np.ndarray:
    """Chande Momentum Oscillator (CMO)
    
    Args:
        close: Close prices
        period: Lookback period
        
    Returns:
        numpy.ndarray: CMO values (-100 to 100)
    """
    close = pd.Series(close)
    
    # Calculate price changes
    diff = close.diff()
    
    # Calculate sum of up and down moves
    up = diff.where(diff > 0, 0).rolling(window=period).sum()
    down = -diff.where(diff < 0, 0).rolling(window=period).sum()
    
    # Calculate CMO
    cmo = 100 * ((up - down) / (up + down + 1e-10))
    
    return cmo.values

def detrended_price_oscillator_momentum(close: Union[pd.Series, np.ndarray],
                                      period: int = 20) -> np.ndarray:
    """Detrended Price Oscillator (DPO) Momentum
    
    This is a momentum-based version of the DPO that shows the difference
    between the price and a simple moving average shifted back in time.
    
    Args:
        close: Close prices
        period: Lookback period
        
    Returns:
        numpy.ndarray: DPO values
    """
    close = pd.Series(close)
    
    # Calculate SMA and shift it back by (period/2 + 1) periods
    shift_amount = (period // 2) + 1
    sma = close.rolling(window=period).mean().shift(shift_amount)
    
    # Calculate DPO
    dpo = close - sma
    
    return dpo.values
