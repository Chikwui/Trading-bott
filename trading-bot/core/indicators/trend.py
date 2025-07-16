"""Trend indicators for technical analysis."""
import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple, List

def adx(high: Union[pd.Series, np.ndarray],
        low: Union[pd.Series, np.ndarray],
        close: Union[pd.Series, np.ndarray],
        window: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average Directional Index (ADX)
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        window: Lookback period
        
    Returns:
        Tuple of (plus_di, minus_di, adx) where:
        - plus_di: Plus Directional Indicator
        - minus_di: Minus Directional Indicator
        - adx: Average Directional Index
    """
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    
    # Calculate True Range (TR)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate +DM and -DM
    up = high.diff()
    down = low.diff().abs()
    
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    
    # Smooth the values
    tr_smooth = tr.rolling(window=window).sum()
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=window).sum()
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=window).sum()
    
    # Calculate +DI and -DI
    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    
    # Calculate DX and ADX
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
    adx = dx.rolling(window=window).mean()
    
    return plus_di.values, minus_di.values, adx.values

def adxr(high: Union[pd.Series, np.ndarray],
         low: Union[pd.Series, np.ndarray],
         close: Union[pd.Series, np.ndarray],
         window: int = 14,
         adx_window: int = 14) -> np.ndarray:
    """Average Directional Movement Rating (ADXR)
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        window: Lookback period for ADX
        adx_window: Lookback period for ADXR
        
    Returns:
        numpy.ndarray: ADXR values
    """
    # Get ADX values
    _, _, adx_vals = adx(high, low, close, window)
    
    # Calculate ADXR
    adxr_vals = (adx_vals + np.roll(adx_vals, adx_window)) / 2
    
    return adxr_vals

def aroon(high: Union[pd.Series, np.ndarray],
          low: Union[pd.Series, np.ndarray],
          window: int = 25) -> Tuple[np.ndarray, np.ndarray]:
    """Aroon Indicator
    
    Args:
        high: High prices
        low: Low prices
        window: Lookback period
        
    Returns:
        Tuple of (aroon_up, aroon_down) where:
        - aroon_up: Aroon Up line (0-100)
        - aroon_down: Aroon Down line (0-100)
    """
    high = pd.Series(high)
    low = pd.Series(low)
    
    # Calculate days since highest high and lowest low
    days_since_high = high.rolling(window=window + 1).apply(
        lambda x: window - x.argmax(), raw=True
    )
    days_since_low = low.rolling(window=window + 1).apply(
        lambda x: window - x.argmin(), raw=True
    )
    
    # Calculate Aroon Up and Down
    aroon_up = 100 * (window - days_since_high) / window
    aroon_down = 100 * (window - days_since_low) / window
    
    return aroon_up.values, aroon_down.values

def cci_trend(high: Union[pd.Series, np.ndarray],
              low: Union[pd.Series, np.ndarray],
              close: Union[pd.Series, np.ndarray],
              window: int = 20,
              constant: float = 0.015) -> np.ndarray:
    """Commodity Channel Index (CCI) for Trend Analysis
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        window: Lookback period
        constant: Scaling constant (typically 0.015)
        
    Returns:
        numpy.ndarray: CCI values
    """
    # Use the CCI implementation from oscillators.py
    from .oscillators import cci
    return cci(high, low, close, window, constant)

def dpo(close: Union[pd.Series, np.ndarray], window: int = 20) -> np.ndarray:
    """Detrended Price Oscillator (DPO)
    
    Args:
        close: Close prices
        window: Lookback period
        
    Returns:
        numpy.ndarray: DPO values
    """
    close = pd.Series(close)
    
    # Calculate SMA and shift it back by (window/2 + 1) periods
    shift_amount = (window // 2) + 1
    sma = close.rolling(window=window).mean().shift(shift_amount)
    
    # Calculate DPO
    dpo = close - sma
    
    return dpo.values

def ichimoku_cloud(high: Union[pd.Series, np.ndarray],
                  low: Union[pd.Series, np.ndarray],
                  conversion_period: int = 9,
                  base_period: int = 26,
                  span_b_period: int = 52,
                  displacement: int = 26) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ichimoku Cloud
    
    Args:
        high: High prices
        low: Low prices
        conversion_period: Tenkan-sen period (default: 9)
        base_period: Kijun-sen period (default: 26)
        span_b_period: Senkou Span B period (default: 52)
        displacement: Chikou Span displacement (default: 26)
        
    Returns:
        Tuple of (conversion, base, span_a, span_b, chikou) where:
        - conversion: Tenkan-sen (Conversion Line)
        - base: Kijun-sen (Base Line)
        - span_a: Senkou Span A (Leading Span A)
        - span_b: Senkou Span B (Leading Span B)
        - chikou: Chikou Span (Lagging Span)
    """
    high = pd.Series(high)
    low = pd.Series(low)
    
    # Conversion Line (Tenkan-sen)
    high_conv = high.rolling(window=conversion_period).max()
    low_conv = low.rolling(window=conversion_period).min()
    conversion = (high_conv + low_conv) / 2
    
    # Base Line (Kijun-sen)
    high_base = high.rolling(window=base_period).max()
    low_base = low.rolling(window=base_period).min()
    base = (high_base + low_base) / 2
    
    # Leading Span A (Senkou Span A)
    span_a = ((conversion + base) / 2).shift(base_period)
    
    # Leading Span B (Senkou Span B)
    high_span_b = high.rolling(window=span_b_period).max()
    low_span_b = low.rolling(window=span_b_period).min()
    span_b = ((high_span_b + low_span_b) / 2).shift(base_period)
    
    # Lagging Span (Chikou Span)
    chikou = close.shift(-displacement)
    
    return conversion.values, base.values, span_a.values, span_b.values, chikou.values

def kst_trend(close: Union[pd.Series, np.ndarray],
              roc1: int = 10, roc2: int = 15, roc3: int = 20, roc4: int = 30,
              sma1: int = 10, sma2: int = 10, sma3: int = 10, sma4: int = 15) -> np.ndarray:
    """Know Sure Thing (KST) for Trend Analysis
    
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
        
    Returns:
        numpy.ndarray: KST values
    """
    # Use the KST implementation from oscillators.py
    from .oscillators import kst_oscillator
    kst_vals, _ = kst_oscillator(close, roc1, roc2, roc3, roc4, sma1, sma2, sma3, sma4)
    return kst_vals

def macd_trend(close: Union[pd.Series, np.ndarray],
               fast_period: int = 12,
               slow_period: int = 26,
               signal_period: int = 9) -> np.ndarray:
    """Moving Average Convergence Divergence (MACD) for Trend Analysis
    
    Args:
        close: Close prices
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period
        
    Returns:
        numpy.ndarray: MACD line values
    """
    # Use the MACD implementation from momentum.py
    from .momentum import macd
    macd_line, _, _ = macd(close, fast_period, slow_period, signal_period)
    return macd_line

def mass_index(high: Union[pd.Series, np.ndarray],
               low: Union[pd.Series, np.ndarray],
               window: int = 9,
               window_ema: int = 25) -> np.ndarray:
    """Mass Index
    
    Identifies trend reversals by measuring the narrowing and widening of the range between
    high and low prices.
    
    Args:
        high: High prices
        low: Low prices
        window: Lookback period for high-low range
        window_ema: EMA period for the ratio
        
    Returns:
        numpy.ndarray: Mass Index values
    """
    high = pd.Series(high)
    low = pd.Series(low)
    
    # Calculate high-low range
    high_low_range = high - low
    
    # Calculate 9-period EMA of the range
    ema1 = high_low_range.ewm(span=window, adjust=False).mean()
    
    # Calculate 9-period EMA of ema1
    ema2 = ema1.ewm(span=window, adjust=False).mean()
    
    # Calculate ratio and sum over window_ema periods
    ratio = ema1 / (ema2 + 1e-10)
    mass_index = ratio.rolling(window=window_ema).sum()
    
    return mass_index.values

def parabolic_sar(high: Union[pd.Series, np.ndarray],
                 low: Union[pd.Series, np.ndarray],
                 acceleration: float = 0.02,
                 maximum: float = 0.2) -> np.ndarray:
    """Parabolic SAR (Stop and Reverse)
    
    Args:
        high: High prices
        low: Low prices
        acceleration: Acceleration factor (default: 0.02)
        maximum: Maximum acceleration (default: 0.2)
        
    Returns:
        numpy.ndarray: Parabolic SAR values
    """
    high = np.asarray(high)
    low = np.asarray(low)
    
    # Initialize arrays
    psar = np.zeros_like(high)
    trend = np.ones_like(high)
    af = acceleration
    ep = low[0]
    
    # Initial values
    psar[0] = low[0] - (high[0] - low[0]) * 0.1
    
    for i in range(1, len(high)):
        # Update SAR value
        psar[i] = psar[i-1] + trend[i-1] * af * (ep - psar[i-1])
        
        # Check for trend changes
        if trend[i-1] == 1:
            # Uptrend
            if low[i] < psar[i]:
                trend[i] = -1
                psar[i] = max(high[i-1], high[i-2] if i > 1 else high[i-1])
                ep = low[i]
                af = acceleration
            else:
                trend[i] = 1
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + acceleration, maximum)
                if low[i] < psar[i]:
                    psar[i] = low[i]
        else:
            # Downtrend
            if high[i] > psar[i]:
                trend[i] = 1
                psar[i] = min(low[i-1], low[i-2] if i > 1 else low[i-1])
                ep = high[i]
                af = acceleration
            else:
                trend[i] = -1
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + acceleration, maximum)
                if high[i] > psar[i]:
                    psar[i] = high[i]
    
    return psar

def qstick(open_price: Union[pd.Series, np.ndarray],
           close: Union[pd.Series, np.ndarray],
           window: int = 14) -> np.ndarray:
    """Qstick Indicator
    
    Measures the strength of the trend by comparing the current close to the open.
    
    Args:
        open_price: Open prices
        close: Close prices
        window: Lookback period
        
    Returns:
        numpy.ndarray: Qstick values
    """
    open_price = pd.Series(open_price)
    close = pd.Series(close)
    
    # Calculate close - open
    diff = close - open_price
    
    # Calculate simple moving average of the difference
    qstick = diff.rolling(window=window).mean()
    
    return qstick.values

def vortex(high: Union[pd.Series, np.ndarray],
           low: Union[pd.Series, np.ndarray],
           close: Union[pd.Series, np.ndarray],
           window: int = 14) -> Tuple[np.ndarray, np.ndarray]:
    """Vortex Indicator
    
    Identifies the start of new trends or continuation of trends.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        window: Lookback period
        
    Returns:
        Tuple of (vortex_plus, vortex_minus) where:
        - vortex_plus: Positive trend indicator
        - vortex_minus: Negative trend indicator
    """
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    
    # Calculate VM+ (upward movement)
    vm_plus = high.diff()
    vm_plus[vm_plus < 0] = 0
    
    # Calculate VM- (downward movement)
    vm_minus = low.diff().abs()
    vm_minus[vm_minus < 0] = 0
    
    # Calculate True Range
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Calculate sums over the window
    sum_vm_plus = vm_plus.rolling(window=window).sum()
    sum_vm_minus = vm_minus.rolling(window=window).sum()
    sum_tr = tr.rolling(window=window).sum()
    
    # Calculate Vortex indicators
    vortex_plus = sum_vm_plus / (sum_tr + 1e-10)
    vortex_minus = sum_vm_minus / (sum_tr + 1e-10)
    
    return vortex_plus.values, vortex_minus.values

def trix(close: Union[pd.Series, np.ndarray],
         window: int = 15) -> np.ndarray:
    """TRIX Indicator
    
    Shows the percent rate of change of a triple exponentially smoothed moving average.
    
    Args:
        close: Close prices
        window: Lookback period for the EMA
        
    Returns:
        numpy.ndarray: TRIX values as percentage
    """
    close = pd.Series(close)
    
    # Triple EMA
    ema1 = close.ewm(span=window, adjust=False).mean()
    ema2 = ema1.ewm(span=window, adjust=False).mean()
    ema3 = ema2.ewm(span=window, adjust=False).mean()
    
    # Calculate TRIX as percentage
    trix = 100 * (ema3.diff() / ema3.shift(1))
    
    return trix.values
