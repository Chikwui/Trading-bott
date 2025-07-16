"""Volume indicators for technical analysis."""
import numpy as np
import pandas as pd
from typing import Optional, Union, Tuple

def obv(close: Union[pd.Series, np.ndarray],
        volume: Union[pd.Series, np.ndarray]) -> np.ndarray:
    """On-Balance Volume (OBV)
    
    Args:
        close: Close prices
        volume: Trading volume
        
    Returns:
        numpy.ndarray: OBV values
    """
    close = pd.Series(close)
    volume = pd.Series(volume)
    
    # Calculate price direction
    direction = np.sign(close.diff())
    direction[0] = 0  # Set first value to 0
    
    # Calculate OBV
    obv = (direction * volume).cumsum()
    
    return obv.values

def cmf(high: Union[pd.Series, np.ndarray],
        low: Union[pd.Series, np.ndarray],
        close: Union[pd.Series, np.ndarray],
        volume: Union[pd.Series, np.ndarray],
        window: int = 20) -> np.ndarray:
    """Chaikin Money Flow (CMF)
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Trading volume
        window: Lookback period
        
    Returns:
        numpy.ndarray: CMF values (-1 to 1)
    """
    high = np.asarray(high)
    low = np.asarray(low)
    close = np.asarray(close)
    volume = np.asarray(volume)
    
    # Calculate Money Flow Multiplier
    mf_multiplier = ((close - low) - (high - close)) / (high - low + 1e-10)
    
    # Calculate Money Flow Volume
    mf_volume = mf_multiplier * volume
    
    # Calculate CMF
    cmf = pd.Series(mf_volume).rolling(window=window).sum() / \
          pd.Series(volume).rolling(window=window).sum()
    
    return cmf.values

def vwap_band(high: Union[pd.Series, np.ndarray],
              low: Union[pd.Series, np.ndarray],
              close: Union[pd.Series, np.ndarray],
              volume: Union[pd.Series, np.ndarray],
              window: int = 20,
              num_std: float = 2.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Volume Weighted Average Price (VWAP) Bands
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Trading volume
        window: Lookback period
        num_std: Number of standard deviations for the bands
        
    Returns:
        Tuple of (upper, middle, lower) bands
    """
    high = np.asarray(high)
    low = np.asarray(low)
    close = np.asarray(close)
    volume = np.asarray(volume)
    
    # Calculate typical price
    typical_price = (high + low + close) / 3
    
    # Calculate VWAP (middle band)
    cum_tp_vol = (typical_price * volume).cumsum()
    cum_vol = volume.cumsum()
    vwap = cum_tp_vol / cum_vol
    
    # Calculate standard deviation
    sq_diff = (typical_price - vwap) ** 2
    std = np.sqrt((sq_diff * volume).rolling(window=window).sum() / \
                 volume.rolling(window=window).sum())
    
    # Calculate bands
    upper = vwap + (std * num_std)
    lower = vwap - (std * num_std)
    
    return upper.values, vwap.values, lower.values

def vwap_oscillator(close: Union[pd.Series, np.ndarray],
                   vwap: Union[pd.Series, np.ndarray]) -> np.ndarray:
    """VWAP Oscillator
    
    Shows the percentage difference between the close price and VWAP.
    
    Args:
        close: Close prices
        vwap: VWAP values
        
    Returns:
        numpy.ndarray: VWAP Oscillator values as percentage
    """
    close = np.asarray(close)
    vwap = np.asarray(vwap)
    
    return ((close - vwap) / vwap) * 100

def volume_price_trend(close: Union[pd.Series, np.ndarray],
                     volume: Union[pd.Series, np.ndarray]) -> np.ndarray:
    """Volume Price Trend (VPT)
    
    Args:
        close: Close prices
        volume: Trading volume
        
    Returns:
        numpy.ndarray: VPT values
    """
    close = pd.Series(close)
    volume = pd.Series(volume)
    
    # Calculate percentage change in price
    pct_change = close.pct_change()
    
    # Calculate VPT
    vpt = (pct_change * volume).cumsum()
    
    return vpt.values

def money_flow_index(high: Union[pd.Series, np.ndarray],
                    low: Union[pd.Series, np.ndarray],
                    close: Union[pd.Series, np.ndarray],
                    volume: Union[pd.Series, np.ndarray],
                    window: int = 14) -> np.ndarray:
    """Money Flow Index (MFI)
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Trading volume
        window: Lookback period
        
    Returns:
        numpy.ndarray: MFI values (0-100)
    """
    high = np.asarray(high)
    low = np.asarray(low)
    close = np.asarray(close)
    volume = np.asarray(volume)
    
    # Calculate typical price
    typical_price = (high + low + close) / 3
    
    # Calculate money flow
    money_flow = typical_price * volume
    
    # Calculate positive and negative money flow
    positive_flow = np.zeros_like(money_flow)
    negative_flow = np.zeros_like(money_flow)
    
    for i in range(1, len(typical_price)):
        if typical_price[i] > typical_price[i-1]:
            positive_flow[i] = money_flow[i]
        elif typical_price[i] < typical_price[i-1]:
            negative_flow[i] = money_flow[i]
    
    # Calculate money ratio
    positive_flow_sum = pd.Series(positive_flow).rolling(window=window).sum()
    negative_flow_sum = pd.Series(negative_flow).rolling(window=window).sum()
    
    money_ratio = positive_flow_sum / (negative_flow_sum + 1e-10)
    
    # Calculate MFI
    mfi = 100 - (100 / (1 + money_ratio))
    
    return mfi.values

def ease_of_movement(high: Union[pd.Series, np.ndarray],
                    low: Union[pd.Series, np.ndarray],
                    volume: Union[pd.Series, np.ndarray],
                    window: int = 14) -> np.ndarray:
    """Ease of Movement (EMV)
    
    Args:
        high: High prices
        low: Low prices
        volume: Trading volume
        window: Lookback period for smoothing
        
    Returns:
        numpy.ndarray: EMV values
    """
    high = np.asarray(high)
    low = np.asarray(low)
    volume = np.asarray(volume)
    
    # Calculate distance moved
    distance = ((high + low) / 2) - ((np.roll(high, 1) + np.roll(low, 1)) / 2)
    
    # Calculate box ratio
    box_ratio = volume / (high - low + 1e-10)
    
    # Calculate raw EMV
    emv = distance / box_ratio
    
    # Smooth EMV
    emv_smoothed = pd.Series(emv).rolling(window=window).mean()
    
    return emv_smoothed.values

def volume_weighted_average_price(high: Union[pd.Series, np.ndarray],
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
    high = np.asarray(high)
    low = np.asarray(low)
    close = np.asarray(close)
    volume = np.asarray(volume)
    
    # Calculate typical price
    typical_price = (high + low + close) / 3
    
    # Calculate VWAP
    cum_tp_vol = np.cumsum(typical_price * volume)
    cum_vol = np.cumsum(volume)
    vwap = cum_tp_vol / cum_vol
    
    return vwap

def volume_oscillator(volume: Union[pd.Series, np.ndarray],
                     short_window: int = 10,
                     long_window: int = 20) -> np.ndarray:
    """Volume Oscillator
    
    Shows the difference between two volume moving averages as a percentage.
    
    Args:
        volume: Trading volume
        short_window: Short MA period
        long_window: Long MA period
        
    Returns:
        numpy.ndarray: Volume Oscillator values as percentage
    """
    volume = pd.Series(volume)
    
    # Calculate moving averages
    short_ma = volume.rolling(window=short_window).mean()
    long_ma = volume.rolling(window=long_window).mean()
    
    # Calculate oscillator
    vo = ((short_ma - long_ma) / long_ma) * 100
    
    return vo.values

def klinger_oscillator_volume(high: Union[pd.Series, np.ndarray],
                            low: Union[pd.Series, np.ndarray],
                            close: Union[pd.Series, np.ndarray],
                            volume: Union[pd.Series, np.ndarray],
                            short_period: int = 34,
                            long_period: int = 55) -> np.ndarray:
    """Klinger Volume Oscillator (KVO)
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Trading volume
        short_period: Short EMA period
        long_period: Long EMA period
        
    Returns:
        numpy.ndarray: KVO values
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
    
    # Calculate KVO
    short_ema = pd.Series(vf).ewm(span=short_period, adjust=False).mean()
    long_ema = pd.Series(vf).ewm(span=long_period, adjust=False).mean()
    kvo = short_ema - long_ema
    
    return kvo.values
