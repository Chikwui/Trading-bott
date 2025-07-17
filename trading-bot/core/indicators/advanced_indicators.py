"""
Advanced Technical Analysis Indicators

This module provides additional technical indicators beyond the basic ones,
including volume-based indicators, advanced oscillators, and custom indicators.
"""
from typing import Union, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from enum import Enum
from .ta_indicators import IndicatorType, IndicatorResult, _fillna, _ema, _sma

class IchimokuLine(Enum):
    TENKAN = "tenkan"
    KIJUN = "kijun"
    SENKOU_A = "senkou_a"
    SENKOU_B = "senkou_b"
    CHIKOU = "chikou"

class VolumeProfileLevels:
    """Container for Volume Profile levels."""
    def __init__(self, 
                 point_of_control: float,
                 value_area_high: float,
                 value_area_low: float,
                 volume_profile: Dict[float, float]):
        self.point_of_control = point_of_control
        self.value_area_high = value_area_high
        self.value_area_low = value_area_low
        self.volume_profile = volume_profile

def ichimoku_cloud(high: Union[np.ndarray, pd.Series],
                  low: Union[np.ndarray, pd.Series],
                  close: Union[np.ndarray, pd.Series],
                  tenkan_period: int = 9,
                  kijun_period: int = 26,
                  senkou_span_b_period: int = 52,
                  chikou_shift: int = 26,
                  senkou_shift: int = 26) -> Dict[str, IndicatorResult]:
    """
    Ichimoku Kinko Hyo (Ichimoku Cloud) indicator.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        tenkan_period: Period for Tenkan-sen (conversion line)
        kijun_period: Period for Kijun-sen (base line)
        senkou_span_b_period: Period for Senkou Span B
        chikou_shift: Shift for Chikou Span (lagging span)
        senkou_shift: Shift for Senkou Span (leading span)
        
    Returns:
        Dictionary with Ichimoku components
    """
    if isinstance(high, pd.Series):
        high = high.values
    if isinstance(low, pd.Series):
        low = low.values
    if isinstance(close, pd.Series):
        close = close.values
    
    # Tenkan-sen (Conversion Line)
    tenkan_high = pd.Series(high).rolling(window=tenkan_period).max().values
    tenkan_low = pd.Series(low).rolling(window=tenkan_period).min().values
    tenkan_sen = (tenkan_high + tenkan_low) / 2
    
    # Kijun-sen (Base Line)
    kijun_high = pd.Series(high).rolling(window=kijun_period).max().values
    kijun_low = pd.Series(low).rolling(window=kijun_period).min().values
    kijun_sen = (kijun_high + kijun_low) / 2
    
    # Senkou Span A (Leading Span A)
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2)
    
    # Senkou Span B (Leading Span B)
    span_b_high = pd.Series(high).rolling(window=senkou_span_b_period).max().values
    span_b_low = pd.Series(low).rolling(window=senkou_span_b_period).min().values
    senkou_span_b = (span_b_high + span_b_low) / 2
    
    # Shift Senkou Spans forward
    senkou_span_a = np.roll(senkou_span_a, senkou_shift)
    senkou_span_b = np.roll(senkou_span_b, senkou_shift)
    senkou_span_a[:senkou_shift] = np.nan
    senkou_span_b[:senkou_shift] = np.nan
    
    # Chikou Span (Lagging Span)
    chikou_span = np.roll(close, -chikou_shift)
    chikou_span[-chikou_shift:] = np.nan
    
    params = {
        'tenkan_period': tenkan_period,
        'kijun_period': kijun_period,
        'senkou_span_b_period': senkou_span_b_period,
        'chikou_shift': chikou_shift,
        'senkou_shift': senkou_shift
    }
    
    return {
        'tenkan_sen': IndicatorResult(
            values=tenkan_sen,
            name='Ichimoku_Tenkan',
            indicator_type=IndicatorType.TREND,
            params=params
        ),
        'kijun_sen': IndicatorResult(
            values=kijun_sen,
            name='Ichimoku_Kijun',
            indicator_type=IndicatorType.TREND,
            params=params
        ),
        'senkou_span_a': IndicatorResult(
            values=senkou_span_a,
            name='Ichimoku_Senkou_A',
            indicator_type=IndicatorType.TREND,
            params=params
        ),
        'senkou_span_b': IndicatorResult(
            values=senkou_span_b,
            name='Ichimoku_Senkou_B',
            indicator_type=IndicatorType.TREND,
            params=params
        ),
        'chikou_span': IndicatorResult(
            values=chikou_span,
            name='Ichimoku_Chikou',
            indicator_type=IndicatorType.TREND,
            params=params
        )
    }

def parabolic_sar(high: Union[np.ndarray, pd.Series],
                 low: Union[np.ndarray, pd.Series],
                 acceleration: float = 0.02,
                 maximum: float = 0.2) -> IndicatorResult:
    """
    Parabolic Stop and Reverse (SAR) indicator.
    
    Args:
        high: High prices
        low: Low prices
        acceleration: Acceleration factor
        maximum: Maximum acceleration
        
    Returns:
        IndicatorResult with SAR values
    """
    if isinstance(high, pd.Series):
        high = high.values
    if isinstance(low, pd.Series):
        low = low.values
    
    sar = np.full_like(high, np.nan)
    trend = np.ones_like(high)
    af = acceleration
    ep = low[0]
    hp = high[0]
    lp = low[0]
    
    sar[0] = low[0] - (high[0] - low[0]) * 0.2
    
    for i in range(1, len(high)):
        if trend[i-1] < 0:
            # Downtrend
            sar[i] = sar[i-1] - (sar[i-1] - hp) * af
            
            if high[i] > hp:
                hp = high[i]
                af = min(af + acceleration, maximum)
                
            if high[i] > sar[i]:
                trend[i] = 1
                sar[i] = hp
                hp = high[i]
                af = acceleration
        else:
            # Uptrend
            sar[i] = sar[i-1] + (lp - sar[i-1]) * af
            
            if low[i] < lp:
                lp = low[i]
                af = min(af + acceleration, maximum)
                
            if low[i] < sar[i]:
                trend[i] = -1
                sar[i] = lp
                lp = low[i]
                af = acceleration
    
    params = {'acceleration': acceleration, 'maximum': maximum}
    return IndicatorResult(
        values=sar,
        name='Parabolic_SAR',
        indicator_type=IndicatorType.TREND,
        params=params
    )

def adx(high: Union[np.ndarray, pd.Series],
       low: Union[np.ndarray, pd.Series],
       close: Union[np.ndarray, pd.Series],
       window: int = 14) -> Dict[str, IndicatorResult]:
    """
    Average Directional Index (ADX) with +DI and -DI.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        window: Period for calculation
        
    Returns:
        Dictionary with ADX, +DI, and -DI
    """
    if isinstance(high, pd.Series):
        high = high.values
    if isinstance(low, pd.Series):
        low = low.values
    if isinstance(close, pd.Series):
        close = close.values
    
    # Calculate True Range (TR)
    prev_close = np.roll(close, 1)
    prev_close[0] = np.nan
    
    tr1 = high - low
    tr2 = np.abs(high - prev_close)
    tr3 = np.abs(low - prev_close)
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    # Calculate +DM and -DM
    up = high - np.roll(high, 1)
    down = np.roll(low, 1) - low
    
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    
    # Smooth the values
    def wilders_smoothing(series, window):
        result = np.zeros_like(series)
        result[0] = np.nan
        
        for i in range(1, len(series)):
            if i < window:
                result[i] = np.nan
            elif i == window:
                result[i] = np.mean(series[1:window+1])
            else:
                result[i] = (result[i-1] * (window - 1) + series[i]) / window
                
        return result
    
    tr_smooth = wilders_smoothing(tr, window)
    plus_dm_smooth = wilders_smoothing(plus_dm, window)
    minus_dm_smooth = wilders_smoothing(minus_dm, window)
    
    # Calculate +DI and -DI
    plus_di = 100 * (plus_dm_smooth / tr_smooth)
    minus_di = 100 * (minus_dm_smooth / tr_smooth)
    
    # Calculate DX and ADX
    dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di))
    adx_values = _sma(dx, window)
    
    params = {'window': window}
    
    return {
        'adx': IndicatorResult(
            values=adx_values,
            name='ADX',
            indicator_type=IndicatorType.TREND,
            params=params
        ),
        'plus_di': IndicatorResult(
            values=plus_di,
            name='+DI',
            indicator_type=IndicatorType.TREND,
            params=params
        ),
        'minus_di': IndicatorResult(
            values=minus_di,
            name='-DI',
            indicator_type=IndicatorType.TREND,
            params=params
        )
    }

def volume_profile(prices: Union[np.ndarray, pd.Series],
                  volumes: Union[np.ndarray, pd.Series],
                  price_bins: int = 20) -> VolumeProfileLevels:
    """
    Calculate Volume Profile.
    
    Args:
        prices: Price data (typically close or typical price)
        volumes: Volume data
        price_bins: Number of price bins
        
    Returns:
        VolumeProfileLevels object with POC, value area, and volume distribution
    """
    if isinstance(prices, pd.Series):
        prices = prices.values
    if isinstance(volumes, pd.Series):
        volumes = volumes.values
    
    # Calculate price range and create bins
    min_price = np.nanmin(prices)
    max_price = np.nanmax(prices)
    bin_size = (max_price - min_price) / price_bins
    
    if bin_size == 0:  # Handle case where all prices are the same
        return VolumeProfileLevels(
            point_of_control=min_price,
            value_area_high=min_price,
            value_area_low=min_price,
            volume_profile={min_price: np.sum(volumes)}
        )
    
    # Create price bins and calculate volume in each bin
    bins = np.linspace(min_price, max_price, price_bins + 1)
    bin_indices = np.digitize(prices, bins) - 1
    bin_indices = np.clip(bin_indices, 0, price_bins - 1)
    
    volume_dist = {}
    for i in range(price_bins):
        bin_vol = np.sum(volumes[bin_indices == i])
        price_level = (bins[i] + bins[i+1]) / 2
        volume_dist[price_level] = bin_vol
    
    # Find Point of Control (POC) - price level with highest volume
    poc_price = max(volume_dist.items(), key=lambda x: x[1])[0]
    
    # Calculate Value Area (70% of total volume)
    total_volume = sum(volume_dist.values())
    target_volume = total_volume * 0.7
    
    # Sort price levels by volume in descending order
    sorted_levels = sorted(volume_dist.items(), key=lambda x: x[1], reverse=True)
    
    value_volume = 0
    value_prices = []
    
    for price, vol in sorted_levels:
        if value_volume >= target_volume:
            break
        value_volume += vol
        value_prices.append(price)
    
    if value_prices:
        value_area_high = max(value_prices)
        value_area_low = min(value_prices)
    else:
        value_area_high = value_area_low = poc_price
    
    return VolumeProfileLevels(
        point_of_control=poc_price,
        value_area_high=value_area_high,
        value_area_low=value_area_low,
        volume_profile=volume_dist
    )
