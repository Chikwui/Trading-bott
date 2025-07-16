"""Technical indicators for trading strategies."""
import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Union

# Import indicator functions
from .moving_averages import (
    sma, ema, dema, tema, wma, vwap, hull_moving_average, kama, zlema, t3
)

from .oscillators import (
    rsi, stoch, stoch_rsi, williams_r, cci, awesome_oscillator, klinger_oscillator,
    ultimate_oscillator, choppiness_index, kst_oscillator, detrended_price_oscillator
)

from .volatility import (
    atr, bollinger_bands, keltner_channels, donchian_channels, average_true_range_pct,
    chaikin_volatility, historical_volatility, ulcer_index
)

from .momentum import (
    macd, ppo, roc, roc_pct, tsi, kst, stoch_oscillator, awesome_oscillator_momentum,
    chande_momentum_oscillator, detrended_price_oscillator_momentum
)

from .volume import (
    obv, cmf, vwap_band, vwap_oscillator, volume_price_trend, money_flow_index,
    ease_of_movement, volume_weighted_average_price, volume_oscillator, klinger_oscillator_volume
)

from .trend import (
    adx, adxr, aroon, cci_trend, dpo, ichimoku_cloud, kst_trend, macd_trend,
    mass_index, parabolic_sar, qstick, vortex, trix
)

from .utils import (
    cross_above, cross_below, cross, highest, lowest, change, 
    percent_change, rolling_apply, zscore, rolling_zscore, rolling_rank,
    rolling_quantile, rolling_std, rolling_var, rolling_skew, rolling_kurt
)

# Re-export all indicators for easy access
__all__ = [
    # Moving Averages
    'sma', 'ema', 'dema', 'tema', 'wma', 'vwap', 'hull_moving_average', 'kama', 'zlema', 't3',
    
    # Oscillators
    'rsi', 'stoch', 'stoch_rsi', 'williams_r', 'cci', 'awesome_oscillator', 
    'klinger_oscillator', 'ultimate_oscillator', 'choppiness_index', 'kst_oscillator',
    'detrended_price_oscillator',
    
    # Volatility
    'atr', 'bollinger_bands', 'keltner_channels', 'donchian_channels', 'average_true_range_pct',
    'chaikin_volatility', 'historical_volatility', 'ulcer_index',
    
    # Momentum
    'macd', 'ppo', 'roc', 'roc_pct', 'tsi', 'kst', 'stoch_oscillator', 'awesome_oscillator_momentum',
    'chande_momentum_oscillator', 'detrended_price_oscillator_momentum',
    
    # Volume
    'obv', 'cmf', 'vwap_band', 'vwap_oscillator', 'volume_price_trend', 'money_flow_index',
    'ease_of_movement', 'volume_weighted_average_price', 'volume_oscillator', 'klinger_oscillator_volume',
    
    # Trend
    'adx', 'adxr', 'aroon', 'cci_trend', 'dpo', 'ichimoku_cloud', 'kst_trend', 'macd_trend',
    'mass_index', 'parabolic_sar', 'qstick', 'vortex', 'trix',
    
    # Utils
    'cross_above', 'cross_below', 'cross', 'highest', 'lowest', 'change', 
    'percent_change', 'rolling_apply', 'zscore', 'rolling_zscore', 'rolling_rank',
    'rolling_quantile', 'rolling_std', 'rolling_var', 'rolling_skew', 'rolling_kurt'
]
