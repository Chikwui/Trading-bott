"""
Multi-Timeframe Strategy

This strategy combines signals from multiple timeframes to generate more robust
trading signals. It uses higher timeframes to determine the trend and lower
timeframes for entry and exit timing.
"""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from enum import Enum
from .base_strategy import BaseStrategy, PositionState, SignalType, SignalStrength
from core.indicators.ta_indicators import (
    moving_average, rsi, macd, bollinger_bands, atr, stochastic_oscillator,
    MovingAverageType, IndicatorType
)
from core.indicators.advanced_indicators import (
    ichimoku_cloud, parabolic_sar, adx, volume_profile, VolumeProfileLevels
)

class TimeframeRelation(Enum):
    """Relationship between timeframes."""
    ALIGNED = 1      # All timeframes agree on direction
    NEUTRAL = 0      # Timeframes are mixed or neutral
    CONFLICTED = -1  # Timeframes conflict in direction

class MultiTimeframeStrategy(BaseStrategy):
    """
    Multi-Timeframe Trading Strategy
    
    This strategy uses multiple timeframes to determine the overall trend and 
    generate trading signals. It combines:
    - Higher timeframes (e.g., 4H, 1D) for trend direction
    - Medium timeframes (e.g., 1H) for confirmation
    - Lower timeframes (e.g., 15M, 5M) for precise entries/exits
    
    Parameters:
    -----------
    timeframes : List[str]
        List of timeframes to use, from highest to lowest (e.g., ['4H', '1H', '15M'])
    trend_ma_periods : Dict[str, int]
        Moving average periods for trend detection on each timeframe
    entry_rsi_period : int
        RSI period for entry signals
    exit_rsi_period : int
        RSI period for exit signals
    atr_period : int
        Period for ATR calculation for stop loss and position sizing
    risk_reward_ratio : float
        Risk-reward ratio for take profit levels
    """
    
    def __init__(self, params: Optional[Dict] = None):
        super().__init__(params)
        self.name = "MultiTimeframeStrategy"
        self.timeframes = params.get('timeframes', ['4H', '1H', '15M'])
        self.trend_ma_periods = params.get('trend_ma_periods', {'4H': 50, '1H': 20, '15M': 10})
        self.entry_rsi_period = params.get('entry_rsi_period', 14)
        self.exit_rsi_period = params.get('exit_rsi_period', 5)
        self.atr_period = params.get('atr_period', 14)
        self.risk_reward_ratio = params.get('risk_reward_ratio', 2.0)
        self.trend_strength_threshold = params.get('trend_strength_threshold', 0.7)
        self.confirmation_timeframes = params.get('confirmation_timeframes', 2)  # Number of timeframes needed for confirmation
        
        # Data storage for each timeframe
        self.tf_data = {tf: None for tf in self.timeframes}
        self.tf_indicators = {tf: {} for tf in self.timeframes}
        self.tf_signals = {tf: {'trend': 0, 'momentum': 0, 'volatility': 0} for tf in self.timeframes}
        
        # Initialize with default parameters
        self._set_default_params()
        if params:
            self._update_params(params)
    
    def _set_default_params(self):
        """Set default strategy parameters."""
        super()._set_default_params()
        self.default_params.update({
            'timeframes': ['4H', '1H', '15M'],
            'trend_ma_periods': {'4H': 50, '1H': 20, '15M': 10},
            'entry_rsi_period': 14,
            'exit_rsi_period': 5,
            'atr_period': 14,
            'risk_reward_ratio': 2.0,
            'trend_strength_threshold': 0.7,
            'confirmation_timeframes': 2,
            'use_ichimoku': True,
            'use_adx': True,
            'adx_threshold': 25,
            'max_trend_age': 20,  # Max bars to consider trend valid
            'min_trend_strength': 0.5,
            'max_volatility': 0.02,  # Max ATR/price ratio
            'min_volume_ratio': 0.8,  # Min volume relative to average
        })
    
    def initialize(self, data: pd.DataFrame):
        """
        Initialize the strategy with historical data.
        
        Args:
            data: DataFrame with OHLCV data for the primary timeframe
        """
        super().initialize(data)
        self.primary_tf = self.timeframes[0]  # Highest timeframe is primary
        self.data = data
        
        # Initialize indicators for primary timeframe
        self._calculate_indicators(data, self.primary_tf)
        
        # For now, we assume higher timeframe data is already pre-processed
        # In a real implementation, you would load data for each timeframe
        
        self.initialized = True
    
    def _calculate_indicators(self, data: pd.DataFrame, timeframe: str):
        """Calculate indicators for a specific timeframe."""
        if data is None or len(data) < max(self.trend_ma_periods.values()) + 10:
            return
        
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']
        
        # Store the timeframe data
        self.tf_data[timeframe] = data
        
        # Calculate moving averages for trend
        ma_fast = moving_average(close, window=self.trend_ma_periods[timeframe] // 2, 
                               ma_type=MovingAverageType.EMA)
        ma_slow = moving_average(close, window=self.trend_ma_periods[timeframe], 
                               ma_type=MovingAverageType.EMA)
        
        # Calculate RSI for entry/exit signals
        rsi_entry = rsi(close, window=self.entry_rsi_period)
        rsi_exit = rsi(close, window=self.exit_rsi_period)
        
        # Calculate ATR for volatility and position sizing
        atr_value = atr(high, low, close, window=self.atr_period)
        
        # Calculate MACD for trend confirmation
        macd_line, signal_line, hist = macd(close)
        
        # Store indicators
        self.tf_indicators[timeframe] = {
            'ma_fast': ma_fast,
            'ma_slow': ma_slow,
            'rsi_entry': rsi_entry,
            'rsi_exit': rsi_exit,
            'atr': atr_value,
            'macd_line': macd_line,
            'macd_signal': signal_line,
            'macd_hist': hist
        }
        
        # Advanced indicators if enabled
        if self.default_params.get('use_ichimoku', True):
            ichimoku = ichimoku_cloud(high, low, close)
            self.tf_indicators[timeframe].update(ichimoku)
        
        if self.default_params.get('use_adx', True):
            adx_result = adx(high, low, close)
            self.tf_indicators[timeframe].update(adx_result)
    
    def _analyze_trend(self, timeframe: str) -> Tuple[int, float]:
        """
        Analyze trend for a specific timeframe.
        
        Returns:
            Tuple of (trend_direction, trend_strength)
            - trend_direction: 1 (up), -1 (down), or 0 (neutral)
            - trend_strength: 0.0 to 1.0 indicating trend strength
        """
        if timeframe not in self.tf_indicators or not self.tf_indicators[timeframe]:
            return 0, 0.0
        
        indicators = self.tf_indicators[timeframe]
        close = self.tf_data[timeframe]['close']
        
        # Initialize trend components
        trend_components = []
        strength_components = []
        
        # 1. Moving Average Crossover
        if 'ma_fast' in indicators and 'ma_slow' in indicators:
            ma_fast = indicators['ma_fast'].iloc[-1] if hasattr(indicators['ma_fast'], 'iloc') else indicators['ma_fast'][-1]
            ma_slow = indicators['ma_slow'].iloc[-1] if hasattr(indicators['ma_slow'], 'iloc') else indicators['ma_slow'][-1]
            
            ma_trend = 1 if ma_fast > ma_slow else -1
            ma_strength = abs(ma_fast - ma_slow) / (ma_slow + 1e-10)
            
            trend_components.append(ma_trend)
            strength_components.append(ma_strength)
        
        # 2. MACD
        if 'macd_hist' in indicators:
            macd_hist = indicators['macd_hist']
            if len(macd_hist) >= 2:
                macd_trend = 1 if macd_hist[-1] > 0 and macd_hist[-1] > macd_hist[-2] else -1
                macd_strength = abs(macd_hist[-1])
                
                trend_components.append(macd_trend)
                strength_components.append(macd_strength)
        
        # 3. ADX (if enabled)
        if self.default_params.get('use_adx', True) and 'ADX' in indicators:
            adx_value = indicators['ADX']
            adx_value = adx_value.iloc[-1] if hasattr(adx_value, 'iloc') else adx_value[-1]
            
            if adx_value > self.default_params.get('adx_threshold', 25):
                if '+DI' in indicators and '-DI' in indicators:
                    di_plus = indicators['+DI']
                    di_minus = indicators['-DI']
                    di_plus = di_plus.iloc[-1] if hasattr(di_plus, 'iloc') else di_plus[-1]
                    di_minus = di_minus.iloc[-1] if hasattr(di_minus, 'iloc') else di_minus[-1]
                    
                    adx_trend = 1 if di_plus > di_minus else -1
                    adx_strength = adx_value / 100.0  # Normalize to 0-1
                    
                    trend_components.append(adx_trend)
                    strength_components.append(adx_strength)
        
        # 4. Ichimoku Cloud (if enabled)
        if self.default_params.get('use_ichimoku', True) and 'Ichimoku_Tenkan' in indicators:
            tenkan = indicators['Ichimoku_Tenkan']
            kijun = indicators['Ichimoku_Kijun']
            senkou_a = indicators['Ichimoku_Senkou_A']
            senkou_b = indicators['Ichimoku_Senkou_B']
            
            tenkan = tenkan.iloc[-1] if hasattr(tenkan, 'iloc') else tenkan[-1]
            kijun = kijun.iloc[-1] if hasattr(kijun, 'iloc') else kijun[-1]
            
            # For Senkou spans, we need to look ahead (shifted forward)
            senkou_a = senkou_a.iloc[-26] if hasattr(senkou_a, 'iloc') else senkou_a[-26] if len(senkou_a) > 26 else 0
            senkou_b = senkou_b.iloc[-26] if hasattr(senkou_b, 'iloc') else senkou_b[-26] if len(senkou_b) > 26 else 0
            
            current_close = close.iloc[-1] if hasattr(close, 'iloc') else close[-1]
            
            ichimoku_trend = 0
            ichimoku_strength = 0.0
            
            # Price above cloud is bullish, below is bearish
            if current_close > max(senkou_a, senkou_b):
                ichimoku_trend = 1
                ichimoku_strength = (current_close - max(senkou_a, senkou_b)) / current_close
            elif current_close < min(senkou_a, senkou_b):
                ichimoku_trend = -1
                ichimoku_strength = (min(senkou_a, senkou_b) - current_close) / current_close
            
            # Tenkan/Kijun crossover
            if tenkan > kijun:
                ichimoku_trend = max(ichimoku_trend, 0) + 1
            elif tenkan < kijun:
                ichimoku_trend = min(ichimoku_trend, 0) - 1
            
            if ichimoku_trend != 0:
                trend_components.append(1 if ichimoku_trend > 0 else -1)
                strength_components.append(abs(ichimoku_strength))
        
        # Calculate overall trend
        if not trend_components:
            return 0, 0.0
        
        # Weighted average of trend components
        total_strength = sum(strength_components)
        if total_strength == 0:
            return 0, 0.0
        
        weighted_trend = sum(t * s for t, s in zip(trend_components, strength_components)) / total_strength
        avg_strength = sum(strength_components) / len(strength_components)
        
        trend_direction = 1 if weighted_trend > 0.2 else -1 if weighted_trend < -0.2 else 0
        trend_strength = min(1.0, avg_strength * 2)  # Scale to 0-1 range
        
        return trend_direction, trend_strength
    
    def _get_timeframe_relation(self) -> TimeframeRelation:
        """
        Determine the relationship between timeframes.
        
        Returns:
            TimeframeRelation indicating if timeframes are aligned, neutral, or conflicted
        """
        trends = []
        strengths = []
        
        for tf in self.timeframes:
            direction, strength = self._analyze_trend(tf)
            trends.append(direction)
            strengths.append(strength)
        
        # Count trend directions
        up_count = trends.count(1)
        down_count = trends.count(-1)
        neutral_count = trends.count(0)
        
        # Calculate average strength of non-neutral trends
        avg_strength = sum(s for s, t in zip(strengths, trends) if t != 0) / \
                      (len(trends) - neutral_count + 1e-10)
        
        # Determine relationship
        if up_count >= self.confirmation_timeframes and avg_strength > self.trend_strength_threshold:
            return TimeframeRelation.ALIGNED, 1, avg_strength
        elif down_count >= self.confirmation_timeframes and avg_strength > self.trend_strength_threshold:
            return TimeframeRelation.ALIGNED, -1, avg_strength
        elif neutral_count == len(trends):
            return TimeframeRelation.NEUTRAL, 0, 0.0
        else:
            return TimeframeRelation.CONFLICTED, 0, 0.0
    
    def calculate_signals(self, data: pd.DataFrame) -> Dict:
        """
        Calculate trading signals based on multi-timeframe analysis.
        
        Args:
            data: Latest market data (OHLCV) for the primary timeframe
            
        Returns:
            Dictionary containing signals and other information
        """
        if not self.initialized:
            self.initialize(data)
        
        # Update primary timeframe data
        self.data = data
        self._calculate_indicators(data, self.primary_tf)
        
        # Analyze trend across all timeframes
        tf_relation, trend_direction, trend_strength = self._get_timeframe_relation()
        
        # Initialize signal
        signal = {
            'signal': SignalType.NONE,
            'strength': SignalStrength.MODERATE,
            'price': data['close'].iloc[-1],
            'timestamp': data.index[-1],
            'indicators': {},
            'metadata': {
                'timeframe_relation': tf_relation.name,
                'trend_direction': 'UP' if trend_direction > 0 else 'DOWN' if trend_direction < 0 else 'NEUTRAL',
                'trend_strength': trend_strength,
                'timeframes': {}
            }
        }
        
        # Store individual timeframe analysis
        for tf in self.timeframes:
            direction, strength = self._analyze_trend(tf)
            signal['metadata']['timeframes'][tf] = {
                'trend': 'UP' if direction > 0 else 'DOWN' if direction < 0 else 'NEUTRAL',
                'strength': strength
            }
        
        # Only generate signals if timeframes are aligned
        if tf_relation == TimeframeRelation.ALIGNED:
            # Get indicators for entry/exit
            indicators = self.tf_indicators[self.primary_tf]
            close = data['close']
            rsi_entry = indicators['rsi_entry']
            
            # Check for entry signals
            if trend_direction > 0:  # Uptrend
                # Look for oversold conditions to enter long
                rsi_value = rsi_entry.iloc[-1] if hasattr(rsi_entry, 'iloc') else rsi_entry[-1]
                
                if rsi_value < (100 - self.default_params.get('rsi_overbought', 70)):
                    signal['signal'] = SignalType.LONG_ENTRY
                    signal['strength'] = SignalStrength.STRONG if trend_strength > 0.7 else SignalStrength.MODERATE
            
            elif trend_direction < 0:  # Downtrend
                # Look for overbought conditions to enter short
                rsi_value = rsi_entry.iloc[-1] if hasattr(rsi_entry, 'iloc') else rsi_entry[-1]
                
                if rsi_value > self.default_params.get('rsi_overbought', 70):
                    signal['signal'] = SignalType.SHORT_ENTRY
                    signal['strength'] = SignalStrength.STRONG if trend_strength > 0.7 else SignalStrength.MODERATE
        
        # Check for exit signals on existing positions
        elif self.position_state != PositionState.FLAT:
            # Use faster RSI for exit signals
            rsi_exit = indicators['rsi_exit']
            rsi_value = rsi_exit.iloc[-1] if hasattr(rsi_exit, 'iloc') else rsi_exit[-1]
            
            if (self.position_state == PositionState.LONG and 
                ((trend_direction < 0 and tf_relation == TimeframeRelation.ALIGNED) or
                 rsi_value > self.default_params.get('rsi_overbought', 70))):
                signal['signal'] = SignalType.LONG_EXIT
                signal['strength'] = SignalStrength.STRONG
            
            elif (self.position_state == PositionState.SHORT and 
                  ((trend_direction > 0 and tf_relation == TimeframeRelation.ALIGNED) or
                   rsi_value < (100 - self.default_params.get('rsi_overbought', 70)))):
                signal['signal'] = SignalType.SHORT_EXIT
                signal['strength'] = SignalStrength.STRONG
        
        # Update stop loss and take profit levels
        if signal['signal'] in [SignalType.LONG_ENTRY, SignalType.SHORT_ENTRY]:
            current_price = signal['price']
            atr_value = indicators['atr']
            atr_value = atr_value.iloc[-1] if hasattr(atr_value, 'iloc') else atr_value[-1]
            
            if signal['signal'] == SignalType.LONG_ENTRY:
                stop_loss = current_price - (atr_value * 1.5)
                take_profit = current_price + (atr_value * 1.5 * self.risk_reward_ratio)
            else:  # SHORT_ENTRY
                stop_loss = current_price + (atr_value * 1.5)
                take_profit = current_price - (atr_value * 1.5 * self.risk_reward_ratio)
            
            signal['metadata'].update({
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'atr': atr_value
            })
        
        # Add indicators to signal
        for name, indicator in indicators.items():
            if hasattr(indicator, 'iloc'):
                signal['indicators'][name] = indicator.iloc[-1] if len(indicator) > 0 else 0
            elif isinstance(indicator, (list, np.ndarray)) and len(indicator) > 0:
                signal['indicators'][name] = indicator[-1]
            else:
                signal['indicators'][name] = indicator
        
        # Store the signal
        self.signals.append(signal)
        
        return signal
