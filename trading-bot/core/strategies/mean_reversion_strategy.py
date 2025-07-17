"""
Mean Reversion Strategy

This strategy identifies overbought/oversold conditions based on the assumption that
prices will revert to their mean over time. It's particularly effective in range-bound
markets and uses statistical measures like Z-scores and Bollinger Bands to identify
extreme price movements.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from enum import Enum
from .base_strategy import BaseStrategy, PositionState, SignalType, SignalStrength
from core.indicators.ta_indicators import (
    bollinger_bands, rsi, atr, moving_average, 
    MovingAverageType, IndicatorType, stochastic_oscillator
)
from core.indicators.advanced_indicators import (
    keltner_channels, donchian_channels, volume_profile
)

class MeanReversionType(Enum):
    """Type of mean reversion strategy."""
    BOLLINGER = "bollinger"
    KELTNER = "keltner"
    DONCHIAN = "donchian"
    ZSCORE = "zscore"
    COMBINED = "combined"

class MeanReversionStrategy(BaseStrategy):
    """
    Mean Reversion Trading Strategy
    
    This strategy identifies overbought/oversold conditions and takes positions
    expecting prices to revert to their mean. It's particularly effective in
    range-bound markets.
    
    Parameters:
    -----------
    reversion_type : str or MeanReversionType
        Type of mean reversion strategy to use
    lookback : int
        Lookback period for calculating mean and standard deviation
    std_dev : float
        Number of standard deviations for Bollinger Bands
    rsi_period : int
        Period for RSI calculation
    rsi_overbought : float
        RSI level considered overbought (default: 70)
    rsi_oversold : float
        RSI level considered oversold (default: 30)
    atr_period : int
        Period for ATR calculation for position sizing
    atr_multiplier : float
        Multiplier for ATR-based stop loss
    min_holding_period : int
        Minimum number of bars to hold a position
    max_holding_period : int
        Maximum number of bars to hold a position
    """
    
    def __init__(self, params: Optional[Dict] = None):
        super().__init__(params)
        self.name = "MeanReversionStrategy"
        self.reversion_type = MeanReversionType(params.get('reversion_type', MeanReversionType.COMBINED))
        self.lookback = params.get('lookback', 20)
        self.std_dev = params.get('std_dev', 2.0)
        self.rsi_period = params.get('rsi_period', 14)
        self.rsi_overbought = params.get('rsi_overbought', 70)
        self.rsi_oversold = params.get('rsi_oversold', 30)
        self.atr_period = params.get('atr_period', 14)
        self.atr_multiplier = params.get('atr_multiplier', 2.0)
        self.min_holding_period = params.get('min_holding_period', 5)
        self.max_holding_period = params.get('max_holding_period', 20)
        self.entry_time = None
        self.entry_price = None
        
        # Initialize with default parameters
        self._set_default_params()
        if params:
            self._update_params(params)
    
    def _set_default_params(self):
        """Set default strategy parameters."""
        super()._set_default_params()
        self.default_params.update({
            'reversion_type': MeanReversionType.COMBINED,
            'lookback': 20,
            'std_dev': 2.0,
            'rsi_period': 14,
            'rsi_overbought': 70,
            'rsi_oversold': 30,
            'atr_period': 14,
            'atr_multiplier': 2.0,
            'min_holding_period': 5,
            'max_holding_period': 20,
            'use_adx_filter': True,
            'adx_period': 14,
            'adx_threshold': 25,
            'use_volume_filter': True,
            'volume_ma_period': 20,
            'min_volume_ratio': 0.8,
            'use_time_filter': True,
            'session_start': '09:30',
            'session_end': '16:00',
            'max_position_size': 0.1,  # Max 10% of account per trade
            'risk_per_trade': 0.01,    # 1% risk per trade
            'trailing_stop': True,
            'trail_percent': 0.5,      # 0.5% trailing stop
            'use_volatility_filter': True,
            'volatility_period': 20,
            'max_volatility': 0.02,    # Max 2% ATR/price ratio
            'use_correlation_filter': False,
            'correlation_period': 20,
            'min_correlation': 0.7,
            'use_news_filter': False,
            'news_impact_threshold': 0.7,
            'use_sentiment': False,
            'sentiment_threshold': 0.6
        })
    
    def initialize(self, data: pd.DataFrame):
        """
        Initialize the strategy with historical data.
        
        Args:
            data: DataFrame with OHLCV data
        """
        super().initialize(data)
        
        # Calculate indicators
        self._calculate_indicators(data)
        
        self.initialized = True
    
    def _calculate_indicators(self, data: pd.DataFrame):
        """Calculate all required indicators."""
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']
        
        # Moving averages
        self.sma = moving_average(close, window=self.lookback)
        self.ema = moving_average(close, window=self.lookback//2, ma_type=MovingAverageType.EMA)
        
        # Bollinger Bands
        self.bb_upper, self.bb_middle, self.bb_lower = bollinger_bands(
            close, window=self.lookback, num_std=self.std_dev
        )
        
        # Keltner Channels
        self.keltner_upper, self.keltner_middle, self.keltner_lower = keltner_channels(
            high, low, close, window=self.lookback, multiplier=self.std_dev
        )
        
        # Donchian Channels
        self.donchian_upper, self.donchian_lower = donchian_channels(
            high, low, window=self.lookback
        )
        
        # RSI
        self.rsi = rsi(close, window=self.rsi_period)
        
        # ATR for position sizing and stop loss
        self.atr = atr(high, low, close, window=self.atr_period)
        
        # Volume indicators
        self.volume_ma = moving_average(volume, window=self.default_params['volume_ma_period'])
        
        # ADX for trend strength
        if self.default_params['use_adx_filter']:
            adx_result = adx(high, low, close, window=self.default_params['adx_period'])
            self.adx = adx_result['ADX']
            self.di_plus = adx_result['+DI']
            self.di_minus = adx_result['-DI']
        
        # Volatility
        self.volatility = close.pct_change().rolling(window=self.default_params['volatility_period']).std()
        
        # Store the latest values
        self.latest_close = close.iloc[-1] if hasattr(close, 'iloc') else close[-1]
        self.latest_high = high.iloc[-1] if hasattr(high, 'iloc') else high[-1]
        self.latest_low = low.iloc[-1] if hasattr(low, 'iloc') else low[-1]
    
    def _is_trending(self) -> bool:
        """Check if the market is trending based on ADX."""
        if not self.default_params['use_adx_filter']:
            return False
            
        adx_value = self.adx.iloc[-1] if hasattr(self.adx, 'iloc') else self.adx[-1]
        return adx_value > self.default_params['adx_threshold']
    
    def _get_volume_confirmation(self) -> bool:
        """Check if volume confirms the signal."""
        if not self.default_params['use_volume_filter']:
            return True
            
        volume_ratio = (self.volume_ma.iloc[-1] / self.volume_ma.iloc[-2] if hasattr(self.volume_ma, 'iloc')
                       else self.volume_ma[-1] / self.volume_ma[-2])
        return volume_ratio >= self.default_params['min_volume_ratio']
    
    def _is_in_session(self, timestamp) -> bool:
        """Check if the current time is within the trading session."""
        if not self.default_params['use_time_filter']:
            return True
            
        if hasattr(timestamp, 'time'):  # If it's a datetime object
            time = timestamp.time()
        else:  # If it's a string or numeric timestamp
            time = pd.to_datetime(timestamp).time()
            
        session_start = pd.to_datetime(self.default_params['session_start']).time()
        session_end = pd.to_datetime(self.default_params['session_end']).time()
        
        return session_start <= time <= session_end
    
    def _get_zscore(self, window: int = 20) -> float:
        """Calculate the Z-score of the latest price."""
        if hasattr(self.data['close'], 'iloc'):
            prices = self.data['close'].iloc[-window:]
        else:
            prices = self.data['close'][-window:]
            
        mean = np.mean(prices)
        std = np.std(prices)
        
        if std == 0:
            return 0
            
        latest_price = prices.iloc[-1] if hasattr(prices, 'iloc') else prices[-1]
        return (latest_price - mean) / std
    
    def _get_signal_strength(self, signal_type: SignalType) -> SignalStrength:
        """Determine the strength of the signal based on multiple factors."""
        # Base strength
        strength = SignalStrength.MODERATE
        
        # Check RSI
        rsi_value = self.rsi.iloc[-1] if hasattr(self.rsi, 'iloc') else self.rsi[-1]
        
        # Check Z-score
        zscore = self._get_zscore()
        
        # Check volume
        volume_ratio = (self.volume_ma.iloc[-1] / self.volume_ma.iloc[-2] if hasattr(self.volume_ma, 'iloc')
                       else self.volume_ma[-1] / self.volume_ma[-2])
        
        # Increase strength for extreme RSI and high volume
        if ((signal_type == SignalType.LONG_ENTRY and rsi_value < (self.rsi_oversold - 10)) or
            (signal_type == SignalType.SHORT_ENTRY and rsi_value > (self.rsi_overbought + 10))) and \
           volume_ratio > 1.5 and abs(zscore) > 2.5:
            strength = SignalStrength.VERY_STRONG
        elif ((signal_type == SignalType.LONG_ENTRY and rsi_value < self.rsi_oversold) or
              (signal_type == SignalType.SHORT_ENTRY and rsi_value > self.rsi_overbought)) and \
             volume_ratio > 1.2 and abs(zscore) > 2.0:
            strength = SignalStrength.STRONG
        elif volume_ratio < 0.8 or abs(zscore) < 1.0:
            strength = SignalStrength.WEAK
            
        return strength
    
    def _check_exit_conditions(self, current_price: float) -> bool:
        """Check if exit conditions are met for the current position."""
        if self.position_state == PositionState.FLAT:
            return False
            
        # Check holding period
        current_time = len(self.data) - 1
        bars_held = current_time - self.entry_time if self.entry_time is not None else 0
        
        if bars_held >= self.max_holding_period:
            return True
            
        # Check profit/loss
        if self.position_state == PositionState.LONG:
            if current_price <= self.stop_loss:
                return True
            if current_price >= self.take_profit:
                return True
                
            # Update trailing stop for long positions
            if self.default_params['trailing_stop']:
                trail_amount = current_price * (self.default_params['trail_percent'] / 100)
                new_stop = current_price - trail_amount
                self.stop_loss = max(self.stop_loss, new_stop)
                
        elif self.position_state == PositionState.SHORT:
            if current_price >= self.stop_loss:
                return True
            if current_price <= self.take_profit:
                return True
                
            # Update trailing stop for short positions
            if self.default_params['trailing_stop']:
                trail_amount = current_price * (self.default_params['trail_percent'] / 100)
                new_stop = current_price + trail_amount
                self.stop_loss = min(self.stop_loss, new_stop)
        
        return False
    
    def calculate_signals(self, data: pd.DataFrame) -> Dict:
        """
        Calculate trading signals based on mean reversion strategy.
        
        Args:
            data: Latest market data (OHLCV)
            
        Returns:
            Dictionary containing signals and other information
        """
        if not self.initialized:
            self.initialize(data)
        
        # Update data and indicators
        self.data = data
        self._calculate_indicators(data)
        
        # Check exit conditions first
        current_price = self.latest_close
        should_exit = self._check_exit_conditions(current_price)
        
        if should_exit:
            signal_type = SignalType.LONG_EXIT if self.position_state == PositionState.LONG else SignalType.SHORT_EXIT
            return {
                'signal': signal_type,
                'strength': SignalStrength.MODERATE,
                'price': current_price,
                'timestamp': data.index[-1],
                'indicators': {
                    'rsi': self.rsi.iloc[-1] if hasattr(self.rsi, 'iloc') else self.rsi[-1],
                    'atr': self.atr.iloc[-1] if hasattr(self.atr, 'iloc') else self.atr[-1],
                    'zscore': self._get_zscore()
                },
                'metadata': {
                    'exit_reason': 'Holding period or stop/target reached',
                    'holding_bars': len(self.data) - 1 - (self.entry_time if self.entry_time is not None else 0)
                }
            }
        
        # Skip if market is trending strongly (not suitable for mean reversion)
        if self._is_trending() and self.default_params['use_adx_filter']:
            return {
                'signal': SignalType.NONE,
                'strength': SignalStrength.WEAK,
                'price': current_price,
                'timestamp': data.index[-1],
                'indicators': {},
                'metadata': {'reason': 'Strong trend detected, avoiding mean reversion'}
            }
        
        # Check volume confirmation
        if not self._get_volume_confirmation():
            return {
                'signal': SignalType.NONE,
                'strength': SignalStrength.WEAK,
                'price': current_price,
                'timestamp': data.index[-1],
                'indicators': {},
                'metadata': {'reason': 'Volume confirmation failed'}
            }
        
        # Check if in trading session
        if not self._is_in_session(data.index[-1]):
            return {
                'signal': SignalType.NONE,
                'strength': SignalStrength.WEAK,
                'price': current_price,
                'timestamp': data.index[-1],
                'indicators': {},
                'metadata': {'reason': 'Outside trading session'}
            }
        
        # Check volatility
        if self.default_params['use_volatility_filter']:
            atr_ratio = (self.atr.iloc[-1] / current_price if hasattr(self.atr, 'iloc') 
                        else self.atr[-1] / current_price)
            if atr_ratio > self.default_params['max_volatility']:
                return {
                    'signal': SignalType.NONE,
                    'strength': SignalStrength.WEAK,
                    'price': current_price,
                    'timestamp': data.index[-1],
                    'indicators': {},
                    'metadata': {'reason': 'High volatility, avoiding trades'}
                }
        
        # Generate signals based on the selected reversion type
        signal = self._generate_signal()
        
        # If we have a valid signal, set up stop loss and take profit
        if signal['signal'] in [SignalType.LONG_ENTRY, SignalType.SHORT_ENTRY]:
            atr_value = self.atr.iloc[-1] if hasattr(self.atr, 'iloc') else self.atr[-1]
            
            if signal['signal'] == SignalType.LONG_ENTRY:
                # For long entries, set stop below recent low and target based on risk/reward
                stop_loss = current_price - (atr_value * self.atr_multiplier)
                take_profit = current_price + (atr_value * self.atr_multiplier * self.default_params['risk_per_trade'] * 100)
            else:  # SHORT_ENTRY
                # For short entries, set stop above recent high and target based on risk/reward
                stop_loss = current_price + (atr_value * self.atr_multiplier)
                take_profit = current_price - (atr_value * self.atr_multiplier * self.default_params['risk_per_trade'] * 100)
            
            # Update position state
            self.entry_price = current_price
            self.entry_time = len(self.data) - 1
            self.stop_loss = stop_loss
            self.take_profit = take_profit
            
            # Add to signal
            signal['metadata'].update({
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'atr': atr_value,
                'risk_reward_ratio': self.default_params['risk_per_trade'] * 100
            })
        
        # Store the signal
        self.signals.append(signal)
        
        return signal
    
    def _generate_signal(self) -> Dict:
        """Generate trading signal based on the selected mean reversion type."""
        current_price = self.latest_close
        rsi_value = self.rsi.iloc[-1] if hasattr(self.rsi, 'iloc') else self.rsi[-1]
        zscore = self._get_zscore()
        
        # Initialize default signal
        signal = {
            'signal': SignalType.NONE,
            'strength': SignalStrength.MODERATE,
            'price': current_price,
            'timestamp': self.data.index[-1],
            'indicators': {
                'rsi': rsi_value,
                'zscore': zscore,
                'atr': self.atr.iloc[-1] if hasattr(self.atr, 'iloc') else self.atr[-1]
            },
            'metadata': {
                'strategy': str(self.reversion_type),
                'lookback': self.lookback
            }
        }
        
        # Check each reversion type
        if self.reversion_type == MeanReversionType.BOLLINGER:
            bb_upper = self.bb_upper.iloc[-1] if hasattr(self.bb_upper, 'iloc') else self.bb_upper[-1]
            bb_lower = self.bb_lower.iloc[-1] if hasattr(self.bb_lower, 'iloc') else self.bb_lower[-1]
            
            if current_price <= bb_lower and rsi_value < self.rsi_oversold:
                signal['signal'] = SignalType.LONG_ENTRY
                signal['strength'] = self._get_signal_strength(SignalType.LONG_ENTRY)
                signal['metadata']['condition'] = 'Price below lower Bollinger Band'
            elif current_price >= bb_upper and rsi_value > self.rsi_overbought:
                signal['signal'] = SignalType.SHORT_ENTRY
                signal['strength'] = self._get_signal_strength(SignalType.SHORT_ENTRY)
                signal['metadata']['condition'] = 'Price above upper Bollinger Band'
                
        elif self.reversion_type == MeanReversionType.KELTNER:
            keltner_upper = self.keltner_upper.iloc[-1] if hasattr(self.keltner_upper, 'iloc') else self.keltner_upper[-1]
            keltner_lower = self.keltner_lower.iloc[-1] if hasattr(self.keltner_lower, 'iloc') else self.keltner_lower[-1]
            
            if current_price <= keltner_lower and rsi_value < self.rsi_oversold:
                signal['signal'] = SignalType.LONG_ENTRY
                signal['strength'] = self._get_signal_strength(SignalType.LONG_ENTRY)
                signal['metadata']['condition'] = 'Price below lower Keltner Channel'
            elif current_price >= keltner_upper and rsi_value > self.rsi_overbought:
                signal['signal'] = SignalType.SHORT_ENTRY
                signal['strength'] = self._get_signal_strength(SignalType.SHORT_ENTRY)
                signal['metadata']['condition'] = 'Price above upper Keltner Channel'
                
        elif self.reversion_type == MeanReversionType.DONCHIAN:
            donchian_upper = self.donchian_upper.iloc[-1] if hasattr(self.donchian_upper, 'iloc') else self.donchian_upper[-1]
            donchian_lower = self.donchian_lower.iloc[-1] if hasattr(self.donchian_lower, 'iloc') else self.donchian_lower[-1]
            
            if current_price <= donchian_lower and rsi_value < self.rsi_oversold:
                signal['signal'] = SignalType.LONG_ENTRY
                signal['strength'] = self._get_signal_strength(SignalType.LONG_ENTRY)
                signal['metadata']['condition'] = 'Price below lower Donchian Channel'
            elif current_price >= donchian_upper and rsi_value > self.rsi_overbought:
                signal['signal'] = SignalType.SHORT_ENTRY
                signal['strength'] = self._get_signal_strength(SignalType.SHORT_ENTRY)
                signal['metadata']['condition'] = 'Price above upper Donchian Channel'
                
        elif self.reversion_type == MeanReversionType.ZSCORE:
            if zscore < -2.0 and rsi_value < self.rsi_oversold:
                signal['signal'] = SignalType.LONG_ENTRY
                signal['strength'] = self._get_signal_strength(SignalType.LONG_ENTRY)
                signal['metadata']['condition'] = f'Z-score ({zscore:.2f}) below -2.0'
            elif zscore > 2.0 and rsi_value > self.rsi_overbought:
                signal['signal'] = SignalType.SHORT_ENTRY
                signal['strength'] = self._get_signal_strength(SignalType.SHORT_ENTRY)
                signal['metadata']['condition'] = f'Z-score ({zscore:.2f}) above 2.0'
                
        elif self.reversion_type == MeanReversionType.COMBINED:
            # Combined approach using multiple indicators
            bb_upper = self.bb_upper.iloc[-1] if hasattr(self.bb_upper, 'iloc') else self.bb_upper[-1]
            bb_lower = self.bb_lower.iloc[-1] if hasattr(self.bb_lower, 'iloc') else self.bb_lower[-1]
            keltner_upper = self.keltner_upper.iloc[-1] if hasattr(self.keltner_upper, 'iloc') else self.keltner_upper[-1]
            keltner_lower = self.keltner_lower.iloc[-1] if hasattr(self.keltner_lower, 'iloc') else self.keltner_lower[-1]
            
            # Check for long entry (oversold conditions)
            bb_long = current_price <= bb_lower
            keltner_long = current_price <= keltner_lower
            rsi_long = rsi_value < self.rsi_oversold
            zscore_long = zscore < -2.0
            
            # Check for short entry (overbought conditions)
            bb_short = current_price >= bb_upper
            keltner_short = current_price >= keltner_upper
            rsi_short = rsi_value > self.rsi_overbought
            zscore_short = zscore > 2.0
            
            # Require confirmation from multiple indicators
            long_conditions = [bb_long, keltner_long, rsi_long, zscore_long]
            short_conditions = [bb_short, keltner_short, rsi_short, zscore_short]
            
            if sum(long_conditions) >= 3:  # At least 3 out of 4 conditions met
                signal['signal'] = SignalType.LONG_ENTRY
                signal['strength'] = self._get_signal_strength(SignalType.LONG_ENTRY)
                signal['metadata']['condition'] = 'Multiple oversold conditions met'
                signal['metadata']['conditions_met'] = sum(long_conditions)
                
            elif sum(short_conditions) >= 3:  # At least 3 out of 4 conditions met
                signal['signal'] = SignalType.SHORT_ENTRY
                signal['strength'] = self._get_signal_strength(SignalType.SHORT_ENTRY)
                signal['metadata']['condition'] = 'Multiple overbought conditions met'
                signal['metadata']['conditions_met'] = sum(short_conditions)
        
        return signal
