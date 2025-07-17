"""
Statistical Arbitrage Strategy

This strategy identifies and exploits price divergences between correlated assets
using statistical methods like cointegration, correlation analysis, and z-score
based trading signals. It's particularly effective for pairs trading and basket
trading strategies.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from scipy.stats import zscore, pearsonr
from .base_strategy import BaseStrategy, PositionState, SignalType, SignalStrength
from core.indicators.ta_indicators import atr, moving_average, MovingAverageType

class CointegrationMethod(Enum):
    """Method for testing cointegration."""
    ENGLE_GRANGER = "engle_granger"
    JOHANSEN = "johansen"

class HedgeRatioMethod(Enum):
    """Method for calculating hedge ratio."""
    OLS = "ols"
    TSM = "total_least_squares"
    KALMAN = "kalman"

class StatisticalArbitrageStrategy(BaseStrategy):
    """
    Statistical Arbitrage Strategy
    
    This strategy identifies trading opportunities by looking for temporary price
discrepancies between correlated assets. It uses statistical methods to identify
pairs or baskets of assets that tend to move together and generates signals when
their prices diverge beyond historical norms.
    
    Parameters:
    -----------
    assets : List[str]
        List of asset symbols to trade
    lookback : int
        Lookback period for statistical calculations
    zscore_entry : float
        Z-score threshold for entering trades
    zscore_exit : float
        Z-score threshold for exiting trades
    max_position_size : float
        Maximum position size as a percentage of account
    stop_loss : float
        Stop loss as a multiple of the standard deviation
    """
    
    def __init__(self, params: Optional[Dict] = None):
        super().__init__(params)
        self.name = "StatisticalArbitrageStrategy"
        self.assets = params.get('assets', [])
        self.lookback = params.get('lookback', 20)
        self.zscore_entry = params.get('zscore_entry', 2.0)
        self.zscore_exit = params.get('zscore_exit', 0.5)
        self.max_position_size = params.get('max_position_size', 0.1)  # 10% of account
        self.stop_loss = params.get('stop_loss', 3.0)  # 3 std dev stop loss
        self.entry_time = None
        self.entry_zscore = None
        self.hedge_ratio = None
        self.spread_mean = None
        self.spread_std = None
        self.coint_pvalue = None
        self.correlation = None
        
        # Initialize with default parameters
        self._set_default_params()
        if params:
            self._update_params(params)
    
    def _set_default_params(self):
        """Set default strategy parameters."""
        super()._set_default_params()
        self.default_params.update({
            'assets': ['SPY', 'QQQ'],  # Default pair
            'lookback': 20,
            'zscore_entry': 2.0,
            'zscore_exit': 0.5,
            'max_position_size': 0.1,
            'stop_loss': 3.0,
            'min_correlation': 0.7,
            'max_coint_pvalue': 0.05,
            'min_coint_period': 30,
            'use_kalman_filter': True,
            'kalman_obs_cov': 0.001,
            'kalman_trans_cov': 0.001,
            'use_volatility_scaling': True,
            'volatility_lookback': 20,
            'max_volatility': 0.02,  # Max 2% daily volatility
            'use_volume_filter': True,
            'min_volume_ratio': 0.8,
            'use_time_filter': True,
            'session_start': '09:30',
            'session_end': '16:00',
            'use_dynamic_position_sizing': True,
            'position_risk_percent': 1.0,  # 1% risk per trade
            'max_leverage': 3.0,
            'use_trailing_stop': True,
            'trail_percent': 0.5,  # 0.5% trailing stop
            'use_news_filter': False,
            'news_impact_threshold': 0.7,
            'use_sentiment': False,
            'sentiment_threshold': 0.6
        })
    
    def initialize(self, data: Dict[str, pd.DataFrame]):
        """
        Initialize the strategy with historical data.
        
        Args:
            data: Dictionary of DataFrames with OHLCV data for each asset
        """
        super().initialize(data)
        self.data = data
        
        # Check if we have enough data
        if len(self.assets) < 2:
            raise ValueError("At least 2 assets are required for statistical arbitrage")
        
        # Check if all assets have data
        for asset in self.assets:
            if asset not in data:
                raise ValueError(f"No data provided for asset: {asset}")
        
        # Calculate initial hedge ratio and spread statistics
        self._calculate_hedge_ratio()
        self._calculate_spread_stats()
        
        self.initialized = True
    
    def _calculate_hedge_ratio(self):
        """Calculate the hedge ratio between assets."""
        # Get price series for all assets
        prices = pd.DataFrame()
        for asset in self.assets:
            prices[asset] = self.data[asset]['close']
        
        # Use the first asset as the dependent variable
        y = prices[self.assets[0]]
        X = prices[self.assets[1:]]
        
        if self.default_params['use_kalman_filter']:
            self.hedge_ratio = self._kalman_filter_hedge_ratio(y, X)
        else:
            # Simple OLS regression
            X = sm.add_constant(X)  # Add constant term
            model = sm.OLS(y, X).fit()
            self.hedge_ratio = model.params[1:]  # Skip the constant
    
    def _kalman_filter_hedge_ratio(self, y: pd.Series, X: pd.DataFrame) -> np.ndarray:
        """
        Calculate hedge ratio using Kalman filter.
        
        Args:
            y: Dependent variable (prices of asset 1)
            X: Independent variables (prices of other assets)
            
        Returns:
            Array of hedge ratios
        """
        n_assets = X.shape[1]
        
        # Initialize Kalman filter parameters
        # State transition matrix (identity)
        F = np.eye(n_assets)
        
        # Observation matrix (current prices of other assets)
        H = X.values.reshape(-1, n_assets)
        
        # State covariance
        P = np.eye(n_assets)
        
        # Process noise covariance
        Q = np.eye(n_assets) * self.default_params['kalman_trans_cov']
        
        # Observation noise covariance
        R = np.eye(1) * self.default_params['kalman_obs_cov']
        
        # Initial state (hedge ratios)
        x = np.ones(n_assets) / n_assets
        
        # Apply Kalman filter
        for i in range(len(y)):
            # Prediction
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q
            
            # Update
            y_pred = H[i] @ x_pred
            y_actual = y.iloc[i]
            y_error = y_actual - y_pred
            
            S = H[i] @ P_pred @ H[i].T + R
            K = P_pred @ H[i].T / S
            
            x = x_pred + K * y_error
            P = (np.eye(n_assets) - np.outer(K, H[i])) @ P_pred
        
        return x
    
    def _calculate_spread_stats(self):
        """Calculate spread statistics (mean and standard deviation)."""
        # Get price series for all assets
        prices = pd.DataFrame()
        for asset in self.assets:
            prices[asset] = self.data[asset]['close']
        
        # Calculate spread
        spread = prices[self.assets[0]].copy()
        for i, asset in enumerate(self.assets[1:], 1):
            spread -= self.hedge_ratio[i-1] * prices[asset]
        
        # Calculate statistics
        self.spread_mean = spread.rolling(window=self.lookback).mean()
        self.spread_std = spread.rolling(window=self.lookback).std()
        self.zscores = (spread - self.spread_mean) / (self.spread_std + 1e-10)
        
        # Calculate cointegration and correlation
        self._check_cointegration(prices)
        self._calculate_correlation(prices)
    
    def _check_cointegration(self, prices: pd.DataFrame):
        """Check for cointegration between assets."""
        if len(prices) < self.default_params['min_coint_period']:
            self.coint_pvalue = 1.0
            return
        
        # Test cointegration between all pairs
        pvalues = []
        for i in range(len(self.assets)):
            for j in range(i+1, len(self.assets)):
                _, pvalue, _ = coint(
                    prices[self.assets[i]], 
                    prices[self.assets[j]]
                )
                pvalues.append(pvalue)
        
        # Use the minimum p-value
        self.coint_pvalue = min(pvalues) if pvalues else 1.0
    
    def _calculate_correlation(self, prices: pd.DataFrame):
        """Calculate correlation between assets."""
        if len(prices) < 2:
            self.correlation = 0.0
            return
        
        # Calculate correlation matrix
        corr_matrix = prices.corr()
        
        # Get average correlation (excluding diagonal)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        self.correlation = corr_matrix.where(mask).mean().mean()
    
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
    
    def _get_volume_confirmation(self) -> bool:
        """Check if volume confirms the signal."""
        if not self.default_params['use_volume_filter']:
            return True
            
        for asset in self.assets:
            volume = self.data[asset]['volume']
            volume_ma = moving_average(volume, window=self.default_params['volume_ma_period'])
            
            if len(volume) < 2 or len(volume_ma) < 2:
                return False
                
            volume_ratio = volume.iloc[-1] / volume_ma.iloc[-1] if hasattr(volume, 'iloc') else \
                          volume[-1] / volume_ma[-1]
            
            if volume_ratio < self.default_params['min_volume_ratio']:
                return False
                
        return True
    
    def _check_volatility(self) -> bool:
        """Check if volatility is within acceptable limits."""
        if not self.default_params['use_volatility_scaling']:
            return True
            
        for asset in self.assets:
            returns = self.data[asset]['close'].pct_change()
            volatility = returns.rolling(window=self.default_params['volatility_lookback']).std()
            
            if len(volatility) == 0:
                continue
                
            current_vol = volatility.iloc[-1] if hasattr(volatility, 'iloc') else volatility[-1]
            
            if current_vol > self.default_params['max_volatility']:
                return False
                
        return True
    
    def _calculate_position_size(self) -> float:
        """Calculate position size based on risk parameters."""
        if not self.default_params['use_dynamic_position_sizing']:
            return self.default_params['max_position_size']
            
        # Calculate volatility-based position sizing
        volatilities = []
        for asset in self.assets:
            returns = self.data[asset]['close'].pct_change()
            vol = returns.std()
            volatilities.append(vol)
        
        # Use inverse volatility weighting
        weights = 1 / (np.array(volatilities) + 1e-10)
        weights /= weights.sum()  # Normalize to sum to 1
        
        # Apply risk per trade
        position_size = min(
            self.default_params['position_risk_percent'] / 100,
            self.default_params['max_position_size']
        )
        
        return position_size * weights[0]  # Return weight for the first asset
    
    def _generate_signal(self) -> Dict:
        """Generate trading signal based on spread z-score."""
        if len(self.zscores) < 2:
            return {
                'signal': SignalType.NONE,
                'strength': SignalStrength.WEAK,
                'price': {asset: self.data[asset]['close'].iloc[-1] for asset in self.assets},
                'timestamp': next(iter(self.data.values())).index[-1],
                'indicators': {},
                'metadata': {'reason': 'Insufficient data for z-score calculation'}
            }
        
        current_zscore = self.zscores.iloc[-1] if hasattr(self.zscores, 'iloc') else self.zscores[-1]
        
        # Initialize signal
        signal = {
            'signal': SignalType.NONE,
            'strength': SignalStrength.MODERATE,
            'price': {asset: self.data[asset]['close'].iloc[-1] for asset in self.assets},
            'timestamp': next(iter(self.data.values())).index[-1],
            'indicators': {
                'zscore': current_zscore,
                'spread_mean': self.spread_mean.iloc[-1] if hasattr(self.spread_mean, 'iloc') else self.spread_mean[-1],
                'spread_std': self.spread_std.iloc[-1] if hasattr(self.spread_std, 'iloc') else self.spread_std[-1],
                'correlation': self.correlation,
                'coint_pvalue': self.coint_pvalue
            },
            'metadata': {
                'assets': self.assets,
                'hedge_ratio': self.hedge_ratio.tolist() if hasattr(self.hedge_ratio, 'tolist') else self.hedge_ratio,
                'lookback': self.lookback,
                'entry_zscore': self.entry_zscore
            }
        }
        
        # Check cointegration and correlation
        if (self.coint_pvalue > self.default_params['max_coint_pvalue'] or 
            abs(self.correlation) < self.default_params['min_correlation']):
            signal['metadata']['reason'] = 'Weak cointegration/correlation'
            return signal
        
        # Check if we're in a valid trading session
        if not self._is_in_session(signal['timestamp']):
            signal['metadata']['reason'] = 'Outside trading session'
            return signal
        
        # Check volume confirmation
        if not self._get_volume_confirmation():
            signal['metadata']['reason'] = 'Volume confirmation failed'
            return signal
        
        # Check volatility
        if not self._check_volatility():
            signal['metadata']['reason'] = 'High volatility'
            return signal
        
        # Generate signals based on z-score
        if self.position_state == PositionState.FLAT:
            # Long signal (spread is too low)
            if current_zscore <= -self.zscore_entry:
                signal['signal'] = SignalType.LONG_ENTRY
                signal['strength'] = SignalStrength.STRONG if abs(current_zscore) > self.zscore_entry * 1.5 else SignalStrength.MODERATE
                signal['metadata']['condition'] = f'Z-score ({current_zscore:.2f}) below -{self.zscore_entry}'
                self.entry_zscore = current_zscore
            # Short signal (spread is too high)
            elif current_zscore >= self.zscore_entry:
                signal['signal'] = SignalType.SHORT_ENTRY
                signal['strength'] = SignalStrength.STRONG if abs(current_zscore) > self.zscore_entry * 1.5 else SignalStrength.MODERATE
                signal['metadata']['condition'] = f'Z-score ({current_zscore:.2f}) above {self.zscore_entry}'
                self.entry_zscore = current_zscore
        else:
            # Check exit conditions
            exit_long = (self.position_state == PositionState.LONG and 
                        (current_zscore >= -self.zscore_exit or 
                         (self.default_params['use_trailing_stop'] and 
                          current_zscore <= self.entry_zscore * 0.8)))  # Trailing stop
            
            exit_short = (self.position_state == PositionState.SHORT and 
                         (current_zscore <= self.zscore_exit or 
                          (self.default_params['use_trailing_stop'] and 
                           current_zscore >= self.entry_zscore * 0.8)))  # Trailing stop
            
            # Stop loss
            stop_loss_hit = False
            if self.position_state == PositionState.LONG and current_zscore <= -self.stop_loss:
                exit_long = True
                stop_loss_hit = True
            elif self.position_state == PositionState.SHORT and current_zscore >= self.stop_loss:
                exit_short = True
                stop_loss_hit = True
            
            if exit_long:
                signal['signal'] = SignalType.LONG_EXIT
                signal['metadata']['exit_reason'] = 'Stop loss hit' if stop_loss_hit else 'Take profit/Exit condition met'
            elif exit_short:
                signal['signal'] = SignalType.SHORT_EXIT
                signal['metadata']['exit_reason'] = 'Stop loss hit' if stop_loss_hit else 'Take profit/Exit condition met'
        
        # Add position sizing
        if signal['signal'] in [SignalType.LONG_ENTRY, SignalType.SHORT_ENTRY]:
            position_size = self._calculate_position_size()
            signal['metadata']['position_size'] = position_size
            
            # Calculate stop loss and take profit levels
            if signal['signal'] == SignalType.LONG_ENTRY:
                stop_loss_level = -self.stop_loss
                take_profit_level = -self.zscore_exit
            else:  # SHORT_ENTRY
                stop_loss_level = self.stop_loss
                take_profit_level = self.zscore_exit
            
            signal['metadata'].update({
                'stop_loss': stop_loss_level,
                'take_profit': take_profit_level,
                'risk_reward_ratio': abs((take_profit_level - current_zscore) / (current_zscore - stop_loss_level))
            })
        
        return signal
    
    def calculate_signals(self, data: Dict[str, pd.DataFrame]) -> Dict:
        """
        Calculate trading signals based on statistical arbitrage.
        
        Args:
            data: Dictionary of DataFrames with OHLCV data for each asset
            
        Returns:
            Dictionary containing signals and other information
        """
        if not self.initialized:
            self.initialize(data)
        
        # Update data
        self.data = data
        
        # Recalculate hedge ratio and spread statistics
        self._calculate_hedge_ratio()
        self._calculate_spread_stats()
        
        # Generate signal
        signal = self._generate_signal()
        
        # Update position state if we're entering a trade
        if signal['signal'] in [SignalType.LONG_ENTRY, SignalType.SHORT_ENTRY]:
            self.entry_time = len(next(iter(data.values()))) - 1
            self.position_state = PositionState.LONG if signal['signal'] == SignalType.LONG_ENTRY else PositionState.SHORT
        elif signal['signal'] in [SignalType.LONG_EXIT, SignalType.SHORT_EXIT]:
            self.position_state = PositionState.FLAT
            self.entry_time = None
            self.entry_zscore = None
        
        # Store the signal
        self.signals.append(signal)
        
        return signal
