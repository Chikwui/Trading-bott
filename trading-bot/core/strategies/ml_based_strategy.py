"""
Machine Learning-Based Trading Strategy

This strategy uses machine learning models to predict price movements based on
technical indicators and other features. It can be trained on historical data
and used to generate trading signals.
"""
import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from .base_strategy import BaseStrategy, PositionState, SignalType, SignalStrength
from core.indicators.ta_indicators import (
    moving_average, rsi, macd, bollinger_bands, atr, stochastic_oscillator,
    MovingAverageType, IndicatorType
)
from core.indicators.advanced_indicators import (
    ichimoku_cloud, parabolic_sar, adx, volume_profile, VolumeProfileLevels
)

class ModelType(Enum):
    """Type of machine learning model to use."""
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    SVM = "svm"

class FeatureSet(Enum):
    """Predefined feature sets for the ML model."""
    BASIC = "basic"           # Basic price and volume features
    TECHNICAL = "technical"   # Common technical indicators
    ADVANCED = "advanced"     # Advanced indicators and patterns
    ALL = "all"               # All available features

class MLBasedStrategy(BaseStrategy):
    """
    Machine Learning-Based Trading Strategy
    
    This strategy uses machine learning to predict price movements based on
    technical indicators and other features. It can be trained on historical
    data and used to generate trading signals.
    
    Parameters:
    -----------
    model_type : str or ModelType
        Type of machine learning model to use
    feature_set : str or FeatureSet
        Set of features to use for training/prediction
    lookback : int
        Number of past bars to include in features
    target_bars : int
        Number of bars ahead to predict (1 = next bar)
    threshold : float
        Confidence threshold for taking a trade (0.5-1.0)
    retrain_interval : int
        Number of bars between model retraining (0 = never retrain)
    """
    
    def __init__(self, params: Optional[Dict] = None):
        super().__init__(params)
        self.name = "MLBasedStrategy"
        self.model_type = ModelType(params.get('model_type', ModelType.XGBOOST))
        self.feature_set = FeatureSet(params.get('feature_set', FeatureSet.ADVANCED))
        self.lookback = params.get('lookback', 14)
        self.target_bars = params.get('target_bars', 1)
        self.threshold = params.get('threshold', 0.6)
        self.retrain_interval = params.get('retrain_interval', 100)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.last_retrain = 0
        self.training_data = None
        
        # Model parameters
        self.model_params = params.get('model_params', {})
        
        # Initialize with default parameters
        self._set_default_params()
        if params:
            self._update_params(params)
    
    def _set_default_params(self):
        """Set default strategy parameters."""
        super()._set_default_params()
        self.default_params.update({
            'model_type': ModelType.XGBOOST,
            'feature_set': FeatureSet.ADVANCED,
            'lookback': 14,
            'target_bars': 1,
            'threshold': 0.6,
            'retrain_interval': 100,
            'model_params': {
                'n_estimators': 100,
                'max_depth': 5,
                'learning_rate': 0.1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': 42
            },
            'train_test_split': 0.8,
            'cross_validation': 5,
            'feature_importance_threshold': 0.01,
            'use_pca': False,
            'pca_components': 0.95,
            'early_stopping_rounds': 20,
            'class_weights': 'balanced',
            'use_smote': False,
            'save_model_path': 'models/ml_strategy',
            'load_model_path': None
        })
    
    def initialize(self, data: pd.DataFrame):
        """
        Initialize the strategy with historical data.
        
        Args:
            data: DataFrame with OHLCV data
        """
        super().initialize(data)
        
        # Prepare features and target
        self._prepare_features(data)
        
        # Load or train model
        if self.default_params.get('load_model_path') and os.path.exists(self.default_params['load_model_path']):
            self._load_model(self.default_params['load_model_path'])
        else:
            self.train_model(data)
        
        self.initialized = True
    
    def _prepare_features(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for the ML model.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Tuple of (features, target)
        """
        df = data.copy()
        
        # Calculate returns (target)
        df['returns'] = df['close'].pct_change(self.target_bars).shift(-self.target_bars)
        df['target'] = (df['returns'] > 0).astype(int)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Calculate features based on feature_set
        features = pd.DataFrame(index=df.index)
        
        # 1. Basic price features
        if self.feature_set in [FeatureSet.BASIC, FeatureSet.TECHNICAL, 
                               FeatureSet.ADVANCED, FeatureSet.ALL]:
            # Price returns
            for period in [1, 3, 5, 10]:
                features[f'return_{period}'] = df['close'].pct_change(period)
            
            # Volatility
            features['volatility_10'] = df['close'].pct_change().rolling(window=10).std()
            features['volatility_20'] = df['close'].pct_change().rolling(window=20).std()
            
            # Volume features
            features['volume_change'] = df['volume'].pct_change()
            features['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        # 2. Technical indicators
        if self.feature_set in [FeatureSet.TECHNICAL, FeatureSet.ADVANCED, FeatureSet.ALL]:
            # RSI
            features['rsi_14'] = rsi(df['close'], window=14)
            features['rsi_7'] = rsi(df['close'], window=7)
            
            # MACD
            macd_line, signal_line, _ = macd(df['close'])
            features['macd'] = macd_line
            features['macd_signal'] = signal_line
            features['macd_hist'] = macd_line - signal_line
            
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = bollinger_bands(df['close'])
            features['bb_upper'] = bb_upper
            features['bb_middle'] = bb_middle
            features['bb_lower'] = bb_lower
            features['bb_width'] = (bb_upper - bb_lower) / bb_middle
            
            # ATR
            features['atr'] = atr(df['high'], df['low'], df['close'])
            
            # Stochastic Oscillator
            stoch_k, stoch_d = stochastic_oscillator(df['high'], df['low'], df['close'])
            features['stoch_k'] = stoch_k
            features['stoch_d'] = stoch_d
        
        # 3. Advanced indicators
        if self.feature_set in [FeatureSet.ADVANCED, FeatureSet.ALL]:
            # Ichimoku Cloud
            ichimoku = ichimoku_cloud(df['high'], df['low'], df['close'])
            for name, values in ichimoku.items():
                features[f'ichimoku_{name}'] = values
            
            # ADX
            adx_result = adx(df['high'], df['low'], df['close'])
            features['adx'] = adx_result['ADX']
            features['di_plus'] = adx_result['+DI']
            features['di_minus'] = adx_result['-DI']
            
            # Parabolic SAR
            features['sar'] = parabolic_sar(df['high'], df['low'])
            
            # Volume Profile
            vp = volume_profile(df['close'], df['volume'])
            features['vp_poc'] = vp.point_of_control
            features['vp_vah'] = vp.value_area_high
            features['vp_val'] = vp.value_area_low
        
        # 4. Additional features for ALL
        if self.feature_set == FeatureSet.ALL:
            # Price patterns (simplified)
            features['is_doji'] = self._is_doji(df)
            features['is_engulfing'] = self._is_engulfing(df)
            features['is_hammer'] = self._is_hammer(df)
            
            # Time-based features
            features['hour'] = df.index.hour
            features['day_of_week'] = df.index.dayofweek
            features['month'] = df.index.month
            
            # Volatility ratios
            features['vr_14_50'] = features['volatility_14'] / features['volatility_50']
            
            # Price/MA ratios
            for period in [20, 50, 100, 200]:
                ma = moving_average(df['close'], window=period)
                features[f'price_ma_{period}_ratio'] = df['close'] / ma - 1
        
        # Add lagged features
        if self.lookback > 1:
            for col in features.columns:
                for lag in range(1, self.lookback):
                    features[f'{col}_lag{lag}'] = features[col].shift(lag)
        
        # Drop remaining NaN values
        features = features.dropna()
        
        # Align with target
        common_index = features.index.intersection(df.index)
        features = features.loc[common_index]
        target = df.loc[common_index, 'target']
        
        # Store prepared data
        self.features = features
        self.target = target
        
        return features, target
    
    def _is_doji(self, df: pd.DataFrame, threshold: float = 0.1) -> pd.Series:
        """Identify Doji candle pattern."""
        body_size = abs(df['close'] - df['open'])
        total_range = df['high'] - df['low']
        doji = (body_size / (total_range + 1e-10) < threshold).astype(int)
        return doji
    
    def _is_engulfing(self, df: pd.DataFrame) -> pd.Series:
        """Identify Engulfing candle pattern."""
        prev_close = df['close'].shift(1)
        prev_open = df['open'].shift(1)
        
        bullish = (df['close'] > df['open']) & \
                  (df['open'] < prev_close) & \
                  (df['close'] > prev_open) & \
                  ((df['close'] - df['open']) > (prev_open - prev_close))
        
        bearish = (df['close'] < df['open']) & \
                  (df['open'] > prev_close) & \
                  (df['close'] < prev_open) & \
                  ((df['open'] - df['close']) > (prev_close - prev_open))
        
        engulfing = pd.Series(0, index=df.index)
        engulfing[bullish] = 1
        engulfing[bearish] = -1
        
        return engulfing
    
    def _is_hammer(self, df: pd.DataFrame) -> pd.Series:
        """Identify Hammer/Inverted Hammer candle pattern."""
        body_size = abs(df['close'] - df['open'])
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        total_range = df['high'] - df['low']
        
        # Hammer (lower shadow at least 2x body, little to no upper shadow)
        hammer = (lower_shadow > 2 * body_size) & \
                 (upper_shadow < body_size * 0.5) & \
                 (df['close'] > df['open'])
        
        # Inverted Hammer (upper shadow at least 2x body, little to no lower shadow)
        inv_hammer = (upper_shadow > 2 * body_size) & \
                     (lower_shadow < body_size * 0.5) & \
                     (df['close'] < df['open'])
        
        hammer_pattern = pd.Series(0, index=df.index)
        hammer_pattern[hammer] = 1
        hammer_pattern[inv_hammer] = -1
        
        return hammer_pattern
    
    def train_model(self, data: pd.DataFrame):
        """
        Train the machine learning model.
        
        Args:
            data: DataFrame with OHLCV data for training
        """
        # Prepare features and target
        X, y = self._prepare_features(data)
        
        # Split into train/test sets
        train_size = int(len(X) * self.default_params['train_test_split'])
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Scale features
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize model
        model_type = self.model_type
        if model_type == ModelType.RANDOM_FOREST:
            self.model = RandomForestClassifier(**self.model_params)
        elif model_type == ModelType.GRADIENT_BOOSTING:
            self.model = GradientBoostingClassifier(**self.model_params)
        elif model_type == ModelType.XGBOOST:
            self.model = xgb.XGBClassifier(**self.model_params)
        elif model_type == ModelType.LIGHTGBM:
            self.model = lgb.LGBMClassifier(**self.model_params)
        elif model_type == ModelType.SVM:
            self.model = SVC(probability=True, **self.model_params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Train model
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],
            early_stopping_rounds=self.default_params.get('early_stopping_rounds', None),
            verbose=False
        )
        
        # Evaluate model
        train_pred = self.model.predict_proba(X_train_scaled)[:, 1]
        test_pred = self.model.predict_proba(X_test_scaled)[:, 1]
        
        train_accuracy = accuracy_score(y_train, train_pred > 0.5)
        test_accuracy = accuracy_score(y_test, test_pred > 0.5)
        
        print(f"Model training complete:")
        print(f"  - Train Accuracy: {train_accuracy:.4f}")
        print(f"  - Test Accuracy: {test_accuracy:.4f}")
        
        # Store feature importances
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = dict(zip(X.columns, self.model.feature_importances_))
        
        # Save model if path is provided
        if self.default_params.get('save_model_path'):
            os.makedirs(os.path.dirname(self.default_params['save_model_path']), exist_ok=True)
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler,
                'feature_importance': self.feature_importance,
                'params': self.default_params
            }, self.default_params['save_model_path'])
            print(f"Model saved to {self.default_params['save_model_path']}")
    
    def _load_model(self, path: str):
        """
        Load a trained model from disk.
        
        Args:
            path: Path to the saved model
        """
        try:
            saved_data = joblib.load(path)
            self.model = saved_data['model']
            self.scaler = saved_data['scaler']
            self.feature_importance = saved_data.get('feature_importance', {})
            print(f"Model loaded from {path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def calculate_signals(self, data: pd.DataFrame) -> Dict:
        """
        Calculate trading signals using the trained ML model.
        
        Args:
            data: Latest market data (OHLCV)
            
        Returns:
            Dictionary containing signals and other information
        """
        if not self.initialized:
            self.initialize(data)
        
        # Prepare features for prediction
        features, _ = self._prepare_features(data)
        if len(features) == 0:
            return {
                'signal': SignalType.NONE,
                'strength': SignalStrength.WEAK,
                'price': data['close'].iloc[-1],
                'timestamp': data.index[-1],
                'indicators': {},
                'metadata': {'error': 'Insufficient data for prediction'}
            }
        
        # Get the most recent feature vector
        X = features.iloc[[-1]]
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        proba = self.model.predict_proba(X_scaled)[0]
        prediction = self.model.predict(X_scaled)[0]
        confidence = max(proba)
        
        # Determine signal
        signal = SignalType.NONE
        strength = SignalStrength.WEAK
        
        if confidence > self.threshold:
            if prediction == 1:  # Buy signal
                signal = SignalType.LONG_ENTRY
                strength = SignalStrength.STRONG if confidence > 0.8 else SignalStrength.MODERATE
            else:  # Sell signal
                signal = SignalType.SHORT_ENTRY
                strength = SignalStrength.STRONG if confidence > 0.8 else SignalStrength.MODERATE
        
        # Check for exit signals on existing positions
        if self.position_state != PositionState.FLAT:
            if ((self.position_state == PositionState.LONG and signal == SignalType.SHORT_ENTRY) or
                (self.position_state == PositionState.SHORT and signal == SignalType.LONG_ENTRY)):
                signal = SignalType.LONG_EXIT if self.position_state == PositionState.LONG else SignalType.SHORT_EXIT
        
        # Prepare result
        result = {
            'signal': signal,
            'strength': strength,
            'price': data['close'].iloc[-1],
            'timestamp': data.index[-1],
            'indicators': {
                'prediction': prediction,
                'confidence': confidence,
                'proba_up': proba[1],
                'proba_down': proba[0]
            },
            'metadata': {
                'model_type': str(self.model_type),
                'feature_set': str(self.feature_set),
                'lookback': self.lookback,
                'threshold': self.threshold
            }
        }
        
        # Add feature importances if available
        if self.feature_importance:
            top_features = sorted(
                [(k, v) for k, v in self.feature_importance.items()],
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5]  # Top 5 features
            result['metadata']['top_features'] = dict(top_features)
        
        # Store the signal
        self.signals.append(result)
        
        # Retrain model if needed
        if (self.retrain_interval > 0 and 
            len(self.signals) - self.last_retrain >= self.retrain_interval):
            print("Retraining model...")
            self.train_model(data)
            self.last_retrain = len(self.signals)
        
        return result
