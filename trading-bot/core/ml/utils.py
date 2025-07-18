"""
ML Utilities

This module provides utility functions for the machine learning pipeline,
including data preprocessing, feature engineering, and evaluation metrics.
"""
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve, roc_curve, auc
)
from scipy import stats
import talib
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

# Type aliases
ArrayLike = Union[np.ndarray, pd.Series, pd.DataFrame]

@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    # Technical indicators
    use_rsi: bool = True
    use_macd: bool = True
    use_bollinger: bool = True
    use_atr: bool = True
    use_obv: bool = True
    use_ema: bool = True
    use_stoch: bool = False
    use_adx: bool = False
    use_cci: bool = False
    
    # Statistical features
    use_returns: bool = True
    use_volatility: bool = True
    use_skew: bool = False
    use_kurtosis: bool = False
    use_correlation: bool = False
    
    # Time-based features
    use_time_features: bool = True
    use_weekday: bool = True
    use_month: bool = True
    use_hour: bool = False
    use_is_quarter_end: bool = True
    
    # Target transformation
    target_lookahead: int = 5  # Number of periods to look ahead for target
    target_threshold: float = 0.001  # Minimum return to consider as a signal
    
    # Feature scaling
    scale_features: bool = True
    scaler_type: str = 'standard'  # 'standard', 'minmax', 'robust', or None
    
    # Feature selection
    feature_importance_threshold: float = 0.0  # Keep features above this importance score

class FeatureEngineer:
    """Feature engineering for financial time series data."""
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """Initialize with optional configuration."""
        self.config = config if config is not None else FeatureConfig()
        self.scaler = None
        self.feature_importances_ = None
        self.selected_features_ = None
    
    def fit(self, df: pd.DataFrame) -> 'FeatureEngineer':
        """Fit the feature engineer on the training data."""
        # Calculate feature importances if needed
        if self.config.feature_importance_threshold > 0:
            self._calculate_feature_importances(df)
        
        # Fit scaler if needed
        if self.config.scale_features and self.config.scaler_type != 'none':
            self._fit_scaler(df)
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering transformations."""
        df = df.copy()
        
        # Add technical indicators
        df = self._add_technical_indicators(df)
        
        # Add statistical features
        df = self._add_statistical_features(df)
        
        # Add time-based features
        if self.config.use_time_features:
            df = self._add_time_features(df)
        
        # Scale features if enabled
        if self.config.scale_features and self.scaler is not None:
            df = self._scale_features(df)
        
        # Select features based on importance
        if self.selected_features_ is not None:
            df = df[self.selected_features_]
        
        return df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform the data."""
        return self.fit(df).transform(df)
    
    def create_target(self, prices: pd.Series) -> pd.Series:
        """Create target variable for supervised learning."""
        # Calculate future returns
        future_returns = prices.pct_change(self.config.target_lookahead).shift(-self.config.target_lookahead)
        
        # Create binary target (1 for positive return, 0 otherwise)
        target = (future_returns > self.config.target_threshold).astype(int)
        
        # Drop NaN values
        target = target[target.notna()]
        
        return target
    
    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the dataframe."""
        # Ensure we have required columns
        if 'close' not in df.columns:
            raise ValueError("DataFrame must contain 'close' column for technical indicators")
        
        # RSI (Relative Strength Index)
        if self.config.use_rsi:
            df['rsi'] = talib.RSI(df['close'], timeperiod=14)
        
        # MACD (Moving Average Convergence Divergence)
        if self.config.use_macd:
            macd, macd_signal, _ = talib.MACD(df['close'])
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_hist'] = macd - macd_signal
        
        # Bollinger Bands
        if self.config.use_bollinger:
            upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20)
            df['bb_upper'] = upper
            df['bb_middle'] = middle
            df['bb_lower'] = lower
            df['bb_width'] = (upper - lower) / middle  # Bollinger Band Width
        
        # ATR (Average True Range)
        if self.config.use_atr and 'high' in df.columns and 'low' in df.columns:
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        
        # OBV (On-Balance Volume)
        if self.config.use_obv and 'volume' in df.columns:
            df['obv'] = talib.OBV(df['close'], df['volume'])
        
        # EMA (Exponential Moving Average)
        if self.config.use_ema:
            df['ema_12'] = talib.EMA(df['close'], timeperiod=12)
            df['ema_26'] = talib.EMA(df['close'], timeperiod=26)
            df['ema_50'] = talib.EMA(df['close'], timeperiod=50)
            df['ema_200'] = talib.EMA(df['close'], timeperiod=200)
        
        # Stochastic Oscillator
        if self.config.use_stoch and 'high' in df.columns and 'low' in df.columns:
            slowk, slowd = talib.STOCH(
                df['high'], df['low'], df['close'],
                fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0
            )
            df['stoch_k'] = slowk
            df['stoch_d'] = slowd
        
        # ADX (Average Directional Index)
        if self.config.use_adx and 'high' in df.columns and 'low' in df.columns:
            df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        
        # CCI (Commodity Channel Index)
        if self.config.use_cci and 'high' in df.columns and 'low' in df.columns:
            df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=14)
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features to the dataframe."""
        # Returns
        if self.config.use_returns:
            df['return'] = df['close'].pct_change()
            df['log_return'] = np.log1p(df['return'])
            
            # Rolling returns
            for window in [5, 10, 20, 50]:
                df[f'return_{window}d'] = df['close'].pct_change(window)
        
        # Volatility
        if self.config.use_volatility and 'return' in df.columns:
            for window in [5, 10, 20, 50]:
                df[f'volatility_{window}d'] = df['return'].rolling(window).std() * np.sqrt(252)  # Annualized
        
        # Skewness and kurtosis
        if self.config.use_skew and 'return' in df.columns:
            for window in [20, 50, 100]:
                df[f'skew_{window}d'] = df['return'].rolling(window).skew()
        
        if self.config.use_kurtosis and 'return' in df.columns:
            for window in [20, 50, 100]:
                df[f'kurtosis_{window}d'] = df['return'].rolling(window).kurt()
        
        # Correlation with market (if available)
        if self.config.use_correlation and 'market_return' in df.columns and 'return' in df.columns:
            for window in [20, 50, 100]:
                df[f'corr_market_{window}d'] = df['return'].rolling(window).corr(df['market_return'])
        
        return df
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features to the dataframe."""
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex for time features")
        
        # Time of day features
        if self.config.use_hour:
            df['hour'] = df.index.hour
        
        # Day of week
        if self.config.use_weekday:
            df['weekday'] = df.index.weekday
        
        # Month of year
        if self.config.use_month:
            df['month'] = df.index.month
        
        # Quarter end flag
        if self.config.use_is_quarter_end:
            df['is_quarter_end'] = (df.index + pd.offsets.MonthEnd(0)).month % 3 == 0
        
        return df
    
    def _fit_scaler(self, df: pd.DataFrame) -> None:
        """Fit the feature scaler."""
        if self.config.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.config.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.config.scaler_type == 'robust':
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.config.scaler_type}")
        
        # Fit the scaler on numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        self.scaler.fit(df[numeric_cols])
    
    def _scale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale features using the fitted scaler."""
        if self.scaler is None:
            raise RuntimeError("Scaler has not been fitted. Call fit() first.")
        
        # Scale numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        scaled_values = self.scaler.transform(df[numeric_cols])
        
        # Create a new DataFrame with scaled values
        df_scaled = df.copy()
        df_scaled[numeric_cols] = scaled_values
        
        return df_scaled
    
    def _calculate_feature_importances(self, df: pd.DataFrame) -> None:
        """Calculate feature importances using a simple model."""
        from sklearn.ensemble import RandomForestClassifier
        
        # Create target variable
        target = self.create_target(df['close'])
        
        # Align features and target
        features = df.drop(columns=['close']).select_dtypes(include=[np.number])
        features, target = self._align_features_and_target(features, target)
        
        # Train a simple model to get feature importances
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(features, target)
        
        # Store feature importances
        self.feature_importances_ = pd.Series(
            model.feature_importances_,
            index=features.columns
        ).sort_values(ascending=False)
        
        # Select features above threshold
        self.selected_features_ = self.feature_importances_[
            self.feature_importances_ > self.config.feature_importance_threshold
        ].index.tolist()
    
    @staticmethod
    def _align_features_and_target(
        features: pd.DataFrame,
        target: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Align features and target, handling missing values."""
        # Combine features and target into a single DataFrame
        df = features.copy()
        df['target'] = target
        
        # Drop rows with any NaN values
        df = df.dropna()
        
        # Separate features and target
        return df.drop(columns=['target']), df['target']

def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    task: str = 'classification'
) -> Dict[str, float]:
    """
    Calculate evaluation metrics for classification or regression.
    
    Args:
        y_true: True target values
        y_pred: Predicted values
        y_prob: Predicted probabilities (for classification)
        task: 'classification' or 'regression'
        
    Returns:
        Dictionary of metrics
    """
    if task == 'classification':
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
        }
        
        if y_prob is not None:
            try:
                metrics.update({
                    'roc_auc': roc_auc_score(y_true, y_prob),
                    'pr_auc': average_precision_score(y_true, y_prob)
                })
            except Exception as e:
                logger.warning(f"Could not calculate AUC metrics: {e}")
    
    else:  # regression
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': _mean_absolute_percentage_error(y_true, y_pred)
        }
    
    return metrics

def _mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate MAPE (Mean Absolute Percentage Error)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0  # Avoid division by zero
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def create_lookback_dataset(
    X: np.ndarray,
    y: np.ndarray,
    lookback: int = 10,
    forecast_horizon: int = 1,
    step: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a dataset with lookback windows for time series forecasting.
    
    Args:
        X: Input features (n_samples, n_features)
        y: Target values (n_samples,)
        lookback: Number of time steps to look back
        forecast_horizon: Number of time steps to forecast ahead
        step: Step size between windows
        
    Returns:
        Tuple of (X_windows, y_windows) where:
        - X_windows: Shape (n_windows, lookback, n_features)
        - y_windows: Shape (n_windows, forecast_horizon)
    """
    n_samples, n_features = X.shape
    X_windows = []
    y_windows = []
    
    for i in range(0, n_samples - lookback - forecast_horizon + 1, step):
        X_windows.append(X[i:(i + lookback)])
        y_windows.append(y[(i + lookback):(i + lookback + forecast_horizon)])
    
    return np.array(X_windows), np.array(y_windows)

def plot_feature_importance(
    importances: Union[np.ndarray, pd.Series],
    feature_names: Optional[List[str]] = None,
    top_n: int = 20,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Plot feature importances.
    
    Args:
        importances: Feature importances (either array or Series with feature names as index)
        feature_names: List of feature names (required if importances is array)
        top_n: Number of top features to show
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if isinstance(importances, pd.Series):
        importance_df = importances.sort_values(ascending=False).head(top_n)
    else:
        if feature_names is None or len(feature_names) != len(importances):
            raise ValueError("feature_names must be provided and match the length of importances")
        importance_df = pd.Series(importances, index=feature_names)
        importance_df = importance_df.sort_values(ascending=False).head(top_n)
    
    fig, ax = plt.subplots(figsize=figsize)
    importance_df.plot(kind='barh', ax=ax)
    ax.set_title('Feature Importances')
    ax.set_xlabel('Importance')
    plt.tight_layout()
    
    return fig

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: Optional[List[str]] = None,
    normalize: bool = False,
    cmap: str = 'Blues',
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot a confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: List of class names
        normalize: Whether to normalize the confusion matrix
        cmap: Colormap for the plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if classes is None:
        classes = np.unique(np.concatenate([y_true, y_pred]))
    
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, annot=True, fmt='.2f' if normalize else 'd',
        cmap=cmap, square=True, cbar=True,
        xticklabels=classes, yticklabels=classes,
        ax=ax
    )
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.tight_layout()
    
    return fig

def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot ROC curve.
    
    Args:
        y_true: True binary labels
        y_prob: Target scores (probability estimates of the positive class)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc='lower right')
    plt.tight_layout()
    
    return fig

def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot precision-recall curve.
    
    Args:
        y_true: True binary labels
        y_prob: Target scores (probability estimates of the positive class)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.step(recall, precision, where='post', color='b', alpha=0.2, lw=2)
    ax.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.set_title(f'Precision-Recall Curve (AP = {avg_precision:.2f})')
    plt.tight_layout()
    
    return fig
