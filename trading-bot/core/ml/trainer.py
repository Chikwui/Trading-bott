"""
ML Model Trainer for Trading

This module provides functionality to train, validate, and optimize machine learning models
for trading using historical MT5 data.
"""
# Standard library imports
import json
import logging
import os
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

# Third-party imports
import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score
)
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Dense, LSTM, GRU, Dropout, BatchNormalization, Input,
    MultiHeadAttention, LayerNormalization, Add, Flatten, Conv1D, MaxPooling1D
)
from tensorflow.keras.optimizers import Adam, RMSprop, Nadam
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
)
from tensorflow.keras.regularizers import l1_l2

# Local imports
from core.ml.models import (
    BaseModel, XGBoostModel, LightGBMModel, LSTMModel, TransformerModel
)
from core.data.mt5_fetcher import MT5Fetcher, MT5Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

@dataclass
class ModelConfig:
    """Configuration for model training and evaluation."""
    # Model parameters
    model_type: str = 'xgboost'  # xgboost, lightgbm, lstm, transformer
    task_type: str = 'regression'  # regression or classification
    
    # Feature engineering
    lookback: int = 50
    horizon: int = 5
    target_column: str = 'target'
    feature_columns: List[str] = field(default_factory=lambda: [
        'open', 'high', 'low', 'close', 'volume',
        'returns', 'volatility', 'rsi', 'macd', 'macd_signal',
        'bb_upper', 'bb_middle', 'bb_lower', 'atr', 'vwap'
    ])
    
    # Training parameters
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42
    n_trials: int = 100
    n_jobs: int = -1
    
    # Model hyperparameters
    model_params: Dict[str, Any] = field(default_factory=dict)
    
    # Paths
    model_dir: str = "models"
    tensorboard_log_dir: str = "logs/tensorboard"
    
    def __post_init__(self):
        """Initialize directories and validate configuration."""
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.tensorboard_log_dir, exist_ok=True)
        
        if self.task_type not in ['regression', 'classification']:
            raise ValueError("task_type must be either 'regression' or 'classification'")


class DataPreprocessor:
    """Handles data preprocessing and feature engineering for trading data."""
    
    def __init__(
        self,
        feature_columns: List[str],
        target_column: str,
        timestamp_column: str = 'timestamp',
        scale_features: bool = True,
        scale_target: bool = False,
        scaler_type: str = 'standard',
        sequence_length: int = 10,
        test_size: float = 0.2,
        val_size: float = 0.1,
        random_state: int = 42
    ):
        """Initialize the data preprocessor.
        
        Args:
            feature_columns: List of feature column names
            target_column: Name of the target column
            timestamp_column: Name of the timestamp column
            scale_features: Whether to scale features
            scale_target: Whether to scale target variable
            scaler_type: Type of scaler ('standard', 'minmax', 'robust')
            sequence_length: Length of sequences for time series models
            test_size: Proportion of data to use for testing
            val_size: Proportion of training data to use for validation
            random_state: Random seed for reproducibility
        """
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.timestamp_column = timestamp_column
        self.scale_features = scale_features
        self.scale_target = scale_target
        self.scaler_type = scaler_type
        self.sequence_length = sequence_length
        self.test_size = test_size
        self.val_size = val_size
        self.random_state = random_state
        
        # Initialize scalers
        self.feature_scaler = self._get_scaler(scaler_type)
        self.target_scaler = self._get_scaler(scaler_type) if scale_target else None
        
        # State variables
        self._is_fitted = False
    
    def _get_scaler(self, scaler_type: str):
        """Get the appropriate scaler based on type."""
        if scaler_type == 'standard':
            return StandardScaler()
        elif scaler_type == 'minmax':
            return MinMaxScaler()
        elif scaler_type == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"Unsupported scaler type: {scaler_type}")
    
    def preprocess(
        self,
        data: pd.DataFrame,
        is_training: bool = True
    ) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
        """Preprocess the input data.
        
        Args:
            data: Input DataFrame with raw data
            is_training: Whether this is training data
            
        Returns:
            Dictionary containing preprocessed data and metadata
        """
        logger.info("Starting data preprocessing...")
        
        # Make a copy to avoid modifying the original data
        df = data.copy()
        
        # Ensure timestamp is in datetime format
        if self.timestamp_column in df.columns:
            df[self.timestamp_column] = pd.to_datetime(df[self.timestamp_column])
            df = df.sort_values(by=self.timestamp_column)
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Feature engineering
        df = self._engineer_features(df)
        
        # Split into features and target
        X = df[self.feature_columns].values
        y = df[self.target_column].values
        
        # Scale features
        if self.scale_features:
            if is_training or not self._is_fitted:
                X = self.feature_scaler.fit_transform(X)
                self._is_fitted = True
            else:
                X = self.feature_scaler.transform(X)
        
        # Scale target if needed
        if self.scale_target:
            if is_training or not self._is_fitted:
                y = self.target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
            else:
                y = self.target_scaler.transform(y.reshape(-1, 1)).flatten()
        
        # Split into train/validation/test sets
        if is_training:
            return self._train_val_test_split(X, y, df)
        else:
            return {
                'X': X,
                'y': y,
                'timestamps': df[self.timestamp_column].values if self.timestamp_column in df.columns else None,
                'feature_names': self.feature_columns,
                'target_name': self.target_column
            }
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the DataFrame."""
        # Forward fill for time series data
        df = df.ffill()
        
        # If there are still missing values, fill with mean for numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mean())
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer additional features for the model."""
        # Add technical indicators as features
        if 'close' in df.columns:
            # Simple Moving Averages
            df['sma_5'] = df['close'].rolling(window=5).mean()
            df['sma_20'] = df['close'].rolling(window=20).mean()
            
            # Exponential Moving Averages
            df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['bb_upper'] = df['close'].rolling(window=20).mean() + 2 * df['close'].rolling(window=20).std()
            df['bb_lower'] = df['close'].rolling(window=20).mean() - 2 * df['close'].rolling(window=20).std()
            
            # Update feature columns with new features
            new_features = ['sma_5', 'sma_20', 'ema_12', 'ema_26', 'macd', 'macd_signal', 'rsi', 'bb_upper', 'bb_lower']
            self.feature_columns = list(set(self.feature_columns + new_features))
        
        # Add time-based features
        if self.timestamp_column in df.columns:
            df['hour'] = df[self.timestamp_column].dt.hour
            df['day_of_week'] = df[self.timestamp_column].dt.dayofweek
            df['day_of_month'] = df[self.timestamp_column].dt.day
            df['month'] = df[self.timestamp_column].dt.month
            
            # Update feature columns with time features
            time_features = ['hour', 'day_of_week', 'day_of_month', 'month']
            self.feature_columns = list(set(self.feature_columns + time_features))
        
        # Drop any remaining NaN values that might have been introduced
        df = df.dropna()
        
        return df
    
    def _train_val_test_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        df: pd.DataFrame
    ) -> Dict[str, Union[np.ndarray, pd.DataFrame]]:
        """Split data into training, validation, and test sets."""
        # For time series data, we need to maintain temporal order
        n_samples = len(X)
        test_size = int(n_samples * self.test_size)
        val_size = int((n_samples - test_size) * self.val_size)
        
        # Split into train/validation/test
        X_train, X_val, X_test = X[:-(test_size + val_size)], X[-(test_size + val_size):-test_size], X[-test_size:]
        y_train, y_val, y_test = y[:-(test_size + val_size)], y[-(test_size + val_size):-test_size], y[-test_size:]
        
        # Get timestamps if available
        if self.timestamp_column in df.columns:
            timestamps = df[self.timestamp_column].values
            timestamps_train = timestamps[:-(test_size + val_size)]
            timestamps_val = timestamps[-(test_size + val_size):-test_size]
            timestamps_test = timestamps[-test_size:]
        else:
            timestamps_train = timestamps_val = timestamps_test = None
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'timestamps_train': timestamps_train,
            'timestamps_val': timestamps_val,
            'timestamps_test': timestamps_test,
            'feature_names': self.feature_columns,
            'target_name': self.target_column
        }
    
    def create_sequences(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series models.
        
        Args:
            X: Input features (n_samples, n_features)
            y: Target values (n_samples,)
            
        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        X_sequences, y_sequences = [], []
        
        for i in range(len(X) - self.sequence_length):
            X_sequences.append(X[i:(i + self.sequence_length)])
            y_sequences.append(y[i + self.sequence_length - 1])
        
        return np.array(X_sequences), np.array(y_sequences)
    
    def save(self, path: str) -> None:
        """Save the preprocessor to disk."""
        state = {
            'feature_columns': self.feature_columns,
            'target_column': self.target_column,
            'timestamp_column': self.timestamp_column,
            'scale_features': self.scale_features,
            'scale_target': self.scale_target,
            'scaler_type': self.scaler_type,
            'sequence_length': self.sequence_length,
            'test_size': self.test_size,
            'val_size': self.val_size,
            'random_state': self.random_state,
            'is_fitted': self._is_fitted,
            'feature_scaler': joblib.dumps(self.feature_scaler) if self._is_fitted else None,
            'target_scaler': joblib.dumps(self.target_scaler) if (self._is_fitted and self.scale_target) else None
        }
        
        with open(path, 'wb') as f:
            joblib.dump(state, f)
    
    @classmethod
    def load(cls, path: str) -> 'DataPreprocessor':
        """Load a preprocessor from disk."""
        with open(path, 'rb') as f:
            state = joblib.load(f)
        
        # Create a new instance
        preprocessor = cls(
            feature_columns=state['feature_columns'],
            target_column=state['target_column'],
            timestamp_column=state['timestamp_column'],
            scale_features=state['scale_features'],
            scale_target=state['scale_target'],
            scaler_type=state['scaler_type'],
            sequence_length=state['sequence_length'],
            test_size=state['test_size'],
            val_size=state['val_size'],
            random_state=state['random_state']
        )
        
        # Restore state
        preprocessor._is_fitted = state['is_fitted']
        
        if preprocessor._is_fitted:
            preprocessor.feature_scaler = joblib.loads(state['feature_scaler'])
            if preprocessor.scale_target:
                preprocessor.target_scaler = joblib.loads(state['target_scaler'])
        
        return preprocessor


class ModelTrainer:
    """Handles model training, validation, and evaluation."""
    
    def __init__(
        self,
        model_class: type,
        model_params: Dict[str, Any],
        preprocessor: DataPreprocessor,
        task_type: str = 'regression',
        output_dir: str = 'models',
        experiment_name: str = None,
        use_sequences: bool = False
    ):
        """Initialize the model trainer.
        
        Args:
            model_class: The model class to use (e.g., XGBoostModel, LSTMModel)
            model_params: Parameters to pass to the model constructor
            preprocessor: Preprocessor for the data
            task_type: Type of task ('regression' or 'classification')
            output_dir: Directory to save model outputs
            experiment_name: Name for the experiment (used for saving results)
            use_sequences: Whether to use sequences for time series models
        """
        self.model_class = model_class
        self.model_params = model_params
        self.preprocessor = preprocessor
        self.task_type = task_type
        self.use_sequences = use_sequences
        
        # Set up output directory
        self.experiment_name = experiment_name or f"{model_class.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.output_dir = Path(output_dir) / self.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = self._init_model()
        
        # Training history
        self.history = None
        self.evaluation_metrics = None
    
    def _init_model(self) -> BaseModel:
        """Initialize the model with the given parameters."""
        return self.model_class(**self.model_params)
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        **fit_params
    ) -> Dict[str, Any]:
        """Train the model on the given data.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            **fit_params: Additional parameters to pass to the model's fit method
            
        Returns:
            Dictionary containing training history and metrics
        """
        logger.info(f"Training {self.model_class.__name__}...")
        
        # Create sequences if needed
        if self.use_sequences:
            X_train_seq, y_train_seq = self.preprocessor.create_sequences(X_train, y_train)
            if X_val is not None and y_val is not None:
                X_val_seq, y_val_seq = self.preprocessor.create_sequences(X_val, y_val)
                val_data = (X_val_seq, y_val_seq)
            else:
                val_data = None
            
            # Train the model
            self.history = self.model.fit(
                X_train_seq, y_train_seq,
                X_val=val_data,
                **fit_params
            )
        else:
            # Train the model
            val_data = (X_val, y_val) if (X_val is not None and y_val is not None) else None
            self.history = self.model.fit(
                X_train, y_train,
                X_val=val_data,
                **fit_params
            )
        
        # Save the trained model
        self.save_model()
        
        return self.history
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        set_name: str = 'test'
    ) -> Dict[str, float]:
        """Evaluate the model on the given data.
        
        Args:
            X: Input features
            y: True target values
            set_name: Name of the dataset (e.g., 'train', 'val', 'test')
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.use_sequences:
            X_seq, y_true = self.preprocessor.create_sequences(X, y)
        else:
            X_seq, y_true = X, y
        
        # Make predictions
        y_pred = self.model.predict(X_seq)
        
        # Calculate metrics
        metrics = {}
        
        if self.task_type == 'regression':
            metrics.update({
                f'{set_name}_mse': mean_squared_error(y_true, y_pred),
                f'{set_name}_rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                f'{set_name}_mae': mean_absolute_error(y_true, y_pred),
                f'{set_name}_r2': r2_score(y_true, y_pred)
            })
        else:  # classification
            y_pred_bin = (y_pred > 0.5).astype(int)
            
            metrics.update({
                f'{set_name}_accuracy': accuracy_score(y_true, y_pred_bin),
                f'{set_name}_precision': precision_score(y_true, y_pred_bin, zero_division=0),
                f'{set_name}_recall': recall_score(y_true, y_pred_bin, zero_division=0),
                f'{set_name}_f1': f1_score(y_true, y_pred_bin, zero_division=0),
                f'{set_name}_roc_auc': roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else float('nan')
            })
        
        # Save metrics
        if self.evaluation_metrics is None:
            self.evaluation_metrics = {}
        self.evaluation_metrics.update(metrics)
        
        # Save metrics to file
        metrics_file = self.output_dir / 'evaluation_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(self.evaluation_metrics, f, indent=2)
        
        logger.info(f"{set_name.capitalize()} metrics: {metrics}")
        return metrics
    
    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 5,
        **fit_params
    ) -> Dict[str, List[float]]:
        """Perform time series cross-validation.
        
        Args:
            X: Input features
            y: Target values
            n_splits: Number of cross-validation folds
            **fit_params: Additional parameters to pass to the fit method
            
        Returns:
            Dictionary of cross-validation metrics
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        metrics = {}
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            logger.info(f"Training fold {fold + 1}/{n_splits}...")
            
            # Split data
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            self.model = self._init_model()  # Reset model for each fold
            self.train(X_train, y_train, X_val, y_val, **fit_params)
            
            # Evaluate on validation set
            fold_metrics = self.evaluate(X_val, y_val, f'fold_{fold + 1}')
            
            # Update metrics
            for metric_name, value in fold_metrics.items():
                if metric_name not in metrics:
                    metrics[metric_name] = []
                metrics[metric_name].append(value)
        
        # Calculate mean and std of metrics
        cv_metrics = {}
        for metric_name, values in metrics.items():
            cv_metrics[f'cv_mean_{metric_name}'] = np.mean(values)
            cv_metrics[f'cv_std_{metric_name}'] = np.std(values)
        
        # Save CV metrics
        if self.evaluation_metrics is None:
            self.evaluation_metrics = {}
        self.evaluation_metrics.update(cv_metrics)
        
        # Save metrics to file
        metrics_file = self.output_dir / 'cv_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(cv_metrics, f, indent=2)
        
        logger.info(f"Cross-validation metrics: {cv_metrics}")
        return cv_metrics
    
    def save_model(self) -> None:
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No model has been trained yet.")
        
        # Save model
        model_file = self.output_dir / 'model'
        self.model.save(str(model_file))
        
        # Save preprocessor
        preprocessor_file = self.output_dir / 'preprocessor.pkl'
        self.preprocessor.save(str(preprocessor_file))
        
        # Save model info
        model_info = {
            'model_class': self.model_class.__name__,
            'model_params': self.model_params,
            'task_type': self.task_type,
            'use_sequences': self.use_sequences,
            'feature_columns': self.preprocessor.feature_columns,
            'target_column': self.preprocessor.target_column,
            'timestamp_column': self.preprocessor.timestamp_column,
            'created_at': datetime.now().isoformat()
        }
        
        with open(self.output_dir / 'model_info.json', 'w') as f:
            json.dump(model_info, f, indent=2)
        
        logger.info(f"Model saved to {self.output_dir}")
    
    @classmethod
    def load_model(cls, model_dir: str) -> 'ModelTrainer':
        """Load a trained model from disk.
        
        Args:
            model_dir: Directory containing the saved model
            
        Returns:
            Loaded ModelTrainer instance
        """
        model_dir = Path(model_dir)
        
        # Load model info
        with open(model_dir / 'model_info.json', 'r') as f:
            model_info = json.load(f)
        
        # Get model class
        model_class = {
            'XGBoostModel': XGBoostModel,
            'LightGBMModel': LightGBMModel,
            'LSTMModel': LSTMModel,
            'TransformerModel': TransformerModel
        }[model_info['model_class']]
        
        # Create preprocessor
        preprocessor = DataPreprocessor.load(model_dir / 'preprocessor.pkl')
        
        # Create trainer instance
        trainer = cls(
            model_class=model_class,
            model_params=model_info['model_params'],
            preprocessor=preprocessor,
            task_type=model_info['task_type'],
            output_dir=str(model_dir.parent),
            experiment_name=model_dir.name,
            use_sequences=model_info.get('use_sequences', False)
        )
        
        # Load model weights
        trainer.model = model_class.load(str(model_dir / 'model'))
        
        # Load metrics if available
        metrics_file = model_dir / 'evaluation_metrics.json'
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                trainer.evaluation_metrics = json.load(f)
        
        return trainer


def train_pipeline(
    data: pd.DataFrame,
    model_class: type,
    model_params: Dict[str, Any],
    feature_columns: List[str],
    target_column: str,
    task_type: str = 'regression',
    use_sequences: bool = False,
    sequence_length: int = 10,
    output_dir: str = 'models',
    experiment_name: str = None,
    cross_validate: bool = False,
    n_splits: int = 5,
    **preprocessor_kwargs
) -> ModelTrainer:
    """Complete training pipeline.
    
    Args:
        data: Input DataFrame with features and target
        model_class: Model class to use
        model_params: Parameters for the model
        feature_columns: List of feature column names
        target_column: Name of the target column
        task_type: Type of task ('regression' or 'classification')
        use_sequences: Whether to use sequences for time series models
        sequence_length: Length of sequences for time series models
        output_dir: Directory to save model outputs
        experiment_name: Name for the experiment
        cross_validate: Whether to perform cross-validation
        n_splits: Number of cross-validation folds
        **preprocessor_kwargs: Additional arguments for DataPreprocessor
        
    Returns:
        Trained ModelTrainer instance
    """
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        feature_columns=feature_columns,
        target_column=target_column,
        sequence_length=sequence_length,
        **preprocessor_kwargs
    )
    
    # Preprocess data
    processed_data = preprocessor.preprocess(data)
    
    # Initialize trainer
    trainer = ModelTrainer(
        model_class=model_class,
        model_params=model_params,
        preprocessor=preprocessor,
        task_type=task_type,
        output_dir=output_dir,
        experiment_name=experiment_name,
        use_sequences=use_sequences
    )
    
    # Train model
    trainer.train(
        processed_data['X_train'],
        processed_data['y_train'],
        processed_data.get('X_val'),
        processed_data.get('y_val')
    )
    
    # Evaluate on test set
    if 'X_test' in processed_data and 'y_test' in processed_data:
        trainer.evaluate(
            processed_data['X_test'],
            processed_data['y_test'],
            'test'
        )
    
    # Perform cross-validation if requested
    if cross_validate:
        trainer.cross_validate(
            np.vstack([processed_data['X_train'], processed_data.get('X_val', np.array([]))]),
            np.concatenate([processed_data['y_train'], processed_data.get('y_val', np.array([]))]),
            n_splits=n_splits
        )
    
    return trainer


def train_model_on_mt5_data(
    symbol: str,
    timeframe: str = "1h",
    lookback: int = 50,
    horizon: int = 5,
    model_type: str = "xgboost",
    task_type: str = "regression",
    from_date: str = "2020-01-01",
    to_date: str = "2023-12-31",
    test_size: float = 0.2,
    n_trials: int = 50
) -> ModelTrainer:
    """
    Train a model on MT5 data with hyperparameter tuning.
    
    Args:
        symbol: Trading symbol (e.g., 'EURUSD')
        timeframe: Timeframe for the data (e.g., '1h', '4h', '1d')
        lookback: Number of lookback periods for features
        horizon: Number of periods ahead to predict
        model_type: Type of model to train ('xgboost', 'lightgbm', 'lstm', 'transformer')
        task_type: Type of task ('regression' or 'classification')
        from_date: Start date for training data
        to_date: End date for training data
        test_size: Fraction of data to use for testing
        n_trials: Number of hyperparameter optimization trials
        
    Returns:
        Trained ModelTrainer instance
    """
    # Initialize MT5 fetcher
    mt5_config = MT5Config()
    fetcher = MT5Fetcher(mt5_config)
    
    try:
        # Fetch data
        df = fetcher.fetch_rates(
            symbol=symbol,
            timeframe=timeframe,
            from_date=from_date,
            to_date=to_date,
            save_raw=True
        )
        
        if df is None or df.empty:
            raise ValueError(f"No data found for {symbol} {timeframe} from {from_date} to {to_date}")
        
        # Preprocess data
        X, y = fetcher.preprocess_data(
            df=df,
            symbol=symbol,
            timeframe=timeframe,
            lookback=lookback,
            horizon=horizon,
            save_processed=True
        )
        
        if X.empty or y.empty:
            raise ValueError("Failed to preprocess data")
        
        # Configure model
        config = ModelConfig(
            model_type=model_type,
            task_type=task_type,
            lookback=lookback,
            horizon=horizon,
            test_size=test_size,
            n_trials=n_trials
        )
        
        # Initialize trainer
        trainer = ModelTrainer(
            model_class={
                'xgboost': XGBoostModel,
                'lightgbm': LightGBMModel,
                'lstm': LSTMModel,
                'transformer': TransformerModel
            }[model_type],
            model_params=config.model_params,
            preprocessor=DataPreprocessor(
                feature_columns=config.feature_columns,
                target_column=config.target_column,
                sequence_length=config.lookback,
                test_size=config.test_size
            ),
            task_type=config.task_type,
            output_dir=config.model_dir,
            experiment_name=f"{model_type}_{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            use_sequences=config.model_type in ['lstm', 'transformer']
        )
        
        # Split data
        X_train, X_test, y_train, y_test = trainer.preprocessor._train_val_test_split(X, y, pd.DataFrame())
        
        # Perform hyperparameter tuning
        logger.info(f"Starting hyperparameter tuning for {model_type} model...")
        tuning_results = trainer.hyperparameter_tuning(
            X_train, y_train,
            n_trials=n_trials,
            direction='minimize' if task_type == 'regression' else 'maximize'
        )
        
        # Train final model with best parameters
        logger.info("Training final model with best parameters...")
        trainer.train(X_train, y_train, X_test, y_test)
        
        # Evaluate on test set
        test_metrics = trainer.evaluate(X_test, y_test, prefix='test_')
        logger.info(f"Test metrics: {test_metrics}")
        
        # Save model
        model_path = trainer.save_model()
        logger.info(f"Model saved to {model_path}")
        
        return trainer
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}", exc_info=True)
        raise
    finally:
        fetcher.disconnect()


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description='Train a model on MT5 data')
    parser.add_argument('--symbol', type=str, default='EURUSD', help='Trading symbol')
    parser.add_argument('--timeframe', type=str, default='1h', help='Timeframe')
    parser.add_argument('--model', type=str, default='xgboost', 
                        choices=['xgboost', 'lightgbm', 'lstm', 'transformer'],
                        help='Model type')
    parser.add_argument('--task', type=str, default='regression',
                        choices=['regression', 'classification'],
                        help='Task type')
    parser.add_argument('--lookback', type=int, default=50, help='Lookback periods')
    parser.add_argument('--horizon', type=int, default=5, help='Forecast horizon')
    parser.add_argument('--from_date', type=str, default='2020-01-01', help='Start date')
    parser.add_argument('--to_date', type=str, default='2023-12-31', help='End date')
    parser.add_argument('--trials', type=int, default=50, help='Number of hyperparameter trials')
    
    args = parser.parse_args()
    
    # Train model
    trainer = train_model_on_mt5_data(
        symbol=args.symbol,
        timeframe=args.timeframe,
        lookback=args.lookback,
        horizon=args.horizon,
        model_type=args.model,
        task_type=args.task,
        from_date=args.from_date,
        to_date=args.to_date,
        n_trials=args.trials
    )
