"""
Model Training Pipeline

This module implements a flexible pipeline for training, validating, and evaluating
machine learning models for trading applications.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import yaml
import logging
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
import optuna
from optuna.samplers import TPESampler

from .models import (
    BaseModel,
    XGBoostModel,
    LightGBMModel,
    LSTMModel,
    TransformerModel
)

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for model training and evaluation."""
    model_type: str  # xgboost, lightgbm, lstm, transformer
    params: Dict[str, Any] = field(default_factory=dict)
    features: List[str] = field(default_factory=list)
    target: str = 'target'
    task: str = 'classification'  # 'classification' or 'regression'
    train_test_split: float = 0.8
    cross_validation_folds: int = 5
    early_stopping_rounds: int = 50
    random_state: int = 42
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_dir: str = 'models'
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.model_type in ['xgboost', 'lightgbm', 'lstm', 'transformer'], \
            f"Unsupported model type: {self.model_type}"
        assert 0 < self.train_test_split < 1, "train_test_split must be between 0 and 1"
        assert self.cross_validation_folds > 1, "cross_validation_folds must be > 1"
        
        # Create output directory if it doesn't exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

class ModelPipeline:
    """End-to-end pipeline for training and evaluating ML models."""
    
    def __init__(self, config: ModelConfig):
        """Initialize the pipeline with configuration."""
        self.config = config
        self.model = self._init_model()
        self.feature_importances_ = None
        self.cv_results_ = None
        
    def _init_model(self) -> BaseModel:
        """Initialize the appropriate model based on configuration."""
        model_map = {
            'xgboost': XGBoostModel,
            'lightgbm': LightGBMModel,
            'lstm': LSTMModel,
            'transformer': TransformerModel
        }
        return model_map[self.config.model_type](self.config)
    
    def train(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        eval_set: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> Dict[str, Any]:
        """
        Train the model on the provided data.
        
        Args:
            X: Feature matrix
            y: Target values
            eval_set: Optional validation set (X_val, y_val)
            
        Returns:
            Dictionary containing training metrics and results
        """
        logger.info(f"Training {self.config.model_type} model...")
        
        # Train the model
        train_metrics = self.model.fit(X, y, eval_set=eval_set)
        
        # Get feature importances if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances_ = self.model.feature_importances_
        
        return {
            'train_metrics': train_metrics,
            'feature_importances': self.feature_importances_
        }
    
    def cross_validate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        n_splits: int = 5
    ) -> Dict[str, Any]:
        """
        Perform time-series cross-validation.
        
        Args:
            X: Feature matrix
            y: Target values
            n_splits: Number of CV folds
            
        Returns:
            Dictionary containing CV metrics and results
        """
        logger.info(f"Performing {n_splits}-fold time series cross-validation...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train on this fold
            self.model = self._init_model()  # Fresh model for each fold
            self.model.fit(X_train, y_train, eval_set=(X_val, y_val))
            
            # Evaluate on validation set
            y_pred = self.model.predict(X_val)
            fold_metrics = self._calculate_metrics(y_val, y_pred)
            metrics.append(fold_metrics)
            
            logger.info(f"Fold {fold} metrics: {fold_metrics}")
        
        # Aggregate metrics across folds
        avg_metrics = {
            f'mean_{k}': np.mean([m[k] for m in metrics]) 
            for k in metrics[0].keys()
        }
        
        self.cv_results_ = {
            'fold_metrics': metrics,
            'mean_metrics': avg_metrics
        }
        
        return self.cv_results_
    
    def optimize_hyperparameters(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        param_space: Dict[str, Any],
        n_trials: int = 100,
        direction: str = 'maximize'
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            X: Feature matrix
            y: Target values
            param_space: Dictionary defining the parameter search space
            n_trials: Number of optimization trials
            direction: Optimization direction ('minimize' or 'maximize')
            
        Returns:
            Dictionary containing optimization results
        """
        def objective(trial):
            # Sample parameters
            params = {}
            for name, space in param_space.items():
                if space['type'] == 'categorical':
                    params[name] = trial.suggest_categorical(name, space['values'])
                elif space['type'] == 'float':
                    params[name] = trial.suggest_float(
                        name, space['low'], space['high'], log=space.get('log', False)
                    )
                elif space['type'] == 'int':
                    params[name] = trial.suggest_int(
                        name, space['low'], space['high'], log=space.get('log', False)
                    )
            
            # Update model with sampled parameters
            self.model = self._init_model()
            self.model.set_params(params)
            
            # Perform cross-validation
            cv_results = self.cross_validate(X, y, n_splits=3)  # Fewer folds for speed
            
            # Return the metric to optimize (e.g., accuracy, negative MSE)
            return cv_results['mean_metrics']['mean_accuracy']
        
        # Run optimization
        study = optuna.create_study(
            direction=direction,
            sampler=TPESampler(seed=self.config.random_state)
        )
        study.optimize(objective, n_trials=n_trials)
        
        # Update model with best parameters
        self.model = self._init_model()
        self.model.set_params(study.best_params)
        
        return {
            'best_params': study.best_params,
            'best_value': study.best_value,
            'trials': study.trials_dataframe()
        }
    
    def evaluate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray]
    ) -> Dict[str, float]:
        """
        Evaluate the model on test data.
        
        Args:
            X: Test feature matrix
            y: True target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.model.predict(X)
        return self._calculate_metrics(y, y_pred)
    
    def save(self, path: Optional[str] = None) -> str:
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save the model. If None, uses config output_dir.
            
        Returns:
            Path where the model was saved
        """
        if path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            path = f"{self.config.output_dir}/{self.config.model_type}_{timestamp}.joblib"
        
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(self.model, path)
        logger.info(f"Model saved to {path}")
        
        # Save config
        config_path = str(path).replace('.joblib', '_config.yaml')
        with open(config_path, 'w') as f:
            yaml.safe_dump(self.config.__dict__, f)
        
        return path
    
    @classmethod
    def load(cls, path: str) -> 'ModelPipeline':
        """
        Load a trained model from disk.
        
        Args:
            path: Path to the saved model
            
        Returns:
            Loaded ModelPipeline instance
        """
        # Load model
        model = joblib.load(path)
        
        # Load config
        config_path = str(path).replace('.joblib', '_config.yaml')
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Create config and pipeline
        config = ModelConfig(**config_dict)
        pipeline = cls(config)
        pipeline.model = model
        
        return pipeline
    
    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """Calculate evaluation metrics."""
        if self.config.task == 'classification':
            return {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }
        else:  # regression
            return {
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred)
            }
