"""
Model Implementations

This module contains the base model class and specific model implementations
for the trading system's machine learning pipeline.
"""
# Standard library imports
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple, Union

# Third-party imports
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from tensorflow.keras import Model, layers
from tensorflow.keras.callbacks import (
    EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
)
from tensorflow.keras.layers import (
    GRU, LSTM, Add, BatchNormalization, Conv1D, Dense, Dropout, Flatten, Input,
    Layer, LayerNormalization, MaxPooling1D, MultiHeadAttention
)
from tensorflow.keras.optimizers import Adam

class BaseModel(ABC):
    """Abstract base class for all model implementations."""
    
    @abstractmethod
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        X_val: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train the model on the given data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save the model to disk."""
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'BaseModel':
        """Load a saved model from disk."""
        pass
    
    def __str__(self) -> str:
        """String representation of the model."""
        return self.__class__.__name__


class XGBoostModel(BaseModel):
    """XGBoost model implementation."""
    
    def __init__(
        self,
        task_type: str = 'regression',
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        objective: Optional[str] = None,
        **kwargs
    ):
        """Initialize the XGBoost model.
        
        Args:
            task_type: Type of task ('regression' or 'classification')
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Learning rate
            objective: Custom objective function (overrides task_type if provided)
            **kwargs: Additional XGBoost parameters
        """
        self.task_type = task_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.objective = objective
        self.params = kwargs
        
        # Set default parameters based on task type
        if self.objective is None:
            if self.task_type == 'regression':
                self.objective = 'reg:squarederror'
            elif self.task_type == 'classification':
                self.objective = 'binary:logistic'
            else:
                raise ValueError(f"Unsupported task type: {self.task_type}")
        
        self.model = None
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        X_val: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train the XGBoost model."""
        # Prepare data
        dtrain = xgb.DMatrix(X, label=y)
        
        # Prepare validation data if provided
        eval_set = []
        if X_val is not None:
            X_val, y_val = X_val
            dval = xgb.DMatrix(X_val, label=y_val)
            eval_set = [(dtrain, 'train'), (dval, 'eval')]
        
        # Set parameters
        params = {
            'objective': self.objective,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'n_estimators': self.n_estimators,
            'verbosity': 1,
            'n_jobs': -1,
            **self.params
        }
        
        # Train model
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=eval_set,
            early_stopping_rounds=50 if X_val is not None else None,
            verbose_eval=kwargs.get('verbose', 10)
        )
        
        # Get feature importances
        self.feature_importances_ = self.model.get_score(importance_type='weight')
        
        return {
            'model': 'xgboost',
            'params': params,
            'feature_importances': self.feature_importances_
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the XGBoost model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
    
    def save(self, path: str) -> None:
        """Save the XGBoost model to disk."""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        self.model.save_model(path + ".json")
        
        # Save additional metadata
        metadata = {
            'task_type': self.task_type,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'objective': self.objective,
            'params': self.params,
            'feature_importances': self.feature_importances_
        }
        
        with open(path + ".meta", 'wb') as f:
            joblib.dump(metadata, f)
    
    @classmethod
    def load(cls, path: str) -> 'XGBoostModel':
        """Load a saved XGBoost model from disk."""
        # Load metadata
        with open(path + ".meta", 'rb') as f:
            metadata = joblib.load(f)
        
        # Create model instance
        model = cls(
            task_type=metadata['task_type'],
            n_estimators=metadata['n_estimators'],
            max_depth=metadata['max_depth'],
            learning_rate=metadata['learning_rate'],
            objective=metadata['objective'],
            **metadata['params']
        )
        
        # Load model
        model.model = xgb.Booster()
        model.model.load_model(path + ".json")
        model.feature_importances_ = metadata['feature_importances']
        
        return model


class LightGBMModel(BaseModel):
    """LightGBM model implementation."""
    
    def __init__(
        self,
        task_type: str = 'regression',
        n_estimators: int = 100,
        max_depth: int = -1,  # -1 means no limit
        learning_rate: float = 0.1,
        num_leaves: int = 31,
        objective: Optional[str] = None,
        **kwargs
    ):
        """Initialize the LightGBM model.
        
        Args:
            task_type: Type of task ('regression' or 'classification')
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth (-1 for no limit)
            learning_rate: Learning rate
            num_leaves: Maximum number of leaves in one tree
            objective: Custom objective function (overrides task_type if provided)
            **kwargs: Additional LightGBM parameters
        """
        self.task_type = task_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.objective = objective
        self.params = kwargs
        
        # Set default parameters based on task type
        if self.objective is None:
            if self.task_type == 'regression':
                self.objective = 'regression'
            elif self.task_type == 'classification':
                self.objective = 'binary'
            else:
                raise ValueError(f"Unsupported task type: {self.task_type}")
        
        self.model = None
        self.feature_importances_ = None
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        X_val: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train the LightGBM model."""
        # Prepare data
        train_data = lgb.Dataset(X, label=y, free_raw_data=False)
        
        # Prepare validation data if provided
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None:
            X_val, y_val = X_val
            valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data, free_raw_data=False)
            valid_sets.append(valid_data)
            valid_names.append('valid')
        
        # Set parameters
        params = {
            'objective': self.objective,
            'boosting_type': 'gbdt',  # dart, goss
            'num_leaves': self.num_leaves,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'n_estimators': self.n_estimators,
            'verbosity': 1,
            'n_jobs': -1,
            'importance_type': 'gain',
            **self.params
        }
        
        # Train model
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            early_stopping_rounds=50 if X_val is not None else None,
            verbose_eval=kwargs.get('verbose', 10)
        )
        
        # Get feature importances
        self.feature_importances_ = dict(zip(
            [f'f{i}' for i in range(X.shape[1])],
            self.model.feature_importance(importance_type='gain')
        ))
        
        return {
            'model': 'lightgbm',
            'params': params,
            'feature_importances': self.feature_importances_
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the LightGBM model."""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        return self.model.predict(X)
    
    def save(self, path: str) -> None:
        """Save the LightGBM model to disk."""
        if self.model is None:
            raise ValueError("Model has not been trained yet")
            
        self.model.save_model(path + ".txt")
        
        # Save additional metadata
        metadata = {
            'task_type': self.task_type,
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'num_leaves': self.num_leaves,
            'objective': self.objective,
            'params': self.params,
            'feature_importances': self.feature_importances_
        }
        
        with open(path + ".meta", 'wb') as f:
            joblib.dump(metadata, f)
    
    @classmethod
    def load(cls, path: str) -> 'LightGBMModel':
        """Load a saved LightGBM model from disk."""
        # Load metadata
        with open(path + ".meta", 'rb') as f:
            metadata = joblib.load(f)
        
        # Create model instance
        model = cls(
            task_type=metadata['task_type'],
            n_estimators=metadata['n_estimators'],
            max_depth=metadata['max_depth'],
            learning_rate=metadata['learning_rate'],
            num_leaves=metadata['num_leaves'],
            objective=metadata['objective'],
            **metadata['params']
        )
        
        # Load model
        model.model = lgb.Booster(model_file=path + ".txt")
        model.feature_importances_ = metadata['feature_importances']
        
        return model


class TransformerBlock(Layer):
    """Transformer block with self-attention and feed-forward layers."""
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, rate: float = 0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation='relu'),
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
    
    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class LSTMModel(BaseModel):
    """LSTM-based deep learning model for time series prediction."""
    
    def __init__(
        self,
        input_shape: Tuple[int, int],
        task_type: str = 'regression',
        units: Tuple[int, ...] = (64, 32),
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        **kwargs
    ):
        """Initialize the LSTM model.
        
        Args:
            input_shape: Shape of input data (timesteps, features)
            task_type: Type of task ('regression' or 'classification')
            units: Number of units in each LSTM layer
            dropout_rate: Dropout rate
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            epochs: Number of training epochs
            **kwargs: Additional parameters
        """
        self.input_shape = input_shape
        self.task_type = task_type
        self.units = units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.params = kwargs
        
        # Build model
        self.model = self._build_model()
    
    def _build_model(self) -> tf.keras.Model:
        """Build the LSTM model architecture."""
        inputs = Input(shape=self.input_shape)
        x = inputs
        
        # Add LSTM layers
        for i, units in enumerate(self.units):
            return_sequences = (i < len(self.units) - 1)
            x = LSTM(
                units=units,
                return_sequences=return_sequences,
                dropout=self.dropout_rate,
                recurrent_dropout=self.dropout_rate * 0.5,
                kernel_regularizer=tf.keras.regularizers.l2(1e-4)
            )(x)
            
            # Add batch normalization
            if i < len(self.units) - 1:
                x = BatchNormalization()(x)
        
        # Add output layer
        if self.task_type == 'regression':
            outputs = Dense(1, activation='linear')(x)
            loss = 'mse'
            metrics = ['mae']
        else:  # classification
            outputs = Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs)
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        return model
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        X_val: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train the LSTM model."""
        callbacks = []
        
        # Add callbacks
        if X_val is not None:
            X_val, y_val = X_val
            
            # Early stopping
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stopping)
            
            # Reduce learning rate on plateau
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=10,
                min_lr=1e-6,
                verbose=1
            )
            callbacks.append(reduce_lr)
            
            # Model checkpoint
            checkpoint = ModelCheckpoint(
                'best_model.h5',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                verbose=0
            )
            callbacks.append(checkpoint)
            
            validation_data = (X_val, y_val)
        else:
            validation_data = None
        
        # Train model
        history = self.model.fit(
            X, y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=kwargs.get('verbose', 1),
            shuffle=False
        )
        
        # Load best weights if validation data was provided
        if X_val is not None:
            self.model.load_weights('best_model.h5')
        
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the LSTM model."""
        return self.model.predict(X, verbose=0).flatten()
    
    def save(self, path: str) -> None:
        """Save the LSTM model to disk."""
        self.model.save_weights(path + ".h5")
        
        # Save model architecture and parameters
        model_config = {
            'input_shape': self.input_shape,
            'task_type': self.task_type,
            'units': self.units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'params': self.params
        }
        
        with open(path + ".json", 'w') as f:
            json.dump(model_config, f)
    
    @classmethod
    def load(cls, path: str) -> 'LSTMModel':
        """Load a saved LSTM model from disk."""
        # Load model configuration
        with open(path + ".json", 'r') as f:
            model_config = json.load(f)
        
        # Create model instance
        model = cls(**model_config)
        
        # Load weights
        model.model.load_weights(path + ".h5")
        
        return model


class TransformerModel(BaseModel):
    """Transformer-based model for time series prediction."""
    
    def __init__(
        self,
        input_shape: Tuple[int, int],
        task_type: str = 'regression',
        num_heads: int = 4,
        ff_dim: int = 128,
        num_transformer_blocks: int = 2,
        mlp_units: Tuple[int, ...] = (128, 64),
        dropout_rate: float = 0.1,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        **kwargs
    ):
        """Initialize the Transformer model.
        
        Args:
            input_shape: Shape of input data (timesteps, features)
            task_type: Type of task ('regression' or 'classification')
            num_heads: Number of attention heads
            ff_dim: Hidden layer size in feed-forward network
            num_transformer_blocks: Number of transformer blocks
            mlp_units: Number of units in each dense layer of the MLP head
            dropout_rate: Dropout rate
            learning_rate: Learning rate for optimizer
            batch_size: Batch size for training
            epochs: Number of training epochs
            **kwargs: Additional parameters
        """
        self.input_shape = input_shape
        self.task_type = task_type
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.mlp_units = mlp_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.params = kwargs
        
        # Build model
        self.model = self._build_model()
    
    def _build_model(self) -> tf.keras.Model:
        """Build the Transformer model architecture."""
        inputs = Input(shape=self.input_shape)
        
        # Add positional encoding
        x = inputs
        
        # Create multiple transformer blocks
        for _ in range(self.num_transformer_blocks):
            x = TransformerBlock(
                embed_dim=self.input_shape[1],
                num_heads=self.num_heads,
                ff_dim=self.ff_dim,
                rate=self.dropout_rate
            )(x)
        
        # Global average pooling
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        
        # Add MLP head
        for dim in self.mlp_units:
            x = Dense(dim, activation='relu')(x)
            x = Dropout(self.dropout_rate)(x)
        
        # Add output layer
        if self.task_type == 'regression':
            outputs = Dense(1, activation='linear')(x)
            loss = 'mse'
            metrics = ['mae']
        else:  # classification
            outputs = Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
            metrics = ['accuracy']
        
        # Create and compile model
        model = Model(inputs=inputs, outputs=outputs)
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        
        return model
    
    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        X_val: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Train the Transformer model."""
        callbacks = []
        
        # Add callbacks
        if X_val is not None:
            X_val, y_val = X_val
            
            # Early stopping
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=20,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stopping)
            
            # Reduce learning rate on plateau
            reduce_lr = ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=10,
                min_lr=1e-6,
                verbose=1
            )
            callbacks.append(reduce_lr)
            
            # Model checkpoint
            checkpoint = ModelCheckpoint(
                'best_transformer.h5',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True,
                verbose=0
            )
            callbacks.append(checkpoint)
            
            validation_data = (X_val, y_val)
        else:
            validation_data = None
        
        # Train model
        history = self.model.fit(
            X, y,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=kwargs.get('verbose', 1),
            shuffle=False
        )
        
        # Load best weights if validation data was provided
        if X_val is not None:
            self.model.load_weights('best_transformer.h5')
        
        return history.history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with the Transformer model."""
        return self.model.predict(X, verbose=0).flatten()
    
    def save(self, path: str) -> None:
        """Save the Transformer model to disk."""
        self.model.save_weights(path + ".h5")
        
        # Save model architecture and parameters
        model_config = {
            'input_shape': self.input_shape,
            'task_type': self.task_type,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'num_transformer_blocks': self.num_transformer_blocks,
            'mlp_units': self.mlp_units,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'params': self.params
        }
        
        with open(path + ".json", 'w') as f:
            json.dump(model_config, f)
    
    @classmethod
    def load(cls, path: str) -> 'TransformerModel':
        """Load a saved Transformer model from disk."""
        # Load model configuration
        with open(path + ".json", 'r') as f:
            model_config = json.load(f)
        
        # Create model instance
        model = cls(**model_config)
        
        # Load weights
        model.model.load_weights(path + ".h5")
        
        return model
