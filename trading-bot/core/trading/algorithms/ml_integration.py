"""
ML Integration for Execution Algorithms

This module provides integration between ML models and execution algorithms,
allowing for data-driven execution decisions and parameter optimization.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from decimal import Decimal

# Import ML components
from core.ml import ModelPipeline, ModelConfig, FeatureEngineer
from core.ml.models import XGBoostModel, LightGBMModel, LSTMModel, TransformerModel
from core.ml.utils import calculate_metrics, prepare_sequences, create_lookback_dataset

# Import execution algorithm base class
from .base import ExecutionAlgorithm

logger = logging.getLogger(__name__)

@dataclass
class MLExecutionConfig:
    """Configuration for ML-powered execution."""
    # Model configuration
    model_type: str = 'xgboost'  # xgboost, lightgbm, lstm, transformer
    feature_window: int = 100  # Number of historical data points for features
    prediction_horizon: int = 10  # Number of steps ahead to predict
    
    # Training parameters
    train_interval: int = 3600  # Seconds between model retraining
    train_window: int = 10000  # Number of samples for training
    
    # Feature engineering
    feature_columns: List[str] = field(default_factory=lambda: [
        'open', 'high', 'low', 'close', 'volume',
        'bid', 'ask', 'spread', 'order_book_imbalance'
    ])
    target_columns: List[str] = field(default_factory=lambda: [
        'price_change', 'execution_quality'
    ])
    
    # Model hyperparameters
    model_params: Dict[str, Any] = field(default_factory=lambda: {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    })

class MLExecutionMixin:
    """Mixin class that adds ML capabilities to execution algorithms."""
    
    def __init__(self, *args, ml_config: Optional[Dict] = None, **kwargs):
        """Initialize ML execution mixin."""
        super().__init__(*args, **kwargs)
        self.ml_config = MLExecutionConfig(**(ml_config or {}))
        self.ml_initialized = False
        self.model_pipeline = None
        self.feature_engineer = None
        self.last_training_time = 0
        self.market_data_buffer = []
        self.prediction_cache = {}
        
    async def initialize_ml_models(self):
        """Initialize ML models and feature engineering."""
        if self.ml_initialized:
            return
            
        logger.info("Initializing ML models for execution...")
        
        # Initialize feature engineer
        self.feature_engineer = FeatureEngineer(
            window_size=self.ml_config.feature_window,
            feature_columns=self.ml_config.feature_columns,
            target_columns=self.ml_config.target_columns
        )
        
        # Initialize model based on configuration
        model_classes = {
            'xgboost': XGBoostModel,
            'lightgbm': LightGBMModel,
            'lstm': LSTMModel,
            'transformer': TransformerModel
        }
        
        model_class = model_classes.get(self.ml_config.model_type)
        if not model_class:
            raise ValueError(f"Unsupported model type: {self.ml_config.model_type}")
            
        # Create model pipeline
        self.model_pipeline = ModelPipeline(
            model=model_class(**self.ml_config.model_params),
            config=ModelConfig(
                feature_window=self.ml_config.feature_window,
                prediction_horizon=self.ml_config.prediction_horizon,
                train_interval=self.ml_config.train_interval,
                train_window=self.ml_config.train_window
            ),
            feature_engineer=self.feature_engineer
        )
        
        self.ml_initialized = True
        logger.info("ML models initialized successfully")
    
    async def update_market_data(self, market_data: Dict[str, Any]):
        """Update market data buffer with new data point."""
        if not self.ml_initialized:
            await self.initialize_ml_models()
            
        # Add timestamp if not present
        if 'timestamp' not in market_data:
            market_data['timestamp'] = datetime.utcnow()
            
        self.market_data_buffer.append(market_data)
        
        # Keep only the most recent data points
        if len(self.market_data_buffer) > self.ml_config.train_window * 2:
            self.market_data_buffer = self.market_data_buffer[-self.ml_config.train_window * 2:]
    
    async def train_models(self):
        """Train or retrain ML models if needed."""
        if not self.ml_initialized or not self.model_pipeline:
            await self.initialize_ml_models()
            
        current_time = datetime.utcnow().timestamp()
        if current_time - self.last_training_time < self.ml_config.train_interval:
            return
            
        if len(self.market_data_buffer) < self.ml_config.train_window:
            logger.warning(f"Not enough data for training. Have {len(self.market_data_buffer)} samples, need {self.ml_config.train_window}")
            return
            
        logger.info("Starting model training...")
        
        try:
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame(self.market_data_buffer)
            
            # Train the model
            self.model_pipeline.train(df)
            self.last_training_time = current_time
            logger.info("Model training completed successfully")
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}", exc_info=True)
    
    async def predict_market_conditions(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict market conditions using ML models."""
        if not self.ml_initialized or not self.model_pipeline:
            await self.initialize_ml_models()
            
        # Check if we have a recent prediction in cache
        cache_key = tuple(sorted(market_data.items()))
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]
            
        try:
            # Prepare data for prediction
            df = pd.DataFrame([market_data])
            
            # Make prediction
            prediction = self.model_pipeline.predict(df)
            
            # Cache the prediction
            self.prediction_cache[cache_key] = prediction
            
            # Clear cache if it gets too large
            if len(self.prediction_cache) > 1000:
                self.prediction_cache.clear()
                
            return prediction
            
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}", exc_info=True)
            return {}
    
    async def optimize_execution_parameters(self, order: 'Order', market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize execution parameters using ML predictions."""
        # Get market predictions
        predictions = await self.predict_market_conditions(market_data)
        
        if not predictions:
            return {}
            
        # Basic optimization logic - can be customized per algorithm
        optimized_params = {
            'aggressiveness': 0.5,  # Default value
            'size_multiplier': 1.0,
            'time_horizon': 60,  # seconds
            'risk_adjustment': 1.0
        }
        
        # Example: Adjust parameters based on predicted market conditions
        if 'volatility' in predictions and predictions['volatility'] > 0.7:
            optimized_params['aggressiveness'] = 0.8
            optimized_params['size_multiplier'] = 0.8
            optimized_params['risk_adjustment'] = 1.2
        elif 'liquidity' in predictions and predictions['liquidity'] < 0.3:
            optimized_params['aggressiveness'] = 0.3
            optimized_params['size_multiplier'] = 0.5
            optimized_params['time_horizon'] = 120
            
        return optimized_params

class MLEnhancedIcebergExecutor(MLExecutionMixin, ExecutionAlgorithm):
    """Iceberg execution algorithm enhanced with ML predictions."""
    
    def __init__(self, *args, **kwargs):
        """Initialize the ML-enhanced iceberg executor."""
        ml_config = kwargs.pop('ml_config', {})
        super().__init__(*args, **kwargs)
        MLExecutionMixin.__init__(self, *args, ml_config=ml_config, **kwargs)
        
        # Initialize any additional state specific to Iceberg
        self.order_slices = {}
        self.slice_timers = {}
        
    async def execute(self, order: 'Order', params: Dict[str, Any] = None) -> 'Order':
        """Execute an order using ML-enhanced Iceberg algorithm."""
        # Initialize ML models if not already done
        if not self.ml_initialized:
            await self.initialize_ml_models()
            
        # Get market data
        market_data = await self.exchange_adapter.get_market_data(order.symbol)
        
        # Update market data and train models if needed
        await self.update_market_data(market_data)
        await self.train_models()
        
        # Optimize execution parameters using ML
        execution_params = await self.optimize_execution_parameters(order, market_data)
        
        # Merge with any provided parameters
        if params:
            execution_params.update(params)
            
        # Call parent class execute with optimized parameters
        return await super().execute(order, execution_params)
    
    async def _execute_slice(self, order: 'Order', params: Dict[str, Any]) -> bool:
        """Execute a single slice of the iceberg order."""
        # Get current market data
        market_data = await self.exchange_adapter.get_market_data(order.symbol)
        
        # Get ML predictions for this slice
        predictions = await self.predict_market_conditions(market_data)
        
        # Adjust slice size and aggressiveness based on predictions
        if predictions:
            # Example: Reduce size if high adverse selection risk
            if predictions.get('adverse_selection_risk', 0) > 0.7:
                params['size'] = params.get('size', 0) * 0.8
                params['aggressiveness'] = params.get('aggressiveness', 0.5) * 1.2
                
            # Example: Increase size if high liquidity and low volatility
            elif (predictions.get('liquidity', 0) > 0.7 and 
                  predictions.get('volatility', 0) < 0.3):
                params['size'] = params.get('size', 0) * 1.2
        
        # Call parent class implementation
        return await super()._execute_slice(order, params)
