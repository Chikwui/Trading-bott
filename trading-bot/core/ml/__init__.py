"""
ML Module for Trading Bot

This module contains machine learning components for the trading system,
including model training, validation, and inference pipelines.
"""
from .pipeline import ModelPipeline, ModelConfig, FeatureEngineer
from .models import (
    XGBoostModel,
    LightGBMModel,
    LSTMModel,
    TransformerModel
)
from .utils import (
    calculate_metrics,
    prepare_sequences,
    create_lookback_dataset
)

__version__ = '0.1.0'
__all__ = [
    'ModelPipeline',
    'ModelConfig',
    'FeatureEngineer',
    'XGBoostModel',
    'LightGBMModel',
    'LSTMModel',
    'TransformerModel',
    'calculate_metrics',
    'prepare_sequences',
    'create_lookback_dataset'
]
