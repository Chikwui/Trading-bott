import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from config.settings import Settings
from utils.logger import logger

# Initialize settings
settings = Settings()

class SignalPredictor:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or settings.MODEL_PATH
        self._model = None
        self._model_loaded = False
        
        # Don't load the model during initialization
        if self.model_path:
            logger.info(f"Initialized SignalPredictor with model path: {self.model_path}")
            if not Path(self.model_path).exists():
                logger.warning(f"Model file not found at {self.model_path}")
        else:
            logger.warning("No model path provided. Running in fallback mode.")

    @property
    def model(self):
        """Lazy load the model when first needed"""
        if not self._model_loaded and self.model_path and Path(self.model_path).exists():
            self._load_model()
        return self._model

    def _load_model(self):
        """Load the model with proper error handling"""
        if self._model_loaded:
            return
            
        try:
            logger.info(f"Loading model from {self.model_path}")
            self._model = joblib.load(self.model_path)
            logger.info(f"Successfully loaded model from {self.model_path}")
            self._model_loaded = True
        except Exception as e:
            logger.error(f"Error loading model from {self.model_path}: {str(e)}")
            logger.warning("Running in fallback mode with no model")
            self._model = None

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on input features.
        
        Args:
            features: DataFrame containing the input features for prediction
            
        Returns:
            DataFrame with columns ['symbol', 'signal', 'confidence']
            where signal is 1 (BUY), -1 (SELL), or 0 (HOLD)
        """
        if features.empty:
            logger.warning("Empty features DataFrame provided")
            return pd.DataFrame(columns=['symbol', 'signal', 'confidence'])
            
        logger.debug(f"Generating predictions for {len(features)} samples")
        
        try:
            if self.model is not None:
                # Check if model has predict_proba method
                if hasattr(self.model, 'predict_proba'):
                    proba = self.model.predict_proba(features)
                    # Assuming binary classification with classes [0, 1]
                    confidence = np.max(proba, axis=1)
                    predictions = self.model.predict(features)
                else:
                    # Fallback for models without predict_proba
                    predictions = self.model.predict(features)
                    confidence = np.ones_like(predictions, dtype=float) * 0.6  # Default confidence
                
                # Create results DataFrame
                results = pd.DataFrame({
                    'symbol': features.get('symbol', 'UNKNOWN'),
                    'signal': predictions,
                    'confidence': confidence
                })
                return results
                
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            logger.debug("Falling back to neutral predictions")
        
        # Fallback: return neutral signals
        return pd.DataFrame({
            'symbol': features.get('symbol', 'UNKNOWN'),
            'signal': 0,
            'confidence': 0.0
        }, index=features.index)
