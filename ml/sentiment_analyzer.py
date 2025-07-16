from pathlib import Path
import joblib
import logging
from config.settings import Settings
from utils.logger import logger

# Initialize settings
settings = Settings()

class SentimentAnalyzer:
    def __init__(self, model_name=None):
        self.model_name = model_name or settings.SENTIMENT_MODEL
        self._model_loaded = False
        self._model = None
        self._tokenizer = None
        self.use_huggingface = "/" in str(self.model_name) if self.model_name else False
        
        # Don't load the model during initialization
        logger.info(f"Initialized SentimentAnalyzer with model: {self.model_name}")
        logger.info("Model will be loaded on first use")

    @property
    def model(self):
        """Lazy load the model when first needed"""
        if not self._model_loaded and self.model_name:
            self._load_model()
        return self._model

    def _load_model(self):
        """Load the model with proper error handling"""
        if self._model_loaded:
            return

        try:
            if self.use_huggingface:
                logger.info(f"Loading Hugging Face model: {self.model_name}")
                from transformers import pipeline
                
                # Load with explicit device mapping to avoid CUDA issues
                device = -1  # Default to CPU
                try:
                    import torch
                    if torch.cuda.is_available():
                        device = 0  # Use first GPU if available
                except ImportError:
                    pass

                self._model = pipeline(
                    "sentiment-analysis",
                    model=self.model_name,
                    tokenizer=self.model_name,
                    device=device,
                    max_length=512,
                    truncation=True
                )
                logger.info(f"Successfully loaded Hugging Face model: {self.model_name}")
            
            elif self.model_name and Path(self.model_name).exists():
                logger.info(f"Loading local sentiment model from {self.model_name}")
                self._model = joblib.load(self.model_name)
                logger.info(f"Successfully loaded local model from {self.model_name}")
            
            else:
                logger.warning(f"Sentiment model not found at {self.model_name}")
                logger.warning("Using default sentiment analysis (always neutral)")
                
        except Exception as e:
            logger.error(f"Error loading sentiment model: {str(e)}")
            logger.warning("Using default sentiment analysis (always neutral)")
        
        self._model_loaded = True

    def analyze(self, text: str) -> float:
        """
        Analyze sentiment of the given text.
        Returns a float between -1 (bearish) and 1 (bullish).
        """
        if not text or not isinstance(text, str) or not text.strip():
            return 0.0
            
        logger.debug(f"Analyzing sentiment for text: {text[:100]}...")  # Log first 100 chars
        
        try:
            if self.model is not None and hasattr(self, 'use_huggingface') and self.use_huggingface:
                # Process with Hugging Face model
                result = self._model(text[:512])[0]  # Limit input length to 512 tokens
                score = result['score']
                # Convert sentiment score to -1 to 1 range
                if result['label'] == 'NEGATIVE':
                    return -score
                return score
                
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            logger.debug("Falling back to neutral sentiment")
                
        # Default neutral sentiment if no model is loaded or on error
        return 0.0
