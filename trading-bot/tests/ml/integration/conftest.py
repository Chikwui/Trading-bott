"""
Configuration and fixtures for ML Pipeline integration tests.
"""
import os
import shutil
import tempfile
from pathlib import Path

import pytest
import numpy as np
import pandas as pd

from core.ml.registry import get_model_registry
from core.ml.feature_store import get_feature_store
from core.ml.monitoring import get_monitor

@pytest.fixture(scope="module")
def test_environment():
    """Set up test environment with temporary directories."""
    # Create temporary directories
    test_dir = Path(tempfile.mkdtemp())
    registry_dir = test_dir / "model_registry"
    feature_store_dir = test_dir / "feature_store"
    monitoring_dir = test_dir / "monitoring"
    
    # Initialize components
    registry = get_model_registry("file", registry_path=str(registry_dir))
    feature_store = get_feature_store("file", store_path=str(feature_store_dir))
    monitor = get_monitor("file", log_dir=str(monitoring_dir))
    
    yield {
        'test_dir': test_dir,
        'registry': registry,
        'feature_store': feature_store,
        'monitor': monitor
    }
    
    # Clean up
    shutil.rmtree(test_dir, ignore_errors=True)

@pytest.fixture
def sample_price_data():
    """Generate sample price data for testing."""
    np.random.seed(42)
    n_samples = 100
    dates = pd.date_range(end=pd.Timestamp.now(), periods=n_samples)
    
    # Generate random walk for price data
    returns = np.random.normal(0.001, 0.02, n_samples)
    prices = np.cumprod(1 + returns) * 100
    
    # Create OHLCV data
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_samples).astype(int)
    })
    
    # Ensure high > low
    df['high'] = df[['open', 'high']].max(axis=1)
    df['low'] = df[['open', 'low']].min(axis=1)
    
    # Create target (1 if price goes up next period, else 0)
    df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
    return df.dropna()

@pytest.fixture
def price_feature_view(test_environment, sample_price_data):
    """Create a feature view with price data for testing."""
    feature_store = test_environment['feature_store']
    
    # Create feature view
    feature_view_name = "price_features"
    features = [
        {"name": "open", "dtype": "float"},
        {"name": "high", "dtype": "float"},
        {"name": "low", "dtype": "float"},
        {"name": "close", "dtype": "float"},
        {"name": "volume", "dtype": "float"},
        {"name": "target", "dtype": "int"}
    ]
    
    feature_store.create_feature_view(
        name=feature_view_name,
        features=features,
        description="Price movement features",
        tags=["price", "movement"]
    )
    
    # Add some technical indicators
    df = sample_price_data.copy()
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    
    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Drop NA values from indicators
    df = df.dropna()
    
    # Store features
    feature_store.write_features(
        feature_view_name=feature_view_name,
        data=df,
        timestamp_column='timestamp'
    )
    
    return {
        'feature_view_name': feature_view_name,
        'features': features,
        'data': df
    }
