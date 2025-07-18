"""
Test suite for ML integration with the trading system.

This file covers:
- ML model training and validation
- Feature engineering
- Model inference in trading decisions
- Performance metrics and monitoring
"""
import asyncio
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import MagicMock, patch, AsyncMock

from core.ml.pipeline import MLPipeline
from core.ml.models import XGBoostModel, LSTMModel
from core.ml.feature_engineering import FeatureEngineer
from core.data.historical_data import HistoricalDataFetcher
from core.trading.order import OrderSide, OrderType, TimeInForce

# Test configuration
TEST_SYMBOL = "BTC/USD"
TEST_START_DATE = datetime(2023, 1, 1)
TEST_END_DATE = datetime(2023, 12, 31)

@pytest.fixture
def sample_ohlcv_data():
    """Generate sample OHLCV data for testing."""
    date_range = pd.date_range(start=TEST_START_DATE, end=TEST_END_DATE, freq='1H')
    np.random.seed(42)
    
    # Generate random walk for prices
    log_returns = np.random.normal(0, 0.01, len(date_range))
    prices = 50000 * np.exp(np.cumsum(log_returns))
    
    # Create DataFrame with OHLCV data
    data = pd.DataFrame({
        'timestamp': date_range,
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, len(date_range)))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, len(date_range)))),
        'close': prices,
        'volume': np.random.lognormal(mean=10, sigma=1, size=len(date_range))
    })
    
    data['close'] = (data['high'] + data['low'] + data['close']) / 3  # More realistic close
    return data

@pytest.fixture
def mock_historical_data(sample_ohlcv_data):
    """Mock historical data fetcher."""
    class MockHistoricalDataFetcher:
        async def fetch_ohlcv(self, symbol, timeframe, start_date, end_date):
            mask = (sample_ohlcv_data['timestamp'] >= start_date) & \
                   (sample_ohlcv_data['timestamp'] <= end_date)
            return sample_ohlcv_data[mask].copy()
    
    return MockHistoricalDataFetcher()

class TestMLPipeline:
    """Tests for the ML pipeline integration."""
    
    @pytest.mark.asyncio
    async def test_pipeline_training(self, mock_historical_data):
        """Test end-to-end ML pipeline training."""
        # Initialize pipeline
        pipeline = MLPipeline(
            model_type='xgboost',
            symbol=TEST_SYMBOL,
            timeframe='1h',
            features=['sma_20', 'rsi_14', 'macd'],
            target='next_return',
            test_size=0.2,
            random_state=42
        )
        
        # Train model
        model, metrics = await pipeline.train(
            start_date=TEST_START_DATE,
            end_date=TEST_END_DATE,
            historical_data=mock_historical_data
        )
        
        # Verify model was trained
        assert model is not None
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        
        # Verify metrics are reasonable
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1'] <= 1
    
    @pytest.mark.asyncio
    async def test_model_inference(self, mock_historical_data, sample_ohlcv_data):
        """Test model inference on new data."""
        # Initialize and train model
        pipeline = MLPipeline(
            model_type='xgboost',
            symbol=TEST_SYMBOL,
            timeframe='1h',
            features=['sma_20', 'rsi_14', 'macd'],
            target='next_return',
            test_size=0.2,
            random_state=42
        )
        
        model, _ = await pipeline.train(
            start_date=TEST_START_DATE,
            end_date=TEST_END_DATE - timedelta(days=30),  # Leave out last month for testing
            historical_data=mock_historical_data
        )
        
        # Prepare test data (last month)
        test_data = sample_ohlcv_data[sample_ohlcv_data['timestamp'] >= TEST_END_DATE - timedelta(days=30)].copy()
        
        # Generate predictions
        predictions = await pipeline.predict(
            data=test_data,
            model=model
        )
        
        # Verify predictions
        assert len(predictions) == len(test_data)
        assert all(0 <= p <= 1 for p in predictions)  # Assuming binary classification

class TestTradingIntegration:
    """Tests for ML model integration with trading system."""
    
    @pytest.fixture
    def mock_order_manager(self):
        """Mock order manager for testing."""
        manager = AsyncMock()
        manager.submit_order = AsyncMock()
        return manager
    
    @pytest.mark.asyncio
    async def test_ml_signal_generation(self, mock_historical_data, mock_order_manager):
        """Test ML-based signal generation and order submission."""
        # Initialize ML pipeline
        pipeline = MLPipeline(
            model_type='xgboost',
            symbol=TEST_SYMBOL,
            timeframe='1h',
            features=['sma_20', 'rsi_14', 'macd'],
            target='next_return',
            test_size=0.2,
            random_state=42
        )
        
        # Train model
        model, _ = await pipeline.train(
            start_date=TEST_START_DATE,
            end_date=TEST_END_DATE - timedelta(days=30),
            historical_data=mock_historical_data
        )
        
        # Mock market data
        current_market_data = {
            'timestamp': [datetime.utcnow()],
            'open': [50000],
            'high': [50500],
            'low': [49500],
            'close': [50200],
            'volume': [1000]
        }
        
        # Generate features for current market data
        feature_engineer = FeatureEngineer()
        features = feature_engineer.transform(pd.DataFrame(current_market_data))
        
        # Get model prediction
        prediction = model.predict_proba(features)[0][1]  # Probability of positive return
        
        # Generate signal based on prediction
        if prediction > 0.7:  # Strong buy signal
            side = OrderSide.BUY
            quantity = Decimal('0.1')
        elif prediction < 0.3:  # Strong sell signal
            side = OrderSide.SELL
            quantity = Decimal('0.1')
        else:  # No clear signal
            side = None
        
        # Submit order if we have a signal
        if side is not None:
            order = {
                'symbol': TEST_SYMBOL,
                'side': side,
                'order_type': OrderType.MARKET,
                'quantity': quantity,
                'time_in_force': TimeInForce.GTC
            }
            await mock_order_manager.submit_order(order)
        
        # Verify order was submitted if we had a strong signal
        if prediction > 0.7 or prediction < 0.3:
            mock_order_manager.submit_order.assert_called_once()
        else:
            mock_order_manager.submit_order.assert_not_called()

class TestPerformanceMonitoring:
    """Tests for ML model performance monitoring."""
    
    @pytest.mark.asyncio
    async def test_performance_tracking(self, mock_historical_data):
        """Test tracking of ML model performance over time."""
        # Initialize pipeline with performance tracking
        pipeline = MLPipeline(
            model_type='xgboost',
            symbol=TEST_SYMBOL,
            timeframe='1h',
            features=['sma_20', 'rsi_14', 'macd'],
            target='next_return',
            test_size=0.2,
            random_state=42,
            enable_performance_tracking=True
        )
        
        # Train initial model
        model, initial_metrics = await pipeline.train(
            start_date=TEST_START_DATE,
            end_date=TEST_START_DATE + timedelta(days=180),  # First 6 months
            historical_data=mock_historical_data
        )
        
        # Simulate periodic retraining and performance tracking
        performance_metrics = []
        window_size = 30  # days
        retrain_interval = 7  # days
        
        for i in range(0, 180, retrain_interval):
            start_date = TEST_START_DATE + timedelta(days=i)
            end_date = start_date + timedelta(days=window_size)
            
            # Retrain model on sliding window
            model, metrics = await pipeline.train(
                start_date=start_date,
                end_date=end_date,
                historical_data=mock_historical_data
            )
            
            # Track performance
            performance_metrics.append({
                'date': end_date,
                'accuracy': metrics['accuracy'],
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1': metrics['f1']
            })
        
        # Verify performance metrics were tracked
        assert len(performance_metrics) > 0
        
        # Convert to DataFrame for analysis
        perf_df = pd.DataFrame(performance_metrics)
        
        # Verify metrics are within expected ranges
        assert all(0 <= perf_df['accuracy'] <= 1)
        assert all(0 <= perf_df['precision'] <= 1)
        assert all(0 <= perf_df['recall'] <= 1)
        assert all(0 <= perf_df['f1'] <= 1)
        
        # Check for significant performance degradation
        assert perf_df['accuracy'].iloc[-1] >= perf_df['accuracy'].iloc[0] * 0.9  # Within 10% of initial accuracy

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
