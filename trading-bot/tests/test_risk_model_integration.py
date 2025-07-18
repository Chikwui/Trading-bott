"""
Integration tests for risk model integration with the trading system.

These tests verify that the advanced risk models (VaR, CVaR, Factor) are properly
integrated with the trading system through the ExposureManager.
"""

import asyncio
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from core.risk.exposure_manager import ExposureManager, MT5MarketDataProvider, Position
from core.risk.advanced_risk_models import (
    VaRModel, CVaRModel, FactorRiskModel, RiskModelType
)
from core.market.enums import AssetClass

# Mock data for testing
SAMPLE_MARKET_DATA = {
    'AAPL': pd.DataFrame({
        'close': [100, 101, 102, 101, 100, 99, 101, 103, 105, 104],
        'volume': [1000, 1200, 1100, 900, 800, 1000, 1200, 1300, 1100, 1000],
        'open': [99, 100, 101, 102, 100, 98, 100, 102, 104, 105],
        'high': [101, 102, 103, 102, 101, 100, 102, 104, 106, 105],
        'low': [99, 100, 101, 100, 99, 97, 99, 101, 104, 103],
    }),
    'MSFT': pd.DataFrame({
        'close': [200, 202, 201, 203, 202, 201, 203, 205, 207, 206],
        'volume': [500, 600, 550, 450, 400, 500, 600, 650, 550, 500],
        'open': [199, 200, 202, 201, 200, 200, 202, 204, 206, 207],
        'high': [201, 203, 203, 204, 203, 202, 204, 206, 208, 207],
        'low': [198, 200, 200, 200, 199, 198, 200, 204, 206, 205],
    })
}

# Mock position data
SAMPLE_POSITIONS = [
    Position(
        symbol="AAPL",
        quantity=100,
        price=150.0,
        asset_class=AssetClass.EQUITY,
        sector="Technology",
        region="US",
        beta=1.2
    ),
    Position(
        symbol="MSFT",
        quantity=50,
        price=300.0,
        asset_class=AssetClass.EQUITY,
        sector="Technology",
        region="US",
        beta=1.1
    )
]

@pytest.fixture
def mock_market_data_provider():
    """Create a mock market data provider for testing."""
    mock = AsyncMock(spec=MT5MarketDataProvider)
    
    async def mock_get_current_price(symbol):
        if symbol == 'AAPL':
            return 150.0
        elif symbol == 'MSFT':
            return 300.0
        return None
    
    async def mock_get_historical_data(symbol, timeframe, start_date, end_date=None, count=None):
        if symbol in SAMPLE_MARKET_DATA:
            return SAMPLE_MARKET_DATA[symbol].copy()
        return pd.DataFrame()
    
    mock.get_current_price = mock_get_current_price
    mock.get_historical_data = mock_get_historical_data
    return mock

@pytest.fixture
def risk_models():
    """Create a list of risk models for testing."""
    return [
        VaRModel(confidence_level=0.95, method='historical'),
        CVaRModel(confidence_level=0.95),
        FactorRiskModel(factors=['market', 'size', 'value'])
    ]

@pytest.fixture
async def exposure_manager(mock_market_data_provider, risk_models):
    """Create an ExposureManager instance for testing."""
    manager = ExposureManager(
        portfolio_value=1_000_000,
        market_data_provider=mock_market_data_provider,
        risk_models=risk_models,
        lookback_days=10
    )
    
    # Add sample positions
    for position in SAMPLE_POSITIONS:
        manager.positions[position.symbol] = position
    
    yield manager
    
    # Cleanup
    await manager.close()

@pytest.mark.asyncio
async def test_risk_model_integration(exposure_manager, mock_market_data_provider):
    """Test integration of risk models with ExposureManager."""
    # Initial update of market data and risk metrics
    await exposure_manager._update_market_data()
    await exposure_manager._update_risk_metrics()
    
    # Verify market data was updated
    assert 'AAPL' in exposure_manager.market_data
    assert 'MSFT' in exposure_manager.market_data
    assert not exposure_manager.market_data['AAPL'].empty
    assert not exposure_manager.market_data['MSFT'].empty
    
    # Verify risk metrics were calculated
    assert 'models' in exposure_manager.risk_metrics
    assert 'VaRModel' in exposure_manager.risk_metrics['models']
    assert 'CVaRModel' in exposure_manager.risk_metrics['models']
    assert 'FactorRiskModel' in exposure_manager.risk_metrics['models']
    
    # Verify position values and weights
    assert 'position_values' in exposure_manager.risk_metrics
    assert 'position_weights' in exposure_manager.risk_metrics
    
    # Check that position values are calculated correctly
    aapl_value = exposure_manager.risk_metrics['position_values'].get('AAPL')
    msft_value = exposure_manager.risk_metrics['position_values'].get('MSFT')
    assert aapl_value is not None
    assert msft_value is not None
    assert aapl_value == 150.0 * 100  # 100 shares * $150
    assert msft_value == 300.0 * 50   # 50 shares * $300

@pytest.mark.asyncio
async def test_var_model_calculation(exposure_manager):
    """Test VaR model calculation through ExposureManager."""
    # Update market data and calculate risk metrics
    await exposure_manager._update_market_data()
    await exposure_manager._update_risk_metrics()
    
    # Get VaR model results
    var_result = exposure_manager.risk_metrics['models'].get('VaRModel', {})
    
    # Verify VaR result structure
    assert isinstance(var_result, dict)
    assert 'value' in var_result
    assert 'confidence_level' in var_result
    assert 'model_type' in var_result
    assert 'parameters' in var_result
    
    # Verify VaR value is reasonable (negative return at confidence level)
    assert var_result['value'] > 0  # VaR is typically positive (absolute value of loss)
    assert var_result['confidence_level'] == 0.95
    assert var_result['model_type'] == RiskModelType.HISTORICAL_VAR.value

@pytest.mark.asyncio
async def test_cvar_model_calculation(exposure_manager):
    """Test CVaR model calculation through ExposureManager."""
    # Update market data and calculate risk metrics
    await exposure_manager._update_market_data()
    await exposure_manager._update_risk_metrics()
    
    # Get CVaR model results
    cvar_result = exposure_manager.risk_metrics['models'].get('CVaRModel', {})
    
    # Verify CVaR result structure
    assert isinstance(cvar_result, dict)
    assert 'value' in cvar_result
    assert 'confidence_level' in cvar_result
    assert 'model_type' in cvar_result
    
    # Verify CVaR value is reasonable (should be greater than or equal to VaR)
    var_result = exposure_manager.risk_metrics['models'].get('VaRModel', {})
    if 'value' in var_result and 'value' in cvar_result:
        assert cvar_result['value'] >= var_result['value']  # CVaR >= VaR
    
    assert cvar_result['confidence_level'] == 0.95
    assert cvar_result['model_type'] == RiskModelType.CVAR.value

@pytest.mark.asyncio
async def test_factor_model_calculation(exposure_manager):
    """Test Factor risk model calculation through ExposureManager."""
    # Update market data and calculate risk metrics
    await exposure_manager._update_market_data()
    await exposure_manager._update_risk_metrics()
    
    # Get Factor model results
    factor_result = exposure_manager.risk_metrics['models'].get('FactorRiskModel', {})
    
    # Verify Factor model result structure
    assert isinstance(factor_result, dict)
    assert 'value' in factor_result
    assert 'model_type' in factor_result
    assert 'additional_metrics' in factor_result
    
    # Verify factor exposures and contributions
    additional_metrics = factor_result.get('additional_metrics', {})
    assert 'factor_exposures' in additional_metrics
    assert 'risk_contributions' in additional_metrics
    
    # Verify exposures are calculated for each factor
    factor_exposures = additional_metrics['factor_exposures']
    assert 'market' in factor_exposures
    assert 'size' in factor_exposures
    assert 'value' in factor_exposures

@pytest.mark.asyncio
async def test_portfolio_metrics_update(exposure_manager):
    """Test that portfolio-level metrics are updated correctly."""
    # Initial update of market data and risk metrics
    await exposure_manager._update_market_data()
    await exposure_manager._update_risk_metrics()
    
    # Verify portfolio metrics
    assert 'portfolio_value' in exposure_manager.risk_metrics
    assert 'position_weights' in exposure_manager.risk_metrics
    
    # Calculate expected portfolio value
    expected_value = sum(
        pos.quantity * (150.0 if pos.symbol == 'AAPL' else 300.0)
        for pos in SAMPLE_POSITIONS
    )
    
    assert abs(exposure_manager.risk_metrics['portfolio_value'] - expected_value) < 1e-6
    
    # Verify position weights sum to 1 (or very close due to floating point)
    weights = sum(exposure_manager.risk_metrics['position_weights'].values())
    assert abs(weights - 1.0) < 1e-6

@pytest.mark.asyncio
async def test_risk_metrics_update_frequency(exposure_manager, monkeypatch):
    """Test that risk metrics are updated at the expected frequency."""
    # Mock the update methods to track calls
    update_market_data_calls = 0
    update_risk_metrics_calls = 0
    
    original_update_market_data = exposure_manager._update_market_data
    original_update_risk_metrics = exposure_manager._update_risk_metrics
    
    async def mock_update_market_data():
        nonlocal update_market_data_calls
        update_market_data_calls += 1
        await original_update_market_data()
    
    async def mock_update_risk_metrics():
        nonlocal update_risk_metrics_calls
        update_risk_metrics_calls += 1
        await original_update_risk_metrics()
    
    # Replace methods with mocks
    exposure_manager._update_market_data = mock_update_market_data
    exposure_manager._update_risk_metrics = mock_update_risk_metrics
    
    # Start the periodic update task
    task = asyncio.create_task(exposure_manager._periodic_update())
    
    try:
        # Wait for a few update cycles
        await asyncio.sleep(10)  # Should be enough for at least one update cycle
        
        # Stop the task
        exposure_manager._stop_event.set()
        await task
        
        # Verify that both methods were called at least once
        assert update_market_data_calls > 0
        assert update_risk_metrics_calls > 0
        
    finally:
        # Ensure task is cancelled if test fails
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
