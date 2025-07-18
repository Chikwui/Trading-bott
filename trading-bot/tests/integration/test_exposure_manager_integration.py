"""
Integration tests for ExposureManager using real MT5 market data.

These tests require:
1. MT5 terminal to be installed and running
2. Valid MT5 account credentials
3. Internet connection for market data
"""

import os
import sys
import asyncio
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
import pytest
import MetaTrader5 as mt5
import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.risk.exposure_manager import (
    ExposureManager, 
    Position, 
    MT5MarketDataProvider,
    VaRModel,
    CVaRModel,
    FactorRiskModel
)
from core.risk.circuit_breakers import CircuitBreaker
from core.market.instrument import InstrumentType, InstrumentMetadata

# Skip these tests if MT5 is not available or not connected
MT5_AVAILABLE = False
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = mt5.initialize()
except (ImportError, RuntimeError):
    pass

# Test symbols with different asset classes
TEST_SYMBOLS = [
    "EURUSD",    # Forex
    "XAUUSD",    # Commodity
    "BTCUSD",    # Crypto
    "US500"      # Index
]

# Skip integration tests if MT5 is not available
pytestmark = pytest.mark.skipif(
    not MT5_AVAILABLE, 
    reason="MT5 not available or not connected"
)

class TestExposureManagerIntegration(unittest.IsolatedAsyncioTestCase):
    """Integration tests for ExposureManager with real MT5 market data."""
    
    @classmethod
    async def asyncSetUpClass(cls):
        """Set up test environment once before all tests."""
        # Initialize MT5 connection
        if not mt5.initialize():
            raise RuntimeError(f"MT5 initialize() failed: {mt5.last_error()}")
        
        # Create market data provider
        cls.provider = MT5MarketDataProvider(TEST_SYMBOLS)
        await cls.provider.connect()
        
        # Initialize risk models
        cls.risk_models = [
            VaRModel(confidence_level=0.95),
            CVaRModel(confidence_level=0.95),
            FactorRiskModel()
        ]
        
        # Initialize exposure manager with $1M portfolio
        cls.manager = ExposureManager(
            portfolio_value=1_000_000,
            base_currency="USD",
            risk_free_rate=0.0,
            lookback_days=60,
            market_data_provider=cls.provider,
            risk_models=cls.risk_models
        )
    
    @classmethod
    async def asyncTearDownClass(cls):
        """Clean up after all tests."""
        await cls.manager.close()
        mt5.shutdown()
    
    async def test_initialization(self):
        """Test that the exposure manager initializes correctly."""
        self.assertIsNotNone(self.manager)
        self.assertEqual(self.manager.portfolio_value, 1_000_000)
        self.assertEqual(self.manager.base_currency, "USD")
    
    async def test_market_data_connection(self):
        """Test that we can connect to MT5 and get market data."""
        # Test getting historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=10)
        
        for symbol in TEST_SYMBOLS:
            data = await self.provider.get_historical_data(
                symbol=symbol,
                timeframe='D1',
                start_date=start_date,
                end_date=end_date
            )
            self.assertIsNotNone(data)
            self.assertFalse(data.empty)
            self.assertIn('close', data.columns)
    
    async def test_position_management(self):
        """Test adding and managing positions with real market data."""
        # Add a position (1% risk)
        position = Position(
            symbol="EURUSD",
            entry_price=1.1000,
            size=100000,  # 1 standard lot
            stop_loss=1.0900,
            take_profit=1.1200,
            entry_time=datetime.now(),
            metadata={"strategy": "mean_reversion"}
        )
        
        # Add position to manager
        self.manager.add_position(position)
        
        # Check position was added
        self.assertIn("EURUSD", self.manager.positions)
        self.assertEqual(len(self.manager.positions), 1)
        
        # Update market data and risk metrics
        await self.manager._update_market_data()
        await self.manager._update_risk_metrics()
        
        # Check risk metrics were updated
        self.assertIn("portfolio_metrics", self.manager.risk_metrics)
        self.assertIn("EURUSD", self.manager.risk_metrics.get("position_metrics", {}))
    
    async def test_risk_metrics_calculation(self):
        """Test that risk metrics are calculated correctly."""
        # Clear existing positions
        self.manager.positions = {}
        
        # Add multiple positions for correlation testing
        symbols = ["EURUSD", "GBPUSD", "USDJPY"]
        positions = []
        
        for symbol in symbols:
            position = Position(
                symbol=symbol,
                entry_price=1.0,  # Will be updated with real prices
                size=50000,  # 0.5 standard lot each
                stop_loss=0.95,
                take_profit=1.05,
                entry_time=datetime.now(),
                metadata={"strategy": "correlation_test"}
            )
            positions.append(position)
        
        # Add positions and update market data
        for position in positions:
            self.manager.add_position(position)
        
        # Update market data and risk metrics
        await self.manager._update_market_data()
        await self.manager._update_risk_metrics()
        
        # Check that risk metrics were calculated
        metrics = self.manager.risk_metrics
        self.assertIn("portfolio_metrics", metrics)
        
        # Check that correlation matrix was calculated
        self.assertIsNotNone(self.manager.correlation_matrix)
        self.assertEqual(
            set(self.manager.correlation_matrix.index), 
            set(symbols)
        )
    
    async def test_circuit_breaker_activation(self):
        """Test that circuit breakers activate correctly."""
        # Clear existing positions
        self.manager.positions = {}
        
        # Add a position with tight stop loss
        position = Position(
            symbol="XAUUSD",
            entry_price=1800.0,
            size=100,  # 1 mini lot
            stop_loss=1799.0,  # Very tight stop
            take_profit=2000.0,
            entry_time=datetime.now(),
            metadata={"strategy": "circuit_breaker_test"}
        )
        
        # Add position and update market data
        self.manager.add_position(position)
        await self.manager._update_market_data()
        
        # Check initial state
        self.assertFalse(self.manager.is_circuit_breaker_triggered())
        
        # Simulate a large price move that would trigger stop loss
        # Note: In a real test, we'd need to mock the price feed or use a test account
        # For integration testing, we'll just verify the circuit breaker logic
        
        # Manually trigger a large drawdown
        self.manager.portfolio_value = 900000  # 10% drawdown
        
        # Check that circuit breaker would trigger
        self.assertTrue(self.manager.is_circuit_breaker_triggered())
    
    async def test_real_time_updates(self):
        """Test that the exposure manager handles real-time updates."""
        # Clear existing positions
        self.manager.positions = {}
        
        # Add a position
        position = Position(
            symbol="BTCUSD",
            entry_price=30000.0,
            size=0.1,  # 0.1 BTC
            stop_loss=29000.0,
            take_profit=35000.0,
            entry_time=datetime.now(),
            metadata={"strategy": "crypto_test"}
        )
        
        self.manager.add_position(position)
        
        # Simulate real-time updates for 5 seconds
        start_time = datetime.now()
        while (datetime.now() - start_time).seconds < 5:
            await self.manager._update_market_data()
            await self.manager._update_risk_metrics()
            await asyncio.sleep(1)  # Update every second
        
        # Verify that metrics were updated
        self.assertIn("BTCUSD", self.manager.risk_metrics.get("position_metrics", {}))
        
        # Verify that we have recent market data
        self.assertIn("BTCUSD", self.manager.market_data)
        self.assertFalse(self.manager.market_data["BTCUSD"].empty)

if __name__ == "__main__":
    unittest.main()
