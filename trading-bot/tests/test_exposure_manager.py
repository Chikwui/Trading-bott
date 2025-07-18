"""
Unit tests for the ExposureManager class.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from core.risk.exposure_manager import (
    ExposureManager,
    Position,
    AssetClass,
    RiskThresholds,
    CircuitBreaker,
    PositionSizer,
    VaRModel,
    CVaRModel,
    FactorRiskModel,
    MarketDataProvider
)


class MockMarketDataProvider(MarketDataProvider):
    """Mock market data provider for testing."""
    
    def __init__(self):
        self.connected = False
        self.prices = {}
        self.historical_data = {}
        self.callbacks = {}
    
    async def connect(self) -> None:
        self.connected = True
    
    async def disconnect(self) -> None:
        self.connected = False
    
    async def get_prices(self, symbols: list) -> dict:
        return {s: self.prices.get(s, 100.0) for s in symbols}
    
    async def get_historical_data(self, symbol: str, timeframe: str = "1d", limit: int = 1000) -> pd.DataFrame:
        if symbol in self.historical_data:
            return self.historical_data[symbol]
        return pd.DataFrame()
    
    async def subscribe_to_updates(self, symbols: list, callback: callable) -> None:
        for symbol in symbols:
            self.callbacks[symbol] = callback


class TestExposureManager(unittest.IsolatedAsyncioTestCase):
    """Test cases for the ExposureManager class."""
    
    async def asyncSetUp(self):
        """Set up test environment."""
        self.market_data = MockMarketDataProvider()
        self.risk_models = [VaRModel(), CVaRModel(), FactorRiskModel()]
        self.thresholds = RiskThresholds(
            max_position_size_pct=0.1,  # 10%
            max_sector_exposure_pct=0.3,  # 30%
            max_drawdown_daily_pct=0.05,  # 5%
            max_drawdown_total_pct=0.15,  # 15%
            max_leverage=3.0,  # 3x
            min_liquidity_usd=10000.0  # $10k
        )
        
        self.manager = ExposureManager(
            market_data_provider=self.market_data,
            risk_models=self.risk_models,
            thresholds=self.thresholds,
            update_interval=1  # 1 second for faster tests
        )
        
        # Set up some test market data
        self.market_data.prices = {
            'AAPL': 150.0,
            'MSFT': 300.0,
            'GOOGL': 2800.0,
            'BTC-USD': 50000.0
        }
        
        # Start the manager
        await self.manager.start()
    
    async def asyncTearDown(self):
        """Clean up after tests."""
        await self.manager.stop()
    
    async def test_add_position_success(self):
        """Test adding a position successfully."""
        # Initial portfolio value is 0, so we need to set it first
        self.manager.portfolio_value = 100000.0  # $100k portfolio
        
        result = await self.manager.add_position(
            symbol='AAPL',
            quantity=50,  # $7,500 position (7.5% of portfolio)
            entry_price=150.0,
            asset_class=AssetClass.EQUITY,
            sector='Technology',
            region='US',
            stop_loss=135.0,
            take_profit=165.0,
            beta=1.2
        )
        
        self.assertTrue(result['success'])
        self.assertIn('AAPL', self.manager.positions)
        self.assertEqual(self.manager.positions['AAPL'].quantity, 50)
        self.assertEqual(self.manager.portfolio_value, 100000.0)  # Shouldn't change
    
    async def test_add_position_exceeds_position_size(self):
        """Test adding a position that exceeds maximum position size."""
        self.manager.portfolio_value = 100000.0
        
        # Try to add a position worth 15% of portfolio (max is 10%)
        result = await self.manager.add_position(
            symbol='AAPL',
            quantity=100,  # $15,000 position (15% of portfolio)
            entry_price=150.0,
            asset_class=AssetClass.EQUITY
        )
        
        self.assertFalse(result['success'])
        self.assertIn('exceeds maximum', result['message'])
        self.assertNotIn('AAPL', self.manager.positions)
    
    async def test_add_position_exceeds_sector_exposure(self):
        """Test adding a position that exceeds sector exposure limit."""
        self.manager.portfolio_value = 100000.0
        
        # Add first position in Technology sector (9% of portfolio)
        await self.manager.add_position(
            symbol='MSFT',
            quantity=30,  # $9,000 position (9% of portfolio)
            entry_price=300.0,
            asset_class=AssetClass.EQUITY,
            sector='Technology'
        )
        
        # Try to add another position in Technology sector that would exceed 30% limit
        result = await self.manager.add_position(
            symbol='AAPL',
            quantity=150,  # $22,500 position (22.5% of portfolio)
            entry_price=150.0,
            asset_class=AssetClass.EQUITY,
            sector='Technology'
        )
        
        self.assertFalse(result['success'])
        self.assertIn('Sector Technology exposure', result['message'])
        self.assertIn('MSFT', self.manager.positions)
        self.assertNotIn('AAPL', self.manager.positions)
    
    async def test_circuit_breaker_drawdown(self):
        """Test circuit breaker triggering on drawdown."""
        # Set up initial portfolio
        self.manager.portfolio_value = 100000.0
        self.manager.peak_portfolio_value = 100000.0
        
        # Add a position
        await self.manager.add_position(
            symbol='AAPL',
            quantity=50,  # $7,500 position
            entry_price=150.0,
            asset_class=AssetClass.EQUITY
        )
        
        # Simulate a large drop in price (16% drawdown, over 15% threshold)
        self.market_data.prices['AAPL'] = 126.0  # 16% drop
        await self.manager.update_positions()
        
        # Circuit breaker should be triggered
        self.assertTrue(self.manager.circuit_breaker.triggered)
        self.assertIn('drawdown', self.manager.circuit_breaker.trigger_reason)
        
        # Try to add another position (should be blocked)
        result = await self.manager.add_position(
            symbol='MSFT',
            quantity=10,
            entry_price=300.0,
            asset_class=AssetClass.EQUITY
        )
        
        self.assertFalse(result['success'])
        self.assertIn('circuit breaker', result['message'])
    
    async def test_position_sizing(self):
        """Test position sizing calculations."""
        self.manager.portfolio_value = 100000.0
        
        # Test position size with stop loss
        size, meta = self.manager.position_sizer.calculate_position_size(
            symbol='AAPL',
            entry_price=150.0,
            stop_loss=135.0,  # 10% risk
            account_equity=100000.0,
            portfolio={},
            market_data={},
            risk_per_trade_pct=1.0  # Risk 1% of account
        )
        
        # Expected: 1% of $100k = $1,000 risk
        # $15 risk per share ($150 - $135)
        # $1,000 / $15 = 66.67 shares
        self.assertAlmostEqual(size, 66.0, delta=1.0)  # Rounded to whole shares
        self.assertAlmostEqual(meta['position_value'], size * 150.0, delta=1.0)
        self.assertAlmostEqual(meta['risk_amount'], 1000.0, delta=1.0)
    
    async def test_leverage_check(self):
        """Test leverage constraints."""
        self.manager.portfolio_value = 100000.0
        
        # Add positions up to 2x leverage
        await self.manager.add_position(
            symbol='AAPL',
            quantity=1000,  # $150,000 position (1.5x)
            entry_price=150.0,
            asset_class=AssetClass.EQUITY
        )
        
        # Try to add another position that would exceed 3x leverage
        result = await self.manager.add_position(
            symbol='MSFT',
            quantity=500,  # $150,000 additional position (would be 3x total)
            entry_price=300.0,
            asset_class=AssetClass.EQUITY
        )
        
        # Should be allowed (exactly at 3x limit)
        self.assertTrue(result['success'])
        
        # Try to add one more position (should exceed 3x leverage)
        result = await self.manager.add_position(
            symbol='GOOGL',
            quantity=10,  # $28,000 additional position
            entry_price=2800.0,
            asset_class=AssetClass.EQUITY
        )
        
        self.assertFalse(result['success'])
        self.assertIn('leverage', result['message'].lower())
    
    async def test_market_data_updates(self):
        """Test that market data updates are processed correctly."""
        self.manager.portfolio_value = 100000.0
        
        # Add a position
        await self.manager.add_position(
            symbol='AAPL',
            quantity=50,  # $7,500 position
            entry_price=150.0,
            asset_class=AssetClass.EQUITY
        )
        
        # Simulate price update
        self.market_data.prices['AAPL'] = 160.0  # $500 gain
        
        # Wait for update (update_interval is 1 second in setup)
        await asyncio.sleep(1.1)
        
        # Check that position was updated
        self.assertEqual(self.manager.positions['AAPL'].current_price, 160.0)
        self.assertAlmostEqual(self.manager.portfolio_value, 100500.0, delta=1.0)
    
    async def test_risk_metrics_calculation(self):
        """Test that risk metrics are calculated correctly."""
        # Set up historical data for VaR calculation
        dates = pd.date_range(end=datetime.now(), periods=100)
        returns = np.random.normal(0.0005, 0.015, 100)  # Daily returns with 1.5% vol
        prices = 100 * np.exp(np.cumsum(returns))
        
        self.market_data.historical_data['AAPL'] = pd.DataFrame({
            'open': prices,
            'high': prices * 1.01,
            'low': prices * 0.99,
            'close': prices,
            'volume': np.random.lognormal(10, 1, 100)
        }, index=dates)
        
        # Add a position
        self.manager.portfolio_value = 100000.0
        await self.manager.add_position(
            symbol='AAPL',
            quantity=100,  # $15,000 position
            entry_price=150.0,
            asset_class=AssetClass.EQUITY,
            beta=1.2
        )
        
        # Calculate risk metrics
        metrics = await self.manager.calculate_risk_metrics()
        
        # Check that all risk models returned results
        self.assertIn('var', metrics)
        self.assertIn('cvar', metrics)
        self.assertIn('factor_exposures', metrics)
        
        # Check that VaR is reasonable (between 0.5% and 10% of position value)
        position_value = 100 * 150.0  # 100 shares * $150
        self.assertGreater(metrics['var'], 0.005 * position_value)
        self.assertLess(metrics['var'], 0.10 * position_value)
        
        # CVaR should be greater than VaR
        self.assertGreater(metrics['cvar'], metrics['var'])
    
    async def test_reset_circuit_breaker(self):
        """Test resetting the circuit breaker."""
        # Trigger circuit breaker
        self.manager.portfolio_value = 100000.0
        self.manager.peak_portfolio_value = 100000.0
        
        # Add a position and simulate a large drop
        await self.manager.add_position(
            symbol='BTC-USD',
            quantity=1,  # $50,000 position
            entry_price=50000.0,
            asset_class=AssetClass.CRYPTO
        )
        
        # 20% drop (exceeds 15% threshold)
        self.market_data.prices['BTC-USD'] = 40000.0
        await self.manager.update_positions()
        
        self.assertTrue(self.manager.circuit_breaker.triggered)
        
        # Reset circuit breaker
        self.manager.reset_circuit_breaker()
        self.assertFalse(self.manager.circuit_breaker.triggered)
        
        # Should be able to add positions again
        result = await self.manager.add_position(
            symbol='AAPL',
            quantity=10,
            entry_price=150.0,
            asset_class=AssetClass.EQUITY
        )
        
        self.assertTrue(result['success'])


class TestCircuitBreaker(unittest.TestCase):
    """Test cases for the CircuitBreaker class."""
    
    def setUp(self):
        """Set up test environment."""
        self.thresholds = RiskThresholds()
        self.cb = CircuitBreaker(self.thresholds)
    
    def test_initial_state(self):
        """Test initial state of circuit breaker."""
        self.assertFalse(self.cb.triggered)
        self.assertEqual(self.cb.trigger_reason, "")
        self.assertIsNone(self.cb.trigger_time)
    
    def test_trigger_and_reset(self):
        """Test triggering and resetting the circuit breaker."""
        reason = "Test reason"
        self.cb.trigger(reason)
        
        self.assertTrue(self.cb.triggered)
        self.assertEqual(self.cb.trigger_reason, reason)
        self.assertIsNotNone(self.cb.trigger_time)
        
        # Reset
        self.cb.reset()
        self.assertFalse(self.cb.triggered)
        self.assertEqual(self.cb.trigger_reason, "")
        self.assertIsNone(self.cb.trigger_time)
    
    def test_check_position_size(self):
        """Test position size checking."""
        position = MagicMock()
        position.market_value = 6000.0  # $6,000
        
        # Test within limits (5% of $100k = $5k max, but this is checked in ExposureManager)
        # The actual check is done against the threshold, not the position directly
        portfolio_value = 100000.0
        result, msg = self.cb.check_position_size(position, portfolio_value)
        self.assertTrue(result)
        self.assertEqual(msg, "")
    
    def test_check_leverage(self):
        """Test leverage checking."""
        # Test within limits (3x max leverage)
        result, msg = self.cb.check_leverage(200000.0, 100000.0)  # 2x leverage
        self.assertTrue(result)
        self.assertEqual(msg, "")
        
        # Test exceeding limits
        result, msg = self.cb.check_leverage(400000.0, 100000.0)  # 4x leverage
        self.assertFalse(result)
        self.assertIn("exceeds maximum", msg)


class TestPositionSizer(unittest.TestCase):
    """Test cases for the PositionSizer class."""
    
    def setUp(self):
        """Set up test environment."""
        self.risk_model = MagicMock()
        self.thresholds = RiskThresholds()
        self.sizer = PositionSizer(self.risk_model, self.thresholds)
    
    def test_calculate_position_size_with_stop_loss(self):
        """Test position size calculation with stop loss."""
        size, meta = self.sizer.calculate_position_size(
            symbol='AAPL',
            entry_price=100.0,
            stop_loss=95.0,  # 5% risk per share
            account_equity=100000.0,
            portfolio={},
            market_data={},
            risk_per_trade_pct=1.0  # Risk 1% of account ($1,000)
        )
        
        # Expected: $1,000 risk / $5 risk per share = 200 shares
        self.assertEqual(size, 200.0)
        self.assertAlmostEqual(meta['position_value'], 20000.0)  # 200 * $100
        self.assertAlmostEqual(meta['risk_amount'], 1000.0)  # 1% of $100k
    
    def test_calculate_position_size_without_stop_loss(self):
        """Test position size calculation without stop loss."""
        size, meta = self.sizer.calculate_position_size(
            symbol='AAPL',
            entry_price=100.0,
            stop_loss=None,
            account_equity=100000.0,
            portfolio={},
            market_data={},
            risk_per_trade_pct=1.0  # Risk 1% of account ($1,000)
        )
        
        # Without stop loss, should use fixed percentage (1% of equity / entry price)
        expected_size = (100000.0 * 0.01) / 100.0  # $1,000 / $100 = 10 shares
        self.assertEqual(size, expected_size)
    
    def test_position_size_rounding(self):
        """Test that position size is rounded to nearest lot size."""
        # Mock the _round_to_lot_size method
        with patch.object(self.sizer, '_round_to_lot_size', return_value=150.0):
            size, _ = self.sizer.calculate_position_size(
                symbol='AAPL',
                entry_price=100.0,
                stop_loss=95.0,
                account_equity=100000.0,
                portfolio={},
                market_data={},
                risk_per_trade_pct=1.0
            )
            
            self.assertEqual(size, 150.0)  # Should use the mocked rounded value
    
    def test_risk_reward_ratio(self):
        """Test risk/reward ratio calculation."""
        # Test with entry=100, stop=95, target=110
        # Risk = 5 (100-95), Reward = 10 (110-100)
        # Ratio = 10/5 = 2.0
        ratio = self.sizer._calculate_risk_reward_ratio(100.0, 95.0, 110.0)
        self.assertEqual(ratio, 2.0)
        
        # Test with no stop loss
        ratio = self.sizer._calculate_risk_reward_ratio(100.0, 100.0, 110.0)
        self.assertEqual(ratio, 0.0)
        
        # Test with no take profit
        ratio = self.sizer._calculate_risk_reward_ratio(100.0, 95.0, 100.0)
        self.assertEqual(ratio, 0.0)


if __name__ == '__main__':
    unittest.main()
