"""
Unit tests for risk models in the trading bot.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, call
from core.risk.exposure_manager import (
    VaRModel, 
    CVaRModel, 
    FactorRiskModel,
    Position,
    AssetClass,
    RiskThresholds,
    CircuitBreaker
)

class TestVaRModel(unittest.TestCase):
    """Test cases for the Value at Risk (VaR) model."""
    
    def setUp(self):
        """Set up test data."""
        # Create test positions
        self.positions = {
            'AAPL': Position(
                symbol='AAPL',
                quantity=100,
                entry_price=150.0,
                current_price=150.0,
                asset_class=AssetClass.EQUITY,
                sector='Technology',
                beta=1.2
            ),
            'MSFT': Position(
                symbol='MSFT',
                quantity=50,
                entry_price=300.0,
                current_price=300.0,
                asset_class=AssetClass.EQUITY,
                sector='Technology',
                beta=0.9
            )
        }
        
        # Create mock market data with some correlation
        dates = pd.date_range(end=datetime.now(), periods=252)  # 1 year of trading days
        np.random.seed(42)
        
        # Generate correlated returns
        corr_matrix = np.array([
            [1.0, 0.7],
            [0.7, 1.0]
        ])
        
        # Cholesky decomposition for correlated random numbers
        L = np.linalg.cholesky(corr_matrix)
        uncorrelated = np.random.normal(0, 0.015, (len(dates), 2))  # 1.5% daily vol
        correlated = np.dot(uncorrelated, L.T)
        
        # Create price series
        aapl_prices = 150 * np.exp(np.cumsum(correlated[:, 0]))
        msft_prices = 300 * np.exp(np.cumsum(correlated[:, 1]))
        
        self.market_data = {
            'AAPL': pd.DataFrame({
                'open': aapl_prices,
                'high': aapl_prices * 1.01,
                'low': aapl_prices * 0.99,
                'close': aapl_prices,
                'volume': np.random.lognormal(10, 1, len(dates))
            }, index=dates),
            'MSFT': pd.DataFrame({
                'open': msft_prices,
                'high': msft_prices * 1.01,
                'low': msft_prices * 0.99,
                'close': msft_prices,
                'volume': np.random.lognormal(10, 1, len(dates))
            }, index=dates)
        }
        
        self.model = VaRModel(confidence_level=0.95, lookback_days=252)
    
    def test_var_calculation(self):
        """Test VaR calculation with historical simulation."""
        result = self.model.calculate_risk_metrics(
            positions=self.positions,
            prices={'AAPL': 150.0, 'MSFT': 300.0},
            market_data=self.market_data
        )
        
        self.assertIn('var', result)
        self.assertIsInstance(result['var'], float)
        self.assertGreater(result['var'], 0)  # VaR should be positive
        
        # Check that VaR is within reasonable bounds (e.g., between 0.5% and 10% of portfolio)
        portfolio_value = sum(pos.market_value for pos in self.positions.values())
        self.assertGreater(result['var'], 0.005 * portfolio_value)
        self.assertLess(result['var'], 0.10 * portfolio_value)
    
    def test_empty_positions(self):
        """Test with empty positions."""
        result = self.model.calculate_risk_metrics(
            positions={},
            prices={},
            market_data=self.market_data
        )
        self.assertEqual(result, {})
    
    def test_insufficient_data(self):
        """Test with insufficient market data."""
        result = self.model.calculate_risk_metrics(
            positions=self.positions,
            prices={'AAPL': 150.0, 'MSFT': 300.0},
            market_data={}
        )
        self.assertEqual(result, {})
    
    def test_single_position(self):
        """Test VaR calculation with a single position."""
        single_pos = {'AAPL': self.positions['AAPL']}
        result = self.model.calculate_risk_metrics(
            positions=single_pos,
            prices={'AAPL': 150.0},
            market_data={'AAPL': self.market_data['AAPL']}
        )
        
        self.assertIn('var', result)
        self.assertGreater(result['var'], 0)


class TestCVaRModel(unittest.TestCase):
    """Test cases for the Conditional Value at Risk (CVaR) model."""
    
    def setUp(self):
        """Set up test data."""
        self.positions = {
            'AAPL': Position(
                symbol='AAPL',
                quantity=100,
                entry_price=150.0,
                current_price=150.0,
                asset_class=AssetClass.EQUITY
            )
        }
        
        # Mock VaR model
        self.var_model = MagicMock()
        self.var_model.calculate_risk_metrics.return_value = {
            'var': 0.05,  # 5% VaR
            'historical_returns': pd.Series([-0.01, -0.02, -0.03, -0.04, -0.05, -0.06, -0.07])
        }
        
        # Create CVaR model with mocked VaR model
        self.model = CVaRModel(confidence_level=0.95)
        self.model.var_model = self.var_model
    
    def test_cvar_calculation(self):
        """Test CVaR calculation."""
        result = self.model.calculate_risk_metrics(
            positions=self.positions,
            prices={'AAPL': 150.0},
            market_data={}
        )
        
        self.assertIn('cvar', result)
        self.assertIn('var', result)
        self.assertIsInstance(result['cvar'], float)
        self.assertGreater(result['cvar'], result['var'])  # CVaR should be more extreme than VaR
    
    def test_no_tail_observations(self):
        """Test when there are no observations in the tail."""
        # Mock VaR model with no tail observations
        self.var_model.calculate_risk_metrics.return_value = {
            'var': 0.10,  # 10% VaR
            'historical_returns': pd.Series([0.01, 0.02, 0.03])  # All positive returns
        }
        
        result = self.model.calculate_risk_metrics(
            positions=self.positions,
            prices={'AAPL': 150.0},
            market_data={}
        )
        
        self.assertEqual(result['cvar'], 0)  # CVaR should be 0 when no tail observations
    
    def test_empty_positions(self):
        """Test with empty positions."""
        result = self.model.calculate_risk_metrics(
            positions={},
            prices={},
            market_data={}
        )
        self.assertEqual(result, {})


class TestFactorRiskModel(unittest.TestCase):
    """Test cases for the Factor Risk model."""
    
    def setUp(self):
        """Set up test data."""
        self.positions = {
            'AAPL': Position(
                symbol='AAPL',
                quantity=100,
                entry_price=150.0,
                current_price=150.0,
                asset_class=AssetClass.EQUITY,
                sector='Technology',
                beta=1.2
            ),
            'MSFT': Position(
                symbol='MSFT',
                quantity=50,
                entry_price=300.0,
                current_price=300.0,
                asset_class=AssetClass.EQUITY,
                sector='Technology',
                beta=0.9
            )
        }
        
        # Create mock market data
        dates = pd.date_range(end=datetime.now(), periods=100)
        np.random.seed(42)
        
        # Generate some fake factor returns
        self.factor_names = ['market', 'size', 'value', 'momentum', 'volatility']
        self.factor_returns = {
            factor: pd.Series(
                np.random.normal(0, 0.01, len(dates)),
                index=dates
            )
            for factor in self.factor_names
        }
        
        self.model = FactorRiskModel(factors=self.factor_names)
        
        # Mock the factor exposure simulation
        self.exposure_patcher = patch.object(
            self.model, 
            '_simulate_factor_exposure',
            side_effect=lambda s, f: {
                'market': 1.2,
                'size': 0.8,
                'value': -0.5,
                'momentum': 0.3,
                'volatility': -0.2
            }.get(f, 0.0)
        )
        self.mock_exposure = self.exposure_patcher.start()
    
    def tearDown(self):
        """Clean up patches."""
        self.exposure_patcher.stop()
    
    def test_factor_exposures(self):
        """Test factor exposure calculation."""
        result = self.model.calculate_risk_metrics(
            positions=self.positions,
            prices={'AAPL': 150.0, 'MSFT': 300.0},
            market_data={},
            factor_returns=self.factor_returns
        )
        
        self.assertIn('factor_exposures', result)
        self.assertIn('factor_contributions', result)
        self.assertIn('factor_volatility', result)
        
        # Check that exposures are calculated for all factors
        self.assertEqual(set(result['factor_exposures'].keys()), set(self.factor_names))
        
        # Check that contributions sum to a reasonable value
        total_contribution = sum(abs(v) for v in result['factor_contributions'].values())
        self.assertGreater(total_contribution, 0)
    
    def test_simulated_factor_returns(self):
        """Test the factor return simulation."""
        # Create minimal market data
        dates = pd.date_range(end=datetime.now(), periods=10)
        market_data = {
            'AAPL': pd.DataFrame({
                'close': [100 + i for i in range(10)],
                'volume': [1000] * 10
            }, index=dates)
        }
        
        # This will trigger the simulated factor returns
        result = self.model.calculate_risk_metrics(
            positions={'AAPL': self.positions['AAPL']},
            prices={'AAPL': 105.0},
            market_data=market_data
        )
        
        # Should still get results even with simulated factor returns
        self.assertIn('factor_exposures', result)
        self.assertEqual(len(result['factor_exposures']), len(self.factor_names))
    
    def test_empty_positions(self):
        """Test with empty positions."""
        result = self.model.calculate_risk_metrics(
            positions={},
            prices={},
            market_data={}
        )
        self.assertEqual(result, {})


class TestCircuitBreaker(unittest.TestCase):
    """Test cases for the CircuitBreaker class."""
    
    def setUp(self):
        """Set up test data."""
        self.thresholds = RiskThresholds(
            max_position_size_pct=0.05,       # 5% max position size
            max_sector_exposure_pct=0.25,     # 25% max sector exposure
            max_drawdown_daily_pct=0.05,      # 5% max daily drawdown
            max_drawdown_total_pct=0.15,      # 15% max total drawdown
            max_leverage=3.0,                 # 3x max leverage
            min_liquidity_usd=10000.0         # $10k min liquidity
        )
        self.circuit_breaker = CircuitBreaker(self.thresholds)
    
    def test_position_size_check(self):
        """Test position size limit check."""
        # Create a position that's too large (10% of portfolio, limit is 5%)
        position = Position(
            symbol='AAPL',
            quantity=1000,
            entry_price=100.0,
            current_price=100.0,
            asset_class=AssetClass.EQUITY
        )
        portfolio_value = 100000.0  # $100k portfolio
        
        # Check should fail
        result, msg = self.circuit_breaker.check_position_size(position, portfolio_value)
        self.assertFalse(result)
        self.assertIn("exceeds maximum", msg)
        
        # Check with valid position size (2% of portfolio)
        position.quantity = 20  # $2k position in $100k portfolio (2%)
        result, msg = self.circuit_breaker.check_position_size(position, portfolio_value)
        self.assertTrue(result)
        self.assertEqual(msg, "")
    
    def test_sector_exposure_check(self):
        """Test sector exposure limit check."""
        # Create some positions in the same sector
        positions = {
            'AAPL': Position(
                symbol='AAPL',
                quantity=100,
                entry_price=150.0,
                current_price=150.0,
                asset_class=AssetClass.EQUITY,
                sector='Technology'
            ),
            'MSFT': Position(
                symbol='MSFT',
                quantity=50,
                entry_price=300.0,
                current_price=300.0,
                asset_class=AssetClass.EQUITY,
                sector='Technology'
            )
        }
        
        # Total portfolio value = $30k, Tech sector = $30k (100% exposure, limit is 25%)
        result, msg = self.circuit_breaker.check_sector_exposure(positions, 'Technology')
        self.assertFalse(result)
        self.assertIn("exceeds maximum", msg)
        
        # Add positions in other sectors to dilute the Tech exposure
        positions['JPM'] = Position(
            symbol='JPM',
            quantity=200,
            entry_price=150.0,
            current_price=150.0,
            asset_class=AssetClass.EQUITY,
            sector='Financials'
        )
        
        # Now Tech is $30k out of $60k (50% exposure, still over 25% limit)
        result, msg = self.circuit_breaker.check_sector_exposure(positions, 'Technology')
        self.assertFalse(result)
        
        # Add more non-Tech positions to get Tech under 25%
        positions['XOM'] = Position(
            symbol='XOM',
            quantity=200,
            entry_price=100.0,
            current_price=100.0,
            asset_class=AssetClass.COMMODITY,
            sector='Energy'
        )
        
        # Now Tech is $30k out of $80k (37.5% exposure, still over 25% limit)
        positions['JPM'].quantity = 400  # Increase Financials to $60k
        
        # Now Tech is $30k out of $120k (25% exposure, at limit)
        result, msg = self.circuit_breaker.check_sector_exposure(positions, 'Technology')
        self.assertTrue(result)  # At limit is still acceptable
    
    def test_drawdown_checks(self):
        """Test drawdown limit checks."""
        # Test daily drawdown check
        peak_value = 100000.0
        current_value = 94000.0  # 6% drawdown (over 5% limit)
        
        result, msg = self.circuit_breaker.check_drawdown(
            peak_value, current_value, is_daily=True
        )
        self.assertFalse(result)
        self.assertIn("Daily drawdown", msg)
        
        # Test total drawdown check
        current_value = 83000.0  # 17% drawdown (over 15% limit)
        result, msg = self.circuit_breaker.check_drawdown(
            peak_value, current_value, is_daily=False
        )
        self.assertFalse(result)
        self.assertIn("Total drawdown", msg)
        
        # Test within limits
        current_value = 97000.0  # 3% drawdown
        result, msg = self.circuit_breaker.check_drawdown(
            peak_value, current_value, is_daily=True
        )
        self.assertTrue(result)
        self.assertEqual(msg, "")
    
    def test_leverage_check(self):
        ""Test leverage limit check."""
        # Test over leverage
        total_position_value = 400000.0  # $400k positions
        equity = 100000.0  # $100k equity = 4x leverage (over 3x limit)
        
        result, msg = self.circuit_breaker.check_leverage(total_position_value, equity)
        self.assertFalse(result)
        self.assertIn("exceeds maximum", msg)
        
        # Test within limits
        total_position_value = 200000.0  # 2x leverage
        result, msg = self.circuit_breaker.check_leverage(total_position_value, equity)
        self.assertTrue(result)
        self.assertEqual(msg, "")
    
    def test_trigger_and_reset(self):
        ""Test circuit breaker trigger and reset functionality."""
        self.assertFalse(self.circuit_breaker.triggered)
        self.assertIsNone(self.circuit_breaker.trigger_time)
        
        # Trigger the circuit breaker
        reason = "Test trigger"
        self.circuit_breaker.trigger(reason)
        
        self.assertTrue(self.circuit_breaker.triggered)
        self.assertEqual(self.circuit_breaker.trigger_reason, reason)
        self.assertIsNotNone(self.circuit_breaker.trigger_time)
        
        # Reset the circuit breaker
        self.circuit_breaker.reset()
        
        self.assertFalse(self.circuit_breaker.triggered)
        self.assertEqual(self.circuit_breaker.trigger_reason, "")
        self.assertIsNone(self.circuit_breaker.trigger_time)


class TestPositionSizer(unittest.TestCase):
    """Test cases for the PositionSizer class."""
    
    def setUp(self):
        ""Set up test data."""
        self.risk_model = MagicMock()
        self.thresholds = RiskThresholds()
        self.sizer = PositionSizer(self.risk_model, self.thresholds)
        
        # Mock position and market data
        self.positions = {
            'AAPL': Position(
                symbol='AAPL',
                quantity=100,
                entry_price=150.0,
                current_price=150.0,
                asset_class=AssetClass.EQUITY
            )
        }
        
        self.market_data = {
            'AAPL': pd.DataFrame({
                'close': [145.0, 146.0, 147.0, 148.0, 149.0, 150.0],
                'volume': [1000] * 6
            })
        }
    
    def test_calculate_position_size_with_stop_loss(self):
        ""Test position size calculation with stop loss."""
        # Test case: $100k account, 1% risk, $150 entry, $140 stop ($10 risk per share)
        # Expected position size: ($100k * 1%) / $10 = 100 shares
        size, details = self.sizer.calculate_position_size(
            symbol='AAPL',
            entry_price=150.0,
            stop_loss=140.0,
            account_equity=100000.0,
            portfolio=self.positions,
            market_data=self.market_data,
            risk_per_trade_pct=1.0
        )
        
        expected_size = 100.0  # ($100k * 1%) / ($150 - $140)
        self.assertAlmostEqual(size, expected_size, places=2)
        self.assertAlmostEqual(details['position_value'], expected_size * 150.0, places=2)
        self.assertAlmostEqual(details['risk_amount'], 1000.0, places=2)  # 1% of $100k
        self.assertAlmostEqual(details['risk_reward_ratio'], 1.0, places=2)  # (160-150)/(150-140) = 1.0
    
    def test_calculate_position_size_without_stop_loss(self):
        ""Test position size calculation without stop loss."""
        # Without stop loss, should use fixed percentage of equity
        size, details = self.sizer.calculate_position_size(
            symbol='AAPL',
            entry_price=150.0,
            stop_loss=None,
            account_equity=100000.0,
            portfolio=self.positions,
            market_data=self.market_data,
            risk_per_trade_pct=1.0
        )
        
        # Should be 1% of equity divided by entry price
        expected_size = (100000.0 * 0.01) / 150.0
        self.assertAlmostEqual(size, expected_size, places=2)
    
    def test_position_size_with_leverage_limit(self):
        ""Test position size respects leverage limits."""
        # Test with existing positions that already use most of the available margin
        # $100k account, 3x max leverage = $300k total position value allowed
        # Existing position: 100 shares * $150 = $15k
        # Available for new position: $300k - $15k = $285k
        
        size, details = self.sizer.calculate_position_size(
            symbol='MSFT',
            entry_price=300.0,
            stop_loss=270.0,
            account_equity=100000.0,
            portfolio=self.positions,
            market_data=self.market_data,
            risk_per_trade_pct=1.0
        )
        
        # Should be limited by available margin, not by risk calculation
        max_position_value = 285000.0  # $300k - $15k existing
        expected_max_shares = max_position_value / 300.0  # MSFT at $300
        
        # Position size should be the minimum of risk-based size and margin-based size
        self.assertLessEqual(size * 300.0, max_position_value)
    
    def test_round_to_lot_size(self):
        ""Test rounding to lot size."""
        # Test with default lot size of 1.0
        size = self.sizer._round_to_lot_size(123.456, 'AAPL')
        self.assertEqual(size, 123.0)
        
        # Test with custom lot size
        with patch.object(self.sizer, '_get_lot_size', return_value=0.1):
            size = self.sizer._round_to_lot_size(123.456, 'AAPL')
            self.assertEqual(size, 123.4)
    
    def test_risk_reward_ratio_calculation(self):
        ""Test risk/reward ratio calculation."""
        # Test normal case
        ratio = self.sizer._calculate_risk_reward_ratio(
            entry=100.0,
            stop_loss=95.0,
            take_profit=110.0
        )
        self.assertAlmostEqual(ratio, 2.0)  # (110-100)/(100-95) = 2.0
        
        # Test zero risk (should return 0)
        ratio = self.sizer._calculate_risk_reward_ratio(
            entry=100.0,
            stop_loss=100.0,  # No risk
            take_profit=110.0
        )
        self.assertEqual(ratio, 0.0)


if __name__ == '__main__':
    unittest.main()
