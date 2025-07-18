"""
Unit tests for advanced risk models.

This module contains comprehensive tests for the risk models implemented in
`core.risk.advanced_risk_models`.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime
from unittest.mock import patch, MagicMock

# Import the models to test
from core.risk.advanced_risk_models import (
    VaRModel,
    CVaRModel,
    FactorRiskModel,
    RiskModelType
)

class TestVaRModel(unittest.TestCase):
    """Test cases for the VaRModel class."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.returns = np.random.normal(0.0005, 0.02, 1000)
        self.var_model = VaRModel(confidence_level=0.95)
    
    def test_historical_var_calculation(self):
        """Test historical VaR calculation."""
        var_model = VaRModel(confidence_level=0.95, method='historical')
        result = var_model.calculate(self.returns)
        
        # Basic validation
        self.assertIsInstance(result.value, float)
        self.assertGreater(result.value, 0)
        self.assertEqual(result.model_type, RiskModelType.HISTORICAL_VAR)
        self.assertEqual(result.confidence_level, 0.95)
        
        # Check additional metrics
        self.assertIn('min_return', result.additional_metrics)
        self.assertIn('max_return', result.additional_metrics)
        self.assertIn('mean_return', result.additional_metrics)
        self.assertIn('std_dev', result.additional_metrics)
    
    def test_parametric_var_calculation(self):
        """Test parametric VaR calculation."""
        var_model = VaRModel(confidence_level=0.99, method='parametric')
        result = var_model.calculate(self.returns)
        
        # Basic validation
        self.assertIsInstance(result.value, float)
        self.assertGreater(result.value, 0)
        self.assertEqual(result.model_type, RiskModelType.PARAMETRIC_VAR)
        self.assertEqual(result.confidence_level, 0.99)
        
        # Check parameters
        self.assertIn('mean', result.parameters)
        self.assertIn('std_dev', result.parameters)
        self.assertIn('z_score', result.parameters)
        
        # Check additional metrics
        self.assertIn('skewness', result.additional_metrics)
        self.assertIn('kurtosis', result.additional_metrics)
        self.assertIn('is_normal', result.additional_metrics)
    
    def test_monte_carlo_var_calculation(self):
        """Test Monte Carlo VaR calculation."""
        var_model = VaRModel(
            confidence_level=0.95, 
            method='monte_carlo',
            num_simulations=5000
        )
        result = var_model.calculate(self.returns, time_horizon=5)
        
        # Basic validation
        self.assertIsInstance(result.value, float)
        self.assertGreater(result.value, 0)
        self.assertEqual(result.model_type, RiskModelType.MONTE_CARLO_VAR)
        
        # Check parameters
        self.assertEqual(result.parameters['num_simulations'], 5000)
        self.assertEqual(result.parameters['time_horizon'], 5)
        
        # Check additional metrics
        self.assertIn('min_simulated_return', result.additional_metrics)
        self.assertIn('max_simulated_return', result.additional_metrics)
        self.assertIn('mean_simulated_return', result.additional_metrics)
    
    def test_invalid_method(self):
        """Test initialization with invalid method raises error."""
        with self.assertRaises(ValueError):
            VaRModel(method='invalid_method')
    
    def test_insufficient_data(self):
        """Test with insufficient data raises error."""
        with self.assertRaises(ValueError):
            self.var_model.calculate(np.array([0.01]))


class TestCVaRModel(unittest.TestCase):
    """Test cases for the CVaRModel class."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.returns = np.random.normal(0.0005, 0.02, 1000)
        self.cvar_model = CVaRModel(confidence_level=0.95)
    
    def test_cvar_calculation(self):
        """Test CVaR calculation."""
        result = self.cvar_model.calculate(self.returns)
        
        # Basic validation
        self.assertIsInstance(result.value, float)
        self.assertGreater(result.value, 0)
        self.assertEqual(result.model_type, RiskModelType.CVAR)
        
        # CVaR should be greater than or equal to VaR
        self.assertGreaterEqual(result.value, result.additional_metrics['var'])
        
        # Check additional metrics
        self.assertIn('var', result.additional_metrics)
        self.assertIn('tail_observations', result.additional_metrics)
        self.assertIn('tail_min', result.additional_metrics)
        self.assertIn('tail_max', result.additional_metrics)
    
    def test_cvar_with_no_tail_observations(self):
        """Test CVaR when there are no returns in the tail."""
        # Create returns that are all positive (no losses)
        positive_returns = np.abs(np.random.normal(0.01, 0.005, 1000))
        result = self.cvar_model.calculate(positive_returns)
        
        # Should still return a valid result using fallback
        self.assertIsInstance(result.value, float)
        self.assertGreaterEqual(result.value, 0)


class TestFactorRiskModel(unittest.TestCase):
    """Test cases for the FactorRiskModel class."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        
        # Create sample assets and factors
        self.assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']
        self.factors = ['market', 'size', 'value', 'momentum', 'volatility']
        
        # Generate random factor exposures
        self.factor_exposures = {
            asset: {factor: np.random.normal(0, 1) for factor in self.factors}
            for asset in self.assets
        }
        
        # Generate random factor returns
        self.factor_returns = pd.DataFrame(
            np.random.multivariate_normal(
                mean=[0.0005] * len(self.factors),
                cov=np.eye(len(self.factors)) * 0.0001,
                size=1000
            ),
            columns=self.factors
        )
        
        # Generate specific risk
        self.specific_risk = {
            asset: np.random.uniform(0.01, 0.03) 
            for asset in self.assets
        }
        
        # Create position values
        self.position_values = {
            asset: (i + 1) * 1_000_000  # 1M to 5M positions
            for i, asset in enumerate(self.assets)
        }
        
        # Create model instance
        self.factor_model = FactorRiskModel(
            factors=self.factors,
            confidence_level=0.95,
            risk_free_rate=0.02
        )
    
    def test_factor_risk_calculation(self):
        """Test factor risk model calculation."""
        result = self.factor_model.calculate(
            returns=pd.Series(np.random.normal(0.0005, 0.02, 1000)),
            position_values=self.position_values,
            factor_exposures=self.factor_exposures,
            factor_returns=self.factor_returns,
            specific_risk=self.specific_risk
        )
        
        # Basic validation
        self.assertIsInstance(result.value, float)
        self.assertGreater(result.value, 0)
        self.assertEqual(result.model_type, RiskModelType.FACTOR_MODEL)
        
        # Check parameters
        self.assertEqual(result.parameters['num_factors'], len(self.factors))
        self.assertEqual(result.parameters['num_assets'], len(self.assets))
        
        # Check additional metrics
        self.assertIn('factor_risk', result.additional_metrics)
        self.assertIn('specific_risk', result.additional_metrics)
        self.assertIn('risk_contributions', result.additional_metrics)
        self.assertIn('factor_exposures', result.additional_metrics)
        
        # Check risk contributions
        risk_contrib = result.additional_metrics['risk_contributions']
        self.assertIn('factor_contributions', risk_contrib)
        self.assertIn('asset_contributions', risk_contrib)
        self.assertIn('specific_risk_contribution', risk_contrib)
        
        # Factor contributions should sum to 1 (or close to it)
        factor_contrib_sum = sum(risk_contrib['factor_contributions'].values())
        self.assertAlmostEqual(factor_contrib_sum, 1.0, delta=0.1)  # Allow some numerical error
    
    def test_missing_factor_returns(self):
        """Test with missing factor returns (should still work but with limited functionality)."""
        result = self.factor_model.calculate(
            returns=pd.Series(np.random.normal(0.0005, 0.02, 1000)),
            position_values=self.position_values,
            factor_exposures=self.factor_exposures
            # No factor_returns or specific_risk provided
        )
        
        # Should still return a result, but with limited information
        self.assertIsInstance(result.value, float)
        self.assertTrue(np.isnan(result.additional_metrics['specific_risk']))
    
    def test_single_asset_portfolio(self):
        """Test with a single asset portfolio."""
        single_asset = self.assets[0]
        result = self.factor_model.calculate(
            returns=pd.Series(np.random.normal(0.0005, 0.02, 1000)),
            position_values={single_asset: 1_000_000},
            factor_exposures={single_asset: self.factor_exposures[single_asset]},
            factor_returns=self.factor_returns,
            specific_risk={single_asset: self.specific_risk[single_asset]}
        )
        
        # Basic validation
        self.assertIsInstance(result.value, float)
        self.assertGreater(result.value, 0)


if __name__ == '__main__':
    unittest.main()
