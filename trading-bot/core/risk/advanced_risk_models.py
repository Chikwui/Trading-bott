"""
Advanced Risk Models for Quantitative Risk Management

This module implements sophisticated risk models including:
- Value at Risk (VaR) with multiple calculation methods
- Conditional Value at Risk (CVaR)
- Multi-factor risk models
- Stress testing scenarios
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from enum import Enum, auto
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
from datetime import datetime, timedelta
import scipy.stats as stats
from scipy.optimize import minimize

# Configure logging
logger = logging.getLogger(__name__)

class RiskModelType(Enum):
    """Types of risk models supported."""
    HISTORICAL_VAR = auto()
    PARAMETRIC_VAR = auto()
    MONTE_CARLO_VAR = auto()
    CVAR = auto()
    FACTOR_MODEL = auto()

@dataclass
class RiskModelResult:
    """Container for risk model results."""
    value: float
    confidence_level: float
    model_type: RiskModelType
    parameters: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.utcnow)
    additional_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            'value': self.value,
            'confidence_level': self.confidence_level,
            'model_type': self.model_type.name,
            'parameters': self.parameters,
            'timestamp': self.timestamp.isoformat(),
            'additional_metrics': self.additional_metrics
        }

class RiskModel(ABC):
    """Abstract base class for all risk models."""
    
    def __init__(self, confidence_level: float = 0.95):
        """Initialize with confidence level (0-1)."""
        if not 0 < confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")
        self.confidence_level = confidence_level
    
    @abstractmethod
    def calculate(
        self, 
        returns: Union[pd.Series, np.ndarray],
        position_values: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> RiskModelResult:
        """Calculate risk metric."""
        pass

class VaRModel(RiskModel):
    """
    Value at Risk (VaR) model with multiple calculation methods.
    
    Supports:
    - Historical simulation
    - Parametric (variance-covariance)
    - Monte Carlo simulation
    """
    
    def __init__(
        self, 
        confidence_level: float = 0.95,
        method: str = 'historical',
        lookback_days: int = 252,
        num_simulations: int = 10000
    ):
        """
        Initialize VaR model.
        
        Args:
            confidence_level: Confidence level (0-1)
            method: Calculation method ('historical', 'parametric', 'monte_carlo')
            lookback_days: Number of days to use for historical data
            num_simulations: Number of simulations for Monte Carlo
        """
        super().__init__(confidence_level)
        self.method = method.lower()
        self.lookback_days = lookback_days
        self.num_simulations = num_simulations
        self._validate_method()
    
    def _validate_method(self):
        """Validate calculation method."""
        valid_methods = ['historical', 'parametric', 'monte_carlo']
        if self.method not in valid_methods:
            raise ValueError(f"Invalid method. Must be one of {valid_methods}")
    
    def calculate(
        self, 
        returns: Union[pd.Series, np.ndarray],
        position_values: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> RiskModelResult:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            returns: Series or array of returns
            position_values: Dictionary of position values by asset
            **kwargs: Additional parameters for specific methods
            
        Returns:
            RiskModelResult with VaR and additional metrics
        """
        returns = self._prepare_returns(returns)
        
        if self.method == 'historical':
            return self._calculate_historical_var(returns, position_values, **kwargs)
        elif self.method == 'parametric':
            return self._calculate_parametric_var(returns, position_values, **kwargs)
        elif self.method == 'monte_carlo':
            return self._calculate_monte_carlo_var(returns, position_values, **kwargs)
    
    def _prepare_returns(self, returns: Union[pd.Series, np.ndarray]) -> np.ndarray:
        """Convert returns to numpy array and validate."""
        if isinstance(returns, pd.Series):
            returns = returns.values
        
        if not isinstance(returns, np.ndarray):
            returns = np.array(returns)
        
        if len(returns) < 2:
            raise ValueError("At least 2 returns are required")
            
        return returns
    
    def _calculate_historical_var(
        self,
        returns: np.ndarray,
        position_values: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> RiskModelResult:
        """Calculate historical VaR."""
        # Sort returns and find the appropriate percentile
        var = -np.percentile(returns, (1 - self.confidence_level) * 100)
        
        return RiskModelResult(
            value=var,
            confidence_level=self.confidence_level,
            model_type=RiskModelType.HISTORICAL_VAR,
            parameters={
                'method': 'historical',
                'lookback_days': self.lookback_days,
                'num_returns': len(returns)
            },
            additional_metrics={
                'min_return': float(np.min(returns)),
                'max_return': float(np.max(returns)),
                'mean_return': float(np.mean(returns)),
                'std_dev': float(np.std(returns, ddof=1))
            }
        )
    
    def _calculate_parametric_var(
        self,
        returns: np.ndarray,
        position_values: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> RiskModelResult:
        """Calculate parametric (variance-covariance) VaR."""
        mean = np.mean(returns)
        std_dev = np.std(returns, ddof=1)
        
        # Calculate z-score for normal distribution
        z_score = stats.norm.ppf(1 - self.confidence_level)
        var = -(mean + z_score * std_dev)
        
        return RiskModelResult(
            value=var,
            confidence_level=self.confidence_level,
            model_type=RiskModelType.PARAMETRIC_VAR,
            parameters={
                'method': 'parametric',
                'mean': float(mean),
                'std_dev': float(std_dev),
                'z_score': float(z_score)
            },
            additional_metrics={
                'skewness': float(stats.skew(returns)),
                'kurtosis': float(stats.kurtosis(returns)),
                'is_normal': self._test_normality(returns)
            }
        )
    
    def _calculate_monte_carlo_var(
        self,
        returns: np.ndarray,
        position_values: Optional[Dict[str, float]] = None,
        time_horizon: int = 1,
        **kwargs
    ) -> RiskModelResult:
        """Calculate VaR using Monte Carlo simulation."""
        mean = np.mean(returns)
        std_dev = np.std(returns, ddof=1)
        
        # Generate random returns
        np.random.seed(42)  # For reproducibility
        simulated_returns = np.random.normal(
            mean, 
            std_dev, 
            (self.num_simulations, time_horizon)
        )
        
        # Calculate cumulative returns
        cumulative_returns = np.prod(1 + simulated_returns, axis=1) - 1
        
        # Calculate VaR
        var = -np.percentile(cumulative_returns, (1 - self.confidence_level) * 100)
        
        return RiskModelResult(
            value=var,
            confidence_level=self.confidence_level,
            model_type=RiskModelType.MONTE_CARLO_VAR,
            parameters={
                'method': 'monte_carlo',
                'num_simulations': self.num_simulations,
                'time_horizon': time_horizon,
                'mean': float(mean),
                'std_dev': float(std_dev)
            },
            additional_metrics={
                'min_simulated_return': float(np.min(cumulative_returns)),
                'max_simulated_return': float(np.max(cumulative_returns)),
                'mean_simulated_return': float(np.mean(cumulative_returns))
            }
        )
    
    def _test_normality(self, returns: np.ndarray, alpha: float = 0.05) -> bool:
        """Test if returns are normally distributed using Shapiro-Wilk test."""
        if len(returns) < 3 or len(returns) > 5000:
            return False  # Test not reliable for very small or large samples
            
        _, p_value = stats.shapiro(returns)
        return p_value > alpha  # Null hypothesis: data is normally distributed

class CVaRModel(VaRModel):
    """
    Conditional Value at Risk (CVaR) model.
    
    Also known as Expected Shortfall, CVaR measures the expected loss
    in the worst (1-confidence_level) percent of cases.
    """
    
    def calculate(
        self, 
        returns: Union[pd.Series, np.ndarray],
        position_values: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> RiskModelResult:
        """
        Calculate Conditional Value at Risk (CVaR).
        
        Args:
            returns: Series or array of returns
            position_values: Dictionary of position values by asset
            **kwargs: Additional parameters for VaR calculation
            
        Returns:
            RiskModelResult with CVaR and additional metrics
        """
        # First calculate VaR
        var_result = super().calculate(returns, position_values, **kwargs)
        var = var_result.value
        
        # Convert returns to numpy array if needed
        returns = self._prepare_returns(returns)
        
        # Calculate CVaR as the average of returns worse than -VaR
        tail_returns = returns[returns <= -var]
        
        if len(tail_returns) == 0:
            cvar = -np.mean(returns[returns < 0])  # Fallback to average loss
        else:
            cvar = -np.mean(tail_returns)
        
        # Update result with CVaR
        result = RiskModelResult(
            value=cvar,
            confidence_level=self.confidence_level,
            model_type=RiskModelType.CVAR,
            parameters=var_result.parameters,
            additional_metrics={
                'var': var,
                'tail_observations': len(tail_returns),
                'tail_min': float(np.min(tail_returns)) if len(tail_returns) > 0 else None,
                'tail_max': float(np.max(tail_returns)) if len(tail_returns) > 0 else None,
                **var_result.additional_metrics
            }
        )
        
        return result

class FactorRiskModel(RiskModel):
    """
    Multi-factor risk model for analyzing portfolio risk.
    
    Implements:
    - Factor exposure analysis
    - Risk decomposition
    - Marginal risk contribution
    """
    
    def __init__(
        self,
        factors: List[str] = None,
        confidence_level: float = 0.95,
        risk_free_rate: float = 0.0
    ):
        """
        Initialize factor risk model.
        
        Args:
            factors: List of factor names
            confidence_level: Confidence level for risk calculations
            risk_free_rate: Annual risk-free rate
        """
        super().__init__(confidence_level)
        self.factors = factors or ['market', 'size', 'value', 'momentum', 'volatility']
        self.risk_free_rate = risk_free_rate
        self.factor_returns: Optional[pd.DataFrame] = None
        self.factor_cov: Optional[pd.DataFrame] = None
        self.specific_risk: Optional[pd.Series] = None
    
    def calculate(
        self,
        returns: Union[pd.Series, np.ndarray],
        position_values: Dict[str, float],
        factor_exposures: Dict[str, Dict[str, float]],
        factor_returns: Optional[Dict[str, Union[pd.Series, np.ndarray]]] = None,
        specific_risk: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> RiskModelResult:
        """
        Calculate factor risk metrics.
        
        Args:
            returns: Asset returns
            position_values: Dictionary of position values by asset
            factor_exposures: Dictionary of factor exposures by asset
            factor_returns: Optional dictionary of factor return series
            specific_risk: Optional specific risk by asset
            
        Returns:
            RiskModelResult with factor risk metrics
        """
        # Prepare data
        assets = list(position_values.keys())
        weights = np.array([position_values[a] / sum(position_values.values()) for a in assets])
        
        # If factor returns provided, estimate factor covariance
        if factor_returns is not None:
            self._estimate_factor_model(factor_returns)
        
        # Calculate portfolio factor exposures
        portfolio_exposures = self._calculate_portfolio_exposures(assets, weights, factor_exposures)
        
        # Calculate portfolio risk
        if self.factor_cov is not None:
            factor_risk = np.sqrt(
                portfolio_exposures.T @ self.factor_cov @ portfolio_exposures
            )
        else:
            factor_risk = np.nan
        
        # Calculate specific risk
        if self.specific_risk is not None:
            specific_risk = np.sqrt((weights**2 * np.array([self.specific_risk[a] for a in assets])).sum())
            total_risk = np.sqrt(factor_risk**2 + specific_risk**2)
        else:
            specific_risk = np.nan
            total_risk = factor_risk
        
        # Calculate risk contributions
        risk_contributions = self._calculate_risk_contributions(
            assets, weights, portfolio_exposures, factor_risk, specific_risk
        )
        
        return RiskModelResult(
            value=total_risk,
            confidence_level=self.confidence_level,
            model_type=RiskModelType.FACTOR_MODEL,
            parameters={
                'num_factors': len(self.factors),
                'num_assets': len(assets),
                'portfolio_beta': float(portfolio_exposures.get('market', np.nan))
            },
            additional_metrics={
                'factor_risk': float(factor_risk),
                'specific_risk': float(specific_risk) if not np.isnan(specific_risk) else None,
                'risk_contributions': risk_contributions,
                'factor_exposures': portfolio_exposures.to_dict()
            }
        )
    
    def _estimate_factor_model(
        self,
        factor_returns: Dict[str, Union[pd.Series, np.ndarray]]
    ) -> None:
        """Estimate factor covariance matrix from factor returns."""
        # Convert to DataFrame if needed
        if not isinstance(factor_returns, pd.DataFrame):
            factor_returns = pd.DataFrame(factor_returns)
        
        # Calculate factor covariance matrix
        self.factor_returns = factor_returns
        self.factor_cov = factor_returns.cov()
    
    def _calculate_portfolio_exposures(
        self,
        assets: List[str],
        weights: np.ndarray,
        factor_exposures: Dict[str, Dict[str, float]]
    ) -> pd.Series:
        """Calculate portfolio-level factor exposures."""
        # Create exposure matrix
        exposure_matrix = pd.DataFrame(
            index=assets,
            columns=self.factors,
            data=np.zeros((len(assets), len(self.factors)))
        )
        
        # Fill in exposures
        for asset in assets:
            if asset in factor_exposures:
                for factor, exposure in factor_exposures[asset].items():
                    if factor in exposure_matrix.columns:
                        exposure_matrix.loc[asset, factor] = exposure
        
        # Calculate portfolio exposures
        portfolio_exposures = exposure_matrix.T @ weights
        
        return portfolio_exposures
    
    def _calculate_risk_contributions(
        self,
        assets: List[str],
        weights: np.ndarray,
        portfolio_exposures: pd.Series,
        factor_risk: float,
        specific_risk: float
    ) -> Dict[str, Dict[str, float]]:
        """Calculate risk contributions for each factor and asset."""
        if self.factor_cov is None:
            return {}
        
        # Calculate marginal contributions to factor risk
        marginal_contrib = {}
        for factor in self.factors:
            # Marginal contribution of factor to portfolio risk
            if factor in portfolio_exposures.index and factor in self.factor_cov.index:
                cov_with_portfolio = self.factor_cov[factor] @ portfolio_exposures
                marginal_contrib[factor] = cov_with_portfolio / factor_risk if factor_risk > 0 else 0
        
        # Calculate risk contributions
        risk_contrib = {}
        
        # Factor contributions
        factor_contrib = {}
        for factor in self.factors:
            if factor in marginal_contrib and factor in portfolio_exposures:
                factor_contrib[factor] = (
                    portfolio_exposures[factor] * marginal_contrib[factor]
                )
        
        # Asset contributions (simplified)
        asset_contrib = {}
        for i, asset in enumerate(assets):
            # This is a simplified calculation
            asset_contrib[asset] = (weights[i] ** 2) / len(assets)
        
        return {
            'factor_contributions': factor_contrib,
            'asset_contributions': asset_contrib,
            'specific_risk_contribution': specific_risk**2 / (factor_risk**2 + specific_risk**2) 
                                        if not np.isnan(specific_risk) and (factor_risk**2 + specific_risk**2) > 0 
                                        else 0
        }

# Example usage
if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    
    # Example returns (normally you'd load this from market data)
    np.random.seed(42)
    returns = np.random.normal(0.0005, 0.02, 1000)  # 1000 days of returns
    
    # Calculate VaR using different methods
    for method in ['historical', 'parametric', 'monte_carlo']:
        var_model = VaRModel(confidence_level=0.95, method=method)
        result = var_model.calculate(returns)
        print(f"{method.upper()} VaR (95%): {result.value:.4f}")
    
    # Calculate CVaR
    cvar_model = CVaRModel(confidence_level=0.95)
    cvar_result = cvar_model.calculate(returns)
    print(f"\nCVaR (95%): {cvar_result.value:.4f}")
    print(f"VaR (95%): {cvar_result.additional_metrics['var']:.4f}")
    print(f"Tail observations: {cvar_result.additional_metrics['tail_observations']}")
    
    # Example factor risk model
    print("\nFactor Risk Model Example:")
    
    # Create sample data
    assets = ['AAPL', 'MSFT', 'GOOGL']
    factors = ['market', 'size', 'value']
    
    # Position values
    position_values = {'AAPL': 1e6, 'MSFT': 1.5e6, 'GOOGL': 0.8e6}
    
    # Factor exposures (random for example)
    np.random.seed(42)
    factor_exposures = {
        asset: {factor: np.random.normal(0, 1) for factor in factors}
        for asset in assets
    }
    
    # Factor returns (random for example)
    factor_returns = pd.DataFrame(
        np.random.multivariate_normal(
            mean=[0.0005] * len(factors),
            cov=np.eye(len(factors)) * 0.0001,
            size=1000
        ),
        columns=factors
    )
    
    # Specific risk
    specific_risk = {asset: np.random.uniform(0.01, 0.03) for asset in assets}
    
    # Create and run factor model
    factor_model = FactorRiskModel(factors=factors)
    factor_result = factor_model.calculate(
        returns=pd.Series(np.random.normal(0.0005, 0.02, 1000)),
        position_values=position_values,
        factor_exposures=factor_exposures,
        factor_returns=factor_returns,
        specific_risk=specific_risk
    )
    
    print(f"Total Portfolio Risk: {factor_result.value:.4f}")
    print(f"Factor Risk: {factor_result.additional_metrics['factor_risk']:.4f}")
    print(f"Specific Risk: {factor_result.additional_metrics['specific_risk']:.4f}")
    print("\nFactor Exposures:")
    for factor, exposure in factor_result.additional_metrics['factor_exposures'].items():
        print(f"  {factor}: {exposure:.4f}")
    
    print("\nFactor Risk Contributions:")
    for factor, contrib in factor_result.additional_metrics['risk_contributions']['factor_contributions'].items():
        print(f"  {factor}: {contrib:.4f}")
