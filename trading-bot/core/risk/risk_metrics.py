"""
Advanced risk metrics calculation and monitoring.

This module provides comprehensive risk metrics including:
- Value at Risk (VaR)
- Conditional VaR (CVaR)
- Maximum Drawdown
- Risk-adjusted return metrics
- Correlation analysis
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class ConfidenceLevel(Enum):
    """Standard confidence levels for risk calculations."""
    P95 = 0.95
    P97_5 = 0.975
    P99 = 0.99
    P99_5 = 0.995
    P99_9 = 0.999


@dataclass
class RiskMetricsResult:
    """Container for risk metrics results."""
    var: float  # Value at Risk
    cvar: float  # Conditional Value at Risk
    max_drawdown: float
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    calmar_ratio: Optional[float] = None
    ulcer_index: Optional[float] = None


class RiskMetrics:
    """
    Advanced risk metrics calculator for portfolio analysis.
    
    Implements various risk measures including:
    - Historical and parametric VaR/CVaR
    - Maximum drawdown and drawdown duration
    - Risk-adjusted return metrics (Sharpe, Sortino, Calmar)
    - Correlation and covariance analysis
    """
    
    def __init__(
        self,
        returns: Optional[pd.Series] = None,
        confidence_level: float = 0.95,
        risk_free_rate: float = 0.0,
        annual_factor: int = 252,  # Trading days in a year
    ):
        """
        Initialize the risk metrics calculator.
        
        Args:
            returns: Series of periodic returns (daily by default)
            confidence_level: Confidence level for VaR/CVaR (0-1)
            risk_free_rate: Annual risk-free rate
            annual_factor: Number of periods in a year (default: 252 trading days)
        """
        self.returns = returns if returns is not None else pd.Series(dtype=float)
        self.confidence_level = confidence_level
        self.risk_free_rate = risk_free_rate
        self.annual_factor = annual_factor
        
    def calculate_all_metrics(self) -> RiskMetricsResult:
        """Calculate and return all risk metrics."""
        if self.returns.empty:
            raise ValueError("No returns data provided")
            
        return RiskMetricsResult(
            var=self.calculate_var(),
            cvar=self.calculate_cvar(),
            max_drawdown=self.calculate_max_drawdown(),
            sharpe_ratio=self.calculate_sharpe_ratio(),
            sortino_ratio=self.calculate_sortino_ratio(),
            calmar_ratio=self.calculate_calmar_ratio(),
            ulcer_index=self.calculate_ulcer_index()
        )
    
    def calculate_var(self, method: str = 'historical') -> float:
        """
        Calculate Value at Risk (VaR).
        
        Args:
            method: 'historical' or 'parametric'
            
        Returns:
            VaR as a positive number (loss amount)
        """
        if method == 'historical':
            return self._calculate_historical_var()
        elif method == 'parametric':
            return self._calculate_parametric_var()
        else:
            raise ValueError(f"Unsupported VaR method: {method}")
    
    def _calculate_historical_var(self) -> float:
        """Calculate historical VaR from return distribution."""
        return float(-np.percentile(
            self.returns.dropna(), 
            (1 - self.confidence_level) * 100
        ))
    
    def _calculate_parametric_var(self) -> float:
        """Calculate parametric (Gaussian) VaR."""
        from scipy.stats import norm
        mu = np.mean(self.returns)
        sigma = np.std(self.returns)
        return float(-(mu + sigma * norm.ppf(1 - self.confidence_level)))
    
    def calculate_cvar(self) -> float:
        """Calculate Conditional Value at Risk (CVaR)."""
        var = self.calculate_var()
        cvar = -self.returns[self.returns <= -var].mean()
        return float(cvar if not np.isnan(cvar) else var)
    
    def calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown from peak."""
        cum_returns = (1 + self.returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        return float(drawdowns.min())
    
    def calculate_sharpe_ratio(self, annualize: bool = True) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            annualize: Whether to annualize the ratio
            
        Returns:
            Sharpe ratio (higher is better)
        """
        excess_returns = self.returns - (self.risk_free_rate / self.annual_factor)
        sharpe = np.mean(excess_returns) / np.std(excess_returns, ddof=1)
        
        if annualize and len(self.returns) > 1:
            sharpe *= np.sqrt(self.annual_factor)
            
        return float(sharpe)
    
    def calculate_sortino_ratio(self, annualize: bool = True) -> float:
        """
        Calculate Sortino ratio (Sharpe ratio using downside deviation).
        
        Args:
            annualize: Whether to annualize the ratio
            
        Returns:
            Sortino ratio (higher is better)
        """
        excess_returns = self.returns - (self.risk_free_rate / self.annual_factor)
        downside_returns = excess_returns[excess_returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf')
            
        downside_std = np.std(downside_returns, ddof=1)
        sortino = np.mean(excess_returns) / downside_std if downside_std != 0 else float('inf')
        
        if annualize and len(self.returns) > 1:
            sortino *= np.sqrt(self.annual_factor)
            
        return float(sortino)
    
    def calculate_calmar_ratio(self, years_lookback: int = 3) -> float:
        """
        Calculate Calmar ratio (return over max drawdown).
        
        Args:
            years_lookback: Number of years to consider for max drawdown
            
        Returns:
            Calmar ratio (higher is better)
        """
        if len(self.returns) < 2:
            return 0.0
            
        # Use last N years of data if specified
        periods = years_lookback * self.annual_factor
        returns = self.returns[-int(periods):] if len(self.returns) > periods else self.returns
        
        # Calculate annualized return
        annual_return = (1 + returns).prod() ** (self.annual_factor / len(returns)) - 1
        
        # Calculate max drawdown
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min())
        
        return float(annual_return / max_drawdown) if max_drawdown != 0 else float('inf')
    
    def calculate_ulcer_index(self) -> float:
        """
        Calculate Ulcer Index - a measure of downside volatility.
        
        Returns:
            Ulcer Index (lower is better)
        """
        if self.returns.empty:
            return 0.0
            
        cum_returns = (1 + self.returns).cumprod()
        rolling_max = cum_returns.expanding().max()
        drawdowns = (cum_returns - rolling_max) / rolling_max
        squared_drawdowns = drawdowns ** 2
        
        return float(np.sqrt(squared_drawdowns.mean()))
    
    def calculate_correlation_matrix(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate correlation matrix for multiple return series.
        
        Args:
            returns_df: DataFrame with returns (columns = assets, index = dates)
            
        Returns:
            Correlation matrix
        """
        return returns_df.corr()
    
    def calculate_beta(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
    ) -> float:
        """
        Calculate portfolio beta relative to a benchmark.
        
        Args:
            portfolio_returns: Series of portfolio returns
            benchmark_returns: Series of benchmark returns
            
        Returns:
            Beta coefficient
        """
        # Align the series by index
        aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        if len(aligned) < 2:
            return 1.0
            
        cov_matrix = np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1])
        return float(cov_matrix[0, 1] / cov_matrix[1, 1])
    
    def update_returns(self, new_returns: pd.Series):
        """Update the returns series with new data."""
        if not isinstance(new_returns, pd.Series):
            new_returns = pd.Series(new_returns)
        self.returns = pd.concat([self.returns, new_returns]).drop_duplicates()
    
    def rolling_metrics(
        self,
        window: int = 21,  # 1 month of trading days
        step: int = 5,     # Weekly updates
    ) -> pd.DataFrame:
        """
        Calculate rolling risk metrics.
        
        Args:
            window: Number of periods in each window
            step: Number of periods to move forward between windows
            
        Returns:
            DataFrame with rolling metrics
        """
        metrics = []
        
        for i in range(0, len(self.returns) - window + 1, step):
            window_returns = self.returns.iloc[i:i + window]
            if len(window_returns) < window:
                continue
                
            rm = RiskMetrics(
                returns=window_returns,
                confidence_level=self.confidence_level,
                risk_free_rate=self.risk_free_rate,
                annual_factor=self.annual_factor
            )
            
            try:
                metrics.append({
                    'date': window_returns.index[-1],
                    'var': rm.calculate_var(),
                    'cvar': rm.calculate_cvar(),
                    'max_drawdown': rm.calculate_max_drawdown(),
                    'sharpe': rm.calculate_sharpe_ratio(),
                    'sortino': rm.calculate_sortino_ratio(),
                    'ulcer': rm.calculate_ulcer_index()
                })
            except Exception as e:
                print(f"Error calculating metrics for window {i}: {e}")
                
        return pd.DataFrame(metrics).set_index('date')


"""
Advanced risk metrics and analytics.

This module implements various risk measurement techniques:
- Value at Risk (VaR)
- Conditional Value at Risk (CVaR)
- Maximum Drawdown (MDD)
- Risk-adjusted return metrics (Sharpe, Sortino, Calmar)
- Risk attribution analysis
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union
from scipy import stats


class ConfidenceLevel(Enum):
    """Standard confidence levels for risk calculations."""
    P90 = 0.90
    P95 = 0.95
    P97_5 = 0.975
    P99 = 0.99
    P99_5 = 0.995
    P99_9 = 0.999


class ReturnType(Enum):
    """Types of returns for risk calculations."""
    SIMPLE = auto()
    LOG = auto()
    ARITHMETIC = auto()
    EXCESS = auto()


@dataclass
class RiskMetricsConfig:
    """Configuration for risk metrics calculations."""
    confidence_level: float = 0.95
    lookback_period: int = 252  # Trading days in a year
    risk_free_rate: float = 0.0
    return_type: ReturnType = ReturnType.SIMPLE
    annualization_factor: int = 252
    min_observations: int = 30
    outlier_threshold: float = 3.0  # Z-score threshold for outliers
    

class RiskMetrics:
    """
    Advanced risk metrics calculator for trading strategies and portfolios.
    
    Features:
    - Value at Risk (VaR) - Historical, Parametric, and Monte Carlo methods
    - Conditional VaR (Expected Shortfall)
    - Maximum Drawdown analysis
    - Risk-adjusted return metrics (Sharpe, Sortino, Calmar, etc.)
    - Risk attribution and decomposition
    - Stress testing and scenario analysis
    - Correlation and beta calculations
    - Risk contribution analysis
    """
    
    def __init__(self, config: Optional[RiskMetricsConfig] = None):
        """
        Initialize the risk metrics calculator.
        
        Args:
            config: Risk metrics configuration. If None, uses defaults.
        """
        self.config = config or RiskMetricsConfig()
        self._validate_config()
        
        # Cache for intermediate calculations
        self._cache: Dict[str, Any] = {}
    
    def _validate_config(self) -> None:
        """Validate the configuration parameters."""
        if not 0 < self.config.confidence_level < 1:
            raise ValueError("Confidence level must be between 0 and 1")
        if self.config.lookback_period < 1:
            raise ValueError("Lookback period must be positive")
        if self.config.min_observations < 1:
            raise ValueError("Minimum observations must be positive")
    
    def calculate_returns(
        self,
        prices: Union[pd.Series, np.ndarray],
        return_type: Optional[ReturnType] = None
    ) -> np.ndarray:
        """
        Calculate returns from price data.
        
        Args:
            prices: Series or array of price data
            return_type: Type of returns to calculate. If None, uses config default.
            
        Returns:
            Array of returns
        """
        return_type = return_type or self.config.return_type
        prices = np.asarray(prices)
        
        if return_type == ReturnType.SIMPLE:
            returns = np.diff(prices) / prices[:-1]
        elif return_type == ReturnType.LOG:
            returns = np.diff(np.log(prices))
        elif return_type == ReturnType.ARITHMETIC:
            returns = (prices[1:] - prices[:-1]) / prices[0]
        elif return_type == ReturnType.EXCESS:
            returns = np.diff(prices) / prices[:-1] - self.config.risk_free_rate / self.config.annualization_factor
        else:
            raise ValueError(f"Unsupported return type: {return_type}")
            
        return returns
    
    def value_at_risk(
        self,
        returns: Union[pd.Series, np.ndarray],
        confidence_level: Optional[float] = None,
        method: str = 'historical',
        window: Optional[int] = None,
        parametric_dist: str = 'normal'
    ) -> float:
        """
        Calculate Value at Risk (VaR) using the specified method.
        
        Args:
            returns: Array of returns
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            method: Calculation method ('historical', 'parametric', 'monte_carlo')
            window: Lookback window for historical VaR
            parametric_dist: Distribution for parametric VaR ('normal', 't', 'laplace')
            
        Returns:
            Value at Risk as a positive number
        """
        confidence_level = confidence_level or self.config.confidence_level
        window = window or len(returns)
        returns = np.asarray(returns[-window:])  # Use most recent 'window' returns
        
        if method == 'historical':
            # Historical VaR (non-parametric)
            return -np.percentile(returns, 100 * (1 - confidence_level))
            
        elif method == 'parametric':
            # Parametric VaR (assumes distribution)
            if parametric_dist == 'normal':
                mu, sigma = np.mean(returns), np.std(returns, ddof=1)
                z_score = stats.norm.ppf(1 - confidence_level)
                return -(mu + z_score * sigma)
                
            elif parametric_dist == 't':
                # Student's t-distribution
                df, mu, sigma = stats.t.fit(returns)
                t_score = stats.t.ppf(1 - confidence_level, df)
                return -(mu + t_score * sigma)
                
            elif parametric_dist == 'laplace':
                # Laplace distribution
                loc, scale = stats.laplace.fit(returns)
                laplace_quantile = stats.laplace.ppf(1 - confidence_level, loc=loc, scale=scale)
                return -laplace_quantile
                
            else:
                raise ValueError(f"Unsupported distribution: {parametric_dist}")
                
        elif method == 'monte_carlo':
            # Monte Carlo VaR (simulation-based)
            n_simulations = 10000
            mu, sigma = np.mean(returns), np.std(returns, ddof=1)
            simulated_returns = np.random.normal(mu, sigma, n_simulations)
            return -np.percentile(simulated_returns, 100 * (1 - confidence_level))
            
        else:
            raise ValueError(f"Unsupported VaR method: {method}")
    
    def conditional_var(
        self,
        returns: Union[pd.Series, np.ndarray],
        confidence_level: Optional[float] = None,
        method: str = 'historical'
    ) -> float:
        """
        Calculate Conditional Value at Risk (CVaR) or Expected Shortfall.
        
        Args:
            returns: Array of returns
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            method: Calculation method ('historical', 'parametric')
            
        Returns:
            Conditional VaR as a positive number
        """
        confidence_level = confidence_level or self.config.confidence_level
        returns = np.asarray(returns)
        
        if method == 'historical':
            # Historical CVaR (average of losses beyond VaR)
            var = self.value_at_risk(returns, confidence_level, method='historical')
            losses = returns[returns < -var]
            return -np.mean(losses) if len(losses) > 0 else 0.0
            
        elif method == 'parametric':
            # Parametric CVaR for normal distribution
            mu, sigma = np.mean(returns), np.std(returns, ddof=1)
            alpha = 1 - confidence_level
            
            # For normal distribution
            z_alpha = stats.norm.ppf(alpha)
            cvar = -(mu - sigma * (stats.norm.pdf(z_alpha) / alpha))
            return cvar
            
        else:
            raise ValueError(f"Unsupported CVaR method: {method}")
    
    def max_drawdown(self, prices: Union[pd.Series, np.ndarray]) -> float:
        """
        Calculate maximum drawdown from price series.
        
        Args:
            prices: Array of prices
            
        Returns:
            Maximum drawdown as a positive number (e.g., 0.15 for 15%)
        """
        prices = np.asarray(prices)
        peak = prices[0]
        max_dd = 0.0
        
        for price in prices[1:]:
            if price > peak:
                peak = price
            else:
                dd = (peak - price) / peak
                if dd > max_dd:
                    max_dd = dd
                    
        return max_dd
    
    def sharpe_ratio(
        self,
        returns: Union[pd.Series, np.ndarray],
        risk_free_rate: Optional[float] = None,
        annualize: bool = True
    ) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            returns: Array of returns
            risk_free_rate: Annual risk-free rate. If None, uses config value.
            annualize: Whether to annualize the ratio
            
        Returns:
            Sharpe ratio
        """
        risk_free_rate = risk_free_rate or self.config.risk_free_rate
        returns = np.asarray(returns)
        
        if len(returns) < 2:
            return 0.0
            
        excess_returns = returns - (risk_free_rate / self.config.annualization_factor)
        sharpe = np.mean(excess_returns) / np.std(excess_returns, ddof=1)
        
        if annualize:
            sharpe *= np.sqrt(self.config.annualization_factor)
            
        return sharpe
    
    def sortino_ratio(
        self,
        returns: Union[pd.Series, np.ndarray],
        risk_free_rate: Optional[float] = None,
        annualize: bool = True,
        target_return: float = 0.0
    ) -> float:
        """
        Calculate Sortino ratio (risk-adjusted return using downside deviation).
        
        Args:
            returns: Array of returns
            risk_free_rate: Annual risk-free rate. If None, uses config value.
            annualize: Whether to annualize the ratio
            target_return: Minimum acceptable return (MAR)
            
        Returns:
            Sortino ratio
        """
        risk_free_rate = risk_free_rate or self.config.risk_free_rate
        returns = np.asarray(returns)
        
        if len(returns) < 2:
            return 0.0
            
        excess_returns = returns - (risk_free_rate / self.config.annualization_factor)
        downside_returns = np.minimum(0, returns - target_return)
        downside_std = np.std(downside_returns, ddof=1)
        
        if downside_std == 0:
            return 0.0
            
        sortino = np.mean(excess_returns) / downside_std
        
        if annualize:
            sortino *= np.sqrt(self.config.annualization_factor)
            
        return sortino
    
    def calmar_ratio(
        self,
        prices: Union[pd.Series, np.ndarray],
        risk_free_rate: Optional[float] = None,
        annualize: bool = True
    ) -> float:
        """
        Calculate Calmar ratio (return over maximum drawdown).
        
        Args:
            prices: Array of prices
            risk_free_rate: Annual risk-free rate. If None, uses config value.
            annualize: Whether to use annualized returns
            
        Returns:
            Calmar ratio
        """
        risk_free_rate = risk_free_rate or self.config.risk_free_rate
        returns = self.calculate_returns(prices)
        
        if len(returns) < 1:
            return 0.0
            
        # Calculate annualized return
        total_return = (prices[-1] - prices[0]) / prices[0]
        if annualize and len(returns) > 1:
            total_return = (1 + total_return) ** (self.config.annualization_factor / len(returns)) - 1
        
        # Calculate max drawdown
        mdd = self.max_drawdown(prices)
        
        if mdd == 0:
            return 0.0
            
        # Calculate excess return over risk-free rate
        excess_return = total_return - (risk_free_rate if annualize else risk_free_rate / self.config.annualization_factor)
        
        return excess_return / mdd
    
    def beta(
        self,
        asset_returns: Union[pd.Series, np.ndarray],
        market_returns: Union[pd.Series, np.ndarray]
    ) -> float:
        """
        Calculate beta (sensitivity to market movements).
        
        Args:
            asset_returns: Returns of the asset
            market_returns: Returns of the market/index
            
        Returns:
            Beta coefficient
        """
        asset_returns = np.asarray(asset_returns)
        market_returns = np.asarray(market_returns)
        
        if len(asset_returns) != len(market_returns):
            raise ValueError("Asset and market returns must have the same length")
            
        if len(asset_returns) < 2:
            return 0.0
            
        # Calculate covariance and variance
        cov_matrix = np.cov(asset_returns, market_returns, ddof=1)
        if cov_matrix[1, 1] == 0:
            return 0.0
            
        return cov_matrix[0, 1] / cov_matrix[1, 1]
    
    def risk_contribution(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Calculate risk contribution of each asset in a portfolio.
        
        Args:
            weights: Portfolio weights (should sum to 1)
            cov_matrix: Covariance matrix of returns
            
        Returns:
            Array of risk contributions
        """
        weights = np.asarray(weights)
        if not np.isclose(np.sum(weights), 1.0, rtol=1e-5):
            raise ValueError("Weights must sum to 1")
            
        portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        if portfolio_vol == 0:
            return np.zeros_like(weights)
            
        # Marginal risk contribution
        mrc = (cov_matrix @ weights) / portfolio_vol
        
        # Risk contribution
        return weights * mrc
    
    def stress_test(
        self,
        returns: Union[pd.Series, np.ndarray],
        scenarios: Dict[str, Dict[str, float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Perform stress testing under different market scenarios.
        
        Args:
            returns: Historical returns
            scenarios: Dictionary of scenario names to parameter changes
                     (e.g., {'crash': {'vol_multiplier': 2.0, 'return_shift': -0.1}})
                     
        Returns:
            Dictionary of scenario results
        """
        returns = np.asarray(returns)
        results = {}
        
        for scenario_name, params in scenarios.items():
            scenario_returns = returns.copy()
            
            # Apply scenario transformations
            if 'vol_multiplier' in params:
                # Scale volatility
                vol_mult = params['vol_multiplier']
                mean_ret = np.mean(scenario_returns)
                std_ret = np.std(scenario_returns, ddof=1)
                scenario_returns = (scenario_returns - mean_ret) * vol_mult + mean_ret
                
            if 'return_shift' in params:
                # Shift returns
                scenario_returns += params['return_shift'] / self.config.annualization_factor
                
            # Calculate metrics for the scenario
            scenario_var = self.value_at_risk(scenario_returns)
            scenario_cvar = self.conditional_var(scenario_returns)
            scenario_vol = np.std(scenario_returns, ddof=1) * np.sqrt(self.config.annualization_factor)
            
            results[scenario_name] = {
                'var': scenario_var,
                'cvar': scenario_cvar,
                'volatility': scenario_vol,
                'mean_return': np.mean(scenario_returns) * self.config.annualization_factor,
                'sharpe': self.sharpe_ratio(scenario_returns),
                'sortino': self.sortino_ratio(scenario_returns)
            }
            
        return results
