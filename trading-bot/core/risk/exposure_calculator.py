"""
Exposure Calculator - Portfolio-level exposure and risk calculations.
Handles correlation, concentration, and advanced risk metrics.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from .position_manager import PositionManager

logger = logging.getLogger(__name__)

@dataclass
class ExposureMetrics:
    """Container for exposure metrics."""
    # Position-based metrics
    total_exposure: float = 0.0
    net_exposure: float = 0.0
    gross_exposure: float = 0.0
    
    # Risk metrics
    var_95: Optional[float] = None
    cvar_95: Optional[float] = None
    max_drawdown: float = 0.0
    
    # Concentration metrics
    sector_exposure: Dict[str, float] = None
    asset_class_exposure: Dict[str, float] = None
    top_positions: List[Tuple[str, float]] = None
    
    # Liquidity metrics
    adv_utilization: Dict[str, float] = None  # % of ADV
    liquidity_impact: Dict[str, float] = None  # Estimated impact cost

class ExposureCalculator:
    ""
    Calculates portfolio exposure and risk metrics.
    """
    
    def __init__(self, position_manager: PositionManager):
        """
        Initialize ExposureCalculator.
        
        Args:
            position_manager: Instance of PositionManager
        """
        self.pm = position_manager
        self.historical_returns = pd.DataFrame()
        self.correlation_matrix = pd.DataFrame()
        
    def calculate_exposure_metrics(self) -> ExposureMetrics:
        """Calculate comprehensive exposure metrics."""
        metrics = ExposureMetrics(
            sector_exposure={},
            asset_class_exposure={},
            adv_utilization={},
            liquidity_impact={}
        )
        
        # Get basic position metrics
        exposure = self.pm.get_exposure()
        metrics.total_exposure = exposure['total_exposure']
        metrics.net_exposure = exposure['net_exposure']
        metrics.gross_exposure = sum(
            abs(pos.market_value) 
            for pos in self.pm.positions.values()
        )
        
        # Calculate concentration metrics
        self._calculate_concentration_metrics(metrics, exposure)
        
        # Calculate risk metrics if we have historical data
        if not self.historical_returns.empty:
            self._calculate_risk_metrics(metrics)
            
        return metrics
    
    def _calculate_concentration_metrics(
        self, 
        metrics: ExposureMetrics,
        exposure: Dict
    ) -> None:
        """Calculate position concentration metrics."""
        # Sort positions by absolute value
        sorted_positions = sorted(
            exposure['exposure_per_symbol'].items(),
            key=lambda x: abs(x[1]['value']),
            reverse=True
        )
        
        # Get top 5 positions
        metrics.top_positions = [
            (sym, data['value']) 
            for sym, data in sorted_positions[:5]
        ]
        
        # Calculate sector exposure (example implementation)
        # This would be populated from instrument metadata in a real system
        for symbol, data in exposure['exposure_per_symbol'].items():
            # Placeholder - in practice, get sector from instrument metadata
            sector = self._get_sector_for_symbol(symbol)
            metrics.sector_exposure[sector] = \
                metrics.sector_exposure.get(sector, 0) + abs(data['value'])
    
    def _calculate_risk_metrics(self, metrics: ExposureMetrics) -> None:
        """Calculate Value at Risk (VaR) and Conditional VaR (CVaR)."""
        try:
            # Calculate portfolio returns
            portfolio_returns = self.historical_returns.dot(
                np.array([pos.quantity for pos in self.pm.positions.values()])
            )
            
            # Calculate VaR and CVaR at 95% confidence
            metrics.var_95 = np.percentile(portfolio_returns, 5)
            metrics.cvar_95 = portfolio_returns[portfolio_returns <= metrics.var_95].mean()
            
            # Calculate max drawdown
            cum_returns = (1 + portfolio_returns).cumprod()
            running_max = cum_returns.cummax()
            drawdowns = (cum_returns - running_max) / running_max
            metrics.max_drawdown = drawdowns.min()
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
    
    def update_historical_returns(
        self, 
        returns: pd.DataFrame,
        lookback: int = 252
    ) -> None:
        """
        Update historical returns data for risk calculations.
        
        Args:
            returns: DataFrame with historical returns (columns = symbols, index = datetime)
            lookback: Number of periods to keep in history
        """
        self.historical_returns = returns.tail(lookback)
        self.correlation_matrix = self.historical_returns.corr()
    
    def calculate_liquidity_metrics(
        self,
        adv_data: Dict[str, float],
        impact_model: callable
    ) -> Dict[str, float]:
        """
        Calculate liquidity metrics for positions.
        
        Args:
            adv_data: Dict of {symbol: average_daily_volume}
            impact_model: Function that estimates impact cost
            
        Returns:
            Dict with liquidity metrics
        """
        metrics = {}
        for symbol, pos in self.pm.positions.items():
            if symbol not in adv_data:
                continue
                
            adv = adv_data[symbol]
            position_adv_ratio = abs(pos.quantity) / adv if adv > 0 else 0
            impact_cost = impact_model(pos.quantity, adv)
            
            metrics[symbol] = {
                'adv_utilization': position_adv_ratio,
                'impact_cost': impact_cost,
                'liquidity_score': 1.0 / (1.0 + position_adv_ratio * 10)
            }
            
        return metrics
    
    def _get_sector_for_symbol(self, symbol: str) -> str:
        """Get sector for a given symbol (placeholder implementation)."""
        # In a real system, this would query a market data service
        # or database to get the sector for the symbol
        return "Technology"  # Placeholder
