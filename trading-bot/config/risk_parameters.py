"""
Risk management parameters and configurations.
"""
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
from datetime import time
from dataclasses import dataclass, field
from config.settings import RISK_CONFIG


class RiskLevel(Enum):
    """Risk level classification."""
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    EXTREME = auto()


@dataclass
class AssetClassRiskParameters:
    """Risk parameters for a specific asset class."""
    max_leverage: int = 10
    position_size_limit: float = 0.1  # Max position size as % of portfolio
    daily_loss_limit: float = 0.02  # 2% daily loss limit
    max_drawdown: float = 0.1  # 10% max drawdown
    volatility_threshold: float = 0.05  # 5% volatility threshold for position reduction
    min_risk_reward_ratio: float = 1.5  # Minimum risk/reward ratio
    
    # Session-based adjustments (UTC times)
    session_risk_multipliers: Dict[str, float] = field(
        default_factory=lambda: {
            'asian': 0.8,      # 00:00-08:00 UTC
            'london': 1.0,     # 08:00-16:00 UTC
            'new_york': 1.0,   # 13:00-21:00 UTC
            'overlap': 1.2,    # 13:00-16:00 UTC (London-NY overlap)
            'overnight': 0.5   # 21:00-00:00 UTC
        }
    )
    
    def get_session_multiplier(self, current_time: time) -> float:
        """Get risk multiplier based on current market session."""
        hour = current_time.hour
        
        if 0 <= hour < 8:    # Asian session
            return self.session_risk_multipliers['asian']
        elif 8 <= hour < 13:  # London session
            return self.session_risk_multipliers['london']
        elif 13 <= hour < 16:  # London-NY overlap
            return self.session_risk_multipliers['overlap']
        elif 16 <= hour < 21:  # NY session
            return self.session_risk_multipliers['new_york']
        else:  # Overnight
            return self.session_risk_multipliers['overnight']


class RiskParameters:
    """Global risk management parameters."""
    def __init__(self):
        # Default risk parameters
        self.max_risk_per_trade: float = RISK_CONFIG['max_risk_per_trade']
        self.max_daily_drawdown: float = RISK_CONFIG['max_daily_drawdown']
        self.max_portfolio_risk: float = RISK_CONFIG['max_portfolio_risk']
        self.volatility_lookback: int = RISK_CONFIG['volatility_lookback']
        self.atr_multiplier: float = RISK_CONFIG['atr_multiplier']
        self.max_leverage: int = RISK_CONFIG['max_leverage']
        
        # Asset class specific parameters
        self.asset_class_parameters = {
            'forex': AssetClassRiskParameters(
                max_leverage=30,
                position_size_limit=0.2,
                daily_loss_limit=0.02,
                max_drawdown=0.1,
                volatility_threshold=0.05
            ),
            'crypto': AssetClassRiskParameters(
                max_leverage=10,
                position_size_limit=0.15,
                daily_loss_limit=0.03,
                max_drawdown=0.15,
                volatility_threshold=0.08
            ),
            'commodities': AssetClassRiskParameters(
                max_leverage=20,
                position_size_limit=0.15,
                daily_loss_limit=0.025,
                max_drawdown=0.12,
                volatility_threshold=0.06
            ),
            'indices': AssetClassRiskParameters(
                max_leverage=20,
                position_size_limit=0.15,
                daily_loss_limit=0.02,
                max_drawdown=0.1,
                volatility_threshold=0.05
            )
        }
    
    def get_asset_class_parameters(self, asset_class: str) -> AssetClassRiskParameters:
        """Get risk parameters for a specific asset class."""
        return self.asset_class_parameters.get(asset_class.lower(), AssetClassRiskParameters())
    
    def get_max_leverage(self, asset_class: str) -> int:
        """Get maximum leverage for an asset class."""
        return min(
            self.max_leverage,
            self.get_asset_class_parameters(asset_class).max_leverage
        )
    
    def get_position_size_limit(self, asset_class: str) -> float:
        """Get position size limit for an asset class."""
        return min(
            self.max_risk_per_trade,
            self.get_asset_class_parameters(asset_class).position_size_limit
        )


# Circuit breaker configuration
class CircuitBreakerLevel(Enum):
    """Circuit breaker levels for risk management."""
    NORMAL = auto()
    WARNING = auto()
    PARTIAL_CLOSE = auto()
    FULL_STOP = auto()


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breakers."""
    # Drawdown thresholds (as % of account balance)
    warning_threshold: float = 0.05  # 5% drawdown
    partial_close_threshold: float = 0.08  # 8% drawdown
    full_stop_threshold: float = 0.12  # 12% drawdown
    
    # Position reduction amounts (as % of current position)
    warning_reduction: float = 0.25  # 25% reduction
    partial_close_amount: float = 0.5  # 50% reduction
    
    # Cool-off periods (in minutes)
    warning_cooldown: int = 60  # 1 hour
    partial_close_cooldown: int = 240  # 4 hours
    full_stop_cooldown: int = 1440  # 24 hours
    
    # Volatility thresholds (as %)
    volatility_spike_threshold: float = 3.0  # 3 standard deviations
    
    def get_breach_level(self, drawdown: float) -> Optional[CircuitBreakerLevel]:
        """Determine if any circuit breaker levels have been breached."""
        if drawdown >= self.full_stop_threshold:
            return CircuitBreakerLevel.FULL_STOP
        elif drawdown >= self.partial_close_threshold:
            return CircuitBreakerLevel.PARTIAL_CLOSE
        elif drawdown >= self.warning_threshold:
            return CircuitBreakerLevel.WARNING
        return None


# Initialize global instances
risk_parameters = RiskParameters()
circuit_breaker_config = CircuitBreakerConfig()
