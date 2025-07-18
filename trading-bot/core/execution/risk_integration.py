"""
Risk Integration Module

This module integrates the risk management system with the trading execution system,
providing monitoring, alerts, and historical risk analysis.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import logging
import pandas as pd
import numpy as np
from decimal import Decimal

from .broker import Broker, BrokerType
from .order import Order, OrderSide, OrderStatus
from .position import Position
from ..market.risk import RiskManager, ExposureManager, ExposureType
from ..market.circuit_breaker import CircuitBreakerManager, CircuitBreaker, CircuitBreakerConfig, BreakerType
from ..market.position_sizing import PositionSizingParameters, calculate_kelly_position, calculate_volatility_adjusted_position

logger = logging.getLogger(__name__)

@dataclass
class RiskIntegrationConfig:
    """Configuration for risk integration."""
    # Position sizing
    default_risk_per_trade: float = 0.01  # 1% risk per trade
    max_position_size_pct: float = 0.1    # 10% of portfolio per position
    
    # Exposure limits
    max_asset_class_exposure: float = 0.3  # 30% per asset class
    max_sector_exposure: float = 0.2      # 20% per sector
    max_leverage: float = 3.0             # 3x leverage max
    
    # Circuit breakers
    max_daily_drawdown_pct: float = 0.05  # 5% daily drawdown
    max_position_drawdown_pct: float = 0.1  # 10% position drawdown
    
    # Monitoring
    risk_check_interval: float = 1.0  # seconds
    history_retention_days: int = 30   # days to keep risk history

class RiskIntegration:
    """
    Integrates risk management with the trading execution system.
    
    This class connects the risk management components (position sizing,
    exposure management, circuit breakers) with the trading execution system.
    """
    
    def __init__(
        self,
        broker: Broker,
        config: Optional[RiskIntegrationConfig] = None
    ):
        """
        Initialize the risk integration.
        
        Args:
            broker: The broker instance to integrate with
            config: Risk integration configuration
        """
        self.broker = broker
        self.config = config or RiskIntegrationConfig()
        
        # Initialize risk management components
        self.risk_manager = RiskManager()
        self.exposure_manager = ExposureManager(portfolio_value=0.0)
        self.circuit_breaker_manager = CircuitBreakerManager()
        
        # State tracking
        self._is_running = False
        self._task: Optional[asyncio.Task] = None
        self._last_risk_check: Optional[datetime] = None
        
        # Initialize circuit breakers
        self._setup_circuit_breakers()
        
        # Historical data
        self.risk_history: List[Dict[str, Any]] = []
        self.alert_history: List[Dict[str, Any]] = []
    
    async def start(self) -> None:
        """Start the risk integration service."""
        if self._is_running:
            logger.warning("Risk integration is already running")
            return
            
        self._is_running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Risk integration started")
    
    async def stop(self) -> None:
        """Stop the risk integration service."""
        if not self._is_running:
            return
            
        self._is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Risk integration stopped")
    
    def _setup_circuit_breakers(self) -> None:
        """Initialize circuit breakers with default configurations."""
        # Daily drawdown breaker
        daily_dd_config = CircuitBreakerConfig(
            breaker_type=BreakerType.DRAWDOWN,
            threshold=self.config.max_daily_drawdown_pct,
            lookback_period=timedelta(days=1),
            cooldown_period=timedelta(hours=1),
            params={"action": "reduce_risk"}
        )
        
        daily_dd_breaker = CircuitBreaker(
            name="daily_drawdown",
            config=daily_dd_config,
            check_condition=self._check_daily_drawdown
        )
        
        self.circuit_breaker_manager.add_breaker("daily_drawdown", daily_dd_breaker)
        
        # Position drawdown breaker
        position_dd_config = CircuitBreakerConfig(
            breaker_type=BreakerType.DRAWDOWN,
            threshold=self.config.max_position_drawdown_pct,
            lookback_period=timedelta(hours=1),
            cooldown_period=timedelta(minutes=30),
            params={"action": "close_position"}
        )
        
        position_dd_breaker = CircuitBreaker(
            name="position_drawdown",
            config=position_dd_config,
            check_condition=self._check_position_drawdown
        )
        
        self.circuit_breaker_manager.add_breaker("position_drawdown", position_dd_breaker)
    
    async def _run_loop(self) -> None:
        """Main risk monitoring loop."""
        while self._is_running:
            try:
                await self._check_risk()
                await asyncio.sleep(self.config.risk_check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in risk monitoring loop: {e}")
                await asyncio.sleep(5)  # Prevent tight loop on error
    
    async def _check_risk(self) -> None:
        """Perform a comprehensive risk check."""
        try:
            # Update portfolio value
            portfolio_value = await self._get_portfolio_value()
            self.exposure_manager.portfolio_value = portfolio_value
            
            # Update positions and exposures
            await self._update_positions_and_exposures()
            
            # Check circuit breakers
            await self.circuit_breaker_manager.check_all()
            
            # Log risk metrics
            await self._log_risk_metrics()
            
            self._last_risk_check = datetime.utcnow()
            
        except Exception as e:
            logger.error(f"Error during risk check: {e}", exc_info=True)
    
    async def _get_portfolio_value(self) -> float:
        """Get the current portfolio value from the broker."""
        try:
            account = await self.broker.get_account()
            return float(account.equity or 0.0)
        except Exception as e:
            logger.error(f"Error getting portfolio value: {e}")
            return 0.0
    
    async def _update_positions_and_exposures(self) -> None:
        """Update position and exposure information."""
        try:
            positions = await self.broker.get_positions()
            
            for position in positions:
                # Update exposure manager
                exposures = {
                    ExposureType.ASSET_CLASS: [position.asset_class],
                    ExposureType.SECTOR: [position.sector],
                    ExposureType.INSTRUMENT: [position.symbol]
                }
                
                self.exposure_manager.update_position(
                    position_id=position.id,
                    notional_value=abs(position.notional_value),
                    exposures=exposures
                )
                
                # Update risk manager with position metrics
                self.risk_manager.update_position(
                    position_id=position.id,
                    symbol=position.symbol,
                    quantity=position.quantity,
                    entry_price=position.entry_price,
                    current_price=position.current_price,
                    side=position.side
                )
                
        except Exception as e:
            logger.error(f"Error updating positions and exposures: {e}")
    
    async def _log_risk_metrics(self) -> None:
        """Log current risk metrics to history."""
        try:
            # Get current time
            now = datetime.utcnow()
            
            # Get portfolio value
            portfolio_value = self.exposure_manager.portfolio_value
            
            # Get exposure report
            exposure_report = self.exposure_manager.get_exposure_report()
            
            # Get risk metrics
            risk_metrics = self.risk_manager.get_risk_metrics()
            
            # Create risk snapshot
            snapshot = {
                "timestamp": now,
                "portfolio_value": portfolio_value,
                "exposures": exposure_report,
                "risk_metrics": risk_metrics,
                "circuit_breakers": {
                    name: {
                        "state": breaker.state.name,
                        "last_triggered": breaker.last_triggered,
                        "trigger_count": breaker.trigger_count
                    }
                    for name, breaker in self.circuit_breaker_manager.breakers.items()
                }
            }
            
            # Add to history
            self.risk_history.append(snapshot)
            
            # Clean up old history
            cutoff = now - timedelta(days=self.config.history_retention_days)
            self.risk_history = [r for r in self.risk_history if r["timestamp"] >= cutoff]
            
        except Exception as e:
            logger.error(f"Error logging risk metrics: {e}")
    
    def _check_daily_drawdown(self, breaker: CircuitBreaker) -> bool:
        """Check if daily drawdown exceeds threshold."""
        if not self.risk_history:
            return False
            
        # Get today's risk snapshots
        today = datetime.utcnow().date()
        today_snapshots = [
            s for s in self.risk_history 
            if s["timestamp"].date() == today
        ]
        
        if not today_snapshots:
            return False
            
        # Calculate max drawdown for today
        max_equity = max(s["portfolio_value"] for s in today_snapshots)
        current_equity = today_snapshots[-1]["portfolio_value"]
        drawdown = (max_equity - current_equity) / max_equity if max_equity > 0 else 0
        
        return drawdown >= self.config.max_daily_drawdown_pct
    
    def _check_position_drawdown(self, breaker: CircuitBreaker) -> bool:
        """Check if any position exceeds drawdown threshold."""
        if not self.risk_history:
            return False
            
        # Get latest risk metrics
        latest_metrics = self.risk_history[-1]["risk_metrics"]
        
        # Check if any position exceeds drawdown threshold
        for position_metrics in latest_metrics.get("positions", []):
            if position_metrics.get("drawdown_pct", 0) >= self.config.max_position_drawdown_pct:
                return True
                
        return False
    
    async def check_order_risk(self, order: Order) -> Tuple[bool, List[str]]:
        """
        Check if an order meets all risk requirements.
        
        Args:
            order: The order to check
            
        Returns:
            Tuple of (is_allowed, list_of_violations)
        """
        violations = []
        
        # 1. Check circuit breakers
        if any(b.state.name == "OPEN" for b in self.circuit_breaker_manager.breakers.values()):
            violations.append("Trading is currently suspended due to risk limits")
            
        # 2. Check position sizing
        position_size_ok, position_violations = self._check_position_size_risk(order)
        violations.extend(position_violations)
        
        # 3. Check exposure limits
        exposure_ok, exposure_violations = await self._check_exposure_risk(order)
        violations.extend(exposure_violations)
        
        return (len(violations) == 0, violations)
    
    def _check_position_size_risk(self, order: Order) -> Tuple[bool, List[str]]:
        """Check if order size is within risk limits."""
        violations = []
        
        # Get position sizing parameters
        params = PositionSizingParameters(
            account_balance=self.exposure_manager.portfolio_value,
            risk_per_trade=self.config.default_risk_per_trade,
            max_position_size=self.config.max_position_size_pct,
            max_leverage=self.config.max_leverage
        )
        
        # Calculate max position size based on risk
        # This is a simplified example - in practice, you'd use more sophisticated logic
        max_size = params.account_balance * params.max_position_size
        order_value = abs(order.quantity * order.price)
        
        if order_value > max_size:
            violations.append(
                f"Order size {order_value:.2f} exceeds maximum position size {max_size:.2f}"
            )
            
        return (len(violations) == 0, violations)
    
    async def _check_exposure_risk(self, order: Order) -> Tuple[bool, List[str]]:
        """Check if order would exceed exposure limits."""
        # This is a placeholder - in practice, you'd get the instrument's
        # asset class and sector from your data model
        exposures = {
            ExposureType.ASSET_CLASS: ["CRYPTO"],  # Example
            ExposureType.SECTOR: ["DEFI"],         # Example
            ExposureType.INSTRUMENT: [order.symbol]
        }
        
        # Calculate order notional value
        order_value = abs(order.quantity * order.price)
        
        # Check exposure limits
        return self.exposure_manager.check_exposure_limits(
            new_position=exposures,
            new_notional=order_value
        )
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get a summary of current risk metrics."""
        if not self.risk_history:
            return {}
            
        latest = self.risk_history[-1]
        
        return {
            "timestamp": latest["timestamp"].isoformat(),
            "portfolio_value": latest["portfolio_value"],
            "total_exposure": sum(
                sum(dim.values()) 
                for dim in latest["exposures"]["exposures"].values()
            ),
            "active_alerts": len(latest["exposures"].get("warnings", [])),
            "circuit_breakers": {
                name: {
                    "state": breaker.state.name,
                    "last_triggered": (
                        breaker.last_triggered.isoformat() 
                        if breaker.last_triggered else None
                    ),
                    "trigger_count": breaker.trigger_count
                }
                for name, breaker in self.circuit_breaker_manager.breakers.items()
            }
        }
