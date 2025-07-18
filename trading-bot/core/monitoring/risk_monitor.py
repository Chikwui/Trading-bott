"""
Risk Monitoring and Alerting System

This module provides real-time monitoring and alerting for risk metrics,
integrating with the risk management components to provide comprehensive
visibility into the trading system's risk profile.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Any, Set, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import logging
import json
from enum import Enum, auto

from ..execution.risk_integration import RiskIntegration
from ..market.circuit_breaker import CircuitBreakerState

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Severity levels for alerts."""
    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()

@dataclass
class Alert:
    """Represents a risk alert."""
    id: str
    title: str
    message: str
    severity: AlertSeverity
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None

class AlertManager:
    """Manages risk alerts and notifications."""
    
    def __init__(self):
        """Initialize the alert manager."""
        self.alerts: Dict[str, Alert] = {}
        self.subscribers: Set[Callable[[Alert], None]] = set()
        self._alert_id_counter = 0
    
    def subscribe(self, callback: Callable[[Alert], None]) -> Callable[[], None]:
        """
        Subscribe to alert notifications.
        
        Args:
            callback: Function to call when a new alert is generated
            
        Returns:
            Unsubscribe function
        """
        self.subscribers.add(callback)
        
        def unsubscribe():
            self.subscribers.discard(callback)
            
        return unsubscribe
    
    def create_alert(
        self,
        title: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.WARNING,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Alert:
        """
        Create and dispatch a new alert.
        
        Args:
            title: Short title for the alert
            message: Detailed message
            severity: Alert severity level
            metadata: Additional metadata
            
        Returns:
            The created alert
        """
        alert_id = f"alert_{self._alert_id_counter}"
        self._alert_id_counter += 1
        
        alert = Alert(
            id=alert_id,
            title=title,
            message=message,
            severity=severity,
            timestamp=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        self.alerts[alert_id] = alert
        self._notify_subscribers(alert)
        
        return alert
    
    def acknowledge_alert(self, alert_id: str, user: str) -> Optional[Alert]:
        """
        Acknowledge an alert.
        
        Args:
            alert_id: ID of the alert to acknowledge
            user: Username of the person acknowledging
            
        Returns:
            The updated alert, or None if not found
        """
        if alert_id not in self.alerts:
            return None
            
        alert = self.alerts[alert_id]
        alert.acknowledged = True
        alert.acknowledged_by = user
        alert.acknowledged_at = datetime.utcnow()
        
        self._notify_subscribers(alert)
        return alert
    
    def _notify_subscribers(self, alert: Alert) -> None:
        """Notify all subscribers of a new or updated alert."""
        for callback in self.subscribers:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert subscriber: {e}", exc_info=True)
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active (unacknowledged) alerts."""
        return [a for a in self.alerts.values() if not a.acknowledged]
    
    def get_alert_history(
        self,
        limit: int = 100,
        severity: Optional[AlertSeverity] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Alert]:
        """
        Get alert history with optional filtering.
        
        Args:
            limit: Maximum number of alerts to return
            severity: Filter by severity level
            start_time: Earliest alert timestamp
            end_time: Latest alert timestamp
            
        Returns:
            List of matching alerts, most recent first
        """
        alerts = list(self.alerts.values())
        
        if severity is not None:
            alerts = [a for a in alerts if a.severity == severity]
            
        if start_time is not None:
            alerts = [a for a in alerts if a.timestamp >= start_time]
            
        if end_time is not None:
            alerts = [a for a in alerts if a.timestamp <= end_time]
        
        # Sort by timestamp, most recent first
        alerts.sort(key=lambda x: x.timestamp, reverse=True)
        
        return alerts[:limit]

class RiskMonitor:
    """
    Monitors risk metrics and generates alerts when thresholds are exceeded.
    
    This class integrates with the RiskIntegration component to provide
    real-time monitoring and alerting for risk metrics.
    """
    
    def __init__(
        self,
        risk_integration: RiskIntegration,
        alert_manager: Optional[AlertManager] = None
    ):
        """
        Initialize the risk monitor.
        
        Args:
            risk_integration: The risk integration instance to monitor
            alert_manager: Optional alert manager for handling alerts
        """
        self.risk_integration = risk_integration
        self.alert_manager = alert_manager or AlertManager()
        self._is_running = False
        self._task: Optional[asyncio.Task] = None
        self._last_check: Optional[datetime] = None
        
        # Track previously triggered alerts to avoid duplicates
        self._active_alert_ids: Set[str] = set()
    
    async def start(self, interval: float = 5.0) -> None:
        """
        Start the risk monitoring service.
        
        Args:
            interval: Seconds between risk checks
        """
        if self._is_running:
            logger.warning("Risk monitor is already running")
            return
            
        self._is_running = True
        self._task = asyncio.create_task(self._monitor_loop(interval))
        logger.info("Risk monitor started")
    
    async def stop(self) -> None:
        """Stop the risk monitoring service."""
        if not self._is_running:
            return
            
        self._is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("Risk monitor stopped")
    
    async def _monitor_loop(self, interval: float) -> None:
        """Main monitoring loop."""
        while self._is_running:
            try:
                await self.check_risk()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in risk monitoring loop: {e}", exc_info=True)
                await asyncio.sleep(min(30, interval))  # Back off on error
    
    async def check_risk(self) -> Dict[str, Any]:
        """
        Perform a comprehensive risk check and generate alerts.
        
        Returns:
            Dictionary of risk metrics and alerts
        """
        try:
            # Get current risk summary
            risk_summary = self.risk_integration.get_risk_summary()
            
            # Check for circuit breaker states
            await self._check_circuit_breakers(risk_summary)
            
            # Check exposure limits
            await self._check_exposures(risk_summary)
            
            # Check position risks
            await self._check_position_risks(risk_summary)
            
            # Update last check time
            self._last_check = datetime.utcnow()
            
            return risk_summary
            
        except Exception as e:
            logger.error(f"Error during risk check: {e}", exc_info=True)
            return {}
    
    async def _check_circuit_breakers(self, risk_summary: Dict[str, Any]) -> None:
        """Check circuit breaker states and generate alerts."""
        if "circuit_breakers" not in risk_summary:
            return
            
        for name, cb_info in risk_summary["circuit_breakers"].items():
            alert_id = f"cb_{name}"
            
            if cb_info["state"] == "OPEN":
                if alert_id not in self._active_alert_ids:
                    self.alert_manager.create_alert(
                        title=f"Circuit Breaker Triggered: {name}",
                        message=(
                            f"Circuit breaker '{name}' has been triggered. "
                            f"State: {cb_info['state']}, "
                            f"Trigger count: {cb_info.get('trigger_count', 0)}"
                        ),
                        severity=AlertSeverity.CRITICAL,
                        metadata={
                            "type": "circuit_breaker",
                            "name": name,
                            "state": cb_info["state"],
                            "trigger_count": cb_info.get("trigger_count", 0),
                            "last_triggered": cb_info.get("last_triggered")
                        }
                    )
                    self._active_alert_ids.add(alert_id)
            else:
                if alert_id in self._active_alert_ids:
                    self.alert_manager.create_alert(
                        title=f"Circuit Breaker Reset: {name}",
                        message=f"Circuit breaker '{name}' has been reset to {cb_info['state']} state.",
                        severity=AlertSeverity.INFO,
                        metadata={
                            "type": "circuit_breaker_reset",
                            "name": name,
                            "state": cb_info["state"]
                        }
                    )
                    self._active_alert_ids.discard(alert_id)
    
    async def _check_exposures(self, risk_summary: Dict[str, Any]) -> None:
        """Check exposure limits and generate alerts."""
        if "exposures" not in risk_summary:
            return
            
        exposures = risk_summary.get("exposures", {})
        
        # Check for exposure warnings
        for warning in exposures.get("warnings", []):
            alert_id = f"exp_{hash(warning)}"
            
            if alert_id not in self._active_alert_ids:
                self.alert_manager.create_alert(
                    title="Exposure Warning",
                    message=warning,
                    severity=AlertSeverity.WARNING,
                    metadata={
                        "type": "exposure_warning",
                        "message": warning
                    }
                )
                self._active_alert_ids.add(alert_id)
    
    async def _check_position_risks(self, risk_summary: Dict[str, Any]) -> None:
        """Check position-level risks and generate alerts."""
        if "risk_metrics" not in risk_summary:
            return
            
        metrics = risk_summary["risk_metrics"]
        
        # Check for high drawdown positions
        for position in metrics.get("positions", []):
            if position.get("drawdown_pct", 0) > 0.05:  # 5% drawdown
                alert_id = f"dd_{position.get('symbol')}_{position.get('position_id')}"
                
                if alert_id not in self._active_alert_ids:
                    self.alert_manager.create_alert(
                        title=f"High Drawdown: {position.get('symbol')}",
                        message=(
                            f"Position {position.get('position_id')} ({position.get('symbol')}) "
                            f"has {position.get('drawdown_pct', 0):.2%} drawdown. "
                            f"Current PnL: {position.get('unrealized_pnl', 0):.2f} "
                            f"({position.get('unrealized_pnl_pct', 0):.2%})"
                        ),
                        severity=AlertSeverity.WARNING,
                        metadata={
                            "type": "position_drawdown",
                            "position_id": position.get("position_id"),
                            "symbol": position.get("symbol"),
                            "drawdown_pct": position.get("drawdown_pct"),
                            "unrealized_pnl": position.get("unrealized_pnl"),
                            "unrealized_pnl_pct": position.get("unrealized_pnl_pct")
                        }
                    )
                    self._active_alert_ids.add(alert_id)

class RiskDashboard:
    """
    Provides a dashboard for monitoring risk metrics and alerts.
    
    This class can be used to build a web-based or CLI dashboard
    for monitoring the trading system's risk profile.
    """
    
    def __init__(
        self,
        risk_monitor: RiskMonitor,
        update_interval: float = 1.0
    ):
        """
        Initialize the risk dashboard.
        
        Args:
            risk_monitor: The risk monitor instance to visualize
            update_interval: Seconds between updates
        """
        self.risk_monitor = risk_monitor
        self.update_interval = update_interval
        self._is_running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the dashboard update loop."""
        if self._is_running:
            return
            
        self._is_running = True
        self._task = asyncio.create_task(self._update_loop())
    
    async def stop(self) -> None:
        """Stop the dashboard update loop."""
        if not self._is_running:
            return
            
        self._is_running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
    
    async def _update_loop(self) -> None:
        """Dashboard update loop."""
        while self._is_running:
            try:
                await self.render()
                await asyncio.sleep(self.update_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error updating dashboard: {e}")
                await asyncio.sleep(1)  # Prevent tight loop on error
    
    async def render(self) -> None:
        """Render the dashboard."""
        # This is a basic console-based implementation.
        # In a real application, this could be a web-based dashboard.
        
        # Clear screen
        print("\033[H\033[J", end="")
        
        # Get current time
        now = datetime.utcnow()
        print(f"=== Risk Dashboard === {now.isoformat()} ===\n")
        
        # Get risk summary
        risk_summary = self.risk_monitor.risk_integration.get_risk_summary()
        
        # Display portfolio summary
        print("=== Portfolio Summary ===")
        print(f"Portfolio Value: ${risk_summary.get('portfolio_value', 0):,.2f}")
        print(f"Total Exposure: ${risk_summary.get('total_exposure', 0):,.2f}")
        print(f"Active Alerts: {len(self.risk_monitor.alert_manager.get_active_alerts())}")
        print()
        
        # Display circuit breakers
        print("=== Circuit Breakers ===")
        for name, cb in risk_summary.get('circuit_breakers', {}).items():
            state = cb.get('state', 'UNKNOWN')
            last_triggered = cb.get('last_triggered', 'Never')
            trigger_count = cb.get('trigger_count', 0)
            
            state_color = "\033[92m"  # Green
            if state == "OPEN":
                state_color = "\033[91m"  # Red
            elif state == "HALF_OPEN":
                state_color = "\033[93m"  # Yellow
                
            print(f"{name}: {state_color}{state}\033[0m, "
                  f"Last Triggered: {last_triggered}, "
                  f"Count: {trigger_count}")
        
        # Display active alerts
        active_alerts = self.risk_monitor.alert_manager.get_active_alerts()
        if active_alerts:
            print("\n=== Active Alerts ===")
            for i, alert in enumerate(active_alerts[:5], 1):  # Show max 5 alerts
                print(f"{i}. [{alert.severity.name}] {alert.title}")
                print(f"   {alert.message}")
                if alert.metadata:
                    print(f"   Metadata: {json.dumps(alert.metadata, indent=4)}")
        
        print("\nPress Ctrl+C to exit...")

# Example usage
async def example():
    """Example usage of the risk monitoring system."""
    from ..execution.broker import Broker, BrokerType
    from ..execution.risk_integration import RiskIntegration, RiskIntegrationConfig
    
    # Initialize components
    broker = Broker(BrokerType.SIMULATION)
    risk_integration = RiskIntegration(broker, RiskIntegrationConfig())
    risk_monitor = RiskMonitor(risk_integration)
    dashboard = RiskDashboard(risk_monitor)
    
    # Start services
    await risk_integration.start()
    await risk_monitor.start()
    await dashboard.start()
    
    try:
        # Keep the example running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # Clean up
        await dashboard.stop()
        await risk_monitor.stop()
        await risk_integration.stop()

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example())
