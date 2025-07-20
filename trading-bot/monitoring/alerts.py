"""
Alerting system for the trading platform.

This module provides a flexible and extensible alerting system that integrates
with various notification channels (email, Slack, PagerDuty, etc.) and supports
complex alert conditions and deduplication.
"""
from __future__ import annotations

import asyncio
import importlib
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from types import ModuleType
from typing import (
    Any, Dict, List, Optional, Set, Tuple, Callable, Awaitable,
    Union, Type, TypeVar, Generic, cast, TYPE_CHECKING
)

import aiohttp
from pydantic import BaseModel, Field, validator

from .config import config

# Lazy import metrics to avoid circular imports
if TYPE_CHECKING:
    from . import metrics

logger = logging.getLogger(__name__)

# Lazy metrics module
_metrics_module: Optional[ModuleType] = None

def get_metrics() -> 'metrics.MetricsCollector':
    """Lazily import and return the metrics module.
    
    Returns:
        The metrics module
    """
    global _metrics_module
    if _metrics_module is None:
        from . import metrics as metrics_mod
        _metrics_module = metrics_mod
    return _metrics_module.metrics

# Type aliases
T = TypeVar('T')
AlertCondition = Callable[['AlertContext'], Awaitable[bool]]

class AlertSeverity(str, Enum):
    """Severity levels for alerts."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    ERROR = "error"

class AlertStatus(str, Enum):
    """Status of an alert."""
    FIRING = "firing"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"
    ACKNOWLEDGED = "acknowledged"

class AlertContext(BaseModel):
    """Context for alert evaluation and processing."""
    name: str
    severity: AlertSeverity
    status: AlertStatus = AlertStatus.FIRING
    summary: str
    description: str = ""
    labels: Dict[str, str] = Field(default_factory=dict)
    annotations: Dict[str, str] = Field(default_factory=dict)
    starts_at: datetime = Field(default_factory=datetime.utcnow)
    ends_at: Optional[datetime] = None
    generator_url: Optional[str] = None
    fingerprint: Optional[str] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat() + 'Z' if v else None
        }
    
    @validator('fingerprint', always=True, pre=True)
    def generate_fingerprint(cls, v, values):
        """Generate a fingerprint if not provided."""
        if v is not None:
            return v
            
        # Create a stable fingerprint based on name and labels
        import hashlib
        import json
        
        fingerprint_data = {
            'name': values.get('name', ''),
            'labels': values.get('labels', {})
        }
        
        return hashlib.sha256(
            json.dumps(fingerprint_data, sort_keys=True).encode('utf-8')
        ).hexdigest()

class AlertReceiver(ABC):
    """Base class for alert receivers."""
    
    @abstractmethod
    async def send(self, alert: AlertContext) -> bool:
        """Send an alert.
        
        Args:
            alert: The alert to send
            
        Returns:
            bool: True if the alert was sent successfully
        """
        pass

class ConsoleReceiver(AlertReceiver):
    """Simple console-based alert receiver for debugging."""
    
    async def send(self, alert: AlertContext) -> bool:
        """Print alert to console."""
        print(f"[ALERT] {alert.severity.upper()}: {alert.summary}")
        if alert.description:
            print(f"  {alert.description}")
        if alert.labels:
            print("  Labels:", json.dumps(alert.labels, indent=2))
        return True

class WebhookReceiver(AlertReceiver):
    """Send alerts to a webhook endpoint."""
    
    def __init__(
        self,
        url: str,
        timeout: float = 5.0,
        headers: Optional[Dict[str, str]] = None,
        template: Optional[Callable[[AlertContext], Dict[str, Any]]] = None
    ):
        """Initialize the webhook receiver.
        
        Args:
            url: Webhook URL
            timeout: Request timeout in seconds
            headers: Additional headers to include
            template: Function to format the alert payload
        """
        self.url = url
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.headers = headers or {}
        self.template = template or self._default_template
    
    def _default_template(self, alert: AlertContext) -> Dict[str, Any]:
        """Default template for webhook payload."""
        return {
            'status': alert.status.value,
            'alerts': [{
                'status': alert.status.value,
                'labels': alert.labels,
                'annotations': alert.annotations,
                'startsAt': alert.starts_at.isoformat() + 'Z',
                'endsAt': alert.ends_at.isoformat() + 'Z' if alert.ends_at else None,
                'generatorURL': alert.generator_url or '',
                'fingerprint': alert.fingerprint
            }]
        }
    
    async def send(self, alert: AlertContext) -> bool:
        """Send alert to webhook."""
        payload = self.template(alert)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.url,
                    json=payload,
                    headers={
                        'Content-Type': 'application/json',
                        **self.headers
                    },
                    timeout=self.timeout
                ) as response:
                    if response.status >= 400:
                        logger.error(
                            "Failed to send alert to webhook: %s %s",
                            response.status,
                            await response.text()
                        )
                        return False
                    return True
        except Exception as e:
            logger.exception("Error sending alert to webhook: %s", str(e))
            return False

class AlertManager:
    """Manages alert routing and processing."""
    
    def __init__(self, metrics_enabled: bool = True):
        """Initialize the alert manager.
        
        Args:
            metrics_enabled: Whether to enable metrics collection
        """
        self.receivers: Dict[str, AlertReceiver] = {}
        self.routes: List[Tuple[Callable[[AlertContext], bool], List[str]]] = []
        self.silenced: Dict[str, datetime] = {}
        self.inhibited: Set[str] = set()
        self.mutex = asyncio.Lock()
        self.metrics_enabled = metrics_enabled
        
        # Track alert state
        self.active_alerts: Dict[str, AlertContext] = {}
        self.alert_history: List[Tuple[datetime, AlertContext]] = []
        self.history_limit = 1000
        
        # Register default console receiver
        self.add_receiver('console', ConsoleReceiver())
    
    async def _record_alert_metrics(self, alert: AlertContext) -> None:
        """Record metrics for an alert.
        
        Args:
            alert: The alert to record metrics for
        """
        if not self.metrics_enabled:
            return
            
        try:
            metrics = get_metrics()
            metrics.incr("alerts_total", labels={
                'name': alert.name,
                'severity': alert.severity.value,
                'status': alert.status.value
            })
        except Exception as e:
            logger.warning("Failed to record alert metrics: %s", str(e))
    
    async def send_alert(self, alert: AlertContext) -> bool:
        """Send an alert through all matching receivers.
        
        Args:
            alert: The alert to send
            
        Returns:
            bool: True if the alert was sent successfully to at least one receiver
        """
        # Record metrics before sending
        await self._record_alert_metrics(alert)
        
        async with self.mutex:
            # Check if alert is silenced
            if alert.name in self.silenced:
                if datetime.utcnow() < self.silenced[alert.name]:
                    logger.debug("Alert %s is silenced", alert.name)
                    alert.status = AlertStatus.SUPPRESSED
                else:
                    # Clean up expired silence
                    self.silenced.pop(alert.name)
            
            # Update active alerts
            self.active_alerts[alert.fingerprint] = alert
            
            # Add to history (with limit)
            self.alert_history.append((datetime.utcnow(), alert))
            if len(self.alert_history) > self.history_limit:
                self.alert_history.pop(0)
            
            # Find matching receivers
            receivers = set()
            for match_func, route_receivers in self.routes:
                if match_func(alert):
                    for receiver_name in route_receivers:
                        if receiver_name in self.receivers:
                            receivers.add(self.receivers[receiver_name])
            
            # If no receivers matched, use all
            if not receivers and self.receivers:
                receivers = set(self.receivers.values())
            
            # Send to all matching receivers
            if not receivers:
                logger.warning("No receivers configured for alert: %s", alert.name)
                return False
            
            # Send alerts in parallel
            results = await asyncio.gather(
                *[receiver.send(alert) for receiver in receivers],
                return_exceptions=True
            )
            
            # Log any failures
            for receiver, result in zip(receivers, results):
                if isinstance(result, Exception):
                    logger.error(
                        "Error sending alert to %s: %s",
                        receiver.__class__.__name__,
                        str(result)
                    )
            
            return any(not isinstance(r, Exception) and r for r in results)
    
    async def resolve_alert(self, alert_name: str) -> bool:
        """Resolve an active alert.
        
        Args:
            alert_name: Name of the alert to resolve
            
        Returns:
            bool: True if the alert was found and resolved
        """
        async with self.mutex:
            resolved = False
            now = datetime.utcnow()
            
            for fingerprint, alert in list(self.active_alerts.items()):
                if alert.name == alert_name and alert.status == AlertStatus.FIRING:
                    alert.status = AlertStatus.RESOLVED
                    alert.ends_at = now
                    await self.send_alert(alert)
                    resolved = True
            
            return resolved
    
    def get_active_alerts(self) -> List[AlertContext]:
        """Get all active alerts."""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, limit: int = 100) -> List[AlertContext]:
        """Get recent alert history."""
        return [alert for _, alert in self.alert_history[-limit:]]

    def add_receiver(self, name: str, receiver: AlertReceiver) -> None:
        """Add a receiver.
        
        Args:
            name: Name of the receiver
            receiver: Receiver instance
        """
        self.receivers[name] = receiver
    
    def add_route(
        self,
        match_func: Callable[[AlertContext], bool],
        receivers: List[str]
    ) -> None:
        """Add a routing rule.
        
        Args:
            match_func: Function that returns True if the alert should be sent to the receivers
            receivers: List of receiver names
        """
        self.routes.append((match_func, receivers))
    
    def silence(self, alert_name: str, duration: int = 3600) -> None:
        """Silence alerts matching the given name.
        
        Args:
            alert_name: Name of the alert to silence
            duration: Duration in seconds to silence for (0 = indefinitely)
        """
        if duration > 0:
            self.silenced[alert_name] = datetime.utcnow() + timedelta(seconds=duration)
        else:
            self.silenced[alert_name] = datetime.max
    
    def unsilence(self, alert_name: str) -> None:
        """Remove silence for an alert.
        
        Args:
            alert_name: Name of the alert to unsilence
        """
        self.silenced.pop(alert_name, None)

# Global instance with metrics enabled by default
alert_manager = AlertManager(metrics_enabled=True)

def alert(
    name: str,
    severity: AlertSeverity = AlertSeverity.WARNING,
    summary: Optional[str] = None,
    **kwargs
) -> AlertContext:
    """Create and send an alert.
    
    Args:
        name: Name of the alert
        severity: Severity level
        summary: Brief description of the alert
        **kwargs: Additional fields for the alert context
        
    Returns:
        The created alert context
    """
    if summary is None:
        summary = name
    
    alert_ctx = AlertContext(
        name=name,
        severity=severity,
        summary=summary,
        **kwargs
    )
    
    # Schedule the alert to be sent
    asyncio.create_task(alert_manager.send_alert(alert_ctx))
    
    return alert_ctx

def critical_alert(name: str, **kwargs) -> AlertContext:
    """Create and send a critical alert."""
    return alert(name, AlertSeverity.CRITICAL, **kwargs)

def error_alert(name: str, **kwargs) -> AlertContext:
    """Create and send an error alert."""
    return alert(name, AlertSeverity.ERROR, **kwargs)

def warning_alert(name: str, **kwargs) -> AlertContext:
    """Create and send a warning alert."""
    return alert(name, AlertSeverity.WARNING, **kwargs)

def info_alert(name: str, **kwargs) -> AlertContext:
    """Create and send an informational alert."""
    return alert(name, AlertSeverity.INFO, **kwargs)
