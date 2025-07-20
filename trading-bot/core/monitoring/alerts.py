"""
Alerting system for the trading bot.

This module provides functionality for creating, managing, and dispatching alerts
for important events and data quality issues.
"""
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Callable, Awaitable, Union

from pydantic import BaseModel, Field, validator

from .metrics_server import MetricsRegistry

logger = logging.getLogger(__name__)

class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"

class AlertType(Enum):
    """Types of alerts."""
    DATA_QUALITY = "data_quality"
    PERFORMANCE = "performance"
    SECURITY = "security"
    SYSTEM = "system"
    TRADING = "trading"
    RISK = "risk"

@dataclass
class AlertContext:
    """Context for an alert."""
    source: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

class Alert(BaseModel):
    """An alert that can be raised and tracked."""
    id: str = Field(default_factory=lambda: f"alert_{int(time.time() * 1000)}")
    title: str
    message: str
    level: AlertLevel
    type: AlertType
    status: AlertStatus = AlertStatus.ACTIVE
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[float] = None
    resolved_at: Optional[float] = None
    context: Optional[AlertContext] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
        }
    
    @validator('level', pre=True)
    def validate_level(cls, v):
        if isinstance(v, str):
            return AlertLevel(v.lower())
        return v
    
    @validator('type', pre=True)
    def validate_type(cls, v):
        if isinstance(v, str):
            return AlertType(v.lower())
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the alert to a dictionary."""
        data = self.dict()
        data['created_at'] = datetime.fromtimestamp(self.created_at).isoformat()
        data['updated_at'] = datetime.fromtimestamp(self.updated_at).isoformat()
        
        if self.acknowledged_at:
            data['acknowledged_at'] = datetime.fromtimestamp(self.acknowledged_at).isoformat()
        
        if self.resolved_at:
            data['resolved_at'] = datetime.fromtimestamp(self.resolved_at).isoformat()
        
        if self.context and hasattr(self.context, 'timestamp'):
            data['context']['timestamp'] = datetime.fromtimestamp(self.context.timestamp).isoformat()
        
        return data
    
    def acknowledge(self, user: Optional[str] = None) -> None:
        """Acknowledge the alert."""
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_by = user or "system"
        self.acknowledged_at = time.time()
        self.updated_at = time.time()
    
    def resolve(self) -> None:
        """Mark the alert as resolved."""
        self.status = AlertStatus.RESOLVED
        self.resolved_at = time.time()
        self.updated_at = time.time()
    
    def suppress(self) -> None:
        """Suppress the alert."""
        self.status = AlertStatus.SUPPRESSED
        self.updated_at = time.time()

class AlertRule(BaseModel):
    """A rule that defines when an alert should be triggered."""
    id: str
    name: str
    description: str
    condition: Callable[[Dict[str, Any]], bool]
    alert_config: Dict[str, Any]
    enabled: bool = True
    cooldown: int = 300  # seconds
    last_triggered: Optional[float] = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def should_trigger(self, data: Dict[str, Any]) -> bool:
        """Check if the alert should be triggered."""
        if not self.enabled:
            return False
        
        # Check cooldown
        if self.last_triggered and (time.time() - self.last_triggered) < self.cooldown:
            return False
        
        # Check condition
        try:
            result = self.condition(data)
            if result:
                self.last_triggered = time.time()
            return result
        except Exception as e:
            logger.error(f"Error evaluating alert rule {self.id}: {e}")
            return False
    
    def create_alert(self, context: Optional[Dict[str, Any]] = None) -> Alert:
        """Create an alert from this rule."""
        return Alert(
            title=self.alert_config.get('title', 'Alert'),
            message=self.alert_config.get('message', ''),
            level=AlertLevel(self.alert_config.get('level', 'info')),
            type=AlertType(self.alert_config.get('type', 'system')),
            context=AlertContext(
                source=self.alert_config.get('source', 'alert_rule'),
                metadata={
                    'rule_id': self.id,
                    'rule_name': self.name,
                    **(context or {})
                }
            )
        )

class AlertManager(metaclass=type('_Singleton', (), {'_instance': None})):
    """Manages alerts and alert rules."""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._alerts: Dict[str, Alert] = {}
        self._rules: Dict[str, AlertRule] = {}
        self._handlers: List[Callable[[Alert], Awaitable[None]]] = []
        self._metrics = MetricsRegistry()
        self._max_alerts = 1000  # Maximum number of alerts to keep in memory
        self._initialized = True
        
        # Register default alert rules
        self._register_default_rules()
    
    def _register_default_rules(self) -> None:
        """Register default alert rules."""
        # High latency alert
        self.add_rule(AlertRule(
            id="high_latency",
            name="High Processing Latency",
            description="Alert when processing latency exceeds threshold",
            condition=lambda data: data.get('latency', 0) > 1.0,  # 1 second
            alert_config={
                'title': 'High Processing Latency',
                'message': 'Processing latency has exceeded 1 second',
                'level': 'warning',
                'type': 'performance',
                'source': 'alert_manager',
            },
            cooldown=300  # 5 minutes
        ))
        
        # Data quality alert
        self.add_rule(AlertRule(
            id="data_quality_issue",
            name="Data Quality Issue",
            description="Alert when data quality issues are detected",
            condition=lambda data: data.get('data_quality_score', 1.0) < 0.8,
            alert_config={
                'title': 'Data Quality Issue',
                'message': 'Data quality score has dropped below threshold',
                'level': 'error',
                'type': 'data_quality',
                'source': 'data_validator',
            }
        ))
        
        # System error alert
        self.add_rule(AlertRule(
            id="system_error",
            name="System Error",
            description="Alert when a system error occurs",
            condition=lambda data: data.get('error_count', 0) > 0,
            alert_config={
                'title': 'System Error',
                'message': 'A system error has occurred',
                'level': 'error',
                'type': 'system',
                'source': 'system_monitor',
            }
        ))
    
    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        self._rules[rule.id] = rule
        logger.info(f"Added alert rule: {rule.name} ({rule.id})")
    
    def remove_rule(self, rule_id: str) -> None:
        """Remove an alert rule."""
        if rule_id in self._rules:
            del self._rules[rule_id]
            logger.info(f"Removed alert rule: {rule_id}")
    
    def get_rule(self, rule_id: str) -> Optional[AlertRule]:
        """Get an alert rule by ID."""
        return self._rules.get(rule_id)
    
    def get_rules(self) -> List[AlertRule]:
        """Get all alert rules."""
        return list(self._rules.values())
    
    def add_handler(self, handler: Callable[[Alert], Awaitable[None]]) -> None:
        """Add an alert handler."""
        self._handlers.append(handler)
    
    def remove_handler(self, handler: Callable[[Alert], Awaitable[None]]) -> None:
        """Remove an alert handler."""
        if handler in self._handlers:
            self._handlers.remove(handler)
    
    async def evaluate_rules(self, data: Dict[str, Any]) -> List[Alert]:
        """Evaluate all rules against the given data and return triggered alerts."""
        triggered_alerts = []
        
        for rule in self._rules.values():
            if rule.should_trigger(data):
                alert = rule.create_alert(context={
                    'data': data,
                    'rule_id': rule.id,
                    'rule_name': rule.name,
                })
                triggered_alerts.append(alert)
        
        # Process triggered alerts
        for alert in triggered_alerts:
            await self.trigger_alert(alert)
        
        return triggered_alerts
    
    async def trigger_alert(self, alert: Union[Alert, Dict[str, Any]]) -> Alert:
        """Trigger an alert."""
        if isinstance(alert, dict):
            alert = Alert(**alert)
        
        # Add to alerts dictionary
        self._alerts[alert.id] = alert
        
        # Update metrics
        self._metrics.counter(
            'alerts_total',
            'Total number of alerts triggered',
            ['level', 'type', 'status']
        ).inc(1, {
            'level': alert.level.value,
            'type': alert.type.value,
            'status': alert.status.value
        })
        
        # Call handlers
        for handler in self._handlers:
            try:
                await handler(alert)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
        
        # Log the alert
        logger.log(
            getattr(logging, alert.level.upper(), logging.INFO),
            f"Alert triggered: {alert.title} - {alert.message}"
        )
        
        # Clean up old alerts
        self._cleanup_alerts()
        
        return alert
    
    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get an alert by ID."""
        return self._alerts.get(alert_id)
    
    def get_alerts(
        self,
        status: Optional[AlertStatus] = None,
        level: Optional[AlertLevel] = None,
        alert_type: Optional[AlertType] = None,
        limit: int = 100
    ) -> List[Alert]:
        """Get alerts matching the given filters."""
        alerts = list(self._alerts.values())
        
        if status is not None:
            alerts = [a for a in alerts if a.status == status]
        
        if level is not None:
            alerts = [a for a in alerts if a.level == level]
        
        if alert_type is not None:
            alerts = [a for a in alerts if a.type == alert_type]
        
        # Sort by creation time (newest first)
        alerts.sort(key=lambda a: a.created_at, reverse=True)
        
        return alerts[:limit]
    
    async def acknowledge_alert(self, alert_id: str, user: Optional[str] = None) -> Optional[Alert]:
        """Acknowledge an alert."""
        alert = self._alerts.get(alert_id)
        if alert:
            alert.acknowledge(user)
            await self.trigger_alert(alert)  # Update the alert
            return alert
        return None
    
    async def resolve_alert(self, alert_id: str) -> Optional[Alert]:
        """Resolve an alert."""
        alert = self._alerts.get(alert_id)
        if alert:
            alert.resolve()
            await self.trigger_alert(alert)  # Update the alert
            return alert
        return None
    
    def _cleanup_alerts(self) -> None:
        """Clean up old alerts if we've reached the maximum number."""
        if len(self._alerts) <= self._max_alerts:
            return
        
        # Remove the oldest alerts
        alerts = sorted(self._alerts.values(), key=lambda a: a.created_at)
        for alert in alerts[:len(self._alerts) - self._max_alerts]:
            del self._alerts[alert.id]

# Global alert manager instance
alert_manager = AlertManager()

"""
Alert Manager for Trading Events

This module provides functionality to manage and send alerts for important trading events.
"""
import logging
from typing import Dict, List, Optional, Union, Any
from enum import Enum, auto
from dataclasses import dataclass, asdict
import json
import requests
from datetime import datetime
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Severity levels for alerts."""
    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()
    ERROR = auto()
    SUCCESS = auto()

@dataclass
class Alert:
    """Represents a trading alert."""
    title: str
    message: str
    severity: AlertSeverity
    timestamp: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'title': self.title,
            'message': self.message,
            'severity': self.severity.name,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }
    
    def to_json(self) -> str:
        """Convert alert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

class AlertManager:
    """Manages and dispatches trading alerts."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the alert manager.
        
        Args:
            config: Configuration dictionary with alert settings
        """
        self.config = config or {}
        self.alert_history: List[Alert] = []
        self.max_history = self.config.get('max_history', 1000)
        self.enabled = self.config.get('enabled', True)
        
        # Configure alert methods
        self.console_enabled = self.config.get('console', True)
        self.logfile_enabled = self.config.get('logfile', True)
        self.webhook_url = self.config.get('webhook_url')
        self.logfile_path = Path(self.config.get('logfile_path', 'logs/alerts.log'))
        
        # Ensure log directory exists
        if self.logfile_enabled:
            self.logfile_path.parent.mkdir(parents=True, exist_ok=True)
    
    def send_alert(self, alert: Alert) -> bool:
        """Send an alert through all configured channels.
        
        Args:
            alert: The alert to send
            
        Returns:
            bool: True if all alert methods succeeded
        """
        if not self.enabled:
            return False
            
        # Add to history
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history:
            self.alert_history.pop(0)
        
        success = True
        
        # Console output
        if self.console_enabled:
            success &= self._send_to_console(alert)
            
        # Log file
        if self.logfile_enabled:
            success &= self._write_to_logfile(alert)
            
        # Webhook
        if self.webhook_url:
            success &= self._send_webhook(alert)
            
        return success
    
    def _send_to_console(self, alert: Alert) -> bool:
        """Send alert to console."""
        try:
            log_msg = f"[{alert.timestamp}] {alert.severity.name}: {alert.title} - {alert.message}"
            if alert.severity == AlertSeverity.CRITICAL or alert.severity == AlertSeverity.ERROR:
                logger.error(log_msg)
            elif alert.severity == AlertSeverity.WARNING:
                logger.warning(log_msg)
            elif alert.severity == AlertSeverity.SUCCESS:
                logger.info(f"✅ {log_msg}")
            else:
                logger.info(log_msg)
            return True
        except Exception as e:
            logger.error(f"Failed to send alert to console: {e}")
            return False
    
    def _write_to_logfile(self, alert: Alert) -> bool:
        """Write alert to log file."""
        try:
            with open(self.logfile_path, 'a') as f:
                f.write(f"{alert.to_json()}\n")
            return True
        except Exception as e:
            logger.error(f"Failed to write alert to log file: {e}")
            return False
    
    def _send_webhook(self, alert: Alert) -> bool:
        """Send alert to webhook URL."""
        if not self.webhook_url:
            return False
            
        try:
            headers = {'Content-Type': 'application/json'}
            if 'api_key' in self.config:
                headers['Authorization'] = f"Bearer {self.config['api_key']}"
                
            response = requests.post(
                self.webhook_url,
                json=alert.to_dict(),
                headers=headers,
                timeout=10
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to send alert to webhook: {e}")
            return False

# Default alert manager instance
alert_manager = AlertManager()

def send_alert(
    title: str,
    message: str,
    severity: AlertSeverity = AlertSeverity.INFO,
    metadata: Optional[Dict[str, Any]] = None,
    manager: Optional[AlertManager] = None
) -> bool:
    """Helper function to send an alert.
    
    Args:
        title: Short title of the alert
        message: Detailed message
        severity: Alert severity level
        metadata: Additional metadata
        manager: Optional AlertManager instance (uses default if None)
        
    Returns:
        bool: True if alert was sent successfully
    """
    if manager is None:
        manager = alert_manager
        
    alert = Alert(
        title=title,
        message=message,
        severity=severity,
        metadata=metadata or {}
    )
    
    return manager.send_alert(alert)

# Predefined alert types
def send_trade_alert(
    symbol: str,
    side: str,
    quantity: float,
    price: float,
    order_id: str,
    is_simulated: bool = False,
    manager: Optional[AlertManager] = None
) -> bool:
    """Send a trade execution alert."""
    sim_text = " (SIMULATED)" if is_simulated else ""
    return send_alert(
        title=f"Trade Executed{sim_text}",
        message=f"{side.upper()} {quantity} {symbol} @ {price}",
        severity=AlertSeverity.SUCCESS,
        metadata={
            'type': 'trade',
            'symbol': symbol,
            'side': side,
            'quantity': quantity,
            'price': price,
            'order_id': order_id,
            'is_simulated': is_simulated
        },
        manager=manager
    )

def send_risk_alert(
    message: str,
    level: str = "warning",
    metadata: Optional[Dict[str, Any]] = None,
    manager: Optional[AlertManager] = None
) -> bool:
    """Send a risk-related alert."""
    severity = {
        'info': AlertSeverity.INFO,
        'warning': AlertSeverity.WARNING,
        'critical': AlertSeverity.CRITICAL
    }.get(level.lower(), AlertSeverity.WARNING)
    
    return send_alert(
        title=f"Risk Alert: {level.upper()}",
        message=message,
        severity=severity,
        metadata={
            'type': 'risk',
            'risk_level': level,
            **(metadata or {})
        },
        manager=manager
    )

def send_system_alert(
    message: str,
    severity: AlertSeverity = AlertSeverity.ERROR,
    component: Optional[str] = None,
    manager: Optional[AlertManager] = None
) -> bool:
    """Send a system-level alert."""
    title = "System Alert"
    if component:
        title = f"{component} - {title}"
        
    return send_alert(
        title=title,
        message=message,
        severity=severity,
        metadata={
            'type': 'system',
            'component': component or 'unknown'
        },
        manager=manager
    )
