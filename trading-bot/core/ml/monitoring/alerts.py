"""
Alerting System for ML Monitoring

This module provides a flexible and extensible alerting system for ML model monitoring.
It includes alert definitions, handlers, and rule-based alert generation.
"""

import json
import logging
import smtplib
import socket
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Union, Callable, Type, TypeVar, ClassVar
from typing_extensions import Protocol
import requests
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)

class AlertLevel(Enum):    
    """Severity levels for alerts."""
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()

    def __str__(self) -> str:
        return self.name

class AlertStatus(Enum):
    """Status of an alert."""
    TRIGGERED = auto()
    ACKNOWLEDGED = auto()
    RESOLVED = auto()
    SUPPRESSED = auto()

    def __str__(self) -> str:
        return self.name

@dataclass
class AlertContext:
    """Contextual information about where an alert was generated."""
    source: str = "unknown"
    component: Optional[str] = None
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

class Alert(BaseModel):
    """Base class for all alerts in the monitoring system."""
    
    # Required fields
    name: str
    message: str
    level: AlertLevel = AlertLevel.INFO
    context: AlertContext = field(default_factory=AlertContext)
    
    # Optional fields
    status: AlertStatus = AlertStatus.TRIGGERED
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolved_by: Optional[str] = None
    
    # Alert metadata
    fingerprint: Optional[str] = None
    deduplication_key: Optional[str] = None
    suppress_duplicates: bool = True
    suppress_window: int = 3600  # seconds
    
    # Alert details
    details: Dict[str, Any] = field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
            AlertLevel: lambda x: x.name,
            AlertStatus: lambda x: x.name
        }
    
    def __post_init__(self) -> None:
        """Initialize alert with default values."""
        # Generate a fingerprint for deduplication if not provided
        if not self.fingerprint:
            self.fingerprint = self._generate_fingerprint()
        
        # Set deduplication key if not provided
        if not self.deduplication_key:
            self.deduplication_key = f"{self.name}:{self.fingerprint}"
    
    def _generate_fingerprint(self) -> str:
        """Generate a fingerprint for alert deduplication."""
        import hashlib
        fingerprint_data = {
            'name': self.name,
            'message': self.message,
            'level': self.level.name,
            'source': self.context.source,
            'component': self.context.component,
            'model': f"{self.context.model_name}:{self.context.model_version}"
        }
        return hashlib.md5(
            json.dumps(fingerprint_data, sort_keys=True).encode('utf-8')
        ).hexdigest()
    
    def acknowledge(self, user: Optional[str] = None) -> None:
        """Mark the alert as acknowledged."""
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_at = datetime.utcnow()
        self.acknowledged_by = user or "system"
        self.updated_at = datetime.utcnow()
    
    def resolve(self, user: Optional[str] = None) -> None:
        """Mark the alert as resolved."""
        self.status = AlertStatus.RESOLVED
        self.resolved_at = datetime.utcnow()
        self.resolved_by = user or "system"
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'name': self.name,
            'message': self.message,
            'level': self.level.name,
            'status': self.status.name,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'acknowledged_by': self.acknowledged_by,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'resolved_by': self.resolved_by,
            'context': {
                'source': self.context.source,
                'component': self.context.component,
                'model_name': self.context.model_name,
                'model_version': self.context.model_version,
                'metadata': self.context.metadata
            },
            'details': self.details,
            'fingerprint': self.fingerprint,
            'deduplication_key': self.deduplication_key
        }
    
    def to_json(self) -> str:
        """Convert alert to JSON string."""
        return json.dumps(self.to_dict(), default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Alert':
        """Create an alert from a dictionary."""
        context_data = data.pop('context', {})
        context = AlertContext(
            source=context_data.get('source', 'unknown'),
            component=context_data.get('component'),
            model_name=context_data.get('model_name'),
            model_version=context_data.get('model_version'),
            metadata=context_data.get('metadata', {})
        )
        
        return cls(
            name=data['name'],
            message=data['message'],
            level=AlertLevel[data.get('level', 'INFO')],
            context=context,
            status=AlertStatus[data.get('status', 'TRIGGERED')],
            created_at=datetime.fromisoformat(data['created_at']) if 'created_at' in data else None,
            updated_at=datetime.fromisoformat(data['updated_at']) if 'updated_at' in data else None,
            acknowledged_at=datetime.fromisoformat(data['acknowledged_at']) if data.get('acknowledged_at') else None,
            acknowledged_by=data.get('acknowledged_by'),
            resolved_at=datetime.fromisoformat(data['resolved_at']) if data.get('resolved_at') else None,
            resolved_by=data.get('resolved_by'),
            fingerprint=data.get('fingerprint'),
            deduplication_key=data.get('deduplication_key'),
            details=data.get('details', {})
        )

class AlertHandler(ABC):
    """Base class for all alert handlers."""
    
    def __init__(self, min_level: AlertLevel = AlertLevel.INFO):
        self.min_level = min_level
        self.last_sent: Dict[str, datetime] = {}
    
    @abstractmethod
    def handle(self, alert: Alert) -> bool:
        """Handle an alert.
        
        Args:
            alert: The alert to handle
            
        Returns:
            bool: True if the alert was handled successfully, False otherwise
        """
        pass
    
    def should_handle(self, alert: Alert) -> bool:
        """Determine if this handler should process the alert."""
        # Check minimum level
        if alert.level.value < self.min_level.value:
            return False
        
        # Check for duplicate suppression
        if alert.suppress_duplicates and alert.deduplication_key:
            last_sent = self.last_sent.get(alert.deduplication_key)
            if last_sent and (datetime.utcnow() - last_sent).total_seconds() < alert.suppress_window:
                return False
        
        return True
    
    def mark_sent(self, alert: Alert) -> None:
        """Mark an alert as sent."""
        if alert.deduplication_key:
            self.last_sent[alert.deduplication_key] = datetime.utcnow()

class ConsoleAlertHandler(AlertHandler):
    """Handler that writes alerts to the console."""
    
    def __init__(self, min_level: AlertLevel = AlertLevel.INFO, 
                 format_string: Optional[str] = None):
        super().__init__(min_level)
        self.format_string = format_string or "[{level}] {name}: {message}"
    
    def handle(self, alert: Alert) -> bool:
        if not self.should_handle(alert):
            return False
        
        try:
            # Format the alert message
            message = self.format_string.format(
                name=alert.name,
                level=alert.level.name,
                message=alert.message,
                source=alert.context.source,
                component=alert.context.component or 'unknown',
                model=f"{alert.context.model_name or 'unknown'}:{alert.context.model_version or 'unknown'}",
                timestamp=alert.created_at.isoformat()
            )
            
            # Print to console with appropriate log level
            if alert.level == AlertLevel.DEBUG:
                logger.debug(message)
            elif alert.level == AlertLevel.INFO:
                logger.info(message)
            elif alert.level == AlertLevel.WARNING:
                logger.warning(message)
            elif alert.level == AlertLevel.ERROR:
                logger.error(message)
            else:  # CRITICAL
                logger.critical(message)
            
            self.mark_sent(alert)
            return True
            
        except Exception as e:
            logger.error(f"Error handling console alert: {e}")
            return False

class EmailAlertHandler(AlertHandler):
    """Handler that sends alerts via email."""
    
    def __init__(
        self,
        smtp_server: str,
        smtp_port: int,
        sender_email: str,
        sender_password: str,
        recipient_emails: List[str],
        subject_prefix: str = "[ML Monitoring] ",
        min_level: AlertLevel = AlertLevel.WARNING,
        use_tls: bool = True,
        timeout: int = 10
    ):
        super().__init__(min_level)
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.recipient_emails = recipient_emails
        self.subject_prefix = subject_prefix
        self.use_tls = use_tls
        self.timeout = timeout
    
    def handle(self, alert: Alert) -> bool:
        if not self.should_handle(alert) or not self.recipient_emails:
            return False
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = ", ".join(self.recipient_emails)
            msg['Subject'] = f"{self.subject_prefix}{alert.name} - {alert.level.name}"
            
            # Create HTML content
            html = f"""
            <html>
            <body>
                <h2>{alert.name}</h2>
                <p><strong>Level:</strong> {alert.level.name}</p>
                <p><strong>Status:</strong> {alert.status.name}</p>
                <p><strong>Time:</strong> {alert.created_at}</p>
                <p><strong>Source:</strong> {alert.context.source}</p>
                <p><strong>Component:</strong> {alert.context.component or 'N/A'}</p>
                <p><strong>Model:</strong> {alert.context.model_name or 'N/A'} (v{alert.context.model_version or '?'})</p>
                <hr>
                <h3>Message</h3>
                <p>{alert.message}</p>
            """
            
            # Add details if present
            if alert.details:
                html += "<h3>Details</h3><pre>"
                html += json.dumps(alert.details, indent=2, default=str)
                html += "</pre>"
            
            html += "</body></html>"
            
            # Attach HTML content
            msg.attach(MIMEText(html, 'html'))
            
            # Connect to SMTP server and send
            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=self.timeout) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            self.mark_sent(alert)
            return True
            
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
            return False

class SlackAlertHandler(AlertHandler):
    """Handler that sends alerts to a Slack channel."""
    
    def __init__(
        self,
        webhook_url: str,
        channel: Optional[str] = None,
        username: str = "ML Monitoring",
        icon_emoji: Optional[str] = None,
        min_level: AlertLevel = AlertLevel.WARNING,
        timeout: int = 10
    ):
        super().__init__(min_level)
        self.webhook_url = webhook_url
        self.channel = channel
        self.username = username
        self.icon_emoji = icon_emoji or self._get_default_emoji()
        self.timeout = timeout
    
    def _get_default_emoji(self) -> str:
        return {
            AlertLevel.DEBUG: ":loudspeaker:",
            AlertLevel.INFO: ":information_source:",
            AlertLevel.WARNING: ":warning:",
            AlertLevel.ERROR: ":x:",
            AlertLevel.CRITICAL: ":fire:"
        }.get(self.min_level, ":loudspeaker:")
    
    def _get_color(self, level: AlertLevel) -> str:
        return {
            AlertLevel.DEBUG: "#808080",    # Gray
            AlertLevel.INFO: "#36a64f",     # Green
            AlertLevel.WARNING: "#ffcc00",  # Yellow
            AlertLevel.ERROR: "#ff6600",    # Orange
            AlertLevel.CRITICAL: "#ff0000"  # Red
        }.get(level, "#808080")
    
    def handle(self, alert: Alert) -> bool:
        if not self.should_handle(alert):
            return False
        
        try:
            # Create Slack message
            payload = {
                'username': self.username,
                'icon_emoji': self.icon_emoji,
                'attachments': [
                    {
                        'fallback': f"[{alert.level.name}] {alert.name}: {alert.message}",
                        'color': self._get_color(alert.level),
                        'title': f"{alert.name}",
                        'text': alert.message,
                        'fields': [
                            {
                                'title': 'Level',
                                'value': alert.level.name,
                                'short': True
                            },
                            {
                                'title': 'Status',
                                'value': alert.status.name,
                                'short': True
                            },
                            {
                                'title': 'Source',
                                'value': alert.context.source,
                                'short': True
                            },
                            {
                                'title': 'Component',
                                'value': alert.context.component or 'N/A',
                                'short': True
                            },
                            {
                                'title': 'Model',
                                'value': f"{alert.context.model_name or 'N/A'} (v{alert.context.model_version or '?'})",
                                'short': True
                            },
                            {
                                'title': 'Time',
                                'value': alert.created_at.strftime('%Y-%m-%d %H:%M:%S UTC'),
                                'short': True
                            }
                        ],
                        'footer': f"Fingerprint: {alert.fingerprint}",
                        'ts': alert.created_at.timestamp()
                    }
                ]
            }
            
            # Add details if present
            if alert.details:
                payload['attachments'][0]['fields'].extend([
                    {
                        'title': 'Details',
                        'value': f"```{json.dumps(alert.details, indent=2, default=str)}```",
                        'short': False
                    }
                ])
            
            # Add channel if specified
            if self.channel:
                payload['channel'] = self.channel
            
            # Send request to Slack webhook
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=self.timeout
            )
            
            if response.status_code != 200:
                raise Exception(f"Slack API error: {response.status_code} - {response.text}")
            
            self.mark_sent(alert)
            return True
            
        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")
            return False

class AlertManager:
    """Manages alert handlers and routes alerts to the appropriate handlers."""
    
    def __init__(self):
        self.handlers: List[AlertHandler] = []
        self.alert_history: List[Alert] = []
        self.max_history: int = 1000
    
    def add_handler(self, handler: AlertHandler) -> None:
        """Add an alert handler."""
        if handler not in self.handlers:
            self.handlers.append(handler)
    
    def remove_handler(self, handler: AlertHandler) -> None:
        """Remove an alert handler."""
        if handler in self.handlers:
            self.handlers.remove(handler)
    
    def clear_handlers(self) -> None:
        """Remove all alert handlers."""
        self.handlers.clear()
    
    def handle_alert(self, alert: Alert) -> bool:
        """Handle an alert by sending it to all registered handlers."""
        if not isinstance(alert, Alert):
            logger.error(f"Invalid alert type: {type(alert)}")
            return False
        
        # Update timestamps
        alert.updated_at = datetime.utcnow()
        
        # Add to history
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history:
            self.alert_history.pop(0)
        
        # Route to handlers
        results = []
        for handler in self.handlers:
            try:
                result = handler.handle(alert)
                results.append(result)
            except Exception as e:
                logger.error(f"Error in alert handler {handler.__class__.__name__}: {e}")
                results.append(False)
        
        return any(results)
    
    def get_recent_alerts(
        self, 
        limit: int = 100, 
        min_level: Optional[AlertLevel] = None,
        status: Optional[AlertStatus] = None
    ) -> List[Alert]:
        """Get recent alerts, optionally filtered by level and status."""
        alerts = self.alert_history[-limit:] if limit > 0 else self.alert_history
        
        if min_level is not None:
            alerts = [a for a in alerts if a.level.value >= min_level.value]
        
        if status is not None:
            alerts = [a for a in alerts if a.status == status]
        
        return alerts

# Alert rule system
class AlertRule(ABC):
    """Base class for alert rules."""
    
    def __init__(
        self, 
        name: str,
        description: str = "",
        enabled: bool = True,
        min_severity: AlertLevel = AlertLevel.WARNING,
        cooldown: int = 300  # seconds
    ):
        self.name = name
        self.description = description
        self.enabled = enabled
        self.min_severity = min_severity
        self.cooldown = cooldown
        self.last_triggered: Dict[str, datetime] = {}
    
    @abstractmethod
    def evaluate(self, data: Any) -> Optional[Alert]:
        """Evaluate the rule against the given data and return an alert if triggered."""
        pass
    
    def is_on_cooldown(self, key: str) -> bool:
        """Check if the rule is on cooldown for the given key."""
        if not self.cooldown:
            return False
            
        last_trigger = self.last_triggered.get(key)
        if not last_trigger:
            return False
            
        return (datetime.utcnow() - last_trigger).total_seconds() < self.cooldown
    
    def mark_triggered(self, key: str) -> None:
        """Mark the rule as triggered for the given key."""
        self.last_triggered[key] = datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert rule to dictionary."""
        return {
            'name': self.name,
            'description': self.description,
            'enabled': self.enabled,
            'min_severity': self.min_severity.name,
            'cooldown': self.cooldown,
            'type': self.__class__.__name__
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AlertRule':
        """Create a rule from a dictionary."""
        # This is a factory method that creates the appropriate rule type
        rule_type = data.pop('type', '')
        rule_class = globals().get(rule_type)
        
        if not rule_class or not issubclass(rule_class, AlertRule):
            raise ValueError(f"Unknown rule type: {rule_type}")
        
        # Convert string enums back to enum values
        if 'min_severity' in data and isinstance(data['min_severity'], str):
            data['min_severity'] = AlertLevel[data['min_severity']]
        
        return rule_class(**data)

# Example rule implementations
class ThresholdAlertRule(AlertRule):
    """Alert when a metric crosses a threshold."""
    
    def __init__(
        self,
        name: str,
        metric_name: str,
        threshold: float,
        condition: str = '>',  # '>', '>=', '<', '<=', '==', '!='
        window_size: int = 1,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.metric_name = metric_name
        self.threshold = threshold
        self.condition = condition
        self.window_size = window_size
    
    def evaluate(self, data: Dict[str, Any]) -> Optional[Alert]:
        if not self.enabled or self.metric_name not in data:
            return None
            
        # Get the metric value(s)
        value = data[self.metric_name]
        
        # Handle windowed metrics (lists/arrays)
        if isinstance(value, (list, tuple)) and self.window_size > 1:
            values = value[-self.window_size:]
            value = sum(values) / len(values)  # Simple average for now
        
        # Check condition
        condition_met = False
        if self.condition == '>':
            condition_met = value > self.threshold
        elif self.condition == '>=':
            condition_met = value >= self.threshold
        elif self.condition == '<':
            condition_met = value < self.threshold
        elif self.condition == '<=':
            condition_met = value <= self.threshold
        elif self.condition == '==':
            condition_met = value == self.threshold
        elif self.condition == '!=':
            condition_met = value != self.threshold
        
        if not condition_met:
            return None
        
        # Check cooldown
        if self.is_on_cooldown(f"{self.metric_name}_{self.condition}_{self.threshold}"):
            return None
        
        # Create alert
        alert = Alert(
            name=self.name,
            message=f"{self.metric_name} {self.condition} {self.threshold} (value: {value:.4f})",
            level=self.min_severity,
            context=AlertContext(
                source="threshold_rule",
                component="monitoring",
                metadata={
                    'metric': self.metric_name,
                    'value': value,
                    'threshold': self.threshold,
                    'condition': self.condition,
                    'window_size': self.window_size
                }
            ),
            details={
                'metric': self.metric_name,
                'value': float(value) if hasattr(value, '__float__') else str(value),
                'threshold': self.threshold,
                'condition': self.condition,
                'window_size': self.window_size
            }
        )
        
        # Mark as triggered
        self.mark_triggered(f"{self.metric_name}_{self.condition}_{self.threshold}")
        
        return alert

class AnomalyAlertRule(AlertRule):
    """Alert when an anomaly is detected in a metric."""
    
    def __init__(
        self,
        name: str,
        metric_name: str,
        threshold: float = 3.0,  # Number of standard deviations
        window_size: int = 10,
        min_samples: int = 30,   # Minimum samples before alerting
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.metric_name = metric_name
        self.threshold = threshold
        self.window_size = window_size
        self.min_samples = min_samples
        self.values: List[float] = []
    
    def evaluate(self, data: Dict[str, Any]) -> Optional[Alert]:
        if not self.enabled or self.metric_name not in data:
            return None
            
        # Get the metric value
        value = data[self.metric_name]
        
        # Add to history
        self.values.append(float(value))
        
        # Keep only the most recent values
        if len(self.values) > self.window_size * 2:  # Keep some history
            self.values = self.values[-self.window_size:]
        
        # Not enough data yet
        if len(self.values) < self.min_samples:
            return None
        
        # Calculate statistics
        import numpy as np
        values = np.array(self.values)
        mean = np.mean(values)
        std = np.std(values)
        
        # Skip if standard deviation is zero (constant data)
        if std == 0:
            return None
        
        # Calculate z-score
        z_score = abs((value - mean) / std) if std != 0 else 0
        
        # Check if anomaly
        if z_score < self.threshold:
            return None
        
        # Check cooldown
        if self.is_on_cooldown(f"{self.metric_name}_anomaly_{self.threshold}"):
            return None
        
        # Create alert
        alert = Alert(
            name=self.name,
            message=f"Anomaly detected in {self.metric_name}: {value:.4f} (z-score: {z_score:.2f})",
            level=self.min_severity,
            context=AlertContext(
                source="anomaly_rule",
                component="monitoring",
                metadata={
                    'metric': self.metric_name,
                    'value': value,
                    'z_score': z_score,
                    'threshold': self.threshold,
                    'window_size': self.window_size
                }
            ),
            details={
                'metric': self.metric_name,
                'value': float(value) if hasattr(value, '__float__') else str(value),
                'z_score': float(z_score),
                'threshold': self.threshold,
                'window_size': self.window_size,
                'mean': float(mean),
                'std': float(std)
            }
        )
        
        # Mark as triggered
        self.mark_triggered(f"{self.metric_name}_anomaly_{self.threshold}")
        
        return alert

# Factory function for creating rules from configuration
def create_alert_rule(config: Dict[str, Any]) -> AlertRule:
    """Create an alert rule from a configuration dictionary."""
    rule_type = config.get('type', '')
    
    if rule_type == 'threshold':
        return ThresholdAlertRule(
            name=config['name'],
            metric_name=config['metric_name'],
            threshold=config['threshold'],
            condition=config.get('condition', '>'),
            window_size=config.get('window_size', 1),
            description=config.get('description', ''),
            enabled=config.get('enabled', True),
            min_severity=AlertLevel[config.get('min_severity', 'WARNING')],
            cooldown=config.get('cooldown', 300)
        )
    elif rule_type == 'anomaly':
        return AnomalyAlertRule(
            name=config['name'],
            metric_name=config['metric_name'],
            threshold=config.get('threshold', 3.0),
            window_size=config.get('window_size', 10),
            min_samples=config.get('min_samples', 30),
            description=config.get('description', ''),
            enabled=config.get('enabled', True),
            min_severity=AlertLevel[config.get('min_severity', 'WARNING')],
            cooldown=config.get('cooldown', 300)
        )
    else:
        raise ValueError(f"Unknown alert rule type: {rule_type}")

# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create alert manager
    alert_manager = AlertManager()
    
    # Add console handler
    console_handler = ConsoleAlertHandler(min_level=AlertLevel.DEBUG)
    alert_manager.add_handler(console_handler)
    
    # Example: Create and trigger an alert
    alert = Alert(
        name="HighLatencyAlert",
        message="API latency exceeded threshold",
        level=AlertLevel.WARNING,
        context=AlertContext(
            source="api_gateway",
            component="request_processing",
            model_name="recommendation_engine",
            model_version="1.2.3"
        ),
        details={
            'endpoint': '/api/recommend',
            'latency_ms': 1250,
            'threshold_ms': 1000,
            'request_id': 'req_12345'
        }
    )
    
    # Handle the alert
    alert_manager.handle_alert(alert)
    
    # Example: Using a threshold rule
    rule = ThresholdAlertRule(
        name="HighErrorRate",
        description="Alert when error rate exceeds 5%",
        metric_name="error_rate",
        threshold=0.05,
        condition=">",
        min_severity=AlertLevel.ERROR
    )
    
    # Simulate evaluating the rule
    metrics = {"error_rate": 0.07}  # 7% error rate
    alert = rule.evaluate(metrics)
    if alert:
        alert_manager.handle_alert(alert)
