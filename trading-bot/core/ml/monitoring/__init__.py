"""
ML Monitoring Package

This package provides comprehensive monitoring capabilities for machine learning models in production.
It includes drift detection, explainability, performance tracking, and alerting.
"""

__version__ = "1.0.0"

# Core exports
from .drift_detector import (
    BaseDriftDetector,
    StatisticalDriftDetector,
    ModelDriftDetector,
    DriftDetectorFactory,
    DriftResult,
    DriftType
)

from ..explainability.shap_explainer import (
    SHAPExplainer,
    ExplanationResult,
    ExplanationType
)

from .metrics import (
    ModelMetrics,
    PerformanceMetrics,
    DataQualityMetrics
)

from .alerts import (
    Alert,
    AlertLevel,
    AlertManager,
    AlertRule,
    NotificationHandler
)

from .pipeline import (
    ModelMonitor,
    MonitoringConfig,
    MonitoringPipeline,
    DataStream,
    ModelInput,
    ModelOutput,
    PredictionRecord
)

# Re-export commonly used types for convenience
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import numpy as np

# Set up logging
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Suppress noisy library logs
logging.getLogger('shap').setLevel(logging.WARNING)
logging.getLogger('alibi_detect').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Initialize global components
try:
    from ..versioning import ModelVersionManager
    version_manager = ModelVersionManager()
except ImportError:
    logger.warning("Model versioning not available. Some features may be limited.")
    version_manager = None

alert_manager = AlertManager()

# Add default notification handlers
try:
    from .notifications import (
        ConsoleNotificationHandler,
        EmailNotificationHandler,
        SlackNotificationHandler
    )
    
    # Add console handler by default
    console_handler = ConsoleNotificationHandler()
    alert_manager.add_handler(console_handler)
    
    # Try to add email handler if configured
    try:
        from ..config import settings
        if settings.EMAIL_ENABLED:
            email_handler = EmailNotificationHandler(
                smtp_server=settings.SMTP_SERVER,
                smtp_port=settings.SMTP_PORT,
                sender_email=settings.EMAIL_SENDER,
                sender_password=settings.EMAIL_PASSWORD,
                recipient_emails=settings.ALERT_RECIPIENTS
            )
            alert_manager.add_handler(email_handler)
    except ImportError:
        logger.debug("Email notifications not configured")
    
    # Try to add Slack handler if configured
    try:
        if settings.SLACK_WEBHOOK_URL:
            slack_handler = SlackNotificationHandler(
                webhook_url=settings.SLACK_WEBHOOK_URL,
                channel=settings.SLACK_CHANNEL,
                username=settings.SLACK_USERNAME
            )
            alert_manager.add_handler(slack_handler)
    except (ImportError, AttributeError):
        logger.debug("Slack notifications not configured")
        
except ImportError as e:
    logger.warning(f"Failed to initialize notification handlers: {e}")

# Add default alert rules
try:
    from .rules import (
        DataDriftRule,
        PerformanceDegradationRule,
        DataQualityRule,
        ResourceUsageRule
    )
    
    # Add default rules if none exist
    if not alert_manager.get_rules():
        alert_manager.add_rule(DataDriftRule())
        alert_manager.add_rule(PerformanceDegradationRule())
        alert_manager.add_rule(DataQualityRule())
        alert_manager.add_rule(ResourceUsageRule())
        
except ImportError as e:
    logger.warning(f"Failed to initialize default alert rules: {e}")

# Export common types for type hints
ModelType = Any
DataFrame = pd.DataFrame
NDArray = np.ndarray

"""
Monitoring Module

This module provides functionality for monitoring ML models in production,
including performance metrics, drift detection, and alerting.
"""

import os
from typing import Optional

from .base import Monitor
from .file_monitor import FileMonitor

__all__ = ['Monitor', 'FileMonitor', 'get_monitor']

def get_monitor(
    monitor_type: str = 'file',
    log_dir: Optional[str] = None,
    **kwargs
) -> Monitor:
    """Get a monitoring instance.
    
    Args:
        monitor_type: Type of monitor to create ('file')
        log_dir: Directory to store logs (defaults to 'monitoring' in current directory)
        **kwargs: Additional arguments for the monitor constructor
        
    Returns:
        An instance of Monitor
    """
    if log_dir is None:
        log_dir = os.path.join(os.getcwd(), 'monitoring')
    
    if monitor_type == 'file':
        return FileMonitor(log_dir=log_dir, **kwargs)
    else:
        raise ValueError(f"Unsupported monitor type: {monitor_type}")
