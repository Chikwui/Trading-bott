"""
Monitoring System Configuration

This module contains configuration settings for the trading system monitoring.
"""
from typing import Dict, Any
from pydantic import BaseSettings, Field, validator
import os
from pathlib import Path


class MonitoringConfig(BaseSettings):
    """Configuration for the monitoring system."""
    
    # Server configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    RELOAD: bool = False
    
    # Dashboard settings
    DASHBOARD_TITLE: str = "Trading System Monitor"
    DASHBOARD_THEME: str = "BOOTSTRAP"
    REFRESH_INTERVAL: int = 5000  # milliseconds
    
    # Metrics settings
    METRICS_PREFIX: str = "trading_bot_"
    DEFAULT_BUCKETS = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    
    # Paths
    STATIC_DIR: Path = Path(__file__).parent / "static"
    TEMPLATES_DIR: Path = Path(__file__).parent / "templates"
    
    # Authentication (set these in environment variables for production)
    AUTH_ENABLED: bool = False
    AUTH_USERNAME: str = "admin"
    AUTH_PASSWORD: str = ""
    
    # Alerting
    ALERT_THRESHOLDS: Dict[str, Any] = {
        "circuit_breaker_open": {
            "warning": 1,
            "critical": 3,
        },
        "event_processing_latency_p99": {
            "warning": 0.5,  # seconds
            "critical": 1.0,  # seconds
        },
        "queue_size": {
            "warning": 1000,
            "critical": 5000,
        },
    }
    
    @validator('STATIC_DIR', 'TEMPLATES_DIR')
    def ensure_dirs_exist(cls, v: Path) -> Path:
        """Ensure required directories exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v
    
    class Config:
        env_prefix = "MONITORING_"
        case_sensitive = False


# Create a singleton instance
config = MonitoringConfig()

__all__ = ["config"]
