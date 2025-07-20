"""
Monitoring System Configuration

This module contains configuration settings for the trading system monitoring.
"""
from typing import Dict, Any, List, Optional, Union, Literal
from pydantic import Field, validator, AnyHttpUrl, HttpUrl, field_validator, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict
import os
import json
from pathlib import Path
import secrets

class SecurityConfig(BaseSettings):
    """Security-related configuration."""
    # Authentication
    AUTH_ENABLED: bool = False
    API_KEYS: List[str] = ["default-insecure-key"]
    
    # Rate limiting
    RATE_LIMIT: str = "100/minute"
    
    # CORS
    BACKEND_CORS_ORIGINS: List[Union[str, AnyHttpUrl]] = [
        "http://localhost:8000",
        "http://localhost:3000",
    ]
    
    # Security headers
    SECURE_HEADERS: bool = True
    HSTS_ENABLED: bool = True
    HSTS_SECONDS: int = 3600  # 1 hour
    
    # Session
    SESSION_SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    SESSION_COOKIE_NAME: str = "trading_bot_session"
    SESSION_COOKIE_HTTPONLY: bool = True
    SESSION_COOKIE_SECURE: bool = False  # Set to True in production with HTTPS
    SESSION_COOKIE_SAMESITE: str = "lax"
    SESSION_LIFETIME: int = 3600 * 24 * 7  # 1 week
    
    # Password hashing
    PASSWORD_HASH_ALGORITHM: str = "bcrypt"
    PASSWORD_HASH_ITERATIONS: int = 12
    
    # Pydantic v2 model config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding='utf-8',
        env_prefix="SECURITY_",
        env_nested_delimiter="__",
        case_sensitive=True,
        extra='ignore'
    )
    
    @field_validator('BACKEND_CORS_ORIGINS', mode='before')
    @classmethod
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

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
    DEFAULT_BUCKETS: List[float] = [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
    
    # Paths
    STATIC_DIR: Path = Path(__file__).parent / "static"
    TEMPLATES_DIR: Path = Path(__file__).parent / "templates"
    
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
    
    # Nested configs
    SECURITY: SecurityConfig = Field(default_factory=SecurityConfig)
    
    # Pydantic v2 model config
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding='utf-8',
        env_prefix="MONITORING_",
        env_nested_delimiter="__",
        case_sensitive=False,
        extra='ignore',
        validate_default=True,
        json_schema_extra={
            "example": {
                "HOST": "0.0.0.0",
                "PORT": 8000,
                "DEBUG": False,
                "RELOAD": False,
                "DASHBOARD_TITLE": "Trading System Monitor",
                "DASHBOARD_THEME": "BOOTSTRAP",
                "REFRESH_INTERVAL": 5000,
                "METRICS_PREFIX": "trading_bot_"
            }
        }
    )
    
    @field_validator('DEFAULT_BUCKETS', mode='before')
    @classmethod
    def parse_buckets(cls, v: Union[str, List[float]]) -> List[float]:
        if isinstance(v, str):
            try:
                return [float(x.strip()) for x in v.split(",")]
            except (ValueError, AttributeError):
                pass
        return v
    
    @field_validator('ALERT_THRESHOLDS', mode='before')
    @classmethod
    def parse_alert_thresholds(cls, v: Union[str, Dict]) -> Dict:
        if isinstance(v, str):
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                pass
        return v
    
    @field_validator('STATIC_DIR', 'TEMPLATES_DIR', mode='after')
    @classmethod
    def ensure_dirs_exist(cls, v: Path) -> Path:
        v.mkdir(parents=True, exist_ok=True)
        return v

# Create singleton instances
config = MonitoringConfig()
security_config = config.SECURITY

__all__ = ["config", "security_config"]
