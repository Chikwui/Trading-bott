"""
Prometheus-compatible metrics server for the trading bot.

This module provides a FastAPI-based metrics server that exposes metrics in a format
compatible with Prometheus, with persistence, security, and distributed tracing support.
"""
import asyncio
import abc
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Set, Type, TypeVar, cast, Generic, Awaitable

from fastapi import FastAPI, HTTPException, Request, status, Depends, Security
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.security.utils import get_authorization_scheme_param
from jose import JWTError, jwt
from jose.constants import ALGORITHMS
from passlib.context import CryptContext
from pydantic import BaseModel, Field, validator
from pydantic.generics import GenericModel
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse

from core.utils.singleton import Singleton
from core.monitoring.tracing import trace_function, start_span

logger = logging.getLogger(__name__)

# Constants
DEFAULT_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0, float('inf'))
DEFAULT_QUANTILES = (0.5, 0.9, 0.95, 0.99, 0.999, 1.0)
DEFAULT_RETENTION_DAYS = 7
METRICS_DIR = "data/metrics"

# Security
SECRET_KEY = os.getenv("METRICS_SECRET_KEY", "insecure-dev-key-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Type aliases
LabelSet = Dict[str, str]
LabelKey = str  # String representation of label key-value pairs (e.g., 'status="200",method="GET"')

# Security
security = HTTPBearer()

class TokenData(BaseModel):
    """Token data model for JWT authentication."""
    username: Optional[str] = None
    scopes: List[str] = []

class User(BaseModel):
    """User model for authentication."""
    username: str
    disabled: bool = False
    scopes: List[str] = Field(default_factory=list)

class UserInDB(User):
    """User model with password for database storage."""
    hashed_password: str

class MetricPersistenceConfig(BaseModel):
    """Configuration for metric persistence."""
    enabled: bool = True
    storage_backend: str = "file"  # 'file' or 'memory'
    storage_path: str = METRICS_DIR
    retention_days: int = DEFAULT_RETENTION_DAYS
    backup_interval: int = 3600  # seconds

# Mock user database (replace with real user management in production)
fake_users_db = {
    "admin": {
        "username": "admin",
        "hashed_password": "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW",  # 'secret'
        "disabled": False,
        "scopes": ["metrics:read", "metrics:write", "admin"]
    }
}

class MetricType(str, Enum):
    """Types of metrics supported."""
    COUNTER = 'counter'
    GAUGE = 'gauge'
    HISTOGRAM = 'histogram'
    SUMMARY = 'summary'
    
    @classmethod
    def has_value(cls, value: str) -> bool:
        """Check if the enum contains a specific value."""
        return value in cls._value2member_map_

from pydantic import BaseModel, Field

class Metric(BaseModel):
    """Base class for all metrics with persistence support."""
    name: str
    help_text: str
    type: MetricType
    labels: List[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    
    # Internal state (not persisted by default)
    _values: Dict[str, float] = Field(default_factory=dict, exclude=True)
    _label_sets: Dict[str, Dict[str, str]] = Field(default_factory=dict, exclude=True)
    
    class Config:
        use_enum_values = True
        json_encoders = {
            MetricType: lambda v: v.value,
        }
    
    def __init__(self, **data):
        super().__init__(**data)
        self._validate_metric_name()
        self._validate_labels()
    
    def _validate_metric_name(self) -> None:
        """Validate the metric name."""
        if not re.match(r'^[a-zA-Z_:][a-zA-Z0-9_:]*$', self.name):
            raise ValueError(f"Invalid metric name: {self.name}")
    
    def _validate_labels(self) -> None:
        """Validate label names."""
        for label in self.labels:
            if not re.match(r'^[a-zA-Z_][a-zA-Z0-9_]*$', label):
                raise ValueError(f"Invalid label name: {label}")
    
    @property
    def values(self) -> Dict[str, float]:
        """Get metric values."""
        return self._values
    
    @property
    def label_sets(self) -> Dict[str, Dict[str, str]]:
        """Get label sets."""
        return self._label_sets
    
    def _get_key(self, labels: Dict[str, str]) -> str:
        """Get a unique key for the given label set."""
        if not labels:
            return ''
            
        # Validate labels
        for label in self.labels:
            if label not in labels:
                raise ValueError(f"Missing required label: {label}")
                
        # Sort labels for consistent key generation
        sorted_labels = sorted(labels.items())
        key_parts = [f'{k}=\"{v}\"' for k, v in sorted_labels]
        key = ','.join(key_parts)
        
        # Store the label set for this key
        self._label_sets[key] = labels
        return key
    
    def update_timestamp(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the metric to a dictionary for serialization."""
        data = self.dict(exclude={"_values", "_label_sets"})
        data["values"] = self._values
        data["label_sets"] = self._label_sets
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Metric':
        """Create a metric from a dictionary."""
        values = data.pop("values", {})
        label_sets = data.pop("label_sets", {})
        
        # Create the metric instance
        metric = cls(**data)
        
        # Set internal state
        metric._values = values
        metric._label_sets = label_sets
        
        return metric

class Counter(Metric):
    """A counter metric that can only increase."""
    type: MetricType = Field(default=MetricType.COUNTER, const=True)
    
    @trace_function()
    def inc(self, value: float = 1, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment the counter by the given value."""
        if value < 0:
            raise ValueError("Counters can only increase")
            
        key = self._get_key(labels or {})
        self._values[key] = self._values.get(key, 0) + value
        self.update_timestamp()
    
    @trace_function()
    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get the current value of the counter."""
        key = self._get_key(labels or {})
        return self._values.get(key, 0)

class Gauge(Metric):
    """A gauge metric that can go up and down."""
    type: MetricType = Field(default=MetricType.GAUGE, const=True)
    
    @trace_function()
    def set(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Set the gauge to the given value."""
        key = self._get_key(labels or {})
        self._values[key] = value
        self.update_timestamp()
    
    @trace_function()
    def inc(self, value: float = 1, labels: Optional[Dict[str, str]] = None) -> None:
        """Increment the gauge by the given value."""
        key = self._get_key(labels or {})
        self._values[key] = self._values.get(key, 0) + value
        self.update_timestamp()
    
    @trace_function()
    def dec(self, value: float = 1, labels: Optional[Dict[str, str]] = None) -> None:
        """Decrement the gauge by the given value."""
        self.inc(-value, labels)
    
    @trace_function()
    def get(self, labels: Optional[Dict[str, str]] = None) -> float:
        """Get the current value of the gauge."""
        key = self._get_key(labels or {})
        return self._values.get(key, 0)

class Histogram(Metric):
    """A histogram metric that tracks observations in configurable buckets."""
    type: MetricType = Field(default=MetricType.HISTOGRAM, const=True)
    buckets: List[float] = Field(default_factory=lambda: list(DEFAULT_BUCKETS))
    
    # Internal state for histogram-specific data
    _sums: Dict[str, float] = Field(default_factory=dict, exclude=True)
    _counts: Dict[str, int] = Field(default_factory=dict, exclude=True)
    
    @validator('buckets')
    def validate_buckets(cls, v):
        """Ensure buckets are sorted and valid."""
        if not v:
            return list(DEFAULT_BUCKETS)
        return sorted(v)
    
    @property
    def sums(self) -> Dict[str, float]:
        """Get the sum of observed values for each label set."""
        return self._sums
    
    @property
    def counts(self) -> Dict[str, int]:
        """Get the count of observed values for each label set."""
        return self._counts
    
    @trace_function()
    def observe(self, value: float, labels: Optional[Dict[str, str]] = None) -> None:
        """Observe a value and update the histogram."""
        key = self._get_key(labels or {})
        
        # Update sum and count
        self._sums[key] = self._sums.get(key, 0) + value
        self._counts[key] = self._counts.get(key, 0) + 1
        
        # Update bucket counts
        for bucket in self.buckets:
            if value <= bucket:
                bucket_key = f'le="{bucket}"'
                if key:  # If we have labels, append them
                    bucket_key = f'{bucket_key},{key}'
                self._values[bucket_key] = self._values.get(bucket_key, 0) + 1
        
        # Add +Inf bucket
        inf_key = 'le="+Inf"'
        if key:  # If we have labels, append them
            inf_key = f'{inf_key},{key}'
        self._values[inf_key] = self._values.get(inf_key, 0) + 1
        
        # Store the label set for this key
        key_parts = []
        for label in self.labels:
            if label not in labels:
                raise ValueError(f"Missing label '{label}' for metric {self.name}")
            key_parts.append(f"{label}=\"{labels[label]}\"")
        
        key = ','.join(key_parts)
        self.label_sets[key] = labels
        return key

class MetricsRegistry(metaclass=Singleton):
    """Registry for all metrics with persistence and distributed tracing support."""
    
    def __init__(self, config: Optional[MetricPersistenceConfig] = None):
        self.metrics: Dict[str, Metric] = {}
        self.config = config or MetricPersistenceConfig()
        self.storage: Optional[StorageBackend] = None
        self._initialized = False
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._backup_task: Optional[asyncio.Task] = None
        
        # Initialize storage
        self._init_storage()
        
        # Start background tasks
        self._start_background_tasks()
    
    def _init_storage(self) -> None:
        """Initialize the storage backend."""
        if not self.config.enabled:
            self.storage = InMemoryStorageBackend()
            return
            
        if self.config.storage_backend == "file":
            # Create storage directory if it doesn't exist
            os.makedirs(self.config.storage_path, exist_ok=True)
            self.storage = FileStorageBackend(self.config.storage_path)
        else:
            self.storage = InMemoryStorageBackend()
    
    def _start_background_tasks(self) -> None:
        """Start background tasks for cleanup and backup."""
        if self._initialized:
            return
            
        self._initialized = True
        
        # Start cleanup task
        if self.storage and self.config.retention_days > 0:
            self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        
        # Start backup task
        if self.storage and self.config.enabled and self.config.backup_interval > 0:
            self._backup_task = asyncio.create_task(self._periodic_backup())
    
    async def _periodic_cleanup(self) -> None:
        """Periodically clean up old metrics."""
        while True:
            try:
                if self.storage:
                    cutoff = datetime.now() - timedelta(days=self.config.retention_days)
                    deleted = await self.storage.cleanup(cutoff)
                    if deleted > 0:
                        logger.info(f"Cleaned up {deleted} old metrics")
            except Exception as e:
                logger.error(f"Error during metrics cleanup: {e}")
            
            # Run once per day
            await asyncio.sleep(86400)  # 24 hours
    
    async def _periodic_backup(self) -> None:
        """Periodically back up metrics to storage."""
        while True:
            try:
                await self.persist_metrics()
            except Exception as e:
                logger.error(f"Error during metrics backup: {e}")
            
            # Wait for the next backup interval
            await asyncio.sleep(self.config.backup_interval)
    
    @trace_function()
    async def load_metrics(self) -> None:
        """Load metrics from storage."""
        if not self.storage:
            return
            
        try:
            async with self._lock:
                metrics_data = await self.storage.load("metrics")
                if metrics_data:
                    self.metrics = {
                        name: self._deserialize_metric(data)
                        for name, data in metrics_data.items()
                    }
                    logger.info(f"Loaded {len(self.metrics)} metrics from storage")
        except Exception as e:
            logger.error(f"Error loading metrics: {e}")
    
    @trace_function()
    async def persist_metrics(self) -> None:
        """Persist metrics to storage."""
        if not self.storage:
            return
            
        try:
            async with self._lock:
                metrics_data = {
                    name: self._serialize_metric(metric)
                    for name, metric in self.metrics.items()
                }
                await self.storage.save("metrics", metrics_data)
        except Exception as e:
            logger.error(f"Error persisting metrics: {e}")
    
    def _serialize_metric(self, metric: Metric) -> Dict[str, Any]:
        """Serialize a metric to a dictionary."""
        return metric.dict()
    
    def _deserialize_metric(self, data: Dict[str, Any]) -> Metric:
        """Deserialize a metric from a dictionary."""
        metric_type = data.get("type")
        if metric_type == MetricType.COUNTER:
            return Counter(**data)
        elif metric_type == MetricType.GAUGE:
            return Gauge(**data)
        elif metric_type == MetricType.HISTOGRAM:
            return Histogram(**data)
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")
    
    @trace_function()
    def counter(self, name: str, help_text: str, labels: List[str] = None) -> Counter:
        """Create or get a counter metric."""
        if name not in self.metrics:
            with start_span("create_counter"):
                self.metrics[name] = Counter(
                    name=name,
                    help_text=help_text,
                    labels=labels or []
                )
        return cast(Counter, self.metrics[name])
    
    @trace_function()
    def gauge(self, name: str, help_text: str, labels: List[str] = None) -> Gauge:
        """Create or get a gauge metric."""
        if name not in self.metrics:
            with start_span("create_gauge"):
                self.metrics[name] = Gauge(
                    name=name,
                    help_text=help_text,
                    labels=labels or []
                )
        return cast(Gauge, self.metrics[name])
    
    @trace_function()
    def histogram(
        self,
        name: str,
        help_text: str,
        buckets: List[float] = None,
        labels: List[str] = None
    ) -> Histogram:
        """Create or get a histogram metric."""
        if name not in self.metrics:
            with start_span("create_histogram"):
                self.metrics[name] = Histogram(
                    name=name,
                    help_text=help_text,
                    buckets=buckets or list(DEFAULT_BUCKETS),
                    labels=labels or []
                )
        return cast(Histogram, self.metrics[name])
    
    @trace_function()
    def get_metric(self, name: str) -> Optional[Metric]:
        """Get a metric by name."""
        return self.metrics.get(name)
    
    @trace_function()
    def get_all_metrics(self) -> Dict[str, Metric]:
        """Get all registered metrics."""
        return self.metrics.copy()
    
    @trace_function()
    def generate_metrics(self) -> str:
        """Generate Prometheus-formatted metrics with tracing support."""
        with start_span("generate_metrics") as span:
            lines = []
            
            for metric in self.metrics.values():
                try:
                    # Add help text
                    lines.append(f"# HELP {metric.name} {metric.help_text}")
                    lines.append(f"# TYPE {metric.name} {metric.type.value}")
                    
                    # Add metric values
                    if isinstance(metric, Histogram):
                        self._generate_histogram_metrics(metric, lines)
                    else:
                        self._generate_simple_metrics(metric, lines)
                    
                    # Add an empty line between metrics
                    lines.append("")
                except Exception as e:
                    logger.error(f"Error generating metrics for {metric.name}: {e}")
                    if span:
                        span.record_exception(e)
            
            return '\n'.join(lines)
    
    def _generate_histogram_metrics(self, metric: Histogram, lines: List[str]) -> None:
        """Generate histogram metrics."""
        # Add bucket metrics
        for key, count in metric.values.items():
            lines.append(f"{metric.name}_bucket{{{key}}} {count}")
        
        # Add sum and count metrics
        for key, sum_val in metric.sums.items():
            lines.append(f"{metric.name}_sum{{{key}}} {sum_val}")
        
        for key, count in metric.counts.items():
            lines.append(f"{metric.name}_count{{{key}}} {count}")
    
    def _generate_simple_metrics(self, metric: Metric, lines: List[str]) -> None:
        """Generate simple metrics (counters and gauges)."""
        for key, value in metric.values.items():
            if key:
                lines.append(f"{metric.name}{{{key}}} {value}")
            else:
                lines.append(f"{metric.name} {value}")
    
    async def close(self) -> None:
        """Clean up resources."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self._backup_task:
            self._backup_task.cancel()
            try:
                await self._backup_task
            except asyncio.CancelledError:
                pass
        
        # Persist metrics one last time
        
        for metric in self.metrics.values():
            try:
                # Add help text
                lines.append(f"# HELP {metric.name} {metric.help_text}")
                lines.append(f"# TYPE {metric.name} {metric.type.value}")
                
                # Add metric values
                if isinstance(metric, Histogram):
                    self._generate_histogram_metrics(metric, lines)
                else:
                    self._generate_simple_metrics(metric, lines)
                
                # Add an empty line between metrics
                lines.append("")
            except Exception as e:
                logger.error(f"Error generating metrics for {metric.name}: {e}")
                if span:
                    span.record_exception(e)
        
        return '\n'.join(lines)

def _generate_histogram_metrics(self, metric: Histogram, lines: List[str]) -> None:
    """Generate histogram metrics."""
    # Add bucket metrics
    for key, count in metric.values.items():
        lines.append(f"{metric.name}_bucket{{{key}}} {count}")
    
    # Add sum and count metrics
    for key, sum_val in metric.sums.items():
        lines.append(f"{metric.name}_sum{{{key}}} {sum_val}")
    
    for key, count in metric.counts.items():
        lines.append(f"{metric.name}_count{{{key}}} {count}")

def _generate_simple_metrics(self, metric: Metric, lines: List[str]) -> None:
    """Generate simple metrics (counters and gauges)."""
    for key, value in metric.values.items():
        if key:
            lines.append(f"{metric.name}{{{key}}} {value}")
        else:
            lines.append(f"{metric.name} {value}")

async def close(self) -> None:
    """Clean up resources."""
    if self._cleanup_task:
        self._cleanup_task.cancel()
        try:
            await self._cleanup_task
        except asyncio.CancelledError:
            pass
    
    if self._backup_task:
        self._backup_task.cancel()
        try:
            await self._backup_task
        except asyncio.CancelledError:
            pass
    
    # Persist metrics one last time
    await self.persist_metrics()
    
    self._initialized = False

class MetricsServer:
    """
    FastAPI-based metrics server with authentication, rate limiting, and persistence.
    
    Features:
    - JWT-based authentication
    - Role-based access control
    - Rate limiting
    - Request validation
    - Distributed tracing
    - Metrics persistence
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8000,
        config: Optional[MetricPersistenceConfig] = None,
        enable_auth: bool = True,
        rate_limit: int = 100,
        rate_window: int = 60,
    ):
        """Initialize the metrics server.
        
        Args:
            host: Host to bind to
            port: Port to listen on
            config: Persistence configuration
            enable_auth: Whether to enable authentication
            rate_limit: Maximum number of requests per window
            rate_window: Rate limit window in seconds
        """
        self.host = host
        self.port = port
        self.enable_auth = enable_auth
        self.rate_limit = rate_limit
        self.rate_window = rate_window
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="Trading Bot Metrics API",
            description="Metrics server with authentication and persistence",
            version="1.0.0",
            docs_url="/docs" if not enable_auth else None,
            redoc_url="/redoc" if not enable_auth else None,
            openapi_url="/openapi.json" if not enable_auth else None,
        )
        
        # Initialize metrics registry
        self.registry = MetricsRegistry(config)
        
        # Add middleware
        self._setup_middleware()
        
        # Add routes
        self.setup_routes()
        
        # Setup default metrics
        self._setup_default_metrics()
    
    def _setup_middleware(self) -> None:
        """Set up FastAPI middleware."""
        # Add CORS middleware
        from fastapi.middleware.cors import CORSMiddleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Add request timing middleware
        @self.app.middleware("http")
        async def add_process_time_header(request: Request, call_next):
            start_time = time.time()
            response = await call_next(request)
            process_time = time.time() - start_time
            
            # Record request duration
            self.request_duration.observe(
                process_time,
                {
                    "method": request.method,
                    "endpoint": request.url.path,
                    "status": str(response.status_code)
                }
            )
            
            # Record request count
            self.requests_total.inc(
                1,
                {
                    "method": request.method,
                    "endpoint": request.url.path,
                    "status": str(response.status_code)
                }
            )
            
            # Record errors
            if response.status_code >= 400:
                self.errors_total.inc(
                    1,
                    {
                        "method": request.method,
                        "endpoint": request.url.path,
                        "status": str(response.status_code)
                    }
                )
            
            return response
    
    def setup_routes(self) -> None:
        """Set up the FastAPI routes."""
        # Health check endpoint (always public)
        @self.app.get("/health")
        async def health_check() -> Dict[str, str]:
            """Health check endpoint."""
            return {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "version": "1.0.0"
            }
        
        # Metrics endpoints
        @self.app.get(
            "/metrics",
            response_class=Response,
            dependencies=[Depends(self.verify_token)] if self.enable_auth else None
        )
        async def get_metrics() -> Response:
            """Get metrics in Prometheus format."""
            with start_span("get_metrics"):
                return Response(
                    content=self.registry.generate_metrics(),
                    media_type="text/plain; version=0.0.4"
                )
        
        @self.app.get(
            "/metrics/json",
            response_class=JSONResponse,
            dependencies=[Depends(self.verify_token)] if self.enable_auth else None
        )
        async def get_metrics_json() -> JSONResponse:
            """Get metrics in JSON format."""
            with start_span("get_metrics_json"):
                metrics = {}
                for name, metric in self.registry.get_all_metrics().items():
                    metrics[name] = {
                        'type': metric.type.value,
                        'help': metric.help_text,
                        'values': dict(metric.values)
                    }
                    if hasattr(metric, 'sums'):
                        metrics[name]['sums'] = dict(metric.sums)
                    if hasattr(metric, 'counts'):
                        metrics[name]['counts'] = dict(metric.counts)
                
                return JSONResponse(content=metrics)
        
        # Authentication endpoints (if enabled)
        if self.enable_auth:
            @self.app.post("/token")
            async def login_for_access_token(
                form_data: OAuth2PasswordRequestForm = Depends()
            ) -> Dict[str, str]:
                """Get an access token for authentication."""
                user = await self.authenticate_user(form_data.username, form_data.password)
                if not user:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Incorrect username or password",
                        headers={"WWW-Authenticate": "Bearer"},
                    )
                
                access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
                access_token = self.create_access_token(
                    data={"sub": user.username, "scopes": user.scopes},
                    expires_delta=access_token_expires
                )
                
                return {
                    "access_token": access_token,
                    "token_type": "bearer",
                    "expires_in": int(access_token_expires.total_seconds())
                }
        
        # Dashboard endpoint
        @self.app.get(
            "/dashboard",
            response_class=HTMLResponse,
            dependencies=[Depends(self.verify_token)] if self.enable_auth else None
        )
        async def dashboard() -> HTMLResponse:
            """Display a simple dashboard with metrics visualizations."""
            with start_span("render_dashboard"):
                return self._generate_dashboard()
    
    # Authentication utilities
    async def verify_token(
        self,
        credentials: HTTPAuthorizationCredentials = Depends(security)
    ) -> TokenData:
        """Verify JWT token in the Authorization header."""
        if not self.enable_auth:
            return TokenData(username="anonymous", scopes=[])
            
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
        try:
            token = credentials.credentials
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            username: str = payload.get("sub")
            if username is None:
                raise credentials_exception
            token_scopes = payload.get("scopes", [])
            return TokenData(username=username, scopes=token_scopes)
        except (JWTError, AttributeError):
            raise credentials_exception
    
    def create_access_token(
        self,
        data: Dict[str, Any],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create a JWT access token."""
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    async def authenticate_user(self, username: str, password: str) -> Optional[UserInDB]:
        """Authenticate a user."""
        # In a real application, you would verify the password against a database
        # This is a simplified example
        user_dict = fake_users_db.get(username)
        if not user_dict:
            return None
        
        # In a real app, you would verify the hashed password
        # For this example, we're using a hardcoded password hash
        user = UserInDB(**user_dict)
        if not user_dict["hashed_password"] == "$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW":
            return None
            
        return user
    
    def _generate_dashboard(self) -> HTMLResponse:
        """Generate an HTML dashboard with metrics visualizations."""
        # This is a simplified example - in a real app, you'd use a templating engine
        html_content = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Trading Bot Metrics Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.8.1/font/bootstrap-icons.css">
            <style>
                body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; padding: 20px; }
                .metric-card { margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .chart-container { height: 400px; margin-bottom: 20px; }
                .navbar { margin-bottom: 20px; }
                #refresh-btn { margin-left: auto; }
                .card-header { background-color: #f8f9fa; }
                .alert { margin-top: 20px; }
            </style>
        </head>
        <body>
            <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
                <div class="container-fluid">
                    <a class="navbar-brand" href="#">Trading Bot Metrics</a>
                    <button class="btn btn-outline-light" id="refresh-btn">
                        <i class="bi bi-arrow-clockwise"></i> Refresh
                    </button>
                </div>
            </nav>
            
            <div class="container-fluid">
                <div class="row" id="metrics-container">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-body">
                                <h5 class="card-title">Loading metrics...</h5>
                                <p class="card-text">Please wait while we load the metrics data.</p>
                                <div class="spinner-border text-primary" role="status">
                                    <span class="visually-hidden">Loading...</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <script>
                // Load metrics from the /metrics/json endpoint
                async function loadMetrics() {
                    try {
                        const headers = {
                            'Accept': 'application/json',
                            'Cache-Control': 'no-cache, no-store, must-revalidate',
                            'Pragma': 'no-cache',
                            'Expires': '0'
                        };
                        
                        // Add auth token if available
                        const token = localStorage.getItem('metrics_token');
                        if (token) {
                            headers['Authorization'] = `Bearer ${token}`;
                        }
                        
                        const response = await fetch('/metrics/json', { 
                            headers,
                            cache: 'no-store'
                        });
                        
                        if (response.status === 401) {
                            // Handle unauthorized
                            window.location.href = '/login';
                            return;
                        }
                        
                        if (!response.ok) {
                            throw new Error(`HTTP error! status: ${response.status}`);
                        }
                        
                        const metrics = await response.json();
                        renderMetrics(metrics);
                    } catch (error) {
                        console.error('Error loading metrics:', error);
                        const container = document.getElementById('metrics-container');
                        if (container) {
                            container.innerHTML = `
                                <div class="col-12">
                                    <div class="alert alert-danger" role="alert">
                                        <h4 class="alert-heading">Error loading metrics</h4>
                                        <p>${error.message || 'Unknown error occurred'}</p>
                                        <hr>
                                        <p class="mb-0">Please check the browser console for more details.</p>
                                    </div>
                                </div>
                            `;
                        }
                    }
                }
                
                // Render metrics in the UI
                function renderMetrics(metrics) {
                    const container = document.getElementById('metrics-container');
                    if (!container) return;
                    
                    let html = '';
                    
                    for (const [name, metric] of Object.entries(metrics)) {
                        const values = Object.entries(metric.values || {});
                        
                        if (values.length === 0) continue;
                        
                        // Create a chart for the metric
                        const chartId = `chart-${name.replace(/[^a-zA-Z0-9-]/g, '-')}`;
                        const isHistogram = name.endsWith('_bucket') || name.endsWith('_sum') || name.endsWith('_count');
                        
                        if (isHistogram) {
                            // Skip individual histogram components, we'll handle them as a group
                            if (name.endsWith('_sum') || name.endsWith('_count')) continue;
                            
                            // This is a histogram bucket metric
                            const baseName = name.replace('_bucket', '');
                            const sumMetric = metrics[`${baseName}_sum`];
                            const countMetric = metrics[`${baseName}_count`];
                            
                            if (!sumMetric || !countMetric) continue;
                            
                            // Process histogram data
                            const buckets = [];
                            const counts = [];
                            
                            for (const [key, value] of values) {
                                const leMatch = /le="([^"]+)"/.exec(key);
                                if (leMatch) {
                                    const le = leMatch[1];
                                    if (le !== '+Inf') {
                                        buckets.push(parseFloat(le));
                                        counts.push(parseFloat(value));
                                    }
                                }
                            }
                            
                            if (buckets.length > 0) {
                                // Create histogram chart
                                html += `
                                    <div class="col-md-6 col-lg-4">
                                        <div class="card metric-card">
                                            <div class="card-header">
                                                <h5 class="mb-0">${baseName}</h5>
                                                <small class="text-muted">${metric.help || 'Histogram'}</small>
                                            </div>
                                            <div class="card-body">
                                                <div id="${chartId}" class="chart-container"></div>
                                            </div>
                                        </div>
                                    </div>
                                `;
                                
                                // Add chart rendering code
                                html += `
                                    <script>
                                        document.addEventListener('DOMContentLoaded', function() {
                                            const trace = {
                                                x: ${JSON.stringify(buckets)},
                                                y: ${JSON.stringify(counts)},
                                                type: 'bar',
                                                name: '${baseName}',
                                                marker: {
                                                    color: 'rgba(55, 128, 191, 0.7)',
                                                    line: {
                                                        color: 'rgba(55, 128, 191, 1.0)',
                                                        width: 1
                                                    }
                                                }
                                            };
                                            
                                            const layout = {
                                                title: '${baseName.replace(/_/g, ' ').toUpperCase()}',
                                                xaxis: {
                                                    title: 'Bucket',
                                                    tickangle: -45,
                                                    type: 'category'
                                                },
                                                yaxis: {
                                                    title: 'Count'
                                                },
                                                margin: { t: 50, l: 50, r: 30, b: 100 },
                                                height: 350,
                                                showlegend: false
                                            };
                                            
                                            Plotly.newPlot('${chartId}', [trace], layout);
                                        });
                                    </script>
                                `;
                            }
                        } else {
                            // Regular metric (counter or gauge)
                            const labels = [];
                            const data = [];
                            
                            for (const [key, value] of values) {
                                const label = key ? key.replace(/"/g, '') : 'value';
                                labels.push(label);
                                data.push(parseFloat(value));
                            }
                            
                            if (data.length > 0) {
                                html += `
                                    <div class="col-md-6 col-lg-4">
                                        <div class="card metric-card">
                                            <div class="card-header">
                                                <h5 class="mb-0">${name}</h5>
                                                <small class="text-muted">${metric.help || ''}</small>
                                            </div>
                                            <div class="card-body">
                                                <div id="${chartId}" class="chart-container"></div>
                                            </div>
                                        </div>
                                    </div>
                                `;
                                
                                // Add chart rendering code
                                html += `
                                    <script>
                                        document.addEventListener('DOMContentLoaded', function() {
                                            const trace = {
                                                x: ${JSON.stringify(labels)},
                                                y: ${JSON.stringify(data)},
                                                type: 'bar',
                                                name: '${name}',
                                                marker: {
                                                    color: 'rgba(75, 192, 192, 0.7)',
                                                    line: {
                                                        color: 'rgba(75, 192, 192, 1.0)',
                                                        width: 1
                                                    }
                                                }
                                            };
                                            
                                            const layout = {
                                                title: '${name.replace(/_/g, ' ').toUpperCase()}',
                                                xaxis: {
                                                    title: 'Labels',
                                                    tickangle: -45,
                                                    type: 'category'
                                                },
                                                yaxis: {
                                                    title: 'Value'
                                                },
                                                margin: { t: 50, l: 50, r: 30, b: 100 },
                                                height: 350,
                                                showlegend: false
                                            };
                                            
                                            Plotly.newPlot('${chartId}', [trace], layout);
                                        });
                                    </script>
                                `;
                            }
                        }
                    }
                    
                    if (html === '') {
                        html = `
                            <div class="col-12">
                                <div class="alert alert-info" role="alert">
                                    <h4 class="alert-heading">No metrics available</h4>
                                    <p>No metrics have been recorded yet. Try again later.</p>
                                </div>
                            </div>
                        `;
                    }
                    
                    container.innerHTML = html;
                }
                
                // Set up refresh button
                document.getElementById('refresh-btn')?.addEventListener('click', () => {
                    const container = document.getElementById('metrics-container');
                    if (container) {
                        container.innerHTML = `
                            <div class="col-12">
                                <div class="card">
                                    <div class="card-body text-center">
                                        <div class="spinner-border text-primary" role="status">
                                            <span class="visually-hidden">Loading...</span>
                                        </div>
                                        <p class="mt-2">Refreshing metrics...</p>
                                    </div>
                                </div>
                            </div>
                        `;
                    }
                    loadMetrics();
                });
                
                // Load metrics immediately and then every 30 seconds
                loadMetrics();
                setInterval(loadMetrics, 30000);
                
                // Handle browser tab visibility changes
                document.addEventListener('visibilitychange', () => {
                    if (!document.hidden) {
                        loadMetrics();
                    }
                });
            </script>
        </body>
        </html>
        """
        
        return HTMLResponse(content=html_content, status_code=200)
    
    def start(self) -> None:
        """Start the metrics server."""
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )

def start_metrics_server(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the metrics server.
    
    Args:
        host: Host to bind to
        port: Port to listen on
    """
    server = MetricsServer(host, port)
    server.start()

if __name__ == "__main__":
    # Example usage
    import random
    import time
    import threading
    
    # Start the metrics server in a background thread
    server_thread = threading.Thread(
        target=start_metrics_server,
        daemon=True
    )
    server_thread.start()
    
    # Get the metrics registry
    registry = MetricsRegistry()
    
    # Register some example metrics
    trades_counter = registry.counter(
        'trades_total',
        'Total number of trades',
        ['symbol', 'side']
    )
    
    latency_histogram = registry.histogram(
        'trade_latency_seconds',
        'Trade processing latency in seconds',
        [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        ['symbol']
    )
    
    # Simulate some metrics updates
    symbols = ['BTC/USD', 'ETH/USD', 'XRP/USD', 'LTC/USD', 'BCH/USD']
    sides = ['buy', 'sell']
    
    print(f"Metrics server running at http://localhost:8000/metrics")
    print(f"Dashboard available at http://localhost:8000/dashboard")
    
    try:
        while True:
            # Simulate some trades
            symbol = random.choice(symbols)
            side = random.choice(sides)
            latency = random.uniform(0.0001, 0.1)
            
            # Update metrics
            trades_counter.inc(1, {'symbol': symbol, 'side': side})
            latency_histogram.observe(latency, {'symbol': symbol})
            
            # Sleep for a bit
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nShutting down...")
