"""
Security middleware and utilities for the monitoring system.
"""
import time
from typing import Optional, List, Dict, Any, Callable
from functools import wraps
from fastapi import Request, HTTPException, status
from fastapi.security import APIKeyHeader
from fastapi.security.utils import get_authorization_scheme_param
from jose import JWTError, jwt
from jose.exceptions import JWTClaimsError, ExpiredSignatureError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from .config import config

# Rate limiter instance
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[config.RATE_LIMIT]
)

# API Key Header
api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
    description="API key for authentication"
)

class SecurityError(Exception):
    """Base class for security-related errors."""
    pass

class RateLimitError(SecurityError):
    """Raised when rate limit is exceeded."""
    pass

class AuthenticationError(SecurityError):
    """Raised when authentication fails."""
    pass

def get_api_key(request: Request) -> Optional[str]:
    """Extract API key from request headers."""
    api_key = request.headers.get("X-API-Key")
    if not api_key:
        # Check Authorization header as fallback
        auth = request.headers.get("Authorization")
        if auth:
            scheme, param = get_authorization_scheme_param(auth)
            if scheme.lower() == "bearer":
                api_key = param
    return api_key

async def verify_api_key(api_key: str) -> bool:
    """Verify if the provided API key is valid."""
    if not config.AUTH_ENABLED:
        return True
        
    if not api_key:
        return False
        
    return api_key in config.API_KEYS

def api_key_required(func: Callable):
    """Decorator to enforce API key authentication."""
    @wraps(func)
    async def wrapper(*args, **kwargs):
        if not config.AUTH_ENABLED:
            return await func(*args, **kwargs)
            
        request = kwargs.get('request')
        if not request:
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break
                    
        if not request:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )
            
        api_key = get_api_key(request)
        if not api_key or not await verify_api_key(api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        return await func(*args, **kwargs)
    return wrapper

def setup_security(app):
    """Configure security middleware and handlers."""
    # Rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
    # Security headers
    @app.middleware("http")
    async def add_security_headers(request: Request, call_next):
        response = await call_next(request)
        
        # Security headers
        headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
        }
        
        # Add HSTS if enabled
        if config.SECURE_HEADERS and config.HSTS_ENABLED:
            headers["Strict-Transport-Security"] = f"max-age={config.HSTS_SECONDS}; includeSubDomains"
        
        # Add headers to response
        for key, value in headers.items():
            response.headers[key] = value
            
        return response
