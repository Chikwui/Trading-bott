"""Core exceptions for the trading system."""


class TradingError(Exception):
    """Base exception for all trading-related errors."""
    pass


class OrderError(TradingError):
    """Raised when there is an error with order operations."""
    pass


class OCOOrderError(OrderError):
    """Raised when there is an error with OCO orders."""
    pass


class OrderNotFoundError(OrderError):
    """Raised when an order is not found."""
    pass


class OrderValidationError(OrderError):
    """Raised when order validation fails."""
    pass


class RiskCheckFailed(OrderError):
    """Raised when a risk check fails during order validation."""
    pass


class ExchangeError(TradingError):
    """Raised for exchange-related errors."""
    pass


class RateLimitError(ExchangeError):
    """Raised when rate limits are exceeded."""
    pass


class InsufficientFundsError(ExchangeError):
    """Raised when there are insufficient funds for an operation."""
    pass


class ConnectionError(ExchangeError):
    """Raised when there are connection issues with the exchange."""
    pass
