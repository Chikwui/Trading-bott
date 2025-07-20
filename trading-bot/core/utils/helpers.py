"""Utility helper functions for the trading system."""
import logging
from typing import Any, Dict, Optional, Type, TypeVar

T = TypeVar('T')

def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance.
    
    Args:
        name: Name of the logger (usually __name__ of the calling module)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        # Configure basic logging if no handlers are configured
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    return logger

def validate_not_none(value: Any, name: str) -> None:
    """Validate that a value is not None.
    
    Args:
        value: Value to validate
        name: Name of the parameter for error message
        
    Raises:
        ValueError: If value is None
    """
    if value is None:
        raise ValueError(f"{name} cannot be None")

def validate_positive(value: float, name: str) -> None:
    """Validate that a numeric value is positive.
    
    Args:
        value: Value to validate
        name: Name of the parameter for error message
        
    Raises:
        ValueError: If value is not positive
    """
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_price(price: float, name: str = "Price") -> None:
    """Validate that a price is valid (positive and finite).
    
    Args:
        price: Price to validate
        name: Name of the price parameter for error message
        
    Raises:
        ValueError: If price is not valid
    """
    if not isinstance(price, (int, float)):
        raise ValueError(f"{name} must be a number, got {type(price).__name__}")
    if not float('-inf') < price < float('inf'):
        raise ValueError(f"{name} must be a finite number, got {price}")
    if price <= 0:
        raise ValueError(f"{name} must be positive, got {price}")


def validate_quantity(quantity: float, name: str = "Quantity") -> None:
    """Validate that a quantity is valid (positive and finite).
    
    Args:
        quantity: Quantity to validate
        name: Name of the quantity parameter for error message
        
    Raises:
        ValueError: If quantity is not valid
    """
    if not isinstance(quantity, (int, float)):
        raise ValueError(f"{name} must be a number, got {type(quantity).__name__}")
    if not float('-inf') < quantity < float('inf'):
        raise ValueError(f"{name} must be a finite number, got {quantity}")
    if quantity <= 0:
        raise ValueError(f"{name} must be positive, got {quantity}")

def to_decimal(value: Any, precision: int = 8) -> str:
    """Convert a value to a decimal string with specified precision.
    
    Args:
        value: Value to convert (int, float, Decimal, or string)
        precision: Number of decimal places to round to
        
    Returns:
        String representation of the decimal value
    """
    from decimal import Decimal, ROUND_HALF_UP
    
    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise ValueError("Cannot convert empty string to decimal")
    
    try:
        dec = Decimal(str(value))
        return str(round(dec, precision).quantize(
            Decimal('1e-{0}'.format(precision)),
            rounding=ROUND_HALF_UP
        ))
    except Exception as e:
        raise ValueError(f"Failed to convert {value} to decimal: {str(e)}")

def from_dict(data: Dict[str, Any], cls: Type[T]) -> T:
    """Convert a dictionary to a dataclass instance.
    
    Args:
        data: Dictionary containing field values
        cls: Dataclass type to instantiate
        
    Returns:
        Instance of the specified dataclass
    """
    import inspect
    from dataclasses import fields
    
    field_names = {f.name for f in fields(cls) if f.init}
    filtered_data = {k: v for k, v in data.items() if k in field_names}
    return cls(**filtered_data)

def to_dict(obj: Any) -> Dict[str, Any]:
    """Convert a dataclass instance to a dictionary.
    
    Args:
        obj: Dataclass instance to convert
        
    Returns:
        Dictionary representation of the dataclass
    """
    from dataclasses import asdict, is_dataclass
    
    if is_dataclass(obj):
        return asdict(obj)
    elif hasattr(obj, '__dict__'):
        return dict(obj.__dict__)
    else:
        raise ValueError(f"Cannot convert {type(obj).__name__} to dict")
