"""
Instrument metadata and definitions for trading.
"""
from dataclasses import dataclass
from datetime import time
from typing import Optional, List, Dict, Any


@dataclass
class InstrumentMetadata:
    """Metadata for a financial instrument."""
    symbol: str
    asset_class: str
    exchange: str
    min_tick_size: float
    min_order_size: float
    max_leverage: int
    trading_hours: str
    description: str = ""
    is_active: bool = True
    lot_size: float = 1.0
    margin_required: float = 0.0
    commission: float = 0.0
    price_precision: int = 5
    quantity_precision: int = 2
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'asset_class': self.asset_class,
            'exchange': self.exchange,
            'min_tick_size': self.min_tick_size,
            'min_order_size': self.min_order_size,
            'max_leverage': self.max_leverage,
            'trading_hours': self.trading_hours,
            'description': self.description,
            'is_active': self.is_active,
            'lot_size': self.lot_size,
            'margin_required': self.margin_required,
            'commission': self.commission,
            'price_precision': self.price_precision,
            'quantity_precision': self.quantity_precision
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'InstrumentMetadata':
        """Create from dictionary."""
        return cls(**data)
    
    def __str__(self) -> str:
        return f"{self.symbol} ({self.asset_class}) on {self.exchange}"
