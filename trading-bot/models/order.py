"""
Order-related data models.
"""
from datetime import datetime
from enum import Enum, auto
from typing import Optional, List, Dict, Any
from decimal import Decimal
from .base import BaseModel

class OrderType(str, Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TAKE_PROFIT = "take_profit"
    TAKE_PROFIT_LIMIT = "take_profit_limit"
    TRAILING_STOP = "trailing_stop"
    FOK = "fok"  # Fill or Kill
    IOC = "ioc"  # Immediate or Cancel


class OrderSide(str, Enum):
    """Order sides."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    """Order statuses."""
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    PENDING_CANCEL = "pending_cancel"
    SUSPENDED = "suspended"
    CALCULATED = "calculated"  # For conditional orders
    
    @classmethod
    def is_terminal(cls, status: 'OrderStatus') -> bool:
        """Check if the status is terminal (no further changes)."""
        return status in [
            cls.FILLED,
            cls.CANCELED,
            cls.REJECTED,
            cls.EXPIRED
        ]


class Order(BaseModel):
    """Order model representing a trading order."""
    
    id: str
    client_order_id: Optional[str] = None
    symbol: str
    type: OrderType
    side: OrderSide
    status: OrderStatus = OrderStatus.NEW
    
    # Order amounts
    amount: float  # Base currency amount
    price: Optional[float] = None  # Price per unit
    stop_price: Optional[float] = None  # For stop orders
    average: Optional[float] = None  # Average filled price
    filled: float = 0.0  # Filled amount
    remaining: Optional[float] = None  # Remaining amount to be filled
    cost: Optional[float] = None  # Total order cost (quote currency)
    
    # Fees
    fee: Optional[Dict[str, Any]] = None  # {'currency': 'USDT', 'cost': 0.1, 'rate': 0.001}
    
    # Timestamps
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    
    # Additional info
    info: Dict[str, Any] = {}
    
    @property
    def is_open(self) -> bool:
        """Check if the order is still open."""
        return self.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]
    
    @property
    def is_closed(self) -> bool:
        """Check if the order is closed."""
        return not self.is_open
    
    @property
    def is_filled(self) -> bool:
        """Check if the order is completely filled."""
        return self.status == OrderStatus.FILLED
    
    @property
    def is_canceled(self) -> bool:
        """Check if the order was canceled."""
        return self.status == OrderStatus.CANCELED
    
    @property
    def is_rejected(self) -> bool:
        """Check if the order was rejected."""
        return self.status == OrderStatus.REJECTED
    
    @property
    def is_expired(self) -> bool:
        """Check if the order expired."""
        return self.status == OrderStatus.EXPIRED
    
    @property
    def filled_value(self) -> float:
        """Get the total value of the filled portion in quote currency."""
        if self.filled == 0:
            return 0.0
            
        if self.average is not None:
            return self.filled * self.average
        elif self.price is not None:
            return self.filled * self.price
        else:
            return 0.0
    
    @property
    def remaining_value(self) -> float:
        """Get the total value of the remaining portion in quote currency."""
        remaining = self.remaining if self.remaining is not None else (self.amount - self.filled)
        if remaining <= 0 or self.price is None:
            return 0.0
        return remaining * self.price
    
    def update(self, update: 'Order') -> None:
        """Update order with new data."""
        for field in [
            'status', 'amount', 'price', 'stop_price', 'average', 
            'filled', 'remaining', 'cost', 'fee', 'updated_at', 'closed_at', 'info'
        ]:
            if hasattr(update, field) and getattr(update, field) is not None:
                setattr(self, field, getattr(update, field))
