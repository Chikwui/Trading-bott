"""
A minimal version of the Order class for testing purposes.
"""
from enum import Enum, auto
from decimal import Decimal
from datetime import datetime

class OrderSide(str, Enum):
    """Order side (buy or sell)."""
    BUY = "buy"
    SELL = "sell"

class OrderType(str, Enum):
    """Order type (market, limit, etc.)."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class OrderStatus(str, Enum):
    """Order status."""
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class Order:
    """A minimal Order class for testing."""
    
    def __init__(
        self,
        symbol: str,
        side: OrderSide,
        type: OrderType,
        quantity: Decimal,
        price: Decimal = None,
        client_order_id: str = None,
        **kwargs
    ):
        self.symbol = symbol
        self.side = side
        self.type = type
        self.quantity = quantity
        self.price = price
        self.client_order_id = client_order_id or f"test_order_{datetime.now().timestamp()}"
        self.status = OrderStatus.NEW
        self.filled_quantity = Decimal("0.0")
        self.remaining_quantity = quantity
        self.average_filled_price = None
        self.timestamp = datetime.utcnow()
        self.last_updated = self.timestamp
        
        # Store any additional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __repr__(self):
        return (
            f"<Order symbol='{self.symbol}' side='{self.side}' type='{self.type}' "
            f"quantity={self.quantity} price={self.price} status='{self.status}'>"
        )
