"""
Advanced order state machine and order management system.
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum, auto
from typing import (
    Any, Callable, ClassVar, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union
)

from ..market.instrument import Instrument
from ..utils.helpers import get_logger

logger = get_logger(__name__)

class OrderStatus(Enum):
    """Order status enumeration with support for advanced order types."""
    # Initial states
    PENDING_NEW = "PENDING_NEW"          # Order received by system, not yet sent to exchange
    PENDING_ACTIVATION = "PENDING_ACTIVATION"  # Conditional order waiting for activation
    
    # Active states
    NEW = "NEW"                          # Order accepted by exchange
    PARTIALLY_FILLED = "PARTIALLY_FILLED"  # Partially filled
    
    # Pending states (awaiting response)
    PENDING_CANCEL = "PENDING_CANCEL"        # Request to cancel received
    PENDING_REPLACE = "PENDING_REPLACE"      # Request to modify received
    PENDING_ACTIVATE = "PENDING_ACTIVATE"    # Request to activate conditional order
    
    # Terminal states
    FILLED = "FILLED"                  # Order completely filled
    CANCELED = "CANCELED"              # Order canceled by user
    REJECTED = "REJECTED"              # Order rejected by exchange
    EXPIRED = "EXPIRED"                # Order expired (time in force)
    SUSPENDED = "SUSPENDED"            # Order suspended by exchange
    TRIGGERED = "TRIGGERED"            # Conditional order triggered
    
    # Synthetic states (not from exchange)
    ERROR = "ERROR"                    # Error processing order
    CANCELED_SIBLING = "CANCELED_SIBLING"  # Canceled due to sibling order execution (OCO)
    
    def is_active(self) -> bool:
        """Check if the order is in an active state."""
        return self in {
            OrderStatus.NEW,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.PENDING_CANCEL,
            OrderStatus.PENDING_REPLACE
        }
    
    def is_terminal(self) -> bool:
        """Check if the order is in a terminal state."""
        return self in {
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED,
            OrderStatus.SUSPENDED,
            OrderStatus.ERROR
        }
    
    def is_pending(self) -> bool:
        """Check if the order is in a pending state."""
        return self in {
            OrderStatus.PENDING_NEW,
            OrderStatus.PENDING_CANCEL,
            OrderStatus.PENDING_REPLACE
        }

class OrderSide(Enum):
    """Order side."""
    BUY = "BUY"
    SELL = "SELL"

class OrderType(Enum):
    """Order type with support for advanced order types."""
    # Basic order types
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    
    # Advanced order types
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"
    TRAILING_STOP_LIMIT = "TRAILING_STOP_LIMIT"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"
    
    # Advanced multi-leg orders
    OCO = "OCO"                        # One-Cancels-Other
    BRACKET = "BRACKET"                # Bracket order (entry + take profit + stop loss)
    IF_TOUCHED = "IF_TOUCHED"          # If Touched order
    ONE_TRIGGERS_OCO = "ONE_TRIGGERS_OCO"  # OTO (One-Triggers-OCO)
    
    # Special order types
    SETTLE_POSITION = "SETTLE_POSITION"
    TRAILING_STOP_MARKET = "TRAILING_STOP_MARKET"  # Trailing stop with market execution

class TimeInForce(Enum):
    """Time in force for an order."""
    GTC = "GTC"  # Good Till Cancel
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    GTD = "GTD"  # Good Till Date
    DAY = "DAY"  # Day order
    GTD_EXT = "GTD_EXT"  # Good Till Date Extended Hours
    
class OrderLinkType(Enum):
    """Type of relationship between linked orders (OCO, Bracket, etc.)."""
    NONE = "NONE"
    OCO = "OCO"                    # One-Cancels-Other
    BRACKET = "BRACKET"            # Bracket order (entry + take profit + stop loss)
    TAKE_PROFIT = "TAKE_PROFIT"    # Take profit order
    STOP_LOSS = "STOP_LOSS"        # Stop loss order
    ENTRY = "ENTRY"                # Entry order (for bracket)
    TRAILING_STOP = "TRAILING_STOP"  # Trailing stop order
    IF_TOUCHED = "IF_TOUCHED"      # If Touched order
    OTO = "OTO"                    # One-Triggers-OCO


class OrderRejectReason(Enum):
    """Reasons for order rejection with enhanced error codes."""
    UNKNOWN = "UNKNOWN"
    RISK_CHECK_FAILED = "RISK_CHECK_FAILED"
    INSUFFICIENT_BALANCE = "INSUFFICIENT_BALANCE"
    UNSUPPORTED_ORDER = "UNSUPPORTED_ORDER"
    DUPLICATE_ORDER = "DUPLICATE_ORDER"
    INVALID_QUANTITY = "INVALID_QUANTITY"
    INVALID_PRICE = "INVALID_PRICE"
    INVALID_STOP_PRICE = "INVALID_STOP_PRICE"
    INVALID_TIF = "INVALID_TIF"
    INVALID_ORDER_TYPE = "INVALID_ORDER_TYPE"
    INVALID_LINKED_ORDER = "INVALID_LINKED_ORDER"
    INVALID_ORDER_GROUP = "INVALID_ORDER_GROUP"
    INVALID_ORDER_RELATIONSHIP = "INVALID_ORDER_RELATIONSHIP"
    
class OrderEventType(Enum):
    """Order event types."""
    CREATED = "CREATED"
    UPDATED = "UPDATED"
    STATUS_CHANGED = "STATUS_CHANGED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELED = "CANCELED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    SUSPENDED = "SUSPENDED"
    ERROR = "ERROR"
    MODIFY_REQUESTED = "MODIFY_REQUESTED"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    
@dataclass
class OrderEvent:
    """Order event data."""
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: Optional[OrderEventType] = None
    order_id: Optional[str] = None
    client_order_id: Optional[str] = None
    status: Optional[OrderStatus] = None
    filled_quantity: Optional[Decimal] = None
    remaining_quantity: Optional[Decimal] = None
    avg_fill_price: Optional[Decimal] = None
    last_fill_price: Optional[Decimal] = None
    last_fill_quantity: Optional[Decimal] = None
    last_fill_time: Optional[datetime] = None
    reject_reason: Optional[OrderRejectReason] = None
    reject_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'event_id': self.event_id,
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type.value if self.event_type else None,
            'order_id': self.order_id,
            'client_order_id': self.client_order_id,
            'status': self.status.value if self.status else None,
            'filled_quantity': str(self.filled_quantity) if self.filled_quantity is not None else None,
            'remaining_quantity': str(self.remaining_quantity) if self.remaining_quantity is not None else None,
            'avg_fill_price': str(self.avg_fill_price) if self.avg_fill_price is not None else None,
            'last_fill_price': str(self.last_fill_price) if self.last_fill_price is not None else None,
            'last_fill_quantity': str(self.last_fill_quantity) if self.last_fill_quantity is not None else None,
            'last_fill_time': self.last_fill_time.isoformat() if self.last_fill_time else None,
            'reject_reason': self.reject_reason.value if self.reject_reason else None,
            'reject_message': self.reject_message,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OrderEvent':
        """Create from dictionary."""
        return cls(
            event_id=data.get('event_id', str(uuid.uuid4())),
            timestamp=datetime.fromisoformat(data['timestamp']) if 'timestamp' in data else datetime.now(timezone.utc),
            event_type=OrderEventType(data['event_type']) if data.get('event_type') else None,
            order_id=data.get('order_id'),
            client_order_id=data.get('client_order_id'),
            status=OrderStatus(data['status']) if data.get('status') else None,
            filled_quantity=Decimal(data['filled_quantity']) if data.get('filled_quantity') else None,
            remaining_quantity=Decimal(data['remaining_quantity']) if data.get('remaining_quantity') else None,
            avg_fill_price=Decimal(data['avg_fill_price']) if data.get('avg_fill_price') else None,
            last_fill_price=Decimal(data['last_fill_price']) if data.get('last_fill_price') else None,
            last_fill_quantity=Decimal(data['last_fill_quantity']) if data.get('last_fill_quantity') else None,
            last_fill_time=datetime.fromisoformat(data['last_fill_time']) if data.get('last_fill_time') else None,
            reject_reason=OrderRejectReason(data['reject_reason']) if data.get('reject_reason') else None,
            reject_message=data.get('reject_message'),
            metadata=data.get('metadata', {})
        )

class OrderStateError(Exception):
    """Order state transition error."""
    pass

class OrderValidationError(Exception):
    """Order validation error."""
    pass

class OrderCancelError(Exception):
    """Order cancellation error."""
    pass

class OrderModifyError(Exception):
    """Order modification error."""
    pass

class OrderExpiredError(OrderStateError):
    """Order has expired."""
    pass

class OrderRejectedError(OrderStateError):
    """Order was rejected."""
    pass
