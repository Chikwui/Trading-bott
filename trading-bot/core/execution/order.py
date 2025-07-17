"""
Order Management Module

This module defines the Order class and related enums for order management.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional, Dict, Any
import uuid


class OrderType(Enum):
    """Types of orders that can be placed."""
    MARKET = auto()      # Market order (execute immediately at best available price)
    LIMIT = auto()       # Limit order (execute at specified price or better)
    STOP = auto()        # Stop order (becomes market order when price is reached)
    STOP_LIMIT = auto()  # Stop-limit order (becomes limit order when price is reached)
    TRAILING_STOP = auto()  # Trailing stop order (adjusts stop price as market moves)
    ICEBERG = auto()     # Iceberg order (large order split into smaller visible quantities)
    TWAP = auto()        # Time-Weighted Average Price order
    VWAP = auto()        # Volume-Weighted Average Price order
    

class OrderSide(Enum):
    """Order side (buy or sell)."""
    BUY = auto()
    SELL = auto()
    
    def is_buy(self) -> bool:
        return self == OrderSide.BUY
    
    def is_sell(self) -> bool:
        return self == OrderSide.SELL


class OrderStatus(Enum):
    """Current status of an order."""
    NEW = auto()            # Order has been created but not yet sent to broker
    PENDING_NEW = auto()    # Order has been sent to broker but not yet acknowledged
    PARTIALLY_FILLED = auto()  # Order has been partially filled
    FILLED = auto()         # Order has been completely filled
    CANCELED = auto()       # Order has been canceled
    REJECTED = auto()       # Order was rejected by broker
    EXPIRED = auto()        # Order expired (time in force)
    SUSPENDED = auto()      # Order is suspended (regulatory or other reasons)
    CALCULATED = auto()     # Order has been completed for the day (e.g., VWAP)
    DONE_FOR_DAY = auto()   # Order is done for the day
    
    def is_active(self) -> bool:
        """Check if the order is still active."""
        return self in [
            OrderStatus.NEW,
            OrderStatus.PENDING_NEW,
            OrderStatus.PARTIALLY_FILLED
        ]
    
    def is_done(self) -> bool:
        """Check if the order is completely done (filled, canceled, rejected, etc.)."""
        return not self.is_active()


class TimeInForce(Enum):
    """Time in force for orders."""
    DAY = auto()            # Order is good for the trading day
    GTC = auto()            # Good Till Canceled
    IOC = auto()            # Immediate or Cancel
    FOK = auto()            # Fill or Kill
    GTD = auto()            # Good Till Date (requires expiration time)
    AT_THE_OPEN = auto()    # Market on open
    AT_THE_CLOSE = auto()   # Market on close


@dataclass
class Order:
    """
    Represents a trading order.
    
    Attributes:
        symbol: Trading symbol (e.g., 'AAPL')
        order_type: Type of order (MARKET, LIMIT, etc.)
        side: Order side (BUY or SELL)
        quantity: Number of shares/contracts
        limit_price: Limit price (required for LIMIT orders)
        stop_price: Stop price (required for STOP orders)
        time_in_force: Order time in force
        order_id: Unique order identifier
        status: Current order status
        filled_quantity: Number of shares/contracts filled
        avg_fill_price: Average fill price
        created_at: Timestamp when order was created
        updated_at: Timestamp when order was last updated
        client_order_id: Client-specific order ID
        parent_order_id: ID of parent order (for OCO, bracket orders)
        strategy_id: ID of the strategy that generated the order
        metadata: Additional order metadata
    """
    symbol: str
    order_type: OrderType
    side: OrderSide
    quantity: float
    
    # Optional fields with defaults
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.DAY
    order_id: str = field(default_factory=lambda: f"ord_{uuid.uuid4().hex[:8]}")
    status: OrderStatus = OrderStatus.NEW
    filled_quantity: float = 0.0
    avg_fill_price: Optional[float] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    client_order_id: Optional[str] = None
    parent_order_id: Optional[str] = None
    strategy_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate order parameters after initialization."""
        if self.quantity <= 0:
            raise ValueError("Order quantity must be positive")
            
        if self.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and self.limit_price is None:
            raise ValueError(f"Limit price is required for {self.order_type.name} orders")
            
        if self.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and self.stop_price is None:
            raise ValueError(f"Stop price is required for {self.order_type.name} orders")
    
    def is_buy(self) -> bool:
        """Check if this is a buy order."""
        return self.side == OrderSide.BUY
    
    def is_sell(self) -> bool:
        """Check if this is a sell order."""
        return self.side == OrderSide.SELL
    
    def is_active(self) -> bool:
        """Check if the order is still active."""
        return self.status.is_active()
    
    def is_done(self) -> bool:
        """Check if the order is completely done."""
        return self.status.is_done()
    
    def is_filled(self) -> bool:
        """Check if the order is completely filled."""
        return self.status == OrderStatus.FILLED
    
    def remaining_quantity(self) -> float:
        """Calculate remaining quantity to be filled."""
        return max(0.0, self.quantity - self.filled_quantity)
    
    def update_status(self, new_status: OrderStatus, filled_qty: float = None, 
                     fill_price: float = None, timestamp: datetime = None):
        """
        Update the order status and fill information.
        
        Args:
            new_status: New order status
            filled_qty: New filled quantity (if any)
            fill_price: Fill price (if any)
            timestamp: Timestamp of the update (defaults to now)
        """
        timestamp = timestamp or datetime.utcnow()
        
        # Update filled quantity if provided
        if filled_qty is not None:
            self.filled_quantity = filled_qty
            
            # If fully filled, update status accordingly
            if self.filled_quantity >= self.quantity:
                new_status = OrderStatus.FILLED
                self.filled_quantity = self.quantity
        
        # Update average fill price if provided
        if fill_price is not None and filled_qty is not None and filled_qty > 0:
            if self.avg_fill_price is None:
                self.avg_fill_price = fill_price
            else:
                total_value = (self.avg_fill_price * (self.filled_quantity - filled_qty) +
                             fill_price * filled_qty)
                self.avg_fill_price = total_value / self.filled_quantity
        
        # Update status and timestamp
        self.status = new_status
        self.updated_at = timestamp
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary representation."""
        return {
            'order_id': self.order_id,
            'client_order_id': self.client_order_id,
            'symbol': self.symbol,
            'order_type': self.order_type.name,
            'side': self.side.name,
            'quantity': self.quantity,
            'filled_quantity': self.filled_quantity,
            'remaining_quantity': self.remaining_quantity(),
            'limit_price': self.limit_price,
            'stop_price': self.stop_price,
            'avg_fill_price': self.avg_fill_price,
            'status': self.status.name,
            'time_in_force': self.time_in_force.name,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'parent_order_id': self.parent_order_id,
            'strategy_id': self.strategy_id,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Order':
        """Create an Order instance from a dictionary."""
        # Convert string enums back to enum values
        data = data.copy()
        data['order_type'] = OrderType[data['order_type']]
        data['side'] = OrderSide[data['side']]
        data['status'] = OrderStatus[data['status']]
        data['time_in_force'] = TimeInForce[data['time_in_force']]
        
        # Convert string timestamps back to datetime objects
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if isinstance(data.get('updated_at'), str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        return cls(**data)


class OrderList:
    """Container for managing a collection of orders."""
    
    def __init__(self):
        self.orders = {}
        self._order_id_index = {}
        self._client_order_id_index = {}
        self._symbol_index = {}
        self._status_index = {}
    
    def add(self, order: Order):
        """Add an order to the list."""
        self.orders[order.order_id] = order
        self._update_indexes(order)
    
    def get(self, order_id: str) -> Optional[Order]:
        """Get an order by its ID."""
        return self.orders.get(order_id)
    
    def get_by_client_order_id(self, client_order_id: str) -> Optional[Order]:
        """Get an order by its client order ID."""
        order_id = self._client_order_id_index.get(client_order_id)
        return self.get(order_id) if order_id else None
    
    def get_all(self) -> List[Order]:
        """Get all orders."""
        return list(self.orders.values())
    
    def get_active_orders(self) -> List[Order]:
        """Get all active orders."""
        return [order for order in self.orders.values() if order.is_active()]
    
    def get_orders_by_symbol(self, symbol: str) -> List[Order]:
        """Get all orders for a specific symbol."""
        order_ids = self._symbol_index.get(symbol, set())
        return [self.orders[oid] for oid in order_ids if oid in self.orders]
    
    def get_orders_by_status(self, status: OrderStatus) -> List[Order]:
        """Get all orders with a specific status."""
        order_ids = self._status_index.get(status, set())
        return [self.orders[oid] for oid in order_ids if oid in self.orders]
    
    def update(self, order: Order):
        """Update an existing order."""
        if order.order_id not in self.orders:
            raise KeyError(f"Order {order.order_id} not found")
        
        # Remove old indexes
        old_order = self.orders[order.order_id]
        self._remove_from_indexes(old_order)
        
        # Update order and reindex
        self.orders[order.order_id] = order
        self._update_indexes(order)
    
    def remove(self, order_id: str) -> bool:
        """Remove an order by its ID."""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        del self.orders[order_id]
        self._remove_from_indexes(order)
        return True
    
    def _update_indexes(self, order: Order):
        """Update all indexes for an order."""
        # Order ID index (always exists)
        self._order_id_index[order.order_id] = order
        
        # Client order ID index (if exists)
        if order.client_order_id:
            self._client_order_id_index[order.client_order_id] = order.order_id
        
        # Symbol index
        if order.symbol not in self._symbol_index:
            self._symbol_index[order.symbol] = set()
        self._symbol_index[order.symbol].add(order.order_id)
        
        # Status index
        if order.status not in self._status_index:
            self._status_index[order.status] = set()
        self._status_index[order.status].add(order.order_id)
    
    def _remove_from_indexes(self, order: Order):
        """Remove an order from all indexes."""
        # Client order ID index
        if order.client_order_id and order.client_order_id in self._client_order_id_index:
            del self._client_order_id_index[order.client_order_id]
        
        # Symbol index
        if order.symbol in self._symbol_index:
            self._symbol_index[order.symbol].discard(order.order_id)
            if not self._symbol_index[order.symbol]:
                del self._symbol_index[order.symbol]
        
        # Status index
        if order.status in self._status_index:
            self._status_index[order.status].discard(order.order_id)
            if not self._status_index[order.status]:
                del self._status_index[order.status]
