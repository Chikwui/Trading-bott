"""
Advanced order types and execution logic for the trading system.
"""
from __future__ import annotations

import asyncio
import enum
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum, auto
from typing import (
    Any, Dict, List, Optional, Set, Tuple, Callable, Awaitable, Union,
    DefaultDict
)

logger = logging.getLogger(__name__)

class OrderSide(Enum):
    """Order side (buy/sell)."""
    BUY = "BUY"
    SELL = "SELL"
    
    def opposite(self) -> 'OrderSide':
        """Get the opposite side."""
        return OrderSide.SELL if self == OrderSide.BUY else OrderSide.BUY

class OrderType(Enum):
    """Order types."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_LIMIT = "TAKE_PROFIT_LIMIT"
    OCO = "OCO"  # One-Cancels-Other
    BRACKET = "BRACKET"
    IF_TOUCHED = "IF_TOUCHED"

class TimeInForce(Enum):
    """Time in force for orders."""
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate or Cancel
    FOK = "FOK"  # Fill or Kill
    GTD = "GTD"  # Good Till Date
    DAY = "DAY"  # Day order
    GTC_EXT = "GTC_EXT"  # GTC with extended hours

class OrderStatus(Enum):
    """Order status."""
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    PENDING_CANCEL = "PENDING_CANCEL"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    EXPIRED_IN_MATCH = "EXPIRED_IN_MATCH"
    TRIGGERED = "TRIGGERED"  # For stop/take profit orders
    
    def is_active(self) -> bool:
        """Check if the order is still active."""
        return self in [
            OrderStatus.NEW,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.PENDING_CANCEL,
            OrderStatus.TRIGGERED
        ]

class Order:
    """Base order class."""
    
    def __init__(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        price: Optional[Decimal] = None,
        stop_price: Optional[Decimal] = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        client_order_id: Optional[str] = None,
        parent_order_id: Optional[str] = None,
        reduce_only: bool = False,
        hidden: bool = False,
        post_only: bool = False,
        iceberg_qty: Optional[Decimal] = None,
        timestamp: Optional[datetime] = None,
        **kwargs
    ):
        """Initialize an order."""
        self.symbol = symbol.upper()
        self.side = side if isinstance(side, OrderSide) else OrderSide(side.upper())
        self.order_type = order_type if isinstance(order_type, OrderType) else OrderType(order_type.upper())
        self.quantity = Decimal(quantity) if not isinstance(quantity, Decimal) else quantity
        self.price = Decimal(price) if price is not None else None
        self.stop_price = Decimal(stop_price) if stop_price is not None else None
        self.time_in_force = time_in_force if isinstance(time_in_force, TimeInForce) else TimeInForce(time_in_force.upper())
        self.client_order_id = client_order_id or f"{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        self.parent_order_id = parent_order_id
        self.reduce_only = bool(reduce_only)
        self.hidden = bool(hidden)
        self.post_only = bool(post_only)
        self.iceberg_qty = Decimal(iceberg_qty) if iceberg_qty is not None else None
        self.timestamp = timestamp or datetime.now(timezone.utc)
        
        # Execution fields
        self.status = OrderStatus.NEW
        self.executed_quantity = Decimal(0)
        self.cumulative_quote_qty = Decimal(0)
        self.avg_price = Decimal(0)
        self.last_executed_quantity = Decimal(0)
        self.last_executed_price = Decimal(0)
        self.commission = Decimal(0)
        self.commission_asset = ""
        self.update_time = self.timestamp
        self.working_time = self.timestamp
        
        # Additional metadata
        self.metadata = kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert order to dictionary."""
        return {
            'symbol': self.symbol,
            'side': self.side.value,
            'type': self.order_type.value,
            'quantity': str(self.quantity),
            'price': str(self.price) if self.price is not None else None,
            'stop_price': str(self.stop_price) if self.stop_price is not None else None,
            'time_in_force': self.time_in_force.value,
            'client_order_id': self.client_order_id,
            'parent_order_id': self.parent_order_id,
            'reduce_only': self.reduce_only,
            'hidden': self.hidden,
            'post_only': self.post_only,
            'iceberg_qty': str(self.iceberg_qty) if self.iceberg_qty is not None else None,
            'status': self.status.value,
            'executed_quantity': str(self.executed_quantity),
            'cumulative_quote_qty': str(self.cumulative_quote_qty),
            'avg_price': str(self.avg_price) if self.avg_price else None,
            'last_executed_quantity': str(self.last_executed_quantity) if self.last_executed_quantity else None,
            'last_executed_price': str(self.last_executed_price) if self.last_executed_price else None,
            'commission': str(self.commission) if self.commission else None,
            'commission_asset': self.commission_asset,
            'timestamp': self.timestamp.isoformat(),
            'update_time': self.update_time.isoformat(),
            'working_time': self.working_time.isoformat(),
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Order':
        """Create an order from a dictionary."""
        order = cls(
            symbol=data['symbol'],
            side=OrderSide(data['side']),
            order_type=OrderType(data['type']),
            quantity=Decimal(data['quantity']),
            price=Decimal(data['price']) if data.get('price') is not None else None,
            stop_price=Decimal(data['stop_price']) if data.get('stop_price') is not None else None,
            time_in_force=TimeInForce(data.get('time_in_force', 'GTC')),
            client_order_id=data.get('client_order_id'),
            parent_order_id=data.get('parent_order_id'),
            reduce_only=bool(data.get('reduce_only', False)),
            hidden=bool(data.get('hidden', False)),
            post_only=bool(data.get('post_only', False)),
            iceberg_qty=Decimal(data['iceberg_qty']) if data.get('iceberg_qty') is not None else None,
            timestamp=datetime.fromisoformat(data['timestamp']) if 'timestamp' in data else datetime.now(timezone.utc)
        )
        
        # Set execution fields
        if 'status' in data:
            order.status = OrderStatus(data['status'])
        if 'executed_quantity' in data:
            order.executed_quantity = Decimal(data['executed_quantity'])
        if 'cumulative_quote_qty' in data:
            order.cumulative_quote_qty = Decimal(data['cumulative_quote_qty'])
        if 'avg_price' in data and data['avg_price'] is not None:
            order.avg_price = Decimal(data['avg_price'])
        if 'last_executed_quantity' in data and data['last_executed_quantity'] is not None:
            order.last_executed_quantity = Decimal(data['last_executed_quantity'])
        if 'last_executed_price' in data and data['last_executed_price'] is not None:
            order.last_executed_price = Decimal(data['last_executed_price'])
        if 'commission' in data and data['commission'] is not None:
            order.commission = Decimal(data['commission'])
        if 'commission_asset' in data:
            order.commission_asset = data['commission_asset']
        if 'update_time' in data and data['update_time'] is not None:
            order.update_time = datetime.fromisoformat(data['update_time'])
        if 'working_time' in data and data['working_time'] is not None:
            order.working_time = datetime.fromisoformat(data['working_time'])
        
        # Set metadata
        if 'metadata' in data and isinstance(data['metadata'], dict):
            order.metadata.update(data['metadata'])
        
        return order
    
    def update_execution(
        self,
        executed_qty: Decimal,
        executed_price: Decimal,
        commission: Decimal = Decimal(0),
        commission_asset: str = "",
        timestamp: Optional[datetime] = None
    ) -> None:
        """Update order execution details."""
        self.last_executed_quantity = executed_qty
        self.last_executed_price = executed_price
        self.executed_quantity += executed_qty
        self.cumulative_quote_qty += executed_qty * executed_price
        self.avg_price = self.cumulative_quote_qty / self.executed_quantity if self.executed_quantity > 0 else Decimal(0)
        self.commission += commission
        self.commission_asset = commission_asset or self.commission_asset
        self.update_time = timestamp or datetime.now(timezone.utc)
        
        # Update status
        if self.executed_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
        elif self.executed_quantity > 0:
            self.status = OrderStatus.PARTIALLY_FILLED
    
    def cancel(self, timestamp: Optional[datetime] = None) -> None:
        """Cancel the order."""
        if self.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
            return
            
        self.status = OrderStatus.CANCELED
        self.update_time = timestamp or datetime.now(timezone.utc)
    
    def is_active(self) -> bool:
        """Check if the order is active."""
        return self.status.is_active()
    
    def remaining_quantity(self) -> Decimal:
        """Get remaining quantity to be filled."""
        return max(Decimal(0), self.quantity - self.executed_quantity)
    
    def is_triggered(self, price: Decimal) -> bool:
        """Check if the order should be triggered at the given price."""
        if self.order_type == OrderType.STOP:
            if self.side == OrderSide.BUY:
                return price >= self.stop_price
            else:
                return price <= self.stop_price
        elif self.order_type == OrderType.TAKE_PROFIT:
            if self.side == OrderSide.BUY:
                return price <= self.stop_price
            else:
                return price >= self.stop_price
        return False

class BracketOrder:
    """Bracket order (entry order with take profit and stop loss)."""
    
    def __init__(
        self,
        symbol: str,
        side: Union[OrderSide, str],
        quantity: Union[Decimal, str, float],
        entry_price: Optional[Union[Decimal, str, float]] = None,
        stop_loss: Optional[Union[Decimal, str, float]] = None,
        take_profit: Optional[Union[Decimal, str, float]] = None,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        time_in_force: Union[TimeInForce, str] = TimeInForce.GTC,
        client_order_id: Optional[str] = None,
        **kwargs
    ):
        """Initialize a bracket order."""
        self.symbol = symbol.upper()
        self.side = side if isinstance(side, OrderSide) else OrderSide(side.upper())
        self.quantity = Decimal(quantity) if not isinstance(quantity, Decimal) else quantity
        self.entry_price = Decimal(entry_price) if entry_price is not None else None
        self.stop_loss = Decimal(stop_loss) if stop_loss is not None else None
        self.take_profit = Decimal(take_profit) if take_profit is not None else None
        self.stop_loss_pct = float(stop_loss_pct) if stop_loss_pct is not None else None
        self.take_profit_pct = float(take_profit_pct) if take_profit_pct is not None else None
        self.time_in_force = time_in_force if isinstance(time_in_force, TimeInForce) else TimeInForce(time_in_force.upper())
        self.client_order_id = client_order_id or f"BRACKET_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        self.entry_order: Optional[Order] = None
        self.stop_loss_order: Optional[Order] = None
        self.take_profit_order: Optional[Order] = None
        self.status: str = "NEW"
        self.created_at: datetime = datetime.now(timezone.utc)
        self.updated_at: datetime = self.created_at
        self.metadata: Dict[str, Any] = kwargs
        
        # Calculate stop loss/take profit from percentages if needed
        if entry_price is not None:
            self._calculate_sl_tp()
    
    def _calculate_sl_tp(self) -> None:
        """Calculate stop loss and take profit prices from percentages."""
        if self.entry_price is None:
            return
            
        if self.stop_loss_pct is not None and self.stop_loss is None:
            if self.side == OrderSide.BUY:
                self.stop_loss = self.entry_price * (1 - Decimal(str(self.stop_loss_pct)) / 100)
            else:
                self.stop_loss = self.entry_price * (1 + Decimal(str(self.stop_loss_pct)) / 100)
        
        if self.take_profit_pct is not None and self.take_profit is None:
            if self.side == OrderSide.BUY:
                self.take_profit = self.entry_price * (1 + Decimal(str(self.take_profit_pct)) / 100)
            else:
                self.take_profit = self.entry_price * (1 - Decimal(str(self.take_profit_pct)) / 100)
    
    def create_orders(self) -> Tuple[Order, Order, Order]:
        """Create entry, stop loss, and take profit orders."""
        if self.entry_price is None:
            raise ValueError("Entry price must be set to create orders")
            
        # Create entry order
        self.entry_order = Order(
            symbol=self.symbol,
            side=self.side,
            order_type=OrderType.LIMIT if self.entry_price else OrderType.MARKET,
            quantity=self.quantity,
            price=self.entry_price if self.entry_price else None,
            time_in_force=self.time_in_force,
            client_order_id=f"{self.client_order_id}_ENTRY",
            parent_order_id=self.client_order_id,
            metadata={"bracket_id": self.client_order_id, "order_type": "ENTRY"}
        )
        
        # Create stop loss order
        if self.stop_loss is not None:
            self.stop_loss_order = Order(
                symbol=self.symbol,
                side=self.side.opposite(),
                order_type=OrderType.STOP,
                quantity=self.quantity,
                stop_price=self.stop_loss,
                time_in_force=self.time_in_force,
                client_order_id=f"{self.client_order_id}_STOP_LOSS",
                parent_order_id=self.client_order_id,
                reduce_only=True,
                metadata={"bracket_id": self.client_order_id, "order_type": "STOP_LOSS"}
            )
        
        # Create take profit order
        if self.take_profit is not None:
            self.take_profit_order = Order(
                symbol=self.symbol,
                side=self.side.opposite(),
                order_type=OrderType.TAKE_PROFIT,
                quantity=self.quantity,
                price=self.take_profit,
                stop_price=self.take_profit,
                time_in_force=self.time_in_force,
                client_order_id=f"{self.client_order_id}_TAKE_PROFIT",
                parent_order_id=self.client_order_id,
                reduce_only=True,
                metadata={"bracket_id": self.client_order_id, "order_type": "TAKE_PROFIT"}
            )
        
        return self.entry_order, self.stop_loss_order, self.take_profit_order
    
    def update_status(self) -> None:
        """Update the status of the bracket order based on its child orders."""
        if not self.entry_order:
            self.status = "NEW"
            return
            
        if self.entry_order.status == OrderStatus.FILLED:
            self.status = "ACTIVE"
        elif self.entry_order.status in [OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
            self.status = "CANCELED"
        
        if self.status == "ACTIVE":
            if self.stop_loss_order and self.stop_loss_order.status == OrderStatus.FILLED:
                self.status = "STOPPED"
            elif self.take_profit_order and self.take_profit_order.status == OrderStatus.FILLED:
                self.status = "TAKEN"
        
        self.updated_at = datetime.now(timezone.utc)
    
    def cancel(self) -> None:
        """Cancel all active orders in the bracket."""
        for order in [self.entry_order, self.stop_loss_order, self.take_profit_order]:
            if order and order.is_active():
                order.cancel()
        
        self.status = "CANCELED"
        self.updated_at = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert bracket order to dictionary."""
        return {
            'client_order_id': self.client_order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': str(self.quantity),
            'entry_price': str(self.entry_price) if self.entry_price is not None else None,
            'stop_loss': str(self.stop_loss) if self.stop_loss is not None else None,
            'take_profit': str(self.take_profit) if self.take_profit is not None else None,
            'stop_loss_pct': self.stop_loss_pct,
            'take_profit_pct': self.take_profit_pct,
            'time_in_force': self.time_in_force.value,
            'status': self.status,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'entry_order': self.entry_order.to_dict() if self.entry_order else None,
            'stop_loss_order': self.stop_loss_order.to_dict() if self.stop_loss_order else None,
            'take_profit_order': self.take_profit_order.to_dict() if self.take_profit_order else None,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BracketOrder':
        """Create a bracket order from a dictionary."""
        bracket = cls(
            symbol=data['symbol'],
            side=OrderSide(data['side']),
            quantity=Decimal(data['quantity']),
            entry_price=Decimal(data['entry_price']) if data.get('entry_price') is not None else None,
            stop_loss=Decimal(data['stop_loss']) if data.get('stop_loss') is not None else None,
            take_profit=Decimal(data['take_profit']) if data.get('take_profit') is not None else None,
            stop_loss_pct=data.get('stop_loss_pct'),
            take_profit_pct=data.get('take_profit_pct'),
            time_in_force=TimeInForce(data.get('time_in_force', 'GTC')),
            client_order_id=data.get('client_order_id'),
            **data.get('metadata', {})
        )
        
        if 'entry_order' in data and data['entry_order'] is not None:
            bracket.entry_order = Order.from_dict(data['entry_order'])
        if 'stop_loss_order' in data and data['stop_loss_order'] is not None:
            bracket.stop_loss_order = Order.from_dict(data['stop_loss_order'])
        if 'take_profit_order' in data and data['take_profit_order'] is not None:
            bracket.take_profit_order = Order.from_dict(data['take_profit_order'])
        
        bracket.status = data.get('status', 'NEW')
        if 'created_at' in data:
            bracket.created_at = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data:
            bracket.updated_at = datetime.fromisoformat(data['updated_at'])
        
        return bracket

class OCOOrder:
    """One-Cancels-Other (OCO) order."""
    
    def __init__(
        self,
        symbol: str,
        side: Union[OrderSide, str],
        quantity: Union[Decimal, str, float],
        price: Union[Decimal, str, float],
        stop_price: Union[Decimal, str, float],
        stop_limit_price: Optional[Union[Decimal, str, float]] = None,
        time_in_force: Union[TimeInForce, str] = TimeInForce.GTC,
        client_order_id: Optional[str] = None,
        **kwargs
    ):
        """Initialize an OCO order."""
        self.symbol = symbol.upper()
        self.side = side if isinstance(side, OrderSide) else OrderSide(side.upper())
        self.quantity = Decimal(quantity) if not isinstance(quantity, Decimal) else quantity
        self.price = Decimal(price) if not isinstance(price, Decimal) else price
        self.stop_price = Decimal(stop_price) if not isinstance(stop_price, Decimal) else stop_price
        self.stop_limit_price = Decimal(stop_limit_price) if stop_limit_price is not None else None
        self.time_in_force = time_in_force if isinstance(time_in_force, TimeInForce) else TimeInForce(time_in_force.upper())
        self.client_order_id = client_order_id or f"OCO_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        self.limit_order: Optional[Order] = None
        self.stop_order: Optional[Order] = None
        self.status: str = "NEW"
        self.created_at: datetime = datetime.now(timezone.utc)
        self.updated_at: datetime = self.created_at
        self.metadata: Dict[str, Any] = kwargs
    
    def create_orders(self) -> Tuple[Order, Order]:
        """Create the limit and stop orders for the OCO."""
        # Create limit order
        self.limit_order = Order(
            symbol=self.symbol,
            side=self.side,
            order_type=OrderType.LIMIT,
            quantity=self.quantity,
            price=self.price,
            time_in_force=self.time_in_force,
            client_order_id=f"{self.client_order_id}_LIMIT",
            parent_order_id=self.client_order_id,
            metadata={"oco_id": self.client_order_id, "order_type": "LIMIT"}
        )
        
        # Create stop order (or stop-limit)
        order_type = OrderType.STOP_LIMIT if self.stop_limit_price is not None else OrderType.STOP
        
        self.stop_order = Order(
            symbol=self.symbol,
            side=self.side,
            order_type=order_type,
            quantity=self.quantity,
            price=self.stop_limit_price if order_type == OrderType.STOP_LIMIT else self.stop_price,
            stop_price=self.stop_price,
            time_in_force=self.time_in_force,
            client_order_id=f"{self.client_order_id}_STOP",
            parent_order_id=self.client_order_id,
            metadata={"oco_id": self.client_order_id, "order_type": order_type.name}
        )
        
        return self.limit_order, self.stop_order
    
    def update_status(self) -> None:
        """Update the status of the OCO order based on its child orders."""
        if not self.limit_order or not self.stop_order:
            self.status = "NEW"
            return
            
        if self.limit_order.status == OrderStatus.FILLED:
            self.status = "FILLED"
            # Cancel the other order
            if self.stop_order.is_active():
                self.stop_order.cancel()
        elif self.stop_order.status == OrderStatus.FILLED:
            self.status = "STOPPED"
            # Cancel the other order
            if self.limit_order.is_active():
                self.limit_order.cancel()
        elif self.limit_order.status in [OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED] and \
             self.stop_order.status in [OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
            self.status = "CANCELED"
        
        self.updated_at = datetime.now(timezone.utc)
    
    def cancel(self) -> None:
        """Cancel both orders in the OCO."""
        if self.limit_order and self.limit_order.is_active():
            self.limit_order.cancel()
        if self.stop_order and self.stop_order.is_active():
            self.stop_order.cancel()
        
        self.status = "CANCELED"
        self.updated_at = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert OCO order to dictionary."""
        return {
            'client_order_id': self.client_order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': str(self.quantity),
            'price': str(self.price),
            'stop_price': str(self.stop_price),
            'stop_limit_price': str(self.stop_limit_price) if self.stop_limit_price is not None else None,
            'time_in_force': self.time_in_force.value,
            'status': self.status,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'limit_order': self.limit_order.to_dict() if self.limit_order else None,
            'stop_order': self.stop_order.to_dict() if self.stop_order else None,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OCOOrder':
        """Create an OCO order from a dictionary."""
        oco = cls(
            symbol=data['symbol'],
            side=OrderSide(data['side']),
            quantity=Decimal(data['quantity']),
            price=Decimal(data['price']),
            stop_price=Decimal(data['stop_price']),
            stop_limit_price=Decimal(data['stop_limit_price']) if data.get('stop_limit_price') is not None else None,
            time_in_force=TimeInForce(data.get('time_in_force', 'GTC')),
            client_order_id=data.get('client_order_id'),
            **data.get('metadata', {})
        )
        
        if 'limit_order' in data and data['limit_order'] is not None:
            oco.limit_order = Order.from_dict(data['limit_order'])
        if 'stop_order' in data and data['stop_order'] is not None:
            oco.stop_order = Order.from_dict(data['stop_order'])
        
        oco.status = data.get('status', 'NEW')
        if 'created_at' in data:
            oco.created_at = datetime.fromisoformat(data['created_at'])
        if 'updated_at' in data:
            oco.updated_at = datetime.fromisoformat(data['updated_at'])
        
        return oco
