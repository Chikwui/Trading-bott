"""Advanced order types and their relationships."""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Union
from uuid import uuid4

from decimal import Decimal

from ..models.order import Order, OrderStatus, OrderType, OrderSide


class AdvancedOrderType(Enum):
    """Types of advanced orders."""
    OCO = "OCO"  # One-Cancels-Other
    BRACKET = "BRACKET"  # Entry with take-profit and stop-loss
    TRAILING_STOP = "TRAILING_STOP"
    ICEBERG = "ICEBERG"
    TWAP = "TWAP"  # Time-Weighted Average Price
    VWAP = "VWAP"  # Volume-Weighted Average Price


@dataclass
class AdvancedOrder:
    """Base class for advanced order types."""
    id: str = field(default_factory=lambda: str(uuid4()))
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    status: OrderStatus = OrderStatus.NEW
    order_type: AdvancedOrderType = None
    parent_order: Optional[Order] = None
    child_orders: List[Order] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def update_status(self) -> None:
        """Update the status based on child orders."""
        if not self.child_orders:
            return

        # Get the most recent update time from child orders
        self.updated_at = max(
            (order.updated_at for order in self.child_orders if order.updated_at),
            default=self.updated_at
        )

        # Determine overall status
        statuses = {order.status for order in self.child_orders}
        
        if any(s.is_failed() for s in statuses):
            self.status = OrderStatus.REJECTED
        elif all(s.is_filled() for s in statuses):
            self.status = OrderStatus.FILLED
        elif any(s.is_filled() for s in statuses):
            self.status = OrderStatus.PARTIALLY_FILLED
        elif any(s.is_active() for s in statuses):
            self.status = OrderStatus.NEW
        else:
            self.status = OrderStatus.CANCELED


@dataclass
class OCOOrder(AdvancedOrder):
    """One-Cancels-Other order."""
    def __post_init__(self):
        self.order_type = AdvancedOrderType.OCO
    
    def add_orders(self, order1: Order, order2: Order) -> None:
        """Add the two orders that form the OCO pair."""
        if len(self.child_orders) >= 2:
            raise ValueError("OCO order already has maximum number of child orders")
        
        self.child_orders.extend([order1, order2])
        order1.parent_order_id = self.id
        order2.parent_order_id = self.id
        order1.metadata['oco_group'] = self.id
        order2.metadata['oco_group'] = self.id
        
        # Set OCO relationship
        order1.metadata['oco_other_order_id'] = order2.id
        order2.metadata['oco_other_order_id'] = order1.id


@dataclass
class BracketOrder(AdvancedOrder):
    """Bracket order (entry with take-profit and stop-loss)."""
    def __post_init__(self):
        self.order_type = AdvancedOrderType.BRACKET
    
    def add_orders(self, entry: Order, take_profit: Order, stop_loss: Order) -> None:
        """Add entry, take-profit, and stop-loss orders."""
        if len(self.child_orders) > 0:
            raise ValueError("Bracket order already has child orders")
        
        self.child_orders.extend([entry, take_profit, stop_loss])
        
        # Set parent-child relationships
        entry.parent_order_id = self.id
        take_profit.parent_order_id = self.id
        stop_loss.parent_order_id = self.id
        
        # Add metadata for relationship tracking
        entry.metadata['bracket_group'] = self.id
        take_profit.metadata['bracket_group'] = self.id
        stop_loss.metadata['bracket_group'] = self.id
        
        # Mark take-profit and stop-loss as OCO
        take_profit.metadata['oco_other_order_id'] = stop_loss.id
        stop_loss.metadata['oco_other_order_id'] = take_profit.id
        
        # Set entry order as parent of take-profit and stop-loss
        take_profit.parent_order_id = entry.id
        stop_loss.parent_order_id = entry.id


class AdvancedOrderManager:
    """Manages advanced order types and their relationships."""
    
    def __init__(self):
        self.orders: Dict[str, AdvancedOrder] = {}
        self.order_index: Dict[str, str] = {}  # Maps order_id -> advanced_order_id
        self._lock = asyncio.Lock()
    
    async def create_oco_order(
        self,
        order1: Order,
        order2: Order
    ) -> OCOOrder:
        """Create a new OCO order."""
        async with self._lock:
            oco = OCOOrder()
            oco.add_orders(order1, order2)
            self.orders[oco.id] = oco
            self.order_index[order1.id] = oco.id
            self.order_index[order2.id] = oco.id
            return oco
    
    async def create_bracket_order(
        self,
        entry: Order,
        take_profit: Order,
        stop_loss: Order
    ) -> BracketOrder:
        """Create a new bracket order."""
        async with self._lock:
            bracket = BracketOrder()
            bracket.add_orders(entry, take_profit, stop_loss)
            self.orders[bracket.id] = bracket
            for order in [entry, take_profit, stop_loss]:
                self.order_index[order.id] = bracket.id
            return bracket
    
    async def get_advanced_order(self, order_id: str) -> Optional[AdvancedOrder]:
        """Get the advanced order containing the given order ID."""
        return self.orders.get(self.order_index.get(order_id))
    
    async def handle_order_update(self, order: Order) -> None:
        """Update the status of the advanced order when a child order updates."""
        if order.id not in self.order_index:
            return
            
        adv_order = await self.get_advanced_order(order.id)
        if not adv_order:
            return
            
        # Update the advanced order status
        adv_order.update_status()
        
        # If this is an OCO order and one order was filled/cancelled,
        # cancel the other order
        if isinstance(adv_order, OCOOrder) and order.status.is_terminal():
            await self._handle_oco_completion(adv_order, order)
    
    async def _handle_oco_completion(self, oco: OCOOrder, completed_order: Order) -> None:
        """Handle completion of one order in an OCO pair."""
        other_order_id = completed_order.metadata.get('oco_other_order_id')
        if not other_order_id:
            return
            
        # Find the other order in the OCO pair
        other_order = next(
            (o for o in oco.child_orders if o.id == other_order_id and o.status.is_active()),
            None
        )
        
        if other_order:
            # Cancel the other order
            other_order.cancel()
            # Update the advanced order status
            oco.update_status()
