"""
Enhanced Order class with state machine integration.

This module provides an Order class that uses the StateMachine
for robust state management and validation.
"""
from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Type, TypeVar, Union
from uuid import uuid4

from pydantic import BaseModel, Field, validator

from core.trading.state_machine import StateMachine, StateTransitionError
from core.trading.order_states import OrderStatus, OrderSide, OrderType, TimeInForce

T = TypeVar('T', bound='Order')

class OrderStateMachine(StateMachine):
    """State machine for order lifecycle management."""
    
    # Define valid state transitions
    TRANSITION_PENDING_TO_NEW = (OrderStatus.PENDING_NEW, OrderStatus.NEW)
    TRANSITION_NEW_TO_PARTIALLY_FILLED = (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED)
    TRANSITION_PARTIALLY_FILLED_TO_FILLED = (OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED)
    TRANSITION_NEW_TO_FILLED = (OrderStatus.NEW, OrderStatus.FILLED)
    TRANSITION_NEW_TO_CANCELED = (OrderStatus.NEW, OrderStatus.CANCELED)
    TRANSITION_PARTIALLY_FILLED_TO_CANCELED = (OrderStatus.PARTIALLY_FILLED, OrderStatus.CANCELED)
    TRANSITION_NEW_TO_REJECTED = (OrderStatus.NEW, OrderStatus.REJECTED)
    TRANSITION_PENDING_CANCEL_TO_CANCELED = (OrderStatus.PENDING_CANCEL, OrderStatus.CANCELED)
    TRANSITION_PENDING_REPLACE_TO_REPLACED = (OrderStatus.PENDING_REPLACE, OrderStatus.REPLACED)
    
    def __init__(self, order: 'Order'):
        """Initialize with an order instance."""
        super().__init__(initial_state=OrderStatus.PENDING_NEW.value)
        self.order = order
    
    async def validate_transition(self, to_state: str, **kwargs) -> None:
        """Validate a state transition with order-specific rules."""
        await super().validate_transition(to_state, **kwargs)
        
        # Additional validation based on order type and state
        if to_state == OrderStatus.FILLED:
            if self.order.remaining_quantity > 0:
                raise StateTransitionError(
                    self.state,
                    to_state,
                    f"Cannot fill order with {self.order.remaining_quantity} remaining"
                )
    
    async def on_transition(
        self,
        from_state: str,
        to_state: str,
        **kwargs
    ) -> None:
        """Handle post-transition logic."""
        # Update timestamps
        now = datetime.utcnow()
        
        if to_state == OrderStatus.FILLED:
            self.order.filled_at = now
        elif to_state == OrderStatus.CANCELED:
            self.order.canceled_at = now
        elif to_state == OrderStatus.REJECTED:
            self.order.rejected_at = now
        
        self.order.updated_at = now
        
        # Call any registered hooks
        if hasattr(self.order, f'on_{to_state.lower()}'):
            handler = getattr(self.order, f'on_{to_state.lower()}')
            if callable(handler):
                await handler(**kwargs)

class Order(BaseModel):
    """Enhanced Order class with state machine integration."""
    
    # Core order fields
    id: str = Field(default_factory=lambda: f"ord_{uuid4().hex[:16]}")
    client_order_id: Optional[str] = None
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: Decimal
    filled_quantity: Decimal = Decimal('0')
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: TimeInForce = TimeInForce.GTC
    status: OrderStatus = OrderStatus.PENDING_NEW
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    filled_at: Optional[datetime] = None
    canceled_at: Optional[datetime] = None
    rejected_at: Optional[datetime] = None
    
    # State machine
    _state_machine: Optional[OrderStateMachine] = None
    _redis = None  # Will be set by class method
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            Decimal: str,
            datetime: lambda v: v.isoformat()
        }
    
    def __init__(self, **data):
        """Initialize order with state machine."""
        super().__init__(**data)
        self._state_machine = OrderStateMachine(self)
        if self._redis:
            self._state_machine.set_redis(self._redis)
    
    @classmethod
    def set_redis(cls, redis):
        """Set Redis client for distributed locking."""
        cls._redis = redis
    
    @property
    def state_machine(self) -> OrderStateMachine:
        """Get the state machine instance."""
        if self._state_machine is None:
            self._state_machine = OrderStateMachine(self)
            if self._redis:
                self._state_machine.set_redis(self._redis)
        return self._state_machine
    
    @property
    def remaining_quantity(self) -> Decimal:
        """Get remaining quantity to be filled."""
        return self.quantity - self.filled_quantity
    
    @property
    def is_active(self) -> bool:
        """Check if the order is in an active state."""
        return self.status in {
            OrderStatus.NEW,
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.PENDING_CANCEL,
            OrderStatus.PENDING_REPLACE
        }
    
    async def update_status(self, new_status: OrderStatus, **kwargs) -> bool:
        """Update order status with state machine validation."""
        try:
            success = await self.state_machine.transition_to(
                new_status.value,
                **kwargs
            )
            if success:
                self.status = new_status
            return success
        except StateTransitionError as e:
            # Log the error but don't crash
            # In a real system, you'd want to handle this more gracefully
            print(f"Invalid state transition: {e}")
            return False
    
    async def fill(self, quantity: Decimal, price: Decimal) -> bool:
        """Fill part of the order."""
        if quantity <= 0:
            raise ValueError("Fill quantity must be positive")
            
        if quantity > self.remaining_quantity:
            raise ValueError("Fill quantity exceeds remaining quantity")
            
        # Update filled quantity
        self.filled_quantity += quantity
        
        # Update status based on fill amount
        if self.filled_quantity >= self.quantity:
            return await self.update_status(OrderStatus.FILLED, fill_price=price)
        elif self.status != OrderStatus.PARTIALLY_FILLED:
            return await self.update_status(OrderStatus.PARTIALLY_FILLED, fill_price=price)
            
        return True
    
    async def cancel(self) -> bool:
        """Cancel the order."""
        if self.status not in {OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED}:
            raise ValueError(f"Cannot cancel order in state {self.status}")
            
        return await self.update_status(OrderStatus.CANCELED)
    
    async def reject(self, reason: str = "") -> bool:
        """Reject the order."""
        if self.status != OrderStatus.PENDING_NEW:
            raise ValueError(f"Cannot reject order in state {self.status}")
            
        return await self.update_status(OrderStatus.REJECTED, reason=reason)
    
    # State transition hooks
    async def on_filled(self, **kwargs) -> None:
        """Called when order is filled."""
        # Update average fill price if provided
        if 'fill_price' in kwargs:
            self.price = Decimal(str(kwargs['fill_price']))
    
    async def on_canceled(self, **kwargs) -> None:
        """Called when order is canceled."""
        pass  # Add custom logic here
    
    async def on_rejected(self, **kwargs) -> None:
        """Called when order is rejected."""
        self.reject_reason = kwargs.get('reason', '')

class AdvancedOrderStateMachine(StateMachine):
    """State machine for order lifecycle management with comprehensive transitions."""
    
    def __init__(self, order: 'Order'):
        super().__init__(initial_state=OrderStatus.DRAFT.value)
        self.order = order
        self._setup_transitions()
    
    def _setup_transitions(self) -> None:
        """Configure all valid state transitions and their handlers."""
        # Draft -> Pending Submit
        self.add_transition_handler(
            OrderStatus.DRAFT, 
            OrderStatus.PENDING_SUBMIT,
            self._on_submit
        )
        
        # Pending Submit -> Pending New
        self.add_transition_handler(
            OrderStatus.PENDING_SUBMIT,
            OrderStatus.PENDING_NEW,
            self._on_acknowledge
        )
        
        # Pending New -> Active
        self.add_transition_handler(
            OrderStatus.PENDING_NEW,
            OrderStatus.ACTIVE,
            self._on_activate
        )
        
        # Active -> Partially Filled
        self.add_transition_handler(
            OrderStatus.ACTIVE,
            OrderStatus.PARTIALLY_FILLED,
            self._on_partial_fill
        )
        
        # Partially Filled -> Filled
        self.add_transition_handler(
            OrderStatus.PARTIALLY_FILLED,
            OrderStatus.FILLED,
            self._on_fill
        )
        
        # Active -> Filled (for immediate fills)
        self.add_transition_handler(
            OrderStatus.ACTIVE,
            OrderStatus.FILLED,
            self._on_fill
        )
        
        # Any -> Canceled
        for state in OrderStatus:
            if state not in {OrderStatus.CANCELED, OrderStatus.FILLED, OrderStatus.REJECTED, OrderStatus.EXPIRED}:
                self.add_transition_handler(
                    state,
                    OrderStatus.CANCELED,
                    self._on_cancel
                )
    
    # Transition handlers with metrics and logging
    async def _on_submit(self, **kwargs: Any) -> None:
        """Handle order submission."""
        self.order.submitted_at = datetime.now()
        print(f"Order {self.order.id} submitted for processing")
    
    async def _on_acknowledge(self, **kwargs: Any) -> None:
        """Handle exchange acknowledgment."""
        self.order.acknowledged_at = datetime.now()
        print(f"Order {self.order.id} acknowledged by exchange")
    
    async def _on_activate(self, **kwargs: Any) -> None:
        """Handle order activation."""
        self.order.activated_at = datetime.now()
        print(f"Order {self.order.id} is now active")
    
    async def _on_partial_fill(self, **kwargs: Any) -> None:
        """Handle partial fill."""
        fill_qty = kwargs.get('quantity', Decimal('0'))
        fill_price = kwargs.get('price')
        
        if fill_qty:
            self.order.filled_quantity += fill_qty
            
            # Update average fill price
            if fill_price is not None:
                total_value = (self.order.avg_fill_price or Decimal('0')) * (
                    self.order.filled_quantity - fill_qty
                ) + (fill_price * fill_qty)
                self.order.avg_fill_price = total_value / self.order.filled_quantity
        
        print(
            f"Order {self.order.id} partially filled: "
            f"{fill_qty} @ {fill_price or 'market'}"
        )
    
    async def _on_fill(self, **kwargs: Any) -> None:
        """Handle order fill completion."""
        fill_qty = kwargs.get('quantity')
        fill_price = kwargs.get('price')
        
        if fill_qty:
            self.order.filled_quantity = fill_qty
            if fill_price is not None:
                self.order.avg_fill_price = fill_price
        
        self.order.filled_at = datetime.now()
        print(f"Order {self.order.id} fully filled")
    
    async def _on_cancel(self, **kwargs: Any) -> None:
        """Handle order cancellation."""
        self.order.canceled_at = datetime.now()
        self.order.cancel_reason = kwargs.get('reason')
        print(f"Order {self.order.id} canceled: {self.order.cancel_reason or 'No reason provided'}")
    
    # Validation methods
    def _is_valid_transition(self, old_state: OrderStatus, new_state: OrderStatus) -> bool:
        """Validate state transitions."""
        # Terminal states cannot be changed
        if old_state.is_terminal and old_state != new_state:
            print(
                f"Cannot transition from terminal state {old_state} to {new_state}"
            )
            return False
            
        # Check for valid transitions
        valid_transitions = {
            OrderStatus.DRAFT: {OrderStatus.PENDING_SUBMIT},
            OrderStatus.PENDING_SUBMIT: {OrderStatus.PENDING_NEW, OrderStatus.REJECTED},
            OrderStatus.PENDING_NEW: {OrderStatus.ACTIVE, OrderStatus.REJECTED, OrderStatus.CANCELED},
            OrderStatus.ACTIVE: {
                OrderStatus.PARTIALLY_FILLED, 
                OrderStatus.FILLED, 
                OrderStatus.PENDING_CANCEL,
                OrderStatus.CANCELED,
                OrderStatus.EXPIRED
            },
            OrderStatus.PARTIALLY_FILLED: {
                OrderStatus.PARTIALLY_FILLED,  # For multiple fills
                OrderStatus.FILLED,
                OrderStatus.PENDING_CANCEL,
                OrderStatus.CANCELED,
                OrderStatus.EXPIRED
            },
            OrderStatus.PENDING_CANCEL: {OrderStatus.CANCELED, OrderStatus.REJECTED},
            OrderStatus.PENDING_REPLACE: {OrderStatus.ACTIVE, OrderStatus.REJECTED},
        }
        
        allowed = new_state in valid_transitions.get(old_state, set())
        if not allowed:
            print(
                f"Invalid transition: {old_state} -> {new_state}"
            )
        
        return allowed

class AdvancedOrder(BaseModel):
    """Enhanced order model with state machine integration."""
    
    id: str = Field(default_factory=lambda: f"ord_{uuid.uuid4().hex[:16]}")
    client_order_id: Optional[str] = None
    symbol: str
    side: str  # BUY/SELL
    order_type: str  # MARKET, LIMIT, etc.
    quantity: Decimal
    filled_quantity: Decimal = Decimal('0')
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    time_in_force: str = "GTC"
    status: str = OrderStatus.DRAFT.value
    
    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now())
    updated_at: datetime = Field(default_factory=lambda: datetime.now())
    submitted_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    activated_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    canceled_at: Optional[datetime] = None
    rejected_at: Optional[datetime] = None
    
    # State tracking
    state_machine: Optional[AdvancedOrderStateMachine] = None
    state_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Additional fields
    avg_fill_price: Optional[Decimal] = None
    cancel_reason: Optional[str] = None
    reject_reason: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: str,
        }
    
    def __init__(self, **data):
        super().__init__(**data)
        self.state_machine = AdvancedOrderStateMachine(self)
    
    async def submit(self) -> None:
        """Submit the order for processing."""
        await self.state_machine.transition_to(OrderStatus.PENDING_SUBMIT)
    
    async def acknowledge(self) -> None:
        """Acknowledge order receipt from exchange."""
        await self.state_machine.transition_to(OrderStatus.PENDING_NEW)
    
    async def activate(self) -> None:
        """Activate the order on the exchange."""
        await self.state_machine.transition_to(OrderStatus.ACTIVE)
    
    async def fill(self, quantity: Optional[Decimal] = None, price: Optional[Decimal] = None) -> None:
        """Fill the order or part of it."""
        if quantity is None:
            quantity = self.quantity - self.filled_quantity
        
        remaining = self.quantity - self.filled_quantity
        
        if quantity >= remaining:
            await self.state_machine.transition_to(
                OrderStatus.FILLED,
                quantity=remaining,
                price=price
            )
        else:
            await self.state_machine.transition_to(
                OrderStatus.PARTIALLY_FILLED,
                quantity=quantity,
                price=price
            )
    
    async def cancel(self, reason: Optional[str] = None) -> None:
        """Cancel the order."""
        await self.state_machine.transition_to(
            OrderStatus.CANCELED,
            reason=reason
        )
    
    async def reject(self, reason: str) -> None:
        """Reject the order."""
        await self.state_machine.transition_to(
            OrderStatus.REJECTED,
            reason=reason
        )
    
    # State change hooks
    async def on_state_change(self, event: Dict[str, Any]) -> None:
        """Handle state changes."""
        self.status = event['to_state']
        self.updated_at = datetime.now()
        self.state_history.append(event)
        
        # Emit event if needed
        await self._emit_event('state_change', event)
    
    async def _emit_event(self, event_type: str, data: Any) -> None:
        """Emit an order event."""
        # This would be connected to an event bus in a real implementation
        pass

# Example usage
async def example_usage():
    """Example of using the order state machine."""
    order = AdvancedOrder(
        symbol="BTC/USD",
        side="BUY",
        order_type="LIMIT",
        quantity=Decimal("1.0"),
        price=Decimal("50000.00")
    )
    
    try:
        await order.submit()
        await order.acknowledge()
        await order.activate()
        await order.fill(quantity=Decimal("0.5"), price=Decimal("50001.00"))
        await order.fill(quantity=Decimal("0.5"), price=Decimal("50002.00"))
        print(f"Order {order.id} completed successfully")
    except Exception as e:
        print(f"Order failed: {e}")
        await order.cancel(reason=str(e))
