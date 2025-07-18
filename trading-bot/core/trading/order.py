"""
Advanced Order Management System

This module provides a comprehensive order management system with support for:
- Market, limit, stop, stop-limit orders
- Advanced order types: OCO, Bracket, Trailing Stop
- Order state machine with full lifecycle tracking
- Order validation and risk checks
- Order modification and cancellation
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field, validator


class OrderType(str, Enum):
    """Supported order types."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"
    OCO = "OCO"
    BRACKET = "BRACKET"


class OrderSide(str, Enum):
    """Order side (buy/sell)."""
    BUY = "BUY"
    SELL = "SELL"


class OrderStatus(str, Enum):
    """Order status states."""
    NEW = "NEW"
    PENDING_NEW = "PENDING_NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    PENDING_CANCEL = "PENDING_CANCEL"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    EXPIRED = "EXPIRED"
    SUSPENDED = "SUSPENDED"
    CALCULATED = "CALCULATED"
    DONE_FOR_DAY = "DONE_FOR_DAY"


class TimeInForce(str, Enum):
    """Time in force options."""
    DAY = "DAY"
    GTC = "GTC"  # Good Till Cancel
    IOC = "IOC"   # Immediate or Cancel
    FOK = "FOK"   # Fill or Kill
    GTD = "GTD"   # Good Till Date
    OPG = "OPG"   # At the Opening
    CLS = "CLS"   # At the Close


@dataclass
class OrderLeg:
    """Represents a single leg of a complex order."""
    order_type: OrderType
    side: OrderSide
    quantity: Decimal
    limit_price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    trailing_offset: Optional[Decimal] = None
    trailing_offset_type: str = "amount"  # 'amount' or 'percentage'


class Order(BaseModel):
    """
    Core order class representing a trading order with full lifecycle management.
    
    Attributes:
        id: Unique order identifier
        symbol: Trading pair symbol (e.g., 'BTC/USD')
        side: Buy or sell side
        order_type: Type of order (MARKET, LIMIT, etc.)
        quantity: Order quantity
        limit_price: Optional limit price
        stop_price: Optional stop price
        trailing_offset: Optional trailing offset
        trailing_offset_type: Type of trailing offset ('amount' or 'percentage')
        time_in_force: Time in force setting
        client_order_id: Optional client-defined order ID
        strategy_id: Optional strategy identifier
        oco_group_id: Optional OCO group ID
        parent_order_id: Optional parent order ID
        status: Current order status
        filled_quantity: Quantity filled so far
        avg_fill_price: Average fill price
        created_at: Order creation timestamp
        updated_at: Last update timestamp
        state_history: History of state changes
    """
    # Required fields
    id: str = Field(default_factory=lambda: f"ord_{uuid.uuid4().hex}", description="Unique order identifier")
    symbol: str = Field(..., description="Trading pair symbol")
    side: OrderSide = Field(..., description="Order side (BUY/SELL)")
    order_type: OrderType = Field(..., description="Type of order")
    quantity: Decimal = Field(..., gt=0, description="Order quantity")
    
    # Optional fields with defaults
    limit_price: Optional[Decimal] = Field(None, ge=0, description="Limit price for limit orders")
    stop_price: Optional[Decimal] = Field(None, ge=0, description="Stop price for stop orders")
    trailing_offset: Optional[Decimal] = Field(None, ge=0, description="Trailing offset amount")
    trailing_offset_type: str = Field("amount", pattern="^(amount|percentage)$", 
                                    description="Type of trailing offset ('amount' or 'percentage')")
    time_in_force: TimeInForce = Field(TimeInForce.DAY, description="Time in force setting")
    client_order_id: Optional[str] = Field(None, description="Client-defined order ID")
    strategy_id: Optional[str] = Field(None, description="Strategy identifier")
    
    # OCO/Bracket specific
    oco_group_id: Optional[str] = Field(None, description="OCO group ID")
    oco_orders: List['Order'] = Field(default_factory=list, exclude=True, 
                                    description="List of OCO orders")
    parent_order_id: Optional[str] = Field(None, description="Parent order ID")
    child_orders: List['Order'] = Field(default_factory=list, exclude=True, 
                                      description="List of child orders")
    
    # State tracking
    status: OrderStatus = Field(OrderStatus.NEW, description="Current order status")
    filled_quantity: Decimal = Field(Decimal("0"), ge=0, description="Filled quantity")
    avg_fill_price: Optional[Decimal] = Field(None, ge=0, description="Average fill price")
    created_at: datetime = Field(default_factory=datetime.utcnow, 
                               description="Order creation timestamp")
    updated_at: datetime = Field(default_factory=datetime.utcnow, 
                               description="Last update timestamp")
    
    # State history (excluded from model dump by default)
    state_history: List[Tuple[datetime, OrderStatus, str]] = Field(
        default_factory=list, 
        exclude=True,
        description="History of state changes"
    )
    
    # Internal flags (excluded from model)
    is_modifying: bool = Field(False, exclude=True, description="Modification in progress flag")
    original_values: Dict = Field(default_factory=dict, exclude=True, 
                                description="Original values before modification")

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat(),
            Decimal: lambda v: str(v)
        }

    def __init__(self, **data):
        super().__init__(**data)
        self.state_history = []
        self.is_modifying = False
        self.original_values = {}
        self._record_state_change("Order created")

    def _record_state_change(self, note: str = ""):
        """Record a state change in the order's history."""
        now = datetime.utcnow()
        self.state_history.append((now, self.status, note))
        self.updated_at = now

    def validate(self) -> Tuple[bool, List[str]]:
        """Validate order parameters."""
        errors = []
        
        # Basic validations
        if self.quantity <= 0:
            errors.append("Quantity must be positive")
            
        if self.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and not self.limit_price:
            errors.append(f"Limit price required for {self.order_type}")
            
        if self.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and not self.stop_price:
            errors.append(f"Stop price required for {self.order_type}")
            
        if self.order_type == OrderType.TRAILING_STOP and not self.trailing_offset:
            errors.append("Trailing offset required for TRAILING_STOP")
            
        # Complex order validations
        if self.oco_orders and len(self.oco_orders) != 1:
            errors.append("OCO orders must have exactly one other order")
            
        return len(errors) == 0, errors

    def modify(self, **kwargs) -> bool:
        """Modify order parameters."""
        if self.is_modifying:
            return False
            
        self.is_modifying = True
        
        try:
            # Store original values for rollback
            original_values = {k: getattr(self, k) for k in kwargs.keys()}
            
            # Update fields
            for key, value in kwargs.items():
                if hasattr(self, key) and not key.startswith('_'):
                    setattr(self, key, value)
            
            # Validate modified order
            is_valid, errors = self.validate()
            if not is_valid:
                # Rollback on validation failure
                for k, v in original_values.items():
                    setattr(self, k, v)
                return False
                
            self._record_state_change("Order modified")
            return True
            
        finally:
            self.is_modifying = False

    def _finalize_modification(self, success: bool = True):
        """Finalize the modification process."""
        self.is_modifying = False
        if not success:
            # Restore original values on failure
            for key, value in self.original_values.items():
                setattr(self, key, value)
        self.original_values = {}
        self._record_state_change("Order modified" if success else "Order modification failed")

    def cancel(self) -> bool:
        """Cancel the order."""
        if self.status not in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]:
            return False
            
        self.status = OrderStatus.PENDING_CANCEL
        self._record_state_change("Cancellation requested")
        return True

    def ack_cancel(self) -> bool:
        """Acknowledge order cancellation."""
        if self.status != OrderStatus.PENDING_CANCEL:
            return False
            
        self.status = OrderStatus.CANCELLED
        self._record_state_change("Order cancelled")
        return True

    def fill(self, quantity: Decimal, price: Decimal) -> bool:
        """Process a fill for this order."""
        if self.status not in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]:
            return False
            
        self.filled_quantity += quantity
        
        # Update average fill price
        if self.avg_fill_price is None:
            self.avg_fill_price = price
        else:
            self.avg_fill_price = (
                (self.avg_fill_price * (self.filled_quantity - quantity)) + 
                (price * quantity)
            ) / self.filled_quantity
        
        # Update status
        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
            self._record_state_change("Order filled")
        else:
            self.status = OrderStatus.PARTIALLY_FILLED
            self._record_state_change("Order partially filled")
            
        return True

    def to_dict(self) -> Dict:
        """Convert order to dictionary."""
        data = self.model_dump(exclude_none=True)
        data['side'] = self.side.value
        data['order_type'] = self.order_type.value
        data['status'] = self.status.value
        data['time_in_force'] = self.time_in_force.value
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        
        # Handle Decimal serialization
        for field in ['quantity', 'limit_price', 'stop_price', 'trailing_offset', 'filled_quantity', 'avg_fill_price']:
            if field in data and data[field] is not None:
                if isinstance(data[field], Decimal):
                    data[field] = str(data[field])
                
        return data

    @classmethod
    def create_market_order(
        cls,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        **kwargs
    ) -> 'Order':
        """Create a market order."""
        return cls(
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            **kwargs
        )

    @classmethod
    def create_limit_order(
        cls,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        limit_price: Decimal,
        **kwargs
    ) -> 'Order':
        """Create a limit order."""
        return cls(
            symbol=symbol,
            side=side,
            order_type=OrderType.LIMIT,
            quantity=quantity,
            limit_price=limit_price,
            **kwargs
        )

    @classmethod
    def create_oco_order(
        cls,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        limit_price: Decimal,
        stop_price: Decimal,
        stop_limit_price: Optional[Decimal] = None,
        **kwargs
    ) -> 'Order':
        """Create an OCO (One-Cancels-Other) order."""
        oco_id = f"oco_{uuid.uuid4().hex}"
        
        # Create the main order (limit order)
        main_order = cls.create_limit_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            limit_price=limit_price,
            oco_group_id=oco_id,
            **kwargs
        )
        
        # Create the OCO order (stop or stop-limit)
        if stop_limit_price is not None:
            oco_order = cls(
                symbol=symbol,
                side=side,
                order_type=OrderType.STOP_LIMIT,
                quantity=quantity,
                limit_price=stop_limit_price,
                stop_price=stop_price,
                oco_group_id=oco_id,
                **kwargs
            )
        else:
            oco_order = cls(
                symbol=symbol,
                side=side,
                order_type=OrderType.STOP,
                quantity=quantity,
                stop_price=stop_price,
                oco_group_id=oco_id,
                **kwargs
            )
        
        # Link the orders
        main_order.oco_orders = [oco_order]
        oco_order.oco_orders = [main_order]
        
        return main_order

    @classmethod
    def create_bracket_order(
        cls,
        symbol: str,
        side: OrderSide,
        quantity: Decimal,
        limit_price: Decimal,
        take_profit_price: Decimal,
        stop_loss_price: Decimal,
        **kwargs
    ) -> 'Order':
        """Create a bracket order with take profit and stop loss."""
        bracket_id = f"brk_{uuid.uuid4().hex}"
        
        # Main order (entry)
        main_order = cls.create_limit_order(
            symbol=symbol,
            side=side,
            quantity=quantity,
            limit_price=limit_price,
            parent_order_id=bracket_id,
            **kwargs
        )
        
        # Take profit order
        take_profit = cls.create_limit_order(
            symbol=symbol,
            side=OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY,
            quantity=quantity,
            limit_price=take_profit_price,
            parent_order_id=bracket_id,
            **kwargs
        )
        
        # Stop loss order
        stop_loss = cls(
            symbol=symbol,
            side=OrderSide.SELL if side == OrderSide.BUY else OrderSide.BUY,
            order_type=OrderType.STOP,
            quantity=quantity,
            stop_price=stop_loss_price,
            parent_order_id=bracket_id,
            **kwargs
        )
        
        # Link the orders
        main_order.child_orders = [take_profit, stop_loss]
        
        return main_order
