"""
Advanced position management system for tracking and managing trading positions.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Set, Tuple, Callable, Awaitable, Union,
    DefaultDict
)
import uuid

from .order_types import Order, OrderSide, OrderStatus, OrderType, TimeInForce
from .advanced_orders import AdvancedOrder, BracketOrder, OCOOrder, AdvancedOrderType
from core.utils.trade_logger import (
    TradeEventType,
    log_position_update,
    log_order_fill,
    log_risk_check
)

logger = logging.getLogger(__name__)

class PositionStatus(Enum):
    """Position status."""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    LIQUIDATED = "LIQUIDATED"
    HEDGED = "HEDGED"

class Position:
    """Represents a trading position with advanced order support."""
    
    def __init__(
        self,
        symbol: str,
        side: Union[OrderSide, str],
        quantity: Union[Decimal, str, float],
        entry_price: Union[Decimal, str, float],
        entry_time: Optional[datetime] = None,
        position_id: Optional[str] = None,
        account_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        **kwargs
    ):
        """Initialize a position with advanced order support."""
        self.position_id = position_id or f"POS_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        self.symbol = symbol.upper()
        self.side = side if isinstance(side, OrderSide) else OrderSide(side.upper())
        self.quantity = Decimal(quantity) if not isinstance(quantity, Decimal) else quantity
        self.entry_price = Decimal(entry_price) if not isinstance(entry_price, Decimal) else entry_price
        self.entry_time = entry_time or datetime.now(timezone.utc)
        self.exit_price: Optional[Decimal] = None
        self.exit_time: Optional[datetime] = None
        self.realized_pnl: Decimal = Decimal(0)
        self.unrealized_pnl: Decimal = Decimal(0)
        self.commission: Decimal = Decimal(0)
        self.funding_rate: Decimal = Decimal(0)
        self.leverage: int = 1
        self.status: PositionStatus = PositionStatus.OPEN
        self.account_id = account_id
        self.strategy_id = strategy_id
        self.tags: Set[str] = set()
        self.metadata: Dict[str, Any] = kwargs
        self.orders: List[Order] = []
        self.update_time: datetime = self.entry_time
        
        # For tracking partial closes and advanced orders
        self.original_quantity: Decimal = self.quantity
        self.closed_quantity: Decimal = Decimal(0)
        self.avg_entry_price: Decimal = self.entry_price
        self.avg_exit_price: Optional[Decimal] = None
        self.advanced_order_id: Optional[str] = None
        self.parent_order_id: Optional[str] = None
        self.child_orders: List[Order] = []
    
    def update_market_price(self, mark_price: Union[Decimal, str, float]) -> None:
        """Update the position with the current market price and recalculate P&L."""
        mark_price = Decimal(mark_price) if not isinstance(mark_price, Decimal) else mark_price
        
        if self.status != PositionStatus.OPEN:
            return
            
        if self.side == OrderSide.LONG:
            self.unrealized_pnl = (mark_price - self.avg_entry_price) * self.quantity
        else:  # SHORT
            self.unrealized_pnl = (self.avg_entry_price - mark_price) * self.quantity
        
        # Apply leverage if needed
        if self.leverage > 1:
            self.unrealized_pnl *= self.leverage
        
        self.update_time = datetime.now(timezone.utc)
    
    def add_order(self, order: Order) -> None:
        """Add an order to the position and update position state."""
        if order.status not in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]:
            return
            
        self.orders.append(order)
        
        # Update position based on order
        if order.side == self.side:
            # Adding to position
            self._handle_increasing_position(order)
        else:
            # Reducing position
            self._handle_decreasing_position(order)
            
        # Update metadata for advanced orders
        if hasattr(order, 'advanced_order_id'):
            self.advanced_order_id = order.advanced_order_id
        if hasattr(order, 'parent_order_id'):
            self.parent_order_id = order.parent_order_id
            
        self.update_time = datetime.now(timezone.utc)
    
    def _handle_increasing_position(self, order: Order) -> None:
        """Handle logic for increasing an existing position."""
        filled_qty = order.filled_quantity or Decimal(0)
        filled_price = order.filled_price or self.avg_entry_price
        
        # Update average entry price
        total_quantity = self.quantity + filled_qty
        self.avg_entry_price = (
            (self.quantity * self.avg_entry_price + filled_qty * filled_price) / 
            total_quantity
        )
        self.quantity = total_quantity
        
        # Update original quantity if this is a new position
        if self.original_quantity == 0:
            self.original_quantity = filled_qty
    
    def _handle_decreasing_position(self, order: Order) -> None:
        """Handle logic for decreasing an existing position."""
        filled_qty = order.filled_quantity or Decimal(0)
        filled_price = order.filled_price or self.avg_exit_price or Decimal(0)
        
        if filled_qty >= self.quantity:
            # Full close
            self._close_position(filled_price, order.timestamp or datetime.now(timezone.utc))
        else:
            # Partial close
            self._reduce_position(filled_qty, filled_price)
    
    def _close_position(self, exit_price: Decimal, exit_time: datetime) -> None:
        """Fully close the position."""
        self.exit_price = exit_price
        self.exit_time = exit_time
        self.status = PositionStatus.CLOSED
        self.closed_quantity = self.quantity
        self.quantity = Decimal(0)
        
        # Calculate final P&L
        if self.side == OrderSide.LONG:
            self.realized_pnl = (self.exit_price - self.avg_entry_price) * self.original_quantity
        else:  # SHORT
            self.realized_pnl = (self.avg_entry_price - self.exit_price) * self.original_quantity
        
        # Apply leverage if needed
        if self.leverage > 1:
            self.realized_pnl *= self.leverage
    
    def _reduce_position(self, reduce_qty: Decimal, reduce_price: Decimal) -> None:
        """Partially reduce the position."""
        # Calculate P&L for the reduced portion
        if self.side == OrderSide.LONG:
            pnl = (reduce_price - self.avg_entry_price) * reduce_qty
        else:  # SHORT
            pnl = (self.avg_entry_price - reduce_price) * reduce_qty
        
        # Apply leverage if needed
        if self.leverage > 1:
            pnl *= self.leverage
            
        self.realized_pnl += pnl
        self.quantity -= reduce_qty
        self.closed_quantity += reduce_qty
    
    def add_child_order(self, order: Order) -> None:
        """Add a child order (e.g., take-profit, stop-loss) to this position."""
        self.child_orders.append(order)
        
    def get_active_child_orders(self) -> List[Order]:
        """Get all active child orders for this position."""
        return [o for o in self.child_orders if o.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary for serialization."""
        return {
            'position_id': self.position_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': str(self.quantity),
            'entry_price': str(self.entry_price),
            'exit_price': str(self.exit_price) if self.exit_price else None,
            'entry_time': self.entry_time.isoformat(),
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'realized_pnl': str(self.realized_pnl),
            'unrealized_pnl': str(self.unrealized_pnl),
            'status': self.status.value,
            'leverage': self.leverage,
            'account_id': self.account_id,
            'strategy_id': self.strategy_id,
            'advanced_order_id': self.advanced_order_id,
            'parent_order_id': self.parent_order_id,
            'update_time': self.update_time.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Position':
        """Create a Position from a dictionary."""
        position = cls(
            symbol=data['symbol'],
            side=data['side'],
            quantity=Decimal(data['quantity']),
            entry_price=Decimal(data['entry_price']),
            position_id=data['position_id'],
            account_id=data.get('account_id'),
            strategy_id=data.get('strategy_id')
        )
        
        if 'exit_price' in data and data['exit_price'] is not None:
            position.exit_price = Decimal(data['exit_price'])
        if 'exit_time' in data and data['exit_time'] is not None:
            position.exit_time = datetime.fromisoformat(data['exit_time'])
        if 'realized_pnl' in data:
            position.realized_pnl = Decimal(data['realized_pnl'])
        if 'unrealized_pnl' in data:
            position.unrealized_pnl = Decimal(data['unrealized_pnl'])
        if 'status' in data:
            position.status = PositionStatus(data['status'])
        if 'leverage' in data:
            position.leverage = int(data['leverage'])
        if 'advanced_order_id' in data:
            position.advanced_order_id = data['advanced_order_id']
        if 'parent_order_id' in data:
            position.parent_order_id = data['parent_order_id']
        if 'update_time' in data:
            position.update_time = datetime.fromisoformat(data['update_time'])
            
        return position


class PositionManager:
    """Manages multiple trading positions with advanced order support."""
    
    def __init__(self, account_id: Optional[str] = None):
        """Initialize the position manager."""
        self.positions: Dict[str, Position] = {}
        self.account_id = account_id
        self._position_lock = asyncio.Lock()
        self._order_to_position: Dict[str, str] = {}  # order_id -> position_id
        self._symbol_positions: DefaultDict[str, Set[str]] = defaultdict(set)  # symbol -> set of position_ids
        self._strategy_positions: DefaultDict[Optional[str], Set[str]] = defaultdict(set)  # strategy_id -> set of position_ids
        
        # For advanced order tracking
        self._advanced_order_positions: Dict[str, str] = {}  # advanced_order_id -> position_id
    
    async def create_position(
        self,
        symbol: str,
        side: Union[OrderSide, str],
        quantity: Union[Decimal, str, float],
        entry_price: Union[Decimal, str, float],
        entry_time: Optional[datetime] = None,
        position_id: Optional[str] = None,
        account_id: Optional[str] = None,
        strategy_id: Optional[str] = None,
        **kwargs
    ) -> Position:
        """Create and track a new position."""
        position = Position(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            entry_time=entry_time or datetime.utcnow(),
            position_id=position_id,
            account_id=account_id or self.account_id,
            strategy_id=strategy_id,
            **kwargs
        )
        
        async with self._position_lock:
            self.positions[position.position_id] = position
            self._symbol_positions[symbol].add(position.position_id)
            if strategy_id:
                self._strategy_positions[strategy_id].add(position.position_id)
        
        # Log position creation
        log_position_update(
            position_id=position.position_id,
            symbol=position.symbol,
            side=position.side.name,
            size=float(position.quantity),
            entry_price=float(position.entry_price),
            unrealized_pnl=float(position.unrealized_pnl) if hasattr(position, 'unrealized_pnl') else 0.0,
            event_type=TradeEventType.POSITION_OPENED,
            strategy_id=position.strategy_id,
            account_id=position.account_id
        )
        
        return position

    async def update_position(
        self,
        position_id: str,
        order: Optional[Order] = None,
        price: Optional[Union[Decimal, str, float]] = None,
        mark_price: Optional[Union[Decimal, str, float]] = None,
        timestamp: Optional[datetime] = None
    ) -> Optional[Position]:
        """Update a position with an order or market data."""
        async with self._position_lock:
            if position_id not in self.positions:
                return None
                
            position = self.positions[position_id]
            original_status = position.status
            
            # Update position with order if provided
            if order:
                # Map order to position
                self._order_to_position[order.order_id] = position_id
                
                # Update position based on order status
                if order.status in [OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED]:
                    position.add_order(order)
                    
                    # Log order fill
                    log_order_fill(
                        order_id=order.order_id,
                        symbol=order.symbol,
                        side=order.side.name,
                        filled_quantity=float(order.filled_quantity) if hasattr(order, 'filled_quantity') else float(order.quantity),
                        fill_price=float(order.avg_fill_price) if hasattr(order, 'avg_fill_price') else float(order.price or 0),
                        commission=float(order.commission) if hasattr(order, 'commission') else 0.0,
                        is_partial=order.status == OrderStatus.PARTIALLY_FILLED,
                        remaining_quantity=float(order.remaining_quantity) if hasattr(order, 'remaining_quantity') else 0.0
                    )
            
            # Update market price if provided
            if mark_price is not None:
                position.update_market_price(mark_price)
            elif price is not None:
                position.update_market_price(price)
            
            # Log position update if status changed
            if position.status != original_status:
                event_type = {
                    PositionStatus.OPEN: TradeEventType.POSITION_OPENED,
                    PositionStatus.CLOSED: TradeEventType.POSITION_CLOSED,
                    PositionStatus.LIQUIDATED: TradeEventType.POSITION_LIQUIDATED
                }.get(position.status, TradeEventType.POSITION_MODIFIED)
                
                log_position_update(
                    position_id=position.position_id,
                    symbol=position.symbol,
                    side=position.side.name,
                    size=float(position.quantity),
                    entry_price=float(position.entry_price),
                    exit_price=float(position.exit_price) if position.exit_price else None,
                    unrealized_pnl=float(position.unrealized_pnl) if hasattr(position, 'unrealized_pnl') else 0.0,
                    realized_pnl=float(position.realized_pnl) if hasattr(position, 'realized_pnl') else 0.0,
                    event_type=event_type,
                    strategy_id=position.strategy_id,
                    account_id=position.account_id,
                    metadata={
                        'status': position.status.name,
                        'reason': getattr(position, 'close_reason', None)
                    }
                )
            
            # Clean up if position is closed
            if position.status != PositionStatus.OPEN and position_id in self.positions:
                self._cleanup_position(position_id)
            
            return position

    async def close_position(
        self,
        position_id: str,
        price: Union[Decimal, str, float],
        status: PositionStatus = PositionStatus.CLOSED,
        timestamp: Optional[datetime] = None,
        reason: Optional[str] = None,
        **kwargs
    ) -> bool:
        """Close a position with the given price and status."""
        async with self._position_lock:
            if position_id not in self.positions:
                return False
                
            position = self.positions[position_id]
            
            # Skip if already closed
            if position.status != PositionStatus.OPEN:
                return False
            
            # Set close price and status
            position.exit_price = Decimal(str(price)) if not isinstance(price, Decimal) else price
            position.exit_time = timestamp or datetime.utcnow()
            position.status = status
            
            # Calculate P&L
            if hasattr(position, 'calculate_pnl'):
                position.calculate_pnl()
            
            # Log position closure
            log_position_update(
                position_id=position.position_id,
                symbol=position.symbol,
                side=position.side.name,
                size=float(position.quantity),
                entry_price=float(position.entry_price),
                exit_price=float(position.exit_price),
                unrealized_pnl=0.0,  # Should be 0 for closed positions
                realized_pnl=float(position.realized_pnl) if hasattr(position, 'realized_pnl') else 0.0,
                event_type=TradeEventType.POSITION_CLOSED,
                strategy_id=position.strategy_id,
                account_id=position.account_id,
                metadata={
                    'status': status.name,
                    'reason': reason,
                    **kwargs
                }
            )
            
            # Clean up position
            self._cleanup_position(position_id)
            
            return True

    def _cleanup_position(self, position_id: str) -> None:
        """Clean up position references."""
        if position_id not in self.positions:
            return
            
        position = self.positions[position_id]
        
        # Remove from symbol index
        if position.symbol in self._symbol_positions:
            self._symbol_positions[position.symbol].discard(position_id)
            if not self._symbol_positions[position.symbol]:
                del self._symbol_positions[position.symbol]
        
        # Remove from strategy index
        if position.strategy_id in self._strategy_positions:
            self._strategy_positions[position.strategy_id].discard(position_id)
            if not self._strategy_positions[position.strategy_id]:
                del self._strategy_positions[position.strategy_id]
        
        # Remove from order mapping
        for order_id, pos_id in list(self._order_to_position.items()):
            if pos_id == position_id:
                del self._order_to_position[order_id]
        
        # Remove from positions
        del self.positions[position_id]
    
    async def get_position(self, position_id: str) -> Optional[Position]:
        """Get a position by ID."""
        return self.positions.get(position_id)
    
    async def get_positions(
        self,
        symbol: Optional[str] = None,
        status: Optional[Union[PositionStatus, str]] = None,
        strategy_id: Optional[str] = None,
        tag: Optional[str] = None
    ) -> List[Position]:
        """
        Get positions matching the given filters.
        
        Args:
            symbol: Filter by symbol
            status: Filter by status (OPEN, CLOSED, etc.)
            strategy_id: Filter by strategy ID
            tag: Filter by tag
            
        Returns:
            List of matching positions
        """
        # Convert status to enum if it's a string
        if status is not None and isinstance(status, str):
            status = PositionStatus(status.upper())
        
        # Get initial set of position IDs based on the most specific filter
        if symbol and symbol in self._symbol_positions:
            position_ids = set(self._symbol_positions[symbol])
        elif strategy_id is not None and strategy_id in self._strategy_positions:
            position_ids = set(self._strategy_positions[strategy_id])
        else:
            position_ids = set(self.positions.keys())
        
        # Apply filters
        result = []
        for pos_id in position_ids:
            position = self.positions.get(pos_id)
            if not position:
                continue
                
            if status is not None and position.status != status:
                continue
                
            if tag is not None and tag not in position.tags:
                continue
                
            result.append(position)
        
        return result
    
    async def get_open_position(
        self,
        symbol: str,
        side: Optional[Union[OrderSide, str]] = None
    ) -> Optional[Position]:
        """
        Get an open position for the given symbol and optional side.
        
        Args:
            symbol: Trading symbol
            side: Optional side (BUY/SELL)
            
        Returns:
            Matching Position or None if not found
        """
        if side is not None and not isinstance(side, OrderSide):
            side = OrderSide(side.upper())
            
        positions = await self.get_positions(symbol=symbol, status=PositionStatus.OPEN)
        
        if side is None and positions:
            return positions[0]
            
        for position in positions:
            if position.side == side:
                return position
                
        return None
    
    async def get_net_position(self, symbol: str) -> Decimal:
        """
        Get the net position size for a symbol.
        
        Returns:
            Net position size (positive for long, negative for short)
        """
        positions = await self.get_positions(symbol=symbol, status=PositionStatus.OPEN)
        net_position = Decimal(0)
        
        for position in positions:
            if position.side == OrderSide.LONG:
                net_position += position.quantity
            else:  # SHORT
                net_position -= position.quantity
                
        return net_position
    
    async def get_total_pnl(self) -> Dict[str, Decimal]:
        """
        Get total P&L across all positions.
        
        Returns:
            Dictionary with 'realized', 'unrealized', and 'total' P&L
        """
        total_realized = Decimal(0)
        total_unrealized = Decimal(0)
        
        for position in self.positions.values():
            total_realized += position.realized_pnl
            if position.status == PositionStatus.OPEN:
                total_unrealized += position.unrealized_pnl
        
        return {
            'realized': total_realized,
            'unrealized': total_unrealized,
            'total': total_realized + total_unrealized
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position manager to dictionary."""
        return {
            'account_id': self.account_id,
            'positions': {pid: pos.to_dict() for pid, pos in self.positions.items()},
            'order_to_position': self._order_to_position,
            'advanced_order_positions': self._advanced_order_positions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PositionManager':
        """Create a position manager from a dictionary."""
        manager = cls(account_id=data.get('account_id'))
        
        # Restore positions
        for pos_data in data.get('positions', {}).values():
            position = Position.from_dict(pos_data)
            manager.positions[position.position_id] = position
            
            # Rebuild indices
            manager._symbol_positions[position.symbol].add(position.position_id)
            manager._strategy_positions[position.strategy_id].add(position.position_id)
            
            # Rebuild order to position mapping
            for order in position.orders:
                manager._order_to_position[order.order_id] = position.position_id
        
        # Restore advanced order positions
        manager._advanced_order_positions = data.get('advanced_order_positions', {})
        
        return manager
