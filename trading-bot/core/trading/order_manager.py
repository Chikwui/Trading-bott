"""
Advanced Order Management System with state management, modification, and slippage modeling.
"""
from __future__ import annotations

import asyncio
import logging
import random
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from enum import Enum
from typing import (
    Any, AsyncGenerator, Awaitable, Callable, DefaultDict, Dict, List, 
    Optional, Set, Tuple, Type, TypeVar, Union, cast
)

from ..market.instrument import Instrument
from ..market.order_book import OrderBook, OrderBookSnapshot
from ..utils.helpers import get_logger
from .advanced_orders import (
    AdvancedOrder, AdvancedOrderManager, BracketOrder, OCOOrder, AdvancedOrderType
)
from .order import (
    OrderBase, OrderEvent, OrderEventType, OrderRejectReason, OrderSide, 
    OrderStatus, OrderType, TimeInForce, create_order, OrderId, OrderGroupId
)
from .order_states import OrderLinkType
from .order_states import OrderValidationError

logger = get_logger(__name__)

T = TypeVar('T', bound='OrderBase')
OrderCallback = Callable[[OrderBase, OrderEvent], Awaitable[None]]
SlippageModel = Callable[[OrderBase, 'OrderBookSnapshot'], Decimal]

class OrderManagerError(Exception):
    """Base exception for order manager errors."""
    pass

class OrderNotFoundError(OrderManagerError):
    """Order not found in the order manager."""
    pass

class OrderAlreadyExistsError(OrderManagerError):
    """Order with the same ID already exists."""
    pass

class OrderRoutingError(OrderManagerError):
    """Error routing order to exchange."""
    pass

class SlippageModelType(Enum):
    """Slippage model types."""
    NONE = "none"
    CONSTANT = "constant"
    PERCENTAGE = "percentage"
    VOLUME_WEIGHTED = "volume_weighted"
    IMPLIED_VOLATILITY = "implied_volatility"
    CUSTOM = "custom"

@dataclass
class SlippageConfig:
    """Configuration for slippage modeling."""
    model_type: SlippageModelType = SlippageModelType.CONSTANT
    constant: Decimal = Decimal('0.0005')  # 0.05% default slippage
    percentage: Decimal = Decimal('0.001')  # 0.1% default
    min_slippage: Decimal = Decimal('0.0001')  # 0.01% min
    max_slippage: Decimal = Decimal('0.05')    # 5% max
    volatility_factor: Decimal = Decimal('1.0')  # Multiplier for vol-based models
    custom_model: Optional[SlippageModel] = None

@dataclass
class OrderManagerConfig:
    """Configuration for the order manager."""
    default_slippage_model: SlippageConfig = field(default_factory=SlippageConfig)
    max_orders_per_symbol: int = 1000
    max_open_orders: int = 10000
    max_order_age: timedelta = timedelta(days=7)
    auto_cancel_expired: bool = True
    enable_slippage: bool = True
    enable_requeue: bool = True
    requeue_delay: float = 0.1  # seconds
    fill_latency: Dict[str, float] = field(
        default_factory=lambda: {
            'min': 0.001,  # 1ms
            'max': 0.1,    # 100ms
            'avg': 0.01    # 10ms
        }
    )

class OrderManager:
    """
    Advanced Order Management System that handles order routing, state management,
    modification, cancellation, expiration, and slippage modeling.
    
    Supports advanced order types including OCO (One-Cancels-Other), Bracket orders,
    and other conditional order types with relationship management.
    """
    
    def __init__(
        self,
        exchange_adapter: Any,  # Would be a proper ExchangeAdapter type in practice
        order_book: Any,  # OrderBook type would be used here
        position_manager: Optional[Any] = None,  # PositionManager instance
        config: Optional[dict] = None
    ):
        """Initialize the order manager."""
        self.exchange_adapter = exchange_adapter
        self.order_book = order_book
        self.config = config or {}
        
        # Order storage and indexing
        self._orders: Dict[str, OrderBase] = {}
        self._order_index: Dict[Tuple[str, str], str] = {}  # (symbol, client_order_id) -> order_id
        self._symbol_orders: DefaultDict[str, Set[str]] = defaultdict(set)
        self._status_orders: DefaultDict[OrderStatus, Set[str]] = defaultdict(set)
        self._group_orders: DefaultDict[OrderGroupId, Set[str]] = defaultdict(set)  # Group ID -> order IDs
        self._parent_children: DefaultDict[str, Set[str]] = defaultdict(set)  # Parent order ID -> child order IDs
        
        # Position management
        self.position_manager = position_manager
        
        # Event handling
        self._event_handlers: List[OrderCallback] = []
        self._pending_orders: asyncio.Queue[Tuple[OrderBase, asyncio.Future]] = asyncio.Queue()
        
        # Advanced order management
        self.advanced_order_manager = AdvancedOrderManager()
        
        # Task tracking
        self._is_running = False
        self._main_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Statistics
        self.stats = {
            'orders_created': 0,
            'orders_filled': 0,
            'orders_canceled': 0,
            'orders_rejected': 0,
            'orders_expired': 0,
            'total_slippage': Decimal('0'),
            'total_volume': Decimal('0')
        }
        
    # ========== Public API ==========
    
    async def start(self) -> None:
        """Start the order manager."""
        if self._is_running:
            return
            
        self._is_running = True
        self._main_task = asyncio.create_task(self._run())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("Order Manager started")
    
    async def stop(self) -> None:
        """Stop the order manager."""
        if not self._is_running:
            return
            
        self._is_running = False
        
        if self._main_task:
            self._main_task.cancel()
            try:
                await self._main_task
            except asyncio.CancelledError:
                pass
                
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
                
        logger.info("Order Manager stopped")
    
    def add_event_handler(self, handler: OrderCallback) -> None:
        """Add an order event handler."""
        if handler not in self._event_handlers:
            self._event_handlers.append(handler)
    
    def remove_event_handler(self, handler: OrderCallback) -> None:
        """Remove an order event handler."""
        if handler in self._event_handlers:
            self._event_handlers.remove(handler)
    
    async def submit_order(self, order: OrderBase) -> OrderBase:
        """Submit a new order to the order manager.
        
        Args:
            order: The order to submit
            
        Returns:
            The submitted order with updated status
            
        Raises:
            OrderAlreadyExistsError: If an order with the same ID already exists
            OrderValidationError: If the order fails validation
        """
        if order.client_order_id in self._orders:
            raise OrderAlreadyExistsError(f"Order with ID {order.client_order_id} already exists")
            
        # Validate the order
        order.validate()
        
        # Add to tracking
        self._add_order_to_indices(order)
        
        # Handle order relationships
        if order.order_group_id:
            self._group_orders[order.order_group_id].add(order.client_order_id)
            
        if order.parent_order_id:
            self._parent_children[order.parent_order_id].add(order.client_order_id)
        
        # Queue for processing
        future = asyncio.Future()
        await self._pending_orders.put((order, future))
        
        # Wait for the order to be processed
        return await future
        
    async def cancel_order(self, order_id: str, cancel_children: bool = False) -> bool:
        """Cancel an existing order.
        
        Args:
            order_id: ID of the order to cancel
            cancel_children: If True, also cancel any child orders
            
        Returns:
            bool: True if the order was canceled, False if it was already in a terminal state
            
        Raises:
            OrderNotFoundError: If the order is not found
        """
        order = self._orders.get(order_id)
        if not order:
            raise OrderNotFoundError(f"Order {order_id} not found")
            
        if order.status.is_terminal():
            return False
            
        # Update status to PENDING_CANCEL
        await order.cancel()
        
        # Update indices
        self._status_orders[OrderStatus.PENDING_CANCEL].add(order_id)
        self._status_orders[order.status].discard(order_id)
        
        # Cancel child orders if requested
        if cancel_children and order_id in self._parent_children:
            for child_id in list(self._parent_children[order_id]):
                try:
                    await self.cancel_order(child_id, cancel_children=True)
                except Exception as e:
                    logger.error(f"Error canceling child order {child_id}: {e}")
        
        return True
    
    async def cancel_orders(
        self, 
        symbol: Optional[str] = None, 
        side: Optional[OrderSide] = None,
        order_type: Optional[OrderType] = None,
        cancel_children: bool = True
    ) -> int:
        """Cancel multiple orders based on filters.
        
        Args:
            symbol: Only cancel orders for this symbol (if None, all symbols)
            side: Only cancel orders with this side (if None, all sides)
            order_type: Only cancel orders of this type (if None, all types)
            cancel_children: If True, also cancel any child orders
            
        Returns:
            int: Number of orders canceled
        """
        count = 0
        order_ids = list(self._orders.keys())
        
        for order_id in order_ids:
            try:
                order = self._orders[order_id]
                
                # Skip if the order doesn't match the filters
                if symbol and order.instrument.symbol != symbol:
                    continue
                if side and order.side != side:
                    continue
                if order_type and order.order_type != order_type:
                    continue
                if order.status.is_terminal():
                    continue
                    
                # Cancel the order
                if await self.cancel_order(order_id, cancel_children=cancel_children):
                    count += 1
                    
            except Exception as e:
                logger.error(f"Error canceling order {order_id}: {e}")
                
        return count
    
    async def get_order(self, order_id: str) -> Optional[OrderBase]:
        """Get an order by ID.
        
        Args:
            order_id: The ID of the order to retrieve
            
        Returns:
            The order if found, None otherwise
        """
        return self._orders.get(order_id)
        
    async def get_orders(
        self, 
        symbol: Optional[str] = None, 
        status: Optional[OrderStatus] = None,
        order_type: Optional[OrderType] = None,
        side: Optional[OrderSide] = None
    ) -> List[OrderBase]:
        """Get orders matching the specified filters.
        
        Args:
            symbol: Filter by symbol
            status: Filter by status
            order_type: Filter by order type
            side: Filter by order side
            
        Returns:
            List of matching orders
        """
        result = []
        
        # Optimize by using indices when possible
        if status is not None and status in self._status_orders:
            order_ids = self._status_orders[status]
        else:
            order_ids = self._orders.keys()
            
        for order_id in order_ids:
            order = self._orders.get(order_id)
            if not order:
                continue
                
            if symbol and order.instrument.symbol != symbol:
                continue
                
            if order_type and order.order_type != order_type:
                continue
                
            if side and order.side != side:
                continue
                
            result.append(order)
            
        return result
    
    async def get_order_group(self, group_id: OrderGroupId) -> List[OrderBase]:
        """Get all orders in a group.
        
        Args:
            group_id: The group ID to retrieve orders for
            
        Returns:
            List of orders in the group
        """
        if group_id not in self._group_orders:
            return []
            
        return [self._orders[order_id] for order_id in self._group_orders[group_id] 
                if order_id in self._orders]
    
    async def get_child_orders(self, parent_order_id: str) -> List[OrderBase]:
        """Get all child orders of a parent order.
        
        Args:
            parent_order_id: The ID of the parent order
            
        Returns:
            List of child orders
        """
        if parent_order_id not in self._parent_children:
            return []
            
        return [self._orders[order_id] for order_id in self._parent_children[parent_order_id]
                if order_id in self._orders]
    
    async def create_oco_order(
        self,
        order1: OrderBase,
        order2: OrderBase,
        **kwargs
    ) -> OCOOrder:
        """
        Create a One-Cancels-Other (OCO) order.
        
        Args:
            order1: First order in the OCO pair
            order2: Second order in the OCO pair
            **kwargs: Additional arguments for the OCO order
            
        Returns:
            The created OCOOrder instance
        """
        # Validate orders
        if order1.instrument.symbol != order2.instrument.symbol:
            raise ValueError("Both orders in OCO pair must be for the same symbol")
            
        if order1.side == order2.side:
            raise ValueError("OCO orders must be on opposite sides")
            
        if order1.quantity != order2.quantity:
            raise ValueError("OCO orders must have the same quantity")
            
        # Create the OCO order
        oco_order = await self.advanced_order_manager.create_oco_order(order1, order2)
        
        # Submit both orders
        await self.submit_order(order1)
        await self.submit_order(order2)
        
        return oco_order
    
    async def create_bracket_order(
        self,
        entry: OrderBase,
        take_profit: OrderBase,
        stop_loss: OrderBase,
        **kwargs
    ) -> BracketOrder:
        """
        Create a bracket order (entry with take-profit and stop-loss).
        
        Args:
            entry: Entry order
            take_profit: Take-profit order
            stop_loss: Stop-loss order
            **kwargs: Additional arguments for the bracket order
            
        Returns:
            The created BracketOrder instance
        """
        # Validate orders
        if entry.instrument.symbol != take_profit.instrument.symbol or entry.instrument.symbol != stop_loss.instrument.symbol:
            raise ValueError("All orders in bracket must be for the same symbol")
            
        if entry.side not in [OrderSide.BUY, OrderSide.SELL]:
            raise ValueError("Entry order must be a BUY or SELL")
            
        if take_profit.side == entry.side or stop_loss.side == entry.side:
            raise ValueError("Take-profit and stop-loss must be on the opposite side of entry")
            
        if take_profit.quantity != entry.quantity or stop_loss.quantity != entry.quantity:
            raise ValueError("All orders in bracket must have the same quantity")
            
        # Create the bracket order
        bracket_order = await self.advanced_order_manager.create_bracket_order(
            entry, take_profit, stop_loss
        )
        
        # Submit the entry order first
        await self.submit_order(entry)
        
        # Take-profit and stop-loss will be submitted when entry is filled
        
        return bracket_order
    
    # ========== Internal Methods ==========
    
    def _add_order_to_indices(self, order: OrderBase) -> None:
        """Add an order to all tracking indices."""
        self._orders[order.client_order_id] = order
        self._symbol_orders[order.instrument.symbol].add(order.client_order_id)
        self._status_orders[order.status].add(order.client_order_id)
        self._order_index[(order.instrument.symbol, order.client_order_id)] = order.client_order_id
        
        if order.order_group_id:
            self._group_orders[order.order_group_id].add(order.client_order_id)
    
    def _remove_order_from_indices(self, order_id: str) -> None:
        """Remove an order from all tracking indices.
        
        Args:
            order_id: The ID of the order to remove
        """
        if order_id not in self._orders:
            return
            
        order = self._orders[order_id]
        
        # Remove from symbol index
        self._symbol_orders[order.instrument.symbol].discard(order_id)
        
        # Remove from status index
        for status_orders in self._status_orders.values():
            status_orders.discard(order_id)
            
        # Remove from order index
        self._order_index.pop((order.instrument.symbol, order_id), None)
        
        # Remove from group index if it's part of a group
        if order.order_group_id and order_id in self._group_orders[order.order_group_id]:
            self._group_orders[order.order_group_id].discard(order_id)
            if not self._group_orders[order.order_group_id]:
                del self._group_orders[order.order_group_id]
                
        # Remove from parent-child index
        if order_id in self._parent_children:
            # If this order has children, they become orphaned
            for child_id in self._parent_children[order_id]:
                if child_id in self._orders:
                    self._orders[child_id].parent_order_id = None
            del self._parent_children[order_id]
            
        # Remove from main orders dictionary
        del self._orders[order_id]
    
    async def _apply_slippage(self, order: OrderBase) -> None:
        """Apply slippage to an order's price if applicable.
        
        Args:
            order: The order to apply slippage to
            
        Note:
            Only applies to limit orders with a price and when slippage is enabled
        """
        if not self.config.enable_slippage:
            return
            
        if order.order_type == OrderType.MARKET:
            return  # Market orders don't need slippage applied
            
        if not order.limit_price:
            return  # No price to adjust
            
        # Get order book snapshot
        snapshot = await self.order_book.get_snapshot(order.instrument.symbol)
        if not snapshot:
            return
            
        # Calculate slippage based on model
        slippage = self._calculate_slippage(order, snapshot)
        if not slippage:
            return
            
        # Adjust limit price
        if order.side == OrderSide.BUY:
            new_price = order.limit_price * (1 + slippage)
        else:  # SELL
            new_price = order.limit_price * (1 - slippage)
        
        # Round to appropriate precision
        tick_size = order.instrument.tick_size or Decimal('0.00000001')
        new_price = (new_price / tick_size).quantize(1, rounding=ROUND_UP) * tick_size
        
        # Log the price adjustment
        logger.debug(
            f"Applied slippage of {slippage*100:.4f}% to order {order.client_order_id}: "
            f"{order.limit_price} -> {new_price}"
        )
        
        # Update order price
        order.limit_price = new_price
        self.stats['total_slippage'] += abs(order.quantity * slippage)
    
    def _calculate_slippage(
        self,
        order: OrderBase,
        snapshot: 'OrderBookSnapshot'
    ) -> Decimal:
        """Calculate slippage for an order."""
        config = self.config.default_slippage_model
        
        if config.model_type == SlippageModelType.NONE:
            return Decimal('0')
            
        elif config.model_type == SlippageModelType.CONSTANT:
            return config.constant
            
        elif config.model_type == SlippageModelType.PERCENTAGE:
            return config.percentage
            
        elif config.model_type == SlippageModelType.VOLUME_WEIGHTED:
            # Calculate volume-weighted slippage based on order book depth
            if order.side == OrderSide.BUY:
                levels = snapshot.asks
            else:  # SELL
                levels = snapshot.bids
                
            if not levels:
                return config.min_slippage
                
            total_volume = sum(level.quantity for level in levels)
            if total_volume <= 0:
                return config.min_slippage
                
            # Calculate impact
            remaining_qty = order.quantity
            impact = Decimal('0')
            
            for level in levels:
                if remaining_qty <= 0:
                    break
                    
                fill_qty = min(remaining_qty, level.quantity)
                price_impact = abs(level.price - levels[0].price) / levels[0].price
                impact += price_impact * (fill_qty / order.quantity)
                remaining_qty -= fill_qty
                
            # Apply volatility factor
            impact *= config.volatility_factor
            
            # Apply bounds
            return max(config.min_slippage, min(impact, config.max_slippage))
            
        elif config.model_type == SlippageModelType.IMPLIED_VOLATILITY:
            # Simplified IV-based slippage model
            # In a real implementation, this would use options market data
            iv = Decimal('0.3')  # Placeholder for implied volatility
            slippage = iv * Decimal('0.1')  # Simple scaling
            return max(config.min_slippage, min(slippage, config.max_slippage))
            
        elif config.model_type == SlippageModelType.CUSTOM and config.custom_model:
            return config.custom_model(order, snapshot)
            
        return Decimal('0')
    
    async def _on_order_event(self, order: OrderBase, event: OrderEvent) -> None:
        """Handle order events."""
        # Update statistics
        if event.event_type == OrderEventType.FILLED:
            self.stats['orders_filled'] += 1
            self.stats['total_volume'] += order.filled_quantity
        elif event.event_type == OrderEventType.CANCELED:
            self.stats['orders_canceled'] += 1
        elif event.event_type == OrderEventType.REJECTED:
            self.stats['orders_rejected'] += 1
        elif event.event_type == OrderEventType.EXPIRED:
            self.stats['orders_expired'] += 1
        
        # Forward to registered handlers
        for handler in self._event_handlers:
            try:
                await handler(order, event)
            except Exception as e:
                logger.error(f"Error in order event handler: {e}", exc_info=True)
    
    async def _process_order_fill(self, order: OrderBase) -> None:
        """Process an order fill and update related positions."""
        if not order.filled_quantity or order.filled_quantity <= 0:
            return
            
        # Update position if position manager is available
        if self.position_manager:
            try:
                position = await self.position_manager.update_position(
                    position_id=order.position_id,
                    order=order,
                    price=order.filled_price
                )
                
                # If this is a new position (no position_id was set)
                if not order.position_id and position:
                    order.position_id = position.position_id
                    
            except Exception as e:
                logger.error(f"Error updating position for order {order.client_order_id}: {e}", exc_info=True)
        
        # Update statistics
        self.stats['orders_filled'] += 1
        self.stats['total_volume'] += order.filled_quantity
        
        # Notify listeners
        await self._notify_order_updated(order, 'FILLED', {
            'filled_quantity': order.filled_quantity,
            'filled_price': order.filled_price,
            'remaining_quantity': order.remaining_quantity
        })
        
        # Handle advanced order relationships
        if hasattr(order, 'advanced_order_id') and order.advanced_order_id:
            await self._handle_advanced_order_fill(order)

    async def _handle_advanced_order_fill(self, order: OrderBase) -> None:
        """Handle order fill for advanced orders (OCO, Bracket, etc.)."""
        if not hasattr(order, 'advanced_order_id') or not order.advanced_order_id:
            return
            
        try:
            # Get the advanced order
            adv_order = self.advanced_order_manager.get_order(order.advanced_order_id)
            if not adv_order:
                return
                
            # Handle different advanced order types
            if isinstance(adv_order, OCOOrder):
                await self._handle_oco_order_fill(adv_order, order)
            elif isinstance(adv_order, BracketOrder):
                await self._handle_bracket_order_fill(adv_order, order)
                
        except Exception as e:
            logger.error(f"Error handling advanced order fill: {e}", exc_info=True)

    async def _handle_oco_order_fill(self, oco_order: OCOOrder, filled_order: OrderBase) -> None:
        """Handle order fill for an OCO (One-Cancels-Other) order."""
        # Cancel the other order in the OCO pair
        other_order_id = None
        if filled_order.order_id == oco_order.order1_id:
            other_order_id = oco_order.order2_id
        elif filled_order.order_id == oco_order.order2_id:
            other_order_id = oco_order.order1_id
            
        if other_order_id:
            try:
                await self.cancel_order(other_order_id)
            except Exception as e:
                logger.warning(f"Failed to cancel OCO counterparty order {other_order_id}: {e}")

    async def _handle_bracket_order_fill(self, bracket_order: BracketOrder, filled_order: OrderBase) -> None:
        """Handle order fill for a Bracket order."""
        # If the entry order was filled, activate the take-profit and stop-loss orders
        if filled_order.order_id == bracket_order.entry_order_id:
            try:
                # Submit take-profit order
                tp_order = bracket_order.take_profit_order
                if tp_order and tp_order.status == OrderStatus.NEW:
                    await self.submit_order(tp_order)
                    
                # Submit stop-loss order
                sl_order = bracket_order.stop_loss_order
                if sl_order and sl_order.status == OrderStatus.NEW:
                    await self.submit_order(sl_order)
                    
            except Exception as e:
                logger.error(f"Failed to submit bracket child orders: {e}", exc_info=True)
        
        # If either take-profit or stop-loss was filled, cancel the other one
        elif filled_order.order_id in [bracket_order.take_profit_order_id, bracket_order.stop_loss_order_id]:
            other_order_id = (
                bracket_order.stop_loss_order_id 
                if filled_order.order_id == bracket_order.take_profit_order_id
                else bracket_order.take_profit_order_id
            )
            
            if other_order_id:
                try:
                    other_order = await self.get_order(other_order_id)
                    if other_order and other_order.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]:
                        await self.cancel_order(other_order_id)
                except Exception as e:
                    logger.warning(f"Failed to cancel bracket counterparty order {other_order_id}: {e}")

    async def _run(self) -> None:
        """Main order processing loop."""
        while self._is_running:
            try:
                # Process order events
                await asyncio.sleep(0.01)  # Small delay to prevent tight loop
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in order manager main loop: {e}", exc_info=True)
                await asyncio.sleep(1)  # Prevent tight loop on errors
    
    async def _cleanup_loop(self) -> None:
        """Clean up expired and completed orders."""
        while self._is_running:
            try:
                now = datetime.now(timezone.utc)
                to_remove = []
                
                # Find orders to clean up
                for order in self._orders.values():
                    # Skip active orders
                    if order.status.is_active():
                        continue
                        
                    # Check if order is too old
                    age = now - order.updated_at
                    if age > self.config.max_order_age:
                        to_remove.append(order)
                        continue
                        
                    # Check for expired orders
                    if (self.config.auto_cancel_expired and 
                        order.time_in_force == TimeInForce.GTD and 
                        order.expire_time and 
                        now >= order.expire_time):
                        await order.expire()
                
                # Remove old orders
                for order in to_remove:
                    self._unregister_order(order)
                    logger.debug(f"Removed old order: {order.client_order_id}")
                
                # Sleep for a while before next cleanup
                await asyncio.sleep(60)  # Check every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in order cleanup loop: {e}", exc_info=True)
                await asyncio.sleep(5)  # Prevent tight loop on errors
