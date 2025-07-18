"""
Advanced Order Execution Service with smart order routing and execution algorithms.
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP, ROUND_DOWN
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Callable, Awaitable, Any

from ..market.order_book import OrderBook, OrderBookSnapshot
from ..utils.helpers import get_logger
from .order import (
    OrderBase, OrderEvent, OrderEventType, OrderRejectReason, 
    OrderSide, OrderStatus, OrderType, TimeInForce, create_order
)
from .order_states import OrderValidationError

logger = get_logger(__name__)

class ExecutionAlgorithm(Enum):
    """Supported execution algorithms."""
    MARKET = "market"
    LIMIT = "limit"
    TWAP = "twap"       # Time-Weighted Average Price
    VWAP = "vwap"       # Volume-Weighted Average Price
    ICEBERG = "iceberg" # Iceberg order
    SNIPER = "sniper"   # Aggressive immediate execution
    DARK = "dark"       # Dark pool execution


@dataclass
class ExecutionConfig:
    """Configuration for order execution."""
    algorithm: ExecutionAlgorithm = ExecutionAlgorithm.MARKET
    max_slippage: Decimal = Decimal('0.001')  # 0.1%
    max_retries: int = 3
    retry_delay: float = 0.1  # seconds
    aggressive_until_fill: bool = False
    post_only: bool = False
    hidden: bool = False
    iceberg_visible: Optional[Decimal] = None
    time_window: float = 10.0  # seconds for TWAP/VWAP
    slice_interval: float = 1.0  # seconds between slices
    min_slice_size: Decimal = Decimal('0.01')  # minimum size per slice
    max_visible_size: Optional[Decimal] = None
    ioc: bool = False  # Immediate or Cancel
    fok: bool = False  # Fill or Kill
    aon: bool = False  # All or None
    allow_partial: bool = True
    priority: int = 1  # 1-10, higher is more aggressive
    
    def __post_init__(self):
        if self.ioc and self.fok:
            raise ValueError("Cannot specify both IOC and FOK")
        if self.iceberg_visible is not None and self.iceberg_visible <= 0:
            raise ValueError("Iceberg visible size must be positive")


class OrderExecutionService:
    """
    Advanced order execution service that handles order routing, execution algorithms,
    and smart order handling with retries and error recovery.
    """
    
    def __init__(
        self,
        exchange_adapter: Any,  # Exchange-specific adapter
        order_book: OrderBook,
        config: Optional[ExecutionConfig] = None
    ):
        """Initialize the execution service."""
        self.exchange_adapter = exchange_adapter
        self.order_book = order_book
        self.config = config or ExecutionConfig()
        self._active_orders: Dict[str, asyncio.Task] = {}
        self._order_callbacks: Dict[str, Callable[[OrderBase, OrderEvent], Awaitable[None]]] = {}
        self._is_running = False
        self._lock = asyncio.Lock()
        
    async def start(self) -> None:
        """Start the execution service."""
        if self._is_running:
            return
        self._is_running = True
        logger.info("Order Execution Service started")
    
    async def stop(self) -> None:
        """Stop the execution service and cancel all active orders."""
        if not self._is_running:
            return
            
        self._is_running = False
        
        # Cancel all active execution tasks
        for order_id, task in self._active_orders.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        self._active_orders.clear()
        logger.info("Order Execution Service stopped")
    
    async def execute_order(
        self, 
        order: OrderBase, 
        config: Optional[ExecutionConfig] = None,
        callback: Optional[Callable[[OrderBase, OrderEvent], Awaitable[None]]] = None
    ) -> OrderBase:
        """Execute an order using the specified execution algorithm.
        
        Args:
            order: The order to execute
            config: Execution configuration (overrides default)
            callback: Callback for order events
            
        Returns:
            The order with updated status
        """
        if not self._is_running:
            raise RuntimeError("Execution service is not running")
            
        config = config or self.config
        
        # Validate the order
        try:
            order.validate()
        except OrderValidationError as e:
            await self._handle_order_error(order, f"Order validation failed: {e}")
            return order
        
        # Register callback if provided
        if callback:
            self._order_callbacks[order.client_order_id] = callback
        
        # Create and store execution task
        task = asyncio.create_task(self._execute_order_async(order, config))
        self._active_orders[order.client_order_id] = task
        
        # Clean up task when done
        task.add_done_callback(
            lambda t, oid=order.client_order_id: self._cleanup_task(oid, t)
        )
        
        return order
    
    async def _execute_order_async(
        self, 
        order: OrderBase, 
        config: ExecutionConfig
    ) -> None:
        """Execute an order using the specified algorithm."""
        try:
            if config.algorithm == ExecutionAlgorithm.MARKET:
                await self._execute_market(order, config)
            elif config.algorithm == ExecutionAlgorithm.LIMIT:
                await self._execute_limit(order, config)
            elif config.algorithm == ExecutionAlgorithm.TWAP:
                await self._execute_twap(order, config)
            elif config.algorithm == ExecutionAlgorithm.VWAP:
                await self._execute_vwap(order, config)
            elif config.algorithm == ExecutionAlgorithm.ICEBERG:
                await self._execute_iceberg(order, config)
            elif config.algorithm == ExecutionAlgorithm.SNIPER:
                await self._execute_sniper(order, config)
            elif config.algorithm == ExecutionAlgorithm.DARK:
                await self._execute_dark(order, config)
            else:
                raise ValueError(f"Unsupported execution algorithm: {config.algorithm}")
                
        except asyncio.CancelledError:
            logger.info(f"Order {order.client_order_id} execution cancelled")
            await self._handle_order_cancelled(order)
            
        except Exception as e:
            logger.error(f"Error executing order {order.client_order_id}: {e}", exc_info=True)
            await self._handle_order_error(order, f"Execution failed: {e}")
    
    async def _execute_market(self, order: OrderBase, config: ExecutionConfig) -> None:
        """Execute a market order."""
        # For market orders, we want immediate execution at the best available price
        await order.acknowledge()
        
        # Get current market price from order book
        snapshot = await self.order_book.get_snapshot(order.instrument.symbol)
        if not snapshot or not snapshot.bids or not snapshot.asks:
            await self._handle_order_error(order, "No market data available")
            return
        
        # Determine best price based on side
        if order.side == OrderSide.BUY:
            best_price = snapshot.asks[0].price
            price_with_slippage = best_price * (1 + config.max_slippage)
        else:  # SELL
            best_price = snapshot.bids[0].price
            price_with_slippage = best_price * (1 - config.max_slippage)
        
        # Execute at the best price with slippage protection
        executed_price = await self._execute_at_price(
            order, 
            price_with_slippage, 
            aggressive=True,
            config=config
        )
        
        if executed_price is not None:
            await order.fill(executed_price)
        else:
            await self._handle_order_error(order, "Failed to execute market order")
    
    async def _execute_limit(
        self, 
        order: OrderBase, 
        config: ExecutionConfig
    ) -> None:
        """Execute a limit order with optional IOC/FOK/AON handling."""
        if not order.limit_price:
            await self._handle_order_error(order, "Limit price required for limit order")
            return
            
        await order.acknowledge()
        
        # For IOC/FOK orders, we need to check if we can get immediate execution
        if config.ioc or config.fok:
            snapshot = await self.order_book.get_snapshot(order.instrument.symbol)
            if not snapshot:
                await self._handle_order_error(order, "No market data available")
                return
                
            # Check if we can get immediate execution
            if order.side == OrderSide.BUY:
                best_ask = snapshot.asks[0].price if snapshot.asks else None
                can_execute = best_ask and order.limit_price >= best_ask
            else:  # SELL
                best_bid = snapshot.bids[0].price if snapshot.bids else None
                can_execute = best_bid and order.limit_price <= best_bid
                
            if not can_execute:
                await self._handle_order_rejected(
                    order, 
                    "Immediate execution not available",
                    OrderRejectReason.IMMEDIATE_EXECUTION_NOT_AVAILABLE
                )
                return
                
            # For FOK, we need to check if we can get the full quantity
            if config.fok:
                # Simplified check - in practice, we'd need to sum up the order book
                await self._handle_order_error(order, "FOK not fully implemented")
                return
        
        # For regular limit orders, submit to the exchange
        executed_price = await self._execute_at_price(
            order,
            order.limit_price,
            aggressive=config.aggressive_until_fill,
            config=config
        )
        
        if executed_price is not None:
            await order.fill(executed_price)
    
    async def _execute_twap(
        self, 
        order: OrderBase, 
        config: ExecutionConfig
    ) -> None:
        """Execute an order using TWAP (Time-Weighted Average Price) algorithm."""
        await order.acknowledge()
        
        total_quantity = order.quantity
        remaining_quantity = total_quantity
        slice_quantity = max(
            total_quantity / (config.time_window / config.slice_interval),
            config.min_slice_size
        )
        
        start_time = time.time()
        end_time = start_time + config.time_window
        
        while remaining_quantity > 0 and time.time() < end_time:
            # Calculate next slice time
            next_slice_time = time.time() + config.slice_interval
            
            # Determine quantity for this slice
            slice_qty = min(slice_quantity, remaining_quantity)
            if slice_qty < config.min_slice_size and remaining_quantity > 0:
                # If we have less than min size but still have quantity, use the remaining
                slice_qty = remaining_quantity
            
            if slice_qty <= 0:
                break
                
            # Create and execute slice
            slice_order = create_order(
                order_type=OrderType.MARKET,  # or LIMIT with price
                instrument=order.instrument,
                side=order.side,
                quantity=slice_qty,
                client_order_id=f"{order.client_order_id}_slice_{int(time.time())}",
                parent_order_id=order.client_order_id
            )
            
            # Execute slice
            await self._execute_market(slice_order, config)
            
            # Update remaining quantity
            filled_qty = slice_order.filled_quantity if slice_order.filled_quantity else Decimal('0')
            remaining_quantity -= filled_qty
            
            # Wait for next slice
            sleep_time = max(0, next_slice_time - time.time())
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        if remaining_quantity > 0:
            logger.warning(
                f"TWAP order {order.client_order_id} partially filled: "
                f"{total_quantity - remaining_quantity}/{total_quantity}"
            )
    
    async def _execute_vwap(
        self, 
        order: OrderBase, 
        config: ExecutionConfig
    ) -> None:
        """Execute an order using VWAP (Volume-Weighted Average Price) algorithm."""
        # Similar to TWAP but uses volume profiles
        await order.acknowledge()
        logger.warning("VWAP execution not fully implemented")
        # Implementation would analyze volume profiles and adjust slice sizes accordingly
        
    async def _execute_iceberg(
        self, 
        order: OrderBase, 
        config: ExecutionConfig
    ) -> None:
        """Execute an iceberg order (shows only a portion of the total quantity)."""
        await order.acknowledge()
        visible_size = config.iceberg_visible or (order.quantity * Decimal('0.1'))  # Default 10%
        
        remaining_qty = order.quantity
        while remaining_qty > 0:
            # Determine next slice size
            slice_size = min(visible_size, remaining_qty)
            slice_order = create_order(
                order_type=OrderType.LIMIT,
                instrument=order.instrument,
                side=order.side,
                quantity=slice_size,
                limit_price=order.limit_price,
                client_order_id=f"{order.client_order_id}_iceberg_{int(time.time())}",
                parent_order_id=order.client_order_id
            )
            
            # Execute slice
            await self._execute_limit(slice_order, config)
            
            # Update remaining quantity
            filled_qty = slice_order.filled_quantity if slice_order.filled_quantity else Decimal('0')
            remaining_qty -= filled_qty
            
            # Wait a bit before next slice
            await asyncio.sleep(1.0)
    
    async def _execute_sniper(
        self, 
        order: OrderBase, 
        config: ExecutionConfig
    ) -> None:
        """Execute aggressively to capture the current price."""
        await order.acknowledge()
        
        # Get current market price
        snapshot = await self.order_book.get_snapshot(order.instrument.symbol)
        if not snapshot or not snapshot.bids or not snapshot.asks:
            await self._handle_order_error(order, "No market data available")
            return
        
        # Be aggressive with the price to ensure execution
        if order.side == OrderSide.BUY:
            price = snapshot.asks[0].price * (1 + Decimal('0.001'))  # Pay a bit more
        else:  # SELL
            price = snapshot.bids[0].price * (1 - Decimal('0.001'))  # Accept a bit less
        
        # Execute immediately
        executed_price = await self._execute_at_price(
            order,
            price,
            aggressive=True,
            config=config
        )
        
        if executed_price is not None:
            await order.fill(executed_price)
    
    async def _execute_dark(
        self, 
        order: OrderBase, 
        config: ExecutionConfig
    ) -> None:
        """Execute using dark pool matching if available."""
        await order.acknowledge()
        logger.warning("Dark pool execution not implemented")
        # Implementation would interact with dark pool matching engine
    
    async def _execute_at_price(
        self,
        order: OrderBase,
        price: Decimal,
        aggressive: bool,
        config: ExecutionConfig
    ) -> Optional[Decimal]:
        """Execute an order at a specific price."""
        # In a real implementation, this would interact with the exchange
        try:
            # Simulate execution with some randomness
            await asyncio.sleep(random.uniform(0.01, 0.1))
            
            # 90% chance of execution at or better than the limit price
            if random.random() < 0.9:
                # Add some small random slippage
                slippage = random.uniform(-0.0005, 0.0005)  # ±0.05%
                executed_price = price * (1 + Decimal(str(slippage)))
                return executed_price.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)
            
            return None
            
        except Exception as e:
            logger.error(f"Error executing order {order.client_order_id} at {price}: {e}")
            return None
    
    async def _handle_order_error(
        self, 
        order: OrderBase, 
        message: str,
        reject_reason: Optional[OrderRejectReason] = None
    ) -> None:
        """Handle order execution errors."""
        logger.error(f"Order {order.client_order_id} error: {message}")
        
        if reject_reason:
            await order.reject(reason=reject_reason, message=message)
        else:
            await order.cancel(reason=message)
        
        await self._notify_order_event(order, OrderEvent(
            event_type=OrderEventType.REJECTED if reject_reason else OrderEventType.CANCELLED,
            timestamp=datetime.now(timezone.utc),
            reason=reject_reason,
            message=message
        ))
    
    async def _handle_order_rejected(
        self,
        order: OrderBase,
        message: str,
        reject_reason: OrderRejectReason
    ) -> None:
        """Handle order rejection."""
        await self._handle_order_error(order, message, reject_reason)
    
    async def _handle_order_cancelled(self, order: OrderBase) -> None:
        """Handle order cancellation."""
        logger.info(f"Order {order.client_order_id} was cancelled")
        await order.cancel(reason="Execution cancelled")
        
        await self._notify_order_event(order, OrderEvent(
            event_type=OrderEventType.CANCELLED,
            timestamp=datetime.now(timezone.utc),
            message="Execution cancelled"
        ))
    
    async def _notify_order_event(
        self, 
        order: OrderBase, 
        event: OrderEvent
    ) -> None:
        """Notify registered callbacks of an order event."""
        if order.client_order_id in self._order_callbacks:
            try:
                await self._order_callbacks[order.client_order_id](order, event)
            except Exception as e:
                logger.error(f"Error in order callback: {e}", exc_info=True)
    
    def _cleanup_task(self, order_id: str, task: asyncio.Task) -> None:
        """Clean up a completed execution task."""
        if order_id in self._active_orders and self._active_orders[order_id] == task:
            del self._active_orders[order_id]
        
        if order_id in self._order_callbacks:
            del self._order_callbacks[order_id]
        
        # Log any exceptions
        if task.done() and task.exception():
            logger.error(
                f"Error in execution task for order {order_id}: {task.exception()}",
                exc_info=task.exception()
            )
