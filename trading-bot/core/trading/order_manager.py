"""
Advanced Order Management System with state management, modification, and slippage modeling.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum, auto
from typing import (
    Any,
    AsyncGenerator,
    Awaitable,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import numpy as np
from prometheus_client import start_http_server

from core.metrics import metrics
from core.trading.oco_order import OCOOrder, OCOOrderConfig, OCOOrderStatus
from core.trading.order import Order, OrderSide, OrderStatus, OrderType
from core.trading.performance import (
    LockFreeOrderBook,
    ObjectPool,
    OrderBookEntry,
    PerformanceMonitor,
    RateLimiter,
)
from core.utils.trade_logger import (
    TradeEventType,
    log_order_submit,
    log_order_fill,
    log_order_error,
    log_risk_check
)

logger = logging.getLogger(__name__)
T = TypeVar('T')


class OrderManager:
    """High-performance order management system with OCO order support.
    
    Features:
    - Lock-free order book for high throughput
    - Object pooling for reduced GC pressure
    - Fine-grained metrics and monitoring
    - Rate limiting and backpressure handling
    - Advanced OCO order management
    """
    
    def __init__(
        self,
        exchange_adapter: Any,
        max_orders_per_second: int = 100,
        metrics_port: int = 8000,
        enable_metrics: bool = True,
    ):
        """Initialize the OrderManager.
        
        Args:
            exchange_adapter: Adapter for exchange communication
            max_orders_per_second: Maximum order submission rate
            metrics_port: Port for Prometheus metrics server
            enable_metrics: Whether to enable Prometheus metrics collection
        """
        self.exchange = exchange_adapter
        self.orders: Dict[str, Order] = {}
        self.oco_orders: Dict[str, OCOOrder] = {}
        self.order_id_to_oco: Dict[str, str] = {}
        self.order_book = LockFreeOrderBook()
        self.performance_monitor = PerformanceMonitor()
        self.rate_limiter = RateLimiter(
            rate=max_orders_per_second,
            capacity=max_orders_per_second * 2
        )
        
        # Object pools
        self.order_pool = ObjectPool(Order, max_size=10_000)
        self.oco_order_pool = ObjectPool(OCOOrder, max_size=1_000)
        
        # Task management
        self._cleanup_task: Optional[asyncio.Task] = None
        self._is_running = False
        self._lock = asyncio.Lock()
        
        # Start metrics server if enabled
        self.enable_metrics = enable_metrics
        if self.enable_metrics:
            try:
                start_http_server(metrics_port)
                logger.info(f"Started metrics server on port {metrics_port}")
            except Exception as e:
                logger.warning(f"Failed to start metrics server: {e}")
                self.enable_metrics = False
        
    async def start(self) -> None:
        """Start the order manager."""
        if self._is_running:
            return
            
        self._is_running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("OrderManager started")
    
    async def stop(self) -> None:
        """Stop the order manager."""
        self._is_running = False
        
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
                
        logger.info("OrderManager stopped")
    
    async def _cleanup_loop(self) -> None:
        """Background task to clean up old orders and update metrics."""
        while self._is_running:
            try:
                await self._cleanup_old_orders()
                await self._update_metrics()
                await asyncio.sleep(60)  # Run every minute
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}", exc_info=True)
                await asyncio.sleep(5)
    
    async def _cleanup_old_orders(self) -> None:
        """Clean up old completed/canceled orders."""
        now = time.time()
        max_age = 3600  # 1 hour
        
        async with self._lock:
            # Clean up regular orders
            to_remove = [
                order_id for order_id, order in self.orders.items()
                if order.status in (OrderStatus.FILLED, OrderStatus.CANCELED) 
                and (now - order.updated_at.timestamp()) > max_age
            ]
            
            for order_id in to_remove:
                order = self.orders.pop(order_id, None)
                if order:
                    await self.order_pool.release(order)
            
            # Clean up OCO orders
            oco_to_remove = [
                oco_id for oco_id, oco_order in self.oco_orders.items()
                if oco_order.status in (OCOOrderStatus.FILLED, OCOOrderStatus.CANCELED)
                and (now - oco_order.updated_at.timestamp()) > max_age
            ]
            
            for oco_id in oco_to_remove:
                oco_order = self.oco_orders.pop(oco_id, None)
                if oco_order:
                    for order_id in [oco_order.entry_order_id, oco_order.stop_loss_id, oco_order.take_profit_id]:
                        if order_id and order_id in self.order_id_to_oco:
                            del self.order_id_to_oco[order_id]
                    await self.oco_order_pool.release(oco_order)
    
    async def _update_metrics(self) -> None:
        """Update Prometheus metrics."""
        # Count orders by status and symbol
        status_counts: Dict[Tuple[str, str, str], int] = defaultdict(int)
        active_orders: Dict[Tuple[str, str], int] = defaultdict(int)
        
        for order in self.orders.values():
            status = order.status.value.lower()
            status_counts[('regular', status, order.symbol)] += 1
            if order.status not in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED):
                active_orders[(order.symbol, order.side.value.lower())] += 1
        
        for oco_order in self.oco_orders.values():
            status = oco_order.status.value.lower()
            symbol = oco_order.symbol
            status_counts[('oco', status, symbol)] += 1
            if oco_order.status not in (OCOOrderStatus.FILLED, OCOOrderStatus.CANCELED):
                active_orders[(symbol, 'oco')] += 1
        
        # Update metrics
        for (order_type, status, symbol), count in status_counts.items():
            ORDERS_TOTAL.labels(type=order_type, status=status, symbol=symbol).inc(count)
            
        for (symbol, side), count in active_orders.items():
            ACTIVE_ORDERS.labels(symbol=symbol, side=side).set(count)
    
    async def execute_order(self, order: Order) -> None:
        """Execute an order through the exchange."""
        try:
            start_time = time.monotonic()
            
            # Log order submission
            log_order_submit(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side.name,
                order_type=order.order_type.name,
                quantity=float(order.quantity),
                price=float(order.price) if order.price else None,
                time_in_force=order.time_in_force.name if hasattr(order, 'time_in_force') else 'GTC',
                strategy=order.strategy_id or 'unknown'
            )
            
            # Execute the order
            response = await self.exchange.execute_order(order)
            
            # Update order status based on response
            if response.get('status') == 'FILLED':
                order.status = OrderStatus.FILLED
                fill_price = Decimal(str(response.get('price', 0)))
                fill_quantity = Decimal(str(response.get('filled_quantity', order.quantity)))
                
                # Log order fill
                log_order_fill(
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side.name,
                    filled_quantity=float(fill_quantity),
                    fill_price=float(fill_price),
                    commission=float(response.get('commission', 0)),
                    is_partial=fill_quantity < order.quantity,
                    remaining_quantity=float(order.quantity - fill_quantity)
                )
                
            elif response.get('status') == 'PARTIALLY_FILLED':
                order.status = OrderStatus.PARTIALLY_FILLED
                fill_quantity = Decimal(str(response.get('filled_quantity', 0)))
                
                log_order_fill(
                    order_id=order.order_id,
                    symbol=order.symbol,
                    side=order.side.name,
                    filled_quantity=float(fill_quantity),
                    fill_price=float(Decimal(str(response.get('price', 0)))),
                    commission=float(response.get('commission', 0)),
                    is_partial=True,
                    remaining_quantity=float(order.quantity - fill_quantity)
                )
                
            elif response.get('status') == 'REJECTED':
                order.status = OrderStatus.REJECTED
                error_msg = response.get('error', 'Unknown error')
                
                log_order_error(
                    order_id=order.order_id,
                    symbol=order.symbol,
                    error_type='ORDER_REJECTED',
                    error_message=error_msg,
                    order_data={
                        'quantity': float(order.quantity),
                        'price': float(order.price) if order.price else None,
                        'order_type': order.order_type.name,
                        'time_in_force': order.time_in_force.name if hasattr(order, 'time_in_force') else 'GTC'
                    }
                )
                
            # Update metrics
            latency = (time.monotonic() - start_time) * 1000  # in ms
            metrics.order_latency.observe(latency, labels={
                'order_type': order.order_type.name,
                'status': order.status.name
            })
            
        except Exception as e:
            logger.error(f"Error executing order {order.order_id}: {str(e)}", exc_info=True)
            
            log_order_error(
                order_id=order.order_id,
                symbol=order.symbol,
                error_type='EXECUTION_ERROR',
                error_message=str(e),
                order_data={
                    'quantity': float(order.quantity),
                    'price': float(order.price) if order.price else None,
                    'order_type': order.order_type.name,
                    'time_in_force': order.time_in_force.name if hasattr(order, 'time_in_force') else 'GTC'
                }
            )
            
            order.status = OrderStatus.REJECTED
            order.error = str(e)
    
    async def submit_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        quantity: Decimal,
        price: Optional[Decimal] = None,
        client_order_id: Optional[str] = None,
        **kwargs
    ) -> Order:
        """Submit a new order with rate limiting and performance monitoring."""
        start_time = time.monotonic()
        operation = f"submit_order_{order_type.value.lower()}"
        
        try:
            # Apply rate limiting
            if not await self.rate_limiter.acquire(resource='order_submission'):
                raise RuntimeError("Rate limit exceeded for order submission")
            
            # Create and submit order
            order_id = client_order_id or f"order_{uuid.uuid4().hex}"
            
            # Prepare order parameters
            order_params = {
                'order_id': order_id,
                'symbol': symbol,
                'side': side,
                'order_type': order_type,
                'quantity': quantity,
                'limit_price': price,  # Pass price as limit_price
                'status': OrderStatus.NEW
            }
            
            # Update with any additional kwargs, allowing them to override defaults
            order_params.update(kwargs)
            
            # Create the order
            order = await self.order_pool.acquire(**order_params)
            
            # Add to order book
            await self.order_book.add_order(
                OrderBookEntry(
                    order_id=order.id,
                    price=order.limit_price,  # Changed from order.price to order.limit_price
                    quantity=order.quantity
                ),
                is_bid=order.side == OrderSide.BUY,
                symbol=order.symbol
            )
            
            # Store order
            async with self._lock:
                self.orders[order_id] = order
            
            # Log and update metrics
            latency = time.monotonic() - start_time
            await self.performance_monitor.record_latency(operation, latency)
            metrics.order_latency.labels(operation=operation).observe(latency)
            
            log_order_submit(order)
            
            logger.info(
                f"Submitted {order_type.value} {side.value} order: "
                f"{order_id} for {quantity} {symbol} at {price or 'market'}"
            )
            
            return order
            
        except Exception as e:
            latency = time.monotonic() - start_time
            await self.performance_monitor.record_latency(f"{operation}_error", latency)
            logger.error(f"Error submitting order: {e}", exc_info=True)
            raise
    
    async def submit_oco_order(
        self,
        config: OCOOrderConfig,
        client_order_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> OCOOrder:
        """Submit a new OCO (One-Cancels-Other) order with atomic guarantees."""
        start_time = time.monotonic()
        oco_id = client_order_id or f"oco_{uuid.uuid4().hex}"
        
        try:
            # Create OCO order
            oco_order = await self.oco_order_pool.acquire(
                oco_id=oco_id,
                config=config,
                status=OCOOrderStatus.NEW,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                metadata=metadata or {}
            )
            
            # Submit entry order
            entry_order = await self.submit_order(
                symbol=config.symbol,
                side=config.side,
                order_type=config.entry_type,
                quantity=config.quantity,
                price=config.entry_price,
                client_order_id=f"{oco_id}_entry"
            )
            
            # Submit stop loss order (opposite side of entry)
            stop_side = OrderSide.SELL if config.side == OrderSide.BUY else OrderSide.BUY
            stop_order = await self.submit_order(
                symbol=config.symbol,
                side=stop_side,
                order_type=OrderType.STOP_LOSS,
                quantity=config.quantity,
                price=config.stop_loss_price,
                client_order_id=f"{oco_id}_stop"
            )
            
            # Submit take profit order (same side as entry)
            take_profit_order = await self.submit_order(
                symbol=config.symbol,
                side=stop_side,  # Opposite of entry for profit taking
                order_type=OrderType.LIMIT,
                quantity=config.quantity,
                price=config.take_profit_price,
                client_order_id=f"{oco_id}_take"
            )
            
            # Update OCO order with child order IDs
            oco_order.entry_order_id = entry_order.order_id
            oco_order.stop_loss_id = stop_order.order_id
            oco_order.take_profit_id = take_profit_order.order_id
            
            # Store OCO order and update mappings
            async with self._lock:
                self.oco_orders[oco_id] = oco_order
                self.order_id_to_oco[entry_order.order_id] = oco_id
                self.order_id_to_oco[stop_order.order_id] = oco_id
                self.order_id_to_oco[take_profit_order.order_id] = oco_id
            
            # Log and update metrics
            latency = time.monotonic() - start_time
            await self.performance_monitor.record_latency("submit_oco_order", latency)
            ORDER_LATENCY.labels(operation="submit_oco_order").observe(latency)
            
            log_order_submit(oco_order)
            
            logger.info(
                f"Submitted OCO order {oco_id} with entry {entry_order.order_id}, "
                f"stop {stop_order.order_id}, take profit {take_profit_order.order_id}"
            )
            
            return oco_order
            
        except Exception as e:
            latency = time.monotonic() - start_time
            await self.performance_monitor.record_latency("submit_oco_order_error", latency)
            logger.error(f"Error submitting OCO order: {e}", exc_info=True)
            
            # Clean up any partially created orders
            for order_id in [
                getattr(entry_order, 'order_id', None),
                getattr(stop_order, 'order_id', None),
                getattr(take_profit_order, 'order_id', None)
            ]:
                if order_id and order_id in self.orders:
                    await self.cancel_order(order_id)
            
            raise
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order with proper cleanup."""
        start_time = time.monotonic()
        
        try:
            # Check if this is part of an OCO order
            oco_id = self.order_id_to_oco.get(order_id)
            
            if oco_id and oco_id in self.oco_orders:
                # Cancel entire OCO order
                return await self.cancel_oco_order(oco_id)
            
            # Regular order cancellation
            order = self.orders.get(order_id)
            if not order:
                logger.warning(f"Order {order_id} not found")
                return False
                
            # Skip if already completed
            if order.status in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED):
                return False
                
            # Update order status
            order.status = OrderStatus.CANCELED
            order.updated_at = datetime.utcnow()
            
            # Remove from order book if it exists
            await self.order_book.remove_order(
                order_id=order_id,
                price=order.price,
                is_bid=order.side == OrderSide.BUY,
                symbol=order.symbol
            )
            
            # Log and update metrics
            latency = time.monotonic() - start_time
            await self.performance_monitor.record_latency("cancel_order", latency)
            ORDER_LATENCY.labels(operation="cancel_order").observe(latency)
            
            log_order_fill(order, TradeEventType.CANCELED)
            
            logger.info(f"Canceled order {order_id}")
            return True
            
        except Exception as e:
            latency = time.monotonic() - start_time
            await self.performance_monitor.record_latency("cancel_order_error", latency)
            logger.error(f"Error canceling order {order_id}: {e}", exc_info=True)
            return False
    
    async def cancel_oco_order(self, oco_id: str) -> bool:
        """Cancel an OCO order and all its child orders."""
        start_time = time.monotonic()
        
        try:
            oco_order = self.oco_orders.get(oco_id)
            if not oco_order:
                logger.warning(f"OCO order {oco_id} not found")
                return False
                
            # Skip if already completed
            if oco_order.status in (OCOOrderStatus.FILLED, OCOOrderStatus.CANCELED):
                return False
                
            # Cancel all child orders
            success = True
            for order_id in [
                oco_order.entry_order_id,
                oco_order.stop_loss_id,
                oco_order.take_profit_id
            ]:
                if order_id and order_id in self.orders:
                    if not await self.cancel_order(order_id):
                        success = False
            
            # Update OCO order status
            if success:
                oco_order.status = OCOOrderStatus.CANCELED
                oco_order.updated_at = datetime.utcnow()
            
            # Log and update metrics
            latency = time.monotonic() - start_time
            await self.performance_monitor.record_latency("cancel_oco_order", latency)
            ORDER_LATENCY.labels(operation="cancel_oco_order").observe(latency)
            
            log_order_fill(oco_order, TradeEventType.CANCELED)
            
            logger.info(f"Canceled OCO order {oco_id}")
            return success
            
        except Exception as e:
            latency = time.monotonic() - start_time
            await self.performance_monitor.record_latency("cancel_oco_order_error", latency)
            logger.error(f"Error canceling OCO order {oco_id}: {e}", exc_info=True)
            return False
    
    async def get_order(self, order_id: str) -> Optional[Union[Order, OCOOrder]]:
        """Get an order by ID, checking both regular and OCO orders."""
        # Check regular orders
        order = self.orders.get(order_id)
        if order:
            return order
            
        # Check OCO orders
        oco_id = self.order_id_to_oco.get(order_id)
        if oco_id:
            return self.oco_orders.get(oco_id)
            
        return None
    
    async def get_oco_order(self, oco_id: str) -> Optional[OCOOrder]:
        """Get an OCO order by ID."""
        return self.oco_orders.get(oco_id)
    
    async def list_orders(
        self,
        symbol: Optional[str] = None,
        status: Optional[OrderStatus] = None,
        limit: int = 100
    ) -> List[Order]:
        """List orders with optional filtering."""
        result = []
        
        for order in self.orders.values():
            if symbol and order.symbol != symbol:
                continue
            if status and order.status != status:
                continue
                
            result.append(order)
            if len(result) >= limit:
                break
                
        return result
    
    async def list_oco_orders(
        self,
        symbol: Optional[str] = None,
        status: Optional[OCOOrderStatus] = None,
        limit: int = 100
    ) -> List[OCOOrder]:
        """List OCO orders with optional filtering."""
        result = []
        
        for oco_order in self.oco_orders.values():
            if symbol and oco_order.symbol != symbol:
                continue
            if status and oco_order.status != status:
                continue
                
            result.append(oco_order)
            if len(result) >= limit:
                break
                
        return result
    
    async def on_order_update(self, order_update: Dict[str, Any]) -> None:
        """Handle order update from exchange."""
        start_time = time.monotonic()
        order_id = order_update.get('order_id')
        
        if not order_id:
            logger.warning("Received order update without order_id")
            return
            
        try:
            # Update order status
            order = await self.get_order(order_id)
            if not order:
                logger.warning(f"Received update for unknown order: {order_id}")
                return
                
            # Handle regular order update
            if isinstance(order, Order):
                # Update order fields
                order.status = OrderStatus(order_update['status'])
                order.filled_quantity = Decimal(str(order_update.get('filled_quantity', 0)))
                order.avg_fill_price = Decimal(str(order_update.get('avg_fill_price', 0)))
                order.updated_at = datetime.utcnow()
                
                # Handle OCO order logic if this is part of one
                oco_id = self.order_id_to_oco.get(order_id)
                if oco_id and oco_id in self.oco_orders:
                    await self._handle_oco_order_update(oco_id, order_id, order.status)
                
                # Remove from order book if completed
                if order.status in (OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED):
                    await self.order_book.remove_order(
                        order_id=order_id,
                        price=order.price or Decimal('0'),
                        is_bid=order.side == OrderSide.BUY
                    )
                
                log_order_fill(order, TradeEventType.UPDATED)
            
            # Log and update metrics
            latency = time.monotonic() - start_time
            await self.performance_monitor.record_latency("order_update", latency)
            ORDER_LATENCY.labels(operation="order_update").observe(latency)
            
            logger.debug(f"Processed order update for {order_id}: {order_update}")
            
        except Exception as e:
            latency = time.monotonic() - start_time
            await self.performance_monitor.record_latency("order_update_error", latency)
            logger.error(f"Error processing order update {order_id}: {e}", exc_info=True)
    
    async def _handle_oco_order_update(
        self,
        oco_id: str,
        order_id: str,
        new_status: OrderStatus
    ) -> None:
        """Handle order update for an OCO order component."""
        oco_order = self.oco_orders.get(oco_id)
        if not oco_order:
            return
            
        # Update OCO order status based on component order status
        if new_status == OrderStatus.FILLED:
            if order_id == oco_order.entry_order_id:
                # Entry order filled, update OCO status
                oco_order.status = OCOOrderStatus.ENTRY_FILLED
            elif order_id in (oco_order.stop_loss_id, oco_order.take_profit_id):
                # Stop loss or take profit filled, complete OCO
                oco_order.status = OCOOrderStatus.FILLED
                
                # Cancel the other order if it's still active
                other_order_id = (
                    oco_order.take_profit_id 
                    if order_id == oco_order.stop_loss_id 
                    else oco_order.stop_loss_id
                )
                if other_order_id and other_order_id in self.orders:
                    other_order = self.orders[other_order_id]
                    if other_order.status not in (
                        OrderStatus.FILLED, 
                        OrderStatus.CANCELED, 
                        OrderStatus.REJECTED
                    ):
                        await self.cancel_order(other_order_id)
        
        oco_order.updated_at = datetime.utcnow()
        log_order_fill(oco_order, TradeEventType.UPDATED)
