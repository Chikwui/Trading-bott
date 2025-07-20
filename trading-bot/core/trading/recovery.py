"""
Order recovery and reconciliation service.

Handles recovery from system failures, network partitions, and other
edge cases to ensure order state consistency.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Set, Tuple, TypeVar, Union

from loguru import logger
import aioredis

from core.trading.order_state import Order, OrderStatus
from core.trading.state_machine import StateTransitionError
from core.utils.distributed_lock import DistributedLock

T = TypeVar('T', bound='OrderRecoveryService')

class RecoveryError(Exception):
    """Base class for recovery errors."""
    pass

class OrderRecoveryService:
    """
    Service for recovering and reconciling order state after failures.
    
    Features:
    - Periodic state reconciliation
    - Orphaned order detection
    - State repair utilities
    - Conflict resolution
    """
    
    def __init__(
        self,
        redis: aioredis.Redis,
        exchange_adapter: Any,
        reconciliation_interval: float = 300.0,  # 5 minutes
        max_recovery_attempts: int = 3
    ):
        """Initialize the recovery service."""
        self.redis = redis
        self.exchange = exchange_adapter
        self.reconciliation_interval = reconciliation_interval
        self.max_attempts = max_recovery_attempts
        self._running = False
        self._reconciliation_task = None
        
        # Register Redis callbacks
        self.redis.register_script("""
        local key = KEYS[1]
        local value = ARGV[1]
        local ttl = tonumber(ARGV[2])
        
        if redis.call('setnx', key, value) == 1 then
            redis.call('pexpire', key, ttl)
            return 1
        else
            return 0
        end
        """)
    
    async def start(self) -> None:
        """Start the recovery service."""
        if self._running:
            return
            
        self._running = True
        self._reconciliation_task = asyncio.create_task(self._reconciliation_loop())
        logger.info("Order recovery service started")
    
    async def stop(self) -> None:
        """Stop the recovery service."""
        if not self._running:
            return
            
        self._running = False
        
        if self._reconciliation_task:
            self._reconciliation_task.cancel()
            try:
                await self._reconciliation_task
            except asyncio.CancelledError:
                pass
                
        logger.info("Order recovery service stopped")
    
    async def _reconciliation_loop(self) -> None:
        """Background task for periodic reconciliation."""
        while self._running:
            try:
                await asyncio.sleep(self.reconciliation_interval)
                await self.reconcile_orders()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in reconciliation loop: {e}")
                await asyncio.sleep(60)  # Back off on error
    
    async def reconcile_orders(self) -> int:
        """
        Reconcile local order state with exchange state.
        
        Returns:
            int: Number of orders reconciled
        """
        # In a real implementation, this would:
        # 1. Query all active orders from local storage
        # 2. Fetch current state from exchange
        # 3. Resolve any discrepancies
        # 4. Return count of reconciled orders
        
        # This is a simplified version
        logger.info("Starting order reconciliation")
        reconciled = 0
        
        # Get active orders from local storage
        active_orders = await self._get_active_orders()
        
        for order in active_orders:
            try:
                if await self._needs_reconciliation(order):
                    await self._reconcile_order(order)
                    reconciled += 1
            except Exception as e:
                logger.error(f"Error reconciling order {order.id}: {e}")
        
        logger.info(f"Order reconciliation complete: {reconciled} orders reconciled")
        return reconciled
    
    async def _get_active_orders(self) -> List[Order]:
        """Get all active orders from local storage."""
        # In a real implementation, this would query your database
        # For now, return an empty list as a placeholder
        return []
    
    async def _needs_reconciliation(self, order: Order) -> bool:
        """Check if an order needs reconciliation."""
        # Check if order is in a terminal state
        if order.status in {
            OrderStatus.FILLED,
            OrderStatus.CANCELED,
            OrderStatus.REJECTED,
            OrderStatus.EXPIRED
        }:
            return False
            
        # Check if order is older than threshold
        max_age = datetime.utcnow() - timedelta(minutes=5)
        if order.updated_at < max_age:
            return True
            
        return False
    
    async def _reconcile_order(self, order: Order) -> bool:
        """Reconcile a single order with the exchange."""
        # Use distributed lock to prevent concurrent reconciliation
        lock = DistributedLock(
            redis=self.redis,
            lock_key=f"reconcile:{order.id}",
            timeout=60,  # 1 minute timeout
            blocking=True,
            block_timeout=10  # Wait up to 10 seconds for lock
        )
        
        try:
            if not await lock.acquire():
                logger.warning(f"Could not acquire lock for order {order.id}")
                return False
                
            # Get current state from exchange
            exchange_order = await self.exchange.get_order(order.id)
            
            if not exchange_order:
                # Order not found on exchange, mark as canceled
                if order.status != OrderStatus.CANCELED:
                    await order.update_status(OrderStatus.CANCELED, reason="Not found on exchange")
                return True
                
            # Update local state to match exchange
            if exchange_order.status != order.status:
                await order.update_status(
                    exchange_order.status,
                    exchange_data=exchange_order.raw_data
                )
                
            return True
            
        except Exception as e:
            logger.error(f"Error reconciling order {order.id}: {e}")
            return False
            
        finally:
            await lock.release()
    
    async def repair_order_state(self, order: Order) -> bool:
        """Attempt to repair an order's state."""
        attempts = 0
        
        while attempts < self.max_attempts:
            try:
                # Get current state from exchange
                exchange_order = await self.exchange.get_order(order.id)
                
                if not exchange_order:
                    # Order doesn't exist on exchange
                    if order.status not in {
                        OrderStatus.CANCELED,
                        OrderStatus.REJECTED,
                        OrderStatus.EXPIRED
                    }:
                        await order.update_status(
                            OrderStatus.CANCELED,
                            reason="Order not found on exchange"
                        )
                    return True
                
                # Update local state to match exchange
                if exchange_order.status != order.status:
                    await order.update_status(
                        exchange_order.status,
                        exchange_data=exchange_order.raw_data
                    )
                
                return True
                
            except Exception as e:
                attempts += 1
                logger.warning(
                    f"Attempt {attempts} to repair order {order.id} failed: {e}"
                )
                
                if attempts >= self.max_attempts:
                    logger.error(
                        f"Failed to repair order {order.id} after "
                        f"{self.max_attempts} attempts"
                    )
                    return False
                
                # Exponential backoff
                await asyncio.sleep(2 ** attempts)
        
        return False
