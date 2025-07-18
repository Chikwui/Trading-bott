"""
Advanced trade reconciliation system to ensure consistency between local and broker records.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Set, Tuple, Callable, Awaitable, Union,
    DefaultDict
)
import difflib
import hashlib
import json

from .order_types import Order, OrderStatus, OrderSide, OrderType
from .position_manager import Position, PositionStatus, PositionManager

logger = logging.getLogger(__name__)

class ReconciliationStatus(Enum):
    """Reconciliation status."""
    PENDING = "PENDING"           # Initial state, not yet reconciled
    MATCHED = "MATCHED"           # Local and broker records match
    MISSING_LOCAL = "MISSING_LOCAL"       # Order exists on broker but not locally
    MISSING_BROKER = "MISSING_BROKER"     # Order exists locally but not on broker
    DISCREPANCY = "DISCREPANCY"   # Order exists in both but with discrepancies
    RESOLVED = "RESOLVED"         # Discrepancy has been resolved

class DiscrepancyType(Enum):
    """Types of discrepancies that can occur."""
    STATUS_MISMATCH = "STATUS_MISMATCH"
    QUANTITY_MISMATCH = "QUANTITY_MISMATCH"
    PRICE_MISMATCH = "PRICE_MISMATCH"
    SIDE_MISMATCH = "SIDE_MISMATCH"
    SYMBOL_MISMATCH = "SYMBOL_MISMATCH"
    DUPLICATE_ORDER = "DUPLICATE_ORDER"
    MISSING_EXECUTION = "MISSING_EXECUTION"
    UNEXPECTED_EXECUTION = "UNEXPECTED_EXECUTION"
    OTHER = "OTHER"

@dataclass
class ReconciliationResult:
    """Result of a reconciliation operation."""
    status: ReconciliationStatus
    local_order: Optional[Order] = None
    broker_order: Optional[Dict[str, Any]] = None
    discrepancies: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved: bool = False
    resolution: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_discrepancy(
        self,
        discrepancy_type: DiscrepancyType,
        field: str,
        local_value: Any,
        broker_value: Any,
        severity: str = "WARNING"
    ) -> None:
        """Add a discrepancy to the result."""
        self.discrepancies.append({
            'type': discrepancy_type.value,
            'field': field,
            'local_value': str(local_value) if isinstance(local_value, Decimal) else local_value,
            'broker_value': str(broker_value) if isinstance(broker_value, Decimal) else broker_value,
            'severity': severity,
            'timestamp': datetime.now(timezone.utc).isoformat()
        })
        
        if self.status != ReconciliationStatus.DISCREPANCY:
            self.status = ReconciliationStatus.DISCREPANCY
    
    def resolve(self, resolution: str) -> None:
        """Mark the discrepancy as resolved."""
        self.status = ReconciliationStatus.RESOLVED
        self.resolved = True
        self.resolution = resolution
        self.timestamp = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'status': self.status.value,
            'local_order': self.local_order.to_dict() if self.local_order else None,
            'broker_order': self.broker_order,
            'discrepancies': self.discrepancies,
            'timestamp': self.timestamp.isoformat(),
            'resolved': self.resolved,
            'resolution': self.resolution,
            'metadata': self.metadata
        }

class TradeReconciler:
    """
    Advanced trade reconciliation system that ensures consistency between
    local trading records and broker records.
    """
    
    def __init__(
        self,
        position_manager: PositionManager,
        reconciliation_window: int = 7,  # days
        max_retries: int = 3,
        retry_delay: float = 1.0,
        tolerance: float = 1e-8  # For decimal comparisons
    ):
        """Initialize the trade reconciler."""
        self.position_manager = position_manager
        self.reconciliation_window = reconciliation_window
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.tolerance = Decimal(str(tolerance))
        self.reconciliation_history: List[ReconciliationResult] = []
        self._reconciliation_lock = asyncio.Lock()
        
        # Cache for storing broker orders to reduce API calls
        self._broker_orders_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_expiry: Dict[str, float] = {}
        self._cache_ttl = 300  # 5 minutes
    
    async def reconcile_orders(
        self,
        local_orders: List[Order],
        broker_orders: List[Dict[str, Any]],
        force_full: bool = False
    ) -> List[ReconciliationResult]:
        """
        Reconcile local orders with broker orders.
        
        Args:
            local_orders: List of local order objects
            broker_orders: List of order dictionaries from the broker
            force_full: If True, perform a full reconciliation even if not needed
            
        Returns:
            List of reconciliation results
        """
        results: List[ReconciliationResult] = []
        
        # Update broker orders cache
        self._update_broker_orders_cache(broker_orders)
        
        # Create mappings for faster lookups
        local_orders_by_id = {order.client_order_id: order for order in local_orders}
        broker_orders_by_id = {order['client_order_id']: order for order in broker_orders if 'client_order_id' in order}
        
        # Check for missing or duplicate orders
        local_ids = set(local_orders_by_id.keys())
        broker_ids = set(broker_orders_by_id.keys())
        
        # Find orders that exist locally but not on the broker
        missing_on_broker = local_ids - broker_ids
        for order_id in missing_on_broker:
            result = ReconciliationResult(
                status=ReconciliationStatus.MISSING_BROKER,
                local_order=local_orders_by_id[order_id],
                broker_order=None
            )
            result.add_discrepancy(
                DiscrepancyType.MISSING_EXECUTION,
                'order_id',
                order_id,
                None,
                severity="ERROR"
            )
            results.append(result)
        
        # Find orders that exist on the broker but not locally
        missing_locally = broker_ids - local_ids
        for order_id in missing_locally:
            result = ReconciliationResult(
                status=ReconciliationStatus.MISSING_LOCAL,
                local_order=None,
                broker_order=broker_orders_by_id[order_id]
            )
            result.add_discrepancy(
                DiscrepancyType.UNEXPECTED_EXECUTION,
                'order_id',
                None,
                order_id,
                severity="ERROR"
            )
            results.append(result)
        
        # Compare orders that exist in both places
        common_ids = local_ids & broker_ids
        for order_id in common_ids:
            local_order = local_orders_by_id[order_id]
            broker_order = broker_orders_by_id[order_id]
            
            result = await self._compare_orders(local_order, broker_order)
            results.append(result)
            
            # If there are discrepancies, try to resolve them
            if result.status == ReconciliationStatus.DISCREPANCY:
                await self._resolve_discrepancies(result)
        
        # Store results in history
        async with self._reconciliation_lock:
            self.reconciliation_history.extend(results)
            # Keep only the most recent history within the window
            cutoff = datetime.now(timezone.utc) - timedelta(days=self.reconciliation_window)
            self.reconciliation_history = [
                r for r in self.reconciliation_history
                if r.timestamp >= cutoff
            ]
        
        return results
    
    async def _compare_orders(
        self,
        local_order: Order,
        broker_order: Dict[str, Any]
    ) -> ReconciliationResult:
        """Compare a local order with a broker order."""
        result = ReconciliationResult(
            status=ReconciliationStatus.MATCHED,
            local_order=local_order,
            broker_order=broker_order
        )
        
        # Check basic fields
        self._check_field_equality(
            result, 'symbol',
            local_order.symbol,
            broker_order.get('symbol'),
            DiscrepancyType.SYMBOL_MISMATCH
        )
        
        self._check_field_equality(
            result, 'side',
            local_order.side.value,
            broker_order.get('side'),
            DiscrepancyType.SIDE_MISMATCH
        )
        
        self._check_decimal_equality(
            result, 'quantity',
            local_order.quantity,
            self._safe_get_decimal(broker_order, 'quantity'),
            DiscrepancyType.QUANTITY_MISMATCH
        )
        
        self._check_decimal_equality(
            result, 'price',
            local_order.price,
            self._safe_get_decimal(broker_order, 'price'),
            DiscrepancyType.PRICE_MISMATCH
        )
        
        # Check status
        broker_status = self._map_broker_status(broker_order.get('status'))
        if local_order.status != broker_status:
            result.add_discrepancy(
                DiscrepancyType.STATUS_MISMATCH,
                'status',
                local_order.status.value,
                broker_order.get('status'),
                severity="WARNING" if local_order.status.is_active() and broker_status.is_active() else "ERROR"
            )
        
        # Check execution details for filled orders
        if local_order.status == OrderStatus.FILLED or broker_status == OrderStatus.FILLED:
            self._check_decimal_equality(
                result, 'executed_quantity',
                local_order.executed_quantity,
                self._safe_get_decimal(broker_order, 'executed_quantity'),
                DiscrepancyType.QUANTITY_MISMATCH
            )
            
            self._check_decimal_equality(
                result, 'avg_price',
                local_order.avg_price,
                self._safe_get_decimal(broker_order, 'avg_price'),
                DiscrepancyType.PRICE_MISMATCH
            )
        
        return result
    
    async def _resolve_discrepancies(self, result: ReconciliationResult) -> None:
        """Attempt to resolve discrepancies between local and broker orders."""
        if not result.discrepancies:
            return
        
        # Group discrepancies by type
        discrepancy_types = {d['type'] for d in result.discrepancies}
        
        # Handle different types of discrepancies
        if DiscrepancyType.STATUS_MISMATCH in discrepancy_types:
            await self._resolve_status_mismatch(result)
        
        if DiscrepancyType.QUANTITY_MISMATCH in discrepancy_types:
            await self._resolve_quantity_mismatch(result)
        
        if DiscrepancyType.PRICE_MISMATCH in discrepancy_types:
            await self._resolve_price_mismatch(result)
        
        # If we've resolved all discrepancies, update the status
        if not any(not d.get('resolved', False) for d in result.discrepancies):
            result.resolve("All discrepancies resolved")
    
    async def _resolve_status_mismatch(self, result: ReconciliationResult) -> None:
        """Resolve status mismatches between local and broker orders."""
        if not result.local_order or not result.broker_order:
            return
        
        broker_status = self._map_broker_status(result.broker_order.get('status'))
        
        # If the broker order is in a terminal state but local isn't, update local
        if broker_status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
            result.local_order.status = broker_status
            result.local_order.update_time = datetime.now(timezone.utc)
            
            # Update the corresponding position if needed
            if broker_status == OrderStatus.FILLED:
                await self._update_position_from_order(result.local_order, result.broker_order)
            
            result.resolve(f"Updated local order status to {broker_status.value}")
    
    async def _resolve_quantity_mismatch(self, result: ReconciliationResult) -> None:
        """Resolve quantity mismatches between local and broker orders."""
        if not result.local_order or not result.broker_order:
            return
        
        broker_qty = self._safe_get_decimal(result.broker_order, 'quantity')
        broker_exec_qty = self._safe_get_decimal(result.broker_order, 'executed_quantity')
        
        # If the broker has more up-to-date execution info, update local
        if broker_exec_qty is not None and broker_exec_qty > result.local_order.executed_quantity:
            result.local_order.executed_quantity = broker_exec_qty
            
            # If fully filled, update status
            if broker_exec_qty >= result.local_order.quantity:
                result.local_order.status = OrderStatus.FILLED
            
            result.local_order.update_time = datetime.now(timezone.utc)
            
            # Update position if needed
            await self._update_position_from_order(result.local_order, result.broker_order)
            
            result.resolve(f"Updated local executed quantity to {broker_exec_qty}")
    
    async def _resolve_price_mismatch(self, result: ReconciliationResult) -> None:
        """Resolve price mismatches between local and broker orders."""
        if not result.local_order or not result.broker_order:
            return
        
        broker_avg_price = self._safe_get_decimal(result.broker_order, 'avg_price')
        
        # If we have an average price from the broker, use it
        if broker_avg_price is not None and broker_avg_price > 0:
            result.local_order.avg_price = broker_avg_price
            result.local_order.update_time = datetime.now(timezone.utc)
            
            # Update position if needed
            await self._update_position_from_order(result.local_order, result.broker_order)
            
            result.resolve(f"Updated local average price to {broker_avg_price}")
    
    async def _update_position_from_order(
        self,
        local_order: Order,
        broker_order: Dict[str, Any]
    ) -> None:
        """Update position based on order execution."""
        if local_order.status != OrderStatus.FILLED:
            return
        
        # Get the position for this order
        position = await self.position_manager.get_position(local_order.client_order_id)
        if not position:
            # Try to find an open position for this symbol and side
            position = await self.position_manager.get_open_position(
                local_order.symbol,
                local_order.side
            )
        
        if position:
            # Update the position with the order
            await self.position_manager.update_position(
                position.position_id,
                order=local_order,
                price=local_order.avg_price
            )
    
    def _check_field_equality(
        self,
        result: ReconciliationResult,
        field: str,
        local_value: Any,
        broker_value: Any,
        discrepancy_type: DiscrepancyType
    ) -> None:
        """Check if a field matches between local and broker orders."""
        if local_value != broker_value:
            result.add_discrepancy(
                discrepancy_type,
                field,
                local_value,
                broker_value,
                severity="ERROR"
            )
    
    def _check_decimal_equality(
        self,
        result: ReconciliationResult,
        field: str,
        local_value: Optional[Decimal],
        broker_value: Optional[Decimal],
        discrepancy_type: DiscrepancyType
    ) -> None:
        """Check if decimal fields are approximately equal."""
        if local_value is None and broker_value is None:
            return
            
        if local_value is None or broker_value is None:
            result.add_discrepancy(
                discrepancy_type,
                field,
                local_value,
                broker_value,
                severity="ERROR"
            )
            return
            
        if abs(local_value - broker_value) > self.tolerance:
            result.add_discrepancy(
                discrepancy_type,
                field,
                local_value,
                broker_value,
                severity="ERROR"
            )
    
    def _map_broker_status(self, status: Optional[str]) -> OrderStatus:
        """Map broker-specific status to our OrderStatus enum."""
        if not status:
            return OrderStatus.NEW
            
        status = status.upper()
        
        # Common status mappings
        status_map = {
            'NEW': OrderStatus.NEW,
            'PARTIALLY_FILLED': OrderStatus.PARTIALLY_FILLED,
            'FILLED': OrderStatus.FILLED,
            'CANCELED': OrderStatus.CANCELED,
            'PENDING_CANCEL': OrderStatus.PENDING_CANCEL,
            'REJECTED': OrderStatus.REJECTED,
            'EXPIRED': OrderStatus.EXPIRED,
            'EXPIRED_IN_MATCH': OrderStatus.EXPIRED_IN_MATCH,
            'TRIGGERED': OrderStatus.TRIGGERED,
            'ACTIVE': OrderStatus.NEW,
            'DONE': OrderStatus.FILLED,
            'CLOSED': OrderStatus.FILLED,
            'OPEN': OrderStatus.NEW,
            'PENDING': OrderStatus.NEW,
            'CANCELLED': OrderStatus.CANCELED,
            'PARTIALLY_FILLED': OrderStatus.PARTIALLY_FILLED,
            'PARTIAL_FILL': OrderStatus.PARTIALLY_FILLED,
            'FILL': OrderStatus.FILLED,
            'EXECUTED': OrderStatus.FILLED,
            'EXECUTION': OrderStatus.FILLED,
            'TRADE': OrderStatus.FILLED,
            'TRADED': OrderStatus.FILLED,
            'MATCHED': OrderStatus.FILLED,
        }
        
        return status_map.get(status, OrderStatus.NEW)
    
    def _safe_get_decimal(self, data: Dict[str, Any], key: str) -> Optional[Decimal]:
        """Safely get a decimal value from a dictionary."""
        value = data.get(key)
        if value is None:
            return None
        
        try:
            if isinstance(value, Decimal):
                return value
            elif isinstance(value, (int, float, str)):
                return Decimal(str(value))
            else:
                return None
        except (ValueError, TypeError):
            return None
    
    def _update_broker_orders_cache(self, broker_orders: List[Dict[str, Any]]) -> None:
        """Update the broker orders cache."""
        now = time.time()
        
        # Add new orders to cache
        for order in broker_orders:
            if 'client_order_id' in order:
                order_id = order['client_order_id']
                self._broker_orders_cache[order_id] = order
                self._cache_expiry[order_id] = now + self._cache_ttl
        
        # Remove expired orders from cache
        expired = [
            order_id for order_id, expiry in self._cache_expiry.items()
            if expiry < now
        ]
        
        for order_id in expired:
            self._broker_orders_cache.pop(order_id, None)
            self._cache_expiry.pop(order_id, None)
    
    async def get_recent_discrepancies(
        self,
        hours: int = 24,
        status: Optional[ReconciliationStatus] = None
    ) -> List[ReconciliationResult]:
        """Get recent discrepancies from the reconciliation history."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        
        results = [
            r for r in self.reconciliation_history
            if r.timestamp >= cutoff and 
               (status is None or r.status == status) and
               (r.status == ReconciliationStatus.DISCREPANCY or not r.resolved)
        ]
        
        return results
    
    async def get_unresolved_discrepancies(self) -> List[ReconciliationResult]:
        """Get all unresolved discrepancies."""
        return [
            r for r in self.reconciliation_history
            if not r.resolved and r.status == ReconciliationStatus.DISCREPANCY
        ]
    
    async def get_reconciliation_report(self) -> Dict[str, Any]:
        """Generate a reconciliation report."""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=self.reconciliation_window)
        
        recent_results = [r for r in self.reconciliation_history if r.timestamp >= cutoff]
        
        # Count by status
        status_counts = {}
        for status in ReconciliationStatus:
            status_counts[status.value] = sum(1 for r in recent_results if r.status == status)
        
        # Count by discrepancy type
        discrepancy_counts = {}
        for result in recent_results:
            for d in result.discrepancies:
                d_type = d['type']
                discrepancy_counts[d_type] = discrepancy_counts.get(d_type, 0) + 1
        
        # Get unresolved discrepancies
        unresolved = await self.get_unresolved_discrepancies()
        
        return {
            'timestamp': now.isoformat(),
            'window_days': self.reconciliation_window,
            'total_reconciliations': len(recent_results),
            'status_counts': status_counts,
            'discrepancy_counts': discrepancy_counts,
            'unresolved_count': len(unresolved),
            'unresolved_discrepancies': [r.to_dict() for r in unresolved[:10]]  # Limit to 10
        }
