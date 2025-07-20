"""
Order monitoring integration for the trading system.

This module provides integration between the OrderManager and OrderMonitor
for comprehensive order lifecycle monitoring and alerting.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Dict, List, Optional, Any, Callable, Awaitable, TypeVar, Union

from core.monitoring.order_monitor import OrderMonitor, StateTransitionEvent
from core.trading.order import Order
from core.trading.order_manager import OrderManager
from core.trading.oco_order import OCOOrder
from core.trading.types import OrderSide, OrderType, TimeInForce
from core.utils.metrics import timer, counter, histogram

logger = logging.getLogger(__name__)
T = TypeVar('T', bound='OrderMonitorIntegration')

class OrderMonitorIntegration:
    """Integration between OrderManager and OrderMonitor."""
    
    def __init__(
        self,
        order_manager: OrderManager,
        order_monitor: Optional[OrderMonitor] = None,
        enable_metrics: bool = True,
        enable_alerting: bool = True,
        default_alert_rules: bool = True
    ):
        """Initialize the order monitoring integration.
        
        Args:
            order_manager: The OrderManager instance to monitor
            order_monitor: Optional OrderMonitor instance (will create one if not provided)
            enable_metrics: Whether to enable metrics collection
            enable_alerting: Whether to enable alerting
            default_alert_rules: Whether to load default alert rules
        """
        self.order_manager = order_manager
        self.order_monitor = order_monitor or OrderMonitor()
        self.enable_metrics = enable_metrics
        self.enable_alerting = enable_alerting
        
        # Register event handlers
        self._register_event_handlers()
        
        # Load default alert rules if enabled
        if enable_alerting and default_alert_rules:
            self._load_default_alert_rules()
    
    def _register_event_handlers(self) -> None:
        """Register event handlers with the OrderManager."""
        self.order_manager.on_order_created.add_handler(self._on_order_created)
        self.order_manager.on_order_updated.add_handler(self._on_order_updated)
        self.order_manager.on_order_filled.add_handler(self._on_order_filled)
        self.order_manager.on_order_canceled.add_handler(self._on_order_canceled)
        self.order_manager.on_order_rejected.add_handler(self._on_order_rejected)
        self.order_manager.on_oco_order_created.add_handler(self._on_oco_order_created)
        self.order_manager.on_oco_order_updated.add_handler(self._on_oco_order_updated)
        self.order_manager.on_oco_order_completed.add_handler(self._on_oco_order_completed)
    
    def _load_default_alert_rules(self) -> None:
        """Load default alert rules for order monitoring."""
        default_rules = [
            # High latency alert
            {
                'name': 'High Order Submission Latency',
                'condition': "event.to_state == 'SUBMITTED' and event.metadata.get('submission_latency_ms', 0) > 500",
                'severity': 'warning',
                'message': 'High order submission latency detected',
                'cooldown_seconds': 300
            },
            # Order rejection alert
            {
                'name': 'Order Rejection',
                'condition': "event.to_state == 'REJECTED'",
                'severity': 'critical',
                'message': 'Order was rejected by the exchange',
                'cooldown_seconds': 0
            },
            # Partial fill alert
            {
                'name': 'Large Order Partially Filled',
                'condition': "event.to_state == 'PARTIALLY_FILLED' and order.quantity > 10",
                'severity': 'info',
                'message': 'Large order partially filled',
                'cooldown_seconds': 60
            },
            # Slippage alert
            {
                'name': 'High Slippage Detected',
                'condition': "event.to_state == 'FILLED' and 'slippage' in event.metadata and event.metadata['slippage'] > 0.005",
                'severity': 'warning',
                'message': 'High slippage detected on order fill',
                'cooldown_seconds': 60
            }
        ]
        
        self.order_monitor.load_alert_rules(default_rules)
    
    async def start(self) -> None:
        """Start the order monitoring system."""
        if self.enable_metrics or self.enable_alerting:
            await self.order_monitor.start()
    
    # Event handlers
    async def _on_order_created(self, order: Order) -> None:
        """Handle order creation event."""
        if not self.enable_metrics:
            return
            
        event = StateTransitionEvent(
            from_state='NEW',
            to_state='CREATED',
            timestamp=datetime.now(timezone.utc),
            reason='Order created',
            metadata={
                'order_id': order.id,
                'client_order_id': order.client_order_id,
                'symbol': order.symbol,
                'side': order.side,
                'order_type': order.order_type,
                'quantity': float(order.quantity),
                'price': float(order.price) if order.price else None,
                'time_in_force': order.time_in_force
            }
        )
        
        await self.order_monitor.process_order_event(event, order)
    
    async def _on_order_updated(self, order: Order, old_status: str, new_status: str) -> None:
        """Handle order status update event."""
        if not self.enable_metrics:
            return
            
        event = StateTransitionEvent(
            from_state=old_status,
            to_state=new_status,
            timestamp=datetime.now(timezone.utc),
            reason='Order status updated',
            metadata={
                'order_id': order.id,
                'client_order_id': order.client_order_id,
                'symbol': order.symbol,
                'side': order.side,
                'order_type': order.order_type,
                'filled_quantity': float(order.filled_quantity) if order.filled_quantity else 0.0,
                'remaining_quantity': float(order.remaining_quantity) if hasattr(order, 'remaining_quantity') else None,
                'avg_fill_price': float(order.avg_fill_price) if hasattr(order, 'avg_fill_price') and order.avg_fill_price else None
            }
        )
        
        await self.order_monitor.process_order_event(event, order)
    
    async def _on_order_filled(
        self, 
        order: Order, 
        fill_quantity: Decimal, 
        fill_price: Decimal,
        is_complete: bool
    ) -> None:
        """Handle order fill event."""
        if not self.enable_metrics:
            return
            
        status = 'FILLED' if is_complete else 'PARTIALLY_FILLED'
        
        # Calculate slippage if possible
        metadata = {
            'fill_quantity': float(fill_quantity),
            'fill_price': float(fill_price),
            'is_complete': is_complete
        }
        
        if hasattr(order, 'price') and order.price and fill_price:
            slippage = abs(float(fill_price) - float(order.price)) / float(order.price)
            metadata['slippage'] = slippage
        
        event = StateTransitionEvent(
            from_state=order.status,
            to_state=status,
            timestamp=datetime.now(timezone.utc),
            reason='Order filled' if is_complete else 'Order partially filled',
            metadata=metadata
        )
        
        await self.order_monitor.process_order_event(event, order)
    
    async def _on_order_canceled(self, order: Order, reason: Optional[str] = None) -> None:
        """Handle order cancellation event."""
        if not self.enable_metrics:
            return
            
        event = StateTransitionEvent(
            from_state=order.status,
            to_state='CANCELED',
            timestamp=datetime.now(timezone.utc),
            reason=reason or 'Order canceled',
            metadata={
                'cancel_reason': reason,
                'filled_quantity': float(order.filled_quantity) if order.filled_quantity else 0.0,
                'remaining_quantity': float(order.remaining_quantity) if hasattr(order, 'remaining_quantity') else None
            }
        )
        
        await self.order_monitor.process_order_event(event, order)
    
    async def _on_order_rejected(self, order: Order, reason: str) -> None:
        """Handle order rejection event."""
        if not self.enable_metrics:
            return
            
        event = StateTransitionEvent(
            from_state=order.status,
            to_state='REJECTED',
            timestamp=datetime.now(timezone.utc),
            reason=reason,
            metadata={
                'reject_reason': reason,
                'order_type': order.order_type,
                'quantity': float(order.quantity) if order.quantity else None,
                'price': float(order.price) if order.price else None
            }
        )
        
        await self.order_monitor.process_order_event(event, order)
    
    async def _on_oco_order_created(self, oco_order: OCOOrder) -> None:
        """Handle OCO order creation event."""
        if not self.enable_metrics:
            return
            
        event = StateTransitionEvent(
            from_state='NEW',
            to_state='OCO_CREATED',
            timestamp=datetime.now(timezone.utc),
            reason='OCO order created',
            metadata={
                'oco_order_id': oco_order.id,
                'symbol': oco_order.symbol,
                'quantity': float(oco_order.quantity),
                'limit_price': float(oco_order.limit_price) if oco_order.limit_price else None,
                'stop_price': float(oco_order.stop_price) if oco_order.stop_price else None,
                'stop_limit_price': float(oco_order.stop_limit_price) if oco_order.stop_limit_price else None
            }
        )
        
        # We'll use the main order for metrics
        main_order = oco_order.entry_order or oco_order.stop_loss_order or oco_order.take_profit_order
        if main_order:
            await self.order_monitor.process_order_event(event, main_order)
    
    async def _on_oco_order_updated(self, oco_order: OCOOrder, old_status: str, new_status: str) -> None:
        """Handle OCO order status update event."""
        if not self.enable_metrics:
            return
            
        event = StateTransitionEvent(
            from_state=old_status,
            to_state=new_status,
            timestamp=datetime.now(timezone.utc),
            reason='OCO order status updated',
            metadata={
                'oco_order_id': oco_order.id,
                'symbol': oco_order.symbol,
                'status': oco_order.status,
                'filled_quantity': float(oco_order.filled_quantity) if oco_order.filled_quantity else 0.0
            }
        )
        
        # We'll use the main order for metrics
        main_order = oco_order.entry_order or oco_order.stop_loss_order or oco_order.take_profit_order
        if main_order:
            await self.order_monitor.process_order_event(event, main_order)
    
    async def _on_oco_order_completed(self, oco_order: OCOOrder, reason: str) -> None:
        """Handle OCO order completion event."""
        if not self.enable_metrics:
            return
            
        event = StateTransitionEvent(
            from_state=oco_order.status,
            to_state='OCO_COMPLETED',
            timestamp=datetime.now(timezone.utc),
            reason=f'OCO order completed: {reason}',
            metadata={
                'oco_order_id': oco_order.id,
                'symbol': oco_order.symbol,
                'completion_reason': reason,
                'filled_quantity': float(oco_order.filled_quantity) if oco_order.filled_quantity else 0.0
            }
        )
        
        # We'll use the main order for metrics
        main_order = oco_order.entry_order or oco_order.stop_loss_order or oco_order.take_profit_order
        if main_order:
            await self.order_monitor.process_order_event(event, main_order)

# Example usage
async def example_usage():
    """Example of using the OrderMonitorIntegration."""
    from core.trading.order_manager import OrderManager
    from core.trading.order import Order, OrderSide, OrderType, TimeInForce
    
    # Create order manager and monitor
    order_manager = OrderManager()
    monitor_integration = OrderMonitorIntegration(
        order_manager=order_manager,
        enable_metrics=True,
        enable_alerting=True
    )
    
    # Start monitoring
    await monitor_integration.start()
    
    # Example order
    order = Order(
        symbol="BTC/USD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("1.0"),
        price=Decimal("50000.00"),
        time_in_force=TimeInForce.GTC
    )
    
    # Simulate order flow
    await order_manager.submit_order(order)
    
    # Simulate order updates
    await order_manager.update_order_status(order, "PARTIALLY_FILLED")
    await order_manager.update_order_status(order, "FILLED")

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
