"""
Order execution handling with market hours awareness.
"""
from __future__ import annotations

import logging
import queue
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple, Callable

from core.instruments import InstrumentMetadata
from core.calendar import MarketCalendar

logger = logging.getLogger(__name__)


class ExecutionHandler:
    """Handles order execution with market hours awareness."""
    
    def __init__(
        self,
        calendar: Optional[MarketCalendar] = None,
        timezone: str = "UTC",
        max_queue_size: int = 1000,
        **kwargs
    ):
        """Initialize the execution handler.
        
        Args:
            calendar: Market calendar
            timezone: Default timezone
            max_queue_size: Maximum size of the order queue
            **kwargs: Additional broker-specific parameters
        """
        self.calendar = calendar
        self.timezone = timezone
        self.max_queue_size = max_queue_size
        self.broker_params = kwargs
        
        # Order management
        self.order_queue = queue.Queue(maxsize=max_queue_size)
        self.orders: Dict[str, Dict] = {}  # order_id -> order
        self.positions: Dict[Tuple[str, str], Dict] = {}  # (account_id, symbol) -> position
        
        # Threading
        self._running = False
        self._thread = None
        
        # Statistics
        self.executed_orders = 0
        self.rejected_orders = 0
        self.total_commission = 0.0
    
    def submit_order(self, order: Dict[str, Any]) -> bool:
        """Submit an order for execution.
        
        Args:
            order: Order details with required fields:
                - symbol: Instrument symbol
                - order_type: 'market', 'limit', 'stop', etc.
                - side: 'buy' or 'sell'
                - quantity: Number of units
                - price: Limit/stop price if applicable
                - account_id: Trading account ID
                - order_id: Unique order ID
                - time_in_force: 'GTC', 'IOC', 'FOK', etc.
                
        Returns:
            bool: True if order was accepted, False otherwise
        """
        # Basic validation
        required_fields = ['symbol', 'order_type', 'side', 'quantity', 'account_id', 'order_id']
        for field in required_fields:
            if field not in order:
                logger.error(f"Order missing required field: {field}")
                return False
                
        # Add timestamp if not present
        if 'timestamp' not in order:
            order['timestamp'] = datetime.now(timezone.utc)
            
        # Add to order queue
        try:
            self.order_queue.put_nowait(order)
            self.orders[order['order_id']] = order
            logger.info(f"Order {order['order_id']} submitted: {order}")
            return True
        except queue.Full:
            logger.error("Order queue full. Order rejected.")
            self.rejected_orders += 1
            return False
    
    def cancel_order(self, order_id: str, account_id: str) -> bool:
        """Cancel an existing order.
        
        Args:
            order_id: Order ID to cancel
            account_id: Account ID that owns the order
            
        Returns:
            bool: True if cancellation was successful, False otherwise
        """
        if order_id not in self.orders:
            logger.warning(f"Order {order_id} not found")
            return False
            
        order = self.orders[order_id]
        if order['account_id'] != account_id:
            logger.warning(f"Order {order_id} does not belong to account {account_id}")
            return False
            
        # In a real implementation, this would send a cancellation request to the broker
        # For now, we'll just mark it as cancelled
        if 'status' not in order or order['status'] == 'new':
            order['status'] = 'cancelled'
            order['updated_at'] = datetime.now(timezone.utc)
            logger.info(f"Order {order_id} cancelled")
            return True
            
        logger.warning(f"Cannot cancel order {order_id} with status {order.get('status')}")
        return False
    
    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of an order.
        
        Args:
            order_id: Order ID to check
            
        Returns:
            Order status or None if order not found
        """
        return self.orders.get(order_id)
    
    def get_position(self, account_id: str, symbol: str) -> Dict[str, Any]:
        """Get the current position for an account and symbol.
        
        Args:
            account_id: Account ID
            symbol: Instrument symbol
            
        Returns:
            Position details or empty dict if no position
        """
        return self.positions.get((account_id, symbol), {
            'symbol': symbol,
            'quantity': 0,
            'avg_price': 0.0,
            'unrealized_pnl': 0.0,
            'realized_pnl': 0.0,
            'timestamp': datetime.now(timezone.utc)
        })
    
    def _process_orders(self) -> None:
        """Process orders from the queue."""
        while self._running:
            try:
                order = self.order_queue.get(timeout=1.0)
                self._execute_order(order)
                self.order_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing order: {e}", exc_info=True)
    
    def _execute_order(self, order: Dict[str, Any]) -> None:
        """Execute a single order.
        
        In a real implementation, this would communicate with a broker API.
        For now, we'll simulate execution.
        """
        order_id = order['order_id']
        symbol = order['symbol']
        account_id = order['account_id']
        
        # Update order status
        order['status'] = 'filled'
        order['filled_quantity'] = order['quantity']
        order['filled_avg_price'] = order.get('price', 100.0)  # Simulated price
        order['commission'] = max(1.0, order['quantity'] * 0.001)  # Simulated commission
        order['updated_at'] = datetime.now(timezone.utc)
        
        # Update position
        position_key = (account_id, symbol)
        position = self.positions.get(position_key, {
            'symbol': symbol,
            'quantity': 0,
            'avg_price': 0.0,
            'unrealized_pnl': 0.0,
            'realized_pnl': 0.0,
            'timestamp': datetime.now(timezone.utc)
        })
        
        # Calculate new position
        old_quantity = position['quantity']
        old_avg_price = position['avg_price']
        new_quantity = old_quantity + (order['quantity'] if order['side'] == 'buy' else -order['quantity'])
        
        # Update position
        if new_quantity == 0:
            # Position closed
            position['quantity'] = 0
            position['avg_price'] = 0.0
            position['realized_pnl'] += (order['filled_avg_price'] - old_avg_price) * old_quantity
        else:
            # Position updated
            if (order['side'] == 'buy' and old_quantity >= 0) or (order['side'] == 'sell' and old_quantity <= 0):
                # Adding to position in same direction
                position['avg_price'] = (
                    (old_quantity * old_avg_price) + 
                    (order['quantity'] * order['filled_avg_price'])
                ) / (old_quantity + order['quantity'])
            else:
                # Reversing or partially closing position
                position['realized_pnl'] += (
                    order['quantity'] * 
                    (order['filled_avg_price'] - old_avg_price) * 
                    (1 if order['side'] == 'sell' else -1)
                )
            
            position['quantity'] = new_quantity
        
        position['timestamp'] = datetime.now(timezone.utc)
        self.positions[position_key] = position
        
        # Update statistics
        self.executed_orders += 1
        self.total_commission += order.get('commission', 0.0)
        
        logger.info(f"Order {order_id} executed: {order}")
    
    def start(self) -> None:
        """Start the execution handler."""
        if self._running:
            return
            
        self._running = True
        self._thread = threading.Thread(target=self._process_orders, daemon=True)
        self._thread.start()
        logger.info("Execution handler started")
    
    def stop(self) -> None:
        """Stop the execution handler."""
        self._running = False
        
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            
        logger.info("Execution handler stopped")
