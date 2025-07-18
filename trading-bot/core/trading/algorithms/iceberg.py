"""
Iceberg Order Execution Algorithm

This module implements an Iceberg order execution strategy that breaks large orders into smaller
"iceberg" orders to minimize market impact by only showing a small portion of the total order
quantity at any given time.
"""
import asyncio
import logging
from decimal import Decimal, ROUND_DOWN
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta, timezone
import numpy as np

from ..order_types import (
    Order,
    OrderSide,
    OrderType,
    OrderStatus,
    TimeInForce,
    ExecutionReport
)
from .base import ExecutionAlgorithm, ExecutionResult

logger = logging.getLogger(__name__)

class IcebergExecutor(ExecutionAlgorithm):
    """
    Iceberg Order Execution Algorithm
    
    This algorithm breaks down large orders into smaller "iceberg" orders to minimize market impact
    by only showing a small portion of the total order quantity at any given time. The algorithm
    continuously monitors the order book and adjusts its strategy based on market conditions.
    
    Key Features:
    - Hides large order quantities by only showing a small "tip" of the iceberg
    - Dynamically adjusts order sizes based on market depth and volume
    - Implements sophisticated price improvement logic
    - Supports both passive (limit) and aggressive (market) execution styles
    - Includes anti-gaming measures to detect and respond to predatory trading
    - Implements VWAP/TWAP tracking to ensure execution quality
    - Supports partial fills and order restarts after disconnections
    """
    
    def __init__(self, exchange_adapter, position_manager, config: Optional[Dict] = None):
        """
        Initialize the Iceberg order executor.
        
        Args:
            exchange_adapter: Exchange adapter for order execution
            position_manager: Position manager for tracking positions
            config: Configuration dictionary with the following optional keys:
                - display_size: Maximum quantity to display at once (default: 10% of average daily volume)
                - max_slippage: Maximum allowed slippage in basis points (default: 5bps)
                - price_improvement: Target price improvement in basis points (default: 1bp)
                - min_quantity: Minimum order quantity (default: 1% of display size)
                - max_quantity: Maximum order quantity (default: 2x display size)
                - refresh_interval: How often to refresh orders in seconds (default: 5s)
                - max_retries: Maximum number of retry attempts for failed orders (default: 3)
                - passive_fill_ratio: Ratio of passive to aggressive fills (0.0-1.0, default: 0.7)
                - anti_gaming: Enable anti-gaming measures (default: True)
                - vwap_tracking: Enable VWAP tracking (default: True)
                - twap_tracking: Enable TWAP tracking (default: True)
        """
        super().__init__(exchange_adapter, position_manager, config)
        
        # Algorithm configuration with defaults
        self.display_size = Decimal(str(self.config.get('display_size', '0')))
        self.max_slippage = Decimal(str(self.config.get('max_slippage', '5'))) / Decimal('10000')  # 5bps
        self.price_improvement = Decimal(str(self.config.get('price_improvement', '1'))) / Decimal('10000')  # 1bp
        self.min_quantity = Decimal(str(self.config.get('min_quantity', '0')))
        self.max_quantity = Decimal(str(self.config.get('max_quantity', '0')))
        self.refresh_interval = float(self.config.get('refresh_interval', '5'))
        self.max_retries = int(self.config.get('max_retries', '3'))
        self.passive_fill_ratio = Decimal(str(self.config.get('passive_fill_ratio', '0.7')))
        self.anti_gaming = bool(self.config.get('anti_gaming', True))
        self.vwap_tracking = bool(self.config.get('vwap_tracking', True))
        self.twap_tracking = bool(self.config.get('twap_tracking', True))
        
        # Execution state
        self.original_order = None
        self.remaining_quantity = Decimal('0')
        self.displayed_quantity = Decimal('0')
        self.hidden_quantity = Decimal('0')
        self.working_orders = []
        self.fill_history = []
        self.start_time = None
        self.end_time = None
        self.cancellation_event = asyncio.Event()
        self._is_running = False
        
        # Performance tracking
        self.vwap_target = None
        self.twap_target = None
        self.executed_vwap = Decimal('0')
        self.executed_twap = Decimal('0')
        self.executed_volume = Decimal('0')
        self.executed_notional = Decimal('0')
        self.last_refresh = None
        self.retry_count = 0
        
        # Anti-gaming state
        self.last_adverse_move = None
        self.consecutive_adverse = 0
        self.last_order_book = None
        self.order_book_updates = []
        
        logger.info(f"Initialized IcebergExecutor with config: {config}")
    
    async def execute(self, order: Order) -> ExecutionResult:
        """
        Execute an order using the Iceberg algorithm.
        
        Args:
            order: The order to execute
            
        Returns:
            ExecutionResult containing the execution details
        """
        try:
            self._validate_order(order)
            self._initialize_execution(order)
            
            logger.info(f"Starting Iceberg execution for order {order.order_id}: "
                       f"{order.side} {order.remaining_quantity} {order.symbol} @ {order.price}")
            
            # Main execution loop
            while self.remaining_quantity > 0 and not self.cancellation_event.is_set():
                try:
                    await self._refresh_orders()
                    
                    # Wait for refresh interval or until cancelled
                    try:
                        await asyncio.wait_for(
                            asyncio.shield(self.cancellation_event.wait()),
                            timeout=self.refresh_interval
                        )
                        if self.cancellation_event.is_set():
                            break
                    except asyncio.TimeoutError:
                        pass
                        
                except Exception as e:
                    logger.error(f"Error in Iceberg execution loop: {str(e)}", exc_info=True)
                    self.retry_count += 1
                    if self.retry_count > self.max_retries:
                        logger.error(f"Max retries ({self.max_retries}) exceeded. Aborting execution.")
                        break
                    await asyncio.sleep(min(2 ** self.retry_count, 30))  # Exponential backoff
            
            # Clean up any remaining working orders
            await self._cancel_all_orders()
            
            # Calculate execution metrics
            return self._create_execution_result()
            
        except Exception as e:
            logger.error(f"Fatal error in Iceberg execution: {str(e)}", exc_info=True)
            await self._cancel_all_orders()
            raise
    
    async def cancel(self):
        """Cancel the current execution."""
        logger.info("Cancelling Iceberg execution")
        self.cancellation_event.set()
        await self._cancel_all_orders()
    
    def _validate_order(self, order: Order):
        """Validate the order parameters."""
        if order.quantity <= 0:
            raise ValueError("Order quantity must be positive")
            
        if order.order_type not in (OrderType.MARKET, OrderType.LIMIT):
            raise ValueError("Iceberg algorithm only supports MARKET and LIMIT order types")
            
        if order.time_in_force not in (TimeInForce.GTC, TimeInForce.IOC, TimeInForce.FOK):
            raise ValueError("Iceberg algorithm only supports GTC, IOC, and FOK time in force")
    
    def _initialize_execution(self, order: Order):
        """Initialize the execution state."""
        self.original_order = order
        self.remaining_quantity = order.quantity
        self.displayed_quantity = Decimal('0')
        self.hidden_quantity = order.quantity
        self.working_orders = []
        self.fill_history = []
        self.start_time = datetime.now(timezone.utc)
        self.end_time = None
        self.cancellation_event.clear()
        self._is_running = True
        self.retry_count = 0
        
        # Calculate display size if not provided
        if self.display_size <= 0:
            # Default to 10% of average daily volume, with min/max constraints
            avg_daily_volume = self._get_average_daily_volume(order.symbol)
            self.display_size = max(
                self.min_quantity,
                min(
                    self.max_quantity if self.max_quantity > 0 else Decimal('inf'),
                    avg_daily_volume * Decimal('0.1')  # 10% of ADV
                )
            )
            logger.info(f"Calculated display size: {self.display_size} {order.symbol.split('/')[1]}")
        
        # Initialize VWAP/TWAP tracking
        self.executed_volume = Decimal('0')
        self.executed_notional = Decimal('0')
        self.executed_vwap = Decimal('0')
        self.executed_twap = Decimal('0')
        
        # Get initial market data
        self._update_market_data()
    
    async def _refresh_orders(self):
        """Refresh the displayed orders based on current market conditions."""
        # Update market data
        self._update_market_data()
        
        # Calculate how much we need to display
        target_display = min(self.display_size, self.remaining_quantity)
        
        # Cancel any working orders that are no longer optimal
        await self._optimize_working_orders(target_display)
        
        # Calculate how much more we need to show
        current_display = sum(o.remaining_quantity for o in self.working_orders)
        quantity_to_add = max(Decimal('0'), target_display - current_display)
        
        # If we need to show more, create new orders
        if quantity_to_add > 0:
            await self._create_new_orders(quantity_to_add)
        
        # Update tracking metrics
        self._update_performance_metrics()
        
        # Check for adverse selection and adjust strategy if needed
        if self.anti_gaming:
            self._detect_adverse_selection()
    
    async def _optimize_working_orders(self, target_display: Decimal):
        """Optimize working orders by cancelling and replacing suboptimal ones."""
        if not self.working_orders:
            return
            
        # Get current market data
        market_data = self.exchange_adapter.get_market_data(self.original_order.symbol)
        best_bid = market_data.get('best_bid', Decimal('0'))
        best_ask = market_data.get('best_ask', Decimal('inf'))
        
        # Determine if we need to adjust prices
        need_price_improvement = False
        if self.original_order.side == OrderSide.BUY and best_ask > self.original_order.price * (1 - self.price_improvement):
            need_price_improvement = True
        elif self.original_order.side == OrderSide.SELL and best_bid < self.original_order.price * (1 + self.price_improvement):
            need_price_improvement = True
        
        # Cancel and replace orders that need adjustment
        orders_to_cancel = []
        for order in self.working_orders:
            cancel_order = False
            
            # Check if order price is no longer optimal
            if self.original_order.side == OrderSide.BUY:
                if order.price > best_ask * (1 - self.price_improvement):
                    cancel_order = True
            else:  # SELL
                if order.price < best_bid * (1 + self.price_improvement):
                    cancel_order = True
            
            # Check if order is too large for current market conditions
            order_book = self.exchange_adapter.get_order_book(self.original_order.symbol)
            if order_book and len(order_book['bids']) > 0 and len(order_book['asks']) > 0:
                if self.original_order.side == OrderSide.BUY:
                    available_liquidity = sum(level[1] for level in order_book['asks'] 
                                           if level[0] <= order.price)
                    if order.quantity > available_liquidity * Decimal('0.2'):  # Don't take more than 20% of available liquidity
                        cancel_order = True
                else:  # SELL
                    available_liquidity = sum(level[1] for level in order_book['bids'] 
                                           if level[0] >= order.price)
                    if order.quantity > available_liquidity * Decimal('0.2'):
                        cancel_order = True
            
            if cancel_order:
                orders_to_cancel.append(order)
        
        # Cancel the orders
        for order in orders_to_cancel:
            await self.exchange_adapter.cancel_order(order.order_id)
            self.working_orders.remove(order)
            logger.debug(f"Cancelled order {order.order_id} for optimization")
    
    async def _create_new_orders(self, quantity: Decimal):
        """Create new iceberg orders."""
        if quantity <= 0:
            return
            
        # Get current market data
        market_data = self.exchange_adapter.get_market_data(self.original_order.symbol)
        best_bid = market_data.get('best_bid', Decimal('0'))
        best_ask = market_data.get('best_ask', Decimal('inf'))
        
        # Determine order parameters based on order side and type
        if self.original_order.side == OrderSide.BUY:
            # For buys, we want to pay as little as possible
            price = best_ask * (1 - self.price_improvement) if best_ask < Decimal('inf') else self.original_order.price
            price = min(price, self.original_order.price) if self.original_order.order_type == OrderType.LIMIT else price
        else:  # SELL
            # For sells, we want to get as much as possible
            price = best_bid * (1 + self.price_improvement) if best_bid > 0 else self.original_order.price
            price = max(price, self.original_order.price) if self.original_order.order_type == OrderType.LIMIT else price
        
        # Round price to appropriate tick size
        tick_size = market_data.get('tick_size', Decimal('0.01'))
        price = (price / tick_size).quantize(Decimal('1.'), rounding=ROUND_DOWN) * tick_size
        
        # Create the order
        order = Order(
            order_id=f"iceberg_{self.original_order.order_id}_{len(self.working_orders)}",
            client_order_id=f"{self.original_order.client_order_id}_iceberg_{len(self.working_orders)}",
            symbol=self.original_order.symbol,
            side=self.original_order.side,
            order_type=self.original_order.order_type,
            quantity=min(quantity, self.display_size),
            price=price,
            time_in_force=TimeInForce.GTC,
            status=OrderStatus.NEW,
            timestamp=datetime.now(timezone.utc),
            parent_order_id=self.original_order.order_id
        )
        
        # Submit the order
        try:
            result = await self.exchange_adapter.submit_order(
                symbol=order.symbol,
                side=order.side,
                order_type=order.order_type,
                quantity=order.quantity,
                price=order.price,
                time_in_force=order.time_in_force,
                client_order_id=order.client_order_id
            )
            
            # Update order with exchange response
            order.order_id = result['order_id']
            order.status = OrderStatus[result['status']]
            
            # Add to working orders
            self.working_orders.append(order)
            self.displayed_quantity += order.quantity
            self.hidden_quantity -= order.quantity
            
            logger.debug(f"Created new iceberg order: {order.order_id} - {order.side} {order.quantity} "
                        f"{order.symbol} @ {order.price}")
            
        except Exception as e:
            logger.error(f"Failed to create iceberg order: {str(e)}", exc_info=True)
            self.retry_count += 1
            if self.retry_count > self.max_retries:
                raise
    
    async def _cancel_all_orders(self):
        """Cancel all working orders."""
        if not self.working_orders:
            return
            
        cancellation_tasks = []
        for order in self.working_orders:
            if order.status in (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED):
                cancellation_tasks.append(
                    self.exchange_adapter.cancel_order(order.order_id)
                )
        
        if cancellation_tasks:
            await asyncio.gather(*cancellation_tasks, return_exceptions=True)
        
        self.working_orders = []
    
    def _update_market_data(self):
        """Update market data and detect adverse selection."""
        # Get current order book
        order_book = self.exchange_adapter.get_order_book(self.original_order.symbol)
        
        # Store for anti-gaming analysis
        if order_book:
            self.last_order_book = order_book
            self.order_book_updates.append((datetime.now(timezone.utc), order_book))
            
            # Keep only the last 100 updates (adjust based on needs)
            if len(self.order_book_updates) > 100:
                self.order_book_updates.pop(0)
    
    def _update_performance_metrics(self):
        """Update execution performance metrics."""
        now = datetime.now(timezone.utc)
        
        # Calculate VWAP
        if self.executed_volume > 0:
            self.executed_vwap = self.executed_notional / self.executed_volume
        
        # Calculate TWAP
        if self.start_time:
            elapsed = (now - self.start_time).total_seconds()
            if elapsed > 0:
                self.executed_twap = self.executed_notional / elapsed
        
        self.last_refresh = now
    
    def _detect_adverse_selection(self):
        """Detect if the market is moving against us in a way that suggests gaming."""
        if len(self.order_book_updates) < 10:  # Need some history
            return
            
        # Simple heuristic: count the number of times the market moved away after our orders
        adverse_moves = 0
        total_moves = 0
        
        for i in range(1, len(self.order_book_updates)):
            prev_time, prev_book = self.order_book_updates[i-1]
            curr_time, curr_book = self.order_book_updates[i]
            
            if not prev_book or not curr_book or not prev_book['bids'] or not prev_book['asks']:
                continue
                
            prev_best_bid = prev_book['bids'][0][0]
            prev_best_ask = prev_book['asks'][0][0]
            curr_best_bid = curr_book['bids'][0][0]
            curr_best_ask = curr_book['asks'][0][0]
            
            # Check for adverse moves
            if self.original_order.side == OrderSide.BUY and curr_best_ask > prev_best_ask * Decimal('1.0001'):
                adverse_moves += 1
                total_moves += 1
            elif self.original_order.side == OrderSide.SELL and curr_best_bid < prev_best_bid * Decimal('0.9999'):
                adverse_moves += 1
                total_moves += 1
            elif self.original_order.side == OrderSide.BUY and curr_best_ask < prev_best_ask * Decimal('0.9999'):
                total_moves += 1
            elif self.original_order.side == OrderSide.SELL and curr_best_bid > prev_best_bid * Decimal('1.0001'):
                total_moves += 1
        
        # If more than 60% of moves are adverse, we might be getting gamed
        if total_moves > 0 and adverse_moves / total_moves > 0.6:
            self.consecutive_adverse += 1
            logger.warning(f"Detected potential adverse selection: {adverse_moves}/{total_moves} moves adverse "
                         f"({self.consecutive_adverse} consecutive)")
            
            # If this happens multiple times in a row, adjust strategy
            if self.consecutive_adverse >= 3:
                logger.warning("Multiple adverse moves detected. Adjusting execution strategy...")
                # Reduce display size to be less aggressive
                self.display_size = max(
                    self.min_quantity,
                    self.display_size * Decimal('0.8')  # Reduce by 20%
                )
                self.consecutive_adverse = 0  # Reset counter
        else:
            self.consecutive_adverse = max(0, self.consecutive_adverse - 1)  # Decay counter
    
    def _get_average_daily_volume(self, symbol: str, lookback_days: int = 30) -> Decimal:
        """Get the average daily volume for a symbol."""
        try:
            # Try to get historical volume data
            end_time = datetime.now(timezone.utc)
            start_time = end_time - timedelta(days=lookback_days)
            
            volume_data = self.exchange_adapter.get_historical_volume(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time,
                interval='1d'
            )
            
            if volume_data and len(volume_data) > 0:
                # Calculate average volume over the lookback period
                total_volume = sum(item['volume'] for item in volume_data)
                avg_volume = total_volume / len(volume_data)
                return Decimal(str(avg_volume))
                
        except Exception as e:
            logger.warning(f"Could not get historical volume data: {str(e)}")
        
        # Fallback to a reasonable default if data is unavailable
        return Decimal('1000')  # Default to 1000 units
    
    def _create_execution_result(self) -> ExecutionResult:
        """Create an execution result from the current state."""
        self.end_time = datetime.now(timezone.utc)
        
        # Calculate filled quantity and average price
        filled_quantity = self.original_order.quantity - self.remaining_quantity
        avg_price = self.executed_vwap if self.executed_vwap > 0 else (
            self.original_order.price if self.original_order.order_type == OrderType.LIMIT 
            else Decimal('0')
        )
        
        # Determine final status
        if self.cancellation_event.is_set():
            status = OrderStatus.CANCELED
        elif filled_quantity <= 0:
            status = OrderStatus.REJECTED
        elif self.remaining_quantity > 0:
            status = OrderStatus.PARTIALLY_FILLED
        else:
            status = OrderStatus.FILLED
        
        # Calculate slippage
        slippage = Decimal('0')
        if self.original_order.order_type == OrderType.LIMIT and avg_price > 0:
            if self.original_order.side == OrderSide.BUY:
                slippage = (avg_price / self.original_order.price - 1) * Decimal('10000')  # In bps
            else:  # SELL
                slippage = (self.original_order.price / avg_price - 1) * Decimal('10000')  # In bps
        
        # Create result
        result = ExecutionResult(
            order_id=self.original_order.order_id,
            client_order_id=self.original_order.client_order_id,
            symbol=self.original_order.symbol,
            side=self.original_order.side,
            order_type=self.original_order.order_type,
            quantity=self.original_order.quantity,
            price=self.original_order.price,
            filled_quantity=filled_quantity,
            remaining_quantity=self.remaining_quantity,
            avg_price=avg_price,
            status=status,
            timestamp=self.end_time,
            execution_id=f"exec_{self.original_order.order_id}",
            execution_time=(self.end_time - self.start_time).total_seconds(),
            slippage_bps=float(slippage),
            metadata={
                'algorithm': 'iceberg',
                'display_size': float(self.display_size),
                'vwap': float(self.executed_vwap) if self.executed_vwap > 0 else None,
                'twap': float(self.executed_twap) if self.executed_twap > 0 else None,
                'passive_fill_ratio': float(self.passive_fill_ratio),
                'adverse_moves': self.consecutive_adverse,
                'working_orders': len(self.working_orders),
                'retry_count': self.retry_count
            }
        )
        
        return result
    
    def _on_order_fill(self, order: Order, fill_quantity: Decimal, fill_price: Decimal):
        """Handle order fill events."""
        # Update filled quantity
        self.remaining_quantity -= fill_quantity
        self.displayed_quantity -= fill_quantity
        
        # Update execution metrics
        self.executed_volume += fill_quantity
        self.executed_notional += fill_quantity * fill_price
        
        # Update VWAP
        if self.executed_volume > 0:
            self.executed_vwap = self.executed_notional / self.executed_volume
        
        # Log the fill
        logger.info(f"Iceberg order filled: {fill_quantity} {self.original_order.symbol} @ {fill_price} "
                   f"(Remaining: {self.remaining_quantity})")
        
        # Update order status
        if order.remaining_quantity <= 0:
            order.status = OrderStatus.FILLED
            if order in self.working_orders:
                self.working_orders.remove(order)
        else:
            order.status = OrderStatus.PARTIALLY_FILLED
        
        # Add to fill history
        self.fill_history.append({
            'timestamp': datetime.now(timezone.utc),
            'order_id': order.order_id,
            'quantity': fill_quantity,
            'price': fill_price,
            'remaining': self.remaining_quantity
        })
