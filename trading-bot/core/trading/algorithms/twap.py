""
TWAP (Time-Weighted Average Price) Execution Algorithm.

This module implements a TWAP execution algorithm that slices a large order into smaller
chunks and executes them evenly over a specified time period to minimize market impact.
"""
import asyncio
import logging
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from typing import Dict, Any, Optional, Tuple

from .base import ExecutionAlgorithm
from ...order_types import Order, OrderStatus, OrderType, TimeInForce

logger = logging.getLogger(__name__)

class TWAPExecutor(ExecutionAlgorithm):
    """
    Time-Weighted Average Price (TWAP) execution algorithm.
    
    Splits a large order into smaller slices and executes them at regular intervals
    over a specified time horizon to minimize market impact.
    """
    
    DEFAULT_CONFIG = {
        'duration_seconds': 300,  # 5 minutes default duration
        'slices': 10,  # Number of slices
        'randomize_slices': True,  # Randomize slice sizes slightly
        'max_slice_pct_volume': 0.1,  # Max slice size as % of average volume
        'min_slice_size': 0.01,  # Minimum size per slice (in base currency)
        'max_slice_size': None,  # Maximum size per slice (None for no limit)
        'allow_market_orders': True,  # Allow market orders for execution
        'limit_order_spread': 0.0005,  # 5 bps inside mid for limit orders
        'slippage_tolerance': 0.001,  # 10 bps max slippage
    }
    
    def __init__(
        self,
        exchange_adapter: Any,
        position_manager: Any,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the TWAP executor.
        
        Args:
            exchange_adapter: Exchange adapter for order execution
            position_manager: Position manager for risk checks
            config: Configuration overrides
        """
        super().__init__(exchange_adapter, position_manager, config)
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self._active_orders: Dict[str, asyncio.Task] = {}
        self._is_running = False
    
    async def execute(
        self,
        order: Order,
        params: Optional[Dict[str, Any]] = None
    ) -> Order:
        """Execute an order using the TWAP algorithm.
        
        Args:
            order: The order to execute
            params: Additional execution parameters
            
        Returns:
            The executed order with updated status
        """
        params = params or {}
        self._is_running = True
        
        try:
            # Update order status to indicate execution has started
            order.status = OrderStatus.WORKING
            
            # Get execution parameters with overrides from params
            duration = params.get('duration_seconds', self.config['duration_seconds'])
            num_slices = params.get('slices', self.config['slices'])
            randomize = params.get('randomize_slices', self.config['randomize_slices'])
            
            # Calculate execution window
            start_time = datetime.utcnow()
            end_time = start_time + timedelta(seconds=duration)
            
            # Calculate slice sizes and intervals
            slice_size = self._calculate_slice_size(
                order.quantity,
                num_slices,
                min_size=Decimal(str(self.config['min_slice_size'])),
                max_size=(
                    Decimal(str(self.config['max_slice_size'])) 
                    if self.config['max_slice_size'] is not None else None
                )
            )
            
            interval = self._calculate_slice_interval(
                start_time, end_time, num_slices, randomize
            )
            
            logger.info(
                f"Executing TWAP order {order.order_id}: "
                f"{order.quantity} {order.symbol} over {duration}s in {num_slices} slices, "
                f"{slice_size} per slice, every {interval:.2f}s"
            )
            
            # Execute slices
            remaining_quantity = order.quantity
            slice_num = 0
            
            while remaining_quantity > 0 and self._is_running and slice_num < num_slices:
                # Calculate quantity for this slice
                current_slice_size = min(slice_size, remaining_quantity)
                
                # Apply volume constraints
                current_slice_size = self._apply_volume_constraints(
                    order.symbol, current_slice_size
                )
                
                if current_slice_size <= 0:
                    logger.warning("Slice size is zero or negative, stopping execution")
                    break
                
                # Execute the slice
                slice_order = await self._execute_slice(
                    order=order,
                    quantity=current_slice_size,
                    slice_num=slice_num,
                    total_slices=num_slices,
                    params=params
                )
                
                # Update remaining quantity
                filled_qty = slice_order.filled_quantity
                remaining_quantity -= filled_qty
                
                # Update parent order
                order.filled_quantity += filled_qty
                order.remaining_quantity = remaining_quantity
                
                # Calculate average filled price
                if slice_order.filled_price:
                    if order.filled_quantity > 0:
                        order.filled_price = (
                            (order.filled_price * (order.filled_quantity - filled_qty) +
                             slice_order.filled_price * filled_qty) /
                            order.filled_quantity
                        )
                    else:
                        order.filled_price = slice_order.filled_price
                
                # Update order status
                order.status = (
                    OrderStatus.FILLED 
                    if remaining_quantity <= 0 
                    else OrderStatus.PARTIALLY_FILLED
                )
                
                # Log progress
                logger.info(
                    f"TWAP slice {slice_num + 1}/{num_slices} executed: "
                    f"{filled_qty}/{current_slice_size} filled at {slice_order.filled_price or 'N/A'}, "
                    f"{remaining_quantity} remaining"
                )
                
                # Wait for next slice (except after last slice)
                slice_num += 1
                if remaining_quantity > 0 and slice_num < num_slices and self._is_running:
                    await asyncio.sleep(interval)
            
            # Final status update
            if remaining_quantity <= 0:
                order.status = OrderStatus.FILLED
                logger.info(f"TWAP order {order.order_id} fully executed")
            else:
                order.status = OrderStatus.PARTIALLY_FILLED
                logger.warning(
                    f"TWAP order {order.order_id} partially filled: "
                    f"{order.filled_quantity}/{order.quantity}"
                )
            
            return order
            
        except asyncio.CancelledError:
            logger.info(f"TWAP execution for order {order.order_id} was cancelled")
            order.status = OrderStatus.CANCELED
            raise
            
        except Exception as e:
            logger.error(f"Error in TWAP execution for order {order.order_id}: {str(e)}")
            order.status = OrderStatus.REJECTED
            order.error_message = str(e)
            raise
            
        finally:
            self._is_running = False
    
    async def _execute_slice(
        self,
        order: Order,
        quantity: Decimal,
        slice_num: int,
        total_slices: int,
        params: Dict[str, Any]
    ) -> Order:
        """Execute a single TWAP slice.
        
        Args:
            order: Parent order
            quantity: Quantity for this slice
            slice_num: Current slice number (0-based)
            total_slices: Total number of slices
            params: Execution parameters
            
        Returns:
            The executed slice order
        """
        # Create a child order for this slice
        slice_order = order.create_child(
            quantity=quantity,
            order_id=f"{order.order_id}_slice_{slice_num}",
            parent_order_id=order.order_id,
            metadata={
                'slice_num': slice_num,
                'total_slices': total_slices,
                'algo': 'twap',
                **order.metadata
            }
        )
        
        # Determine order type and price
        use_market = self.config.get('allow_market_orders', True)
        
        if use_market:
            slice_order.order_type = OrderType.MARKET
        else:
            slice_order.order_type = OrderType.LIMIT
            # Calculate limit price based on current market
            market = self._get_market_conditions(order.symbol)
            spread = self.config.get('limit_order_spread', 0.0005)
            
            if order.side == 'BUY':
                # For buys, we want to pay less than the ask
                slice_order.price = market['best_ask'] * (1 - Decimal(str(spread)))
            else:  # SELL
                # For sells, we want to get more than the bid
                slice_order.price = market['best_bid'] * (1 + Decimal(str(spread)))
            
            slice_order.price = slice_order.price.quantize(Decimal('0.00000001'))
        
        # Execute the slice order
        try:
            # Submit to exchange
            result = await self.exchange_adapter.submit_order(
                symbol=slice_order.symbol,
                side=slice_order.side,
                order_type=slice_order.order_type,
                quantity=slice_order.quantity,
                price=slice_order.price,
                time_in_force=TimeInForce.IOC if use_market else TimeInForce.GTC,
                client_order_id=slice_order.client_order_id,
                **slice_order.metadata
            )
            
            # Update slice order with execution details
            self._update_order_from_execution(slice_order, result)
            
            # Update position if needed
            if hasattr(self, 'position_manager') and slice_order.filled_quantity > 0:
                await self.position_manager.update_position(
                    position_id=slice_order.position_id,
                    order=slice_order,
                    price=slice_order.filled_price
                )
            
            return slice_order
            
        except Exception as e:
            logger.error(f"Error executing TWAP slice {slice_num}: {str(e)}")
            slice_order.status = OrderStatus.REJECTED
            slice_order.error_message = str(e)
            return slice_order
    
    def _apply_volume_constraints(
        self,
        symbol: str,
        quantity: Decimal
    ) -> Decimal:
        """Apply volume-based constraints to slice size.
        
        Args:
            symbol: Trading pair symbol
            quantity: Proposed slice size
            
        Returns:
            Adjusted slice size respecting volume constraints
        """
        if 'max_slice_pct_volume' not in self.config:
            return quantity
            
        # Get market data
        market = self._get_market_conditions(symbol)
        if 'volume_24h' not in market:
            return quantity
            
        # Calculate max size based on % of daily volume
        max_pct = Decimal(str(self.config['max_slice_pct_volume']))
        max_volume = market['volume_24h'] * max_pct
        
        # Apply minimum size
        min_size = Decimal(str(self.config.get('min_slice_size', '0.01')))
        
        return max(min(quantity, max_volume), min_size)
    
    async def cancel(self) -> bool:
        """Cancel the TWAP execution.
        
        Returns:
            True if cancellation was successful, False otherwise
        """
        self._is_running = False
        
        # Cancel any active slice orders
        if hasattr(self, '_active_orders'):
            for task in self._active_orders.values():
                if not task.done():
                    task.cancel()
            
            # Wait for all tasks to complete
            if self._active_orders:
                await asyncio.wait(
                    list(self._active_orders.values()),
                    return_when=asyncio.ALL_COMPLETED
                )
        
        return True
