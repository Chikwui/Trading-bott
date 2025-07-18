"""
VWAP (Volume-Weighted Average Price) Execution Algorithm.

This module implements a VWAP execution algorithm that slices orders based on historical
volume profiles to minimize market impact while tracking the volume-weighted average price.
"""
import asyncio
import logging
from datetime import datetime, timedelta, time
from decimal import Decimal, ROUND_DOWN
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

from .base import ExecutionAlgorithm
from ...order_types import Order, OrderStatus, OrderType, TimeInForce, OrderSide

logger = logging.getLogger(__name__)

class VWAPExecutor(ExecutionAlgorithm):
    """
    Volume-Weighted Average Price (VWAP) execution algorithm.
    
    Executes orders by following historical volume profiles to minimize market impact
    while tracking the volume-weighted average price.
    """
    
    DEFAULT_CONFIG = {
        'duration': '1d',  # Default execution duration (1d, 4h, 1h, etc.)
        'time_slice': '1m',  # Time slice for volume buckets (1m, 5m, 15m, etc.)
        'max_slippage': 0.001,  # Maximum allowed slippage (0.1%)
        'aggressiveness': 0.5,  # 0-1, where 1 is most aggressive
        'allow_market_orders': True,
        'limit_order_spread': 0.0005,  # 5 bps inside mid for limit orders
        'min_order_size': 0.01,  # Minimum order size in base currency
        'max_retries': 3,  # Maximum number of retry attempts per slice
    }
    
    def __init__(
        self,
        exchange_adapter: Any,
        position_manager: Any,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the VWAP executor.
        
        Args:
            exchange_adapter: Exchange adapter for order execution
            position_manager: Position manager for risk checks
            config: Configuration overrides
        """
        super().__init__(exchange_adapter, position_manager, config)
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self._volume_profile: List[Dict[str, Any]] = []
        self._current_bucket = 0
        self._is_running = False
    
    async def execute(
        self,
        order: Order,
        params: Optional[Dict[str, Any]] = None
    ) -> Order:
        """Execute an order using the VWAP algorithm.
        
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
            
            # Load historical volume profile
            await self._load_volume_profile(order.symbol, params)
            
            if not self._volume_profile:
                raise ValueError("Failed to load volume profile")
            
            # Calculate total volume and target participation rate
            total_volume = sum(bucket['volume'] for bucket in self._volume_profile)
            if total_volume <= 0:
                raise ValueError("No trading volume in selected period")
            
            # Adjust target quantity based on available liquidity
            target_quantity = order.quantity
            max_quantity = total_volume * Decimal(str(self.config.get('max_participation', 0.1)))  # Default 10% max participation
            
            if target_quantity > max_quantity:
                logger.warning(
                    f"Order quantity {target_quantity} exceeds maximum {max_quantity} "
                    f"(max {self.config.get('max_participation', 10)}% of volume)"
                )
                if not params.get('allow_oversize', False):
                    target_quantity = max_quantity
            
            # Execute slices based on volume profile
            executed_quantity = Decimal('0')
            total_slices = len(self._volume_profile)
            
            logger.info(
                f"Executing VWAP order {order.order_id}: "
                f"{target_quantity} {order.symbol} over {total_slices} time slices"
            )
            
            for i, bucket in enumerate(self._volume_profile):
                if not self._is_running:
                    logger.info(f"VWAP execution for order {order.order_id} was cancelled")
                    order.status = OrderStatus.CANCELED
                    break
                
                # Calculate target quantity for this time slice
                bucket_pct = bucket['volume'] / total_volume
                target_slice = (target_quantity * Decimal(str(bucket_pct))).quantize(
                    Decimal('0.00000001'),
                    rounding=ROUND_DOWN
                )
                
                if target_slice < Decimal(str(self.config['min_order_size'])):
                    logger.debug(f"Skipping small slice {target_slice} < {self.config['min_order_size']}")
                    continue
                
                # Execute slice
                slice_order = await self._execute_slice(
                    order=order,
                    quantity=target_slice,
                    bucket=bucket,
                    slice_num=i,
                    total_slices=total_slices,
                    params=params
                )
                
                # Update executed quantity
                executed_quantity += slice_order.filled_quantity
                
                # Update parent order
                order.filled_quantity = executed_quantity
                order.remaining_quantity = order.quantity - executed_quantity
                
                # Calculate average filled price
                if slice_order.filled_price:
                    if order.filled_quantity > 0:
                        order.filled_price = (
                            (order.filled_price * (order.filled_quantity - slice_order.filled_quantity) +
                             slice_order.filled_price * slice_order.filled_quantity) /
                            order.filled_quantity
                        )
                    else:
                        order.filled_price = slice_order.filled_price
                
                # Update order status
                order.status = (
                    OrderStatus.FILLED 
                    if executed_quantity >= order.quantity * Decimal('0.999')  # Allow for rounding errors
                    else OrderStatus.PARTIALLY_FILLED
                )
                
                # Log progress
                logger.info(
                    f"VWAP slice {i+1}/{total_slices} executed: "
                    f"{slice_order.filled_quantity}/{target_slice} filled at {slice_order.filled_price or 'N/A'}, "
                    f"{order.remaining_quantity} remaining"
                )
                
                # Wait for next time slice (if not the last one)
                if i < total_slices - 1 and self._is_running:
                    next_bucket_time = self._volume_profile[i+1]['start_time']
                    sleep_time = (next_bucket_time - datetime.utcnow()).total_seconds()
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
            
            # Final status update
            if executed_quantity >= order.quantity * Decimal('0.999'):  # Allow for rounding errors
                order.status = OrderStatus.FILLED
                logger.info(f"VWAP order {order.order_id} fully executed")
            else:
                order.status = OrderStatus.PARTIALLY_FILLED
                logger.warning(
                    f"VWAP order {order.order_id} partially filled: "
                    f"{executed_quantity}/{order.quantity}"
                )
            
            return order
            
        except asyncio.CancelledError:
            logger.info(f"VWAP execution for order {order.order_id} was cancelled")
            order.status = OrderStatus.CANCELED
            raise
            
        except Exception as e:
            logger.error(f"Error in VWAP execution for order {order.order_id}: {str(e)}")
            order.status = OrderStatus.REJECTED
            order.error_message = str(e)
            raise
            
        finally:
            self._is_running = False
    
    async def _load_volume_profile(
        self,
        symbol: str,
        params: Dict[str, Any]
    ) -> None:
        """Load historical volume profile for the given symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            params: Execution parameters
        """
        # Get execution duration and time slice from params or config
        duration_str = params.get('duration', self.config['duration'])
        time_slice = params.get('time_slice', self.config['time_slice'])
        
        # Parse duration (e.g., '1d', '4h', '30m')
        duration = self._parse_duration(duration_str)
        
        # Calculate start and end times
        end_time = datetime.utcnow()
        start_time = end_time - duration
        
        # Get historical volume data from exchange
        # This is a placeholder - implement actual data retrieval based on your exchange
        try:
            # Try to get volume profile from exchange
            self._volume_profile = await self.exchange_adapter.get_historical_volume_profile(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time,
                interval=time_slice
            )
            
            # If no data, fall back to a default profile
            if not self._volume_profile:
                self._volume_profile = self._create_default_profile(duration, time_slice)
                
        except Exception as e:
            logger.warning(
                f"Failed to load volume profile from exchange: {str(e)}. "
                "Using default profile."
            )
            self._volume_profile = self._create_default_profile(duration, time_slice)
        
        # Normalize volumes to sum to 1.0
        total_volume = sum(bucket['volume'] for bucket in self._volume_profile)
        if total_volume > 0:
            for bucket in self._volume_profile:
                bucket['volume_pct'] = bucket['volume'] / total_volume
        
        logger.debug(f"Loaded volume profile with {len(self._volume_profile)} time slices")
    
    def _create_default_profile(
        self,
        duration: timedelta,
        time_slice: str
    ) -> List[Dict[str, Any]]:
        """Create a default volume profile based on typical market patterns.
        
        Args:
            duration: Total execution duration
            time_slice: Time slice for buckets
            
        Returns:
            List of volume buckets
        """
        # Parse time slice (e.g., '5m' -> 5 minutes)
        slice_minutes = int(time_slice.rstrip('m'))
        total_minutes = int(duration.total_seconds() / 60)
        num_slices = total_minutes // slice_minutes
        
        if num_slices <= 0:
            num_slices = 1
        
        # Create a U-shaped volume profile (higher at market open/close)
        profile = []
        now = datetime.utcnow()
        
        for i in range(num_slices):
            # Calculate position in the day (0-1)
            pos = i / max(1, num_slices - 1)
            
            # U-shaped volume curve (higher at open/close)
            volume = 0.3 + 1.4 * (1 - abs(2 * pos - 1))
            
            # Add some random noise
            volume *= 0.8 + 0.4 * np.random.random()
            
            # Calculate bucket times
            start_time = now - (num_slices - i) * timedelta(minutes=slice_minutes)
            end_time = start_time + timedelta(minutes=slice_minutes)
            
            profile.append({
                'start_time': start_time,
                'end_time': end_time,
                'volume': Decimal(str(volume)),
                'volume_pct': Decimal('0'),  # Will be normalized later
                'vwap': Decimal('0'),  # Not available in default profile
                'trades': 0  # Not available in default profile
            })
        
        return profile
    
    def _parse_duration(self, duration_str: str) -> timedelta:
        """Parse a duration string into a timedelta.
        
        Args:
            duration_str: Duration string (e.g., '1d', '4h', '30m')
            
        Returns:
            Corresponding timedelta
        """
        if not duration_str:
            return timedelta(hours=1)  # Default to 1 hour
            
        try:
            value = int(duration_str[:-1])
            unit = duration_str[-1].lower()
            
            if unit == 'd':
                return timedelta(days=value)
            elif unit == 'h':
                return timedelta(hours=value)
            elif unit == 'm':
                return timedelta(minutes=value)
            else:
                raise ValueError(f"Unknown duration unit: {unit}")
                
        except (ValueError, IndexError):
            logger.warning(f"Invalid duration format: {duration_str}. Using default 1h.")
            return timedelta(hours=1)
    
    async def _execute_slice(
        self,
        order: Order,
        quantity: Decimal,
        bucket: Dict[str, Any],
        slice_num: int,
        total_slices: int,
        params: Dict[str, Any]
    ) -> Order:
        """Execute a single VWAP slice.
        
        Args:
            order: Parent order
            quantity: Quantity for this slice
            bucket: Volume bucket data
            slice_num: Current slice number
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
                'algo': 'vwap',
                'bucket_start': bucket['start_time'].isoformat(),
                'bucket_end': bucket['end_time'].isoformat(),
                'bucket_volume_pct': float(bucket.get('volume_pct', 0)),
                **order.metadata
            }
        )
        
        # Determine order type and price
        use_market = self.config.get('allow_market_orders', True)
        aggressiveness = float(params.get('aggressiveness', self.config.get('aggressiveness', 0.5)))
        
        if use_market and aggressiveness >= 0.8:
            # Use market orders for high aggressiveness
            slice_order.order_type = OrderType.MARKET
        else:
            # Use limit orders with price improvement
            slice_order.order_type = OrderType.LIMIT
            
            # Get current market data
            market = self._get_market_conditions(order.symbol)
            
            if order.side == OrderSide.BUY:
                # For buys, we want to pay less than the ask
                spread = Decimal(str(self.config.get('limit_order_spread', 0.0005)))
                price_improvement = spread * Decimal(str(1.0 - aggressiveness))
                slice_order.price = market['best_ask'] * (1 - price_improvement)
            else:  # SELL
                # For sells, we want to get more than the bid
                spread = Decimal(str(self.config.get('limit_order_spread', 0.0005)))
                price_improvement = spread * Decimal(str(1.0 - aggressiveness))
                slice_order.price = market['best_bid'] * (1 + price_improvement)
            
            slice_order.price = slice_order.price.quantize(Decimal('0.00000001'))
        
        # Execute the slice order
        max_retries = int(params.get('max_retries', self.config.get('max_retries', 3)))
        retry_delay = float(params.get('retry_delay', 1.0))
        
        for attempt in range(max_retries + 1):
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
                
                # If we got a fill, we're done
                if slice_order.filled_quantity > 0:
                    # Update position if needed
                    if hasattr(self, 'position_manager'):
                        await self.position_manager.update_position(
                            position_id=slice_order.position_id,
                            order=slice_order,
                            price=slice_order.filled_price
                        )
                    break
                
                # If we didn't get a fill, try again with a more aggressive price
                if attempt < max_retries and not use_market:
                    # Increase price improvement for next attempt
                    if order.side == OrderSide.BUY:
                        slice_order.price *= Decimal('0.999')  # Move price up
                    else:
                        slice_order.price *= Decimal('1.001')  # Move price down
                    
                    logger.debug(
                        f"Retry {attempt+1}/{max_retries} for slice {slice_num}: "
                        f"adjusting price to {slice_order.price}"
                    )
                    
                    # Wait before retry
                    await asyncio.sleep(retry_delay)
                
            except Exception as e:
                logger.error(f"Error executing VWAP slice {slice_num} (attempt {attempt+1}): {str(e)}")
                if attempt >= max_retries:
                    slice_order.status = OrderStatus.REJECTED
                    slice_order.error_message = str(e)
                else:
                    await asyncio.sleep(retry_delay)
        
        return slice_order
    
    async def cancel(self) -> bool:
        """Cancel the VWAP execution.
        
        Returns:
            True if cancellation was successful, False otherwise
        """
        self._is_running = False
        return True
