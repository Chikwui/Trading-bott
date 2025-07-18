"""
Time-Weighted Average Price (TWAP) Execution Strategy.

This module implements a TWAP execution strategy that slices a large order into smaller
chunks and executes them at regular intervals to achieve the time-weighted average price.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timedelta, time as dt_time
from decimal import Decimal, ROUND_HALF_UP, ROUND_DOWN
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any, Union, Deque

import numpy as np
import pandas as pd

from core.execution.base import (
    ExecutionClient,
    ExecutionParameters,
    ExecutionResult,
    ExecutionStyle,
    ExecutionState,
    ExecutionReport
)
from core.market.data import MarketDataService, BarData, TickerData
from core.risk.manager import RiskManager
from core.ml.model_registry import ModelRegistry, ModelType
from core.trading.order import Order, OrderSide, OrderType, OrderStatus, TimeInForce
from core.utils.connection_pool import ConnectionPool

logger = logging.getLogger(__name__)

@dataclass
class TWAPParameters:
    """Parameters for TWAP execution strategy."""
    
    # Time interval between child orders (in seconds)
    interval_seconds: int = 300  # 5 minutes
    
    # Maximum number of child orders to create
    max_child_orders: int = 12  # 1 hour of 5-min intervals
    
    # Whether to use dynamic intervals based on market conditions
    use_adaptive_intervals: bool = True
    
    # Minimum interval between orders (in seconds)
    min_interval_seconds: int = 30  # 30 seconds
    
    # Maximum interval between orders (in seconds)
    max_interval_seconds: int = 600  # 10 minutes
    
    # Whether to use limit orders (True) or market orders (False)
    use_limit_orders: bool = True
    
    # Price tolerance for limit orders (in basis points from current price)
    limit_order_tolerance_bps: int = 2  # 0.02%
    
    # Maximum spread to trade at (in basis points)
    max_allowed_spread_bps: int = 20  # 0.2%
    
    # Time in force for child orders (in seconds)
    order_timeout_seconds: int = 60
    
    # Whether to randomize order sizes within a range to avoid detection
    randomize_order_sizes: bool = True
    
    # Randomization range for order sizes (e.g., 0.2 means +/- 20%)
    order_size_randomization: float = 0.2  # +/- 20%
    
    # Whether to adjust for market impact
    adjust_for_market_impact: bool = True
    
    # Market impact coefficient (higher = more conservative)
    market_impact_coefficient: float = 1.0
    
    # Whether to enable anti-gaming logic
    enable_anti_gaming: bool = True
    
    # Anti-gaming: minimum time between orders (ms)
    min_order_interval_ms: int = 100
    
    # Whether to enable fill probability prediction
    enable_fill_probability: bool = True
    
    # Minimum fill probability for limit orders (0-1)
    min_fill_probability: float = 0.7
    
    # Whether to enable real-time adaptation
    enable_realtime_adaptation: bool = True
    
    # Adaptation sensitivity (0-1)
    adaptation_sensitivity: float = 0.5


class TWAPExecutionClient(ExecutionClient):
    """
    TWAP (Time-Weighted Average Price) Execution Strategy.
    
    This strategy breaks up large orders into smaller child orders that are executed
    at regular time intervals, with adaptive sizing based on market conditions.
    """
    
    def __init__(
        self,
        client_id: str,
        market_data: MarketDataService,
        risk_manager: RiskManager,
        model_registry: Optional[ModelRegistry] = None,
        default_params: Optional[ExecutionParameters] = None,
        twap_params: Optional[TWAPParameters] = None,
        connection_pool: Optional[ConnectionPool] = None
    ):
        super().__init__(
            client_id=client_id,
            market_data=market_data,
            risk_manager=risk_manager,
            model_registry=model_registry,
            default_params=default_params
        )
        self.twap_params = twap_params or TWAPParameters()
        self._scheduled_tasks: Dict[str, asyncio.Task] = {}
        self._connection_pool = connection_pool
        self._last_order_time: Dict[str, datetime] = {}
        self._order_count: Dict[str, int] = {}
        self._execution_stats: Dict[str, Any] = {
            'total_orders': 0,
            'filled_orders': 0,
            'cancelled_orders': 0,
            'rejected_orders': 0,
            'avg_fill_price': {},
            'avg_slippage': {},
            'twap_slippage': {},
            'participation_rate': {},
            'execution_duration': {}
        }
        self._market_impact_model = None
        self._fill_probability_model = None
        
        # Initialize ML models if available
        self._init_ml_models()
    
    def _init_ml_models(self) -> None:
        """Initialize ML models for execution optimization."""
        if self.model_registry:
            try:
                # Load market impact model if available
                self._market_impact_model = self.model_registry.get_model(
                    ModelType.MARKET_IMPACT,
                    version='latest'
                )
                
                # Load fill probability model if available
                self._fill_probability_model = self.model_registry.get_model(
                    ModelType.FILL_PROBABILITY,
                    version='latest'
                )
                
                logger.info("Loaded ML models for TWAP execution optimization")
                
            except Exception as e:
                logger.warning(f"Failed to load ML models: {e}", exc_info=True)
    
    async def _execute_strategy(
        self,
        order: Order,
        params: ExecutionParameters,
        result: ExecutionResult
    ) -> ExecutionResult:
        """
        Execute the order using TWAP strategy.
        
        Args:
            order: The parent order to execute
            params: Execution parameters
            result: Execution result to update
            
        Returns:
            Updated execution result
        """
        try:
            execution_start = datetime.utcnow()
            symbol = order.symbol
            
            # Calculate order schedule
            schedule = self._calculate_order_schedule(order, params)
            
            # Execute orders according to schedule
            child_orders = await self._execute_schedule(order, schedule, params)
            
            # Update result with execution details
            result = self._update_execution_result(
                order,
                child_orders,
                result,
                execution_start
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in TWAP execution: {e}", exc_info=True)
            await self.cancel_all_orders()
            raise
    
    def _calculate_order_schedule(
        self,
        order: Order,
        params: ExecutionParameters
    ) -> List[Dict[str, Any]]:
        """
        Calculate the order execution schedule.
        
        Args:
            order: The parent order
            params: Execution parameters
            
        Returns:
            List of order schedule entries with timing and size
        """
        total_quantity = float(order.quantity)
        
        # Calculate number of child orders
        num_orders = min(
            self.twap_params.max_child_orders,
            int((24 * 60 * 60) / self.twap_params.interval_seconds)  # Max 24 hours
        )
        
        # Calculate base order size
        base_size = total_quantity / num_orders
        
        # Generate schedule
        schedule = []
        remaining_quantity = total_quantity
        current_time = datetime.utcnow()
        
        for i in range(num_orders):
            # Calculate order time
            interval = self._get_adaptive_interval(i, num_orders)
            order_time = current_time + timedelta(seconds=interval * i)
            
            # Calculate order size (last order takes remaining quantity)
            if i == num_orders - 1:
                order_size = remaining_quantity
            else:
                order_size = base_size
                if self.twap_params.randomize_order_sizes:
                    # Add random variation to order size
                    variation = 1 + (np.random.random() * 2 - 1) * self.twap_params.order_size_randomization
                    order_size *= variation
                
                # Ensure we don't exceed remaining quantity
                order_size = min(order_size, remaining_quantity)
            
            if order_size <= 0:
                continue
                
            remaining_quantity -= order_size
            
            schedule.append({
                'time': order_time,
                'quantity': Decimal(str(round(order_size, 8))),
                'executed': False,
                'order_id': None
            })
            
            if remaining_quantity <= 0:
                break
        
        return schedule
    
    def _get_adaptive_interval(self, order_index: int, total_orders: int) -> float:
        """
        Calculate adaptive time interval between orders.
        
        Args:
            order_index: Current order index (0-based)
            total_orders: Total number of orders
            
        Returns:
            Time interval in seconds
        """
        if not self.twap_params.use_adaptive_intervals:
            return self.twap_params.interval_seconds
        
        # Simple adaptive logic - can be enhanced with market conditions
        base_interval = self.twap_params.interval_seconds
        
        # Adjust interval based on order progress
        progress = order_index / max(1, total_orders - 1)  # 0 to 1
        
        # Start and end with shorter intervals, longer in the middle
        if progress < 0.2 or progress > 0.8:
            # Beginning and end: faster execution
            interval = base_interval * 0.8
        else:
            # Middle: slower execution
            interval = base_interval * 1.2
        
        # Apply bounds
        return max(
            self.twap_params.min_interval_seconds,
            min(interval, self.twap_params.max_interval_seconds)
        )
    
    async def _execute_schedule(
        self,
        parent_order: Order,
        schedule: List[Dict[str, Any]],
        params: ExecutionParameters
    ) -> List[Order]:
        """
        Execute orders according to the schedule.
        
        Args:
            parent_order: The parent order
            schedule: Order execution schedule
            params: Execution parameters
            
        Returns:
            List of child orders created
        """
        child_orders = []
        
        for entry in schedule:
            # Wait until scheduled time
            wait_time = (entry['time'] - datetime.utcnow()).total_seconds()
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            
            # Check if parent order is still active
            if parent_order.status not in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]:
                logger.info(f"Parent order {parent_order.id} is no longer active. Stopping execution.")
                break
            
            # Create and execute child order
            child_order = self._create_child_order(parent_order, entry['quantity'], params)
            
            try:
                # Apply anti-gaming measures
                await self._apply_anti_gaming_measures(parent_order.symbol)
                
                # Execute order
                await self._execute_child_order(child_order, params)
                child_orders.append(child_order)
                entry['executed'] = True
                entry['order_id'] = child_order.id
                
                # Update execution stats
                self._update_order_stats(child_order)
                
            except Exception as e:
                logger.error(f"Error executing child order: {e}", exc_info=True)
                # Continue with next order even if one fails
                continue
        
        return child_orders
    
    def _create_child_order(
        self,
        parent_order: Order,
        quantity: Decimal,
        params: ExecutionParameters
    ) -> Order:
        """
        Create a child order from the parent order.
        
        Args:
            parent_order: The parent order
            quantity: Order quantity
            params: Execution parameters
            
        Returns:
            New child order
        """
        order_type = OrderType.LIMIT if self.twap_params.use_limit_orders else OrderType.MARKET
        
        # Calculate limit price if using limit orders
        limit_price = None
        if order_type == OrderType.LIMIT:
            # Get current market price
            ticker = asyncio.get_event_loop().run_until_complete(
                self.market_data.get_ticker(parent_order.symbol)
            )
            if ticker and 'last' in ticker:
                price = Decimal(str(ticker['last']))
                # Apply tolerance
                tolerance = price * Decimal(str(self.twap_params.limit_order_tolerance_bps / 10000))
                if parent_order.side == OrderSide.BUY:
                    limit_price = price - tolerance
                else:  # SELL
                    limit_price = price + tolerance
        
        return Order(
            symbol=parent_order.symbol,
            side=parent_order.side,
            quantity=quantity,
            order_type=order_type,
            price=limit_price,
            time_in_force=TimeInForce.GTC,
            parent_order_id=parent_order.id,
            client_order_id=f"{parent_order.id}_{len(self._child_orders) + 1}"
        )
    
    async def _execute_child_order(
        self,
        order: Order,
        params: ExecutionParameters
    ) -> Order:
        """
        Execute a single child order.
        
        Args:
            order: The order to execute
            params: Execution parameters
            
        Returns:
            The executed order with updated status
        """
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Check market conditions
                if not await self._check_market_conditions(order):
                    await asyncio.sleep(1)  # Wait and retry
                    continue
                
                # Execute order
                if self._connection_pool:
                    async with self._connection_pool.acquire() as conn:
                        await conn.execute_order(order)
                else:
                    # Fallback to direct execution
                    await self.market_data.execute_order(order)
                
                # Update order status
                if order.status == OrderStatus.FILLED:
                    self._filled_quantity += order.filled_quantity
                    self._filled_value += order.filled_quantity * (order.avg_fill_price or order.price or Decimal('0'))
                
                return order
                
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to execute order after {max_retries} attempts: {e}")
                    order.status = OrderStatus.REJECTED
                    raise
                
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(1)  # Exponential backoff could be used here
    
    async def _check_market_conditions(self, order: Order) -> bool:
        """
        Check if market conditions are favorable for order execution.
        
        Args:
            order: The order to check
            
        Returns:
            bool: True if conditions are favorable, False otherwise
        """
        try:
            # Get current ticker data
            ticker = await self.market_data.get_ticker(order.symbol)
            if not ticker or 'ask' not in ticker or 'bid' not in ticker:
                return False
            
            # Check spread
            spread = (ticker['ask'] - ticker['bid']) / ticker['bid'] * 10000  # in bps
            if spread > self.twap_params.max_allowed_spread_bps:
                logger.warning(f"Spread too wide: {spread:.1f} bps > {self.twap_params.max_allowed_spread_bps} bps")
                return False
            
            # Check fill probability if model is available
            if self._fill_probability_model and self.twap_params.enable_fill_probability:
                features = self._get_fill_probability_features(order, ticker)
                fill_prob = await self._fill_probability_model.predict(features)
                
                if fill_prob < self.twap_params.min_fill_probability:
                    logger.debug(f"Low fill probability: {fill_prob:.2f} < {self.twap_params.min_fill_probability}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking market conditions: {e}", exc_info=True)
            return False
    
    def _get_fill_probability_features(self, order: Order, ticker: Dict[str, float]) -> Dict[str, Any]:
        """
        Get features for fill probability prediction.
        
        Args:
            order: The order
            ticker: Current ticker data
            
        Returns:
            Dictionary of feature names and values
        """
        spread = (ticker['ask'] - ticker['bid']) / ticker['bid'] * 10000  # bps
        mid_price = (ticker['ask'] + ticker['bid']) / 2
        
        return {
            'symbol': order.symbol,
            'side': order.side.value,
            'quantity': float(order.quantity),
            'price': float(order.price or mid_price),
            'spread_bps': spread,
            'mid_price': mid_price,
            'volume': ticker.get('volume', 0),
            'time_of_day': datetime.utcnow().time().strftime('%H%M'),
            'day_of_week': datetime.utcnow().weekday(),
            'order_type': order.order_type.value,
            'price_distance_bps': abs(float(order.price or mid_price) - mid_price) / mid_price * 10000
        }
    
    async def _apply_anti_gaming_measures(self, symbol: str) -> None:
        """
        Apply anti-gaming measures like rate limiting.
        
        Args:
            symbol: The trading symbol
        """
        if not self.twap_params.enable_anti_gaming:
            return
        
        now = datetime.utcnow()
        last_time = self._last_order_time.get(symbol)
        
        if last_time:
            # Calculate time since last order
            elapsed = (now - last_time).total_seconds() * 1000  # ms
            min_interval = self.twap_params.min_order_interval_ms
            
            if elapsed < min_interval:
                # Sleep for remaining time
                sleep_time = (min_interval - elapsed) / 1000  # to seconds
                await asyncio.sleep(sleep_time)
        
        # Update last order time
        self._last_order_time[symbol] = datetime.utcnow()
    
    def _update_order_stats(self, order: Order) -> None:
        """
        Update execution statistics.
        
        Args:
            order: The executed order
        """
        symbol = order.symbol
        
        # Initialize symbol stats if needed
        if symbol not in self._execution_stats['avg_fill_price']:
            self._execution_stats['avg_fill_price'][symbol] = []
            self._execution_stats['avg_slippage'][symbol] = []
            self._execution_stats['twap_slippage'][symbol] = []
        
        # Update stats
        self._execution_stats['total_orders'] += 1
        
        if order.status == OrderStatus.FILLED:
            self._execution_stats['filled_orders'] += 1
            self._execution_stats['avg_fill_price'][symbol].append(float(order.avg_fill_price or 0))
            
            # Calculate slippage if we have a reference price
            if order.price and order.avg_fill_price:
                if order.side == OrderSide.BUY:
                    slippage = float((order.avg_fill_price / order.price - 1) * 10000)  # bps
                else:  # SELL
                    slippage = float((1 - order.avg_fill_price / order.price) * 10000)  # bps
                
                self._execution_stats['avg_slippage'][symbol].append(slippage)
        
        elif order.status == OrderStatus.CANCELED:
            self._execution_stats['cancelled_orders'] += 1
        elif order.status == OrderStatus.REJECTED:
            self._execution_stats['rejected_orders'] += 1
    
    def _update_execution_result(
        self,
        parent_order: Order,
        child_orders: List[Order],
        result: ExecutionResult,
        start_time: datetime
    ) -> ExecutionResult:
        """
        Update the execution result with final metrics.
        
        Args:
            parent_order: The parent order
            child_orders: List of child orders
            result: Execution result to update
            start_time: When execution started
            
        Returns:
            Updated execution result
        """
        # Calculate execution metrics
        exec_time = (datetime.utcnow() - start_time).total_seconds()
        filled_orders = [o for o in child_orders if o.status == OrderStatus.FILLED]
        filled_qty = sum(float(o.filled_quantity) for o in filled_orders)
        
        # Calculate TWAP price
        twap_price = 0.0
        if filled_orders:
            total_value = sum(float(o.filled_quantity * (o.avg_fill_price or Decimal('0'))) for o in filled_orders)
            twap_price = total_value / filled_qty if filled_qty > 0 else 0.0
        
        # Calculate slippage vs arrival price
        arrival_price = float(parent_order.price or 0)
        twap_slippage_bps = 0.0
        if arrival_price > 0 and twap_price > 0:
            if parent_order.side == OrderSide.BUY:
                twap_slippage_bps = (twap_price / arrival_price - 1) * 10000  # bps
            else:  # SELL
                twap_slippage_bps = (1 - twap_price / arrival_price) * 10000  # bps
        
        # Update result metadata
        return result.update({
            'strategy': 'TWAP',
            'child_orders_created': len(child_orders),
            'child_orders_filled': len(filled_orders),
            'total_quantity': float(parent_order.quantity),
            'filled_quantity': filled_qty,
            'fill_rate': filled_qty / float(parent_order.quantity) if parent_order.quantity > 0 else 0,
            'twap_price': twap_price,
            'arrival_price': arrival_price,
            'twap_slippage_bps': twap_slippage_bps,
            'execution_time_seconds': exec_time,
            'start_time': start_time.isoformat(),
            'end_time': datetime.utcnow().isoformat(),
            'parameters': {
                'use_limit_orders': self.twap_params.use_limit_orders,
                'interval_seconds': self.twap_params.interval_seconds,
                'max_child_orders': self.twap_params.max_child_orders,
                'use_adaptive_intervals': self.twap_params.use_adaptive_intervals,
                'enable_anti_gaming': self.twap_params.enable_anti_gaming
            }
        })
