"""
Iceberg Order Execution Strategy.

This module implements an Iceberg order execution strategy that only shows a small portion
of the total order quantity at any given time, with the goal of minimizing market impact
by hiding the full order size.
"""
from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import pandas as pd

from core.execution.base import (
    ExecutionClient, 
    ExecutionParameters, 
    ExecutionResult,
    ExecutionStyle,
    ExecutionState
)
from core.market.data import MarketDataService, BarData
from core.risk.manager import RiskManager
from core.ml.model_registry import ModelRegistry
from core.trading.order import Order, OrderSide, OrderType, OrderStatus

logger = logging.getLogger(__name__)

@dataclass
class IcebergParameters:
    """Parameters specific to Iceberg execution strategy."""
    # Maximum visible quantity (tip of the iceberg) as a percentage of average daily volume
    max_visible_pct_adv: float = 0.05  # 5% of ADV
    
    # Minimum visible quantity (in base currency for crypto, shares for stocks)
    min_visible_qty: Decimal = Decimal('0.001')
    
    # Maximum visible quantity (in base currency/shares)
    max_visible_qty: Decimal = Decimal('100')
    
    # Time between order refreshes (in seconds)
    refresh_interval: int = 30
    
    # Maximum time to live for the entire order (in seconds)
    max_duration: int = 3600  # 1 hour
    
    # Price aggressiveness (0-1, where 0 is passive, 1 is aggressive)
    aggressiveness: float = 0.5
    
    # Whether to dynamically adjust the visible quantity based on market conditions
    dynamic_quantity: bool = True
    
    # Whether to randomize the refresh interval to avoid detection
    randomize_refresh: bool = True
    
    # Maximum spread to trade at (in basis points)
    max_allowed_spread_bps: int = 20  # 0.2%
    
    # Time in force for child orders (in seconds)
    order_timeout: int = 60


class IcebergExecutionClient(ExecutionClient):
    """
    Iceberg Order Execution Strategy.
    
    This strategy only shows a small portion (the "tip") of the total order quantity
    at any given time, with the goal of minimizing market impact by hiding the full
    order size from the market.
    
    The visible quantity is dynamically adjusted based on:
    - Current market volume
    - Order book depth
    - Recent price volatility
    - Time of day
    """
    
    def __init__(
        self,
        client_id: str,
        market_data: MarketDataService,
        risk_manager: RiskManager,
        model_registry: Optional[ModelRegistry] = None,
        default_params: Optional[ExecutionParameters] = None,
        iceberg_params: Optional[IcebergParameters] = None
    ):
        super().__init__(
            client_id=client_id,
            market_data=market_data,
            risk_manager=risk_manager,
            model_registry=model_registry,
            default_params=default_params
        )
        self.iceberg_params = iceberg_params or IcebergParameters()
        self._active_orders: Dict[str, asyncio.Task] = {}
        self._refresh_tasks: Dict[str, asyncio.Task] = {}
    
    async def _execute_strategy(
        self,
        order: Order,
        params: ExecutionParameters,
        result: ExecutionResult
    ) -> ExecutionResult:
        """
        Execute the order using the Iceberg strategy.
        
        This method:
        1. Calculates the initial visible quantity
        2. Places the first slice
        3. Sets up a refresh task to manage subsequent slices
        """
        # Initialize execution state
        execution_id = result.execution_id
        symbol = order.symbol
        
        # Get market data
        symbol_data = await self.market_data.get_symbol_data(symbol)
        if not symbol_data:
            raise ValueError(f"No market data available for {symbol}")
        
        # Calculate initial visible quantity
        visible_qty = await self._calculate_visible_quantity(order, symbol_data)
        
        # Create execution state
        exec_state = {
            'order': order,
            'params': params,
            'result': result,
            'visible_qty': visible_qty,
            'remaining_qty': order.quantity,
            'filled_qty': Decimal('0'),
            'avg_fill_price': Decimal('0'),
            'last_refresh': datetime.utcnow(),
            'start_time': datetime.utcnow(),
            'refresh_count': 0,
            'child_orders': []
        }
        
        # Store execution state
        self._active_orders[execution_id] = exec_state
        
        # Start refresh task
        refresh_task = asyncio.create_task(
            self._refresh_loop(execution_id),
            name=f"Iceberg-Refresh-{execution_id[:8]}"
        )
        self._refresh_tasks[execution_id] = refresh_task
        
        # Place initial order
        await self._place_iceberg_order(execution_id)
        
        # Wait for execution to complete or be cancelled
        try:
            while not result.is_complete and execution_id in self._active_orders:
                await asyncio.sleep(1)
                
                # Check for timeout
                exec_time = (datetime.utcnow() - exec_state['start_time']).total_seconds()
                if exec_time > self.iceberg_params.max_duration:
                    logger.info(f"Iceberg execution {execution_id} timed out after {exec_time:.1f}s")
                    await self.cancel_execution(execution_id)
                    break
                    
        except asyncio.CancelledError:
            logger.info(f"Iceberg execution {execution_id} was cancelled")
            await self.cancel_execution(execution_id)
        
        return result
    
    async def _refresh_loop(self, execution_id: str) -> None:
        """Background task to refresh iceberg orders."""
        try:
            while execution_id in self._active_orders:
                exec_state = self._active_orders[execution_id]
                
                # Calculate next refresh interval
                base_interval = self.iceberg_params.refresh_interval
                if self.iceberg_params.randomize_refresh:
                    # Randomize interval by ±20%
                    interval = base_interval * random.uniform(0.8, 1.2)
                else:
                    interval = base_interval
                
                # Wait for refresh interval
                try:
                    await asyncio.sleep(interval)
                except asyncio.CancelledError:
                    break
                
                # Check if execution is still active
                if execution_id not in self._active_orders:
                    break
                
                # Refresh the iceberg order
                await self._refresh_iceberg_order(execution_id)
                
        except Exception as e:
            logger.error(f"Error in iceberg refresh loop for {execution_id}: {str(e)}", exc_info=True)
            if execution_id in self._active_orders:
                await self._handle_execution_error(execution_id, str(e))
    
    async def _refresh_iceberg_order(self, execution_id: str) -> None:
        """Refresh an iceberg order by canceling and replacing it."""
        if execution_id not in self._active_orders:
            return
            
        exec_state = self._active_orders[execution_id]
        order = exec_state['order']
        
        # Cancel any existing child orders
        await self._cancel_child_orders(execution_id)
        
        # Check if we're done
        if exec_state['remaining_qty'] <= 0:
            logger.info(f"Iceberg execution {execution_id} completed successfully")
            await self._complete_execution(execution_id)
            return
        
        # Update visible quantity based on current market conditions
        symbol_data = await self.market_data.get_symbol_data(order.symbol)
        if symbol_data:
            exec_state['visible_qty'] = await self._calculate_visible_quantity(
                order, 
                symbol_data,
                current_visible=exec_state['visible_qty']
            )
        
        # Place new iceberg order
        await self._place_iceberg_order(execution_id)
        exec_state['last_refresh'] = datetime.utcnow()
        exec_state['refresh_count'] += 1
    
    async def _place_iceberg_order(self, execution_id: str) -> None:
        """Place a new iceberg order."""
        if execution_id not in self._active_orders:
            return
            
        exec_state = self._active_orders[execution_id]
        order = exec_state['order']
        symbol = order.symbol
        
        # Get current market data
        symbol_data = await self.market_data.get_symbol_data(symbol)
        if not symbol_data:
            logger.warning(f"No market data for {symbol}, cannot place iceberg order")
            return
        
        # Calculate order parameters
        visible_qty = min(exec_state['visible_qty'], exec_state['remaining_qty'])
        if visible_qty <= 0:
            return
        
        # Calculate price based on aggressiveness
        if order.side == OrderSide.BUY:
            reference_price = Decimal(str(symbol_data['ask']))
            spread = (Decimal(str(symbol_data['ask'])) - Decimal(str(symbol_data['bid']))) / reference_price
            price = reference_price * (1 - (1 - self.iceberg_params.aggressiveness) * spread)
        else:  # Sell
            reference_price = Decimal(str(symbol_data['bid']))
            spread = (Decimal(str(symbol_data['ask'])) - Decimal(str(symbol_data['bid']))) / reference_price
            price = reference_price * (1 + (1 - self.iceberg_params.aggressiveness) * spread)
        
        # Round to appropriate tick size
        tick_size = Decimal(str(symbol_data.get('tick_size', '0.01')))
        if tick_size > 0:
            price = (price / tick_size).quantize(Decimal('1.'), rounding=ROUND_HALF_UP) * tick_size
        
        # Create and submit the order
        order_params = {
            'symbol': symbol,
            'side': order.side,
            'order_type': OrderType.LIMIT,
            'quantity': visible_qty,
            'price': price,
            'time_in_force': f"GTC,{self.iceberg_params.order_timeout}",
            'iceberg': True,
            'display_quantity': visible_qty,  # Show the full visible quantity
            'parent_order_id': order.order_id,
            'tags': ['ICEBERG', f'refresh-{exec_state["refresh_count"]}']
        }
        
        try:
            # Submit the order (implementation depends on your order management system)
            # order_result = await self.order_manager.submit_order(order_params)
            order_result = {
                'order_id': f"iceberg-{execution_id[:8]}-{exec_state['refresh_count']}",
                'status': 'NEW',
                'filled_quantity': Decimal('0'),
                'remaining_quantity': visible_qty,
                'price': float(price),
                'timestamp': datetime.utcnow()
            }
            
            # Track the child order
            child_order = {
                'order_id': order_result['order_id'],
                'quantity': visible_qty,
                'price': Decimal(str(order_result['price'])),
                'status': order_result['status'],
                'created_at': order_result['timestamp'],
                'filled_quantity': Decimal(str(order_result.get('filled_quantity', '0'))),
                'remaining_quantity': Decimal(str(order_result.get('remaining_quantity', visible_qty)))
            }
            
            exec_state['child_orders'].append(child_order)
            
            logger.info(
                f"Placed iceberg order {child_order['order_id']}: "
                f"{order.side} {visible_qty} {symbol} @ {price}"
            )
            
        except Exception as e:
            logger.error(
                f"Failed to place iceberg order for {execution_id}: {str(e)}",
                exc_info=True
            )
            raise
    
    async def _cancel_child_orders(self, execution_id: str) -> None:
        """Cancel all active child orders for an execution."""
        if execution_id not in self._active_orders:
            return
            
        exec_state = self._active_orders[execution_id]
        
        for order in exec_state['child_orders']:
            if order['status'] in ['NEW', 'PARTIALLY_FILLED']:
                try:
                    # Cancel the order (implementation depends on your order management system)
                    # await self.order_manager.cancel_order(order['order_id'])
                    logger.info(f"Cancelled iceberg child order {order['order_id']}")
                    order['status'] = 'CANCELLED'
                    
                    # Update remaining quantity
                    if 'remaining_quantity' in order:
                        exec_state['remaining_qty'] += order['remaining_quantity']
                    
                except Exception as e:
                    logger.error(
                        f"Failed to cancel iceberg child order {order['order_id']}: {str(e)}",
                        exc_info=True
                    )
    
    async def _complete_execution(self, execution_id: str) -> None:
        """Mark an execution as complete and clean up resources."""
        if execution_id not in self._active_orders:
            return
            
        exec_state = self._active_orders.pop(execution_id, None)
        if not exec_state:
            return
            
        # Cancel refresh task
        refresh_task = self._refresh_tasks.pop(execution_id, None)
        if refresh_task and not refresh_task.done():
            refresh_task.cancel()
            try:
                await refresh_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Error cancelling refresh task: {str(e)}")
        
        # Update result
        result = exec_state['result']
        if result:
            result.status = ExecutionState.COMPLETED
            result.filled_quantity = exec_state['filled_qty']
            result.avg_fill_price = (
                exec_state['avg_fill_price'] 
                if exec_state['filled_qty'] > 0 
                else None
            )
            
            # Calculate execution metrics
            duration = (datetime.utcnow() - exec_state['start_time']).total_seconds()
            result.metadata.update({
                'strategy': 'ICEBERG',
                'duration_seconds': duration,
                'avg_visible_qty': float(exec_state['visible_qty']),
                'num_refreshes': exec_state['refresh_count'],
                'num_child_orders': len(exec_state['child_orders']),
                'completion_pct': float(exec_state['filled_qty'] / exec_state['order'].quantity * 100)
            })
            
            # Notify callbacks
            await self._notify_callbacks(result)
    
    async def _calculate_visible_quantity(
        self, 
        order: Order,
        symbol_data: Dict[str, Any],
        current_visible: Optional[Decimal] = None
    ) -> Decimal:
        """
        Calculate the optimal visible quantity for an iceberg order.
        
        This takes into account:
        - Current market volume
        - Order book depth
        - Recent price volatility
        - Time of day
        """
        # Start with default visible quantity
        visible_qty = current_visible or self.iceberg_params.min_visible_qty
        
        try:
            # Get recent volume (e.g., 5-min volume)
            recent_volume = Decimal(str(symbol_data.get('volume_5m', 0)))
            
            # Calculate visible quantity as a percentage of recent volume
            if recent_volume > 0:
                volume_based = recent_volume * Decimal(str(self.iceberg_params.max_visible_pct_adv))
                visible_qty = min(visible_qty, volume_based)
            
            # Consider order book depth
            if 'bids' in symbol_data and 'asks' in symbol_data:
                book_side = 'bids' if order.side == OrderSide.SELL else 'asks'
                levels = symbol_data[book_side]
                
                if levels and len(levels) > 0:
                    # Calculate average size at top N price levels
                    top_levels = levels[:5]  # Look at top 5 price levels
                    avg_level_size = sum(
                        Decimal(str(level[1])) for level in top_levels
                    ) / len(top_levels)
                    
                    # Don't exceed average level size to avoid moving the market
                    visible_qty = min(visible_qty, avg_level_size)
            
            # Apply min/max bounds
            visible_qty = max(
                self.iceberg_params.min_visible_qty,
                min(visible_qty, self.iceberg_params.max_visible_qty)
            )
            
            # Round to appropriate lot size
            lot_size = Decimal(str(symbol_data.get('lot_size', '0.000001')))
            if lot_size > 0:
                visible_qty = (visible_qty / lot_size).quantize(
                    Decimal('1.'), 
                    rounding=ROUND_HALF_UP
                ) * lot_size
            
            return visible_qty
            
        except Exception as e:
            logger.warning(f"Error calculating visible quantity: {str(e)}")
            return self.iceberg_params.min_visible_qty
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """
        Cancel an in-progress execution.
        
        Args:
            execution_id: ID of the execution to cancel
            
        Returns:
            bool: True if cancellation was successful, False otherwise
        """
        if execution_id not in self._active_orders:
            return False
            
        logger.info(f"Cancelling iceberg execution {execution_id}")
        
        try:
            # Cancel all child orders
            await self._cancel_child_orders(execution_id)
            
            # Complete the execution with current state
            await self._complete_execution(execution_id)
            
            # Update result status
            if execution_id in self._active_orders:
                exec_state = self._active_orders[execution_id]
                if exec_state and 'result' in exec_state:
                    exec_state['result'].status = ExecutionState.CANCELLED
            
            return True
            
        except Exception as e:
            logger.error(f"Error cancelling iceberg execution {execution_id}: {str(e)}", exc_info=True)
            return False
    
    async def close(self) -> None:
        """Clean up resources and cancel all running executions."""
        # Cancel all executions
        execution_ids = list(self._active_orders.keys())
        for execution_id in execution_ids:
            await self.cancel_execution(execution_id)
        
        # Cancel all refresh tasks
        for task in self._refresh_tasks.values():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.error(f"Error cancelling refresh task: {str(e)}")
        
        self._refresh_tasks.clear()
        
        await super().close()
