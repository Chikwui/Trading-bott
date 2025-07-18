"""
Order Execution Service with support for various execution algorithms.

This module provides an OrderExecutionService that handles order execution
using different algorithms (TWAP, VWAP, etc.) and manages order routing.
"""
import asyncio
import logging
from decimal import Decimal
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field
from enum import Enum, auto

from .order_types import Order, OrderSide, OrderType, TimeInForce, OrderStatus
from .position_manager import PositionManager

logger = logging.getLogger(__name__)

class ExecutionAlgorithm(str, Enum):
    """Supported execution algorithms."""
    MARKET = "market"
    LIMIT = "limit"
    TWAP = "twap"
    VWAP = "vwap"
    ICEBERG = "iceberg"
    SNIPER = "sniper"


@dataclass
class ExecutionParameters:
    """Parameters for order execution."""
    algorithm: ExecutionAlgorithm = ExecutionAlgorithm.MARKET
    urgency: int = 1  # 1-5, where 5 is most aggressive
    max_slippage: Optional[Decimal] = None
    max_retries: int = 3
    retry_delay: float = 0.5
    routing_strategy: str = "default"
    algorithm_params: Dict[str, Any] = field(default_factory=dict)


class OrderExecutionService:
    """
    Service responsible for executing orders using various algorithms.
    
    Handles order routing, execution algorithms, and real-time metrics.
    """
    
    def __init__(
        self,
        exchange_adapter: Any,
        position_manager: PositionManager,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the execution service.
        
        Args:
            exchange_adapter: Adapter for exchange communication
            position_manager: For position tracking and risk checks
            config: Configuration dictionary
        """
        self.exchange_adapter = exchange_adapter
        self.position_manager = position_manager
        self.config = config or {}
        
        # Active executions
        self._active_executions: Dict[str, asyncio.Task] = {}
        self._execution_lock = asyncio.Lock()
        
        # Metrics
        self.metrics = {
            'orders_executed': 0,
            'shares_executed': Decimal('0'),
            'notional_value': Decimal('0'),
            'slippage': Decimal('0'),
            'latency': []
        }
    
    async def execute_order(
        self,
        order: Order,
        execution_params: Optional[ExecutionParameters] = None
    ) -> Order:
        """Execute an order using the specified algorithm.
        
        Args:
            order: The order to execute
            execution_params: Optional execution parameters
            
        Returns:
            The executed order with updated status
        """
        if execution_params is None:
            execution_params = ExecutionParameters()
        
        # Validate order
        await self._validate_order(order)
        
        # Select execution algorithm
        executor = self._get_executor(execution_params.algorithm)
        
        # Execute the order
        start_time = datetime.utcnow()
        try:
            executed_order = await executor(order, execution_params)
            
            # Update metrics
            await self._update_metrics(executed_order, start_time)
            
            return executed_order
            
        except Exception as e:
            logger.error(f"Error executing order {order.order_id}: {str(e)}")
            order.status = OrderStatus.REJECTED
            order.error_message = str(e)
            raise
    
    async def cancel_execution(self, order_id: str) -> bool:
        """Cancel an in-progress execution.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            True if cancellation was successful, False otherwise
        """
        async with self._execution_lock:
            execution_task = self._active_executions.get(order_id)
            if execution_task and not execution_task.done():
                execution_task.cancel()
                try:
                    await execution_task
                except asyncio.CancelledError:
                    logger.info(f"Execution for order {order_id} was cancelled")
                    return True
                except Exception as e:
                    logger.error(f"Error cancelling execution for order {order_id}: {e}")
                    return False
        return False
    
    def _get_executor(
        self,
        algorithm: ExecutionAlgorithm
    ) -> Callable[[Order, ExecutionParameters], Any]:
        """Get the executor function for the specified algorithm."""
        executors = {
            ExecutionAlgorithm.MARKET: self._execute_market,
            ExecutionAlgorithm.TWAP: self._execute_twap,
            ExecutionAlgorithm.VWAP: self._execute_vwap,
            ExecutionAlgorithm.ICEBERG: self._execute_iceberg,
            ExecutionAlgorithm.SNIPER: self._execute_sniper,
        }
        return executors.get(algorithm, self._execute_market)
    
    async def _execute_market(
        self,
        order: Order,
        params: ExecutionParameters
    ) -> Order:
        """Execute an order using market execution."""
        # Simple market execution
        order.execution_algorithm = ExecutionAlgorithm.MARKET
        
        # Submit to exchange
        result = await self.exchange_adapter.submit_order(
            symbol=order.symbol,
            side=order.side,
            order_type=OrderType.MARKET,
            quantity=order.quantity,
            time_in_force=order.time_in_force or TimeInForce.IOC,
            client_order_id=order.client_order_id,
            **order.metadata
        )
        
        # Update order with execution details
        self._update_order_from_execution(order, result)
        return order
    
    async def _execute_twap(
        self,
        order: Order,
        params: ExecutionParameters
    ) -> Order:
        """Execute an order using TWAP (Time-Weighted Average Price) algorithm."""
        from .algorithms.twap import TWAPExecutor
        
        executor = TWAPExecutor(
            exchange_adapter=self.exchange_adapter,
            position_manager=self.position_manager,
            config=params.algorithm_params
        )
        
        return await executor.execute(order, params)
    
    async def _execute_vwap(
        self,
        order: Order,
        params: ExecutionParameters
    ) -> Order:
        """Execute an order using VWAP (Volume-Weighted Average Price) algorithm."""
        from .algorithms.vwap import VWAPExecutor
        
        executor = VWAPExecutor(
            exchange_adapter=self.exchange_adapter,
            position_manager=self.position_manager,
            config=params.algorithm_params
        )
        
        return await executor.execute(order, params)
    
    async def _execute_iceberg(
        self,
        order: Order,
        params: ExecutionParameters
    ) -> Order:
        """Execute an order using Iceberg algorithm."""
        from .algorithms.iceberg import IcebergExecutor
        
        executor = IcebergExecutor(
            exchange_adapter=self.exchange_adapter,
            position_manager=self.position_manager,
            config=params.algorithm_params
        )
        
        return await executor.execute(order, params)
    
    async def _execute_sniper(
        self,
        order: Order,
        params: ExecutionParameters
    ) -> Order:
        """Execute an order using Sniper algorithm (aggressive execution)."""
        from .algorithms.sniper import SniperExecutor
        
        executor = SniperExecutor(
            exchange_adapter=self.exchange_adapter,
            position_manager=self.position_manager,
            config=params.algorithm_params
        )
        
        return await executor.execute(order, params)
    
    async def _validate_order(self, order: Order) -> None:
        """Validate an order before execution."""
        if order.status != OrderStatus.NEW:
            raise ValueError(f"Cannot execute order {order.order_id} with status {order.status}")
        
        if order.quantity <= 0:
            raise ValueError(f"Invalid order quantity: {order.quantity}")
        
        # Add more validations as needed
    
    def _update_order_from_execution(self, order: Order, execution: Dict[str, Any]) -> None:
        """Update order with execution details."""
        order.status = OrderStatus(execution.get('status', 'REJECTED').upper())
        order.filled_quantity = Decimal(str(execution.get('filled_quantity', 0)))
        order.remaining_quantity = order.quantity - order.filled_quantity
        order.filled_price = Decimal(str(execution.get('filled_price', 0)))
        order.fees = Decimal(str(execution.get('fees', 0)))
        order.execution_time = execution.get('timestamp', datetime.utcnow())
        
        # Update position if position manager is available
        if hasattr(self, 'position_manager') and order.filled_quantity > 0:
            asyncio.create_task(
                self.position_manager.update_position(
                    position_id=order.position_id,
                    order=order,
                    price=order.filled_price
                )
            )
    
    async def _update_metrics(self, order: Order, start_time: datetime) -> None:
        """Update execution metrics."""
        if order.status != OrderStatus.FILLED:
            return
            
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        self.metrics['orders_executed'] += 1
        self.metrics['shares_executed'] += order.filled_quantity
        self.metrics['notional_value'] += order.filled_quantity * order.filled_price
        self.metrics['latency'].append(execution_time)
        
        # Keep only the last 1000 latency measurements
        if len(self.metrics['latency']) > 1000:
            self.metrics['latency'] = self.metrics['latency'][-1000:]
    
    async def get_execution_metrics(self) -> Dict[str, Any]:
        """Get current execution metrics."""
        return {
            **self.metrics,
            'avg_latency': (
                sum(self.metrics['latency']) / len(self.metrics['latency']) 
                if self.metrics['latency'] else 0
            )
        }
