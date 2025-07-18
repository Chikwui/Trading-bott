"""
Execution Service

This module provides a high-level service for order execution that manages different
execution strategies (VWAP, TWAP, Iceberg, etc.) and provides a unified interface
for the trading system to execute orders with various execution algorithms.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Type, TypeVar, Any, Callable, Awaitable, Union

from core.execution.base import ExecutionClient, ExecutionResult, ExecutionParameters
from core.execution.strategies.vwap import VWAPExecutionClient, VWAPParameters
from core.market.data import MarketDataService
from core.ml.model_registry import ModelRegistry
from core.risk.manager import RiskManager
from core.trading.order import Order, OrderSide, OrderStatus

logger = logging.getLogger(__name__)

T = TypeVar('T', bound='ExecutionService')


class ExecutionStrategy(str, Enum):
    """Available execution strategies."""
    VWAP = "vwap"
    TWAP = "twap"
    ICEBERG = "iceberg"
    SNIPER = "sniper"
    MARKET = "market"
    LIMIT = "limit"


@dataclass
class ExecutionConfig:
    """Configuration for execution service."""
    default_strategy: ExecutionStrategy = ExecutionStrategy.VWAP
    max_concurrent_orders: int = 100
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds
    enable_ml: bool = True
    default_slippage_bps: int = 5  # 0.05%
    default_partial_fill: bool = True
    
    # Strategy-specific configurations
    vwap_params: Optional[VWAPParameters] = None
    
    def get_strategy_params(self, strategy: ExecutionStrategy) -> Any:
        """Get parameters for a specific strategy."""
        if strategy == ExecutionStrategy.VWAP:
            return self.vwap_params or VWAPParameters()
        # Add other strategy params here
        return None


class ExecutionService:
    """
    High-level execution service that manages different execution strategies.
    
    This service provides a unified interface for executing orders using various
    execution algorithms (VWAP, TWAP, Iceberg, etc.) and handles retries, error
    recovery, and monitoring.
    """
    
    def __init__(
        self,
        client_id: str,
        market_data: MarketDataService,
        risk_manager: RiskManager,
        model_registry: Optional[ModelRegistry] = None,
        config: Optional[ExecutionConfig] = None
    ):
        """
        Initialize the execution service.
        
        Args:
            client_id: Unique client identifier
            market_data: Market data service instance
            risk_manager: Risk manager instance
            model_registry: Optional model registry for ML-based execution
            config: Execution configuration
        """
        self.client_id = client_id
        self.market_data = market_data
        self.risk_manager = risk_manager
        self.model_registry = model_registry
        self.config = config or ExecutionConfig()
        
        # Active execution clients by order ID
        self._clients: Dict[str, ExecutionClient] = {}
        self._lock = asyncio.Lock()
        
        # Event callbacks
        self._callbacks: Dict[str, List[Callable[[ExecutionResult], Awaitable[None]]]] = {
            'on_execution_start': [],
            'on_execution_update': [],
            'on_execution_complete': [],
            'on_execution_error': []
        }
        
        logger.info(f"Initialized ExecutionService for client {client_id}")
    
    async def execute_order(
        self,
        order: Order,
        strategy: Optional[ExecutionStrategy] = None,
        params: Optional[ExecutionParameters] = None,
        **kwargs
    ) -> str:
        """
        Execute an order using the specified strategy.
        
        Args:
            order: The order to execute
            strategy: Execution strategy to use (defaults to config.default_strategy)
            params: Execution parameters (overrides defaults if provided)
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            Execution ID for tracking
            
        Raises:
            ValueError: If the order or strategy is invalid
        """
        if not order or not order.is_valid():
            raise ValueError("Invalid order")
        
        strategy = strategy or self.config.default_strategy
        params = params or ExecutionParameters()
        
        # Create execution client based on strategy
        client = self._create_execution_client(strategy, **kwargs)
        execution_id = f"exe_{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}_{order.order_id[-6:]}"
        
        # Register callbacks
        async def on_update(result: ExecutionResult) -> None:
            await self._handle_execution_update(execution_id, result)
        
        # Store client
        async with self._lock:
            self._clients[execution_id] = client
        
        try:
            # Start execution
            await self._notify_callbacks('on_execution_start', ExecutionResult(
                execution_id=execution_id,
                client_order_id=order.client_order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                status='pending',
                metadata={
                    'strategy': strategy.value,
                    'order_type': order.order_type.value,
                    'time_in_force': order.time_in_force,
                    **params.dict()
                }
            ))
            
            # Execute the order
            client_execution_id = await client.execute_order(order, params, on_update)
            
            logger.info(
                f"Started {strategy.value.upper()} execution {execution_id} for order {order.order_id} "
                f"({order.side} {order.quantity} {order.symbol} @ {order.price or 'MKT'})"
            )
            
            return execution_id
            
        except Exception as e:
            logger.error(f"Failed to start execution {execution_id}: {str(e)}", exc_info=True)
            await self._handle_execution_error(execution_id, str(e))
            raise
    
    def _create_execution_client(
        self,
        strategy: ExecutionStrategy,
        **kwargs
    ) -> ExecutionClient:
        """
        Create an execution client for the specified strategy.
        
        Args:
            strategy: Execution strategy
            **kwargs: Strategy-specific parameters
            
        Returns:
            Initialized execution client
            
        Raises:
            ValueError: If the strategy is not supported
        """
        client_id = f"{self.client_id}_{strategy.value}_{datetime.utcnow().strftime('%H%M%S%f')}"
        
        if strategy == ExecutionStrategy.VWAP:
            # Get VWAP parameters from config or kwargs
            vwap_params = kwargs.get('vwap_params')
            if vwap_params is None and hasattr(self.config, 'vwap_params'):
                vwap_params = self.config.vwap_params
            
            return VWAPExecutionClient(
                client_id=client_id,
                market_data=self.market_data,
                risk_manager=self.risk_manager,
                model_registry=self.model_registry if self.config.enable_ml else None,
                vwap_params=vwap_params or VWAPParameters()
            )
        
        # Add other strategy clients here
        # elif strategy == ExecutionStrategy.TWAP:
        #     return TWAPExecutionClient(...)
        # elif strategy == ExecutionStrategy.ICEBERG:
        #     return IcebergExecutionClient(...)
        # elif strategy == ExecutionStrategy.SNIPER:
        #     return SniperExecutionClient(...)
        # elif strategy in (ExecutionStrategy.MARKET, ExecutionStrategy.LIMIT):
        #     return BasicExecutionClient(...)
        
        else:
            raise ValueError(f"Unsupported execution strategy: {strategy}")
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """
        Cancel an in-progress execution.
        
        Args:
            execution_id: ID of the execution to cancel
            
        Returns:
            bool: True if cancellation was successful, False otherwise
        """
        client = self._clients.get(execution_id)
        if not client:
            logger.warning(f"No active execution found for ID: {execution_id}")
            return False
        
        try:
            logger.info(f"Cancelling execution {execution_id}")
            return await client.cancel_execution(execution_id)
        except Exception as e:
            logger.error(f"Error cancelling execution {execution_id}: {str(e)}", exc_info=True)
            return False
    
    async def get_execution_status(self, execution_id: str) -> Optional[ExecutionResult]:
        """
        Get the current status of an execution.
        
        Args:
            execution_id: ID of the execution to check
            
        Returns:
            ExecutionResult if found, None otherwise
        """
        client = self._clients.get(execution_id)
        if not client:
            return None
        
        return await client.get_execution_status(execution_id)
    
    async def _handle_execution_update(self, execution_id: str, result: ExecutionResult) -> None:
        """Handle execution updates from clients."""
        try:
            # Update execution state
            if result.status in ('completed', 'cancelled', 'error'):
                async with self._lock:
                    if execution_id in self._clients:
                        client = self._clients.pop(execution_id)
                        await client.close()
            
            # Notify callbacks
            event_type = 'on_execution_complete' if result.status in ('completed', 'cancelled') else 'on_execution_update'
            await self._notify_callbacks(event_type, result)
            
            # Log completion
            if result.status == 'completed':
                logger.info(
                    f"Execution {execution_id} completed: "
                    f"Filled {result.filled_quantity}/{result.quantity} "
                    f"@ avg price {result.avg_fill_price}"
                )
            elif result.status == 'cancelled':
                logger.info(f"Execution {execution_id} was cancelled")
            
        except Exception as e:
            logger.error(f"Error handling execution update: {str(e)}", exc_info=True)
            await self._handle_execution_error(execution_id, str(e))
    
    async def _handle_execution_error(self, execution_id: str, error: str) -> None:
        """Handle execution errors."""
        try:
            # Clean up client
            async with self._lock:
                if execution_id in self._clients:
                    client = self._clients.pop(execution_id)
                    await client.close()
            
            # Create error result
            result = ExecutionResult(
                execution_id=execution_id,
                client_order_id='',
                symbol='',
                side=OrderSide.BUY,  # Dummy value
                quantity=Decimal('0'),
                status='error',
                metadata={'error': error}
            )
            
            # Notify callbacks
            await self._notify_callbacks('on_execution_error', result)
            
            logger.error(f"Execution {execution_id} failed: {error}")
            
        except Exception as e:
            logger.error(f"Error handling execution error: {str(e)}", exc_info=True)
    
    def register_callback(
        self,
        event_type: str,
        callback: Callable[[ExecutionResult], Awaitable[None]]
    ) -> None:
        """
        Register a callback for execution events.
        
        Args:
            event_type: One of 'on_execution_start', 'on_execution_update', 
                      'on_execution_complete', 'on_execution_error'
            callback: Async function that takes an ExecutionResult
        """
        if event_type not in self._callbacks:
            raise ValueError(f"Invalid event type: {event_type}")
        
        self._callbacks[event_type].append(callback)
    
    async def _notify_callbacks(
        self,
        event_type: str,
        result: ExecutionResult
    ) -> None:
        """Notify all registered callbacks of an event."""
        if event_type not in self._callbacks:
            logger.warning(f"No callbacks registered for event type: {event_type}")
            return
        
        for callback in self._callbacks[event_type]:
            try:
                await callback(result)
            except Exception as e:
                logger.error(f"Error in {event_type} callback: {str(e)}", exc_info=True)
    
    async def close(self) -> None:
        """Clean up resources and cancel all running executions."""
        logger.info("Shutting down ExecutionService...")
        
        # Cancel all running executions
        execution_ids = list(self._clients.keys())
        for execution_id in execution_ids:
            await self.cancel_execution(execution_id)
        
        # Clear callbacks
        self._callbacks.clear()
        
        logger.info("ExecutionService shutdown complete")
