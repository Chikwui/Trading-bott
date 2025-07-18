"""
Base execution client with ML integration for advanced order handling.

This module provides the foundation for all execution strategies with built-in
ML model integration for predictive order routing and execution optimization.
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Type, TypeVar, Union, Any, Callable, Awaitable

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator

from core.ml.model_registry import ModelRegistry
from core.market.data import MarketDataService
from core.risk.manager import RiskManager
from core.trading.order import Order, OrderSide, OrderType, OrderStatus

logger = logging.getLogger(__name__)

T = TypeVar('T', bound='ExecutionClient')
ExecutionCallback = Callable[['ExecutionResult'], Awaitable[None]]


class ExecutionStyle(str, Enum):
    """Execution style determines how aggressively to execute orders."""
    AGGRESSIVE = "aggressive"
    NEUTRAL = "neutral"
    PASSIVE = "passive"
    DARK = "dark"  # For dark pool execution


class ExecutionState(str, Enum):
    """Execution state machine states."""
    PENDING = "pending"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ERROR = "error"


@dataclass
class ExecutionParameters:
    """Parameters for order execution."""
    style: ExecutionStyle = ExecutionStyle.NEUTRAL
    urgency: float = 0.5  # 0-1 scale, 1 being most urgent
    max_slippage: Optional[float] = None  # Max acceptable slippage in basis points
    participation_rate: Optional[float] = None  # % of volume to target
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    time_zone: str = "UTC"
    market_impact_limit: Optional[float] = None  # Max acceptable market impact
    price_improvement_target: Optional[float] = None  # Target price improvement in bps
    
    # ML model parameters
    use_ml_routing: bool = True
    ml_model_version: Optional[str] = None
    ml_features: Optional[Dict[str, Any]] = None


@dataclass
class ExecutionResult:
    """Result of an execution operation."""
    order_id: str
    client_order_id: str
    symbol: str
    side: OrderSide
    quantity: Decimal
    filled_quantity: Decimal = Decimal("0")
    avg_fill_price: Optional[Decimal] = None
    fees: Decimal = Decimal("0")
    status: ExecutionState = ExecutionState.PENDING
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    @property
    def is_complete(self) -> bool:
        """Check if execution is complete."""
        return self.status in {
            ExecutionState.COMPLETED, 
            ExecutionState.CANCELLED, 
            ExecutionState.ERROR
        }
    
    def update(self, **kwargs) -> ExecutionResult:
        """Update result fields and return a new instance."""
        new_data = self.__dict__.copy()
        new_data.update(kwargs)
        new_data['updated_at'] = datetime.utcnow()
        return self.__class__(**new_data)


class ExecutionClient(ABC):
    """
    Abstract base class for all execution clients.
    
    This class provides the interface for executing orders with different strategies
    and integrates with ML models for optimal execution.
    """
    
    def __init__(
        self,
        client_id: str,
        market_data: MarketDataService,
        risk_manager: RiskManager,
        model_registry: Optional[ModelRegistry] = None,
        default_params: Optional[ExecutionParameters] = None
    ):
        self.client_id = client_id
        self.market_data = market_data
        self.risk_manager = risk_manager
        self.model_registry = model_registry
        self.default_params = default_params or ExecutionParameters()
        self._executions: Dict[str, asyncio.Task] = {}
        self._callbacks: List[ExecutionCallback] = []
        self._state: ExecutionState = ExecutionState.PENDING
        self._lock = asyncio.Lock()
    
    @property
    def state(self) -> ExecutionState:
        """Get current execution state."""
        return self._state
    
    async def execute_order(
        self,
        order: Order,
        params: Optional[ExecutionParameters] = None,
        callback: Optional[ExecutionCallback] = None
    ) -> str:
        """
        Execute an order using the specified parameters.
        
        Args:
            order: The order to execute
            params: Execution parameters (uses defaults if None)
            callback: Optional callback for execution events
            
        Returns:
            Execution ID for tracking
        """
        execution_id = f"exe_{uuid.uuid4().hex[:16]}"
        params = params or self.default_params
        
        # Register callback if provided
        if callback:
            self._callbacks.append(callback)
        
        # Create and store execution task
        task = asyncio.create_task(
            self._execute_order(execution_id, order, params),
            name=f"Execution-{execution_id}"
        )
        self._executions[execution_id] = task
        
        # Add cleanup callback
        task.add_done_callback(
            lambda t, eid=execution_id: self._cleanup_execution(eid, t)
        )
        
        return execution_id
    
    async def _execute_order(
        self,
        execution_id: str,
        order: Order,
        params: ExecutionParameters
    ) -> ExecutionResult:
        """Internal method to handle order execution."""
        result = ExecutionResult(
            execution_id=execution_id,
            client_order_id=order.client_order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            status=ExecutionState.ACTIVE
        )
        
        try:
            # Pre-execution validation
            await self._pre_execute_checks(order, params)
            
            # Execute using the strategy
            result = await self._execute_strategy(order, params, result)
            
            # Update to completed if fully filled
            if result.filled_quantity >= order.quantity:
                result = result.update(
                    status=ExecutionState.COMPLETED,
                    updated_at=datetime.utcnow()
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Execution {execution_id} failed: {str(e)}", exc_info=True)
            return result.update(
                status=ExecutionState.ERROR,
                metadata={"error": str(e), "traceback": str(e.__traceback__)}
            )
        finally:
            # Notify callbacks of final state
            await self._notify_callbacks(result)
    
    @abstractmethod
    async def _execute_strategy(
        self,
        order: Order,
        params: ExecutionParameters,
        result: ExecutionResult
    ) -> ExecutionResult:
        """
        Execute the order using the specific strategy.
        
        Subclasses must implement this method to provide specific execution logic.
        """
        pass
    
    async def _pre_execute_checks(
        self,
        order: Order,
        params: ExecutionParameters
    ) -> None:
        """Perform pre-execution validation and checks."""
        # Check order status
        if order.status not in {OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED}:
            raise ValueError(f"Cannot execute order in state: {order.status}")
        
        # Check market hours if needed
        if not await self.market_data.is_market_open(order.symbol):
            logger.warning(f"Market is closed for {order.symbol}")
        
        # Run risk checks
        risk_check = await self.risk_manager.check_order(order)
        if not risk_check.passed:
            raise ValueError(f"Risk check failed: {risk_check.reason}")
        
        # Validate execution parameters
        self._validate_execution_params(params)
    
    def _validate_execution_params(self, params: ExecutionParameters) -> None:
        """Validate execution parameters."""
        if params.participation_rate is not None and not (0 < params.participation_rate <= 1):
            raise ValueError("Participation rate must be between 0 and 1")
        
        if params.start_time and params.end_time and params.start_time >= params.end_time:
            raise ValueError("Start time must be before end time")
    
    async def _get_ml_prediction(
        self,
        symbol: str,
        features: Dict[str, Any],
        model_version: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get ML model prediction for execution optimization.
        
        Args:
            symbol: Trading symbol
            features: Input features for the model
            model_version: Specific model version to use
            
        Returns:
            Dictionary containing model predictions
        """
        if not self.model_registry:
            logger.warning("No model registry available, skipping ML prediction")
            return {}
        
        try:
            model = await self.model_registry.get_latest_model(
                f"execution_{symbol}", 
                version=model_version
            )
            if model:
                return await model.predict(features)
            return {}
        except Exception as e:
            logger.error(f"ML prediction failed: {e}", exc_info=True)
            return {}
    
    async def _notify_callbacks(self, result: ExecutionResult) -> None:
        """Notify all registered callbacks of an execution event."""
        for callback in self._callbacks:
            try:
                await callback(result)
            except Exception as e:
                logger.error(f"Error in execution callback: {e}", exc_info=True)
    
    def _cleanup_execution(self, execution_id: str, task: asyncio.Task) -> None:
        """Clean up completed or failed execution tasks."""
        if execution_id in self._executions:
            del self._executions[execution_id]
        
        # Handle task exceptions
        if task.done() and task.exception():
            logger.error(
                f"Execution {execution_id} failed with error",
                exc_info=task.exception()
            )
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """
        Cancel an in-progress execution.
        
        Args:
            execution_id: ID of the execution to cancel
            
        Returns:
            bool: True if cancellation was successful
        """
        task = self._executions.get(execution_id)
        if not task or task.done():
            return False
        
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            logger.info(f"Execution {execution_id} was cancelled")
            return True
        except Exception as e:
            logger.error(f"Error cancelling execution {execution_id}: {e}")
            return False
        
        return True
    
    async def get_execution_status(self, execution_id: str) -> Optional[ExecutionResult]:
        """
        Get the current status of an execution.
        
        Args:
            execution_id: ID of the execution to check
            
        Returns:
            ExecutionResult if found, None otherwise
        """
        task = self._executions.get(execution_id)
        if not task:
            return None
        
        if task.done():
            try:
                return task.result()
            except Exception as e:
                return ExecutionResult(
                    execution_id=execution_id,
                    client_order_id="",
                    symbol="",
                    side=OrderSide.BUY,  # Dummy value
                    quantity=Decimal("0"),
                    status=ExecutionState.ERROR,
                    metadata={"error": str(e)}
                )
        
        # For in-progress executions, return a status indicating it's still running
        return ExecutionResult(
            execution_id=execution_id,
            client_order_id="",
            symbol="",
            side=OrderSide.BUY,  # Dummy value
            quantity=Decimal("0"),
            status=ExecutionState.ACTIVE
        )
    
    async def close(self) -> None:
        """Clean up resources and cancel all running executions."""
        # Cancel all running executions
        for execution_id in list(self._executions.keys()):
            await self.cancel_execution(execution_id)
        
        # Clear callbacks
        self._callbacks.clear()
        
        # Update state
        self._state = ExecutionState.CANCELLED
        
        logger.info(f"Execution client {self.client_id} closed")
