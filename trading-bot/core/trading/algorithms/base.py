""
Base classes for execution algorithms."""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from decimal import Decimal

from ...order_types import Order, OrderStatus, TimeInForce
from ...position_manager import PositionManager


class ExecutionAlgorithm(ABC):
    """Base class for all execution algorithms."""
    
    def __init__(
        self,
        exchange_adapter: Any,
        position_manager: PositionManager,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the execution algorithm.
        
        Args:
            exchange_adapter: Adapter for exchange communication
            position_manager: For position tracking and risk checks
            config: Algorithm-specific configuration
        """
        self.exchange_adapter = exchange_adapter
        self.position_manager = position_manager
        self.config = config or {}
        self._is_running = False
    
    @abstractmethod
    async def execute(
        self,
        order: Order,
        params: Any = None
    ) -> Order:
        """Execute an order using this algorithm.
        
        Args:
            order: The order to execute
            params: Algorithm-specific parameters
            
        Returns:
            The executed order with updated status
        """
        pass
    
    async def cancel(self) -> bool:
        """Cancel the current execution.
        
        Returns:
            True if cancellation was successful, False otherwise
        """
        self._is_running = False
        return True
    
    def _calculate_slice_size(
        self,
        total_quantity: Decimal,
        num_slices: int,
        min_size: Optional[Decimal] = None,
        max_size: Optional[Decimal] = None
    ) -> Decimal:
        """Calculate the size of each execution slice.
        
        Args:
            total_quantity: Total quantity to execute
            num_slices: Number of slices to divide into
            min_size: Minimum size per slice (optional)
            max_size: Maximum size per slice (optional)
            
        Returns:
            Size of each slice
        """
        if num_slices <= 0:
            raise ValueError("Number of slices must be positive")
            
        slice_size = total_quantity / Decimal(num_slices)
        
        if min_size is not None and slice_size < min_size:
            slice_size = min_size
            
        if max_size is not None and slice_size > max_size:
            slice_size = max_size
            
        return slice_size.quantize(Decimal('0.00000001'))
    
    def _calculate_slice_interval(
        self,
        start_time: datetime,
        end_time: datetime,
        num_slices: int,
        randomize: bool = False
    ) -> float:
        """Calculate the interval between execution slices.
        
        Args:
            start_time: When to start execution
            end_time: When to complete execution
            num_slices: Number of slices
            randomize: Whether to randomize intervals slightly
            
        Returns:
            Interval in seconds between slices
        """
        if num_slices <= 1:
            return 0.0
            
        total_seconds = (end_time - start_time).total_seconds()
        interval = total_seconds / (num_slices - 1)
        
        if randomize:
            # Randomize interval by ±10%
            import random
            variation = interval * 0.1
            interval += random.uniform(-variation, variation)
        
        return max(0.1, interval)  # Minimum 100ms between slices
    
    def _get_market_conditions(self, symbol: str) -> Dict[str, Any]:
        """Get current market conditions for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            Dictionary with market data like spread, volume, etc.
        """
        # This is a placeholder - implement actual market data retrieval
        return {
            'spread': Decimal('0.0005'),  # 5 basis points
            'volume_24h': Decimal('1000'),  # Base currency
            'vwap': Decimal('50000'),  # Volume-weighted average price
            'last_price': Decimal('50000'),
            'best_bid': Decimal('49999.8'),
            'best_ask': Decimal('50000.2'),
        }
    
    def _calculate_aggressive_price(
        self,
        side: str,
        reference_price: Decimal,
        spread_fraction: float = 0.0
    ) -> Decimal:
        """Calculate an aggressive price for immediate execution.
        
        Args:
            side: 'buy' or 'sell'
            reference_price: Current market price
            spread_fraction: Fraction of spread to cross (0.0-1.0)
            
        Returns:
            Aggressive price for immediate execution
        """
        spread_fraction = max(0.0, min(1.0, spread_fraction))
        market = self._get_market_conditions('')
        
        if side.lower() == 'buy':
            # For buys, we pay up to the ask
            price = market['best_ask']
            if spread_fraction > 0 and 'best_bid' in market:
                # Cross part of the spread
                spread = market['best_ask'] - market['best_bid']
                price = market['best_bid'] + (spread * Decimal(str(spread_fraction)))
        else:  # sell
            # For sells, we pay down to the bid
            price = market['best_bid']
            if spread_fraction > 0 and 'best_ask' in market:
                # Cross part of the spread
                spread = market['best_ask'] - market['best_bid']
                price = market['best_ask'] - (spread * Decimal(str(spread_fraction)))
        
        return price.quantize(Decimal('0.00000001'))
