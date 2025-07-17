"""
Broker Interface Module

Defines the abstract base class for all broker implementations.
Supports both live and backtesting modes.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
import logging
from decimal import Decimal

from ..models import Order, OrderType, OrderSide, OrderStatus, TimeInForce, Position

logger = logging.getLogger(__name__)

class BrokerType(Enum):
    """Type of broker implementation."""
    LIVE = auto()
    BACKTEST = auto()
    SIMULATION = auto()

class BrokerError(Exception):
    """Base exception for broker-related errors."""
    pass

class ConnectionError(BrokerError):
    """Raised when there's an issue connecting to the broker."""
    pass

class OrderError(BrokerError):
    """Raised when there's an issue with order execution or management."""
    pass

class MarketDataError(BrokerError):
    """Raised when there's an issue with market data retrieval."""
    pass

class Broker(ABC):
    """
    Abstract base class for all broker implementations.
    
    This class defines the interface that all broker implementations must follow.
    It supports both live and backtesting modes.
    """
    
    def __init__(self, broker_type: BrokerType, config: Dict[str, Any] = None):
        """
        Initialize the broker.
        
        Args:
            broker_type: Type of broker (LIVE, BACKTEST, SIMULATION)
            config: Configuration dictionary for the broker
        """
        self.broker_type = broker_type
        self.config = config or {}
        self.connected = False
        self.initialized = False
        self._last_update = None
        
        # Initialize logging
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def connect(self) -> bool:
        """
        Connect to the broker's API.
        
        Returns:
            bool: True if connection was successful, False otherwise
            
        Raises:
            ConnectionError: If connection fails
        """
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """
        Disconnect from the broker's API.
        """
        pass
    
    @abstractmethod
    async def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Dict containing account information (balance, equity, margin, etc.)
            
        Raises:
            BrokerError: If account info cannot be retrieved
        """
        pass
    
    @abstractmethod
    async def place_order(self, order: Order) -> str:
        """
        Place a new order.
        
        Args:
            order: Order to place
            
        Returns:
            str: Order ID assigned by the broker
            
        Raises:
            OrderError: If order placement fails
        """
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            bool: True if cancellation was successful, False otherwise
            
        Raises:
            OrderError: If order cancellation fails
        """
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get the status of an order.
        
        Args:
            order_id: ID of the order to check
            
        Returns:
            Dict containing order status information
            
        Raises:
            OrderError: If order status cannot be retrieved
        """
        pass
    
    @abstractmethod
    async def get_positions(self, symbol: str = None) -> List[Dict[str, Any]]:
        """
        Get current open positions.
        
        Args:
            symbol: Optional symbol to filter positions
            
        Returns:
            List of position dictionaries
            
        Raises:
            BrokerError: If positions cannot be retrieved
        """
        pass
    
    @abstractmethod
    async def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start: datetime = None, 
        end: datetime = None, 
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Get historical market data.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe for the data (e.g., '1m', '1h', '1d')
            start: Start time for the data
            end: End time for the data
            limit: Maximum number of data points to return
            
        Returns:
            List of OHLCV data points
            
        Raises:
            MarketDataError: If historical data cannot be retrieved
        """
        pass
    
    @abstractmethod
    async def get_current_price(self, symbol: str) -> float:
        """
        Get the current market price for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current market price
            
        Raises:
            MarketDataError: If price cannot be retrieved
        """
        pass
    
    @abstractmethod
    async def get_order_book(self, symbol: str, depth: int = 10) -> Dict[str, Any]:
        """
        Get the current order book for a symbol.
        
        Args:
            symbol: Trading symbol
            depth: Number of price levels to return
            
        Returns:
            Order book data including bids and asks
            
        Raises:
            MarketDataError: If order book cannot be retrieved
        """
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if the broker is connected.
        
        Returns:
            bool: True if connected, False otherwise
        """
        return self.connected
    
    @abstractmethod
    def get_broker_info(self) -> Dict[str, Any]:
        """
        Get information about the broker.
        
        Returns:
            Dict containing broker information
        """
        return {
            'broker_type': self.broker_type.name,
            'connected': self.connected,
            'last_update': self._last_update,
            'config': self.config
        }
    
    def __str__(self) -> str:
        """String representation of the broker."""
        info = self.get_broker_info()
        return f"{self.__class__.__name__}({info})"
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return self.__str__()
