"""
Base exchange interface and common functionality.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import asyncio
import time
import hmac
import hashlib
import json
from datetime import datetime
from decimal import Decimal

from core.utils.logger import get_logger
from core.models.order import Order, OrderType, OrderSide, OrderStatus
from core.models.ticker import Ticker
from core.models.ohlcv import OHLCV
from core.models.balance import Balance
from core.models.position import Position

logger = get_logger(__name__)

class ExchangeError(Exception):
    """Base exception for exchange-related errors."""
    pass

class ExchangeConnectionError(ExchangeError):
    """Raised when there is a connection error with the exchange."""
    pass

class ExchangeAPIError(ExchangeError):
    """Raised when the exchange API returns an error."""
    pass

class InsufficientFunds(ExchangeError):
    """Raised when there are insufficient funds to place an order."""
    pass

class OrderNotFound(ExchangeError):
    """Raised when an order is not found."""
    pass

class BaseExchange(ABC):
    """Base class for exchange implementations."""
    
    def __init__(self, 
                 api_key: str = None, 
                 api_secret: str = None, 
                 api_passphrase: str = None,
                 sandbox: bool = False):
        """Initialize the exchange client.
        
        Args:
            api_key: API key for the exchange
            api_secret: API secret for the exchange
            api_passphrase: API passphrase (if required by the exchange)
            sandbox: Whether to use the sandbox/testnet environment
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.api_passphrase = api_passphrase
        self.sandbox = sandbox
        self._session = None
        self._ws = None
        self._last_request_time = 0
        self._rate_limit_semaphore = asyncio.Semaphore(10)  # Default rate limit
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the exchange."""
        pass
    
    @property
    @abstractmethod
    def base_url(self) -> str:
        """Return the base URL for the exchange's REST API."""
        pass
    
    @property
    @abstractmethod
    def ws_url(self) -> str:
        """Return the WebSocket URL for the exchange."""
        pass
    
    async def _rate_limit(self) -> None:
        """Handle rate limiting."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < 0.1:  # 10 requests per second by default
            await asyncio.sleep(0.1 - elapsed)
        self._last_request_time = time.time()
    
    def _generate_signature(self, method: str, path: str, data: dict = None) -> str:
        """Generate a signature for authenticated requests.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            path: API endpoint path
            data: Request payload
            
        Returns:
            Signature string
        """
        timestamp = str(int(time.time() * 1000))
        message = timestamp + method.upper() + path
        
        if data:
            if method.upper() == 'GET':
                message += '?' + '&'.join(f"{k}={v}" for k, v in sorted(data.items()))
            else:
                message += json.dumps(data, separators=(',', ':'))
        
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        return signature, timestamp
    
    @abstractmethod
    async def connect(self) -> None:
        """Connect to the exchange."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close the connection to the exchange."""
        pass
    
    @abstractmethod
    async def get_ticker(self, symbol: str) -> Ticker:
        """Get the current ticker for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            Ticker object with price and volume information
        """
        pass
    
    @abstractmethod
    async def get_ohlcv(self, 
                       symbol: str, 
                       timeframe: str = '1h', 
                       limit: int = 100,
                       since: int = None) -> List[OHLCV]:
        """Get OHLCV (Open, High, Low, Close, Volume) data.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for the OHLCV data (e.g., '1h', '4h', '1d')
            limit: Number of candles to return
            since: Timestamp in milliseconds for the start time
            
        Returns:
            List of OHLCV objects
        """
        pass
    
    @abstractmethod
    async def get_balance(self) -> Dict[str, Balance]:
        """Get the account balance.
        
        Returns:
            Dictionary of Balance objects keyed by currency
        """
        pass
    
    @abstractmethod
    async def get_positions(self) -> List[Position]:
        """Get open positions.
        
        Returns:
            List of Position objects
        """
        pass
    
    @abstractmethod
    async def create_order(self, 
                         symbol: str, 
                         order_type: OrderType, 
                         side: OrderSide,
                         amount: float,
                         price: float = None,
                         params: dict = None) -> Order:
        """Create a new order.
        
        Args:
            symbol: Trading pair symbol
            order_type: Type of order (MARKET, LIMIT, etc.)
            side: Order side (BUY or SELL)
            amount: Amount of the base currency to buy/sell
            price: Price for limit orders
            params: Additional parameters specific to the exchange
            
        Returns:
            Order object with the order details
        """
        pass
    
    @abstractmethod
    async def get_order(self, order_id: str, symbol: str = None) -> Order:
        """Get order details.
        
        Args:
            order_id: Order ID
            symbol: Trading pair symbol (optional, required by some exchanges)
            
        Returns:
            Order object with the order details
        """
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str, symbol: str = None) -> bool:
        """Cancel an order.
        
        Args:
            order_id: Order ID
            symbol: Trading pair symbol (optional, required by some exchanges)
            
        Returns:
            True if the order was successfully canceled, False otherwise
        """
        pass
    
    @abstractmethod
    async def get_open_orders(self, symbol: str = None) -> List[Order]:
        """Get all open orders.
        
        Args:
            symbol: Trading pair symbol (optional)
            
        Returns:
            List of Order objects
        """
        pass
    
    @abstractmethod
    async def get_order_book(self, symbol: str, limit: int = 100) -> dict:
        """Get the order book for a symbol.
        
        Args:
            symbol: Trading pair symbol
            limit: Maximum number of orders to return
            
        Returns:
            Dictionary with 'bids' and 'asks' lists
        """
        pass
    
    @abstractmethod
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[dict]:
        """Get recent trades for a symbol.
        
        Args:
            symbol: Trading pair symbol
            limit: Maximum number of trades to return
            
        Returns:
            List of trade dictionaries
        """
        pass
    
    # WebSocket methods
    @abstractmethod
    async def watch_ticker(self, symbol: str) -> Ticker:
        """Watch the ticker for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Yields:
            Ticker objects as they are received
        """
        pass
    
    @abstractmethod
    async def watch_ohlcv(self, symbol: str, timeframe: str = '1h') -> OHLCV:
        """Watch OHLCV data for a symbol.
        
        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe for the OHLCV data
            
        Yields:
            OHLCV objects as they are received
        """
        pass
    
    @abstractmethod
    async def watch_order_book(self, symbol: str) -> dict:
        """Watch the order book for a symbol.
        
        Args:
            symbol: Trading pair symbol
            
        Yields:
            Order book updates as they are received
        """
        pass
    
    @abstractmethod
    async def watch_orders(self, symbol: str = None) -> Order:
        """Watch for order updates.
        
        Args:
            symbol: Trading pair symbol (optional)
            
        Yields:
            Order objects as they are updated
        """
        pass
    
    @abstractmethod
    async def watch_balance(self) -> Dict[str, Balance]:
        """Watch for balance updates.
        
        Yields:
            Dictionary of Balance objects as they are updated
        """
        pass
    
    @abstractmethod
    async def watch_positions(self) -> List[Position]:
        """Watch for position updates.
        
        Yields:
            List of Position objects as they are updated
        """
        pass
