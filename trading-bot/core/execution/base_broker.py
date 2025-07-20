"""
Base broker interface defining the contract for all broker implementations.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
from decimal import Decimal
from datetime import datetime
from enum import Enum, auto


class OrderType(Enum):
    """Supported order types."""
    MARKET = auto()
    LIMIT = auto()
    STOP = auto()
    STOP_LIMIT = auto()
    TRAILING_STOP = auto()
    OCO = auto()  # One-Cancels-Other


class OrderSide(Enum):
    """Order side (buy/sell)."""
    BUY = auto()
    SELL = auto()


class OrderStatus(Enum):
    """Order status."""
    NEW = auto()
    PARTIALLY_FILLED = auto()
    FILLED = auto()
    CANCELED = auto()
    REJECTED = auto()
    EXPIRED = auto()


class TimeInForce(Enum):
    """Time in force for orders."""
    GTC = auto()  # Good Till Cancel
    IOC = auto()  # Immediate or Cancel
    FOK = auto()  # Fill or Kill
    DAY = auto()  # Day order
    GTD = auto()  # Good Till Date


class Order:
    """Order representation."""
    
    def __init__(
        self,
        symbol: str,
        order_type: OrderType,
        side: OrderSide,
        quantity: Union[Decimal, float, str],
        price: Optional[Union[Decimal, float, str]] = None,
        stop_price: Optional[Union[Decimal, float, str]] = None,
        time_in_force: TimeInForce = TimeInForce.GTC,
        expire_time: Optional[datetime] = None,
        client_order_id: Optional[str] = None,
    ):
        self.symbol = symbol.upper()
        self.order_type = order_type
        self.side = side
        self.quantity = Decimal(str(quantity))
        self.price = Decimal(str(price)) if price is not None else None
        self.stop_price = Decimal(str(stop_price)) if stop_price is not None else None
        self.time_in_force = time_in_force
        self.expire_time = expire_time
        self.client_order_id = client_order_id
        
        # Will be set by broker
        self.order_id: Optional[str] = None
        self.status: OrderStatus = OrderStatus.NEW
        self.filled_quantity: Decimal = Decimal('0')
        self.avg_fill_price: Optional[Decimal] = None
        self.commission: Optional[Decimal] = None
        self.created_at: datetime = datetime.utcnow()
        self.updated_at: datetime = self.created_at
        self.metadata: Dict = {}


class Position:
    """Position representation."""
    
    def __init__(
        self,
        symbol: str,
        quantity: Union[Decimal, float, str],
        avg_price: Union[Decimal, float, str],
        leverage: int = 1,
        unrealized_pnl: Optional[Union[Decimal, float, str]] = None,
        realized_pnl: Optional[Union[Decimal, float, str]] = None,
    ):
        self.symbol = symbol.upper()
        self.quantity = Decimal(str(quantity))
        self.avg_price = Decimal(str(avg_price))
        self.leverage = int(leverage)
        self.unrealized_pnl = Decimal(str(unrealized_pnl)) if unrealized_pnl is not None else None
        self.realized_pnl = Decimal(str(realized_pnl)) if realized_pnl is not None else None
        self.timestamp: datetime = datetime.utcnow()


class AccountInfo:
    """Account information."""
    
    def __init__(
        self,
        account_id: str,
        balance: Union[Decimal, float, str],
        equity: Optional[Union[Decimal, float, str]] = None,
        available_funds: Optional[Union[Decimal, float, str]] = None,
        initial_margin: Optional[Union[Decimal, float, str]] = None,
        maintenance_margin: Optional[Union[Decimal, float, str]] = None,
        currency: str = 'USD',
        leverage: int = 1,
    ):
        self.account_id = str(account_id)
        self.balance = Decimal(str(balance))
        self.equity = Decimal(str(equity)) if equity is not None else self.balance
        self.available_funds = (
            Decimal(str(available_funds)) 
            if available_funds is not None 
            else self.balance
        )
        self.initial_margin = Decimal(str(initial_margin)) if initial_margin is not None else None
        self.maintenance_margin = (
            Decimal(str(maintenance_margin)) 
            if maintenance_margin is not None 
            else None
        )
        self.currency = currency.upper()
        self.leverage = int(leverage)
        self.timestamp: datetime = datetime.utcnow()


class BaseBroker(ABC):
    """
    Abstract base class defining the broker interface.
    
    All broker implementations must inherit from this class and implement all abstract methods.
    """
    
    @abstractmethod
    async def connect(self) -> None:
        """Initialize connection to the broker."""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Close connection to the broker."""
        pass
    
    @abstractmethod
    async def get_account_info(self) -> AccountInfo:
        """
        Get current account information.
        
        Returns:
            AccountInfo: Current account information
        """
        pass
    
    @abstractmethod
    async def get_positions(self) -> Dict[str, Position]:
        """
        Get all open positions.
        
        Returns:
            Dict[str, Position]: Dictionary of symbol -> Position
        """
        pass
    
    @abstractmethod
    async def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position for a specific symbol.
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL')
            
        Returns:
            Position if exists, None otherwise
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
            OrderRejected: If order is rejected by the broker
        """
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            bool: True if cancellation was successful
        """
        pass
    
    @abstractmethod
    async def get_order_status(self, order_id: str) -> OrderStatus:
        """
        Get status of an order.
        
        Args:
            order_id: ID of the order
            
        Returns:
            OrderStatus: Current status of the order
        """
        pass
    
    @abstractmethod
    async def get_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """
        Get all open orders, optionally filtered by symbol.
        
        Args:
            symbol: Optional symbol to filter by
            
        Returns:
            List[Order]: List of open orders
        """
        pass
    
    @abstractmethod
    async def get_historical_data(
        self,
        symbol: str,
        interval: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 500,
    ) -> List[Dict]:
        """
        Get historical price data.
        
        Args:
            symbol: Trading symbol
            interval: Time interval (e.g., '1m', '1h', '1d')
            start_time: Start time (default: now - 1 day)
            end_time: End time (default: now)
            limit: Maximum number of candles to return
            
        Returns:
            List of OHLCV candles
        """
        pass
    
    @abstractmethod
    async def get_order_book(self, symbol: str, depth: int = 20) -> Dict:
        """
        Get order book (market depth) for a symbol.
        
        Args:
            symbol: Trading symbol
            depth: Number of price levels to return
            
        Returns:
            Dict with 'bids' and 'asks' lists
        """
        pass
    
    @abstractmethod
    async def get_ticker(self, symbol: str) -> Dict:
        """
        Get 24h ticker price change statistics.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict with price and volume information
        """
        pass
    
    @abstractmethod
    async def get_balances(self) -> Dict[str, Dict]:
        """
        Get all account balances.
        
        Returns:
            Dict of asset -> balance info
        """
        pass
    
    @abstractmethod
    async def get_leverage(self, symbol: str) -> int:
        """
        Get current leverage for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            int: Current leverage (e.g., 10 for 10x)
        """
        pass
    
    @abstractmethod
    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """
        Set leverage for a symbol.
        
        Args:
            symbol: Trading symbol
            leverage: Desired leverage (e.g., 10 for 10x)
            
        Returns:
            bool: True if successful
        """
        pass
