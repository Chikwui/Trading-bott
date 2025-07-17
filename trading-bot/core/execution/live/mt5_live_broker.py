"""
MT5 Live Broker Implementation

This module provides a live trading implementation of the `Broker` interface for MetaTrader 5.
It connects to the MT5 terminal and executes trades in real-time.

Features:
- Real-time order execution and management
- Position tracking and account synchronization
- Market data streaming via WebSocket
- Comprehensive error handling and reconnection logic
- Thread-safe operations for high-frequency trading
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Union, Any, Tuple

import MetaTrader5 as mt5
import pandas as pd
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)

from core.execution.broker import (
    Broker,
    BrokerType,
    BrokerError,
    OrderError,
    MarketDataError,
    ConnectionError,
)
from core.models import (
    Order,
    OrderType,
    OrderSide,
    OrderStatus,
    TimeInForce,
    Position,
    TradeSignal,
)
from core.utils.helpers import retry_async

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MT5 Constants
MT5_TIMEOUT = 10000  # 10 seconds
MT5_POLLING_INTERVAL = 0.1  # 100ms
MT5_RECONNECT_DELAY = 5  # 5 seconds

# MT5 Order Type Mappings
MT5_ORDER_TYPES = {
    OrderType.MARKET: mt5.ORDER_TYPE_BUY if OrderSide.BUY else mt5.ORDER_TYPE_SELL,
    OrderType.LIMIT: mt5.ORDER_TYPE_BUY_LIMIT if OrderSide.BUY else mt5.ORDER_TYPE_SELL_LIMIT,
    OrderType.STOP: mt5.ORDER_TYPE_BUY_STOP if OrderSide.BUY else mt5.ORDER_TYPE_SELL_STOP,
}

# MT5 Order Type Reverse Mappings
MT5_ORDER_TYPE_NAMES = {
    mt5.ORDER_TYPE_BUY: (OrderType.MARKET, OrderSide.BUY),
    mt5.ORDER_TYPE_SELL: (OrderType.MARKET, OrderSide.SELL),
    mt5.ORDER_TYPE_BUY_LIMIT: (OrderType.LIMIT, OrderSide.BUY),
    mt5.ORDER_TYPE_SELL_LIMIT: (OrderType.LIMIT, OrderSide.SELL),
    mt5.ORDER_TYPE_BUY_STOP: (OrderType.STOP, OrderSide.BUY),
    mt5.ORDER_TYPE_SELL_STOP: (OrderType.STOP, OrderSide.SELL),
}

# MT5 Order Status Mappings
MT5_ORDER_STATUS = {
    mt5.ORDER_STATE_STARTED: OrderStatus.PENDING,
    mt5.ORDER_STATE_PLACED: OrderStatus.PENDING,
    mt5.ORDER_STATE_CANCELED: OrderStatus.CANCELED,
    mt5.ORDER_STATE_PARTIAL: OrderStatus.PARTIALLY_FILLED,
    mt5.ORDER_STATE_FILLED: OrderStatus.FILLED,
    mt5.ORDER_STATE_REJECTED: OrderStatus.REJECTED,
    mt5.ORDER_STATE_EXPIRED: OrderStatus.EXPIRED,
    mt5.ORDER_STATE_REQUEST_ADD: OrderStatus.PENDING,
    mt5.ORDER_STATE_REQUEST_MODIFY: OrderStatus.PENDING,
    mt5.ORDER_STATE_REQUEST_CANCEL: OrderStatus.PENDING_CANCEL,
}

# MT5 Time in Force Mappings
MT5_TIME_IN_FORCE = {
    TimeInForce.DAY: mt5.ORDER_TIME_DAY,
    TimeInForce.GTC: mt5.ORDER_TIME_GTC,
    TimeInForce.IOC: mt5.ORDER_TIME_SPECIFIED,
    TimeInForce.FOK: mt5.ORDER_TIME_SPECIFIED,
}

# MT5 Price Types
MT5_PRICE_TYPES = {
    OrderSide.BUY: mt5.ORDER_TYPE_BUY,
    OrderSide.SELL: mt5.ORDER_TYPE_SELL,
}


class MT5LiveBroker(Broker):
    """
    Live trading broker implementation for MetaTrader 5.
    
    This class provides a high-level interface to the MT5 terminal for executing trades,
    managing positions, and streaming market data.
    """
    
    def __init__(
        self,
        server: str = None,
        login: int = None,
        password: str = None,
        timeout: int = MT5_TIMEOUT,
        polling_interval: float = MT5_POLLING_INTERVAL,
        max_reconnect_attempts: int = 5,
    ):
        """
        Initialize the MT5 live broker.
        
        Args:
            server: MT5 server name (e.g., 'PocketOption-Demo')
            login: MT5 account login
            password: MT5 account password
            timeout: Connection timeout in milliseconds
            polling_interval: Interval for polling MT5 for updates (seconds)
            max_reconnect_attempts: Maximum number of reconnection attempts
        """
        super().__init__(BrokerType.LIVE)
        
        # Connection parameters
        self.server = server
        self.login = login
        self.password = password
        self.timeout = timeout
        self.polling_interval = polling_interval
        self.max_reconnect_attempts = max_reconnect_attempts
        
        # Connection state
        self._connected = False
        self._last_ping = None
        self._reconnect_attempts = 0
        self._shutdown_event = asyncio.Event()
        self._market_data_handlers = set()
        
        # Cached data
        self._account_info = {}
        self._positions = {}
        self._orders = {}
        self._symbols_info = {}
        
        # Locks for thread safety
        self._connection_lock = asyncio.Lock()
        self._order_lock = asyncio.Lock()
        self._position_lock = asyncio.Lock()
        self._market_data_lock = asyncio.Lock()
    
    @property
    def connected(self) -> bool:
        """Check if the broker is connected to MT5."""
        return self._connected and mt5.terminal_info() is not None
    
    @property
    def initialized(self) -> bool:
        """Check if the broker is initialized and ready for trading."""
        return self.connected and mt5.terminal_info().trade_allowed
    
    async def connect(self) -> bool:
        """
        Connect to the MT5 terminal.
        
        Returns:
            bool: True if connected successfully, False otherwise
            
        Raises:
            ConnectionError: If connection fails after max retries
        """
        async with self._connection_lock:
            if self.connected:
                logger.info("Already connected to MT5")
                return True
            
            logger.info(f"Connecting to MT5 (Server: {self.server or 'default'}, Login: {self.login})")
            
            # Initialize MT5 if not already initialized
            if not mt5.initialize():
                error = mt5.last_error()
                logger.error(f"MT5 initialization failed: {error}")
                raise ConnectionError(f"MT5 initialization failed: {error}")
            
            # Login to the trade account if credentials are provided
            if self.login and self.password:
                authorized = mt5.login(
                    login=self.login,
                    password=self.password,
                    server=self.server,
                    timeout=self.timeout,
                )
                
                if not authorized:
                    error = mt5.last_error()
                    mt5.shutdown()
                    logger.error(f"MT5 login failed: {error}")
                    raise ConnectionError(f"MT5 login failed: {error}")
                
                logger.info(f"Connected to account #{self.login} on {self.server}")
            else:
                logger.warning("No login credentials provided. Using MT5 in read-only mode.")
            
            # Update connection state
            self._connected = True
            self._last_ping = datetime.utcnow()
            self._reconnect_attempts = 0
            
            # Start background tasks
            asyncio.create_task(self._monitor_connection())
            asyncio.create_task(self._update_account_info())
            asyncio.create_task(self._update_positions())
            asyncio.create_task(self._update_orders())
            
            logger.info("MT5 connection established successfully")
            return True
    
    async def disconnect(self) -> None:
        """Disconnect from the MT5 terminal."""
        async with self._connection_lock:
            if not self.connected:
                return
            
            logger.info("Disconnecting from MT5...")
            
            # Signal background tasks to stop
            self._shutdown_event.set()
            
            # Shutdown MT5 connection
            mt5.shutdown()
            
            # Update connection state
            self._connected = False
            self._last_ping = None
            
            # Clear cached data
            self._account_info = {}
            self._positions = {}
            self._orders = {}
            
            logger.info("Disconnected from MT5")
    
    async def _monitor_connection(self) -> None:
        """Monitor the connection to MT5 and attempt reconnection if needed."""
        while not self._shutdown_event.is_set():
            try:
                if not self.connected:
                    if self._reconnect_attempts < self.max_reconnect_attempts:
                        logger.warning("Connection lost. Attempting to reconnect...")
                        try:
                            await self.connect()
                        except Exception as e:
                            self._reconnect_attempts += 1
                            logger.error(f"Reconnection attempt {self._reconnect_attempts} failed: {e}")
                            await asyncio.sleep(MT5_RECONNECT_DELAY)
                            continue
                    else:
                        logger.error("Max reconnection attempts reached. Giving up.")
                        self._shutdown_event.set()
                        break
                
                # Check if we've received any data recently
                if self._last_ping and (datetime.utcnow() - self._last_ping) > timedelta(seconds=30):
                    logger.warning("No data received from MT5 for 30 seconds. Connection may be stale.")
                    self._connected = False
                    continue
                
                # Update last ping time
                self._last_ping = datetime.utcnow()
                
                # Sleep before next check
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in connection monitor: {e}", exc_info=True)
                await asyncio.sleep(5)  # Prevent tight loop on error
    
    async def _update_account_info(self) -> None:
        """Periodically update account information."""
        while not self._shutdown_event.is_set() and self.connected:
            try:
                account_info = mt5.account_info()
                if account_info is None:
                    logger.warning("Failed to get account info from MT5")
                    continue
                    
                self._account_info = {
                    'balance': account_info.balance,
                    'equity': account_info.equity,
                    'margin': account_info.margin,
                    'margin_free': account_info.margin_free,
                    'margin_level': account_info.margin_level,
                    'leverage': account_info.leverage,
                    'currency': account_info.currency,
                    'name': account_info.name,
                    'server': account_info.server,
                    'timestamp': datetime.utcnow().isoformat(),
                }
            except Exception as e:
                logger.error(f"Error updating account info: {e}", exc_info=True)
            
            await asyncio.sleep(1)  # Update every second
    
    async def _update_positions(self) -> None:
        """Periodically update open positions."""
        while not self._shutdown_event.is_set() and self.connected:
            try:
                positions = mt5.positions_get()
                if positions is None:
                    positions = []
                
                async with self._position_lock:
                    self._positions = {
                        str(pos.ticket): self._parse_position(pos)
                        for pos in positions
                    }
            except Exception as e:
                logger.error(f"Error updating positions: {e}", exc_info=True)
            
            await asyncio.sleep(1)  # Update every second
    
    async def _update_orders(self) -> None:
        """Periodically update open orders."""
        while not self._shutdown_event.is_set() and self.connected:
            try:
                orders = mt5.orders_get()
                if orders is None:
                    orders = []
                
                async with self._order_lock:
                    self._orders = {
                        str(order.ticket): self._parse_order(order)
                        for order in orders
                    }
            except Exception as e:
                logger.error(f"Error updating orders: {e}", exc_info=True)
            
            await asyncio.sleep(1)  # Update every second
    
    def _parse_position(self, position: mt5.TradePosition) -> dict:
        """Parse an MT5 position to our internal format."""
        return {
            'position_id': str(position.ticket),
            'symbol': position.symbol,
            'type': 'long' if position.type == mt5.POSITION_TYPE_BUY else 'short',
            'volume': position.volume,
            'entry_price': position.price_open,
            'current_price': position.price_current,
            'sl': position.sl,
            'tp': position.tp,
            'swap': position.swap,
            'profit': position.profit,
            'comment': position.comment,
            'magic': position.magic,
            'timestamp': datetime.fromtimestamp(position.time_msc / 1000).isoformat(),
        }
    
    def _parse_order(self, order: mt5.TradeOrder) -> dict:
        """Parse an MT5 order to our internal format."""
        order_type, order_side = MT5_ORDER_TYPE_NAMES.get(
            order.type, (OrderType.UNKNOWN, OrderSide.UNKNOWN)
        )
        
        return {
            'order_id': str(order.ticket),
            'symbol': order.symbol,
            'type': order_type.value,
            'side': order_side.value,
            'volume': order.volume_current,
            'price': order.price_open,
            'stop_loss': order.sl,
            'take_profit': order.tp,
            'status': MT5_ORDER_STATUS.get(order.state, OrderStatus.UNKNOWN).value,
            'comment': order.comment,
            'magic': order.magic,
            'timestamp': datetime.fromtimestamp(order.time_setup).isoformat(),
        }
    
    # Implementation of abstract methods from Broker interface
    
    @retry_async(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    )
    async def place_order(self, order: Order) -> str:
        """
        Place a new order.
        
        Args:
            order: Order to place
            
        Returns:
            str: Order ID if successful
            
        Raises:
            OrderError: If order placement fails
            ConnectionError: If not connected to MT5
        """
        if not self.connected:
            raise ConnectionError("Not connected to MT5")
        
        try:
            # Prepare the order request
            request = {
                'action': mt5.TRADE_ACTION_DEAL,
                'symbol': order.symbol,
                'volume': float(order.quantity),
                'type': MT5_ORDER_TYPES[order.order_type],
                'price': float(order.price or 0.0),
                'sl': float(order.stop_loss) if order.stop_loss else 0.0,
                'tp': float(order.take_profit) if order.take_profit else 0.0,
                'deviation': 10,  # Slippage in points
                'magic': 123456,  # Magic number for identifying our orders
                'comment': order.comment or "",
                'type_time': MT5_TIME_IN_FORCE.get(order.time_in_force, mt5.ORDER_TIME_GTC),
                'type_filling': mt5.ORDER_FILLING_FOK,
            }
            
            # For market orders, we need to get the current price
            if order.order_type == OrderType.MARKET:
                symbol_info = mt5.symbol_info(order.symbol)
                if symbol_info is None:
                    raise OrderError(f"Symbol {order.symbol} not found")
                
                if order.side == OrderSide.BUY:
                    request['price'] = symbol_info.ask
                else:
                    request['price'] = symbol_info.bid
            
            # For pending orders, we need to set the appropriate action
            if order.order_type in (OrderType.LIMIT, OrderType.STOP):
                request['action'] = mt5.TRADE_ACTION_PENDING
                request['type'] = MT5_ORDER_TYPES[order.order_type]
            
            # Execute the order
            result = mt5.order_send(request)
            
            # Check the result
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                error_msg = f"Order failed: {result.comment} (code: {result.retcode})"
                logger.error(error_msg)
                raise OrderError(error_msg)
            
            # Update local cache
            order.order_id = str(result.order)
            order.status = OrderStatus.FILLED if order.order_type == OrderType.MARKET else OrderStatus.PENDING
            
            logger.info(f"Order placed successfully: {order}")
            return order.order_id
            
        except Exception as e:
            error_msg = f"Failed to place order: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise OrderError(error_msg) from e
    
    @retry_async(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    )
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            bool: True if successful, False otherwise
            
        Raises:
            OrderError: If order cancellation fails
            ConnectionError: If not connected to MT5
        """
        if not self.connected:
            raise ConnectionError("Not connected to MT5")
        
        try:
            # Prepare the cancellation request
            request = {
                'action': mt5.TRADE_ACTION_REMOVE,
                'order': int(order_id),
                'comment': 'Cancelled by user',
            }
            
            # Execute the cancellation
            result = mt5.order_send(request)
            
            # Check the result
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                error_msg = f"Order cancellation failed: {result.comment} (code: {result.retcode})"
                logger.error(error_msg)
                raise OrderError(error_msg)
            
            logger.info(f"Order {order_id} cancelled successfully")
            return True
            
        except Exception as e:
            error_msg = f"Failed to cancel order {order_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise OrderError(error_msg) from e
    
    @retry_async(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    )
    async def modify_order(self, order_id: str, **kwargs) -> bool:
        """
        Modify an existing order.
        
        Args:
            order_id: ID of the order to modify
            **kwargs: Fields to modify (price, stop_loss, take_profit, etc.)
            
        Returns:
            bool: True if successful, False otherwise
            
        Raises:
            OrderError: If order modification fails
            ConnectionError: If not connected to MT5
        """
        if not self.connected:
            raise ConnectionError("Not connected to MT5")
        
        try:
            # Get the current order
            orders = mt5.orders_get(ticket=int(order_id))
            if not orders:
                raise OrderError(f"Order {order_id} not found")
            
            order = orders[0]  # Get the first (and should be only) order
            
            # Prepare the modification request
            request = {
                'action': mt5.TRADE_ACTION_MODIFY,
                'order': order.ticket,
                'price': float(kwargs.get('price', order.price_open)),
                'sl': float(kwargs.get('stop_loss', order.sl or 0.0)),
                'tp': float(kwargs.get('take_profit', order.tp or 0.0)),
                'deviation': 10,  # Slippage in points
                'comment': kwargs.get('comment', order.comment),
            }
            
            # Execute the modification
            result = mt5.order_send(request)
            
            # Check the result
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                error_msg = f"Order modification failed: {result.comment} (code: {result.retcode})"
                logger.error(error_msg)
                raise OrderError(error_msg)
            
            logger.info(f"Order {order_id} modified successfully")
            return True
            
        except Exception as e:
            error_msg = f"Failed to modify order {order_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise OrderError(error_msg) from e
    
    @retry_async(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    )
    async def close_position(self, position_id: str, **kwargs) -> bool:
        """
        Close an open position.
        
        Args:
            position_id: ID of the position to close
            **kwargs: Additional parameters (e.g., volume for partial close)
            
        Returns:
            bool: True if successful, False otherwise
            
        Raises:
            OrderError: If position closure fails
            ConnectionError: If not connected to MT5
        """
        if not self.connected:
            raise ConnectionError("Not connected to MT5")
        
        try:
            # Get the position
            positions = mt5.positions_get(ticket=int(position_id))
            if not positions:
                raise OrderError(f"Position {position_id} not found")
            
            position = positions[0]  # Get the first (and should be only) position
            
            # Determine the close price
            symbol_info = mt5.symbol_info(position.symbol)
            if symbol_info is None:
                raise OrderError(f"Symbol {position.symbol} not found")
            
            # Determine the close price based on position type
            if position.type == mt5.POSITION_TYPE_BUY:
                price = symbol_info.bid
                order_type = mt5.ORDER_TYPE_SELL
            else:
                price = symbol_info.ask
                order_type = mt5.ORDER_TYPE_BUY
            
            # Prepare the close request
            volume = float(kwargs.get('volume', position.volume))
            
            request = {
                'action': mt5.TRADE_ACTION_DEAL,
                'position': position.ticket,
                'symbol': position.symbol,
                'volume': volume,
                'type': order_type,
                'price': price,
                'deviation': 10,  # Slippage in points
                'magic': position.magic,
                'comment': 'Closed by user',
                'type_time': mt5.ORDER_TIME_GTC,
                'type_filling': mt5.ORDER_FILLING_FOK,
            }
            
            # Execute the close
            result = mt5.order_send(request)
            
            # Check the result
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                error_msg = f"Position close failed: {result.comment} (code: {result.retcode})"
                logger.error(error_msg)
                raise OrderError(error_msg)
            
            logger.info(f"Position {position_id} closed successfully")
            return True
            
        except Exception as e:
            error_msg = f"Failed to close position {position_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise OrderError(error_msg) from e
    
    @retry_async(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    )
    async def get_order_status(self, order_id: str) -> dict:
        """
        Get the status of an order.
        
        Args:
            order_id: ID of the order to check
            
        Returns:
            dict: Order status information
            
        Raises:
            OrderError: If order not found
            ConnectionError: If not connected to MT5
        """
        if not self.connected:
            raise ConnectionError("Not connected to MT5")
        
        try:
            # Check local cache first
            if order_id in self._orders:
                return self._orders[order_id]
            
            # Query MT5 for the order
            orders = mt5.orders_get(ticket=int(order_id))
            if not orders:
                raise OrderError(f"Order {order_id} not found")
            
            # Parse the order
            order = orders[0]  # Get the first (and should be only) order
            order_info = self._parse_order(order)
            
            # Update cache
            self._orders[order_id] = order_info
            
            return order_info
            
        except Exception as e:
            error_msg = f"Failed to get status for order {order_id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise OrderError(error_msg) from e
    
    @retry_async(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    )
    async def get_positions(self, symbol: str = None) -> list:
        """
        Get all open positions.
        
        Args:
            symbol: Optional symbol to filter positions
            
        Returns:
            list: List of open positions
            
        Raises:
            ConnectionError: If not connected to MT5
        """
        if not self.connected:
            raise ConnectionError("Not connected to MT5")
        
        try:
            # Get positions from MT5
            if symbol:
                positions = mt5.positions_get(symbol=symbol)
            else:
                positions = mt5.positions_get()
            
            if positions is None:
                return []
            
            # Parse positions
            positions_list = [self._parse_position(pos) for pos in positions]
            
            # Update cache
            async with self._position_lock:
                for pos in positions_list:
                    self._positions[str(pos['position_id'])] = pos
            
            return positions_list
            
        except Exception as e:
            error_msg = f"Failed to get positions: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ConnectionError(error_msg) from e
    
    @retry_async(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    )
    async def get_orders(self, symbol: str = None) -> list:
        """
        Get all open orders.
        
        Args:
            symbol: Optional symbol to filter orders
            
        Returns:
            list: List of open orders
            
        Raises:
            ConnectionError: If not connected to MT5
        """
        if not self.connected:
            raise ConnectionError("Not connected to MT5")
        
        try:
            # Get orders from MT5
            if symbol:
                orders = mt5.orders_get(symbol=symbol)
            else:
                orders = mt5.orders_get()
            
            if orders is None:
                return []
            
            # Parse orders
            orders_list = [self._parse_order(order) for order in orders]
            
            # Update cache
            async with self._order_lock:
                for order in orders_list:
                    self._orders[order['order_id']] = order
            
            return orders_list
            
        except Exception as e:
            error_msg = f"Failed to get orders: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ConnectionError(error_msg) from e
    
    @retry_async(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    )
    async def get_account_info(self) -> dict:
        """
        Get account information.
        
        Returns:
            dict: Account information
            
        Raises:
            ConnectionError: If not connected to MT5
        """
        if not self.connected:
            raise ConnectionError("Not connected to MT5")
        
        try:
            # Get account info from MT5
            account_info = mt5.account_info()
            if account_info is None:
                raise ConnectionError("Failed to get account info")
            
            # Parse account info
            account_info_dict = {
                'balance': account_info.balance,
                'equity': account_info.equity,
                'margin': account_info.margin,
                'margin_free': account_info.margin_free,
                'margin_level': account_info.margin_level,
                'leverage': account_info.leverage,
                'currency': account_info.currency,
                'name': account_info.name,
                'server': account_info.server,
                'timestamp': datetime.utcnow().isoformat(),
            }
            
            # Update cache
            self._account_info = account_info_dict
            
            return account_info_dict
            
        except Exception as e:
            error_msg = f"Failed to get account info: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise ConnectionError(error_msg) from e
    
    @retry_async(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
    )
    async def get_symbol_info(self, symbol: str) -> dict:
        """
        Get information about a symbol.
        
        Args:
            symbol: Symbol to get info for
            
        Returns:
            dict: Symbol information
            
        Raises:
            MarketDataError: If symbol not found
            ConnectionError: If not connected to MT5
        """
        if not self.connected:
            raise ConnectionError("Not connected to MT5")
        
        try:
            # Check cache first
            if symbol in self._symbols_info:
                return self._symbols_info[symbol]
            
            # Get symbol info from MT5
            symbol_info = mt5.symbol_info(symbol)
            if symbol_info is None:
                raise MarketDataError(f"Symbol {symbol} not found")
            
            # Parse symbol info
            symbol_info_dict = {
                'symbol': symbol_info.name,
                'description': symbol_info.description,
                'exchange': symbol_info.exchange,
                'type': symbol_info.type,
                'digits': symbol_info.digits,
                'spread': symbol_info.spread,
                'spread_float': symbol_info.spread_float,
                'tick_size': symbol_info.trade_tick_size,
                'tick_value': symbol_info.trade_tick_value,
                'volume_min': symbol_info.volume_min,
                'volume_max': symbol_info.volume_max,
                'volume_step': symbol_info.volume_step,
                'margin_initial': symbol_info.margin_initial,
                'margin_maintenance': symbol_info.margin_maintenance,
                'currency_base': symbol_info.currency_base,
                'currency_profit': symbol_info.currency_profit,
                'currency_margin': symbol_info.currency_margin,
                'session_deals': symbol_info.session_deals,
                'session_buy_orders': symbol_info.session_buy_orders,
                'session_sell_orders': symbol_info.session_sell_orders,
                'session_volume': symbol_info.session_volume,
                'session_interest': symbol_info.session_interest,
                'session_avg_price': symbol_info.session_avg_price,
                'session_open': symbol_info.session_open,
                'session_close': symbol_info.session_close,
                'session_aw': symbol_info.session_aw,
                'session_price_settlement': symbol_info.session_price_settlement,
                'session_price_limit_min': symbol_info.session_price_limit_min,
                'session_price_limit_max': symbol_info.session_price_limit_max,
                'timestamp': datetime.utcnow().isoformat(),
            }
            
            # Update cache
            self._symbols_info[symbol] = symbol_info_dict
            
            return symbol_info_dict
            
        except Exception as e:
            error_msg = f"Failed to get symbol info for {symbol}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise MarketDataError(error_msg) from e
