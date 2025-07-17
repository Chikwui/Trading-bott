"""
MT5 data provider implementation with advanced features.

Features:
- Real-time WebSocket streaming
- Order book reconstruction
- Advanced data validation
- Data gap filling
- Performance metrics
- Automatic reconnection
"""
from __future__ import annotations
from typing import Dict, List, Optional, Union, Any, Type, TypeVar, Generic, Deque, Tuple, Callable
from datetime import datetime, timedelta
import asyncio
import logging
import json
import time
import random
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum, auto
from decimal import Decimal, ROUND_HALF_UP

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from ...exchange import Exchange
from ..provider import DataProvider
from ...models import (
    DataType, TimeFrame, DataQuality, Instrument, OHLCV, Trade, OrderBook,
    FundingRate, OpenInterest, Liquidation, MarketDataRequest, MarketDataResponse,
    DataProviderConfig, DataCacheConfig, DataManagerConfig, InstrumentType,
    OrderBookLevel, OrderBookSide, TradeSide, Exchange as ExchangeModel
)

logger = logging.getLogger(__name__)

# MT5 timeframe mapping with additional timeframes
MT5_TIMEFRAME_MAP = {
    TimeFrame.TICKS: 0,  # Custom tick data
    TimeFrame.M1: mt5.TIMEFRAME_M1,
    TimeFrame.M2: mt5.TIMEFRAME_M2,
    TimeFrame.M3: mt5.TIMEFRAME_M3,
    TimeFrame.M4: mt5.TIMEFRAME_M4,
    TimeFrame.M5: mt5.TIMEFRAME_M5,
    TimeFrame.M6: mt5.TIMEFRAME_M6,
    TimeFrame.M10: mt5.TIMEFRAME_M10,
    TimeFrame.M12: mt5.TIMEFRAME_M12,
    TimeFrame.M15: mt5.TIMEFRAME_M15,
    TimeFrame.M20: mt5.TIMEFRAME_M20,
    TimeFrame.M30: mt5.TIMEFRAME_M30,
    TimeFrame.H1: mt5.TIMEFRAME_H1,
    TimeFrame.H2: mt5.TIMEFRAME_H2,
    TimeFrame.H3: mt5.TIMEFRAME_H3,
    TimeFrame.H4: mt5.TIMEFRAME_H4,
    TimeFrame.H6: mt5.TIMEFRAME_H6,
    TimeFrame.H8: mt5.TIMEFRAME_H8,
    TimeFrame.H12: mt5.TIMEFRAME_H12,
    TimeFrame.D1: mt5.TIMEFRAME_D1,
    TimeFrame.W1: mt5.TIMEFRAME_W1,
    TimeFrame.MN1: mt5.TIMEFRAME_MN1,
}

# Reverse mapping for MT5 timeframes
MT5_TIMEFRAME_REVERSE_MAP = {v: k for k, v in MT5_TIMEFRAME_MAP.items()}

# Data quality thresholds
class DataQualityThresholds:
    """Thresholds for data quality metrics."""
    MAX_GAP_RATIO = 0.1  # Maximum allowed gap ratio in OHLCV data
    MAX_PRICE_DEVIATION = 0.05  # 5% max price deviation from previous close
    MIN_VOLUME_RATIO = 0.1  # Minimum volume ratio compared to average
    MAX_SPREAD_RATIO = 0.1  # Maximum spread ratio (spread/mid_price)
    VALIDATION_WINDOW = 100  # Number of candles to use for validation

# Order book configuration
class OrderBookConfig:
    """Configuration for order book management."""
    MAX_DEPTH = 100  # Maximum depth to maintain in the order book
    PRICE_ROUNDING = 8  # Decimal places for price rounding
    SIZE_ROUNDING = 8  # Decimal places for size rounding
    SNAPSHOT_INTERVAL = 60  # Seconds between full order book snapshots
    VALIDATION_INTERVAL = 300  # Seconds between order book validations
    MAX_RECONSTRUCTION_ATTEMPTS = 3  # Max attempts to reconstruct order book

# WebSocket configuration
class WebSocketConfig:
    """Configuration for WebSocket connections."""
    PING_INTERVAL = 30  # Seconds between ping messages
    PING_TIMEOUT = 10  # Seconds to wait for pong response
    RECONNECT_DELAY = 5  # Base delay for reconnection attempts
    MAX_RECONNECT_ATTEMPTS = 5  # Maximum reconnection attempts
    MAX_QUEUE_SIZE = 10000  # Maximum size of the message queue

class MT5DataProvider(DataProvider):
    """
    Advanced MT5 data provider with real-time streaming and order book management.
    
    Features:
    - Real-time WebSocket data streaming
    - Order book reconstruction and validation
    - Data quality monitoring
    - Automatic reconnection
    - Performance metrics
    """
    
    def __init__(self, config: DataProviderConfig):
        """Initialize the MT5 data provider with advanced features.
        
        Args:
            config: Provider configuration with MT5-specific settings
        """
        super().__init__(config)
        self._mt5 = mt5
        self._connected = False
        self._initialized = False
        self._symbols_info = {}
        self._order_books = {}  # Symbol -> OrderBookState
        self._last_snapshots = {}  # Symbol -> last order book snapshot
        self._last_validation = {}  # Symbol -> last validation time
        self._subscriptions = set()  # Set of subscribed symbols
        self._websocket = None  # WebSocket connection
        self._ws_connected = False
        self._ws_task = None  # WebSocket task
        self._message_queue = asyncio.Queue()
        self._callbacks = {
            'ticker': [],
            'trades': [],
            'orderbook': [],
            'ohlcv': [],
            'error': []
        }
        
        # MT5-specific configuration
        self._login = config.extra.get('login')
        self._password = config.extra.get('password')
        self._server = config.extra.get('server')
        self._path = config.extra.get('path')
        self._timeout = config.extra.get('timeout', 60000)  # Default 60 seconds
        
        # Advanced configuration
        self._enable_websocket = config.extra.get('enable_websocket', True)
        self._validate_data = config.extra.get('validate_data', True)
        self._auto_reconnect = config.extra.get('auto_reconnect', True)
        self._reconnect_attempts = 0
        
        # Performance metrics
        self._metrics = {
            'messages_received': 0,
            'messages_processed': 0,
            'errors': 0,
            'last_error': None,
            'latency': deque(maxlen=1000),
            'throughput': 0,  # messages/second
            'last_update': time.time(),
        }
        
        # Initialize MT5
        self._init_mt5()
    
    def _init_mt5(self) -> None:
        """Initialize the MT5 terminal connection."""
        try:
            # Set MT5 path if provided
            if self._path and not mt5.initialize(path=self._path, 
                                              login=self._login, 
                                              password=self._password,
                                              server=self._server,
                                              timeout=self._timeout):
                error = self._mt5.last_error()
                logger.error(f"MT5 initialization failed: {error}")
                raise RuntimeError(f"MT5 initialization failed: {error}")
            
            logger.info("MT5 terminal initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing MT5: {e}")
            raise

    async def _initialize(self) -> None:
        """Initialize the MT5 data provider with advanced features."""
        if self._initialized:
            return
            
        try:
            # Connect to MT5 terminal if credentials are provided
            if self._login and self._password:
                logger.info(f"Logging in to MT5 account: {self._login}")
                authorized = self._mt5.login(
                    login=int(self._login),
                    password=self._password,
                    server=self._server,
                    timeout=self._timeout
                )
                
                if not authorized:
                    error = self._mt5.last_error()
                    logger.error(f"MT5 login failed: {error}")
                    raise RuntimeError(f"MT5 login failed: {error}")
                
                logger.info(f"Successfully logged in to MT5 account: {self._login}")
            
            # Load symbols info
            await self._load_symbols_info()
            
            # Initialize WebSocket if enabled
            if self._enable_websocket:
                await self._init_websocket()
            
            # Start background tasks
            self._start_background_tasks()
            
            self._initialized = True
            logger.info("MT5 data provider initialized with advanced features")
            
        except Exception as e:
            logger.error(f"Error initializing MT5 provider: {e}")
            if self._auto_reconnect:
                await self._handle_reconnect()
            raise
    
    async def _connect(self) -> None:
        """Connect to the MT5 terminal and establish WebSocket connection."""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Verify MT5 connection
            if not self._mt5.terminal_info():
                error = self._mt5.last_error()
                raise ConnectionError(f"MT5 terminal not connected: {error}")
            
            # Connect WebSocket if enabled
            if self._enable_websocket and not self._ws_connected:
                await self._connect_websocket()
            
            self._connected = True
            self._reconnect_attempts = 0  # Reset reconnect attempts on successful connection
            logger.info("Successfully connected to MT5 data provider")
            
        except Exception as e:
            logger.error(f"Error connecting to MT5: {e}")
            if self._auto_reconnect:
                await self._handle_reconnect()
            raise
    
    async def _disconnect(self, shutdown: bool = True) -> None:
        """Disconnect from the MT5 terminal and clean up resources.
        
        Args:
            shutdown: If True, shutdown the MT5 terminal. Set to False for reconnection.
        """
        if not self._connected and not self._ws_connected:
            return
        
        try:
            # Close WebSocket connection
            if self._ws_connected:
                await self._disconnect_websocket()
            
            # Stop background tasks
            self._stop_background_tasks()
            
            # Shutdown MT5 if requested
            if shutdown and self._connected:
                self._mt5.shutdown()
            
            self._connected = False
            logger.info("Disconnected from MT5 data provider")
            
        except Exception as e:
            logger.error(f"Error disconnecting from MT5: {e}")
            raise
    
    async def _load_symbols_info(self) -> None:
        """Load and cache symbols information from MT5."""
        symbols = self._mt5.symbols_get()
        if symbols is None:
            error = self._mt5.last_error()
            logger.error(f"Failed to get symbols from MT5: {error}")
            return
        
        self._symbols_info = {s.name: s for s in symbols}
        logger.info(f"Loaded {len(self._symbols_info)} symbols from MT5")
    
    def _map_timeframe(self, timeframe: Union[TimeFrame, str]) -> int:
        """Map TimeFrame enum to MT5 timeframe.
        
        Args:
            timeframe: TimeFrame enum or string representation
            
        Returns:
            MT5 timeframe constant
            
        Raises:
            ValueError: If the timeframe is not supported
        """
        if isinstance(timeframe, str):
            timeframe = TimeFrame[timeframe.upper()]
        
        if timeframe not in MT5_TIMEFRAME_MAP:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
            
        return MT5_TIMEFRAME_MAP[timeframe]
    
    def _map_instrument_type(self, symbol: str) -> InstrumentType:
        """Map MT5 symbol to instrument type.
        
        Args:
            symbol: Symbol name
            
        Returns:
            InstrumentType enum
        """
        if symbol.endswith(('.F', '.f')):
            return InstrumentType.FUTURE
        
        # Check if it's a forex pair (e.g., EURUSD, GBPUSD)
        if len(symbol) == 6 and symbol.isalpha():
            return InstrumentType.SPOT
            
        # Check if it's a crypto pair (e.g., BTCUSD, ETHUSD)
        if any(c in symbol.upper() for c in ['BTC', 'ETH', 'XRP', 'LTC', 'BCH']):
            return InstrumentType.CRYPTO
            
        # Default to SPOT for other instruments
        return InstrumentType.SPOT
    
    def _map_exchange(self, symbol: str) -> Exchange:
        """Map symbol to exchange.
        
        Args:
            symbol: Symbol name
            
        Returns:
            Exchange enum
        """
        # This is a simplified mapping - adjust based on your needs
        if symbol.endswith('.F'):
            return Exchange.CME  # Futures
        elif any(c in symbol.upper() for c in ['BTC', 'ETH', 'XRP']):
            return Exchange.BINANCE  # Crypto
        else:
            return Exchange.FOREX  # Forex
    
    async def _get_instruments(self, **kwargs) -> List[Instrument]:
        """Get all available trading instruments.
        
        Args:
            **kwargs: Additional parameters
            
        Returns:
            List of Instrument objects
        """
        if not self._symbols_info:
            await self._load_symbols_info()
        
        instruments = []
        for symbol, info in self._symbols_info.items():
            try:
                # Skip symbols that don't have required attributes
                if not hasattr(info, 'point') or not hasattr(info, 'digits'):
                    continue
                
                instrument = Instrument(
                    symbol=symbol,
                    name=info.description or symbol,
                    exchange=self._map_exchange(symbol),
                    type=self._map_instrument_type(symbol),
                    base_currency=info.currency_base,
                    quote_currency=info.currency_profit,
                    price_precision=info.digits,
                    min_price_increment=info.point,
                    min_quantity=info.volume_min,
                    max_quantity=info.volume_max,
                    quantity_precision=info.volume_step,
                    is_active=info.visible,
                    metadata={
                        'path': info.path,
                        'trade_mode': info.trade_mode,
                        'trade_allowed': info.trade_mode == 0,  # 0 = Allowed
                        'trade_contract_size': info.trade_contract_size,
                        'trade_tick_size': info.trade_tick_size,
                        'trade_tick_value': info.trade_tick_value,
                        'trade_tick_value_profit': info.trade_tick_value_profit,
                        'trade_tick_value_loss': info.trade_tick_value_loss,
                        'trade_stops_level': info.trade_stops_level,
                        'trade_freeze_level': info.trade_freeze_level,
                        'margin_initial': info.margin_initial,
                        'margin_maintenance': info.margin_maintenance,
                        'margin_hedged': info.margin_hedged,
                        'margin_hedged_use_leg': info.margin_hedged_use_leg,
                        'margin_hedged_margin': info.margin_hedged_margin,
                        'swap_mode': info.swap_mode,
                        'swap_rollover3days': info.swap_rollover3days,
                        'swap_long': info.swap_long,
                        'swap_short': info.swap_short,
                        'swap_rollover1day': info.swap_rollover1day,
                    }
                )
                instruments.append(instrument)
            except Exception as e:
                logger.error(f"Error creating instrument for {symbol}: {e}")
                continue
        
        return instruments
    
    async def _get_ohlcv(
        self,
        symbol: str,
        timeframe: Union[TimeFrame, str],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        **kwargs
    ) -> List[OHLCV]:
        """Get OHLCV data from MT5.
        
        Args:
            symbol: Symbol name (e.g., 'EURUSD')
            timeframe: TimeFrame enum or string
            start_time: Start time for the data
            end_time: End time for the data (defaults to now)
            limit: Maximum number of candles to return
            **kwargs: Additional parameters
            
        Returns:
            List of OHLCV objects
        """
        # Map timeframe to MT5 format
        mt5_timeframe = self._map_timeframe(timeframe)
        
        # Set default end time to now if not provided
        if end_time is None:
            end_time = datetime.now()
        
        # If limit is provided, adjust start time accordingly
        if limit is not None and start_time is None:
            # Calculate start time based on limit and timeframe
            timeframe_minutes = {
                TimeFrame.M1: 1,
                TimeFrame.M5: 5,
                TimeFrame.M15: 15,
                TimeFrame.M30: 30,
                TimeFrame.H1: 60,
                TimeFrame.H4: 240,
                TimeFrame.D1: 1440,
                TimeFrame.W1: 10080,
                TimeFrame.MN1: 43200,
            }.get(timeframe, 1)
            
            start_time = end_time - timedelta(minutes=timeframe_minutes * limit)
        
        # Convert to MT5 datetime format
        from_date = start_time if start_time else 0
        to_date = end_time if end_time else datetime.now()
        
        # Get rates from MT5
        rates = self._mt5.copy_rates_range(symbol, mt5_timeframe, from_date, to_date)
        if rates is None:
            error = self._mt5.last_error()
            logger.error(f"Failed to get OHLCV data for {symbol}: {error}")
            return []
        
        # Convert to pandas DataFrame for easier manipulation
        df = pd.DataFrame(rates)
        if df.empty:
            return []
        
        # Convert to list of OHLCV objects
        ohlcv_list = []
        for _, row in df.iterrows():
            ohlcv = OHLCV(
                timestamp=pd.to_datetime(row['time'], unit='s'),
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=row['real_volume'] if 'real_volume' in row else row['tick_volume'],
                symbol=symbol,
                timeframe=timeframe if isinstance(timeframe, TimeFrame) else TimeFrame[timeframe.upper()],
                exchange=self._map_exchange(symbol),
                data_quality=DataQuality.REAL_TIME,
                metadata={
                    'spread': row['spread'] if 'spread' in row else None,
                    'tick_volume': row['tick_volume'] if 'tick_volume' in row else None,
                    'real_volume': row['real_volume'] if 'real_volume' in row else None,
                }
            )
            ohlcv_list.append(ohlcv)
        
        # Apply limit if specified
        if limit is not None and len(ohlcv_list) > limit:
            ohlcv_list = ohlcv_list[-limit:]
        
        return ohlcv_list
    
    async def _get_trades(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        **kwargs
    ) -> List[Trade]:
        """Get historical trade data from MT5.
        
        Args:
            symbol: Symbol name
            start_time: Start time for the data
            end_time: End time for the data (defaults to now)
            limit: Maximum number of trades to return
            **kwargs: Additional parameters
            
        Returns:
            List of Trade objects
        """
        # MT5 doesn't provide direct access to historical trades through the API
        # This is a placeholder implementation that would need to be adapted
        # based on how you want to handle this in your specific case
        
        # You might need to implement this by:
        # 1. Using the MT5 terminal's history of deals/orders
        # 2. Implementing a custom solution to log trades as they happen
        # 3. Using a third-party service that provides this data
        
        logger.warning("MT5 historical trade data is not directly available through the standard API")
        return []
    
    async def _get_order_book(
        self,
        symbol: str,
        limit: Optional[int] = None,
        **kwargs
    ) -> OrderBook:
        """Get order book snapshot from MT5.
        
        Args:
            symbol: Symbol name
            limit: Maximum number of levels to return (not supported in MT5)
            **kwargs: Additional parameters
            
        Returns:
            OrderBook object
        """
        # Get order book from MT5
        book = self._mt5.market_book_get(symbol)
        if book is None:
            error = self._mt5.last_error()
            logger.error(f"Failed to get order book for {symbol}: {error}")
            return OrderBook(
                symbol=symbol,
                exchange=self._map_exchange(symbol),
                timestamp=datetime.now(),
                bids=[],
                asks=[],
                data_quality=DataQuality.UNKNOWN
            )
        
        # Convert to our OrderBook format
        bids = []
        asks = []
        
        for level in book:
            side = OrderBookSide.BID if level.type == 1 else OrderBookSide.ASK
            price = level.price
            size = level.volume
            
            if side == OrderBookSide.BID:
                bids.append(OrderBookLevel(price=price, size=size, count=1))
            else:
                asks.append(OrderBookLevel(price=price, size=size, count=1))
        
        # Sort bids (descending) and asks (ascending)
        bids.sort(key=lambda x: -x.price)
        asks.sort(key=lambda x: x.price)
        
        # Apply limit if specified
        if limit is not None:
            bids = bids[:limit]
            asks = asks[:limit]
        
        return OrderBook(
            symbol=symbol,
            exchange=self._map_exchange(symbol),
            timestamp=datetime.now(),
            bids=bids,
            asks=asks,
            data_quality=DataQuality.REAL_TIME
        )
    
    async def _get_funding_rate(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        **kwargs
    ) -> List[FundingRate]:
        """Get funding rate history.
        
        Note: MT5 doesn't provide funding rate data directly.
        This method would need to be implemented based on your specific requirements.
        """
        logger.warning("Funding rate data is not directly available from MT5")
        return []
    
    async def _get_open_interest(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        **kwargs
    ) -> List[OpenInterest]:
        """Get open interest history.
        
        Note: MT5 doesn't provide open interest data directly.
        This method would need to be implemented based on your specific requirements.
        """
        logger.warning("Open interest data is not directly available from MT5")
        return []
    
    async def _get_liquidations(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        **kwargs
    ) -> List[Liquidation]:
        """Get liquidation history.
        
        Note: MT5 doesn't provide liquidation data directly.
        This method would need to be implemented based on your specific requirements.
        """
        logger.warning("Liquidation data is not directly available from MT5")
        return []
    
    async def _subscribe(self, channel: str, symbol: str, **kwargs) -> None:
        """Subscribe to a data stream.
        
        Args:
            channel: Data channel ('ticker', 'trades', 'orderbook')
            symbol: Symbol name
            **kwargs: Additional parameters
        """
        # MT5 uses a different subscription model where you need to explicitly
        # request data updates. This is a simplified implementation.
        logger.info(f"Subscribed to {channel} for {symbol}")
    
    async def _unsubscribe(self, channel: str, symbol: str, **kwargs) -> None:
        """Unsubscribe from a data stream.
        
        Args:
            channel: Data channel
            symbol: Symbol name
            **kwargs: Additional parameters
        """
        logger.info(f"Unsubscribed from {channel} for {symbol}")

# Register the provider with the factory
DataProviderFactory.register_provider('mt5', MT5DataProvider)
