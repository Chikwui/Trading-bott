"""
Advanced Market Data Service for real-time data streaming and order book management.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Set, Tuple, Callable, Awaitable, Union,
    DefaultDict
)
from collections import defaultdict
import json

import numpy as np
import pandas as pd

from .order_book import OrderBook, OrderBookUpdate, OrderBookManager, OrderBookSide
from .websocket_service import WebSocketClient, WebSocketServer, WebSocketMessage
from .data_validation import DataNormalizer, DataValidator, DataQualityChecker, DataSchemaType
from ..instruments import Instrument, InstrumentType

logger = logging.getLogger(__name__)

class MarketDataService:
    """
    Unified market data service that handles real-time data streaming,
    order book management, and data validation.
    """
    
    def __init__(
        self,
        instruments: List[Instrument],
        websocket_host: str = '0.0.0.0',
        websocket_port: int = 8765,
        max_order_book_depth: int = 1000,
        validate_data: bool = True,
        enable_websocket: bool = True,
        **kwargs
    ):
        """
        Initialize the market data service.
        
        Args:
            instruments: List of instruments to track
            websocket_host: Host for the WebSocket server
            websocket_port: Port for the WebSocket server
            max_order_book_depth: Maximum depth for order books
            validate_data: Whether to validate incoming data
            enable_websocket: Whether to enable WebSocket server
        """
        self.instruments = {i.symbol: i for i in instruments}
        self.max_order_book_depth = max_order_book_depth
        self.validate_data = validate_data
        self.enable_websocket = enable_websocket
        
        # Core components
        self.order_book_manager = OrderBookManager()
        self.data_normalizer = DataNormalizer()
        self.data_validator = DataValidator()
        self.quality_checker = DataQualityChecker()
        
        # WebSocket server
        self.websocket_server: Optional[WebSocketServer] = None
        self.websocket_host = websocket_host
        self.websocket_port = websocket_port
        
        # Data storage
        self.market_data: Dict[str, Dict] = {}
        self.order_books: Dict[str, OrderBook] = {}
        self.last_updates: Dict[str, float] = {}
        
        # Callbacks
        self._callbacks: DefaultDict[str, List[Callable]] = defaultdict(list)
        
        # Initialize order books for each instrument
        self._init_order_books()
    
    async def start(self) -> None:
        """Start the market data service."""
        logger.info("Starting Market Data Service...")
        
        # Start WebSocket server if enabled
        if self.enable_websocket:
            self.websocket_server = WebSocketServer(
                host=self.websocket_host,
                port=self.websocket_port
            )
            await self.websocket_server.start()
            logger.info(f"WebSocket server started on ws://{self.websocket_host}:{self.websocket_port}")
        
        # Start order books
        await self.order_book_manager.start_all()
        logger.info(f"Started {len(self.order_books)} order books")
        
        logger.info("Market Data Service started")
    
    async def stop(self) -> None:
        """Stop the market data service."""
        logger.info("Stopping Market Data Service...")
        
        # Stop WebSocket server
        if self.websocket_server:
            await self.websocket_server.stop()
        
        # Stop order books
        await self.order_book_manager.stop_all()
        
        logger.info("Market Data Service stopped")
    
    def _init_order_books(self) -> None:
        """Initialize order books for all instruments."""
        for symbol, instrument in self.instruments.items():
            self.order_books[symbol] = self.order_book_manager.get_order_book(
                symbol=symbol,
                price_precision=instrument.price_precision,
                size_precision=instrument.volume_precision,
                max_depth=self.max_order_book_depth,
                validate=self.validate_data
            )
    
    async def process_market_data(
        self,
        data: Dict[str, Any],
        data_type: DataSchemaType,
        source: Optional[str] = None
    ) -> bool:
        """
        Process incoming market data.
        
        Args:
            data: Raw market data
            data_type: Type of market data
            source: Data source (e.g., 'mt5', 'binance')
            
        Returns:
            bool: True if processing was successful
        """
        try:
            # Normalize data
            normalized = await self.data_normalizer.normalize(
                data=data,
                schema_type=data_type,
                source=source
            )
            
            # Validate data
            if self.validate_data:
                is_valid, errors = await self.data_validator.validate(
                    data=normalized,
                    schema_type=data_type
                )
                if not is_valid:
                    logger.error(f"Data validation failed: {errors}")
                    return False
            
            # Update market data store
            symbol = normalized.get('symbol')
            if not symbol:
                logger.error("No symbol in market data")
                return False
            
            self.market_data[symbol] = normalized
            self.last_updates[symbol] = time.time()
            
            # Process based on data type
            if data_type == DataSchemaType.ORDER_BOOK:
                await self._process_order_book_update(normalized)
            elif data_type == DataSchemaType.TRADE:
                await self._process_trade(normalized)
            elif data_type == DataSchemaType.OHLCV:
                await self._process_ohlcv(normalized)
            
            # Broadcast update via WebSocket
            if self.websocket_server:
                await self._broadcast_update(data_type, normalized)
            
            # Trigger callbacks
            await self._trigger_callbacks(data_type, normalized)
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing market data: {e}", exc_info=True)
            return False
    
    async def _process_order_book_update(self, data: Dict[str, Any]) -> None:
        """Process an order book update."""
        symbol = data['symbol']
        if symbol not in self.order_books:
            logger.warning(f"No order book for symbol: {symbol}")
            return
        
        order_book = self.order_books[symbol]
        
        # Process bids
        for level in data.get('bids', []):
            update = OrderBookUpdate(
                timestamp=time.time(),
                symbol=symbol,
                side=OrderBookSide.BID,
                price=Decimal(str(level['price'])),
                volume=Decimal(str(level['volume'])),
                is_snapshot='is_snapshot' in data and data['is_snapshot']
            )
            order_book.queue_update(update)
        
        # Process asks
        for level in data.get('asks', []):
            update = OrderBookUpdate(
                timestamp=time.time(),
                symbol=symbol,
                side=OrderBookSide.ASK,
                price=Decimal(str(level['price'])),
                volume=Decimal(str(level['volume'])),
                is_snapshot='is_snapshot' in data and data['is_snapshot']
            )
            order_book.queue_update(update)
    
    async def _process_trade(self, data: Dict[str, Any]) -> None:
        """Process a trade update."""
        # Update last trade price
        symbol = data['symbol']
        if symbol not in self.market_data:
            self.market_data[symbol] = {}
        
        self.market_data[symbol].update({
            'last_trade_price': data['price'],
            'last_trade_volume': data['volume'],
            'last_trade_side': data['side'],
            'last_trade_time': data['timestamp']
        })
    
    async def _process_ohlcv(self, data: Dict[str, Any]) -> None:
        """Process an OHLCV update."""
        symbol = data['symbol']
        if symbol not in self.market_data:
            self.market_data[symbol] = {}
        
        self.market_data[symbol].update({
            'open': data['open'],
            'high': data['high'],
            'low': data['low'],
            'close': data['close'],
            'volume': data['volume'],
            'timeframe': data['timeframe'],
            'last_ohlcv_update': data['timestamp']
        })
    
    async def _broadcast_update(
        self, 
        data_type: DataSchemaType, 
        data: Dict[str, Any]
    ) -> None:
        """Broadcast an update to WebSocket clients."""
        if not self.websocket_server:
            return
        
        message = {
            'type': data_type.value,
            'data': data,
            'timestamp': time.time()
        }
        
        try:
            await self.websocket_server.broadcast(message)
        except Exception as e:
            logger.error(f"Error broadcasting WebSocket message: {e}")
    
    async def _trigger_callbacks(
        self, 
        data_type: DataSchemaType, 
        data: Dict[str, Any]
    ) -> None:
        """Trigger registered callbacks for the given data type."""
        callbacks = self._callbacks.get(data_type.value, [])
        
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(data)
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"Error in market data callback: {e}")
    
    def subscribe(
        self, 
        data_type: Union[DataSchemaType, str], 
        callback: Callable[[Dict[str, Any]], Awaitable[None] | None]
    ) -> None:
        """
        Subscribe to market data updates.
        
        Args:
            data_type: Type of data to subscribe to
            callback: Callback function to receive updates
        """
        if isinstance(data_type, DataSchemaType):
            data_type = data_type.value
        
        self._callbacks[data_type].append(callback)
    
    def unsubscribe(
        self, 
        data_type: Union[DataSchemaType, str], 
        callback: Callable[[Dict[str, Any]], Awaitable[None] | None]
    ) -> None:
        """
        Unsubscribe from market data updates.
        
        Args:
            data_type: Type of data to unsubscribe from
            callback: Callback function to remove
        """
        if isinstance(data_type, DataSchemaType):
            data_type = data_type.value
        
        if callback in self._callbacks[data_type]:
            self._callbacks[data_type].remove(callback)
    
    async def get_order_book(
        self, 
        symbol: str, 
        depth: int = 10
    ) -> Optional[Dict[str, Any]]:
        """
        Get the current order book for a symbol.
        
        Args:
            symbol: Symbol to get order book for
            depth: Maximum depth to return
            
        Returns:
            Order book data or None if not available
        """
        if symbol not in self.order_books:
            return None
            
        return self.order_books[symbol].get_snapshot(depth)
    
    async def get_market_data(
        self, 
        symbol: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get the latest market data for a symbol.
        
        Args:
            symbol: Symbol to get data for
            
        Returns:
            Market data or None if not available
        """
        return self.market_data.get(symbol)
    
    async def get_last_update_time(
        self, 
        symbol: str
    ) -> Optional[float]:
        """
        Get the timestamp of the last update for a symbol.
        
        Args:
            symbol: Symbol to check
            
        Returns:
            Timestamp of last update or None if never updated
        """
        return self.last_updates.get(symbol)
    
    async def get_websocket_url(self) -> Optional[str]:
        """
        Get the WebSocket server URL.
        
        Returns:
            WebSocket URL or None if not enabled
        """
        if not self.enable_websocket or not self.websocket_server:
            return None
            
        return f"ws://{self.websocket_host}:{self.websocket_port}"

# Example usage
async def example_usage():
    from datetime import datetime, timezone
    from decimal import Decimal
    
    # Create some test instruments
    class TestInstrument:
        def __init__(self, symbol, price_precision=8, volume_precision=8):
            self.symbol = symbol
            self.price_precision = price_precision
            self.volume_precision = volume_precision
    
    instruments = [
        TestInstrument("BTCUSDT"),
        TestInstrument("ETHUSDT")
    ]
    
    # Create and start the service
    service = MarketDataService(instruments)
    await service.start()
    
    try:
        # Example order book update
        order_book_update = {
            'symbol': 'BTCUSDT',
            'bids': [
                {'price': '50000.00', 'volume': '1.5', 'orders': 10},
                {'price': '49999.50', 'volume': '2.0', 'orders': 5},
            ],
            'asks': [
                {'price': '50001.00', 'volume': '0.75', 'orders': 8},
                {'price': '50001.50', 'volume': '1.25', 'orders': 3},
            ],
            'timestamp': datetime.now(timezone.utc).timestamp(),
            'exchange': 'test',
            'is_snapshot': True
        }
        
        # Process the update
        await service.process_market_data(
            data=order_book_update,
            data_type=DataSchemaType.ORDER_BOOK,
            source='test'
        )
        
        # Get the order book
        order_book = await service.get_order_book('BTCUSDT')
        print("Order Book:", order_book)
        
        # Keep running to process WebSocket messages
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        pass
    finally:
        await service.stop()

if __name__ == "__main__":
    import asyncio
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the example
    asyncio.run(example_usage())
