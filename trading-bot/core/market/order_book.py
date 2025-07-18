"""
Advanced Order Book implementation with real-time updates and validation.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Dict, List, Optional, Set, Callable, Deque, Tuple
from collections import defaultdict, deque
import time
import json
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class OrderBookSide(Enum):
    BID = "bid"
    ASK = "ask"

@dataclass(order=True)
class PriceLevel:
    """Represents a price level in the order book."""
    price: Decimal
    volume: Decimal
    orders: int = 1
    
    def update(self, volume: Decimal, is_new_order: bool = True) -> None:
        """Update the price level with new volume."""
        self.volume += volume
        if is_new_order:
            self.orders += 1

@dataclass
class OrderBookUpdate:
    """Represents an update to the order book."""
    timestamp: float
    symbol: str
    side: OrderBookSide
    price: Decimal
    volume: Decimal
    is_snapshot: bool = False
    
    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp,
            'symbol': self.symbol,
            'side': self.side.value,
            'price': float(self.price),
            'volume': float(self.volume),
            'is_snapshot': self.is_snapshot
        }

class OrderBook:
    """High-performance order book implementation with validation and normalization."""
    
    def __init__(
        self,
        symbol: str,
        price_precision: int = 5,
        size_precision: int = 8,
        max_depth: int = 1000,
        validate: bool = True
    ):
        self.symbol = symbol
        self.price_precision = price_precision
        self.size_precision = size_precision
        self.max_depth = max_depth
        self.validate = validate
        
        # Order book state
        self.bids: Dict[Decimal, PriceLevel] = {}
        self.asks: Dict[Decimal, PriceLevel] = {}
        self.last_update: float = time.time()
        self.sequence = 0
        self._subscribers: Set[Callable[[OrderBookUpdate], None]] = set()
        self._update_queue: asyncio.Queue = asyncio.Queue(maxsize=10000)
        self._update_task: Optional[asyncio.Task] = None
        
    def normalize_price(self, price: float | Decimal | str) -> Decimal:
        """Normalize price to the required precision."""
        if not isinstance(price, Decimal):
            price = Decimal(str(price))
        return price.quantize(
            Decimal('1e-{}'.format(self.price_precision)),
            rounding=ROUND_HALF_UP
        )
    
    def normalize_volume(self, volume: float | Decimal | str) -> Decimal:
        """Normalize volume to the required precision."""
        if not isinstance(volume, Decimal):
            volume = Decimal(str(volume))
        return volume.quantize(
            Decimal('1e-{}'.format(self.size_precision)),
            rounding=ROUND_HALF_UP
        )
        
    def validate_update(self, side: OrderBookSide, price: Decimal, volume: Decimal) -> bool:
        """Validate an order book update."""
        if volume < 0:
            logger.warning(f"Invalid volume: {volume}")
            return False
            
        if price <= 0:
            logger.warning(f"Invalid price: {price}")
            return False
            
        # Check for crossed book
        if self.bids and self.asks:
            best_bid = max(self.bids.keys()) if self.bids else Decimal('0')
            best_ask = min(self.asks.keys()) if self.asks else Decimal('inf')
            
            if side == OrderBookSide.BID and price >= best_ask:
                logger.warning(f"Crossed book - bid {price} >= ask {best_ask}")
                return False
                
            if side == OrderBookSide.ASK and price <= best_bid:
                logger.warning(f"Crossed book - ask {price} <= bid {best_bid}")
                return False
                
        return True
    
    async def start(self) -> None:
        """Start the order book update processor."""
        if self._update_task is None or self._update_task.done():
            self._update_task = asyncio.create_task(self._process_updates())
    
    async def stop(self) -> None:
        """Stop the order book update processor."""
        if self._update_task and not self._update_task.done():
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
    
    async def _process_updates(self) -> None:
        """Process order book updates from the queue."""
        while True:
            try:
                update = await self._update_queue.get()
                self.apply_update(update)
                self._update_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error processing update: {e}", exc_info=True)
    
    def queue_update(self, update: OrderBookUpdate) -> None:
        """Queue an order book update for processing."""
        try:
            self._update_queue.put_nowait(update)
        except asyncio.QueueFull:
            logger.warning("Order book update queue full, dropping update")
    
    def apply_update(self, update: OrderBookUpdate) -> None:
        """Apply an update to the order book."""
        try:
            price = self.normalize_price(update.price)
            volume = self.normalize_volume(update.volume)
            
            if self.validate and not self.validate_update(update.side, price, volume):
                return
            
            book_side = self.bids if update.side == OrderBookSide.BID else self.asks
            
            if volume == 0:
                # Remove price level
                book_side.pop(price, None)
            else:
                # Update or add price level
                if price in book_side:
                    book_side[price].update(volume, update.is_snapshot)
                else:
                    book_side[price] = PriceLevel(price, volume)
            
            self.last_update = time.time()
            self.sequence += 1
            
            # Notify subscribers
            self._notify_subscribers(update)
            
        except Exception as e:
            logger.error(f"Failed to apply update: {e}", exc_info=True)
    
    def get_snapshot(self, depth: int = 100) -> dict:
        """Get a snapshot of the order book."""
        bids = sorted(
            [(float(price), float(level.volume), level.orders) 
             for price, level in self.bids.items()],
            reverse=True
        )
        asks = sorted(
            [(float(price), float(level.volume), level.orders) 
             for price, level in self.asks.items()]
        )
        
        return {
            'symbol': self.symbol,
            'timestamp': self.last_update,
            'sequence': self.sequence,
            'bids': bids[:depth],
            'asks': asks[:depth],
            'is_snapshot': True
        }
    
    def subscribe(self, callback: Callable[[OrderBookUpdate], None]) -> None:
        """Subscribe to order book updates."""
        self._subscribers.add(callback)
    
    def unsubscribe(self, callback: Callable[[OrderBookUpdate], None]) -> None:
        """Unsubscribe from order book updates."""
        self._subscribers.discard(callback)
    
    def _notify_subscribers(self, update: OrderBookUpdate) -> None:
        """Notify all subscribers of an update."""
        for callback in list(self._subscribers):
            try:
                callback(update)
            except Exception as e:
                logger.error(f"Error in order book subscriber: {e}", exc_info=True)
                self.unsubscribe(callback)

class OrderBookManager:
    """Manages multiple order books with WebSocket support."""
    
    def __init__(self):
        self.order_books: Dict[str, OrderBook] = {}
        self._websocket_clients: Set[WebSocket] = set()
    
    def get_order_book(self, symbol: str, **kwargs) -> OrderBook:
        """Get or create an order book for a symbol."""
        if symbol not in self.order_books:
            self.order_books[symbol] = OrderBook(symbol, **kwargs)
        return self.order_books[symbol]
    
    async def start_all(self) -> None:
        """Start all order books."""
        for book in self.order_books.values():
            await book.start()
    
    async def stop_all(self) -> None:
        """Stop all order books."""
        for book in self.order_books.values():
            await book.stop()
    
    def subscribe_websocket(self, websocket: WebSocket) -> None:
        """Subscribe a WebSocket client to order book updates."""
        self._websocket_clients.add(websocket)
    
    def unsubscribe_websocket(self, websocket: WebSocket) -> None:
        """Unsubscribe a WebSocket client."""
        self._websocket_clients.discard(websocket)
    
    async def broadcast_update(self, update: OrderBookUpdate) -> None:
        """Broadcast an update to all WebSocket clients."""
        if not self._websocket_clients:
            return
            
        message = json.dumps(update.to_dict())
        for client in list(self._websocket_clients):
            try:
                await client.send_text(message)
            except Exception as e:
                logger.error(f"Error sending WebSocket update: {e}")
                self.unsubscribe_websocket(client)

class WebSocket:
    """WebSocket client interface for order book updates."""
    async def send_text(self, data: str) -> None:
        """Send text data over the WebSocket."""
        raise NotImplementedError("WebSocket.send_text must be implemented by subclasses")
