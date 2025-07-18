"""
High-performance WebSocket service for real-time market data streaming.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Set, Callable, Awaitable, Union
from dataclasses import dataclass, field
from decimal import Decimal
import ssl
import uuid
import zlib

import aiohttp
import ujson
import websockets
from websockets import WebSocketServerProtocol
from websockets.exceptions import ConnectionClosed

from .order_book import OrderBook, OrderBookUpdate, OrderBookSide

logger = logging.getLogger(__name__)

class WebSocketMessage:
    """Standardized WebSocket message format."""
    
    def __init__(
        self,
        msg_type: str,
        data: Any,
        timestamp: Optional[float] = None,
        request_id: Optional[str] = None
    ):
        self.msg_type = msg_type
        self.data = data
        self.timestamp = timestamp or time.time()
        self.request_id = request_id or str(uuid.uuid4())
    
    def to_dict(self) -> dict:
        return {
            'type': self.msg_type,
            'data': self.data,
            'timestamp': self.timestamp,
            'request_id': self.request_id
        }
    
    def to_json(self) -> str:
        return ujson.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> WebSocketMessage:
        data = ujson.loads(json_str)
        return cls(
            msg_type=data['type'],
            data=data['data'],
            timestamp=data.get('timestamp'),
            request_id=data.get('request_id')
        )

class WebSocketClient:
    """WebSocket client for connecting to market data feeds."""
    
    def __init__(
        self,
        url: str,
        api_key: Optional[str] = None,
        subscriptions: Optional[List[Dict]] = None,
        reconnect_interval: int = 5,
        max_reconnect_attempts: int = 10,
        ping_interval: int = 30,
        ping_timeout: int = 10,
        compression: bool = True
    ):
        self.url = url
        self.api_key = api_key
        self.subscriptions = subscriptions or []
        self.reconnect_interval = reconnect_interval
        self.max_reconnect_attempts = max_reconnect_attempts
        self.ping_interval = ping_interval
        self.ping_timeout = ping_timeout
        self.compression = compression
        
        self._ws: Optional[WebSocketServerProtocol] = None
        self._reconnect_attempts = 0
        self._running = False
        self._message_handlers: Dict[str, Callable[[dict], Awaitable[None]]] = {}
        self._subscription_callbacks: Dict[str, Callable[[dict], None]] = {}
        self._connect_task: Optional[asyncio.Task] = None
        self._ping_task: Optional[asyncio.Task] = None
    
    async def connect(self) -> None:
        """Connect to the WebSocket server."""
        if self._connect_task and not self._connect_task.done():
            logger.warning("Connection already in progress")
            return
            
        self._connect_task = asyncio.create_task(self._connect_loop())
    
    async def _connect_loop(self) -> None:
        """Handle connection and reconnection logic."""
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        extra_headers = {}
        if self.api_key:
            extra_headers['Authorization'] = f"Bearer {self.api_key}"
        
        while self._reconnect_attempts < self.max_reconnect_attempts:
            try:
                async with websockets.connect(
                    self.url,
                    ssl=ssl_context,
                    extra_headers=extra_headers,
                    compression='deflate' if self.compression else None,
                    ping_interval=self.ping_interval,
                    ping_timeout=self.ping_timeout
                ) as websocket:
                    self._ws = websocket
                    self._reconnect_attempts = 0
                    self._running = True
                    
                    # Start ping task
                    self._ping_task = asyncio.create_task(self._ping_loop())
                    
                    # Resubscribe to channels
                    await self._resubscribe()
                    
                    # Start message processing
                    await self._process_messages()
                    
            except (ConnectionError, OSError, asyncio.TimeoutError) as e:
                logger.error(f"WebSocket connection error: {e}")
                self._reconnect_attempts += 1
                await self._handle_reconnect()
            except Exception as e:
                logger.error(f"Unexpected error in WebSocket connection: {e}")
                self._reconnect_attempts += 1
                await self._handle_reconnect()
            finally:
                await self._cleanup()
    
    async def _process_messages(self) -> None:
        """Process incoming WebSocket messages."""
        if not self._ws:
            return
            
        try:
            async for message in self._ws:
                try:
                    if isinstance(message, bytes):
                        message = zlib.decompress(message).decode('utf-8')
                    
                    data = ujson.loads(message)
                    msg_type = data.get('type')
                    
                    # Handle message type
                    if msg_type in self._message_handlers:
                        await self._message_handlers[msg_type](data)
                    
                    # Handle subscription callbacks
                    if 'channel' in data and data['channel'] in self._subscription_callbacks:
                        self._subscription_callbacks[data['channel']](data)
                        
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    
        except ConnectionClosed as e:
            logger.warning(f"WebSocket connection closed: {e}")
            self._running = False
        except Exception as e:
            logger.error(f"Error in message processing: {e}")
            self._running = False
    
    async def _ping_loop(self) -> None:
        """Send periodic pings to keep the connection alive."""
        while self._running and self._ws:
            try:
                await asyncio.wait_for(
                    self._ws.ping(),
                    timeout=self.ping_timeout
                )
                await asyncio.sleep(self.ping_interval)
            except (asyncio.TimeoutError, ConnectionError):
                logger.warning("Ping timeout, reconnecting...")
                self._running = False
                break
            except Exception as e:
                logger.error(f"Error in ping loop: {e}")
                self._running = False
                break
    
    async def _resubscribe(self) -> None:
        """Resubscribe to all channels after reconnection."""
        if not self.subscriptions or not self._ws:
            return
            
        for subscription in self.subscriptions:
            try:
                await self._ws.send(ujson.dumps(subscription))
            except Exception as e:
                logger.error(f"Failed to resubscribe: {e}")
    
    async def _handle_reconnect(self) -> None:
        """Handle reconnection logic."""
        if self._reconnect_attempts >= self.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            self._running = False
            return
            
        logger.info(f"Reconnecting in {self.reconnect_interval} seconds...")
        await asyncio.sleep(self.reconnect_interval)
    
    async def _cleanup(self) -> None:
        """Clean up resources."""
        if self._ping_task and not self._ping_task.done():
            self._ping_task.cancel()
            try:
                await self._ping_task
            except asyncio.CancelledError:
                pass
        
        self._ws = None
        self._ping_task = None
    
    def on_message(self, msg_type: str) -> Callable:
        """Decorator to register a message handler."""
        def decorator(func: Callable[[dict], Awaitable[None]]) -> Callable:
            self._message_handlers[msg_type] = func
            return func
        return decorator
    
    def on_subscription(self, channel: str) -> Callable:
        """Decorator to register a subscription callback."""
        def decorator(func: Callable[[dict], None]) -> Callable:
            self._subscription_callbacks[channel] = func
            return func
        return decorator
    
    async def send(self, data: Union[dict, str]) -> None:
        """Send data through the WebSocket connection."""
        if not self._ws:
            raise ConnectionError("WebSocket is not connected")
            
        if isinstance(data, dict):
            data = ujson.dumps(data)
            
        await self._ws.send(data)
    
    async def subscribe(self, channel: str, **kwargs) -> None:
        """Subscribe to a channel."""
        subscription = {
            'event': 'subscribe',
            'channel': channel,
            **kwargs
        }
        await self.send(subscription)
    
    async def unsubscribe(self, channel: str) -> None:
        """Unsubscribe from a channel."""
        await self.send({
            'event': 'unsubscribe',
            'channel': channel
        })
    
    async def close(self) -> None:
        """Close the WebSocket connection."""
        self._running = False
        if self._ws:
            await self._ws.close()
        if self._connect_task and not self._connect_task.done():
            self._connect_task.cancel()
            try:
                await self._connect_task
            except asyncio.CancelledError:
                pass

class WebSocketServer:
    """WebSocket server for broadcasting market data to clients."""
    
    def __init__(self, host: str = '0.0.0.0', port: int = 8765):
        self.host = host
        self.port = port
        self.clients: Set[WebSocketServerProtocol] = set()
        self.server: Optional[asyncio.Server] = None
        self._broadcast_queue: asyncio.Queue = asyncio.Queue()
        self._broadcast_task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the WebSocket server."""
        self.server = await websockets.serve(
            self._handle_client,
            self.host,
            self.port,
            compression='deflate',
            ping_interval=30,
            ping_timeout=10
        )
        
        # Start broadcast task
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())
        
        logger.info(f"WebSocket server started on ws://{self.host}:{self.port}")
    
    async def stop(self) -> None:
        """Stop the WebSocket server."""
        if self._broadcast_task and not self._broadcast_task.done():
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass
        
        if self.server:
            self.server.close()
            await self.server.wait_closed()
    
    async def _handle_client(self, websocket: WebSocketServerProtocol, path: str) -> None:
        """Handle a new WebSocket client connection."""
        self.clients.add(websocket)
        client_ip = websocket.remote_address[0] if websocket.remote_address else 'unknown'
        logger.info(f"New WebSocket connection from {client_ip}")
        
        try:
            # Keep the connection open
            async for _ in websocket:
                pass
                
        except ConnectionClosed:
            logger.info(f"Client {client_ip} disconnected")
        except Exception as e:
            logger.error(f"Error with client {client_ip}: {e}")
        finally:
            self.clients.discard(websocket)
    
    async def broadcast(self, message: Union[dict, str]) -> None:
        """Broadcast a message to all connected clients."""
        await self._broadcast_queue.put(message)
    
    async def _broadcast_loop(self) -> None:
        """Process broadcast messages in the background."""
        while True:
            try:
                message = await self._broadcast_queue.get()
                
                if not self.clients:
                    continue
                    
                if isinstance(message, dict):
                    message = ujson.dumps(message)
                
                # Send to all connected clients
                tasks = [
                    client.send(message)
                    for client in self.clients
                    if client.open
                ]
                
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in broadcast loop: {e}")
                await asyncio.sleep(1)  # Prevent tight loop on errors
