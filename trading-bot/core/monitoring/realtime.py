"""
Real-time Data Updates

This module handles real-time data streaming to the dashboard using Server-Sent Events (SSE).
"""
import asyncio
import json
import logging
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Callable, Awaitable

from fastapi import Request
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)

class RealtimeManager:
    """Manages real-time data subscriptions and broadcasting."""
    
    def __init__(self):
        """Initialize the realtime manager."""
        self.subscriptions: Dict[str, Set[asyncio.Queue]] = defaultdict(set)
        self.last_updates: Dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._running = True
        self._broadcast_task = asyncio.create_task(self._broadcast_loop())
    
    async def subscribe(self, channels: List[str]) -> asyncio.Queue:
        """Subscribe to one or more data channels.
        
        Args:
            channels: List of channel names to subscribe to
            
        Returns:
            asyncio.Queue: Queue that will receive updates
        """
        queue = asyncio.Queue(maxsize=100)
        
        async with self._lock:
            for channel in channels:
                self.subscriptions[channel].add(queue)
                # Send the last known state if available
                if channel in self.last_updates:
                    await queue.put({
                        'channel': channel,
                        'data': self.last_updates[channel],
                        'timestamp': datetime.utcnow().isoformat()
                    })
        
        logger.debug(f"New subscription to channels: {channels}")
        return queue
    
    async def unsubscribe(self, queue: asyncio.Queue):
        """Unsubscribe a queue from all channels."""
        async with self._lock:
            for channel in list(self.subscriptions.keys()):
                self.subscriptions[channel].discard(queue)
    
    async def publish(self, channel: str, data: Any):
        """Publish data to a channel.
        
        Args:
            channel: Channel name
            data: Data to publish (must be JSON-serializable)
        """
        async with self._lock:
            self.last_updates[channel] = data
            
            if channel not in self.subscriptions:
                return
            
            message = {
                'channel': channel,
                'data': data,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Convert to JSON once
            message_json = json.dumps(message)
            
            # Send to all subscribers
            dead_queues = []
            for queue in self.subscriptions[channel]:
                try:
                    if queue.full():
                        # Drop the oldest message if queue is full
                        try:
                            queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    queue.put_nowait(message_json)
                except Exception as e:
                    logger.warning(f"Error sending to queue: {e}")
                    dead_queues.append(queue)
            
            # Clean up dead queues
            if dead_queues:
                self.subscriptions[channel] = {
                    q for q in self.subscriptions[channel]
                    if q not in dead_queues
                }
    
    async def _broadcast_loop(self):
        """Background task to handle broadcasting updates."""
        while self._running:
            try:
                # Periodically clean up empty channels
                async with self._lock:
                    empty_channels = [
                        channel for channel, queues in self.subscriptions.items()
                        if not queues
                    ]
                    for channel in empty_channels:
                        del self.subscriptions[channel]
                
                await asyncio.sleep(30)  # Cleanup interval
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in broadcast loop: {e}")
                await asyncio.sleep(5)  # Prevent tight loop on errors
    
    async def stream_response(self, request: Request, channels: List[str]):
        """Generate a Server-Sent Events (SSE) response.
        
        Args:
            request: FastAPI request object
            channels: List of channels to subscribe to
            
        Yields:
            str: SSE-formatted messages
        """
        queue = await self.subscribe(channels)
        
        try:
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    logger.debug("Client disconnected")
                    break
                
                try:
                    # Wait for a message with timeout
                    try:
                        message = await asyncio.wait_for(queue.get(), timeout=30.0)
                        yield f"data: {message}\n\n"
                    except asyncio.TimeoutError:
                        # Send a heartbeat to keep the connection alive
                        yield ":keepalive\n\n"
                        # Check if client is still connected
                        if await request.is_disconnected():
                            break
                        continue
                    
                    # Allow other tasks to run
                    await asyncio.sleep(0)
                    
                except asyncio.CancelledError:
                    logger.debug("Stream cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in stream: {e}")
                    yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
                    await asyncio.sleep(1)  # Prevent tight loop on errors
                    
        finally:
            await self.unsubscribe(queue)
            logger.debug("Client unsubscribed")
    
    async def close(self):
        """Clean up resources."""
        self._running = False
        if self._broadcast_task:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass
        
        # Clear all subscriptions
        async with self._lock:
            self.subscriptions.clear()
            self.last_updates.clear()

# Global instance
realtime_manager = RealtimeManager()

# Helper functions
async def publish_update(channel: str, data: Any):
    """Publish an update to a channel."""
    await realtime_manager.publish(channel, data)

async def get_realtime_stream(request: Request, channels: List[str]):
    """Get a real-time event stream for the specified channels."""
    return await realtime_manager.stream_response(request, channels)
