"""
Market data service for handling real-time and historical market data.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
import pandas as pd
from core.market import MarketDataHandler
from core.instruments import InstrumentMetadata

logger = logging.getLogger(__name__)

class MarketDataService:
    """Service for managing market data streams and subscriptions."""
    
    def __init__(self, data_handler: MarketDataHandler):
        """Initialize the market data service.
        
        Args:
            data_handler: Market data handler instance
        """
        self.data_handler = data_handler
        self.subscriptions: Dict[str, List[Callable]] = {}
        self._is_running = False
        self._tasks: Dict[str, asyncio.Task] = {}
    
    async def start(self) -> None:
        """Start the market data service."""
        self._is_running = True
        logger.info("Market Data Service started")
    
    async def stop(self) -> None:
        """Stop the market data service."""
        self._is_running = False
        for task in self._tasks.values():
            task.cancel()
        await asyncio.gather(*self._tasks.values(), return_exceptions=True)
        logger.info("Market Data Service stopped")
    
    def subscribe(
        self,
        symbol: str,
        callback: Callable[[Dict[str, Any]], None],
        interval: str = '1m'
    ) -> None:
        """Subscribe to market data updates for a symbol.
        
        Args:
            symbol: Instrument symbol to subscribe to
            callback: Callback function to receive updates
            interval: Data interval (e.g., '1m', '5m', '1h')
        """
        if symbol not in self.subscriptions:
            self.subscriptions[symbol] = []
            self._start_data_stream(symbol, interval)
        
        if callback not in self.subscriptions[symbol]:
            self.subscriptions[symbol].append(callback)
    
    def unsubscribe(self, symbol: str, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Unsubscribe from market data updates.
        
        Args:
            symbol: Instrument symbol
            callback: Callback function to remove
        """
        if symbol in self.subscriptions and callback in self.subscriptions[symbol]:
            self.subscriptions[symbol].remove(callback)
            
            if not self.subscriptions[symbol]:
                self._stop_data_stream(symbol)
    
    def _start_data_stream(self, symbol: str, interval: str) -> None:
        """Start a data stream for the given symbol.
        
        Args:
            symbol: Instrument symbol
            interval: Data interval
        """
        if symbol in self._tasks:
            return
            
        async def _stream_data():
            while self._is_running:
                try:
                    # Get latest market data
                    data = await self.data_handler.get_latest_bar(symbol, interval)
                    
                    # Notify subscribers
                    if symbol in self.subscriptions and data is not None:
                        for callback in self.subscriptions[symbol]:
                            try:
                                callback(data)
                            except Exception as e:
                                logger.error(f"Error in market data callback: {e}")
                    
                    # Sleep based on interval
                    await asyncio.sleep(self._get_interval_seconds(interval))
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in market data stream for {symbol}: {e}")
                    await asyncio.sleep(5)  # Backoff on error
        
        self._tasks[symbol] = asyncio.create_task(_stream_data())
    
    def _stop_data_stream(self, symbol: str) -> None:
        """Stop the data stream for a symbol.
        
        Args:
            symbol: Instrument symbol
        """
        if symbol in self._tasks:
            self._tasks[symbol].cancel()
            del self._tasks[symbol]
            del self.subscriptions[symbol]
    
    async def get_historical_data(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = '1m'
    ) -> pd.DataFrame:
        """Get historical market data.
        
        Args:
            symbol: Instrument symbol
            start: Start datetime
            end: End datetime
            interval: Data interval
            
        Returns:
            DataFrame with historical data
        """
        return await self.data_handler.get_historical_bars(
            symbol=symbol,
            start=start,
            end=end,
            interval=interval
        )
    
    @staticmethod
    def _get_interval_seconds(interval: str) -> int:
        """Convert interval string to seconds."""
        unit = interval[-1].lower()
        value = int(interval[:-1])
        
        if unit == 's':
            return value
        elif unit == 'm':
            return value * 60
        elif unit == 'h':
            return value * 3600
        elif unit == 'd':
            return value * 86400
        else:
            return 60  # Default to 1 minute
