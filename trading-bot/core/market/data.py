"""
Market data handling with calendar-aware data streaming.
"""
from __future__ import annotations

import logging
import queue
import threading
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Callable, Set

import pandas as pd

from core.instruments import InstrumentMetadata
from core.calendar import MarketCalendar

logger = logging.getLogger(__name__)


class MarketDataHandler:
    """Handles market data streaming and storage with calendar awareness."""
    
    def __init__(
        self,
        instruments: List[InstrumentMetadata],
        calendar: Optional[MarketCalendar] = None,
        timezone: str = "UTC",
        max_queue_size: int = 10000
    ):
        """Initialize the market data handler.
        
        Args:
            instruments: List of instruments to track
            calendar: Market calendar
            timezone: Default timezone
            max_queue_size: Maximum size of the data queue
        """
        self.instruments = {instr.symbol: instr for instr in instruments}
        self.calendar = calendar
        self.timezone = timezone
        self.max_queue_size = max_queue_size
        
        # Data storage
        self.ticks: Dict[str, queue.Queue] = {}
        self.bars: Dict[str, Dict[str, queue.Queue]] = {}
        self.latest_ticks: Dict[str, Dict[str, Any]] = {}
        self.historical_data: Dict[str, pd.DataFrame] = {}
        
        # Threading
        self._running = False
        self._threads: List[threading.Thread] = []
        
        # Initialize data structures
        for symbol in self.instruments:
            self.ticks[symbol] = queue.Queue(maxsize=max_queue_size)
            self.bars[symbol] = {}
            self.latest_ticks[symbol] = {}
    
    def subscribe(self, symbol: str, timeframe: str = "1m") -> bool:
        """Subscribe to market data for a symbol and timeframe.
        
        Args:
            symbol: Instrument symbol
            timeframe: Bar timeframe (e.g., '1m', '1h', '1d')
            
        Returns:
            bool: True if subscription was successful
        """
        if symbol not in self.instruments:
            logger.warning(f"Cannot subscribe to unknown symbol: {symbol}")
            return False
            
        if timeframe not in self.bars[symbol]:
            self.bars[symbol][timeframe] = queue.Queue(maxsize=self.max_queue_size)
            logger.info(f"Subscribed to {symbol} {timeframe} data")
            return True
            
        return False
    
    def unsubscribe(self, symbol: str, timeframe: Optional[str] = None) -> None:
        """Unsubscribe from market data.
        
        Args:
            symbol: Instrument symbol
            timeframe: Optional specific timeframe to unsubscribe from
        """
        if symbol in self.bars:
            if timeframe:
                if timeframe in self.bars[symbol]:
                    del self.bars[symbol][timeframe]
                    logger.info(f"Unsubscribed from {symbol} {timeframe} data")
            else:
                self.bars[symbol].clear()
                logger.info(f"Unsubscribed from all {symbol} data")
    
    def on_tick(self, symbol: str, tick: Dict[str, Any]) -> None:
        """Process a new tick.
        
        Args:
            symbol: Instrument symbol
            tick: Tick data with 'bid', 'ask', 'last', 'volume', 'timestamp'
        """
        if symbol not in self.instruments:
            logger.warning(f"Received tick for unknown symbol: {symbol}")
            return
            
        # Add timestamp if not present
        if 'timestamp' not in tick:
            tick['timestamp'] = datetime.now(timezone.utc)
            
        # Update latest tick
        self.latest_ticks[symbol] = tick
        
        # Add to tick queue
        try:
            self.ticks[symbol].put_nowait(tick)
        except queue.Full:
            logger.warning(f"Tick queue full for {symbol}, dropping tick")
    
    def on_bar(self, symbol: str, timeframe: str, bar: Dict[str, Any]) -> None:
        """Process a new bar.
        
        Args:
            symbol: Instrument symbol
            timeframe: Bar timeframe (e.g., '1m', '1h')
            bar: Bar data with 'open', 'high', 'low', 'close', 'volume', 'timestamp'
        """
        if symbol not in self.instruments:
            logger.warning(f"Received bar for unknown symbol: {symbol}")
            return
            
        if timeframe not in self.bars[symbol]:
            logger.warning(f"Received bar for unsubscribed timeframe: {symbol} {timeframe}")
            return
            
        # Add timestamp if not present
        if 'timestamp' not in bar:
            bar['timestamp'] = datetime.now(timezone.utc)
            
        # Add to appropriate bar queue
        try:
            self.bars[symbol][timeframe].put_nowait(bar)
        except queue.Full:
            logger.warning(f"Bar queue full for {symbol} {timeframe}, dropping bar")
    
    def get_latest_tick(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the latest tick for a symbol.
        
        Args:
            symbol: Instrument symbol
            
        Returns:
            Latest tick data or None if not available
        """
        return self.latest_ticks.get(symbol)
    
    def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 1000
    ) -> Optional[pd.DataFrame]:
        """Get historical price data.
        
        Args:
            symbol: Instrument symbol
            timeframe: Bar timeframe (e.g., '1m', '1h')
            start: Start datetime
            end: End datetime
            limit: Maximum number of bars to return
            
        Returns:
            DataFrame with historical data or None if not available
        """
        # In a real implementation, this would fetch historical data from a database
        # For now, return an empty DataFrame with the expected columns
        return pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    def start(self) -> None:
        """Start the market data handler."""
        if self._running:
            return
            
        self._running = True
        
        # Start data collection threads
        # In a real implementation, this would connect to a data feed
        logger.info("Market data handler started")
    
    def stop(self) -> None:
        """Stop the market data handler."""
        self._running = False
        
        # Stop all data collection threads
        for thread in self._threads:
            if thread.is_alive():
                thread.join(timeout=1.0)
                
        logger.info("Market data handler stopped")
