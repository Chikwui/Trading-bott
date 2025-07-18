"""
MT5 Data Fetcher

This module provides functionality to fetch and manage market data from MetaTrader 5.
"""
# Standard library imports
import os
import time
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

# Third-party imports
import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import pytz

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Timeframe(Enum):
    """MT5 timeframe constants."""
    M1 = mt5.TIMEFRAME_M1
    M2 = mt5.TIMEFRAME_M2
    M3 = mt5.TIMEFRAME_M3
    M4 = mt5.TIMEFRAME_M4
    M5 = mt5.TIMEFRAME_M5
    M6 = mt5.TIMEFRAME_M6
    M10 = mt5.TIMEFRAME_M10
    M12 = mt5.TIMEFRAME_M12
    M15 = mt5.TIMEFRAME_M15
    M20 = mt5.TIMEFRAME_M20
    M30 = mt5.TIMEFRAME_M30
    H1 = mt5.TIMEFRAME_H1
    H2 = mt5.TIMEFRAME_H2
    H3 = mt5.TIMEFRAME_H3
    H4 = mt5.TIMEFRAME_H4
    H6 = mt5.TIMEFRAME_H6
    H8 = mt5.TIMEFRAME_H8
    H12 = mt5.TIMEFRAME_H12
    D1 = mt5.TIMEFRAME_D1
    W1 = mt5.TIMEFRAME_W1
    MN1 = mt5.TIMEFRAME_MN1

@dataclass
class MT5Config:
    """Configuration for MT5 connection and data fetching."""
    server: str = ""
    login: int = 0
    password: str = ""
    path: str = "C:\\Program Files\\MetaTrader 5\\terminal64.exe"
    timeout: int = 60000  # Connection timeout in milliseconds
    portable: bool = False
    cache_dir: str = "data/mt5_cache"
    max_retries: int = 3
    retry_delay: int = 5  # seconds
    symbols: List[str] = field(default_factory=lambda: ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"])
    timeframes: List[Timeframe] = field(default_factory=lambda: [Timeframe.M5, Timeframe.M15, Timeframe.H1])
    update_interval: int = 60  # seconds
    max_bars: int = 10000  # Maximum number of bars to fetch
    utc_offset: int = 0  # UTC offset in hours

class MT5Fetcher:
    """Handles fetching and managing market data from MetaTrader 5."""
    
    def __init__(self, config: Optional[MT5Config] = None):
        """Initialize the MT5 fetcher with configuration."""
        self.config = config or MT5Config()
        self.connected = False
        self.last_update: Dict[str, Dict[Timeframe, datetime]] = {}
        self._initialize_cache()
        
    def _initialize_cache(self) -> None:
        """Initialize the cache directory."""
        os.makedirs(self.config.cache_dir, exist_ok=True)
    
    def connect(self) -> bool:
        """Connect to the MT5 terminal."""
        if mt5.initialize(
            path=self.config.path,
            login=self.config.login,
            password=self.config.password,
            server=self.config.server,
            timeout=self.config.timeout,
            portable=self.config.portable
        ):
            self.connected = True
            logger.info("Successfully connected to MT5 terminal")
            return True
        else:
            error = mt5.last_error()
            logger.error(f"Failed to connect to MT5: {error}")
            return False
    
    def disconnect(self) -> None:
        """Disconnect from the MT5 terminal."""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("Disconnected from MT5 terminal")
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
    
    def _to_mt5_time(self, dt: datetime) -> datetime:
        """Convert local datetime to MT5 timezone."""
        return dt - timedelta(hours=self.config.utc_offset)
    
    def _from_mt5_time(self, dt: datetime) -> datetime:
        """Convert MT5 timezone to local datetime."""
        return dt + timedelta(hours=self.config.utc_offset)
    
    def _get_cached_data_path(self, symbol: str, timeframe: Timeframe) -> str:
        """Get the path to the cached data file."""
        return os.path.join(self.config.cache_dir, f"{symbol}_{timeframe.name}.parquet")
    
    def _load_cached_data(self, symbol: str, timeframe: Timeframe) -> Optional[pd.DataFrame]:
        """Load cached data from disk."""
        cache_path = self._get_cached_data_path(symbol, timeframe)
        if os.path.exists(cache_path):
            try:
                df = pd.read_parquet(cache_path)
                if not df.empty:
                    df.index = pd.to_datetime(df.index)
                    logger.debug(f"Loaded {len(df)} cached bars for {symbol} {timeframe.name}")
                    return df
            except Exception as e:
                logger.warning(f"Failed to load cached data for {symbol} {timeframe.name}: {e}")
        return None
    
    def _save_to_cache(self, df: pd.DataFrame, symbol: str, timeframe: Timeframe) -> None:
        """Save data to cache."""
        if df is not None and not df.empty:
            try:
                cache_path = self._get_cached_data_path(symbol, timeframe)
                df.to_parquet(cache_path)
                logger.debug(f"Saved {len(df)} bars to cache for {symbol} {timeframe.name}")
            except Exception as e:
                logger.error(f"Failed to save cache for {symbol} {timeframe.name}: {e}")
    
    def _merge_data(self, old_data: Optional[pd.DataFrame], new_data: pd.DataFrame) -> pd.DataFrame:
        """Merge old and new data, removing duplicates."""
        if old_data is None or old_data.empty:
            return new_data
        
        # Combine and remove duplicates
        combined = pd.concat([old_data, new_data])
        combined = combined[~combined.index.duplicated(keep='last')]
        combined = combined.sort_index()
        
        return combined
    
    def _fetch_bars(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        n_bars: Optional[int] = None
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data from MT5."""
        if not self.connected and not self.connect():
            return None
        
        # Select the symbol
        if not mt5.symbol_select(symbol, True):
            logger.error(f"Failed to select symbol {symbol}")
            return None
        
        # Convert datetime to MT5 timezone
        if start_time:
            start_time = self._to_mt5_time(start_time)
        if end_time:
            end_time = self._to_mt5_time(end_time)
        
        # Fetch data with retries
        for attempt in range(self.config.max_retries):
            try:
                rates = mt5.copy_rates_range(
                    symbol,
                    timeframe.value,
                    start_time or 0,
                    end_time or mt5.TIMEFRAME_FRAME_MODE_TIME_CURRENT
                )
                
                if rates is None or len(rates) == 0:
                    logger.warning(f"No data returned for {symbol} {timeframe.name}, attempt {attempt + 1}")
                    time.sleep(self.config.retry_delay)
                    continue
                
                # Convert to DataFrame
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df = df.set_index('time')
                df.index = df.index.tz_localize('UTC')
                
                # Limit number of bars if specified
                if n_bars and len(df) > n_bars:
                    df = df.iloc[-n_bars:]
                
                # Convert index back to local time
                df.index = df.index.tz_convert(None)
                
                # Rename columns to lowercase
                df = df.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'tick_volume': 'Volume',
                    'spread': 'Spread',
                    'real_volume': 'RealVolume'
                })
                
                # Add symbol and timeframe as columns
                df['Symbol'] = symbol
                df['Timeframe'] = timeframe.name
                
                # Calculate typical price and add it as a feature
                df['TypicalPrice'] = (df['High'] + df['Low'] + df['Close']) / 3
                
                logger.info(f"Fetched {len(df)} bars for {symbol} {timeframe.name}")
                return df
                
            except Exception as e:
                logger.error(f"Error fetching data (attempt {attempt + 1}): {e}")
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    logger.error(f"Failed to fetch data after {self.config.max_retries} attempts")
        
        return None
    
    def get_historical_data(
        self,
        symbol: str,
        timeframe: Timeframe,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        n_bars: Optional[int] = None,
        use_cache: bool = True,
        update_cache: bool = True
    ) -> Optional[pd.DataFrame]:
        """Get historical OHLCV data with caching support."""
        # Load cached data if available
        cached_data = None
        if use_cache:
            cached_data = self._load_cached_data(symbol, timeframe)
        
        # Determine time range for fetching new data
        fetch_start = start_time
        if cached_data is not None and not cached_data.empty:
            last_cached_time = cached_data.index[-1].to_pydatetime()
            fetch_start = last_cached_time
            
            # If we only need recent data, limit the fetch
            if n_bars:
                fetch_start = None  # We'll fetch all data and then take the last n_bars
        
        # Fetch new data
        new_data = self._fetch_bars(symbol, timeframe, fetch_start, end_time, n_bars)
        
        if new_data is None or new_data.empty:
            logger.warning(f"No new data fetched for {symbol} {timeframe.name}")
            return cached_data
        
        # Merge with cached data
        if cached_data is not None and not cached_data.empty:
            combined_data = self._merge_data(cached_data, new_data)
        else:
            combined_data = new_data
        
        # Apply time range filter
        if start_time:
            combined_data = combined_data[combined_data.index >= start_time]
        if end_time:
            combined_data = combined_data[combined_data.index <= end_time]
        
        # Limit number of bars if specified
        if n_bars and len(combined_data) > n_bars:
            combined_data = combined_data.iloc[-n_bars:]
        
        # Update cache
        if update_cache and (not use_cache or len(new_data) > 0):
            self._save_to_cache(combined_data, symbol, timeframe)
        
        return combined_data
    
    def get_multiple_symbols(
        self,
        symbols: List[str],
        timeframes: List[Timeframe],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        n_bars: Optional[int] = None,
        use_cache: bool = True,
        update_cache: bool = True
    ) -> Dict[str, Dict[Timeframe, pd.DataFrame]]:
        """Get data for multiple symbols and timeframes."""
        result = {}
        
        for symbol in symbols:
            result[symbol] = {}
            for tf in timeframes:
                data = self.get_historical_data(
                    symbol=symbol,
                    timeframe=tf,
                    start_time=start_time,
                    end_time=end_time,
                    n_bars=n_bars,
                    use_cache=use_cache,
                    update_cache=update_cache
                )
                if data is not None and not data.empty:
                    result[symbol][tf] = data
                    self.last_update[symbol] = self.last_update.get(symbol, {})
                    self.last_update[symbol][tf] = datetime.now()
        
        return result
    
    def get_latest_bar(self, symbol: str, timeframe: Timeframe) -> Optional[pd.Series]:
        """Get the latest bar for a symbol and timeframe."""
        data = self.get_historical_data(symbol, timeframe, n_bars=1)
        if data is not None and not data.empty:
            return data.iloc[-1]
        return None
    
    def get_account_info(self) -> Optional[Dict[str, Any]]:
        """Get account information."""
        if not self.connected and not self.connect():
            return None
        
        account_info = mt5.account_info()._asdict()
        return account_info
    
    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get information about a symbol."""
        if not self.connected and not self.connect():
            return None
        
        info = mt5.symbol_info(symbol)
        if info is not None:
            return info._asdict()
        return None
    
    def get_available_symbols(self, group: str = "*") -> List[str]:
        """Get a list of available symbols."""
        if not self.connected and not self.connect():
            return []
        
        symbols = mt5.symbols_get(group)
        if symbols is None:
            return []
        
        return [s.name for s in symbols]
    
    def get_server_time(self) -> Optional[datetime]:
        """Get the current server time."""
        if not self.connected and not self.connect():
            return None
        
        time = mt5.symbol_info_tick("EURUSD").time if mt5.symbol_select("EURUSD") else None
        if time is not None:
            return self._from_mt5_time(time)
        return None


def download_historical_data(
    config: MT5Config,
    symbols: Optional[List[str]] = None,
    timeframes: Optional[List[Timeframe]] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    update_existing: bool = True
) -> Dict[str, Dict[Timeframe, pd.DataFrame]]:
    """Download historical data for the specified symbols and timeframes.
    
    Args:
        config: MT5 configuration
        symbols: List of symbols to download (default: from config)
        timeframes: List of timeframes to download (default: from config)
        start_date: Start date for historical data
        end_date: End date for historical data (default: now)
        update_existing: Whether to update existing cached data
        
    Returns:
        Dictionary of DataFrames with the downloaded data
    """
    symbols = symbols or config.symbols
    timeframes = timeframes or config.timeframes
    end_date = end_date or datetime.now()
    
    with MT5Fetcher(config) as fetcher:
        data = fetcher.get_multiple_symbols(
            symbols=symbols,
            timeframes=timeframes,
            start_time=start_date,
            end_time=end_date,
            use_cache=not update_existing,
            update_cache=True
        )
    
    return data


def stream_realtime_data(
    config: MT5Config,
    callback: callable,
    symbols: Optional[List[str]] = None,
    timeframes: Optional[List[Timeframe]] = None,
    interval: int = 60,
    max_iterations: Optional[int] = None
) -> None:
    """Stream real-time market data.
    
    Args:
        config: MT5 configuration
        callback: Function to call with new data
        symbols: List of symbols to stream (default: from config)
        timeframes: List of timeframes to stream (default: from config)
        interval: Update interval in seconds
        max_iterations: Maximum number of iterations (None for infinite)
    """
    symbols = symbols or config.symbols
    timeframes = timeframes or config.timeframes
    
    with MT5Fetcher(config) as fetcher:
        iteration = 0
        while max_iterations is None or iteration < max_iterations:
            start_time = time.time()
            iteration += 1
            
            try:
                # Get latest data
                data = fetcher.get_multiple_symbols(
                    symbols=symbols,
                    timeframes=timeframes,
                    n_bars=1,  # Only get the latest bar
                    use_cache=False,
                    update_cache=False
                )
                
                # Call the callback with the new data
                if data:
                    callback(data)
                
                # Calculate sleep time to maintain the desired interval
                elapsed = time.time() - start_time
                sleep_time = max(0, interval - elapsed)
                time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                logger.info("Streaming stopped by user")
                break
            except Exception as e:
                logger.error(f"Error in streaming loop: {e}")
                time.sleep(min(60, interval))  # Wait before retrying
