"""Data provider interface and implementations."""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any, Type, TypeVar, Generic
from datetime import datetime
import asyncio
import logging
from ..exchange.base import BaseExchange as Exchange
from .models import (
    DataType, TimeFrame, DataQuality, Instrument, OHLCV, Trade, OrderBook,
    FundingRate, OpenInterest, Liquidation, MarketDataRequest, MarketDataResponse,
    DataProviderConfig, DataCacheConfig, DataManagerConfig, InstrumentType
)

logger = logging.getLogger(__name__)
T = TypeVar('T')

class DataProvider(ABC):
    """Abstract base class for all data providers."""
    
    def __init__(self, config: DataProviderConfig):
        """Initialize the data provider with configuration.
        
        Args:
            config: Configuration for the data provider
        """
        self.config = config
        self._connected = False
        self._initialized = False
        self._subscriptions: Dict[str, Any] = {}
        
    @property
    def name(self) -> str:
        """Get the name of the data provider."""
        return self.config.provider_name
    
    @property
    def is_connected(self) -> bool:
        """Check if the provider is connected."""
        return self._connected
    
    @property
    def is_initialized(self) -> bool:
        """Check if the provider is initialized."""
        return self._initialized
    
    async def initialize(self) -> None:
        """Initialize the data provider."""
        if self._initialized:
            return
        
        try:
            await self._initialize()
            self._initialized = True
            logger.info(f"Initialized data provider: {self.name}")
        except Exception as e:
            logger.error(f"Failed to initialize data provider {self.name}: {e}")
            raise
    
    async def connect(self) -> None:
        """Connect to the data provider."""
        if self._connected:
            return
        
        if not self._initialized:
            await self.initialize()
        
        try:
            await self._connect()
            self._connected = True
            logger.info(f"Connected to data provider: {self.name}")
        except Exception as e:
            logger.error(f"Failed to connect to data provider {self.name}: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Disconnect from the data provider."""
        if not self._connected:
            return
        
        try:
            await self._disconnect()
            self._connected = False
            logger.info(f"Disconnected from data provider: {self.name}")
        except Exception as e:
            logger.error(f"Error disconnecting from data provider {self.name}: {e}")
            raise
    
    async def get_instruments(self, **kwargs) -> List[Instrument]:
        """Get all available trading instruments.
        
        Args:
            **kwargs: Additional parameters for the request
            
        Returns:
            List of available trading instruments
        """
        if not self._connected:
            await self.connect()
            
        try:
            return await self._get_instruments(**kwargs)
        except Exception as e:
            logger.error(f"Failed to get instruments from {self.name}: {e}")
            raise
    
    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: Union[TimeFrame, str],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        **kwargs
    ) -> List[OHLCV]:
        """Get OHLCV data.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe for the data
            start_time: Start time for the data
            end_time: End time for the data
            limit: Maximum number of data points to return
            **kwargs: Additional parameters for the request
            
        Returns:
            List of OHLCV data points
        """
        if not self._connected:
            await self.connect()
            
        try:
            return await self._get_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Failed to get OHLCV data from {self.name}: {e}")
            raise
    
    async def get_trades(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        **kwargs
    ) -> List[Trade]:
        """Get trade data.
        
        Args:
            symbol: Trading pair symbol
            start_time: Start time for the data
            end_time: End time for the data
            limit: Maximum number of trades to return
            **kwargs: Additional parameters for the request
            
        Returns:
            List of trade data points
        """
        if not self._connected:
            await self.connect()
            
        try:
            return await self._get_trades(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Failed to get trades from {self.name}: {e}")
            raise
    
    async def get_order_book(
        self,
        symbol: str,
        limit: Optional[int] = None,
        **kwargs
    ) -> OrderBook:
        """Get order book snapshot.
        
        Args:
            symbol: Trading pair symbol
            limit: Maximum number of order book levels to return
            **kwargs: Additional parameters for the request
            
        Returns:
            Order book snapshot
        """
        if not self._connected:
            await self.connect()
            
        try:
            return await self._get_order_book(
                symbol=symbol,
                limit=limit,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Failed to get order book from {self.name}: {e}")
            raise
    
    async def get_funding_rate(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        **kwargs
    ) -> List[FundingRate]:
        """Get funding rate history.
        
        Args:
            symbol: Trading pair symbol
            start_time: Start time for the data
            end_time: End time for the data
            limit: Maximum number of data points to return
            **kwargs: Additional parameters for the request
            
        Returns:
            List of funding rate data points
        """
        if not self._connected:
            await self.connect()
            
        try:
            return await self._get_funding_rate(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Failed to get funding rate from {self.name}: {e}")
            raise
    
    async def get_open_interest(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        **kwargs
    ) -> List[OpenInterest]:
        """Get open interest history.
        
        Args:
            symbol: Trading pair symbol
            start_time: Start time for the data
            end_time: End time for the data
            limit: Maximum number of data points to return
            **kwargs: Additional parameters for the request
            
        Returns:
            List of open interest data points
        """
        if not self._connected:
            await self.connect()
            
        try:
            return await self._get_open_interest(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Failed to get open interest from {self.name}: {e}")
            raise
    
    async def get_liquidations(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        **kwargs
    ) -> List[Liquidation]:
        """Get liquidation history.
        
        Args:
            symbol: Trading pair symbol
            start_time: Start time for the data
            end_time: End time for the data
            limit: Maximum number of liquidations to return
            **kwargs: Additional parameters for the request
            
        Returns:
            List of liquidation events
        """
        if not self._connected:
            await self.connect()
            
        try:
            return await self._get_liquidations(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Failed to get liquidations from {self.name}: {e}")
            raise
    
    async def subscribe(self, channel: str, symbol: str, **kwargs) -> None:
        """Subscribe to a data stream.
        
        Args:
            channel: Data channel to subscribe to (e.g., 'ticker', 'trades', 'orderbook')
            symbol: Trading pair symbol
            **kwargs: Additional parameters for the subscription
        """
        if not self._connected:
            await self.connect()
            
        try:
            await self._subscribe(channel=channel, symbol=symbol, **kwargs)
            logger.info(f"Subscribed to {channel} for {symbol} on {self.name}")
        except Exception as e:
            logger.error(f"Failed to subscribe to {channel} for {symbol} on {self.name}: {e}")
            raise
    
    async def unsubscribe(self, channel: str, symbol: str, **kwargs) -> None:
        """Unsubscribe from a data stream.
        
        Args:
            channel: Data channel to unsubscribe from
            symbol: Trading pair symbol
            **kwargs: Additional parameters for the unsubscription
        """
        if not self._connected:
            return
            
        try:
            await self._unsubscribe(channel=channel, symbol=symbol, **kwargs)
            logger.info(f"Unsubscribed from {channel} for {symbol} on {self.name}")
        except Exception as e:
            logger.error(f"Failed to unsubscribe from {channel} for {symbol} on {self.name}: {e}")
            raise
    
    # Abstract methods that must be implemented by subclasses
    
    @abstractmethod
    async def _initialize(self) -> None:
        """Initialize the data provider (implementation-specific)."""
        pass
    
    @abstractmethod
    async def _connect(self) -> None:
        """Connect to the data provider (implementation-specific)."""
        pass
    
    @abstractmethod
    async def _disconnect(self) -> None:
        """Disconnect from the data provider (implementation-specific)."""
        pass
    
    @abstractmethod
    async def _get_instruments(self, **kwargs) -> List[Instrument]:
        """Get all available trading instruments (implementation-specific)."""
        pass
    
    @abstractmethod
    async def _get_ohlcv(
        self,
        symbol: str,
        timeframe: Union[TimeFrame, str],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        **kwargs
    ) -> List[OHLCV]:
        """Get OHLCV data (implementation-specific)."""
        pass
    
    @abstractmethod
    async def _get_trades(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        **kwargs
    ) -> List[Trade]:
        """Get trade data (implementation-specific)."""
        pass
    
    @abstractmethod
    async def _get_order_book(
        self,
        symbol: str,
        limit: Optional[int] = None,
        **kwargs
    ) -> OrderBook:
        """Get order book snapshot (implementation-specific)."""
        pass
    
    @abstractmethod
    async def _get_funding_rate(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        **kwargs
    ) -> List[FundingRate]:
        """Get funding rate history (implementation-specific)."""
        pass
    
    @abstractmethod
    async def _get_open_interest(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        **kwargs
    ) -> List[OpenInterest]:
        """Get open interest history (implementation-specific)."""
        pass
    
    @abstractmethod
    async def _get_liquidations(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        **kwargs
    ) -> List[Liquidation]:
        """Get liquidation history (implementation-specific)."""
        pass
    
    @abstractmethod
    async def _subscribe(self, channel: str, symbol: str, **kwargs) -> None:
        """Subscribe to a data stream (implementation-specific)."""
        pass
    
    @abstractmethod
    async def _unsubscribe(self, channel: str, symbol: str, **kwargs) -> None:
        """Unsubscribe from a data stream (implementation-specific)."""
        pass

class DataProviderFactory:
    """Factory for creating data provider instances."""
    
    _providers: Dict[str, Type[DataProvider]] = {}
    
    @classmethod
    def register_provider(cls, name: str, provider_class: Type[DataProvider]) -> None:
        """Register a data provider class.
        
        Args:
            name: Provider name
            provider_class: DataProvider subclass
        """
        if not issubclass(provider_class, DataProvider):
            raise TypeError(f"Provider class must be a subclass of DataProvider")
        cls._providers[name.lower()] = provider_class
    
    @classmethod
    def create_provider(cls, config: DataProviderConfig) -> DataProvider:
        """Create a data provider instance.
        
        Args:
            config: Data provider configuration
            
        Returns:
            DataProvider instance
            
        Raises:
            ValueError: If the provider is not registered
        """
        provider_name = config.provider_name.lower()
        if provider_name not in cls._providers:
            raise ValueError(f"No data provider registered with name: {provider_name}")
        
        return cls._providers[provider_name](config)
