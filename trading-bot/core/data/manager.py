"""Data manager for handling multiple data providers and caching."""
from __future__ import annotations
from typing import Dict, List, Optional, Union, Any, Type, TypeVar, Generic, cast
from datetime import datetime
import asyncio
import logging
from .provider import DataProvider, DataProviderFactory
from .cache import DataCache
from .models import (
    DataType, TimeFrame, DataQuality, Instrument, OHLCV, Trade, OrderBook,
    FundingRate, OpenInterest, Liquidation, MarketDataRequest, MarketDataResponse,
    DataProviderConfig, DataCacheConfig, DataManagerConfig, Exchange, InstrumentType
)

logger = logging.getLogger(__name__)
T = TypeVar('T')

class DataManager:
    """Manager for handling multiple data providers and caching."""
    
    def __init__(self, config: DataManagerConfig):
        """Initialize the data manager.
        
        Args:
            config: Configuration for the data manager
        """
        self.config = config
        self._providers: Dict[str, DataProvider] = {}
        self._default_provider: Optional[str] = config.default_provider
        self._cache = DataCache(config.cache)
        self._initialized = False
        
    @property
    def is_initialized(self) -> bool:
        """Check if the data manager is initialized."""
        return self._initialized
    
    async def initialize(self) -> None:
        """Initialize the data manager and all providers."""
        if self._initialized:
            return
        
        # Initialize all providers
        for name, provider_config in self.config.providers.items():
            try:
                provider = DataProviderFactory.create_provider(provider_config)
                await provider.initialize()
                self._providers[name] = provider
                logger.info(f"Initialized data provider: {name}")
            except Exception as e:
                logger.error(f"Failed to initialize data provider {name}: {e}")
                raise
        
        # Set default provider if not set
        if not self._default_provider and self._providers:
            self._default_provider = next(iter(self._providers.keys()))
        
        self._initialized = True
        logger.info(f"Data manager initialized with {len(self._providers)} providers")
    
    async def connect(self) -> None:
        """Connect to all data providers."""
        if not self._initialized:
            await self.initialize()
        
        for name, provider in self._providers.items():
            try:
                await provider.connect()
                logger.info(f"Connected to data provider: {name}")
            except Exception as e:
                logger.error(f"Failed to connect to data provider {name}: {e}")
                raise
    
    async def disconnect(self) -> None:
        """Disconnect from all data providers."""
        for name, provider in self._providers.items():
            try:
                await provider.disconnect()
                logger.info(f"Disconnected from data provider: {name}")
            except Exception as e:
                logger.error(f"Error disconnecting from data provider {name}: {e}")
    
    def get_provider(self, provider_name: Optional[str] = None) -> DataProvider:
        """Get a data provider by name.
        
        Args:
            provider_name: Name of the provider, or None to use default
            
        Returns:
            DataProvider instance
            
        Raises:
            ValueError: If the provider is not found
        """
        name = provider_name or self._default_provider
        if not name:
            raise ValueError("No provider specified and no default provider set")
        
        if name not in self._providers:
            raise ValueError(f"No data provider found with name: {name}")
        
        return self._providers[name]
    
    async def get_instruments(
        self,
        exchange: Optional[Union[Exchange, str]] = None,
        provider: Optional[str] = None,
        **kwargs
    ) -> List[Instrument]:
        """Get all available trading instruments.
        
        Args:
            exchange: Filter by exchange
            provider: Data provider to use
            **kwargs: Additional parameters for the request
            
        Returns:
            List of available trading instruments
        """
        cache_key = f"instruments:{exchange or 'all'}"
        
        # Try to get from cache first
        cached = await self._cache.get(cache_key)
        if cached is not None:
            return cached
        
        # Get from provider
        provider_inst = self.get_provider(provider)
        instruments = await provider_inst.get_instruments(**kwargs)
        
        # Apply filters
        if exchange is not None:
            exchange_name = exchange.value if isinstance(exchange, Exchange) else exchange
            instruments = [i for i in instruments if i.exchange.value.lower() == exchange_name.lower()]
        
        # Cache the result
        await self._cache.set(cache_key, instruments)
        
        return instruments
    
    async def get_ohlcv(
        self,
        symbol: str,
        timeframe: Union[TimeFrame, str],
        exchange: Optional[Union[Exchange, str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        provider: Optional[str] = None,
        use_cache: bool = True,
        **kwargs
    ) -> List[OHLCV]:
        """Get OHLCV data.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe for the data
            exchange: Exchange to get data from
            start_time: Start time for the data
            end_time: End time for the data
            limit: Maximum number of data points to return
            provider: Data provider to use
            use_cache: Whether to use cached data if available
            **kwargs: Additional parameters for the request
            
        Returns:
            List of OHLCV data points
        """
        # Generate cache key
        exchange_name = exchange.value if isinstance(exchange, Exchange) else exchange
        cache_key = f"ohlcv:{exchange_name or 'any'}:{symbol}:{timeframe}:{start_time}:{end_time}:{limit}"
        
        # Try to get from cache first
        if use_cache:
            cached = await self._cache.get(cache_key)
            if cached is not None:
                return cached
        
        # Get from provider
        provider_inst = self.get_provider(provider)
        ohlcv_data = await provider_inst.get_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            **kwargs
        )
        
        # Cache the result
        if use_cache:
            await self._cache.set(cache_key, ohlcv_data)
        
        return ohlcv_data
    
    async def get_trades(
        self,
        symbol: str,
        exchange: Optional[Union[Exchange, str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        provider: Optional[str] = None,
        use_cache: bool = True,
        **kwargs
    ) -> List[Trade]:
        """Get trade data.
        
        Args:
            symbol: Trading pair symbol
            exchange: Exchange to get data from
            start_time: Start time for the data
            end_time: End time for the data
            limit: Maximum number of trades to return
            provider: Data provider to use
            use_cache: Whether to use cached data if available
            **kwargs: Additional parameters for the request
            
        Returns:
            List of trade data points
        """
        # Generate cache key
        exchange_name = exchange.value if isinstance(exchange, Exchange) else exchange
        cache_key = f"trades:{exchange_name or 'any'}:{symbol}:{start_time}:{end_time}:{limit}"
        
        # Try to get from cache first
        if use_cache:
            cached = await self._cache.get(cache_key)
            if cached is not None:
                return cached
        
        # Get from provider
        provider_inst = self.get_provider(provider)
        trades = await provider_inst.get_trades(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            **kwargs
        )
        
        # Cache the result
        if use_cache:
            await self._cache.set(cache_key, trades)
        
        return trades
    
    async def get_order_book(
        self,
        symbol: str,
        exchange: Optional[Union[Exchange, str]] = None,
        limit: Optional[int] = None,
        provider: Optional[str] = None,
        use_cache: bool = True,
        **kwargs
    ) -> OrderBook:
        """Get order book snapshot.
        
        Args:
            symbol: Trading pair symbol
            exchange: Exchange to get data from
            limit: Maximum number of order book levels to return
            provider: Data provider to use
            use_cache: Whether to use cached data if available
            **kwargs: Additional parameters for the request
            
        Returns:
            Order book snapshot
        """
        # Generate cache key
        exchange_name = exchange.value if isinstance(exchange, Exchange) else exchange
        cache_key = f"orderbook:{exchange_name or 'any'}:{symbol}:{limit}"
        
        # Try to get from cache first
        if use_cache:
            cached = await self._cache.get(cache_key)
            if cached is not None:
                return cached
        
        # Get from provider
        provider_inst = self.get_provider(provider)
        orderbook = await provider_inst.get_order_book(
            symbol=symbol,
            limit=limit,
            **kwargs
        )
        
        # Cache the result with a short TTL since order books change frequently
        if use_cache:
            await self._cache.set(cache_key, orderbook, ttl=5)  # 5 seconds TTL
        
        return orderbook
    
    async def get_funding_rate(
        self,
        symbol: str,
        exchange: Optional[Union[Exchange, str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        provider: Optional[str] = None,
        use_cache: bool = True,
        **kwargs
    ) -> List[FundingRate]:
        """Get funding rate history.
        
        Args:
            symbol: Trading pair symbol
            exchange: Exchange to get data from
            start_time: Start time for the data
            end_time: End time for the data
            limit: Maximum number of data points to return
            provider: Data provider to use
            use_cache: Whether to use cached data if available
            **kwargs: Additional parameters for the request
            
        Returns:
            List of funding rate data points
        """
        # Generate cache key
        exchange_name = exchange.value if isinstance(exchange, Exchange) else exchange
        cache_key = f"funding_rate:{exchange_name or 'any'}:{symbol}:{start_time}:{end_time}:{limit}"
        
        # Try to get from cache first
        if use_cache:
            cached = await self._cache.get(cache_key)
            if cached is not None:
                return cached
        
        # Get from provider
        provider_inst = self.get_provider(provider)
        funding_rates = await provider_inst.get_funding_rate(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            **kwargs
        )
        
        # Cache the result
        if use_cache:
            await self._cache.set(cache_key, funding_rates, ttl=300)  # 5 minutes TTL
        
        return funding_rates
    
    async def get_open_interest(
        self,
        symbol: str,
        exchange: Optional[Union[Exchange, str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        provider: Optional[str] = None,
        use_cache: bool = True,
        **kwargs
    ) -> List[OpenInterest]:
        """Get open interest history.
        
        Args:
            symbol: Trading pair symbol
            exchange: Exchange to get data from
            start_time: Start time for the data
            end_time: End time for the data
            limit: Maximum number of data points to return
            provider: Data provider to use
            use_cache: Whether to use cached data if available
            **kwargs: Additional parameters for the request
            
        Returns:
            List of open interest data points
        """
        # Generate cache key
        exchange_name = exchange.value if isinstance(exchange, Exchange) else exchange
        cache_key = f"open_interest:{exchange_name or 'any'}:{symbol}:{start_time}:{end_time}:{limit}"
        
        # Try to get from cache first
        if use_cache:
            cached = await self._cache.get(cache_key)
            if cached is not None:
                return cached
        
        # Get from provider
        provider_inst = self.get_provider(provider)
        open_interest = await provider_inst.get_open_interest(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            **kwargs
        )
        
        # Cache the result
        if use_cache:
            await self._cache.set(cache_key, open_interest, ttl=60)  # 1 minute TTL
        
        return open_interest
    
    async def get_liquidations(
        self,
        symbol: str,
        exchange: Optional[Union[Exchange, str]] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        provider: Optional[str] = None,
        use_cache: bool = True,
        **kwargs
    ) -> List[Liquidation]:
        """Get liquidation history.
        
        Args:
            symbol: Trading pair symbol
            exchange: Exchange to get data from
            start_time: Start time for the data
            end_time: End time for the data
            limit: Maximum number of liquidations to return
            provider: Data provider to use
            use_cache: Whether to use cached data if available
            **kwargs: Additional parameters for the request
            
        Returns:
            List of liquidation events
        """
        # Generate cache key
        exchange_name = exchange.value if isinstance(exchange, Exchange) else exchange
        cache_key = f"liquidations:{exchange_name or 'any'}:{symbol}:{start_time}:{end_time}:{limit}"
        
        # Try to get from cache first
        if use_cache:
            cached = await self._cache.get(cache_key)
            if cached is not None:
                return cached
        
        # Get from provider
        provider_inst = self.get_provider(provider)
        liquidations = await provider_inst.get_liquidations(
            symbol=symbol,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            **kwargs
        )
        
        # Cache the result with a short TTL since liquidations are time-sensitive
        if use_cache:
            await self._cache.set(cache_key, liquidations, ttl=10)  # 10 seconds TTL
        
        return liquidations
    
    async def get_market_data(
        self,
        request: MarketDataRequest,
        provider: Optional[str] = None,
        use_cache: bool = True
    ) -> MarketDataResponse:
        """Get market data using a request object.
        
        Args:
            request: Market data request
            provider: Data provider to use
            use_cache: Whether to use cached data if available
            
        Returns:
            Market data response
        """
        # Map data type to the appropriate method
        handler_map = {
            DataType.OHLCV: self.get_ohlcv,
            DataType.TRADES: self.get_trades,
            DataType.ORDER_BOOK: self.get_order_book,
            DataType.FUNDING_RATE: self.get_funding_rate,
            DataType.OPEN_INTEREST: self.get_open_interest,
            DataType.LIQUIDATIONS: self.get_liquidations,
        }
        
        if request.data_type not in handler_map:
            raise ValueError(f"Unsupported data type: {request.data_type}")
        
        handler = handler_map[request.data_type]
        data = await handler(
            symbol=request.symbol,
            exchange=request.exchange,
            timeframe=request.timeframe,
            start_time=request.start_time,
            end_time=request.end_time,
            limit=request.limit,
            provider=provider,
            use_cache=use_cache,
            **request.params
        )
        
        return MarketDataResponse(
            request=request,
            data=data,
            data_quality=DataQuality.REAL_TIME  # TODO: Determine actual data quality
        )
    
    async def subscribe(
        self,
        channel: str,
        symbol: str,
        exchange: Optional[Union[Exchange, str]] = None,
        provider: Optional[str] = None,
        **kwargs
    ) -> None:
        """Subscribe to a data stream.
        
        Args:
            channel: Data channel to subscribe to (e.g., 'ticker', 'trades', 'orderbook')
            symbol: Trading pair symbol
            exchange: Exchange to subscribe to
            provider: Data provider to use
            **kwargs: Additional parameters for the subscription
        """
        provider_inst = self.get_provider(provider)
        await provider_inst.subscribe(
            channel=channel,
            symbol=symbol,
            exchange=exchange,
            **kwargs
        )
    
    async def unsubscribe(
        self,
        channel: str,
        symbol: str,
        exchange: Optional[Union[Exchange, str]] = None,
        provider: Optional[str] = None,
        **kwargs
    ) -> None:
        """Unsubscribe from a data stream.
        
        Args:
            channel: Data channel to unsubscribe from
            symbol: Trading pair symbol
            exchange: Exchange to unsubscribe from
            provider: Data provider to use
            **kwargs: Additional parameters for the unsubscription
        """
        provider_inst = self.get_provider(provider)
        await provider_inst.unsubscribe(
            channel=channel,
            symbol=symbol,
            exchange=exchange,
            **kwargs
        )
    
    # Context manager support
    async def __aenter__(self) -> 'DataManager':
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.disconnect()
