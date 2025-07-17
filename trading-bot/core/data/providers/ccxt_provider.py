"""CCXT data provider implementation."""
from __future__ import annotations
from typing import Dict, List, Optional, Union, Any, Type, TypeVar, Tuple
from datetime import datetime, timedelta
import asyncio
import logging
import ccxt.async_support as ccxt
import pandas as pd
import numpy as np
from ...exchange import Exchange
from ..provider import DataProvider
from ...models import (
    DataType, TimeFrame, DataQuality, Instrument, OHLCV, Trade, OrderBook,
    FundingRate, OpenInterest, Liquidation, MarketDataRequest, MarketDataResponse,
    DataProviderConfig, DataCacheConfig, DataManagerConfig, InstrumentType,
    OrderBookLevel, OrderBookSide, TradeSide, Exchange as ExchangeModel
)

logger = logging.getLogger(__name__)

# CCXT timeframe mapping
CCXT_TIMEFRAME_MAP = {
    TimeFrame.M1: '1m',
    TimeFrame.M5: '5m',
    TimeFrame.M15: '15m',
    TimeFrame.M30: '30m',
    TimeFrame.H1: '1h',
    TimeFrame.H4: '4h',
    TimeFrame.D1: '1d',
    TimeFrame.W1: '1w',
    TimeFrame.MN1: '1M',
}

# Reverse mapping for CCXT timeframes
CCXT_TIMEFRAME_REVERSE_MAP = {v: k for k, v in CCXT_TIMEFRAME_MAP.items()}

# Map CCXT exchange names to our Exchange enum
EXCHANGE_MAP = {
    'binance': Exchange.BINANCE,
    'binanceusdm': Exchange.BINANCE_FUTURES,
    'binancecoinm': Exchange.BINANCE_COIN_M_FUTURES,
    'bitmex': Exchange.BITMEX,
    'bybit': Exchange.BYBIT,
    'ftx': Exchange.FTX,
    'kraken': Exchange.KRAKEN,
    'kucoin': Exchange.KUCOIN,
    'huobi': Exchange.HUOBI,
    'okx': Exchange.OKX,
    'bitfinex': Exchange.BITFINEX,
    'coinbase': Exchange.COINBASE,
    'bitstamp': Exchange.BITSTAMP,
    'gemini': Exchange.GEMINI,
    'poloniex': Exchange.POLONIEX,
    'bitflyer': Exchange.BITFLYER,
    'bitso': Exchange.BITSO,
    'cex': Exchange.CEX,
    'hitbtc': Exchange.HITBTC,
    'bitmart': Exchange.BITMART,
    'bitrue': Exchange.BITRUE,
    'mexc': Exchange.MEXC,
    'phemex': Exchange.PHEMEX,
    'gateio': Exchange.GATEIO,
    'lbank': Exchange.LBANK,
    'whitebit': Exchange.WHITEBIT,
    'bitget': Exchange.BITGET,
    'coinex': Exchange.COINEX,
    'bittrex': Exchange.BITTREX,
    'probit': Exchange.PROBIT,
    'ascendex': Exchange.ASCENDEX,
    'bigone': Exchange.BIGONE,
    'bitvavo': Exchange.BITVAVO,
    'bitz': Exchange.BITZ,
    'btcalpha': Exchange.BTCALPHA,
    'btcmarkets': Exchange.BTCMARKETS,
    'bw': Exchange.BW,
    'coincheck': Exchange.COINCHECK,
    'coinmate': Exchange.COINMATE,
    'crex24': Exchange.CREX24,
    'currencycom': Exchange.CURRENCYCOM,
    'digifinex': Exchange.DIGIFINEX,
    'eqonex': Exchange.EQONEX,
    'exmo': Exchange.EXMO,
    'flowbtc': Exchange.FLOWBTC,
    'ftxus': Exchange.FTXUS,
    'independentreserve': Exchange.INDEPENDENTRESERVE,
    'itbit': Exchange.ITBIT,
    'luno': Exchange.LUNO,
    'ndax': Exchange.NDAX,
    'novadax': Exchange.NOVADAX,
    'oceanex': Exchange.OCEANEX,
    'paribu': Exchange.PARIBU,
    'ripio': Exchange.RIPIO,
    'stex': Exchange.STEX,
    'therock': Exchange.THEROCK,
    'tidex': Exchange.TIDEX,
    'timex': Exchange.TIMEX,
    'upbit': Exchange.UPBIT,
    'wavesexchange': Exchange.WAVESEXCHANGE,
    'wazirx': Exchange.WAZIRX,
    'yobit': Exchange.YOBIT,
    'zaif': Exchange.ZAIF,
    'zb': Exchange.ZB,
}

class CCXTDataProvider(DataProvider):
    """CCXT data provider implementation."""
    
    def __init__(self, config: DataProviderConfig):
        """Initialize the CCXT data provider.
        
        Args:
            config: Provider configuration
        """
        super().__init__(config)
        self._exchange = None
        self._connected = False
        self._initialized = False
        self._markets = {}
        self._symbols_info = {}
        
        # Extract exchange ID from config
        self._exchange_id = config.extra.get('exchange_id', 'binance').lower()
        
        # CCXT-specific configuration
        self._api_key = config.extra.get('api_key')
        self._api_secret = config.extra.get('api_secret')
        self._password = config.extra.get('password')
        self._uid = config.extra.get('uid')
        self._options = config.extra.get('options', {})
        self._timeout = config.extra.get('timeout', 30000)  # Default 30 seconds
        
        # Rate limiting
        self._rate_limit = config.extra.get('rate_limit', True)
        self._enable_rate_limit = config.extra.get('enable_rate_limit', True)
        
        # Initialize the exchange
        self._init_exchange()
    
    def _init_exchange(self) -> None:
        """Initialize the CCXT exchange instance."""
        # Get the exchange class
        exchange_class = getattr(ccxt, self._exchange_id, None)
        if exchange_class is None:
            raise ValueError(f"Unsupported exchange: {self._exchange_id}")
        
        # Create exchange config
        config = {
            'apiKey': self._api_key,
            'secret': self._api_secret,
            'password': self._password,
            'uid': self._uid,
            'timeout': self._timeout,
            'enableRateLimit': self._enable_rate_limit,
            'options': self._options,
        }
        
        # Remove None values
        config = {k: v for k, v in config.items() if v is not None}
        
        # Create exchange instance
        self._exchange = exchange_class(config)
        
        # Set rate limiter if needed
        if self._rate_limit and not self._enable_rate_limit:
            self._exchange.enableRateLimit = True
    
    async def _initialize(self) -> None:
        """Initialize the CCXT data provider."""
        if self._initialized:
            return
        
        try:
            # Load markets
            await self._load_markets()
            
            self._initialized = True
            logger.info(f"CCXT data provider initialized for {self._exchange_id}")
        except Exception as e:
            logger.error(f"Failed to initialize CCXT provider: {e}")
            raise
    
    async def _connect(self) -> None:
        """Connect to the exchange."""
        if not self._initialized:
            await self.initialize()
        
        self._connected = True
    
    async def _disconnect(self) -> None:
        """Disconnect from the exchange."""
        if not self._connected:
            return
        
        try:
            if self._exchange:
                await self._exchange.close()
        except Exception as e:
            logger.error(f"Error disconnecting from CCXT exchange: {e}")
        finally:
            self._connected = False
    
    async def _load_markets(self) -> None:
        """Load and cache markets from the exchange."""
        if not self._exchange:
            raise RuntimeError("Exchange not initialized")
        
        try:
            await self._exchange.load_markets()
            self._markets = self._exchange.markets
            self._symbols_info = {}
            
            # Convert to our format
            for symbol, market in self._markets.items():
                self._symbols_info[symbol] = self._parse_market_info(market)
            
            logger.info(f"Loaded {len(self._markets)} markets from {self._exchange_id}")
        except Exception as e:
            logger.error(f"Failed to load markets from {self._exchange_id}: {e}")
            raise
    
    def _parse_market_info(self, market: dict) -> dict:
        """Parse CCXT market info into our format."""
        return {
            'symbol': market['symbol'],
            'base': market['base'],
            'quote': market['quote'],
            'base_id': market['baseId'],
            'quote_id': market['quoteId'],
            'active': market['active'],
            'precision': market.get('precision', {}),
            'limits': market.get('limits', {}),
            'taker': market.get('taker', 0),
            'maker': market.get('maker', 0),
            'percentage': market.get('percentage', False),
            'tierBased': market.get('tierBased', False),
            'fee_loaded': market.get('fee_loaded', False),
            'info': market.get('info', {})
        }
    
    def _map_timeframe(self, timeframe: Union[TimeFrame, str]) -> str:
        """Map TimeFrame enum to CCXT timeframe string.
        
        Args:
            timeframe: TimeFrame enum or string representation
            
        Returns:
            CCXT timeframe string
            
        Raises:
            ValueError: If the timeframe is not supported
        """
        if isinstance(timeframe, str):
            timeframe = TimeFrame[timeframe.upper()]
        
        if timeframe not in CCXT_TIMEFRAME_MAP:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
            
        return CCXT_TIMEFRAME_MAP[timeframe]
    
    def _map_instrument_type(self, market: dict) -> InstrumentType:
        """Map CCXT market to instrument type.
        
        Args:
            market: CCXT market info
            
        Returns:
            InstrumentType enum
        """
        if market.get('future', False):
            return InstrumentType.FUTURE
        elif market.get('swap', False):
            return InstrumentType.PERPETUAL
        elif market.get('option', False):
            return InstrumentType.OPTION
        elif market.get('spot', False):
            return InstrumentType.SPOT
        else:
            # Try to determine from symbol or other attributes
            symbol = market.get('symbol', '').upper()
            if 'PERP' in symbol or 'SWAP' in symbol:
                return InstrumentType.PERPETUAL
            elif 'FUT' in symbol or 'USD' in symbol or 'USDT' in symbol:
                return InstrumentType.FUTURE
            else:
                return InstrumentType.SPOT
    
    def _map_exchange(self) -> Exchange:
        """Map CCXT exchange to our Exchange enum.
        
        Returns:
            Exchange enum
        """
        return EXCHANGE_MAP.get(self._exchange_id, Exchange.UNKNOWN)
    
    async def _get_instruments(self, **kwargs) -> List[Instrument]:
        """Get all available trading instruments.
        
        Args:
            **kwargs: Additional parameters
            
        Returns:
            List of Instrument objects
        """
        if not self._markets:
            await self._load_markets()
        
        instruments = []
        for symbol, market in self._markets.items():
            try:
                # Skip inactive markets
                if not market.get('active', True):
                    continue
                
                # Parse precision
                precision = market.get('precision', {})
                price_precision = precision.get('price', 8)
                amount_precision = precision.get('amount', 8)
                cost_precision = precision.get('cost', 8)
                
                # Parse limits
                limits = market.get('limits', {})
                amount_limits = limits.get('amount', {})
                price_limits = limits.get('price', {})
                cost_limits = limits.get('cost', {})
                
                # Create instrument
                instrument = Instrument(
                    symbol=market['symbol'],
                    name=f"{market['base']}/{market['quote']}",
                    exchange=self._map_exchange(),
                    type=self._map_instrument_type(market),
                    base_currency=market['base'],
                    quote_currency=market['quote'],
                    price_precision=price_precision,
                    min_price_increment=10 ** -price_precision,
                    min_quantity=amount_limits.get('min'),
                    max_quantity=amount_limits.get('max'),
                    quantity_precision=amount_precision,
                    min_notional=cost_limits.get('min'),
                    is_active=market.get('active', True),
                    metadata={
                        'precision': precision,
                        'limits': limits,
                        'taker': market.get('taker'),
                        'maker': market.get('maker'),
                        'percentage': market.get('percentage'),
                        'tierBased': market.get('tierBased'),
                        'info': market.get('info', {})
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
        """Get OHLCV data from the exchange.
        
        Args:
            symbol: Symbol name (e.g., 'BTC/USDT')
            timeframe: TimeFrame enum or string
            start_time: Start time for the data
            end_time: End time for the data (defaults to now)
            limit: Maximum number of candles to return
            **kwargs: Additional parameters
            
        Returns:
            List of OHLCV objects
        """
        if not self._exchange:
            raise RuntimeError("Exchange not initialized")
        
        # Map timeframe
        tf = self._map_timeframe(timeframe)
        
        # Convert to timestamp in milliseconds
        since = None
        if start_time:
            since = int(start_time.timestamp() * 1000)
        
        # Fetch OHLCV data
        try:
            ohlcv_data = await self._exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=tf,
                since=since,
                limit=limit,
                params=kwargs
            )
            
            # Convert to our format
            ohlcv_list = []
            for item in ohlcv_data:
                timestamp = datetime.fromtimestamp(item[0] / 1000)
                ohlcv = OHLCV(
                    timestamp=timestamp,
                    open=item[1],
                    high=item[2],
                    low=item[3],
                    close=item[4],
                    volume=item[5],
                    symbol=symbol,
                    timeframe=timeframe if isinstance(timeframe, TimeFrame) else TimeFrame[timeframe.upper()],
                    exchange=self._map_exchange(),
                    data_quality=DataQuality.REAL_TIME,
                    metadata={
                        'exchange': self._exchange_id,
                        'timeframe': tf,
                        'timestamp': item[0]
                    }
                )
                ohlcv_list.append(ohlcv)
            
            return ohlcv_list
            
        except Exception as e:
            logger.error(f"Failed to get OHLCV data for {symbol}: {e}")
            raise
    
    async def _get_trades(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        **kwargs
    ) -> List[Trade]:
        """Get historical trade data from the exchange.
        
        Args:
            symbol: Symbol name
            start_time: Start time for the data
            end_time: End time for the data (defaults to now)
            limit: Maximum number of trades to return
            **kwargs: Additional parameters
            
        Returns:
            List of Trade objects
        """
        if not self._exchange:
            raise RuntimeError("Exchange not initialized")
        
        # Convert to timestamp in milliseconds
        since = None
        if start_time:
            since = int(start_time.timestamp() * 1000)
        
        # Fetch trades
        try:
            trades = await self._exchange.fetch_trades(
                symbol=symbol,
                since=since,
                limit=limit,
                params=kwargs
            )
            
            # Convert to our format
            trade_list = []
            for trade in trades:
                timestamp = datetime.fromtimestamp(trade['timestamp'] / 1000)
                trade_obj = Trade(
                    trade_id=trade.get('id'),
                    symbol=trade['symbol'],
                    exchange=self._map_exchange(),
                    price=trade['price'],
                    quantity=trade['amount'],
                    side=TradeSide.BUY if trade['side'] == 'buy' else TradeSide.SELL,
                    timestamp=timestamp,
                    data_quality=DataQuality.REAL_TIME,
                    metadata={
                        'exchange': self._exchange_id,
                        'order': trade.get('order'),
                        'type': trade.get('type'),
                        'taker_or_maker': trade.get('takerOrMaker'),
                        'fee': trade.get('fee'),
                        'info': trade.get('info', {})
                    }
                )
                trade_list.append(trade_obj)
            
            # Apply end_time filter if needed
            if end_time:
                trade_list = [t for t in trade_list if t.timestamp <= end_time]
            
            # Apply limit if specified
            if limit is not None and len(trade_list) > limit:
                trade_list = trade_list[-limit:]
            
            return trade_list
            
        except Exception as e:
            logger.error(f"Failed to get trades for {symbol}: {e}")
            raise
    
    async def _get_order_book(
        self,
        symbol: str,
        limit: Optional[int] = None,
        **kwargs
    ) -> OrderBook:
        """Get order book snapshot from the exchange.
        
        Args:
            symbol: Symbol name
            limit: Maximum number of levels to return
            **kwargs: Additional parameters
            
        Returns:
            OrderBook object
        """
        if not self._exchange:
            raise RuntimeError("Exchange not initialized")
        
        try:
            # Fetch order book
            orderbook = await self._exchange.fetch_order_book(
                symbol=symbol,
                limit=limit,
                params=kwargs
            )
            
            # Convert bids and asks to our format
            bids = [
                OrderBookLevel(price=price, size=size, count=1)
                for price, size in orderbook.get('bids', [])
            ]
            
            asks = [
                OrderBookLevel(price=price, size=size, count=1)
                for price, size in orderbook.get('asks', [])
            ]
            
            return OrderBook(
                symbol=symbol,
                exchange=self._map_exchange(),
                timestamp=datetime.now(),
                bids=bids,
                asks=asks,
                data_quality=DataQuality.REAL_TIME,
                metadata={
                    'exchange': self._exchange_id,
                    'nonce': orderbook.get('nonce'),
                    'datetime': orderbook.get('datetime'),
                    'timestamp': orderbook.get('timestamp'),
                    'info': orderbook.get('info', {})
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to get order book for {symbol}: {e}")
            raise
    
    async def _get_funding_rate(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        **kwargs
    ) -> List[FundingRate]:
        """Get funding rate history.
        
        Args:
            symbol: Symbol name
            start_time: Start time for the data
            end_time: End time for the data (defaults to now)
            limit: Maximum number of data points to return
            **kwargs: Additional parameters
            
        Returns:
            List of FundingRate objects
        """
        if not self._exchange:
            raise RuntimeError("Exchange not initialized")
        
        try:
            # Check if the exchange supports funding rate
            if not hasattr(self._exchange, 'fetch_funding_rate_history'):
                logger.warning(f"Exchange {self._exchange_id} does not support fetching funding rate history")
                return []
            
            # Convert to timestamp in milliseconds
            since = int(start_time.timestamp() * 1000) if start_time else None
            
            # Fetch funding rate history
            funding_rates = await self._exchange.fetch_funding_rate_history(
                symbol=symbol,
                since=since,
                limit=limit,
                params=kwargs
            )
            
            # Convert to our format
            result = []
            for rate in funding_rates:
                timestamp = datetime.fromtimestamp(rate['timestamp'] / 1000)
                
                # Skip if outside the time range
                if start_time and timestamp < start_time:
                    continue
                if end_time and timestamp > end_time:
                    continue
                
                funding_rate = FundingRate(
                    symbol=symbol,
                    exchange=self._map_exchange(),
                    timestamp=timestamp,
                    rate=rate['fundingRate'],
                    next_funding_time=datetime.fromtimestamp(rate['nextFundingTime'] / 1000) if 'nextFundingTime' in rate else None,
                    data_quality=DataQuality.REAL_TIME,
                    metadata={
                        'exchange': self._exchange_id,
                        'info': rate.get('info', {})
                    }
                )
                result.append(funding_rate)
            
            # Apply limit if specified
            if limit is not None and len(result) > limit:
                result = result[-limit:]
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get funding rate for {symbol}: {e}")
            raise
    
    async def _get_open_interest(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        **kwargs
    ) -> List[OpenInterest]:
        """Get open interest history.
        
        Args:
            symbol: Symbol name
            start_time: Start time for the data
            end_time: End time for the data (defaults to now)
            limit: Maximum number of data points to return
            **kwargs: Additional parameters
            
        Returns:
            List of OpenInterest objects
        """
        if not self._exchange:
            raise RuntimeError("Exchange not initialized")
        
        try:
            # Check if the exchange supports open interest
            if not hasattr(self._exchange, 'fetch_open_interest_history'):
                logger.warning(f"Exchange {self._exchange_id} does not support fetching open interest history")
                return []
            
            # Convert to timestamp in milliseconds
            since = int(start_time.timestamp() * 1000) if start_time else None
            
            # Fetch open interest history
            oi_data = await self._exchange.fetch_open_interest_history(
                symbol=symbol,
                timeframe=None,  # Some exchanges support timeframe-based OI
                since=since,
                limit=limit,
                params=kwargs
            )
            
            # Convert to our format
            result = []
            for oi in oi_data:
                timestamp = datetime.fromtimestamp(oi['timestamp'] / 1000)
                
                # Skip if outside the time range
                if start_time and timestamp < start_time:
                    continue
                if end_time and timestamp > end_time:
                    continue
                
                open_interest = OpenInterest(
                    symbol=symbol,
                    exchange=self._map_exchange(),
                    timestamp=timestamp,
                    open_interest=oi['openInterestValue'] if 'openInterestValue' in oi else oi['openInterest'],
                    data_quality=DataQuality.REAL_TIME,
                    metadata={
                        'exchange': self._exchange_id,
                        'info': oi.get('info', {})
                    }
                )
                result.append(open_interest)
            
            # Apply limit if specified
            if limit is not None and len(result) > limit:
                result = result[-limit:]
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to get open interest for {symbol}: {e}")
            raise
    
    async def _get_liquidations(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        **kwargs
    ) -> List[Liquidation]:
        """Get liquidation history.
        
        Note: Not all exchanges provide this data through CCXT.
        This is a placeholder implementation that would need to be adapted.
        """
        logger.warning(f"Liquidation data is not directly available from {self._exchange_id} via CCXT")
        return []
    
    async def _subscribe(self, channel: str, symbol: str, **kwargs) -> None:
        """Subscribe to a data stream.
        
        Args:
            channel: Data channel ('ticker', 'trades', 'orderbook')
            symbol: Symbol name
            **kwargs: Additional parameters
        """
        # CCXT doesn't directly support WebSocket subscriptions in the standard way
        # This would need to be implemented using the exchange's WebSocket API directly
        logger.info(f"Subscribed to {channel} for {symbol} (not implemented in CCXT)")
    
    async def _unsubscribe(self, channel: str, symbol: str, **kwargs) -> None:
        """Unsubscribe from a data stream.
        
        Args:
            channel: Data channel
            symbol: Symbol name
            **kwargs: Additional parameters
        """
        logger.info(f"Unsubscribed from {channel} for {symbol} (not implemented in CCXT)")

# Register the provider with the factory
DataProviderFactory.register_provider('ccxt', CCXTDataProvider)
