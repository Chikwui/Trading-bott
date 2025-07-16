"""Data models for market data and related entities."""
from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from pydantic import BaseModel, Field, validator, HttpUrl
import pandas as pd
import numpy as np

class TimeFrame(str, Enum):
    """Supported timeframes for market data."""
    TICK = "tick"
    SECOND = "1s"
    MINUTE_1 = "1m"
    MINUTE_3 = "3m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    MINUTE_30 = "30m"
    HOUR_1 = "1h"
    HOUR_2 = "2h"
    HOUR_4 = "4h"
    HOUR_6 = "6h"
    HOUR_8 = "8h"
    HOUR_12 = "12h"
    DAY_1 = "1d"
    DAY_3 = "3d"
    WEEK_1 = "1w"
    MONTH_1 = "1M"

class DataType(str, Enum):
    """Types of market data."""
    OHLCV = "ohlcv"
    TRADES = "trades"
    ORDER_BOOK = "order_book"
    FUNDING_RATE = "funding_rate"
    OPEN_INTEREST = "open_interest"
    LIQUIDATIONS = "liquidations"
    MARKET_DEPTH = "market_depth"

class DataQuality(str, Enum):
    """Data quality indicators."""
    REAL_TIME = "realtime"
    DELAYED = "delayed"
    HISTORICAL = "historical"
    SYNTHETIC = "synthetic"
    INTERPOLATED = "interpolated"

class Exchange(str, Enum):
    """Supported exchanges."""
    BINANCE = "binance"
    BINANCE_FUTURES = "binance_futures"
    COINBASE = "coinbase"
    KRAKEN = "kraken"
    BYBIT = "bybit"
    DERIBIT = "deribit"
    OKX = "okx"
    BITMEX = "bitmex"
    FTX = "ftx"
    MT5 = "mt5"
    TESTNET = "testnet"

class InstrumentType(str, Enum):
    """Types of financial instruments."""
    SPOT = "spot"
    FUTURES = "futures"
    PERPETUAL = "perpetual"
    OPTION = "option"
    FOREX = "forex"
    COMMODITY = "commodity"
    INDEX = "index"
    STOCK = "stock"
    CRYPTO = "cryptocurrency"

class Instrument(BaseModel):
    """Financial instrument representation."""
    symbol: str
    exchange: Exchange
    type: InstrumentType
    base_currency: str
    quote_currency: str
    is_active: bool = True
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    tick_size: Optional[float] = None
    min_qty: Optional[float] = None
    max_qty: Optional[float] = None
    step_size: Optional[float] = None
    contract_size: Optional[float] = None
    expiry: Optional[datetime] = None
    strike: Optional[float] = None
    option_type: Optional[str] = None  # 'call' or 'put'
    lot_size: Optional[float] = None
    margin_currency: Optional[str] = None
    leverage: Optional[float] = None
    is_inverse: bool = False
    is_quanto: bool = False
    maker_fee: Optional[float] = None
    taker_fee: Optional[float] = None
    funding_rate: Optional[float] = None
    next_funding_time: Optional[datetime] = None
    open_interest: Optional[float] = None
    volume_24h: Optional[float] = None
    price_change_24h: Optional[float] = None
    last_updated: Optional[datetime] = None
    metadata: Dict[str, Any] = {}

class OHLCV(BaseModel):
    """OHLCV (Open-High-Low-Close-Volume) data point."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    quote_volume: Optional[float] = None
    trades: Optional[int] = None
    vwap: Optional[float] = None
    resolution: TimeFrame
    symbol: str
    exchange: Exchange
    data_quality: DataQuality = DataQuality.REAL_TIME

class Trade(BaseModel):
    """Trade data point."""
    timestamp: datetime
    price: float
    amount: float
    side: str  # 'buy' or 'sell'
    symbol: str
    exchange: Exchange
    trade_id: Optional[str] = None
    order_id: Optional[str] = None
    taker_side: Optional[str] = None  # 'maker' or 'taker'
    fee: Optional[float] = None
    fee_currency: Optional[str] = None
    data_quality: DataQuality = DataQuality.REAL_TIME

class OrderBook(BaseModel):
    """Order book snapshot."""
    timestamp: datetime
    bids: List[Tuple[float, float]]  # [(price, amount), ...]
    asks: List[Tuple[float, float]]  # [(price, amount), ...]
    symbol: str
    exchange: Exchange
    data_quality: DataQuality = DataQuality.REAL_TIME

class FundingRate(BaseModel):
    """Funding rate data point."""
    timestamp: datetime
    symbol: str
    exchange: Exchange
    funding_rate: float
    funding_time: datetime
    funding_rate_8h: Optional[float] = None
    predicted_funding_rate: Optional[float] = None
    data_quality: DataQuality = DataQuality.REAL_TIME

class OpenInterest(BaseModel):
    """Open interest data point."""
    timestamp: datetime
    symbol: str
    exchange: Exchange
    open_interest: float
    open_value: Optional[float] = None
    data_quality: DataQuality = DataQuality.REAL_TIME

class Liquidation(BaseModel):
    """Liquidation data point."""
    timestamp: datetime
    symbol: str
    exchange: Exchange
    price: float
    quantity: float
    side: str  # 'long' or 'short'
    value: Optional[float] = None
    data_quality: DataQuality = DataQuality.REAL_TIME

class MarketDataRequest(BaseModel):
    """Request for market data."""
    symbol: str
    exchange: Exchange
    data_type: DataType
    timeframe: Optional[TimeFrame] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    limit: Optional[int] = 1000
    params: Dict[str, Any] = {}

class MarketDataResponse(BaseModel):
    """Response containing market data."""
    request: MarketDataRequest
    data: Union[List[OHLCV], List[Trade], List[OrderBook], List[FundingRate], List[OpenInterest], List[Liquidation]]
    next_page_token: Optional[str] = None
    data_quality: DataQuality = DataQuality.REAL_TIME

class DataProviderConfig(BaseModel):
    """Configuration for data providers."""
    provider_name: str
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    api_passphrase: Optional[str] = None
    ws_url: Optional[str] = None
    rest_url: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3
    rate_limit: Optional[int] = None
    rate_period: Optional[int] = None
    verify_ssl: bool = True
    proxy: Optional[Dict[str, str]] = None
    headers: Dict[str, str] = {}
    extra: Dict[str, Any] = {}

class DataCacheConfig(BaseModel):
    """Configuration for data caching."""
    enabled: bool = True
    ttl: int = 3600  # seconds
    max_size: int = 10000  # max number of items
    backend: str = "memory"  # 'memory', 'redis', 'disk', etc.
    path: Optional[str] = None  # for disk cache
    host: Optional[str] = None  # for redis/memcached
    port: Optional[int] = None  # for redis/memcached
    db: Optional[int] = None  # for redis
    password: Optional[str] = None  # for redis
    prefix: str = "trading_bot:"  # cache key prefix

class DataManagerConfig(BaseModel):
    """Configuration for data manager."""
    default_provider: str
    providers: Dict[str, DataProviderConfig]
    cache: DataCacheConfig
    default_timeframe: TimeFrame = TimeFrame.MINUTE_15
    default_limit: int = 1000
    max_retries: int = 3
    retry_delay: int = 1  # seconds
    timeout: int = 30  # seconds
    verify_ssl: bool = True
    proxy: Optional[Dict[str, str]] = None
    headers: Dict[str, str] = {}
    extra: Dict[str, Any] = {}
