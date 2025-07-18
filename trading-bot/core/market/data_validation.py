"""
Advanced data validation and normalization for market data.
"""
from __future__ import annotations

import re
import json
import logging
from datetime import datetime, timezone, timedelta
from decimal import Decimal, ROUND_HALF_UP, DecimalException
from enum import Enum
from typing import (
    Any, Dict, List, Optional, Tuple, Union, Callable, TypeVar, Type, 
    get_type_hints, get_origin, get_args
)
import numpy as np
import pandas as pd
from pydantic import BaseModel, validator, root_validator, Field, create_model

logger = logging.getLogger(__name__)

T = TypeVar('T')

class DataQuality(str, Enum):
    """Data quality indicators."""
    REAL_TIME = 'real_time'      # Direct exchange feed
    DELAYED = 'delayed'         # Delayed data
    SNAPSHOT = 'snapshot'       # End-of-day snapshot
    ESTIMATED = 'estimated'     # Estimated/calculated values
    INCOMPLETE = 'incomplete'   # Missing some fields
    STALE = 'stale'            # Data is old
    
class DataSchemaType(str, Enum):
    """Supported data schema types."""
    TICK = 'tick'
    OHLCV = 'ohlcv'
    ORDER_BOOK = 'order_book'
    TRADE = 'trade'
    FUNDING_RATE = 'funding_rate'
    OPEN_INTEREST = 'open_interest'
    LIQUIDATION = 'liquidation'

class DataNormalizer:
    """Handles normalization of market data across different sources."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the data normalizer with optional configuration."""
        self.config = config or {}
        self._validators: Dict[DataSchemaType, Callable] = {
            DataSchemaType.TICK: self._normalize_tick,
            DataSchemaType.OHLCV: self._normalize_ohlcv,
            DataSchemaType.ORDER_BOOK: self._normalize_order_book,
            DataSchemaType.TRADE: self._normalize_trade,
            DataSchemaType.FUNDING_RATE: self._normalize_funding_rate,
            DataSchemaType.OPEN_INTEREST: self._normalize_open_interest,
            DataSchemaType.LIQUIDATION: self._normalize_liquidation,
        }
    
    async def normalize(
        self, 
        data: Any, 
        schema_type: DataSchemaType,
        source: Optional[str] = None,
        **kwargs
    ) -> Any:
        """Normalize market data according to the specified schema."""
        if schema_type not in self._validators:
            raise ValueError(f"Unsupported schema type: {schema_type}")
        
        try:
            normalizer = self._validators[schema_type]
            return await normalizer(data, source=source, **kwargs)
        except Exception as e:
            logger.error(f"Error normalizing {schema_type} data: {e}")
            raise
    
    async def _normalize_tick(
        self, 
        tick: Dict[str, Any], 
        source: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Normalize tick data."""
        # Source-specific normalization
        if source == 'mt5':
            return {
                'symbol': tick.get('symbol', '').upper(),
                'bid': self._to_decimal(tick.get('bid')),
                'ask': self._to_decimal(tick.get('ask')),
                'last': self._to_decimal(tick.get('last')),
                'volume': self._to_decimal(tick.get('volume', 0)),
                'timestamp': self._parse_timestamp(tick.get('time', 0)),
                'exchange': 'MT5',
                'data_quality': DataQuality.REAL_TIME
            }
        else:
            # Generic normalization
            return {
                'symbol': str(tick.get('symbol', '')).upper(),
                'bid': self._to_decimal(tick.get('bid')),
                'ask': self._to_decimal(tick.get('ask')),
                'last': self._to_decimal(tick.get('last') or tick.get('price')),
                'volume': self._to_decimal(tick.get('volume', 0)),
                'timestamp': self._parse_timestamp(tick.get('timestamp') or tick.get('time')),
                'exchange': str(tick.get('exchange', '')),
                'data_quality': DataQuality(tick.get('data_quality', DataQuality.REAL_TIME))
            }
    
    async def _normalize_ohlcv(
        self, 
        ohlcv: Dict[str, Any], 
        source: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Normalize OHLCV data."""
        timeframe = kwargs.get('timeframe', '1m')
        
        if source == 'mt5':
            return {
                'symbol': ohlcv.get('symbol', '').upper(),
                'open': self._to_decimal(ohlcv.get('open')),
                'high': self._to_decimal(ohlcv.get('high')),
                'low': self._to_decimal(ohlcv.get('low')),
                'close': self._to_decimal(ohlcv.get('close')),
                'volume': self._to_decimal(ohlcv.get('real_volume') or ohlcv.get('volume', 0)),
                'trades': int(ohlcv.get('trades', 0)),
                'timeframe': timeframe,
                'timestamp': self._parse_timestamp(ohlcv.get('time', 0)),
                'exchange': 'MT5',
                'data_quality': DataQuality.REAL_TIME
            }
        else:
            return {
                'symbol': str(ohlcv.get('symbol', '')).upper(),
                'open': self._to_decimal(ohlcv.get('open')),
                'high': self._to_decimal(ohlcv.get('high')),
                'low': self._to_decimal(ohlcv.get('low')),
                'close': self._to_decimal(ohlcv.get('close')),
                'volume': self._to_decimal(ohlcv.get('volume', 0)),
                'trades': int(ohlcv.get('trades', 0)),
                'timeframe': str(ohlcv.get('timeframe', timeframe)),
                'timestamp': self._parse_timestamp(ohlcv.get('timestamp') or ohlcv.get('time')),
                'exchange': str(ohlcv.get('exchange', '')),
                'data_quality': DataQuality(ohlcv.get('data_quality', DataQuality.REAL_TIME))
            }
    
    async def _normalize_order_book(
        self, 
        book: Dict[str, Any], 
        source: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Normalize order book data."""
        if source == 'mt5':
            bids = [
                {
                    'price': self._to_decimal(level.get('price')),
                    'volume': self._to_decimal(level.get('volume', 0)),
                    'orders': int(level.get('orders', 1))
                }
                for level in book.get('bids', [])
            ]
            
            asks = [
                {
                    'price': self._to_decimal(level.get('price')),
                    'volume': self._to_decimal(level.get('volume', 0)),
                    'orders': int(level.get('orders', 1))
                }
                for level in book.get('asks', [])
            ]
            
            return {
                'symbol': book.get('symbol', '').upper(),
                'bids': sorted(bids, key=lambda x: x['price'], reverse=True),
                'asks': sorted(asks, key=lambda x: x['price']),
                'timestamp': self._parse_timestamp(book.get('time', 0)),
                'exchange': 'MT5',
                'data_quality': DataQuality.REAL_TIME
            }
        else:
            return {
                'symbol': str(book.get('symbol', '')).upper(),
                'bids': [
                    {
                        'price': self._to_decimal(level.get('price')),
                        'volume': self._to_decimal(level.get('volume', 0)),
                        'orders': int(level.get('orders', 1))
                    }
                    for level in book.get('bids', [])
                ],
                'asks': [
                    {
                        'price': self._to_decimal(level.get('price')),
                        'volume': self._to_decimal(level.get('volume', 0)),
                        'orders': int(level.get('orders', 1))
                    }
                    for level in book.get('asks', [])
                ],
                'timestamp': self._parse_timestamp(book.get('timestamp') or book.get('time')),
                'exchange': str(book.get('exchange', '')),
                'data_quality': DataQuality(book.get('data_quality', DataQuality.REAL_TIME))
            }
    
    async def _normalize_trade(
        self, 
        trade: Dict[str, Any], 
        source: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Normalize trade data."""
        if source == 'mt5':
            return {
                'trade_id': str(trade.get('ticket', '')),
                'symbol': trade.get('symbol', '').upper(),
                'price': self._to_decimal(trade.get('price')),
                'volume': self._to_decimal(trade.get('volume', 0)),
                'side': 'buy' if trade.get('type') in [0, 4] else 'sell',
                'timestamp': self._parse_timestamp(trade.get('time', 0)),
                'exchange': 'MT5',
                'data_quality': DataQuality.REAL_TIME
            }
        else:
            return {
                'trade_id': str(trade.get('trade_id') or trade.get('id', '')),
                'symbol': str(trade.get('symbol', '')).upper(),
                'price': self._to_decimal(trade.get('price')),
                'volume': self._to_decimal(trade.get('volume', 0)),
                'side': str(trade.get('side', '')).lower(),
                'timestamp': self._parse_timestamp(trade.get('timestamp') or trade.get('time')),
                'exchange': str(trade.get('exchange', '')),
                'data_quality': DataQuality(trade.get('data_quality', DataQuality.REAL_TIME))
            }
    
    async def _normalize_funding_rate(
        self, 
        rate: Dict[str, Any], 
        source: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Normalize funding rate data."""
        return {
            'symbol': str(rate.get('symbol', '')).upper(),
            'funding_rate': self._to_decimal(rate.get('funding_rate') or rate.get('rate')),
            'funding_time': self._parse_timestamp(rate.get('funding_time') or rate.get('time')),
            'next_funding_time': self._parse_timestamp(rate.get('next_funding_time') or rate.get('next_time')),
            'exchange': str(rate.get('exchange', '')),
            'data_quality': DataQuality(rate.get('data_quality', DataQuality.REAL_TIME))
        }
    
    async def _normalize_open_interest(
        self, 
        oi: Dict[str, Any], 
        source: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Normalize open interest data."""
        return {
            'symbol': str(oi.get('symbol', '')).upper(),
            'open_interest': self._to_decimal(oi.get('open_interest') or oi.get('oi')),
            'timestamp': self._parse_timestamp(oi.get('timestamp') or oi.get('time')),
            'exchange': str(oi.get('exchange', '')),
            'data_quality': DataQuality(oi.get('data_quality', DataQuality.REAL_TIME))
        }
    
    async def _normalize_liquidation(
        self, 
        liq: Dict[str, Any], 
        source: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Normalize liquidation data."""
        return {
            'symbol': str(liq.get('symbol', '')).upper(),
            'price': self._to_decimal(liq.get('price')),
            'quantity': self._to_decimal(liq.get('quantity') or liq.get('qty')),
            'side': str(liq.get('side', '')).lower(),
            'timestamp': self._parse_timestamp(liq.get('timestamp') or liq.get('time')),
            'exchange': str(liq.get('exchange', '')),
            'data_quality': DataQuality(liq.get('data_quality', DataQuality.REAL_TIME))
        }
    
    def _to_decimal(self, value: Any, precision: int = 18) -> Decimal:
        """Convert a value to Decimal with specified precision."""
        if value is None:
            return Decimal('0')
            
        try:
            if isinstance(value, (str, float, int)):
                return Decimal(str(value)).quantize(
                    Decimal(f"1e-{precision}"),
                    rounding=ROUND_HALF_UP
                )
            elif isinstance(value, Decimal):
                return value.quantize(
                    Decimal(f"1e-{precision}"),
                    rounding=ROUND_HALF_UP
                )
            else:
                return Decimal('0')
        except (ValueError, TypeError, DecimalException):
            return Decimal('0')
    
    def _parse_timestamp(self, timestamp: Any) -> datetime:
        """Parse a timestamp from various formats."""
        if timestamp is None:
            return datetime.now(timezone.utc)
            
        if isinstance(timestamp, (int, float)):
            # Handle Unix timestamps (seconds or milliseconds)
            if timestamp > 1e12:  # Likely milliseconds
                timestamp = timestamp / 1000
            return datetime.fromtimestamp(timestamp, timezone.utc)
        elif isinstance(timestamp, str):
            # Try to parse ISO format
            try:
                return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                pass
                
            # Try other common formats
            for fmt in [
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S',
                '%Y-%m-%dT%H:%M:%S%z',
                '%Y-%m-%d %H:%M:%S.%f',
                '%Y-%m-%dT%H:%M:%S.%f',
                '%Y-%m-%dT%H:%M:%S.%f%z'
            ]:
                try:
                    return datetime.strptime(timestamp, fmt).astimezone(timezone.utc)
                except (ValueError, TypeError):
                    continue
        
        # Default to current time if parsing fails
        return datetime.now(timezone.utc)

class DataValidator:
    """Validates market data against defined schemas and rules."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the data validator."""
        self.config = config or {}
        self._schemas: Dict[DataSchemaType, Type[BaseModel]] = {
            DataSchemaType.TICK: self._create_tick_schema(),
            DataSchemaType.OHLCV: self._create_ohlcv_schema(),
            DataSchemaType.ORDER_BOOK: self._create_order_book_schema(),
            DataSchemaType.TRADE: self._create_trade_schema(),
            DataSchemaType.FUNDING_RATE: self._create_funding_rate_schema(),
            DataSchemaType.OPEN_INTEREST: self._create_open_interest_schema(),
            DataSchemaType.LIQUIDATION: self._create_liquidation_schema()
        }
    
    async def validate(
        self, 
        data: Dict[str, Any], 
        schema_type: DataSchemaType,
        **kwargs
    ) -> Tuple[bool, List[str]]:
        """Validate data against the specified schema."""
        if schema_type not in self._schemas:
            return False, [f"Unknown schema type: {schema_type}"]
        
        try:
            schema = self._schemas[schema_type]
            validated = schema(**data)
            return True, []
        except Exception as e:
            return False, [str(e)]
    
    def _create_tick_schema(self) -> Type[BaseModel]:
        """Create a Pydantic model for tick data validation."""
        class TickSchema(BaseModel):
            symbol: str
            bid: Decimal
            ask: Decimal
            last: Decimal
            volume: Decimal
            timestamp: datetime
            exchange: str = ""
            data_quality: DataQuality = DataQuality.REAL_TIME
            
            class Config:
                arbitrary_types_allowed = True
                json_encoders = {
                    Decimal: lambda v: str(v),
                    datetime: lambda v: v.isoformat(),
                }
                
            @validator('symbol')
            def validate_symbol(cls, v):
                if not v or not isinstance(v, str):
                    raise ValueError("Symbol must be a non-empty string")
                return v.upper()
                
            @validator('bid', 'ask', 'last', 'volume', pre=True)
            def validate_decimals(cls, v):
                if not isinstance(v, (Decimal, int, float, str)):
                    raise ValueError("Numeric fields must be numbers or strings")
                try:
                    return Decimal(str(v))
                except Exception as e:
                    raise ValueError(f"Invalid decimal value: {v}") from e
        
        return TickSchema
    
    def _create_ohlcv_schema(self) -> Type[BaseModel]:
        """Create a Pydantic model for OHLCV data validation."""
        class OHLCVSchema(BaseModel):
            symbol: str
            open: Decimal
            high: Decimal
            low: Decimal
            close: Decimal
            volume: Decimal
            trades: int = 0
            timeframe: str = "1m"
            timestamp: datetime
            exchange: str = ""
            data_quality: DataQuality = DataQuality.REAL_TIME
            
            class Config:
                arbitrary_types_allowed = True
                json_encoders = {
                    Decimal: lambda v: str(v),
                    datetime: lambda v: v.isoformat(),
                }
                
            @validator('timeframe')
            def validate_timeframe(cls, v):
                if not re.match(r'^\d+[smhdwM]$', str(v)):
                    raise ValueError("Invalid timeframe format. Use format: 1m, 5m, 1h, 1d, 1w, 1M")
                return v
                
            @validator('open', 'high', 'low', 'close', 'volume', pre=True)
            def validate_decimals(cls, v):
                if not isinstance(v, (Decimal, int, float, str)):
                    raise ValueError("Numeric fields must be numbers or strings")
                try:
                    return Decimal(str(v))
                except Exception as e:
                    raise ValueError(f"Invalid decimal value: {v}") from e
        
        return OHLCVSchema
    
    def _create_order_book_schema(self) -> Type[BaseModel]:
        """Create a Pydantic model for order book validation."""
        class OrderBookLevel(BaseModel):
            price: Decimal
            volume: Decimal
            orders: int = 1
            
            class Config:
                arbitrary_types_allowed = True
                json_encoders = {
                    Decimal: lambda v: str(v),
                }
        
        class OrderBookSchema(BaseModel):
            symbol: str
            bids: List[OrderBookLevel]
            asks: List[OrderBookLevel]
            timestamp: datetime
            exchange: str = ""
            data_quality: DataQuality = DataQuality.REAL_TIME
            
            class Config:
                arbitrary_types_allowed = True
                json_encoders = {
                    Decimal: lambda v: str(v),
                    datetime: lambda v: v.isoformat(),
                }
            
            @validator('bids', 'asks')
            def validate_levels(cls, levels):
                if not levels:
                    return levels
                    
                # Check for crossed book
                if levels and hasattr(levels[0], 'price'):
                    if levels[0].__class__.__name__ == 'OrderBookLevel' and levels[0].price <= 0:
                        raise ValueError("Price must be positive")
                return levels
            
            @root_validator
            def validate_book(cls, values):
                bids = values.get('bids', [])
                asks = values.get('asks', [])
                
                if bids and asks:
                    best_bid = max(level.price for level in bids)
                    best_ask = min(level.price for level in asks)
                    
                    if best_bid >= best_ask:
                        raise ValueError("Crossed order book detected")
                
                return values
        
        return OrderBookSchema
    
    def _create_trade_schema(self) -> Type[BaseModel]:
        """Create a Pydantic model for trade data validation."""
        class TradeSchema(BaseModel):
            trade_id: str
            symbol: str
            price: Decimal
            volume: Decimal
            side: str
            timestamp: datetime
            exchange: str = ""
            data_quality: DataQuality = DataQuality.REAL_TIME
            
            class Config:
                arbitrary_types_allowed = True
                json_encoders = {
                    Decimal: lambda v: str(v),
                    datetime: lambda v: v.isoformat(),
                }
            
            @validator('side')
            def validate_side(cls, v):
                if v.lower() not in ['buy', 'sell']:
                    raise ValueError("Side must be either 'buy' or 'sell'")
                return v.lower()
    
    def _create_funding_rate_schema(self) -> Type[BaseModel]:
        """Create a Pydantic model for funding rate validation."""
        class FundingRateSchema(BaseModel):
            symbol: str
            funding_rate: Decimal
            funding_time: datetime
            next_funding_time: datetime
            exchange: str = ""
            data_quality: DataQuality = DataQuality.REAL_TIME
            
            class Config:
                arbitrary_types_allowed = True
                json_encoders = {
                    Decimal: lambda v: str(v),
                    datetime: lambda v: v.isoformat(),
                }
    
    def _create_open_interest_schema(self) -> Type[BaseModel]:
        """Create a Pydantic model for open interest validation."""
        class OpenInterestSchema(BaseModel):
            symbol: str
            open_interest: Decimal
            timestamp: datetime
            exchange: str = ""
            data_quality: DataQuality = DataQuality.REAL_TIME
            
            class Config:
                arbitrary_types_allowed = True
                json_encoders = {
                    Decimal: lambda v: str(v),
                    datetime: lambda v: v.isoformat(),
                }
    
    def _create_liquidation_schema(self) -> Type[BaseModel]:
        """Create a Pydantic model for liquidation validation."""
        class LiquidationSchema(BaseModel):
            symbol: str
            price: Decimal
            quantity: Decimal
            side: str
            timestamp: datetime
            exchange: str = ""
            data_quality: DataQuality = DataQuality.REAL_TIME
            
            class Config:
                arbitrary_types_allowed = True
                json_encoders = {
                    Decimal: lambda v: str(v),
                    datetime: lambda v: v.isoformat(),
                }
            
            @validator('side')
            def validate_side(cls, v):
                if v.lower() not in ['buy', 'sell']:
                    raise ValueError("Side must be either 'buy' or 'sell'")
                return v.lower()

class DataQualityChecker:
    """Checks and enforces data quality rules."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize the data quality checker."""
        self.config = config or {}
        self._rules: Dict[str, List[Callable]] = {}
    
    async def check_quality(
        self, 
        data: Dict[str, Any], 
        schema_type: DataSchemaType,
        **kwargs
    ) -> Dict[str, Any]:
        """Check data quality and return a quality report."""
        report = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'metrics': {},
            'timestamp': datetime.now(timezone.utc),
            'schema_type': schema_type.value,
            'data_quality': DataQuality.REAL_TIME
        }
        
        # Apply quality checks
        await self._check_freshness(data, report, **kwargs)
        await self._check_completeness(data, report, **kwargs)
        await self._check_anomalies(data, report, **kwargs)
        
        # Update overall validity
        report['is_valid'] = len(report['errors']) == 0
        
        # Set data quality level
        if not report['is_valid']:
            report['data_quality'] = DataQuality.INCOMPLETE
        elif report['warnings']:
            report['data_quality'] = DataQuality.ESTIMATED
        
        return report
    
    async def _check_freshness(
        self, 
        data: Dict[str, Any], 
        report: Dict[str, Any],
        **kwargs
    ) -> None:
        """Check if the data is fresh."""
        max_age = kwargs.get('max_age_seconds', 60)
        
        if 'timestamp' in data and data['timestamp']:
            age = (datetime.now(timezone.utc) - data['timestamp']).total_seconds()
            report['metrics']['age_seconds'] = age
            
            if age > max_age:
                report['warnings'].append(f"Data is {age:.1f} seconds old (max: {max_age}s)")
                report['data_quality'] = DataQuality.STALE
    
    async def _check_completeness(
        self, 
        data: Dict[str, Any], 
        report: Dict[str, Any],
        **kwargs
    ) -> None:
        """Check if all required fields are present and valid."""
        required_fields = {
            'symbol': (str,),
            'timestamp': (datetime, int, float),
        }
        
        for field, types in required_fields.items():
            if field not in data or data[field] is None:
                report['errors'].append(f"Missing required field: {field}")
            elif not isinstance(data[field], types):
                report['errors'].append(
                    f"Invalid type for {field}: {type(data[field])}, expected {types}"
                )
    
    async def _check_anomalies(
        self, 
        data: Dict[str, Any], 
        report: Dict[str, Any],
        **kwargs
    ) -> None:
        """Check for data anomalies."""
        # Price anomalies
        for price_field in ['price', 'open', 'high', 'low', 'close', 'bid', 'ask']:
            if price_field in data and data[price_field] is not None:
                price = float(data[price_field])
                if price <= 0:
                    report['errors'].append(f"Invalid {price_field}: {price} (must be positive)")
        
        # Volume anomalies
        if 'volume' in data and data['volume'] is not None:
            volume = float(data['volume'])
            if volume < 0:
                report['errors'].append(f"Invalid volume: {volume} (must be non-negative)")
        
        # OHLC consistency
        if all(f in data for f in ['open', 'high', 'low', 'close']):
            o, h, l, c = data['open'], data['high'], data['low'], data['close']
            if h < l:
                report['errors'].append("High price is less than low price")
            if o > h or o < l:
                report['warnings'].append("Open price is outside high/low range")
            if c > h or c < l:
                report['warnings'].append("Close price is outside high/low range")
