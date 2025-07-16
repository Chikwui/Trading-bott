"""
Instrument factory for creating instrument instances.
"""
from typing import Dict, Any, Optional, Type, TypeVar, List
from datetime import datetime, time
import logging

from .metadata import (
    InstrumentMetadata,
    AssetClass,
    InstrumentType,
    TradingHours
)

logger = logging.getLogger(__name__)
T = TypeVar('T', bound='InstrumentFactory')


class InstrumentFactory:
    """Factory for creating instrument instances."""
    
    # Default trading hours for different asset classes
    DEFAULT_TRADING_HOURS = {
        AssetClass.FOREX: {
            'open_time': time(0, 0),
            'close_time': time(23, 59, 59),
            'timezone': 'UTC',
            'days': list(range(7)),  # Monday (0) to Sunday (6)
            'is_24h': True
        },
        AssetClass.CRYPTO: {
            'open_time': time(0, 0),
            'close_time': time(23, 59, 59),
            'timezone': 'UTC',
            'days': list(range(7)),  # 24/7
            'is_24h': True
        },
        AssetClass.COMMODITY: {
            'open_time': time(0, 0),
            'close_time': time(23, 59, 59),
            'timezone': 'UTC',
            'days': list(range(5)),  # Monday to Friday
            'is_24h': True
        },
        AssetClass.INDEX: {
            'open_time': time(0, 0),
            'close_time': time(23, 59, 59),
            'timezone': 'UTC',
            'days': list(range(5)),  # Monday to Friday
            'is_24h': True
        },
        AssetClass.STOCK: {
            'open_time': time(9, 30),  # 9:30 AM
            'close_time': time(16, 0),  # 4:00 PM
            'timezone': 'America/New_York',
            'days': list(range(5)),  # Monday to Friday
            'is_24h': False
        }
    }
    
    # Default instrument configurations by asset class and type
    DEFAULTS = {
        AssetClass.FOREX: {
            'tick_size': 0.00001,
            'tick_value': 1.0,
            'min_lot_size': 0.01,
            'max_lot_size': 100.0,
            'lot_step': 0.01,
            'leverage': 30.0,
            'margin_required': 0.01,  # 1% margin
            'tags': ['forex', 'fx']
        },
        AssetClass.CRYPTO: {
            'tick_size': 0.01,
            'tick_value': 1.0,
            'min_lot_size': 0.0001,
            'max_lot_size': 1000.0,
            'lot_step': 0.0001,
            'leverage': 10.0,
            'margin_required': 0.1,  # 10% margin
            'tags': ['crypto', 'digital_assets']
        },
        AssetClass.COMMODITY: {
            'tick_size': 0.01,
            'tick_value': 1.0,
            'min_lot_size': 0.1,
            'max_lot_size': 100.0,
            'lot_step': 0.1,
            'leverage': 20.0,
            'margin_required': 0.05,  # 5% margin
            'tags': ['commodities']
        },
        AssetClass.INDEX: {
            'tick_size': 0.1,
            'tick_value': 1.0,
            'min_lot_size': 0.01,
            'max_lot_size': 100.0,
            'lot_step': 0.01,
            'leverage': 20.0,
            'margin_required': 0.05,  # 5% margin
            'tags': ['indices', 'index']
        },
        AssetClass.STOCK: {
            'tick_size': 0.01,
            'tick_value': 1.0,
            'min_lot_size': 1.0,
            'max_lot_size': 10000.0,
            'lot_step': 1.0,
            'leverage': 4.0,  # Typical pattern day trading leverage
            'margin_required': 0.25,  # 25% margin
            'tags': ['stocks', 'equities']
        }
    }
    
    @classmethod
    def create_instrument(
        cls,
        symbol: str,
        name: str,
        asset_class: AssetClass,
        instrument_type: Optional[InstrumentType] = None,
        base_currency: Optional[str] = None,
        quote_currency: Optional[str] = None,
        exchange: Optional[str] = None,
        **kwargs
    ) -> InstrumentMetadata:
        """
        Create a new instrument with default values based on asset class.
        
        Args:
            symbol: Instrument symbol (e.g., 'EURUSD', 'BTCUSDT')
            name: Full name of the instrument
            asset_class: Asset class (e.g., FOREX, CRYPTO, COMMODITY)
            instrument_type: Type of instrument (e.g., MAJOR, MINOR, COIN, TOKEN)
            base_currency: Base currency (e.g., 'EUR', 'BTC')
            quote_currency: Quote currency (e.g., 'USD', 'USDT')
            exchange: Exchange where the instrument is traded
            **kwargs: Additional instrument parameters to override defaults
            
        Returns:
            InstrumentMetadata: A new instrument instance
        """
        # Get defaults for the asset class
        defaults = cls.DEFAULTS.get(asset_class, {}).copy()
        
        # Get trading hours for the asset class
        trading_hours_data = cls.DEFAULT_TRADING_HOURS.get(asset_class, {})
        trading_hours = TradingHours(**trading_hours_data)
        
        # Extract tags from defaults
        tags = set(defaults.pop('tags', []))
        
        # Add additional tags from kwargs
        if 'tags' in kwargs:
            tags.update(kwargs.pop('tags'))
        
        # Add asset class and type as tags
        tags.add(asset_class.value.lower())
        if instrument_type:
            tags.add(instrument_type.value.lower())
        
        # Create instrument with defaults and overrides
        instrument = InstrumentMetadata(
            symbol=symbol.upper(),
            name=name,
            asset_class=asset_class,
            instrument_type=instrument_type,
            base_currency=base_currency,
            quote_currency=quote_currency,
            exchange=exchange,
            trading_hours=trading_hours,
            tags=tags,
            **{**defaults, **kwargs}
        )
        
        return instrument
    
    @classmethod
    def create_crypto_pair(
        cls,
        base_currency: str,
        quote_currency: str,
        leverage: float = 10.0,
        **kwargs
    ) -> InstrumentMetadata:
        """
        Create a cryptocurrency trading pair.
        
        Args:
            base_currency: The base cryptocurrency (e.g., 'BTC')
            quote_currency: The quote currency (e.g., 'USDT')
            leverage: The leverage to use (default: 10.0)
            **kwargs: Additional arguments to pass to InstrumentMetadata
            
        Returns:
            InstrumentMetadata: The created crypto pair
        """
        symbol = f"{base_currency}{quote_currency}"
        
        # Determine instrument type based on the currencies
        instrument_type = InstrumentType.COIN
        if quote_currency in ['USDT', 'USDC', 'DAI', 'BUSD']:
            instrument_type = InstrumentType.STABLECOIN
        
        # Default trading hours for crypto (24/7)
        trading_hours = TradingHours(
            open_time=time(0, 0),
            close_time=time(23, 59, 59),
            timezone='UTC',
            days=list(range(7)),  # 7 days a week
            is_24h=True
        )
        
        # Create the instrument
        return InstrumentMetadata(
            symbol=symbol,
            name=f"{base_currency}/{quote_currency}",
            asset_class=AssetClass.CRYPTO,
            instrument_type=instrument_type,
            base_currency=base_currency,
            quote_currency=quote_currency,
            exchange=kwargs.pop('exchange', 'Binance'),
            lot_size=1.0,
            min_lot_size=0.0001,
            max_lot_size=1000.0,
            lot_step=0.0001,
            tick_size=0.01,
            tick_value=1.0,
            margin_required=1.0 / leverage if leverage > 0 else 0.0,
            leverage=leverage,
            trading_hours=trading_hours,
            tags={"crypto", f"crypto_{instrument_type.value.lower()}"},
            is_active=True,
            **kwargs
        )
    
    @classmethod
    def create_forex_pair(
        cls,
        base_currency: str,
        quote_currency: str,
        leverage: float = 30.0,
        instrument_type: Optional[InstrumentType] = None,
        **kwargs
    ) -> InstrumentMetadata:
        """
        Create a forex currency pair instrument.
        
        Args:
            base_currency: Base currency (e.g., 'EUR', 'GBP')
            quote_currency: Quote currency (e.g., 'USD', 'JPY')
            leverage: The leverage to use (default: 30.0)
            instrument_type: Type of forex pair (MAJOR, MINOR, EXOTIC, CROSS)
            **kwargs: Additional instrument parameters
            
        Returns:
            InstrumentMetadata: A new forex pair instrument
        """
        # Determine instrument type if not provided
        if instrument_type is None:
            major_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'USDCAD', 'AUDUSD', 'NZDUSD']
            symbol = f"{base_currency}{quote_currency}"
            
            if symbol in major_pairs:
                instrument_type = InstrumentType.MAJOR
            elif 'USD' in symbol and len(symbol) == 6:  # Minor pairs include USD
                instrument_type = InstrumentType.MINOR
            else:
                instrument_type = InstrumentType.CROSS
        
        # Add forex-specific tags
        tags = kwargs.pop('tags', set())
        if isinstance(tags, (list, set)):
            tags = set(tags)
        else:
            tags = set()
        
        tags.update(['forex', 'fx', f'fx_{instrument_type.value.lower()}'])
        
        # Create instrument
        return cls.create_instrument(
            symbol=f"{base_currency}{quote_currency}",
            name=f"{base_currency}/{quote_currency}",
            asset_class=AssetClass.FOREX,
            instrument_type=instrument_type,
            base_currency=base_currency,
            quote_currency=quote_currency,
            leverage=leverage,
            tags=tags,
            **kwargs
        )
    
    @classmethod
    def create_crypto_pair(
        cls,
        base_currency: str,
        quote_currency: str = 'USDT',
        leverage: float = 10.0,
        instrument_type: Optional[InstrumentType] = None,
        **kwargs
    ) -> InstrumentMetadata:
        """
        Create a cryptocurrency trading pair instrument.
        
        Args:
            base_currency: Base cryptocurrency (e.g., 'BTC', 'ETH')
            quote_currency: Quote currency (default: 'USDT')
            instrument_type: Type of crypto instrument (COIN, TOKEN, STABLECOIN)
            **kwargs: Additional instrument parameters
            
        Returns:
            InstrumentMetadata: A new crypto pair instrument
        """
        # Determine instrument type if not provided
        if instrument_type is None:
            stablecoins = ['USDT', 'USDC', 'DAI', 'BUSD', 'TUSD', 'PAX', 'USDP']
            
            if base_currency in stablecoins or quote_currency in stablecoins:
                instrument_type = InstrumentType.STABLECOIN
            else:
                instrument_type = InstrumentType.COIN  # Default to COIN
        
        # Add crypto-specific tags
        tags = kwargs.pop('tags', set())
        if isinstance(tags, (list, set)):
            tags = set(tags)
        else:
            tags = set()
        
        tags.update(['crypto', 'digital_assets', f'crypto_{instrument_type.value.lower()}'])
        
        # Create instrument
        return cls.create_instrument(
            symbol=f"{base_currency}{quote_currency}",
            name=f"{base_currency}/{quote_currency}",
            asset_class=AssetClass.CRYPTO,
            instrument_type=instrument_type,
            base_currency=base_currency,
            quote_currency=quote_currency,
            tags=tags,
            **kwargs
        )
    
    @classmethod
    def create_commodity(
        cls,
        symbol: str,
        name: str,
        commodity_type: InstrumentType = InstrumentType.METAL,
        **kwargs
    ) -> InstrumentMetadata:
        """
        Create a commodity instrument.
        
        Args:
            symbol: Commodity symbol (e.g., 'XAUUSD', 'CL')
            name: Full name of the commodity
            commodity_type: Type of commodity (METAL, ENERGY, AGRICULTURAL)
            **kwargs: Additional instrument parameters
            
        Returns:
            InstrumentMetadata: A new commodity instrument
        """
        # Add commodity-specific tags
        tags = kwargs.pop('tags', set())
        if isinstance(tags, (list, set)):
            tags = set(tags)
        else:
            tags = set()
        
        tags.update(['commodity', f'commodity_{commodity_type.value.lower()}'])
        
        # Create instrument
        return cls.create_instrument(
            symbol=symbol,
            name=name,
            asset_class=AssetClass.COMMODITY,
            instrument_type=commodity_type,
            tags=tags,
            **kwargs
        )
    
    @classmethod
    def create_index(
        cls,
        symbol: str,
        name: str,
        **kwargs
    ) -> InstrumentMetadata:
        """
        Create a stock index instrument.
        
        Args:
            symbol: Index symbol (e.g., 'SPX500', 'US30')
            name: Full name of the index
            **kwargs: Additional instrument parameters
            
        Returns:
            InstrumentMetadata: A new index instrument
        """
        # Add index-specific tags
        tags = kwargs.pop('tags', set())
        if isinstance(tags, (list, set)):
            tags = set(tags)
        else:
            tags = set()
        
        tags.update(['index', 'indices'])
        
        # Create instrument
        return cls.create_instrument(
            symbol=symbol,
            name=name,
            asset_class=AssetClass.INDEX,
            instrument_type=InstrumentType.INDEX,
            tags=tags,
            **kwargs
        )
    
    @classmethod
    def create_stock(
        cls,
        symbol: str,
        name: str,
        exchange: str,
        **kwargs
    ) -> InstrumentMetadata:
        """
        Create a stock instrument.
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'MSFT')
            name: Full company name
            exchange: Exchange where the stock is listed (e.g., 'NASDAQ', 'NYSE')
            **kwargs: Additional instrument parameters
            
        Returns:
            InstrumentMetadata: A new stock instrument
        """
        # Add stock-specific tags
        tags = kwargs.pop('tags', set())
        if isinstance(tags, (list, set)):
            tags = set(tags)
        else:
            tags = set()
        
        tags.update(['stock', 'equity', f'exchange_{exchange.lower()}'])
        
        # Create instrument
        return cls.create_instrument(
            symbol=symbol,
            name=name,
            asset_class=AssetClass.STOCK,
            instrument_type=InstrumentType.STOCK,
            exchange=exchange,
            tags=tags,
            **kwargs
        )
