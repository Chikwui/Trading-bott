"""
Instrument metadata and management system.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Any, Type, TypeVar, ClassVar
from datetime import time, datetime, date
import logging
from pathlib import Path
import json
import yaml

# Configure logging
logger = logging.getLogger(__name__)

T = TypeVar('T', bound='InstrumentMetadata')


class AssetClass(str, Enum):
    """Asset class enumeration."""
    FOREX = "forex"
    CRYPTO = "crypto"
    COMMODITY = "commodity"
    INDEX = "index"
    STOCK = "stock"
    ETF = "etf"
    FUTURE = "future"
    OPTION = "option"
    BOND = "bond"


class InstrumentType(str, Enum):
    """Instrument type enumeration."""
    SPOT = "spot"  # Spot trading instruments
    # Forex types
    MAJOR = "major"
    MINOR = "minor"
    EXOTIC = "exotic"
    CROSS = "cross"
    
    # Crypto types
    COIN = "coin"
    TOKEN = "token"
    STABLECOIN = "stablecoin"
    
    # Commodity types
    METAL = "metal"
    ENERGY = "energy"
    AGRICULTURAL = "agricultural"
    
    # Other types
    INDEX = "index"
    FUTURE = "future"
    OPTION = "option"
    ETF = "etf"
    STOCK = "stock"
    BOND = "bond"


class TradingHours:
    """Trading hours for an instrument with calendar system integration."""
    
    def __init__(
        self,
        open_time: time = time(0, 0),
        close_time: time = time(23, 59, 59, 999999),
        timezone: str = "UTC",
        days: List[int] = None,
        is_24h: bool = False,
        asset_class: Optional[str] = None,
        exchange: Optional[str] = None
    ):
        """Initialize trading hours.
        
        Args:
            open_time: Market open time
            close_time: Market close time
            timezone: Timezone for the trading hours
            days: List of weekdays (0=Monday, 6=Sunday) when market is open
            is_24h: Whether the market is open 24/7
            asset_class: Asset class for auto-selecting calendar
            exchange: Exchange name for auto-selecting calendar
        """
        self.open_time = open_time
        self.close_time = close_time
        self.timezone = timezone
        self.days = days or list(range(7))  # 0=Monday, 6=Sunday
        self.is_24h = is_24h
        self._asset_class = asset_class
        self._exchange = exchange
        self._calendar = None
    
    @property
    def calendar(self):
        """Get or create the market calendar for these trading hours."""
        if self._calendar is None:
            from core.calendar import CalendarFactory
            if self._asset_class:
                # Create appropriate calendar based on asset class
                self._calendar = CalendarFactory.get_calendar(
                    asset_class=self._asset_class,
                    timezone=self.timezone,
                    exchange=self._exchange
                )
        return self._calendar
    
    def is_market_open(self, dt: Optional[datetime] = None) -> bool:
        """Check if market is currently open.
        
        Uses the calendar system if available, otherwise falls back to basic time checks.
        """
        if self.is_24h:
            return True
            
        dt = dt or datetime.utcnow()
        
        try:
            # Use the calendar system if available
            if self.calendar:
                return self.calendar.is_session_open(dt)
        except Exception as e:
            logger.warning(f"Error checking market open with calendar: {e}")
            
        # Fall back to basic time check
        current_time = dt.time()
        return (
            dt.weekday() in self.days and
            self.open_time <= current_time <= self.close_time
        )
    
    def time_to_next_session(self, dt: Optional[datetime] = None) -> float:
        """Time in seconds until next trading session starts."""
        dt = dt or datetime.utcnow()
        
        if self.is_24h:
            return 0.0
            
        try:
            # Use the calendar system if available
            if self.calendar:
                if self.calendar.is_session_open(dt):
                    return 0.0
                    
                next_session = self.calendar.next_trading_day(dt)
                next_session = datetime.combine(next_session, self.open_time)
                return (next_session - dt).total_seconds()
        except Exception as e:
            logger.warning(f"Error calculating next session with calendar: {e}")
            
        # Fall back to basic calculation
        if self.is_market_open(dt):
            return 0.0
            
        # Find next trading day
        next_day = dt
        while True:
            next_day = datetime.combine(
                next_day.date() + timedelta(days=1),
                time(0, 0)
            )
            if next_day.weekday() in self.days:
                break
        
        # Calculate time until next session
        next_session = datetime.combine(next_day.date(), self.open_time)
        return (next_session - dt).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'open_time': self.open_time.isoformat(),
            'close_time': self.close_time.isoformat(),
            'timezone': self.timezone,
            'days': self.days,
            'is_24h': self.is_24h,
            'asset_class': self._asset_class,
            'exchange': self._exchange
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingHours':
        """Create from dictionary."""
        open_time = time.fromisoformat(data['open_time']) if isinstance(data['open_time'], str) else data['open_time']
        close_time = time.fromisoformat(data['close_time']) if isinstance(data['close_time'], str) else data['close_time']
        
        return cls(
            open_time=open_time,
            close_time=close_time,
            timezone=data.get('timezone', 'UTC'),
            days=data.get('days', list(range(7))),
            is_24h=data.get('is_24h', False),
            asset_class=data.get('asset_class'),
            exchange=data.get('exchange')
        )
    
    @classmethod
    def for_asset_class(
        cls,
        asset_class: str,
        exchange: Optional[str] = None,
        timezone: Optional[str] = None
    ) -> 'TradingHours':
        """Create trading hours for a specific asset class.
        
        Args:
            asset_class: The asset class (e.g., 'forex', 'equity', 'crypto')
            exchange: Optional exchange name
            timezone: Optional timezone override
            
        Returns:
            A TradingHours instance configured for the asset class
        """
        # Get the default timezone for the asset class if not specified
        if timezone is None:
            if asset_class.lower() in ['forex', 'fx', 'crypto', 'cryptocurrency']:
                timezone = 'UTC'
            elif asset_class.lower() in ['equity', 'stock', 'stocks']:
                timezone = 'America/New_York'
            elif asset_class.lower() in ['futures', 'commodity', 'commodities']:
                timezone = 'America/Chicago'
            else:
                timezone = 'UTC'
        
        # Configure based on asset class
        if asset_class.lower() in ['forex', 'fx']:
            return cls(
                open_time=time(0, 0),
                close_time=time(23, 59, 59, 999999),
                timezone=timezone,
                days=list(range(7)),  # 24/5 (handled by calendar)
                is_24h=True,
                asset_class=asset_class,
                exchange=exchange
            )
        elif asset_class.lower() in ['crypto', 'cryptocurrency']:
            return cls(
                open_time=time(0, 0),
                close_time=time(23, 59, 59, 999999),
                timezone=timezone,
                days=list(range(7)),  # 24/7
                is_24h=True,
                asset_class=asset_class,
                exchange=exchange
            )
        elif asset_class.lower() in ['equity', 'stock', 'stocks']:
            return cls(
                open_time=time(9, 30),  # 9:30 AM
                close_time=time(16, 0),  # 4:00 PM
                timezone=timezone,
                days=list(range(5)),  # Monday-Friday
                is_24h=False,
                asset_class=asset_class,
                exchange=exchange
            )
        elif asset_class.lower() in ['futures', 'commodity', 'commodities']:
            return cls(
                open_time=time(18, 0),  # 6:00 PM previous day
                close_time=time(17, 0),  # 5:00 PM current day
                timezone=timezone,
                days=list(range(7)),  # Sunday-Friday
                is_24h=False,
                asset_class=asset_class,
                exchange=exchange
            )
        else:
            # Default to 24/5 for unknown asset classes
            return cls(
                open_time=time(0, 0),
                close_time=time(23, 59, 59, 999999),
                timezone=timezone,
                days=list(range(5)),  # Monday-Friday
                is_24h=True,
                asset_class=asset_class,
                exchange=exchange
            )


@dataclass
class InstrumentMetadata:
    """Metadata for a trading instrument with calendar system integration."""
    symbol: str
    name: str
    asset_class: AssetClass
    instrument_type: Optional[InstrumentType] = None
    base_currency: Optional[str] = None
    quote_currency: Optional[str] = None
    exchange: Optional[str] = None
    lot_size: float = 1.0
    min_lot_size: float = 0.01
    max_lot_size: float = 100.0
    lot_step: float = 0.01
    tick_size: float = 0.00001
    tick_value: float = 1.0
    margin_required: float = 0.0
    leverage: float = 1.0
    trading_hours: TradingHours = field(default_factory=TradingHours)
    tags: Set[str] = field(default_factory=set)
    attributes: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    
    # Class variable to store all instrument instances
    _registry: ClassVar[Dict[str, 'InstrumentMetadata']] = {}
    
    def __post_init__(self):
        """Post-initialization setup."""
        # If trading_hours is not initialized with asset class info, update it
        if not hasattr(self.trading_hours, '_asset_class') or not self.trading_hours._asset_class:
            self.trading_hours = TradingHours.for_asset_class(
                asset_class=self.asset_class.value,
                exchange=self.exchange
            )
        
        # Ensure symbol is uppercase
        self.symbol = self.symbol.upper()
        
        # Register the instrument
        self._register()
    
    def _register(self) -> None:
        """Register the instrument in the global registry."""
        self._registry[self.symbol] = self
    
    def add_tag(self, tag: str) -> None:
        """Add a tag to the instrument."""
        self.tags.add(tag.lower())
        self.updated_at = datetime.utcnow()
    
    def remove_tag(self, tag: str) -> None:
        """Remove a tag from the instrument."""
        self.tags.discard(tag.lower())
        self.updated_at = datetime.utcnow()
    
    def has_tag(self, tag: str) -> bool:
        """Check if instrument has a specific tag."""
        return tag.lower() in self.tags
    
    def has_any_tag(self, tags: List[str]) -> bool:
        """Check if instrument has any of the specified tags."""
        tag_set = {t.lower() for t in tags}
        return len(self.tags.intersection(tag_set)) > 0
    
    def has_all_tags(self, tags: List[str]) -> bool:
        """Check if instrument has all of the specified tags."""
        tag_set = {t.lower() for t in tags}
        return self.tags.issuperset(tag_set)
    
    def set_attribute(self, key: str, value: Any) -> None:
        """Set a custom attribute."""
        self.attributes[key] = value
        self.updated_at = datetime.utcnow()
    
    def get_attribute(self, key: str, default: Any = None) -> Any:
        """Get a custom attribute."""
        return self.attributes.get(key, default)
    
    def is_tradable(self) -> bool:
        """Check if the instrument is currently tradable."""
        return self.is_active and self.trading_hours.is_market_open()
    
    def time_to_next_session(self) -> float:
        """Time in seconds until the next trading session starts."""
        return self.trading_hours.time_to_next_session()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert instrument to dictionary."""
        data = asdict(self)
        
        # Handle special types
        data['asset_class'] = self.asset_class.value
        data['instrument_type'] = self.instrument_type.value if self.instrument_type else None
        data['trading_hours'] = self.trading_hours.to_dict()
        data['tags'] = list(self.tags)
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        
        return data
    
    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Create instrument from dictionary."""
        # Handle special conversions
        kwargs = data.copy()
        
        # Convert string enums back to enum instances
        if 'asset_class' in kwargs and isinstance(kwargs['asset_class'], str):
            kwargs['asset_class'] = AssetClass(kwargs['asset_class'])
        
        if 'instrument_type' in kwargs and kwargs['instrument_type'] and isinstance(kwargs['instrument_type'], str):
            kwargs['instrument_type'] = InstrumentType(kwargs['instrument_type'])
        
        # Convert trading hours
        if 'trading_hours' in kwargs and kwargs['trading_hours']:
            kwargs['trading_hours'] = TradingHours.from_dict(kwargs['trading_hours'])
        
        # Convert tags to set
        if 'tags' in kwargs and isinstance(kwargs['tags'], list):
            kwargs['tags'] = set(tag.lower() for tag in kwargs['tags'])
        else:
            kwargs['tags'] = set()
        
        # Convert string timestamps to datetime
        for time_field in ['created_at', 'updated_at']:
            if time_field in kwargs and isinstance(kwargs[time_field], str):
                kwargs[time_field] = datetime.fromisoformat(kwargs[time_field])
        
        return cls(**kwargs)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> Dict[str, 'InstrumentMetadata']:
        """Load instruments from a JSON or YAML file."""
        path = Path(file_path)
        
        if not path.exists():
            logger.warning(f"Instrument file not found: {file_path}")
            return {}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            if path.suffix.lower() == '.json':
                data = json.load(f)
            elif path.suffix.lower() in ('.yaml', '.yml'):
                data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        
        instruments = {}
        for item in data.get('instruments', []):
            try:
                instrument = cls.from_dict(item)
                instruments[instrument.symbol] = instrument
            except Exception as e:
                logger.error(f"Failed to load instrument: {e}", exc_info=True)
        
        return instruments
    
    @classmethod
    def save_to_file(
        cls,
        instruments: Dict[str, 'InstrumentMetadata'],
        file_path: str,
        format: str = 'json'
    ) -> None:
        """Save instruments to a file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'instruments': [inst.to_dict() for inst in instruments.values()],
            'metadata': {
                'generated_at': datetime.utcnow().isoformat(),
                'count': len(instruments)
            }
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            if format.lower() == 'json':
                json.dump(data, f, indent=2, ensure_ascii=False)
            elif format.lower() in ('yaml', 'yml'):
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            else:
                raise ValueError(f"Unsupported format: {format}")
    
    @classmethod
    def get_registry(cls) -> Dict[str, 'InstrumentMetadata']:
        """Get the global instrument registry."""
        return cls._registry
    
    @classmethod
    def clear_registry(cls) -> None:
        """Clear the global instrument registry."""
        cls._registry = {}


class InstrumentManager:
    """Manager for instrument metadata."""
    def __init__(self):
        self.instruments: Dict[str, InstrumentMetadata] = {}
        self._tag_index: Dict[str, Set[str]] = {}  # tag -> set of symbols
        self._asset_class_index: Dict[AssetClass, Set[str]] = {}  # asset_class -> set of symbols
    
    def add_instrument(self, instrument: InstrumentMetadata) -> None:
        """Add or update an instrument."""
        self.instruments[instrument.symbol] = instrument
        
        # Update tag index
        for tag in instrument.tags:
            if tag not in self._tag_index:
                self._tag_index[tag] = set()
            self._tag_index[tag].add(instrument.symbol)
        
        # Update asset class index
        if instrument.asset_class not in self._asset_class_index:
            self._asset_class_index[instrument.asset_class] = set()
        self._asset_class_index[instrument.asset_class].add(instrument.symbol)
    
    def get_instrument(self, symbol: str) -> Optional[InstrumentMetadata]:
        """Get an instrument by symbol."""
        return self.instruments.get(symbol.upper())
    
    def get_instruments_by_tag(self, tag: str) -> List[InstrumentMetadata]:
        """Get all instruments with a specific tag."""
        symbols = self._tag_index.get(tag.lower(), set())
        return [self.instruments[sym] for sym in symbols if sym in self.instruments]
    
    def get_instruments_by_asset_class(
        self,
        asset_class: AssetClass
    ) -> List[InstrumentMetadata]:
        """Get all instruments of a specific asset class."""
        symbols = self._asset_class_index.get(asset_class, set())
        return [self.instruments[sym] for sym in symbols if sym in self.instruments]
    
    def get_all_tags(self) -> Set[str]:
        """Get all unique tags across all instruments."""
        return set(self._tag_index.keys())
    
    def search_instruments(
        self,
        symbol: Optional[str] = None,
        asset_class: Optional[AssetClass] = None,
        tags: Optional[List[str]] = None,
        tradable_only: bool = False
    ) -> List[InstrumentMetadata]:
        """Search for instruments matching the given criteria."""
        results = []
        
        # Filter by symbol if provided
        if symbol:
            symbol = symbol.upper()
            if symbol in self.instruments:
                results.append(self.instruments[symbol])
            return results
        
        # Start with all instruments if no filters
        if not asset_class and not tags:
            results = list(self.instruments.values())
        # Filter by asset class if provided
        elif asset_class and not tags:
            results = self.get_instruments_by_asset_class(asset_class)
        # Filter by tags if provided
        elif tags and not asset_class:
            tag_sets = [set(self._tag_index.get(tag.lower(), set())) for tag in tags]
            if not tag_sets:
                return []
            
            # Get intersection of all tag sets
            symbols = set.intersection(*tag_sets)
            results = [self.instruments[sym] for sym in symbols if sym in self.instruments]
        # Filter by both asset class and tags
        else:
            asset_symbols = set(self._asset_class_index.get(asset_class, set()))
            tag_sets = [set(self._tag_index.get(tag.lower(), set())) for tag in tags or []]
            
            if tag_sets:
                # Get intersection of asset class and all tag sets
                symbols = asset_symbols.intersection(*tag_sets)
            else:
                symbols = asset_symbols
            
            results = [self.instruments[sym] for sym in symbols if sym in self.instruments]
        
        # Filter by tradable status if requested
        if tradable_only:
            results = [inst for inst in results if inst.is_tradable()]
        
        return results
    
    def load_from_file(self, file_path: str) -> int:
        """Load instruments from a file."""
        instruments = InstrumentMetadata.load_from_file(file_path)
        count = 0
        
        for symbol, instrument in instruments.items():
            self.add_instrument(instrument)
            count += 1
        
        return count
    
    def save_to_file(self, file_path: str, format: str = 'json') -> None:
        """Save instruments to a file."""
        InstrumentMetadata.save_to_file(self.instruments, file_path, format)
    
    def clear(self) -> None:
        """Clear all instruments and indices."""
        self.instruments.clear()
        self._tag_index.clear()
        self._asset_class_index.clear()
