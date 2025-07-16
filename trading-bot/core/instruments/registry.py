"""
Instrument registry for managing instrument instances.
"""
from datetime import datetime
from typing import Dict, List, Optional, Set, Type, TypeVar, Any
import logging
from pathlib import Path
import json
import yaml

from .metadata import (
    InstrumentMetadata,
    AssetClass,
    InstrumentType,
    TradingHours
)
from .factory import InstrumentFactory

logger = logging.getLogger(__name__)
T = TypeVar('T', bound='InstrumentRegistry')


class InstrumentRegistry:
    """Registry for managing instrument instances with hierarchical tagging."""
    
    def __init__(self):
        """Initialize the registry."""
        self._instruments: Dict[str, InstrumentMetadata] = {}
        self._tag_index: Dict[str, Set[str]] = {}  # tag -> set of symbols
        self._asset_class_index: Dict[AssetClass, Set[str]] = {}  # asset_class -> set of symbols
        self._type_index: Dict[InstrumentType, Set[str]] = {}  # instrument_type -> set of symbols
        self._exchange_index: Dict[str, Set[str]] = {}  # exchange -> set of symbols
        self._factory = InstrumentFactory()
    
    def __contains__(self, symbol: str) -> bool:
        """Check if an instrument exists in the registry."""
        return symbol.upper() in self._instruments
    
    def __getitem__(self, symbol: str) -> InstrumentMetadata:
        """Get an instrument by symbol."""
        return self.get_instrument(symbol)
    
    def __len__(self) -> int:
        """Get the number of instruments in the registry."""
        return len(self._instruments)
    
    def __iter__(self):
        """Iterate over all instruments in the registry."""
        return iter(self._instruments.values())
    
    @property
    def symbols(self) -> List[str]:
        """Get a list of all instrument symbols."""
        return list(self._instruments.keys())
    
    @property
    def instruments(self) -> List[InstrumentMetadata]:
        """Get a list of all instruments."""
        return list(self._instruments.values())
    
    def add_instrument(self, instrument: InstrumentMetadata) -> None:
        """
        Add an instrument to the registry.
        
        Args:
            instrument: The instrument to add
            
        Raises:
            ValueError: If an instrument with the same symbol already exists
        """
        symbol = instrument.symbol.upper()
        
        if symbol in self._instruments:
            raise ValueError(f"Instrument with symbol '{symbol}' already exists")
        
        # Add to main registry
        self._instruments[symbol] = instrument
        
        # Update tag index
        for tag in instrument.tags:
            self._add_to_index(self._tag_index, tag.lower(), symbol)
        
        # Update asset class index
        self._add_to_index(self._asset_class_index, instrument.asset_class, symbol)
        
        # Update type index if type is specified
        if instrument.instrument_type:
            self._add_to_index(self._type_index, instrument.instrument_type, symbol)
        
        # Update exchange index if exchange is specified
        if instrument.exchange:
            self._add_to_index(self._exchange_index, instrument.exchange.lower(), symbol)
    
    def update_instrument(self, instrument: InstrumentMetadata) -> None:
        """
        Update an existing instrument in the registry.
        
        Args:
            instrument: The instrument with updated values
            
        Raises:
            KeyError: If the instrument does not exist in the registry
        """
        symbol = instrument.symbol.upper()
        
        if symbol not in self._instruments:
            raise KeyError(f"Instrument with symbol '{symbol}' not found in registry")
        
        # Get the old instrument for index cleanup
        old_instrument = self._instruments[symbol]
        
        # Remove old indices
        for tag in old_instrument.tags:
            self._remove_from_index(self._tag_index, tag.lower(), symbol)
        
        self._remove_from_index(self._asset_class_index, old_instrument.asset_class, symbol)
        
        if old_instrument.instrument_type:
            self._remove_from_index(self._type_index, old_instrument.instrument_type, symbol)
        
        if old_instrument.exchange:
            self._remove_from_index(self._exchange_index, old_instrument.exchange.lower(), symbol)
        
        # Update the instrument
        self._instruments[symbol] = instrument
        
        # Add new indices
        for tag in instrument.tags:
            self._add_to_index(self._tag_index, tag.lower(), symbol)
        
        self._add_to_index(self._asset_class_index, instrument.asset_class, symbol)
        
        if instrument.instrument_type:
            self._add_to_index(self._type_index, instrument.instrument_type, symbol)
        
        if instrument.exchange:
            self._add_to_index(self._exchange_index, instrument.exchange.lower(), symbol)
    
    def remove_instrument(self, symbol: str) -> bool:
        """
        Remove an instrument from the registry.
        
        Args:
            symbol: The symbol of the instrument to remove
            
        Returns:
            bool: True if the instrument was removed, False if it didn't exist
        """
        symbol = symbol.upper()
        
        if symbol not in self._instruments:
            return False
        
        instrument = self._instruments[symbol]
        
        # Remove from indices
        for tag in instrument.tags:
            self._remove_from_index(self._tag_index, tag.lower(), symbol)
        
        self._remove_from_index(self._asset_class_index, instrument.asset_class, symbol)
        
        if instrument.instrument_type:
            self._remove_from_index(self._type_index, instrument.instrument_type, symbol)
        
        if instrument.exchange:
            self._remove_from_index(self._exchange_index, instrument.exchange.lower(), symbol)
        
        # Remove from main registry
        del self._instruments[symbol]
        
        return True
    
    def get_instrument(self, symbol: str) -> InstrumentMetadata:
        """
        Get an instrument by symbol.
        
        Args:
            symbol: The symbol of the instrument to retrieve
            
        Returns:
            InstrumentMetadata: The requested instrument
            
        Raises:
            KeyError: If the instrument is not found
        """
        symbol = symbol.upper()
        
        if symbol not in self._instruments:
            raise KeyError(f"Instrument with symbol '{symbol}' not found in registry")
        
        return self._instruments[symbol]
    
    def get_instruments_by_tag(self, tag: str) -> List[InstrumentMetadata]:
        """
        Get all instruments with the specified tag.
        
        Args:
            tag: The tag to filter by (case-insensitive)
            
        Returns:
            List[InstrumentMetadata]: A list of matching instruments
        """
        symbols = self._tag_index.get(tag.lower(), set())
        return [self._instruments[symbol] for symbol in symbols]
    
    def get_instruments_by_asset_class(self, asset_class: AssetClass) -> List[InstrumentMetadata]:
        """
        Get all instruments of the specified asset class.
        
        Args:
            asset_class: The asset class to filter by
            
        Returns:
            List[InstrumentMetadata]: A list of matching instruments
        """
        symbols = self._asset_class_index.get(asset_class, set())
        return [self._instruments[symbol] for symbol in symbols]
    
    def get_instruments_by_type(self, instrument_type: InstrumentType) -> List[InstrumentMetadata]:
        """
        Get all instruments of the specified type.
        
        Args:
            instrument_type: The instrument type to filter by
            
        Returns:
            List[InstrumentMetadata]: A list of matching instruments
        """
        symbols = self._type_index.get(instrument_type, set())
        return [self._instruments[symbol] for symbol in symbols]
    
    def get_instruments_by_exchange(self, exchange: str) -> List[InstrumentMetadata]:
        """
        Get all instruments from the specified exchange.
        
        Args:
            exchange: The exchange to filter by (case-insensitive)
            
        Returns:
            List[InstrumentMetadata]: A list of matching instruments
        """
        symbols = self._exchange_index.get(exchange.lower(), set())
        return [self._instruments[symbol] for symbol in symbols]
    
    def get_all_tags(self) -> Set[str]:
        """
        Get all unique tags in the registry.
        
        Returns:
            Set[str]: A set of all unique tags
        """
        return set(self._tag_index.keys())
    
    def search_instruments(
        self,
        symbol: Optional[str] = None,
        asset_class: Optional[AssetClass] = None,
        instrument_type: Optional[InstrumentType] = None,
        exchange: Optional[str] = None,
        tags: Optional[List[str]] = None,
        active_only: bool = False,
        tradable_only: bool = False
    ) -> List[InstrumentMetadata]:
        """
        Search for instruments matching the specified criteria.
        
        Args:
            symbol: Filter by symbol (case-insensitive, partial match)
            asset_class: Filter by asset class
            instrument_type: Filter by instrument type
            exchange: Filter by exchange (case-insensitive)
            tags: Filter by tags (must match all specified tags)
            active_only: If True, only return active instruments
            tradable_only: If True, only return tradable instruments
            
        Returns:
            List[InstrumentMetadata]: A list of matching instruments
        """
        # Start with all instruments
        if symbol:
            # Fast path: direct symbol lookup
            symbol = symbol.upper()
            if symbol in self._instruments:
                results = [self._instruments[symbol]]
            else:
                # Partial symbol match
                results = [
                    inst for sym, inst in self._instruments.items()
                    if symbol.upper() in sym
                ]
        else:
            results = list(self._instruments.values())
        
        # Apply filters
        if asset_class is not None:
            results = [inst for inst in results if inst.asset_class == asset_class]
        
        if instrument_type is not None:
            results = [inst for inst in results if inst.instrument_type == instrument_type]
        
        if exchange is not None:
            exchange = exchange.lower()
            results = [
                inst for inst in results
                if inst.exchange and inst.exchange.lower() == exchange
            ]
        
        if tags:
            tag_set = {tag.lower() for tag in tags}
            results = [
                inst for inst in results
                if tag_set.issubset({t.lower() for t in inst.tags})
            ]
        
        if active_only:
            results = [inst for inst in results if inst.is_active]
        
        if tradable_only:
            results = [inst for inst in results if inst.is_tradable()]
        
        return results
    
    def load_from_file(self, file_path: str) -> int:
        """
        Load instruments from a JSON or YAML file.
        
        Args:
            file_path: Path to the file to load
            
        Returns:
            int: Number of instruments loaded
            
        Raises:
            ValueError: If the file format is not supported
        """
        path = Path(file_path)
        
        if not path.exists():
            logger.warning(f"Instrument file not found: {file_path}")
            return 0
        
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            if path.suffix.lower() == '.json':
                data = json.load(f)
            elif path.suffix.lower() in ('.yaml', '.yml'):
                data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")
        
        # Process instruments
        count = 0
        for item in data.get('instruments', []):
            try:
                instrument = InstrumentMetadata.from_dict(item)
                self.add_instrument(instrument)
                count += 1
            except Exception as e:
                logger.error(f"Failed to load instrument: {e}", exc_info=True)
        
        return count
    
    def save_to_file(self, file_path: str, format: str = 'json') -> None:
        """
        Save instruments to a file.
        
        Args:
            file_path: Path to the output file
            format: Output format ('json' or 'yaml')
            
        Raises:
            ValueError: If the format is not supported
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare data
        data = {
            'instruments': [inst.to_dict() for inst in self._instruments.values()],
            'metadata': {
                'generated_at': str(datetime.utcnow()),
                'count': len(self._instruments)
            }
        }
        
        # Write to file
        with open(file_path, 'w', encoding='utf-8') as f:
            if format.lower() == 'json':
                json.dump(data, f, indent=2, ensure_ascii=False)
            elif format.lower() in ('yaml', 'yml'):
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            else:
                raise ValueError(f"Unsupported format: {format}")
    
    def clear(self) -> None:
        """Clear all instruments from the registry."""
        self._instruments.clear()
        self._tag_index.clear()
        self._asset_class_index.clear()
        self._type_index.clear()
        self._exchange_index.clear()
    
    def _add_to_index(self, index: Dict[Any, Set[str]], key: Any, symbol: str) -> None:
        """Add a symbol to the specified index."""
        if key not in index:
            index[key] = set()
        index[key].add(symbol)
    
    def _remove_from_index(self, index: Dict[Any, Set[str]], key: Any, symbol: str) -> None:
        """Remove a symbol from the specified index."""
        if key in index:
            index[key].discard(symbol)
            if not index[key]:  # Remove empty sets
                del index[key]
