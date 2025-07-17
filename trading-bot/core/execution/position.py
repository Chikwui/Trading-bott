"""
Position Management Module

This module defines the Position class and related enums for position management.
"""
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Optional, Dict, List, Any
import uuid


class PositionStatus(Enum):
    """Status of a trading position."""
    OPEN = auto()        # Position is currently open
    CLOSED = auto()      # Position has been fully closed
    PARTIALLY_CLOSED = auto()  # Position has been partially closed
    LIQUIDATED = auto()  # Position was liquidated (margin call, etc.)
    EXPIRED = auto()     # Position expired (for options, futures, etc.)


class PositionSide(Enum):
    """Side of a position (long or short)."""
    LONG = auto()
    SHORT = auto()
    
    def is_long(self) -> bool:
        return self == PositionSide.LONG
    
    def is_short(self) -> bool:
        return self == PositionSide.SHORT
    
    @classmethod
    def from_order_side(cls, order_side: 'OrderSide') -> 'PositionSide':
        """Convert OrderSide to PositionSide."""
        if order_side == OrderSide.BUY:
            return cls.LONG
        elif order_side == OrderSide.SELL:
            return cls.SHORT
        raise ValueError(f"Invalid order side: {order_side}")


@dataclass
class Position:
    """
    Represents a trading position.
    
    A position is created when an order is filled and represents an open
    position in the market. It tracks the current state of the position,
    including P&L, entry/exit prices, and other metrics.
    
    Attributes:
        position_id: Unique identifier for the position
        symbol: Trading symbol (e.g., 'AAPL')
        side: Position side (LONG or SHORT)
        quantity: Current position size (can be fractional)
        entry_price: Weighted average entry price
        current_price: Current market price
        status: Current position status
        unrealized_pnl: Current unrealized P&L
        realized_pnl: Realized P&L from closed portions of the position
        entry_time: Timestamp when position was opened
        exit_time: Timestamp when position was closed (if applicable)
        strategy_id: ID of the strategy that opened the position
        metadata: Additional position metadata
    """
    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    
    # Optional fields with defaults
    position_id: str = field(default_factory=lambda: f"pos_{uuid.uuid4().hex[:8]}")
    current_price: Optional[float] = None
    status: PositionStatus = PositionStatus.OPEN
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    entry_time: datetime = field(default_factory=datetime.utcnow)
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    strategy_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize position and validate parameters."""
        if self.quantity <= 0:
            raise ValueError("Position quantity must be positive")
        
        if self.entry_price <= 0:
            raise ValueError("Entry price must be positive")
        
        # Initialize current price to entry price if not provided
        if self.current_price is None:
            self.current_price = self.entry_price
        
        # Calculate initial unrealized P&L
        self.update_pnl(self.current_price)
    
    def is_open(self) -> bool:
        """Check if the position is currently open."""
        return self.status == PositionStatus.OPEN
    
    def is_closed(self) -> bool:
        """Check if the position is closed."""
        return self.status in [PositionStatus.CLOSED, PositionStatus.LIQUIDATED, PositionStatus.EXPIRED]
    
    def is_long(self) -> bool:
        """Check if this is a long position."""
        return self.side == PositionSide.LONG
    
    def is_short(self) -> bool:
        """Check if this is a short position."""
        return self.side == PositionSide.SHORT
    
    def update_pnl(self, current_price: float) -> float:
        """
        Update the position's unrealized P&L based on the current price.
        
        Args:
            current_price: Current market price of the asset
            
        Returns:
            The updated unrealized P&L
        """
        self.current_price = current_price
        
        if self.is_long():
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        else:  # SHORT
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity
            
        return self.unrealized_pnl
    
    def pnl_pct(self) -> float:
        """Calculate the P&L as a percentage of position value."""
        if self.entry_price == 0:
            return 0.0
        
        if self.is_long():
            return ((self.current_price - self.entry_price) / self.entry_price) * 100.0
        else:  # SHORT
            return ((self.entry_price - self.current_price) / self.entry_price) * 100.0
    
    def update_position(
        self, 
        quantity: float, 
        price: float, 
        is_reducing: bool = False
    ) -> 'PositionUpdateResult':
        """
        Update the position with a new fill.
        
        Args:
            quantity: Quantity of the fill (positive for long, negative for short)
            price: Fill price
            is_reducing: Whether this is a reducing trade (closing part of the position)
            
        Returns:
            PositionUpdateResult with details about the update
        """
        if self.is_closed():
            raise ValueError("Cannot update a closed position")
        
        # Calculate the direction of the fill relative to the position
        fill_side = PositionSide.LONG if quantity > 0 else PositionSide.SHORT
        
        # Calculate the effect on the position
        if not is_reducing and fill_side == self.side:
            # Increasing the position in the same direction
            return self._increase_position(abs(quantity), price)
        elif is_reducing and fill_side != self.side:
            # Reducing the position (partial close)
            return self._reduce_position(abs(quantity), price)
        elif fill_side != self.side:
            # Flipping the position (closing and opening in opposite direction)
            return self._flip_position(abs(quantity), price)
        else:
            # Shouldn't happen if is_reducing is set correctly
            raise ValueError("Invalid position update parameters")
    
    def _increase_position(self, quantity: float, price: float) -> 'PositionUpdateResult':
        """Increase the position size."""
        # Calculate new average entry price
        total_quantity = self.quantity + quantity
        total_cost = (self.quantity * self.entry_price) + (quantity * price)
        new_avg_price = total_cost / total_quantity
        
        # Update position
        old_quantity = self.quantity
        old_avg_price = self.entry_price
        
        self.quantity = total_quantity
        self.entry_price = new_avg_price
        
        # Update P&L
        self.update_pnl(self.current_price)
        
        return PositionUpdateResult(
            position_id=self.position_id,
            old_quantity=old_quantity,
            new_quantity=self.quantity,
            old_avg_price=old_avg_price,
            new_avg_price=new_avg_price,
            realized_pnl=0.0,  # No realized P&L when increasing position
            is_closed=False
        )
    
    def _reduce_position(self, quantity: float, price: float) -> 'PositionUpdateResult':
        """Reduce the position size."""
        if quantity > self.quantity:
            raise ValueError("Cannot reduce position by more than current size")
        
        # Calculate realized P&L for the reduced portion
        if self.is_long():
            realized_pnl = (price - self.entry_price) * quantity
        else:  # SHORT
            realized_pnl = (self.entry_price - price) * quantity
        
        # Update position
        old_quantity = self.quantity
        old_avg_price = self.entry_price
        
        self.quantity -= quantity
        self.realized_pnl += realized_pnl
        
        # Check if position is fully closed
        is_closed = self.quantity <= 0.000001  # Handle floating point precision
        if is_closed:
            self.quantity = 0.0
            self.status = PositionStatus.CLOSED
            self.exit_time = datetime.utcnow()
            self.exit_price = price
        
        # Update P&L
        self.update_pnl(self.current_price)
        
        return PositionUpdateResult(
            position_id=self.position_id,
            old_quantity=old_quantity,
            new_quantity=self.quantity,
            old_avg_price=old_avg_price,
            new_avg_price=self.entry_price,
            realized_pnl=realized_pnl,
            is_closed=is_closed
        )
    
    def _flip_position(self, quantity: float, price: float) -> 'PositionUpdateResult':
        """Flip the position to the opposite side."""
        # First close the existing position
        close_result = self._reduce_position(self.quantity, price)
        
        # Then open a new position in the opposite direction
        new_side = PositionSide.SHORT if self.is_long() else PositionSide.LONG
        flip_quantity = quantity - close_result.old_quantity
        
        if flip_quantity > 0:
            self.side = new_side
            self.quantity = flip_quantity
            self.entry_price = price
            self.status = PositionStatus.OPEN
            self.entry_time = datetime.utcnow()
            self.exit_time = None
            self.exit_price = None
        
        return PositionUpdateResult(
            position_id=self.position_id,
            old_quantity=close_result.old_quantity,
            new_quantity=self.quantity,
            old_avg_price=close_result.old_avg_price,
            new_avg_price=self.entry_price,
            realized_pnl=close_result.realized_pnl,
            is_closed=close_result.is_closed,
            side_flipped=flip_quantity > 0,
            new_side=self.side if flip_quantity > 0 else None
        )
    
    def close_position(self, price: float) -> 'PositionUpdateResult':
        """Close the entire position at the given price."""
        if self.is_closed():
            raise ValueError("Position is already closed")
        
        return self._reduce_position(self.quantity, price)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary representation."""
        return {
            'position_id': self.position_id,
            'symbol': self.symbol,
            'side': self.side.name,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'current_price': self.current_price,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'pnl_pct': self.pnl_pct(),
            'status': self.status.name,
            'entry_time': self.entry_time.isoformat(),
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'exit_price': self.exit_price,
            'strategy_id': self.strategy_id,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Position':
        """Create a Position instance from a dictionary."""
        # Convert string enums back to enum values
        data = data.copy()
        data['side'] = PositionSide[data['side']]
        data['status'] = PositionStatus[data['status']]
        
        # Convert string timestamps back to datetime objects
        if isinstance(data.get('entry_time'), str):
            data['entry_time'] = datetime.fromisoformat(data['entry_time'])
        if isinstance(data.get('exit_time'), str):
            data['exit_time'] = datetime.fromisoformat(data['exit_time'])
        
        return cls(**data)


@dataclass
class PositionUpdateResult:
    """Result of a position update operation."""
    position_id: str
    old_quantity: float
    new_quantity: float
    old_avg_price: float
    new_avg_price: float
    realized_pnl: float
    is_closed: bool
    side_flipped: bool = False
    new_side: Optional[PositionSide] = None


class PositionManager:
    """Manager for tracking and updating positions."""
    
    def __init__(self):
        self.positions = {}  # position_id -> Position
        self.symbol_to_position = {}  # symbol -> List[position_id]
    
    def get_position(self, position_id: str) -> Optional[Position]:
        """Get a position by its ID."""
        return self.positions.get(position_id)
    
    def get_positions_by_symbol(self, symbol: str) -> List[Position]:
        """Get all positions for a specific symbol."""
        position_ids = self.symbol_to_position.get(symbol, [])
        return [self.positions[pid] for pid in position_ids if pid in self.positions]
    
    def get_open_positions(self, symbol: str = None) -> List[Position]:
        """Get all open positions, optionally filtered by symbol."""
        if symbol:
            positions = self.get_positions_by_symbol(symbol)
        else:
            positions = list(self.positions.values())
        
        return [p for p in positions if p.is_open()]
    
    def update_position(
        self, 
        symbol: str, 
        quantity: float, 
        price: float, 
        is_reducing: bool = False,
        strategy_id: str = None
    ) -> PositionUpdateResult:
        """
        Update or create a position based on a fill.
        
        Args:
            symbol: Trading symbol
            quantity: Fill quantity (positive for long, negative for short)
            price: Fill price
            is_reducing: Whether this is a reducing trade (closing part of the position)
            strategy_id: ID of the strategy that generated the fill
            
        Returns:
            PositionUpdateResult with details about the update
        """
        if quantity == 0:
            raise ValueError("Quantity cannot be zero")
        
        # Get existing open position for this symbol and strategy
        open_positions = self.get_open_positions(symbol)
        
        if strategy_id:
            open_positions = [p for p in open_positions if p.strategy_id == strategy_id]
        
        # If no open position, create a new one
        if not open_positions:
            if is_reducing:
                raise ValueError("Cannot reduce non-existent position")
                
            position = Position(
                symbol=symbol,
                side=PositionSide.LONG if quantity > 0 else PositionSide.SHORT,
                quantity=abs(quantity),
                entry_price=price,
                strategy_id=strategy_id
            )
            
            self._add_position(position)
            
            return PositionUpdateResult(
                position_id=position.position_id,
                old_quantity=0.0,
                new_quantity=position.quantity,
                old_avg_price=0.0,
                new_avg_price=position.entry_price,
                realized_pnl=0.0,
                is_closed=False
            )
        
        # If multiple open positions, use the first one (shouldn't normally happen)
        if len(open_positions) > 1:
            # In a more sophisticated system, you might want to handle this differently
            # For now, we'll just log a warning and use the first position
            import warnings
            warnings.warn(f"Multiple open positions found for {symbol}, using the first one")
        
        position = open_positions[0]
        return position.update_position(quantity, price, is_reducing)
    
    def _add_position(self, position: Position):
        """Add a new position to the manager."""
        self.positions[position.position_id] = position
        
        if position.symbol not in self.symbol_to_position:
            self.symbol_to_position[position.symbol] = []
        
        if position.position_id not in self.symbol_to_position[position.symbol]:
            self.symbol_to_position[position.symbol].append(position.position_id)
    
    def close_position(self, position_id: str, price: float) -> PositionUpdateResult:
        """Close a position by its ID."""
        position = self.get_position(position_id)
        if not position:
            raise ValueError(f"Position {position_id} not found")
        
        if position.is_closed():
            raise ValueError(f"Position {position_id} is already closed")
        
        result = position.close_position(price)
        
        # Clean up if position is closed
        if result.is_closed and position_id in self.positions:
            # Don't remove from symbol_to_position to maintain history
            pass
        
        return result
    
    def update_prices(self, prices: Dict[str, float]):
        """Update all positions with current market prices."""
        for symbol, price in prices.items():
            for position in self.get_positions_by_symbol(symbol):
                position.update_pnl(price)
    
    def get_total_pnl(self) -> float:
        """Get the total P&L across all positions."""
        return sum(p.realized_pnl + p.unrealized_pnl for p in self.positions.values())
    
    def get_pnl_breakdown(self) -> Dict[str, float]:
        """Get a breakdown of P&L by symbol."""
        pnl_by_symbol = {}
        
        for position in self.positions.values():
            if position.symbol not in pnl_by_symbol:
                pnl_by_symbol[position.symbol] = 0.0
            
            pnl_by_symbol[position.symbol] += position.realized_pnl + position.unrealized_pnl
        
        return pnl_by_symbol
