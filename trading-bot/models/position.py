"""
Position-related data models.
"""
from datetime import datetime
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple
from decimal import Decimal
from .base import BaseModel

class PositionSide(str, Enum):
    """Position sides."""
    LONG = "long"
    SHORT = "short"
    BOTH = "both"  # For exchanges that don't distinguish between long/short

class PositionStatus(str, Enum):
    """Position statuses."""
    OPEN = "open"
    CLOSED = "closed"
    LIQUIDATED = "liquidated"

class Position(BaseModel):
    """Trading position model."""
    
    id: Optional[str] = None
    symbol: str
    side: PositionSide
    status: PositionStatus = PositionStatus.OPEN
    
    # Position sizing
    size: float  # Position size in base currency
    entry_price: float  # Average entry price
    mark_price: Optional[float] = None  # Current mark price
    liquidation_price: Optional[float] = None
    
    # PnL
    unrealized_pnl: Optional[float] = None
    realized_pnl: Optional[float] = 0.0
    pnl_percentage: Optional[float] = None
    
    # Leverage and margin
    leverage: float = 1.0
    initial_margin: Optional[float] = None  # Initial margin required
    maintenance_margin: Optional[float] = None  # Maintenance margin required
    margin_ratio: Optional[float] = None  # Current margin ratio
    
    # Funding
    funding_rate: Optional[float] = None  # Current funding rate
    next_funding_time: Optional[datetime] = None  # Next funding time
    
    # Timestamps
    opened_at: datetime
    updated_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    
    # Additional info
    info: Dict[str, Any] = {}
    
    @property
    def notional_value(self) -> float:
        """Calculate the notional value of the position in quote currency."""
        if self.mark_price is not None:
            return abs(self.size) * self.mark_price
        return 0.0
    
    @property
    def entry_value(self) -> float:
        """Calculate the entry value of the position in quote currency."""
        return abs(self.size) * self.entry_price
    
    @property
    def is_open(self) -> bool:
        """Check if the position is open."""
        return self.status == PositionStatus.OPEN
    
    @property
    def is_closed(self) -> bool:
        """Check if the position is closed."""
        return not self.is_open
    
    @property
    def is_long(self) -> bool:
        """Check if the position is long."""
        return self.side == PositionSide.LONG
    
    @property
    def is_short(self) -> bool:
        """Check if the position is short."""
        return self.side == PositionSide.SHORT
    
    def calculate_pnl(self, current_price: Optional[float] = None) -> Tuple[float, float]:
        """Calculate PnL for the position.
        
        Args:
            current_price: Current market price (if None, use mark_price)
            
        Returns:
            Tuple of (unrealized_pnl, pnl_percentage)
        """
        if current_price is None:
            current_price = self.mark_price
            
        if current_price is None:
            return 0.0, 0.0
            
        if self.size == 0:
            return 0.0, 0.0
            
        if self.is_long:
            price_diff = current_price - self.entry_price
        else:  # short
            price_diff = self.entry_price - current_price
            
        unrealized_pnl = price_diff * abs(self.size)
        pnl_percentage = (price_diff / self.entry_price) * 100.0 * (1.0 if self.is_long else -1.0)
        
        return unrealized_pnl, pnl_percentage
    
    def update(self, update: 'Position') -> None:
        """Update position with new data."""
        for field in [
            'status', 'size', 'entry_price', 'mark_price', 'liquidation_price',
            'unrealized_pnl', 'realized_pnl', 'pnl_percentage', 'leverage',
            'initial_margin', 'maintenance_margin', 'margin_ratio',
            'funding_rate', 'next_funding_time', 'updated_at', 'closed_at', 'info'
        ]:
            if hasattr(update, field) and getattr(update, field) is not None:
                setattr(self, field, getattr(update, field))
