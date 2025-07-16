"""
Ticker data model.
"""
from datetime import datetime
from typing import Optional
from .base import BaseModel

class Ticker(BaseModel):
    """Ticker data for a trading pair."""
    
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last: float
    bid_volume: Optional[float] = None
    ask_volume: Optional[float] = None
    base_volume: Optional[float] = None
    quote_volume: Optional[float] = None
    vwap: Optional[float] = None  # Volume Weighted Average Price
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    change: Optional[float] = None
    percentage: Optional[float] = None
    average: Optional[float] = None
    
    @property
    def spread(self) -> float:
        """Calculate the spread between bid and ask prices."""
        return self.ask - self.bid
    
    @property
    def spread_percentage(self) -> float:
        """Calculate the spread as a percentage of the mid price."""
        if self.bid == 0:
            return 0.0
        return (self.spread / self.bid) * 100.0
    
    @property
    def mid_price(self) -> float:
        """Calculate the mid price between bid and ask."""
        return (self.bid + self.ask) / 2
