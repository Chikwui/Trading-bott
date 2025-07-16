"""
OHLCV (Open, High, Low, Close, Volume) data model.
"""
from datetime import datetime
from typing import Optional
from .base import BaseModel

class OHLCV(BaseModel):
    """OHLCV (Open, High, Low, Close, Volume) data point."""
    
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None  # Volume Weighted Average Price
    trade_count: Optional[int] = None
    
    @property
    def body(self) -> float:
        """Get the absolute size of the candle body."""
        return abs(self.close - self.open)
    
    @property
    def upper_shadow(self) -> float:
        """Get the size of the upper shadow/wick."""
        return self.high - max(self.open, self.close)
    
    @property
    def lower_shadow(self) -> float:
        """Get the size of the lower shadow/wick."""
        return min(self.open, self.close) - self.low
    
    @property
    def is_bullish(self) -> bool:
        """Check if the candle is bullish (close > open)."""
        return self.close > self.open
    
    @property
    def is_bearish(self) -> bool:
        """Check if the candle is bearish (close < open)."""
        return self.close < self.open
    
    @property
    def is_doji(self) -> bool:
        """Check if the candle is a doji (open ≈ close)."""
        return abs(self.close - self.open) <= (self.high - self.low) * 0.1  # 10% threshold
    
    @property
    def range(self) -> float:
        """Get the price range (high - low)."""
        return self.high - self.low
    
    @property
    def body_percentage(self) -> float:
        """Get the body size as a percentage of the total range."""
        if self.range == 0:
            return 0.0
        return (self.body / self.range) * 100.0
