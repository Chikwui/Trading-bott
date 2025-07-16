"""
Trade-related data models.
"""
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
from decimal import Decimal
from .base import BaseModel
from .order import OrderSide, OrderType

class TradeSide(str, Enum):
    """Trade sides."""
    BUY = "buy"
    SELL = "sell"

class Liquidity(str, Enum):
    """Trade liquidity (taker or maker)."""
    TAKER = "taker"
    MAKER = "maker"

class Trade(BaseModel):
    """Trade model representing an executed trade."""
    
    id: str  # Trade ID
    order_id: str  # Associated order ID
    symbol: str
    side: TradeSide
    price: float
    amount: float  # Base currency amount
    cost: float  # Quote currency cost (price * amount)
    
    # Timestamps
    timestamp: datetime
    datetime: Optional[datetime] = None  # Alias for timestamp
    
    # Fees
    fee: Optional[Dict[str, Any]] = None  # {'currency': 'USDT', 'cost': 0.1, 'rate': 0.001}
    
    # Additional info
    taker_or_maker: Optional[Liquidity] = None  # 'taker' or 'maker'
    fee_currency: Optional[str] = None  # Deprecated, use fee['currency'] instead
    fee_cost: Optional[float] = None  # Deprecated, use fee['cost'] instead
    fee_rate: Optional[float] = None  # Deprecated, use fee['rate'] instead
    
    # Additional metadata
    info: Dict[str, Any] = {}
    
    def __init__(self, **data):
        # Ensure datetime is set from timestamp if not provided
        if 'datetime' not in data and 'timestamp' in data:
            data['datetime'] = data['timestamp']
        super().__init__(**data)
    
    @property
    def is_buy(self) -> bool:
        """Check if this is a buy trade."""
        return self.side == TradeSide.BUY
    
    @property
    def is_sell(self) -> bool:
        """Check if this is a sell trade."""
        return self.side == TradeSide.SELL
    
    @property
    def is_taker(self) -> bool:
        """Check if this was a taker trade."""
        return self.taker_or_maker == Liquidity.TAKER
    
    @property
    def is_maker(self) -> bool:
        """Check if this was a maker trade."""
        return self.taker_or_maker == Liquidity.MAKER
    
    @property
    def fee_currency_code(self) -> Optional[str]:
        """Get the fee currency code."""
        if self.fee and 'currency' in self.fee:
            return self.fee['currency']
        return self.fee_currency
    
    @property
    def fee_amount(self) -> float:
        """Get the fee amount."""
        if self.fee and 'cost' in self.fee:
            return self.fee['cost']
        return self.fee_cost or 0.0
    
    @property
    def fee_rate_decimal(self) -> float:
        """Get the fee rate as a decimal."""
        if self.fee and 'rate' in self.fee:
            return self.fee['rate']
        return self.fee_rate or 0.0
    
    @property
    def fee_percentage(self) -> float:
        """Get the fee as a percentage of the trade value."""
        if self.cost == 0:
            return 0.0
        return (self.fee_amount / self.cost) * 100.0
    
    @classmethod
    def from_order_fill(cls, order: 'Order', fill: Dict[str, Any]) -> 'Trade':
        """Create a Trade from an order fill.
        
        Args:
            order: The order that was filled
            fill: The fill data from the exchange
            
        Returns:
            A new Trade instance
        """
        return cls(
            id=fill.get('id', ''),
            order_id=order.id,
            symbol=order.symbol,
            side=TradeSide.BUY if order.side == OrderSide.BUY else TradeSide.SELL,
            price=float(fill.get('price', 0)),
            amount=float(fill.get('amount', 0)),
            cost=float(fill.get('cost', 0)),
            timestamp=fill.get('timestamp', datetime.utcnow()),
            fee=fill.get('fee'),
            taker_or_maker=fill.get('takerOrMaker'),
            info=fill
        )
