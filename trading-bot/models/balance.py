"""
Balance and account-related data models.
"""
from datetime import datetime
from typing import Dict, List, Optional
from decimal import Decimal
from .base import BaseModel

class Balance(BaseModel):
    """Account balance for a specific currency."""
    
    currency: str
    free: float = 0.0  # Available balance
    used: float = 0.0   # Amount in open orders
    total: float = 0.0  # Total balance (free + used)
    
    # Additional metadata
    last_updated: Optional[datetime] = None
    
    def __init__(self, **data):
        super().__init__(**data)
        # Ensure total is always free + used if not explicitly set
        if 'total' not in data:
            self.total = self.free + self.used
    
    @property
    def available(self) -> float:
        """Alias for free balance."""
        return self.free
    
    @property
    def locked(self) -> float:
        """Alias for used balance."""
        return self.used
    
    def update(self, free: float = None, used: float = None, total: float = None) -> None:
        """Update balance values."""
        if free is not None:
            self.free = free
        if used is not None:
            self.used = used
        if total is not None:
            self.total = total
        elif free is not None or used is not None:
            self.total = self.free + self.used
        
        self.last_updated = datetime.utcnow()


class AccountInfo(BaseModel):
    """Complete account information including all balances."""
    
    id: Optional[str] = None
    balances: Dict[str, Balance] = {}
    maker_commission: float = 0.0  # In basis points (0.1% = 10)
    taker_commission: float = 0.0  # In basis points (0.1% = 10)
    buyer_commission: float = 0.0  # In basis points (0.1% = 10)
    seller_commission: float = 0.0  # In basis points (0.1% = 10)
    can_trade: bool = False
    can_withdraw: bool = False
    can_deposit: bool = False
    update_time: Optional[datetime] = None
    
    def get_balance(self, currency: str) -> Balance:
        """Get balance for a specific currency."""
        currency = currency.upper()
        if currency not in self.balances:
            self.balances[currency] = Balance(currency=currency)
        return self.balances[currency]
    
    def update_balance(self, currency: str, free: float = None, used: float = None, total: float = None) -> None:
        """Update balance for a specific currency."""
        balance = self.get_balance(currency)
        balance.update(free=free, used=used, total=total)
    
    def get_free_balance(self, currency: str) -> float:
        """Get free balance for a specific currency."""
        return self.get_balance(currency).free
    
    def get_used_balance(self, currency: str) -> float:
        """Get used balance for a specific currency."""
        return self.get_balance(currency).used
    
    def get_total_balance(self, currency: str) -> float:
        """Get total balance for a specific currency."""
        return self.get_balance(currency).total
    
    def get_balances(self) -> Dict[str, Balance]:
        """Get all non-zero balances."""
        return {k: v for k, v in self.balances.items() if v.total > 0 or v.used > 0}
