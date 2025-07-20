"""
Position validation and risk checks.
"""
from decimal import Decimal, InvalidOperation
from typing import Dict, Optional, Tuple, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ValidationError(Exception):
    """Raised when a validation check fails."""
    pass

class PositionValidator:
    """
    Validates positions against risk parameters and business rules.
    """
    
    def __init__(
        self,
        max_position_size: Union[Decimal, float, int],
        max_position_value: Union[Decimal, float, int],
        allowed_instruments: Optional[set] = None,
        trading_hours: Optional[Dict[str, Tuple[datetime.time, datetime.time]]] = None,
    ):
        """
        Initialize the position validator.
        
        Args:
            max_position_size: Maximum position size in units
            max_position_value: Maximum position value in account currency
            allowed_instruments: Set of allowed instrument symbols (None = all allowed)
            trading_hours: Dict of symbol -> (open_time, close_time) for each instrument
        """
        self.max_position_size = Decimal(str(max_position_size))
        self.max_position_value = Decimal(str(max_position_value))
        self.allowed_instruments = allowed_instruments
        self.trading_hours = trading_hours or {}
    
    def validate_new_position(
        self,
        symbol: str,
        quantity: Union[Decimal, float, int, str],
        price: Optional[Union[Decimal, float, int, str]] = None,
        current_positions: Optional[Dict[str, Dict]] = None,
        account_balance: Optional[Union[Decimal, float, int, str]] = None,
        current_time: Optional[datetime] = None,
    ) -> Tuple[bool, str]:
        """
        Validate a new position.
        
        Args:
            symbol: Instrument symbol
            quantity: Position quantity (positive for long, negative for short)
            price: Current market price (required for value checks)
            current_positions: Current open positions {symbol: {quantity, ...}}
            account_balance: Current account balance (required for position value checks)
            current_time: Current time (for trading hours validation)
            
        Returns:
            Tuple of (is_valid, reason)
            - is_valid: True if position is valid
            - reason: Description of any validation failure
        """
        try:
            # Convert inputs to Decimal for precise arithmetic
            quantity = Decimal(str(quantity))
            price = Decimal(str(price)) if price is not None else None
            
            # Check if trading is allowed for this instrument
            if not self._is_instrument_allowed(symbol):
                return False, f"Trading not allowed for instrument: {symbol}"
            
            # Check trading hours
            if not self._is_within_trading_hours(symbol, current_time):
                return False, f"Outside trading hours for {symbol}"
            
            # Check position size limits
            if not self._validate_position_size(symbol, quantity):
                return False, f"Position size exceeds maximum allowed size of {self.max_position_size}"
            
            # Check position value limits if price is provided
            if price is not None:
                position_value = abs(quantity * price)
                if position_value > self.max_position_value:
                    return False, f"Position value {position_value} exceeds maximum allowed value of {self.max_position_value}"
                
                # Check position value against account balance if provided
                if account_balance is not None:
                    account_balance = Decimal(str(account_balance))
                    if position_value > account_balance * Decimal('0.9'):  # Max 90% of balance
                        return False, "Position value exceeds 90% of account balance"
            
            # Check aggregate exposure if current positions are provided
            if current_positions is not None and price is not None:
                if not self._validate_aggregate_exposure(symbol, quantity, price, current_positions):
                    return False, "Position would exceed aggregate exposure limits"
            
            return True, ""
            
        except (ValueError, InvalidOperation) as e:
            logger.error(f"Validation error: {e}", exc_info=True)
            return False, f"Invalid input parameters: {str(e)}"
    
    def _is_instrument_allowed(self, symbol: str) -> bool:
        """Check if trading is allowed for the given instrument."""
        if self.allowed_instruments is None:
            return True
        return symbol.upper() in self.allowed_instruments
    
    def _is_within_trading_hours(self, symbol: str, current_time: Optional[datetime] = None) -> bool:
        """Check if current time is within trading hours for the instrument."""
        if not self.trading_hours or symbol.upper() not in self.trading_hours:
            return True  # No trading hours defined for this symbol
            
        if current_time is None:
            current_time = datetime.now().time()
        else:
            current_time = current_time.time()
            
        open_time, close_time = self.trading_hours[symbol.upper()]
        return open_time <= current_time <= close_time
    
    def _validate_position_size(self, symbol: str, quantity: Decimal) -> bool:
        """Validate position size against limits."""
        return abs(quantity) <= self.max_position_size
    
    def _validate_aggregate_exposure(
        self,
        symbol: str,
        new_quantity: Decimal,
        price: Decimal,
        current_positions: Dict[str, Dict]
    ) -> bool:
        """
        Validate that the new position doesn't exceed aggregate exposure limits.
        
        This is a placeholder for more complex exposure calculations that might consider:
        - Sector exposure
        - Asset class exposure
        - Correlation between positions
        - Portfolio-level risk metrics
        """
        # Simple implementation: just check total notional exposure
        total_exposure = abs(new_quantity * price)
        
        # Add exposure from other positions
        for pos_symbol, pos in current_positions.items():
            if pos_symbol != symbol and 'quantity' in pos and 'price' in pos:
                total_exposure += abs(Decimal(str(pos['quantity'])) * Decimal(str(pos['price'])))
        
        # Don't allow total exposure to exceed 5x account value (example)
        return total_exposure <= self.max_position_value * Decimal('5')


# Example usage:
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize validator with some limits
    validator = PositionValidator(
        max_position_size=1000,  # Max 1000 units per position
        max_position_value=100000,  # Max $100k per position
        allowed_instruments={"AAPL", "MSFT", "GOOGL"},
        trading_hours={
            "AAPL": (datetime.strptime("09:30", "%H:%M").time(), 
                     datetime.strptime("16:00", "%H:%M").time()),
        }
    )
    
    # Test some validations
    print(validator.validate_new_position("AAPL", 100, 150.0))  # Valid
    print(validator.validate_new_position("TSLA", 100, 150.0))  # Not allowed
    print(validator.validate_new_position("AAPL", 10000, 150.0))  # Size too large
