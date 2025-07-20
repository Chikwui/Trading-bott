"""
Position Manager - Core position and exposure management system.
Handles position sizing, exposure limits, risk calculations, and validation.
"""
from dataclasses import dataclass, field, asdict
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
from datetime import datetime, time
import logging

from core.risk.validation import PositionValidator, ValidationError

logger = logging.getLogger(__name__)

@dataclass
class Position:
    """Represents a single trading position with comprehensive tracking."""
    symbol: str
    quantity: Decimal
    entry_price: Decimal
    current_price: Decimal
    entry_time: datetime = field(default_factory=datetime.utcnow)
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def market_value(self) -> Decimal:
        """Current market value of the position."""
        return self.quantity * self.current_price
    
    @property
    def cost_basis(self) -> Decimal:
        """Total cost basis of the position."""
        return abs(self.quantity * self.entry_price)
    
    @property
    def unrealized_pnl(self) -> Decimal:
        """Unrealized profit/loss."""
        return (self.current_price - self.entry_price) * self.quantity
    
    @property
    def unrealized_pnl_pct(self) -> Decimal:
        """Unrealized P&L as a percentage of cost basis."""
        if self.cost_basis == Decimal('0'):
            return Decimal('0')
        return (self.unrealized_pnl / self.cost_basis) * Decimal('100')
    
    @property
    def is_long(self) -> bool:
        """Whether this is a long position."""
        return self.quantity > Decimal('0')
    
    @property
    def is_short(self) -> bool:
        """Whether this is a short position."""
        return self.quantity < Decimal('0')
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary."""
        result = asdict(self)
        result['market_value'] = float(self.market_value)
        result['cost_basis'] = float(self.cost_basis)
        result['unrealized_pnl'] = float(self.unrealized_pnl)
        result['unrealized_pnl_pct'] = float(self.unrealized_pnl_pct)
        result['is_long'] = self.is_long
        result['is_short'] = self.is_short
        return result
    timestamp: datetime = field(default_factory=datetime.utcnow)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    meta: dict = field(default_factory=dict)

    @property
    def market_value(self) -> float:
        """Current market value of the position."""
        return self.quantity * self.current_price
    
    @property
    def pnl(self) -> float:
        """Unrealized P&L for the position."""
        return (self.current_price - self.entry_price) * self.quantity

class PositionManager:
    """
    Advanced position management with comprehensive validation and risk controls.
    
    Features:
    - Position validation with configurable limits
    - Risk-based position sizing
    - Exposure tracking and limits
    - P&L calculation
    - Trading hours enforcement
    - Instrument allowlisting
    """
    
    def __init__(
        self,
        account_balance: Union[float, str, Decimal],
        risk_per_trade: float = 0.02,
        max_position_size: float = 0.1,
        max_position_value: Optional[float] = None,
        max_portfolio_risk: float = 0.2,
        allowed_instruments: Optional[set] = None,
        trading_hours: Optional[Dict[str, Tuple[time, time]]] = None,
    ):
        """
        Initialize PositionManager with comprehensive risk controls.
        
        Args:
            account_balance: Current account balance in base currency
            risk_per_trade: Maximum risk per trade as a fraction of account balance (default: 2%)
            max_position_size: Maximum position size as a fraction of account balance (default: 10%)
            max_position_value: Maximum absolute position value (optional, overrides size-based limit)
            max_portfolio_risk: Maximum portfolio risk as a fraction of account balance (default: 20%)
            allowed_instruments: Set of allowed instrument symbols (None = all allowed)
            trading_hours: Dict of symbol -> (open_time, close_time) for each instrument
        """
        self.positions: Dict[str, Position] = {}
        self.account_balance = Decimal(str(account_balance))
        self.risk_per_trade = Decimal(str(risk_per_trade))
        self.max_position_size = Decimal(str(max_position_size))
        self.max_position_value = (
            Decimal(str(max_position_value)) 
            if max_position_value is not None 
            else self.account_balance * self.max_position_size
        )
        self.max_portfolio_risk = Decimal(str(max_portfolio_risk))
        
        # Initialize validator
        self.validator = PositionValidator(
            max_position_size=self.max_position_value,
            max_position_value=self.max_position_value,
            allowed_instruments=allowed_instruments,
            trading_hours=trading_hours,
        )
        
    def calculate_position_size(
        self, 
        symbol: str, 
        entry_price: Union[float, str, Decimal],
        stop_loss: Union[float, str, Decimal],
        risk_amount: Optional[Union[float, str, Decimal]] = None,
        current_time: Optional[datetime] = None,
    ) -> Tuple[Decimal, Decimal]:
        """
        Calculate optimal position size based on risk parameters with validation.
        
        Args:
            symbol: Trading symbol
            entry_price: Intended entry price
            stop_loss: Stop loss price
            risk_amount: Custom risk amount (optional, overrides default)
            current_time: Current time for trading hours validation
            
        Returns:
            Tuple of (position_size, risk_amount)
            
        Raises:
            ValidationError: If position would violate risk parameters
        """
        # Convert inputs to Decimal for precise arithmetic
        entry_price = Decimal(str(entry_price))
        stop_loss = Decimal(str(stop_loss))
        
        # Calculate risk amount if not provided
        if risk_amount is None:
            risk_amount = self.account_balance * self.risk_per_trade
        else:
            risk_amount = Decimal(str(risk_amount))
        
        # Calculate price risk and position size
        price_risk = abs(entry_price - stop_loss)
        if price_risk <= 0:
            return Decimal('0'), Decimal('0')
        
        # Base position size calculation
        position_size = (risk_amount / price_risk) * entry_price
        
        # Get current positions for exposure calculation
        current_positions = {
            sym: {'quantity': pos.quantity, 'price': pos.current_price}
            for sym, pos in self.positions.items()
        }
        
        # Validate the position
        is_valid, reason = self.validator.validate_new_position(
            symbol=symbol,
            quantity=position_size,
            price=entry_price,
            current_positions=current_positions,
            account_balance=float(self.account_balance),
            current_time=current_time,
        )
        
        if not is_valid:
            raise ValidationError(f"Position validation failed: {reason}")
        
        # Apply position size limits
        max_size_by_balance = self.account_balance * self.max_position_size
        max_size_by_value = self.max_position_value / entry_price if entry_price > 0 else Decimal('0')
        max_size = min(max_size_by_balance, max_size_by_value)
        
        position_size = min(abs(position_size), max_size)
        
        # Ensure position size has the correct sign
        if entry_price < stop_loss:  # Short position
            position_size = -position_size
        
        return position_size, risk_amount
        
    def add_position(
        self, 
        symbol: str, 
        quantity: Union[float, str, Decimal],
        entry_price: Union[float, str, Decimal],
        current_price: Optional[Union[float, str, Decimal]] = None,
        stop_loss: Optional[Union[float, str, Decimal]] = None,
        take_profit: Optional[Union[float, str, Decimal]] = None,
        meta: Optional[dict] = None,
        validate: bool = True,
        current_time: Optional[datetime] = None,
    ) -> Position:
        """
        Add a new position with comprehensive validation.
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL')
            quantity: Position quantity (positive for long, negative for short)
            entry_price: Entry price
            current_price: Current market price (defaults to entry_price if None)
            stop_loss: Stop loss price (optional)
            take_profit: Take profit price (optional)
            meta: Additional position metadata (optional)
            validate: Whether to validate the position before adding
            current_time: Current time for validation (defaults to now if None)
            
        Returns:
            The created Position object
            
        Raises:
            ValidationError: If position validation fails
            ValueError: For invalid inputs
        """
        # Convert inputs to Decimal for consistency
        quantity = Decimal(str(quantity))
        entry_price = Decimal(str(entry_price))
        current_price = Decimal(str(current_price)) if current_price is not None else entry_price
        
        # Check for existing position
        if symbol in self.positions:
            logger.warning(f"Position for {symbol} already exists. Updating instead of adding.")
            return self.update_position(
                symbol=symbol,
                quantity=quantity,
                current_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                meta=meta,
                validate=validate,
                current_time=current_time,
            )
        
        # Validate the new position if requested
        if validate:
            current_positions = {
                sym: {'quantity': pos.quantity, 'price': pos.current_price}
                for sym, pos in self.positions.items()
            }
            
            is_valid, reason = self.validator.validate_new_position(
                symbol=symbol,
                quantity=quantity,
                price=entry_price,
                current_positions=current_positions,
                account_balance=float(self.account_balance),
                current_time=current_time,
            )
            
            if not is_valid:
                raise ValidationError(f"Cannot add position: {reason}")
        
        # Create and store the new position
        position = Position(
            symbol=symbol.upper(),
            quantity=quantity,
            entry_price=entry_price,
            current_price=current_price,
            stop_loss=Decimal(str(stop_loss)) if stop_loss is not None else None,
            take_profit=Decimal(str(take_profit)) if take_profit is not None else None,
            meta=meta or {}
        )
        
        self.positions[symbol.upper()] = position
        logger.info(f"Added new position: {position}")
        return position
        
    def update_prices(self, price_updates: Dict[str, Union[float, str, Decimal]]):
        """
        Update current prices for positions.
        
        Args:
            price_updates: Dictionary of symbol -> new_price
        """
        for symbol, price in price_updates.items():
            symbol = symbol.upper()
            if symbol in self.positions:
                self.positions[symbol].current_price = Decimal(str(price))
                self.positions[symbol].meta['last_updated'] = datetime.utcnow()
                
                # Check for stop loss/take profit hits
                self._check_exit_conditions(symbol)
                
    def _check_exit_conditions(self, symbol: str) -> bool:
        """
        Check if exit conditions (stop loss/take profit) are met for a position.
        
        Args:
            symbol: Symbol to check
            
        Returns:
            bool: True if position was closed due to exit condition
        """
        if symbol not in self.positions:
            return False
            
        position = self.positions[symbol]
        current_price = position.current_price
        
        # Check stop loss
        if position.stop_loss is not None:
            if (position.is_long and current_price <= position.stop_loss) or \
               (position.is_short and current_price >= position.stop_loss):
                logger.info(f"Stop loss triggered for {symbol} at {current_price}")
                self.close_position(symbol, reason="stop_loss")
                return True
        
        # Check take profit
        if position.take_profit is not None:
            if (position.is_long and current_price >= position.take_profit) or \
               (position.is_short and current_price <= position.take_profit):
                logger.info(f"Take profit triggered for {symbol} at {current_price}")
                self.close_position(symbol, reason="take_profit")
                return True
                
        return False
    
    def close_position(
        self, 
        symbol: str, 
        price: Optional[Union[float, str, Decimal]] = None,
        reason: Optional[str] = None
    ) -> Optional[Position]:
        """
        Close an open position.
        
        Args:
            symbol: Symbol of position to close
            price: Optional exit price (uses current price if None)
            reason: Reason for closing (for logging)
            
        Returns:
            The closed Position, or None if no position was open
        """
        symbol = symbol.upper()
        if symbol not in self.positions:
            return None
            
        position = self.positions[symbol]
        
        # Update to the exit price if provided
        if price is not None:
            position.current_price = Decimal(str(price))
        
        # Log the position close
        pnl = position.unrealized_pnl
        pnl_pct = position.unrealized_pnl_pct
        
        logger.info(
            f"Closing {symbol} position: "
            f"PnL={pnl:.2f} ({pnl_pct:.2f}%) "
            f"(Reason: {reason or 'manual'})"
        )
        
        # Remove the position
        closed_position = self.positions.pop(symbol)
        
        # Update account balance with realized P&L
        self.account_balance += pnl
        
        return closed_position
    
    def get_portfolio_value(self) -> Decimal:
        """
        Calculate total portfolio value (account balance + open positions P&L).
        
        Returns:
            Total portfolio value in account currency
        """
        total_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        return self.account_balance + total_pnl
    
    def get_exposure(self) -> Dict[str, Any]:
        """
        Calculate current exposure metrics with comprehensive position data.
        
        Returns:
            Dictionary with exposure metrics:
            - total_value: Total portfolio value (account + unrealized P&L)
            - total_exposure: Total notional exposure (absolute value of all positions)
            - net_exposure: Net notional exposure (sum of all positions)
            - exposure_ratio: Total exposure / Portfolio value
            - exposure_per_symbol: Detailed exposure per symbol
            - risk_metrics: Various risk metrics
        """
        total_value = self.get_portfolio_value()
        total_exposure = Decimal('0')
        net_exposure = Decimal('0')
        long_exposure = Decimal('0')
        short_exposure = Decimal('0')
        
        exposure_per_symbol = {}
        
        for symbol, position in self.positions.items():
            exposure = position.market_value
            exposure_data = position.to_dict()
            exposure_data['exposure'] = float(abs(exposure))
            exposure_per_symbol[symbol] = exposure_data
            
            total_exposure += abs(exposure)
            net_exposure += exposure
            
            if position.is_long:
                long_exposure += exposure
            else:
                short_exposure += abs(exposure)
        
        # Calculate exposure ratios
        exposure_ratio = (total_exposure / total_value) if total_value > 0 else Decimal('0')
        long_ratio = (long_exposure / total_value) if total_value > 0 else Decimal('0')
        short_ratio = (short_exposure / total_value) if total_value > 0 else Decimal('0')
        
        # Calculate risk metrics
        risk_metrics = {
            'var_95': None,  # Would require historical data
            'max_drawdown': None,  # Would require historical data
            'exposure_ratio': float(exposure_ratio),
            'long_exposure_ratio': float(long_ratio),
            'short_exposure_ratio': float(short_ratio),
            'position_count': len(self.positions),
        }
        
        return {
            'total_value': float(total_value),
            'total_exposure': float(total_exposure),
            'net_exposure': float(net_exposure),
            'exposure_ratio': float(exposure_ratio),
            'exposure_per_symbol': exposure_per_symbol,
            'risk_metrics': risk_metrics,
            'timestamp': datetime.utcnow().isoformat(),
        }
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """
        Get position by symbol.
        
        Args:
            symbol: Symbol to look up
            
        Returns:
            Position if found, None otherwise
        """
        return self.positions.get(symbol.upper())
    
    def get_positions(self) -> Dict[str, Position]:
        """
        Get all open positions.
        
        Returns:
            Dictionary of symbol -> Position
        """
        return self.positions.copy()
    
    def update_position(
        self,
        symbol: str,
        quantity: Optional[Union[float, str, Decimal]] = None,
        current_price: Optional[Union[float, str, Decimal]] = None,
        stop_loss: Optional[Union[float, str, Decimal]] = None,
        take_profit: Optional[Union[float, str, Decimal]] = None,
        meta: Optional[dict] = None,
        validate: bool = True,
        current_time: Optional[datetime] = None,
    ) -> Optional[Position]:
        """
        Update an existing position.
        
        Args:
            symbol: Symbol of position to update
            quantity: New quantity (None to keep current)
            current_price: New current price (None to keep current)
            stop_loss: New stop loss (None to keep current, np.nan to remove)
            take_profit: New take profit (None to keep current, np.nan to remove)
            meta: Metadata updates (shallow merged with existing)
            validate: Whether to validate the update
            current_time: Current time for validation
            
        Returns:
            Updated Position if found, None otherwise
        """
        symbol = symbol.upper()
        if symbol not in self.positions:
            return None
            
        position = self.positions[symbol]
        
        # Update fields if provided
        if quantity is not None:
            position.quantity = Decimal(str(quantity))
            
        if current_price is not None:
            position.current_price = Decimal(str(current_price))
            
        if stop_loss is not None:
            position.stop_loss = Decimal(str(stop_loss)) if not isinstance(stop_loss, float) or not np.isnan(stop_loss) else None
            
        if take_profit is not None:
            position.take_profit = Decimal(str(take_profit)) if not isinstance(take_profit, float) or not np.isnan(take_profit) else None
        
        # Update metadata
        if meta is not None:
            position.meta.update(meta)
        
        # Validate the updated position if requested
        if validate:
            current_positions = {
                sym: {'quantity': pos.quantity, 'price': pos.current_price}
                for sym, pos in self.positions.items()
                if sym != symbol  # Exclude current position from validation
            }
            
            is_valid, reason = self.validator.validate_new_position(
                symbol=symbol,
                quantity=position.quantity,
                price=position.current_price,
                current_positions=current_positions,
                account_balance=float(self.account_balance),
                current_time=current_time,
            )
            
            if not is_valid:
                raise ValidationError(f"Cannot update position: {reason}")
        
        position.meta['last_updated'] = datetime.utcnow()
        return position
