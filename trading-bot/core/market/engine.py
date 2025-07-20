"""
Trading engine that respects market hours and holidays.

This module provides the TradingEngine class which manages trading operations
while respecting market hours and holidays. It handles order submission,
market status checks, and integrates with various components like data handlers,
risk managers, and execution handlers.
"""
from __future__ import annotations

import json
import logging
import queue
import time
from datetime import datetime, time, timezone, date
from typing import Dict, List, Optional, Type, Any, Protocol, runtime_checkable, Tuple, Union
from dataclasses import dataclass, field
import time
import uuid
import threading
from enum import Enum, auto
from collections import defaultdict

from core.instruments import InstrumentMetadata
from core.calendar import MarketCalendar, CalendarError, CalendarFactory

# Define protocol for execution handler
@runtime_checkable
class ExecutionHandler(Protocol):
    """Protocol for execution handlers."""
    def execute_order(self, order: Dict[str, Any]) -> bool:
        """Execute a trading order.
        
        Args:
            order: Dictionary containing order details
            
        Returns:
            bool: True if order execution was successful, False otherwise
        """
        ...

# Define protocol for risk manager
@runtime_checkable
class RiskManager(Protocol):
    """Protocol for risk management components."""
    def check_order_risk(self, order: Dict[str, Any]) -> bool:
        """Check if an order passes all risk checks.
        
        Args:
            order: Order details to check
            
        Returns:
            bool: True if order passes all risk checks, False otherwise
        """
        ...

# Define protocol for data handler
@runtime_checkable
class DataHandler(Protocol):
    """Protocol for market data handlers."""
    def get_latest_bar(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get the latest market data bar for a symbol.
        
        Args:
            symbol: Trading symbol to get data for
            
        Returns:
            Optional dictionary with market data or None if not available
        """
        ...

# Define protocol for portfolio management
@runtime_checkable
class PortfolioManager(Protocol):
    """Protocol for portfolio management components."""
    def update_portfolio(self, market_data: Dict[str, Any]) -> None:
        """Update portfolio with latest market data.
        
        Args:
            market_data: Latest market data update
        """
        ...

logger = logging.getLogger(__name__)

@dataclass
class TradingStatus:
    """Data class representing trading status for an instrument."""
    symbol: str
    is_market_open: bool
    time_to_next_session: float
    next_session_start: Optional[datetime]
    current_time: str
    timezone: str


class TradingEngine:
    """Trading engine that manages order execution respecting market hours and holidays.
    
    The TradingEngine coordinates between different components of the trading system:
    - Market data handling
    - Order execution
    - Portfolio management
    - Risk management
    - Market calendar awareness
    
    It ensures that all trading activities respect market hours and holidays,
    and provides a clean interface for strategy implementation.
    """
    
    def __init__(
        self,
        data_handler: Optional[DataHandler] = None,
        execution_handler: Optional[ExecutionHandler] = None,
        portfolio: Optional[PortfolioManager] = None,
        risk_manager: Optional[RiskManager] = None,
        calendar: Optional[MarketCalendar] = None,
        timezone: str = "UTC"
    ) -> None:
        """Initialize the trading engine with required components.
        
        Args:
            data_handler: Component responsible for providing market data.
            execution_handler: Handler for order execution with brokers/exchanges.
            portfolio: Manager for tracking and updating portfolio state.
            risk_manager: Component for performing risk checks on orders.
            calendar: Market calendar instance. If not provided, a default
                     forex calendar will be used.
            timezone: Timezone for the trading engine (default: "UTC").
                     
        Raises:
            ValueError: If required components are not provided or invalid.
            CalendarError: If there's an issue initializing the market calendar.
        """
        self.data_handler = data_handler
        self.execution_handler = execution_handler
        self.portfolio = portfolio
        self.risk_manager = risk_manager
        self.timezone = timezone
        
        try:
            # Initialize market calendar
            self.calendar = calendar or CalendarFactory.get_calendar(
                asset_class="forex",  # Default, can be overridden per instrument
                timezone=timezone
            )
            
            # Track instruments and their specific calendars
            self.instruments: Dict[str, InstrumentMetadata] = {}
            self.instrument_calendars: Dict[str, MarketCalendar] = {}
            
            # Trading state
            self.is_running = False
            self._shutdown_requested = False
            
            logger.info(
                "TradingEngine initialized with timezone: %s",
                timezone
            )
            
        except CalendarError as e:
            logger.critical("Failed to initialize market calendar: %s", e)
            raise
        except Exception as e:
            logger.critical("Unexpected error initializing TradingEngine: %s", e)
            raise
    
    def add_instrument(self, instrument: InstrumentMetadata) -> None:
        """Add an instrument to the trading engine.
        
        This method registers a new trading instrument with the engine, setting up
        the appropriate market calendar based on the instrument's asset class.
        
        Args:
            instrument: The instrument metadata to add to the trading engine.
            
        Raises:
            ValueError: If the instrument is invalid or already exists.
            CalendarError: If there's an issue creating the instrument's market calendar.
            
        Example:
            >>> metadata = InstrumentMetadata(symbol="AAPL", ...)
            >>> engine.add_instrument(metadata)
        """
        if not instrument or not instrument.symbol:
            raise ValueError("Invalid instrument: missing symbol")
            
        symbol = instrument.symbol
        is_new_instrument = symbol not in self.instruments
        
        if not is_new_instrument:
            logger.warning(
                "Instrument with symbol %s is already registered",
                symbol
            )
            return  # Don't proceed with calendar registration for existing instruments
            
        self.instruments[symbol] = instrument
        
        # Register the instrument with the calendar
        self.calendar.add_instrument(instrument)
        
        # Get or create a calendar for this instrument
        try:
            calendar = CalendarFactory.get_calendar_for_instrument(instrument)
            self.instrument_calendars[symbol] = calendar
            logger.info(
                "Added instrument %s with %s calendar",
                symbol,
                instrument.asset_class or 'default'
            )
            
        except CalendarError as e:
            logger.warning(
                "Failed to create calendar for %s: %s. Using default calendar.",
                symbol,
                str(e)
            )
            self.instrument_calendars[symbol] = self.calendar
            
        except Exception as e:
            logger.error(
                "Unexpected error adding instrument %s: %s",
                symbol,
                str(e),
                exc_info=True
            )
            # Clean up if there was an error
            if symbol in self.instruments:
                del self.instruments[symbol]
            if symbol in self.instrument_calendars:
                del self.instrument_calendars[symbol]
            raise
    
    def remove_instrument(self, symbol: str) -> bool:
        """Remove an instrument from the trading engine.
        
        This method removes an instrument and its associated calendar from the engine.
        Any pending orders or positions for this instrument should be handled separately.
        
        Args:
            symbol: The symbol of the instrument to remove.
            
        Returns:
            bool: True if the instrument was found and removed, False otherwise.
            
        Example:
            >>> engine.remove_instrument("AAPL")
            True
        """
        if not symbol:
            logger.warning("Attempted to remove instrument with empty symbol")
            return False
            
        instrument_removed = symbol in self.instruments
        calendar_removed = symbol in self.instrument_calendars
        
        # Remove from calendar first
        if instrument_removed or calendar_removed:
            self.calendar.remove_instrument(symbol)
            
        if instrument_removed:
            del self.instruments[symbol]
            logger.debug("Removed instrument: %s", symbol)
            
        if calendar_removed:
            del self.instrument_calendars[symbol]
            logger.debug("Removed calendar for instrument: %s", symbol)
            
        if not instrument_removed and not calendar_removed:
            logger.debug("Instrument %s not found for removal", symbol)
            return False
            
        return True
    
    def get_instrument(self, symbol: str) -> Optional[InstrumentMetadata]:
        """Get an instrument by its symbol.
        
        Args:
            symbol: The trading symbol of the instrument to retrieve.
                
        Returns:
            Optional[InstrumentMetadata]: The instrument metadata if found, None otherwise.
            
        Example:
            >>> instrument = engine.get_instrument("AAPL")
            >>> if instrument:
            ...     print(f"Found instrument: {instrument.name}")
        """
        if not symbol:
            logger.warning("Empty symbol provided to get_instrument")
            return None
            
        return self.instruments.get(symbol)
    
    def has_instrument(self, symbol: str) -> bool:
        """Check if an instrument with the given symbol exists.
        
        Args:
            symbol: The trading symbol to check.
                
        Returns:
            bool: True if the instrument exists, False otherwise.
            
        Example:
            >>> if engine.has_instrument("AAPL"):
            ...     print("AAPL is registered")
        """
        if not symbol:
            return False
            
        return symbol in self.instruments
    
    def list_instruments(self) -> List[str]:
        """Get a list of all registered instrument symbols.
        
        Returns:
            List[str]: A list of all registered instrument symbols.
            
        Example:
            >>> symbols = engine.list_instruments()
            >>> print(f"Registered instruments: {', '.join(symbols)}")
        """
        return list(self.instruments.keys())
    
    def is_market_open(
        self, 
        symbol: str, 
        dt: Optional[datetime] = None
    ) -> bool:
        """Check if the market is currently open for the specified instrument.
        
        This method checks if the given datetime (or current time if not provided)
        falls within the trading hours for the specified instrument, taking into
        account market holidays and special trading sessions.
        
        Args:
            symbol: The trading symbol of the instrument to check.
            dt: Optional datetime to check (in UTC). If not provided, uses current time.
                
        Returns:
            bool: True if the market is open for trading, False otherwise.
            
        Raises:
            ValueError: If the symbol is not registered with the engine.
            
        Example:
            >>> engine.is_market_open("AAPL")
            True
            >>> import datetime
            >>> dt = datetime.datetime(2023, 12, 25, tzinfo=datetime.timezone.utc)
            >>> engine.is_market_open("AAPL", dt=dt)
            False  # Christmas Day
        """
        if not symbol:
            logger.warning("Empty symbol provided to is_market_open")
            return False
            
        if symbol not in self.instruments and symbol not in self.instrument_calendars:
            logger.warning("Symbol %s not found in registered instruments", symbol)
            return False
            
        dt = dt or datetime.now(timezone.utc)
        
        # Get the appropriate calendar - use instrument-specific if available
        calendar = self.instrument_calendars.get(symbol, self.calendar)
        
        try:
            is_open = calendar.is_session_open(dt)
            logger.debug(
                "Market status for %s at %s: %s",
                symbol,
                dt.isoformat(),
                "OPEN" if is_open else "CLOSED"
            )
            return is_open
            
        except CalendarError as e:
            logger.error(
                "Calendar error checking market status for %s: %s",
                symbol,
                str(e)
            )
            return False
            
        except Exception as e:
            logger.error(
                "Unexpected error checking market status for %s: %s",
                symbol,
                str(e),
                exc_info=True
            )
            return False
    
    def time_to_next_session(
        self, 
        symbol: str, 
        dt: Optional[datetime] = None
    ) -> float:
        """Calculate time remaining until the next trading session starts.
        
        This method determines how long until the next trading session begins for
        the specified instrument. If the market is currently open, returns 0.
        
        Args:
            symbol: The trading symbol of the instrument to check.
            dt: Optional datetime to check from (in UTC). If not provided, uses current time.
                
        Returns:
            float: Number of seconds until the next trading session starts. Returns 0 if
                 the market is currently open.
                 
        Raises:
            ValueError: If the symbol is not registered with the engine.
            
        Example:
            >>> seconds = engine.time_to_next_session("AAPL")
            >>> if seconds > 0:
            ...     print(f"Market opens in {seconds/3600:.1f} hours")
            ... else:
            ...     print("Market is currently open")
        """
        if not symbol:
            raise ValueError("Symbol cannot be empty")
            
        if symbol not in self.instruments and symbol not in self.instrument_calendars:
            raise ValueError(f"No instrument found with symbol: {symbol}")
            
        dt = dt or datetime.now(timezone.utc)
        
        # Get the appropriate calendar - use instrument-specific if available
        calendar = self.instrument_calendars.get(symbol, self.calendar)
        
        try:
            # Check if market is currently open
            if calendar.is_session_open(dt):
                return 0.0
                
            # Get the next session start time
            next_session = calendar.next_session(symbol=symbol, dt=dt)
            
            if not next_session or not isinstance(next_session, datetime):
                logger.warning("No valid next session found for %s", symbol)
                return 0.0
                
            # Calculate time difference in seconds
            time_diff = (next_session - dt).total_seconds()
            
            # Ensure we don't return negative values
            return max(0.0, time_diff)
            
        except CalendarError as e:
            logger.error(
                "Calendar error calculating time to next session for %s: %s",
                symbol,
                str(e)
            )
            raise
            
        except Exception as e:
            logger.error(
                "Unexpected error calculating time to next session for %s: %s",
                symbol,
                str(e),
                exc_info=True
            )
            raise
    
    def get_trading_status(self, symbol: str) -> TradingStatus:
        """Get the current trading status for an instrument.
        
        This method provides detailed information about the current trading status
        of the specified instrument, including whether the market is open and
        when the next trading session begins.
        
        Args:
            symbol: The trading symbol of the instrument to check.
                
        Returns:
            TradingStatus: An object containing the trading status details.
            
        Raises:
            ValueError: If the symbol is not registered with the engine.
            
        Example:
            >>> status = engine.get_trading_status("AAPL")
            >>> print(f"Market is {'OPEN' if status.is_market_open else 'CLOSED'}")
            >>> if not status.is_market_open:
            ...     print(f"Next session in {status.time_to_next_session/3600:.1f} hours")
        """
        if not symbol:
            raise ValueError("Symbol cannot be empty")
            
        if symbol not in self.instruments and symbol not in self.instrument_calendars:
            raise ValueError(f"No instrument found with symbol: {symbol}")
            
        dt = datetime.now(timezone.utc)
        
        # Get the appropriate calendar - use instrument-specific if available
        calendar = self.instrument_calendars.get(symbol, self.calendar)
        
        try:
            # Check if market is open
            is_open = calendar.is_session_open(dt)
            
            # Get next session start time
            next_session = None
            time_to_next = 0.0
            
            if not is_open:
                next_session = calendar.next_session(symbol=symbol, dt=dt)
                if next_session and isinstance(next_session, datetime):
                    time_to_next = (next_session - dt).total_seconds()
                    time_to_next = max(0.0, time_to_next)
            
            # Create and return status object
            return TradingStatus(
                symbol=symbol,
                is_market_open=is_open,
                time_to_next_session=time_to_next,
                next_session_start=next_session,
                current_time=dt.isoformat(),
                timezone=self.timezone
            )
            
        except CalendarError as e:
            logger.error(
                "Calendar error getting trading status for %s: %s",
                symbol,
                str(e)
            )
            raise
            
        except Exception as e:
            logger.error(
                "Unexpected error getting trading status for %s: %s",
                symbol,
                str(e),
                exc_info=True
            )
            raise
    
    def get_market_hours(
        self, 
        symbol: str, 
        date: Optional[date] = None
    ) -> Tuple[datetime, datetime]:
        """Get the market open and close times for a specific date.
        
        This method retrieves the scheduled market open and close times for the
        specified instrument and date.
        
        Args:
            symbol: The trading symbol of the instrument to check.
            date: The date to get market hours for. If not provided, uses current date.
                
        Returns:
            Tuple[datetime, datetime]: A tuple of (market_open, market_close) times.
                Both times are timezone-aware datetime objects in UTC.
                
        Raises:
            ValueError: If the symbol is not registered or no market hours available.
            
        Example:
            >>> market_open, market_close = engine.get_market_hours("AAPL")
            >>> print(f"Market hours: {market_open} to {market_close}")
        """
        if not symbol:
            raise ValueError("Symbol cannot be empty")
            
        if symbol not in self.instruments and symbol not in self.instrument_calendars:
            raise ValueError(f"No instrument found with symbol: {symbol}")
            
        date = date or datetime.now(timezone.utc).date()
        
        # Get the appropriate calendar - use instrument-specific if available
        calendar = self.instrument_calendars.get(symbol, self.calendar)
        
        try:
            market_open, market_close = calendar.get_market_hours(symbol=symbol, date=date)
            
            if not isinstance(market_open, datetime) or not isinstance(market_close, datetime):
                raise ValueError("Invalid market hours returned from calendar")
                
            # Ensure timezone awareness
            if market_open.tzinfo is None:
                market_open = market_open.replace(tzinfo=timezone.utc)
            if market_close.tzinfo is None:
                market_close = market_close.replace(tzinfo=timezone.utc)
                
            return market_open, market_close
            
        except CalendarError as e:
            logger.error(
                "Calendar error getting market hours for %s on %s: %s",
                symbol,
                date.isoformat(),
                str(e)
            )
            raise
            
        except Exception as e:
            logger.error(
                "Unexpected error getting market hours for %s on %s: %s",
                symbol,
                date.isoformat(),
                str(e),
                exc_info=True
            )
            raise
    
    def time_to_next_session(
        self, 
        symbol: str, 
        dt: Optional[datetime] = None
    ) -> float:
        """Calculate time remaining until the next trading session starts.
        
        This method determines how long until the next trading session begins for
        the specified instrument. If the market is currently open, returns 0.
        
        Args:
            symbol: The trading symbol of the instrument to check.
            dt: Optional reference datetime (in UTC). If not provided, uses current time.
                
        Returns:
            float: Number of seconds until the next trading session starts.
                  Returns 0 if the market is currently open.
                  Returns float('inf') if there's an error or the symbol is invalid.
            
        Example:
            >>> # At 4:00 PM ET on a weekday (after market close)
            >>> engine.time_to_next_session("SPY")
            54000.0  # 15 hours until next market open (9:30 AM ET next day)
        """
        if not symbol:
            logger.warning("Empty symbol provided to time_to_next_session")
            return float('inf')
            
        if symbol not in self.instruments and symbol not in self.instrument_calendars:
            logger.warning("Symbol %s not found in registered instruments", symbol)
            return float('inf')
            
        dt = dt or datetime.now(timezone.utc)
        calendar = self.instrument_calendars.get(symbol, self.calendar)
        
        try:
            # If market is currently open, return 0
            if calendar.is_session_open(dt):
                return 0.0
                
            # Get next trading day and its market hours
            next_session_date = calendar.next_trading_day(dt)
            try:
                market_open, _ = calendar.get_market_hours(next_session_date)
                # Calculate time difference in seconds
                time_until_open = (market_open - dt).total_seconds()
            except (ValueError, AttributeError) as e:
                logger.warning(
                    "Could not get market hours for %s on %s: %s",
                    symbol, next_session_date, str(e)
                )
                # Fallback: assume market opens at 9:30 AM in the calendar's timezone
                market_open = calendar.timezone.localize(
                    datetime.combine(next_session_date, time(9, 30))
                ).astimezone(timezone.utc)
                time_until_open = (market_open - dt).total_seconds()
            
            logger.debug(
                "Next session for %s starts in %.1f hours (%s)",
                symbol,
                time_until_open / 3600,
                next_session_open.isoformat()
            )
            
            return max(0.0, time_until_open)  # Ensure non-negative
            
        except Exception as e:
            logger.error(
                "Error calculating time to next session for %s: %s",
                symbol, str(e),
                exc_info=True
            )
            return float('inf')
    
    def get_market_hours(
        self, 
        symbol: str, 
        date: Optional[date] = None
    ) -> Tuple[datetime, datetime]:
        """Get the market open and close times for a specific date.
        
        This method retrieves the scheduled market open and close times for the
        specified instrument and date.
        
        Args:
            symbol: The trading symbol of the instrument to check.
            date: The date to get market hours for. If not provided, uses current date.
                
        Returns:
            Tuple[datetime, datetime]: A tuple of (market_open, market_close) times.
                Both times are timezone-aware datetime objects in UTC.
                
        Raises:
            ValueError: If the symbol is not registered or no market hours available.
            
        Example:
            >>> market_open, market_close = engine.get_market_hours("AAPL")
            >>> print(f"Market hours: {market_open} to {market_close}")
        """
        if not symbol:
            raise ValueError("Symbol cannot be empty")
            
        if symbol not in self.instruments and symbol not in self.instrument_calendars:
            raise ValueError(f"No instrument found with symbol: {symbol}")
            
        date = date or datetime.now(timezone.utc).date()
        
        # Get the appropriate calendar - use instrument-specific if available
        calendar = self.instrument_calendars.get(symbol, self.calendar)
        
        try:
            market_open, market_close = calendar.get_market_hours(symbol=symbol, date=date)
            
            if not isinstance(market_open, datetime) or not isinstance(market_close, datetime):
                raise ValueError("Invalid market hours returned from calendar")
                
            # Ensure timezone awareness
            if market_open.tzinfo is None:
                market_open = market_open.replace(tzinfo=timezone.utc)
            if market_close.tzinfo is None:
                market_close = market_close.replace(tzinfo=timezone.utc)
                
            return market_open, market_close
            
        except CalendarError as e:
            logger.error(
                "Calendar error getting market hours for %s on %s: %s",
                symbol,
                date.isoformat(),
                str(e)
            )
            raise
            
        except Exception as e:
            logger.error(
                "Unexpected error getting market hours for %s on %s: %s",
                symbol,
                date.isoformat(),
                str(e),
                exc_info=True
            )
            raise
            
        except (KeyError, IndexError) as e:
            logger.error(
                "Error finding next session for %s: %s",
                symbol,
                str(e)
            )
            return float('inf')
            
        except CalendarError as e:
            logger.error(
                "Calendar error for %s: %s",
                symbol,
                str(e)
            )
            return float('inf')
            
        except Exception as e:
            logger.error(
                "Unexpected error calculating next session for %s: %s",
                symbol,
                str(e),
                exc_info=True
            )
            return float('inf')
    
    class OrderError(Exception):
        """Base exception for order-related errors."""
        pass

    class OrderValidationError(OrderError):
        """Raised when an order fails validation."""
        pass

    class OrderRejectedError(OrderError):
        """Raised when an order is rejected by the risk manager or exchange."""
        pass

    class OrderExecutionError(OrderError):
        """Raised when there's an error executing an order."""
        pass

    def _validate_order_parameters(self, order: Dict[str, Any]) -> None:
        """Validate order parameters.
        
        Args:
            order: The order to validate.
            
        Raises:
            OrderValidationError: If the order is invalid.
        """
        if not isinstance(order, dict):
            raise self.OrderValidationError("Order must be a dictionary")
            
        if not order.get('symbol'):
            raise self.OrderValidationError("Order must specify a symbol")
            
        if not isinstance(order.get('quantity'), (int, float)) or order['quantity'] <= 0:
            raise self.OrderValidationError("Order quantity must be a positive number")
            
        valid_order_types = {'market', 'limit', 'stop', 'stop_limit', 'trailing_stop'}
        if order.get('order_type') not in valid_order_types:
            raise self.OrderValidationError(
                f"Invalid order type. Must be one of: {', '.join(valid_order_types)}"
            )
            
        valid_sides = {'buy', 'sell'}
        if order.get('side') not in valid_sides:
            raise self.OrderValidationError(
                f"Invalid order side. Must be one of: {', '.join(valid_sides)}"
            )
            
        if order.get('order_type') in {'limit', 'stop_limit'} and 'price' not in order:
            raise self.OrderValidationError("Price is required for limit and stop-limit orders")
            
        if 'price' in order and (not isinstance(order['price'], (int, float)) or order['price'] <= 0):
            raise self.OrderValidationError("Price must be a positive number")

    def _enrich_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Enrich order with additional metadata.
        
        Args:
            order: The order to enrich.
            
        Returns:
            The enriched order dictionary.
        """
        enriched = order.copy()
        enriched['timestamp'] = datetime.now(timezone.utc).isoformat()
        enriched['order_id'] = str(uuid.uuid4())
        return enriched

    def _execute_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an order through the execution handler.
        
        Args:
            order: The order to execute.
            
        Returns:
            The execution result.
            
        Raises:
            OrderExecutionError: If there's an error executing the order.
        """
        if not self.execution_handler:
            raise self.OrderExecutionError("No execution handler configured")
            
        try:
            result = self.execution_handler.execute_order(order)
            if not result or not result.get('order_id'):
                raise self.OrderExecutionError("Invalid response from execution handler")
            return result
        except Exception as e:
            raise self.OrderExecutionError(f"Failed to execute order: {str(e)}") from e

    def _check_risk_limits(self, order: Dict[str, Any]) -> bool:
        """Check order against risk limits.
        
        Args:
            order: The order to check.
            
        Returns:
            bool: True if the order passes risk checks, False otherwise.
            
        Raises:
            OrderRejectedError: If the order is rejected by the risk manager.
        """
        if not self.risk_manager:
            return True
            
        try:
            if not self.risk_manager.check_order_risk(order):
                raise self.OrderRejectedError("Order rejected by risk manager")
            return True
        except Exception as e:
            raise self.OrderRejectedError(f"Risk check failed: {str(e)}") from e

    def _check_market_hours(self, symbol: str) -> None:
        """Check if the market is open for trading.
        
        Args:
            symbol: The symbol to check.
            
        Raises:
            OrderRejectedError: If the market is closed.
        """
        if not self.is_market_open(symbol):
            time_to_open = self.time_to_next_session(symbol)
            raise self.OrderRejectedError(
                f"Market is closed for {symbol}. "
                f"Next session in {time_to_open/3600:.1f} hours."
            )

    def submit_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Submit a trading order to the execution system.
        
        This method handles the complete order submission workflow including:
        1. Input validation
        2. Market hours verification
        3. Risk management checks
        4. Order execution
        
        Args:
            order: Dictionary containing order details with the following keys:
                - symbol: The trading symbol (required)
                - quantity: Number of shares/contracts (required, must be positive)
                - order_type: Type of order (required, one of: 'market', 'limit', 'stop', 'stop_limit')
                - side: 'buy' or 'sell' (required)
                - price: Required for limit and stop-limit orders
                - time_in_force: Order time in force (e.g., 'day', 'gtc')
                - Additional broker-specific parameters
                
        Returns:
            Dict containing the order execution result with at least:
                - order_id: Unique identifier for the order
                - status: Current status of the order
                - symbol: The trading symbol
                - timestamp: When the order was submitted
                - Additional broker-specific fields
            
        Raises:
            OrderValidationError: If required order parameters are missing or invalid.
            OrderRejectedError: If the order is rejected by risk management or market is closed.
            OrderExecutionError: If there's an error executing the order.
            RuntimeError: If the trading engine is not running or not properly configured.
            
        Example:
            >>> order = {
            ...     'symbol': 'AAPL',
            ...     'quantity': 100,
            ...     'order_type': 'limit',
            ...     'side': 'buy',
            ...     'price': 150.0,
            ...     'time_in_force': 'day'
            ... }
            >>> result = engine.submit_order(order)
            >>> print(f"Order {result['order_id']} submitted: {result['status']}")
        """
        if not self.is_running:
            raise RuntimeError("Trading engine is not running")
            
        logger.info("Submitting order", extra={'order': order})
        
        try:
            # 1. Validate order parameters
            self._validate_order_parameters(order)
            
            # 2. Check market hours
            self._check_market_hours(order['symbol'])
            
            # 3. Enrich order with metadata
            enriched_order = self._enrich_order(order)
            
            # 4. Check risk limits
            self._check_risk_limits(enriched_order)
            
            # 5. Execute the order
            result = self._execute_order(enriched_order)
            
            logger.info("Order submitted successfully", 
                       extra={'order_id': result['order_id'], 'status': result.get('status')})
            return result
            
        except (self.OrderValidationError, self.OrderRejectedError, self.OrderExecutionError) as e:
            logger.warning(f"Order submission failed: {str(e)}", 
                         extra={'order': order, 'error': str(e)})
            raise
        except Exception as e:
            logger.error("Unexpected error submitting order", 
                        extra={'order': order}, exc_info=True)
            raise self.OrderExecutionError(f"Unexpected error: {str(e)}") from e
    
    def start(self) -> None:
        """Start the trading engine and begin processing events.
        
        This method initializes the trading engine's main event loop, which:
        1. Processes incoming market data
        2. Handles order execution
        3. Manages risk controls
        4. Updates portfolio positions
        
        The method runs in the current thread and will block until stop() is called.
        
        Raises:
            RuntimeError: If the engine is already running or if required
                         components are not initialized.
        """
        if self.is_running:
            logger.warning("Trading engine is already running")
            return
            
        # Validate required components
        if not self.execution_handler:
            error_msg = "Cannot start trading: No execution handler configured"
            logger.critical(error_msg)
            raise RuntimeError(error_msg)
            
        if not self.data_handler:
            logger.warning("No data handler configured - running in order-only mode")
            
        self.is_running = True
        self._shutdown_requested = False
        
        logger.info("Starting trading engine with timezone: %s", self.timezone)
        
        # Main trading loop
        loop_count = 0
        while not self._shutdown_requested:
            loop_start = time.time()
            loop_count += 1
            
            try:
                # Process pending events
                self._process_events()
                
                # Log status periodically
                if loop_count % 60 == 0:  # Every ~minute at normal loop speed
                    self._log_engine_status()
                
                # Calculate time for consistent loop timing
                loop_time = time.time() - loop_start
                sleep_time = max(0, 1.0 - loop_time)  # Target 1 second per loop
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                    
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received, shutting down...")
                self.stop()
                
            except Exception as e:
                logger.error(
                    "Error in trading loop (attempt %d): %s",
                    loop_count,
                    str(e),
                    exc_info=True
                )
                # Add increasing delay on repeated errors
                error_delay = min(5, loop_count / 10)  # Max 5 seconds
                time.sleep(error_delay)
                
                # If we've had many errors, consider shutting down
                if loop_count > 100 and (loop_count % 10) == 0:
                    logger.critical(
                        "Multiple errors detected. Consider shutting down and investigating."
                    )
        
        logger.info("Trading engine main loop ended")
    
    def stop(self) -> None:
        """Stop the trading engine."""
        self._shutdown_requested = True
        self.is_running = False
        logger.info("Trading engine stopped")
    
    def _process_events(self) -> None:
        """Process pending events (orders, market data, etc.)."""
        # This is a placeholder. In a real implementation, this would:
        # 1. Process any pending orders
        # 2. Update portfolio with latest market data
        # 3. Run strategy logic
        # 4. Handle any risk management checks
        pass
    
    def get_trading_status(self, symbol: str) -> Dict[str, Any]:
        """Get trading status for an instrument.
        
        Args:
            symbol: Instrument symbol
            
        Returns:
            Dict with trading status information
        """
        is_open = self.is_market_open(symbol)
        time_to_open = 0.0 if is_open else self.time_to_next_session(symbol)
        
        return {
            'symbol': symbol,
            'is_market_open': is_open,
            'time_to_next_session': time_to_open,
            'next_session_start': None,  # Could be enhanced to return actual datetime
            'current_time': datetime.now(timezone.utc).isoformat(),
            'timezone': self.timezone
        }
        
    def _process_events(self) -> None:
        """Process all pending events in the event queue.
        
        This method continuously processes events from the event queue until it's empty.
        It handles various event types including market data, order updates, and signals.
        
        Events are processed in a non-blocking manner, and any exceptions during
        event processing are caught and logged without stopping the engine.
        
        Event types handled:
        - 'MARKET_DATA': Process new market data updates
        - 'ORDER': Handle order-related events (fills, cancels, etc.)
        - 'SIGNAL': Process trading signals
        - 'ERROR': Handle error conditions
        
        Returns:
            None
            
        Example:
            >>> engine._process_events()  # Called internally by the main loop
        """
        if not hasattr(self, 'event_queue') or self.event_queue is None:
            logger.warning("Event queue not initialized")
            return
            
        processed_count = 0
        max_events_per_cycle = 1000  # Prevent event queue starvation
        
        try:
            # Process events with a limit to prevent blocking too long
            while processed_count < max_events_per_cycle:
                try:
                    # Get event with a small timeout to prevent hanging
                    event = self.event_queue.get_nowait()
                    processed_count += 1
                    
                    # Log event processing at debug level
                    logger.debug(
                        "Processing event: %s (type: %s)",
                        event.get('event_id', 'no_id'),
                        event.get('type', 'unknown')
                    )
                    
                    # Handle different event types
                    event_type = event.get('type')
                    if event_type == 'MARKET_DATA':
                        self._handle_market_data(event)
                    elif event_type == 'ORDER':
                        self._handle_order_event(event)
                    elif event_type == 'SIGNAL':
                        self._handle_signal(event)
                    elif event_type == 'ERROR':
                        self._handle_error(event)
                    else:
                        logger.warning("Unknown event type: %s", event_type)
                        
                    # Mark event as processed
                    self.event_queue.task_done()
                    
                except queue.Empty:
                    # No more events to process
                    break
                    
                except Exception as e:
                    logger.error(
                        "Error processing event: %s",
                        str(e),
                        exc_info=True
                    )
                    # Continue with next event even if one fails
                    continue
                    
            if processed_count > 0:
                logger.debug("Processed %d events", processed_count)
                
        except Exception as e:
            logger.error(
                "Unexpected error in event processing: %s",
                str(e),
                exc_info=True
            )
    
    def _handle_market_data(self, event: Dict[str, Any]) -> None:
        """Handle incoming market data events.
        
        This method processes market data updates, which may include:
        - OHLCV (Open, High, Low, Close, Volume) data
        - Order book updates
        - Trade ticks
        - Market depth updates
        
        Args:
            event: Dictionary containing market data with at least:
                - 'symbol': The trading symbol
                - 'data': The market data payload
                - 'timestamp': When the data was generated
                
        Returns:
            None
            
        Raises:
            ValueError: If required fields are missing from the event
        """
        try:
            symbol = event.get('symbol')
            data = event.get('data', {})
            
            if not symbol or not data:
                raise ValueError("Market data event missing required fields")
                
            # Update portfolio with latest market data if available
            if hasattr(self, 'portfolio') and self.portfolio is not None:
                self.portfolio.update_portfolio({
                    'symbol': symbol,
                    'data': data,
                    'timestamp': event.get('timestamp', datetime.now(timezone.utc))
                })
                
            logger.debug("Processed market data for %s", symbol)
            
        except Exception as e:
            logger.error(
                "Error processing market data for %s: %s",
                event.get('symbol', 'unknown'),
                str(e),
                exc_info=True
            )
    
    def _handle_order_event(self, event: Dict[str, Any]) -> None:
        """Handle order-related events (fills, cancels, rejects, etc.).
        
        This method processes order events from the execution handler,
        including order fills, cancellations, and rejections.
        
        Args:
            event: Dictionary containing order event details with at least:
                - 'order_id': Unique identifier for the order
                - 'status': Current status of the order
                - 'symbol': The trading symbol
                - Additional order-specific fields
                
        Returns:
            None
        """
        try:
            order_id = event.get('order_id')
            status = event.get('status')
            
            if not order_id or not status:
                raise ValueError("Order event missing required fields")
                
            # Log order status change
            logger.info(
                "Order %s status update: %s",
                order_id,
                status
            )
            
            # Update portfolio with order fill if applicable
            if status.upper() == 'FILLED' and hasattr(self, 'portfolio') and self.portfolio is not None:
                self.portfolio.update_order(event)
                
        except Exception as e:
            logger.error(
                "Error processing order event %s: %s",
                event.get('order_id', 'unknown'),
                str(e),
                exc_info=True
            )
    
    def _handle_signal(self, event: Dict[str, Any]) -> None:
        """Process trading signals generated by strategies.
        
        This method handles trading signals, which may result in new orders
        being submitted to the execution handler after passing risk checks.
        
        Args:
            event: Dictionary containing signal details with at least:
                - 'signal_type': Type of signal (e.g., 'BUY', 'SELL', 'EXIT')
                - 'symbol': The trading symbol
                - 'strength': Signal strength/confidence (0.0 to 1.0)
                - Additional signal-specific parameters
                
        Returns:
            None
        """
        try:
            signal_type = event.get('signal_type', '').upper()
            symbol = event.get('symbol')
            
            if not signal_type or not symbol:
                raise ValueError("Signal event missing required fields")
                
            logger.debug(
                "Processing %s signal for %s (strength: %.2f)",
                signal_type,
                symbol,
                float(event.get('strength', 0.0))
            )
            
            # Here you would typically convert the signal to an order
            # and submit it through the execution handler
            # For example:
            # order = self._create_order_from_signal(event)
            # if order:
            #     self.submit_order(order)
                
        except Exception as e:
            logger.error(
                "Error processing signal for %s: %s",
                event.get('symbol', 'unknown'),
                str(e),
                exc_info=True
            )
    
    def _handle_error(self, event: Dict[str, Any]) -> None:
        """Handle error events from various components.
        
        This method processes error events and takes appropriate action,
        which may include logging, alerting, or triggering recovery procedures.
        
        Args:
            event: Dictionary containing error details with at least:
                - 'error_type': Type/category of error
                - 'message': Human-readable error message
                - 'source': Component that generated the error
                - 'timestamp': When the error occurred
                - Additional error-specific fields
                
        Returns:
            None
        """
        try:
            error_type = event.get('error_type', 'UNKNOWN')
            source = event.get('source', 'unknown')
            message = event.get('message', 'No error details provided')
            
            # Log error with appropriate level based on severity
            log_level = event.get('severity', 'error').lower()
            log_method = getattr(logger, log_level, logger.error)
            
            log_method(
                "[%s] %s error in %s: %s",
                error_type.upper(),
                source.upper(),
                message,
                extra={
                    'error_details': event.get('details', {}),
                    'timestamp': event.get('timestamp', datetime.now(timezone.utc).isoformat())
                }
            )
            
            # Potentially trigger alerts or recovery procedures here
            if error_type.upper() in ['FATAL', 'CRITICAL']:
                logger.critical("Critical error detected, considering emergency shutdown")
                # self.stop()  # Uncomment to enable emergency shutdown on critical errors
                
        except Exception as e:
            # Ensure we don't crash the engine while handling errors
            logger.critical("Error in error handler: %s", str(e), exc_info=True)
    
    def _log_engine_status(self) -> None:
        """Log detailed status information about the trading engine.
        
        This method collects and logs various metrics about the engine's state,
        including performance statistics, queue depths, and component status.
        
        The status is logged at INFO level and includes:
        - Engine running state and uptime
        - Component status (data handler, execution handler, risk manager)
        - Queue statistics (event queue depth)
        - Performance metrics (event processing rates)
        - Instrument and position counts
        
        This method is typically called periodically by the main trading loop.
        
        Returns:
            None
            
        Example:
            >>> engine._log_engine_status()  # Called internally by the main loop
        """
        if not hasattr(self, 'start_time'):
            self.start_time = time.time()
            
        # Calculate uptime
        uptime_seconds = time.time() - self.start_time
        hours, remainder = divmod(uptime_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime_str = f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"
        
        # Get queue depth if available
        queue_depth = 0
        if hasattr(self, 'event_queue') and self.event_queue is not None:
            queue_depth = self.event_queue.qsize()
            
        # Get position count if available
        position_count = 0
        if hasattr(self, 'portfolio') and self.portfolio is not None:
            position_count = len(self.portfolio.positions)
            
        # Get event processing stats if available
        if not hasattr(self, '_events_processed'):
            self._events_processed = 0
            self._last_event_count_time = time.time()
            
        time_since_last = time.time() - self._last_event_count_time
        if time_since_last > 0:
            events_per_second = self._events_processed / time_since_last
        else:
            events_per_second = 0.0
            
        # Prepare status dictionary
        status = {
            'status': 'RUNNING' if self.is_running else 'STOPPED',
            'uptime': uptime_str,
            'instruments': len(self.instruments),
            'positions': position_count,
            'queue_depth': queue_depth,
            'events_per_second': round(events_per_second, 2),
            'timezone': str(self.timezone),
            'components': {
                'data_handler': self.data_handler is not None,
                'execution_handler': self.execution_handler is not None,
                'risk_manager': self.risk_manager is not None,
                'portfolio': hasattr(self, 'portfolio') and self.portfolio is not None
            },
            'last_update': datetime.now(timezone.utc).isoformat()
        }
        
        # Log the status in a structured way
        logger.info("Trading Engine Status: %s", json.dumps(status, indent=2))
        
        # Reset counters
        self._events_processed = 0
        self._last_event_count_time = time.time()
