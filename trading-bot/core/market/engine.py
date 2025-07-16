"""
Trading engine that respects market hours and holidays.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Type, Any, Callable

from core.instruments import InstrumentMetadata
from core.calendar import MarketCalendar, CalendarFactory, CalendarError

logger = logging.getLogger(__name__)


class TradingEngine:
    """Trading engine that respects market hours and holidays."""
    
    def __init__(
        self,
        data_handler: Any = None,
        execution_handler: Any = None,
        portfolio: Any = None,
        risk_manager: Any = None,
        calendar: Optional[MarketCalendar] = None,
        timezone: str = "UTC"
    ):
        """Initialize the trading engine.
        
        Args:
            data_handler: Market data handler
            execution_handler: Order execution handler
            portfolio: Portfolio management
            risk_manager: Risk management
            calendar: Market calendar (will be created if not provided)
            timezone: Default timezone for the engine
        """
        self.data_handler = data_handler
        self.execution_handler = execution_handler
        self.portfolio = portfolio
        self.risk_manager = risk_manager
        self.timezone = timezone
        
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
    
    def add_instrument(self, instrument: InstrumentMetadata) -> None:
        """Add an instrument to the trading engine.
        
        Args:
            instrument: The instrument to add
        """
        self.instruments[instrument.symbol] = instrument
        
        # Get or create a calendar for this instrument
        try:
            calendar = CalendarFactory.get_calendar_for_instrument(instrument)
            self.instrument_calendars[instrument.symbol] = calendar
            logger.info(f"Added instrument {instrument.symbol} with {instrument.asset_class} calendar")
        except Exception as e:
            logger.warning(
                f"Failed to create calendar for {instrument.symbol}: {e}. "
                "Using default calendar."
            )
            self.instrument_calendars[instrument.symbol] = self.calendar
    
    def remove_instrument(self, symbol: str) -> None:
        """Remove an instrument from the trading engine.
        
        Args:
            symbol: Symbol of the instrument to remove
        """
        if symbol in self.instruments:
            del self.instruments[symbol]
        if symbol in self.instrument_calendars:
            del self.instrument_calendars[symbol]
    
    def is_market_open(self, symbol: str, dt: Optional[datetime] = None) -> bool:
        """Check if the market is open for a specific instrument.
        
        Args:
            symbol: Instrument symbol
            dt: Optional datetime to check (defaults to now)
            
        Returns:
            bool: True if market is open, False otherwise
        """
        dt = dt or datetime.now(timezone.utc)
        
        # Get the appropriate calendar
        calendar = self.instrument_calendars.get(symbol, self.calendar)
        
        try:
            return calendar.is_session_open(dt)
        except Exception as e:
            logger.error(f"Error checking market open for {symbol}: {e}")
            return False
    
    def time_to_next_session(self, symbol: str, dt: Optional[datetime] = None) -> float:
        """Get time in seconds until the next trading session starts.
        
        Args:
            symbol: Instrument symbol
            dt: Optional reference datetime (defaults to now)
            
        Returns:
            float: Seconds until next session, or 0 if market is open
        """
        dt = dt or datetime.now(timezone.utc)
        
        # Get the appropriate calendar
        calendar = self.instrument_calendars.get(symbol, self.calendar)
        
        try:
            if calendar.is_session_open(dt):
                return 0.0
                
            next_session = calendar.next_trading_day(dt)
            next_session = datetime.combine(next_session, datetime.min.time(), tzinfo=dt.tzinfo)
            return (next_session - dt).total_seconds()
            
        except Exception as e:
            logger.error(f"Error calculating next session for {symbol}: {e}")
            return float('inf')
    
    def submit_order(self, order: Dict[str, Any]) -> bool:
        """Submit an order to the trading engine.
        
        Args:
            order: Order details including 'symbol', 'quantity', 'order_type', etc.
            
        Returns:
            bool: True if order was submitted successfully, False otherwise
        """
        symbol = order.get('symbol')
        if not symbol:
            logger.error("Order missing required 'symbol' field")
            return False
            
        # Check if market is open for this instrument
        if not self.is_market_open(symbol):
            logger.warning(f"Market is closed for {symbol}. Order not submitted.")
            return False
            
        # Check risk management
        if self.risk_manager and not self.risk_manager.check_order_risk(order):
            logger.warning(f"Order for {symbol} rejected by risk manager")
            return False
            
        # Submit order through execution handler
        if self.execution_handler:
            return self.execution_handler.execute_order(order)
            
        logger.warning("No execution handler configured. Order not submitted.")
        return False
    
    def start(self) -> None:
        """Start the trading engine."""
        if self.is_running:
            logger.warning("Trading engine is already running")
            return
            
        self.is_running = True
        self._shutdown_requested = False
        logger.info("Trading engine started")
        
        # Main trading loop
        while not self._shutdown_requested:
            try:
                self._process_events()
            except Exception as e:
                logger.error(f"Error in trading loop: {e}", exc_info=True)
                # Add a small delay to prevent tight loop on errors
                import time
                time.sleep(1)
    
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
