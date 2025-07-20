"""
Trade Execution Logger

This module provides a structured logging utility for tracking trade execution events,
including order submissions, fills, modifications, and errors.
"""
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from enum import Enum
import inspect
import socket
import uuid

# Configure logging
logger = logging.getLogger(__name__)

class TradeLogLevel(Enum):
    """Log levels for trade events."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class TradeEventType(Enum):
    """Types of trade events that can be logged."""
    # Order events
    ORDER_SUBMIT = "order_submit"
    ORDER_ACCEPTED = "order_accepted"
    ORDER_REJECTED = "order_rejected"
    ORDER_FILLED = "order_filled"
    ORDER_PARTIALLY_FILLED = "order_partially_filled"
    ORDER_CANCELED = "order_canceled"
    ORDER_EXPIRED = "order_expired"
    ORDER_MODIFIED = "order_modified"
    
    # Position events
    POSITION_OPENED = "position_opened"
    POSITION_MODIFIED = "position_modified"
    POSITION_CLOSED = "position_closed"
    
    # Risk events
    RISK_CHECK_PASSED = "risk_check_passed"
    RISK_CHECK_FAILED = "risk_check_failed"
    RISK_LIMIT_BREACHED = "risk_limit_breached"
    
    # System events
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"
    CONNECTION_ESTABLISHED = "connection_established"
    CONNECTION_LOST = "connection_lost"
    
    # Error events
    EXECUTION_ERROR = "execution_error"
    VALIDATION_ERROR = "validation_error"
    TIMEOUT_ERROR = "timeout_error"
    
    # Strategy events
    SIGNAL_GENERATED = "signal_generated"
    SIGNAL_CANCELED = "signal_canceled"
    
    # Account events
    BALANCE_UPDATE = "balance_update"
    MARGIN_UPDATE = "margin_update"

class TradeLogger:
    """Structured logger for trade execution events."""
    
    def __init__(self, log_dir: str = "logs/trades", max_file_size: int = 100 * 1024 * 1024):
        """Initialize the trade logger.
        
        Args:
            log_dir: Directory to store trade log files
            max_file_size: Maximum size of each log file in bytes (default: 100MB)
        """
        self.log_dir = Path(log_dir)
        self.max_file_size = max_file_size
        self.current_log_file = None
        self.hostname = socket.gethostname()
        self.process_id = str(uuid.uuid4())[:8]  # Short unique ID for this process
        
        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize the current log file
        self._rotate_log_file()
    
    def _rotate_log_file(self) -> None:
        """Rotate the log file if it exceeds the maximum size."""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        self.current_log_file = self.log_dir / f"trades_{timestamp}.jsonl"
        
        # If the file exists and is too large, create a new one
        if self.current_log_file.exists() and self.current_log_file.stat().st_size > self.max_file_size:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
            self.current_log_file = self.log_dir / f"trades_{timestamp}.jsonl"
    
    def _get_caller_info(self) -> Dict[str, str]:
        """Get information about the function that called the logger."""
        frame = inspect.currentframe()
        try:
            # Go up 3 frames to get past the logging wrapper methods
            for _ in range(3):
                frame = frame.f_back
                if frame is None:
                    break
            
            if frame is not None:
                return {
                    "file": frame.f_code.co_filename,
                    "function": frame.f_code.co_name,
                    "line": frame.f_lineno,
                }
        except Exception:
            pass
        
        return {"file": "unknown", "function": "unknown", "line": 0}
    
    def _log_event(
        self,
        event_type: TradeEventType,
        level: TradeLogLevel,
        message: str,
        data: Optional[Dict[str, Any]] = None,
        order_id: Optional[str] = None,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
    ) -> None:
        """Log a trade event with structured data.
        
        Args:
            event_type: Type of trade event
            level: Log level
            message: Human-readable message
            data: Additional event-specific data
            order_id: Associated order ID (if any)
            symbol: Trading symbol (if any)
            strategy: Strategy name (if any)
        """
        try:
            # Get caller information
            caller = self._get_caller_info()
            
            # Create the log entry
            log_entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "level": level.value,
                "event_type": event_type.value,
                "message": message,
                "hostname": self.hostname,
                "process_id": self.process_id,
                "source": {
                    "file": caller["file"],
                    "function": caller["function"],
                    "line": caller["line"],
                },
                "context": {
                    "order_id": order_id,
                    "symbol": symbol,
                    "strategy": strategy,
                },
                "data": data or {},
            }
            
            # Remove None values
            log_entry = {k: v for k, v in log_entry.items() if v is not None}
            log_entry["context"] = {k: v for k, v in log_entry["context"].items() if v is not None}
            
            # Convert to JSON and write to log file
            log_line = json.dumps(log_entry) + "\n"
            
            # Rotate log file if needed
            if self.current_log_file.exists() and self.current_log_file.stat().st_size + len(log_line) > self.max_file_size:
                self._rotate_log_file()
            
            # Write to log file
            with open(self.current_log_file, "a", encoding="utf-8") as f:
                f.write(log_line)
            
            # Also log to standard logging
            log_method = getattr(logger, level.value.lower(), logger.info)
            log_method(f"{event_type.value.upper()}: {message}", extra={"data": data})
            
        except Exception as e:
            logger.error(f"Failed to log trade event: {e}", exc_info=True)
    
    # Convenience methods for common log types
    
    def log_order_submit(
        self,
        order_id: str,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: Optional[float] = None,
        stop_price: Optional[float] = None,
        time_in_force: str = "GTC",
        strategy: Optional[str] = None,
        **kwargs
    ) -> None:
        """Log an order submission event."""
        data = {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "order_type": order_type,
            "quantity": quantity,
            "price": price,
            "stop_price": stop_price,
            "time_in_force": time_in_force,
            **kwargs
        }
        
        self._log_event(
            event_type=TradeEventType.ORDER_SUBMIT,
            level=TradeLogLevel.INFO,
            message=f"Order submitted: {side} {quantity} {symbol} @ {price or 'MARKET'}",
            data=data,
            order_id=order_id,
            symbol=symbol,
            strategy=strategy
        )
    
    def log_order_fill(
        self,
        order_id: str,
        symbol: str,
        side: str,
        filled_quantity: float,
        fill_price: float,
        commission: float,
        remaining_quantity: float = 0,
        is_partial: bool = False,
        **kwargs
    ) -> None:
        """Log an order fill event."""
        event_type = TradeEventType.ORDER_PARTIALLY_FILLED if is_partial else TradeEventType.ORDER_FILLED
        
        data = {
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "filled_quantity": filled_quantity,
            "fill_price": fill_price,
            "commission": commission,
            "remaining_quantity": remaining_quantity,
            "is_partial": is_partial,
            **kwargs
        }
        
        self._log_event(
            event_type=event_type,
            level=TradeLogLevel.INFO,
            message=f"Order {'partially ' if is_partial else ''}filled: {side} {filled_quantity} {symbol} @ {fill_price}",
            data=data,
            order_id=order_id,
            symbol=symbol
        )
    
    def log_order_error(
        self,
        order_id: str,
        symbol: str,
        error_type: str,
        error_message: str,
        order_data: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> None:
        """Log an order error event."""
        data = {
            "order_id": order_id,
            "symbol": symbol,
            "error_type": error_type,
            "error_message": error_message,
            "order_data": order_data,
            **kwargs
        }
        
        self._log_event(
            event_type=TradeEventType.EXECUTION_ERROR,
            level=TradeLogLevel.ERROR,
            message=f"Order error: {error_type} - {error_message}",
            data=data,
            order_id=order_id,
            symbol=symbol
        )
    
    def log_risk_check(
        self,
        check_name: str,
        passed: bool,
        message: str,
        symbol: Optional[str] = None,
        strategy: Optional[str] = None,
        **kwargs
    ) -> None:
        """Log a risk check result."""
        event_type = TradeEventType.RISK_CHECK_PASSED if passed else TradeEventType.RISK_CHECK_FAILED
        
        data = {
            "check_name": check_name,
            "passed": passed,
            "message": message,
            **kwargs
        }
        
        self._log_event(
            event_type=event_type,
            level=TradeLogLevel.WARNING if not passed else TradeLogLevel.INFO,
            message=f"Risk check {'passed' if passed else 'failed'}: {check_name} - {message}",
            data=data,
            symbol=symbol,
            strategy=strategy
        )
    
    def log_position_update(
        self,
        position_id: str,
        symbol: str,
        side: str,
        size: float,
        entry_price: float,
        unrealized_pnl: float,
        event_type: TradeEventType = TradeEventType.POSITION_MODIFIED,
        **kwargs
    ) -> None:
        """Log a position update event."""
        data = {
            "position_id": position_id,
            "symbol": symbol,
            "side": side,
            "size": size,
            "entry_price": entry_price,
            "unrealized_pnl": unrealized_pnl,
            **kwargs
        }
        
        action = {
            TradeEventType.POSITION_OPENED: "opened",
            TradeEventType.POSITION_MODIFIED: "modified",
            TradeEventType.POSITION_CLOSED: "closed"
        }.get(event_type, "updated")
        
        self._log_event(
            event_type=event_type,
            level=TradeLogLevel.INFO,
            message=f"Position {action}: {side} {size} {symbol} @ {entry_price}",
            data=data,
            symbol=symbol
        )

# Global instance
trade_logger = TradeLogger()

# Convenience functions for direct logging

def log_order_submit(*args, **kwargs):
    """Convenience function to log an order submission."""
    trade_logger.log_order_submit(*args, **kwargs)

def log_order_fill(*args, **kwargs):
    """Convenience function to log an order fill."""
    trade_logger.log_order_fill(*args, **kwargs)

def log_order_error(*args, **kwargs):
    """Convenience function to log an order error."""
    trade_logger.log_order_error(*args, **kwargs)

def log_risk_check(*args, **kwargs):
    """Convenience function to log a risk check result."""
    trade_logger.log_risk_check(*args, **kwargs)

def log_position_update(*args, **kwargs):
    """Convenience function to log a position update."""
    trade_logger.log_position_update(*args, **kwargs)
