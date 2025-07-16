"""
Logging configuration for the trading bot.
"""
import logging
import sys
import os
from pathlib import Path
from typing import Optional, Union
from datetime import datetime

# Log directory
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Log format
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

class ColoredFormatter(logging.Formatter):
    """Custom formatter for colored console output."""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[94m',     # Blue
        'INFO': '\033[92m',      # Green
        'WARNING': '\033[93m',   # Yellow
        'ERROR': '\033[91m',     # Red
        'CRITICAL': '\033[91m',  # Red
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        """Format the log record with colors."""
        if record.levelname in self.COLORS:
            record.msg = (f"{self.COLORS[record.levelname]}{record.msg}"
                         f"{self.COLORS['RESET']}")
        return super().format(record)

def setup_logger(
    name: str,
    level: Union[int, str] = logging.INFO,
    log_to_file: bool = True,
    log_file: Optional[Union[str, Path]] = None,
    log_to_console: bool = True
) -> logging.Logger:
    """Configure and return a logger instance.
    
    Args:
        name: Logger name
        level: Logging level (default: INFO)
        log_to_file: Whether to log to file
        log_file: Path to log file (default: logs/{name}_{date}.log)
        log_to_console: Whether to log to console
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler
    if log_to_file:
        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = LOG_DIR / f"{name}_{timestamp}.log"
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_formatter = logging.Formatter(LOG_FORMAT, DATE_FORMAT)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = ColoredFormatter(
            f"%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt=DATE_FORMAT
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    return logger

# Default logger
def get_logger(name: str = None) -> logging.Logger:
    """Get a configured logger instance.
    
    Args:
        name: Logger name (default: __name__ of caller)
        
    Returns:
        Configured logger instance
    """
    if name is None:
        import inspect
        name = inspect.currentframe().f_back.f_globals.get('__name__', 'root')
    
    return setup_logger(name)

# Example usage
if __name__ == "__main__":
    log = get_logger("example")
    log.debug("Debug message")
    log.info("Info message")
    log.warning("Warning message")
    log.error("Error message")
    log.critical("Critical message")
