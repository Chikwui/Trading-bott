"""
Main entry point for the trading bot.
"""
import asyncio
import argparse
import logging
import signal
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from .app import TradingBot, load_config
from .utils.logger import get_logger
from .config import settings

logger = get_logger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='AI-Powered Trading Bot')
    
    # General options
    parser.add_argument('--config', type=str, default='config/settings.yaml',
                      help='Path to configuration file')
    parser.add_argument('--log-level', type=str, default='INFO',
                      choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                      help='Logging level')
    
    # Trading options
    parser.add_argument('--mode', type=str, choices=['live', 'paper', 'backtest'],
                      default='paper', help='Trading mode')
    parser.add_argument('--symbols', type=str, nargs='+',
                      help='Trading symbols (e.g., BTC/USD, EUR/USD)')
    parser.add_argument('--timeframes', type=str, nargs='+',
                      default=['1h', '4h', '1d'],
                      help='Timeframes to trade (e.g., 1h, 4h, 1d)')
    
    # Strategy options
    parser.add_argument('--strategy', type=str, default='mean_reversion',
                      help='Trading strategy to use')
    parser.add_argument('--risk', type=float, default=0.01,
                      help='Risk per trade (0.01 = 1%)')
    
    # Backtesting options
    parser.add_argument('--backtest', action='store_true',
                      help='Run in backtest mode')
    parser.add_argument('--from-date', type=str,
                      help='Backtest start date (YYYY-MM-DD)')
    parser.add_argument('--to-date', type=str,
                      help='Backtest end date (YYYY-MM-DD)')
    parser.add_argument('--initial-balance', type=float, default=10000.0,
                      help='Initial balance for backtesting')
    
    return parser.parse_args()

def setup_logging(log_level: str = 'INFO') -> None:
    """Configure logging."""
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('trading_bot.log')
        ]
    )

def handle_shutdown(signum, frame) -> None:
    """Handle shutdown signals."""
    logger.info("Shutdown signal received, exiting...")
    sys.exit(0)

async def main() -> None:
    """Main entry point."""
    # Parse command line arguments
    args = parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    logger.info("Starting Trading Bot...")
    logger.info(f"Version: {__import__('trading_bot').__version__}")
    logger.info(f"Trading Mode: {args.mode.upper()}")
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Override config with command line arguments
        if args.mode:
            config['trading']['mode'] = args.mode
        if args.symbols:
            config['trading']['symbols'] = args.symbols
        if args.timeframes:
            config['trading']['timeframes'] = args.timeframes
        if args.risk:
            config['risk_management']['risk_per_trade'] = args.risk
        
        # Initialize and run the bot
        bot = TradingBot(config)
        
        if args.backtest:
            # Run backtest
            logger.info("Running in backtest mode")
            from .backtest import Backtester
            
            backtester = Backtester(
                strategy=args.strategy,
                symbols=args.symbols or config['trading']['symbols'],
                timeframes=args.timeframes,
                initial_balance=args.initial_balance,
                risk_per_trade=args.risk,
                from_date=args.from_date,
                to_date=args.to_date
            )
            
            await backtester.run()
        else:
            # Run live/paper trading
            await bot.start()
            
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Trading Bot stopped")

if __name__ == "__main__":
    asyncio.run(main())
