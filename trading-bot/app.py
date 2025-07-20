"""
Trading Bot - Main Application Entry Point
"""
import asyncio
import signal
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from utils.logger import get_logger
from services.market_data import MarketDataService
from services.signal_service import SignalService
from core.market import MarketDataHandler, ExecutionHandler, Portfolio, RiskManager
from core.calendar import CalendarFactory
from config import settings

logger = get_logger(__name__)

class TradingBot:
    """Main trading bot application."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the trading bot.
        
        Args:
            config: Application configuration
        """
        self.config = config
        self.running = False
        self.market_data_service: Optional[MarketDataService] = None
        self.signal_service: Optional[SignalService] = None
        
        # Initialize core components
        self._init_components()
    
    def _init_components(self) -> None:
        """Initialize all components."""
        logger.info("Initializing components...")
        
        # Initialize market data handler with instruments from config
        from core.market.instrument import InstrumentMetadata
        
        # Create instrument metadata from ASSET_CLASSES config
        instruments = []
        for asset_class, config in settings.ASSET_CLASSES.items():
            for symbol in config['symbols']:
                instruments.append(InstrumentMetadata(
                    symbol=symbol,
                    asset_class=asset_class,
                    exchange='MT5',  # Default exchange
                    min_tick_size=0.00001,  # Default value, adjust as needed
                    min_order_size=0.01,    # Default value, adjust as needed
                    max_leverage=config.get('leverage', 10),
                    trading_hours=config.get('trading_hours', '24/5')
                ))
        
        self.market_data_handler = MarketDataHandler(instruments=instruments)
        
        # Initialize execution handler with MT5 credentials
        self.execution_handler = ExecutionHandler(
            api_key=str(settings.MT5_LOGIN),  # Using MT5 login as API key
            api_secret=settings.MT5_PASSWORD,  # Using MT5 password as API secret
            sandbox=settings.TRADING_MODE == 'paper',
            server=settings.MT5_SERVER,
            path=settings.MT5_PATH,
            timeout=settings.MT5_TIMEOUT,
            portable=settings.MT5_PORTABLE
        )
        
        # Initialize portfolio
        self.portfolio = Portfolio(
            initial_balance=settings.INITIAL_BALANCE,
            risk_per_trade=settings.RISK_PER_TRADE,
            max_drawdown=settings.MAX_DRAWDOWN
        )
        
        # Initialize risk manager
        self.risk_manager = RiskManager(
            max_position_size=settings.MAX_POSITION_SIZE,
            max_leverage=settings.MAX_LEVERAGE,
            daily_loss_limit=settings.DAILY_LOSS_LIMIT
        )
        
        # Initialize market data service
        self.market_data_service = MarketDataService(self.market_data_handler)
        
        # Initialize signal service
        self.signal_service = SignalService()
        
        # Register signal handlers
        self._register_signal_handlers()
    
    def _register_signal_handlers(self) -> None:
        """Register signal handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
    
    def _handle_shutdown(self, signum, frame) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    async def start(self) -> None:
        """Start the trading bot."""
        if self.running:
            logger.warning("Bot is already running")
            return
        
        self.running = True
        logger.info("Starting trading bot...")
        
        try:
            # Start services
            await self.market_data_service.start()
            
            # Main event loop
            while self.running:
                try:
                    # Main trading logic here
                    await asyncio.sleep(1)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}", exc_info=True)
                    await asyncio.sleep(5)  # Backoff on error
                    
        except Exception as e:
            logger.critical(f"Fatal error in trading bot: {e}", exc_info=True)
            raise
            
        finally:
            await self.stop()
    
    async def stop(self) -> None:
        """Stop the trading bot gracefully."""
        if not self.running:
            return
            
        logger.info("Stopping trading bot...")
        self.running = False
        
        # Stop services
        if self.market_data_service:
            await self.market_data_service.stop()
        
        logger.info("Trading bot stopped")

def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from file or environment."""
    # Configuration loading logic
    # This is a placeholder - actual implementation would load from file/env
    return {
        "trading": {
            "mode": "paper",  # or 'live'
            "initial_balance": 10000.0,
            "risk_per_trade": 0.01,  # 1% per trade
            "max_drawdown": 0.1,  # 10% max drawdown
            "max_position_size": 1000.0,
            "max_leverage": 10.0,
            "daily_loss_limit": 0.05  # 5% daily loss limit
        },
        "broker": {
            "api_key": "",  # Will be loaded from environment
            "api_secret": "",  # Will be loaded from environment
            "sandbox": True
        },
        "instruments": ["BTC/USD", "ETH/USD", "AAPL", "SPY"],
        "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d"]
    }

async def main() -> None:
    """Main entry point."""
    try:
        # Load configuration
        config = load_config()
        
        # Create and start bot
        bot = TradingBot(config)
        await bot.start()
        
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    finally:
        logger.info("Application shutdown complete")

if __name__ == "__main__":
    asyncio.run(main())
