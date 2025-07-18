"""
Trading Bot - Phase 1 Initialization with MetaTrader 5 Integration

This script initializes and runs Phase 1 of the trading bot with MT5 integration.
"""
import asyncio
import logging
import signal
import sys
import MetaTrader5 as mt5
from pathlib import Path
from typing import Dict, Any, Optional, List

# Add project root to Python path
sys.path.append(str(Path(__file__).parent))

from utils.logger import setup_logging
from services.market_data import MT5MarketDataService
from services.signal_service import SignalService
from core.execution.mt5_executor import MT5Executor
from core.risk import RiskManager
from core.portfolio import Portfolio
from config import settings

# Configure logging
logger = logging.getLogger(__name__)

class Phase1TradingBot:
    """Phase 1 Trading Bot with MT5 Integration."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Phase 1 components with MT5."""
        self.config = config
        self.running = False
        self.mt5_initialized = False
        
        # Core components
        self.market_data_service: Optional[MT5MarketDataService] = None
        self.signal_service: Optional[SignalService] = None
        self.executor: Optional[MT5Executor] = None
        self.portfolio: Optional[Portfolio] = None
        self.risk_manager: Optional[RiskManager] = None
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
    
    def _initialize_mt5(self) -> bool:
        """Initialize MetaTrader 5 connection."""
        try:
            # Initialize MT5
            if not mt5.initialize(
                path=settings.MT5_PATH,
                portable=settings.MT5_PORTABLE,
                timeout=settings.MT5_TIMEOUT
            ):
                logger.error(f"MT5 initialize() failed: {mt5.last_error()}")
                return False
            
            # Login to the trade account
            authorized = mt5.login(
                login=settings.MT5_LOGIN,
                password=settings.MT5_PASSWORD,
                server=settings.MT5_SERVER
            )
            
            if not authorized:
                logger.error(f"MT5 login failed: {mt5.last_error()}")
                mt5.shutdown()
                return False
            
            logger.info(f"Connected to MT5 account #{settings.MT5_LOGIN}")
            self.mt5_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Error initializing MT5: {e}", exc_info=True)
            return False
    
    def _init_components(self) -> bool:
        """Initialize all Phase 1 components with MT5 integration."""
        try:
            logger.info("Initializing Phase 1 components with MT5...")
            
            # Initialize MT5 connection
            if not self._initialize_mt5():
                return False
            
            # Initialize market data service
            self.market_data_service = MT5MarketDataService(
                symbols=settings.SYMBOLS,
                timeframes=settings.TIMEFRAMES,
                historical_bars=settings.HISTORICAL_BARS
            )
            
            # Initialize portfolio
            account_info = mt5.account_info()
            self.portfolio = Portfolio(
                initial_balance=account_info.balance,
                risk_per_trade=settings.RISK_PER_TRADE,
                max_drawdown=settings.MAX_DRAWDOWN,
                max_position_size=settings.MAX_POSITION_SIZE,
                max_leverage=settings.MAX_LEVERAGE
            )
            
            # Initialize risk manager
            self.risk_manager = RiskManager(
                portfolio=self.portfolio,
                max_daily_loss=settings.DAILY_LOSS_LIMIT,
                max_open_positions=settings.MAX_OPEN_POSITIONS
            )
            
            # Initialize MT5 executor
            self.executor = MT5Executor()
            
            # Initialize signal service
            self.signal_service = SignalService(
                market_data_service=self.market_data_service,
                risk_manager=self.risk_manager,
                executor=self.executor
            )
            
            logger.info("All components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}", exc_info=True)
            return False
    
    async def start(self) -> None:
        """Start the Phase 1 trading bot with MT5."""
        if not self._init_components():
            logger.error("Failed to initialize components. Exiting...")
            return
        
        self.running = True
        logger.info("Starting Phase 1 trading bot with MT5...")
        
        try:
            # Main trading loop
            while self.running:
                try:
                    # Check system health
                    if not await self._check_system_health():
                        logger.error("System health check failed. Shutting down...")
                        break
                    
                    # Process market data
                    await self._process_market_data()
                    
                    # Generate and execute trading signals
                    await self._generate_and_execute_signals()
                    
                    # Update portfolio
                    await self._update_portfolio()
                    
                    # Sleep to prevent CPU overload
                    await asyncio.sleep(1)
                    
                except asyncio.CancelledError:
                    logger.info("Shutdown signal received")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}", exc_info=True)
                    await asyncio.sleep(5)  # Prevent tight loop on error
        
        except Exception as e:
            logger.critical(f"Fatal error: {e}", exc_info=True)
        finally:
            await self.shutdown()
    
    async def _check_system_health(self) -> bool:
        """Check the health of all system components."""
        try:
            # Check MT5 connection
            if not mt5.initialize():
                logger.error("Lost connection to MT5")
                return False
            
            # Check market data service
            if not self.market_data_service.is_connected():
                logger.error("Market data service is not connected")
                return False
            
            # Check risk limits
            if self.risk_manager.is_risk_limit_breached():
                logger.error("Risk limits breached")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def _process_market_data(self) -> None:
        """Process incoming market data from MT5."""
        try:
            # Get latest market data
            market_data = await self.market_data_service.get_latest_data()
            
            # Update portfolio with latest prices
            if market_data:
                self.portfolio.update_prices(market_data)
                
        except Exception as e:
            logger.error(f"Error processing market data: {e}")
    
    async def _generate_and_execute_signals(self) -> None:
        """Generate and execute trading signals."""
        try:
            # Generate trading signals
            signals = await self.signal_service.generate_signals()
            
            # Execute trades based on signals
            if signals:
                for symbol, signal in signals.items():
                    # Execute the trade through MT5
                    result = await self.executor.execute_trade(
                        symbol=symbol,
                        signal_type=signal['type'],
                        price=signal.get('price'),
                        stop_loss=signal.get('stop_loss'),
                        take_profit=signal.get('take_profit'),
                        comment="Auto-trade from Phase 1"
                    )
                    
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        logger.info(f"Trade executed: {result}")
                    else:
                        logger.warning(f"Trade execution failed: {result.comment}")
                    
        except Exception as e:
            logger.error(f"Error in signal generation/execution: {e}")
    
    async def _update_portfolio(self) -> None:
        """Update portfolio with latest positions from MT5."""
        try:
            # Get open positions from MT5
            positions = mt5.positions_get()
            if positions is None:
                positions = []
            
            # Update portfolio with current positions
            self.portfolio.update_positions(positions)
            
        except Exception as e:
            logger.error(f"Error updating portfolio: {e}")
    
    async def shutdown(self) -> None:
        """Gracefully shut down the trading bot."""
        if not self.running:
            return
            
        logger.info("Shutting down Phase 1 trading bot...")
        self.running = False
        
        try:
            # Close all positions if configured
            if settings.CLOSE_POSITIONS_ON_SHUTDOWN:
                logger.info("Closing all open positions...")
                if self.executor:
                    await self.executor.close_all_positions()
            
            # Save portfolio state
            if self.portfolio:
                self.portfolio.save_state()
                
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            # Shutdown MT5 connection
            if self.mt5_initialized:
                mt5.shutdown()
                self.mt5_initialized = False
                
            logger.info("Phase 1 trading bot shutdown complete")
    
    def _handle_shutdown(self, signum, frame) -> None:
        """Handle shutdown signals."""
        logger.info(f"Received shutdown signal {signum}")
        asyncio.create_task(self.shutdown())

async def main():
    """Main entry point for Phase 1 with MT5."""
    # Set up logging
    setup_logging()
    
    # Load configuration
    config = {
        "trading": {
            "mode": "paper",
            "initial_balance": float(settings.INITIAL_BALANCE),
            "risk_per_trade": float(settings.RISK_PER_TRADE),
            "max_drawdown": float(settings.MAX_DRAWDOWN),
            "max_position_size": float(settings.MAX_POSITION_SIZE),
            "max_leverage": float(settings.MAX_LEVERAGE),
            "daily_loss_limit": float(settings.DAILY_LOSS_LIMIT)
        },
        "instruments": settings.SYMBOLS,
        "timeframes": settings.TIMEFRAMES
    }
    
    # Initialize and start the bot
    bot = Phase1TradingBot(config)
    await bot.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutdown requested by user")
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
