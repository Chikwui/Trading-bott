"""
MT5 Data Backfilling Example

This script demonstrates how to use the EnhancedMT5DataProvider with DataBackfiller
to backfill historical OHLCV and trade data from MT5 to local storage.
"""
import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytz

# Add project root to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.data.backfiller import DataBackfiller, LocalStorage
from core.data.providers.mt5_enhanced import EnhancedMT5DataProvider, DataProviderConfig
from core.data.quality import DataQualityChecker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('backfill.log')
    ]
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'mt5': {
        'login': None,          # Your MT5 login
        'password': None,       # Your MT5 password
        'server': None,         # Your MT5 server
        'path': None,           # Path to MT5 terminal (if needed)
        'backfill': {
            'chunk_size': 1000,     # Number of candles/trades per request
            'max_retries': 3,       # Maximum number of retry attempts
            'request_delay': 0.1,   # Delay between requests in seconds
            'max_workers': 5,       # Maximum number of concurrent workers
            'validate_data': True,  # Enable data validation
            'repair_data': True,    # Attempt to repair minor data issues
        }
    },
    'symbols': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD'],
    'timeframes': ['1m', '5m', '15m', '1h', '4h', '1d'],
    'data_dir': 'data/mt5',
    'backfill_days': 30,  # Number of days to backfill
    'backfill_trades': True,  # Whether to backfill trade data
}

async def main():
    """Main function to run the backfilling process."""
    logger.info("Starting MT5 data backfilling...")
    
    # Create data directory if it doesn't exist
    data_dir = Path(CONFIG['data_dir'])
    data_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize data provider
        provider_config = DataProviderConfig(
            name='mt5_enhanced',
            exchange='mt5',
            extra=CONFIG['mt5']
        )
        
        provider = EnhancedMT5DataProvider(provider_config)
        
        # Initialize storage
        storage = LocalStorage(str(data_dir))
        
        # Initialize backfiller
        backfiller = DataBackfiller(
            data_provider=provider,
            storage=storage,
            exchange='mt5',
            max_retries=CONFIG['mt5']['backfill']['max_retries'],
            request_delay=CONFIG['mt5']['backfill']['request_delay'],
            chunk_size=CONFIG['mt5']['backfill']['chunk_size']
        )
        
        # Initialize data quality checker
        quality_checker = DataQualityChecker()
        
        # Calculate date range
        end_date = datetime.now(timezone.utc)
        start_date = end_date - timedelta(days=CONFIG['backfill_days'])
        
        logger.info(f"Backfilling data from {start_date} to {end_date}")
        
        # Backfill OHLCV data for each symbol and timeframe
        for symbol in CONFIG['symbols']:
            logger.info(f"Processing symbol: {symbol}")
            
            # Backfill OHLCV data for each timeframe
            for timeframe in CONFIG['timeframes']:
                try:
                    logger.info(f"Backfilling {symbol} {timeframe} OHLCV data...")
                    
                    await backfiller.backfill_ohlcv(
                        symbol=symbol,
                        timeframe=timeframe,
                        start_date=start_date,
                        end_date=end_date,
                        batch_size=timedelta(days=7)  # 1 week batches
                    )
                    
                    # Verify data quality
                    ohlcv_data = await storage.get_ohlcv(
                        symbol, 
                        timeframe,
                        start_date=start_date,
                        end_date=end_date
                    )
                    
                    if ohlcv_data is not None and not ohlcv_data.empty:
                        quality_report = quality_checker.check_ohlcv_dataframe(
                            ohlcv_data, 
                            symbol, 
                            timeframe
                        )
                        
                        if not quality_report['valid']:
                            logger.warning(
                                f"Data quality issues found for {symbol} {timeframe}: "
                                f"{quality_report['checks']}"
                            )
                    
                except Exception as e:
                    logger.error(
                        f"Error backfilling {symbol} {timeframe} OHLCV data: {e}",
                        exc_info=True
                    )
            
            # Backfill trade data if enabled
            if CONFIG['backfill_trades']:
                try:
                    logger.info(f"Backfilling {symbol} trade data...")
                    
                    await backfiller.backfill_trades(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        batch_size=1000  # 1000 trades per batch
                    )
                    
                    # Verify trade data quality
                    latest_timestamp = await storage.get_latest_timestamp(symbol)
                    if latest_timestamp:
                        logger.info(
                            f"Successfully backfilled trades for {symbol} "
                            f"up to {latest_time}"
                        )
                    
                except Exception as e:
                    logger.error(
                        f"Error backfilling {symbol} trade data: {e}",
                        exc_info=True
                    )
        
        logger.info("Backfilling completed successfully!")
        
    except Exception as e:
        logger.error(f"Fatal error during backfilling: {e}", exc_info=True)
        raise
    
    finally:
        # Cleanup
        if 'provider' in locals():
            await provider.close()
        logger.info("Backfilling process finished")

if __name__ == "__main__":
    asyncio.run(main())
