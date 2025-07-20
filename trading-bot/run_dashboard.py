"""
MT5 Trading Dashboard

This is the main entry point for the MT5 Trading Dashboard.
It initializes the MT5 client, data provider, and dashboard.
"""
import asyncio
import logging
import argparse
import os
from pathlib import Path
import sys

# Add the project root to the Python path
project_root = str(Path(__file__).parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.execution.live.mt5_live_broker import MT5LiveBroker
from core.data.providers.mt5_provider import MT5DataProvider
from core.monitoring.dashboard import init_dashboard
from core.utils.helpers import get_logger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('dashboard.log')
    ]
)

logger = get_logger(__name__)

async def main():
    """Main function to initialize and run the dashboard."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='MT5 Trading Dashboard')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the dashboard on')
    parser.add_argument('--port', type=int, default=8050, help='Port to run the dashboard on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    args = parser.parse_args()

    try:
        logger.info("Starting MT5 Trading Dashboard...")
        
        # Initialize MT5 components
        logger.info("Initializing MT5 components...")
        mt5_broker = MT5LiveBroker(
            server=os.getenv('MT5_SERVER'),
            login=int(os.getenv('MT5_LOGIN', '0')),
            password=os.getenv('MT5_PASSWORD')
        )
        
        mt5_provider = MT5DataProvider()
        
        # Connect to MT5
        logger.info("Connecting to MT5...")
        if not mt5_broker.connect():
            logger.error("Failed to connect to MT5. Please check your credentials and try again.")
            return
        
        # Initialize and run the dashboard
        logger.info(f"Starting dashboard on http://{args.host}:{args.port}")
        init_dashboard(
            mt5_broker=mt5_broker,
            mt5_provider=mt5_provider,
            host=args.host,
            port=args.port,
            debug=args.debug
        )
        
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
    finally:
        # Clean up resources
        if 'mt5_broker' in locals() and mt5_broker.connected:
            mt5_broker.disconnect()
        logger.info("Dashboard stopped")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Dashboard stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
