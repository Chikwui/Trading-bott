"""
Trading Bot Setup Script

This script helps set up the trading bot environment by:
1. Creating necessary directories
2. Initializing logging
3. Verifying dependencies (including MetaTrader 5)
4. Setting up the database
"""
import os
import sys
import subprocess
import logging
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('setup.log')
    ]
)
logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories for the application."""
    dirs = [
        'data',
        'logs',
        'backtest_results',
        'config',
        'strategies'
    ]
    
    for dir_name in dirs:
        try:
            os.makedirs(dir_name, exist_ok=True)
            logger.info(f"Created directory: {dir_name}")
        except Exception as e:
            logger.error(f"Error creating directory {dir_name}: {e}")
            return False
    return True

def check_mt5_installation():
    """Check if MetaTrader 5 is installed and accessible."""
    try:
        import MetaTrader5 as mt5
        
        # Check if MT5 is installed
        if not mt5.initialize():
            logger.error(f"MT5 initialize() failed, error code = {mt5.last_error()}")
            return False
            
        # Get MT5 version
        version = mt5.version()
        logger.info(f"MetaTrader5 package version: {mt5.__version__}")
        logger.info(f"MT5 terminal version: {version}")
        
        mt5.shutdown()
        return True
        
    except ImportError:
        logger.warning("MetaTrader5 package not installed. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "MetaTrader5"])
            logger.info("MetaTrader5 package installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install MetaTrader5: {e}")
            return False
    except Exception as e:
        logger.error(f"Error checking MT5 installation: {e}")
        return False

def check_dependencies():
    """Check if all required dependencies are installed."""
    required = [
        'numpy',
        'pandas',
        'pandas-ta',
        'python-dotenv',
        'PyYAML',
        'redis',
        'requests',
        'websockets',
        'numba',
        'scipy',
        'scikit-learn',
        'aiohttp',
        'pytest',
        'pytest-asyncio',
        'pytest-cov',
        'pytest-mock',
        'ta',
        'matplotlib',
        'seaborn',
        'plotly',
        'tqdm',
        'joblib',
        'python-dateutil',
        'pytz',
        'MetaTrader5'
    ]
    
    missing = []
    for package in required:
        try:
            __import__(package.split('.')[0])
            logger.info(f"✓ {package} is installed")
        except ImportError:
            missing.append(package)
            logger.warning(f"✗ {package} is not installed")
    
    if missing:
        logger.warning(f"\nMissing {len(missing)} dependencies. Installing...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            logger.info("✓ Dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error installing dependencies: {e}")
            return False
    
    return True

def check_redis():
    """Check if Redis server is running."""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        logger.info("✓ Redis server is running")
        return True
    except redis.ConnectionError:
        logger.warning("✗ Redis server is not running")
        logger.info("Please install and start Redis server:")
        logger.info("  Windows: https://github.com/tporadowski/redis/releases")
        logger.info("  macOS: brew install redis && brew services start redis")
        logger.info("  Linux: sudo apt-get install redis-server && sudo service redis-server start")
        return False

def check_environment():
    """Check if environment variables are properly set."""
    load_dotenv()
    required_vars = [
        'MT5_SERVER',
        'MT5_LOGIN',
        'MT5_PASSWORD',
        'MT5_PATH'
    ]
    
    missing = []
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)
    
    if missing:
        logger.warning(f"Missing required environment variables: {', '.join(missing)}")
        logger.info("Please update your .env file with the required MT5 credentials")
        return False
    
    logger.info("✓ Environment variables are properly set")
    return True

def main():
    """Main setup function."""
    logger.info("Starting Trading Bot Setup...\n")
    
    # Create necessary directories
    logger.info("=== Setting up directories ===")
    if not create_directories():
        logger.error("Failed to create directories")
        return False
    
    # Check and install dependencies
    logger.info("\n=== Checking dependencies ===")
    if not check_dependencies():
        logger.error("Failed to install dependencies")
        return False
    
    # Check MetaTrader 5 installation
    logger.info("\n=== Checking MetaTrader 5 ===")
    if not check_mt5_installation():
        logger.error("MetaTrader 5 setup failed. Please ensure MT5 is installed and configured.")
        logger.info("Download MT5 from: https://www.metatrader5.com/en/download")
        return False
    
    # Check Redis server
    logger.info("\n=== Checking Redis server ===")
    if not check_redis():
        logger.warning("Redis server is recommended for optimal performance")
    
    # Check environment variables
    logger.info("\n=== Checking environment variables ===")
    if not check_environment():
        logger.warning("Please update your .env file before running the bot")
    
    logger.info("\n=== Setup Complete ===")
    logger.info("Next steps:")
    logger.info("1. Update your .env file with your MT5 credentials")
    logger.info("2. Make sure MetaTrader 5 is installed and running")
    logger.info("3. Run the bot: python run_phase1.py")
    
    return True

if __name__ == "__main__":
    if main():
        sys.exit(0)
    else:
        sys.exit(1)
