"""
Main configuration settings for the trading bot.
"""
import os
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Logging configuration
LOG_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'standard'
        },
        'file': {
            'level': 'DEBUG',
            'class': 'logging.handlers.RotatingFileHandler',
            'filename': os.path.join(BASE_DIR, 'logs', 'trading_bot.log'),
            'maxBytes': 10 * 1024 * 1024,  # 10MB
            'backupCount': 5,
            'formatter': 'standard'
        },
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
            'propagate': True
        },
    }
}

# Redis configuration
REDIS_CONFIG = {
    'host': os.getenv('REDIS_HOST', 'localhost'),
    'port': int(os.getenv('REDIS_PORT', 6379)),
    'password': os.getenv('REDIS_PASSWORD', ''),
    'db': int(os.getenv('REDIS_DB', 0)),
    'socket_timeout': 5,
    'socket_connect_timeout': 5,
    'retry_on_timeout': True
}

# MT5 configuration - Module level attributes for direct access
MT5_LOGIN = int(os.getenv('MT5_LOGIN', 0))
MT5_PASSWORD = os.getenv('MT5_PASSWORD', '')
MT5_SERVER = os.getenv('MT5_SERVER', '')
MT5_PATH = os.getenv('MT5_PATH', '')
MT5_TIMEOUT = int(os.getenv('MT5_TIMEOUT', 60000))  # 60 seconds
MT5_PORTABLE = os.getenv('MT5_PORTABLE', 'False').lower() in ('true', '1', 't')

# Keep the config dictionary for backward compatibility
MT5_CONFIG = {
    'login': MT5_LOGIN,
    'password': MT5_PASSWORD,
    'server': MT5_SERVER,
    'path': MT5_PATH,
    'timeout': MT5_TIMEOUT,
    'portable': MT5_PORTABLE
}

# Risk management configuration
RISK_CONFIG = {
    'max_risk_per_trade': float(os.getenv('MAX_RISK_PER_TRADE', 0.05)),  # 5%
    'max_daily_drawdown': float(os.getenv('MAX_DAILY_DRAWDOWN', 0.05)),  # 5%
    'max_portfolio_risk': float(os.getenv('MAX_PORTFOLIO_RISK', 0.15)),  # 15%
    'volatility_lookback': int(os.getenv('VOLATILITY_LOOKBACK', 20)),  # periods
    'atr_multiplier': float(os.getenv('ATR_MULTIPLIER', 2.0)),
    'max_leverage': int(os.getenv('MAX_LEVERAGE', 10)),  # 10x
}

# Timeframes configuration
TIMEFRAMES = {
    'M1': 1,      # 1 minute
    'M5': 5,      # 5 minutes
    'M15': 15,    # 15 minutes
    'M30': 30,    # 30 minutes
    'H1': 60,     # 1 hour
    'H2': 120,    # 2 hours
    'H4': 240,    # 4 hours
    'D1': 1440,   # 1 day
    'W1': 10080,  # 1 week
    'MN1': 43200  # 1 month
}

# Asset classes
ASSET_CLASSES = {
    'forex': {
        'symbols': ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD'],
        'leverage': 30,
        'trading_hours': '24/5',
    },
    'crypto': {
        'symbols': ['BTCUSDT', 'ETHUSDT', 'XRPUSDT', 'LTCUSDT', 'BNBUSDT'],
        'leverage': 10,
        'trading_hours': '24/7',
    },
    'commodities': {
        'symbols': ['XAUUSD', 'XAGUSD', 'USOIL', 'UKOIL', 'NATGAS'],
        'leverage': 20,
        'trading_hours': '24/5',
    },
    'indices': {
        'symbols': ['US30', 'US500', 'USTEC', 'UK100', 'GER40'],
        'leverage': 20,
        'trading_hours': '24/5',
    }
}

# Exchange configuration
EXCHANGES = {
    'binance': {
        'api_key': os.getenv('BINANCE_API_KEY', ''),
        'api_secret': os.getenv('BINANCE_API_SECRET', ''),
        'testnet': os.getenv('BINANCE_TESTNET', 'false').lower() == 'true',
    },
    'ftx': {
        'api_key': os.getenv('FTX_API_KEY', ''),
        'api_secret': os.getenv('FTX_API_SECRET', ''),
        'subaccount': os.getenv('FTX_SUBACCOUNT', ''),
    },
}

# Database configuration
DATABASE_CONFIG = {
    'engine': 'sqlite',
    'database': os.path.join(BASE_DIR, 'data', 'trading_bot.db'),
    'echo': False,
}

# Trading mode configuration
TRADING_CONFIG = {
    'auto_trade': os.getenv('AUTO_TRADE', 'true').lower() == 'true',  # Enable automatic trading
    'simulation_mode': os.getenv('SIMULATION_MODE', 'true').lower() == 'true',  # Run in simulation mode
    'paper_trading': os.getenv('PAPER_TRADING', 'true').lower() == 'true',  # Use paper trading account
    'require_confirmation': os.getenv('REQUIRE_CONFIRMATION', 'false').lower() == 'true',  # Manual confirmation
    'max_open_trades': int(os.getenv('MAX_OPEN_TRADES', 10)),  # Maximum number of open trades
    'default_slippage': float(os.getenv('DEFAULT_SLIPPAGE', 0.0005)),  # Default slippage (0.05%)
    'default_commission': float(os.getenv('DEFAULT_COMMISSION', 0.001)),  # Default commission (0.1%)
}

# Paths
PATHS = {
    'data': os.path.join(BASE_DIR, 'data'),
    'logs': os.path.join(BASE_DIR, 'logs'),
    'models': os.path.join(BASE_DIR, 'models'),
    'strategies': os.path.join(BASE_DIR, 'strategies'),
}

# Create necessary directories
for path in PATHS.values():
    os.makedirs(path, exist_ok=True)
