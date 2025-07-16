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

# MT5 configuration
MT5_CONFIG = {
    'login': int(os.getenv('MT5_LOGIN', 0)),
    'password': os.getenv('MT5_PASSWORD', ''),
    'server': os.getenv('MT5_SERVER', ''),
    'timeout': 60000,  # 60 seconds
    'portable': False
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
