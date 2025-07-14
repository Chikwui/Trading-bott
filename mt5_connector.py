import MetaTrader5 as mt5
from config.settings import settings
from utils.logger import logger

def connect():
    """Initialize connection to MT5 terminal."""
    if not mt5.initialize(
        login=settings.MT5_LOGIN,
        password=settings.MT5_PASSWORD,
        server=settings.MT5_SERVER
    ):
        logger.error(f"MT5 initialize failed: {mt5.last_error()}")
        return False
    logger.info("MT5 initialized successfully")
    return True


def shutdown():
    """Shutdown MT5 connection."""
    mt5.shutdown()
    logger.info("MT5 shutdown")
