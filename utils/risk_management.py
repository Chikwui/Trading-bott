from config.settings import settings
from utils.logger import logger


def calculate_position_size(account_balance: float, risk_percent: float, stop_loss_pips: float, pip_value: float = 1.0) -> float:
    """
    Calculate position size (lots) given account balance, risk percent, and stop loss (pips).
    """
    risk_amount = account_balance * (risk_percent / 100)
    if stop_loss_pips <= 0:
        logger.warning("Stop loss pips must be positive")
        return 0.0
    size = risk_amount / (stop_loss_pips * pip_value)
    size = min(size, settings.MAX_OPEN_POSITIONS)
    return round(size, 2)


def validate_trade(size: float) -> bool:
    """Validate trade size against max position limits."""
    if size <= 0:
        logger.error("Calculated position size <= 0")
        return False
    if size > settings.MAX_OPEN_POSITIONS:
        logger.error(f"Position size {size} exceeds max allowed {settings.MAX_OPEN_POSITIONS}")
        return False
    return True
