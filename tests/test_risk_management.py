import pytest
from utils.risk_management import calculate_position_size, validate_trade
from config.settings import settings


def test_calculate_position_size_positive():
    size = calculate_position_size(1000, settings.RISK_PERCENT_PER_TRADE, 10, pip_value=1)
    assert size > 0


def test_validate_trade():
    assert not validate_trade(0)
    assert validate_trade(0.1)
