import pytest
from config.settings import settings


def test_default_settings():
    assert settings.KAFKA_BROKER_URL == "localhost:9092"
    assert settings.KAFKA_TICK_TOPIC == "market_ticks"
    assert settings.KAFKA_SIGNAL_TOPIC == "trade_signals"
    assert settings.MIN_CONFIDENCE_THRESHOLD == 0.7
    assert settings.AUTO_APPROVE_THRESHOLD == 0.85
    assert settings.AUTO_REJECT_THRESHOLD == 0.3
