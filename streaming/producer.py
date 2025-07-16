"""
Message producer for the trading bot.

This module provides a unified interface for publishing messages
using the configured message broker (Redis or Kafka).
"""
from typing import Dict, Any
from .message_broker import get_message_producer

# Get the appropriate producer based on configuration
producer = get_message_producer()

def publish_signal(signal: Dict[str, Any]) -> None:
    """Publish a trade signal to the message broker.
    
    Args:
        signal: Dictionary containing the trade signal data
    """
    from config.settings import settings
    producer.publish(settings.SIGNAL_TOPIC, signal)
