"""
Message consumer for the trading bot.

This module provides a unified interface for consuming messages
using the configured message broker (Redis or Kafka).
"""
import time
from typing import Callable, Dict, Any, Optional
from .message_broker import get_message_consumer
from config.settings import settings
from utils.logger import logger

def consume_ticks(callback: Callable[[Dict[str, Any]], None], 
                poll_timeout_ms: int = 1000, 
                max_retries: int = 3) -> None:
    """
    Consume market tick messages and invoke callback for each tick.
    
    Args:
        callback: Function to call with each tick message
        poll_timeout_ms: Time in milliseconds to wait for new messages (not used in all brokers)
        max_retries: Maximum number of connection retries
    """
    retry_count = 0
    consumer = None
    
    while retry_count < max_retries:
        try:
            # Get the appropriate consumer based on configuration
            consumer = get_message_consumer(topics=[settings.TICK_TOPIC])
            logger.info(f"Subscribed to {settings.TICK_TOPIC}. Waiting for market data...")
            
            # Start consuming messages
            consumer.subscribe(callback=callback)
            
            # If we get here, the consumer was stopped
            break
            
        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                wait_time = min(5 * retry_count, 30)  # Exponential backoff, max 30s
                logger.warning(
                    f"Error in consumer (attempt {retry_count}/{max_retries}): {e}. "
                    f"Retrying in {wait_time} seconds..."
                )
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to start consumer after {max_retries} attempts. Error: {e}")
                raise
                
        finally:
            if consumer:
                try:
                    consumer.stop()
                except Exception as e:
                    logger.error(f"Error stopping consumer: {e}")
                
                # Small delay before next retry
                if retry_count < max_retries:
                    time.sleep(1)
    
    logger.error(f"Failed to connect to Kafka after {max_retries} attempts. Exiting...")
