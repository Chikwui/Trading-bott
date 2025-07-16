"""
Message broker factory and interfaces for the trading bot.

This module provides a unified interface for different message broker implementations
(Redis, Kafka) to be used interchangeably in the trading bot.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Callable, Optional, List
from config.settings import settings
from utils.logger import logger

class MessageProducer(ABC):
    """Abstract base class for message producers."""
    
    @abstractmethod
    def publish(self, topic: str, message: Dict[str, Any]) -> None:
        """Publish a message to a topic.
        
        Args:
            topic: The topic/channel to publish to
            message: The message to publish
        """
        pass

class MessageConsumer(ABC):
    """Abstract base class for message consumers."""
    
    @abstractmethod
    def subscribe(self, topics: List[str], callback: Callable[[Dict[str, Any]], None]) -> None:
        """Subscribe to one or more topics and process messages with the callback.
        
        Args:
            topics: List of topics to subscribe to
            callback: Function to call with each received message
        """
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop consuming messages and clean up resources."""
        pass

def get_message_producer(broker_type: Optional[str] = None) -> MessageProducer:
    """Get a message producer instance based on configuration.
    
    Args:
        broker_type: The type of broker to use ('redis' or 'kafka'). 
                    If None, uses the value from settings.MESSAGE_BROKER
                    
    Returns:
        An instance of a MessageProducer implementation
    """
    broker_type = broker_type or settings.MESSAGE_BROKER.lower()
    
    if broker_type == 'redis':
        from .redis_client import RedisProducer
        return RedisProducer()
    elif broker_type == 'kafka':
        from kafka import KafkaProducer
        from .producer import publish_signal
        # Return a wrapper that matches our interface
        class KafkaProducerWrapper(MessageProducer):
            def publish(self, topic: str, message: Dict[str, Any]) -> None:
                publish_signal(message)
        return KafkaProducerWrapper()
    else:
        raise ValueError(f"Unsupported message broker: {broker_type}")

def get_message_consumer(
    topics: List[str], 
    broker_type: Optional[str] = None
) -> MessageConsumer:
    """Get a message consumer instance based on configuration.
    
    Args:
        topics: List of topics to subscribe to
        broker_type: The type of broker to use ('redis' or 'kafka').
                    If None, uses the value from settings.MESSAGE_BROKER
                    
    Returns:
        An instance of a MessageConsumer implementation
    """
    broker_type = broker_type or settings.MESSAGE_BROKER.lower()
    
    if broker_type == 'redis':
        from .redis_client import RedisConsumer
        return RedisConsumer(topics=topics)
    elif broker_type == 'kafka':
        from kafka import KafkaConsumer
        from .consumer import consume_ticks
        # Return a wrapper that matches our interface
        class KafkaConsumerWrapper(MessageConsumer):
            def __init__(self, topics: List[str]):
                self.topics = topics
                self.running = False
                
            def subscribe(self, callback: Callable[[Dict[str, Any]], None]) -> None:
                self.running = True
                consume_ticks(callback=callback)
                
            def stop(self) -> None:
                self.running = False
                
        return KafkaConsumerWrapper(topics=topics)
    else:
        raise ValueError(f"Unsupported message broker: {broker_type}")

# Default producer/consumer instances for convenience
producer = get_message_producer()

def get_consumer(topics: List[str]) -> MessageConsumer:
    """Get a consumer for the specified topics using the default broker."""
    return get_message_consumer(topics=topics)

def publish_message(topic: str, message: Dict[str, Any]) -> None:
    """Publish a message to the specified topic using the default producer."""
    producer.publish(topic, message)
