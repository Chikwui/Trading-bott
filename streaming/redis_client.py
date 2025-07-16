"""
Redis Pub/Sub client for the trading bot.

This module provides Redis-based implementations of the message producer and consumer
for the trading bot's event streaming needs.
"""
import json
import threading
import logging
import redis
from typing import Callable, Optional, Dict, Any
from config.settings import settings
from utils.logger import logger

class RedisClient:
    """Redis client wrapper with Pub/Sub capabilities."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(RedisClient, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance
    
    def _initialize(self):
        """Initialize the Redis connection."""
        try:
            self.redis_conn = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                password=settings.REDIS_PASSWORD or None,
                decode_responses=True,
                socket_timeout=10,
                socket_connect_timeout=10,
                retry_on_timeout=True,
                health_check_interval=30
            )
            # Test the connection
            self.redis_conn.ping()
            logger.info(f"Connected to Redis at {settings.REDIS_HOST}:{settings.REDIS_PORT}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

class RedisProducer:
    """Redis-based message producer."""
    
    def __init__(self, client: Optional[RedisClient] = None):
        """Initialize the Redis producer.
        
        Args:
            client: Optional RedisClient instance. If not provided, a new one will be created.
        """
        self.client = client or RedisClient()
        self.redis = self.client.redis_conn
    
    def publish(self, topic: str, message: Dict[str, Any]) -> None:
        """Publish a message to a Redis channel.
        
        Args:
            topic: The channel/topic to publish to
            message: The message to publish (will be JSON-serialized)
        """
        try:
            serialized = json.dumps(message)
            self.redis.publish(topic, serialized)
            logger.debug(f"Published to {topic}: {message}")
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            raise

class RedisConsumer:
    """Redis Pub/Sub consumer with message handling."""
    
    def __init__(self, topics: list, client: Optional[RedisClient] = None):
        """Initialize the Redis consumer.
        
        Args:
            topics: List of topics/channels to subscribe to
            client: Optional RedisClient instance. If not provided, a new one will be created.
        """
        self.client = client or RedisClient()
        self.redis = self.client.redis_conn
        self.pubsub = self.redis.pubsub()
        self.topics = topics
        self.running = False
        self.handlers = {}
    
    def subscribe(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Subscribe to topics and start consuming messages.
        
        Args:
            callback: Function to call with each received message
        """
        try:
            # Subscribe to all topics
            self.pubsub.subscribe(*self.topics)
            logger.info(f"Subscribed to topics: {', '.join(self.topics)}")
            
            # Start consuming messages
            self.running = True
            for message in self.pubsub.listen():
                if not self.running:
                    break
                    
                if message['type'] == 'message':
                    try:
                        # Deserialize the message
                        data = json.loads(message['data'])
                        # Call the handler
                        callback(data)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to decode message: {message['data']}")
                    except Exception as e:
                        logger.error(f"Error processing message: {e}")
                        
        except Exception as e:
            logger.error(f"Error in Redis consumer: {e}")
            raise
        finally:
            self.unsubscribe()
    
    def unsubscribe(self) -> None:
        """Unsubscribe from all topics and clean up."""
        if hasattr(self, 'pubsub'):
            self.pubsub.unsubscribe()
            self.running = False
            logger.info("Unsubscribed from all topics")

# Singleton instance for easy access
redis_client = RedisClient()
