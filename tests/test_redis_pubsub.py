"""
Redis Pub/Sub Test Script

This script tests the Redis Pub/Sub functionality by creating a publisher and subscriber
that communicate through a Redis channel.
"""
import redis
import time
import threading
import logging
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RedisPubSubTest:
    """A class to test Redis Pub/Sub functionality."""
    
    def __init__(self, host: str = 'localhost', port: int = 6379, channel: str = 'trading_signals'):
        """Initialize the Redis connection and Pub/Sub handler.
        
        Args:
            host: Redis server hostname
            port: Redis server port
            channel: Channel name for Pub/Sub
        """
        self.redis_conn = redis.Redis(host=host, port=port, decode_responses=True)
        self.pubsub = self.redis_conn.pubsub()
        self.channel = channel
        self.running = False
        
    def publish_messages(self, count: int = 5, delay: float = 1.0) -> None:
        """Publish test messages to the channel.
        
        Args:
            count: Number of messages to publish
            delay: Delay between messages in seconds
        """
        try:
            for i in range(count):
                message = f"Test message {i+1} at {time.ctime()}"
                self.redis_conn.publish(self.channel, message)
                logger.info(f"Published: {message}")
                time.sleep(delay)
        except Exception as e:
            logger.error(f"Error in publisher: {e}")
        finally:
            self.running = False
            
    def subscribe_and_listen(self) -> None:
        """Subscribe to the channel and listen for incoming messages."""
        try:
            self.pubsub.subscribe(self.channel)
            logger.info(f"Subscribed to {self.channel}. Waiting for messages...")
            
            self.running = True
            while self.running:
                message = self.pubsub.get_message(timeout=1.0)
                if message and message['type'] == 'message':
                    logger.info(f"Received: {message['data']}")
                time.sleep(0.01)  # Small sleep to prevent busy waiting
                    
        except Exception as e:
            logger.error(f"Error in subscriber: {e}")
        finally:
            self.running = False
            logger.info("Subscriber stopped")
            
    def run_test(self, message_count: int = 5) -> None:
        """Run the Pub/Sub test.
        
        Args:
            message_count: Number of test messages to send
        """
        try:
            # Start subscriber in a separate thread
            subscriber_thread = threading.Thread(
                target=self.subscribe_and_listen,
                daemon=True
            )
            subscriber_thread.start()
            
            # Give subscriber time to connect
            time.sleep(1)
            
            # Publish test messages
            self.publish_messages(count=message_count)
            
            # Keep the script running to receive messages
            try:
                while self.running:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                logger.info("Test interrupted by user")
            
        except Exception as e:
            logger.error(f"Test failed: {e}")
        finally:
            self.running = False
            if 'subscriber_thread' in locals():
                subscriber_thread.join(timeout=2.0)
            logger.info("Test completed")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Redis Pub/Sub functionality')
    parser.add_argument('--host', default='localhost', help='Redis server host')
    parser.add_argument('--port', type=int, default=6379, help='Redis server port')
    parser.add_argument('--channel', default='trading_signals', help='Pub/Sub channel name')
    parser.add_argument('--count', type=int, default=5, help='Number of test messages to send')
    
    args = parser.parse_args()
    
    # Create and run the test
    test = RedisPubSubTest(host=args.host, port=args.port, channel=args.channel)
    test.run_test(message_count=args.count)
