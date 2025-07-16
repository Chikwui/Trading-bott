"""
Integration test for Redis Pub/Sub in the trading bot.

This script tests the Redis Pub/Sub integration by publishing test messages
and verifying they are received correctly.
"""
import time
import json
import threading
import unittest
from unittest.mock import patch, MagicMock
from config.settings import settings
from streaming.message_broker import get_message_producer, get_message_consumer
from utils.logger import setup_logging

# Configure logging
setup_logging()

class TestRedisIntegration(unittest.TestCase):    
    def setUp(self):
        """Set up test environment."""
        self.test_topic = "test_topic"
        self.test_message = {
            "symbol": "BTCUSD",
            "price": 50000.0,
            "timestamp": int(time.time())
        }
        self.received_messages = []
        self.message_received = threading.Event()
        
    def message_handler(self, message):
        """Handle received messages."""
        self.received_messages.append(message)
        self.message_received.set()
    
    def test_pub_sub_flow(self):
        """Test the publish-subscribe flow with Redis."""
        # Skip if not using Redis
        if settings.MESSAGE_BROKER.lower() != 'redis':
            self.skipTest("Redis is not the configured message broker")
            
        # Create consumer in a separate thread
        consumer = get_message_consumer(topics=[self.test_topic])
        consumer_thread = threading.Thread(
            target=consumer.subscribe,
            args=(self.message_handler,),
            daemon=True
        )
        consumer_thread.start()
        
        # Give consumer time to subscribe
        time.sleep(1)
        
        try:
            # Publish a test message
            producer = get_message_producer()
            producer.publish(self.test_topic, self.test_message)
            
            # Wait for message to be received (with timeout)
            received = self.message_received.wait(timeout=5.0)
            self.assertTrue(received, "Message was not received within timeout")
            
            # Verify the message
            self.assertEqual(len(self.received_messages), 1)
            received_message = self.received_messages[0]
            
            # Check that all expected fields are present
            for key in ["symbol", "price", "timestamp"]:
                self.assertIn(key, received_message)
                self.assertEqual(received_message[key], self.test_message[key])
                
            print("\n✅ Redis Pub/Sub test passed successfully!")
            
        finally:
            # Clean up
            if hasattr(consumer, 'stop'):
                consumer.stop()
            consumer_thread.join(timeout=2.0)

if __name__ == "__main__":
    unittest.main()
