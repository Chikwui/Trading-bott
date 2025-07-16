"""
Test Redis connection and Pub/Sub functionality for the trading bot.

This script tests the Redis connection and verifies that Pub/Sub messaging
is working correctly with the trading bot's configuration.
"""
import time
import json
import threading
import redis
from config.settings import settings

def test_redis_connection():
    """Test Redis server connection."""
    try:
        r = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            password=settings.REDIS_PASSWORD or None,
            decode_responses=True
        )
        # Test the connection
        response = r.ping()
        print(f"✅ Successfully connected to Redis at {settings.REDIS_HOST}:{settings.REDIS_PORT}")
        return r
    except Exception as e:
        print(f"❌ Failed to connect to Redis: {e}")
        return None

def test_pub_sub():
    """Test Redis Pub/Sub functionality."""
    print("\n🔍 Testing Redis Pub/Sub...")
    
    # Create a Redis connection
    r = test_redis_connection()
    if not r:
        return
    
    test_channel = "test_channel"
    test_message = {
        "symbol": "BTCUSD",
        "price": 50000.0,
        "timestamp": int(time.time())
    }
    
    # Create a pubsub object
    pubsub = r.pubsub()
    
    # Subscribe to the test channel
    pubsub.subscribe(test_channel)
    print(f"✅ Subscribed to channel: {test_channel}")
    
    # Start a thread to listen for messages
    def listen_for_messages():
        for message in pubsub.listen():
            if message['type'] == 'message':
                try:
                    data = json.loads(message['data'])
                    print(f"\n📨 Received message:")
                    print(json.dumps(data, indent=2))
                    print("✅ Redis Pub/Sub is working correctly!")
                    break
                except json.JSONDecodeError:
                    print(f"Failed to decode message: {message['data']}")
    
    # Start the listener thread
    listener = threading.Thread(target=listen_for_messages, daemon=True)
    listener.start()
    
    # Give the listener time to subscribe
    time.sleep(1)
    
    # Publish a test message
    print(f"\n📤 Publishing test message to channel '{test_channel}':")
    print(json.dumps(test_message, indent=2))
    r.publish(test_channel, json.dumps(test_message))
    
    # Wait for the message to be received
    listener.join(timeout=5.0)
    
    # Clean up
    pubsub.unsubscribe()
    r.close()

if __name__ == "__main__":
    print("🔌 Testing Redis connection...")
    test_redis_connection()
    test_pub_sub()
