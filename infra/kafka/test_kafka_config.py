import os
import time
from kafka import KafkaProducer, KafkaConsumer
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import KafkaError
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Kafka configuration
KAFKA_BROKER = 'localhost:9092'
TEST_TOPIC = 'trading-test'


def test_producer_performance():
    """Test producer performance and reliability"""
    producer = KafkaProducer(
        bootstrap_servers=[KAFKA_BROKER],
        value_serializer=lambda x: x.encode('utf-8'),
        retries=3,
        request_timeout_ms=30000,
        api_version=(2, 0, 2)
    )

    try:
        # Test message batch
        messages = [f"test-message-{i}" for i in range(1000)]
        start_time = time.time()
        
        for msg in messages:
            producer.send(TEST_TOPIC, msg)
        
        producer.flush()
        end_time = time.time()
        
        logging.info(f"Sent 1000 messages in {end_time - start_time:.2f} seconds")
        logging.info(f"Throughput: {1000 / (end_time - start_time):.2f} messages/sec")
        
    except KafkaError as e:
        logging.error(f"Producer error: {str(e)}")
    finally:
        producer.close()


def test_consumer_lag():
    """Test consumer lag and message delivery"""
    consumer = KafkaConsumer(
        TEST_TOPIC,
        bootstrap_servers=[KAFKA_BROKER],
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='test-group',
        value_deserializer=lambda x: x.decode('utf-8'),
        api_version=(2, 0, 2)
    )

    try:
        # Wait for messages
        start_time = time.time()
        received = 0
        for msg in consumer:
            received += 1
            if received >= 1000:
                break
        
        end_time = time.time()
        logging.info(f"Received 1000 messages in {end_time - start_time:.2f} seconds")
        logging.info(f"Consumer throughput: {1000 / (end_time - start_time):.2f} messages/sec")
        
    except KafkaError as e:
        logging.error(f"Consumer error: {str(e)}")
    finally:
        consumer.close()


def test_topic_creation():
    """Test topic creation and configuration"""
    admin_client = KafkaAdminClient(
        bootstrap_servers=[KAFKA_BROKER],
        api_version=(2, 0, 2)
    )

    try:
        # Create test topic with specific configuration
        topic = NewTopic(
            name=TEST_TOPIC,
            num_partitions=3,
            replication_factor=3,
            topic_configs={
                'retention.ms': '604800000',  # 7 days
                'cleanup.policy': 'delete',
                'min.insync.replicas': '2'
            }
        )
        
        admin_client.create_topics([topic])
        logging.info(f"Successfully created topic {TEST_TOPIC}")
        
        # Verify topic configuration
        topic_configs = admin_client.describe_configs([topic])
        logging.info(f"Topic configuration: {topic_configs}")
        
    except KafkaError as e:
        logging.error(f"Topic creation error: {str(e)}")
    finally:
        admin_client.close()


def main():
    logging.info("Starting Kafka configuration test...")
    
    # Test topic creation
    test_topic_creation()
    
    # Test producer performance
    test_producer_performance()
    
    # Test consumer lag
    test_consumer_lag()
    
    logging.info("Kafka configuration test completed")


if __name__ == '__main__':
    main()
