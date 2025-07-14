import json
from kafka import KafkaProducer
from config.settings import settings
from utils.logger import logger

producer = KafkaProducer(
    bootstrap_servers=[settings.KAFKA_BROKER_URL],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def publish_signal(signal: dict):
    """Publish a trade signal to Kafka."""
    try:
        producer.send(settings.KAFKA_SIGNAL_TOPIC, signal)
        producer.flush()
        logger.info(f"Published signal: {signal}")
    except Exception as e:
        logger.error(f"Failed to publish signal: {e}")
