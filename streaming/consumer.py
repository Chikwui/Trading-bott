import json
from kafka import KafkaConsumer
from config.settings import settings
from utils.logger import logger

def consume_ticks(callback):
    """Consume market tick messages and invoke callback for each tick."""
    consumer = KafkaConsumer(
        settings.KAFKA_TICK_TOPIC,
        bootstrap_servers=[settings.KAFKA_BROKER_URL],
        value_deserializer=lambda m: json.loads(m.decode('utf-8')),
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='ai-trader-consumer'
    )
    for msg in consumer:
        tick = msg.value
        logger.debug(f"Received tick: {tick}")
        try:
            callback(tick)
        except Exception as e:
            logger.error(f"Error processing tick: {e}")
