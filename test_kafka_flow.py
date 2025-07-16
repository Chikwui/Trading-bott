import json
import time
from kafka import KafkaProducer, KafkaConsumer

# Configuration
TOPIC = 'test-topic'
BOOTSTRAP_SERVERS = ['localhost:9092']

print("\n=== Kafka Connectivity Test ===")

# Test Producer
try:
    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    test_messages = [{'msg': f'test{i}'} for i in range(3)]
    for msg in test_messages:
        producer.send(TOPIC, msg)
    producer.flush()
    print("Producer sent messages:", test_messages)
except Exception as e:
    print("Producer error:", e)
    exit(1)

# Give broker a moment
time.sleep(1)

# Test Consumer
try:
    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        auto_offset_reset='earliest',
        consumer_timeout_ms=5000,
        value_deserializer=lambda m: json.loads(m.decode('utf-8'))
    )
    received = [msg.value for msg in consumer]
    consumer.close()
    if received:
        print("Consumer received messages:", received)
    else:
        print("Consumer received no messages. Check broker and topic configuration.")
except Exception as e:
    print("Consumer error:", e)
    exit(1)

print("=== Test Complete ===\n")
