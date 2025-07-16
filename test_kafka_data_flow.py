import json
import time
import random
from datetime import datetime
from kafka import KafkaProducer, KafkaConsumer
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from config.settings import settings

# Initialize database connection
engine = create_engine(settings.DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()
Base = declarative_base()

# Define MarketData model
class MarketData(Base):
    __tablename__ = 'market_data'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String)
    bid = Column(Float)
    ask = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f"<MarketData(symbol={self.symbol}, bid={self.bid}, ask={self.ask})>"

# Create tables if they don't exist
Base.metadata.create_all(engine)

# Kafka configuration for local setup
KAFKA_BOOTSTRAP_SERVERS = 'localhost:9092'

def json_serializer(data):
    return json.dumps(data).encode('utf-8')

def json_deserializer(data):
    return json.loads(data.decode('utf-8'))

def generate_market_data():
    """Generate sample market data for testing."""
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
    base_price = {
        'EURUSD': 1.0800,
        'GBPUSD': 1.2600,
        'USDJPY': 151.50,
        'AUDUSD': 0.6550,
        'USDCAD': 1.3600
    }
    
    data = []
    for symbol in symbols:
        price = base_price[symbol] * (1 + random.uniform(-0.001, 0.001))
        spread = 0.0002 * price  # 2 pips spread
        data.append({
            'symbol': symbol,
            'bid': round(price - spread/2, 5),
            'ask': round(price + spread/2, 5),
            'timestamp': datetime.utcnow().isoformat()
        })
    return data

def produce_market_data():
    """Produce market data to Kafka topic."""
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=json_serializer
    )
    
    try:
        for i in range(5):  # Reduced to 5 batches for testing
            market_data = generate_market_data()
            for data in market_data:
                producer.send('test-topic', data)
                print(f"Produced: {data}")
            time.sleep(1)  # Simulate real-time data
    except Exception as e:
        print(f"Error producing data: {e}")
    finally:
        producer.flush()
        producer.close()

def consume_and_persist():
    """Consume market data and persist to database."""
    consumer = KafkaConsumer(
        'test-topic',
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='market-data-consumer',
        value_deserializer=json_deserializer
    )
    
    try:
        session = Session()
        for message in consumer:
            try:
                data = message.value
                # Convert string timestamp back to datetime
                data['timestamp'] = datetime.fromisoformat(data['timestamp'])
                
                # Create and save market data
                market_data = MarketData(**data)
                session.add(market_data)
                session.commit()
                print(f"Persisted: {market_data}")
                
            except Exception as e:
                session.rollback()
                print(f"Error processing message: {e}")
                
    except KeyboardInterrupt:
        print("Stopping consumer...")
    finally:
        session.close()
        consumer.close()

def verify_data_persistence():
    """Verify that data was correctly persisted to the database."""
    try:
        session = Session()
        count = session.query(MarketData).count()
        latest = session.query(MarketData).order_by(MarketData.timestamp.desc()).first()
        
        print("\n=== Data Persistence Verification ===")
        print(f"Total records in database: {count}")
        if latest:
            print(f"Latest record: {latest}")
        
        # Print sample records
        print("\nSample records:")
        for record in session.query(MarketData).order_by(MarketData.timestamp.desc()).limit(3):
            print(record)
            
    except Exception as e:
        print(f"Error verifying data: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    import threading
    import time
    
    print("=== Starting Kafka Data Flow Test ===")
    
    # Start consumer in a separate thread
    consumer_thread = threading.Thread(target=consume_and_persist, daemon=True)
    consumer_thread.start()
    
    # Give consumer time to initialize
    time.sleep(2)
    
    # Produce test data
    print("\nProducing test data...")
    produce_market_data()
    
    # Give consumer time to process all messages
    print("\nWaiting for data processing...")
    time.sleep(5)
    
    # Verify data was persisted
    verify_data_persistence()
    
    print("\n=== Test Complete ===")
    