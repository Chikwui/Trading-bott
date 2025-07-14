import os
import json
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # Kafka & Streaming
    KAFKA_BROKER_URL = os.getenv("KAFKA_BROKER_URL", "localhost:9092")
    # Kafka topics
    KAFKA_TICK_TOPIC = os.getenv("KAFKA_TICK_TOPIC", "market_ticks")
    KAFKA_SIGNAL_TOPIC = os.getenv("KAFKA_SIGNAL_TOPIC", "trade_signals")

    # Feast feature store
    FEAST_SERVING_URL = os.getenv("FEAST_SERVING_URL", "http://localhost:6566")

    # MLflow tracking
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    # Database
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///ai_trader.db")

    # MetaTrader 5
    MT5_LOGIN = int(os.getenv("MT5_LOGIN", 0))
    MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
    MT5_SERVER = os.getenv("MT5_SERVER", "")

    # Risk Management
    RISK_PERCENT_PER_TRADE = float(os.getenv("RISK_PERCENT_PER_TRADE", 1.0))
    MAX_DAILY_DRAWDOWN_PERCENT = float(os.getenv("MAX_DAILY_DRAWDOWN_PERCENT", 5.0))
    MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", 5))
    ATR_PERIOD = int(os.getenv("ATR_PERIOD", 14))
    ATR_MULTIPLIER = float(os.getenv("ATR_MULTIPLIER", 2.0))

    # Trading Schedule (JSON string)
    TRADING_SCHEDULE = json.loads(os.getenv("TRADING_SCHEDULE", "{}"))

    # Decision thresholds
    MIN_CONFIDENCE_THRESHOLD = float(os.getenv("MIN_CONFIDENCE_THRESHOLD", 0.7))
    AUTO_APPROVE_THRESHOLD = float(os.getenv("AUTO_APPROVE_THRESHOLD", 0.85))
    AUTO_REJECT_THRESHOLD = float(os.getenv("AUTO_REJECT_THRESHOLD", 0.30))

# AI/ML Models
MODEL_PATH = os.getenv("MODEL_PATH", "")
SENTIMENT_MODEL = os.getenv("SENTIMENT_MODEL", "")

settings = Settings()
