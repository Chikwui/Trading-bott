# Configuration

This section describes key configuration options for AI Trader.

## Environment Variables
All options can be set via the `.env` file or environment variables.

```ini
# Kafka & Streaming
KAFKA_BROKER_URL=localhost:9092
KAFKA_TICK_TOPIC=market_ticks
KAFKA_SIGNAL_TOPIC=trade_signals

# Feature Store & MLOps
FEAST_SERVING_URL=http://localhost:6566
MLFLOW_TRACKING_URI=http://localhost:5000

# Database
DATABASE_URL=sqlite:///ai_trader.db
# (or) postgresql://trader:secret@localhost:5432/ai_trader

# MT5
MT5_LOGIN=123456
MT5_PASSWORD=your_password
MT5_SERVER=your_broker_server

# Risk Management
RISK_PERCENT_PER_TRADE=1.0
MAX_DAILY_DRAWDOWN_PERCENT=5.0
MAX_OPEN_POSITIONS=5
ATR_PERIOD=14
ATR_MULTIPLIER=2.0

# Decision Thresholds
MIN_CONFIDENCE_THRESHOLD=0.7
AUTO_APPROVE_THRESHOLD=0.85
AUTO_REJECT_THRESHOLD=0.30

# AI/ML Models
MODEL_PATH=models/signal_predictor.joblib
SENTIMENT_MODEL=ProsusAI/finbert

# Trading Schedule (0=Monday … 4=Friday)
TRADING_SCHEDULE={"0":[["09:30","16:00"]],"1":[["09:30","16:00"]],"2":[["09:30","16:00"]],"3":[["09:30","16:00"]],"4":[["09:30","16:00"]]}
```
