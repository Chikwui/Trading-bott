# AI-Powered Trading Bot with MT5 Integration

A high-performance, multi-asset algorithmic trading platform with AI/ML capabilities, featuring seamless MetaTrader 5 (MT5) integration, comprehensive risk management, and real-time monitoring.

## 🚀 Key Features

### Trading & Execution
- **Multi-Asset Support**: Forex, Cryptocurrencies, Commodities, Indices, and more
- **MT5 Integration**: Full support for MetaTrader 5 with live and demo account capabilities
- **Advanced Order Types**: Market, Limit, Stop, OCO, Bracket, and Iceberg orders
- **Smart Order Routing**: Optimized execution across multiple liquidity providers

### AI/ML Capabilities
- **Predictive Analytics**: ML models for price prediction and signal generation
- **Sentiment Analysis**: News and social media sentiment integration
- **Adaptive Strategies**: Self-optimizing trading strategies using reinforcement learning

### Risk Management
- **Position Sizing**: Dynamic position sizing based on account equity and risk parameters
- **Circuit Breakers**: Automatic trading suspension during extreme market conditions
- **Exposure Limits**: Cross-asset correlation and exposure management
- **Real-time P&L**: Comprehensive profit/loss tracking and reporting

### Monitoring & Analytics
- **Real-time Dashboard**: Web-based monitoring interface with live metrics
- **Distributed Tracing**: End-to-end request tracing for performance analysis
- **Alerting System**: Customizable alerts for key market events and system metrics
- **Comprehensive Logging**: Structured logging with log rotation and remote logging support

### Backtesting & Optimization
- **Multi-timeframe Analysis**: Support for multiple timeframes in backtesting
- **Walk-Forward Testing**: Robust validation of trading strategies
- **Monte Carlo Simulation**: Risk assessment through randomized testing
- **Parameter Optimization**: Grid search and genetic algorithm optimization

### Infrastructure
- **Microservices Architecture**: Scalable and maintainable design
- **Containerized Deployment**: Docker and Kubernetes support
- **High Availability**: Fault-tolerant design with automatic failover
- **Secure API**: JWT authentication and rate limiting

## 🛠 Installation

### Prerequisites
- Python 3.9+
- MetaTrader 5 terminal installed and configured
- Redis server (for caching and pub/sub)
- Docker and Docker Compose (for containerized deployment)

### Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/trading-bot.git
cd trading-bot

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt  # For development

# 4. Install MetaTrader5 package
pip install MetaTrader5

# 5. Set up environment variables
cp .env.example .env
# Edit .env with your configuration
```

### Docker Deployment

```bash
# Build and start all services
docker-compose up -d --build

# View logs
docker-compose logs -f

# Scale workers
docker-compose up -d --scale worker=4
```

### Kubernetes Deployment

```bash
# Apply Kubernetes configurations
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configs/
kubectl apply -f k8s/volumes/
kubectl apply -f k8s/services/

# Deploy applications
kubectl apply -f k8s/deployments/
```

## ⚙️ Configuration

### Configuration Overview

The trading bot uses a combination of environment variables and configuration files for flexible deployment. The main configuration is loaded from the `.env` file in the project root, with sensitive values stored as environment variables in production.

### Quick Start

1. Copy the example configuration:
   ```bash
   cp .env.example .env
   ```
2. Edit `.env` with your specific settings
3. For production, set environment variables directly on your deployment platform

### Configuration Reference

#### Core Application

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `TRADING_MODE` | string | `paper` | Trading mode: `paper` or `live` |
| `LOG_LEVEL` | string | `INFO` | Logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `LOG_TO_FILE` | bool | `true` | Whether to write logs to a file |
| `LOG_FILE` | string | `logs/trading_bot.log` | Path to log file |
| `TIMEZONE` | string | `UTC` | Timezone for all timestamps |

#### MetaTrader 5 (MT5) Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MT5_SERVER` | string | - | MT5 broker server name |
| `MT5_LOGIN` | int | - | MT5 account number |
| `MT5_PASSWORD` | string | - | MT5 account password |
| `MT5_TIMEOUT` | int | `60000` | Connection timeout in milliseconds |
| `MT5_PORTABLE` | bool | `False` | Use portable MT5 installation |
| `MT5_PATH` | string | - | Path to MT5 terminal executable |
| `ORDER_EXECUTION_TIMEOUT` | int | `30` | Seconds to wait for order execution |
| `MAX_ORDER_RETRIES` | int | `3` | Maximum order submission retries |
| `ORDER_CONFIRMATION_TIMEOUT` | int | `10` | Seconds to wait for order confirmation |

#### Redis Configuration

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `REDIS_HOST` | string | `localhost` | Redis server hostname |
| `REDIS_PORT` | int | `6379` | Redis server port |
| `REDIS_PASSWORD` | string | - | Redis authentication password |
| `REDIS_DB` | int | `0` | Redis database number |
| `REDIS_KEY_PREFIX` | string | `trading_bot:` | Prefix for all Redis keys |
| `REDIS_CACHE_TTL` | int | `3600` | Default TTL for cached items in seconds |

#### Message Queue (RabbitMQ)

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `RABBITMQ_HOST` | string | `localhost` | RabbitMQ server hostname |
| `RABBITMQ_PORT` | int | `5672` | RabbitMQ server port |
| `RABBITMQ_USER` | string | `guest` | RabbitMQ username |
| `RABBITMQ_PASSWORD` | string | `guest` | RabbitMQ password |
| `RABBITMQ_VHOST` | string | `/` | RabbitMQ virtual host |
| `RABBITMQ_QUEUE_PREFIX` | string | `trading_bot` | Prefix for all queue names |

#### Monitoring & Alerting

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MONITORING_ENABLED` | bool | `true` | Enable/disable monitoring |
| `METRICS_PORT` | int | `9090` | Port for metrics endpoint |
| `METRICS_PATH` | string | `/metrics` | Path for metrics endpoint |
| `HEALTH_CHECK_PATH` | string | `/health` | Path for health check endpoint |
| `ALERT_MANAGER_URL` | string | - | AlertManager server URL |
| `ALERT_MANAGER_API_KEY` | string | - | API key for AlertManager |
| `ALERT_RECIPIENTS` | string | - | Comma-separated list of alert recipients |
| `ALERT_THRESHOLD_CRITICAL` | int | `5` | Critical alert threshold |
| `ALERT_THRESHOLD_WARNING` | int | `10` | Warning alert threshold |

#### Data Storage

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `DATA_DIR` | string | `./data` | Directory for market data |
| `BACKTEST_RESULTS_DIR` | string | `./backtest_results` | Directory for backtest results |
| `INFLUXDB_URL` | string | - | InfluxDB server URL |
| `INFLUXDB_TOKEN` | string | - | InfluxDB authentication token |
| `INFLUXDB_ORG` | string | - | InfluxDB organization |
| `INFLUXDB_BUCKET` | string | - | InfluxDB bucket name |
| `INFLUXDB_RETENTION` | string | `30d` | Data retention period |

#### Market Data

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `SYMBOLS` | string | `EURUSD,GBPUSD,USDJPY,XAUUSD` | Default trading symbols |
| `TIMEFRAMES` | string | `M1,M5,M15,H1,H4,D1` | Supported timeframes |
| `MARKET_DATA_PROVIDER` | string | `MT5` | Data provider: `MT5`, `CCXT`, etc. |
| `MARKET_DATA_API_KEY` | string | - | API key for market data |
| `MARKET_DATA_SECRET` | string | - | API secret for market data |
| `UPDATE_INTERVAL` | int | `60` | Data update interval in seconds |
| `HISTORICAL_BARS` | int | `1000` | Number of historical bars to load |
| `MAX_DATA_AGE_MINUTES` | int | `15` | Maximum allowed data age |
| `DATA_VALIDATION_ENABLED` | bool | `true` | Enable data validation |

#### Security

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `AUTH_ENABLED` | bool | `true` | Enable authentication |
| `API_KEYS` | string | - | Comma-separated list of valid API keys |
| `JWT_SECRET` | string | - | Secret key for JWT tokens |
| `JWT_ALGORITHM` | string | `HS256` | JWT signing algorithm |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | int | `1440` | JWT token expiration in minutes |
| `RATE_LIMIT` | string | `100/minute` | API rate limit |
| `MAX_REQUESTS_PER_DAY` | int | `10000` | Daily request limit |
| `CORS_ORIGINS` | string | - | Allowed CORS origins |
| `SESSION_SECRET` | string | - | Session secret key |
| `SESSION_COOKIE_SECURE` | bool | `true` | Use secure cookies |
| `SESSION_COOKIE_HTTPONLY` | bool | `true` | HTTP-only cookies |
| `SESSION_COOKIE_SAMESITE` | string | `lax` | SameSite cookie policy |

#### Risk Management

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `INITIAL_BALANCE` | float | `10000.0` | Initial account balance |
| `RISK_PER_TRADE` | float | `0.01` | Risk per trade (0.01 = 1%) |
| `MAX_DRAWDOWN` | float | `0.1` | Maximum drawdown (0.1 = 10%) |
| `MAX_LEVERAGE` | float | `10.0` | Maximum leverage |
| `DAILY_LOSS_LIMIT` | float | `0.05` | Daily loss limit (0.05 = 5%) |
| `ENABLE_CIRCUIT_BREAKERS` | bool | `true` | Enable circuit breakers |
| `CIRCUIT_BREAKER_THRESHOLD` | int | `5` | Failures before opening circuit |
| `CIRCUIT_BREAKER_TIMEOUT` | int | `300` | Seconds before retrying closed circuit |
| `MAX_DAILY_TRADES` | int | `100` | Maximum trades per day |
| `MAX_OPEN_POSITIONS` | int | `10` | Maximum concurrent positions |
| `STOP_LOSS_PIPS` | int | `20` | Default stop loss in pips |
| `TAKE_PROFIT_PIPS` | int | `40` | Default take profit in pips |

### Configuration Files

The configuration is organized into the following directories:

- `config/` - Main configuration directory
  - `settings.py` - Core application settings
  - `strategies/` - Strategy configurations
  - `indicators/` - Technical indicator parameters
  - `risk/` - Risk management settings
  - `brokers/` - Broker-specific configurations
  - `logging.yaml` - Logging configuration

### Logging Configuration

Logging is configured in `config/logging.yaml` with the following features:

- **Console Output**: Colored output with log levels
- **File Output**: JSON-formatted logs with rotation
  - Max file size: 10MB
  - Backup count: 5
  - Compression: Enabled
- **Error Tracking**: Sentry integration (if configured)
- **Structured Logging**: JSON format with timestamps and context

### Environment-Specific Configuration

For different environments (development, staging, production), you can use the following pattern:

1. Create environment-specific files (e.g., `.env.development`, `.env.production`)
2. Set the `ENVIRONMENT` variable to load the appropriate configuration
3. Use `python-dotenv` to load the correct file:
   ```python
   from dotenv import load_dotenv
   load_dotenv(f'.env.{os.getenv("ENVIRONMENT", "development")}')
   ```

### Security Best Practices

1. Never commit sensitive values to version control
2. Use secret management in production (e.g., AWS Secrets Manager, HashiCorp Vault)
3. Set appropriate file permissions on configuration files
4. Rotate API keys and secrets regularly
5. Use environment variables for sensitive data in production

## MT5 Dashboard Setup and Usage

### Prerequisites

- Python 3.8+
- MetaTrader 5 terminal installed and running
- MT5 account credentials

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/trading-bot.git
   cd trading-bot
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your MT5 credentials:
   ```
   MT5_SERVER=YourServerName
   MT5_LOGIN=YourLogin
   MT5_PASSWORD=YourPassword
   ```

### Running the Dashboard

To start the MT5 Trading Dashboard:

```bash
python run_dashboard.py
```

Optional arguments:
- `--host`: Host to run the dashboard on (default: 0.0.0.0)
- `--port`: Port to run the dashboard on (default: 8050)
- `--debug`: Enable debug mode

Example:
```bash
python run_dashboard.py --host 127.0.0.1 --port 8080 --debug
```

The dashboard will be available at `http://localhost:8050` by default.

### Dashboard Features

- **Account Overview**: View balance, equity, margin, and free margin
- **Positions**: Monitor open positions with real-time P&L
- **Orders**: Track open orders and their status
- **Market Data**: View real-time price charts and indicators
- **Alerts**: Get notified of important trading events

## Project Structure

```
trading-bot/
├── core/                    # Core functionality
│   ├── data/               # Data providers and market data handling
│   ├── execution/          # Order execution and broker interfaces
│   ├── monitoring/         # Monitoring and dashboard components
│   └── utils/              # Utility functions and helpers
├── tests/                  # Unit and integration tests
├── .env.example           # Example environment variables
├── requirements.txt       # Python dependencies
└── run_dashboard.py       # Main entry point for the dashboard
```

## Development

### Running Tests

```bash
pytest tests/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## Disclaimer

This software is for educational purposes only. Use at your own risk. The authors are not responsible for any financial losses incurred while using this software. Always test thoroughly with paper trading before using real funds.
