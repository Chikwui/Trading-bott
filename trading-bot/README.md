# AI-Powered Trading Bot with MT5 Integration

A high-frequency, multi-asset trading bot with AI/ML capabilities for algorithmic trading across various financial markets, featuring seamless MetaTrader 5 (MT5) integration for both live and demo trading.

## Features

- **Multi-Asset Support**: Trade Forex, Cryptocurrencies, Commodities, and Indices
- **MT5 Integration**: Full support for MetaTrader 5 platform with both live and demo account capabilities
- **AI/ML Integration**: Advanced signal generation using machine learning models
- **Risk Management**: Comprehensive risk controls and position sizing
- **Backtesting**: Historical strategy testing and optimization with `MT5BacktestBroker`
- **Live Trading**: Production-ready `MT5LiveBroker` for real trading operations
- **Real-time Data**: Live market data processing and analysis
- **Modular Architecture**: Easily extensible with custom strategies and indicators

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/trading-bot.git
   cd trading-bot
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   
   # Install MetaTrader5 package for MT5 integration
   pip install MetaTrader5 pandas numpy
   ```

4. **Install MetaTrader 5**
   - Download and install MT5 from [MetaQuotes](https://www.metatrader5.com/en/download)
   - Open MT5 and log in to your demo or live account
   - Ensure the MT5 terminal is running when using the live broker

5. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your MT5 credentials and settings
   ```

## Configuration

Edit the `.env` file with your configuration:

```ini
# Trading Mode (paper/live)
TRADING_MODE=paper

# MT5 Configuration
MT5_SERVER=YourBrokerServer
MT5_LOGIN=YourAccountNumber
MT5_PASSWORD=YourPassword
MT5_TIMEOUT=10000
MT5_POLLING_INTERVAL=0.1

# Redis Configuration (for advanced features)
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# Risk Parameters
RISK_PER_TRADE=0.01  # 1% risk per trade
MAX_DRAWDOWN=0.1     # 10% max drawdown
LEVERAGE=100         # Default leverage (1:100)

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/trading_bot.log
```

## Project Structure

```
trading-bot/
├── config/                 # Configuration files
│   ├── __init__.py
│   ├── settings.py         # Main settings
│   ├── risk_parameters.py  # Risk management settings
│   └── timeframes.py       # Timeframe definitions
├── core/                   # Core functionality
│   ├── __init__.py
│   ├── execution/          # Broker implementations
│   │   ├── __init__.py
│   │   ├── broker.py       # Base broker interface
│   │   ├── backtest/       # Backtesting implementation
│   │   └── live/           # Live trading implementation
│   │       └── mt5_live_broker.py  # MT5 live trading
│   ├── calendar/           # Market calendar and session management
│   ├── market/             # Market data and execution
│   └── strategy/           # Trading strategies
```

## Usage

### Running the Bot

```bash
# Run in live trading mode (MT5)
python -m trading_bot --mode live --broker mt5

# Run in backtest mode
python -m trading_bot --mode backtest --broker mt5 --strategy mean_reversion --symbols EURUSD,GBPUSD

# Run with specific timeframes
python -m trading_bot --timeframes 1h,4h,1d

# Run with custom config
python -m trading_bot --config config/custom_config.yaml
```

### Example: Using MT5 Live Broker

```python
from core.execution.live.mt5_live_broker import MT5LiveBroker
from core.models import Order, OrderType, OrderSide
import asyncio

async def main():
    # Initialize the broker
    broker = MT5LiveBroker(
        server="YourBrokerServer",
        login=12345678,  # Your MT5 account number
        password="your_password",
        timeout=10000
    )
    
    # Connect to MT5
    if not await broker.connect():
        print("Failed to connect to MT5")
        return
    
    # Place a market order
    order = Order(
        symbol="EURUSD",
        order_type=OrderType.MARKET,
        side=OrderSide.BUY,
        quantity=0.1,  # 0.1 lot
        comment="Test order"
    )
    
    try:
        order_id = await broker.place_order(order)
        print(f"Order placed: {order_id}")
        
        # Get order status
        status = await broker.get_order_status(order_id)
        print(f"Order status: {status}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # Disconnect
    await broker.disconnect()

if __name__ == "__main__":
    asyncio.run(main())
```

## Testing

### Unit Tests

```bash
# Run all unit tests
pytest tests/unit/

# Run MT5 Live Broker tests
pytest tests/unit/execution/test_mt5_live_broker.py -v
```

### Integration Tests

```bash
# Run integration tests (requires MT5 demo account)
pytest tests/integration/execution/test_mt5_live_integration.py -v --log-cli-level=INFO

# Run with specific test
pytest tests/integration/execution/test_mt5_live_integration.py::TestMT5LiveBrokerIntegration::test_market_order_round_trip -v
```

### Test Coverage

To generate a test coverage report:

```bash
pytest --cov=core --cov-report=html
start htmlcov/index.html  # On Windows
open htmlcov/index.html   # On macOS/Linux
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## Disclaimer

This software is for educational purposes only. Use at your own risk. The authors are not responsible for any financial losses incurred while using this software. Always test thoroughly with paper trading before using real funds.
