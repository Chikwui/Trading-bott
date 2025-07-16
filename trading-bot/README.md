# AI-Powered Trading Bot

A high-frequency, multi-asset trading bot with AI/ML capabilities for algorithmic trading across various financial markets.

## Features

- **Multi-Asset Support**: Trade Forex, Cryptocurrencies, Commodities, and Indices
- **AI/ML Integration**: Advanced signal generation using machine learning models
- **Risk Management**: Comprehensive risk controls and position sizing
- **Backtesting**: Historical strategy testing and optimization
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
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys and settings
   ```

## Configuration

Edit the `.env` file with your configuration:

```ini
# Trading Mode (paper/live)
TRADING_MODE=paper

# Exchange API Credentials
EXCHANGE_API_KEY=your_api_key
EXCHANGE_API_SECRET=your_api_secret

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=

# Risk Parameters
RISK_PER_TRADE=0.01  # 1% risk per trade
MAX_DRAWDOWN=0.1     # 10% max drawdown
```

## Usage

### Running the Bot

```bash
python -m trading_bot
```

### Command Line Options

```bash
# Run with custom config
python -m trading_bot --config config/custom_config.yaml

# Run in backtest mode
python -m trading_bot --backtest --strategy mean_reversion --symbols BTC/USD,ETH/USD

# Run with specific timeframes
python -m trading_bot --timeframes 1h,4h,1d
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
│   ├── calendar/           # Market calendar and session management
│   ├── market/             # Market data and execution
│   └── strategy/           # Trading strategies
├── services/               # Service layer
│   ├── __init__.py
│   ├── market_data.py      # Market data service
│   └── signal_service.py   # Signal generation service
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── logger.py           # Logging configuration
│   └── helpers.py          # Helper functions
├── tests/                  # Unit and integration tests
├── .env.example            # Example environment variables
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

This project uses:
- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting

Run the following before committing:

```bash
black .
isort .
flake8
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This software is for educational purposes only. Use at your own risk. The authors are not responsible for any financial losses incurred while using this software. Always test thoroughly with paper trading before using real funds.
