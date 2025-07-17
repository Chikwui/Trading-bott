# Backtest Broker Implementation

This module provides a backtesting implementation of the `Broker` interface that simulates trading on the MT5 platform. It's designed to be used for strategy development, testing, and optimization without risking real capital.

## Features

- **Realistic Market Simulation**: Simulates market conditions including spread, slippage, and commission
- **Order Types**: Supports market, limit, and stop orders
- **Position Management**: Tracks open positions, P&L, and account balance
- **Historical Data**: Uses MT5 historical data for backtesting
- **Risk Management**: Implements stop-loss and take-profit orders
- **Performance Metrics**: Trades, win rate, P&L, and other key metrics

## Components

### `MT5BacktestBroker`

The main broker class that implements the `Broker` interface. It handles order execution, position management, and account updates during backtesting.

#### Key Methods

- `connect()`: Initialize the backtest environment
- `disconnect()`: Clean up resources
- `place_order(order)`: Place a new order
- `cancel_order(order_id)`: Cancel an existing order
- `get_order_status(order_id)`: Get the status of an order
- `get_positions(symbol)`: Get open positions
- `get_account_info()`: Get current account information
- `get_historical_data()`: Get historical price data
- `get_current_price(symbol)`: Get the current market price
- `get_order_book(symbol)`: Get the current order book
- `update_market_data(timestamp)`: Update market data to the specified timestamp

### Factory Function

- `create_mt5_backtest_broker()`: Helper function to create a configured instance of `MT5BacktestBroker`

## Usage Example

```python
from datetime import datetime, timedelta
from core.data.providers.mt5_provider import MT5DataProvider
from core.execution.backtest.mt5_backtest_broker import create_mt5_backtest_broker
from core.models import Order, OrderType, OrderSide, TimeInForce

async def run_backtest():
    # Initialize data provider
    data_provider = MT5DataProvider()
    await data_provider.initialize()
    
    # Create backtest broker
    broker = create_mt5_backtest_broker(
        data_provider=data_provider,
        initial_balance=10000.0,
        leverage=1.0,
        commission=0.0005,  # 0.05% commission
        spread=0.0002,     # 2 pips spread
        slippage=0.0001    # 1 pip slippage
    )
    
    # Connect to the broker
    await broker.connect()
    
    # Example: Place a market order
    order = Order(
        symbol="EURUSD",
        order_type=OrderType.MARKET,
        side=OrderSide.BUY,
        quantity=1000,
        time_in_force=TimeInForce.DAY
    )
    
    order_id = await broker.place_order(order)
    print(f"Order placed: {order_id}")
    
    # Get order status
    status = await broker.get_order_status(order_id)
    print(f"Order status: {status}")
    
    # Get account info
    account_info = await broker.get_account_info()
    print(f"Account balance: {account_info['balance']}")
    
    # Disconnect
    await broker.disconnect()

# Run the backtest
import asyncio
asyncio.run(run_backtest())
```

## Backtesting a Strategy

For a complete example of backtesting a trading strategy, see the example script:

```
python examples/backtest_broker_usage.py
```

This script demonstrates how to:
1. Load historical data
2. Implement a moving average crossover strategy
3. Execute trades using the backtest broker
4. Calculate performance metrics
5. Visualize the results

## Configuration Options

When creating a backtest broker, you can configure the following parameters:

- `data_provider`: The data provider to use for market data (required)
- `initial_balance`: Starting account balance (default: 10,000)
- `leverage`: Account leverage (default: 1.0)
- `commission`: Commission per trade as a percentage (default: 0.0005)
- `spread`: Bid-ask spread in price units (default: 0.0002)
- `slippage`: Slippage per trade in price units (default: 0.0001)

## Testing

Unit and integration tests are available in the `tests/` directory. To run the tests:

```bash
pytest tests/unit/execution/test_mt5_backtest_broker.py
pytest tests/integration/execution/test_execution_flow.py
```

## Notes

- The backtest broker simulates order execution but does not account for all market conditions
- Results may vary from live trading due to factors like liquidity and execution speed
- Always validate strategy performance with forward testing before live trading
