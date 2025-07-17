"""
Example Usage of MT5 Backtest Broker

This script demonstrates how to use the MT5BacktestBroker to backtest a simple moving average
crossover strategy. It shows how to:
1. Set up the backtest environment
2. Load historical data
3. Implement a trading strategy
4. Execute trades using the broker
5. Analyze and visualize the results

Requirements:
- pandas
- matplotlib
- numpy
- core package (from the trading-bot)
"""
import asyncio
import os
import sys
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# Add the project root to the Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from core.data.providers.mt5_provider import MT5DataProvider
from core.execution.backtest.mt5_backtest_broker import create_mt5_backtest_broker
from core.models import Order, OrderType, OrderSide, TimeInForce

# Configuration
SYMBOL = "EURUSD"
TIMEFRAME = "1h"  # 1 hour candles
START_DATE = datetime.utcnow() - timedelta(days=30)  # Last 30 days
END_DATE = datetime.utcnow()
INITIAL_BALANCE = 10000.0  # Initial account balance in USD
LEVERAGE = 1.0  # No leverage
COMMISSION = 0.0005  # 0.05% commission per trade
SPREAD = 0.0002  # 2 pips spread
SLIPPAGE = 0.0001  # 1 pip slippage

# Strategy parameters
FAST_MA_PERIOD = 10  # Fast moving average period
SLOW_MA_PERIOD = 30  # Slow moving average period
RISK_PER_TRADE = 0.02  # 2% risk per trade
STOP_LOSS_ATR_MULTIPLIER = 2.0  # ATR multiplier for stop loss
TAKE_PROFIT_ATR_MULTIPLIER = 3.0  # ATR multiplier for take profit
ATR_PERIOD = 14  # ATR period for volatility calculation

class MovingAverageCrossoverStrategy:
    """Simple Moving Average Crossover Strategy.
    
    This strategy generates buy signals when the fast MA crosses above the slow MA
    and sell signals when the fast MA crosses below the slow MA.
    """
    
    def __init__(self, fast_period: int, slow_period: int):
        """Initialize the strategy with MA periods."""
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.name = f"MA_Crossover_{fast_period}_{slow_period}"
        self.position = 0  # Current position (0 = flat, 1 = long, -1 = short)
        
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate trading signals based on MA crossovers.
        
        Args:
            data: DataFrame with OHLCV data and indicators
            
        Returns:
            DataFrame with signals (1 = buy, -1 = sell, 0 = hold)
        """
        # Calculate moving averages
        data['fast_ma'] = data['close'].rolling(window=self.fast_period).mean()
        data['slow_ma'] = data['close'].rolling(window=self.slow_period).mean()
        
        # Calculate ATR for stop loss and take profit
        high_low = data['high'] - data['low']
        high_close = (data['high'] - data['close'].shift()).abs()
        low_close = (data['low'] - data['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        data['atr'] = true_range.rolling(window=ATR_PERIOD).mean()
        
        # Generate signals
        data['signal'] = 0
        data.loc[data['fast_ma'] > data['slow_ma'], 'signal'] = 1  # Buy signal
        data.loc[data['fast_ma'] < data['slow_ma'], 'signal'] = -1  # Sell signal
        
        # Remove signals where we're already in a position
        data['position'] = data['signal'].diff()
        data.loc[data['position'] == 0, 'signal'] = 0
        
        # Clean up
        data.dropna(inplace=True)
        
        return data
    
    def get_trade_parameters(self, data: pd.DataFrame, current_index: int) -> dict:
        """Calculate trade parameters based on the current market conditions.
        
        Args:
            data: DataFrame with OHLCV data and indicators
            current_index: Current index in the data
            
        Returns:
            Dictionary with trade parameters
        """
        if current_index < max(self.fast_period, self.slow_period, ATR_PERIOD):
            return None
            
        current = data.iloc[current_index]
        prev = data.iloc[current_index - 1]
        
        # Only enter a trade if we have a signal and we're not already in a position
        if current['signal'] == 1 and self.position <= 0:  # Buy signal
            self.position = 1
            entry_price = current['close']
            atr = current['atr']
            
            # Calculate stop loss and take profit based on ATR
            stop_loss = entry_price - (atr * STOP_LOSS_ATR_MULTIPLIER)
            take_profit = entry_price + (atr * TAKE_PROFIT_ATR_MULTIPLIER)
            
            return {
                'side': OrderSide.BUY,
                'price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'atr': atr
            }
            
        elif current['signal'] == -1 and self.position >= 0:  # Sell signal
            self.position = -1
            entry_price = current['close']
            atr = current['atr']
            
            # Calculate stop loss and take profit based on ATR
            stop_loss = entry_price + (atr * STOP_LOSS_ATR_MULTIPLIER)
            take_profit = entry_price - (atr * TAKE_PROFIT_ATR_MULTIPLIER)
            
            return {
                'side': OrderSide.SELL,
                'price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'atr': atr
            }
            
        return None

async def run_backtest():
    """Run the backtest with the MA Crossover strategy."""
    print("Starting backtest...")
    
    # Initialize data provider
    print("Initializing data provider...")
    data_provider = MT5DataProvider()
    await data_provider.initialize()
    
    # Load historical data
    print(f"Loading historical data for {SYMBOL} ({TIMEFRAME}) from {START_DATE} to {END_DATE}...")
    historical_data = await data_provider.get_historical_data(
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        start=START_DATE,
        end=END_DATE,
        limit=1000  # Adjust based on your needs
    )
    
    if not historical_data:
        print("No historical data available. Exiting...")
        return
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(historical_data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Initialize strategy
    print("Initializing strategy...")
    strategy = MovingAverageCrossoverStrategy(
        fast_period=FAST_MA_PERIOD,
        slow_period=SLOW_MA_PERIOD
    )
    
    # Calculate indicators and signals
    print("Calculating indicators and signals...")
    df = strategy.calculate_signals(df)
    
    # Initialize backtest broker
    print("Initializing backtest broker...")
    broker = create_mt5_backtest_broker(
        data_provider=data_provider,
        initial_balance=INITIAL_BALANCE,
        leverage=LEVERAGE,
        commission=COMMISSION,
        spread=SPREAD,
        slippage=SLIPPAGE
    )
    
    # Connect to the broker
    await broker.connect()
    
    # Run the backtest
    print("Running backtest...")
    trades = []
    
    for i in range(len(df)):
        current_time = df.index[i]
        current_price = df['close'].iloc[i]
        
        # Update market data in the broker
        await broker.update_market_data(current_time)
        
        # Get account info
        account_info = await broker.get_account_info()
        
        # Get current positions
        positions = await broker.get_positions(SYMBOL)
        current_position = positions[0]['quantity'] if positions else 0
        
        # Get trade parameters from strategy
        trade_params = strategy.get_trade_parameters(df, i)
        
        if trade_params:
            # Calculate position size based on risk
            atr = trade_params['atr']
            risk_amount = account_info['equity'] * RISK_PER_TRADE
            
            # Calculate position size based on ATR
            if trade_params['side'] == OrderSide.BUY:
                stop_distance = abs(trade_params['price'] - trade_params['stop_loss'])
            else:  # SELL
                stop_distance = abs(trade_params['price'] - trade_params['stop_loss'])
            
            # Avoid division by zero
            if stop_distance > 0:
                position_size = int((risk_amount / stop_distance) * 100000)  # For forex, 1 lot = 100,000 units
                position_size = max(1000, position_size)  # Minimum 0.01 lot
            else:
                position_size = 1000  # Default to 0.01 lot if stop distance is too small
            
            # Close any existing position in the opposite direction
            if (trade_params['side'] == OrderSide.BUY and current_position < 0) or \
               (trade_params['side'] == OrderSide.SELL and current_position > 0):
                # Close existing position
                close_order = Order(
                    symbol=SYMBOL,
                    order_type=OrderType.MARKET,
                    side=OrderSide.SELL if current_position > 0 else OrderSide.BUY,
                    quantity=abs(current_position),
                    time_in_force=TimeInForce.DAY
                )
                
                try:
                    order_id = await broker.place_order(close_order)
                    print(f"Closed position: {order_id}")
                except Exception as e:
                    print(f"Error closing position: {e}")
            
            # Open new position
            order = Order(
                symbol=SYMBOL,
                order_type=OrderType.MARKET,
                side=trade_params['side'],
                quantity=position_size,
                time_in_force=TimeInForce.GTC,
                stop_loss=trade_params['stop_loss'],
                take_profit=trade_params['take_profit']
            )
            
            try:
                order_id = await broker.place_order(order)
                order_status = await broker.get_order_status(order_id)
                
                # Record the trade
                trades.append({
                    'timestamp': current_time,
                    'type': 'LONG' if trade_params['side'] == OrderSide.BUY else 'SHORT',
                    'price': trade_params['price'],
                    'quantity': position_size,
                    'stop_loss': trade_params['stop_loss'],
                    'take_profit': trade_params['take_profit'],
                    'order_id': order_id,
                    'status': order_status['status']
                })
                
                print(f"Placed {trade_params['side']} order: {order_id} at {trade_params['price']:.5f}")
                
            except Exception as e:
                print(f"Error placing order: {e}")
    
    # Close any open positions at the end
    positions = await broker.get_positions(SYMBOL)
    if positions:
        for position in positions:
            close_order = Order(
                symbol=position['symbol'],
                order_type=OrderType.MARKET,
                side=OrderSide.SELL if position['quantity'] > 0 else OrderSide.BUY,
                quantity=abs(position['quantity']),
                time_in_force=TimeInForce.DAY
            )
            
            try:
                order_id = await broker.place_order(close_order)
                print(f"Closed final position: {order_id}")
            except Exception as e:
                print(f"Error closing final position: {e}")
    
    # Get final account info
    final_account_info = await broker.get_account_info()
    
    # Disconnect from the broker
    await broker.disconnect()
    
    # Analyze results
    print("\n=== Backtest Results ===")
    print(f"Initial Balance: ${INITIAL_BALANCE:,.2f}")
    print(f"Final Balance: ${final_account_info['balance']:,.2f}")
    print(f"Profit/Loss: ${final_account_info['balance'] - INITIAL_BALANCE:,.2f} ({(final_account_info['balance'] / INITIAL_BALANCE - 1) * 100:.2f}%)")
    print(f"Total Trades: {len(trades)}")
    
    if trades:
        winning_trades = [t for t in trades if (t['type'] == 'LONG' and t['price'] < t['take_profit']) or 
                                                (t['type'] == 'SHORT' and t['price'] > t['take_profit'])]
        losing_trades = [t for t in trades if t not in winning_trades]
        
        win_rate = (len(winning_trades) / len(trades)) * 100 if trades else 0
        avg_win = np.mean([(t['take_profit'] / t['price'] - 1) * 100 for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([(t['stop_loss'] / t['price'] - 1) * 100 for t in losing_trades]) * -1 if losing_trades else 0
        
        print(f"Winning Trades: {len(winning_trades)} ({win_rate:.1f}%)")
        print(f"Losing Trades: {len(losing_trades)} ({(100 - win_rate):.1f}%)")
        print(f"Average Win: {avg_win:.2f}%")
        print(f"Average Loss: {avg_loss:.2f}%")
        print(f"Risk/Reward Ratio: {abs(avg_win / avg_loss):.2f} (if avg_win/avg_loss > 1, strategy is profitable)")
    
    # Plot results
    plot_results(df, trades)

def plot_results(df: pd.DataFrame, trades: list):
    """Plot the backtest results.
    
    Args:
        df: DataFrame with price data and indicators
        trades: List of trade dictionaries
    """
    plt.figure(figsize=(15, 10))
    
    # Plot price and moving averages
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(df.index, df['close'], label='Close Price', color='black', alpha=0.7)
    ax1.plot(df.index, df['fast_ma'], label=f'Fast MA ({FAST_MA_PERIOD})', color='blue', alpha=0.5)
    ax1.plot(df.index, df['slow_ma'], label=f'Slow MA ({SLOW_MA_PERIOD})', color='red', alpha=0.5)
    
    # Plot trades
    if trades:
        buy_times = [t['timestamp'] for t in trades if t['type'] == 'LONG']
        buy_prices = [t['price'] for t in trades if t['type'] == 'LONG']
        sell_times = [t['timestamp'] for t in trades if t['type'] == 'SHORT']
        sell_prices = [t['price'] for t in trades if t['type'] == 'SHORT']
        
        ax1.scatter(buy_times, buy_prices, color='green', marker='^', s=100, label='Buy Signal')
        ax1.scatter(sell_times, sell_prices, color='red', marker='v', s=100, label='Sell Signal')
    
    ax1.set_title(f'{SYMBOL} Price and Moving Averages')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    # Format x-axis dates
    date_format = DateFormatter('%Y-%m-%d')
    ax1.xaxis.set_major_formatter(date_format)
    plt.xticks(rotation=45)
    
    # Plot ATR
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.plot(df.index, df['atr'], label='ATR (14)', color='purple')
    ax2.set_title('Average True Range (ATR)')
    ax2.set_ylabel('ATR')
    ax2.legend()
    ax2.grid(True)
    
    # Format x-axis dates
    ax2.xaxis.set_major_formatter(date_format)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    asyncio.run(run_backtest())
