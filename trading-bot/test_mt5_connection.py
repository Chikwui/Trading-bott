"""
Test script to verify MT5 connection and basic functionality.

This script requires the following environment variables to be set:
- MT5_LOGIN: Your MT5 account login
- MT5_PASSWORD: Your MT5 account password
- MT5_SERVER: Your MT5 server name
- MT5_PATH: (Optional) Path to MT5 terminal executable
"""
import os
import sys
from datetime import datetime, timedelta
from dotenv import load_dotenv
import MetaTrader5 as mt5
import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Get MT5 credentials from environment variables
MT5_LOGIN = os.getenv('MT5_LOGIN')
MT5_PASSWORD = os.getenv('MT5_PASSWORD')
MT5_SERVER = os.getenv('MT5_SERVER')
MT5_PATH = os.getenv('MT5_PATH')

def test_mt5_connection():
    """Test connection to MT5 terminal and basic functionality."""
    print("Testing MT5 connection...")
    
    # Check if required environment variables are set
    if not all([MT5_LOGIN, MT5_PASSWORD, MT5_SERVER]):
        print("Error: Missing required environment variables")
        print("Please set MT5_LOGIN, MT5_PASSWORD, and MT5_SERVER in your .env file")
        return False
    
    # Initialize MT5 connection
    print(f"Connecting to {MT5_SERVER} as {MT5_LOGIN}...")
    
    # Try to initialize with path if provided
    if MT5_PATH and not mt5.initialize(path=MT5_PATH):
        print(f"MT5 initialize() with path failed: {mt5.last_error()}")
        print("Trying without path...")
        if not mt5.initialize():
            print(f"MT5 initialize() failed: {mt5.last_error()}")
            return False
    elif not mt5.initialize():
        print(f"MT5 initialize() failed: {mt5.last_error()}")
        return False
    
    # Try to log in
    authorized = mt5.login(
        login=int(MT5_LOGIN),
        password=MT5_PASSWORD,
        server=MT5_SERVER
    )
    
    if not authorized:
        print(f"MT5 login failed: {mt5.last_error()}")
        mt5.shutdown()
        return False
    
    try:
        # Test connection status
        account_info = mt5.account_info()
        if account_info is None:
            print("Failed to get account info")
            return False
            
        print("\n=== Connection Successful ===")
        print(f"MT5 version: {mt5.version()}")
        print(f"Terminal Info: {mt5.terminal_info()}")
        print(f"Account Info:")
        print(f"  Login: {account_info.login}")
        print(f"  Name: {account_info.name}")
        print(f"  Balance: {account_info.balance} {account_info.currency}")
        print(f"  Equity: {account_info.equity} {account_info.currency}")
        print(f"  Leverage: 1:{account_info.leverage}")
        
        # Test symbol info
        symbol = "EURUSD"
        print(f"\n=== Symbol Info ===")
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"Failed to get info for {symbol}")
            return False
            
        print(f"Symbol: {symbol}")
        print(f"  Point: {symbol_info.point}")
        print(f"  Digits: {symbol_info.digits}")
        print(f"  Spread: {symbol_info.spread}")
        print(f"  Trade Mode: {symbol_info.trade_mode}")
        print(f"  Trade Stops Level: {symbol_info.trade_stops_level}")
        print(f"  Trade Tick Size: {symbol_info.trade_tick_size}")
        print(f"  Trade Tick Value: {symbol_info.trade_tick_value}")
        
        # Test getting historical data
        print("\n=== Historical Data ===")
        time_from = datetime.now() - timedelta(days=7)
        rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_D1, time_from, 5)
        
        if rates is None:
            print(f"Failed to get historical rates: {mt5.last_error()}")
            return False
            
        print("Historical Data (last 5 days):")
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        print(df[['time', 'open', 'high', 'low', 'close', 'tick_volume']])
        
        # Test market data subscription
        print("\n=== Market Data Subscription ===")
        if not mt5.market_book_add(symbol):
            print(f"Failed to subscribe to market book: {mt5.last_error()}")
        else:
            print(f"Subscribed to {symbol} market book")
            # Get the current order book
            book = mt5.market_book_get(symbol)
            if book is not None and len(book) > 0:
                print(f"Order book depth: {len(book)} levels")
                print(f"Best bid: {book[0].price} x {book[0].volume}")
                print(f"Best ask: {book[-1].price} x {book[-1].volume}")
            mt5.market_book_release(symbol)
            
        # Test symbol selection
        print("\n=== Symbol Selection ===")
        selected = mt5.symbol_select(symbol, True)
        if selected:
            print(f"Successfully selected {symbol}")
            mt5.symbol_select(symbol, False)  # Deselect after test
            
        return True
        
    except Exception as e:
        print(f"Error during MT5 test: {str(e)}")
        return False
    finally:
        # Shutdown MT5 connection
        mt5.shutdown()

if __name__ == "__main__":
    print("MT5 Connection Tester")
    print("====================")
    success = test_mt5_connection()
    print("\nMT5 Connection Test:", "✅ SUCCESS" if success else "❌ FAILED")
    print("====================")
