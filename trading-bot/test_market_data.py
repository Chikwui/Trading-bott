import MetaTrader5 as mt5
import pandas as pd

def test_market_environment():
    print("=== MT5 Market Data Test ===\n")
    
    # Initialize MT5 connection
    if not mt5.initialize():
        print("MT5 initialization failed")
        mt5.shutdown()
        return False
    
    try:
        # Get account info
        account = mt5.account_info()
        if account is None:
            print("Failed to get account info")
            return False
            
        print("\n=== Account Information ===")
        print(f"Login: {account.login}")
        print(f"Name: {account.name}")
        print(f"Server: {account.server}")
        print(f"Leverage: 1:{account.leverage}")
        print(f"Balance: {account.balance:.2f} {account.currency}")
        print(f"Equity: {account.equity:.2f} {account.currency}")
        print(f"Margin: {account.margin:.2f} {account.currency}")
        print(f"Free Margin: {account.margin_free:.2f} {account.currency}")
        
        # Get terminal info
        terminal = mt5.terminal_info()
        print("\n=== Terminal Information ===")
        print(f"Name: {terminal.name}")
        print(f"Company: {terminal.company}")
        print(f"Path: {terminal.path}")
        print(f"Data Path: {terminal.data_path}")
        print(f"Trade Allowed: {bool(terminal.trade_allowed)}")
        # Check if auto_trading_allowed attribute exists
        auto_trading = getattr(terminal, 'auto_trading_allowed', 'Not available')
        print(f"AutoTrading Status: {auto_trading}")
        
        # Check trading permissions
        print("\n=== Trading Permissions ===")
        print(f"Trade Allowed: {bool(terminal.trade_allowed)}")
        print(f"Trading Mode: {'Demo' if terminal.community_connection else 'Real'}")
        
        if not terminal.trade_allowed:
            print("\nWARNING: Trading is not allowed. Please check:")
            print("1. Ensure 'AutoTrading' is enabled in MT5 (Ctrl+O -> Expert Advisors tab)")
            print("2. Check if your account has trading permissions")
            print("3. Verify that your account is properly connected to the broker's server")
        
        # Get symbol info
        symbol = "EURUSD"
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"\n{symbol} not found")
            return False
            
        print(f"\n=== {symbol} Information ===")
        print(f"Bid: {symbol_info.bid}, Ask: {symbol_info.ask}")
        print(f"Spread: {symbol_info.spread} points")
        print(f"Digits: {symbol_info.digits}")
        print(f"Point: {symbol_info.point}")
        print(f"Volume Min: {symbol_info.volume_min}, Volume Max: {symbol_info.volume_max}")
        
        # Get market data
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M1, 0, 10)
        if rates is not None:
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            print("\n=== Recent Market Data ===")
            print(df[['time', 'open', 'high', 'low', 'close', 'tick_volume']].to_string(index=False))
        

        
        return True
        
    except Exception as e:
        print(f"Error during test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        mt5.shutdown()
        print("\nMT5 connection closed")

if __name__ == "__main__":
    success = test_market_environment()
    print("\n=== Test " + ("Completed Successfully" if success else "Failed") + " ===")
