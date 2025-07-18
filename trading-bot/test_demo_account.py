import MetaTrader5 as mt5
import getpass

def test_demo_account():
    print("=== MT5 Demo Account Tester ===\n")
    
    # Get account details
    print("Please enter your demo account details:")
    login = input("Login (account number): ").strip()
    password = getpass.getpass("Password: ").strip()
    server = input("Server (e.g., 'MetaQuotes-Demo' or 'ICMarkets-Demo'): ").strip()
    
    # Initialize MT5 with the provided credentials
    print("\nInitializing MT5 connection...")
    if not mt5.initialize():
        print("MT5 initialization failed")
        print("Error:", mt5.last_error())
        return False
    
    try:
        # Attempt to login
        print(f"\nConnecting to {server} with login {login}...")
        authorized = mt5.login(login=int(login), password=password, server=server)
        
        if not authorized:
            print("Failed to connect to account")
            print("Error:", mt5.last_error())
            return False
            
        # Get account info
        account = mt5.account_info()
        terminal = mt5.terminal_info()
        
        print("\n=== Connection Successful ===")
        print(f"Account: {account.login}")
        print(f"Name: {account.name}")
        print(f"Server: {account.server}")
        print(f"Balance: {account.balance} {account.currency}")
        print(f"Equity: {account.equity} {account.currency}")
        print(f"Leverage: 1:{account.leverage}")
        print(f"Trading Allowed: {bool(terminal.trade_allowed)}")
        print(f"AutoTrading: {getattr(terminal, 'auto_trading_allowed', 'Not available')}")
        
        # Test market data
        print("\nTesting market data...")
        symbols = ["EURUSD", "GBPUSD", "XAUUSD"]
        for symbol in symbols:
            mt5.symbol_select(symbol, True)
            tick = mt5.symbol_info_tick(symbol)
            if tick is not None:
                print(f"{symbol}: Bid={tick.bid}, Ask={tick.ask}")
            else:
                print(f"{symbol}: No data")
        
        return True
        
    except Exception as e:
        print(f"Error during test: {str(e)}")
        return False
        
    finally:
        mt5.shutdown()
        print("\nMT5 connection closed")

if __name__ == "__main__":
    print("MT5 Demo Account Tester")
    print("This script will test connection to an MT5 demo account.\n")
    
    while True:
        success = test_demo_account()
        print("\n=== Test", "Succeeded" if success else "Failed", "===")
        
        again = input("\nTest another account? (y/n): ").strip().lower()
        if again != 'y':
            break
