import MetaTrader5 as mt5
import time
from config.settings import settings

def test_mt5_connection():
    """Test connection to MetaTrader 5 terminal."""
    print("Testing MT5 Connection...")
    print(f"Account: {settings.MT5_LOGIN}")
    print(f"Server: {settings.MT5_SERVER}")
    
    # Initialize MT5 connection
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        return False
    
    print("MT5 initialized successfully.")
    
    # Attempt to log in
    authorized = mt5.login(
        login=settings.MT5_LOGIN,
        password=settings.MT5_PASSWORD,
        server=settings.MT5_SERVER
    )
    
    if authorized:
        print("\nSuccessfully connected to account #{} on {}".format(
            settings.MT5_LOGIN, settings.MT5_SERVER))
        
        # Display account info
        account_info = mt5.account_info()
        if account_info is not None:
            print("\nAccount Info:")
            print(f"  Balance: {account_info.balance}")
            print(f"  Equity: {account_info.equity}")
            print(f"  Margin: {account_info.margin}")
            print(f"  Free Margin: {account_info.margin_free}")
            print(f"  Leverage: 1:{account_info.leverage}")
        
        # Get some symbols
        symbols = mt5.symbols_get()
        print(f"\nAvailable symbols ({min(5, len(symbols))} of {len(symbols)}):")
        for i in range(min(5, len(symbols))):
            print(f"  {symbols[i].name}")
        
        return True
    else:
        print("\nFailed to connect to account #{}, error code: {}".format(
            settings.MT5_LOGIN, mt5.last_error()))
        print("\nTroubleshooting steps:")
        print("1. Make sure MetaTrader 5 terminal is running")
        print("2. Verify your login credentials in the .env file")
        print("3. Check your internet connection")
        print("4. Ensure the server name is correct")
        print("5. Make sure your account is properly set up with the broker")
        return False

if __name__ == "__main__":
    print("Starting MT5 Connection Test...")
    print("-" * 50)
    
    try:
        if test_mt5_connection():
            print("\n" + "=" * 50)
            print("MT5 Connection Test: SUCCESS")
            print("=" * 50)
        else:
            print("\n" + "!" * 50)
            print("MT5 Connection Test: FAILED")
            print("!" * 50)
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
    finally:
        # Shutdown MT5 connection
        mt5.shutdown()
        print("\nMT5 connection closed.")
