import MetaTrader5 as mt5

def test_mt5_connection():
    """
    Test the connection to MetaTrader 5 and display account information.
    
    Returns:
        bool: True if connection and data retrieval was successful, False otherwise
    """
    print("Initializing MT5 connection...")
    
    # Initialize MT5 connection
    if not mt5.initialize():
        print("initialize() failed, error code =", mt5.last_error())
        return False
    
    print("MT5 initialized successfully!")
    
    try:
        # Get account info
        print("\nFetching account information...")
        account_info = mt5.account_info()
        if account_info is None:
            print("Failed to get account info, error code =", mt5.last_error())
            return False
            
        print("\n=== Account Information ===")
        print(f"Login: {account_info.login}")
        print(f"Name: {account_info.name}")
        print(f"Server: {account_info.server}")
        print(f"Leverage: 1:{account_info.leverage}")
        print(f"Balance: {account_info.balance:.2f} {account_info.currency}")
        print(f"Equity: {account_info.equity:.2f} {account_info.currency}")
        print(f"Margin: {account_info.margin:.2f} {account_info.currency}")
        print(f"Free Margin: {account_info.margin_free:.2f} {account_info.currency}")
        
        # Get terminal info
        terminal_info = mt5.terminal_info()
        print("\n=== Terminal Information ===")
        print(f"Name: {terminal_info.name}")
        print(f"Community Account: {getattr(terminal_info, 'community_account', 'N/A')}")
        print(f"Community Connection: {getattr(terminal_info, 'community_connection', 'N/A')}")
        print(f"Connected: {'Yes' if terminal_info.connected else 'No'}")
        print(f"DLLs Allowed: {'Yes' if terminal_info.dlls_allowed else 'No'}")
        print(f"Trade Allowed: {'Yes' if terminal_info.trade_allowed else 'No'}")
        print(f"Trade Mode: {'Demo' if hasattr(terminal_info, 'trade_mode') and terminal_info.trade_mode == 0 else 'Real'}")
        print(f"Path: {terminal_info.path}")
        print(f"Data Path: {terminal_info.data_path}")
        print(f"Common Data Path: {terminal_info.common_data_path}")
        
        # Get symbols
        print("\n=== Available Symbols (First 10) ===")
        symbols = mt5.symbols_get()
        if symbols is not None:
            for i, symbol in enumerate(symbols[:10]):  # Show first 10 symbols
                print(f"{i+1}. {symbol.name}")
        else:
            print("No symbols found or error:", mt5.last_error())
        
        return True
        
    except Exception as e:
        print(f"\nError during MT5 connection test: {str(e)}")
        return False
        
    finally:
        # Shutdown MT5 connection
        print("\nShutting down MT5 connection...")
        mt5.shutdown()
        print("MT5 connection closed.")

if __name__ == "__main__":
    print("=== MT5 Connection Tester ===\n")
    success = test_mt5_connection()
    print("\n=== Test " + ("Completed Successfully" if success else "Failed") + " ===")
