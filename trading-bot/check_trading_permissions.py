import MetaTrader5 as mt5
from datetime import datetime

def check_trading_permissions():
    print("=== MT5 Trading Permissions Check ===\n")
    
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
            
        print("\n=== Account Permissions ===")
        print(f"Login: {account.login}")
        print(f"Trade Mode: {'Demo' if account.trade_mode == 0 else 'Real'}")
        print(f"Leverage: 1:{account.leverage}")
        print(f"Margin Mode: {account.margin_mode}")
        print(f"Trade Allowed: {bool(account.trade_allowed)}")
        print(f"Trade Expert: {bool(account.trade_expert)}")
        
        # Get terminal info
        terminal = mt5.terminal_info()
        print("\n=== Terminal Status ===")
        print(f"Connected: {terminal.connected}")
        print(f"DLLs Allowed: {terminal.dlls_allowed}")
        print(f"Trade Allowed: {terminal.trade_allowed}")
        print(f"AutoTrading: {getattr(terminal, 'auto_trading_allowed', 'Not available')}")
        print(f"Trade Mode: {'Demo' if terminal.community_connection else 'Real'}")
        
        # Check symbol permissions
        symbol = "EURUSD"
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"\n{symbol} not found")
            return False
            
        print(f"\n=== {symbol} Trading Permissions ===")
        print(f"Trade Mode: {symbol_info.trade_mode}")
        print(f"Order Modes: {symbol_info.order_mode}")
        print(f"Order Execution: {symbol_info.order_execution}")
        print(f"Filling Mode: {symbol_info.filling_mode}")
        
        # Check if we can place orders
        print("\n=== Order Placement Test ===")
        
        # Try to get order types
        order_types = mt5.orders_get()
        if order_types is not None:
            print(f"Order types available: {len(order_types)}")
        else:
            print("Failed to get order types")
        
        # Try to get open positions
        positions = mt5.positions_get()
        if positions is not None:
            print(f"Open positions: {len(positions)}")
        else:
            print("No open positions or failed to get positions")
        
        # Check server status
        print("\n=== Server Status ===")
        server_status = mt5.terminal_info().community_connection
        print(f"Connected to server: {bool(server_status)}")
        print(f"Server time: {mt5.terminal_info().time}")
        print(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return True
        
    except Exception as e:
        print(f"Error during permission check: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        mt5.shutdown()
        print("\nMT5 connection closed")

if __name__ == "__main__":
    success = check_trading_permissions()
    print("\n=== Check " + ("Completed" if success else "Failed") + " ===")
