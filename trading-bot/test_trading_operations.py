import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import time

class MT5TradingOperations:
    def __init__(self, symbol="EURUSD", lot_size=0.1, magic_number=123456):
        """
        Initialize trading operations with default parameters
        :param symbol: Trading symbol (default: EURUSD)
        :param lot_size: Trade volume in lots (default: 0.1)
        :param magic_number: Expert Advisor ID (default: 123456)
        """
        self.symbol = symbol
        self.lot_size = lot_size
        self.magic_number = magic_number
        self.slippage = 3  # Allowed slippage in points
        self.deviation = 20  # Max price deviation in points
        
    def initialize_mt5(self):
        """Initialize MT5 connection"""
        if not mt5.initialize():
            print("MT5 initialization failed")
            mt5.shutdown()
            return False
        return True
    
    def get_symbol_info(self):
        """Get detailed information about the trading symbol"""
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            print(f"{self.symbol} not found, can't get symbol info")
            return None
        return symbol_info
    
    def get_current_price(self):
        """Get current bid and ask prices"""
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            print("Failed to get tick")
            return None, None
        return tick.bid, tick.ask
    
    def calculate_position_size(self, risk_percent=1.0, stop_loss_pips=20):
        """
        Calculate position size based on account balance and risk percentage
        :param risk_percent: Percentage of account to risk (default: 1%)
        :param stop_loss_pips: Stop loss in pips (default: 20)
        """
        account_info = mt5.account_info()
        if account_info is None:
            print("Failed to get account info")
            return self.lot_size
            
        symbol_info = self.get_symbol_info()
        if symbol_info is None:
            return self.lot_size
            
        # Calculate position size
        risk_amount = account_info.balance * (risk_percent / 100)
        pip_value = symbol_info.trade_tick_value / symbol_info.trade_tick_size
        pip_size = 0.0001 if 'JPY' not in self.symbol else 0.01
        position_size = (risk_amount / (stop_loss_pips * pip_value * pip_size)) / 10  # Convert to lots
        
        # Round to 2 decimal places and ensure minimum lot size
        position_size = max(round(position_size, 2), 0.01)
        return min(position_size, symbol_info.volume_max)
    
    def place_market_order(self, order_type, lot_size=None, stop_loss=0, take_profit=0, comment="Python Script"):
        """
        Place a market order
        :param order_type: Order type (buy/sell)
        :param lot_size: Trade volume in lots
        :param stop_loss: Stop loss in points (0 = no stop loss)
        :param take_profit: Take profit in points (0 = no take profit)
        :param comment: Order comment
        """
        if lot_size is None:
            lot_size = self.lot_size
            
        # Prepare the trade request
        symbol_info = self.get_symbol_info()
        if symbol_info is None:
            return None
            
        point = mt5.symbol_info(self.symbol).point
        price = mt5.symbol_info_tick(self.symbol).ask if order_type == "buy" else mt5.symbol_info_tick(self.symbol).bid
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot_size,
            "type": mt5.ORDER_TYPE_BUY if order_type == "buy" else mt5.ORDER_TYPE_SELL,
            "price": price,
            "sl": price - stop_loss * point if order_type == "buy" and stop_loss > 0 else 0,
            "tp": price + take_profit * point if order_type == "buy" and take_profit > 0 else 0,
            "deviation": self.deviation,
            "magic": self.magic_number,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,  # Using FILL OR KILL for better compatibility
        }
        
        # For sell orders, adjust SL/TP
        if order_type == "sell" and stop_loss > 0:
            request["sl"] = price + stop_loss * point
        if order_type == "sell" and take_profit > 0:
            request["tp"] = price - take_profit * point
            
        # Send the trading request
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Order failed, retcode={result.retcode}, comment={result.comment}")
            return None
            
        print(f"Order placed successfully, ticket={result.order}")
        return result
    
    def close_position(self, ticket, lot_size=None):
        """Close an open position by ticket"""
        position = mt5.positions_get(ticket=ticket)
        if position is None or len(position) == 0:
            print(f"Position {ticket} not found")
            return False
            
        position = position[0]
        symbol = position.symbol
        order_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        
        price = mt5.symbol_info_tick(symbol).bid if order_type == mt5.ORDER_TYPE_SELL else mt5.symbol_info_tick(symbol).ask
        
        if lot_size is None:
            lot_size = position.volume
            
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "deviation": self.deviation,
            "magic": self.magic_number,
            "comment": "Close position",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_FOK,  # Using FILL OR KILL for better compatibility
        }
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            print(f"Close position failed, retcode={result.retcode}, comment={result.comment}")
            return False
            
        print(f"Position {ticket} closed successfully")
        return True
    
    def get_open_positions(self, symbol=None):
        """Get all open positions"""
        if symbol is None:
            symbol = self.symbol
        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        if positions is None:
            return []
        return positions
    
    def get_account_info(self):
        """Get account information"""
        account_info = mt5.account_info()
        if account_info is None:
            print("Failed to get account info")
            return None
        return account_info
    
    def get_market_data(self, timeframe=mt5.TIMEFRAME_M1, n_bars=100):
        """Get historical market data"""
        rates = mt5.copy_rates_from_pos(self.symbol, timeframe, 0, n_bars)
        if rates is None:
            print("Failed to get market data")
            return None
        return pd.DataFrame(rates)
    
    def run_trading_test(self):
        """Run a complete trading test"""
        print("\n=== Starting Trading Test ===")
        
        # 1. Initialize MT5 connection
        if not self.initialize_mt5():
            return False
            
        try:
            # 2. Display account information
            account = self.get_account_info()
            print(f"\nAccount Info:")
            print(f"Login: {account.login}")
            print(f"Balance: {account.balance:.2f} {account.currency}")
            print(f"Equity: {account.equity:.2f} {account.currency}")
            print(f"Margin Free: {account.margin_free:.2f} {account.currency}")
            
            # 3. Display symbol information
            symbol_info = self.get_symbol_info()
            print(f"\nSymbol Info for {self.symbol}:")
            print(f"Point size: {symbol_info.point}")
            print(f"Digits: {symbol_info.digits}")
            print(f"Spread: {symbol_info.spread} points")
            print(f"Bid: {symbol_info.bid}, Ask: {symbol_info.ask}")
            
            # 4. Calculate position size with 1% risk
            stop_loss_pips = 20
            take_profit_pips = 40
            position_size = self.calculate_position_size(risk_percent=1.0, stop_loss_pips=stop_loss_pips)
            print(f"\nCalculated position size: {position_size} lots (1% risk, {stop_loss_pips} pip SL)")
            
            # 5. Place a buy order
            print("\nPlacing BUY order...")
            order = self.place_market_order(
                order_type="buy",
                lot_size=position_size,
                stop_loss=stop_loss_pips * 10,  # Convert pips to points
                take_profit=take_profit_pips * 10,
                comment="Test BUY order"
            )
            
            if order is None:
                print("Failed to place BUY order")
                return False
                
            # 6. Show open positions
            time.sleep(2)  # Wait for order to be processed
            positions = self.get_open_positions()
            print(f"\nOpen positions: {len(positions)}")
            for pos in positions:
                print(f"Position {pos.ticket}: {pos.symbol} {pos.volume} lots at {pos.price_open}")
            
            # 7. Close the position after a delay
            if len(positions) > 0:
                print("\nWaiting 5 seconds before closing position...")
                time.sleep(5)
                
                for pos in positions:
                    self.close_position(pos.ticket)
            
            # 8. Show final account status
            account = self.get_account_info()
            print(f"\nFinal Account Balance: {account.balance:.2f} {account.currency}")
            print(f"Final Equity: {account.equity:.2f} {account.currency}")
            
            return True
            
        except Exception as e:
            print(f"Error during trading test: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
            
        finally:
            # 9. Shutdown MT5 connection
            mt5.shutdown()
            print("\nMT5 connection closed")

if __name__ == "__main__":
    print("=== MT5 Advanced Trading Operations Test ===\n")
    
    # Initialize with default symbol and lot size
    trader = MT5TradingOperations(symbol="EURUSD", lot_size=0.1)
    
    # Run the trading test
    success = trader.run_trading_test()
    
    print("\n=== Test " + ("Completed Successfully" if success else "Failed") + " ===")
