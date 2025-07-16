import threading
import pandas as pd
import time
from streaming.consumer import consume_ticks
from streaming.producer import publish_signal
from decision_engine import DecisionEngine
from config.settings import settings
from utils.risk_management import calculate_position_size, validate_trade
from mt5_connector import connect, shutdown
from database.session import SessionLocal
from database.models import Trade
from utils.logger import logger
import argparse
import MetaTrader5 as mt5

# Parse trading mode
parser = argparse.ArgumentParser(description="Run AI Trader Bot")
parser.add_argument("--mode", choices=["paper","live"], default="paper", 
                    help="Trading mode: paper or live (deprecated, use --live instead)")
parser.add_argument("--live", action="store_true", 
                    help="Run in live trading mode (overrides --mode if specified)")
args = parser.parse_args()

# Set trading mode
MODE = "live" if args.live else args.mode

# Initialize components
engine = DecisionEngine()
mt5_connected = False

# Database session
db = SessionLocal()

# Example pip value; in real usage derive from symbol
PIP_VALUE = 1.0

# Account state placeholder (to fetch dynamically in real implementation)
ACCOUNT_BALANCE = 10000.0


def process_tick(tick: dict):
    global mt5_connected
    # Convert tick to feature frame (stub)
    features = pd.DataFrame([tick])
    # Decision logic
    decisions = engine.decide(features, news_text=tick.get('news'))
    for dec in decisions:
        symbol = dec.get('symbol')
        signal = dec.get('signal')
        confidence = dec.get('confidence')
        # Position sizing
        size = calculate_position_size(ACCOUNT_BALANCE, settings.RISK_PERCENT_PER_TRADE, tick.get('stop_loss_pips', 10), PIP_VALUE)
        if not validate_trade(size):
            continue
        # Ensure MT5 connection
        if not mt5_connected:
            mt5_connected = connect()
        # Execute live order or simulate paper trade
        if MODE == "live":
            order_type = mt5.ORDER_TYPE_BUY if signal > 0 else mt5.ORDER_TYPE_SELL
            sl = tick.get('price') - (tick.get('stop_loss_pips', 10) * 0.0001) if signal > 0 else tick.get('price') + (tick.get('stop_loss_pips', 10) * 0.0001)
            tp = tick.get('price') + (tick.get('take_profit_pips', 10) * 0.0001) if signal > 0 else tick.get('price') - (tick.get('take_profit_pips', 10) * 0.0001)
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": size,
                "type": order_type,
                "price": tick.get('price'),
                "sl": sl,
                "tp": tp,
                "deviation": 10,
                "magic": getattr(settings, 'MT5_MAGIC_NUMBER', 0),
                "comment": f"AI Trader {MODE}"        
            }
            result = mt5.order_send(request)
            if hasattr(result, 'retcode') and result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"MT5 order_send failed: {result.comment}")
            else:
                logger.info(f"MT5 order sent successfully, order ID: {getattr(result, 'order', None)}")
        else:
            logger.info(f"Paper trade simulated: symbol={symbol}, signal={signal}, size={size}")

        # Publish signal
        publish_signal({'symbol': symbol, 'signal': signal, 'size': size, 'confidence': confidence})
        # Persist trade
        trade = Trade(symbol=symbol, order_type=('BUY' if signal>0 else 'SELL'), size=size,
                      entry_price=tick.get('price'), stop_loss=tick.get('price') - (tick.get('stop_loss_pips',10)*0.0001),
                      take_profit=tick.get('price') + (tick.get('take_profit_pips',10)*0.0001), confidence=confidence)
        db.add(trade)
        db.commit()
        logger.info(f"Executed and recorded trade: {trade.id}")


def main():
    logger.info("Starting AI Trader Bot")
    # Start consumer in separate thread
    consumer_thread = threading.Thread(target=consume_ticks, args=(process_tick,), daemon=True)
    consumer_thread.start()
    try:
        while True:
            time.sleep(1)  # Keep main alive
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        shutdown()


if __name__ == '__main__':
    main()
