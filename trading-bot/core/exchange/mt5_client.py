"""
MT5 Client

This module provides integration with MetaTrader 5 for trading operations,
market data retrieval, and account management.
"""
import os
import time
import logging
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from enum import Enum
import MetaTrader5 as mt5
from dataclasses import dataclass, asdict

from core.utils.helpers import get_logger
from core.config import settings

logger = get_logger(__name__)

class TradeAction(Enum):
    """Trade action types."""
    BUY = mt5.TRADE_ACTION_DEAL
    SELL = mt5.TRADE_ACTION_DEAL
    PENDING_BUY = mt5.TRADE_ACTION_PENDING
    PENDING_SELL = mt5.TRADE_ACTION_PENDING
    MODIFY = mt5.TRADE_ACTION_MODIFY
    REMOVE = mt5.TRADE_ACTION_REMOVE
    CLOSE_BY = mt5.TRADE_ACTION_CLOSE_BY

class OrderType(Enum):
    """Order types."""
    MARKET = mt5.ORDER_TYPE_MARKET
    LIMIT = mt5.ORDER_TYPE_LIMIT
    STOP = mt5.ORDER_TYPE_STOP
    STOP_LIMIT = mt5.ORDER_TYPE_STOP_LIMIT
    BUY_STOP = mt5.ORDER_TYPE_BUY_STOP
    SELL_STOP = mt5.ORDER_TYPE_SELL_STOP
    BUY_LIMIT = mt5.ORDER_TYPE_BUY_LIMIT
    SELL_LIMIT = mt5.ORDER_TYPE_SELL_LIMIT
    BUY_STOP_LIMIT = mt5.ORDER_TYPE_BUY_STOP_LIMIT
    SELL_STOP_LIMIT = mt5.ORDER_TYPE_SELL_STOP_LIMIT
    CLOSE_BY = mt5.ORDER_TYPE_CLOSE_BY

@dataclass
class SymbolInfo:
    """Symbol information."""
    symbol: str
    bid: float
    ask: float
    last: float
    point: float
    digits: int
    spread: int
    volume_min: float
    volume_max: float
    volume_step: float
    trade_tick_size: float
    trade_tick_value: float
    margin_initial: float
    margin_maintenance: float
    swap_mode: int
    swap_long: float
    swap_short: float
    currency_base: str
    currency_profit: str
    currency_margin: str
    
    @classmethod
    def from_mt5(cls, mt5_symbol: mt5.SymbolInfo) -> 'SymbolInfo':
        """Create SymbolInfo from MT5 symbol info."""
        return cls(
            symbol=mt5_symbol.name,
            bid=mt5_symbol.bid,
            ask=mt5_symbol.ask,
            last=mt5_symbol.last,
            point=mt5_symbol.point,
            digits=mt5_symbol.digits,
            spread=mt5_symbol.spread,
            volume_min=mt5_symbol.volume_min,
            volume_max=mt5_symbol.volume_max,
            volume_step=mt5_symbol.volume_step,
            trade_tick_size=mt5_symbol.trade_tick_size,
            trade_tick_value=mt5_symbol.trade_tick_value,
            margin_initial=mt5_symbol.margin_initial,
            margin_maintenance=mt5_symbol.margin_maintenance,
            swap_mode=mt5_symbol.swap_mode,
            swap_long=mt5_symbol.swap_long,
            swap_short=mt5_symbol.swap_short,
            currency_base=mt5_symbol.currency_base,
            currency_profit=mt5_symbol.currency_profit,
            currency_margin=mt5_symbol.currency_margin
        )

@dataclass
class Order:
    """Order information."""
    ticket: int
    symbol: str
    order_type: OrderType
    volume: float
    price: float
    sl: float
    tp: float
    current_price: float
    profit: float
    comment: str
    magic: int
    time_setup: datetime
    time_expiration: datetime
    time_done: datetime
    state: str
    
    @classmethod
    def from_mt5(cls, mt5_order: dict) -> 'Order':
        """Create Order from MT5 order."""
        return cls(
            ticket=mt5_order['ticket'],
            symbol=mt5_order['symbol'],
            order_type=OrderType(mt5_order['type']),
            volume=mt5_order['volume_current'],
            price=mt5_order['price_open'],
            sl=mt5_order['sl'],
            tp=mt5_order['tp'],
            current_price=mt5_order['price_current'],
            profit=mt5_order['profit'],
            comment=mt5_order['comment'],
            magic=mt5_order['magic'],
            time_setup=mt5_order['time_setup'],
            time_expiration=mt5_order['time_expiration'],
            time_done=mt5_order['time_done'],
            state=mt5_order['state']
        )

class MT5Client:
    """MT5 client for trading operations."""
    
    def __init__(self, 
                 server: str = None,
                 login: int = None,
                 password: str = None,
                 timeout: int = 60000,
                 portable: bool = False):
        """Initialize MT5 client.
        
        Args:
            server: MT5 server name
            login: Account login
            password: Account password
            timeout: Connection timeout in milliseconds
            portable: Use portable mode
        """
        self.server = server or settings.MT5_SERVER
        self.login = login or settings.MT5_LOGIN
        self.password = password or settings.MT5_PASSWORD
        self.timeout = timeout
        self.portable = portable
        self.connected = False
        
    async def connect(self) -> bool:
        """Connect to MT5 terminal."""
        try:
            # Initialize MT5
            if not mt5.initialize(portable=self.portable):
                raise ConnectionError(f"MT5 initialize() failed: {mt5.last_error()}")
                
            # Login to account
            if not mt5.login(
                login=self.login,
                password=self.password,
                server=self.server,
                timeout=self.timeout
            ):
                raise ConnectionError(f"MT5 login failed: {mt5.last_error()}")
                
            self.connected = True
            logger.info(f"Connected to MT5 account {self.login} on {self.server}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MT5: {str(e)}")
            self.connected = False
            return False
    
    async def disconnect(self) -> None:
        """Disconnect from MT5 terminal."""
        if self.connected:
            mt5.shutdown()
            self.connected = False
            logger.info("Disconnected from MT5")
    
    async def get_symbol_info(self, symbol: str) -> Optional[SymbolInfo]:
        """Get symbol information."""
        if not self.connected:
            await self.connect()
            
        info = mt5.symbol_info(symbol)
        if info is None:
            logger.error(f"Symbol {symbol} not found")
            return None
            
        return SymbolInfo.from_mt5(info)
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        if not self.connected:
            await self.connect()
            
        account_info = mt5.account_info()._asdict()
        return {
            'balance': account_info['balance'],
            'equity': account_info['equity'],
            'margin': account_info['margin'],
            'free_margin': account_info['margin_free'],
            'margin_level': account_info['margin_level'],
            'leverage': account_info['leverage'],
            'currency': account_info['currency'],
            'name': account_info['name'],
            'server': account_info['server']
        }
    
    async def get_open_orders(self, symbol: str = None) -> List[Order]:
        """Get open orders."""
        if not self.connected:
            await self.connect()
            
        orders = mt5.orders_get(symbol=symbol) if symbol else mt5.orders_get()
        if orders is None:
            return []
            
        return [Order.from_mt5(order._asdict()) for order in orders]
    
    async def get_open_positions(self, symbol: str = None) -> List[Dict]:
        """Get open positions."""
        if not self.connected:
            await self.connect()
            
        positions = mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()
        if positions is None:
            return []
            
        return [pos._asdict() for pos in positions]
    
    async def place_order(
        self,
        symbol: str,
        order_type: OrderType,
        volume: float,
        price: float = None,
        sl: float = None,
        tp: float = None,
        deviation: int = 20,
        magic: int = 0,
        comment: str = "",
        expiration: datetime = None
    ) -> Optional[Dict]:
        """Place a new order."""
        if not self.connected:
            await self.connect()
            
        # Prepare the request
        symbol_info = await self.get_symbol_info(symbol)
        if symbol_info is None:
            return None
            
        point = symbol_info.point
        ask = mt5.symbol_info_tick(symbol).ask
        bid = mt5.symbol_info_tick(symbol).bid
        
        if order_type == OrderType.BUY:
            order_type = mt5.ORDER_TYPE_BUY
            price = ask if price is None else price
        elif order_type == OrderType.SELL:
            order_type = mt5.ORDER_TYPE_SELL
            price = bid if price is None else price
        else:
            raise ValueError(f"Unsupported order type: {order_type}")
        
        # Prepare the request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": deviation,
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }
        
        if expiration:
            request["type_time"] = mt5.ORDER_TIME_SPECIFIED
            request["expiration"] = expiration
        
        # Send the order
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.comment} (code: {result.retcode})")
            return None
            
        return result._asdict()
    
    async def close_position(self, ticket: int, volume: float = None) -> bool:
        """Close an open position."""
        if not self.connected:
            await self.connect()
            
        # Get the position
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            logger.error(f"Position {ticket} not found")
            return False
            
        position = positions[0]._asdict()
        symbol = position['symbol']
        position_type = position['type']
        
        # Prepare the close request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume or position['volume'],
            "type": mt5.ORDER_TYPE_SELL if position_type == 0 else mt5.ORDER_TYPE_BUY,
            "position": ticket,
            "magic": position['magic'],
            "comment": f"Closed by bot",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }
        
        # Send the close request
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Close position failed: {result.comment} (code: {result.retcode})")
            return False
            
        return True
    
    async def get_historical_data(
        self,
        symbol: str,
        timeframe: int,
        from_date: datetime,
        to_date: datetime = None,
        count: int = None
    ) -> pd.DataFrame:
        """Get historical price data."""
        if not self.connected:
            await self.connect()
            
        if to_date is None:
            to_date = datetime.now()
            
        rates = mt5.copy_rates_range(symbol, timeframe, from_date, to_date)
        if rates is None:
            logger.error(f"Failed to get historical data for {symbol}")
            return pd.DataFrame()
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        return df
    
    async def get_last_tick(self, symbol: str) -> Dict:
        """Get the last tick for a symbol."""
        if not self.connected:
            await self.connect()
            
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Failed to get tick for {symbol}")
            return {}
            
        return tick._asdict()
    
    async def subscribe_to_ticks(self, symbols: List[str], callback) -> bool:
        """Subscribe to real-time tick updates."""
        if not self.connected:
            await self.connect()
            
        for symbol in symbols:
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to select symbol {symbol}")
                return False
                
        # Start a background task to poll for ticks
        asyncio.create_task(self._tick_loop(symbols, callback))
        return True
    
    async def _tick_loop(self, symbols: List[str], callback) -> None:
        """Background task to poll for ticks."""
        last_ticks = {symbol: None for symbol in symbols}
        
        while self.connected:
            try:
                for symbol in symbols:
                    tick = await self.get_last_tick(symbol)
                    if tick and tick != last_ticks.get(symbol):
                        last_ticks[symbol] = tick
                        await callback(symbol, tick)
                
                await asyncio.sleep(0.1)  # 100ms update rate
                
            except Exception as e:
                logger.error(f"Error in tick loop: {str(e)}")
                await asyncio.sleep(1)  # Wait before retrying

# Singleton instance
mt5_client = MT5Client()
