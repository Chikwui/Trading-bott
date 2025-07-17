"""
MT5 Backtest Broker Implementation

This module provides a backtesting implementation of the MT5 broker interface.
It simulates MT5's behavior for backtesting purposes.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Any, Union, Tuple, Deque
from datetime import datetime, timedelta
import asyncio
import logging
import random
import numpy as np
import pandas as pd
from decimal import Decimal, ROUND_HALF_UP
from collections import defaultdict, deque

from ...data.providers.mt5_provider import MT5DataProvider, MT5_TIMEFRAME_MAP, MT5_TIMEFRAME_REVERSE_MAP
from ...models import Order, OrderType, OrderSide, OrderStatus, TimeInForce, Position
from ..broker import Broker, BrokerType, BrokerError, OrderError, MarketDataError

logger = logging.getLogger(__name__)

class MT5BacktestBroker(Broker):
    """
    Backtest implementation of the MT5 broker interface.
    
    This broker simulates MT5's behavior for backtesting purposes.
    It uses historical data to simulate order fills and market conditions.
    """
    
    def __init__(
        self, 
        data_provider: MT5DataProvider,
        initial_balance: float = 10000.0,
        leverage: float = 1.0,
        commission: float = 0.0005,  # 0.05% commission
        spread: float = 0.0002,      # 2 pips spread
        slippage: float = 0.0001,    # 1 pip slippage
        **kwargs
    ):
        """
        Initialize the MT5 backtest broker.
        
        Args:
            data_provider: MT5 data provider instance
            initial_balance: Initial account balance in base currency
            leverage: Account leverage (e.g., 1.0 for no leverage, 10.0 for 10x)
            commission: Commission rate per trade (as a decimal, e.g., 0.0005 for 0.05%)
            spread: Bid/ask spread (as a decimal, e.g., 0.0002 for 2 pips)
            slippage: Slippage factor (as a decimal, e.g., 0.0001 for 1 pip)
            **kwargs: Additional broker configuration
        """
        super().__init__(BrokerType.BACKTEST, kwargs)
        
        self.data_provider = data_provider
        self.initial_balance = Decimal(str(initial_balance))
        self.balance = self.initial_balance
        self.leverage = Decimal(str(leverage))
        self.commission_rate = Decimal(str(commission))
        self.spread = Decimal(str(spread))
        self.slippage = Decimal(str(slippage))
        
        # State tracking
        self.positions = {}  # symbol -> Position
        self.orders = {}     # order_id -> Order
        self.order_history = []
        self.trade_history = []
        self.current_time = None
        self._last_tick = {}
        
        # Performance metrics
        self.commissions_paid = Decimal('0')
        self.slippage_cost = Decimal('0')
        self.trades_executed = 0
        
        # Connect to data provider
        self.connected = False
        
    async def connect(self) -> bool:
        """Connect to the backtest broker."""
        try:
            if not self.data_provider.connected:
                await self.data_provider.connect()
            
            self.connected = True
            self.current_time = datetime.utcnow()
            logger.info(f"MT5BacktestBroker connected with initial balance: ${self.initial_balance:,.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to backtest broker: {str(e)}")
            self.connected = False
            raise ConnectionError(f"Failed to connect to backtest broker: {str(e)}") from e
    
    async def disconnect(self) -> None:
        """Disconnect from the backtest broker."""
        self.connected = False
        logger.info("MT5BacktestBroker disconnected")
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get account information."""
        if not self.connected:
            raise BrokerError("Not connected to broker")
            
        # Calculate account metrics
        equity = self.balance
        margin_used = Decimal('0')
        margin_free = equity
        
        # Calculate position values
        for position in self.positions.values():
            if position.quantity > 0:  # Long position
                position_value = position.quantity * Decimal(str(position.entry_price))
                margin_used += position_value / self.leverage
                equity += position.unrealized_pnl
            else:  # Short position
                position_value = abs(position.quantity) * Decimal(str(position.entry_price))
                margin_used += position_value / self.leverage
                equity += position.unrealized_pnl
        
        margin_free = max(Decimal('0'), equity - margin_used)
        
        return {
            'balance': float(self.balance),
            'equity': float(equity),
            'margin_used': float(margin_used),
            'margin_free': float(margin_free),
            'leverage': float(self.leverage),
            'currency': 'USD',
            'timestamp': self.current_time.isoformat()
        }
    
    async def place_order(self, order: Order) -> str:
        """Place a new order."""
        if not self.connected:
            raise OrderError("Not connected to broker")
            
        # Generate order ID if not provided
        if not order.order_id:
            order.order_id = f"order_{len(self.orders) + 1}"
        
        # Set order timestamp
        order.created_at = order.created_at or self.current_time
        order.updated_at = self.current_time
        
        # Validate order
        self._validate_order(order)
        
        # Process order based on type
        if order.order_type == OrderType.MARKET:
            await self._process_market_order(order)
        elif order.order_type in (OrderType.LIMIT, OrderType.STOP):
            await self._process_pending_order(order)
        else:
            raise OrderError(f"Unsupported order type: {order.order_type}")
        
        # Add to orders and history
        self.orders[order.order_id] = order
        self.order_history.append(order)
        
        logger.info(f"Order placed: {order}")
        return order.order_id
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        if not self.connected:
            raise OrderError("Not connected to broker")
            
        if order_id not in self.orders:
            raise OrderError(f"Order {order_id} not found")
            
        order = self.orders[order_id]
        
        # Only pending orders can be canceled
        if order.status not in (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED):
            logger.warning(f"Cannot cancel order {order_id} with status {order.status}")
            return False
        
        # Update order status
        order.status = OrderStatus.CANCELED
        order.updated_at = self.current_time
        
        logger.info(f"Order canceled: {order_id}")
        return True
    
    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Get the status of an order."""
        if not self.connected:
            raise OrderError("Not connected to broker")
            
        if order_id not in self.orders:
            raise OrderError(f"Order {order_id} not found")
            
        order = self.orders[order_id]
        return order.to_dict()
    
    async def get_positions(self, symbol: str = None) -> List[Dict[str, Any]]:
        """Get current open positions."""
        if not self.connected:
            raise BrokerError("Not connected to broker")
            
        if symbol:
            return [pos.to_dict() for pos in self.positions.values() if pos.symbol == symbol]
        return [pos.to_dict() for pos in self.positions.values()]
    
    async def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start: datetime = None, 
        end: datetime = None, 
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Get historical market data."""
        if not self.connected:
            raise MarketDataError("Not connected to broker")
            
        # Convert timeframe to MT5 format
        mt5_timeframe = self._convert_timeframe(timeframe)
        
        # Get data from provider
        data = await self.data_provider.get_historical_data(
            symbol=symbol,
            timeframe=mt5_timeframe,
            start=start,
            end=end,
            limit=limit
        )
        
        return data
    
    async def get_current_price(self, symbol: str) -> float:
        """Get the current market price for a symbol."""
        if not self.connected:
            raise MarketDataError("Not connected to broker")
            
        # Get latest tick data
        tick = await self.data_provider.get_last_tick(symbol)
        if not tick:
            raise MarketDataError(f"No price data available for {symbol}")
            
        # Return mid price (average of bid and ask)
        return (tick['bid'] + tick['ask']) / 2
    
    async def get_order_book(self, symbol: str, depth: int = 10) -> Dict[str, Any]:
        """Get the current order book for a symbol."""
        if not self.connected:
            raise MarketDataError("Not connected to broker")
            
        # Get order book from data provider
        order_book = await self.data_provider.get_order_book(symbol, depth)
        
        # Simulate spread if not available
        if not order_book or 'bids' not in order_book or 'asks' not in order_book:
            price = await self.get_current_price(symbol)
            spread = self.spread * price
            order_book = {
                'bids': [[price - spread/2, 1.0]] * depth,
                'asks': [[price + spread/2, 1.0]] * depth,
                'timestamp': self.current_time.isoformat()
            }
            
        return order_book
    
    def is_connected(self) -> bool:
        """Check if the broker is connected."""
        return self.connected
    
    def get_broker_info(self) -> Dict[str, Any]:
        """Get information about the broker."""
        account_info = {
            'balance': float(self.balance),
            'equity': float(self.balance),
            'margin_used': 0.0,
            'margin_free': float(self.balance),
            'leverage': float(self.leverage),
            'currency': 'USD',
            'timestamp': self.current_time.isoformat() if self.current_time else None
        }
        
        return {
            'broker_type': self.broker_type.name,
            'connected': self.connected,
            'last_update': self._last_update,
            'account': account_info,
            'positions': len(self.positions),
            'open_orders': len([o for o in self.orders.values() if o.status in (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED)]),
            'trades_executed': self.trades_executed,
            'commissions_paid': float(self.commissions_paid),
            'slippage_cost': float(self.slippage_cost)
        }
    
    # Internal methods
    
    def _validate_order(self, order: Order) -> None:
        """Validate order parameters."""
        if order.quantity <= 0:
            raise OrderError("Order quantity must be positive")
            
        if order.symbol not in self.data_provider.available_symbols():
            raise OrderError(f"Symbol {order.symbol} not available")
            
        # Additional validation based on order type
        if order.order_type == OrderType.LIMIT and not order.limit_price:
            raise OrderError("Limit price is required for limit orders")
            
        if order.order_type == OrderType.STOP and not order.stop_price:
            raise OrderError("Stop price is required for stop orders")
    
    async def _process_market_order(self, order: Order) -> None:
        """Process a market order."""
        # Get current price with slippage and spread
        price = await self._get_execution_price(order.symbol, order.side)
        
        # Calculate commission
        order_value = Decimal(str(price)) * Decimal(str(abs(order.quantity)))
        commission = order_value * self.commission_rate
        
        # Update order status
        order.status = OrderStatus.FILLED
        order.avg_fill_price = float(price)
        order.filled_quantity = order.quantity
        order.commission = float(commission)
        order.updated_at = self.current_time
        
        # Update position
        await self._update_position(order, price, commission)
        
        # Update metrics
        self.commissions_paid += commission
        self.trades_executed += 1
        
        logger.info(f"Market order executed: {order.order_id} {order.side} {order.quantity} "
                   f"{order.symbol} @ {price:.5f} (Comm: ${commission:.2f})")
    
    async def _process_pending_order(self, order: Order) -> None:
        """Process a pending order (limit or stop)."""
        # For backtesting, we'll assume the order is placed but not filled yet
        order.status = OrderStatus.NEW
        order.updated_at = self.current_time
        
        logger.info(f"Pending order placed: {order.order_id} {order.order_type} {order.side} "
                   f"{order.quantity} {order.symbol}")
    
    async def _update_position(
        self, 
        order: Order, 
        fill_price: float, 
        commission: Decimal
    ) -> None:
        """Update position based on order fill."""
        symbol = order.symbol
        quantity = order.quantity if order.side == OrderSide.BUY else -order.quantity
        
        if symbol not in self.positions:
            # Create new position
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=fill_price,
                current_price=fill_price,
                unrealized_pnl=0.0,
                realized_pnl=0.0,
                commission=float(commission),
                created_at=self.current_time,
                updated_at=self.current_time
            )
        else:
            # Update existing position
            position = self.positions[symbol]
            
            # Calculate position value before update
            prev_value = position.quantity * Decimal(str(position.entry_price))
            
            # Update position
            if (position.quantity > 0 and quantity > 0) or (position.quantity < 0 and quantity < 0):
                # Adding to position (same direction)
                total_quantity = position.quantity + quantity
                total_value = prev_value + (Decimal(str(quantity)) * Decimal(str(fill_price)))
                position.entry_price = float(total_value / Decimal(str(total_quantity)))
                position.quantity = total_quantity
            else:
                # Reducing or flipping position
                if abs(quantity) <= abs(position.quantity):
                    # Partial close
                    position.quantity += quantity
                    # Update realized P&L
                    pnl = (Decimal(str(fill_price)) - Decimal(str(position.entry_price))) * Decimal(str(abs(quantity)))
                    if position.quantity < 0:  # Was long, now short
                        pnl = -pnl
                    position.realized_pnl += float(pnl)
                else:
                    # Full close and flip
                    pnl = (Decimal(str(fill_price)) - Decimal(str(position.entry_price))) * Decimal(str(abs(position.quantity)))
                    if position.quantity < 0:  # Was long, now short
                        pnl = -pnl
                    position.realized_pnl += float(pnl)
                    
                    # Flip position
                    position.quantity += quantity
                    position.entry_price = fill_price
            
            # Update current price and unrealized P&L
            position.current_price = fill_price
            position.unrealized_pnl = float(
                (Decimal(str(fill_price)) - Decimal(str(position.entry_price))) * 
                Decimal(str(position.quantity))
            )
            
            # Update commission
            position.commission += float(commission)
            position.updated_at = self.current_time
            
            # Remove position if fully closed
            if position.quantity == 0:
                del self.positions[symbol]
    
    async def _get_execution_price(self, symbol: str, side: OrderSide) -> Decimal:
        """Get execution price with slippage and spread."""
        # Get current mid price
        mid_price = Decimal(str(await self.get_current_price(symbol)))
        
        # Calculate spread
        spread = self.spread * mid_price / Decimal('2')
        
        # Apply spread based on order side
        if side == OrderSide.BUY:
            base_price = mid_price + spread  # Pay the ask
        else:  # SELL
            base_price = mid_price - spread  # Get the bid
        
        # Apply random slippage (can be positive or negative)
        slippage = random.uniform(-1, 1) * self.slippage * mid_price
        execution_price = base_price + Decimal(str(slippage))
        
        # Update slippage cost (absolute value)
        self.slippage_cost += abs(Decimal(str(slippage)) * Decimal('1'))  # Assume 1 unit for cost calculation
        
        return execution_price.quantize(Decimal('0.00001'), rounding=ROUND_HALF_UP)
    
    def _convert_timeframe(self, timeframe: str) -> int:
        """Convert timeframe string to MT5 timeframe."""
        # Convert standard timeframe strings to MT5 format
        tf_map = {
            '1m': mt5.TIMEFRAME_M1,
            '5m': mt5.TIMEFRAME_M5,
            '15m': mt5.TIMEFRAME_M15,
            '30m': mt5.TIMEFRAME_M30,
            '1h': mt5.TIMEFRAME_H1,
            '4h': mt5.TIMEFRAME_H4,
            '1d': mt5.TIMEFRAME_D1,
            '1w': mt5.TIMEFRAME_W1,
            '1M': mt5.TIMEFRAME_MN1
        }
        
        return tf_map.get(timeframe, mt5.TIMEFRAME_M1)  # Default to M1 if not found
    
    # Backtest-specific methods
    
    async def update_market_data(self, timestamp: datetime) -> None:
        """Update market data to the specified timestamp."""
        if not self.connected:
            raise BrokerError("Not connected to broker")
            
        self.current_time = timestamp
        
        # Update positions with latest prices
        for symbol, position in self.positions.items():
            try:
                price = await self.get_current_price(symbol)
                position.current_price = price
                position.unrealized_pnl = float(
                    (Decimal(str(price)) - Decimal(str(position.entry_price))) * 
                    Decimal(str(position.quantity))
                )
                position.updated_at = timestamp
            except Exception as e:
                logger.error(f"Error updating position for {symbol}: {str(e)}")
        
        # Check for pending orders that should be triggered
        await self._check_pending_orders()
    
    async def _check_pending_orders(self) -> None:
        """Check if any pending orders should be triggered."""
        if not self.connected:
            return
            
        for order in list(self.orders.values()):
            if order.status not in (OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED):
                continue
                
            if order.order_type == OrderType.LIMIT:
                await self._check_limit_order(order)
            elif order.order_type == OrderType.STOP:
                await self._check_stop_order(order)
    
    async def _check_limit_order(self, order: Order) -> None:
        """Check if a limit order should be filled."""
        try:
            current_price = await self.get_current_price(order.symbol)
            
            # Check if price has crossed the limit
            if (order.side == OrderSide.BUY and current_price <= order.limit_price) or \
               (order.side == OrderSide.SELL and current_price >= order.limit_price):
                # Execute the order at the limit price
                order.status = OrderStatus.FILLED
                order.avg_fill_price = order.limit_price
                order.filled_quantity = order.quantity
                order.updated_at = self.current_time
                
                # Calculate commission
                order_value = Decimal(str(order.limit_price)) * Decimal(str(abs(order.quantity)))
                commission = order_value * self.commission_rate
                order.commission = float(commission)
                
                # Update position
                await self._update_position(order, order.limit_price, commission)
                
                # Update metrics
                self.commissions_paid += commission
                self.trades_executed += 1
                
                logger.info(f"Limit order filled: {order.order_id} {order.side} {order.quantity} "
                          f"{order.symbol} @ {order.limit_price:.5f}")
        except Exception as e:
            logger.error(f"Error checking limit order {order.order_id}: {str(e)}")
    
    async def _check_stop_order(self, order: Order) -> None:
        """Check if a stop order should be triggered."""
        try:
            current_price = await self.get_current_price(order.symbol)
            
            # Check if price has crossed the stop
            if (order.side == OrderSide.BUY and current_price >= order.stop_price) or \
               (order.side == OrderSide.SELL and current_price <= order.stop_price):
                # Execute the order at the current market price
                execution_price = await self._get_execution_price(order.symbol, order.side)
                
                order.status = OrderStatus.FILLED
                order.avg_fill_price = float(execution_price)
                order.filled_quantity = order.quantity
                order.updated_at = self.current_time
                
                # Calculate commission
                order_value = execution_price * Decimal(str(abs(order.quantity)))
                commission = order_value * self.commission_rate
                order.commission = float(commission)
                
                # Update position
                await self._update_position(order, float(execution_price), commission)
                
                # Update metrics
                self.commissions_paid += commission
                self.trades_executed += 1
                
                logger.info(f"Stop order triggered: {order.order_id} {order.side} {order.quantity} "
                          f"{order.symbol} @ {execution_price:.5f}")
        except Exception as e:
            logger.error(f"Error checking stop order {order.order_id}: {str(e)}")

# Factory function for creating MT5 backtest brokers
def create_mt5_backtest_broker(
    data_provider: MT5DataProvider,
    initial_balance: float = 10000.0,
    leverage: float = 1.0,
    commission: float = 0.0005,
    spread: float = 0.0002,
    slippage: float = 0.0001
) -> MT5BacktestBroker:
    """
    Create a new MT5 backtest broker instance.
    
    Args:
        data_provider: MT5 data provider instance
        initial_balance: Initial account balance in base currency
        leverage: Account leverage (e.g., 1.0 for no leverage, 10.0 for 10x)
        commission: Commission rate per trade (as a decimal, e.g., 0.0005 for 0.05%)
        spread: Bid/ask spread (as a decimal, e.g., 0.0002 for 2 pips)
        slippage: Slippage factor (as a decimal, e.g., 0.0001 for 1 pip)
        
    Returns:
        Configured MT5BacktestBroker instance
    """
    return MT5BacktestBroker(
        data_provider=data_provider,
        initial_balance=initial_balance,
        leverage=leverage,
        commission=commission,
        spread=spread,
        slippage=slippage
    )
