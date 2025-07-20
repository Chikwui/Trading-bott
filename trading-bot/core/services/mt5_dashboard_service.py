"""
MT5 Dashboard Service

This module provides a service layer between the dashboard and MT5 components,
handling data retrieval, transformation, and real-time updates.
"""
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set, Union
import logging

from core.utils.helpers import get_logger
from core.monitoring.realtime import publish_update

logger = get_logger(__name__)

class MT5DashboardService:
    """
    Service for providing data to the trading dashboard.
    
    This class acts as a bridge between the dashboard and MT5 components,
    handling data retrieval, transformation, and real-time updates.
    """
    
    def __init__(self, mt5_broker, mt5_provider):
        """
        Initialize the dashboard service.
        
        Args:
            mt5_broker: MT5 live broker instance
            mt5_provider: MT5 data provider instance
        """
        self.mt5_broker = mt5_broker
        self.mt5_provider = mt5_provider
        self._subscribers: Set[Callable[[str, Any], None]] = set()
        self._running = False
        self._update_task = None
        
    async def start(self):
        """Start the dashboard service."""
        if self._running:
            return
            
        self._running = True
        self._update_task = asyncio.create_task(self._update_loop())
        logger.info("MT5 Dashboard Service started")
        
    async def stop(self):
        """Stop the dashboard service."""
        if not self._running:
            return
            
        self._running = False
        if self._update_task:
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        logger.info("MT5 Dashboard Service stopped")
        
    def subscribe(self, callback: Callable[[str, Any], None]):
        """
        Subscribe to real-time updates.
        
        Args:
            callback: Function to call with updates (event_type, data)
        """
        self._subscribers.add(callback)
        
    def unsubscribe(self, callback: Callable[[str, Any], None]):
        """Unsubscribe from updates."""
        self._subscribers.discard(callback)
        
    async def _publish(self, event_type: str, data: Any):
        """Publish an update to all subscribers and real-time clients."""
        # Publish to WebSocket/SSE clients
        await publish_update(event_type, data)
        
        # Call local subscribers
        for callback in list(self._subscribers):
            try:
                callback(event_type, data)
            except Exception as e:
                logger.error(f"Error in subscriber callback: {e}")
                
    async def _update_loop(self):
        """Main update loop for real-time data."""
        last_account_info = None
        last_positions = None
        last_orders = None
        
        while self._running:
            try:
                # Update account info (only if changed)
                account_info = await self.get_account_info()
                if account_info != last_account_info:
                    await self._publish('account_update', account_info)
                    last_account_info = account_info
                
                # Update positions (only if changed)
                positions = await self.get_positions()
                if positions != last_positions:
                    await self._publish('positions_update', positions)
                    last_positions = positions
                
                # Update orders (only if changed)
                orders = await self.get_orders()
                if orders != last_orders:
                    await self._publish('orders_update', orders)
                    last_orders = orders
                
                # Publish connection status
                await self._publish('connection_status', {
                    'is_connected': self.mt5_broker.is_connected() if self.mt5_broker else False,
                    'timestamp': datetime.utcnow().isoformat()
                })
                
                # Sleep before next update (configurable)
                await asyncio.sleep(1)  # More frequent updates for better real-time feel
                
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                await asyncio.sleep(5)  # Back off on error
                
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                await asyncio.sleep(1)  # Prevent tight loop on error
                
    # Data retrieval methods
    async def get_account_info(self) -> Dict[str, Any]:
        """Get current account information with additional metrics."""
        if not self.mt5_broker:
            return {
                'balance': 0,
                'equity': 0,
                'margin': 0,
                'free_margin': 0,
                'margin_level': 0,
                'currency': 'USD',
                'leverage': 0,
                'timestamp': datetime.utcnow().isoformat(),
                'is_connected': False
            }
            
        try:
            account_info = await self.mt5_broker.get_account_info()
            
            # Calculate additional metrics
            positions = await self.get_positions()
            open_pnl = sum(p.get('profit', 0) for p in positions)
            
            return {
                'balance': float(account_info.balance),
                'equity': float(account_info.equity),
                'margin': float(account_info.margin),
                'free_margin': float(account_info.margin_free),
                'margin_level': float(account_info.margin_level) if account_info.margin_level else 0,
                'currency': account_info.currency,
                'leverage': account_info.leverage,
                'open_pnl': open_pnl,
                'timestamp': datetime.utcnow().isoformat(),
                'is_connected': True,
                'server_time': datetime.utcnow().isoformat(),
                'positions_count': len(positions),
                'orders_count': len(await self.get_orders())
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {
                'error': str(e),
                'is_connected': False,
                'timestamp': datetime.utcnow().isoformat()
            }
            
    async def get_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions with additional metrics."""
        if not self.mt5_broker:
            return []
            
        try:
            positions = await self.mt5_broker.get_positions()
            result = []
            
            for pos in positions:
                # Calculate additional metrics
                pnl_pct = ((pos.price_current / pos.price_open) - 1) * 100
                if pos.type == 1:  # SELL position
                    pnl_pct = -pnl_pct
                    
                # Calculate time in trade
                time_in_trade = datetime.utcnow() - datetime.fromtimestamp(pos.time)
                
                result.append({
                    'ticket': pos.ticket,
                    'symbol': pos.symbol,
                    'type': 'BUY' if pos.type == 0 else 'SELL',
                    'volume': float(pos.volume),
                    'open_price': float(pos.price_open),
                    'current_price': float(pos.price_current),
                    'sl': float(pos.sl) if pos.sl else None,
                    'tp': float(pos.tp) if pos.tp else None,
                    'profit': float(pos.profit),
                    'profit_pct': pnl_pct,
                    'swap': float(pos.swap),
                    'commission': float(pos.commission),
                    'time': datetime.fromtimestamp(pos.time).isoformat(),
                    'time_in_trade_seconds': int(time_in_trade.total_seconds()),
                    'time_in_trade': str(time_in_trade).split('.')[0],  # Remove microseconds
                    'comment': pos.comment or ''
                })
                
            return result
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
            
    async def get_orders(self) -> List[Dict[str, Any]]:
        """Get current open orders."""
        try:
            orders = self.mt5_broker.get_orders() or []
            return [{
                'id': str(order.get('ticket', '')),
                'symbol': order.get('symbol', ''),
                'type': self._map_order_type(order.get('type', 0)),
                'volume': float(order.get('volume_current', 0)),
                'price': float(order.get('price_open', 0)),
                'stop_loss': float(order.get('sl', 0)) if order.get('sl', 0) > 0 else None,
                'take_profit': float(order.get('tp', 0)) if order.get('tp', 0) > 0 else None,
                'status': self._map_order_status(order.get('state', '')),
                'timestamp': datetime.utcnow().isoformat()
            } for order in orders]
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            return []
    
    # Helper methods
    def _map_order_type(self, mt5_type: int) -> str:
        """Map MT5 order type to string representation."""
        order_types = {
            0: 'buy',
            1: 'sell',
            2: 'buy_limit',
            3: 'sell_limit',
            4: 'buy_stop',
            5: 'sell_stop'
        }
        return order_types.get(mt5_type, 'unknown')
    
    def _map_order_status(self, mt5_status: str) -> str:
        """Map MT5 order status to string representation."""
        status_map = {
            'Started': 'pending',
            'Placed': 'pending',
            'Canceled': 'cancelled',
            'Partial': 'partially_filled',
            'Filled': 'filled',
            'Rejected': 'rejected',
            'Expired': 'expired'
        }
        return status_map.get(mt5_status, 'unknown')
