"""
Binance exchange implementation.
"""
import asyncio
import time
import hmac
import hashlib
import json
import urllib.parse
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta

import aiohttp
import pandas as pd
from dateutil import parser

from .base import (
    BaseExchange,
    ExchangeError,
    ExchangeConnectionError,
    ExchangeAPIError,
    InsufficientFunds,
    OrderNotFound
)
from ...models import (
    Ticker,
    OHLCV,
    Balance,
    Position,
    Order,
    OrderType,
    OrderSide,
    OrderStatus,
    Trade,
    TradeSide,
    Liquidity
)
from ...utils.logger import get_logger

logger = get_logger(__name__)

class BinanceExchange(BaseExchange):
    """Binance exchange implementation."""
    
    @property
    def name(self) -> str:
        return "binance"
    
    @property
    def base_url(self) -> str:
        if self.sandbox:
            return "https://testnet.binance.vision/api/v3"
        return "https://api.binance.com/api/v3"
    
    @property
    def ws_url(self) -> str:
        if self.sandbox:
            return "wss://testnet.binance.vision/ws"
        return "wss://stream.binance.com:9443/ws"
    
    @property
    def futures_url(self) -> str:
        if self.sandbox:
            return "https://testnet.binancefuture.com/fapi/v1"
        return "https://fapi.binance.com/fapi/v1"
    
    @property
    def futures_ws_url(self) -> str:
        if self.sandbox:
            return "wss://stream.binancefuture.com/ws"
        return "wss://fstream.binance.com/ws"
    
    def __init__(self, 
                 api_key: str = None, 
                 api_secret: str = None, 
                 api_passphrase: str = None,
                 sandbox: bool = False):
        """Initialize the Binance client.
        
        Args:
            api_key: Binance API key
            api_secret: Binance API secret
            api_passphrase: Not used for Binance
            sandbox: Whether to use the testnet
        """
        super().__init__(api_key, api_secret, api_passphrase, sandbox)
        self._session = None
        self._ws = None
        self._last_request_time = 0
        self._rate_limit_semaphore = asyncio.Semaphore(10)
        self._listen_key = None
        self._listen_key_refresh_task = None
    
    async def _request(self, 
                      method: str, 
                      path: str, 
                      params: dict = None, 
                      data: dict = None, 
                      auth: bool = False,
                      futures: bool = False) -> dict:
        """Send a request to the Binance API.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            path: API endpoint path
            params: Query parameters
            data: Request body
            auth: Whether to sign the request
            futures: Whether to use the futures API
            
        Returns:
            Response data as a dictionary
            
        Raises:
            ExchangeConnectionError: If there is a connection error
            ExchangeAPIError: If the API returns an error
        """
        url = f"{self.futures_url if futures else self.base_url}/{path}"
        headers = {
            'Accept': 'application/json',
            'User-Agent': 'TradingBot/1.0',
        }
        
        if auth:
            if not self.api_key or not self.api_secret:
                raise ExchangeError("API key and secret are required for authenticated requests")
            
            headers['X-MBX-APIKEY'] = self.api_key
            
            # Add timestamp to signed requests
            if params is None:
                params = {}
            params['timestamp'] = int(time.time() * 1000)
            
            # Generate signature
            query_string = urllib.parse.urlencode(params)
            signature = hmac.new(
                self.api_secret.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            params['signature'] = signature
        
        try:
            async with self._rate_limit_semaphore:
                await self._rate_limit()
                
                async with aiohttp.ClientSession() as session:
                    async with session.request(
                        method, 
                        url, 
                        params=params, 
                        json=data,
                        headers=headers,
                        timeout=10
                    ) as response:
                        self._last_request_time = time.time()
                        
                        if response.status == 200:
                            return await response.json()
                        else:
                            error_text = await response.text()
                            try:
                                error_data = await response.json()
                                error_msg = error_data.get('msg', error_text)
                                error_code = error_data.get('code', response.status)
                                
                                if error_code == -2010:  # Insufficient balance
                                    raise InsufficientFunds(error_msg)
                                elif error_code == -2013:  # Order does not exist
                                    raise OrderNotFound(error_msg)
                                else:
                                    raise ExchangeAPIError(f"API error {error_code}: {error_msg}")
                                    
                            except (ValueError, KeyError):
                                raise ExchangeAPIError(f"HTTP {response.status}: {error_text}")
                                
        except aiohttp.ClientError as e:
            raise ExchangeConnectionError(f"Connection error: {str(e)}")
        except asyncio.TimeoutError:
            raise ExchangeConnectionError("Request timed out")
    
    async def connect(self) -> None:
        """Connect to the exchange."""
        if self._session is not None and not self._session.closed:
            return
            
        self._session = aiohttp.ClientSession()
        
        # Start WebSocket connection for user data stream
        if self.api_key and self.api_secret:
            await self._start_user_data_stream()
    
    async def close(self) -> None:
        """Close the connection to the exchange."""
        if self._listen_key_refresh_task:
            self._listen_key_refresh_task.cancel()
            self._listen_key_refresh_task = None
            
        if self._ws and not self._ws.closed:
            await self._ws.close()
            self._ws = None
            
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
    
    async def _start_user_data_stream(self) -> None:
        """Start the user data stream."""
        try:
            # Get a listen key
            response = await self._request('POST', 'userDataStream', futures=True)
            self._listen_key = response['listenKey']
            
            # Start a task to keep the listen key alive
            self._listen_key_refresh_task = asyncio.create_task(self._keepalive_listen_key())
            
            # Connect to the user data stream
            ws_url = f"{self.futures_ws_url}/{self._listen_key}"
            self._ws = await self._session.ws_connect(ws_url)
            
            # Start listening for messages
            asyncio.create_task(self._handle_ws_messages())
            
        except Exception as e:
            logger.error(f"Failed to start user data stream: {e}")
    
    async def _keepalive_listen_key(self) -> None:
        """Keep the listen key alive."""
        while True:
            try:
                await asyncio.sleep(30 * 60)  # 30 minutes
                if self._listen_key:
                    await self._request('PUT', 'userDataStream', 
                                      params={'listenKey': self._listen_key}, 
                                      futures=True)
            except asyncio.CancelledError:
                # Clean up the listen key when shutting down
                if self._listen_key:
                    try:
                        await self._request('DELETE', 'userDataStream', 
                                          params={'listenKey': self._listen_key}, 
                                          futures=True)
                    except Exception as e:
                        logger.error(f"Failed to delete listen key: {e}")
                break
            except Exception as e:
                logger.error(f"Failed to keepalive listen key: {e}")
    
    async def _handle_ws_messages(self) -> None:
        """Handle WebSocket messages."""
        if not self._ws:
            return
            
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = json.loads(msg.data)
                    logger.debug(f"WebSocket message: {data}")
                    
                    # Handle different message types
                    if 'e' in data:  # Event type
                        if data['e'] == 'executionReport':
                            # Order update
                            await self._handle_order_update(data)
                        elif data['e'] == 'ACCOUNT_UPDATE':
                            # Balance or position update
                            await self._handle_account_update(data)
                        elif data['e'] == 'ORDER_TRADE_UPDATE':
                            # Order or trade update
                            await self._handle_order_trade_update(data)
                    
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
        finally:
            if self._ws and not self._ws.closed:
                await self._ws.close()
                self._ws = None
    
    async def _handle_order_update(self, data: dict) -> None:
        """Handle order update from WebSocket."""
        # TODO: Implement order update handling
        pass
    
    async def _handle_account_update(self, data: dict) -> None:
        """Handle account update from WebSocket."""
        # TODO: Implement account update handling
        pass
    
    async def _handle_order_trade_update(self, data: dict) -> None:
        """Handle order/trade update from WebSocket."""
        # TODO: Implement order/trade update handling
        pass
    
    # Market Data Methods
    
    async def get_ticker(self, symbol: str) -> Ticker:
        """Get the current ticker for a symbol."""
        data = await self._request('GET', 'ticker/24hr', params={'symbol': symbol.replace('/', '')})
        
        return Ticker(
            symbol=symbol,
            timestamp=datetime.utcfromtimestamp(data['closeTime'] / 1000),
            bid=float(data['bidPrice']),
            ask=float(data['askPrice']),
            last=float(data['lastPrice']),
            bid_volume=float(data['bidQty']),
            ask_volume=float(data['askQty']),
            base_volume=float(data['volume']),
            quote_volume=float(data['quoteVolume']),
            vwap=float(data.get('weightedAvgPrice', 0)),
            open=float(data['openPrice']),
            high=float(data['highPrice']),
            low=float(data['lowPrice']),
            close=float(data['lastPrice']),
            change=float(data['priceChange']),
            percentage=float(data['priceChangePercent'])
        )
    
    async def get_ohlcv(self, 
                       symbol: str, 
                       timeframe: str = '1h', 
                       limit: int = 500,
                       since: int = None) -> List[OHLCV]:
        """Get OHLCV data."""
        # Convert timeframe to Binance format
        tf_mapping = {
            '1m': '1m', '3m': '3m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h', '12h': '12h',
            '1d': '1d', '3d': '3d', '1w': '1w', '1M': '1M'
        }
        
        interval = tf_mapping.get(timeframe)
        if not interval:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        params = {
            'symbol': symbol.replace('/', ''),
            'interval': interval,
            'limit': min(limit, 1000)  # Binance max limit is 1000
        }
        
        if since is not None:
            params['startTime'] = since
        
        klines = await self._request('GET', 'klines', params=params)
        
        return [
            OHLCV(
                symbol=symbol,
                timestamp=datetime.utcfromtimestamp(k[0] / 1000),
                open=float(k[1]),
                high=float(k[2]),
                low=float(k[3]),
                close=float(k[4]),
                volume=float(k[5]),
                vwap=float(k[7]) if len(k) > 7 else None,
                trade_count=int(k[8]) if len(k) > 8 else None
            )
            for k in klines
        ]
    
    async def get_order_book(self, symbol: str, limit: int = 100) -> dict:
        """Get the order book for a symbol."""
        response = await self._request('GET', 'depth', 
                                     params={'symbol': symbol.replace('/', ''), 'limit': limit})
        
        return {
            'bids': [[float(p[0]), float(p[1])] for p in response['bids']],
            'asks': [[float(p[0]), float(p[1])] for p in response['asks']]
        }
    
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[dict]:
        """Get recent trades for a symbol."""
        response = await self._request('GET', 'trades', 
                                     params={'symbol': symbol.replace('/', ''), 'limit': min(limit, 1000)})
        
        return [{
            'id': str(t['id']),
            'timestamp': datetime.utcfromtimestamp(t['time'] / 1000),
            'price': float(t['price']),
            'amount': float(t['qty']),
            'cost': float(t['quoteQty']),
            'side': 'buy' if t['isBuyerMaker'] else 'sell'
        } for t in response]
    
    # Account and Position Management
    
    async def get_balance(self) -> Dict[str, Balance]:
        """Get account balance.
        
        Returns:
            Dictionary of Balance objects keyed by currency symbol
        """
        data = await self._request('GET', 'account', auth=True)
        
        balances = {}
        for asset in data['balances']:
            free = float(asset['free'])
            locked = float(asset['locked'])
            
            if free > 0 or locked > 0:
                balances[asset['asset']] = Balance(
                    currency=asset['asset'],
                    free=free,
                    used=locked,
                    total=free + locked,
                    last_updated=datetime.utcnow()
                )
        
        return balances
    
    async def get_positions(self) -> List[Position]:
        """Get open positions.
        
        Returns:
            List of Position objects
        """
        data = await self._request('GET', 'positionRisk', auth=True, futures=True)
        
        positions = []
        for pos in data:
            position_amt = float(pos['positionAmt'])
            if position_amt == 0:
                continue
                
            positions.append(Position(
                id=pos['symbol'],
                symbol=pos['symbol'],
                side=PositionSide.LONG if position_amt > 0 else PositionSide.SHORT,
                status=PositionStatus.OPEN,
                size=abs(position_amt),
                entry_price=float(pos['entryPrice']),
                mark_price=float(pos['markPrice']),
                liquidation_price=float(pos['liquidationPrice']) if pos['liquidationPrice'] else None,
                unrealized_pnl=float(pos['unRealizedProfit']),
                leverage=float(pos['leverage']),
                margin_ratio=float(pos['marginRatio']) if 'marginRatio' in pos else None,
                opened_at=datetime.utcnow(),  # Binance doesn't provide position open time
                info=pos
            ))
        
        return positions
    
    async def get_account_info(self) -> Dict[str, Any]:
        """Get detailed account information.
        
        Returns:
            Dictionary containing account details
        """
        account = await self._request('GET', 'account', auth=True)
        
        # Get account balances
        balances = {}
        for balance in account['balances']:
            free = float(balance['free'])
            locked = float(balance['locked'])
            if free > 0 or locked > 0:
                balances[balance['asset']] = {
                    'free': free,
                    'locked': locked,
                    'total': free + locked
                }
        
        # Get account permissions and other details
        permissions = account.get('permissions', [])
        
        return {
            'maker_commission': float(account['makerCommission']) / 10000,  # Convert to decimal
            'taker_commission': float(account['takerCommission']) / 10000,
            'buyer_commission': float(account['buyerCommission']) / 100,
            'seller_commission': float(account['sellerCommission']) / 100,
            'can_trade': account.get('canTrade', False),
            'can_withdraw': account.get('canWithdraw', False),
            'can_deposit': account.get('canDeposit', False),
            'update_time': datetime.utcfromtimestamp(account['updateTime'] / 1000),
            'account_type': account.get('accountType', 'SPOT'),
            'permissions': permissions,
            'balances': balances
        }
    
    # Order Management
    
    async def create_order(self, 
                         symbol: str, 
                         order_type: OrderType, 
                         side: OrderSide,
                         amount: float,
                         price: float = None,
                         params: dict = None) -> Order:
        """Create a new order.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            order_type: Type of order (MARKET, LIMIT, etc.)
            side: Order side (BUY or SELL)
            amount: Amount of base currency to buy/sell
            price: Price per unit (required for limit orders)
            params: Additional order parameters
            
        Returns:
            Order object with the created order details
            
        Raises:
            ValueError: If required parameters are missing
            ExchangeAPIError: If the API returns an error
        """
        if params is None:
            params = {}
        
        # Convert order type to Binance format
        type_map = {
            OrderType.MARKET: 'MARKET',
            OrderType.LIMIT: 'LIMIT',
            OrderType.STOP: 'STOP_LOSS',
            OrderType.STOP_LIMIT: 'STOP_LOSS_LIMIT',
            OrderType.TAKE_PROFIT: 'TAKE_PROFIT',
            OrderType.TAKE_PROFIT_LIMIT: 'TAKE_PROFIT_LIMIT',
            OrderType.TRAILING_STOP: 'TRAILING_STOP_MARKET',
            OrderType.FOK: 'LIMIT',
            OrderType.IOC: 'LIMIT'
        }
        
        binance_type = type_map.get(order_type)
        if not binance_type:
            raise ValueError(f"Unsupported order type: {order_type}")
        
        # Prepare order parameters
        order_params = {
            'symbol': symbol.replace('/', '').upper(),
            'side': 'BUY' if side == OrderSide.BUY else 'SELL',
            'type': binance_type,
            'quantity': self._format_amount(symbol, amount),
            'newOrderRespType': 'FULL'  # Get full order details in response
        }
        
        # Add price for limit orders
        if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT, OrderType.TAKE_PROFIT_LIMIT]:
            if price is None:
                raise ValueError("Price is required for limit orders")
            order_params['price'] = self._format_price(symbol, price)
            order_params['timeInForce'] = 'FOK' if order_type == OrderType.FOK else 'IOC' if order_type == OrderType.IOC else 'GTC'
        
        # Add stop price for stop orders
        if order_type in [OrderType.STOP, OrderType.STOP_LIMIT, OrderType.TAKE_PROFIT, OrderType.TAKE_PROFIT_LIMIT]:
            if 'stopPrice' not in params:
                raise ValueError("stopPrice is required for stop/take profit orders")
            order_params['stopPrice'] = self._format_price(symbol, params['stopPrice'])
        
        # Add trailing delta for trailing stop
        if order_type == OrderType.TRAILING_STOP and 'trailingDelta' in params:
            order_params['trailingDelta'] = params['trailingDelta']
        
        # Add any additional parameters
        order_params.update(params)
        
        try:
            # Send the order
            response = await self._request('POST', 'order', data=order_params, auth=True, futures=True)
            return self._parse_order(response)
            
        except ExchangeAPIError as e:
            logger.error(f"Failed to create order: {e}")
            raise
    
    async def cancel_order(self, order_id: str, symbol: str) -> bool:
        """Cancel an order.
        
        Args:
            order_id: ID of the order to cancel
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            bool: True if the order was successfully canceled
            
        Raises:
            OrderNotFound: If the order doesn't exist or is already closed
            ExchangeAPIError: If there's an error canceling the order
        """
        try:
            await self._request('DELETE', 'order', 
                              params={
                                  'symbol': symbol.replace('/', '').upper(),
                                  'orderId': order_id
                              }, 
                              auth=True,
                              futures=True)
            return True
            
        except ExchangeAPIError as e:
            if 'Unknown order' in str(e):
                raise OrderNotFound(f"Order {order_id} not found")
            raise
    
    async def get_order(self, order_id: str, symbol: str) -> Order:
        """Get order details.
        
        Args:
            order_id: ID of the order to fetch
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            Order object with the order details
            
        Raises:
            OrderNotFound: If the order doesn't exist
            ExchangeAPIError: If there's an error fetching the order
        """
        try:
            response = await self._request('GET', 'order', 
                                         params={
                                             'symbol': symbol.replace('/', '').upper(),
                                             'orderId': order_id
                                         }, 
                                         auth=True,
                                         futures=True)
            return self._parse_order(response)
            
        except ExchangeAPIError as e:
            if 'Unknown order' in str(e):
                raise OrderNotFound(f"Order {order_id} not found")
            raise
    
    async def get_open_orders(self, symbol: str = None) -> List[Order]:
        """Get all open orders.
        
        Args:
            symbol: Optional trading pair symbol to filter by
            
        Returns:
            List of Order objects
        """
        params = {}
        if symbol:
            params['symbol'] = symbol.replace('/', '').upper()
            
        response = await self._request('GET', 'openOrders', params=params, auth=True, futures=True)
        return [self._parse_order(o) for o in response]
    
    async def get_orders(self, symbol: str, limit: int = 500) -> List[Order]:
        """Get all account orders (active, canceled, or filled).
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            limit: Maximum number of orders to return (default: 500, max: 1000)
            
        Returns:
            List of Order objects
        """
        params = {
            'symbol': symbol.replace('/', '').upper(),
            'limit': min(limit, 1000)
        }
        
        response = await self._request('GET', 'allOrders', params=params, auth=True, futures=True)
        return [self._parse_order(o) for o in response]
    
    async def get_my_trades(self, 
                          symbol: str, 
                          limit: int = 500,
                          from_id: int = None) -> List[Trade]:
        """Get trades for a specific account and symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            limit: Maximum number of trades to return (default: 500, max: 1000)
            from_id: Trade ID to fetch from (inclusive)
            
        Returns:
            List of Trade objects
        """
        params = {
            'symbol': symbol.replace('/', '').upper(),
            'limit': min(limit, 1000)
        }
        
        if from_id is not None:
            params['fromId'] = from_id
        
        trades = await self._request('GET', 'myTrades', params=params, auth=True, futures=True)
        
        return [
            Trade(
                id=str(trade['id']),
                order_id=str(trade['orderId']),
                symbol=symbol,
                side=TradeSide.BUY if trade['isBuyer'] else TradeSide.SELL,
                price=float(trade['price']),
                amount=float(trade['qty']),
                cost=float(trade['quoteQty']),
                timestamp=datetime.utcfromtimestamp(trade['time'] / 1000),
                fee={
                    'currency': trade['commissionAsset'],
                    'cost': float(trade['commission']),
                    'rate': 0.0  # Binance doesn't provide fee rate in this endpoint
                },
                taker_or_maker=Liquidity.TAKER if trade['isMaker'] else Liquidity.MAKER,
                info=trade
            )
            for trade in trades
        ]
    
    # Helper Methods
    
    def _parse_order(self, data: dict) -> Order:
        """Parse order data from Binance API response.
        
        Args:
            data: Raw order data from Binance API
            
        Returns:
            Order object
        """
        # Map Binance status to our OrderStatus
        status_map = {
            'NEW': OrderStatus.NEW,
            'PARTIALLY_FILLED': OrderStatus.PARTIALLY_FILLED,
            'FILLED': OrderStatus.FILLED,
            'CANCELED': OrderStatus.CANCELED,
            'REJECTED': OrderStatus.REJECTED,
            'EXPIRED': OrderStatus.EXPIRED,
            'PENDING_CANCEL': OrderStatus.PENDING_CANCEL,
            'NEW_INSURANCE': OrderStatus.NEW,
            'NEW_ADL': OrderStatus.NEW
        }
        
        # Map Binance order type to our OrderType
        type_map = {
            'MARKET': OrderType.MARKET,
            'LIMIT': OrderType.LIMIT,
            'STOP': OrderType.STOP,
            'STOP_LOSS': OrderType.STOP,
            'STOP_LOSS_LIMIT': OrderType.STOP_LIMIT,
            'TAKE_PROFIT': OrderType.TAKE_PROFIT,
            'TAKE_PROFIT_LIMIT': OrderType.TAKE_PROFIT_LIMIT,
            'TRAILING_STOP_MARKET': OrderType.TRAILING_STOP,
            'FOK': OrderType.FOK,
            'IOC': OrderType.IOC
        }
        
        # Parse order data
        order = Order(
            id=str(data['orderId']),
            client_order_id=data.get('clientOrderId'),
            symbol=data['symbol'],
            type=type_map.get(data['type'], OrderType.MARKET),
            side=OrderSide.BUY if data['side'] == 'BUY' else OrderSide.SELL,
            status=status_map.get(data['status'], OrderStatus.NEW),
            amount=float(data['origQty']),
            price=float(data['price']) if 'price' in data and data['price'] else None,
            stop_price=float(data['stopPrice']) if 'stopPrice' in data and data['stopPrice'] else None,
            average=float(data['avgPrice']) if 'avgPrice' in data and data['avgPrice'] else None,
            filled=float(data['executedQty']) if 'executedQty' in data else 0.0,
            remaining=float(data['origQty']) - float(data.get('executedQty', 0)),
            cost=float(data['cummulativeQuoteQty']) if 'cummulativeQuoteQty' in data else None,
            created_at=datetime.utcfromtimestamp(data['time'] / 1000) if 'time' in data else None,
            updated_at=datetime.utcfromtimestamp(data['updateTime'] / 1000) if 'updateTime' in data else None,
            info=data
        )
        
        # Add fees if available
        if 'fills' in data and data['fills']:
            total_fee = sum(float(f['commission']) for f in data['fills'] if 'commission' in f)
            if total_fee > 0:
                order.fee = {
                    'currency': data['fills'][0].get('commissionAsset', 'USDT'),
                    'cost': total_fee,
                    'rate': total_fee / float(data['cummulativeQuoteQty']) if 'cummulativeQuoteQty' in data and float(data['cummulativeQuoteQty']) > 0 else 0.0
                }
        
        return order
    
    def _format_amount(self, symbol: str, amount: float) -> str:
        """Format amount according to exchange's precision requirements.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            amount: Amount to format
            
        Returns:
            Formatted amount string
        """
        # This is a simplified version. In a real implementation, you would
        # need to get the precision for each symbol from the exchange info
        return f"{amount:.8f}".rstrip('0').rstrip('.')
    
    def _format_price(self, symbol: str, price: float) -> str:
        """Format price according to exchange's precision requirements.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            price: Price to format
            
        Returns:
            Formatted price string
        """
        # This is a simplified version. In a real implementation, you would
        # need to get the precision for each symbol from the exchange info
        return f"{price:.8f}".rstrip('0').rstrip('.')
    
    # WebSocket Subscription Methods
    
    async def watch_ticker(self, symbol: str) -> Ticker:
        """Watch the ticker for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            Ticker object with the latest data
            
        Note:
            This is a placeholder implementation. The actual implementation
            would use WebSocket to stream ticker updates.
        """
        raise NotImplementedError("WebSocket ticker streaming not implemented")
    
    async def watch_ohlcv(self, symbol: str, timeframe: str = '1h') -> OHLCV:
        """Watch OHLCV data for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe for the OHLCV data (e.g., '1m', '5m', '1h', '1d')
            
        Returns:
            OHLCV object with the latest candle data
            
        Note:
            This is a placeholder implementation. The actual implementation
            would use WebSocket to stream OHLCV updates.
        """
        raise NotImplementedError("WebSocket OHLCV streaming not implemented")
    
    async def watch_order_book(self, symbol: str) -> dict:
        """Watch the order book for a symbol.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            
        Returns:
            Dictionary containing order book data
            
        Note:
            This is a placeholder implementation. The actual implementation
            would use WebSocket to stream order book updates.
        """
        raise NotImplementedError("WebSocket order book streaming not implemented")
    
    async def watch_orders(self, symbol: str = None) -> Order:
        """Watch for order updates.
        
        Args:
            symbol: Optional trading pair symbol to filter by
            
        Returns:
            Order object with the updated order data
            
        Note:
            This is a placeholder implementation. The actual implementation
            would use WebSocket to stream order updates.
        """
        raise NotImplementedError("WebSocket order streaming not implemented")
