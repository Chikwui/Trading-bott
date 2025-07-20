"""
Coinbase exchange implementation.
"""
import asyncio
import time
import hmac
import hashlib
import base64
import json
import uuid
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

class CoinbaseExchange(BaseExchange):
    """Coinbase exchange implementation."""
    
    @property
    def name(self) -> str:
        return "coinbase"
    
    @property
    def base_url(self) -> str:
        if self.sandbox:
            return "https://api-public.sandbox.pro.coinbase.com"
        return "https://api.pro.coinbase.com"
    
    @property
    def ws_url(self) -> str:
        if self.sandbox:
            return "wss://ws-feed-public.sandbox.pro.coinbase.com"
        return "wss://ws-feed.pro.coinbase.com"
    
    def _generate_signature(self, method: str, path: str, body: str = '') -> Dict[str, str]:
        """Generate Coinbase Pro API signature."""
        timestamp = str(time.time())
        message = timestamp + method + path + (body or '')
        hmac_key = base64.b64decode(self.api_secret)
        signature = hmac.new(hmac_key, message.encode('utf-8'), hashlib.sha256)
        signature_b64 = base64.b64encode(signature.digest()).decode('utf-8')
        
        return {
            'CB-ACCESS-KEY': self.api_key,
            'CB-ACCESS-SIGN': signature_b64,
            'CB-ACCESS-TIMESTAMP': timestamp,
            'CB-ACCESS-PASSPHRASE': self.api_passphrase,
            'Content-Type': 'application/json'
        }
    
    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make an authenticated request to the Coinbase Pro API."""
        path = f"/{endpoint.lstrip('/')}"
        url = f"{self.base_url}{path}"
        
        # Prepare request data
        data = kwargs.get('data')
        body = json.dumps(data) if data else ''
        
        # Generate signature
        headers = self._generate_signature(method, path, body)
        
        # Add additional headers
        if 'headers' in kwargs:
            headers.update(kwargs['headers'])
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method, 
                    url, 
                    headers=headers,
                    json=data if data else None,
                    params=kwargs.get('params'),
                    timeout=30
                ) as response:
                    text = await response.text()
                    
                    if response.status == 429:
                        retry_after = int(response.headers.get('Retry-After', 1))
                        logger.warning(f"Rate limited. Retrying after {retry_after} seconds...")
                        await asyncio.sleep(retry_after)
                        return await self._request(method, endpoint, **kwargs)
                        
                    if not response.ok:
                        error_msg = f"API request failed with status {response.status}: {text}"
                        if response.status == 400:
                            error_data = json.loads(text) if text else {}
                            if 'message' in error_data:
                                error_msg = error_data['message']
                            raise ExchangeAPIError(error_msg)
                        elif response.status == 401:
                            raise ExchangeAPIError("Invalid API key/secret/passphrase")
                        elif response.status == 403:
                            raise ExchangeAPIError("Insufficient permissions")
                        elif response.status == 404:
                            raise OrderNotFound("Order not found")
                        else:
                            raise ExchangeAPIError(error_msg)
                    
                    return await response.json() if text else {}
                    
        except asyncio.TimeoutError:
            raise ExchangeConnectionError("Request to Coinbase Pro API timed out")
        except aiohttp.ClientError as e:
            raise ExchangeConnectionError(f"Connection error: {str(e)}")
    
    async def get_ticker(self, symbol: str) -> Ticker:
        """Get ticker for a symbol."""
        response = await self._request('GET', f'products/{symbol}/ticker')
        return Ticker(
            symbol=symbol,
            bid=Decimal(response.get('bid') or 0),
            ask=Decimal(response.get('ask') or 0),
            last=Decimal(response.get('price') or 0),
            volume=Decimal(response.get('volume') or 0),
            timestamp=datetime.utcnow()
        )
    
    async def get_ohlcv(
        self, 
        symbol: str, 
        timeframe: str = '1m', 
        since: Optional[datetime] = None, 
        limit: int = 1000
    ) -> List[OHLCV]:
        """Get OHLCV data."""
        granularity = self._timeframe_to_seconds(timeframe)
        end = datetime.utcnow()
        start = since or (end - timedelta(seconds=granularity * limit))
        
        params = {
            'start': start.isoformat(),
            'end': end.isoformat(),
            'granularity': granularity
        }
        
        response = await self._request('GET', f'products/{symbol}/candles', params=params)
        
        ohlcv_list = []
        for candle in response:
            ohlcv_list.append(OHLCV(
                timestamp=datetime.utcfromtimestamp(candle[0]),
                open=Decimal(str(candle[3])),
                high=Decimal(str(candle[2])),
                low=Decimal(str(candle[1])),
                close=Decimal(str(candle[4])),
                volume=Decimal(str(candle[5]))
            ))
        
        return ohlcv_list
    
    async def get_balance(self) -> Dict[str, Balance]:
        """Get account balances."""
        response = await self._request('GET', 'accounts')
        
        balances = {}
        for acc in response:
            currency = acc['currency']
            balance = Decimal(acc['balance'])
            available = Decimal(acc['available'])
            holds = Decimal(acc['hold'])
            
            balances[currency] = Balance(
                free=available,
                used=holds,
                total=balance
            )
        
        return balances
    
    async def create_order(
        self, 
        symbol: str, 
        order_type: OrderType, 
        side: OrderSide, 
        amount: Decimal,
        price: Optional[Decimal] = None,
        params: Optional[Dict] = None
    ) -> Order:
        """Create a new order."""
        if order_type == OrderType.MARKET and price is not None:
            raise ExchangeError("Market orders cannot have a price")
        if order_type != OrderType.MARKET and price is None:
            raise ExchangeError("Limit/stop orders require a price")
        
        data = {
            'product_id': symbol,
            'side': 'buy' if side == OrderSide.BUY else 'sell',
            'size': str(amount),
            'type': order_type.value.lower()
        }
        
        if order_type != OrderType.MARKET:
            data['price'] = str(price)
        
        if params:
            # Handle additional order parameters
            if 'time_in_force' in params:
                data['time_in_force'] = params['time_in_force']
            if 'post_only' in params:
                data['post_only'] = params['post_only']
            if 'client_oid' in params:
                data['client_oid'] = params['client_oid']
        
        response = await self._request('POST', 'orders', data=data)
        
        return self._parse_order(response)
    
    async def get_order(self, order_id: str, symbol: Optional[str] = None) -> Order:
        """Get order by ID."""
        try:
            response = await self._request('GET', f'orders/{order_id}')
            return self._parse_order(response)
        except OrderNotFound:
            # Try to find in closed orders if not found in open orders
            try:
                response = await self._request('GET', f'orders/{order_id}')
                return self._parse_order(response)
            except Exception:
                raise OrderNotFound(f"Order {order_id} not found")
    
    async def cancel_order(self, order_id: str, symbol: Optional[str] = None) -> bool:
        """Cancel an order."""
        try:
            await self._request('DELETE', f'orders/{order_id}')
            return True
        except OrderNotFound:
            return False
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all open orders."""
        endpoint = 'orders?status=open'
        if symbol:
            endpoint += f'&product_id={symbol}'
            
        response = await self._request('GET', endpoint)
        return [self._parse_order(o) for o in response]
    
    async def get_closed_orders(self, symbol: Optional[str] = None) -> List[Order]:
        """Get all closed orders."""
        endpoint = 'orders?status=done&limit=100'  # Limit to 100 most recent
        if symbol:
            endpoint += f'&product_id={symbol}'
            
        response = await self._request('GET', endpoint)
        return [self._parse_order(o) for o in response]
    
    def _parse_order(self, data: Dict) -> Order:
        """Parse order data from API response."""
        return Order(
            id=data['id'],
            client_order_id=data.get('client_oid', ''),
            symbol=data['product_id'],
            type=OrderType(data['type'].upper()),
            side=OrderSide.BUY if data['side'] == 'buy' else OrderSide.SELL,
            price=Decimal(data.get('price', '0')),
            amount=Decimal(data.get('size', '0')),
            filled=Decimal(data.get('filled_size', '0')),
            remaining=Decimal(data.get('size', '0')) - Decimal(data.get('filled_size', '0')),
            status=OrderStatus(data['status'].upper()),
            timestamp=parser.parse(data['created_at']),
            updated_at=parser.parse(data.get('done_at', data['created_at'])),
            fee=Decimal(data.get('fill_fees', '0')),
            trades=None,  # Will be populated separately if needed
            info=data
        )
    
    async def get_trades(self, symbol: str, since: Optional[datetime] = None, limit: int = 1000) -> List[Trade]:
        """Get recent trades for a symbol."""
        endpoint = f'products/{symbol}/trades'
        params = {}
        
        if since:
            params['after'] = int(since.timestamp() * 1000)
        
        response = await self._request('GET', endpoint, params=params)
        
        trades = []
        for trade in response[:limit]:
            trades.append(Trade(
                id=str(trade['trade_id']),
                order_id=None,  # Not available in public trades
                symbol=symbol,
                price=Decimal(trade['price']),
                amount=Decimal(trade['size']),
                side=TradeSide.BUY if trade['side'] == 'buy' else TradeSide.SELL,
                timestamp=parser.parse(trade['time']),
                fee=None,  # Not available in public trades
                fee_currency=None,
                info=trade
            ))
        
        return trades
    
    async def get_my_trades(
        self, 
        symbol: Optional[str] = None, 
        since: Optional[datetime] = None, 
        limit: int = 100
    ) -> List[Trade]:
        """Get your account's trades."""
        endpoint = 'fills'
        params = {}
        
        if symbol:
            params['product_id'] = symbol
        if since:
            params['start_date'] = since.isoformat()
        
        response = await self._request('GET', endpoint, params=params)
        
        trades = []
        for trade in response[:limit]:
            trades.append(Trade(
                id=str(trade['trade_id']),
                order_id=trade['order_id'],
                symbol=trade['product_id'],
                price=Decimal(trade['price']),
                amount=Decimal(trade['size']),
                side=TradeSide.BUY if trade['side'] == 'buy' else TradeSide.SELL,
                timestamp=parser.parse(trade['created_at']),
                fee=Decimal(trade.get('fee', '0')),
                fee_currency=trade.get('fee_currency', ''),
                info=trade
            ))
        
        return trades
    
    def _timeframe_to_seconds(self, timeframe: str) -> int:
        """Convert timeframe string to seconds."""
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        
        if unit == 'm':
            return value * 60
        elif unit == 'h':
            return value * 3600
        elif unit == 'd':
            return value * 86400
        elif unit == 'w':
            return value * 604800
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
