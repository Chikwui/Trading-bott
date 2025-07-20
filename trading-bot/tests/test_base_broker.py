"""
Tests for the base broker interface.
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal

from core.execution import (
    BaseBroker,
    Order,
    OrderType,
    OrderSide,
    OrderStatus,
    TimeInForce,
    Position,
    AccountInfo,
)


class TestBaseBroker:
    """Test cases for the BaseBroker abstract class."""
    
    @pytest.fixture
    def mock_broker(self):
        """Create a mock broker that implements the abstract methods."""
        class MockBroker(BaseBroker):
            def __init__(self):
                self.connected = False
                self.orders = {}
                self.positions = {}
                self.account_info = AccountInfo(
                    account_id="test-123",
                    balance=Decimal("100000"),
                    equity=Decimal("100000"),
                    available_funds=Decimal("100000"),
                    currency="USD",
                )
                
            async def connect(self):
                self.connected = True
                
            async def disconnect(self):
                self.connected = False
                
            async def get_account_info(self):
                return self.account_info
                
            async def get_positions(self):
                return self.positions
                
            async def get_position(self, symbol):
                return self.positions.get(symbol.upper())
                
            async def place_order(self, order):
                order_id = f"order-{len(self.orders)+1}"
                order.order_id = order_id
                order.status = OrderStatus.NEW
                self.orders[order_id] = order
                return order_id
                
            async def cancel_order(self, order_id):
                if order_id in self.orders:
                    self.orders[order_id].status = OrderStatus.CANCELED
                    return True
                return False
                
            async def get_order_status(self, order_id):
                if order_id in self.orders:
                    return self.orders[order_id].status
                return None
                
            async def get_orders(self, symbol=None):
                if symbol:
                    return [o for o in self.orders.values() 
                           if o.symbol == symbol.upper()]
                return list(self.orders.values())
                
            async def get_historical_data(self, symbol, interval, start_time=None, 
                                       end_time=None, limit=500):
                # Return mock data
                return [{"timestamp": datetime.utcnow(), "open": 100, "high": 101, 
                        "low": 99, "close": 100.5, "volume": 1000}]
                        
            async def get_order_book(self, symbol, depth=20):
                return {"bids": [[100, 1]], "asks": [[101, 1]]}
                
            async def get_ticker(self, symbol):
                return {"symbol": symbol, "last": 100.5, "volume": 1000}
                
            async def get_balances(self):
                return {"USD": {"free": 100000, "used": 0, "total": 100000}}
                
            async def get_leverage(self, symbol):
                return 1
                
            async def set_leverage(self, symbol, leverage):
                return True
                
        return MockBroker()
    
    @pytest.mark.asyncio
    async def test_connect_disconnect(self, mock_broker):
        """Test connection and disconnection."""
        assert not mock_broker.connected
        await mock_broker.connect()
        assert mock_broker.connected
        await mock_broker.disconnect()
        assert not mock_broker.connected
    
    @pytest.mark.asyncio
    async def test_place_order(self, mock_broker):
        """Test placing an order."""
        order = Order(
            symbol="BTC/USD",
            order_type=OrderType.MARKET,
            side=OrderSide.BUY,
            quantity=1.0,
        )
        
        order_id = await mock_broker.place_order(order)
        assert order_id.startswith("order-")
        assert order.status == OrderStatus.NEW
        
        status = await mock_broker.get_order_status(order_id)
        assert status == OrderStatus.NEW
        
        # Test getting all orders
        orders = await mock_broker.get_orders()
        assert len(orders) == 1
        assert orders[0].order_id == order_id
        
        # Test getting orders by symbol
        btc_orders = await mock_broker.get_orders("BTC/USD")
        assert len(btc_orders) == 1
        
        # Test cancelling the order
        result = await mock_broker.cancel_order(order_id)
        assert result is True
        
        status = await mock_broker.get_order_status(order_id)
        assert status == OrderStatus.CANCELED
    
    @pytest.mark.asyncio
    async def test_position_management(self, mock_broker):
        """Test position management."""
        # Initially no positions
        positions = await mock_broker.get_positions()
        assert len(positions) == 0
        
        # Add a position
        position = Position(
            symbol="BTC/USD",
            quantity=1.0,
            avg_price=50000.0,
            unrealized_pnl=1000.0,
        )
        mock_broker.positions["BTC/USD"] = position
        
        # Verify position retrieval
        positions = await mock_broker.get_positions()
        assert len(positions) == 1
        assert "BTC/USD" in positions
        
        single_position = await mock_broker.get_position("btc/usd")  # Test case insensitivity
        assert single_position.symbol == "BTC/USD"
        
        # Test non-existent position
        no_position = await mock_broker.get_position("NONEXISTENT")
        assert no_position is None
    
    @pytest.mark.asyncio
    async def test_account_info(self, mock_broker):
        """Test account information retrieval."""
        account = await mock_broker.get_account_info()
        assert account.account_id == "test-123"
        assert account.balance == Decimal("100000")
        assert account.equity == Decimal("100000")
        assert account.available_funds == Decimal("100000")
        assert account.currency == "USD"
    
    @pytest.mark.asyncio
    async def test_market_data(self, mock_broker):
        """Test market data retrieval."""
        # Test historical data
        candles = await mock_broker.get_historical_data(
            symbol="BTC/USD",
            interval="1h",
            limit=1
        )
        assert len(candles) == 1
        assert "timestamp" in candles[0]
        
        # Test order book
        order_book = await mock_broker.get_order_book("BTC/USD")
        assert "bids" in order_book
        assert "asks" in order_book
        
        # Test ticker
        ticker = await mock_broker.get_ticker("BTC/USD")
        assert ticker["symbol"] == "BTC/USD"
        
    @pytest.mark.asyncio
    async def test_leverage(self, mock_broker):
        """Test leverage management."""
        # Test getting leverage
        leverage = await mock_broker.get_leverage("BTC/USD")
        assert leverage == 1
        
        # Test setting leverage
        result = await mock_broker.set_leverage("BTC/USD", 10)
        assert result is True


if __name__ == "__main__":
    pytest.main(["-v", "test_base_broker.py"])
