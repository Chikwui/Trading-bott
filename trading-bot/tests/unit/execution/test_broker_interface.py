"""
Unit tests for the Broker interface.

These tests verify that all broker implementations conform to the expected interface.
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from core.execution.broker import Broker, BrokerType, BrokerError, OrderError, MarketDataError
from core.models import Order, OrderType, OrderSide, OrderStatus, TimeInForce

# Test broker implementation that implements all abstract methods
class TestBroker(Broker):
    """Test broker implementation for testing the abstract base class."""
    
    async def connect(self) -> bool:
        self.connected = True
        return True
        
    async def disconnect(self) -> None:
        self.connected = False
        
    async def get_account_info(self) -> dict:
        return {"balance": 10000.0, "equity": 10000.0, "margin_used": 0.0}
        
    async def place_order(self, order: Order) -> str:
        order.order_id = "test_order_123"
        return order.order_id
        
    async def cancel_order(self, order_id: str) -> bool:
        return True
        
    async def get_order_status(self, order_id: str) -> dict:
        return {"order_id": order_id, "status": "FILLED"}
        
    async def get_positions(self, symbol: str = None) -> list:
        return [{"symbol": "EURUSD", "quantity": 1000, "entry_price": 1.1000}]
        
    async def get_historical_data(self, symbol: str, timeframe: str, 
                                start: datetime = None, end: datetime = None, 
                                limit: int = 1000) -> list:
        return [{"timestamp": datetime.utcnow(), "open": 1.1, "high": 1.101, 
                "low": 1.099, "close": 1.1005, "volume": 1000}]
                
    async def get_current_price(self, symbol: str) -> float:
        return 1.1005
        
    async def get_order_book(self, symbol: str, depth: int = 10) -> dict:
        return {
            "bids": [[1.1000, 100]],
            "asks": [[1.1005, 100]],
            "timestamp": datetime.utcnow().isoformat()
        }

@pytest.fixture
def test_broker():
    """Fixture that provides a test broker instance."""
    return TestBroker(BrokerType.LIVE)

@pytest.mark.asyncio
async def test_broker_initialization(test_broker):
    """Test broker initialization."""
    assert test_broker is not None
    assert test_broker.broker_type == BrokerType.LIVE
    assert not test_broker.connected
    assert not test_broker.initialized

@pytest.mark.asyncio
async def test_connect_disconnect(test_broker):
    """Test connection and disconnection."""
    # Test connect
    result = await test_broker.connect()
    assert result is True
    assert test_broker.connected
    
    # Test disconnect
    await test_broker.disconnect()
    assert not test_broker.connected

@pytest.mark.asyncio
async def test_get_account_info(test_broker):
    """Test getting account information."""
    await test_broker.connect()
    account_info = await test_broker.get_account_info()
    assert isinstance(account_info, dict)
    assert "balance" in account_info
    assert "equity" in account_info
    assert "margin_used" in account_info

@pytest.mark.asyncio
async def test_place_order(test_broker):
    """Test placing an order."""
    await test_broker.connect()
    
    # Create a test order
    order = Order(
        symbol="EURUSD",
        order_type=OrderType.MARKET,
        side=OrderSide.BUY,
        quantity=1000,
        time_in_force=TimeInForce.DAY
    )
    
    # Place the order
    order_id = await test_broker.place_order(order)
    assert order_id is not None
    assert isinstance(order_id, str)
    assert order_id == "test_order_123"

@pytest.mark.asyncio
async def test_cancel_order(test_broker):
    """Test canceling an order."""
    await test_broker.connect()
    
    # Test cancel order
    result = await test_broker.cancel_order("test_order_123")
    assert result is True

@pytest.mark.asyncio
async def test_get_order_status(test_broker):
    """Test getting order status."""
    await test_broker.connect()
    
    # Test get order status
    status = await test_broker.get_order_status("test_order_123")
    assert status is not None
    assert isinstance(status, dict)
    assert "order_id" in status
    assert "status" in status

@pytest.mark.asyncio
async def test_get_positions(test_broker):
    """Test getting positions."""
    await test_broker.connect()
    
    # Test get all positions
    positions = await test_broker.get_positions()
    assert isinstance(positions, list)
    if positions:  # If positions exist
        assert "symbol" in positions[0]
        assert "quantity" in positions[0]
    
    # Test get positions for specific symbol
    positions = await test_broker.get_positions("EURUSD")
    assert isinstance(positions, list)

@pytest.mark.asyncio
async def test_get_historical_data(test_broker):
    """Test getting historical market data."""
    await test_broker.connect()
    
    # Test with required parameters only
    data = await test_broker.get_historical_data("EURUSD", "1h")
    assert isinstance(data, list)
    if data:  # If data exists
        assert "timestamp" in data[0]
        assert "open" in data[0]
        assert "high" in data[0]
        assert "low" in data[0]
        assert "close" in data[0]
        assert "volume" in data[0]
    
    # Test with all parameters
    end = datetime.utcnow()
    start = end - timedelta(days=7)
    data = await test_broker.get_historical_data(
        symbol="EURUSD",
        timeframe="1d",
        start=start,
        end=end,
        limit=100
    )
    assert isinstance(data, list)

@pytest.mark.asyncio
async def test_get_current_price(test_broker):
    """Test getting the current market price."""
    await test_broker.connect()
    
    price = await test_broker.get_current_price("EURUSD")
    assert isinstance(price, float)
    assert price > 0

@pytest.mark.asyncio
async def test_get_order_book(test_broker):
    """Test getting the order book."""
    await test_broker.connect()
    
    order_book = await test_broker.get_order_book("EURUSD")
    assert isinstance(order_book, dict)
    assert "bids" in order_book
    assert "asks" in order_book
    assert "timestamp" in order_book
    assert isinstance(order_book["bids"], list)
    assert isinstance(order_book["asks"], list)

@pytest.mark.asyncio
async def test_broker_info(test_broker):
    """Test getting broker information."""
    info = test_broker.get_broker_info()
    assert isinstance(info, dict)
    assert "broker_type" in info
    assert "connected" in info
    assert "last_update" in info
    assert info["broker_type"] == "LIVE"

# Test error conditions
@pytest.mark.asyncio
async def test_not_connected_error(test_broker):
    """Test that methods raise BrokerError when not connected."""
    # Test when not connected
    test_broker.connected = False
    
    with pytest.raises(BrokerError):
        await test_broker.get_account_info()
    
    with pytest.raises(OrderError):
        await test_broker.place_order(Order("EURUSD", OrderType.MARKET, OrderSide.BUY, 1000))
    
    with pytest.raises(OrderError):
        await test_broker.cancel_order("test_order_123")
    
    with pytest.raises(OrderError):
        await test_broker.get_order_status("test_order_123")
    
    with pytest.raises(BrokerError):
        await test_broker.get_positions()
    
    with pytest.raises(MarketDataError):
        await test_broker.get_historical_data("EURUSD", "1h")
    
    with pytest.raises(MarketDataError):
        await test_broker.get_current_price("EURUSD")
    
    with pytest.raises(MarketDataError):
        await test_broker.get_order_book("EURUSD")

# Test abstract methods
class IncompleteBroker(Broker):
    """Incomplete broker implementation for testing abstract methods."""
    pass

def test_abstract_methods():
    """Test that abstract methods raise NotImplementedError."""
    broker = IncompleteBroker(BrokerType.LIVE)
    
    with pytest.raises(NotImplementedError):
        asyncio.run(broker.connect())
    
    with pytest.raises(NotImplementedError):
        asyncio.run(broker.disconnect())
    
    with pytest.raises(NotImplementedError):
        asyncio.run(broker.get_account_info())
    
    with pytest.raises(NotImplementedError):
        asyncio.run(broker.place_order(Order("EURUSD", OrderType.MARKET, OrderSide.BUY, 1000)))
    
    with pytest.raises(NotImplementedError):
        asyncio.run(broker.cancel_order("test_order_123"))
    
    with pytest.raises(NotImplementedError):
        asyncio.run(broker.get_order_status("test_order_123"))
    
    with pytest.raises(NotImplementedError):
        asyncio.run(broker.get_positions())
    
    with pytest.raises(NotImplementedError):
        asyncio.run(broker.get_historical_data("EURUSD", "1h"))
    
    with pytest.raises(NotImplementedError):
        asyncio.run(broker.get_current_price("EURUSD"))
    
    with pytest.raises(NotImplementedError):
        asyncio.run(broker.get_order_book("EURUSD"))
