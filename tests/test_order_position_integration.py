"""
Integration tests for OrderManager and PositionManager interaction.
"""
import asyncio
from decimal import Decimal
from datetime import datetime, timezone
import pytest

from unittest.mock import AsyncMock, MagicMock

from core.trading.order_manager import OrderManager
from core.trading.position_manager import PositionManager, PositionStatus
from core.trading.order_types import (
    Order, OrderSide, OrderStatus, OrderType, TimeInForce
)
from core.trading.advanced_orders import OCOOrder, BracketOrder

@pytest.fixture
def mock_exchange_adapter():
    """Create a mock exchange adapter."""
    adapter = AsyncMock()
    adapter.submit_order = AsyncMock(return_value={
        'order_id': 'mock_exchange_id',
        'status': 'FILLED',
        'filled_quantity': '1.0',
        'filled_price': '50000.0',
        'remaining_quantity': '0.0',
        'timestamp': datetime.now(timezone.utc).isoformat()
    })
    adapter.cancel_order = AsyncMock(return_value=True)
    return adapter

@pytest.fixture
def mock_order_book():
    """Create a mock order book."""
    return MagicMock()

@pytest.fixture
async def position_manager():
    """Create a position manager for testing."""
    return PositionManager(account_id="test_account")

@pytest.fixture
async def order_manager(mock_exchange_adapter, mock_order_book, position_manager):
    """Create an order manager with a position manager for testing."""
    manager = OrderManager(
        exchange_adapter=mock_exchange_adapter,
        order_book=mock_order_book,
        position_manager=position_manager,
        config={
            'default_time_in_force': TimeInForce.GTC,
            'max_retry_attempts': 3,
            'retry_delay': 0.1
        }
    )
    await manager.start()
    yield manager
    await manager.stop()

@pytest.mark.asyncio
async def test_market_order_creates_position(order_manager, position_manager):
    """Test that a market buy order creates a new position."""
    # Create a market buy order
    order = Order(
        order_id="test_order_1",
        client_order_id="client_1",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal('1.0'),
        price=Decimal('50000.0'),
        status=OrderStatus.NEW,
        timestamp=datetime.now(timezone.utc)
    )
    
    # Submit the order
    submitted_order = await order_manager.submit_order(order)
    assert submitted_order.status == OrderStatus.FILLED
    
    # Verify position was created
    positions = await position_manager.get_positions(symbol="BTC/USDT")
    assert len(positions) == 1
    
    position = positions[0]
    assert position.symbol == "BTC/USDT"
    assert position.side == OrderSide.BUY
    assert position.quantity == Decimal('1.0')
    assert position.entry_price == Decimal('50000.0')
    assert position.status == PositionStatus.OPEN

@pytest.mark.asyncio
async def test_position_updates_on_partial_fill(order_manager, position_manager):
    """Test that a position is updated correctly on partial order fills."""
    # Create a limit buy order
    order = Order(
        order_id="test_order_2",
        client_order_id="client_2",
        symbol="ETH/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal('10.0'),
        price=Decimal('3000.0'),
        status=OrderStatus.NEW,
        timestamp=datetime.now(timezone.utc)
    )
    
    # Mock a partial fill
    order_manager._exchange_adapter.submit_order = AsyncMock(return_value={
        'order_id': 'mock_exchange_id_2',
        'status': 'PARTIALLY_FILLED',
        'filled_quantity': '3.0',
        'filled_price': '3000.0',
        'remaining_quantity': '7.0',
        'timestamp': datetime.now(timezone.utc).isoformat()
    })
    
    # Submit the order
    submitted_order = await order_manager.submit_order(order)
    assert submitted_order.status == OrderStatus.PARTIALLY_FILLED
    
    # Verify position was created with partial fill
    positions = await position_manager.get_positions(symbol="ETH/USDT")
    assert len(positions) == 1
    assert positions[0].quantity == Decimal('3.0')
    
    # Mock another partial fill
    order_manager._exchange_adapter.submit_order = AsyncMock(return_value={
        'order_id': 'mock_exchange_id_2',
        'status': 'FILLED',
        'filled_quantity': '10.0',
        'filled_price': '3000.0',
        'remaining_quantity': '0.0',
        'timestamp': datetime.now(timezone.utc).isoformat()
    })
    
    # Update the order (simulating exchange update)
    submitted_order.filled_quantity = Decimal('10.0')
    submitted_order.remaining_quantity = Decimal('0.0')
    submitted_order.status = OrderStatus.FILLED
    await order_manager._process_order_fill(submitted_order)
    
    # Verify position was updated
    positions = await position_manager.get_positions(symbol="ETH/USDT")
    assert len(positions) == 1
    assert positions[0].quantity == Decimal('10.0')

@pytest.mark.asyncio
async def test_position_closes_on_opposite_side_order(order_manager, position_manager):
    """Test that a position is closed when an opposite side order is filled."""
    # First, create a long position
    buy_order = Order(
        order_id="test_buy_order",
        client_order_id="client_buy",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal('1.0'),
        price=Decimal('50000.0'),
        status=OrderStatus.FILLED,
        timestamp=datetime.now(timezone.utc)
    )
    
    # Submit the buy order
    await order_manager.submit_order(buy_order)
    
    # Verify position is open
    positions = await position_manager.get_positions(symbol="BTC/USDT")
    assert len(positions) == 1
    assert positions[0].status == PositionStatus.OPEN
    
    # Create a sell order to close the position
    sell_order = Order(
        order_id="test_sell_order",
        client_order_id="client_sell",
        symbol="BTC/USDT",
        side=OrderSide.SELL,
        order_type=OrderType.MARKET,
        quantity=Decimal('1.0'),
        price=Decimal('52000.0'),
        status=OrderStatus.FILLED,
        timestamp=datetime.now(timezone.utc)
    )
    
    # Submit the sell order
    await order_manager.submit_order(sell_order)
    
    # Verify position is closed
    positions = await position_manager.get_positions(symbol="BTC/USDT", status=PositionStatus.OPEN)
    assert len(positions) == 0
    
    # Check closed positions
    closed_positions = await position_manager.get_positions(symbol="BTC/USDT", status=PositionStatus.CLOSED)
    assert len(closed_positions) == 1
    assert closed_positions[0].status == PositionStatus.CLOSED
    assert closed_positions[0].exit_price == Decimal('52000.0')
    assert closed_positions[0].realized_pnl == (52000 - 50000) * Decimal('1.0')

@pytest.mark.asyncio
async def test_oco_order_position_management(order_manager, position_manager):
    """Test position management with OCO (One-Cancels-Other) orders."""
    # Create an OCO order (bracket order with take-profit and stop-loss)
    entry_order = Order(
        order_id="entry_order",
        client_order_id="client_entry",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal('1.0'),
        price=Decimal('50000.0'),
        status=OrderStatus.FILLED,
        timestamp=datetime.now(timezone.utc)
    )
    
    take_profit = Order(
        order_id="take_profit",
        client_order_id="client_tp",
        symbol="BTC/USDT",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=Decimal('1.0'),
        price=Decimal('55000.0'),  # 10% profit target
        status=OrderStatus.NEW,
        timestamp=datetime.now(timezone.utc)
    )
    
    stop_loss = Order(
        order_id="stop_loss",
        client_order_id="client_sl",
        symbol="BTC/USDT",
        side=OrderSide.SELL,
        order_type=OrderType.STOP,
        quantity=Decimal('1.0'),
        price=Decimal('45000.0'),  # 10% stop loss
        stop_price=Decimal('45000.0'),
        status=OrderStatus.NEW,
        timestamp=datetime.now(timezone.utc)
    )
    
    # Create a bracket order
    bracket = BracketOrder(
        entry_order=entry_order,
        take_profit_order=take_profit,
        stop_loss_order=stop_loss
    )
    
    # Submit the bracket order
    await order_manager.advanced_order_manager.add_order(bracket)
    
    # Process the entry fill
    await order_manager._process_order_fill(entry_order)
    
    # Verify position was created
    positions = await position_manager.get_positions(symbol="BTC/USDT", status=PositionStatus.OPEN)
    assert len(positions) == 1
    
    # Simulate take-profit being hit
    take_profit.status = OrderStatus.FILLED
    take_profit.filled_quantity = Decimal('1.0')
    take_profit.filled_price = Decimal('55000.0')
    
    await order_manager._process_order_fill(take_profit)
    
    # Verify position is closed with profit
    positions = await position_manager.get_positions(symbol="BTC/USDT", status=PositionStatus.OPEN)
    assert len(positions) == 0
    
    closed_positions = await position_manager.get_positions(symbol="BTC/USDT", status=PositionStatus.CLOSED)
    assert len(closed_positions) == 1
    assert closed_positions[0].exit_price == Decimal('55000.0')
    assert closed_positions[0].realized_pnl == (55000 - 50000) * Decimal('1.0')
    
    # Verify stop-loss was cancelled
    stop_loss_order = await order_manager.get_order("stop_loss")
    assert stop_loss_order.status == OrderStatus.CANCELED

@pytest.mark.asyncio
async def test_net_position_calculation(order_manager, position_manager):
    """Test net position calculation with multiple orders."""
    # Create multiple orders in both directions
    orders = [
        # Buy 2.0 BTC
        Order(
            order_id=f"buy_1_{i}",
            client_order_id=f"client_buy_{i}",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal('1.0'),
            price=Decimal('50000.0') + Decimal(str(i * 1000)),
            status=OrderStatus.FILLED,
            timestamp=datetime.now(timezone.utc)
        ) for i in range(2)
    ]
    
    # Sell 1.5 BTC
    orders.extend([
        Order(
            order_id=f"sell_{i}",
            client_order_id=f"client_sell_{i}",
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal('0.75'),
            price=Decimal('51000.0') + Decimal(str(i * 1000)),
            status=OrderStatus.FILLED,
            timestamp=datetime.now(timezone.utc)
        ) for i in range(2)
    ])
    
    # Submit all orders
    for order in orders:
        await order_manager.submit_order(order)
    
    # Verify net position is 0.5 BTC long (2.0 - 1.5)
    net_position = await position_manager.get_net_position("BTC/USDT")
    assert net_position == Decimal('0.5')
    
    # Verify we have one open position
    open_positions = await position_manager.get_positions(symbol="BTC/USDT", status=PositionStatus.OPEN)
    assert len(open_positions) == 1
    assert open_positions[0].quantity == Decimal('0.5')
    
    # Verify we have closed position(s) as well
    closed_positions = await position_manager.get_positions(symbol="BTC/USDT", status=PositionStatus.CLOSED)
    assert len(closed_positions) > 0
    
    # Calculate total realized P&L
    total_pnl = sum(p.realized_pnl for p in closed_positions)
    assert total_pnl > 0  # Should be positive since we sold at higher prices
