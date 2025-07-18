"""
Tests for the position management system.
"""
import asyncio
from decimal import Decimal
from datetime import datetime, timezone, timedelta
import pytest

from core.trading.position_manager import Position, PositionManager, PositionStatus
from core.trading.order_types import Order, OrderSide, OrderStatus, OrderType

@pytest.fixture
def sample_order():
    """Create a sample order for testing."""
    return Order(
        order_id=f"order_{datetime.now().timestamp()}",
        client_order_id=f"client_{datetime.now().timestamp()}",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal('1.0'),
        price=Decimal('50000.0'),
        status=OrderStatus.NEW,
        timestamp=datetime.now(timezone.utc)
    )

@pytest.fixture
def position_manager():
    """Create a position manager for testing."""
    return PositionManager(account_id="test_account")

class TestPosition:
    """Test the Position class."""
    
    def test_position_creation(self):
        """Test position creation and basic properties."""
        position = Position(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=Decimal('1.0'),
            entry_price=Decimal('50000.0')
        )
        
        assert position.symbol == "BTC/USDT"
        assert position.side == OrderSide.BUY
        assert position.quantity == Decimal('1.0')
        assert position.entry_price == Decimal('50000.0')
        assert position.status == PositionStatus.OPEN
        assert position.leverage == 1
        assert position.realized_pnl == Decimal('0')
        assert position.unrealized_pnl == Decimal('0')
        
    def test_position_with_leverage(self):
        """Test position with leverage."""
        position = Position(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=Decimal('1.0'),
            entry_price=Decimal('50000.0'),
            leverage=5
        )
        
        assert position.leverage == 5
        
    def test_update_market_price_long(self):
        """Test updating market price for a long position."""
        position = Position(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=Decimal('1.0'),
            entry_price=Decimal('50000.0'),
            leverage=2
        )
        
        # Price increases by 10%
        position.update_market_price(Decimal('55000.0'))
        
        # P&L should be (55000 - 50000) * 1.0 * 2 = 10000
        assert position.unrealized_pnl == Decimal('10000.0')
        
    def test_update_market_price_short(self):
        """Test updating market price for a short position."""
        position = Position(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=Decimal('1.0'),
            entry_price=Decimal('50000.0'),
            leverage=3
        )
        
        # Price decreases by 10%
        position.update_market_price(Decimal('45000.0'))
        
        # P&L should be (50000 - 45000) * 1.0 * 3 = 15000
        assert position.unrealized_pnl == Decimal('15000.0')
        
    def test_add_order_increase_position(self):
        """Test adding an order that increases the position size."""
        position = Position(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=Decimal('1.0'),
            entry_price=Decimal('50000.0')
        )
        
        # Add another buy order
        order = Order(
            order_id="order_2",
            client_order_id="client_2",
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal('0.5'),
            filled_quantity=Decimal('0.5'),
            filled_price=Decimal('51000.0'),
            status=OrderStatus.FILLED,
            timestamp=datetime.now(timezone.utc)
        )
        
        position.add_order(order)
        
        # New average price should be (1.0 * 50000 + 0.5 * 51000) / 1.5 = 50333.33...
        assert position.quantity == Decimal('1.5')
        assert position.avg_entry_price == Decimal('50333.33333333333333333333333')
        
    def test_add_order_decrease_position(self):
        """Test adding an order that decreases the position size."""
        position = Position(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=Decimal('1.0'),
            entry_price=Decimal('50000.0')
        )
        
        # Add a sell order (partial close)
        order = Order(
            order_id="order_2",
            client_order_id="client_2",
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal('0.3'),
            filled_quantity=Decimal('0.3'),
            filled_price=Decimal('52000.0'),
            status=OrderStatus.FILLED,
            timestamp=datetime.now(timezone.utc)
        )
        
        position.add_order(order)
        
        # Position size should decrease, realized P&L should increase
        assert position.quantity == Decimal('0.7')
        assert position.closed_quantity == Decimal('0.3')
        assert position.realized_pnl == (52000 - 50000) * Decimal('0.3')
        
    def test_add_order_close_position(self):
        """Test adding an order that fully closes the position."""
        position = Position(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=Decimal('1.0'),
            entry_price=Decimal('50000.0')
        )
        
        # Add a sell order (full close)
        order = Order(
            order_id="order_2",
            client_order_id="client_2",
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal('1.0'),
            filled_quantity=Decimal('1.0'),
            filled_price=Decimal('52000.0'),
            status=OrderStatus.FILLED,
            timestamp=datetime.now(timezone.utc)
        )
        
        position.add_order(order)
        
        # Position should be closed
        assert position.quantity == Decimal('0')
        assert position.status == PositionStatus.CLOSED
        assert position.realized_pnl == (52000 - 50000) * Decimal('1.0')
        assert position.exit_price == Decimal('52000.0')
        assert position.exit_time is not None

class TestPositionManager:
    """Test the PositionManager class."""
    
    @pytest.mark.asyncio
    async def test_open_position(self, position_manager):
        """Test opening a new position."""
        position = await position_manager.open_position(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=Decimal('1.0'),
            price=Decimal('50000.0'),
            strategy_id="test_strategy"
        )
        
        assert position is not None
        assert position.symbol == "BTC/USDT"
        assert position.side == OrderSide.BUY
        assert position.quantity == Decimal('1.0')
        assert position.entry_price == Decimal('50000.0')
        assert position.strategy_id == "test_strategy"
        
        # Verify position can be retrieved
        retrieved = await position_manager.get_position(position.position_id)
        assert retrieved is not None
        assert retrieved.position_id == position.position_id
    
    @pytest.mark.asyncio
    async def test_update_position_with_order(self, position_manager, sample_order):
        """Test updating a position with an order."""
        # Create a position
        position = await position_manager.open_position(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=Decimal('1.0'),
            price=Decimal('50000.0')
        )
        
        # Update with a filled order
        sample_order.status = OrderStatus.FILLED
        sample_order.filled_quantity = Decimal('0.5')
        sample_order.filled_price = Decimal('51000.0')
        
        updated = await position_manager.update_position(
            position_id=position.position_id,
            order=sample_order
        )
        
        assert updated is not None
        assert updated.quantity == Decimal('1.5')  # 1.0 + 0.5
        
    @pytest.mark.asyncio
    async def test_close_position(self, position_manager):
        """Test closing a position."""
        # Create a position
        position = await position_manager.open_position(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=Decimal('1.0'),
            price=Decimal('50000.0')
        )
        
        # Close the position
        success = await position_manager.close_position(
            position_id=position.position_id,
            price=Decimal('52000.0')
        )
        
        assert success is True
        
        # Verify position is closed
        closed_position = await position_manager.get_position(position.position_id)
        assert closed_position.status == PositionStatus.CLOSED
        assert closed_position.exit_price == Decimal('52000.0')
        assert closed_position.realized_pnl == (52000 - 50000) * Decimal('1.0')
    
    @pytest.mark.asyncio
    async def test_get_positions_filtering(self, position_manager):
        """Test filtering positions by symbol and status."""
        # Create test positions
        btc_long = await position_manager.open_position(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=Decimal('1.0'),
            price=Decimal('50000.0'),
            strategy_id="strategy1"
        )
        
        eth_short = await position_manager.open_position(
            symbol="ETH/USDT",
            side=OrderSide.SELL,
            quantity=Decimal('10.0'),
            price=Decimal('3000.0'),
            strategy_id="strategy2"
        )
        
        # Close one position
        await position_manager.close_position(
            position_id=btc_long.position_id,
            price=Decimal('51000.0')
        )
        
        # Test filters
        open_positions = await position_manager.get_positions(status=PositionStatus.OPEN)
        assert len(open_positions) == 1
        assert open_positions[0].symbol == "ETH/USDT"
        
        btc_positions = await position_manager.get_positions(symbol="BTC/USDT")
        assert len(btc_positions) == 1
        assert btc_positions[0].position_id == btc_long.position_id
        
        strategy2_positions = await position_manager.get_positions(strategy_id="strategy2")
        assert len(strategy2_positions) == 1
        assert strategy2_positions[0].symbol == "ETH/USDT"
    
    @pytest.mark.asyncio
    async def test_get_net_position(self, position_manager):
        """Test getting net position for a symbol."""
        # Create long and short positions for the same symbol
        await position_manager.open_position(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=Decimal('2.0'),
            price=Decimal('50000.0')
        )
        
        await position_manager.open_position(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=Decimal('1.5'),
            price=Decimal('51000.0')
        )
        
        # Net position should be 2.0 - 1.5 = 0.5 (long)
        net_position = await position_manager.get_net_position("BTC/USDT")
        assert net_position == Decimal('0.5')
        
        # Add another short position
        await position_manager.open_position(
            symbol="BTC/USDT",
            side=OrderSide.SELL,
            quantity=Decimal('1.0'),
            price=Decimal('51500.0')
        )
        
        # Net position should now be 2.0 - 1.5 - 1.0 = -0.5 (short)
        net_position = await position_manager.get_net_position("BTC/USDT")
        assert net_position == Decimal('-0.5')
    
    @pytest.mark.asyncio
    async def test_get_total_pnl(self, position_manager):
        """Test getting total P&L across all positions."""
        # Create some positions
        btc_long = await position_manager.open_position(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            quantity=Decimal('1.0'),
            price=Decimal('50000.0')
        )
        
        eth_short = await position_manager.open_position(
            symbol="ETH/USDT",
            side=OrderSide.SELL,
            quantity=Decimal('10.0'),
            price=Decimal('3000.0')
        )
        
        # Update market prices
        await position_manager.update_position(
            position_id=btc_long.position_id,
            mark_price=Decimal('51000.0')  # +1000 unrealized
        )
        
        await position_manager.update_position(
            position_id=eth_short.position_id,
            mark_price=Decimal('2900.0')  # +1000 unrealized (10 * 100)
        )
        
        # Close one position with profit
        await position_manager.close_position(
            position_id=btc_long.position_id,
            price=Decimal('52000.0')  # +2000 realized
        )
        
        # Check total P&L
        pnl = await position_manager.get_total_pnl()
        assert pnl['realized'] == Decimal('2000')  # From closed BTC position
        assert pnl['unrealized'] == Decimal('1000')  # From open ETH position
        assert pnl['total'] == Decimal('3000')  # Sum of realized and unrealized
