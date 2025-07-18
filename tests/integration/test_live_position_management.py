"""
Live integration tests for position management with real exchange adapters.

These tests require valid API credentials and should be run with caution
as they may execute real trades on connected exchange accounts.

Set environment variables:
- EXCHANGE_API_KEY: Your exchange API key
- EXCHANGE_SECRET: Your exchange API secret
- EXCHANGE_PASSPHRASE: (Optional) Your exchange API passphrase if required

WARNING: These tests may execute real trades. Use a test/demo account
with minimal funds and carefully review the test configuration.
"""
import os
import asyncio
import pytest
from decimal import Decimal
from datetime import datetime, timezone
from typing import Dict, Any, Optional

# Import exchange adapters - add more as needed
from core.exchanges.binance import BinanceExchange
from core.trading.position_manager import PositionManager, PositionStatus
from core.trading.order_manager import OrderManager
from core.trading.order_types import Order, OrderSide, OrderStatus, OrderType, TimeInForce

# Skip these tests by default to prevent accidental execution
pytestmark = pytest.mark.integration

class TestLivePositionManagement:
    """Live integration tests for position management with real exchange adapters."""
    
    @pytest.fixture(autouse=True)
    async def setup(self, test_config):
        """Set up test environment with real exchange adapter."""
        # Initialize exchange adapter with test credentials
        self.exchange = BinanceExchange(
            api_key=test_config.EXCHANGE_API_KEY,
            api_secret=test_config.EXCHANGE_SECRET,
            passphrase=test_config.EXCHANGE_PASSPHRASE,
            sandbox=test_config.PAPER_TRADING
        )
        
        # Initialize position and order managers
        self.position_manager = PositionManager(account_id=test_config.TEST_ACCOUNT_ID)
        self.order_manager = OrderManager(
            exchange_adapter=self.exchange,
            order_book=MagicMock(),  # Mock order book for now
            position_manager=self.position_manager,
            config={
                'default_time_in_force': TimeInForce.GTC,
                'max_retry_attempts': test_config.MAX_RETRIES,
                'retry_delay': 0.1
            }
        )
        
        # Start the order manager
        await self.order_manager.start()
        
        # Test setup complete
        yield
        
        # Cleanup: Cancel all open orders and close positions
        await self.cleanup()
        await self.order_manager.stop()
    
    async def cleanup(self):
        """Clean up test environment by canceling all orders and closing positions."""
        # Cancel all open orders
        try:
            orders = await self.exchange.get_open_orders()
            for order in orders:
                await self.exchange.cancel_order(order['order_id'], order['symbol'])
        except Exception as e:
            print(f"Warning: Error during cleanup - {str(e)}")
        
        # Close all open positions
        try:
            positions = await self.position_manager.get_positions(status=PositionStatus.OPEN)
            for position in positions:
                # Create a market order to close the position
                close_side = OrderSide.SELL if position.side == OrderSide.BUY else OrderSide.BUY
                close_order = Order(
                    order_id=f"close_{position.position_id}",
                    client_order_id=f"close_{position.position_id}",
                    symbol=position.symbol,
                    side=close_side,
                    order_type=OrderType.MARKET,
                    quantity=position.quantity,
                    status=OrderStatus.NEW,
                    timestamp=datetime.now(timezone.utc)
                )
                await self.order_manager.submit_order(close_order)
        except Exception as e:
            print(f"Warning: Error closing positions - {str(e)}")
    
    @pytest.mark.asyncio
    async def test_live_market_order_creates_position(self, test_config):
        """Test that a live market buy order creates a position."""
        # Create a market buy order with small quantity
        order = Order(
            order_id=f"test_buy_{int(datetime.now(timezone.utc).timestamp())}",
            client_order_id=f"client_buy_{int(datetime.now(timezone.utc).timestamp())}",
            symbol=test_config.TEST_SYMBOL,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=test_config.TEST_QUANTITY,
            status=OrderStatus.NEW,
            timestamp=datetime.now(timezone.utc)
        )
        
        # Submit the order
        submitted_order = await self.order_manager.submit_order(order)
        assert submitted_order.status == OrderStatus.FILLED
        
        # Verify position was created
        positions = await self.position_manager.get_positions(symbol=test_config.TEST_SYMBOL)
        assert len(positions) == 1
        
        position = positions[0]
        assert position.symbol == test_config.TEST_SYMBOL
        assert position.side == OrderSide.BUY
        assert position.quantity == test_config.TEST_QUANTITY
        assert position.status == PositionStatus.OPEN
        
        return position
    
    @pytest.mark.asyncio
    async def test_live_position_updates_on_multiple_fills(self, test_config):
        """Test position updates with multiple order fills."""
        # First, create a small position
        position = await self.test_live_market_order_creates_position(test_config)
        initial_quantity = position.quantity
        
        # Create another buy order to increase the position
        order = Order(
            order_id=f"test_add_{int(datetime.now(timezone.utc).timestamp())}",
            client_order_id=f"client_add_{int(datetime.now(timezone.utc).timestamp())}",
            symbol=test_config.TEST_SYMBOL,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=test_config.TEST_QUANTITY,
            status=OrderStatus.NEW,
            timestamp=datetime.now(timezone.utc)
        )
        
        # Submit the additional order
        await self.order_manager.submit_order(order)
        
        # Verify position was updated
        positions = await self.position_manager.get_positions(symbol=test_config.TEST_SYMBOL)
        assert len(positions) == 1
        assert positions[0].quantity == initial_quantity + test_config.TEST_QUANTITY
    
    @pytest.mark.asyncio
    async def test_live_position_closing(self, test_config):
        """Test closing a position with an opposite side order."""
        # First, create a position
        position = await self.test_live_market_order_creates_position(test_config)
        position_id = position.position_id
        
        # Create a sell order to close the position
        order = Order(
            order_id=f"test_sell_{int(datetime.now(timezone.utc).timestamp())}",
            client_order_id=f"client_sell_{int(datetime.now(timezone.utc).timestamp())}",
            symbol=test_config.TEST_SYMBOL,
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=test_config.TEST_QUANTITY,
            status=OrderStatus.NEW,
            timestamp=datetime.now(timezone.utc)
        )
        
        # Submit the sell order
        await self.order_manager.submit_order(order)
        
        # Verify position is closed
        positions = await self.position_manager.get_positions(
            symbol=test_config.TEST_SYMBOL, 
            status=PositionStatus.OPEN
        )
        assert len(positions) == 0
        
        # Check closed positions
        closed_positions = await self.position_manager.get_positions(
            symbol=test_config.TEST_SYMBOL,
            status=PositionStatus.CLOSED
        )
        assert len(closed_positions) > 0
        assert any(p.position_id == position_id for p in closed_positions)
    
    @pytest.mark.asyncio
    async def test_live_limit_order_partial_fill(self, test_config):
        """Test position updates with partial order fills."""
        # Create a limit order with a price far from current market to ensure partial fill
        ticker = await self.exchange.get_ticker(test_config.TEST_SYMBOL)
        current_price = Decimal(ticker['last'])
        
        # Use a price that's unlikely to be hit immediately
        limit_price = current_price * Decimal('0.9')  # 10% below current price
        
        order = Order(
            order_id=f"test_limit_{int(datetime.now(timezone.utc).timestamp())}",
            client_order_id=f"client_limit_{int(datetime.now(timezone.utc).timestamp())}",
            symbol=test_config.TEST_SYMBOL,
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=test_config.TEST_QUANTITY * 2,  # Larger quantity for partial fill
            price=limit_price,
            time_in_force=TimeInForce.GTC,
            status=OrderStatus.NEW,
            timestamp=datetime.now(timezone.utc)
        )
        
        # Submit the order
        submitted_order = await self.order_manager.submit_order(order)
        
        # Wait a moment for the order to be processed
        await asyncio.sleep(2)
        
        # Check order status - it should be NEW or PARTIALLY_FILLED
        updated_order = await self.order_manager.get_order(submitted_order.order_id)
        assert updated_order.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]
        
        # If partially filled, verify position was updated
        if updated_order.status == OrderStatus.PARTIALLY_FILLED:
            positions = await self.position_manager.get_positions(
                symbol=test_config.TEST_SYMBOL
            )
            assert len(positions) == 1
            assert positions[0].quantity > 0
            assert positions[0].quantity < test_config.TEST_QUANTITY * 2
        
        # Cleanup: Cancel the order if it's still open
        if updated_order.status in [OrderStatus.NEW, OrderStatus.PARTIALLY_FILLED]:
            await self.order_manager.cancel_order(updated_order.order_id)
    
    @pytest.mark.asyncio
    async def test_live_position_pnl_calculation(self, test_config):
        """Test that P&L is calculated correctly for open positions."""
        # Create a position
        position = await self.test_live_market_order_creates_position(test_config)
        
        # Get current market price
        ticker = await self.exchange.get_ticker(test_config.TEST_SYMBOL)
        current_price = Decimal(ticker['last'])
        
        # Update position with current market price
        await self.position_manager.update_position(
            position_id=position.position_id,
            mark_price=current_price
        )
        
        # Get updated position
        positions = await self.position_manager.get_positions(
            symbol=test_config.TEST_SYMBOL,
            status=PositionStatus.OPEN
        )
        assert len(positions) == 1
        
        # Verify P&L calculation
        position = positions[0]
        if position.side == OrderSide.BUY:
            expected_pnl = (current_price - position.entry_price) * position.quantity
        else:
            expected_pnl = (position.entry_price - current_price) * position.quantity
        
        # Allow for small floating point differences
        assert abs(position.unrealized_pnl - expected_pnl) < Decimal('0.000001')
