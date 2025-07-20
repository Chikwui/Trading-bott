"""Unit tests for the OrderManager class."""
import asyncio
import sys
from decimal import Decimal
from pathlib import Path
from unittest import IsolatedAsyncioTestCase, mock

# Add the project root to the Python path
project_root = str(Path(__file__).parents[3])
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import test helpers first to set up the test environment
from tests.unit.trading.test_helpers import (
    MockExchangeAdapter,
    create_test_order,
    create_test_oco_config,
    create_test_order_manager,
    prometheus_test_registry,
)

# Now import the rest of the modules
from core.trading.order import Order, OrderSide, OrderStatus, OrderType
from core.trading.oco_order import OCOOrderStatus


class TestOrderManager(IsolatedAsyncioTestCase):
    """Test suite for OrderManager functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up class-level test fixtures."""
        # This ensures the Prometheus registry is clean before any tests run
        cls.registry_context = prometheus_test_registry()
        cls.registry_context.__enter__()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up class-level test fixtures."""
        cls.registry_context.__exit__(None, None, None)

    async def asyncSetUp(self):
        """Set up test fixtures."""
        self.exchange = MockExchangeAdapter()
        # Create a new manager with a clean registry for each test
        self.manager = await create_test_order_manager(self.exchange, use_test_registry=False)
        
    async def asyncTearDown(self):
        """Clean up test fixtures."""
        await self.manager.stop()
        
    async def test_submit_order_success(self):
        """Test successful order submission."""
        order = await self.manager.submit_order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000.00")
        )
        
        self.assertEqual(order.status, OrderStatus.NEW)
        self.assertIn(order.order_id, self.manager.orders)
        self.exchange.submit_order.assert_awaited_once()
        
    async def test_submit_order_rate_limiting(self):
        """Test order submission with rate limiting."""
        # Create a manager with very low rate limit
        manager = await create_test_order_manager(max_orders_per_second=1)
        
        try:
            # First order should succeed
            order1 = await manager.submit_order(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("1.0"),
                price=Decimal("50000.00")
            )
            
            # Second order should be rate limited
            with self.assertRaises(RuntimeError, msg="Rate limit exceeded"):
                await manager.submit_order(
                    symbol="ETH/USDT",
                    side=OrderSide.BUY,
                    order_type=OrderType.LIMIT,
                    quantity=Decimal("10.0"),
                    price=Decimal("3000.00")
                )
                
        finally:
            await manager.stop()
            
    async def test_cancel_order_success(self):
        """Test successful order cancellation."""
        # Submit an order
        order = await self.manager.submit_order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000.00")
        )
        
        # Cancel the order
        result = await self.manager.cancel_order(order.order_id)
        self.assertTrue(result)
        self.exchange.cancel_order.assert_awaited_once_with(order.order_id)
        
    async def test_cancel_nonexistent_order(self):
        """Test cancelling a non-existent order."""
        result = await self.manager.cancel_order("nonexistent_order")
        self.assertFalse(result)
        
    async def test_get_order(self):
        """Test retrieving an order by ID."""
        order = await self.manager.submit_order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("1.0"),
            price=Decimal("50000.00")
        )
        
        retrieved = await self.manager.get_order(order.order_id)
        self.assertEqual(retrieved.order_id, order.order_id)
        
    async def test_list_orders_filtering(self):
        """Test filtering orders by symbol and status."""
        # Create test orders
        orders = [
            await self.manager.submit_order(
                symbol="BTC/USDT",
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                quantity=Decimal("1.0"),
                price=Decimal("50000.00")
            ),
            await self.manager.submit_order(
                symbol="ETH/USDT",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal("10.0")
            )
        ]
        
        # Filter by symbol
        btc_orders = await self.manager.list_orders(symbol="BTC/USDT")
        self.assertEqual(len(btc_orders), 1)
        self.assertEqual(btc_orders[0].symbol, "BTC/USDT")
        
        # Filter by order type
        market_orders = await self.manager.list_orders(order_type=OrderType.MARKET)
        self.assertEqual(len(market_orders), 1)
        self.assertEqual(market_orders[0].order_type, OrderType.MARKET)
