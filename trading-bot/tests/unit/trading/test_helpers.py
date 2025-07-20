"""Test helpers for trading system unit tests."""
import asyncio
from decimal import Decimal
from typing import Any, Dict, Optional
from contextlib import contextmanager, nullcontext
from unittest.mock import AsyncMock, MagicMock, patch

# Import prometheus_client first to set up test environment
from prometheus_client import REGISTRY, CollectorRegistry, generate_latest

# Now import the rest of the modules
from core.trading.order import Order, OrderSide, OrderStatus, OrderType
from core.trading.oco_order import OCOOrderConfig, OCOOrderStatus
from core.trading.order_manager import OrderManager


@contextmanager
def prometheus_test_registry():
    """Context manager that provides a clean Prometheus registry for testing.
    
    Creates a new registry for each test and restores the original registry afterward.
    Also ensures the metrics registry is properly reset.
    """
    # Import here to avoid circular imports
    from core.metrics import metrics
    
    # Save the original registry
    original_registry = REGISTRY
    
    # Create a new registry for testing
    test_registry = CollectorRegistry(auto_describe=True)
    
    # Replace the default registry with our test registry
    import prometheus_client.registry
    prometheus_client.registry.REGISTRY = test_registry
    
    # Create a new metrics instance with the test registry
    original_metrics = metrics
    test_metrics = type(metrics)(registry=test_registry)
    
    # Replace the global metrics instance
    import sys
    import importlib
    
    # Save the original module state
    original_metrics_module = sys.modules['core.metrics']
    
    # Create a new module for testing
    test_metrics_module = importlib.import_module('core.metrics')
    test_metrics_module.metrics = test_metrics
    sys.modules['core.metrics'] = test_metrics_module
    
    try:
        # Re-import the module to update the metrics reference
        importlib.reload(test_metrics_module)
        
        # Yield the test registry and metrics instance
        yield test_registry
    finally:
        # Restore the original registry
        prometheus_client.registry.REGISTRY = original_registry
        
        # Clear any collectors from the test registry
        for collector in list(test_registry._collector_to_names.keys()):
            test_registry.unregister(collector)
            
        # Restore the original metrics module
        sys.modules['core.metrics'] = original_metrics_module
        importlib.reload(original_metrics_module)


class MockExchangeAdapter:
    """Mock exchange adapter for testing."""
    
    def __init__(self):
        self.submit_order = AsyncMock()
        self.cancel_order = AsyncMock()
        self.get_order = AsyncMock()
        self.connected = True

    async def connect(self):
        """Mock connect method."""
        self.connected = True

    async def disconnect(self):
        """Mock disconnect method."""
        self.connected = False


def create_test_order(
    order_id: str = "test_order_1",
    symbol: str = "BTC/USDT",
    side: OrderSide = OrderSide.BUY,
    order_type: OrderType = OrderType.LIMIT,
    price: Decimal = Decimal("50000.00"),
    quantity: Decimal = Decimal("1.0"),
    status: OrderStatus = OrderStatus.NEW,
    **kwargs
) -> Order:
    """Create a test order with default values."""
    return Order(
        order_id=order_id,
        symbol=symbol,
        side=side,
        order_type=order_type,
        price=price,
        quantity=quantity,
        status=status,
        **kwargs
    )


def create_test_oco_config(
    symbol: str = "BTC/USDT",
    quantity: Decimal = Decimal("1.0"),
    entry_price: Decimal = Decimal("50000.00"),
    stop_loss_price: Decimal = Decimal("49000.00"),
    take_profit_price: Decimal = Decimal("51000.00"),
    **kwargs
) -> OCOOrderConfig:
    """Create a test OCO order configuration."""
    return OCOOrderConfig(
        symbol=symbol,
        quantity=quantity,
        limit_price=entry_price,
        stop_price=stop_loss_price,
        stop_limit_price=take_profit_price,
        **kwargs
    )


async def create_test_order_manager(
    exchange_adapter: Optional[Any] = None,
    max_orders_per_second: int = 100,
    use_test_registry: bool = True,
    enable_metrics: bool = True
) -> OrderManager:
    """Create and start a test OrderManager instance.
    
    Args:
        exchange_adapter: Optional exchange adapter to use. If None, a MockExchangeAdapter will be created.
        max_orders_per_second: Maximum orders per second for rate limiting.
        use_test_registry: If True, uses a clean Prometheus registry for testing.
        enable_metrics: If False, disables all metrics collection.
            
    Returns:
        OrderManager: Configured and started OrderManager instance.
    """
    if exchange_adapter is None:
        exchange_adapter = MockExchangeAdapter()
    
    # Create a context manager for the test registry if needed
    registry_context = prometheus_test_registry() if use_test_registry else nullcontext()
    
    with registry_context:
        manager = OrderManager(
            exchange_adapter=exchange_adapter,
            max_orders_per_second=max_orders_per_second,
            enable_metrics=enable_metrics
        )
        await manager.start()
        return manager
