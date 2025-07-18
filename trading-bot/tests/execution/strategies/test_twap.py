"""
Unit tests for TWAP (Time-Weighted Average Price) execution strategy.
"""
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
import pandas as pd
import numpy as np

from core.execution.strategies.twap import TWAPExecutionClient, TWAPParameters
from core.execution.base import ExecutionParameters, ExecutionResult, ExecutionState
from core.trading.order import Order, OrderSide, OrderType, OrderStatus, TimeInForce
from core.market.data import TickerData, BarData
from core.risk.manager import RiskManager
from core.ml.model_registry import ModelRegistry, ModelType
from core.utils.connection_pool import ConnectionPool

# Test fixtures
@pytest.fixture
def mock_market_data():
    """Create a mock MarketDataService."""
    mock = AsyncMock()
    
    # Mock ticker data
    ticker = TickerData(
        symbol='BTC/USDT',
        bid=Decimal('50000.00'),
        ask=Decimal('50001.00'),
        last=Decimal('50000.50'),
        volume=Decimal('1000.0'),
        timestamp=datetime.utcnow()
    )
    mock.get_ticker.return_value = ticker
    
    # Mock order execution
    async def mock_execute_order(order):
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.avg_fill_price = Decimal('50000.50')
        return order
    
    mock.execute_order.side_effect = mock_execute_order
    return mock

@pytest.fixture
def mock_risk_manager():
    """Create a mock RiskManager."""
    mock = MagicMock(spec=RiskManager)
    mock.check_order_risk.return_value = True
    return mock

@pytest.fixture
def mock_model_registry():
    """Create a mock ModelRegistry."""
    mock = MagicMock(spec=ModelRegistry)
    
    # Mock fill probability model
    mock_fill_prob = MagicMock()
    mock_fill_prob.predict.return_value = 0.9  # 90% fill probability
    
    mock.get_model.side_effect = lambda model_type, version: {
        ModelType.FILL_PROBABILITY: mock_fill_prob,
        ModelType.MARKET_IMPACT: None
    }.get(model_type, None)
    
    return mock

@pytest.fixture
def default_twap_params():
    """Default TWAP parameters for testing."""
    return TWAPParameters(
        interval_seconds=60,  # 1 minute for faster tests
        max_child_orders=5,
        use_limit_orders=True,
        limit_order_tolerance_bps=5  # 0.05%
    )

@pytest.fixture
async def twap_client(mock_market_data, mock_risk_manager, mock_model_registry, default_twap_params):
    """Create a TWAPExecutionClient instance for testing."""
    client = TWAPExecutionClient(
        client_id='test_client',
        market_data=mock_market_data,
        risk_manager=mock_risk_manager,
        model_registry=mock_model_registry,
        twap_params=default_twap_params
    )
    
    # Add a small sleep to prevent rate limiting in tests
    client.twap_params.min_order_interval_ms = 10
    
    return client

# Test cases
class TestTWAPExecution:
    """Test cases for TWAP execution strategy."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, twap_client):
        """Test TWAP client initialization."""
        assert twap_client.twap_params.interval_seconds == 60
        assert twap_client.twap_params.max_child_orders == 5
        assert twap_client.twap_params.use_limit_orders is True
        
    @pytest.mark.asyncio
    async def test_calculate_order_schedule(self, twap_client):
        """Test order schedule calculation."""
        order = Order(
            symbol='BTC/USDT',
            side=OrderSide.BUY,
            quantity=Decimal('1.0'),
            order_type=OrderType.MARKET
        )
        
        params = ExecutionParameters(
            mode=ExecutionState.LIVE,
            max_slippage_bps=10,
            max_position_pct=0.1,
            max_order_value=100000
        )
        
        schedule = twap_client._calculate_order_schedule(order, params)
        
        # Should create 5 orders of ~0.2 BTC each
        assert len(schedule) == 5
        total_quantity = sum(float(entry['quantity']) for entry in schedule)
        assert abs(total_quantity - 1.0) < 0.0001  # Allow for floating point errors
        
        # Check order times are properly spaced
        for i in range(1, len(schedule)):
            time_diff = (schedule[i]['time'] - schedule[i-1]['time']).total_seconds()
            assert time_diff > 0  # Should be increasing
    
    @pytest.mark.asyncio
    async def test_adaptive_intervals(self, twap_client):
        """Test adaptive interval calculation."""
        # Enable adaptive intervals
        twap_client.twap_params.use_adaptive_intervals = True
        
        # Test beginning of execution (should be faster)
        interval = twap_client._get_adaptive_interval(0, 10)  # First of 10 orders
        assert interval < twap_client.twap_params.interval_seconds  # Should be faster
        
        # Test middle of execution (should be slower)
        interval = twap_client._get_adaptive_interval(5, 10)  # Middle order
        assert interval > twap_client.twap_params.interval_seconds  # Should be slower
        
        # Test end of execution (should be faster)
        interval = twap_client._get_adaptive_interval(9, 10)  # Last order
        assert interval < twap_client.twap_params.interval_seconds  # Should be faster
    
    @pytest.mark.asyncio
    async def test_order_creation(self, twap_client):
        """Test child order creation."""
        parent_order = Order(
            symbol='BTC/USDT',
            side=OrderSide.BUY,
            quantity=Decimal('1.0'),
            order_type=OrderType.MARKET
        )
        
        params = ExecutionParameters(
            mode=ExecutionState.LIVE,
            max_slippage_bps=10,
            max_position_pct=0.1,
            max_order_value=100000
        )
        
        # Test with limit orders
        twap_client.twap_params.use_limit_orders = True
        child_order = twap_client._create_child_order(parent_order, Decimal('0.2'), params)
        
        assert child_order.symbol == 'BTC/USDT'
        assert child_order.side == OrderSide.BUY
        assert child_order.quantity == Decimal('0.2')
        assert child_order.order_type == OrderType.LIMIT
        assert child_order.price is not None  # Should have a limit price
        assert child_order.parent_order_id == parent_order.id
        
        # Test with market orders
        twap_client.twap_params.use_limit_orders = False
        child_order = twap_client._create_child_order(parent_order, Decimal('0.2'), params)
        assert child_order.order_type == OrderType.MARKET
        assert child_order.price is None  # No limit price for market orders
    
    @pytest.mark.asyncio
    async def test_market_conditions_check(self, twap_client):
        """Test market conditions validation."""
        # Create a test order
        order = Order(
            symbol='BTC/USDT',
            side=OrderSide.BUY,
            quantity=Decimal('0.1'),
            order_type=OrderType.LIMIT,
            price=Decimal('50000.00')
        )
        
        # Test with good market conditions
        assert await twap_client._check_market_conditions(order) is True
        
        # Test with wide spread
        mock_ticker = TickerData(
            symbol='BTC/USDT',
            bid=Decimal('50000.00'),
            ask=Decimal('51000.00'),  # Very wide spread
            last=Decimal('50500.00'),
            volume=Decimal('1000.0'),
            timestamp=datetime.utcnow()
        )
        twap_client.market_data.get_ticker.return_value = mock_ticker
        assert await twap_client._check_market_conditions(order) is False
        
        # Test with low fill probability
        twap_client._fill_probability_model.predict.return_value = 0.5  # 50% fill probability
        assert await twap_client._check_market_conditions(order) is False
    
    @pytest.mark.asyncio
    async def test_full_execution_flow(self, twap_client):
        """Test the complete TWAP execution flow."""
        # Create a parent order
        parent_order = Order(
            symbol='BTC/USDT',
            side=OrderSide.BUY,
            quantity=Decimal('1.0'),
            order_type=OrderType.MARKET
        )
        
        # Set execution parameters
        params = ExecutionParameters(
            mode=ExecutionState.LIVE,
            max_slippage_bps=10,
            max_position_pct=0.1,
            max_order_value=100000
        )
        
        # Create execution result
        result = ExecutionResult(
            order_id=parent_order.id,
            status=OrderStatus.NEW,
            symbol=parent_order.symbol,
            side=parent_order.side,
            order_type=parent_order.order_type,
            quantity=parent_order.quantity,
            filled_quantity=Decimal('0'),
            remaining_quantity=parent_order.quantity,
            avg_fill_price=Decimal('0'),
            created_at=datetime.utcnow()
        )
        
        # Execute the strategy
        with patch('asyncio.sleep', new_callable=AsyncMock) as mock_sleep:
            # Speed up the test by not actually sleeping
            mock_sleep.return_value = None
            
            # Execute the strategy
            result = await twap_client._execute_strategy(parent_order, params, result)
        
        # Verify the results
        assert result.status == OrderStatus.FILLED
        assert float(result.filled_quantity) == pytest.approx(1.0, 0.0001)
        assert result.avg_fill_price > Decimal('0')
        
        # Check that child orders were created
        assert len(twap_client._child_orders) == 5  # Should be 5 child orders
        
        # Verify execution metrics
        assert 'twap_price' in result.metadata
        assert 'execution_time_seconds' in result.metadata
        assert 'twap_slippage_bps' in result.metadata
        
        # Check that all child orders were filled
        for child_order in twap_client._child_orders:
            assert child_order.status == OrderStatus.FILLED
    
    @pytest.mark.asyncio
    async def test_anti_gaming_measures(self, twap_client):
        """Test anti-gaming rate limiting."""
        twap_client.twap_params.enable_anti_gaming = True
        twap_client.twap_params.min_order_interval_ms = 100  # 100ms minimum interval
        
        # First order should pass
        start_time = datetime.utcnow()
        await twap_client._apply_anti_gaming_measures('BTC/USDT')
        
        # Second order should be delayed
        await twap_client._apply_anti_gaming_measures('BTC/USDT')
        end_time = datetime.utcnow()
        
        # Should have waited at least 100ms
        elapsed_ms = (end_time - start_time).total_seconds() * 1000
        assert elapsed_ms >= 90  # Allow for some timing variance (90ms of 100ms)
    
    @pytest.mark.asyncio
    async def test_order_size_randomization(self, twap_client):
        """Test order size randomization."""
        twap_client.twap_params.randomize_order_sizes = True
        twap_client.twap_params.order_size_randomization = 0.2  # +/- 20%
        
        order = Order(
            symbol='BTC/USDT',
            side=OrderSide.BUY,
            quantity=Decimal('1.0'),
            order_type=OrderType.MARKET
        )
        
        params = ExecutionParameters()
        schedule = twap_client._calculate_order_schedule(order, params)
        
        # Get all order sizes (except the last one which takes the remaining)
        order_sizes = [float(entry['quantity']) for entry in schedule[:-1]]
        
        # Check that order sizes are within expected range
        expected_size = 0.2  # 1.0 / 5 orders
        min_size = expected_size * 0.8  # -20%
        max_size = expected_size * 1.2  # +20%
        
        for size in order_sizes:
            assert min_size <= size <= max_size
    
    @pytest.mark.asyncio
    async def test_market_impact_adjustment(self, twap_client):
        """Test market impact adjustment."""
        # Enable market impact adjustment
        twap_client.twap_params.adjust_for_market_impact = True
        
        # Create a mock market impact model
        mock_impact_model = MagicMock()
        mock_impact_model.predict.return_value = 0.5  # 0.5% market impact
        twap_client._market_impact_model = mock_impact_model
        
        # Test with a large order that would trigger market impact adjustment
        order = Order(
            symbol='BTC/USDT',
            side=OrderSide.BUY,
            quantity=Decimal('100.0'),  # Large order
            order_type=OrderType.MARKET
        )
        
        params = ExecutionParameters()
        schedule = twap_client._calculate_order_schedule(order, params)
        
        # Should have created more orders to reduce market impact
        assert len(schedule) > 5  # More than the default 5 orders
        
        # Check that market impact model was called
        mock_impact_model.predict.assert_called()
    
    @pytest.mark.asyncio
    async def test_execution_with_connection_pool(self, twap_client):
        """Test execution using a connection pool."""
        # Create a mock connection pool
        mock_connection = AsyncMock()
        mock_connection_pool = MagicMock()
        mock_connection_pool.acquire.return_value.__aenter__.return_value = mock_connection
        
        # Configure the client to use the mock connection pool
        twap_client._connection_pool = mock_connection_pool
        
        # Create a test order
        order = Order(
            symbol='BTC/USDT',
            side=OrderSide.BUY,
            quantity=Decimal('0.1'),
            order_type=OrderType.LIMIT,
            price=Decimal('50000.00')
        )
        
        # Execute the order
        params = ExecutionParameters()
        await twap_client._execute_child_order(order, params)
        
        # Verify that the connection pool was used
        mock_connection_pool.acquire.assert_called_once()
        mock_connection.execute_order.assert_awaited_once()
