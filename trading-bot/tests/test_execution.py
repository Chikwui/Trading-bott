"""
Test suite for the execution module.

This module contains unit tests for the execution strategies and services.
"""
import asyncio
import pytest
from datetime import datetime, timedelta
from decimal import Decimal
from unittest.mock import AsyncMock, MagicMock, patch

from core.execution.base import (
    ExecutionClient, 
    ExecutionParameters, 
    ExecutionResult,
    ExecutionState
)
from core.execution.strategies.vwap import VWAPExecutionClient, VWAPParameters
from core.execution.strategies.iceberg import IcebergExecutionClient, IcebergParameters
from core.execution.service import ExecutionService, ExecutionStrategy, ExecutionConfig
from core.market.data import MarketDataService
from core.ml.model_registry import ModelRegistry
from core.risk.manager import RiskManager
from core.trading.order import Order, OrderSide, OrderType, OrderStatus


@pytest.fixture
def mock_market_data():
    """Fixture for a mock MarketDataService."""
    market_data = AsyncMock(spec=MarketDataService)
    market_data.get_symbol_data.return_value = {
        'bid': 100.0,
        'ask': 100.1,
        'last': 100.05,
        'volume': 1000,
        'bids': [[99.9, 10], [99.8, 15]],
        'asks': [[100.1, 12], [100.2, 18]],
        'volume_5m': 500,
        'tick_size': '0.01',
        'lot_size': '0.0001'
    }
    market_data.get_historical_bars.return_value = pd.DataFrame({
        'open': [100.0, 100.1, 100.2],
        'high': [100.5, 100.6, 100.7],
        'low': [99.5, 99.6, 99.7],
        'close': [100.1, 100.2, 100.3],
        'volume': [1000, 1200, 800],
        'time_of_day': [
            datetime.now().time(),
            (datetime.now() - timedelta(minutes=5)).time(),
            (datetime.now() - timedelta(minutes=10)).time()
        ]
    })
    market_data.get_arrival_price.return_value = 100.0
    market_data.is_market_open.return_value = True
    return market_data


@pytest.fixture
def mock_risk_manager():
    """Fixture for a mock RiskManager."""
    risk_manager = AsyncMock(spec=RiskManager)
    risk_manager.check_order.return_value = MagicMock(passed=True, reason="OK")
    return risk_manager


@pytest.fixture
def mock_model_registry():
    """Fixture for a mock ModelRegistry."""
    model_registry = AsyncMock(spec=ModelRegistry)
    model_registry.get_latest_model.return_value = MagicMock(
        predict=AsyncMock(return_value={'prediction': 0.5})
    )
    return model_registry


@pytest.fixture
def test_order():
    """Fixture for a test order."""
    return Order(
        order_id="test_order_123",
        client_order_id="client_123",
        symbol="BTC/USD",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("1.0"),
        price=Decimal("100.0"),
        time_in_force="GTC"
    )


@pytest.mark.asyncio
async def test_base_execution_client(mock_market_data, mock_risk_manager, test_order):
    """Test the base ExecutionClient functionality."""
    # Create a concrete subclass of ExecutionClient for testing
    class TestExecutionClient(ExecutionClient):
        async def _execute_strategy(self, order, params, result):
            return result.update(
                status=ExecutionState.COMPLETED,
                filled_quantity=order.quantity,
                avg_fill_price=order.price
            )
    
    # Initialize the client
    client = TestExecutionClient(
        client_id="test_client",
        market_data=mock_market_data,
        risk_manager=mock_risk_manager
    )
    
    # Test execute_order
    execution_id = await client.execute_order(test_order)
    assert execution_id is not None
    
    # Test get_execution_status
    result = await client.get_execution_status(execution_id)
    assert result is not None
    assert result.status == ExecutionState.COMPLETED
    assert result.filled_quantity == test_order.quantity
    assert result.avg_fill_price == test_order.price
    
    # Test cancel_execution
    success = await client.cancel_execution(execution_id)
    assert success is True
    
    # Test close
    await client.close()


@pytest.mark.asyncio
async def test_vwap_execution(mock_market_data, mock_risk_manager, test_order):
    """Test VWAP execution strategy."""
    # Initialize VWAP client
    vwap_params = VWAPParameters(
        interval_seconds=60,
        max_child_orders=5,
        volume_participation=0.1,
        use_limit_orders=True
    )
    
    client = VWAPExecutionClient(
        client_id="test_vwap",
        market_data=mock_market_data,
        risk_manager=mock_risk_manager,
        vwap_params=vwap_params
    )
    
    # Mock order execution
    with patch('core.execution.strategies.vwap.VWAPExecutionClient._schedule_child_orders') as mock_schedule:
        mock_schedule.return_value = [{
            'order_id': 'child_1',
            'status': 'FILLED',
            'filled_quantity': Decimal('0.2'),
            'avg_fill_price': Decimal('100.0'),
            'fees': Decimal('0.001')
        }]
        
        # Execute order
        execution_id = await client.execute_order(test_order)
        
        # Let the event loop process the execution
        await asyncio.sleep(0.1)
        
        # Check results
        result = await client.get_execution_status(execution_id)
        assert result is not None
        assert result.status == ExecutionState.COMPLETED
        assert result.filled_quantity == Decimal('0.2')
        assert result.avg_fill_price == Decimal('100.0')
    
    # Clean up
    await client.close()


@pytest.mark.asyncio
async def test_iceberg_execution(mock_market_data, mock_risk_manager, test_order):
    """Test Iceberg execution strategy."""
    # Initialize Iceberg client
    iceberg_params = IcebergParameters(
        max_visible_pct_adv=0.1,
        min_visible_qty=Decimal('0.1'),
        max_visible_qty=Decimal('0.5'),
        refresh_interval=1,
        max_duration=30
    )
    
    client = IcebergExecutionClient(
        client_id="test_iceberg",
        market_data=mock_market_data,
        risk_manager=mock_risk_manager,
        iceberg_params=iceberg_params
    )
    
    # Mock order placement
    with patch('core.execution.strategies.iceberg.IcebergExecutionClient._place_iceberg_order') as mock_place_order:
        mock_place_order.return_value = None
        
        # Execute order
        execution_id = await client.execute_order(test_order)
        
        # Let the refresh loop run once
        await asyncio.sleep(1.5)
        
        # Check that place_order was called
        assert mock_place_order.called
    
    # Test cancellation
    success = await client.cancel_execution(execution_id)
    assert success is True
    
    # Clean up
    await client.close()


@pytest.mark.asyncio
async def test_execution_service(mock_market_data, mock_risk_manager, test_order):
    """Test the high-level ExecutionService."""
    # Initialize execution service
    config = ExecutionConfig(
        default_strategy=ExecutionStrategy.VWAP,
        vwap_params=VWAPParameters(
            interval_seconds=60,
            max_child_orders=5,
            volume_participation=0.1
        )
    )
    
    service = ExecutionService(
        client_id="test_service",
        market_data=mock_market_data,
        risk_manager=mock_risk_manager,
        config=config
    )
    
    # Mock VWAP client
    mock_vwap_client = AsyncMock(spec=VWAPExecutionClient)
    mock_vwap_client.execute_order.return_value = "test_execution_123"
    mock_vwap_client.get_execution_status.return_value = ExecutionResult(
        execution_id="test_execution_123",
        client_order_id=test_order.client_order_id,
        symbol=test_order.symbol,
        side=test_order.side,
        quantity=test_order.quantity,
        filled_quantity=test_order.quantity,
        avg_fill_price=test_order.price,
        status=ExecutionState.COMPLETED
    )
    
    # Patch the client creation
    with patch('core.execution.service.VWAPExecutionClient', return_value=mock_vwap_client):
        # Execute order with default strategy (VWAP)
        execution_id = await service.execute_order(test_order)
        assert execution_id is not None
        
        # Check execution status
        result = await service.get_execution_status(execution_id)
        assert result is not None
        assert result.status == ExecutionState.COMPLETED
        assert result.filled_quantity == test_order.quantity
        
        # Test cancellation
        success = await service.cancel_execution(execution_id)
        assert success is True
    
    # Clean up
    await service.close()


@pytest.mark.asyncio
async def test_execution_with_ml(mock_market_data, mock_risk_manager, mock_model_registry, test_order):
    """Test execution with ML model integration."""
    # Initialize execution service with ML
    config = ExecutionConfig(
        default_strategy=ExecutionStrategy.VWAP,
        enable_ml=True
    )
    
    service = ExecutionService(
        client_id="test_ml",
        market_data=mock_market_data,
        risk_manager=mock_risk_manager,
        model_registry=mock_model_registry,
        config=config
    )
    
    # Mock VWAP client
    mock_vwap_client = AsyncMock(spec=VWAPExecutionClient)
    mock_vwap_client.execute_order.return_value = "ml_execution_123"
    mock_vwap_client.get_execution_status.return_value = ExecutionResult(
        execution_id="ml_execution_123",
        client_order_id=test_order.client_order_id,
        symbol=test_order.symbol,
        side=test_order.side,
        quantity=test_order.quantity,
        filled_quantity=test_order.quantity,
        avg_fill_price=test_order.price,
        status=ExecutionState.COMPLETED,
        metadata={'ml_features': {'prediction': 0.7}}
    )
    
    # Patch the client creation
    with patch('core.execution.service.VWAPExecutionClient', return_value=mock_vwap_client):
        # Execute order with ML features
        execution_id = await service.execute_order(
            test_order,
            params=ExecutionParameters(
                use_ml_routing=True,
                ml_features={'market_condition': 'volatile'}
            )
        )
        
        # Check execution status
        result = await service.get_execution_status(execution_id)
        assert result is not None
        assert 'ml_features' in result.metadata
    
    # Clean up
    await service.close()


@pytest.mark.asyncio
async def test_execution_error_handling(mock_market_data, mock_risk_manager, test_order):
    """Test error handling in the execution service."""
    # Initialize execution service
    service = ExecutionService(
        client_id="test_errors",
        market_data=mock_market_data,
        risk_manager=mock_risk_manager
    )
    
    # Test with invalid order
    with pytest.raises(ValueError):
        await service.execute_order(None)
    
    # Test with unsupported strategy
    with pytest.raises(ValueError):
        await service.execute_order(test_order, strategy="INVALID_STRATEGY")
    
    # Test cancellation of non-existent execution
    success = await service.cancel_execution("nonexistent_id")
    assert success is False
    
    # Test getting status of non-existent execution
    result = await service.get_execution_status("nonexistent_id")
    assert result is None
    
    # Clean up
    await service.close()


@pytest.mark.asyncio
async def test_execution_callbacks(mock_market_data, mock_risk_manager, test_order):
    """Test execution event callbacks."""
    # Initialize execution service
    service = ExecutionService(
        client_id="test_callbacks",
        market_data=mock_market_data,
        risk_manager=mock_risk_manager
    )
    
    # Setup callbacks
    start_called = asyncio.Future()
    update_called = asyncio.Future()
    complete_called = asyncio.Future()
    
    async def on_start(result):
        start_called.set_result(True)
    
    async def on_update(result):
        if result.status == ExecutionState.COMPLETED:
            update_called.set_result(True)
    
    async def on_complete(result):
        complete_called.set_result(True)
    
    # Register callbacks
    service.register_callback('on_execution_start', on_start)
    service.register_callback('on_execution_update', on_update)
    service.register_callback('on_execution_complete', on_complete)
    
    # Mock VWAP client
    mock_vwap_client = AsyncMock(spec=VWAPExecutionClient)
    mock_vwap_client.execute_order.return_value = "callback_test_123"
    mock_vwap_client.get_execution_status.return_value = ExecutionResult(
        execution_id="callback_test_123",
        client_order_id=test_order.client_order_id,
        symbol=test_order.symbol,
        side=test_order.side,
        quantity=test_order.quantity,
        filled_quantity=test_order.quantity,
        avg_fill_price=test_order.price,
        status=ExecutionState.COMPLETED
    )
    
    # Patch the client creation
    with patch('core.execution.service.VWAPExecutionClient', return_value=mock_vwap_client):
        # Execute order
        execution_id = await service.execute_order(test_order)
        
        # Wait for callbacks to be called
        await asyncio.wait_for(start_called, timeout=1.0)
        await asyncio.wait_for(update_called, timeout=1.0)
        await asyncio.wait_for(complete_called, timeout=1.0)
        
        # Verify callbacks were called
        assert start_called.done()
        assert update_called.done()
        assert complete_called.done()
    
    # Clean up
    await service.close()
