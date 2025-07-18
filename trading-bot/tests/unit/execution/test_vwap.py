import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from decimal import Decimal
from datetime import datetime, timedelta
import numpy as np

from core.models.order import Order, OrderSide, OrderType, OrderStatus
from core.execution.strategies.vwap import VWAPExecutionClient
from core.models.market_data import MarketData, Ticker, OrderBook, Trade, Bar
from core.models.execution import ExecutionParameters, ExecutionMode

class MockMarketImpactModel:
    def __init__(self):
        self.updated = False
        self.predicted_impact = 0.0
        
    async def update(self, features):
        self.updated = True
        self.last_features = features
        
    async def predict(self, features):
        return self.predicted_impact

class TestVWAPMarketImpact:
    @pytest.fixture
    def vwap_client(self):
        client = VWAPExecutionClient(
            symbol="BTC/USDT",
            exchange_client=MagicMock(),
            config={
                "max_participation_rate": 0.1,
                "min_order_size": 0.001,
                "max_order_size": 1.0,
                "adaptive_sizing": True,
                "impact_threshold_bps": 10.0
            }
        )
        client._market_impact_model = MockMarketImpactModel()
        return client
    
    @pytest.fixture
    def market_data(self):
        return {
            'bid': 50000.0,
            'ask': 50010.0,
            'last': 50005.0,
            'volume': 100.0,
            'vwap': 49950.0,
            'volatility': 0.02,
            'spread': 10.0,
            'liquidity_score': 0.8
        }
    
    @pytest.fixture
    def order(self):
        return Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=Decimal("0.1"),
            price=Decimal("50000.0"),
            client_order_id="test_order_1"
        )
    
    @pytest.mark.asyncio
    async def test_calculate_market_impact(self, vwap_client, order, market_data):
        # Test with typical market data
        execution_result = {
            'avg_fill_price': 50005.0,
            'filled_quantity': 0.1,
            'status': OrderStatus.FILLED,
            'participation_rate': 0.05,
            'execution_time': datetime.utcnow().timestamp()
        }
        
        # Set arrival price in market data
        market_data['arrival_price'] = 50002.5
        
        # Calculate impact
        impact = vwap_client._calculate_market_impact(
            order, market_data, execution_result
        )
        
        # Verify impact metrics
        assert 'pre_trade_impact' in impact
        assert 'post_trade_impact' in impact
        assert 'temporary_impact' in impact
        assert 'implementation_shortfall' in impact
        
        # Verify calculations (in bps)
        expected_pre_impact = (50002.5 - 50005.0) / 50005.0 * 10000
        assert abs(impact['pre_trade_impact'] - expected_pre_impact) < 0.1
        
        expected_post_impact = (50005.0 - 50002.5) / 50002.5 * 10000
        assert abs(impact['post_trade_impact'] - expected_post_impact) < 0.1
        
        expected_shortfall = (50005.0 - 50000.0) / 50000.0 * 10000
        assert abs(impact['implementation_shortfall'] - expected_shortfall) < 0.1
    
    @pytest.mark.asyncio
    async def test_update_market_impact_model(self, vwap_client, order, market_data):
        # Set up test data
        execution_result = {
            'avg_fill_price': 50005.0,
            'filled_quantity': 0.1,
            'status': OrderStatus.FILLED,
            'participation_rate': 0.05,
            'execution_time': datetime.utcnow().timestamp()
        }
        
        # Update model
        await vwap_client._update_market_impact_model(
            order, market_data, execution_result
        )
        
        # Verify model was updated with expected features
        model = vwap_client._market_impact_model
        assert model.updated is True
        assert 'symbol' in model.last_features
        assert 'side' in model.last_features
        assert 'quantity' in model.last_features
        assert 'participation_rate' in model.last_features
        assert 'volatility' in model.last_features
        assert 'spread' in model.last_features
        assert 'liquidity' in model.last_features
        assert 'time_of_day' in model.last_features
        
        # Verify numeric values
        assert model.last_features['quantity'] == 0.1
        assert model.last_features['participation_rate'] == 0.05
        assert model.last_features['volatility'] == 0.02
        assert model.last_features['spread'] == 10.0
        assert model.last_features['liquidity'] == 0.8
    
    @pytest.mark.asyncio
    async def test_adaptive_order_sizing(self, vwap_client, order, market_data):
        # Set up test
        vwap_client._market_impact_model.predicted_impact = 15.0  # Above threshold
        
        # Test with order size that would cause high impact
        max_size = vwap_client._calculate_adaptive_order_size(
            order=order,
            market_data=market_data,
            target_participation=0.1,
            max_impact_bps=10.0
        )
        
        # Verify size was reduced to stay under impact threshold
        assert float(max_size) < float(order.quantity)
        assert float(max_size) >= float(vwap_client.config['min_order_size'])
        
        # Test with low predicted impact
        vwap_client._market_impact_model.predicted_impact = 5.0  # Below threshold
        max_size = vwap_client._calculate_adaptive_order_size(
            order=order,
            market_data=market_data,
            target_participation=0.1,
            max_impact_bps=10.0
        )
        
        # Verify full size is allowed
        assert float(max_size) == float(order.quantity)
    
    @pytest.mark.asyncio
    async def test_execute_with_market_impact(self, vwap_client, order, market_data):
        # Mock dependencies
        vwap_client._get_market_data = AsyncMock(return_value=market_data)
        vwap_client._submit_order = AsyncMock(return_value=MagicMock(
            status=OrderStatus.FILLED,
            filled_quantity=Decimal("0.1"),
            avg_fill_price=Decimal("50005.0")
        ))
        
        # Execute order
        result = await vwap_client._execute_child_order(
            child_order=order,
            rate={'target_pct': 0.1, 'duration_sec': 60},
            params=ExecutionParameters(
                mode=ExecutionMode.LIVE,
                max_retries=3,
                timeout_sec=30
            )
        )
        
        # Verify order was executed
        assert result.status == OrderStatus.FILLED
        
        # Verify market impact model was updated
        assert vwap_client._market_impact_model.updated is True
        
        # Verify execution metrics were recorded
        assert hasattr(vwap_client, '_execution_metrics')
        assert 'total_orders' in vwap_client._execution_metrics
        assert vwap_client._execution_metrics['total_orders'] > 0
