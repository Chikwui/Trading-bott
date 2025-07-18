import pytest
import asyncio
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch

from core.models.order import Order, OrderSide, OrderStatus, OrderType
from core.execution.strategies.vwap import VWAPExecutionClient
from core.models.execution import ExecutionParameters, ExecutionMode

class MockMarketImpactModel:
    def __init__(self):
        self.updated = False
        self.predicted_impact = 5.0  # Default to low impact
        self.last_prediction = None
        
    async def update(self, features):
        self.updated = True
        self.last_features = features
        
    async def predict(self, features):
        self.last_prediction = features
        return self.predicted_impact

class TestVWAPIntegration:
    @pytest.fixture
    def vwap_client(self):
        client = VWAPExecutionClient(
            symbol="BTC/USDT",
            exchange_client=MagicMock(),
            config={
                "max_participation_rate": 0.2,
                "min_order_size": 0.001,
                "max_order_size": 10.0,
                "adaptive_sizing": True,
                "impact_threshold_bps": 10.0,
                "step_size": 0.000001,
                "price_tick": 0.01,
                "max_slippage_bps": 10.0,
                "max_retries": 3,
                "retry_delay_sec": 1.0,
                "allow_partial_fills": True,
                "enable_shorting": True,
                "max_position_size": 100.0,
                "position_size_pct": 0.1,
                "risk_per_trade_pct": 1.0,
                "max_daily_loss_pct": 5.0,
                "max_drawdown_pct": 10.0,
                "volatility_lookback": 20,
                "liquidity_lookback": 5,
                "vwap_lookback": 30,
                "volume_profile_lookback": 60,
                "market_regime_lookback": 50,
                "impact_model_retrain_interval": 1000,
                "impact_model_warmup_samples": 100,
                "impact_model_features": [
                    'quantity', 'participation_rate', 'volatility',
                    'spread', 'liquidity', 'time_of_day', 'market_regime'
                ],
                "impact_model_target": 'implementation_shortfall',
                "impact_model_type": 'xgb',
                "impact_model_params": {
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'n_estimators': 100,
                    'min_child_weight': 1,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'reg_alpha': 0.1,
                    'reg_lambda': 1.0,
                    'random_state': 42
                }
            }
        )
        
        # Setup mock market impact model
        client._market_impact_model = MockMarketImpactModel()
        
        # Setup mock exchange client
        client.exchange = MagicMock()
        client.exchange.get_order_book = AsyncMock(return_value={
            'bids': [[50000.0, 1.0], [49999.0, 2.0]],
            'asks': [[50001.0, 1.0], [50002.0, 2.0]],
            'timestamp': datetime.utcnow().timestamp()
        })
        
        client.exchange.place_order = AsyncMock(return_value={
            'order_id': 'test_order_123',
            'status': OrderStatus.FILLED,
            'filled_quantity': 0.1,
            'avg_fill_price': 50000.5,
            'fee': 0.001,
            'timestamp': datetime.utcnow().timestamp()
        })
        
        client.exchange.get_ticker = AsyncMock(return_value={
            'bid': 50000.0,
            'ask': 50001.0,
            'last': 50000.5,
            'volume': 100.0,
            'timestamp': datetime.utcnow().timestamp()
        })
        
        client.exchange.get_historical_bars = AsyncMock(return_value=[
            {
                'timestamp': (datetime.utcnow() - timedelta(minutes=i)).timestamp(),
                'open': 50000.0 - i * 10,
                'high': 50010.0 - i * 10,
                'low': 49990.0 - i * 10,
                'close': 50005.0 - i * 10,
                'volume': 100.0 - i,
                'vwap': 50000.0 - i * 10
            }
            for i in range(100)
        ])
        
        return client
    
    @pytest.fixture
    def parent_order(self):
        return Order(
            symbol="BTC/USDT",
            side=OrderSide.BUY,
            type=OrderType.MARKET,
            quantity=Decimal("1.0"),
            price=Decimal("50000.0"),
            client_order_id="test_parent_order_1"
        )
    
    @pytest.fixture
    def execution_params(self):
        return ExecutionParameters(
            mode=ExecutionMode.LIVE,
            max_retries=3,
            timeout_sec=30,
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow() + timedelta(minutes=5)
        )
    
    @pytest.mark.asyncio
    async def test_complete_vwap_flow(self, vwap_client, parent_order, execution_params):
        """Test the complete VWAP execution flow with market impact analysis."""
        # Setup test data
        vwap_client._market_impact_model.predicted_impact = 5.0  # Low impact
        
        # Execute parent order
        await vwap_client.execute_order(parent_order, execution_params)
        
        # Verify order execution
        assert vwap_client.exchange.place_order.await_count > 0
        
        # Verify market impact model was updated
        assert vwap_client._market_impact_model.updated is True
        
        # Verify execution metrics
        assert hasattr(vwap_client, '_execution_metrics')
        assert 'total_orders' in vwap_client._execution_metrics
        assert vwap_client._execution_metrics['total_orders'] > 0
        
        # Verify adaptive sizing was used
        assert vwap_client._market_impact_model.last_prediction is not None
        
        # Verify order completion
        assert parent_order.status == OrderStatus.FILLED or parent_order.filled_quantity > 0
    
    @pytest.mark.asyncio
    async def test_high_impact_scenario(self, vwap_client, parent_order, execution_params):
        """Test behavior when predicted market impact is high."""
        # Set high predicted impact
        vwap_client._market_impact_model.predicted_impact = 20.0  # Above threshold
        
        # Execute parent order
        await vwap_client.execute_order(parent_order, execution_params)
        
        # Verify order size was reduced due to high impact
        place_order_calls = vwap_client.exchange.place_order.await_args_list
        assert len(place_order_calls) > 0
        
        # Get the first child order's quantity
        child_order = place_order_calls[0][0][0]
        assert float(child_order.quantity) < float(parent_order.quantity)
    
    @pytest.mark.asyncio
    async def test_market_regime_handling(self, vwap_client, parent_order, execution_params):
        """Test behavior in different market regimes."""
        # Setup volatile market regime
        vwap_client._detect_market_regime = MagicMock(return_value='high_volatility')
        
        # Execute parent order
        await vwap_client.execute_order(parent_order, execution_params)
        
        # Verify order sizing was adjusted for high volatility
        place_order_calls = vwap_client.exchange.place_order.await_args_list
        assert len(place_order_calls) > 0
        
        # Get the first child order's quantity
        child_order = place_order_calls[0][0][0]
        assert float(child_order.quantity) < float(parent_order.quantity)
    
    @pytest.mark.asyncio
    async def test_error_handling_and_retries(self, vwap_client, parent_order, execution_params):
        """Test error handling and retry logic."""
        # Setup intermittent failures
        vwap_client.exchange.place_order.side_effect = [
            Exception("Temporary error"),
            Exception("Temporary error"),
            {
                'order_id': 'test_order_123',
                'status': OrderStatus.FILLED,
                'filled_quantity': 0.1,
                'avg_fill_price': 50000.5,
                'fee': 0.001,
                'timestamp': datetime.utcnow().timestamp()
            }
        ]
        
        # Execute parent order
        await vwap_client.execute_order(parent_order, execution_params)
        
        # Verify retries were attempted
        assert vwap_client.exchange.place_order.await_count == 3
        
        # Verify final success
        assert parent_order.status == OrderStatus.FILLED or parent_order.filled_quantity > 0
