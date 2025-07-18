"""
Performance benchmarks for VWAP execution strategy.

This module contains benchmarks to measure the performance of the VWAP execution
strategy under various market conditions and configurations.
"""
import asyncio
import time
from datetime import datetime, timedelta
from decimal import Decimal
import pytest
import pandas as pd
import numpy as np
from unittest.mock import AsyncMock, MagicMock, patch

from core.execution.strategies.vwap import VWAPExecutionClient, VWAPParameters
from core.models.order import Order, OrderSide, OrderType, OrderStatus
from core.models.execution import ExecutionParameters, ExecutionMode

class MockMarketDataService:
    """Mock market data service for benchmarking."""
    
    def __init__(self):
        self.bars = {}
        self.tickers = {}
        self._generate_mock_data()
    
    def _generate_mock_data(self):
        """Generate realistic market data for testing."""
        np.random.seed(42)
        dates = pd.date_range(end=datetime.utcnow(), periods=1000, freq='1min')
        base_price = 50000.0
        
        # Generate OHLCV data with realistic patterns
        returns = np.random.normal(0.0001, 0.01, len(dates))
        prices = base_price * (1 + np.cumsum(returns))
        
        # Generate volume with intraday pattern
        hours = np.array([d.hour for d in dates])
        volume_pattern = np.sin((hours - 10) * np.pi / 14) + 1.5  # Peak around 10am and 2pm
        volumes = (np.abs(np.random.normal(100, 20, len(dates))) * volume_pattern).astype(int)
        
        # Create DataFrame
        df = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.002, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.002, len(dates)))),
            'close': prices,
            'volume': volumes
        }, index=dates)
        
        self.bars['1min'] = df
        
        # Current ticker data
        last_row = df.iloc[-1]
        self.tickers['BTC/USDT'] = {
            'bid': float(last_row['close'] * 0.9995),  # 5bps spread
            'ask': float(last_row['close'] * 1.0005),
            'last': float(last_row['close']),
            'volume': int(last_row['volume'])
        }
    
    async def get_historical_bars(self, symbol, timeframe, limit=1000):
        """Get historical bar data."""
        df = self.bars.get(timeframe)
        if df is None:
            return pd.DataFrame()
        return df.iloc[-limit:].copy()
    
    async def get_ticker(self, symbol):
        """Get current ticker data."""
        return self.tickers.get(symbol, {})

class MockExecutionClient:
    """Mock execution client for benchmarking."""
    
    def __init__(self):
        self.orders = {}
        self.order_id = 1
        self.filled_orders = 0
        self.canceled_orders = 0
    
    async def create_order(self, order):
        """Simulate order creation with realistic fill simulation."""
        order.id = str(self.order_id)
        order.status = OrderStatus.NEW
        order.created_at = datetime.utcnow()
        
        # Simulate partial fills
        if np.random.random() > 0.2:  # 80% chance of fill
            fill_ratio = np.random.uniform(0.5, 1.0)  # 50-100% fill ratio
            filled_qty = Decimal(str(round(float(order.quantity) * fill_ratio, 8)))
            
            # Add some latency
            await asyncio.sleep(np.random.uniform(0.001, 0.01))
            
            order.filled_quantity = filled_qty
            order.avg_fill_price = Decimal(str(
                float(order.price or self._get_market_price()) * 
                (1 + np.random.normal(0, 0.0005))  # Small random price improvement
            ))
            order.status = OrderStatus.FILLED if fill_ratio > 0.95 else OrderStatus.PARTIALLY_FILLED
            self.filled_orders += 1
        else:
            # Simulate rejection or timeout
            order.status = OrderStatus.REJECTED if np.random.random() > 0.5 else OrderStatus.CANCELED
            self.canceled_orders += 1
            
        self.orders[order.id] = order
        self.order_id += 1
        return order
    
    def _get_market_price(self):
        """Get simulated market price."""
        return 50000.0 * (1 + np.random.normal(0, 0.001))

class TestVWAPBenchmark:
    """Benchmark tests for VWAP execution strategy."""
    
    @pytest.fixture
    def setup_benchmark(self):
        """Set up benchmark test environment."""
        # Initialize mock services
        market_data = MockMarketDataService()
        execution_client = MockExecutionClient()
        risk_manager = MagicMock()
        
        # Configure VWAP parameters
        vwap_params = VWAPParameters(
            volume_participation=0.1,
            use_limit_orders=True,
            enable_price_improvement=True,
            max_allowed_spread_bps=20,
            enable_volatility_scaling=True,
            enable_anti_gaming=True
        )
        
        # Create VWAP client
        client = VWAPExecutionClient(
            client_id="benchmark_client",
            market_data=market_data,
            risk_manager=risk_manager,
            vwap_params=vwap_params
        )
        
        # Create test order
        order = Order(
            symbol='BTC/USDT',
            side=OrderSide.BUY,
            quantity=Decimal('1.0'),
            order_type=OrderType.MARKET,
            time_in_force='GTC'
        )
        
        execution_params = ExecutionParameters(
            mode=ExecutionMode.DRY_RUN,
            max_slippage_bps=10,
            max_position_pct=0.1,
            max_order_value=100000
        )
        
        return {
            'client': client,
            'order': order,
            'execution_params': execution_params,
            'market_data': market_data,
            'execution_client': execution_client
        }
    
    @pytest.mark.benchmark(group="execution_speed")
    @pytest.mark.asyncio
    async def test_vwap_execution_speed(self, benchmark, setup_benchmark):
        """Benchmark VWAP execution speed."""
        client = setup_benchmark['client']
        order = setup_benchmark['order']
        params = setup_benchmark['execution_params']
        
        # Warm-up run
        await client._execute_strategy(order, params, ExecutionResult())
        
        # Benchmark
        start_time = time.perf_counter()
        
        async def run_execution():
            result = await client._execute_strategy(order, params, ExecutionResult())
            return result
            
        # Run benchmark
        result = await benchmark(run_execution)
        
        # Assertions
        assert result is not None
        assert 'execution_time_ms' in result.metadata
        
        # Log results
        print(f"\nVWAP Execution Benchmark:")
        print(f"- Orders executed: {len(client._child_orders)}")
        print(f"- Total execution time: {result.metadata['execution_time_ms']:.2f}ms")
        print(f"- Avg time per order: {result.metadata['execution_time_ms']/max(1, len(client._child_orders)):.2f}ms")
    
    @pytest.mark.benchmark(group="market_impact")
    @pytest.mark.parametrize("order_size_btc", [0.1, 1.0, 10.0, 100.0])
    @pytest.mark.asyncio
    async def test_market_impact(self, benchmark, setup_benchmark, order_size_btc):
        """Benchmark market impact for different order sizes."""
        setup = setup_benchmark
        client = setup['client']
        order = setup['order']
        params = setup['execution_params']
        
        # Update order size
        order.quantity = Decimal(str(order_size_btc))
        
        # Run benchmark
        start_time = time.perf_counter()
        
        async def run_impact_test():
            result = await client._execute_strategy(order, params, ExecutionResult())
            return result
            
        result = await benchmark(run_impact_test)
        
        # Calculate market impact metrics
        initial_price = float((await client.market_data.get_ticker(order.symbol))['last'])
        final_price = float(result.metadata.get('avg_execution_price', initial_price))
        impact_bps = abs((final_price - initial_price) / initial_price * 10000)
        
        # Log results
        print(f"\nMarket Impact Benchmark ({order_size_btc} BTC):")
        print(f"- Initial price: {initial_price:.2f}")
        print(f"- Avg execution price: {final_price:.2f}")
        print(f"- Market impact: {impact_bps:.2f} bps")
        print(f"- Participation rate: {result.metadata.get('participation_rate', 0)*100:.2f}%")
        print(f"- Slippage: {result.metadata.get('slippage_bps', 0):.2f} bps")
        
        # Assertions
        assert impact_bps < 50.0  # Impact should be less than 0.5%
        assert 'market_impact_bps' in result.metadata
    
    @pytest.mark.benchmark(group="latency")
    @pytest.mark.parametrize("num_orders", [1, 10, 100])
    @pytest.mark.asyncio
    async def test_order_latency(self, benchmark, setup_benchmark, num_orders):
        """Benchmark order submission latency."""
        setup = setup_benchmark
        client = setup['client']
        
        # Create multiple orders
        orders = [
            Order(
                symbol='BTC/USDT',
                side=OrderSide.BUY,
                quantity=Decimal('0.1'),
                order_type=OrderType.MARKET
            ) for _ in range(num_orders)
        ]
        
        # Benchmark
        start_time = time.perf_counter()
        
        async def submit_orders():
            tasks = [client._execute_strategy(order, setup['execution_params'], ExecutionResult()) 
                    for order in orders]
            return await asyncio.gather(*tasks)
            
        results = await benchmark(submit_orders)
        
        # Calculate metrics
        total_time = time.perf_counter() - start_time
        avg_latency = total_time / num_orders * 1000  # ms per order
        
        # Log results
        print(f"\nOrder Latency Benchmark ({num_orders} orders):")
        print(f"- Total time: {total_time*1000:.2f}ms")
        print(f"- Avg latency per order: {avg_latency:.2f}ms")
        print(f"- Orders per second: {num_orders/total_time if total_time > 0 else float('inf'):.2f}")
        
        # Assertions
        assert len(results) == num_orders
        assert avg_latency < 100.0  # Should be under 100ms per order

    @pytest.mark.benchmark(group="scalability")
    @pytest.mark.parametrize("num_symbols", [1, 5, 10])
    @pytest.mark.asyncio
    async def test_multi_symbol_scalability(self, benchmark, setup_benchmark, num_symbols):
        """Test how well the strategy scales with multiple symbols."""
        setup = setup_benchmark
        client = setup['client']
        
        # Create orders for multiple symbols
        symbols = [f"SYM{i}/USDT" for i in range(1, num_symbols + 1)]
        orders = [
            Order(
                symbol=symbol,
                side=OrderSide.BUY,
                quantity=Decimal('0.1'),
                order_type=OrderType.MARKET
            ) for symbol in symbols
        ]
        
        # Update market data for all symbols
        for symbol in symbols:
            if symbol != 'BTC/USDT':
                setup['market_data'].tickers[symbol] = {
                    'bid': 100.0 * (1 - 0.0005),
                    'ask': 100.0 * (1 + 0.0005),
                    'last': 100.0,
                    'volume': 1000.0
                }
        
        # Benchmark
        start_time = time.perf_counter()
        
        async def run_multi_symbol():
            tasks = [client._execute_strategy(order, setup['execution_params'], ExecutionResult()) 
                    for order in orders]
            return await asyncio.gather(*tasks)
            
        results = await benchmark(run_multi_symbol)
        
        # Calculate metrics
        total_time = time.perf_counter() - start_time
        
        # Log results
        print(f"\nMulti-Symbol Scalability ({num_symbols} symbols):")
        print(f"- Total execution time: {total_time:.4f}s")
        print(f"- Avg time per symbol: {total_time/num_symbols*1000:.2f}ms")
        
        # Assertions
        assert len(results) == num_symbols
        assert all(isinstance(r, dict) for r in results)
