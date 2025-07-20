"""Performance benchmarks for OrderManager."""
import asyncio
import time
from decimal import Decimal
import statistics
from typing import List, Dict, Any
import pytest
import pytest_asyncio

from core.trading.order import OrderSide, OrderType
from core.trading.oco_order import OCOOrderConfig
from tests.unit.trading.test_helpers import (
    MockExchangeAdapter,
    create_test_order_manager,
    create_test_oco_config
)

# Benchmark configuration
NUM_ITERATIONS = 1000  # Number of operations per test
BATCH_SIZES = [1, 10, 100]  # Test different batch sizes


class OrderManagerBenchmark:
    """Benchmark suite for OrderManager performance testing."""
    
    @pytest_asyncio.fixture
    async def order_manager(self):
        """Create an OrderManager instance for testing."""
        exchange = MockExchangeAdapter()
        # Set very high rate limit for benchmarks and disable metrics
        manager = await create_test_order_manager(
            exchange_adapter=exchange, 
            max_orders_per_second=1_000_000,
            enable_metrics=False  # Disable metrics for benchmarks
        )
        yield manager
        await manager.stop()
        
    async def order_manager_ctx(self):
        """Async context manager for OrderManager instance."""
        exchange = MockExchangeAdapter()
        # Set very high rate limit for benchmarks and disable metrics
        manager = await create_test_order_manager(
            exchange_adapter=exchange, 
            max_orders_per_second=1_000_000,
            enable_metrics=False  # Disable metrics for benchmarks
        )
        try:
            yield manager
        finally:
            await manager.stop()
    
    @pytest.mark.asyncio
    @pytest.mark.parametrize("batch_size", BATCH_SIZES)
    async def test_order_submission_throughput(self, benchmark, order_manager, batch_size):
        """Benchmark order submission throughput."""
        async def _submit_batch():
            tasks = []
            for i in range(batch_size):
                task = order_manager.submit_order(
                    symbol=f"SYM{i%10}",  # Use 10 different symbols
                    side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                    order_type=OrderType.LIMIT,
                    quantity=Decimal("1.0"),
                    limit_price=Decimal("100.00") + Decimal(str(i % 10))
                )
                tasks.append(asyncio.create_task(task))
            await asyncio.gather(*tasks)
        
        # Warm-up
        await _submit_batch()
        
        # Run benchmark
        times = []
        for _ in range(NUM_ITERATIONS // batch_size):
            start = time.perf_counter()
            await _submit_batch()
            end = time.perf_counter()
            times.append(end - start)
        
        # Calculate and report metrics
        total_orders = NUM_ITERATIONS
        total_time = sum(times)
        orders_per_sec = total_orders / total_time if total_time > 0 else 0
        
        print(f"\nBatch size: {batch_size}")
        print(f"Total orders: {total_orders}")
        print(f"Total time: {total_time:.4f}s")
        print(f"Orders/second: {orders_per_sec:,.0f}")
        print(f"Latency per order: {total_time * 1000 / total_orders:.4f}ms")
        
        # Store benchmark results
        benchmark.extra_info["batch_size"] = batch_size
        benchmark.extra_info["orders_per_second"] = orders_per_sec
        benchmark.extra_info["latency_ms"] = total_time * 1000 / total_orders
    
    @pytest.mark.asyncio
    async def test_oco_order_throughput(self, benchmark, order_manager):
        """Benchmark OCO order submission and processing."""
        async def _submit_oco_batch():
            tasks = []
            for i in range(10):  # Submit 10 OCO orders per batch
                config = create_test_oco_config(
                    symbol=f"SYM{i%5}",  # Use 5 different symbols
                    quantity=Decimal("1.0"),
                    entry_price=Decimal("100.00") + Decimal(str(i)),
                    stop_loss_price=Decimal("95.00") + Decimal(str(i)),
                    take_profit_price=Decimal("105.00") + Decimal(str(i))
                )
                tasks.append(asyncio.create_task(order_manager.submit_oco_order(config)))
            await asyncio.gather(*tasks)
        
        # Warm-up
        await _submit_oco_batch()
        
        # Run benchmark
        times = []
        for _ in range(NUM_ITERATIONS // 10):
            start = time.perf_counter()
            await _submit_oco_batch()
            end = time.perf_counter()
            times.append(end - start)
        
        # Calculate and report metrics
        total_orders = NUM_ITERATIONS * 2  # Each OCO creates 2 orders
        total_time = sum(times)
        orders_per_sec = total_orders / total_time if total_time > 0 else 0
        
        print(f"\nOCO Order Benchmark")
        print(f"Total OCO orders: {NUM_ITERATIONS}")
        print(f"Total orders (including legs): {total_orders}")
        print(f"Total time: {total_time:.4f}s")
        print(f"Orders/second: {orders_per_sec:,.0f}")
        print(f"Latency per OCO order: {total_time * 1000 / (NUM_ITERATIONS):.4f}ms")
        
        # Store benchmark results
        benchmark.extra_info["orders_per_second"] = orders_per_sec
        benchmark.extra_info["latency_ms_per_oco"] = total_time * 1000 / NUM_ITERATIONS
    
    @pytest.mark.asyncio
    async def test_concurrent_order_processing(self, benchmark, order_manager):
        """Test concurrent order processing with multiple symbols."""
        
        async def _submit_orders_for_symbol(symbol: str, num_orders: int):
            tasks = []
            for i in range(num_orders):
                task = order_manager.submit_order(
                    symbol=symbol,
                    side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                    order_type=OrderType.LIMIT,
                    quantity=Decimal("1.0"),
                    price=Decimal("100.00") + Decimal(str(i % 10))
                )
                tasks.append(asyncio.create_task(task))
            await asyncio.gather(*tasks)
        
        symbols = [f"SYM{i}" for i in range(10)]  # 10 different symbols
        orders_per_symbol = NUM_ITERATIONS // len(symbols)
        
        # Warm-up
        await _submit_orders_for_symbol("WARMUP", 10)
        
        # Run benchmark
        start = time.perf_counter()
        
        tasks = []
        for symbol in symbols:
            task = asyncio.create_task(_submit_orders_for_symbol(symbol, orders_per_symbol))
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        end = time.perf_counter()
        
        # Calculate and report metrics
        total_orders = NUM_ITERATIONS
        total_time = end - start
        orders_per_sec = total_orders / total_time if total_time > 0 else 0
        
        print(f"\nConcurrent Order Processing")
        print(f"Symbols: {len(symbols)}")
        print(f"Orders per symbol: {orders_per_symbol}")
        print(f"Total orders: {total_orders}")
        print(f"Total time: {total_time:.4f}s")
        print(f"Orders/second: {orders_per_sec:,.0f}")
        
        # Store benchmark results
        benchmark.extra_info["symbols"] = len(symbols)
        benchmark.extra_info["orders_per_second"] = orders_per_sec
        benchmark.extra_info["total_orders"] = total_orders
        benchmark.extra_info["total_time_seconds"] = total_time


async def run_benchmarks():
    """Run benchmarks and print results."""
    print("Running OrderManager benchmarks...\n")
    
    benchmark = OrderManagerBenchmark()
    # Get the async generator
    order_manager_gen = benchmark.order_manager_ctx()
    # Get the manager from the generator
    manager = await order_manager_gen.__anext__()
    
    try:
        # Create a benchmark instance with extra_info
        benchmark_instance = type('Benchmark', (), {'extra_info': {}})()
        
        # Run benchmarks
        print("=== Order Submission Throughput ===")
        for batch_size in BATCH_SIZES:
            await benchmark.test_order_submission_throughput(
                benchmark_instance, 
                manager, 
                batch_size
            )
    
        print("\n=== OCO Order Throughput ===")
        await benchmark.test_oco_order_throughput(
            benchmark_instance, 
            manager
        )
        
        print("\n=== Concurrent Order Processing ===")
        await benchmark.test_concurrent_order_processing(
            benchmark_instance, 
            manager
        )
    finally:
        # Clean up the generator
        try:
            await order_manager_gen.__anext__()
        except StopAsyncIteration:
            pass

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_benchmarks())
