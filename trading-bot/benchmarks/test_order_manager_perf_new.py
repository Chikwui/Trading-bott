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
        # Set very high rate limit for benchmarks
        manager = await create_test_order_manager(exchange, max_orders_per_second=1_000_000)
        yield manager
        await manager.stop()
        
    def order_manager_ctx(self):
        """Create an async context manager for OrderManager instance."""
        class OrderManagerContext:
            async def __aenter__(self):
                self.exchange = MockExchangeAdapter()
                self.manager = await create_test_order_manager(
                    exchange_adapter=self.exchange, 
                    max_orders_per_second=1_000_000,
                    enable_metrics=False  # Disable metrics to avoid port binding issues
                )
                return self.manager
                
            async def __aexit__(self, exc_type, exc, tb):
                await self.manager.stop()
                
        return OrderManagerContext()
    
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
                    price=Decimal("100.00") + Decimal(str(i % 10))
                )
                tasks.append(asyncio.create_task(task))
            await asyncio.gather(*tasks)
        
        # Warmup
        await _submit_batch()
        
        # Run benchmark
        start_time = time.time()
        for _ in range(NUM_ITERATIONS // batch_size):
            await _submit_batch()
        total_time = time.time() - start_time
        
        # Calculate metrics
        total_orders = NUM_ITERATIONS
        orders_per_sec = total_orders / total_time if total_time > 0 else 0
        
        # Store benchmark results
        benchmark.extra_info["batch_size"] = batch_size
        benchmark.extra_info["orders_per_second"] = orders_per_sec
        benchmark.extra_info["total_orders"] = total_orders
        benchmark.extra_info["total_time_seconds"] = total_time
        
        print(f"\nBatch size: {batch_size}")
        print(f"Total orders: {total_orders:,}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Orders/second: {orders_per_sec:,.2f}")

    @pytest.mark.asyncio
    async def test_oco_order_throughput(self, benchmark, order_manager):
        """Benchmark OCO order submission and processing."""
        async def _submit_oco_batch():
            tasks = []
            for i in range(10):  # Submit 10 OCO orders per batch
                config = OCOOrderConfig(
                    symbol=f"SYM{i%5}",  # Use 5 different symbols
                    side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                    quantity=Decimal("1.0"),
                    price=Decimal("100.00") + Decimal(str(i % 10)),
                    stop_price=Decimal("95.00") - Decimal(str(i % 5)),
                    limit_price=Decimal("105.00") + Decimal(str(i % 5))
                )
                task = order_manager.submit_oco_order(config)
                tasks.append(asyncio.create_task(task))
            await asyncio.gather(*tasks)
        
        # Warmup
        await _submit_oco_batch()
        
        # Run benchmark
        start_time = time.time()
        for _ in range(NUM_ITERATIONS // 10):  # 10 OCO orders per iteration
            await _submit_oco_batch()
        total_time = time.time() - start_time
        
        # Calculate metrics
        total_orders = NUM_ITERATIONS
        orders_per_sec = total_orders / total_time if total_time > 0 else 0
        
        # Store benchmark results
        benchmark.extra_info["orders_per_second"] = orders_per_sec
        benchmark.extra_info["total_orders"] = total_orders
        benchmark.extra_info["total_time_seconds"] = total_time
        
        print(f"\nOCO Order Throughput:")
        print(f"Total OCO orders: {total_orders:,}")
        print(f"Total time: {total_time:.2f}s")
        print(f"OCO orders/second: {orders_per_sec:,.2f}")

    @pytest.mark.asyncio
    async def test_concurrent_order_processing(self, benchmark, order_manager):
        """Test concurrent order processing with multiple symbols."""
        symbols = [f"SYM{i}" for i in range(10)]  # 10 different symbols
        
        async def _submit_concurrent():
            tasks = []
            for symbol in symbols:
                for i in range(5):  # 5 orders per symbol
                    task = order_manager.submit_order(
                        symbol=symbol,
                        side=OrderSide.BUY if i % 2 == 0 else OrderSide.SELL,
                        order_type=OrderType.LIMIT,
                        quantity=Decimal("1.0"),
                        price=Decimal("100.00") + Decimal(str(i % 10))
                    )
                    tasks.append(asyncio.create_task(task))
            await asyncio.gather(*tasks)
        
        # Warmup
        await _submit_concurrent()
        
        # Run benchmark
        start_time = time.time()
        for _ in range(NUM_ITERATIONS // (len(symbols) * 5)):  # 50 orders per iteration
            await _submit_concurrent()
        total_time = time.time() - start_time
        
        # Calculate metrics
        total_orders = NUM_ITERATIONS
        orders_per_sec = total_orders / total_time if total_time > 0 else 0
        
        # Store benchmark results
        benchmark.extra_info["symbols"] = len(symbols)
        benchmark.extra_info["orders_per_second"] = orders_per_sec
        benchmark.extra_info["total_orders"] = total_orders
        benchmark.extra_info["total_time_seconds"] = total_time
        
        print(f"\nConcurrent Order Processing (across {len(symbols)} symbols):")
        print(f"Total orders: {total_orders:,}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Orders/second: {orders_per_sec:,.2f}")

async def run_benchmarks():
    """Run benchmarks and print results."""
    print("Running OrderManager benchmarks...\n")
    
    benchmark = OrderManagerBenchmark()
    async with benchmark.order_manager_ctx() as manager:
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

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_benchmarks())
