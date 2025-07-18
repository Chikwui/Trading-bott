"""
Performance benchmarking for the Sniper execution algorithm.

This script benchmarks the Sniper algorithm under various market conditions,
comparing it against naive execution strategies and measuring key metrics.
"""
import asyncio
import time
import json
import argparse
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import statistics
import warnings

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import stats

# Suppress specific warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Type aliases
Milliseconds = float
BasisPoints = float

class MarketCondition(str, Enum):
    NORMAL = "NORMAL"
    VOLATILE = "VOLATILE"
    ILLIQUID = "ILLIQUID"
    EXTREME = "EXTREME"

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark scenarios."""
    order_sizes: List[float] = field(default_factory=lambda: [0.001, 0.01, 0.1, 1.0])
    slippage_targets: List[int] = field(default_factory=lambda: [5, 10, 20])  # bps
    market_conditions: List[MarketCondition] = field(
        default_factory=lambda: [
            MarketCondition.NORMAL,
            MarketCondition.VOLATILE,
            MarketCondition.ILLIQUID
        ]
    )
    iterations_per_scenario: int = 5
    warmup_iterations: int = 2
    timeout_seconds: float = 30.0
    price_impact_windows: List[int] = field(default_factory=lambda: [1, 5, 15])  # minutes
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary for serialization."""
        return {
            'order_sizes': self.order_sizes,
            'slippage_targets': self.slippage_targets,
            'market_conditions': [c.value for c in self.market_conditions],
            'iterations_per_scenario': self.iterations_per_scenario,
            'warmup_iterations': self.warmup_iterations,
            'timeout_seconds': self.timeout_seconds,
            'price_impact_windows': self.price_impact_windows
        }

@dataclass
class ExecutionMetrics:
    """Container for execution metrics."""
    scenario_id: str
    timestamp: datetime
    success: bool = False
    error: Optional[str] = None
    execution_time_ms: Milliseconds = 0.0
    quantity_requested: float = 0.0
    quantity_executed: float = 0.0
    avg_execution_price: float = 0.0
    slippage_bps: float = 0.0
    participation_rate: float = 0.0
    num_child_orders: int = 0
    variance_impact_bps: float = 0.0
    market_impact_bps: float = 0.0
    price_impact_bps: Dict[int, float] = field(default_factory=dict)  # window -> impact
    volume_imbalance: float = 0.0
    spread_bps: float = 0.0
    volatility_bps: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary for serialization."""
        return {
            'scenario_id': self.scenario_id,
            'timestamp': self.timestamp.isoformat(),
            'success': self.success,
            'error': self.error,
            'execution_time_ms': self.execution_time_ms,
            'quantity_requested': self.quantity_requested,
            'quantity_executed': self.quantity_executed,
            'avg_execution_price': self.avg_execution_price,
            'slippage_bps': self.slippage_bps,
            'participation_rate': self.participation_rate,
            'num_child_orders': self.num_child_orders,
            'variance_impact_bps': self.variance_impact_bps,
            'market_impact_bps': self.market_impact_bps,
            'price_impact_bps': self.price_impact_bps,
            'volume_imbalance': self.volume_imbalance,
            'spread_bps': self.spread_bps,
            'volatility_bps': self.volatility_bps
        }

class BenchmarkResult:
    """Container for benchmark results with statistical analysis."""
    
    def __init__(self, config: BenchmarkConfig):
        self.metrics: List[ExecutionMetrics] = []
        self.config = config
        self.summary_stats: Dict[str, Any] = {}
        self.timestamps: Dict[str, str] = {}
        self.visualizations: Dict[str, str] = {}
    
    def add_metric(self, metric: ExecutionMetrics):
        """Add a metric to the results."""
        self.metrics.append(metric)
    
    def add_timestamp(self, event: str, timestamp: datetime):
        """Add a timestamped event."""
        self.timestamps[event] = timestamp.isoformat()
    
    def calculate_statistics(self):
        """Calculate summary statistics for all metrics."""
        if not self.metrics:
            return
            
        df = self.to_dataframe()
        successful = df[df['success']]
        
        if len(successful) == 0:
            return
        
        # Basic statistics
        self.summary_stats['success_rate'] = df['success'].mean()
        self.summary_stats['total_executions'] = len(df)
        self.summary_stats['successful_executions'] = len(successful)
        
        # Time statistics
        time_cols = ['execution_time_ms', 'slippage_bps', 'market_impact_bps']
        for col in time_cols:
            if col in successful.columns:
                self.summary_stats[f'{col}_stats'] = {
                    'mean': successful[col].mean(),
                    'median': successful[col].median(),
                    'std': successful[col].std(),
                    'min': successful[col].min(),
                    'max': successful[col].max(),
                    'q25': successful[col].quantile(0.25),
                    'q75': successful[col].quantile(0.75)
                }
        
        # Correlation analysis
        if len(successful) > 1:
            corr_cols = ['order_size', 'execution_time_ms', 'slippage_bps', 'market_impact_bps']
            corr_cols = [c for c in corr_cols if c in successful.columns]
            self.summary_stats['correlations'] = successful[corr_cols].corr().to_dict()
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert metrics to pandas DataFrame."""
        if not self.metrics:
            return pd.DataFrame()
            
        # Convert metrics to dicts and handle nested price_impact_bps
        records = []
        for metric in self.metrics:
            metric_dict = metric.to_dict()
            # Flatten price impact windows
            for window, impact in metric_dict.pop('price_impact_bps', {}).items():
                metric_dict[f'price_impact_{window}m_bps'] = impact
            records.append(metric_dict)
            
        return pd.DataFrame(records)
    
    def save(self, output_dir: Path) -> Path:
        """Save results to files."""
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        base_path = output_dir / f"sniper_benchmark_{timestamp}"
        
        # Save metrics to CSV
        df = self.to_dataframe()
        if not df.empty:
            df.to_csv(f"{base_path}_metrics.csv", index=False)
        
        # Save summary statistics
        with open(f"{base_path}_summary.json", 'w') as f:
            json.dump({
                'config': self.config.to_dict(),
                'timestamps': self.timestamps,
                'statistics': self.summary_stats,
                'visualizations': self.visualizations
            }, f, indent=2, default=str)
        
        # Generate and save visualizations
        if not df.empty:
            self._generate_visualizations(df, base_path)
        
        return base_path
    
    def _generate_visualizations(self, df: pd.DataFrame, base_path: Path):
        """Generate and save visualizations."""
        sns.set_theme(style="whitegrid")
        
        # 1. Execution Time vs Order Size
        plt.figure(figsize=(12, 6))
        sns.lineplot(
            data=df[df['success']],
            x='order_size',
            y='execution_time_ms',
            hue='market_condition',
            marker='o'
        )
        plt.xscale('log')
        plt.xlabel('Order Size (log scale)')
        plt.ylabel('Execution Time (ms)')
        plt.title('Execution Time vs Order Size by Market Condition')
        time_plot_path = f"{base_path}_execution_time.png"
        plt.savefig(time_plot_path, bbox_inches='tight', dpi=300)
        self.visualizations['execution_time'] = str(time_plot_path)
        plt.close()
        
        # 2. Slippage Distribution
        plt.figure(figsize=(12, 6))
        sns.boxplot(
            data=df[df['success']],
            x='market_condition',
            y='slippage_bps',
            hue='slippage_target',
            showfliers=False
        )
        plt.axhline(0, color='red', linestyle='--', alpha=0.5)
        plt.title('Slippage Distribution by Market Condition and Target')
        slippage_plot_path = f"{base_path}_slippage.png"
        plt.savefig(slippage_plot_path, bbox_inches='tight', dpi=300)
        self.visualizations['slippage'] = str(slippage_plot_path)
        plt.close()
        
        # 3. Market Impact Analysis
        if 'market_impact_bps' in df.columns:
            plt.figure(figsize=(12, 6))
            sns.scatterplot(
                data=df[df['success']],
                x='order_size',
                y='market_impact_bps',
                hue='market_condition',
                style='slippage_target',
                alpha=0.7
            )
            plt.xscale('log')
            plt.xlabel('Order Size (log scale)')
            plt.ylabel('Market Impact (bps)')
            plt.title('Market Impact vs Order Size')
            impact_plot_path = f"{base_path}_market_impact.png"
            plt.savefig(impact_plot_path, bbox_inches='tight', dpi=300)
            self.visualizations['market_impact'] = str(impact_plot_path)
            plt.close()
        
        # 4. Price Impact Over Time
        price_impact_cols = [c for c in df.columns if c.startswith('price_impact_')]
        if price_impact_cols:
            impact_df = df.melt(
                id_vars=['scenario_id', 'market_condition'],
                value_vars=price_impact_cols,
                var_name='window',
                value_name='impact_bps'
            )
            impact_df['window'] = impact_df['window'].str.extract('(\d+)').astype(int)
            
            plt.figure(figsize=(12, 6))
            sns.lineplot(
                data=impact_df,
                x='window',
                y='impact_bps',
                hue='market_condition',
                marker='o'
            )
            plt.xlabel('Time Window (minutes)')
            plt.ylabel('Price Impact (bps)')
            plt.title('Price Impact Over Time by Market Condition')
            price_impact_path = f"{base_path}_price_impact.png"
            plt.savefig(price_impact_path, bbox_inches='tight', dpi=300)
            self.visualizations['price_impact'] = str(price_impact_path)
            plt.close()

class SniperBenchmark:
    """Benchmark for the Sniper execution algorithm."""
    
    def __init__(self, exchange_id: str, symbol: str, test_mode: bool = True):
        """Initialize the benchmark."""
        self.exchange_id = exchange_id
        self.symbol = symbol
        self.test_mode = test_mode
        self.exchange = None
        self.results = BenchmarkResult(BenchmarkConfig())
        
        # Default benchmark parameters
        self.parameters = BenchmarkConfig()
    
    async def initialize(self):
        """Initialize the exchange connection."""
        self.exchange = ExchangeFactory.create(
            exchange_id=self.exchange_id,
            api_key=os.getenv(f"{self.exchange_id.upper()}_API_KEY"),
            api_secret=os.getenv(f"{self.exchange_id.upper()}_API_SECRET"),
            testnet=self.test_mode
        )
        await self.exchange.initialize()
        
        # Get market info
        self.market_info = await self.exchange.get_market_data(self.symbol)
        self.results.add_parameter('market_info', self.market_info)
        self.results.add_parameter('benchmark_parameters', self.parameters.to_dict())
    
    async def close(self):
        """Close the exchange connection."""
        if self.exchange:
            await self.exchange.close()
    
    async def run_benchmark(self):
        """Run the full benchmark suite."""
        self.results.add_timestamp('benchmark_start', datetime.now(timezone.utc))
        
        # Warm-up phase
        await self._warmup()
        
        # Main benchmark scenarios
        scenarios = self._generate_scenarios()
        
        for scenario in tqdm(scenarios, desc="Running benchmark scenarios"):
            await self._run_scenario(scenario)
        
        # Generate summary metrics
        self.results.calculate_statistics()
        
        self.results.add_timestamp('benchmark_end', datetime.now(timezone.utc))
        return self.results
    
    async def _warmup(self):
        """Run warm-up iterations to stabilize measurements."""
        warmup_order = {
            'order_size': self.parameters.order_sizes[0],
            'slippage_target': self.parameters.slippage_targets[0],
            'market_condition': 'NORMAL'
        }
        
        for _ in range(self.parameters.warmup_iterations):
            await self._execute_scenario(warmup_order)
    
    def _generate_scenarios(self) -> List[Dict]:
        """Generate benchmark scenarios from parameters."""
        scenarios = []
        
        for size in self.parameters.order_sizes:
            for slippage in self.parameters.slippage_targets:
                for condition in self.parameters.market_conditions:
                    for i in range(self.parameters.iterations_per_scenario):
                        scenarios.append({
                            'scenario_id': f"size_{size}_slippage_{slippage}_{condition.value}_{i}",
                            'order_size': size,
                            'slippage_target': slippage,
                            'market_condition': condition,
                            'iteration': i
                        })
        
        return scenarios
    
    async def _run_scenario(self, scenario: Dict):
        """Execute a single benchmark scenario."""
        try:
            # Configure market condition
            await self._simulate_market_condition(scenario['market_condition'])
            
            # Execute the scenario
            result = await self._execute_scenario(scenario)
            
            # Store results
            self.results.add_metric(result)
            
        except Exception as e:
            self.results.add_metric(ExecutionMetrics(
                scenario_id=scenario['scenario_id'],
                timestamp=datetime.now(timezone.utc),
                success=False,
                error=str(e)
            ))
    
    async def _simulate_market_condition(self, condition: MarketCondition):
        """Simulate different market conditions."""
        if condition == MarketCondition.NORMAL:
            # Use current market conditions
            pass
        elif condition == MarketCondition.VOLATILE:
            # Simulate high volatility
            # This is a placeholder - in a real implementation, we might:
            # 1. Look for a volatile period in historical data
            # 2. Use a volatility model to generate synthetic data
            # 3. Adjust the exchange simulator (if in test mode)
            pass
        elif condition == MarketCondition.ILLIQUID:
            # Simulate low liquidity
            # Similar to above, but for low liquidity conditions
            pass
    
    async def _execute_scenario(self, scenario: Dict) -> ExecutionMetrics:
        """Execute a single scenario and return metrics."""
        order_size = Decimal(str(scenario['order_size']))
        slippage_target = scenario['slippage_target']
        
        # Get current market data
        market_data = await self.exchange.get_market_data(self.symbol)
        best_bid = Decimal(str(market_data['best_bid']))
        best_ask = Decimal(str(market_data['best_ask']))
        mid_price = (best_bid + best_ask) / 2
        
        # Create order
        order = Order(
            order_id=f"bench_{int(time.time())}_{scenario['scenario_id']}",
            client_order_id=f"bench_{int(time.time())}_{scenario['scenario_id']}",
            symbol=self.symbol,
            side=OrderSide.BUY,  # Always buy for consistency
            order_type=OrderType.MARKET,
            quantity=order_size,
            time_in_force=TimeInForce.IOC,
            status=OrderStatus.NEW,
            timestamp=datetime.now(timezone.utc)
        )
        
        # Configure Sniper
        sniper = SniperExecutor(
            exchange_adapter=self.exchange,
            position_manager=MagicMock(),
            config={
                'max_slippage': str(slippage_target),
                'urgency': '0.8',
                'max_participation': '0.2',
                'min_slice_size': '0.001',
                'max_slice_size': str(float(order_size) / 2),  # Ensure slicing
                'refresh_interval': '0.1',
                'max_retries': '3',
                'dark_pool_enabled': 'False',
                'anti_gaming': 'True',
                'volatility_adaptive': 'True'
            }
        )
        
        # Execute and measure
        start_time = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                sniper.execute(order),
                timeout=self.parameters.timeout_seconds
            )
            execution_time = time.perf_counter() - start_time
            
            # Calculate metrics
            metrics = ExecutionMetrics(
                scenario_id=scenario['scenario_id'],
                timestamp=datetime.now(timezone.utc),
                success=True,
                execution_time_ms=execution_time * 1000,
                quantity_requested=float(order.quantity),
                quantity_executed=float(result.quantity_executed),
                avg_execution_price=float(result.avg_execution_price),
                slippage_bps=float(result.implementation_shortfall_bps),
                participation_rate=float(result.metadata.get('participation_rate', 0)),
                num_child_orders=len(result.fills),
                variance_impact_bps=self._calculate_variance_impact(result, mid_price),
                market_impact_bps=self._calculate_market_impact(result, mid_price)
            )
            
            return metrics
            
        except asyncio.TimeoutError:
            return ExecutionMetrics(
                scenario_id=scenario['scenario_id'],
                timestamp=datetime.now(timezone.utc),
                success=False,
                error='timeout',
                execution_time_ms=self.parameters.timeout_seconds * 1000
            )
        except Exception as e:
            return ExecutionMetrics(
                scenario_id=scenario['scenario_id'],
                timestamp=datetime.now(timezone.utc),
                success=False,
                error=str(e),
                execution_time_ms=(time.perf_counter() - start_time) * 1000
            )
    
    def _calculate_variance_impact(self, result, mid_price: Decimal) -> float:
        """Calculate the variance impact of the execution."""
        if not result.fills:
            return 0.0
            
        prices = [Decimal(str(fill['price'])) for fill in result.fills]
        avg_price = sum(prices) / len(prices)
        variance = sum((p - avg_price) ** 2 for p in prices) / len(prices)
        return float(variance / mid_price * 10000)  # Convert to basis points
    
    def _calculate_market_impact(self, result, mid_price: Decimal) -> float:
        """Calculate the market impact of the execution."""
        if not result.fills:
            return 0.0
            
        # Simple implementation: percentage deviation from mid-price
        impact = (result.avg_execution_price - mid_price) / mid_price
        return float(impact * 10000)  # Convert to basis points

async def main():
    """Run the benchmark from command line."""
    parser = argparse.ArgumentParser(description='Benchmark Sniper execution algorithm')
    parser.add_argument('--exchange', type=str, default='binance',
                       help='Exchange ID (e.g., binance, ftx)')
    parser.add_argument('--symbol', type=str, default='BTC/USDT',
                       help='Trading symbol (e.g., BTC/USDT)')
    parser.add_argument('--test-mode', action='store_true', default=True,
                       help='Use testnet/sandbox mode')
    parser.add_argument('--output-dir', type=str, default='benchmark_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Run benchmark
    benchmark = SniperBenchmark(
        exchange_id=args.exchange,
        symbol=args.symbol,
        test_mode=args.test_mode
    )
    
    try:
        await benchmark.initialize()
        results = await benchmark.run_benchmark()
        
        # Save results
        output_dir = Path(args.output_dir)
        results_dir = results.save(output_dir)
        print(f"Benchmark completed. Results saved to: {results_dir}")
        
        # Print summary
        print("\n=== Benchmark Summary ===")
        print(f"Success rate: {results.summary_stats.get('success_rate', 0):.2f}%")
        
    finally:
        await benchmark.close()


if __name__ == "__main__":
    asyncio.run(main())
