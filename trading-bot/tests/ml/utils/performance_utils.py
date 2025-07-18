""
Performance analysis utilities for ML pipeline testing.
"""
import os
import time
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Callable, Any, Tuple
from functools import wraps
from pathlib import Path
from pympler import muppy, summary
import seaborn as sns

class PerformanceProfiler:
    """Performance profiling and monitoring utilities."""
    
    def __init__(self, output_dir: str = "test-results/performance"):
        """Initialize the profiler with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.process = psutil.Process(os.getpid())
        self.metrics = {
            'timestamps': [],
            'cpu_percent': [],
            'memory_mb': [],
            'memory_percent': [],
            'read_bytes': [],
            'write_bytes': []
        }
        self.start_time = None
        self.io_start = None
    
    def start(self) -> None:
        """Start performance monitoring."""
        self.start_time = time.time()
        io = self.process.io_counters()
        self.io_start = (io.read_bytes, io.write_bytes)
        self._record_metrics()
    
    def _record_metrics(self) -> None:
        """Record current system metrics."""
        self.metrics['timestamps'].append(time.time() - (self.start_time or 0))
        
        # CPU and memory
        self.metrics['cpu_percent'].append(self.process.cpu_percent())
        mem_info = self.process.memory_info()
        self.metrics['memory_mb'].append(mem_info.rss / (1024 * 1024))  # MB
        self.metrics['memory_percent'].append(self.process.memory_percent())
        
        # I/O
        io = self.process.io_counters()
        if self.io_start:
            read_diff = io.read_bytes - self.io_start[0]
            write_diff = io.write_bytes - self.io_start[1]
            self.metrics['read_bytes'].append(read_diff / (1024 * 1024))  # MB
            self.metrics['write_bytes'].append(write_diff / (1024 * 1024))  # MB
    
    def measure(self, func: Callable) -> Callable:
        """Decorator to measure function performance."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.start()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                self.stop()
                self.analyze(func.__name__)
        return wrapper
    
    def stop(self) -> Dict[str, Any]:
        """Stop monitoring and return results."""
        if self.start_time is None:
            raise RuntimeError("Profiling not started")
        
        self._record_metrics()
        duration = time.time() - self.start_time
        
        results = {
            'duration': duration,
            'avg_cpu': np.mean(self.metrics['cpu_percent']),
            'max_memory_mb': max(self.metrics['memory_mb']),
            'avg_memory_mb': np.mean(self.metrics['memory_mb']),
            'total_read_mb': self.metrics['read_bytes'][-1] if self.metrics['read_bytes'] else 0,
            'total_write_mb': self.metrics['write_bytes'][-1] if self.metrics['write_bytes'] else 0,
            'metrics': self.metrics
        }
        
        return results
    
    def analyze(self, test_name: str) -> Dict[str, Any]:
        """Analyze and visualize performance metrics."""
        results = self.stop()
        
        # Generate plots
        self._generate_plots(test_name, results)
        
        # Generate memory profile
        mem_profile = self._generate_memory_profile()
        
        # Save results to CSV
        self._save_metrics(test_name, results)
        
        return {
            **results,
            'memory_profile': mem_profile
        }
    
    def _generate_plots(self, test_name: str, results: Dict[str, Any]) -> None:
        """Generate performance visualization plots."""
        plt.figure(figsize=(15, 10))
        
        # CPU and Memory Usage
        plt.subplot(2, 1, 1)
        x = results['metrics']['timestamps']
        
        # Plot CPU
        ax1 = plt.gca()
        ax1.plot(x, results['metrics']['cpu_percent'], 'r-', label='CPU %')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('CPU %', color='r')
        ax1.tick_params(axis='y', labelcolor='r')
        
        # Plot Memory
        ax2 = ax1.twinx()
        ax2.plot(x, results['metrics']['memory_mb'], 'b-', label='Memory (MB)')
        ax2.set_ylabel('Memory (MB)', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        
        plt.title(f'Performance Metrics - {test_name}')
        plt.tight_layout()
        
        # Save CPU/Memory plot
        cpu_mem_plot = self.output_dir / f"{test_name}_cpu_memory.png"
        plt.savefig(cpu_mem_plot)
        plt.close()
        
        # I/O Plot
        plt.figure(figsize=(15, 5))
        plt.plot(x, results['metrics']['read_bytes'], 'g-', label='Read (MB)')
        plt.plot(x, results['metrics']['write_bytes'], 'm-', label='Write (MB)')
        plt.xlabel('Time (s)')
        plt.ylabel('I/O (MB)')
        plt.title(f'I/O Operations - {test_name}')
        plt.legend()
        plt.tight_layout()
        
        # Save I/O plot
        io_plot = self.output_dir / f"{test_name}_io.png"
        plt.savefig(io_plot)
        plt.close()
    
    def _generate_memory_profile(self) -> Dict[str, Any]:
        """Generate a memory profile of current process."""
        all_objects = muppy.get_objects()
        summary_data = summary.summarize(all_objects)
        
        # Convert to dict for better serialization
        profile = []
        for row in summary_data:
            profile.append({
                'type': str(row[0]),
                'count': row[1],
                'size_mb': row[2] / (1024 * 1024)  # Convert to MB
            })
        
        return profile
    
    def _save_metrics(self, test_name: str, results: Dict[str, Any]) -> None:
        """Save metrics to CSV file."""
        # Create a DataFrame from metrics
        metrics_df = pd.DataFrame({
            'timestamp': results['metrics']['timestamps'],
            'cpu_percent': results['metrics']['cpu_percent'],
            'memory_mb': results['metrics']['memory_mb'],
            'memory_percent': results['metrics']['memory_percent'],
            'read_mb': results['metrics'].get('read_bytes', [0] * len(results['metrics']['timestamps'])),
            'write_mb': results['metrics'].get('write_bytes', [0] * len(results['metrics']['timestamps']))
        })
        
        # Save to CSV
        csv_path = self.output_dir / f"{test_name}_metrics.csv"
        metrics_df.to_csv(csv_path, index=False)
        
        # Save summary
        summary_path = self.output_dir / f"{test_name}_summary.json"
        summary = {
            'test_name': test_name,
            'duration_seconds': results['duration'],
            'avg_cpu_percent': results['avg_cpu'],
            'max_memory_mb': results['max_memory_mb'],
            'avg_memory_mb': results['avg_memory_mb'],
            'total_read_mb': results['total_read_mb'],
            'total_write_mb': results['total_write_mb'],
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        import json
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

def compare_performance_runs(test_names: List[str], results_dir: str = "test-results/performance") -> None:
    """Compare performance across multiple test runs."""
    summaries = []
    
    for name in test_names:
        summary_path = Path(results_dir) / f"{name}_summary.json"
        if summary_path.exists():
            import json
            with open(summary_path, 'r') as f:
                summary = json.load(f)
                summaries.append(summary)
    
    if not summaries:
        print("No performance data found for comparison")
        return
    
    # Create comparison DataFrame
    df = pd.DataFrame(summaries)
    
    # Plot comparison
    plt.figure(figsize=(15, 8))
    
    # Duration comparison
    plt.subplot(2, 2, 1)
    sns.barplot(x='test_name', y='duration_seconds', data=df)
    plt.title('Execution Time (s)')
    plt.xticks(rotation=45)
    
    # Memory comparison
    plt.subplot(2, 2, 2)
    sns.barplot(x='test_name', y='max_memory_mb', data=df)
    plt.title('Peak Memory Usage (MB)')
    plt.xticks(rotation=45)
    
    # CPU comparison
    plt.subplot(2, 2, 3)
    sns.barplot(x='test_name', y='avg_cpu_percent', data=df)
    plt.title('Average CPU Usage (%)')
    plt.xticks(rotation=45)
    
    # I/O comparison
    plt.subplot(2, 2, 4)
    df_melted = df.melt(id_vars=['test_name'], 
                        value_vars=['total_read_mb', 'total_write_mb'],
                        var_name='io_type', value_name='mb')
    sns.barplot(x='test_name', y='mb', hue='io_type', data=df_melted)
    plt.title('Total I/O (MB)')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save comparison plot
    comparison_plot = Path(results_dir) / "performance_comparison.png"
    plt.savefig(comparison_plot)
    plt.close()
    
    # Print summary table
    print("\nPerformance Comparison Summary:")
    print("-" * 80)
    print(df[['test_name', 'duration_seconds', 'max_memory_mb', 
              'avg_cpu_percent', 'total_read_mb', 'total_write_mb']].to_string(index=False))
    print(f"\nComparison plot saved to: {comparison_plot}")
