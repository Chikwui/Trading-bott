"""
Performance visualization and trend analysis for ML pipeline tests.
"""
import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter

# Configure plotting style
plt.style.use('seaborn')
sns.set_palette("husl")

class PerformanceVisualizer:
    """Visualize performance metrics and trends."""
    
    def __init__(self, results_dir: str = "test-results"):
        """Initialize with results directory."""
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / "reports"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_historical_data(self) -> pd.DataFrame:
        """Load historical performance data from JSON files."""
        data = []
        
        # Find all benchmark result files
        for result_file in sorted(self.results_dir.rglob("benchmark.json")):
            with open(result_file, 'r') as f:
                benchmark_data = json.load(f)
                timestamp = datetime.fromisoformat(benchmark_data.get('datetime', datetime.now().isoformat()))
                
                for bench in benchmark_data.get('benchmarks', []):
                    data.append({
                        'timestamp': timestamp,
                        'test': bench['name'],
                        'duration': bench['stats']['mean'],
                        'min': bench['stats']['min'],
                        'max': bench['stats']['max'],
                        'stddev': bench['stats']['stddev'],
                        'iterations': bench['stats']['rounds']
                    })
        
        # Load additional metrics from summary files
        for summary_file in sorted(self.results_dir.rglob("*_summary.json")):
            if "baseline" in str(summary_file):
                continue
                
            with open(summary_file, 'r') as f:
                summary = json.load(f)
                timestamp = datetime.fromisoformat(summary.get('timestamp', datetime.now().isoformat()))
                test_name = summary_file.stem.replace("_summary", "")
                
                data.append({
                    'timestamp': timestamp,
                    'test': test_name,
                    'duration': summary.get('duration_seconds'),
                    'avg_cpu': summary.get('avg_cpu_percent'),
                    'max_memory_mb': summary.get('max_memory_mb'),
                    'total_read_mb': summary.get('total_read_mb'),
                    'total_write_mb': summary.get('total_write_mb')
                })
        
        return pd.DataFrame(data)
    
    def plot_performance_trends(self, df: pd.DataFrame) -> None:
        """Generate performance trend plots."""
        if df.empty:
            print("No performance data available for visualization")
            return
        
        # Pivot data for plotting
        metrics = ['duration', 'max_memory_mb', 'avg_cpu']
        
        for metric in metrics:
            if metric not in df.columns:
                continue
                
            plt.figure(figsize=(15, 8))
            
            # Filter out tests with insufficient data
            test_counts = df[df[metric].notna()].groupby('test').size()
            valid_tests = test_counts[test_counts > 1].index.tolist()
            
            if not valid_tests:
                continue
                
            # Plot each test separately
            for test in valid_tests:
                test_data = df[(df['test'] == test) & (df[metric].notna())].sort_values('timestamp')
                if len(test_data) < 2:
                    continue
                    
                plt.plot(
                    test_data['timestamp'],
                    test_data[metric],
                    'o-',
                    label=test,
                    alpha=0.7
                )
            
            # Format plot
            plt.title(f'Performance Trend - {metric.replace("_", " ").title()}')
            plt.xlabel('Time')
            plt.ylabel(metric.replace('_', ' ').title())
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            
            # Save figure
            output_file = self.output_dir / f"trend_{metric}.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
    
    def plot_correlation_heatmap(self, df: pd.DataFrame) -> None:
        """Generate correlation heatmap between metrics."""
        # Select numeric columns for correlation
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            return
            
        corr = df[numeric_cols].corr()
        
        # Generate heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            corr,
            annot=True,
            fmt=".2f",
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8}
        )
        
        plt.title('Metric Correlation Matrix')
        plt.tight_layout()
        
        # Save figure
        output_file = self.output_dir / "correlation_heatmap.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_distribution(self, df: pd.DataFrame, metric: str = 'duration') -> None:
        """Generate distribution plot for a metric."""
        if metric not in df.columns:
            return
            
        # Filter out tests with insufficient data
        test_means = df.groupby('test')[metric].agg(['mean', 'count'])
        valid_tests = test_means[test_means['count'] > 1].index.tolist()
        
        if not valid_tests:
            return
            
        # Create distribution plot
        plt.figure(figsize=(15, 8))
        
        for test in valid_tests:
            test_data = df[(df['test'] == test) & (df[metric].notna())][metric]
            sns.kdeplot(test_data, label=test, alpha=0.6, linewidth=2)
        
        plt.title(f'Distribution of {metric.replace("_", " ").title()}')
        plt.xlabel(metric.replace('_', ' ').title())
        plt.ylabel('Density')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save figure
        output_file = self.output_dir / f"distribution_{metric}.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_html_report(self) -> str:
        """Generate an HTML performance report."""
        # Load data
        df = self.load_historical_data()
        
        # Generate plots
        self.plot_performance_trends(df)
        self.plot_correlation_heatmap(df)
        self.plot_distribution(df, 'duration')
        self.plot_distribution(df, 'max_memory_mb')
        
        # Create HTML report
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>ML Pipeline Performance Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2 { color: #2c3e50; }
                .container { max-width: 1200px; margin: 0 auto; }\n                .plot { margin: 30px 0; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
                .plot img { max-width: 100%; height: auto; }
                .summary { background: #f8f9fa; padding: 15px; border-radius: 5px; margin: 20px 0; }
                table { border-collapse: collapse; width: 100%; margin: 20px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>ML Pipeline Performance Report</h1>
                <p>Generated on: {date}</p>
                
                <div class="summary">
                    <h2>Summary</h2>
                    <p>Total test runs: {total_runs}</p>
                    <p>Unique tests: {unique_tests}</p>
                    <p>Time range: {min_date} to {max_date}</p>
                </div>
        """.format(
            date=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            total_runs=len(df) if not df.empty else 0,
            unique_tests=df['test'].nunique() if not df.empty else 0,
            min_date=df['timestamp'].min().strftime('%Y-%m-%d') if not df.empty else 'N/A',
            max_date=df['timestamp'].max().strftime('%Y-%m-%d') if not df.empty else 'N/A'
        )
        
        # Add trend plots
        if not df.empty:
            html += """
                <h2>Performance Trends</h2>
                <div class="plot">
                    <h3>Execution Time</h3>
                    <img src="trend_duration.png" alt="Execution Time Trend">
                </div>
                <div class="plot">
                    <h3>Memory Usage</h3>
                    <img src="trend_max_memory_mb.png" alt="Memory Usage Trend">
                </div>
                <div class="plot">
                    <h3>CPU Usage</h3>
                    <img src="trend_avg_cpu.png" alt="CPU Usage Trend">
                </div>
                
                <h2>Metric Distributions</h2>
                <div class="plot">
                    <h3>Execution Time Distribution</h3>
                    <img src="distribution_duration.png" alt="Execution Time Distribution">
                </div>
                <div class="plot">
                    <h3>Memory Usage Distribution</h3>
                    <img src="distribution_max_memory_mb.png" alt="Memory Usage Distribution">
                </div>
                
                <h2>Metric Correlations</h2>
                <div class="plot">
                    <img src="correlation_heatmap.png" alt="Metric Correlations">
                </div>
            """
        
        # Close HTML
        html += """
            </div>
        </body>
        </html>
        """
        
        # Save HTML report
        output_file = self.output_dir / "performance_report.html"
        with open(output_file, 'w') as f:
            f.write(html)
        
        return str(output_file)

def main():
    """Main entry point for performance visualization."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize performance metrics')
    parser.add_argument('--results-dir', default='test-results',
                      help='Directory containing test results')
    parser.add_argument('--output-dir', default=None,
                      help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = PerformanceVisualizer(
        results_dir=args.results_dir
    )
    
    # Generate and open report
    report_file = visualizer.generate_html_report()
    print(f"Performance report generated: {report_file}")

if __name__ == "__main__":
    main()
