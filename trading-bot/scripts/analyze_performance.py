#!/usr/bin/env python3
"""
Performance analysis and regression detection for ML pipeline tests.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime

# Default configuration
DEFAULT_CONFIG = {
    "baseline_branch": "main",
    "performance_threshold": 1.2,  # 20% regression threshold
    "significance_level": 0.05,    # Statistical significance level
    "min_runs_for_regression": 5,  # Minimum runs to detect regression
    "metrics_to_track": [
        "duration",
        "avg_cpu",
        "max_memory_mb",
        "total_read_mb",
        "total_write_mb"
    ]
}

class PerformanceAnalyzer:
    """Analyze performance test results and detect regressions."""
    
    def __init__(self, results_dir: str = "test-results", config: Optional[Dict] = None):
        """Initialize the analyzer with results directory and configuration."""
        self.results_dir = Path(results_dir)
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.baseline_data = self._load_baseline_data()
        self.current_results = {}
    
    def _load_baseline_data(self) -> Dict[str, Any]:
        """Load baseline performance data."""
        baseline_file = self.results_dir / "baseline" / "performance_baseline.json"
        if baseline_file.exists():
            with open(baseline_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_baseline_data(self, data: Dict[str, Any]) -> None:
        """Save new baseline data."""
        baseline_dir = self.results_dir / "baseline"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        
        baseline_file = baseline_dir / "performance_baseline.json"
        with open(baseline_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_current_results(self) -> Dict[str, Any]:
        """Load current test results."""
        results = {}
        
        # Find all benchmark result files
        for result_file in self.results_dir.rglob("benchmark.json"):
            with open(result_file, 'r') as f:
                data = json.load(f)
                
                # Extract benchmark name and metrics
                for benchmark in data.get('benchmarks', []):
                    name = benchmark['name']
                    if name not in results:
                        results[name] = []
                    
                    results[name].append({
                        'duration': benchmark['stats']['mean'],
                        'min': benchmark['stats']['min'],
                        'max': benchmark['stats']['max'],
                        'stddev': benchmark['stats']['stddev'],
                        'iterations': benchmark['stats']['rounds'],
                        'timestamp': datetime.now().isoformat()
                    })
        
        # Load any additional metrics from test results
        for summary_file in self.results_dir.rglob("*_summary.json"):
            if "baseline" in str(summary_file):
                continue
                
            with open(summary_file, 'r') as f:
                test_name = summary_file.stem.replace("_summary", "")
                if test_name not in results:
                    results[test_name] = []
                
                data = json.load(f)
                results[test_name].append({
                    'duration': data.get('duration_seconds'),
                    'avg_cpu': data.get('avg_cpu_percent'),
                    'max_memory_mb': data.get('max_memory_mb'),
                    'total_read_mb': data.get('total_read_mb'),
                    'total_write_mb': data.get('total_write_mb'),
                    'timestamp': data.get('timestamp', datetime.now().isoformat())
                })
        
        self.current_results = results
        return results
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance and detect regressions."""
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'regressions': [],
            'improvements': [],
            'metrics': {},
            'summary': {}
        }
        
        for test_name, runs in self.current_results.items():
            if not runs:
                continue
                
            # Calculate statistics for current runs
            df = pd.DataFrame(runs)
            current_stats = {
                'count': len(df),
                'mean': df['duration'].mean(),
                'std': df['duration'].std(),
                'min': df['duration'].min(),
                'max': df['duration'].max(),
                'p50': df['duration'].median(),
                'p95': df['duration'].quantile(0.95)
            }
            
            # Compare with baseline if available
            baseline = self.baseline_data.get(test_name, {})
            if baseline:
                baseline_mean = baseline.get('mean', 0)
                if baseline_mean > 0:
                    ratio = current_stats['mean'] / baseline_mean
                    is_regression = ratio > self.config['performance_threshold']
                    
                    # Perform statistical test if enough data points
                    is_significant = False
                    if len(runs) >= 3 and 'values' in baseline and len(baseline['values']) >= 3:
                        _, p_value = stats.ttest_ind(
                            baseline['values'],
                            df['duration'].tolist(),
                            equal_var=False
                        )
                        is_significant = p_value < self.config['significance_level']
                    
                    if is_regression and is_significant:
                        analysis['regressions'].append({
                            'test': test_name,
                            'baseline': baseline_mean,
                            'current': current_stats['mean'],
                            'ratio': ratio,
                            'p_value': float(p_value) if 'p_value' in locals() else None
                        })
                    elif ratio < 1.0 / self.config['performance_threshold'] and is_significant:
                        analysis['improvements'].append({
                            'test': test_name,
                            'baseline': baseline_mean,
                            'current': current_stats['mean'],
                            'ratio': ratio,
                            'p_value': float(p_value) if 'p_value' in locals() else None
                        })
            
            # Update metrics
            analysis['metrics'][test_name] = {
                'current': current_stats,
                'baseline': baseline
            }
        
        # Generate summary
        analysis['summary'] = {
            'total_tests': len(self.current_results),
            'regression_count': len(analysis['regressions']),
            'improvement_count': len(analysis['improvements']),
            'regression_percentage': len(analysis['regressions']) / len(self.current_results) * 100 if self.current_results else 0,
            'threshold': self.config['performance_threshold']
        }
        
        return analysis
    
    def update_baseline(self) -> None:
        """Update baseline with current results."""
        new_baseline = {}
        
        for test_name, runs in self.current_results.items():
            if not runs:
                continue
                
            df = pd.DataFrame(runs)
            values = df['duration'].tolist()
            
            new_baseline[test_name] = {
                'mean': float(df['duration'].mean()),
                'std': float(df['duration'].std()),
                'min': float(df['duration'].min()),
                'max': float(df['duration'].max()),
                'count': len(df),
                'last_updated': datetime.now().isoformat(),
                'values': values[:100]  # Keep last 100 values for statistics
            }
        
        self._save_baseline_data(new_baseline)
        self.baseline_data = new_baseline
    
    def generate_report(self, analysis: Dict[str, Any], output_file: Optional[str] = None) -> str:
        """Generate a markdown report of the analysis."""
        report = ["# Performance Analysis Report\n"]
        
        # Summary
        report.append("## Summary\n")
        report.append(f"- **Total Tests**: {analysis['summary']['total_tests']}")
        report.append(f"- **Regressions**: {analysis['summary']['regression_count']}")
        report.append(f"- **Improvements**: {analysis['summary']['improvement_count']}")
        report.append(f"- **Threshold**: {self.config['performance_threshold']:.1f}x\n")
        
        # Regressions
        if analysis['regressions']:
            report.append("## ⚠️ Performance Regressions\n")
            report.append("| Test | Baseline (s) | Current (s) | Ratio | p-value |")
            report.append("|------|-------------|-------------|-------|---------|")
            
            for reg in analysis['regressions']:
                report.append(
                    f"| {reg['test']} | "
                    f"{reg['baseline']:.4f} | "
                    f"{reg['current']:.4f} | "
                    f"{reg['ratio']:.2f}x | "
                    f"{reg.get('p_value', 'N/A')} |"
                )
        
        # Improvements
        if analysis['improvements']:
            report.append("\n## 🚀 Performance Improvements\n")
            report.append("| Test | Baseline (s) | Current (s) | Ratio | p-value |")
            report.append("|------|-------------|-------------|-------|---------|")
            
            for imp in analysis['improvements']:
                report.append(
                    f"| {imp['test']} | "
                    f"{imp['baseline']:.4f} | "
                    f"{imp['current']:.4f} | "
                    f"{imp['ratio']:.2f}x | "
                    f"{imp.get('p_value', 'N/A')} |"
                )
        
        # Detailed metrics
        report.append("\n## Detailed Metrics\n")
        for test_name, metrics in analysis['metrics'].items():
            report.append(f"### {test_name}\n")
            
            if 'current' in metrics:
                report.append("**Current Run**")
                report.append("```")
                for k, v in metrics['current'].items():
                    if isinstance(v, (int, float)):
                        report.append(f"{k}: {v:.4f}")
                    else:
                        report.append(f"{k}: {v}")
                report.append("```\n")
            
            if metrics.get('baseline'):
                report.append("**Baseline**")
                report.append("```")
                for k, v in metrics['baseline'].items():
                    if k == 'values':
                        continue
                    if isinstance(v, (int, float)):
                        report.append(f"{k}: {v:.4f}")
                    else:
                        report.append(f"{k}: {v}")
                report.append("```")
            
            report.append("")
        
        # Save report
        report_text = "\n".join(report)
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
        
        return report_text

def main():
    """Main entry point for performance analysis."""
    parser = argparse.ArgumentParser(description='Analyze performance test results')
    parser.add_argument('--results-dir', default='test-results',
                      help='Directory containing test results')
    parser.add_argument('--threshold', type=float, default=1.2,
                      help='Performance regression threshold (default: 1.2)')
    parser.add_argument('--update-baseline', action='store_true',
                      help='Update baseline with current results')
    parser.add_argument('--output', default='performance-report.md',
                      help='Output file for the report')
    
    args = parser.parse_args()
    
    # Configure and run analysis
    analyzer = PerformanceAnalyzer(
        results_dir=args.results_dir,
        config={'performance_threshold': args.threshold}
    )
    
    # Load and analyze results
    analyzer.load_current_results()
    analysis = analyzer.analyze_performance()
    
    # Generate report
    report = analyzer.generate_report(analysis, args.output)
    print(f"Report generated: {args.output}")
    
    # Update baseline if requested
    if args.update_baseline:
        analyzer.update_baseline()
        print("Baseline updated with current results")
    
    # Exit with error code if regressions found
    if analysis['regressions']:
        print("\nPerformance regressions detected!")
        sys.exit(1)
    
    print("\nNo significant performance regressions detected")

if __name__ == "__main__":
    main()
