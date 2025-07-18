"""
Tests for performance utilities in the ML pipeline.
"""
import os
import time
import numpy as np
import pandas as pd
from pathlib import Path

import pytest

from tests.ml.utils.performance_utils import PerformanceProfiler, compare_performance_runs

class TestPerformanceUtils:
    """Test performance profiling and monitoring utilities."""
    
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Setup test environment."""
        self.output_dir = tmp_path / "performance"
        self.output_dir.mkdir()
        self.profiler = PerformanceProfiler(output_dir=str(self.output_dir))
    
    def test_profiler_initialization(self):
        """Test profiler initialization."""
        assert self.profiler is not None
        assert self.profiler.output_dir.exists()
    
    def test_profiler_measure_decorator(self):
        """Test the measure decorator."""
        @self.profiler.measure
        def test_function():
            # Simulate work
            time.sleep(0.1)
            return "test"
        
        result = test_function()
        assert result == "test"
        
        # Check that metrics were recorded
        assert len(self.profiler.metrics['timestamps']) > 0
        assert len(self.profiler.metrics['cpu_percent']) > 0
        assert len(self.profiler.metrics['memory_mb']) > 0
    
    def test_manual_profiling(self):
        """Test manual profiling with start/stop."""
        self.profiler.start()
        
        # Simulate some work
        data = np.random.rand(1000, 1000)
        _ = np.dot(data, data.T)
        time.sleep(0.1)
        
        results = self.profiler.stop()
        
        # Check results
        assert 'duration' in results
        assert 'avg_cpu' in results
        assert 'max_memory_mb' in results
        assert 'metrics' in results
        assert results['duration'] > 0
    
    def test_analyze_generates_output_files(self):
        """Test that analyze generates output files."""
        self.profiler.start()
        
        # Simulate some work
        data = pd.DataFrame({
            'a': np.random.rand(1000),
            'b': np.random.rand(1000)
        })
        _ = data.corr()
        time.sleep(0.1)
        
        # Run analysis
        test_name = "test_analysis"
        results = self.profiler.analyze(test_name)
        
        # Check that output files were created
        assert (self.output_dir / f"{test_name}_cpu_memory.png").exists()
        assert (self.output_dir / f"{test_name}_io.png").exists()
        assert (self.output_dir / f"{test_name}_metrics.csv").exists()
        assert (self.output_dir / f"{test_name}_summary.json").exists()
        
        # Check results
        assert 'memory_profile' in results
        assert len(results['memory_profile']) > 0
    
    def test_compare_performance_runs(self, tmp_path):
        """Test performance comparison across multiple runs."""
        # Create test summary files
        test_names = ["test1", "test2", "test3"]
        
        for i, name in enumerate(test_names):
            summary = {
                'test_name': name,
                'duration_seconds': 1.0 + (i * 0.5),
                'avg_cpu_percent': 20.0 + (i * 10),
                'max_memory_mb': 100.0 + (i * 50),
                'avg_memory_mb': 80.0 + (i * 40),
                'total_read_mb': 10.0 + (i * 5),
                'total_write_mb': 5.0 + (i * 2),
                'timestamp': "2023-01-01 12:00:00"
            }
            
            # Write summary file
            import json
            with open(tmp_path / f"{name}_summary.json", 'w') as f:
                json.dump(summary, f)
        
        # Run comparison
        compare_performance_runs(test_names, str(tmp_path))
        
        # Check that comparison plot was created
        assert (tmp_path / "performance_comparison.png").exists()


class TestPerformanceIntegration:
    """Integration tests for performance monitoring."""
    
    def test_feature_store_performance(self, test_environment, tmp_path):
        """Test performance of feature store operations."""
        from core.ml.feature_store import get_feature_store
        
        # Setup profiler
        profiler = PerformanceProfiler(output_dir=str(tmp_path / "feature_store_perf"))
        fs = test_environment['feature_store']
        
        # Create test data
        n_samples = 10000
        data = pd.DataFrame({
            'timestamp': pd.date_range(end=pd.Timestamp.now(), periods=n_samples),
            'feature1': np.random.randn(n_samples),
            'feature2': np.random.rand(n_samples) * 100,
            'target': np.random.randint(0, 2, n_samples)
        })
        
        # Create feature view
        fs.create_feature_view(
            name="perf_test_features",
            features=[
                {"name": "feature1", "dtype": "float"},
                {"name": "feature2", "dtype": "float"},
                {"name": "target", "dtype": "int"}
            ]
        )
        
        # Profile write operation
        @profiler.measure
        def write_features():
            fs.write_features(
                feature_view_name="perf_test_features",
                data=data,
                timestamp_column='timestamp'
            )
        
        write_features()
        
        # Profile read operation
        @profiler.measure
        def read_features():
            return fs.get_features("perf_test_features")
        
        features = read_features()
        assert len(features) == n_samples
        
        # Analyze and save results
        results = profiler.analyze("feature_store_operations")
        
        # Basic assertions
        assert results['duration'] > 0
        assert results['max_memory_mb'] > 0
        assert results['avg_cpu'] >= 0
        
        # Log results
        test_environment['monitor'].log_metrics({
            'feature_store_write_throughput': n_samples / results['duration'],
            'feature_store_memory_usage_mb': results['max_memory_mb']
        }, prefix='performance_')
