"""
Advanced testing for edge cases, performance, and stress testing of the ML Pipeline.
"""
import os
import time
import gc
import random
import string
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

# Test configuration
STRESS_TEST_MULTIPLIER = int(os.getenv('STRESS_TEST_MULTIPLIER', '1'))
PERF_TEST_ITERATIONS = 10
LARGE_DATASET_SIZE = 100000  # Base size, multiplied by STRESS_TEST_MULTIPLIER

class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_feature_store(self, test_environment):
        """Test behavior with empty feature store."""
        fs = test_environment['feature_store']
        
        # Test getting features from non-existent view
        with pytest.raises(ValueError):
            fs.get_features("non_existent_view")
        
        # Test writing empty DataFrame
        empty_df = pd.DataFrame()
        with pytest.raises(ValueError):
            fs.write_features(
                feature_view_name="test_empty",
                data=empty_df,
                timestamp_column='timestamp'
            )
    
    def test_invalid_model_registration(self, test_environment):
        """Test invalid model registration scenarios."""
        registry = test_environment['registry']
        
        # Test registering None model
        with pytest.raises(ValueError):
            registry.register_model(
                name="test_none_model",
                model=None,
                metrics={"accuracy": 0.9}
            )
        
        # Test registering with invalid metrics
        with pytest.raises(ValueError):
            registry.register_model(
                name="test_invalid_metrics",
                model=RandomForestClassifier(),
                metrics={"invalid_metric": "not_a_number"}
            )
    
    def test_monitoring_edge_cases(self, test_environment):
        """Test monitoring with edge cases."""
        monitor = test_environment['monitor']
        
        # Test logging very large values
        monitor.log_metric("large_value_test", float('inf'))
        monitor.log_metric("nan_value_test", float('nan'))
        
        # Test logging with special characters in names
        special_name = "test_metric_!@#$%^&*()_+{}|:<>?"
        monitor.log_metric(special_name, 1.0)
        
        # Test logging with very long strings
        long_string = "x" * 10000
        monitor.log_parameter("long_string_test", long_string)


class TestPerformance:
    """Performance benchmarking tests."""
    
    @pytest.fixture(scope="class")
    def large_dataset(self):
        """Generate a large dataset for performance testing."""
        size = LARGE_DATASET_SIZE * STRESS_TEST_MULTIPLIER
        dates = pd.date_range(end=datetime.now(), periods=size)
        
        data = {
            'timestamp': dates,
            'feature1': np.random.rand(size),
            'feature2': np.random.rand(size) * 100,
            'feature3': np.random.choice(['A', 'B', 'C', 'D'], size=size),
            'target': np.random.randint(0, 2, size=size)
        }
        
        df = pd.DataFrame(data)
        df['feature4'] = df['feature1'] * df['feature2']  # Derived feature
        
        return df
    
    def test_feature_store_write_performance(self, test_environment, large_dataset, benchmark):
        """Benchmark feature store write performance."""
        fs = test_environment['feature_store']
        
        # Create a test feature view
        fs.create_feature_view(
            name="perf_test_features",
            features=[
                {"name": "feature1", "dtype": "float"},
                {"name": "feature2", "dtype": "float"},
                {"name": "feature3", "dtype": "str"},
                {"name": "feature4", "dtype": "float"},
                {"name": "target", "dtype": "int"}
            ]
        )
        
        # Benchmark the write operation
        def write_operation():
            fs.write_features(
                feature_view_name="perf_test_features",
                data=large_dataset,
                timestamp_column='timestamp'
            )
        
        # Run benchmark
        benchmark.pedantic(
            write_operation,
            setup=gc.collect,
            rounds=3,
            iterations=1
        )
    
    def test_model_training_throughput(self, test_environment, large_dataset, benchmark):
        """Benchmark model training throughput."""
        from sklearn.model_selection import train_test_split
        
        # Prepare data
        X = large_dataset[['feature1', 'feature2', 'feature4']]
        y = large_dataset['target']
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        
        def train_model():
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                n_jobs=-1,
                random_state=42
            )
            model.fit(X_train, y_train)
            return model
        
        # Run benchmark
        result = benchmark.pedantic(
            train_model,
            setup=gc.collect,
            rounds=3,
            iterations=1
        )
        
        # Log performance metrics
        test_environment['monitor'].log_metrics({
            'training_throughput': len(X_train) / benchmark.stats['mean'],
            'memory_usage': X_train.memory_usage().sum() / (1024 ** 2)  # MB
        }, prefix='performance_')


class TestStress:
    """Stress testing for the ML pipeline."""
    
    @pytest.mark.stress
    def test_concurrent_feature_writes(self, test_environment):
        """Test concurrent writes to the feature store."""
        import concurrent.futures
        
        fs = test_environment['feature_store']
        num_workers = 10 * STRESS_TEST_MULTIPLIER
        num_writes = 100 * STRESS_TEST_MULTIPLIER
        
        # Create a test feature view
        fs.create_feature_view(
            name="stress_test_features",
            features=[
                {"name": "value", "dtype": "float"},
                {"name": "worker_id", "dtype": "int"}
            ]
        )
        
        def worker(worker_id):
            """Worker function for concurrent writes."""
            for i in range(num_writes):
                df = pd.DataFrame({
                    'timestamp': [datetime.now()],
                    'value': [random.random()],
                    'worker_id': [worker_id]
                })
                fs.write_features(
                    feature_view_name="stress_test_features",
                    data=df,
                    timestamp_column='timestamp'
                )
        
        # Run concurrent workers
        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(worker, i) for i in range(num_workers)]
            concurrent.futures.wait(futures)
        
        duration = time.time() - start_time
        
        # Log performance metrics
        test_environment['monitor'].log_metrics({
            'concurrent_writes_throughput': (num_workers * num_writes) / duration,
            'concurrent_writes_duration': duration,
            'concurrent_workers': num_workers,
            'total_writes': num_workers * num_writes
        }, prefix='stress_')
        
        # Verify all writes succeeded
        features = fs.get_features("stress_test_features")
        assert len(features) >= num_workers * num_writes
    
    @pytest.mark.stress
    def test_large_model_registration(self, test_environment):
        """Test registration of very large models."""
        registry = test_environment['registry']
        monitor = test_environment['monitor']
        
        # Create a very large model (in terms of parameters)
        class LargeModel:
            def __init__(self):
                # Large attribute to increase memory usage
                self.large_array = np.random.rand(10000, 1000)  # ~80MB
                self.coef_ = np.random.rand(10000)
                self.intercept_ = 0.0
            
            def predict(self, X):
                return np.ones(len(X)) if len(X.shape) > 1 else 1
        
        # Test registration and retrieval
        model_name = f"large_model_{int(time.time())}"
        
        start_time = time.time()
        version = registry.register_model(
            name=model_name,
            model=LargeModel(),
            metrics={"test_metric": 0.9},
            model_type="test"
        )
        register_time = time.time() - start_time
        
        # Test retrieval
        start_time = time.time()
        model_info = registry.get_model(model_name, version)
        load_time = time.time() - start_time
        
        # Log performance metrics
        monitor.log_metrics({
            'large_model_register_time': register_time,
            'large_model_load_time': load_time,
            'large_model_size': 10000 * 1000 * 8 / (1024 ** 2)  # MB
        }, prefix='stress_')
        
        # Clean up
        registry.delete_model(model_name, version)
    
    @pytest.mark.stress
    def test_high_frequency_metrics(self, test_environment):
        """Test high-frequency metric logging."""
        monitor = test_environment['monitor']
        num_metrics = 10000 * STRESS_TEST_MULTIPLIER
        
        start_time = time.time()
        
        # Log a large number of metrics
        for i in range(num_metrics):
            monitor.log_metric(
                name=f"high_freq_metric_{i % 100}",  # 100 unique metric names
                value=random.random(),
                tags={"iteration": i}
            )
        
        duration = time.time() - start_time
        
        # Log performance metrics
        monitor.log_metrics({
            'metrics_per_second': num_metrics / duration,
            'total_metrics_logged': num_metrics,
            'duration': duration
        }, prefix='stress_')


# Add a custom marker for stress tests
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "stress: mark test as a stress test"
    )
