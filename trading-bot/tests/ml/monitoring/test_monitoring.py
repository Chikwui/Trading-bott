"""
Tests for the monitoring module.
"""
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

from core.ml.monitoring import get_monitor, Monitor

class TestMonitor(unittest.TestCase):
    """Test cases for the monitoring module."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.monitor: Monitor = get_monitor(log_dir=os.path.join(self.test_dir, 'monitoring'))
        
        # Create test data
        self.now = datetime.utcnow()
        self.test_metrics = {
            'accuracy': 0.95,
            'precision': 0.92,
            'recall': 0.88,
            'f1': 0.90
        }
        
        self.test_params = {
            'learning_rate': 0.001,
            'batch_size': 32,
            'num_epochs': 100
        }
        
        self.test_dist = np.random.normal(0, 1, 1000)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_log_metric(self):
        """Test logging a single metric."""
        # Log a metric
        self.monitor.log_metric('test_metric', 0.5)
        
        # Check that the metric file was created
        metric_files = list((self.test_dir / 'monitoring' / 'metrics').glob('test_metric.csv'))
        self.assertEqual(len(metric_files), 1)
        
        # Check the content
        df = pd.read_csv(metric_files[0])
        self.assertIn('value', df.columns)
        self.assertEqual(len(df), 1)
        self.assertAlmostEqual(df['value'].iloc[0], 0.5)
    
    def test_log_metrics(self):
        """Test logging multiple metrics at once."""
        # Log multiple metrics
        self.monitor.log_metrics(self.test_metrics, prefix='test')
        
        # Check that all metric files were created
        metrics_dir = self.test_dir / 'monitoring' / 'metrics'
        self.assertTrue((metrics_dir / 'test_accuracy.csv').exists())
        self.assertTrue((metrics_dir / 'test_precision.csv').exists())
        self.assertTrue((metrics_dir / 'test_recall.csv').exists())
        self.assertTrue((metrics_dir / 'test_f1.csv').exists())
    
    def test_log_parameter(self):
        """Test logging a single parameter."""
        # Log a parameter
        self.monitor.log_parameter('test_param', 'test_value')
        
        # Check that the parameter file was created
        param_file = self.test_dir / 'monitoring' / 'params' / 'test_param.json'
        self.assertTrue(param_file.exists())
        
        # Check the content
        import json
        with open(param_file, 'r') as f:
            data = json.load(f)
        self.assertEqual(data['name'], 'test_param')
        self.assertEqual(data['value'], 'test_value')
    
    def test_log_parameters(self):
        """Test logging multiple parameters at once."""
        # Log multiple parameters
        self.monitor.log_parameters(self.test_params)
        
        # Check that all parameter files were created
        params_dir = self.test_dir / 'monitoring' / 'params'
        self.assertTrue((params_dir / 'learning_rate.json').exists())
        self.assertTrue((params_dir / 'batch_size.json').exists())
        self.assertTrue((params_dir / 'num_epochs.json').exists())
    
    def test_log_artifact(self):
        """Test logging an artifact."""
        # Create a test file
        test_file = os.path.join(self.test_dir, 'test_artifact.txt')
        with open(test_file, 'w') as f:
            f.write('test artifact content')
        
        # Log the artifact
        self.monitor.log_artifact(test_file, 'test_artifacts/test_artifact.txt')
        
        # Check that the artifact was copied
        artifact_path = self.test_dir / 'monitoring' / 'artifacts' / 'test_artifacts' / 'test_artifact.txt'
        self.assertTrue(artifact_path.exists())
        
        # Check the content
        with open(artifact_path, 'r') as f:
            content = f.read()
        self.assertEqual(content, 'test artifact content')
    
    def test_log_prediction(self):
        """Test logging a prediction."""
        # Log a prediction
        inputs = [1, 2, 3, 4, 5]
        outputs = [0.8, 0.2]
        labels = [1]
        
        self.monitor.log_prediction(
            model_name='test_model',
            version='1.0',
            inputs=inputs,
            outputs=outputs,
            labels=labels,
            metrics={'accuracy': 0.9}
        )
        
        # Check that the prediction was logged
        pred_dir = self.test_dir / 'monitoring' / 'predictions' / 'test_model' / '1.0'
        self.assertTrue(pred_dir.exists())
        
        # Check that a prediction file was created
        pred_files = list(pred_dir.glob('*.json'))
        self.assertGreaterEqual(len(pred_files), 1)
        
        # Check the content of the first prediction file
        import json
        with open(pred_files[0], 'r') as f:
            data = json.load(f)
        
        self.assertEqual(data['model_name'], 'test_model')
        self.assertEqual(data['version'], '1.0')
        self.assertEqual(data['metrics']['accuracy'], 0.9)
    
    def test_log_data_drift(self):
        """Test logging data drift information."""
        # Generate test distributions
        reference_dist = np.random.normal(0, 1, 1000)
        current_dist = np.random.normal(0.5, 1.2, 1000)
        
        # Log data drift
        self.monitor.log_data_drift(
            feature_name='test_feature',
            reference_dist=reference_dist,
            current_dist=current_dist,
            drift_score=0.75
        )
        
        # Check that the drift information was logged
        drift_dir = self.test_dir / 'monitoring' / 'drift' / 'test_feature'
        self.assertTrue(drift_dir.exists())
        
        # Check that a drift file was created
        drift_files = list(drift_dir.glob('*.json'))
        self.assertGreaterEqual(len(drift_files), 1)
        
        # Check the content of the drift file
        import json
        with open(drift_files[0], 'r') as f:
            data = json.load(f)
        
        self.assertEqual(data['feature_name'], 'test_feature')
        self.assertEqual(data['drift_score'], 0.75)
        self.assertIn('reference_stats', data)
        self.assertIn('current_stats', data)
    
    def test_log_performance_metrics(self):
        """Test logging performance metrics."""
        # Log performance metrics
        self.monitor.log_performance_metrics(
            model_name='test_model',
            version='1.0',
            metrics=self.test_metrics
        )
        
        # Check that the metrics were logged
        metrics_dir = self.test_dir / 'monitoring' / 'metrics'
        self.assertTrue((metrics_dir / 'test_model' / 'accuracy.csv').exists())
        self.assertTrue((metrics_dir / 'test_model' / 'precision.csv').exists())
        self.assertTrue((metrics_dir / 'test_model' / 'recall.csv').exists())
        self.assertTrue((metrics_dir / 'test_model' / 'f1.csv').exists())
    
    def test_log_alert(self):
        """Test logging an alert."""
        # Log an alert
        self.monitor.log_alert(
            name='test_alert',
            message='This is a test alert',
            level='WARNING',
            tags={'component': 'test'}
        )
        
        # Check that the alert was logged
        alert_file = self.test_dir / 'monitoring' / 'alerts' / 'alerts.csv'
        self.assertTrue(alert_file.exists())
        
        # Check the content
        df = pd.read_csv(alert_file)
        self.assertGreaterEqual(len(df), 1)
        self.assertEqual(df['name'].iloc[0], 'test_alert')
        self.assertEqual(df['message'].iloc[0], 'This is a test alert')
        self.assertEqual(df['level'].iloc[0], 'WARNING')
    
    def test_get_metric_history(self):
        """Test retrieving metric history."""
        # Log some metrics
        for i in range(5):
            self.monitor.log_metric('test_history', i * 0.1, step=i)
        
        # Get metric history
        history = self.monitor.get_metric_history('test_history')
        
        # Check the results
        self.assertEqual(len(history), 5)
        self.assertEqual([h['value'] for h in history], [0.0, 0.1, 0.2, 0.3, 0.4])
    
    def test_get_alert_history(self):
        """Test retrieving alert history."""
        # Log some alerts
        for i in range(3):
            self.monitor.log_alert(
                name=f'test_alert_{i}',
                message=f'Test alert {i}',
                level='WARNING'
            )
        
        # Get alert history
        history = self.monitor.get_alert_history()
        
        # Check the results
        self.assertEqual(len(history), 3)
        self.assertEqual([h['name'] for h in history], ['test_alert_0', 'test_alert_1', 'test_alert_2'])

if __name__ == "__main__":
    unittest.main()
