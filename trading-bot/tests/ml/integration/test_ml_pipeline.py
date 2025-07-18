"""
Integration tests for the ML Pipeline components.

Tests the interaction between Model Registry, Feature Store, and Monitoring.
"""
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from core.ml.registry import get_model_registry
from core.ml.feature_store import get_feature_store
from core.ml.monitoring import get_monitor

class TestMLPipelineIntegration(unittest.TestCase):
    """Integration tests for the ML Pipeline components."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Create temporary directories
        cls.test_dir = Path(tempfile.mkdtemp())
        cls.registry_dir = cls.test_dir / "model_registry"
        cls.feature_store_dir = cls.test_dir / "feature_store"
        cls.monitoring_dir = cls.test_dir / "monitoring"
        
        # Initialize components
        cls.registry = get_model_registry(
            "file", 
            registry_path=str(cls.registry_dir)
        )
        
        cls.feature_store = get_feature_store(
            "file",
            store_path=str(cls.feature_store_dir),
            partition_format="%Y/%m/%d"
        )
        
        cls.monitor = get_monitor(
            "file",
            log_dir=str(cls.monitoring_dir)
        )
        
        # Generate test data
        cls._generate_test_data()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment."""
        shutil.rmtree(cls.test_dir, ignore_errors=True)
    
    @classmethod
    def _generate_test_data(cls):
        """Generate test data for integration tests."""
        # Create a feature view
        cls.feature_view_name = "price_features"
        cls.features = [
            {"name": "open", "dtype": "float"},
            {"name": "high", "dtype": "float"},
            {"name": "low", "dtype": "float"},
            {"name": "close", "dtype": "float"},
            {"name": "volume", "dtype": "float"},
            {"name": "target", "dtype": "int"}  # 0 for down, 1 for up
        ]
        
        # Create feature view
        cls.feature_store.create_feature_view(
            name=cls.feature_view_name,
            features=cls.features,
            description="Price movement features",
            tags=["price", "movement"]
        )
        
        # Generate synthetic price data
        np.random.seed(42)
        n_samples = 1000
        dates = pd.date_range(end=datetime.now(), periods=n_samples)
        
        # Generate random walk for price data
        returns = np.random.normal(0.001, 0.02, n_samples)
        prices = np.cumprod(1 + returns) * 100  # Start at 100
        
        # Create OHLCV data
        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples))),
            'close': prices,
            'volume': np.random.lognormal(10, 1, n_samples).astype(int)
        })
        
        # Ensure high > low
        df['high'] = df[['open', 'high']].max(axis=1)
        df['low'] = df[['open', 'low']].min(axis=1)
        
        # Create target (1 if price goes up next period, else 0)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        df = df.dropna()
        
        # Store test data
        cls.test_data = df
    
    def test_end_to_end_ml_pipeline(self):
        """Test the complete ML pipeline from feature engineering to model serving."""
        # 1. Feature Engineering
        # Add some technical indicators
        df = self.test_data.copy()
        
        # Simple moving averages
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Drop NA values from indicators
        df = df.dropna()
        
        # 2. Store features in the feature store
        self.feature_store.write_features(
            feature_view_name=self.feature_view_name,
            data=df,
            timestamp_column='timestamp'
        )
        
        # 3. Retrieve features for training
        train_features = [
            'open', 'high', 'low', 'close', 'volume',
            'sma_20', 'sma_50', 'rsi'
        ]
        
        # Get features for the last 90 days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        
        features_df = self.feature_store.get_features(
            feature_view_name=self.feature_view_name,
            start_date=start_date,
            end_date=end_date
        )
        
        # Prepare features and target
        X = features_df[train_features]
        y = features_df['target']
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 4. Train a model
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        
        # Log training parameters
        self.monitor.log_parameters({
            'model_type': 'RandomForestClassifier',
            'n_estimators': 100,
            'random_state': 42,
            'features': ','.join(train_features),
            'train_start_date': start_date.isoformat(),
            'train_end_date': end_date.isoformat(),
            'test_size': 0.2
        })
        
        # Train the model
        model.fit(X_train, y_train)
        
        # 5. Evaluate the model
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score
        )
        
        y_pred = model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        # Log metrics
        self.monitor.log_metrics(metrics, prefix='test_')
        
        # 6. Register the model
        model_name = "price_movement_predictor"
        model_version = self.registry.register_model(
            name=model_name,
            model=model,
            metrics=metrics,
            feature_columns=train_features,
            target_column='target',
            model_type='classifier',
            description="Predicts price movement direction (up/down)",
            tags=["price", "movement", "classification"]
        )
        
        # 7. Test model serving
        # Get the latest model from registry
        loaded_model = self.registry.get_model(model_name, model_version)
        
        # Make predictions on test data
        predictions = loaded_model.predict(X_test)
        
        # Verify predictions
        self.assertEqual(len(predictions), len(X_test))
        self.assertIn(predictions[0], [0, 1])
        
        # 8. Test monitoring
        # Log some predictions
        for i in range(10):
            self.monitor.log_prediction(
                model_name=model_name,
                version=model_version,
                inputs=X_test.iloc[i].to_dict(),
                outputs=int(predictions[i]),
                labels=int(y_test.iloc[i]),
                metrics={
                    'correct': int(predictions[i] == y_test.iloc[i])
                }
            )
        
        # Log data drift (simulated)
        current_data = X_test.sample(100).copy()
        reference_data = X_train.sample(100).copy()
        
        for feature in train_features:
            self.monitor.log_data_drift(
                feature_name=feature,
                reference_dist=reference_data[feature].values,
                current_dist=current_data[feature].values,
                drift_score=np.random.uniform(0, 0.5)  # Simulated drift score
            )
        
        # 9. Verify all components worked together
        # Check model is registered
        model_info = self.registry.get_model(model_name, model_version)
        self.assertIsNotNone(model_info)
        self.assertEqual(model_info['name'], model_name)
        self.assertEqual(model_info['version'], model_version)
        
        # Check features were stored
        stored_features = self.feature_store.get_features(
            feature_view_name=self.feature_view_name,
            start_date=start_date,
            end_date=end_date
        )
        self.assertGreater(len(stored_features), 0)
        
        # Check metrics were logged
        metric_history = self.monitor.get_metric_history('test_accuracy')
        self.assertGreater(len(metric_history), 0)
        
        # Check alerts were created (if any)
        alerts = self.monitor.get_alert_history()
        # We don't assert on the number of alerts as it depends on the test data
    
    def test_feature_store_operations(self):
        """Test feature store operations in detail."""
        # Create a new feature view for this test
        test_view = "test_feature_view"
        
        # Define features
        features = [
            {"name": "feature1", "dtype": "float"},
            {"name": "feature2", "dtype": "int"},
            {"name": "feature3", "dtype": "str"},
            {"name": "target", "dtype": "int"}
        ]
        
        # Create feature view
        version = self.feature_store.create_feature_view(
            name=test_view,
            features=features,
            description="Test feature view",
            tags=["test"]
        )
        
        # Generate test data
        start_date = datetime.now() - timedelta(days=30)
        dates = pd.date_range(start=start_date, periods=100)
        
        test_data = pd.DataFrame({
            'timestamp': dates,
            'feature1': np.random.rand(100),
            'feature2': np.random.randint(0, 10, 100),
            'feature3': np.random.choice(['A', 'B', 'C'], 100),
            'target': np.random.randint(0, 2, 100)
        })
        
        # Write features
        self.feature_store.write_features(
            feature_view_name=test_view,
            data=test_data,
            timestamp_column='timestamp'
        )
        
        # Retrieve features
        retrieved = self.feature_store.get_features(
            feature_view_name=test_view,
            start_date=start_date,
            end_date=datetime.now()
        )
        
        # Verify data
        self.assertEqual(len(retrieved), len(test_data))
        self.assertListEqual(
            sorted(test_data.columns.tolist()),
            sorted(retrieved.columns.tolist())
        )
        
        # Test online features
        online_features = self.feature_store.get_online_features(
            feature_view_name=test_view,
            entity_keys={"feature2": [1, 2, 3]}
        )
        
        self.assertGreater(len(online_features), 0)
        
        # Clean up
        self.feature_store.delete_feature_view(test_view)
        
        # Verify deletion
        with self.assertRaises(ValueError):
            self.feature_store.get_feature_view(test_view)
    
    def test_model_registry_operations(self):
        """Test model registry operations in detail."""
        # Create a test model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        
        # Train on some dummy data
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        model.fit(X, y)
        
        # Register the model
        model_name = "test_model"
        version = self.registry.register_model(
            name=model_name,
            model=model,
            metrics={"accuracy": 0.95, "f1": 0.94},
            feature_columns=[f"feature_{i}" for i in range(5)],
            target_column="target",
            model_type="classifier",
            description="Test model",
            tags=["test"]
        )
        
        # Get model info
        model_info = self.registry.get_model(model_name, version)
        self.assertEqual(model_info["name"], model_name)
        self.assertEqual(model_info["version"], version)
        
        # List models
        models = self.registry.list_models()
        self.assertIn(model_name, [m["name"] for m in models])
        
        # Get model versions
        versions = self.registry.get_model_versions(model_name)
        self.assertIn(version, [v["version"] for v in versions])
        
        # Delete the model
        self.registry.delete_model(model_name, version)
        
        # Verify deletion
        with self.assertRaises(ValueError):
            self.registry.get_model(model_name, version)

if __name__ == "__main__":
    unittest.main()
