"""
Tests for the feature store implementation.
"""
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from core.ml.feature_store import get_feature_store, FeatureStore

class TestFeatureStore(unittest.TestCase):
    """Test cases for the feature store."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.store: FeatureStore = get_feature_store(store_path=os.path.join(self.test_dir, 'test_feature_store'))
        
        # Create test data
        self.now = datetime.utcnow()
        self.dates = [self.now - timedelta(days=i) for i in range(5)]
        
        self.test_data = pd.DataFrame({
            'timestamp': self.dates * 2,  # Two records per date
            'entity_id': [1, 2] * 5,  # Two entities
            'feature1': np.random.rand(10),
            'feature2': np.random.rand(10),
            'target': np.random.randint(0, 2, 10)
        })
        
        # Create a feature view
        self.feature_view_name = "test_features"
        self.features = [
            {"name": "feature1", "dtype": "float"},
            {"name": "feature2", "dtype": "float"},
            {"name": "target", "dtype": "int"}
        ]
        
        self.version = self.store.create_feature_view(
            name=self.feature_view_name,
            features=self.features,
            description="Test feature view",
            tags=["test"]
        )
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_create_feature_view(self):
        """Test creating a feature view."""
        # Test creating a new feature view
        view_name = "test_view_creation"
        version = self.store.create_feature_view(
            name=view_name,
            features=[{"name": "test_feature", "dtype": "float"}],
            description="Test feature view creation"
        )
        
        # Verify the view was created
        view = self.store.get_feature_view(view_name, version)
        self.assertEqual(view["name"], view_name)
        self.assertEqual(view["version"], version)
        self.assertEqual(len(view["features"]), 1)
        self.assertEqual(view["features"][0]["name"], "test_feature")
    
    def test_write_and_retrieve_features(self):
        """Test writing and retrieving features."""
        # Write test data
        result = self.store.write_features(
            feature_view_name=self.feature_view_name,
            data=self.test_data.copy(),
            timestamp_column="timestamp"
        )
        self.assertTrue(result)
        
        # Retrieve features
        features = self.store.get_features(
            feature_view_name=self.feature_view_name,
            start_date=self.now - timedelta(days=10),
            end_date=self.now
        )
        
        # Verify the data
        self.assertEqual(len(features), len(self.test_data))
        self.assertListEqual(
            sorted(features.columns.tolist()),
            sorted(['timestamp', 'entity_id', 'feature1', 'feature2', 'target'])
        )
    
    def test_online_features(self):
        """Test online feature retrieval."""
        # Write test data
        self.store.write_features(
            feature_view_name=self.feature_view_name,
            data=self.test_data,
            timestamp_column="timestamp"
        )
        
        # Get online features for specific entities
        features = self.store.get_online_features(
            feature_view_name=self.feature_view_name,
            entity_keys={"entity_id": [1, 2]}
        )
        
        # Should get the most recent record for each entity
        self.assertEqual(len(features), 2)  # One per entity
        self.assertSetEqual(set(features['entity_id'].tolist()), {1, 2})
    
    def test_feature_stats(self):
        """Test feature statistics calculation."""
        # Write test data
        self.store.write_features(
            feature_view_name=self.feature_view_name,
            data=self.test_data,
            timestamp_column="timestamp"
        )
        
        # Get statistics
        stats = self.store.get_feature_stats(
            feature_view_name=self.feature_view_name,
            feature_names=["feature1", "feature2"]
        )
        
        # Verify statistics
        self.assertIn("feature1", stats)
        self.assertIn("feature2", stats)
        self.assertEqual(stats["feature1"]["count"], len(self.test_data))
        self.assertLessEqual(stats["feature1"]["min"], stats["feature1"]["max"])
    
    def test_list_feature_views(self):
        """Test listing feature views."""
        # Create another feature view
        another_view = "another_test_view"
        self.store.create_feature_view(
            name=another_view,
            features=[{"name": "test", "dtype": "float"}]
        )
        
        # List all feature views
        views = self.store.list_feature_views()
        view_names = {v["name"] for v in views}
        
        self.assertIn(self.feature_view_name, view_names)
        self.assertIn(another_view, view_names)
    
    def test_delete_feature_view(self):
        """Test deleting a feature view."""
        # Delete the test feature view
        result = self.store.delete_feature_view(self.feature_view_name)
        self.assertTrue(result)
        
        # Verify it's gone
        with self.assertRaises(ValueError):
            self.store.get_feature_view(self.feature_view_name)

if __name__ == "__main__":
    unittest.main()
