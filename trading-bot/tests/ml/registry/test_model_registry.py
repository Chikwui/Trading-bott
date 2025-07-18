"""
Tests for the model registry implementation.
"""
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

from core.ml.registry import get_registry, ModelRegistry

class TestModelRegistry(unittest.TestCase):
    """Test cases for the model registry."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_dir = tempfile.mkdtemp()
        self.registry = get_registry(registry_path=os.path.join(self.test_dir, 'test_registry'))
        
        # Create a simple model for testing
        self.model = RandomForestClassifier(n_estimators=10)
        self.model.fit(np.random.rand(10, 5), np.random.randint(0, 2, 10))
        
        # Save model to a temporary file
        self.temp_model_path = os.path.join(self.test_dir, 'test_model.joblib')
        joblib.dump(self.model, self.temp_model_path)
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def test_register_and_retrieve_model(self):
        """Test registering and retrieving a model."""
        # Register the model
        version = self.registry.register_model(
            model_name="test_model",
            model_path=self.temp_model_path,
            metadata={"test": "data"},
            tags=["test"]
        )
        
        # Retrieve the model
        model_path, metadata = self.registry.get_model("test_model", version)
        
        # Verify the model file exists
        self.assertTrue(os.path.exists(model_path))
        
        # Verify metadata
        self.assertEqual(metadata["model_name"], "test_model")
        self.assertEqual(metadata["version"], version)
        self.assertEqual(metadata["metadata"]["test"], "data")
        self.assertIn("test", metadata["tags"])
    
    def test_get_latest_version(self):
        """Test getting the latest version of a model."""
        # Register multiple versions
        versions = []
        for i in range(3):
            version = self.registry.register_model(
                model_name="versioned_model",
                model_path=self.temp_model_path,
                metadata={"iteration": i}
            )
            versions.append(version)
        
        # Get latest version
        latest_version = self.registry.get_latest_version("versioned_model")
        self.assertEqual(latest_version, versions[-1])
    
    def test_list_models(self):
        """Test listing all models."""
        # Register some models
        self.registry.register_model("model1", self.temp_model_path)
        self.registry.register_model("model2", self.temp_model_path)
        
        # List models
        models = self.registry.list_models()
        
        # Verify models are listed
        model_names = {m["model_name"] for m in models}
        self.assertIn("model1", model_names)
        self.assertIn("model2", model_names)
    
    def test_delete_model(self):
        """Test deleting a model version."""
        # Register a model
        version = self.registry.register_model("delete_test", self.temp_model_path)
        
        # Delete the version
        result = self.registry.delete_model("delete_test", version)
        self.assertTrue(result)
        
        # Verify it's gone
        with self.assertRaises(ValueError):
            self.registry.get_model("delete_test", version)
        
        # Test deleting non-existent model
        result = self.registry.delete_model("non_existent")
        self.assertFalse(result)

if __name__ == "__main__":
    unittest.main()
