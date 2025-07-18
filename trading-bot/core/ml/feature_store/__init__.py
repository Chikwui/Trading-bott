"""
Feature Store Module

This module provides functionality for managing feature data,
including storage, versioning, and retrieval of features for model training and inference.
"""

import os
from typing import Optional, Union

from .base import FeatureStore
from .file_feature_store import FileFeatureStore

__all__ = ['FeatureStore', 'FileFeatureStore', 'get_feature_store']

def get_feature_store(
    store_type: str = 'file',
    store_path: Optional[str] = None,
    **kwargs
) -> FeatureStore:
    """Get a feature store instance.
    
    Args:
        store_type: Type of feature store to create ('file')
        store_path: Path to the store directory (defaults to 'feature_store' in current directory)
        **kwargs: Additional arguments for the feature store constructor
        
    Returns:
        An instance of FeatureStore
    """
    if store_path is None:
        store_path = os.path.join(os.getcwd(), 'feature_store')
    
    if store_type == 'file':
        return FileFeatureStore(store_path=store_path, **kwargs)
    else:
        raise ValueError(f"Unsupported feature store type: {store_type}")
