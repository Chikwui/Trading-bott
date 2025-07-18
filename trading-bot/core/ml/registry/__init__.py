"""
Model Registry Module

This module provides functionality for managing machine learning models,
including versioning, storage, and retrieval of trained models.
"""

import os
from typing import Optional

from .base import ModelRegistry
from .file_registry import FileModelRegistry

__all__ = ['ModelRegistry', 'FileModelRegistry', 'get_registry']

def get_registry(registry_path: Optional[str] = None) -> ModelRegistry:
    """Get a model registry instance.
    
    Args:
        registry_path: Path to the registry storage. If None, uses default location.
        
    Returns:
        An instance of ModelRegistry
    """
    if registry_path is None:
        # Default to a directory called 'model_registry' in the current working directory
        registry_path = os.path.join(os.getcwd(), 'model_registry')
    
    return FileModelRegistry(registry_path)
