"""
Base Model Registry Interface

Defines the abstract interface for model registry implementations.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

class ModelRegistry(ABC):
    """Abstract base class for model registries."""
    
    @abstractmethod
    def register_model(
        self, 
        model_name: str, 
        model_path: str, 
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """Register a new model version.
        
        Args:
            model_name: Name of the model
            model_path: Path to the model file or directory
            metadata: Additional model metadata
            tags: List of tags for the model
            
        Returns:
            Version string of the registered model
        """
        pass
    
    @abstractmethod
    def get_model(self, model_name: str, version: Optional[str] = None) -> Tuple[str, Dict]:
        """Retrieve a model and its metadata.
        
        Args:
            model_name: Name of the model
            version: Specific version to retrieve (latest if None)
            
        Returns:
            Tuple of (model_path, metadata)
        """
        pass
    
    @abstractmethod
    def list_models(self) -> List[Dict]:
        """List all registered models with their versions.
        
        Returns:
            List of model metadata dictionaries
        """
        pass
    
    @abstractmethod
    def delete_model(self, model_name: str, version: Optional[str] = None) -> bool:
        """Delete a model version or all versions of a model.
        
        Args:
            model_name: Name of the model
            version: Version to delete (all versions if None)
            
        Returns:
            True if deletion was successful
        """
        pass
    
    @abstractmethod
    def get_latest_version(self, model_name: str) -> Optional[str]:
        """Get the latest version of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Latest version string or None if model not found
        """
        pass
