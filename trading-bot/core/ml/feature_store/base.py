"""
Base Feature Store Interface

Defines the abstract interface for feature store implementations.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import pandas as pd

class FeatureStore(ABC):
    """Abstract base class for feature stores."""
    
    @abstractmethod
    def create_feature_view(
        self,
        name: str,
        features: List[Dict[str, Any]],
        description: str = "",
        tags: Optional[List[str]] = None
    ) -> str:
        """Create a new feature view.
        
        Args:
            name: Name of the feature view
            features: List of feature definitions
            description: Optional description
            tags: Optional list of tags
            
        Returns:
            ID of the created feature view
        """
        pass
    
    @abstractmethod
    def get_feature_view(self, name: str, version: Optional[str] = None) -> Dict:
        """Get a feature view by name and optional version.
        
        Args:
            name: Name of the feature view
            version: Version of the feature view (latest if None)
            
        Returns:
            Feature view metadata
        """
        pass
    
    @abstractmethod
    def write_features(
        self,
        feature_view_name: str,
        data: pd.DataFrame,
        timestamp_column: str = "timestamp",
        **kwargs
    ) -> bool:
        """Write features to the feature store.
        
        Args:
            feature_view_name: Name of the feature view
            data: DataFrame containing the features
            timestamp_column: Name of the timestamp column
            **kwargs: Additional arguments for the storage backend
            
        Returns:
            True if write was successful
        """
        pass
    
    @abstractmethod
    def get_features(
        self,
        feature_view_name: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        entity_keys: Optional[Dict[str, List]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Retrieve features from the feature store.
        
        Args:
            feature_view_name: Name of the feature view
            start_date: Start date for time range query
            end_date: End date for time range query
            entity_keys: Dictionary of entity keys to filter by
            **kwargs: Additional arguments for the query
            
        Returns:
            DataFrame containing the requested features
        """
        pass
    
    @abstractmethod
    def get_online_features(
        self,
        feature_view_name: str,
        entity_keys: Dict[str, List],
        **kwargs
    ) -> pd.DataFrame:
        """Retrieve features for online serving.
        
        Args:
            feature_view_name: Name of the feature view
            entity_keys: Dictionary of entity keys to retrieve features for
            **kwargs: Additional arguments for the query
            
        Returns:
            DataFrame containing the requested features
        """
        pass
    
    @abstractmethod
    def delete_feature_view(self, name: str, version: Optional[str] = None) -> bool:
        """Delete a feature view or version.
        
        Args:
            name: Name of the feature view
            version: Version to delete (all versions if None)
            
        Returns:
            True if deletion was successful
        """
        pass
    
    @abstractmethod
    def list_feature_views(self) -> List[Dict]:
        """List all available feature views.
        
        Returns:
            List of feature view metadata dictionaries
        """
        pass
    
    @abstractmethod
    def get_feature_stats(
        self,
        feature_view_name: str,
        feature_names: Optional[List[str]] = None
    ) -> Dict:
        """Get statistics for features.
        
        Args:
            feature_view_name: Name of the feature view
            feature_names: List of feature names to get stats for (all if None)
            
        Returns:
            Dictionary of feature statistics
        """
        pass
