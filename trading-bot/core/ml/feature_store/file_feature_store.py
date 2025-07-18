"""
File-based implementation of the FeatureStore interface.
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import shutil
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .base import FeatureStore
from ._partitioning import TimePartitioning

class FileFeatureStore(FeatureStore):
    """File-based implementation of the FeatureStore interface."""
    
    def __init__(
        self,
        store_path: str,
        partition_format: str = "%Y/%m/%d",
        file_format: str = "parquet"
    ):
        """Initialize the file-based feature store.
        
        Args:
            store_path: Base directory for the feature store
            partition_format: Time-based partition format
            file_format: File format to use (parquet, csv, etc.)
        """
        self.store_path = Path(store_path)
        self.partitioner = TimePartitioning(partition_format)
        self.file_format = file_format
        self.metadata_file = self.store_path / "metadata.json"
        
        # Ensure store directory exists
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize metadata if it doesn't exist
        if not self.metadata_file.exists():
            with open(self.metadata_file, 'w') as f:
                json.dump({"feature_views": {}}, f, indent=2)
    
    def _load_metadata(self) -> Dict:
        """Load the metadata from disk."""
        with open(self.metadata_file, 'r') as f:
            return json.load(f)
    
    def _save_metadata(self, metadata: Dict) -> None:
        """Save the metadata to disk."""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _get_feature_view_path(self, name: str, version: str) -> Path:
        """Get the directory path for a feature view."""
        return self.store_path / "feature_views" / f"{name}_{version}"
    
    def create_feature_view(
        self,
        name: str,
        features: List[Dict[str, Any]],
        description: str = "",
        tags: Optional[List[str]] = None
    ) -> str:
        """Create a new feature view."""
        metadata = self._load_metadata()
        
        # Generate new version
        version = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        
        # Create feature view directory
        view_path = self._get_feature_view_path(name, version)
        view_path.mkdir(parents=True, exist_ok=True)
        
        # Save feature view metadata
        view_metadata = {
            "name": name,
            "version": version,
            "created_at": datetime.utcnow().isoformat(),
            "features": features,
            "description": description,
            "tags": tags or [],
            "path": str(view_path)
        }
        
        # Update global metadata
        if name not in metadata["feature_views"]:
            metadata["feature_views"][name] = {}
        metadata["feature_views"][name][version] = view_metadata
        self._save_metadata(metadata)
        
        return version
    
    def get_feature_view(self, name: str, version: Optional[str] = None) -> Dict:
        """Get a feature view by name and optional version."""
        metadata = self._load_metadata()
        
        if name not in metadata["feature_views"]:
            raise ValueError(f"Feature view '{name}' not found")
        
        versions = metadata["feature_views"][name]
        
        if version is None:
            # Get latest version
            if not versions:
                raise ValueError(f"No versions found for feature view '{name}'")
            version = max(versions.keys())
        
        if version not in versions:
            raise ValueError(f"Version '{version}' not found for feature view '{name}'")
        
        return versions[version]
    
    def write_features(
        self,
        feature_view_name: str,
        data: pd.DataFrame,
        timestamp_column: str = "timestamp",
        **kwargs
    ) -> bool:
        """Write features to the feature store."""
        # Get the feature view
        view = self.get_feature_view(feature_view_name)
        view_path = Path(view["path"])
        
        # Ensure timestamp column exists
        if timestamp_column not in data.columns:
            raise ValueError(f"Timestamp column '{timestamp_column}' not found in data")
        
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(data[timestamp_column]):
            data[timestamp_column] = pd.to_datetime(data[timestamp_column])
        
        # Group by time partitions and write each partition
        for timestamp, group in data.groupby(pd.Grouper(key=timestamp_column)):
            if pd.isna(timestamp):
                continue
                
            partition = self.partitioner.get_partition_path(timestamp)
            partition_dir = view_path / "data" / partition
            partition_dir.mkdir(parents=True, exist_ok=True)
            
            # Write the data
            file_path = partition_dir / f"data.{self.file_format}"
            
            if self.file_format == "parquet":
                table = pa.Table.from_pandas(group.reset_index(drop=True))
                pq.write_table(table, file_path)
            else:
                group.to_csv(file_path, index=False)
        
        return True
    
    def get_features(
        self,
        feature_view_name: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        entity_keys: Optional[Dict[str, List]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """Retrieve features from the feature store."""
        view = self.get_feature_view(feature_view_name)
        view_path = Path(view["path"]) / "data"
        
        # Get time range
        start_date, end_date = self.partitioner.get_partition_bounds(start_date, end_date)
        
        # Get all relevant partitions
        partitions = self.partitioner.expand_partitions(
            start_date, end_date, str(view_path), self.file_format
        )
        
        # Read and concatenate all partition files
        dfs = []
        for file_path in partitions.values():
            if not os.path.exists(file_path):
                continue
                
            if self.file_format == "parquet":
                table = pq.read_table(file_path)
                df = table.to_pandas()
            else:
                df = pd.read_csv(file_path)
            
            dfs.append(df)
        
        if not dfs:
            return pd.DataFrame(columns=view["features"])
        
        # Combine all data
        result = pd.concat(dfs, ignore_index=True)
        
        # Filter by time range
        time_col = view.get("timestamp_column", "timestamp")
        if time_col in result.columns:
            result = result[
                (result[time_col] >= pd.Timestamp(start_date)) & 
                (result[time_col] <= pd.Timestamp(end_date))
            ]
        
        # Filter by entity keys if provided
        if entity_keys:
            for key, values in entity_keys.items():
                if key in result.columns:
                    result = result[result[key].isin(values)]
        
        return result
    
    def get_online_features(
        self,
        feature_view_name: str,
        entity_keys: Dict[str, List],
        **kwargs
    ) -> pd.DataFrame:
        """Retrieve features for online serving."""
        # For online features, we want the most recent data
        end_date = datetime.utcnow()
        start_date = end_date - pd.Timedelta(days=1)  # Last 24 hours by default
        
        # Get the features with time filter
        features = self.get_features(
            feature_view_name=feature_view_name,
            start_date=start_date,
            end_date=end_date,
            entity_keys=entity_keys,
            **kwargs
        )
        
        if features.empty:
            return features
        
        # For online serving, we want the most recent record per entity
        time_col = self.get_feature_view(feature_view_name).get("timestamp_column", "timestamp")
        if time_col in features.columns:
            # Sort by time descending and take the first record per entity
            entity_cols = list(entity_keys.keys())
            if entity_cols:
                features = features.sort_values(time_col, ascending=False)
                features = features.drop_duplicates(subset=entity_cols, keep='first')
        
        return features
    
    def delete_feature_view(self, name: str, version: Optional[str] = None) -> bool:
        """Delete a feature view or version."""
        metadata = self._load_metadata()
        
        if name not in metadata["feature_views"]:
            return False
        
        if version is None:
            # Delete all versions
            for ver in list(metadata["feature_views"][name].keys()):
                view_path = Path(metadata["feature_views"][name][ver]["path"])
                if view_path.exists():
                    shutil.rmtree(view_path)
            del metadata["feature_views"][name]
        else:
            # Delete specific version
            if version not in metadata["feature_views"][name]:
                return False
            
            view_path = Path(metadata["feature_views"][name][version]["path"])
            if view_path.exists():
                shutil.rmtree(view_path)
            
            del metadata["feature_views"][name][version]
            
            # Remove feature view if no versions left
            if not metadata["feature_views"][name]:
                del metadata["feature_views"][name]
        
        self._save_metadata(metadata)
        return True
    
    def list_feature_views(self) -> List[Dict]:
        """List all available feature views."""
        metadata = self._load_metadata()
        result = []
        
        for name, versions in metadata["feature_views"].items():
            view_versions = []
            for version, view_meta in versions.items():
                view_versions.append({
                    "version": version,
                    "created_at": view_meta["created_at"],
                    "description": view_meta.get("description", ""),
                    "tags": view_meta.get("tags", [])
                })
            
            result.append({
                "name": name,
                "versions": view_versions
            })
        
        return result
    
    def get_feature_stats(
        self,
        feature_view_name: str,
        feature_names: Optional[List[str]] = None
    ) -> Dict:
        """Get statistics for features."""
        view = self.get_feature_view(feature_view_name)
        features = view["features"]
        
        # Get all feature names if not specified
        if feature_names is None:
            feature_names = [f["name"] for f in features]
        
        # Get the data
        df = self.get_features(feature_view_name)
        
        if df.empty:
            return {}
        
        # Calculate basic statistics for each feature
        stats = {}
        for feature in feature_names:
            if feature not in df.columns:
                continue
                
            stats[feature] = {
                "count": int(df[feature].count()),
                "mean": float(df[feature].mean()),
                "std": float(df[feature].std()),
                "min": float(df[feature].min()),
                "25%": float(df[feature].quantile(0.25)),
                "50%": float(df[feature].quantile(0.5)),
                "75%": float(df[feature].quantile(0.75)),
                "max": float(df[feature].max()),
            }
        
        return stats
