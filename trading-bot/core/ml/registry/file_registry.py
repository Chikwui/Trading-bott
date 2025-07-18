"""
File-based implementation of the ModelRegistry interface.
"""
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import uuid
from datetime import datetime

from .base import ModelRegistry

class FileModelRegistry(ModelRegistry):
    """File-based implementation of the ModelRegistry interface."""
    
    def __init__(self, registry_path: str):
        """Initialize the file-based model registry.
        
        Args:
            registry_path: Base directory where models and metadata will be stored
        """
        self.registry_path = Path(registry_path)
        self.metadata_file = self.registry_path / "metadata.json"
        self._ensure_registry_initialized()
    
    def _ensure_registry_initialized(self) -> None:
        """Ensure the registry directory and metadata file exist."""
        self.registry_path.mkdir(parents=True, exist_ok=True)
        if not self.metadata_file.exists():
            with open(self.metadata_file, 'w') as f:
                json.dump({"models": {}}, f)
    
    def _load_metadata(self) -> Dict:
        """Load the metadata from disk."""
        with open(self.metadata_file, 'r') as f:
            return json.load(f)
    
    def _save_metadata(self, metadata: Dict) -> None:
        """Save the metadata to disk."""
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def register_model(
        self, 
        model_name: str, 
        model_path: str, 
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """Register a new model version."""
        # Generate a new version
        version = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        model_id = f"{model_name}_{version}"
        
        # Create model directory
        model_dir = self.registry_path / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model files
        model_src = Path(model_path)
        if model_src.is_file():
            shutil.copy(model_src, model_dir / model_src.name)
            model_path_in_registry = str(model_dir / model_src.name)
        else:
            shutil.copytree(model_src, model_dir, dirs_exist_ok=True)
            model_path_in_registry = str(model_dir)
        
        # Prepare metadata
        model_metadata = {
            "model_name": model_name,
            "version": version,
            "path": model_path_in_registry,
            "created_at": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
            "tags": tags or []
        }
        
        # Update registry metadata
        metadata = self._load_metadata()
        if model_name not in metadata["models"]:
            metadata["models"][model_name] = {}
        metadata["models"][model_name][version] = model_metadata
        self._save_metadata(metadata)
        
        return version
    
    def get_model(self, model_name: str, version: Optional[str] = None) -> Tuple[str, Dict]:
        """Retrieve a model and its metadata."""
        metadata = self._load_metadata()
        
        if model_name not in metadata["models"]:
            raise ValueError(f"Model '{model_name}' not found in registry")
        
        if version is None:
            version = self.get_latest_version(model_name)
            if version is None:
                raise ValueError(f"No versions found for model '{model_name}'")
        
        if version not in metadata["models"][model_name]:
            raise ValueError(f"Version '{version}' not found for model '{model_name}'")
        
        model_info = metadata["models"][model_name][version]
        return model_info["path"], model_info
    
    def list_models(self) -> List[Dict]:
        """List all registered models with their versions."""
        metadata = self._load_metadata()
        result = []
        
        for model_name, versions in metadata["models"].items():
            model_versions = []
            for version, model_info in versions.items():
                model_versions.append({
                    "version": version,
                    "created_at": model_info["created_at"],
                    "tags": model_info.get("tags", [])
                })
            
            result.append({
                "model_name": model_name,
                "versions": model_versions
            })
        
        return result
    
    def delete_model(self, model_name: str, version: Optional[str] = None) -> bool:
        """Delete a model version or all versions of a model."""
        metadata = self._load_metadata()
        
        if model_name not in metadata["models"]:
            return False
        
        if version is None:
            # Delete all versions
            shutil.rmtree(self.registry_path / model_name, ignore_errors=True)
            del metadata["models"][model_name]
        else:
            # Delete specific version
            if version not in metadata["models"][model_name]:
                return False
                
            version_dir = self.registry_path / model_name / version
            if version_dir.exists():
                shutil.rmtree(version_dir)
            
            del metadata["models"][model_name][version]
            
            # Remove model if no versions left
            if not metadata["models"][model_name]:
                del metadata["models"][model_name]
        
        self._save_metadata(metadata)
        return True
    
    def get_latest_version(self, model_name: str) -> Optional[str]:
        """Get the latest version of a model."""
        metadata = self._load_metadata()
        
        if model_name not in metadata["models"] or not metadata["models"][model_name]:
            return None
        
        # Since we're using timestamps as versions, the lexicographically largest is the latest
        return max(metadata["models"][model_name].keys())
