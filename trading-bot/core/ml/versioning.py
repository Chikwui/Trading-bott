"""
Model Versioning and Experiment Tracking Module

This module provides functionality for tracking model versions, experiments,
and managing model artifacts with proper versioning.
"""
import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import mlflow
from dataclasses import dataclass, asdict, field
from enum import Enum
import pandas as pd

class ModelStatus(str, Enum):    
    STAGING = "staging"
    PRODUCTION = "production"n    ARCHIVED = "archived"
    
@dataclass
class ModelVersion:
    """Represents a versioned ML model with metadata."""
    model_id: str
    version: str
    path: str
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    status: ModelStatus = ModelStatus.STAGING
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    description: str = ""
    tags: Dict[str, str] = field(default_factory=dict)

class ModelVersionManager:
    """Manages model versions and their lifecycle."""
    
    def __init__(self, tracking_uri: str = "mlruns"):
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        
    def create_version(
        self,
        model_id: str,
        model,
        metrics: Dict[str, float],
        parameters: Dict[str, Any],
        description: str = "",
        tags: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> ModelVersion:
        """Create a new version of a model."""
        # Generate version hash
        version = hashlib.sha256(
            f"{model_id}{datetime.utcnow().isoformat()}".encode()
        ).hexdigest()[:8]
        
        # Create model version
        model_version = ModelVersion(
            model_id=model_id,
            version=version,
            path=f"models:/{model_id}/{version}",
            metrics=metrics,
            parameters=parameters,
            description=description,
            tags=tags or {}
        )
        
        # Log to MLflow
        with mlflow.start_run():
            # Log parameters and metrics
            mlflow.log_params(parameters)
            mlflow.log_metrics(metrics)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Log additional metadata
            mlflow.set_tag("model_id", model_id)
            mlflow.set_tag("version", version)
            mlflow.set_tag("status", ModelStatus.STAGING.value)
            
            for key, value in (tags or {}).items():
                mlflow.set_tag(key, value)
        
        return model_version
    
    def get_model(self, model_id: str, version: str = None, stage: str = None):
        """Load a specific version or stage of a model."""
        if version and stage:
            raise ValueError("Cannot specify both version and stage")
            
        model_uri = f"models:/{model_id}"
        if version:
            model_uri += f"/{version}"
        elif stage:
            model_uri += f"@{stage}"
            
        return mlflow.pyfunc.load_model(model_uri)
    
    def transition_model_stage(
        self, 
        model_id: str, 
        version: str, 
        new_stage: ModelStatus
    ) -> None:
        """Transition a model version to a new stage."""
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_id,
            version=version,
            stage=new_stage.value
        )
    
    def get_versions(
        self, 
        model_id: str,
        status: Optional[ModelStatus] = None
    ) -> List[ModelVersion]:
        """Get all versions of a model, optionally filtered by status."""
        client = mlflow.tracking.MlflowClient()
        
        if status:
            versions = client.search_model_versions(
                f"name='{model_id}' AND status='{status.value}'"
            )
        else:
            versions = client.search_model_versions(f"name='{model_id}'")
        
        return [
            ModelVersion(
                model_id=model_id,
                version=v.version,
                path=v.source,
                metrics=client.get_run(v.run_id).data.metrics,
                parameters=client.get_run(v.run_id).data.params,
                status=ModelStatus(v.current_stage.lower()),
                created_at=pd.to_datetime(v.creation_timestamp, unit='ms').isoformat(),
                tags=v.tags
            )
            for v in versions
        ]

# Example usage:
if __name__ == "__main__":
    # Initialize version manager
    version_manager = ModelVersionManager()
    
    # Example: Create a new model version
    from sklearn.ensemble import RandomForestClassifier
    
    model = RandomForestClassifier(n_estimators=100)
    metrics = {"accuracy": 0.95, "f1_score": 0.94}
    params = {"n_estimators": 100, "max_depth": 10}
    
    version = version_manager.create_version(
        model_id="price_prediction",
        model=model,
        metrics=metrics,
        parameters=params,
        description="Initial version with basic features",
        tags={"dataset": "hourly_1m", "framework": "sklearn"}
    )
    
    # Transition to production
    version_manager.transition_model_stage(
        model_id="price_prediction",
        version=version.version,
        new_stage=ModelStatus.PRODUCTION
    )
    
    # Get production model
    prod_model = version_manager.get_model("price_prediction", stage=ModelStatus.PRODUCTION)
