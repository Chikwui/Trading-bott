"""
File-based implementation of the monitoring interface.
"""
import os
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any, Union, Callable
import numpy as np
import pandas as pd

from .base import Monitor

class FileMonitor(Monitor):
    """File-based implementation of the monitoring interface."""
    
    def __init__(self, log_dir: str):
        """Initialize the file monitor.
        
        Args:
            log_dir: Base directory for storing logs and metrics
        """
        self.log_dir = Path(log_dir)
        self.metrics_dir = self.log_dir / "metrics"
        self.params_dir = self.log_dir / "params"
        self.artifacts_dir = self.log_dir / "artifacts"
        self.alerts_dir = self.log_dir / "alerts"
        
        # Create directories if they don't exist
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.params_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.alerts_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_metric_file(self, metric_name: str) -> Path:
        """Get the file path for a metric."""
        safe_name = "".join(c if c.isalnum() else "_" for c in metric_name)
        return self.metrics_dir / f"{safe_name}.csv"
    
    def _get_param_file(self, param_name: str) -> Path:
        """Get the file path for a parameter."""
        safe_name = "".join(c if c.isalnum() else "_" for c in param_name)
        return self.params_dir / f"{safe_name}.json"
    
    def _get_alert_file(self) -> Path:
        """Get the file path for alerts."""
        return self.alerts_dir / "alerts.csv"
    
    def log_metric(
        self,
        name: str,
        value: float,
        step: Optional[int] = None,
        timestamp: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Log a metric value."""
        timestamp = timestamp or datetime.utcnow()
        
        # Create metric entry
        entry = {
            "timestamp": timestamp.isoformat(),
            "step": step,
            "value": float(value),
            "tags": json.dumps(tags) if tags else ""
        }
        
        # Get or create metric file
        metric_file = self._get_metric_file(name)
        file_exists = metric_file.exists()
        
        # Write header if file doesn't exist
        with open(metric_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=entry.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(entry)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        timestamp: Optional[datetime] = None,
        prefix: str = "",
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Log multiple metrics at once."""
        timestamp = timestamp or datetime.utcnow()
        
        for name, value in metrics.items():
            full_name = f"{prefix}_{name}" if prefix else name
            self.log_metric(full_name, value, step, timestamp, tags)
    
    def log_parameter(self, name: str, value: Any) -> None:
        """Log a parameter value."""
        param_file = self._get_param_file(name)
        
        # Convert value to JSON-serializable format
        if isinstance(value, (int, float, str, bool, type(None))):
            serializable_value = value
        else:
            try:
                serializable_value = str(value)
            except Exception:
                serializable_value = "[Unserializable value]"
        
        # Write parameter to file
        with open(param_file, 'w') as f:
            json.dump({"name": name, "value": serializable_value}, f, indent=2)
    
    def log_parameters(self, params: Dict[str, Any]) -> None:
        """Log multiple parameters at once."""
        for name, value in params.items():
            self.log_parameter(name, value)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log a local file or directory as an artifact."""
        source = Path(local_path)
        if not source.exists():
            raise ValueError(f"Source path does not exist: {local_path}")
        
        # Determine destination path
        if artifact_path is None:
            artifact_path = source.name
        
        dest = self.artifacts_dir / artifact_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file or directory
        if source.is_file():
            import shutil
            shutil.copy2(source, dest)
        else:
            import shutil
            if dest.exists():
                if dest.is_file():
                    dest.unlink()
                else:
                    shutil.rmtree(dest)
            shutil.copytree(source, dest)
    
    def log_model(
        self,
        model,
        artifact_path: str,
        registered_model_name: Optional[str] = None,
        **kwargs
    ) -> None:
        """Log a model as an artifact."""
        import tempfile
        import joblib
        
        # Create a temporary file to save the model
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
            joblib.dump(model, tmp.name, **kwargs)
            
            # Log the temporary file as an artifact
            self.log_artifact(tmp.name, artifact_path)
            
            # Clean up
            os.unlink(tmp.name)
    
    def log_prediction(
        self,
        model_name: str,
        version: str,
        inputs: Any,
        outputs: Any,
        labels: Optional[Any] = None,
        metrics: Optional[Dict[str, float]] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Log a model prediction."""
        # Create prediction directory if it doesn't exist
        pred_dir = self.log_dir / "predictions" / model_name / version
        pred_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate a unique ID for this prediction
        pred_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
        
        # Save prediction data
        pred_data = {
            "prediction_id": pred_id,
            "timestamp": datetime.utcnow().isoformat(),
            "model_name": model_name,
            "version": version,
            "inputs": str(inputs),
            "outputs": str(outputs),
            "labels": str(labels) if labels is not None else "",
            "metrics": metrics or {},
            "tags": tags or {}
        }
        
        # Save to JSON file
        pred_file = pred_dir / f"{pred_id}.json"
        with open(pred_file, 'w') as f:
            json.dump(pred_data, f, indent=2)
        
        # Log metrics if provided
        if metrics:
            self.log_metrics(
                metrics=metrics,
                tags={"model": model_name, "version": version, **tags or {}}
            )
    
    def log_data_drift(
        self,
        feature_name: str,
        reference_dist: np.ndarray,
        current_dist: np.ndarray,
        drift_score: float,
        timestamp: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Log data drift information for a feature."""
        timestamp = timestamp or datetime.utcnow()
        
        # Create drift directory if it doesn't exist
        drift_dir = self.log_dir / "drift" / feature_name
        drift_dir.mkdir(parents=True, exist_ok=True)
        
        # Create drift data
        drift_data = {
            "timestamp": timestamp.isoformat(),
            "feature_name": feature_name,
            "drift_score": float(drift_score),
            "reference_stats": {
                "mean": float(np.mean(reference_dist)),
                "std": float(np.std(reference_dist)),
                "min": float(np.min(reference_dist)),
                "max": float(np.max(reference_dist)),
                "count": len(reference_dist)
            },
            "current_stats": {
                "mean": float(np.mean(current_dist)),
                "std": float(np.std(current_dist)),
                "min": float(np.min(current_dist)),
                "max": float(np.max(current_dist)),
                "count": len(current_dist)
            },
            "tags": tags or {}
        }
        
        # Save to JSON file
        drift_file = drift_dir / f"{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        with open(drift_file, 'w') as f:
            json.dump(drift_data, f, indent=2)
        
        # Log drift score as a metric
        self.log_metric(
            name=f"drift_{feature_name}",
            value=drift_score,
            timestamp=timestamp,
            tags={"feature": feature_name, **tags or {}}
        )
    
    def log_performance_metrics(
        self,
        model_name: str,
        version: str,
        metrics: Dict[str, float],
        timestamp: Optional[datetime] = None,
        step: Optional[int] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Log performance metrics for a model."""
        timestamp = timestamp or datetime.utcnow()
        
        # Add model info to tags
        all_tags = {"model": model_name, "version": version, **(tags or {})}
        
        # Log each metric
        for metric_name, value in metrics.items():
            full_metric_name = f"{model_name}/{metric_name}"
            self.log_metric(
                name=full_metric_name,
                value=value,
                step=step,
                timestamp=timestamp,
                tags=all_tags
            )
    
    def log_alert(
        self,
        name: str,
        message: str,
        level: str = "WARNING",
        condition: Optional[Callable[[], bool]] = None,
        timestamp: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Log an alert."""
        # Check condition if provided
        if condition is not None and not condition():
            return
        
        timestamp = timestamp or datetime.utcnow()
        
        # Create alert entry
        alert = {
            "timestamp": timestamp.isoformat(),
            "name": name,
            "message": message,
            "level": level.upper(),
            "tags": json.dumps(tags) if tags else ""
        }
        
        # Write to alerts file
        alert_file = self._get_alert_file()
        file_exists = alert_file.exists()
        
        with open(alert_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=alert.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(alert)
    
    def get_metric_history(
        self,
        metric_name: str,
        model_name: Optional[str] = None,
        version: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get the history of a metric."""
        metric_file = self._get_metric_file(metric_name)
        
        if not metric_file.exists():
            return []
        
        # Read metric data
        df = pd.read_csv(metric_file)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter by time range
            if start_time is not None:
                df = df[df['timestamp'] >= start_time]
            if end_time is not None:
                df = df[df['timestamp'] <= end_time]
        
        # Filter by model and version if provided
        if 'tags' in df.columns:
            if model_name is not None:
                df = df[df['tags'].str.contains(f'"model":\s*"{model_name}"', na=False)]
            if version is not None:
                df = df[df['tags'].str.contains(f'"version":\s*"{version}"', na=False)]
        
        # Convert to list of dicts
        return df.to_dict('records')
    
    def get_alert_history(
        self,
        name: Optional[str] = None,
        level: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get the history of alerts."""
        alert_file = self._get_alert_file()
        
        if not alert_file.exists():
            return []
        
        # Read alerts data
        df = pd.read_csv(alert_file)
        
        # Convert timestamp to datetime
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Filter by time range
            if start_time is not None:
                df = df[df['timestamp'] >= start_time]
            if end_time is not None:
                df = df[df['timestamp'] <= end_time]
        
        # Filter by name and level if provided
        if name is not None and 'name' in df.columns:
            df = df[df['name'] == name]
        if level is not None and 'level' in df.columns:
            df = df[df['level'] == level.upper()]
        
        # Convert to list of dicts
        return df.to_dict('records')
