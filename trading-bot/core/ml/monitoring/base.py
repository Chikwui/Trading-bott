"""
Base Monitoring Interface

Defines the abstract interface for monitoring ML models and pipelines.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime
import numpy as np

class Monitor(ABC):
    """Abstract base class for monitoring ML models and pipelines."""
    
    @abstractmethod
    def log_metric(
        self,
        name: str,
        value: float,
        step: Optional[int] = None,
        timestamp: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Log a metric value.
        
        Args:
            name: Name of the metric
            value: Numeric value of the metric
            step: Training step or iteration number
            timestamp: Timestamp of the metric
            tags: Additional tags for the metric
        """
        pass
    
    @abstractmethod
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        timestamp: Optional[datetime] = None,
        prefix: str = "",
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Log multiple metrics at once.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Training step or iteration number
            timestamp: Timestamp of the metrics
            prefix: Optional prefix to add to all metric names
            tags: Additional tags for the metrics
        """
        pass
    
    @abstractmethod
    def log_parameter(self, name: str, value: Any) -> None:
        """Log a parameter value.
        
        Args:
            name: Name of the parameter
            value: Value of the parameter
        """
        pass
    
    @abstractmethod
    def log_parameters(self, params: Dict[str, Any]) -> None:
        """Log multiple parameters at once.
        
        Args:
            params: Dictionary of parameter names to values
        """
        pass
    
    @abstractmethod
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        """Log a local file or directory as an artifact.
        
        Args:
            local_path: Path to the local file or directory
            artifact_path: Optional path within the artifact storage
        """
        pass
    
    @abstractmethod
    def log_model(
        self,
        model,
        artifact_path: str,
        registered_model_name: Optional[str] = None,
        **kwargs
    ) -> None:
        """Log a model as an artifact.
        
        Args:
            model: The model object to log
            artifact_path: Path within the artifact storage
            registered_model_name: Optional name to register the model with
            **kwargs: Additional arguments for model serialization
        """
        pass
    
    @abstractmethod
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
        """Log a model prediction.
        
        Args:
            model_name: Name of the model
            version: Version of the model
            inputs: Model inputs
            outputs: Model outputs/predictions
            labels: Ground truth labels (if available)
            metrics: Dictionary of metrics to log
            tags: Additional tags for the prediction
        """
        pass
    
    @abstractmethod
    def log_data_drift(
        self,
        feature_name: str,
        reference_dist: np.ndarray,
        current_dist: np.ndarray,
        drift_score: float,
        timestamp: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Log data drift information for a feature.
        
        Args:
            feature_name: Name of the feature
            reference_dist: Reference distribution (training data)
            current_dist: Current distribution (production data)
            drift_score: Computed drift score (0-1)
            timestamp: Timestamp of the drift detection
            tags: Additional tags for the drift detection
        """
        pass
    
    @abstractmethod
    def log_performance_metrics(
        self,
        model_name: str,
        version: str,
        metrics: Dict[str, float],
        timestamp: Optional[datetime] = None,
        step: Optional[int] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Log performance metrics for a model.
        
        Args:
            model_name: Name of the model
            version: Version of the model
            metrics: Dictionary of metric names to values
            timestamp: Timestamp of the metrics
            step: Training step or iteration number
            tags: Additional tags for the metrics
        """
        pass
    
    @abstractmethod
    def log_alert(
        self,
        name: str,
        message: str,
        level: str = "WARNING",
        condition: Optional[Callable[[], bool]] = None,
        timestamp: Optional[datetime] = None,
        tags: Optional[Dict[str, str]] = None
    ) -> None:
        """Log an alert.
        
        Args:
            name: Name of the alert
            message: Alert message
            level: Alert level (INFO, WARNING, ERROR, CRITICAL)
            condition: Optional condition function that triggers the alert when True
            timestamp: Timestamp of the alert
            tags: Additional tags for the alert
        """
        pass
    
    @abstractmethod
    def get_metric_history(
        self,
        metric_name: str,
        model_name: Optional[str] = None,
        version: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get the history of a metric.
        
        Args:
            metric_name: Name of the metric
            model_name: Optional name of the model
            version: Optional version of the model
            start_time: Optional start time for filtering
            end_time: Optional end time for filtering
            
        Returns:
            List of metric history entries with timestamps and values
        """
        pass
    
    @abstractmethod
    def get_alert_history(
        self,
        name: Optional[str] = None,
        level: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """Get the history of alerts.
        
        Args:
            name: Optional name of the alert to filter by
            level: Optional level to filter by (INFO, WARNING, ERROR, CRITICAL)
            start_time: Optional start time for filtering
            end_time: Optional end time for filtering
            
        Returns:
            List of alert history entries
        """
        pass
