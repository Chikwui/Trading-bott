"""
Monitoring Pipeline Module

This module implements the core monitoring pipeline for ML models, integrating
drift detection, explainability, metrics, and alerting.
"""

import time
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from typing import (
    Any, Dict, List, Optional, Tuple, Union, Callable, TypeVar,
    Generic, Type, cast, ClassVar
)
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, validator, root_validator

from ..types import ModelType, DataFrame, NDArray
from .drift_detector import DriftResult, DriftType, BaseDriftDetector
from .metrics import ModelMetrics, PerformanceMetrics, DataQualityMetrics
from .alerts import Alert, AlertLevel, AlertManager, AlertRule
from ..explainability.shap_explainer import SHAPExplainer, ExplanationResult

logger = logging.getLogger(__name__)

T = TypeVar('T')

class ModelInput(BaseModel):
    """Represents a single model input with features and optional metadata."""
    features: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
            np.ndarray: lambda arr: arr.tolist(),
            np.integer: int,
            np.floating: float,
        }
        
        @classmethod
        def schema_extra(cls, schema: Dict[str, Any], model: Type['ModelInput']) -> None:
            """Add example to OpenAPI schema."""
            schema['example'] = {
                'features': {'feature1': 1.0, 'feature2': 'value'},
                'timestamp': '2023-01-01T00:00:00Z',
                'metadata': {'source': 'api', 'user_id': 'user123'}
            }

class ModelOutput(BaseModel):
    """Represents a single model prediction with optional explanations."""
    prediction: Union[float, int, str, Dict[str, float], List[float]]
    model_version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    explanation: Optional[Dict[str, Any]] = None
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
            np.ndarray: lambda arr: arr.tolist(),
        }

class PredictionRecord(ModelInput, ModelOutput):
    """Combines model input and output with ground truth when available."""
    ground_truth: Optional[Union[float, int, str, Dict[str, float]]] = None
    prediction_latency: Optional[float] = None  # in seconds
    
    def to_series(self) -> pd.Series:
        """Convert record to pandas Series for analysis."""
        data = {
            **self.features,
            'prediction': self.prediction,
            'model_version': self.model_version,
            'timestamp': self.timestamp,
            **self.metadata
        }
        if self.ground_truth is not None:
            data['ground_truth'] = self.ground_truth
        if self.prediction_latency is not None:
            data['prediction_latency'] = self.prediction_latency
        return pd.Series(data)

@dataclass
class MonitoringConfig:
    """Configuration for the monitoring pipeline."""
    # Drift detection
    drift_enabled: bool = True
    drift_detector: Optional[BaseDriftDetector] = None
    drift_threshold: float = 0.05
    drift_window_size: int = 1000
    
    # Performance monitoring
    performance_metrics: List[str] = field(
        default_factory=lambda: ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    )
    performance_window_size: int = 1000
    
    # Explainability
    explainability_enabled: bool = True
    explainer_type: str = 'auto'  # 'tree', 'linear', 'kernel', 'deep', 'auto'
    explainer_sample_size: int = 100
    
    # Data quality
    data_quality_checks: List[str] = field(
        default_factory=lambda: ['missing_values', 'range_violations', 'type_checks']
    )
    
    # Alerting
    alert_rules: List[AlertRule] = field(default_factory=list)
    
    # Resource monitoring
    monitor_resources: bool = True
    resource_metrics: List[str] = field(
        default_factory=lambda: ['cpu', 'memory', 'gpu', 'latency']
    )
    
    # Storage
    save_predictions: bool = True
    save_explanations: bool = True
    storage_backend: str = 'local'  # 'local', 's3', 'gcs', 'database'
    storage_options: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary, handling non-serializable fields."""
        config_dict = asdict(self)
        # Handle non-serializable fields
        if 'alert_rules' in config_dict:
            config_dict['alert_rules'] = [
                rule.to_dict() if hasattr(rule, 'to_dict') else str(rule)
                for rule in config_dict['alert_rules']
            ]
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MonitoringConfig':
        """Create config from dictionary."""
        # Create a copy to avoid modifying the input
        config_dict = config_dict.copy()
        
        # Handle alert rules if present
        if 'alert_rules' in config_dict and config_dict['alert_rules']:
            from .rules import create_alert_rule
            config_dict['alert_rules'] = [
                create_alert_rule(rule) if isinstance(rule, dict) else rule
                for rule in config_dict['alert_rules']
            ]
            
        return cls(**config_dict)

class ModelMonitor:
    """Main class for monitoring ML models in production."""
    
    def __init__(
        self,
        model: Any,
        config: Optional[MonitoringConfig] = None,
        alert_manager: Optional[AlertManager] = None,
        model_version: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
        target_name: Optional[str] = None
    ) -> None:
        """Initialize the model monitor.
        
        Args:
            model: The trained ML model to monitor
            config: Monitoring configuration
            alert_manager: Alert manager for handling alerts
            model_version: Version of the model being monitored
            feature_names: List of feature names (for better visualization)
            target_name: Name of the target variable (for better visualization)
        """
        self.model = model
        self.config = config or MonitoringConfig()
        self.alert_manager = alert_manager or AlertManager()
        self.model_version = model_version or '1.0.0'
        self.feature_names = feature_names
        self.target_name = target_name
        
        # Initialize components
        self._initialize_components()
        
        # State
        self._prediction_buffer: List[PredictionRecord] = []
        self._last_drift_check: Optional[datetime] = None
        self._last_performance_check: Optional[datetime] = None
        self._last_explanation: Optional[Dict[str, Any]] = None
        
        # Statistics
        self._stats = {
            'predictions_processed': 0,
            'alerts_triggered': 0,
            'drift_detected': 0,
            'last_updated': datetime.utcnow()
        }
    
    def _initialize_components(self) -> None:
        """Initialize monitoring components based on config."""
        # Initialize drift detector if not provided
        if self.config.drift_enabled and self.config.drift_detector is None:
            from .drift_detector import DriftDetectorFactory
            self.config.drift_detector = DriftDetectorFactory.create_detector(
                'ks',  # Default to Kolmogorov-Smirnov test
                threshold=self.config.drift_threshold,
                window_size=self.config.drift_window_size
            )
        
        # Initialize explainer if enabled
        self.explainer: Optional[SHAPExplainer] = None
        if self.config.explainability_enabled:
            try:
                self.explainer = SHAPExplainer(
                    self.model,
                    algorithm=self.config.explainer_type,
                    feature_names=self.feature_names,
                    target_name=self.target_name
                )
            except Exception as e:
                logger.warning(f"Failed to initialize explainer: {e}")
                self.config.explainability_enabled = False
    
    def process_prediction(
        self,
        features: Union[Dict[str, Any], ModelInput, pd.DataFrame],
        prediction: Optional[Any] = None,
        ground_truth: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PredictionRecord:
        """Process a single prediction through the monitoring pipeline.
        
        Args:
            features: Input features as a dictionary, ModelInput, or DataFrame
            prediction: Model prediction (if not provided, will be computed)
            ground_truth: Ground truth value (if available)
            metadata: Additional metadata
            
        Returns:
            PredictionRecord containing the prediction and monitoring results
        """
        start_time = time.time()
        
        # Create prediction record
        if isinstance(features, ModelInput):
            record = PredictionRecord(
                **features.dict(),
                prediction=prediction or self._predict(features.features),
                model_version=self.model_version,
                ground_truth=ground_truth
            )
        else:
            if isinstance(features, pd.DataFrame):
                features = features.iloc[0].to_dict()
                
            record = PredictionRecord(
                features=features,
                prediction=prediction or self._predict(features),
                model_version=self.model_version,
                ground_truth=ground_truth,
                metadata=metadata or {}
            )
        
        # Calculate prediction latency
        record.prediction_latency = time.time() - start_time
        
        # Add to buffer
        self._prediction_buffer.append(record)
        self._stats['predictions_processed'] += 1
        self._stats['last_updated'] = datetime.utcnow()
        
        # Check if we should perform monitoring checks
        self._check_and_perform_monitoring()
        
        return record
    
    def _predict(self, features: Dict[str, Any]) -> Any:
        """Make a prediction using the monitored model."""
        # Convert features to the format expected by the model
        if hasattr(self.model, 'predict'):
            if isinstance(features, dict):
                # Convert dict to DataFrame with single row
                import pandas as pd
                features_df = pd.DataFrame([features])
                return self.model.predict(features_df)[0]
            return self.model.predict(features)[0]
        elif callable(self.model):
            return self.model(features)
        else:
            raise ValueError("Model must be callable or have a predict method")
    
    def _check_and_perform_monitoring(self) -> None:
        """Check if monitoring checks should be performed and execute them."""
        current_time = datetime.utcnow()
        
        # Check for drift
        if (self.config.drift_enabled and 
            (self._last_drift_check is None or 
             (current_time - self._last_drift_check).total_seconds() > 3600) and
            len(self._prediction_buffer) >= self.config.drift_window_size):
            self._check_for_drift()
            self._last_drift_check = current_time
        
        # Check performance if ground truth is available
        if (self._last_performance_check is None or 
            (current_time - self._last_performance_check).total_seconds() > 3600):
            self._check_performance()
            self._last_performance_check = current_time
        
        # Generate explanations periodically if enabled
        if (self.config.explainability_enabled and 
            len(self._prediction_buffer) > 0 and 
            (self._last_explanation is None or 
             (current_time - self._last_explanation.get('timestamp', datetime.min)).total_seconds() > 86400)):
            self._generate_explanations()
    
    def _check_for_drift(self) -> None:
        """Check for data drift in the prediction buffer."""
        if not self.config.drift_detector:
            return
            
        try:
            # Convert buffer to DataFrame
            df = pd.DataFrame([r.features for r in self._prediction_buffer])
            
            # Check for drift
            result = self.config.drift_detector.detect_drift(df)
            
            if result.is_drifted:
                self._stats['drift_detected'] += 1
                alert = Alert(
                    level=AlertLevel.WARNING,
                    message=f"Data drift detected: {result.drift_type} (score={result.drift_score:.4f})",
                    details={
                        'drift_type': result.drift_type,
                        'drift_score': float(result.drift_score),
                        'threshold': float(self.config.drift_threshold),
                        'features': result.feature_scores.to_dict() if hasattr(result, 'feature_scores') else {}
                    },
                    source='drift_detector'
                )
                self.alert_manager.handle_alert(alert)
                
        except Exception as e:
            logger.error(f"Error checking for drift: {e}")
            self.alert_manager.handle_alert(
                Alert(
                    level=AlertLevel.ERROR,
                    message="Error checking for data drift",
                    details={'error': str(e)},
                    source='drift_detector'
                )
            )
    
    def _check_performance(self) -> None:
        """Check model performance on recent predictions with ground truth."""
        try:
            # Get records with ground truth
            records_with_truth = [
                r for r in self._prediction_buffer 
                if r.ground_truth is not None
            ]
            
            if not records_with_truth:
                return
                
            # Convert to DataFrame
            df = pd.DataFrame([r.dict() for r in records_with_truth])
            
            # Calculate performance metrics
            metrics = {}
            y_true = df['ground_truth']
            y_pred = df['prediction']
            
            # Basic regression metrics
            if pd.api.types.is_numeric_dtype(y_true):
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                metrics.update({
                    'mse': mean_squared_error(y_true, y_pred),
                    'mae': mean_absolute_error(y_true, y_pred),
                    'r2': r2_score(y_true, y_pred)
                })
            # Basic classification metrics
            else:
                from sklearn.metrics import (
                    accuracy_score, precision_score, 
                    recall_score, f1_score, roc_auc_score
                )
                metrics.update({
                    'accuracy': accuracy_score(y_true, y_pred),
                    'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                    'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
                })
                
                # ROC AUC for binary classification
                try:
                    if len(set(y_true)) == 2:
                        metrics['roc_auc'] = roc_auc_score(y_true, y_pred)
                except:
                    pass
            
            # Update metrics
            self.performance_metrics = PerformanceMetrics(**metrics)
            
            # Check against thresholds and alert if needed
            self._check_performance_metrics(metrics)
            
        except Exception as e:
            logger.error(f"Error checking performance: {e}")
            self.alert_manager.handle_alert(
                Alert(
                    level=AlertLevel.ERROR,
                    message="Error checking model performance",
                    details={'error': str(e)},
                    source='performance_monitor'
                )
            )
    
    def _check_performance_metrics(self, metrics: Dict[str, float]) -> None:
        """Check performance metrics against thresholds and trigger alerts if needed."""
        # Define performance thresholds (customize based on your use case)
        thresholds = {
            'accuracy': 0.8,
            'precision': 0.7,
            'recall': 0.7,
            'f1': 0.7,
            'roc_auc': 0.7,
            'mse': None,  # Set appropriate thresholds for regression
            'mae': None,
            'r2': 0.6
        }
        
        for metric, value in metrics.items():
            threshold = thresholds.get(metric)
            if threshold is None:
                continue
                
            # For metrics where higher is better
            if metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'r2']:
                if value < threshold:
                    self.alert_manager.handle_alert(
                        Alert(
                            level=AlertLevel.WARNING,
                            message=f"{metric} below threshold: {value:.4f} < {threshold}",
                            details={
                                'metric': metric,
                                'value': value,
                                'threshold': threshold
                            },
                            source='performance_monitor'
                        )
                    )
            # For metrics where lower is better
            elif metric in ['mse', 'mae']:
                if value > threshold:
                    self.alert_manager.handle_alert(
                        Alert(
                            level=AlertLevel.WARNING,
                            message=f"{metric} above threshold: {value:.4f} > {threshold}",
                            details={
                                'metric': metric,
                                'value': value,
                                'threshold': threshold
                            },
                            source='performance_monitor'
                        )
                    )
    
    def _generate_explanations(self) -> None:
        """Generate explanations for recent predictions."""
        if not self.explainer or not self.config.explainability_enabled:
            return
            
        try:
            # Sample recent predictions for explanation
            sample_size = min(self.config.explainer_sample_size, len(self._prediction_buffer))
            sample_indices = np.random.choice(
                len(self._prediction_buffer), 
                size=sample_size, 
                replace=False
            )
            
            sample_records = [self._prediction_buffer[i] for i in sample_indices]
            sample_features = [r.features for r in sample_records]
            
            # Convert to DataFrame
            df = pd.DataFrame(sample_features)
            
            # Generate explanations
            explanation = self.explainer.explain(df)
            
            # Store explanation
            self._last_explanation = {
                'timestamp': datetime.utcnow(),
                'explanation': explanation.to_dict() if hasattr(explanation, 'to_dict') else explanation,
                'sample_size': sample_size
            }
            
            # Log feature importance
            if hasattr(explanation, 'feature_importances'):
                logger.info("Feature importances: %s", explanation.feature_importances)
            
        except Exception as e:
            logger.error(f"Error generating explanations: {e}")
            self.alert_manager.handle_alert(
                Alert(
                    level=AlertLevel.ERROR,
                    message="Error generating model explanations",
                    details={'error': str(e)},
                    source='explainability'
                )
            )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        return {
            'stats': self._stats,
            'config': self.config.to_dict(),
            'last_drift_check': self._last_drift_check.isoformat() if self._last_drift_check else None,
            'last_performance_check': self._last_performance_check.isoformat() if self._last_performance_check else None,
            'buffer_size': len(self._prediction_buffer),
            'performance_metrics': self.performance_metrics.dict() if hasattr(self, 'performance_metrics') else None,
            'has_explanation': self._last_explanation is not None
        }
    
    def save_state(self, path: str) -> None:
        """Save monitor state to disk."""
        import pickle
        import os
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            state = {
                'config': self.config,
                'stats': self._stats,
                'last_drift_check': self._last_drift_check,
                'last_performance_check': self._last_performance_check,
                'prediction_buffer': self._prediction_buffer,
                'last_explanation': self._last_explanation,
                'model_version': self.model_version
            }
            pickle.dump(state, f)
    
    @classmethod
    def load_state(cls, path: str, model: Any) -> 'ModelMonitor':
        """Load monitor state from disk."""
        import pickle
        
        with open(path, 'rb') as f:
            state = pickle.load(f)
            
        monitor = cls(
            model=model,
            config=state.get('config'),
            model_version=state.get('model_version')
        )
        
        monitor._stats = state.get('stats', {})
        monitor._last_drift_check = state.get('last_drift_check')
        monitor._last_performance_check = state.get('last_performance_check')
        monitor._prediction_buffer = state.get('prediction_buffer', [])
        monitor._last_explanation = state.get('last_explanation')
        
        return monitor

class MonitoringPipeline:
    """Orchestrates the end-to-end monitoring workflow."""
    
    def __init__(
        self,
        model: Any,
        config: Optional[MonitoringConfig] = None,
        alert_manager: Optional[AlertManager] = None,
        feature_names: Optional[List[str]] = None,
        target_name: Optional[str] = None
    ) -> None:
        """Initialize the monitoring pipeline.
        
        Args:
            model: The ML model to monitor
            config: Monitoring configuration
            alert_manager: Alert manager for handling alerts
            feature_names: List of feature names
            target_name: Name of the target variable
        """
        self.model = model
        self.config = config or MonitoringConfig()
        self.alert_manager = alert_manager or AlertManager()
        self.feature_names = feature_names
        self.target_name = target_name
        
        # Initialize monitors
        self.monitors: Dict[str, ModelMonitor] = {}
        self._initialize_monitors()
    
    def _initialize_monitors(self) -> None:
        """Initialize model monitors based on configuration."""
        # For now, just create a single monitor for the main model
        # In a real-world scenario, you might have multiple monitors for different models/versions
        self.monitors['default'] = ModelMonitor(
            model=self.model,
            config=self.config,
            alert_manager=self.alert_manager,
            feature_names=self.feature_names,
            target_name=self.target_name
        )
    
    def process_predictions(
        self,
        features: Union[Dict[str, Any], List[Dict[str, Any]], ModelInput, List[ModelInput], pd.DataFrame],
        predictions: Optional[Union[Any, List[Any]]] = None,
        ground_truths: Optional[Union[Any, List[Any]]] = None,
        metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None
    ) -> Union[PredictionRecord, List[PredictionRecord]]:
        """Process one or more predictions through the monitoring pipeline.
        
        Args:
            features: Input features (single or batch)
            predictions: Optional model predictions (if not provided, will be computed)
            ground_truths: Optional ground truth values
            metadata: Optional metadata
            
        Returns:
            PredictionRecord or list of PredictionRecords
        """
        # Handle batch processing
        if isinstance(features, (list, pd.DataFrame)) or (
            hasattr(features, '__len__') and len(features) > 1  # type: ignore
        ):
            return self._process_batch(features, predictions, ground_truths, metadata)
        
        # Single prediction
        return self.monitors['default'].process_prediction(
            features=features,
            prediction=predictions[0] if predictions is not None else None,
            ground_truth=ground_truths[0] if ground_truths is not None else None,
            metadata=metadata[0] if isinstance(metadata, list) else metadata
        )
    
    def _process_batch(
        self,
        features: Union[List[Dict[str, Any]], List[ModelInput], pd.DataFrame],
        predictions: Optional[Union[Any, List[Any]]] = None,
        ground_truths: Optional[Union[Any, List[Any]]] = None,
        metadata: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None
    ) -> List[PredictionRecord]:
        """Process a batch of predictions."""
        if isinstance(features, pd.DataFrame):
            features = features.to_dict('records')
        
        # Ensure ground_truths and metadata are lists of appropriate length
        if ground_truths is not None and not isinstance(ground_truths, (list, np.ndarray)):
            ground_truths = [ground_truths] * len(features)
            
        if metadata is not None and not isinstance(metadata, list):
            metadata = [metadata] * len(features)
        
        # Process each prediction
        records = []
        for i, feature_set in enumerate(features):
            record = self.monitors['default'].process_prediction(
                features=feature_set,
                prediction=predictions[i] if predictions is not None else None,
                ground_truth=ground_truths[i] if ground_truths is not None else None,
                metadata=metadata[i] if metadata is not None else None
            )
            records.append(record)
            
        return records
    
    def get_status(self, monitor_id: str = 'default') -> Dict[str, Any]:
        """Get status of a specific monitor."""
        if monitor_id not in self.monitors:
            raise ValueError(f"Monitor {monitor_id} not found")
            
        return self.monitors[monitor_id].get_status()
    
    def save_state(self, path: str, monitor_id: str = 'default') -> None:
        """Save monitor state to disk."""
        if monitor_id not in self.monitors:
            raise ValueError(f"Monitor {monitor_id} not found")
            
        self.monitors[monitor_id].save_state(path)
    
    @classmethod
    def load_state(
        cls, 
        path: str, 
        model: Any, 
        monitor_id: str = 'default'
    ) -> 'MonitoringPipeline':
        """Load monitor state from disk."""
        pipeline = cls(model)
        pipeline.monitors[monitor_id] = ModelMonitor.load_state(path, model)
        return pipeline

class DataStream:
    """Handles streaming data for online monitoring."""
    
    def __init__(
        self,
        pipeline: MonitoringPipeline,
        buffer_size: int = 1000,
        process_interval: float = 60.0
    ) -> None:
        """Initialize the data stream processor.
        
        Args:
            pipeline: MonitoringPipeline instance
            buffer_size: Maximum number of records to buffer before processing
            process_interval: Time in seconds between processing batches
        """
        self.pipeline = pipeline
        self.buffer_size = buffer_size
        self.process_interval = process_interval
        
        # Buffer for incoming data
        self._buffer: List[Dict[str, Any]] = []
        self._last_process_time = time.time()
    
    def add_data(
        self,
        features: Dict[str, Any],
        prediction: Optional[Any] = None,
        ground_truth: Optional[Any] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add data to the stream buffer."""
        self._buffer.append({
            'features': features,
            'prediction': prediction,
            'ground_truth': ground_truth,
            'metadata': metadata or {}
        })
        
        # Check if we should process the buffer
        self._check_and_process()
    
    def _check_and_process(self) -> None:
        """Check if we should process the buffer and do so if needed."""
        current_time = time.time()
        
        # Process if buffer is full or enough time has passed
        if (len(self._buffer) >= self.buffer_size or 
            (current_time - self._last_process_time) >= self.process_interval):
            self.process_buffer()
    
    def process_buffer(self) -> None:
        """Process all data in the buffer."""
        if not self._buffer:
            return
            
        try:
            # Extract data from buffer
            features = [item['features'] for item in self._buffer]
            predictions = [item['prediction'] for item in self._buffer]
            ground_truths = [item['ground_truth'] for item in self._buffer]
            metadata = [item['metadata'] for item in self._buffer]
            
            # Process through pipeline
            self.pipeline.process_predictions(
                features=features,
                predictions=predictions,
                ground_truths=ground_truths,
                metadata=metadata
            )
            
            # Clear buffer
            self._buffer.clear()
            self._last_process_time = time.time()
            
        except Exception as e:
            logger.error(f"Error processing buffer: {e}")
            # Optionally, implement retry logic here
            
    def flush(self) -> None:
        """Force processing of all remaining data in the buffer."""
        self.process_buffer()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()

# Example usage
if __name__ == "__main__":
    # Example model
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Create monitoring pipeline
    pipeline = MonitoringPipeline(
        model=model,
        feature_names=feature_names,
        target_name='target'
    )
    
    # Process some predictions
    for i in range(100):
        idx = np.random.randint(0, len(X))
        features = dict(zip(feature_names, X[i]))
        
        # Simulate occasional ground truth availability
        ground_truth = y[i] if i % 5 == 0 else None
        
        pipeline.process_predictions(
            features=features,
            ground_truths=ground_truth
        )
    
    # Get status
    status = pipeline.get_status()
    print("\nMonitoring status:")
    print(json.dumps(status, indent=2))
