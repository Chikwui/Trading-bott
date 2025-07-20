"""
Model Monitoring and Drift Detection Module

This module provides functionality for monitoring model performance and detecting data drift.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import warnings
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error

class DriftType(str, Enum):
    """Types of data drift that can be detected."""
    COVARIATE_DRIFT = "covariate_drift"
    PREDICTION_DRIFT = "prediction_drift"
    CONCEPT_DRIFT = "concept_drift"
    DATA_QUALITY_ISSUE = "data_quality_issue"

@dataclass
class DriftAlert:
    """Represents a drift detection alert."""
    drift_type: DriftType
    feature: str
    p_value: float
    threshold: float = 0.05
    timestamp: str = None
    message: str = ""
    
    @property
    def is_drift_detected(self) -> bool:
        """Check if drift is detected based on p-value threshold."""
        return self.p_value < self.threshold
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "drift_type": self.drift_type.value,
            "feature": self.feature,
            "p_value": self.p_value,
            "threshold": self.threshold,
            "timestamp": self.timestamp or pd.Timestamp.utcnow().isoformat(),
            "message": self.message,
            "is_drift_detected": self.is_drift_detected
        }

class DriftDetector:
    """Detects data drift between training and production data."""
    
    def __init__(self, reference_data: pd.DataFrame, threshold: float = 0.05):
        """
        Initialize the drift detector with reference data.
        
        Args:
            reference_data: DataFrame containing the reference/training data
            threshold: P-value threshold for drift detection (default: 0.05)
        """
        self.reference_data = reference_data
        self.threshold = threshold
        self._reference_stats = self._calculate_reference_stats()
    
    def _calculate_reference_stats(self) -> Dict[str, dict]:
        """Calculate reference statistics for each feature."""
        stats = {}
        for col in self.reference_data.columns:
            if pd.api.types.is_numeric_dtype(self.reference_data[col]):
                stats[col] = {
                    'mean': float(self.reference_data[col].mean()),
                    'std': float(self.reference_data[col].std()),
                    'min': float(self.reference_data[col].min()),
                    'max': float(self.reference_data[col].max()),
                    'dtype': 'numeric'
                }
            else:
                # For categorical data
                stats[col] = {
                    'value_counts': self.reference_data[col].value_counts(normalize=True).to_dict(),
                    'dtype': 'categorical'
                }
        return stats
    
    def detect_drift(
        self, 
        current_data: pd.DataFrame,
        predictions: Optional[np.ndarray] = None,
        reference_predictions: Optional[np.ndarray] = None
    ) -> List[DriftAlert]:
        """
        Detect drift between reference and current data.
        
        Args:
            current_data: Current production data to compare against reference
            predictions: Current model predictions (optional, for prediction drift)
            reference_predictions: Reference model predictions (optional, for concept drift)
            
        Returns:
            List of DriftAlert objects
        """
        alerts = []
        
        # Check for covariate drift
        alerts.extend(self._detect_covariate_drift(current_data))
        
        # Check for prediction drift if predictions are provided
        if predictions is not None:
            alerts.extend(self._detect_prediction_drift(predictions, reference_predictions))
        
        return [alert for alert in alerts if alert.is_drift_detected]
    
    def _detect_covariate_drift(self, current_data: pd.DataFrame) -> List[DriftAlert]:
        """Detect drift in feature distributions."""
        alerts = []
        
        for col in self.reference_data.columns:
            if col not in current_data.columns:
                alerts.append(DriftAlert(
                    drift_type=DriftType.DATA_QUALITY_ISSUE,
                    feature=col,
                    p_value=0.0,
                    message=f"Feature {col} is missing in current data"
                ))
                continue
                
            ref_col = self.reference_data[col].dropna()
            curr_col = current_data[col].dropna()
            
            if len(ref_col) == 0 or len(curr_col) == 0:
                continue
                
            if self._reference_stats[col]['dtype'] == 'numeric':
                # Use Kolmogorov-Smirnov test for numeric features
                _, p_value = stats.ks_2samp(ref_col, curr_col)
                
                alerts.append(DriftAlert(
                    drift_type=DriftType.COVARIATE_DRIFT,
                    feature=col,
                    p_value=p_value,
                    threshold=self.threshold,
                    message=f"Numeric feature {col} shows drift (p={p_value:.4f})"
                ))
            else:
                # Use Chi-square test for categorical features
                ref_counts = self._reference_stats[col]['value_counts']
                curr_counts = current_data[col].value_counts(normalize=True).to_dict()
                
                # Align categories
                all_categories = set(ref_counts.keys()).union(set(curr_counts.keys()))
                ref_freq = [ref_counts.get(cat, 0) for cat in all_categories]
                curr_freq = [curr_counts.get(cat, 0) for cat in all_categories]
                
                _, p_value = stats.chisquare(f_obs=curr_freq, f_exp=ref_freq)
                
                alerts.append(DriftAlert(
                    drift_type=DriftType.COVARIATE_DRIFT,
                    feature=col,
                    p_value=p_value,
                    threshold=self.threshold,
                    message=f"Categorical feature {col} shows drift (p={p_value:.4f})"
                ))
        
        return alerts
    
    def _detect_prediction_drift(
        self, 
        predictions: np.ndarray, 
        reference_predictions: Optional[np.ndarray] = None
    ) -> List[DriftAlert]:
        """Detect drift in model predictions."""
        alerts = []
        
        if reference_predictions is not None:
            # Concept drift: Compare current predictions to reference predictions
            if len(predictions) != len(reference_predictions):
                warnings.warn("Length of predictions and reference_predictions do not match")
                return []
                
            # For classification tasks
            if np.issubdtype(predictions.dtype, np.integer) or len(predictions.shape) == 1:
                accuracy = accuracy_score(reference_predictions, predictions)
                f1 = f1_score(reference_predictions, predictions, average='weighted')
                
                alerts.extend([
                    DriftAlert(
                        drift_type=DriftType.CONCEPT_DRIFT,
                        feature="predictions",
                        p_value=1 - accuracy,  # Simple heuristic
                        threshold=self.threshold,
                        message=f"Concept drift detected (accuracy={accuracy:.4f}, F1={f1:.4f})"
                    )
                ])
            else:
                # For regression tasks
                mse = mean_squared_error(reference_predictions, predictions)
                alerts.append(
                    DriftAlert(
                        drift_type=DriftType.CONCEPT_DRIFT,
                        feature="predictions",
                        p_value=1 / (1 + mse),  # Simple heuristic
                        threshold=self.threshold,
                        message=f"Concept drift detected (MSE={mse:.4f})"
                    )
                )
        
        # Check for prediction distribution shift
        if len(predictions) > 0:
            if np.issubdtype(predictions.dtype, np.integer):
                # For classification: compare class distribution
                unique, counts = np.unique(predictions, return_counts=True)
                pred_dist = dict(zip(unique, counts / len(predictions)))
                
                # Compare to reference distribution if available
                if hasattr(self, 'reference_predictions'):
                    ref_unique, ref_counts = np.unique(
                        self.reference_predictions, 
                        return_counts=True
                    )
                    ref_dist = dict(zip(ref_unique, ref_counts / len(self.reference_predictions)))
                    
                    # Calculate KL divergence or similar
                    # For simplicity, we'll use a chi-square test
                    all_classes = set(pred_dist.keys()).union(set(ref_dist.keys()))
                    pred_freq = [pred_dist.get(c, 0) for c in all_classes]
                    ref_freq = [ref_dist.get(c, 0) for c in all_classes]
                    
                    _, p_value = stats.chisquare(f_obs=pred_freq, f_exp=ref_freq)
                    
                    alerts.append(
                        DriftAlert(
                            drift_type=DriftType.PREDICTION_DRIFT,
                            feature="predictions",
                            p_value=p_value,
                            threshold=self.threshold,
                            message=f"Prediction distribution drift detected (p={p_value:.4f})"
                        )
                    )
        
        return alerts

class ModelMonitor:
    """Monitors model performance and data quality in production."""
    
    def __init__(self, model, reference_data: pd.DataFrame, threshold: float = 0.05):
        """
        Initialize the model monitor.
        
        Args:
            model: The trained model to monitor
            reference_data: Reference/training data for drift detection
            threshold: P-value threshold for drift detection
        """
        self.model = model
        self.drift_detector = DriftDetector(reference_data, threshold)
        self.performance_metrics = []
        self.drift_alerts = []
    
    def check_drift(self, X: pd.DataFrame, y_true: Optional[np.ndarray] = None) -> List[DriftAlert]:
        """
        Check for data drift and update performance metrics.
        
        Args:
            X: Input features
            y_true: True labels (optional)
            
        Returns:
            List of drift alerts
        """
        # Get predictions
        y_pred = self.model.predict(X)
        
        # Check for covariate drift
        alerts = self.drift_detector.detect_covariate_drift(X)
        
        # Check for prediction drift if we have true labels
        if y_true is not None:
            alerts.extend(
                self.drift_detector.detect_prediction_drift(y_pred, y_true)
            )
        
        # Store alerts
        self.drift_alerts.extend(alerts)
        return [a for a in alerts if a.is_drift_detected]
    
    def log_performance(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        metrics: dict = None
    ) -> dict:
        """
        Log model performance metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            metrics: Additional metrics to log
            
        Returns:
            Dictionary of logged metrics
        """
        result = {}
        
        # Basic classification metrics
        if np.issubdtype(y_pred.dtype, np.integer) or len(y_pred.shape) == 1:
            result['accuracy'] = accuracy_score(y_true, y_pred)
            result['f1_score'] = f1_score(y_true, y_pred, average='weighted')
        else:
            # Regression metrics
            result['mse'] = mean_squared_error(y_true, y_pred)
            result['rmse'] = np.sqrt(result['mse'])
        
        # Add custom metrics
        if metrics:
            result.update(metrics)
        
        # Add timestamp
        result['timestamp'] = pd.Timestamp.utcnow().isoformat()
        
        # Store metrics
        self.performance_metrics.append(result)
        return result
    
    def get_performance_history(self, window: int = None) -> pd.DataFrame:
        """
        Get performance metrics history.
        
        Args:
            window: Number of most recent entries to return (None for all)
            
        Returns:
            DataFrame of performance metrics over time
        """
        df = pd.DataFrame(self.performance_metrics)
        if window and len(df) > window:
            return df.iloc[-window:]
        return df
    
    def get_drift_alerts(self, recent: bool = True) -> List[DriftAlert]:
        """
        Get drift alerts.
        
        Args:
            recent: If True, only return alerts where is_drift_detected is True
            
        Returns:
            List of drift alerts
        """
        if recent:
            return [a for a in self.drift_alerts if a.is_drift_detected]
        return self.drift_alerts

# Example usage
if __name__ == "__main__":
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    
    # Generate sample data
    X_ref, y_ref = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
    X_curr, y_curr = make_classification(n_samples=100, n_features=20, n_classes=2, random_state=123)
    
    # Train a simple model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_ref, y_ref)
    
    # Initialize monitor
    monitor = ModelMonitor(model, pd.DataFrame(X_ref))
    
    # Check for drift
    alerts = monitor.check_drift(pd.DataFrame(X_curr), y_curr)
    
    # Log performance
    y_pred = model.predict(X_curr)
    metrics = monitor.log_performance(y_curr, y_pred)
    
    print(f"Detected {len(alerts)} drift alerts")
    print("Performance metrics:", metrics)

"""
Drift Detection Module

This module provides functionality for detecting data and concept drift in ML models.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import warnings
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from alibi_detect import KSDrift, CVMDrift, MMDDrift, LSDDDrift
from alibi_detect.saving import save_detector, load_detector
import joblib
import json
from pathlib import Path
from datetime import datetime

# Configure logging
logger = logging.getLogger(__name__)

class DriftType(str, Enum):
    """Types of drift that can be detected."""
    DATA_DRIFT = "data_drift"
    CONCEPT_DRIFT = "concept_drift"
    MODEL_DECAY = "model_decay"
    ANOMALY = "anomaly"

@dataclass
class DriftResult:
    """Container for drift detection results."""
    drift_detected: bool
    drift_type: DriftType
    p_value: Optional[float] = None
    test_statistic: Optional[float] = None
    threshold: Optional[float] = None
    feature_importance: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

class BaseDriftDetector:
    """Base class for drift detectors."""
    
    def __init__(self, name: str, **kwargs):
        self.name = name
        self._is_fitted = False
        
    def fit(self, reference_data: Union[pd.DataFrame, np.ndarray], **kwargs) -> None:
        """Fit the detector on reference data."""
        raise NotImplementedError
        
    def detect_drift(
        self, 
        data: Union[pd.DataFrame, np.ndarray],
        **kwargs
    ) -> DriftResult:
        """Detect drift in the provided data."""
        raise NotImplementedError
        
    def save(self, filepath: str) -> None:
        """Save the detector to disk."""
        raise NotImplementedError
        
    @classmethod
    def load(cls, filepath: str) -> 'BaseDriftDetector':
        """Load a detector from disk."""
        raise NotImplementedError
        
    @property
    def is_fitted(self) -> bool:
        """Check if the detector has been fitted."""
        return self._is_fitted

class StatisticalDriftDetector(BaseDriftDetector):
    """Statistical drift detector using various statistical tests."""
    
    def __init__(
        self,
        name: str = "statistical_drift_detector",
        test_type: str = "ks",  # 'ks', 'cvm', 'mmd', 'lsdd'
        p_val_threshold: float = 0.05,
        correction: str = "bonferroni",
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.test_type = test_type
        self.p_val_threshold = p_val_threshold
        self.correction = correction
        self.reference_data = None
        self.feature_names = None
        
    def fit(self, reference_data: Union[pd.DataFrame, np.ndarray], **kwargs) -> None:
        """Fit the detector on reference data."""
        if isinstance(reference_data, pd.DataFrame):
            self.feature_names = reference_data.columns.tolist()
            reference_data = reference_data.values
            
        self.reference_data = reference_data
        self._is_fitted = True
        
    def detect_drift(
        self, 
        data: Union[pd.DataFrame, np.ndarray],
        **kwargs
    ) -> DriftResult:
        """Detect drift using statistical tests."""
        if not self._is_fitted:
            raise RuntimeError("Detector must be fitted before drift detection")
            
        if isinstance(data, pd.DataFrame):
            data = data.values
            
        # Initialize detector based on test type
        if self.test_type == 'ks':
            detector = KSDrift(
                self.reference_data,
                p_val=self.p_val_threshold,
                corrected_p_val='bonferroni' if self.correction else None
            )
        elif self.test_type == 'cvm':
            detector = CVMDrift(
                self.reference_data,
                p_val=self.p_val_threshold,
                corrected_p_val='bonferroni' if self.correction else None
            )
        elif self.test_type == 'mmd':
            detector = MMDDrift(
                self.reference_data,
                p_val=self.p_val_threshold,
                n_permutations=100
            )
        elif self.test_type == 'lsdd':
            detector = LSDDDrift(
                self.reference_data,
                p_val=self.p_val_threshold
            )
        else:
            raise ValueError(f"Unsupported test type: {self.test_type}")
            
        # Run drift detection
        preds = detector.predict(data)
        
        # Extract feature importance if available
        feature_importance = None
        if hasattr(preds, 'data') and 'feature_score' in preds.data:
            feature_scores = preds.data['feature_score']
            if self.feature_names and len(self.feature_names) == len(feature_scores):
                feature_importance = dict(zip(self.feature_names, feature_scores))
                
        return DriftResult(
            drift_detected=preds['data']['is_drift'] == 1,
            drift_type=DriftType.DATA_DRIFT,
            p_value=float(preds['data']['p_val']),
            test_statistic=float(preds['data']['distance']),
            threshold=self.p_val_threshold,
            feature_importance=feature_importance,
            metadata={
                'test_type': self.test_type,
                'correction': self.correction,
                'n_reference_samples': len(self.reference_data),
                'n_test_samples': len(data)
            }
        )
        
    def save(self, filepath: str) -> None:
        """Save the detector to disk."""
        detector_path = Path(filepath)
        detector_path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config = {
            'name': self.name,
            'test_type': self.test_type,
            'p_val_threshold': self.p_val_threshold,
            'correction': self.correction,
            'feature_names': self.feature_names,
            'is_fitted': self._is_fitted
        }
        
        with open(detector_path / 'config.json', 'w') as f:
            json.dump(config, f)
            
        # Save reference data
        if self.reference_data is not None:
            np.save(detector_path / 'reference_data.npy', self.reference_data)
            
    @classmethod
    def load(cls, filepath: str) -> 'StatisticalDriftDetector':
        """Load a detector from disk."""
        detector_path = Path(filepath)
        
        # Load config
        with open(detector_path / 'config.json', 'r') as f:
            config = json.load(f)
            
        # Create detector instance
        detector = cls(
            name=config['name'],
            test_type=config['test_type'],
            p_val_threshold=config['p_val_threshold'],
            correction=config['correction']
        )
        
        # Load reference data if it exists
        ref_data_path = detector_path / 'reference_data.npy'
        if ref_data_path.exists():
            reference_data = np.load(ref_data_path)
            detector.fit(reference_data)
            
        return detector

class ModelDriftDetector(BaseDriftDetector):
    """Model-based drift detector using classifier-based approaches."""
    
    def __init__(
        self,
        name: str = "model_drift_detector",
        model_type: str = "iforest",  # 'iforest', 'ocsvm', 'lof', 'ee'
        contamination: float = 0.1,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.model_type = model_type
        self.contamination = contamination
        self.model = None
        self.feature_names = None
        
    def fit(self, reference_data: Union[pd.DataFrame, np.ndarray], **kwargs) -> None:
        """Fit the detector on reference data."""
        if isinstance(reference_data, pd.DataFrame):
            self.feature_names = reference_data.columns.tolist()
            reference_data = reference_data.values
            
        # Initialize and fit the model
        if self.model_type == "iforest":
            self.model = IsolationForest(
                contamination=self.contamination,
                random_state=42,
                **kwargs
            )
        elif self.model_type == "ee":
            self.model = EllipticEnvelope(
                contamination=self.contamination,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
            
        self.model.fit(reference_data)
        self._is_fitted = True
        
    def detect_drift(
        self, 
        data: Union[pd.DataFrame, np.ndarray],
        **kwargs
    ) -> DriftResult:
        """Detect drift using the trained model."""
        if not self._is_fitted:
            raise RuntimeError("Detector must be fitted before drift detection")
            
        if isinstance(data, pd.DataFrame):
            data = data.values
            
        # Get anomaly scores (lower means more anomalous)
        scores = self.model.score_samples(data)
        
        # For IsolationForest, scores are negative log anomaly scores
        # Lower scores indicate more anomalous points
        anomaly_mask = scores < np.percentile(scores, self.contamination * 100)
        
        # Calculate drift metrics
        drift_ratio = np.mean(anomaly_mask)
        drift_detected = drift_ratio > self.contamination
        
        # Feature importance (if available)
        feature_importance = None
        if hasattr(self.model, 'feature_importances_'):
            if self.feature_names is not None:
                feature_importance = dict(zip(self.feature_names, self.model.feature_importances_))
                
        return DriftResult(
            drift_detected=drift_detected,
            drift_type=DriftType.MODEL_DECAY,
            p_value=float(1 - drift_ratio),
            test_statistic=float(np.mean(scores)),
            threshold=self.contamination,
            feature_importance=feature_importance,
            metadata={
                'model_type': self.model_type,
                'contamination': self.contamination,
                'n_test_samples': len(data),
                'anomaly_ratio': float(drift_ratio)
            }
        )
        
    def save(self, filepath: str) -> None:
        """Save the detector to disk."""
        detector_path = Path(filepath)
        detector_path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config = {
            'name': self.name,
            'model_type': self.model_type,
            'contamination': self.contamination,
            'feature_names': self.feature_names,
            'is_fitted': self._is_fitted
        }
        
        with open(detector_path / 'config.json', 'w') as f:
            json.dump(config, f)
            
        # Save model
        if self.model is not None:
            joblib.dump(self.model, detector_path / 'model.joblib')
            
    @classmethod
    def load(cls, filepath: str) -> 'ModelDriftDetector':
        """Load a detector from disk."""
        detector_path = Path(filepath)
        
        # Load config
        with open(detector_path / 'config.json', 'r') as f:
            config = json.load(f)
            
        # Create detector instance
        detector = cls(
            name=config['name'],
            model_type=config['model_type'],
            contamination=config['contamination']
        )
        
        # Load model if it exists
        model_path = detector_path / 'model.joblib'
        if model_path.exists():
            detector.model = joblib.load(model_path)
            detector.feature_names = config.get('feature_names')
            detector._is_fitted = config.get('is_fitted', False)
            
        return detector

class DriftDetectorFactory:
    """Factory for creating drift detectors."""
    
    @staticmethod
    def create_detector(
        detector_type: str,
        name: str,
        **kwargs
    ) -> BaseDriftDetector:
        """Create a drift detector of the specified type."""
        detector_type = detector_type.lower()
        
        if detector_type in ['ks', 'cvm', 'mmd', 'lsdd']:
            return StatisticalDriftDetector(
                name=name,
                test_type=detector_type,
                **kwargs
            )
        elif detector_type in ['iforest', 'ee']:
            return ModelDriftDetector(
                name=name,
                model_type=detector_type,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported detector type: {detector_type}")
            
    @staticmethod
    def from_config(config: Dict[str, Any]) -> BaseDriftDetector:
        """Create a detector from a configuration dictionary."""
        detector_type = config.pop('type')
        name = config.pop('name', f"{detector_type}_detector")
        return DriftDetectorFactory.create_detector(detector_type, name, **config)
