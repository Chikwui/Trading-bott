"""
SHAP Explainer for Model Interpretability

This module provides SHAP-based model explainability for understanding model predictions.
"""
import numpy as np
import pandas as pd
import shap
import json
import joblib
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure logging
logger = logging.getLogger(__name__)

class ExplanationType(str, Enum):
    """Types of SHAP explanations."""
    SUMMARY = "summary"
    FORCE = "force"
    DECISION = "decision"
    DEPENDENCE = "dependence"
    BAR = "bar"
    BEESWARM = "beeswarm"
    WATERFALL = "waterfall"

@dataclass
class ExplanationResult:
    """Container for SHAP explanation results."""
    shap_values: np.ndarray
    expected_value: float
    feature_names: List[str]
    data: np.ndarray
    explanation_type: ExplanationType
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert explanation to dictionary."""
        return {
            'shap_values': self.shap_values.tolist(),
            'expected_value': float(self.expected_value),
            'feature_names': self.feature_names,
            'data': self.data.tolist(),
            'explanation_type': self.explanation_type.value,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExplanationResult':
        """Create explanation from dictionary."""
        return cls(
            shap_values=np.array(data['shap_values']),
            expected_value=data['expected_value'],
            feature_names=data['feature_names'],
            data=np.array(data['data']),
            explanation_type=ExplanationType(data['explanation_type']),
            metadata=data.get('metadata', {})
        )

class SHAPExplainer:
    """SHAP-based model explainer for interpretability."""
    
    def __init__(
        self,
        model: Any,
        feature_names: Optional[List[str]] = None,
        model_type: str = 'tree',  # 'tree', 'linear', 'kernel', 'deep', etc.
        **kwargs
    ):
        """
        Initialize the SHAP explainer.
        
        Args:
            model: The trained model to explain
            feature_names: List of feature names
            model_type: Type of model ('tree', 'linear', 'kernel', 'deep', etc.)
            **kwargs: Additional arguments for the SHAP explainer
        """
        self.model = model
        self.feature_names = feature_names or []
        self.model_type = model_type.lower()
        self.explainer = self._create_explainer(**kwargs)
        
    def _create_explainer(self, **kwargs) -> Any:
        """Create the appropriate SHAP explainer based on model type."""
        if self.model_type == 'tree':
            return shap.TreeExplainer(self.model, **kwargs)
        elif self.model_type == 'linear':
            return shap.LinearExplainer(self.model, **kwargs)
        elif self.model_type == 'kernel':
            return shap.KernelExplainer(self.model.predict, **kwargs)
        elif self.model_type == 'deep':
            return shap.DeepExplainer(self.model, **kwargs)
        else:
            # Use the generic explainer as fallback
            return shap.Explainer(self.model, **kwargs)
    
    def explain(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        explanation_type: Union[str, ExplanationType] = ExplanationType.SUMMARY,
        **kwargs
    ) -> ExplanationResult:
        """
        Generate SHAP explanations for the input data.
        
        Args:
            X: Input data to explain
            explanation_type: Type of explanation to generate
            **kwargs: Additional arguments for the explanation
            
        Returns:
            ExplanationResult containing the SHAP values and metadata
        """
        if isinstance(explanation_type, str):
            explanation_type = ExplanationType(explanation_type.lower())
            
        # Convert DataFrame to numpy array if needed
        if isinstance(X, pd.DataFrame):
            if not self.feature_names:
                self.feature_names = X.columns.tolist()
            X_array = X.values
        else:
            X_array = np.array(X)
        
        # Calculate SHAP values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap_values = self.explainer.shap_values(X_array, **kwargs)
            
            # Handle multi-output models
            if isinstance(shap_values, list):
                shap_values = np.array(shap_values)
                if len(shap_values.shape) == 3:  # For multi-class classification
                    shap_values = np.transpose(shap_values, (1, 2, 0))
            
            # Get expected value
            if hasattr(self.explainer, 'expected_value'):
                expected_value = self.explainer.expected_value
                if isinstance(expected_value, np.ndarray):
                    expected_value = expected_value[0]  # Take first output for simplicity
            else:
                expected_value = float(np.mean(self.model.predict(X_array)))
        
        # Create explanation result
        result = ExplanationResult(
            shap_values=shap_values,
            expected_value=expected_value,
            feature_names=self.feature_names,
            data=X_array,
            explanation_type=explanation_type,
            metadata={
                'model_type': self.model_type,
                'n_samples': len(X_array),
                'n_features': X_array.shape[1] if len(X_array.shape) > 1 else 1
            }
        )
        
        return result
    
    def plot_explanation(
        self,
        explanation: ExplanationResult,
        plot_type: Optional[Union[str, ExplanationType]] = None,
        plot_index: int = 0,
        max_display: int = 10,
        output_file: Optional[str] = None,
        **kwargs
    ) -> Optional[go.Figure]:
        """
        Plot SHAP explanations.
        
        Args:
            explanation: ExplanationResult from explain()
            plot_type: Type of plot to generate (defaults to explanation_type)
            plot_index: Index of the instance to plot (for individual explanations)
            max_display: Maximum number of features to display
            output_file: If provided, save the plot to this file
            **kwargs: Additional arguments for the plot
            
        Returns:
            Plotly figure if plotly is available, else None
        """
        if plot_type is None:
            plot_type = explanation.explanation_type
        elif isinstance(plot_type, str):
            plot_type = ExplanationType(plot_type.lower())
        
        try:
            if plot_type == ExplanationType.SUMMARY:
                fig = self._plot_summary(explanation, max_display, **kwargs)
            elif plot_type == ExplanationType.BAR:
                fig = self._plot_bar(explanation, max_display, **kwargs)
            elif plot_type == ExplanationType.WATERFALL:
                fig = self._plot_waterfall(explanation, plot_index, **kwargs)
            elif plot_type in (ExplanationType.FORCE, ExplanationType.DECISION):
                fig = self._plot_force(explanation, plot_index, **kwargs)
            elif plot_type == ExplanationType.DEPENDENCE:
                fig = self._plot_dependence(explanation, **kwargs)
            elif plot_type == ExplanationType.BEESWARM:
                fig = self._plot_beeswarm(explanation, max_display, **kwargs)
            else:
                raise ValueError(f"Unsupported plot type: {plot_type}")
            
            if output_file:
                fig.write_image(output_file)
                
            return fig
            
        except ImportError as e:
            logger.error(f"Failed to create plot: {e}")
            return None
    
    def _plot_summary(
        self,
        explanation: ExplanationResult,
        max_display: int = 10,
        **kwargs
    ) -> go.Figure:
        """Create a summary plot of SHAP values."""
        if len(explanation.shap_values.shape) == 3:  # Multi-class
            shap_values = explanation.shap_values[0]  # Take first class for simplicity
        else:
            shap_values = explanation.shap_values
            
        # Calculate mean absolute SHAP values for each feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Sort features by importance
        if len(explanation.feature_names) == mean_abs_shap.shape[0]:
            sorted_idx = np.argsort(mean_abs_shap)[::-1][:max_display]
            feature_names = [explanation.feature_names[i] for i in sorted_idx]
            values = mean_abs_shap[sorted_idx]
        else:
            values = np.sort(mean_abs_shap)[::-1][:max_display]
            feature_names = [f"Feature {i}" for i in range(len(values))]
        
        # Create bar chart
        fig = go.Figure(go.Bar(
            x=values,
            y=feature_names,
            orientation='h',
            marker_color='#1f77b4'
        ))
        
        fig.update_layout(
            title="SHAP Feature Importance",
            xaxis_title="Mean |SHAP value|",
            yaxis_title="Feature",
            showlegend=False,
            height=400,
            margin=dict(l=100, r=20, t=50, b=50)
        )
        
        return fig
    
    def _plot_bar(
        self,
        explanation: ExplanationResult,
        max_display: int = 10,
        **kwargs
    ) -> go.Figure:
        """Create a bar plot of mean absolute SHAP values."""
        return self._plot_summary(explanation, max_display, **kwargs)
    
    def _plot_waterfall(
        self,
        explanation: ExplanationResult,
        index: int = 0,
        **kwargs
    ) -> go.Figure:
        """Create a waterfall plot for a single prediction."""
        if len(explanation.shap_values.shape) == 3:  # Multi-class
            shap_values = explanation.shap_values[0][index]  # Take first class
        else:
            shap_values = explanation.shap_values[index]
            
        # Sort features by absolute SHAP value
        abs_shap = np.abs(shap_values)
        sorted_idx = np.argsort(-abs_shap)
        
        # Prepare data for waterfall
        feature_names = []
        values = []
        
        for i in sorted_idx:
            if explanation.feature_names and i < len(explanation.feature_names):
                feature_names.append(explanation.feature_names[i])
            else:
                feature_names.append(f"Feature {i}")
            values.append(float(shap_values[i]))
        
        # Add base value and final prediction
        base_value = float(explanation.expected_value)
        final_value = base_value + sum(values)
        
        # Create waterfall chart
        fig = go.Figure(go.Waterfall(
            name="SHAP Values",
            orientation="h",
            measure=["relative"] * len(values) + ["total"],
            x=values + [final_value],
            y=feature_names + ["Prediction"],
            base=base_value,
            decreasing=dict(marker=dict(color="#ff7f0e")),
            increasing=dict(marker=dict(color="#1f77b4")),
            totals=dict(marker=dict(color="#2ca02c"))
        ))
        
        fig.update_layout(
            title=f"SHAP Waterfall Plot (Sample {index})",
            xaxis_title="Model Output",
            yaxis_title="Feature",
            showlegend=False,
            height=600,
            margin=dict(l=150, r=20, t=50, b=50)
        )
        
        return fig
    
    def _plot_force(
        self,
        explanation: ExplanationResult,
        index: int = 0,
        **kwargs
    ) -> go.Figure:
        """Create a force plot for a single prediction."""
        if len(explanation.shap_values.shape) == 3:  # Multi-class
            shap_values = explanation.shap_values[0][index]  # Take first class
        else:
            shap_values = explanation.shap_values[index]
        
        # Prepare data
        if explanation.feature_names and len(explanation.feature_names) == len(shap_values):
            feature_names = explanation.feature_names
        else:
            feature_names = [f"Feature {i}" for i in range(len(shap_values))]
        
        # Sort features by absolute SHAP value
        abs_shap = np.abs(shap_values)
        sorted_idx = np.argsort(-abs_shap)
        
        sorted_features = [feature_names[i] for i in sorted_idx]
        sorted_values = [float(shap_values[i]) for i in sorted_idx]
        
        # Create horizontal bar chart
        fig = go.Figure(go.Bar(
            x=sorted_values,
            y=sorted_features,
            orientation='h',
            marker_color=['#1f77b4' if v > 0 else '#ff7f0e' for v in sorted_values]
        ))
        
        # Add vertical line at x=0
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        
        # Add base value line
        base_value = float(explanation.expected_value)
        fig.add_vline(x=base_value, line_dash="dot", line_color="green", 
                     annotation_text=f"Base Value: {base_value:.4f}")
        
        # Add prediction line
        prediction = base_value + sum(shap_values)
        fig.add_vline(x=prediction, line_dash="dot", line_color="red",
                     annotation_text=f"Prediction: {prediction:.4f}")
        
        fig.update_layout(
            title=f"SHAP Force Plot (Sample {index})",
            xaxis_title="SHAP Value",
            yaxis_title="Feature",
            showlegend=False,
            height=max(300, 30 * len(sorted_features)),
            margin=dict(l=150, r=20, t=50, b=50)
        )
        
        return fig
    
    def _plot_dependence(
        self,
        explanation: ExplanationResult,
        feature_index: int = 0,
        interaction_index: Optional[int] = None,
        **kwargs
    ) -> go.Figure:
        """Create a dependence plot for a feature."""
        if len(explanation.shap_values.shape) == 3:  # Multi-class
            shap_values = explanation.shap_values[0]  # Take first class
        else:
            shap_values = explanation.shap_values
        
        # Get feature data
        if explanation.feature_names:
            feature_name = explanation.feature_names[feature_index]
            if interaction_index is not None:
                interaction_name = explanation.feature_names[interaction_index]
            else:
                interaction_name = None
        else:
            feature_name = f"Feature {feature_index}"
            interaction_name = f"Feature {interaction_index}" if interaction_index is not None else None
        
        # Get data for the selected feature
        feature_data = explanation.data[:, feature_index]
        shap_values_feature = shap_values[:, feature_index]
        
        # Create scatter plot
        fig = go.Figure()
        
        if interaction_index is not None:
            # Color points by interaction feature
            interaction_data = explanation.data[:, interaction_index]
            
            # Create color scale
            fig.add_trace(go.Scatter(
                x=feature_data,
                y=shap_values_feature,
                mode='markers',
                marker=dict(
                    color=interaction_data,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title=interaction_name)
                ),
                hovertemplate=f"{feature_name}: %{{x}}<br>SHAP: %{{y:.4f}}<br>{interaction_name}: %{{marker.color:.4f}}<extra></extra>"
            ))
        else:
            # Simple scatter plot
            fig.add_trace(go.Scatter(
                x=feature_data,
                y=shap_values_feature,
                mode='markers',
                marker=dict(color='#1f77b4'),
                hovertemplate=f"{feature_name}: %{{x}}<br>SHAP: %{{y:.4f}}<extra></extra>"
            ))
        
        # Add trend line
        if len(feature_data) > 1:
            try:
                from scipy import stats
                slope, intercept, r_value, p_value, std_err = stats.linregress(feature_data, shap_values_feature)
                x_range = np.linspace(min(feature_data), max(feature_data), 100)
                y_range = intercept + slope * x_range
                
                fig.add_trace(go.Scatter(
                    x=x_range,
                    y=y_range,
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name=f'Trend (r²={r_value**2:.2f})'
                ))
            except:
                pass
        
        fig.update_layout(
            title=f"SHAP Dependence Plot: {feature_name}" + 
                 (f" vs {interaction_name}" if interaction_name else ""),
            xaxis_title=feature_name,
            yaxis_title=f"SHAP value for {feature_name}",
            showlegend=interaction_index is not None,
            height=500,
            margin=dict(l=50, r=20, t=50, b=50)
        )
        
        return fig
    
    def _plot_beeswarm(
        self,
        explanation: ExplanationResult,
        max_display: int = 10,
        **kwargs
    ) -> go.Figure:
        """Create a beeswarm plot of SHAP values."""
        if len(explanation.shap_values.shape) == 3:  # Multi-class
            shap_values = explanation.shap_values[0]  # Take first class
        else:
            shap_values = explanation.shap_values
        
        # Calculate mean absolute SHAP values for each feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Sort features by importance
        if len(explanation.feature_names) == mean_abs_shap.shape[0]:
            sorted_idx = np.argsort(mean_abs_shap)[::-1][:max_display]
            feature_names = [explanation.feature_names[i] for i in sorted_idx]
            values = shap_values[:, sorted_idx]
        else:
            values = np.take_along_axis(shap_values, np.argsort(-mean_abs_shap)[:max_display], axis=1)
            feature_names = [f"Feature {i}" for i in range(values.shape[1])]
        
        # Create figure with subplots
        fig = make_subplots(rows=1, cols=len(feature_names), 
                           shared_yaxes=True, horizontal_spacing=0.01)
        
        # Add beeswarm plots for each feature
        for i, (feature, feat_name) in enumerate(zip(values.T, feature_names), 1):
            # Add jitter for x-axis
            x_jitter = np.random.normal(0, 0.05, size=len(feature))
            
            # Create color based on value
            colors = ['#1f77b4' if v > 0 else '#ff7f0e' for v in feature]
            
            fig.add_trace(
                go.Scatter(
                    x=x_jitter + i,
                    y=feature,
                    mode='markers',
                    marker=dict(
                        color=colors,
                        size=8,
                        opacity=0.6,
                        line=dict(width=0.5, color='DarkSlateGrey')
                    ),
                    name=feat_name,
                    hoverinfo='y+name',
                    showlegend=False
                ),
                row=1, col=i
            )
            
            # Add box plot for distribution
            fig.add_trace(
                go.Box(
                    y=feature,
                    name=feat_name,
                    boxpoints=False,
                    marker=dict(color='#2ca02c'),
                    line=dict(width=1),
                    showlegend=False
                ),
                row=1, col=i
            )
            
            # Add feature name as x-axis label
            fig.update_xaxes(
                tickvals=[i],
                ticktext=[feat_name],
                row=1, col=i
            )
        
        # Update layout
        fig.update_layout(
            title="SHAP Beeswarm Plot",
            yaxis_title="SHAP Value",
            height=500,
            margin=dict(l=50, r=20, t=50, b=50),
            plot_bgcolor='white'
        )
        
        # Add horizontal line at y=0
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        return fig
    
    def save(self, filepath: str) -> None:
        """
        Save the explainer to disk.
        
        Args:
            filepath: Path to save the explainer
        """
        # Create directory if it doesn't exist
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model separately if it has a save method
        model_path = path.with_suffix('.model.joblib')
        if hasattr(self.model, 'save'):
            self.model.save(str(model_path))
        else:
            joblib.dump(self.model, model_path)
        
        # Save explainer config
        config = {
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'model_path': str(model_path.name)
        }
        
        config_path = path.with_suffix('.config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f)
        
        # Save explainer if it has a save method
        explainer_path = path.with_suffix('.explainer.joblib')
        if hasattr(self.explainer, 'save'):
            self.explainer.save(str(explainer_path))
        else:
            joblib.dump(self.explainer, explainer_path)
    
    @classmethod
    def load(cls, filepath: str) -> 'SHAPExplainer':
        """
        Load an explainer from disk.
        
        Args:
            filepath: Path to the saved explainer
            
        Returns:
            Loaded SHAPExplainer instance
        """
        path = Path(filepath)
        
        # Load config
        config_path = path.with_suffix('.config.json')
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Load model
        model_path = path.parent / config['model_path']
        if model_path.suffix == '.joblib':
            model = joblib.load(model_path)
        else:
            # Handle other model formats if needed
            raise ValueError(f"Unsupported model format: {model_path.suffix}")
        
        # Create explainer instance
        explainer = cls(
            model=model,
            feature_names=config['feature_names'],
            model_type=config['model_type']
        )
        
        # Load explainer if it was saved separately
        explainer_path = path.with_suffix('.explainer.joblib')
        if explainer_path.exists():
            if hasattr(explainer.explainer, 'load'):
                explainer.explainer.load(str(explainer_path))
            else:
                explainer.explainer = joblib.load(explainer_path)
        
        return explainer
