"""
Transformer-based model for time series prediction in trading.

This module implements a transformer architecture specifically designed for financial
time series forecasting, with features like:
- Multi-head self-attention for capturing long-range dependencies
- Positional encoding for sequence information
- Residual connections and layer normalization
- Customizable architecture parameters
"""
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import BaseModel

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """Initialize positional encoding.
        
        Args:
            d_model: Dimension of the model embeddings
            max_len: Maximum sequence length
        """
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch dimension
        self.register_buffer('pe', pe)
    
    def forward(self, x: Tensor) -> Tensor:
        """Add positional encoding to input tensor."""
        return x + self.pe[:, :x.size(1)]

class TransformerModel(BaseModel):
    """Transformer-based model for time series prediction."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "gelu",
        output_activation: Optional[str] = None,
        **kwargs
    ):
        """Initialize the transformer model.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output dimensions
            d_model: Dimension of model embeddings
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            activation: Activation function (gelu/relu)
            output_activation: Output activation function (sigmoid/tanh/None)
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Output layers
        self.output_layer = nn.Linear(d_model, output_dim)
        
        # Output activation
        self.output_activation = None
        if output_activation == 'sigmoid':
            self.output_activation = nn.Sigmoid()
        elif output_activation == 'tanh':
            self.output_activation = nn.Tanh()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Input projection
        x = self.input_proj(x)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        
        # Use the last time step's output for prediction
        x = x[:, -1, :]  # (batch_size, d_model)
        
        # Output layer
        x = self.output_layer(x)  # (batch_size, output_dim)
        
        # Apply output activation if specified
        if self.output_activation is not None:
            x = self.output_activation(x)
            
        return x
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Make predictions on numpy array input."""
        self.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).unsqueeze(0)  # Add batch dimension
            if next(self.parameters()).is_cuda:
                x_tensor = x_tensor.cuda()
            y_pred = self(x_tensor)
            return y_pred.cpu().numpy().squeeze()
    
    def save(self, path: str):
        """Save model to file."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'hyper_parameters': self.hparams
        }, path)
    
    @classmethod
    def load(cls, path: str, **kwargs):
        """Load model from file."""
        checkpoint = torch.load(path, **kwargs)
        model = cls(**checkpoint['hyper_parameters'])
        model.load_state_dict(checkpoint['model_state_dict'])
        return model


class TimeSeriesTransformer(TransformerModel):
    """Specialized transformer for financial time series prediction."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        seq_len: int = 60,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        output_activation: Optional[str] = None,
        num_aux_features: int = 0,
        **kwargs
    ):
        """Initialize the time series transformer.
        
        Args:
            input_dim: Number of input features per time step
            output_dim: Number of output dimensions
            seq_len: Length of input sequence
            d_model: Dimension of model embeddings
            nhead: Number of attention heads
            num_layers: Number of transformer encoder layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout probability
            output_activation: Output activation function
            num_aux_features: Number of auxiliary features (not time-series)
        """
        self.seq_len = seq_len
        self.num_aux_features = num_aux_features
        
        # Calculate total input dimension (time series + auxiliary)
        total_input_dim = input_dim * seq_len + num_aux_features
        
        super().__init__(
            input_dim=total_input_dim,
            output_dim=output_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            output_activation=output_activation,
            **kwargs
        )
        
        # Additional layers for handling auxiliary features
        if num_aux_features > 0:
            self.aux_proj = nn.Linear(num_aux_features, d_model)
    
    def forward(self, x: Tensor, aux: Optional[Tensor] = None) -> Tensor:
        """Forward pass with optional auxiliary features.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            aux: Optional auxiliary features of shape (batch_size, num_aux_features)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        batch_size = x.size(0)
        
        # Flatten the time series data
        x_flat = x.reshape(batch_size, -1)  # (batch_size, seq_len * input_dim)
        
        # Concatenate with auxiliary features if provided
        if aux is not None:
            x_flat = torch.cat([x_flat, aux], dim=1)
        
        # Project to d_model
        x_proj = self.input_proj(x_flat)  # (batch_size, d_model)
        
        # Reshape for transformer (add sequence dimension)
        x_reshaped = x_proj.unsqueeze(1)  # (batch_size, 1, d_model)
        
        # Add positional encoding
        x_encoded = self.pos_encoder(x_reshaped)
        
        # Transformer encoder
        x_transformed = self.transformer_encoder(x_encoded)  # (batch_size, 1, d_model)
        
        # Output layer
        output = self.output_layer(x_transformed.squeeze(1))  # (batch_size, output_dim)
        
        # Apply output activation if specified
        if self.output_activation is not None:
            output = self.output_activation(output)
            
        return output
    
    def predict(self, x: np.ndarray, aux: Optional[np.ndarray] = None) -> np.ndarray:
        """Make predictions on numpy array input with optional auxiliary features."""
        self.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x)
            if aux is not None:
                aux_tensor = torch.FloatTensor(aux)
            else:
                aux_tensor = None
                
            if next(self.parameters()).is_cuda:
                x_tensor = x_tensor.cuda()
                if aux_tensor is not None:
                    aux_tensor = aux_tensor.cuda()
                    
            y_pred = self(x_tensor, aux_tensor)
            return y_pred.cpu().numpy().squeeze()


class MultiHorizonTransformer(TimeSeriesTransformer):
    """Transformer model for multi-horizon time series forecasting."""
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        horizon: int = 1,
        **kwargs
    ):
        """Initialize the multi-horizon transformer.
        
        Args:
            input_dim: Number of input features per time step
            output_dim: Number of output dimensions per horizon
            horizon: Number of time steps to predict ahead
            **kwargs: Additional arguments for TimeSeriesTransformer
        """
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim * horizon,  # Output all predictions at once
            **kwargs
        )
        self.horizon = horizon
        self.output_dim = output_dim
    
    def forward(self, x: Tensor, aux: Optional[Tensor] = None) -> Tensor:
        """Forward pass for multi-horizon prediction."""
        # Get predictions for all horizons
        output = super().forward(x, aux)  # (batch_size, output_dim * horizon)
        
        # Reshape to (batch_size, horizon, output_dim)
        return output.reshape(-1, self.horizon, self.output_dim)
    
    def predict_horizon(self, x: np.ndarray, horizon: int, aux: Optional[np.ndarray] = None) -> np.ndarray:
        """Make predictions for a specific horizon."""
        if horizon < 1 or horizon > self.horizon:
            raise ValueError(f"Horizon must be between 1 and {self.horizon}")
            
        all_preds = self.predict(x, aux)  # (batch_size, horizon, output_dim)
        return all_preds[:, horizon-1]  # Return predictions for the specified horizon
