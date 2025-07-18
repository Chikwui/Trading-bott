"""
Test suite for the transformer-based time series prediction model.
"""
import os
import tempfile
import numpy as np
import pytest
import torch
from torch import nn

from core.ml.models.transformer import (
    PositionalEncoding,
    TransformerModel,
    TimeSeriesTransformer,
    MultiHorizonTransformer
)

# Test data parameters
BATCH_SIZE = 32
SEQ_LEN = 60
INPUT_DIM = 10
OUTPUT_DIM = 1
HORIZON = 5
AUX_DIM = 3

@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Generate random time series data
    x = np.random.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM).astype(np.float32)
    y = np.random.randn(BATCH_SIZE, OUTPUT_DIM).astype(np.float32)
    aux = np.random.randn(BATCH_SIZE, AUX_DIM).astype(np.float32)
    
    return x, y, aux

def test_positional_encoding():
    """Test positional encoding."""
    d_model = 64
    max_len = 100
    pe = PositionalEncoding(d_model, max_len)
    
    # Test output shape
    x = torch.zeros(10, 50, d_model)
    output = pe(x)
    assert output.shape == (10, 50, d_model)
    
    # Test that different positions have different encodings
    assert not torch.allclose(output[0, 0], output[0, 1])
    
    # Test that the same position has the same encoding across batches
    assert torch.allclose(output[0, 0], output[1, 0])

@pytest.mark.parametrize("batch_first", [True, False])
def test_transformer_model_forward(batch_first):
    """Test forward pass of the transformer model."""
    model = TransformerModel(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1,
        batch_first=batch_first
    )
    
    # Test forward pass
    x = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM)
    if not batch_first:
        x = x.transpose(0, 1)  # (seq_len, batch_size, input_dim)
    
    output = model(x)
    assert output.shape == (BATCH_SIZE, OUTPUT_DIM)

def test_transformer_model_save_load(tmp_path):
    """Test saving and loading the transformer model."""
    # Create and save model
    model = TransformerModel(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.1
    )
    
    # Create dummy input
    x = torch.randn(BATCH_SIZE, SEQ_LEN, INPUT_DIM)
    
    # Get predictions before saving
    model.eval()
    with torch.no_grad():
        y_pred_before = model(x).numpy()
    
    # Save and load model
    model_path = os.path.join(tmp_path, "test_model.pt")
    model.save(model_path)
    loaded_model = TransformerModel.load(model_path)
    
    # Get predictions after loading
    loaded_model.eval()
    with torch.no_grad():
        y_pred_after = loaded_model(x).numpy()
    
    # Check that predictions match
    np.testing.assert_allclose(y_pred_before, y_pred_after, rtol=1e-6)

def test_time_series_transformer(sample_data):
    """Test the time series transformer with auxiliary features."""
    x, y, aux = sample_data
    
    model = TimeSeriesTransformer(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        seq_len=SEQ_LEN,
        d_model=64,
        nhead=4,
        num_layers=2,
        num_aux_features=AUX_DIM
    )
    
    # Test forward pass with auxiliary features
    output = model(torch.FloatTensor(x), torch.FloatTensor(aux))
    assert output.shape == (BATCH_SIZE, OUTPUT_DIM)
    
    # Test predict method
    y_pred = model.predict(x, aux)
    assert y_pred.shape == (BATCH_SIZE, OUTPUT_DIM)

def test_multi_horizon_transformer(sample_data):
    """Test the multi-horizon transformer."""
    x, y, aux = sample_data
    
    model = MultiHorizonTransformer(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        seq_len=SEQ_LEN,
        horizon=HORIZON,
        d_model=64,
        nhead=4,
        num_layers=2,
        num_aux_features=AUX_DIM
    )
    
    # Test forward pass
    output = model(torch.FloatTensor(x), torch.FloatTensor(aux))
    assert output.shape == (BATCH_SIZE, HORIZON, OUTPUT_DIM)
    
    # Test predict method
    y_pred = model.predict(x, aux)
    assert y_pred.shape == (BATCH_SIZE, HORIZON, OUTPUT_DIM)
    
    # Test predict_horizon method
    for h in range(1, HORIZON + 1):
        y_pred_h = model.predict_horizon(x, h, aux)
        assert y_pred_h.shape == (BATCH_SIZE, OUTPUT_DIM)
        np.testing.assert_array_equal(y_pred_h, y_pred[:, h-1])

def test_multi_horizon_transformer_training(sample_data):
    """Test training loop for multi-horizon transformer."""
    x, y, aux = sample_data
    
    model = MultiHorizonTransformer(
        input_dim=INPUT_DIM,
        output_dim=OUTPUT_DIM,
        seq_len=SEQ_LEN,
        horizon=HORIZON,
        d_model=64,
        nhead=4,
        num_layers=2,
        num_aux_features=AUX_DIM
    )
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Convert data to tensors
    x_tensor = torch.FloatTensor(x)
    aux_tensor = torch.FloatTensor(aux)
    y_tensor = torch.FloatTensor(y).unsqueeze(1).repeat(1, HORIZON, 1)  # (batch_size, horizon, output_dim)
    
    # Training loop
    model.train()
    losses = []
    
    for _ in range(10):  # Short training for testing
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(x_tensor, aux_tensor)
        
        # Compute loss (average across horizon)
        loss = criterion(outputs, y_tensor)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    
    # Check that loss is decreasing
    assert losses[-1] < losses[0], "Loss should decrease during training"

if __name__ == "__main__":
    pytest.main(["-v", "test_transformer.py"])
