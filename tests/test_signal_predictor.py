import pandas as pd
from ml.signal_predictor import SignalPredictor

def test_predictor_empty_model(tmp_path, monkeypatch):
    # When MODEL_PATH is empty, predictor.model is None
    predictor = SignalPredictor(model_path="")
    df = pd.DataFrame({'symbol': ['EURUSD'], 'feature1': [1.0]})
    result = predictor.predict(df)
    assert result.empty
    assert list(result.columns) == ['symbol', 'signal', 'confidence']
