import joblib
import pandas as pd
from config.settings import MODEL_PATH

class SignalPredictor:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model = joblib.load(model_path) if model_path else None

    def predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Returns a DataFrame with columns ['symbol', 'signal', 'confidence'].
        Signal: 1 for BUY, -1 for SELL, 0 for HOLD.
        """
        if self.model is None:
            return pd.DataFrame(columns=['symbol', 'signal', 'confidence'])

        proba = self.model.predict_proba(features)
        # assuming binary classifier with classes [0, 1]
        confidence = proba.max(axis=1)
        signal_val = [1 if p[1] > 0.5 else -1 for p in proba]
        result = features.copy()
        result['signal'] = signal_val
        result['confidence'] = confidence
        return result[['symbol', 'signal', 'confidence']]
