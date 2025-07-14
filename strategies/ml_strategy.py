from typing import Any
import pandas as pd
from strategies.base_strategy import BaseStrategy
from ml.signal_predictor import SignalPredictor

class MLStrategy(BaseStrategy):
    def __init__(self, config: Any = None):
        super().__init__(config)
        self.name = "MLStrategy"
        self.predictor = SignalPredictor()

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trade signals using ML model predictions."""
        if data.empty:
            return pd.DataFrame(columns=['symbol', 'signal', 'confidence'])
        signals = self.predictor.predict(data)
        return signals
