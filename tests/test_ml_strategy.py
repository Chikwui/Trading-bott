import pandas as pd
from strategies.ml_strategy import MLStrategy


def test_ml_strategy_empty_input():
    strategy = MLStrategy()
    df = pd.DataFrame()
    signals = strategy.generate_signals(df)
    assert signals.empty
    assert list(signals.columns) == ['symbol', 'signal', 'confidence']
