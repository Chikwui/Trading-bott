import pandas as pd


def calculate_sma(data: pd.Series, window: int) -> pd.Series:
    """Calculate simple moving average."""
    return data.rolling(window=window).mean()

# TODO: Add more technical indicators (EMA, ATR, RSI, etc.)
