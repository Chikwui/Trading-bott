import pandas as pd
import numpy as np
from utils.indicators import calculate_sma

def test_calculate_sma():
    s = pd.Series([1, 2, 3, 4, 5])
    sma = calculate_sma(s, window=2)
    assert np.isnan(sma.iloc[0])
    assert sma.iloc[1] == 1.5
    assert sma.iloc[2] == 2.5
