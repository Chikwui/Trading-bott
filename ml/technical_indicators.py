import numpy as np
import pandas as pd

class TechnicalIndicators:
    def __init__(self):
        pass

    def rsi(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        """
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def macd(self, df: pd.DataFrame, fast_window: int = 12, slow_window: int = 26, signal_window: int = 9):
        """
        Calculate MACD and Signal Line.
        Returns tuple of (macd, signal, hist)
        """
        ema_fast = df['close'].ewm(span=fast_window, min_periods=fast_window).mean()
        ema_slow = df['close'].ewm(span=slow_window, min_periods=slow_window).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=signal_window, min_periods=signal_window).mean()
        hist = macd - signal
        return macd, signal, hist

    def bollinger_bands(self, df: pd.DataFrame, window: int = 20, std_dev: float = 2.0):
        """
        Calculate Bollinger Bands.
        Returns tuple of (upper_band, middle_band, lower_band)
        """
        rolling_mean = df['close'].rolling(window=window).mean()
        rolling_std = df['close'].rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * std_dev)
        lower_band = rolling_mean - (rolling_std * std_dev)
        return upper_band, rolling_mean, lower_band

    def atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        """
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        tr = pd.DataFrame({
            'high_low': high_low,
            'high_close': high_close,
            'low_close': low_close
        }).max(axis=1)
        
        return tr.rolling(window=window).mean()

    def adx(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Calculate Average Directional Index (ADX).
        """
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate True Range
        tr = pd.DataFrame({
            'high_low': high - low,
            'high_close': np.abs(high - close.shift()),
            'low_close': np.abs(low - close.shift())
        }).max(axis=1)
        
        # Calculate +DM and -DM
        plus_dm = (high - high.shift()).where(
            (high - high.shift()) > (low.shift() - low), 0
        )
        minus_dm = (low.shift() - low).where(
            (low.shift() - low) > (high - high.shift()), 0
        )
        
        # Calculate +DI and -DI
        plus_di = 100 * (plus_dm.ewm(alpha=1/window).mean() / tr.ewm(alpha=1/window).mean())
        minus_di = 100 * (minus_dm.ewm(alpha=1/window).mean() / tr.ewm(alpha=1/window).mean())
        
        # Calculate DX
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        
        # Calculate ADX
        adx = dx.ewm(alpha=1/window).mean()
        return adx

    def cci(self, df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        Calculate Commodity Channel Index (CCI).
        """
        tp = (df['high'] + df['low'] + df['close']) / 3
        ma = tp.rolling(window=window).mean()
        md = tp.rolling(window=window).apply(lambda x: np.fabs(x - x.mean()).mean())
        return (tp - ma) / (0.015 * md)
