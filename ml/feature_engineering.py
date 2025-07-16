import pandas as pd
import numpy as np
from typing import List, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from ml.technical_indicators import TechnicalIndicators

class FeatureEngineer:
    def __init__(self, features: List[str] = None):
        """
        Initialize feature engineer with optional list of features to extract.
        If None, will use default features.
        """
        self.features = features or [
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'bb_upper', 'bb_middle', 'bb_lower',
            'atr', 'adx', 'cci',
            'ema_10', 'ema_20', 'ema_50',
            'volume_ratio'
        ]
        self.scalers = {}
        self.indicators = TechnicalIndicators()

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators and features to the dataframe.
        First ensures volume column exists by mapping from tick_volume or real_volume.
        """
        # Map volume column if it doesn't exist
        if 'volume' not in df.columns:
            if 'tick_volume' in df.columns:
                df['volume'] = df['tick_volume']
            elif 'real_volume' in df.columns:
                df['volume'] = df['real_volume']
            else:
                raise ValueError("No volume column found in data")

        # Add RSI
        df['rsi'] = self.indicators.rsi(df)
        
        # Add MACD
        macd, macd_signal, macd_hist = self.indicators.macd(df)
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        
        # Add Bollinger Bands
        upper, middle, lower = self.indicators.bollinger_bands(df)
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        
        # Add ATR
        df['atr'] = self.indicators.atr(df)
        
        # Add ADX
        df['adx'] = self.indicators.adx(df)
        
        # Add CCI
        df['cci'] = self.indicators.cci(df)
        
        # Add EMAs
        df['ema_10'] = df['close'].ewm(span=10, min_periods=10).mean()
        df['ema_20'] = df['close'].ewm(span=20, min_periods=20).mean()
        df['ema_50'] = df['close'].ewm(span=50, min_periods=50).mean()
        
        # Add volume features
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        # Drop NaN values created by technical indicators
        df = df.dropna()
        
        return df

    def scale_features(self, df: pd.DataFrame, training: bool = True) -> pd.DataFrame:
        """
        Scale features using StandardScaler.
        If training=True, fit and transform. If training=False, transform only.
        """
        scaled_df = df.copy()
        
        for feature in self.features:
            if feature not in self.scalers:
                self.scalers[feature] = StandardScaler()
            
            if training:
                scaled_df[feature] = self.scalers[feature].fit_transform(
                    df[[feature]]
                ).reshape(-1)
            else:
                scaled_df[feature] = self.scalers[feature].transform(
                    df[[feature]]
                ).reshape(-1)
        
        return scaled_df

    def create_lagged_features(self, df: pd.DataFrame, n_lags: int = 5) -> pd.DataFrame:
        """
        Create lagged features for time series prediction.
        """
        lagged_df = df.copy()
        
        for feature in self.features:
            for lag in range(1, n_lags + 1):
                lagged_df[f'{feature}_lag_{lag}'] = lagged_df[feature].shift(lag)
        
        # Drop NaN values created by lagging
        lagged_df = lagged_df.dropna()
        
        return lagged_df

    def prepare_data(self, df: pd.DataFrame, training: bool = True, n_lags: int = 5) -> Dict:
        """
        Prepare data for ML model training/prediction.
        Returns dict with X (features), y (labels), and scaled data.
        """
        # Add features
        df = self.add_features(df)
        print(f"\n=== After add_features ===")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Scale features
        df = self.scale_features(df, training)
        print(f"\n=== After scale_features ===")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Create lagged features
        df = self.create_lagged_features(df, n_lags)
        print(f"\n=== After create_lagged_features ===")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Prepare features and labels
        X = df[self.features]
        y = df['signal'] if 'signal' in df.columns else None
        
        # Verify shapes
        print(f"\n=== Data Shape Verification ===")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape if y is not None else 'None'}")
        
        # Ensure consistent NaN handling
        if y is not None:
            # Drop rows where either X or y has NaN values
            mask = X.notna().all(axis=1) & y.notna()
            X = X[mask]
            y = y[mask]
            
            # Final shape verification
            print(f"\n=== After NaN Handling ===")
            print(f"X shape: {X.shape}")
            print(f"y shape: {y.shape}")
            
            # Verify shapes match
            if X.shape[0] != len(y):
                raise ValueError(f"Shape mismatch: X has {X.shape[0]} samples but y has {len(y)} samples")
        
        return {
            'X': X,
            'y': y,
            'scaled_data': df
        }
