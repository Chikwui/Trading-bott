import os
import joblib
import pandas as pd
from typing import Dict
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
from ml.feature_engineering import FeatureEngineer
from config.settings import settings
import numpy as np

class ModelTrainer:
    def __init__(self, model_path: str = settings.MODEL_PATH):
        self.model_path = model_path
        self.feature_engineer = FeatureEngineer()
        self.model = None

    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load and preprocess historical market data.
        """
        df = pd.read_csv(data_path)
        print(f"\n=== Data Loading Debug ===")
        print(f"Columns after loading: {df.columns.tolist()}")
        print(f"First few rows:\n{df.head()}")
        
        # Add basic preprocessing if needed
        df = df.dropna()
        print(f"\n=== After dropna ===")
        print(f"Columns after dropna: {df.columns.tolist()}")
        print(f"First few rows after dropna:\n{df.head()}")
        
        return df

    def prepare_features(self, df: pd.DataFrame) -> Dict:
        """
        Prepare features and labels for training.
        """
        print(f"\n=== Feature Preparation Debug ===")
        print(f"Columns before feature engineering: {df.columns.tolist()}")
        print(f"First few rows before feature engineering:\n{df.head()}")
        
        result = self.feature_engineer.prepare_data(df, training=True)
        print(f"\n=== After Feature Engineering ===")
        print(f"Features: {result['X'].columns.tolist()}")
        
        return result

    def train_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train a robust ML model with hyperparameter tuning.
        """
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Initialize model
        rf = RandomForestClassifier(
            n_jobs=-1,
            random_state=42,
            class_weight='balanced'
        )

        # Define hyperparameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }

        # Grid Search with cross-validation
        grid_search = GridSearchCV(
            rf,
            param_grid,
            cv=5,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=2
        )

        # Train model
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_

        # Evaluate on validation set
        y_pred = self.model.predict(X_val)
        
        print("\n=== Model Evaluation ===")
        print("Best Parameters:", grid_search.best_params_)
        print("\nClassification Report:")
        print(classification_report(y_val, y_pred))
        print("\nValidation Accuracy:", accuracy_score(y_val, y_pred))
        print("Validation F1 Score:", f1_score(y_val, y_pred, average='macro'))

    def save_model(self) -> None:
        """
        Save trained model to disk.
        """
        if self.model is not None:
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            joblib.dump(self.model, self.model_path)
            print(f"\nModel saved to {self.model_path}")
        else:
            raise ValueError("Model not trained yet")

    def backtest(self, df: pd.DataFrame) -> Dict:
        """
        Perform backtesting on historical data.
        Returns performance metrics.
        """
        # Prepare test data
        test_data = self.feature_engineer.prepare_data(df, training=False)
        X_test = test_data['X']
        y_test = test_data['y']

        # Predict
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred, average='macro'),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }

        # Calculate trading metrics
        positions = np.sign(y_pred)
        returns = positions * df['return']  # Assuming 'return' column exists
        
        trading_metrics = {
            'total_trades': len(positions[positions != 0]),
            'win_rate': (positions[positions != 0] * y_test > 0).mean(),
            'average_return': returns.mean(),
            'sharpe_ratio': returns.mean() / returns.std()
        }

        return {**metrics, **trading_metrics}

    def train_and_save(self, data_path: str) -> None:
        """
        Complete training pipeline: load data, prepare features, train model, save model.
        """
        print("\n=== Starting Model Training ===")
        
        # Load and prepare data
        df = self.load_data(data_path)
        data = self.prepare_features(df)
        
        # Train model
        self.train_model(data['X'], data['y'])
        
        # Save model
        self.save_model()
        
        # Backtest
        metrics = self.backtest(df)
        print("\n=== Backtest Results ===")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']:.2%}")
        print(f"Average Return: {metrics['average_return']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
