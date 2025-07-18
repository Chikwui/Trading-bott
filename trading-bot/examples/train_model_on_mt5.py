"""
MT5 Model Training Example

This script demonstrates how to use the ML pipeline with MT5 data to train and evaluate
a trading model. It covers the complete workflow from data fetching to model evaluation.
"""
# Standard library imports
import os
import sys
import logging
import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

# Third-party imports
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import core components
from core.data.mt5_fetcher import MT5Fetcher, MT5Config
from core.ml.models import XGBoostModel, LightGBMModel, LSTMModel, TransformerModel
from core.ml.pipeline import ModelPipeline, ModelConfig
from core.ml.trainer import ModelTrainer, DataPreprocessor

# Configure logging
def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Configure logging
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_file = log_dir / f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    
    # Configure third-party loggers
    for logger_name in ['matplotlib', 'tensorflow']:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

# Initialize logger
logger = setup_logging()

def setup_directories() -> Dict[str, Path]:
    """Create necessary directories for the project."""
    base_dir = Path(__file__).parent.parent
    dirs = {
        'data': base_dir / 'data',
        'models': base_dir / 'models',
        'logs': base_dir / 'logs',
        'results': base_dir / 'results'
    }
    
    # Create directories if they don't exist
    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {path}")
    
    return dirs

def fetch_and_prepare_data(
    symbol: str,
    timeframe: str,
    from_date: str,
    to_date: str,
    lookback: int = 50,
    horizon: int = 5,
    features: Optional[List[str]] = None,
    target: str = 'returns',
    save_processed: bool = True
) -> Tuple[pd.DataFrame, pd.Series, DataPreprocessor]:
    """
    Fetch and preprocess data for model training.
    
    Args:
        symbol: Trading symbol (e.g., 'EURUSD')
        timeframe: Timeframe (e.g., '1h', '4h', '1d')
        from_date: Start date in 'YYYY-MM-DD' format
        to_date: End date in 'YYYY-MM-DD' format
        lookback: Number of lookback periods for feature engineering
        horizon: Number of periods ahead to predict
        features: List of feature columns to include
        target: Target variable for prediction
        save_processed: Whether to save processed data
        
    Returns:
        Tuple of (X, y, preprocessor)
    """
    # Initialize MT5 fetcher
    config = MT5Config()
    fetcher = MT5Fetcher(config)
    
    # Default features if not specified
    if features is None:
        features = [
            'open', 'high', 'low', 'close', 'tick_volume', 'spread',
            'returns', 'volatility', 'rsi', 'macd', 'macd_signal',
            'bb_upper', 'bb_middle', 'bb_lower', 'atr', 'vwap'
        ]
    
    # Fetch raw data
    logger.info(f"Fetching {symbol} {timeframe} data from {from_date} to {to_date}")
    df = fetcher.fetch_rates(
        symbol=symbol,
        timeframe=timeframe,
        from_date=from_date,
        to_date=to_date,
        save_raw=True
    )
    
    if df is None or df.empty:
        raise ValueError(f"Failed to fetch data for {symbol} {timeframe}")
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(
        feature_columns=features,
        target_column=target,
        lookback=lookback,
        horizon=horizon,
        scale_features=True,
        scale_target=True,
        scaler_type='standard'
    )
    
    # Preprocess data
    logger.info("Preprocessing data...")
    X, y = preprocessor.preprocess(df)
    
    # Save processed data if requested
    if save_processed:
        output_dir = Path('data/processed')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save features and target
        X.to_parquet(output_dir / f"{symbol}_{timeframe}_X.parquet")
        y.to_parquet(output_dir / f"{symbol}_{timeframe}_y.parquet")
        logger.info(f"Saved processed data to {output_dir}")
    
    return X, y, preprocessor

def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = 'xgboost',
    task_type: str = 'regression',
    test_size: float = 0.2,
    random_state: int = 42,
    output_dir: str = 'models'
) -> ModelTrainer:
    """
    Train a model on the given data.
    
    Args:
        X: Feature matrix
        y: Target values
        model_type: Type of model to train ('xgboost', 'lightgbm', 'lstm', 'transformer')
        task_type: Type of task ('regression' or 'classification')
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        output_dir: Directory to save the trained model
        
    Returns:
        Trained ModelTrainer instance
    """
    # Map model type to model class
    model_classes = {
        'xgboost': XGBoostModel,
        'lightgbm': LightGBMModel,
        'lstm': LSTMModel,
        'transformer': TransformerModel
    }
    
    if model_type not in model_classes:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Model parameters
    model_params = {
        'task_type': task_type,
        'random_state': random_state
    }
    
    # Special parameters for deep learning models
    if model_type == 'lstm':
        model_params.update({
            'input_shape': (X.shape[1], 1),  # (timesteps, features)
            'units': [64, 32],
            'dropout_rate': 0.2,
            'learning_rate': 0.001,
            'epochs': 100,
            'batch_size': 32
        })
    elif model_type == 'transformer':
        model_params.update({
            'input_shape': (X.shape[1], 1),  # (timesteps, features)
            'num_heads': 4,
            'ff_dim': 128,
            'num_transformer_blocks': 2,
            'mlp_units': [128, 64],
            'dropout_rate': 0.1,
            'learning_rate': 0.001,
            'epochs': 100,
            'batch_size': 32
        })
    
    # Initialize model trainer
    trainer = ModelTrainer(
        model_class=model_classes[model_type],
        model_params=model_params,
        preprocessor=None,  # Already preprocessed
        task_type=task_type,
        output_dir=output_dir,
        experiment_name=f"{model_type}_{task_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False, random_state=random_state
    )
    
    # Train model
    logger.info(f"Training {model_type} model for {task_type}...")
    trainer.train(X_train, y_train, X_test, y_test)
    
    # Evaluate model
    logger.info("Evaluating model...")
    metrics = trainer.evaluate(X_test, y_test, set_name='test')
    logger.info(f"Test metrics: {metrics}")
    
    # Save model
    model_path = trainer.save_model()
    logger.info(f"Model saved to {model_path}")
    
    return trainer

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments.
    
    Returns:
        Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description='Train a trading model using MT5 data')
    
    # Data configuration
    parser.add_argument('--symbol', type=str, default='EURUSD',
                       help='Trading symbol (e.g., EURUSD, GBPUSD)')
    parser.add_argument('--timeframe', type=str, default='1h',
                       choices=['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M'],
                       help='Timeframe for the data')
    parser.add_argument('--from-date', type=str, default='2023-01-01',
                       help='Start date for training data (YYYY-MM-DD)')
    parser.add_argument('--to-date', type=str, default='2023-12-31',
                       help='End date for training data (YYYY-MM-DD)')
    
    # Model configuration
    parser.add_argument('--model-type', type=str, default='xgboost',
                       choices=['xgboost', 'lightgbm', 'lstm', 'transformer'],
                       help='Type of model to train')
    parser.add_argument('--task-type', type=str, default='regression',
                       choices=['regression', 'classification'],
                       help='Type of prediction task')
    parser.add_argument('--lookback', type=int, default=50,
                       help='Number of lookback periods for features')
    parser.add_argument('--horizon', type=int, default=5,
                       help='Number of periods ahead to predict')
    
    # Training configuration
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Fraction of data to use for testing')
    parser.add_argument('--random-state', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Logging level')
    
    # MT5 configuration
    parser.add_argument('--mt5-path', type=str, 
                       default='C:\\Program Files\\MetaTrader 5\\terminal64.exe',
                       help='Path to MT5 terminal executable')
    
    return parser.parse_args()

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    default_config = {
        'symbol': 'EURUSD',
        'timeframe': '1h',
        'from_date': '2023-01-01',
        'to_date': '2023-12-31',
        'lookback': 50,
        'horizon': 5,
        'model_type': 'xgboost',
        'task_type': 'regression',
        'test_size': 0.2,
        'random_state': 42,
        'mt5_path': 'C:\\Program Files\\MetaTrader 5\\terminal64.exe'
    }
    
    if config_path and Path(config_path).exists():
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
                logger.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}")
    
    return default_config

def main() -> None:
    """Main function to run the training pipeline."""
    try:
        # Parse command line arguments
        args = parse_arguments()
        
        # Update logger with command line log level
        global logger
        logger = setup_logging(log_level=args.log_level)
        
        # Load configuration
        config = load_config()
        
        # Override config with command line arguments
        for key, value in vars(args).items():
            if value is not None and key.replace('-', '_') in config:
                config[key.replace('-', '_')] = value
        
        # Setup directories
        dirs = setup_directories()
        
        logger.info("Starting training pipeline with configuration:")
        for key, value in config.items():
            logger.info(f"  {key}: {value}")
        
        # Initialize MT5 connection
        logger.info("Initializing MT5 connection...")
        mt5_config = MT5Config(
            path=config['mt5_path'],
            server="",  # Add your server if needed
            login=0,    # Add your login if needed
            password="" # Add your password if needed
        )
        
        with MT5Fetcher(mt5_config) as mt5_fetcher:
            if not mt5_fetcher.connected:
                raise ConnectionError("Failed to connect to MT5 terminal")
            
            logger.info("Successfully connected to MT5 terminal")
            
            # Fetch and prepare data
            logger.info("Fetching and preparing data...")
            X, y, preprocessor = fetch_and_prepare_data(
                symbol=config['symbol'],
                timeframe=config['timeframe'],
                from_date=config['from_date'],
                to_date=config['to_date'],
                lookback=config['lookback'],
                horizon=config['horizon']
            )
            
            if X is None or y is None or len(X) == 0 or len(y) == 0:
                raise ValueError("No data available for training")
            
            logger.info(f"Prepared data: {X.shape[0]} samples with {X.shape[1]} features")
            
            # Train model
            logger.info(f"Training {config['model_type']} model...")
            model = train_model(
                X=X,
                y=y,
                model_type=config['model_type'],
                task_type=config['task_type'],
                test_size=config['test_size'],
                random_state=config['random_state'],
                output_dir=str(dirs['models'])
            )
            
            logger.info("Training completed successfully!")
            
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception("An error occurred during training:")
        sys.exit(1)
    finally:
        # Ensure MT5 connection is properly closed
        if 'mt5_fetcher' in locals() and mt5_fetcher.connected:
            mt5_fetcher.disconnect()
            logger.info("Disconnected from MT5 terminal")

if __name__ == "__main__":
    main()
