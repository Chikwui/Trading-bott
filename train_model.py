from ml.model_trainer import ModelTrainer
import os
from config.settings import settings
import pandas as pd

def main():
    """
    Train ML model using historical market data.
    """
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Process each symbol's data
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD']
    
    for symbol in symbols:
        print(f"\n=== Training model for {symbol} ===")
        
        try:
            # Load data
            data_path = f'data/{symbol}_H1.csv'
            if not os.path.exists(data_path):
                print(f"Data file not found for {symbol}")
                continue
                
            # Train and save model
            trainer.train_and_save(data_path)
            
        except Exception as e:
            print(f"Error training model for {symbol}: {str(e)}")

if __name__ == '__main__':
    main()
