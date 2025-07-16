import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import os
from config.settings import settings
from ml.technical_indicators import TechnicalIndicators

def download_historical_data(symbol: str, timeframe: int, days: int = 365) -> pd.DataFrame:
    """
    Download historical market data using MT5.
    
    Args:
        symbol: Trading symbol (e.g., 'EURUSD')
        timeframe: MT5 timeframe constant (e.g., mt5.TIMEFRAME_H1)
        days: Number of days to download
    
    Returns:
        DataFrame with OHLCV data
    """
    # Initialize MT5
    if not mt5.initialize(
        login=settings.MT5_LOGIN,
        password=settings.MT5_PASSWORD,
        server=settings.MT5_SERVER
    ):
        raise Exception(f"MT5 initialization failed: {mt5.last_error()}")

    try:
        # Calculate time range
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        # Download data
        rates = mt5.copy_rates_range(symbol, timeframe, start_time, end_time)
        
        if rates is None:
            raise Exception(f"Failed to download data for {symbol}")

        # Convert to DataFrame
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        
        # Add return column
        df['return'] = df['close'].pct_change()
        
        return df

    finally:
        mt5.shutdown()

def create_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create trading signals based on simple technical analysis.
    """
    indicators = TechnicalIndicators()
    
    # Add RSI
    df['rsi'] = indicators.rsi(df)
    
    # Add MACD
    macd, macd_signal, _ = indicators.macd(df)
    
    # Generate signals
    df['signal'] = 0
    
    # RSI-based signals
    df.loc[df['rsi'] > 70, 'signal'] = -1  # Sell signal
    df.loc[df['rsi'] < 30, 'signal'] = 1   # Buy signal
    
    # MACD-based signals
    df.loc[macd > macd_signal, 'signal'] = 1
    df.loc[macd < macd_signal, 'signal'] = -1
    
    # Forward fill signals
    df['signal'] = df['signal'].fillna(0)
    df['signal'] = df['signal'].astype(int)
    
    return df

def main():
    """
    Download historical data and prepare it for model training.
    """
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Download data for major pairs
    symbols = ['EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD']
    timeframes = [mt5.TIMEFRAME_H1]  # 1-hour timeframe
    
    for symbol in symbols:
        for timeframe in timeframes:
            print(f"\nDownloading {symbol} data...")
            try:
                # Download data
                df = download_historical_data(symbol, timeframe)
                
                # Create signals
                df = create_signals(df)
                
                # Save to CSV
                filename = f'data/{symbol}_H1.csv'
                df.to_csv(filename, index=False)
                print(f"Data saved to {filename}")
                
            except Exception as e:
                print(f"Error downloading {symbol}: {str(e)}")

if __name__ == '__main__':
    main()
