"""
Technical Analysis Example with MT5 Data Provider

This script demonstrates how to use the MT5 data provider to fetch market data
and apply various technical indicators for analysis.
"""
import asyncio
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add parent directory to path to allow imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data.providers.mt5_provider import MT5DataProvider
from core.data.provider_config import DataProviderConfig
from core.indicators.ta_indicators import (
    moving_average, rsi, macd, bollinger_bands, atr, stochastic_oscillator,
    MovingAverageType, IndicatorType
)

# Configuration
SYMBOL = "EURUSD"
TIMEFRAME = "H1"  # 1-hour candles
START_DATE = datetime.now() - timedelta(days=30)  # Last 30 days
END_DATE = datetime.now()
INDICATOR_WINDOW = 14  # Default window for indicators

# Set up matplotlib for better looking charts
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = (14, 8)

class TechnicalAnalysisExample:
    def __init__(self):
        """Initialize the MT5 data provider."""
        self.mt5_config = DataProviderConfig(
            name="MT5",
            provider_type="mt5",
            enabled=True,
            extra={
                'login': os.getenv('MT5_LOGIN'),
                'password': os.getenv('MT5_PASSWORD'),
                'server': os.getenv('MT5_SERVER'),
                'path': os.getenv('MT5_PATH'),
                'timeout': 60000
            }
        )
        self.mt5 = MT5DataProvider(self.mt5_config)
        self.data = None
        self.indicators = {}
    
    async def initialize(self):
        """Initialize the data provider and load historical data."""
        print("Initializing MT5 data provider...")
        await self.mt5.initialize()
        await self.mt5.connect()
        
        print(f"Fetching {SYMBOL} {TIMEFRAME} data from {START_DATE} to {END_DATE}...")
        self.data = await self.mt5.get_ohlcv(
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            start_time=START_DATE,
            end_time=END_DATE
        )
        
        if not self.data:
            raise ValueError("No data returned from MT5")
        
        print(f"Fetched {len(self.data)} candles")
        
        # Convert to DataFrame for easier manipulation
        self.df = pd.DataFrame([{
            'timestamp': candle.timestamp,
            'open': candle.open,
            'high': candle.high,
            'low': candle.low,
            'close': candle.close,
            'volume': candle.volume
        } for candle in self.data])
        
        # Set timestamp as index
        self.df.set_index('timestamp', inplace=True)
    
    def calculate_indicators(self):
        """Calculate various technical indicators."""
        print("Calculating technical indicators...")
        
        # 1. Moving Averages
        self.indicators['sma_20'] = moving_average(
            self.df['close'], 
            window=20,
            ma_type=MovingAverageType.SMA
        )
        
        self.indicators['ema_50'] = moving_average(
            self.df['close'],
            window=50,
            ma_type=MovingAverageType.EMA
        )
        
        # 2. RSI
        self.indicators['rsi_14'] = rsi(
            self.df['close'],
            window=14
        )
        
        # 3. MACD
        macd_result = macd(
            self.df['close'],
            fast_period=12,
            slow_period=26,
            signal_period=9
        )
        self.indicators.update(macd_result)
        
        # 4. Bollinger Bands
        bb_result = bollinger_bands(
            self.df['close'],
            window=20,
            std_dev=2.0
        )
        self.indicators.update(bb_result)
        
        # 5. ATR
        self.indicators['atr_14'] = atr(
            self.df['high'],
            self.df['low'],
            self.df['close'],
            window=14
        )
        
        # 6. Stochastic Oscillator
        stoch_result = stochastic_oscillator(
            self.df['high'],
            self.df['low'],
            self.df['close'],
            k_window=14,
            d_window=3
        )
        self.indicators.update(stoch_result)
        
        print(f"Calculated {len(self.indicators)} indicators")
    
    def plot_charts(self):
        """Plot price chart with indicators."""
        print("Generating charts...")
        
        # Create figure with subplots
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 16), gridspec_kw={'height_ratios': [3, 1, 1, 1]})
        
        # 1. Price Chart with Moving Averages and Bollinger Bands
        self.df['close'].plot(ax=ax1, color='black', label='Close', alpha=0.7)
        
        # Plot moving averages
        if 'sma_20' in self.indicators:
            ax1.plot(self.df.index, self.indicators['sma_20'].values, 'b-', label='SMA 20', alpha=0.7)
        if 'ema_50' in self.indicators:
            ax1.plot(self.df.index, self.indicators['ema_50'].values, 'r-', label='EMA 50', alpha=0.7)
        
        # Plot Bollinger Bands if available
        if all(k in self.indicators for k in ['BB_Upper', 'BB_Middle', 'BB_Lower']):
            ax1.fill_between(
                self.df.index,
                self.indicators['BB_Upper'].values,
                self.indicators['BB_Lower'].values,
                color='gray',
                alpha=0.2,
                label='Bollinger Bands'
            )
        
        ax1.set_title(f'{SYMBOL} Price with Indicators')
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True)
        
        # 2. RSI
        if 'rsi_14' in self.indicators:
            ax2.plot(self.df.index, self.indicators['rsi_14'].values, 'g-', label='RSI 14')
            ax2.axhline(70, color='r', linestyle='--', alpha=0.3)
            ax2.axhline(30, color='g', linestyle='--', alpha=0.3)
            ax2.set_ylim(0, 100)
            ax2.set_ylabel('RSI')
            ax2.legend()
            ax2.grid(True)
        
        # 3. MACD
        if all(k in self.indicators for k in ['MACD', 'MACD_Signal']):
            ax3.plot(self.df.index, self.indicators['MACD'].values, 'b-', label='MACD')
            ax3.plot(self.df.index, self.indicators['MACD_Signal'].values, 'r-', label='Signal')
            
            # Plot histogram
            ax3.bar(
                self.df.index,
                self.indicators['MACD_Hist'].values,
                color=np.where(self.indicators['MACD_Hist'].values >= 0, 'g', 'r'),
                alpha=0.3,
                label='Histogram'
            )
            
            ax3.axhline(0, color='black', linestyle='-', alpha=0.3)
            ax3.set_ylabel('MACD')
            ax3.legend()
            ax3.grid(True)
        
        # 4. Stochastic Oscillator
        if all(k in self.indicators for k in ['Stoch_%K', 'Stoch_%D']):
            ax4.plot(self.df.index, self.indicators['Stoch_%K'].values, 'b-', label='%K')
            ax4.plot(self.df.index, self.indicators['Stoch_%D'].values, 'r-', label='%D')
            ax4.axhline(80, color='r', linestyle='--', alpha=0.3)
            ax4.axhline(20, color='g', linestyle='--', alpha=0.3)
            ax4.set_ylim(0, 100)
            ax4.set_ylabel('Stochastic')
            ax4.legend()
            ax4.grid(True)
        
        plt.tight_layout()
        
        # Save figure
        os.makedirs('charts', exist_ok=True)
        chart_path = f'charts/{SYMBOL}_{TIMEFRAME}_analysis.png'
        plt.savefig(chart_path)
        print(f"Chart saved to {chart_path}")
        
        # Show plot
        plt.show()
    
    async def run_analysis(self):
        """Run the technical analysis."""
        try:
            await self.initialize()
            self.calculate_indicators()
            self.plot_charts()
            
            # Print summary statistics
            print("\n=== Summary Statistics ===")
            print(f"Time Period: {self.df.index[0].date()} to {self.df.index[-1].date()}")
            print(f"Price Change: {((self.df['close'].iloc[-1] / self.df['close'].iloc[0]) - 1) * 100:.2f}%")
            print(f"Average Daily Range: {self.df['high'].sub(self.df['low']).mean():.5f}")
            
            if 'atr_14' in self.indicators:
                print(f"Average True Range (14): {self.indicators['atr_14'].values[-1]:.5f}")
            
            if 'rsi_14' in self.indicators:
                print(f"Current RSI (14): {self.indicators['rsi_14'].values[-1]:.2f}")
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            raise
        finally:
            await self.mt5.disconnect()

async def main():
    """Main function to run the example."""
    # Check if required environment variables are set
    required_vars = ['MT5_LOGIN', 'MT5_PASSWORD', 'MT5_SERVER']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("Error: The following environment variables must be set:")
        for var in missing_vars:
            print(f"- {var}")
        print("\nPlease create a .env file with these variables or set them in your environment.")
        return
    
    # Run the analysis
    example = TechnicalAnalysisExample()
    await example.run_analysis()

if __name__ == "__main__":
    # Load environment variables from .env file if it exists
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run the async main function
    asyncio.run(main())
