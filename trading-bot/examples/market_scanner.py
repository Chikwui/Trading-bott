"""
Real-time Market Scanner

This script demonstrates a real-time market scanner that identifies potential
trading opportunities across multiple currency pairs and timeframes using
technical indicators.
"""
import asyncio
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

# Add parent directory to path to allow imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data.providers.mt5_provider import MT5DataProvider
from core.data.provider_config import DataProviderConfig
from core.indicators.ta_indicators import (
    moving_average, rsi, macd, bollinger_bands, atr, stochastic_oscillator,
    MovingAverageType, IndicatorType
)
from core.indicators.advanced_indicators import (
    ichimoku_cloud, parabolic_sar, adx, volume_profile, VolumeProfileLevels
)

# Configuration
SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD", "BTCUSD"]
TIMEFRAMES = ["M5", "M15", "H1", "H4"]
SCAN_INTERVAL = 300  # seconds between scans

# Indicator parameters
INDICATOR_PARAMS = {
    'rsi': {'window': 14, 'overbought': 70, 'oversold': 30},
    'macd': {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
    'bollinger': {'window': 20, 'std_dev': 2.0},
    'stochastic': {'k_window': 14, 'd_window': 3, 'overbought': 80, 'oversold': 20},
    'adx': {'window': 14, 'trend_threshold': 25},
    'ichimoku': {'tenkan_period': 9, 'kijun_period': 26, 'senkou_span_b_period': 52}
}

# Set up logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('market_scanner.log')
    ]
)
logger = logging.getLogger(__name__)

class MarketScanner:
    def __init__(self):
        """Initialize the market scanner with MT5 data provider."""
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
        self.scan_results = {}
    
    async def initialize(self):
        """Initialize the data provider."""
        logger.info("Initializing MT5 data provider...")
        await self.mt5.initialize()
        await self.mt5.connect()
        logger.info("MT5 data provider initialized successfully")
    
    async def scan_market(self):
        """Scan the market for trading opportunities."""
        logger.info("Starting market scan...")
        
        for symbol in SYMBOLS:
            for timeframe in TIMEFRAMES:
                try:
                    logger.info(f"Analyzing {symbol} {timeframe}...")
                    
                    # Fetch OHLCV data
                    ohlcv = await self.mt5.get_ohlcv(
                        symbol=symbol,
                        timeframe=timeframe,
                        limit=200  # Get enough data for all indicators
                    )
                    
                    if not ohlcv:
                        logger.warning(f"No data returned for {symbol} {timeframe}")
                        continue
                    
                    # Convert to DataFrame
                    df = self._process_ohlcv(ohlcv)
                    
                    # Calculate indicators
                    indicators = self._calculate_indicators(df)
                    
                    # Generate signals
                    signals = self._generate_signals(df, indicators)
                    
                    # Store results
                    self.scan_results[f"{symbol}_{timeframe}"] = {
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'indicators': indicators,
                        'signals': signals,
                        'timestamp': datetime.now()
                    }
                    
                except Exception as e:
                    logger.error(f"Error analyzing {symbol} {timeframe}: {e}")
        
        logger.info("Market scan completed")
    
    def _process_ohlcv(self, ohlcv) -> pd.DataFrame:
        """Convert OHLCV data to DataFrame."""
        data = []
        for candle in ohlcv:
            data.append({
                'timestamp': candle.timestamp,
                'open': candle.open,
                'high': candle.high,
                'low': candle.low,
                'close': candle.close,
                'volume': candle.volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
    
    def _calculate_indicators(self, df: pd.DataFrame) -> dict:
        """Calculate technical indicators."""
        indicators = {}
        
        # Basic indicators
        indicators['sma_20'] = moving_average(
            df['close'], 
            window=20,
            ma_type=MovingAverageType.SMA
        )
        
        indicators['sma_50'] = moving_average(
            df['close'],
            window=50,
            ma_type=MovingAverageType.SMA
        )
        
        indicators['rsi'] = rsi(
            df['close'],
            window=INDICATOR_PARAMS['rsi']['window']
        )
        
        # MACD
        macd_result = macd(
            df['close'],
            fast_period=INDICATOR_PARAMS['macd']['fast_period'],
            slow_period=INDICATOR_PARAMS['macd']['slow_period'],
            signal_period=INDICATOR_PARAMS['macd']['signal_period']
        )
        indicators.update(macd_result)
        
        # Bollinger Bands
        bb_result = bollinger_bands(
            df['close'],
            window=INDICATOR_PARAMS['bollinger']['window'],
            std_dev=INDICATOR_PARAMS['bollinger']['std_dev']
        )
        indicators.update(bb_result)
        
        # Stochastic Oscillator
        stoch_result = stochastic_oscillator(
            df['high'],
            df['low'],
            df['close'],
            k_window=INDICATOR_PARAMS['stochastic']['k_window'],
            d_window=INDICATOR_PARAMS['stochastic']['d_window']
        )
        indicators.update(stoch_result)
        
        # ADX
        adx_result = adx(
            df['high'],
            df['low'],
            df['close'],
            window=INDICATOR_PARAMS['adx']['window']
        )
        indicators.update(adx_result)
        
        # Ichimoku Cloud
        ichimoku = ichimoku_cloud(
            df['high'],
            df['low'],
            df['close'],
            tenkan_period=INDICATOR_PARAMS['ichimoku']['tenkan_period'],
            kijun_period=INDICATOR_PARAMS['ichimoku']['kijun_period'],
            senkou_span_b_period=INDICATOR_PARAMS['ichimoku']['senkou_span_b_period']
        )
        indicators.update(ichimoku)
        
        # Parabolic SAR
        indicators['sar'] = parabolic_sar(
            df['high'],
            df['low']
        )
        
        # Volume Profile
        indicators['volume_profile'] = volume_profile(
            df['close'],
            df['volume']
        )
        
        return indicators
    
    def _generate_signals(self, df: pd.DataFrame, indicators: dict) -> dict:
        """Generate trading signals based on indicators."""
        signals = {
            'trend': 'NEUTRAL',
            'momentum': 'NEUTRAL',
            'volatility': 'LOW',
            'signals': [],
            'score': 0
        }
        
        current_close = df['close'].iloc[-1]
        
        # Trend signals
        sma_20 = indicators['sma_20'].values[-1]
        sma_50 = indicators['sma_50'].values[-1]
        
        if current_close > sma_20 and sma_20 > sma_50:
            signals['trend'] = 'UPTREND'
            signals['score'] += 1
        elif current_close < sma_20 and sma_20 < sma_50:
            signals['trend'] = 'DOWNTREND'
            signals['score'] -= 1
        
        # RSI signals
        rsi_value = indicators['rsi'].values[-1]
        if not np.isnan(rsi_value):
            if rsi_value > INDICATOR_PARAMS['rsi']['overbought']:
                signals['momentum'] = 'OVERBOUGHT'
                signals['score'] -= 1
            elif rsi_value < INDICATOR_PARAMS['rsi']['oversold']:
                signals['momentum'] = 'OVERSOLD'
                signals['score'] += 1
        
        # MACD signals
        macd_line = indicators['MACD'].values[-1]
        signal_line = indicators['MACD_Signal'].values[-1]
        
        if macd_line > signal_line and indicators['MACD_Hist'].values[-1] > 0:
            signals['signals'].append('MACD_BULLISH_CROSS')
            signals['score'] += 1
        elif macd_line < signal_line and indicators['MACD_Hist'].values[-1] < 0:
            signals['signals'].append('MACD_BEARISH_CROSS')
            signals['score'] -= 1
        
        # Bollinger Bands
        bb_upper = indicators['BB_Upper'].values[-1]
        bb_lower = indicators['BB_Lower'].values[-1]
        
        if current_close > bb_upper:
            signals['volatility'] = 'HIGH'
            signals['signals'].append('PRICE_ABOVE_BB_UPPER')
            signals['score'] -= 1
        elif current_close < bb_lower:
            signals['volatility'] = 'HIGH'
            signals['signals'].append('PRICE_BELOW_BB_LOWER')
            signals['score'] += 1
        
        # ADX trend strength
        adx_value = indicators['adx'].values[-1]
        if not np.isnan(adx_value):
            if adx_value > INDICATOR_PARAMS['adx']['trend_threshold']:
                signals['trend'] = f"STRONG_{signals['trend']}"
                signals['score'] += 1 if 'UPTREND' in signals['trend'] else -1
        
        # Ichimoku Cloud
        tenkan = indicators['Ichimoku_Tenkan'].values[-1]
        kijun = indicators['Ichimoku_Kijun'].values[-1]
        senkou_a = indicators['Ichimoku_Senkou_A'].values[-26]  # 26 periods ahead
        senkou_b = indicators['Ichimoku_Senkou_B'].values[-26]  # 26 periods ahead
        
        if not (np.isnan(tenkan) or np.isnan(kijun) or np.isnan(senkou_a) or np.isnan(senkou_b)):
            if tenkan > kijun and current_close > senkou_a and current_close > senkou_b:
                signals['signals'].append('ICHIMOKU_BULLISH')
                signals['score'] += 2
            elif tenkan < kijun and current_close < senkou_a and current_close < senkou_b:
                signals['signals'].append('ICHIMOKU_BEARISH')
                signals['score'] -= 2
        
        return signals
    
    def display_results(self):
        """Display scan results in a formatted table."""
        if not self.scan_results:
            print("No scan results available.")
            return
        
        # Prepare data for display
        table_data = []
        for key, result in self.scan_results.items():
            symbol = result['symbol']
            timeframe = result['timeframe']
            signals = result['signals']
            
            # Get OHLC data
            ohlcv = self.mt5.get_ohlcv(symbol, timeframe, limit=1)[0]
            
            # Format signals
            signal_text = ", ".join(signals['signals'][:3])  # Show first 3 signals
            if len(signals['signals']) > 3:
                signal_text += f" (+{len(signals['signals']) - 3} more)"
            
            # Add to table
            table_data.append([
                symbol,
                timeframe,
                f"{ohlcv.open:.5f}",
                f"{ohlcv.high:.5f}",
                f"{ohlcv.low:.5f}",
                f"{ohlcv.close:.5f}",
                signals['trend'],
                signals['momentum'],
                signal_text,
                signals['score']
            ])
        
        # Sort by score (descending)
        table_data.sort(key=lambda x: x[-1], reverse=True)
        
        # Display table
        headers = [
            'Symbol', 'TF', 'Open', 'High', 'Low', 'Close', 
            'Trend', 'Momentum', 'Signals', 'Score'
        ]
        print("\n" + "=" * 120)
        print(f"MARKET SCAN RESULTS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 120)
        print(tabulate(table_data, headers=headers, tablefmt='grid'))
        print("=" * 120 + "\n")
    
    async def run(self):
        """Run the market scanner continuously."""
        try:
            await self.initialize()
            
            while True:
                start_time = time.time()
                
                try:
                    await self.scan_market()
                    self.display_results()
                except Exception as e:
                    logger.error(f"Error during market scan: {e}")
                
                # Calculate sleep time
                elapsed = time.time() - start_time
                sleep_time = max(SCAN_INTERVAL - elapsed, 1)
                
                logger.info(f"Next scan in {int(sleep_time)} seconds...")
                await asyncio.sleep(sleep_time)
                
        except KeyboardInterrupt:
            logger.info("Market scanner stopped by user")
        except Exception as e:
            logger.error(f"Fatal error in market scanner: {e}")
        finally:
            await self.mt5.disconnect()

async def main():
    """Main function to run the market scanner."""
    # Check if required environment variables are set
    required_vars = ['MT5_LOGIN', 'MT5_PASSWORD', 'MT5_SERVER']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("Error: The following environment variables must be set:")
        for var in missing_vars:
            print(f"- {var}")
        print("\nPlease create a .env file with these variables or set them in your environment.")
        return
    
    # Run the market scanner
    scanner = MarketScanner()
    await scanner.run()

if __name__ == "__main__":
    # Load environment variables from .env file if it exists
    from dotenv import load_dotenv
    load_dotenv()
    
    # Run the async main function
    asyncio.run(main())
