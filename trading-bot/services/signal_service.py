"""
Signal generation service for trading strategies.
"""
import logging
from typing import Dict, List, Optional, Any, Callable
import pandas as pd
import numpy as np
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Types of trading signals."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    EXIT = "EXIT"

@dataclass
class Signal:
    """Trading signal with metadata."""
    symbol: str
    signal_type: SignalType
    strength: float  # 0.0 to 1.0
    timestamp: datetime
    price: float
    indicators: Dict[str, Any] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary."""
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'strength': self.strength,
            'timestamp': self.timestamp.isoformat(),
            'price': self.price,
            'indicators': self.indicators or {},
            'metadata': self.metadata or {}
        }

class SignalService:
    """Service for generating and managing trading signals."""
    
    def __init__(self):
        """Initialize the signal service."""
        self.subscribers: List[Callable[[Signal], None]] = []
        self.strategies: Dict[str, Callable[[pd.DataFrame], Signal]] = {}
    
    def register_strategy(
        self, 
        name: str, 
        strategy_func: Callable[[pd.DataFrame], Signal]
    ) -> None:
        """Register a new trading strategy.
        
        Args:
            name: Strategy name
            strategy_func: Function that takes market data and returns a Signal
        """
        self.strategies[name] = strategy_func
        logger.info(f"Registered strategy: {name}")
    
    def subscribe(self, callback: Callable[[Signal], None]) -> None:
        """Subscribe to signal updates.
        
        Args:
            callback: Function to call when a new signal is generated
        """
        if callback not in self.subscribers:
            self.subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable[[Signal], None]) -> None:
        """Unsubscribe from signal updates.
        
        Args:
            callback: Callback function to remove
        """
        if callback in self.subscribers:
            self.subscribers.remove(callback)
    
    def generate_signals(
        self, 
        symbol: str, 
        data: pd.DataFrame,
        strategy_name: Optional[str] = None
    ) -> List[Signal]:
        """Generate signals for the given market data.
        
        Args:
            symbol: Instrument symbol
            data: Market data DataFrame
            strategy_name: Optional strategy name (if None, runs all strategies)
            
        Returns:
            List of generated signals
        """
        signals = []
        
        if strategy_name:
            if strategy_name in self.strategies:
                try:
                    signal = self.strategies[strategy_name](data)
                    if signal:
                        signals.append(signal)
                except Exception as e:
                    logger.error(f"Error in strategy '{strategy_name}': {e}")
        else:
            for name, strategy in self.strategies.items():
                try:
                    signal = strategy(data)
                    if signal:
                        signals.append(signal)
                except Exception as e:
                    logger.error(f"Error in strategy '{name}': {e}")
        
        # Notify subscribers
        for signal in signals:
            self._notify_subscribers(signal)
        
        return signals
    
    def _notify_subscribers(self, signal: Signal) -> None:
        """Notify all subscribers about a new signal."""
        for callback in self.subscribers:
            try:
                callback(signal)
            except Exception as e:
                logger.error(f"Error in signal subscriber: {e}")

# Example strategy implementations
def moving_average_crossover(data: pd.DataFrame, 
                          fast_period: int = 10, 
                          slow_period: int = 50) -> Signal:
    """Simple moving average crossover strategy."""
    if len(data) < slow_period:
        return None
    
    close_prices = data['close'].values
    fast_ma = np.mean(close_prices[-fast_period:])
    slow_ma = np.mean(close_prices[-slow_period:])
    
    # Previous values for crossover detection
    prev_fast_ma = np.mean(close_prices[-(fast_period+1):-1])
    prev_slow_ma = np.mean(close_prices[-(slow_period+1):-1])
    
    current_price = close_prices[-1]
    timestamp = data.index[-1].to_pydatetime()
    
    # Check for crossover
    if prev_fast_ma <= prev_slow_ma and fast_ma > slow_ma:
        return Signal(
            symbol=data.attrs.get('symbol', 'UNKNOWN'),
            signal_type=SignalType.BUY,
            strength=0.7,
            timestamp=timestamp,
            price=current_price,
            indicators={
                'fast_ma': fast_ma,
                'slow_ma': slow_ma,
                'strategy': 'MA Crossover'
            }
        )
    elif prev_fast_ma >= prev_slow_ma and fast_ma < slow_ma:
        return Signal(
            symbol=data.attrs.get('symbol', 'UNKNOWN'),
            signal_type=SignalType.SELL,
            strength=0.7,
            timestamp=timestamp,
            price=current_price,
            indicators={
                'fast_ma': fast_ma,
                'slow_ma': slow_ma,
                'strategy': 'MA Crossover'
            }
        )
    
    return None
