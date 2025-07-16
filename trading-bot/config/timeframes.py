"""
Timeframe configurations and utilities.
"""
from enum import Enum, auto
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple


class Timeframe(Enum):
    """Supported timeframes for the trading bot."""
    M1 = (1, "1m", "1 Minute")
    M5 = (5, "5m", "5 Minutes")
    M15 = (15, "15m", "15 Minutes")
    M30 = (30, "30m", "30 Minutes")
    H1 = (60, "1h", "1 Hour")
    H2 = (120, "2h", "2 Hours")
    H4 = (240, "4h", "4 Hours")
    D1 = (1440, "1d", "1 Day")
    W1 = (10080, "1w", "1 Week")
    MN1 = (43200, "1M", "1 Month")
    
    def __init__(self, minutes: int, short_code: str, display_name: str):
        self.minutes = minutes
        self.short_code = short_code
        self.display_name = display_name
    
    @classmethod
    def from_string(cls, value: str) -> 'Timeframe':
        """Get Timeframe enum from string representation."""
        value = value.upper().replace(' ', '').replace('_', '')
        for tf in cls:
            if value in (tf.name, tf.short_code.upper()):
                return tf
        raise ValueError(f"Invalid timeframe: {value}")
    
    def to_timedelta(self) -> timedelta:
        """Convert timeframe to timedelta."""
        return timedelta(minutes=self.minutes)
    
    def is_intraday(self) -> bool:
        """Check if timeframe is intraday (less than 1 day)."""
        return self.minutes < 1440  # 1 day in minutes
    
    def is_swing(self) -> bool:
        """Check if timeframe is suitable for swing trading (1D or higher)."""
        return self.minutes >= 1440  # 1 day or more
    
    def is_scalping(self) -> bool:
        """Check if timeframe is suitable for scalping (1M-15M)."""
        return 1 <= self.minutes <= 15
    
    def is_day_trading(self) -> bool:
        """Check if timeframe is suitable for day trading (30M-4H)."""
        return 30 <= self.minutes <= 240
    
    def get_aggregation_levels(self) -> List['Timeframe']:
        """Get higher timeframes for multi-timeframe analysis."""
        timeframes = list(Timeframe)
        current_idx = timeframes.index(self)
        return timeframes[current_idx+1:]
    
    def get_lower_timeframes(self) -> List['Timeframe']:
        """Get lower timeframes for multi-timeframe analysis."""
        timeframes = list(Timeframe)
        current_idx = timeframes.index(self)
        return timeframes[:current_idx]


class MarketSession(Enum):
    """Market session definitions."""
    ASIAN = (time(0, 0), time(8, 0), "Asian")
    LONDON = (time(8, 0), time(16, 0), "London")
    NEW_YORK = (time(13, 0), time(21, 0), "New York")
    LONDON_NY_OVERLAP = (time(13, 0), time(16, 0), "London-NY Overlap")
    OVERNIGHT = (time(21, 0), time(23, 59, 59), "Overnight")
    
    def __init__(self, start_time: time, end_time: time, display_name: str):
        self.start_time = start_time
        self.end_time = end_time
        self.display_name = display_name
    
    def is_active(self, current_time: Optional[datetime] = None) -> bool:
        """Check if the session is currently active."""
        if current_time is None:
            current_time = datetime.utcnow().time()
        
        # Handle overnight session (crosses midnight)
        if self == MarketSession.OVERNIGHT:
            return current_time >= self.start_time or current_time <= time(4, 0)
        
        return self.start_time <= current_time < self.end_time
    
    @classmethod
    def get_current_session(cls, current_time: Optional[datetime] = None) -> 'MarketSession':
        """Get the current market session."""
        if current_time is None:
            current_time = datetime.utcnow()
        
        current_time = current_time.time()
        
        # Check for session overlaps first
        if MarketSession.LONDON_NY_OVERLAP.is_active(current_time):
            return MarketSession.LONDON_NY_OVERLAP
        
        # Check other sessions
        for session in cls:
            if session != MarketSession.LONDON_NY_OVERLAP and session.is_active(current_time):
                return session
        
        # Default to overnight if no other session is active
        return MarketSession.OVERNIGHT
    
    def get_volatility_multiplier(self) -> float:
        """Get volatility multiplier for the session."""
        multipliers = {
            MarketSession.ASIAN: 0.8,
            MarketSession.LONDON: 1.0,
            MarketSession.NEW_YORK: 1.0,
            MarketSession.LONDON_NY_OVERLAP: 1.2,
            MarketSession.OVERNIGHT: 0.5
        }
        return multipliers[self]


def get_market_sessions() -> Dict[str, Dict]:
    """Get market session definitions."""
    return {
        'asian': {
            'name': 'Asian',
            'start': '00:00',
            'end': '08:00',
            'volatility_multiplier': 0.8,
            'description': 'Tokyo, Singapore, Hong Kong'
        },
        'london': {
            'name': 'London',
            'start': '08:00',
            'end': '16:00',
            'volatility_multiplier': 1.0,
            'description': 'London, Frankfurt, Paris'
        },
        'new_york': {
            'name': 'New York',
            'start': '13:00',
            'end': '21:00',
            'volatility_multiplier': 1.0,
            'description': 'New York, Toronto'
        },
        'london_ny_overlap': {
            'name': 'London-NY Overlap',
            'start': '13:00',
            'end': '16:00',
            'volatility_multiplier': 1.2,
            'description': 'High volatility period'
        },
        'overnight': {
            'name': 'Overnight',
            'start': '21:00',
            'end': '00:00',
            'volatility_multiplier': 0.5,
            'description': 'Low liquidity period'
        }
    }


def get_timeframe_by_minutes(minutes: int) -> Optional[Timeframe]:
    """Get timeframe by number of minutes."""
    for tf in Timeframe:
        if tf.minutes == minutes:
            return tf
    return None


def get_timeframe_by_shortcode(short_code: str) -> Optional[Timeframe]:
    """Get timeframe by short code (e.g., '1h', '4h', '1d')."""
    for tf in Timeframe:
        if tf.short_code.lower() == short_code.lower():
            return tf
    return None


def get_timeframe_by_name(name: str) -> Optional[Timeframe]:
    """Get timeframe by name (e.g., 'M15', 'H1', 'D1')."""
    try:
        return Timeframe[name.upper()]
    except KeyError:
        return None


def get_all_timeframes() -> List[Timeframe]:
    """Get all available timeframes."""
    return list(Timeframe)


def get_intraday_timeframes() -> List[Timeframe]:
    """Get all intraday timeframes (less than 1 day)."""
    return [tf for tf in Timeframe if tf.is_intraday()]


def get_swing_timeframes() -> List[Timeframe]:
    """Get all swing trading timeframes (1 day or more)."""
    return [tf for tf in Timeframe if tf.is_swing()]


def get_scalping_timeframes() -> List[Timeframe]:
    """Get all scalping timeframes (1M-15M)."""
    return [tf for tf in Timeframe if tf.is_scalping()]


def get_day_trading_timeframes() -> List[Timeframe]:
    """Get all day trading timeframes (30M-4H)."""
    return [tf for tf in Timeframe if tf.is_day_trading()]
