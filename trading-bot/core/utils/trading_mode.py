"""
Trading Mode Management

This module provides functionality for managing and validating trading modes
across the application.
"""
from enum import Enum, auto
from typing import Optional, Dict, Any
from dataclasses import dataclass
import logging
from pathlib import Path
import yaml

# Configure logging
logger = logging.getLogger(__name__)

class TradingMode(Enum):
    """Enumeration of supported trading modes."""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"

@dataclass
class TradingModeConfig:
    """Configuration for a trading mode."""
    mode: TradingMode
    risk_multiplier: float = 1.0
    enable_shorting: bool = True
    max_leverage: float = 1.0
    max_drawdown_pct: float = 10.0
    daily_loss_limit_pct: float = 5.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'mode': self.mode.value,
            'risk_multiplier': self.risk_multiplier,
            'enable_shorting': self.enable_shorting,
            'max_leverage': self.max_leverage,
            'max_drawdown_pct': self.max_drawdown_pct,
            'daily_loss_limit_pct': self.daily_loss_limit_pct
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradingModeConfig':
        """Create config from dictionary."""
        return cls(
            mode=TradingMode(data['mode']),
            risk_multiplier=float(data.get('risk_multiplier', 1.0)),
            enable_shorting=bool(data.get('enable_shorting', True)),
            max_leverage=float(data.get('max_leverage', 1.0)),
            max_drawdown_pct=float(data.get('max_drawdown_pct', 10.0)),
            daily_loss_limit_pct=float(data.get('daily_loss_limit_pct', 5.0))
        )

class TradingModeManager:
    """Manages trading mode and its configuration."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_path: Optional[str] = None):
        if self._initialized:
            return
            
        self.config_path = config_path or 'config/trading_mode.yaml'
        self._mode: Optional[TradingMode] = None
        self._config: Optional[TradingModeConfig] = None
        self._initialized = True
        
        # Load initial config
        self._load_config()
    
    def _load_config(self) -> None:
        """Load configuration from file."""
        try:
            config_file = Path(self.config_path)
            if config_file.exists():
                with open(config_file, 'r') as f:
                    config_data = yaml.safe_load(f) or {}
                self._mode = TradingMode(config_data.get('mode', 'paper'))
                self._config = TradingModeConfig.from_dict(config_data)
            else:
                # Default to paper trading
                self._mode = TradingMode.PAPER
                self._config = TradingModeConfig(mode=TradingMode.PAPER)
                self._save_config()
                
        except Exception as e:
            logger.error(f"Failed to load trading mode config: {e}")
            self._mode = TradingMode.PAPER
            self._config = TradingModeConfig(mode=TradingMode.PAPER)
    
    def _save_config(self) -> None:
        """Save current configuration to file."""
        if not self._config:
            return
            
        try:
            config_file = Path(self.config_path)
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w') as f:
                yaml.safe_dump(self._config.to_dict(), f)
                
        except Exception as e:
            logger.error(f"Failed to save trading mode config: {e}")
    
    @property
    def current_mode(self) -> TradingMode:
        """Get current trading mode."""
        if self._mode is None:
            self._load_config()
        return self._mode or TradingMode.PAPER
    
    @current_mode.setter
    def current_mode(self, mode: TradingMode) -> None:
        """Set current trading mode and update config."""
        if not isinstance(mode, TradingMode):
            raise ValueError(f"Invalid trading mode: {mode}")
            
        self._mode = mode
        if self._config:
            self._config.mode = mode
        else:
            self._config = TradingModeConfig(mode=mode)
        
        self._save_config()
        logger.info(f"Trading mode changed to: {mode.value}")
    
    @property
    def config(self) -> TradingModeConfig:
        """Get current trading configuration."""
        if self._config is None:
            self._load_config()
        return self._config or TradingModeConfig(mode=TradingMode.PAPER)
    
    def is_live_trading(self) -> bool:
        """Check if currently in live trading mode."""
        return self.current_mode == TradingMode.LIVE
    
    def is_paper_trading(self) -> bool:
        """Check if currently in paper trading mode."""
        return self.current_mode == TradingMode.PAPER
    
    def is_backtesting(self) -> bool:
        """Check if currently in backtesting mode."""
        return self.current_mode == TradingMode.BACKTEST

# Global instance
trading_mode_manager = TradingModeManager()

def get_trading_mode() -> TradingMode:
    """Get the current trading mode."""
    return trading_mode_manager.current_mode

def is_live_trading() -> bool:
    """Check if currently in live trading mode."""
    return trading_mode_manager.is_live_trading()

def is_paper_trading() -> bool:
    """Check if currently in paper trading mode."""
    return trading_mode_manager.is_paper_trading()

def is_backtesting() -> bool:
    """Check if currently in backtesting mode."""
    return trading_mode_manager.is_backtesting()

def get_trading_config() -> TradingModeConfig:
    """Get the current trading configuration."""
    return trading_mode_manager.config
