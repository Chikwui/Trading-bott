"""
Advanced exposure management system for portfolio risk control.

This module provides tools for monitoring and controlling:
- Position concentration across assets, sectors, and regions
- Correlation risk between positions
- Factor exposures (beta, volatility, style factors)
- Liquidity constraints
- Risk contribution analysis
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any, Set, DefaultDict, Callable
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import logging
import asyncio
from scipy.stats import spearmanr, norm, t
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Type, TypeVar
import pytz

# Configure logging
logger = logging.getLogger(__name__)

# Type variable for market data provider
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

# Abstract base class for market data providers
class MarketDataProvider(ABC):
    """Abstract base class for market data providers."""
    
    @abstractmethod
    async def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: datetime, 
        end_date: Optional[datetime] = None,
        count: Optional[int] = None
    ) -> pd.DataFrame:
        """Get historical price data."""
        pass
    
    @abstractmethod
    async def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        pass
    
    @abstractmethod
    async def subscribe_to_updates(
        self, 
        symbols: List[str], 
        callback: Callable[[Dict[str, float]], None],
        interval: int = 60
    ) -> None:
        """Subscribe to real-time price updates."""
        pass

# MT5 implementation of market data provider
class MT5MarketDataProvider(MarketDataProvider):
    """Market data provider using MetaTrader 5."""
    
    TIMEFRAME_MAP = {
        'M1': mt5.TIMEFRAME_M1,
        'M5': mt5.TIMEFRAME_M5,
        'M15': mt5.TIMEFRAME_M15,
        'H1': mt5.TIMEFRAME_H1,
        'H4': mt5.TIMEFRAME_H4,
        'D1': mt5.TIMEFRAME_D1,
        'W1': mt5.TIMEFRAME_W1,
        'MN1': mt5.TIMEFRAME_MN1,
    }
    
    def __init__(self, symbols: Optional[List[str]] = None):
        if not MT5_AVAILABLE:
            raise RuntimeError("MetaTrader5 package is not available")
        self.connected = False
        self.symbols = symbols or []
        self.subscribers = {}
        self._update_task = None
    
    async def connect(self) -> bool:
        """Connect to MT5 terminal."""
        if not mt5.initialize():
            logger.error(f"MT5 initialize() failed: {mt5.last_error()}")
            return False
        self.connected = True
        return True
    
    async def get_historical_data(
        self, 
        symbol: str, 
        timeframe: str, 
        start_date: datetime, 
        end_date: Optional[datetime] = None,
        count: Optional[int] = None
    ) -> pd.DataFrame:
        """Get historical price data from MT5."""
        if not self.connected:
            await self.connect()
        
        tf = self.TIMEFRAME_MAP.get(timeframe.upper())
        if tf is None:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        
        if count:
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
        else:
            if end_date is None:
                end_date = datetime.now()
            rates = mt5.copy_rates_range(symbol, tf, start_date, end_date)
        
        if rates is None or len(rates) == 0:
            logger.error(f"No data for {symbol} in {timeframe} timeframe")
            return pd.DataFrame()
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        return df
    
    async def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        if not self.connected:
            await self.connect()
        
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Failed to get tick for {symbol}")
            return 0.0
        return (tick.ask + tick.bid) / 2
    
    async def subscribe_to_updates(
        self, 
        symbols: List[str], 
        callback: Callable[[Dict[str, float]], None],
        interval: int = 60
    ) -> None:
        """Subscribe to real-time price updates."""
        self.symbols = list(set(self.symbols + symbols))
        self.subscribers[tuple(symbols)] = (callback, interval)
        
        if self._update_task is None or self._update_task.done():
            self._update_task = asyncio.create_task(self._update_loop())
    
    async def _update_loop(self):
        """Background task to fetch updates at specified intervals."""
        while True:
            try:
                updates = {}
                for symbol in self.symbols:
                    price = await self.get_current_price(symbol)
                    if price > 0:
                        updates[symbol] = price
                
                # Notify all subscribers
                for (symbols, (callback, interval)) in self.subscribers.items():
                    if any(s in updates for s in symbols):
                        callback(updates)
                
                # Sleep for the shortest interval among subscribers
                min_interval = min(interval for _, (_, interval) in self.subscribers.items())
                await asyncio.sleep(min_interval)
                
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying

# Risk model base class
class RiskModel(ABC):
    """Abstract base class for risk models."""
    
    @abstractmethod
    def calculate_risk_metrics(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
        market_data: Dict[str, pd.DataFrame],
        **kwargs
    ) -> Dict[str, Any]:
        """Calculate risk metrics for the given positions and market data."""
        pass

# Implementation of various risk models
class VaRModel(RiskModel):
    """Value at Risk (VaR) model using historical simulation."""
    
    def __init__(self, confidence_level: float = 0.95, lookback_days: int = 252):
        self.confidence_level = confidence_level
        self.lookback_days = lookback_days
    
    def calculate_risk_metrics(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
        market_data: Dict[str, pd.DataFrame],
        **kwargs
    ) -> Dict[str, Any]:
        """Calculate Value at Risk (VaR) using historical simulation."""
        if not positions or not market_data:
            return {}
        
        # Get common index for all symbols
        common_dates = None
        returns = {}
        
        for symbol, df in market_data.items():
            if symbol not in positions:
                continue
                
            # Calculate daily returns
            if 'close' in df.columns and len(df) > 1:
                symbol_returns = df['close'].pct_change().dropna()
                returns[symbol] = symbol_returns
                
                if common_dates is None:
                    common_dates = set(symbol_returns.index)
                else:
                    common_dates = common_dates.intersection(symbol_returns.index)
        
        if not returns or not common_dates:
            return {}
        
        # Align returns to common dates
        aligned_returns = {}
        for symbol, ret in returns.items():
            aligned_returns[symbol] = ret[ret.index.isin(common_dates)]
        
        # Calculate portfolio returns
        portfolio_returns = pd.Series(0.0, index=common_dates)
        total_value = sum(abs(v) for v in positions.values())
        
        for symbol, position in positions.items():
            if symbol in aligned_returns:
                weight = position / total_value if total_value != 0 else 0
                portfolio_returns += aligned_returns[symbol] * weight
        
        # Calculate VaR
        var = -np.percentile(portfolio_returns, (1 - self.confidence_level) * 100)
        
        return {
            'var': var,
            'var_confidence': self.confidence_level,
            'historical_returns': portfolio_returns,
            'lookback_days': self.lookback_days
        }

class CVaRModel(RiskModel):
    """Conditional Value at Risk (CVaR) model."""
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
    
    def calculate_risk_metrics(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
        market_data: Dict[str, pd.DataFrame],
        **kwargs
    ) -> Dict[str, Any]:
        """Calculate Conditional Value at Risk (CVaR)."""
        # First calculate VaR
        var_model = VaRModel(confidence_level=self.confidence_level)
        var_result = var_model.calculate_risk_metrics(positions, prices, market_data)
        
        if not var_result or 'historical_returns' not in var_result:
            return {}
        
        returns = var_result['historical_returns']
        var = var_result['var']
        
        # Calculate CVaR as the average of returns worse than -VaR
        tail_returns = returns[returns <= -var]
        cvar = -tail_returns.mean() if len(tail_returns) > 0 else 0
        
        return {
            'cvar': cvar,
            'var': var,
            'cvar_confidence': self.confidence_level,
            'tail_observations': len(tail_returns)
        }

class FactorRiskModel(RiskModel):
    """Multi-factor risk model for analyzing factor exposures."""
    
    def __init__(self, factors: List[str] = None):
        self.factors = factors or ['market', 'size', 'value', 'momentum', 'volatility']
    
    def calculate_risk_metrics(
        self,
        positions: Dict[str, float],
        prices: Dict[str, float],
        market_data: Dict[str, pd.DataFrame],
        factor_returns: Optional[Dict[str, pd.Series]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Calculate factor exposures and contributions to risk."""
        if not positions or not market_data:
            return {}
        
        # Simulate factor returns if not provided
        if factor_returns is None:
            factor_returns = self._simulate_factor_returns(market_data)
        
        # Calculate factor exposures
        factor_exposures = {f: 0.0 for f in self.factors}
        total_value = sum(abs(v) for v in positions.values())
        
        for symbol, position in positions.items():
            if symbol not in market_data or total_value == 0:
                continue
                
            weight = position / total_value
            
            # In a real implementation, this would use actual factor models
            # Here we simulate factor exposures
            for factor in self.factors:
                # Simulate factor exposure based on symbol characteristics
                exposure = self._simulate_factor_exposure(symbol, factor)
                factor_exposures[factor] += exposure * abs(weight)
        
        # Calculate factor contributions to risk
        factor_volatility = {f: returns.std() for f, returns in factor_returns.items()}
        total_volatility = sum(factor_volatility.values())
        
        factor_contributions = {
            f: (factor_exposures[f] * factor_volatility[f] / total_volatility 
                if total_volatility > 0 else 0)
            for f in self.factors
        }
        
        return {
            'factor_exposures': factor_exposures,
            'factor_contributions': factor_contributions,
            'factor_volatility': factor_volatility
        }
    
    def _simulate_factor_returns(self, market_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """Simulate factor returns for demonstration."""
        # In a real implementation, this would use actual factor return data
        dates = None
        for df in market_data.values():
            if dates is None or len(df) < len(dates):
                dates = df.index
        
        np.random.seed(42)  # For reproducibility
        factor_data = {}
        
        for factor in self.factors:
            # Generate random walk for each factor
            changes = np.random.normal(0, 0.01, len(dates))
            factor_returns = pd.Series(np.cumsum(changes), index=dates)
            factor_data[factor] = factor_returns.pct_change().dropna()
        
        return factor_data
    
    def _simulate_factor_exposure(self, symbol: str, factor: str) -> float:
        """Simulate factor exposure for a symbol."""
        # In a real implementation, this would use actual factor exposures
        # Here we use a hash of the symbol to generate deterministic but random-looking exposures
        symbol_hash = hash(symbol) % 1000 / 1000.0
        factor_hash = hash(factor) % 1000 / 1000.0
        
        # Generate exposure between -1 and 1
        exposure = (symbol_hash + factor_hash) % 2 - 1
        
        # Add some structure to make it more realistic
        if factor == 'market':
            exposure = 0.5 + 0.5 * exposure  # Most stocks have positive market beta
        elif factor == 'size':
            exposure = symbol_hash  # Smaller stocks tend to have higher size factor
        elif factor == 'value':
            exposure = exposure * 0.8  # Less extreme value exposures
        
        return exposure

# Update the ExposureManager to use the new market data and risk models
class ExposureManager:
    """
    Advanced exposure management system for monitoring and controlling portfolio exposures.
    
    Features:
    - Real-time exposure tracking across multiple dimensions
    - Dynamic position limits based on market conditions
    - Correlation analysis and risk contribution
    - Liquidity monitoring
    - Factor exposure analysis
    - Stress testing and scenario analysis
    - Integration with market data providers
    - Advanced risk models (VaR, CVaR, Factor models)
    """
    
    def __init__(
        self,
        portfolio_value: float = 1_000_000,
        base_currency: str = "USD",
        risk_free_rate: float = 0.0,
        lookback_days: int = 60,
        min_liquidity_days: int = 5,
        market_data_provider: Optional[MarketDataProvider] = None,
        risk_models: Optional[List[RiskModel]] = None
    ):
        self.portfolio_value = portfolio_value
        self.base_currency = base_currency
        self.risk_free_rate = risk_free_rate
        self.lookback_days = lookback_days
        self.min_liquidity_days = min_liquidity_days
        
        # Initialize market data provider
        self.market_data_provider = market_data_provider or MT5MarketDataProvider()
        if isinstance(self.market_data_provider, MT5MarketDataProvider):
            asyncio.create_task(self.market_data_provider.connect())
        
        # Initialize risk models
        self.risk_models = risk_models or [
            VaRModel(confidence_level=0.95),
            CVaRModel(confidence_level=0.95),
            FactorRiskModel()
        ]
        
        # Position tracking
        self.positions: Dict[str, Position] = {}
        self.position_history: List[Dict] = []
        
        # Exposure limits
        self.limits: Dict[Tuple[ExposureType, str], ExposureLimit] = {}
        
        # Market data cache
        self.market_data: Dict[str, pd.DataFrame] = {}
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.last_updated: Optional[datetime] = None
        
        # Risk metrics
        self.risk_metrics: Dict[str, Any] = {}
        self.portfolio_beta: float = 0.0
        
        # Initialize default limits
        self._set_default_limits()
        
        # Start background tasks
        self._background_tasks = set()
        self._stop_event = asyncio.Event()
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background tasks for data updates."""
        if hasattr(self, '_update_task') and not self._update_task.done():
            return
            
        self._update_task = asyncio.create_task(self._periodic_update())
    
    async def _periodic_update(self):
        """Periodically update market data and risk metrics."""
        while not self._stop_event.is_set():
            try:
                # Update market data for all positions
                await self._update_market_data()
                
                # Recalculate risk metrics
                await self._update_risk_metrics()
                
                # Wait for next update (e.g., every 5 minutes)
                await asyncio.sleep(300)
                
            except Exception as e:
                logger.error(f"Error in periodic update: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _update_market_data(self):
        """Update market data for all positions."""
        if not self.positions:
            return
            
        symbols = list(self.positions.keys())
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.lookback_days)
        
        for symbol in symbols:
            try:
                # Get historical data
                df = await self.market_data_provider.get_historical_data(
                    symbol=symbol,
                    timeframe='D1',  # Daily data
                    start_date=start_date,
                    end_date=end_date
                )
                
                if not df.empty:
                    self.market_data[symbol] = df
                
            except Exception as e:
                logger.error(f"Error updating market data for {symbol}: {e}")
        
        # Update correlation matrix if we have enough data
        self._update_correlations()
    
    async def _update_risk_metrics(self):
        """
        Update all risk metrics using registered risk models.
        
        This method:
        1. Fetches current market prices for all positions
        2. Calculates position values and weights
        3. Computes returns and other market data metrics
        4. Applies each risk model to the current portfolio
        5. Stores and aggregates the results
        """
        try:
            if not self.positions or not self.market_data:
                logger.debug("No positions or market data available for risk metrics update")
                return
            
            # Get current prices with error handling
            prices = {}
            price_errors = 0
            for symbol in self.positions:
                try:
                    price = await self.market_data_provider.get_current_price(symbol)
                    if price and price > 0:
                        prices[symbol] = price
                    else:
                        price_errors += 1
                        logger.warning(f"Invalid price for {symbol}: {price}")
                except Exception as e:
                    price_errors += 1
                    logger.error(f"Error getting price for {symbol}: {e}")
            
            if not prices:
                logger.warning("No valid prices available for risk calculation")
                return
                
            if price_errors > 0:
                logger.warning(f"Failed to get prices for {price_errors} out of {len(self.positions)} positions")
            
            # Calculate position values and weights
            position_values = {}
            total_value = 0.0
            
            for symbol, position in self.positions.items():
                if symbol in prices:
                    value = position.quantity * prices[symbol]
                    position_values[symbol] = value
                    total_value += value
            
            if total_value <= 0:
                logger.warning("Total portfolio value is zero or negative")
                return
                
            # Calculate position weights
            position_weights = {
                symbol: value / total_value
                for symbol, value in position_values.items()
            }
            
            # Prepare returns data for risk models
            returns_data = {}
            for symbol, data in self.market_data.items():
                if symbol in prices and not data.empty and 'close' in data.columns:
                    returns = data['close'].pct_change().dropna()
                    if not returns.empty:
                        returns_data[symbol] = returns
            
            # Reset risk metrics
            self.risk_metrics = {
                'timestamp': datetime.utcnow(),
                'portfolio_value': total_value,
                'position_values': position_values,
                'position_weights': position_weights,
                'models': {}
            }
            
            # Calculate metrics using each risk model
            for model in self.risk_models:
                try:
                    model_name = model.__class__.__name__
                    logger.debug(f"Calculating risk metrics using {model_name}")
                    
                    # Prepare model-specific inputs
                    model_kwargs = {
                        'returns': returns_data,
                        'position_values': position_values,
                        'position_weights': position_weights,
                        'prices': prices,
                        'market_data': self.market_data
                    }
                    
                    # Special handling for different model types
                    if isinstance(model, FactorRiskModel):
                        # Add factor-specific data
                        model_kwargs.update({
                            'factor_returns': self._get_factor_returns(),
                            'factor_exposures': self._get_factor_exposures()
                        })
                    
                    # Calculate risk metrics
                    if hasattr(model, 'calculate_risk_metrics'):
                        # Legacy interface
                        metrics = model.calculate_risk_metrics(**model_kwargs)
                    else:
                        # New interface with RiskModel base class
                        metrics = model.calculate(**model_kwargs)
                    
                    # Store results
                    self.risk_metrics['models'][model_name] = metrics
                    
                    # Update portfolio-level metrics
                    self._update_portfolio_metrics(model_name, metrics)
                    
                except Exception as e:
                    logger.exception(f"Error in {model.__class__.__name__} calculation: {e}")
            
            # Update correlation matrix
            self._update_correlations()
            
            logger.info(f"Updated risk metrics at {self.risk_metrics['timestamp']}")
            
        except Exception as e:
            logger.exception("Unexpected error in _update_risk_metrics")
            raise
            
    def _get_factor_returns(self) -> Dict[str, pd.Series]:
        """Get factor returns data."""
        # In a real implementation, this would fetch factor returns from a data provider
        # For now, return an empty dict which will trigger model defaults
        return {}
        
    def _get_factor_exposures(self) -> Dict[str, Dict[str, float]]:
        """Get factor exposures for positions."""
        # In a real implementation, this would fetch factor exposures from a model
        # For now, return an empty dict which will trigger model defaults
        return {}
        
    def _update_portfolio_metrics(self, model_name: str, metrics: Any):
        """Update portfolio-level metrics based on model results."""
        if not isinstance(metrics, dict):
            if hasattr(metrics, 'to_dict'):
                metrics = metrics.to_dict()
            else:
                return
                
        # Update portfolio beta if available
        if 'portfolio_beta' in metrics:
            self.portfolio_beta = metrics['portfolio_beta']
        
        # Store additional metrics
        if 'additional_metrics' in metrics:
            self.risk_metrics.setdefault('additional_metrics', {})[model_name] = metrics['additional_metrics']
    
    async def close(self):
        """Clean up resources."""
        self._stop_event.set()
        
        if hasattr(self, '_update_task') and not self._update_task.done():
            self._update_task.cancel()
            try:
                await self._update_task
            except asyncio.CancelledError:
                pass
        
        # Close market data provider if it has a close method
        if hasattr(self.market_data_provider, 'close'):
            await self.market_data_provider.close()
    
    # Add the rest of the existing methods here...
    # [Previous methods like add_position, remove_position, etc.]

# Example usage
async def example_usage():
    # Initialize with MT5 provider
    market_data_provider = MT5MarketDataProvider()
    await market_data_provider.connect()
    
    # Initialize exposure manager with risk models
    risk_models = [
        VaRModel(confidence_level=0.95),
        CVaRModel(confidence_level=0.95),
        FactorRiskModel()
    ]
    
    exposure_manager = ExposureManager(
        portfolio_value=1_000_000,
        market_data_provider=market_data_provider,
        risk_models=risk_models
    )
    
    try:
        # Add some positions
        aapl_position = Position(
            symbol="AAPL",
            quantity=100,
            price=150.0,
            asset_class=AssetClass.EQUITY,
            sector="Technology",
            region="US",
            beta=1.2
        )
        
        msft_position = Position(
            symbol="MSFT",
            quantity=50,
            price=300.0,
            asset_class=AssetClass.EQUITY,
            sector="Technology",
            region="US",
            beta=1.1
        )
        
        exposure_manager.add_position(aapl_position)
        exposure_manager.add_position(msft_position)
        
        # Wait for initial data load
        await asyncio.sleep(5)
        
        # Get risk report
        report = exposure_manager.get_risk_report()
        print("Risk Report:", report)
        
        # Keep running to receive updates
        while True:
            await asyncio.sleep(60)
            
    finally:
        await exposure_manager.close()

if __name__ == "__main__":
    asyncio.run(example_usage())
