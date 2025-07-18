"""
Enhanced VWAP (Volume-Weighted Average Price) Execution Strategy.

This module implements an advanced VWAP execution strategy that breaks up large orders into
smaller child orders to minimize market impact by targeting a dynamic percentage of volume
over time, with ML-based optimizations and real-time adjustments.
"""
from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, time as dt_time
from decimal import Decimal, ROUND_HALF_UP, ROUND_DOWN
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any, Union, Deque

import numpy as np
import pandas as pd
from scipy import stats

from core.execution.base import (
    ExecutionClient, 
    ExecutionParameters, 
    ExecutionResult,
    ExecutionStyle,
    ExecutionState,
    ExecutionReport
)
from core.market.data import MarketDataService, BarData, TickerData
from core.risk.manager import RiskManager
from core.ml.model_registry import ModelRegistry, ModelType
from core.trading.order import Order, OrderSide, OrderType, OrderStatus, TimeInForce
from core.utils.connection_pool import ConnectionPool

logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Market regime classification."""
    TRENDING_UP = auto()
    TRENDING_DOWN = auto()
    MEAN_REVERTING = auto()
    HIGH_VOLATILITY = auto()
    LOW_VOLATILITY = auto()
    
class ExecutionOptimizer:
    """Optimizes execution parameters based on market conditions."""
    
    def __init__(self, model_registry: Optional[ModelRegistry] = None):
        self.model_registry = model_registry
        self._market_regime: Optional[MarketRegime] = None
        self._volatility: float = 0.0
        self._liquidity_score: float = 1.0
        self._spread_ratio: float = 0.0
        
    async def analyze_market_conditions(
        self, 
        symbol: str, 
        market_data: MarketDataService,
        lookback_bars: int = 20
    ) -> None:
        """Analyze current market conditions."""
        try:
            # Get recent market data
            bars = await market_data.get_historical_bars(
                symbol=symbol,
                timeframe='1min',
                limit=lookback_bars
            )
            
            if bars.empty or len(bars) < 10:  # Minimum bars needed
                return
                
            # Calculate basic metrics
            returns = np.log(bars['close'] / bars['close'].shift(1)).dropna()
            self._volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Calculate spread ratio
            ticker = await market_data.get_ticker(symbol)
            if ticker and 'ask' in ticker and 'bid' in ticker and ticker['ask'] > 0:
                mid_price = (ticker['ask'] + ticker['bid']) / 2
                self._spread_ratio = (ticker['ask'] - ticker['bid']) / mid_price
            
            # Simple regime detection (can be enhanced with ML)
            if self._volatility > 0.3:
                self._market_regime = MarketRegime.HIGH_VOLATILITY
            elif self._volatility < 0.1:
                self._market_regime = MarketRegime.LOW_VOLATILITY
            else:
                # Check for trending vs mean-reverting
                z_score = stats.zscore(returns)
                hurst_exp = self._calculate_hurst_exponent(returns)
                
                if abs(z_score[-1]) > 2.0:
                    self._market_regime = (
                        MarketRegime.TRENDING_UP if returns.iloc[-1] > 0 
                        else MarketRegime.TRENDING_DOWN
                    )
                elif hurst_exp < 0.5:
                    self._market_regime = MarketRegime.MEAN_REVERTING
                else:
                    self._market_regime = None
                    
            # Update liquidity score (simplified)
            avg_volume = bars['volume'].mean()
            std_volume = bars['volume'].std()
            self._liquidity_score = min(1.0, avg_volume / (std_volume + 1e-6))
            
        except Exception as e:
            logger.warning(f"Error analyzing market conditions: {e}", exc_info=True)
    
    def _calculate_hurst_exponent(self, returns: pd.Series, max_lag: int = 20) -> float:
        """Calculate the Hurst exponent using R/S analysis."""
        lags = range(2, min(max_lag, len(returns)))
        tau = [np.std(np.subtract(returns[lag:].values, returns[:-lag].values)) 
               for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]
    
    def get_optimal_participation_rate(
        self,
        base_rate: float,
        order_side: OrderSide,
        order_size_ratio: float
    ) -> float:
        """Calculate optimal participation rate based on market conditions."""
        if not self._market_regime:
            return base_rate
            
        # Adjust based on market regime
        adjustment = 1.0
        
        if self._market_regime == MarketRegime.HIGH_VOLATILITY:
            # Be more aggressive in high vol to avoid missing the move
            adjustment *= 1.3
        elif self._market_regime in (MarketRegime.TRENDING_UP, MarketRegime.TRENDING_DOWN):
            # Be more aggressive in trends to minimize slippage
            adjustment *= 1.2
        elif self._market_regime == MarketRegime.MEAN_REVERTING:
            # Be more patient in mean-reverting markets
            adjustment *= 0.8
            
        # Adjust for order size relative to average volume
        size_adjustment = min(1.0, 1.0 / (order_size_ratio + 0.1))
        adjustment *= size_adjustment
        
        # Adjust for spread
        if self._spread_ratio > 0.001:  # 10bps
            spread_adjustment = 1.0 / (1.0 + self._spread_ratio * 100)
            adjustment *= spread_adjustment
            
        # Apply bounds
        adjusted_rate = base_rate * adjustment
        return max(0.01, min(0.5, adjusted_rate))  # Keep within reasonable bounds
    
    def get_limit_order_parameters(
        self,
        side: OrderSide,
        current_price: float,
        spread: float,
        volatility: float
    ) -> Tuple[float, float]:
        """Calculate optimal limit order price and size."""
        # Base aggressiveness based on volatility
        vol_scale = min(1.0, 0.2 / (volatility + 0.01))
        
        # Price improvement factor (how much better than mid price to set limit)
        if side == OrderSide.BUY:
            price_improvement = spread * (0.3 + 0.5 * vol_scale)  # 30-80% of spread
            limit_price = current_price - price_improvement
            limit_price = max(limit_price, current_price * 0.995)  # Max 0.5% below
        else:  # SELL
            price_improvement = spread * (0.3 + 0.5 * vol_scale)  # 30-80% of spread
            limit_price = current_price + price_improvement
            limit_price = min(limit_price, current_price * 1.005)  # Max 0.5% above
            
        # Size adjustment based on volatility and spread
        size_factor = 0.5 + 0.5 * vol_scale  # 50-100% of target size
        
        return limit_price, size_factor

@dataclass
class VWAPParameters:
    """Parameters specific to VWAP execution strategy."""
    # Time interval between child orders (in seconds)
    interval_seconds: int = 300  # 5 minutes
    
    # Maximum number of child orders to create
    max_child_orders: int = 12  # 1 hour of 5-min intervals
    
    # Base percentage of historical volume to target (0-1)
    volume_participation: float = 0.1  # 10% of volume
    
    # Dynamic participation rate adjustment bounds
    min_participation_rate: float = 0.05  # 5% minimum
    max_participation_rate: float = 0.50  # 50% maximum
    
    # Maximum slippage allowed from VWAP (in basis points)
    max_slippage_bps: int = 5  # 0.05%
    
    # Whether to use dynamic participation rate based on ML predictions
    use_dynamic_participation: bool = True
    
    # Whether to adjust for intraday volume profiles
    adjust_for_volume_profile: bool = True
    
    # Whether to use limit orders (True) or market orders (False)
    use_limit_orders: bool = True
    
    # Price tolerance for limit orders (in basis points from current price)
    limit_order_tolerance_bps: int = 2  # 0.02%
    
    # Whether to enable price improvement logic
    enable_price_improvement: bool = True
    
    # Maximum spread to trade at (in basis points)
    max_allowed_spread_bps: int = 20  # 0.2%
    
    # Time in force for child orders (in seconds)
    order_timeout_seconds: int = 60
    
    # Number of standard deviations for dynamic sizing (0 for no adjustment)
    volatility_adjustment_stdev: float = 1.0
    
    # Maximum order size as a multiple of average volume
    max_order_size_volume_ratio: float = 0.2  # 20% of average volume
    
    # Whether to use adaptive intervals based on liquidity
    use_adaptive_intervals: bool = True
    
    # Minimum interval between orders (in seconds)
    min_interval_seconds: int = 60  # 1 minute
    
    # Maximum interval between orders (in seconds)
    max_interval_seconds: int = 600  # 10 minutes
    
    # Whether to use machine learning for price prediction
    use_ml_price_prediction: bool = True
    
    # Whether to randomize order sizes within a range to avoid detection
    randomize_order_sizes: bool = True
    
    # Randomization range for order sizes (e.g., 0.8 means +/- 20%)
    order_size_randomization: float = 0.2  # +/- 20%
    
    # Whether to use volume profile to adjust order timing
    use_volume_profile_timing: bool = True
    
    # Whether to adjust for market impact
    adjust_for_market_impact: bool = True
    
    # Market impact coefficient (higher = more conservative)
    market_impact_coefficient: float = 1.0
    
    # Whether to use reinforcement learning for parameter optimization
    use_rl_optimization: bool = False
    
    # RL model update frequency (in number of trades)
    rl_update_frequency: int = 100
    
    # Whether to enable dark pool routing
    enable_dark_pool_routing: bool = False
    
    # Dark pool participation rate (0-1)
    dark_pool_participation: float = 0.3
    
    # Whether to enable smart order routing
    enable_smart_order_routing: bool = True
    
    # Maximum number of venues to route to
    max_venues: int = 3
    
    # Whether to enable anti-gaming logic
    enable_anti_gaming: bool = True
    
    # Anti-gaming: minimum time between orders (ms)
    min_order_interval_ms: int = 100
    
    # Anti-gaming: maximum order rate (orders/second)
    max_order_rate: float = 5.0
    
    # Whether to enable fill probability prediction
    enable_fill_probability: bool = True
    
    # Minimum fill probability for limit orders (0-1)
    min_fill_probability: float = 0.7
    
    # Whether to enable real-time adaptation
    enable_realtime_adaptation: bool = True
    
    # Adaptation sensitivity (0-1)
    adaptation_sensitivity: float = 0.5
    
    # Whether to enable execution analytics
    enable_execution_analytics: bool = True
    
    # Analytics update frequency (seconds)
    analytics_update_interval: int = 60
    
    # Whether to enable benchmark tracking
    enable_benchmark_tracking: bool = True
    
    # Benchmark type (VWAP, TWAP, Arrival, etc.)
    benchmark_type: str = "VWAP"
    
    # Whether to enable cost prediction
    enable_cost_prediction: bool = True
    
    # Cost prediction horizon (seconds)
    cost_prediction_horizon: int = 300  # 5 minutes
    
    # Whether to enable liquidity seeking
    enable_liquidity_seeking: bool = True
    
    # Liquidity seeking aggressiveness (0-1)
    liquidity_seeking_aggressiveness: float = 0.7
    
    # Whether to enable volatility scaling
    enable_volatility_scaling: bool = True
    
    # Volatility lookback period (in bars)
    volatility_lookback: int = 20
    
    # Whether to enable spread scaling
    enable_spread_scaling: bool = True
    
    # Spread scaling factor
    spread_scaling_factor: float = 1.0
    
    # Whether to enable time scaling
    enable_time_scaling: bool = True
    
    # Time scaling exponent (0=linear, 1=square root of time)
    time_scaling_exponent: float = 0.5


class VWAPExecutionClient(ExecutionClient):
    """
    Enhanced VWAP (Volume-Weighted Average Price) Execution Strategy.
    
    This advanced strategy breaks up large orders into smaller child orders that are executed
    over time, using ML and real-time market data to optimize execution performance.
    
    Key Features:
    - ML-based dynamic participation rate adjustment
    - Real-time market regime detection
    - Adaptive order sizing
    - Anti-gaming measures
    - Smart order routing
    - Dark pool integration
    - Comprehensive analytics and monitoring
    """
    
    def __init__(
        self,
        client_id: str,
        market_data: MarketDataService,
        risk_manager: RiskManager,
        model_registry: Optional[ModelRegistry] = None,
        default_params: Optional[ExecutionParameters] = None,
        vwap_params: Optional[VWAPParameters] = None,
        connection_pool: Optional[ConnectionPool] = None
    ):
        super().__init__(
            client_id=client_id,
            market_data=market_data,
            risk_manager=risk_manager,
            model_registry=model_registry,
            default_params=default_params
        )
        self.vwap_params = vwap_params or VWAPParameters()
        self._scheduled_tasks: Dict[str, asyncio.Task] = {}
        self._optimizer = ExecutionOptimizer(model_registry)
        self._connection_pool = connection_pool
        self._last_order_time: Dict[str, datetime] = {}
        self._order_count: Dict[str, int] = {}
        self._order_rate: Dict[str, float] = {}
        self._execution_stats: Dict[str, Any] = {
            'total_orders': 0,
            'filled_orders': 0,
            'cancelled_orders': 0,
            'rejected_orders': 0,
            'avg_fill_price': {},
            'avg_slippage': {},
            'volume_weighted_price': {},
            'participation_rate': {},
            'market_impact': {},
            'execution_duration': {}
        }
        self._market_impact_model = None
        self._fill_probability_model = None
        self._last_analytics_update = datetime.utcnow()
        
        # Initialize ML models if available
        self._init_ml_models()
    
    def _init_ml_models(self) -> None:
        """Initialize ML models for execution optimization."""
        if self.model_registry:
            try:
                # Load market impact model
                self._market_impact_model = self.model_registry.get_model(
                    ModelType.MARKET_IMPACT,
                    version='latest'
                )
                
                # Load fill probability model
                self._fill_probability_model = self.model_registry.get_model(
                    ModelType.FILL_PROBABILITY,
                    version='latest'
                )
                
                logger.info("Loaded ML models for execution optimization")
                
            except Exception as e:
                logger.warning(f"Failed to load ML models: {e}", exc_info=True)
    
    async def analyze_market_conditions(self, symbol: str) -> None:
        """Analyze current market conditions for optimization."""
        try:
            await self._optimizer.analyze_market_conditions(
                symbol=symbol,
                market_data=self.market_data,
                lookback_bars=min(50, self.vwap_params.volatility_lookback)
            )
            
            # Update execution parameters based on market conditions
            self._update_execution_parameters(symbol)
            
        except Exception as e:
            logger.warning(f"Error in market condition analysis: {e}", exc_info=True)
    
    def _update_execution_parameters(self, symbol: str) -> None:
        """Update execution parameters based on market conditions."""
        # Example: Adjust order size based on volatility
        if self.vwap_params.enable_volatility_scaling:
            volatility = self._optimizer._volatility
            if volatility > 0.3:  # High volatility
                self.vwap_params.volume_participation = min(
                    self.vwap_params.max_participation_rate,
                    self.vwap_params.volume_participation * 0.8  # Reduce size
                )
            elif volatility < 0.1:  # Low volatility
                self.vwap_params.volume_participation = min(
                    self.vwap_params.max_participation_rate,
                    self.vwap_params.volume_participation * 1.2  # Increase size
                )
        
        # Update analytics
        self._update_analytics(symbol)
    
    async def _execute_strategy(
        self,
        order: Order,
        params: ExecutionParameters,
        result: ExecutionResult
    ) -> ExecutionResult:
        """
        Execute the order using enhanced VWAP strategy with ML optimizations.
        
        This method:
        1. Analyzes current market conditions
        2. Calculates the historical volume profile
        3. Determines optimal participation rates using ML
        4. Schedules child orders with adaptive timing
        5. Monitors and adjusts execution in real-time
        6. Tracks performance against benchmarks
        """
        try:
            # Initialize execution context
            execution_start = datetime.utcnow()
            symbol = order.symbol
            
            # Analyze current market conditions
            await self.analyze_market_conditions(symbol)
            
            # Get historical volume profile with adaptive lookback
            volume_profile = await self._get_volume_profile(
                symbol,
                lookback_days=30 if self.vwap_params.adjust_for_volume_profile else 5
            )
            
            # Calculate target participation rates with ML optimization
            participation_rates = await self._calculate_optimal_participation_rates(
                symbol,
                order.quantity,
                volume_profile,
                params
            )
            
            # Apply anti-gaming measures
            if self.vwap_params.enable_anti_gaming:
                participation_rates = self._apply_anti_gaming_measures(
                    symbol,
                    participation_rates
                )
            
            # Schedule child orders with adaptive timing
            child_orders = await self._schedule_child_orders(
                order,
                participation_rates,
                params
            )
            
            # Initialize execution analytics
            self._init_execution_analytics(symbol, order, child_orders)
            
            # Update result with initial state
            result = result.update(
                metadata={
                    "strategy": "Enhanced VWAP",
                    "child_orders_created": len(child_orders),
                    "target_participation": self.vwap_params.volume_participation,
                    "use_dynamic_participation": self.vwap_params.use_dynamic_participation,
                    "use_limit_orders": self.vwap_params.use_limit_orders,
                    "market_regime": self._optimizer._market_regime.name if self._optimizer._market_regime else None,
                    "volatility": float(self._optimizer._volatility) if hasattr(self._optimizer, '_volatility') else None,
                    "liquidity_score": float(self._optimizer._liquidity_score) if hasattr(self._optimizer, '_liquidity_score') else None,
                    "start_time": execution_start.isoformat(),
                    **result.metadata
                }
            )
            
            # Monitor and adjust execution in real-time
            while not self._is_execution_complete(child_orders):
                # Check for completion
                if self._is_execution_complete(child_orders):
                    break
                
                # Update market conditions periodically
                if (datetime.utcnow() - getattr(self, '_last_market_update', datetime.min)).total_seconds() > 60:
                    await self.analyze_market_conditions(symbol)
                    self._last_market_update = datetime.utcnow()
                
                # Adjust orders based on market conditions
                if self.vwap_params.enable_realtime_adaptation:
                    await self._adapt_orders_in_flight(order, child_orders, params)
                
                # Update analytics
                if self.vwap_params.enable_execution_analytics:
                    self._update_execution_analytics(symbol, order, child_orders)
                
                # Sleep before next update
                await asyncio.sleep(1)
            
            # Calculate final execution metrics with detailed analysis
            result = await self._calculate_final_metrics(order, result, child_orders)
            
            # Log execution summary
            self._log_execution_summary(order, result, execution_start)
            
            # Update ML models with execution results
            if self.model_registry and self.vwap_params.use_rl_optimization:
                await self._update_ml_models(order, child_orders, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in VWAP execution: {str(e)}", exc_info=True)
            # Cancel all pending orders on error
            await self.cancel_all_orders()
            raise
    
    async def _get_volume_profile(
        self,
        symbol: str,
        lookback_days: int = 20,
        interval_minutes: int = 5,
        adjust_for_weekday: bool = True
    ) -> pd.DataFrame:
        """
        Get enhanced historical volume profile with ML-based adjustments.
        
        Args:
            symbol: Trading symbol
            lookback_days: Number of days of historical data to use
            interval_minutes: Time interval in minutes
            adjust_for_weekday: Whether to adjust for day-of-week patterns
            
        Returns:
            DataFrame with enhanced volume profile data
        """
        try:
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(days=lookback_days)
            
            # Get historical data with retry logic
            max_retries = 3
            historical_data = None
            
            for attempt in range(max_retries):
                try:
                    historical_data = await self.market_data.get_historical_bars(
                        symbol=symbol,
                        start_time=start_time,
                        end_time=end_time,
                        timeframe=f"{interval_minutes}min"
                    )
                    if not historical_data.empty:
                        break
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    await asyncio.sleep(1)  # Backoff before retry
            
            if historical_data is None or historical_data.empty:
                logger.warning(f"No historical data found for {symbol} after {max_retries} attempts")
                return self._get_default_volume_profile(interval_minutes)
            
            # Calculate time-based features
            historical_data['time_of_day'] = historical_data.index.time
            historical_data['day_of_week'] = historical_data.index.dayofweek
            historical_data['hour'] = historical_data.index.hour
            historical_data['minute'] = historical_data.index.minute
            
            # Calculate volume statistics
            volume_cols = ['time_of_day', 'volume']
            if adjust_for_weekday:
                volume_cols.insert(1, 'day_of_week')
            
            # Group by time of day (and optionally day of week)
            volume_profile = historical_data.groupby(volume_cols[:-1])['volume'].agg(
                ['mean', 'std', 'count', 'min', 'max', 'median']
            ).reset_index()
            
            # Calculate robust z-scores for outlier detection
            q1 = volume_profile['mean'].quantile(0.25)
            q3 = volume_profile['mean'].quantile(0.75)
            iqr = q3 - q1
            volume_profile['z_score'] = (volume_profile['mean'] - volume_profile['mean'].median()) / volume_profile['mean'].std()
            
            # Filter outliers using IQR
            volume_profile = volume_profile[
                (volume_profile['mean'] >= q1 - 1.5 * iqr) & 
                (volume_profile['mean'] <= q3 + 1.5 * iqr)
            ]
            
            # Calculate volume percentage and normalize
            total_volume = volume_profile['mean'].sum()
            volume_profile['volume_pct'] = volume_profile['mean'] / total_volume if total_volume > 0 else 0
            
            # Apply ML-based adjustments if available
            if self.model_registry and hasattr(self, '_volume_model'):
                try:
                    volume_profile = self._apply_volume_model(volume_profile, historical_data)
                except Exception as e:
                    logger.warning(f"Error applying volume model: {e}", exc_info=True)
            
            # Fill any missing time slots
            volume_profile = self._fill_missing_time_slots(volume_profile, interval_minutes)
            
            # Sort and reset index
            sort_cols = ['day_of_week'] if adjust_for_weekday else []
            sort_cols.append('time_of_day')
            volume_profile = volume_profile.sort_values(sort_cols).reset_index(drop=True)
            
            return volume_profile
            
        except Exception as e:
            logger.error(f"Error getting volume profile: {e}", exc_info=True)
            return self._get_default_volume_profile(interval_minutes)
    
    def _get_default_volume_profile(self, interval_minutes: int) -> pd.DataFrame:
        """Generate a default uniform volume profile when no data is available."""
        intervals_per_hour = 60 // interval_minutes
        time_slots = [
            (dt_time(h, m, 0), 1.0 / (24 * intervals_per_hour))
            for h in range(24)
            for m in range(0, 60, interval_minutes)
        ]
        
        return pd.DataFrame({
            'time_of_day': [t[0] for t in time_slots],
            'volume_pct': [t[1] for t in time_slots],
            'mean': [1.0] * len(time_slots),
            'std': [0.1] * len(time_slots),
            'count': [1] * len(time_slots),
            'min': [0.8] * len(time_slots),
            'max': [1.2] * len(time_slots),
            'median': [1.0] * len(time_slots),
            'z_score': [0.0] * len(time_slots)
        })
    
    async def _calculate_optimal_participation_rates(
        self,
        symbol: str,
        quantity: Decimal,
        volume_profile: pd.DataFrame,
        params: ExecutionParameters
    ) -> List[Dict[str, Any]]:
        """
        Calculate optimal participation rates using ML and market conditions.
        
        This method enhances the basic VWAP strategy with:
        - ML-based market impact prediction
        - Real-time liquidity analysis
        - Adaptive order sizing
        - Risk-aware execution
        
        Args:
            symbol: Trading symbol
            quantity: Total quantity to execute
            volume_profile: Enhanced volume profile with ML features
            params: Execution parameters
            
        Returns:
            List of dicts with participation details for each interval
        """
        try:
            if volume_profile.empty:
                return self._get_uniform_participation(quantity)
            
            # Get current market data
            ticker = await self.market_data.get_ticker(symbol)
            if not ticker or 'ask' not in ticker or 'bid' not in ticker:
                logger.warning(f"No valid ticker data for {symbol}, using uniform distribution")
                return self._get_uniform_participation(quantity)
            
            # Calculate order size ratio for participation rate adjustment
            avg_daily_volume = volume_profile['mean'].sum()
            order_size_ratio = float(quantity) / (avg_daily_volume + 1e-6)
            
            # Initialize intervals
            intervals = []
            remaining_qty = quantity
            total_intervals = min(len(volume_profile), self.vwap_params.max_child_orders)
            
            # Get current market conditions
            spread = (ticker['ask'] - ticker['bid']) / ((ticker['ask'] + ticker['bid']) / 2)
            volatility = self._optimizer._volatility if hasattr(self._optimizer, '_volatility') else 0.15
            
            for i in range(total_intervals):
                # Get base participation from volume profile
                vol_pct = volume_profile.iloc[i]['volume_pct']
                
                # Calculate base target quantity
                if i == total_intervals - 1:
                    # Last interval gets remaining quantity
                    target_qty = remaining_qty
                else:
                    target_qty = quantity * Decimal(str(vol_pct))
                    target_qty = min(target_qty, remaining_qty)
                
                # Apply dynamic participation adjustment
                if self.vwap_params.use_dynamic_participation:
                    # Get ML-optimized adjustment factor
                    adjustment_factor = self._get_ml_participation_adjustment(
                        symbol=symbol,
                        target_qty=target_qty,
                        vol_pct=vol_pct,
                        spread=spread,
                        volatility=volatility,
                        order_side=params.side,
                        time_of_day=volume_profile.iloc[i]['time_of_day']
                    )
                    
                    # Apply adjustment with bounds
                    adjusted_qty = target_qty * Decimal(str(adjustment_factor))
                    target_qty = max(
                        Decimal('0'),
                        min(adjusted_qty, remaining_qty)
                    )
                
                # Apply market impact model if available
                if self.vwap_params.adjust_for_market_impact and self._market_impact_model:
                    try:
                        # Predict market impact for this order size
                        impact = self._predict_market_impact(
                            symbol=symbol,
                            quantity=float(target_qty),
                            spread=spread,
                            volatility=volatility,
                            time_of_day=volume_profile.iloc[i]['time_of_day']
                        )
                        
                        # Adjust order size based on predicted impact
                        if impact > 0:
                            impact_factor = 1.0 / (1.0 + impact * self.vwap_params.market_impact_coefficient)
                            target_qty = target_qty * Decimal(str(impact_factor))
                    except Exception as e:
                        logger.warning(f"Error in market impact prediction: {e}", exc_info=True)
                
                # Ensure we don't exceed remaining quantity and respect minimums
                target_qty = max(Decimal('0'), min(target_qty, remaining_qty))
                if target_qty <= 0 and remaining_qty > 0 and i < total_intervals - 1:
                    # Skip this interval if quantity is too small (will be redistributed)
                    continue
                
                # Calculate max quantity based on available liquidity
                max_qty = self._calculate_max_order_size(
                    symbol=symbol,
                    time_of_day=volume_profile.iloc[i]['time_of_day'],
                    volume_profile=volume_profile.iloc[i],
                    current_spread=spread
                )
                
                # Apply randomization if enabled
                if self.vwap_params.randomize_order_sizes and target_qty > 0:
                    random_factor = 1.0 + random.uniform(
                        -self.vwap_params.order_size_randomization,
                        self.vwap_params.order_size_randomization
                    )
                    target_qty = target_qty * Decimal(str(random_factor))
                    target_qty = max(Decimal('0'), min(target_qty, remaining_qty, max_qty))
                
                intervals.append({
                    'target_pct': float(target_qty / quantity) if quantity > 0 else 0,
                    'target_qty': target_qty,
                    'max_qty': max_qty,
                    'start_time': None,  # Will be set when scheduling
                    'end_time': None,    # Will be set when scheduling
                    'original_vol_pct': vol_pct,
                    'adjusted_vol_pct': float(target_qty / quantity) if quantity > 0 else 0,
                    'market_impact': 0.0,  # Will be updated during execution
                    'fill_probability': 0.0  # Will be updated during execution
                })
                
                remaining_qty -= target_qty
                if remaining_qty <= 0:
                    break
            
            # Redistribute any remaining quantity
            if remaining_qty > 0 and intervals:
                self._redistribute_remaining_quantity(intervals, remaining_qty)
            
            return intervals
            
        except Exception as e:
            logger.error(f"Error calculating participation rates: {e}", exc_info=True)
            return self._get_uniform_participation(quantity)
    
    def _get_ml_participation_adjustment(
        self,
        symbol: str,
        target_qty: Decimal,
        vol_pct: float,
        spread: float,
        volatility: float,
        order_side: OrderSide,
        time_of_day: dt_time
    ) -> float:
        """Get ML-based adjustment factor for participation rate."""
        try:
            # Base adjustment from optimizer
            base_rate = self.vwap_params.volume_participation
            order_size_ratio = float(target_qty) / (self._get_avg_daily_volume(symbol) + 1e-6)
            
            # Get market-regime aware adjustment
            adjustment = self._optimizer.get_optimal_participation_rate(
                base_rate=1.0,  # Start with no adjustment
                order_side=order_side,
                order_size_ratio=order_size_ratio
            )
            
            # Apply ML model if available
            if self.model_registry and hasattr(self, '_participation_model'):
                try:
                    # Prepare features for ML model
                    features = {
                        'time_of_day': time_of_day.hour + time_of_day.minute / 60.0,
                        'vol_pct': vol_pct,
                        'spread': spread,
                        'volatility': volatility,
                        'order_size_ratio': order_size_ratio,
                        'is_buy': 1 if order_side == OrderSide.BUY else 0,
                        'day_of_week': datetime.utcnow().weekday(),
                        'market_regime': self._optimizer._market_regime.value if self._optimizer._market_regime else 0,
                        'liquidity_score': getattr(self._optimizer, '_liquidity_score', 1.0)
                    }
                    
                    # Get ML prediction
                    ml_adjustment = self._participation_model.predict(features)
                    
                    # Blend with base adjustment
                    adjustment = 0.7 * adjustment + 0.3 * ml_adjustment
                    
                except Exception as e:
                    logger.warning(f"Error in ML participation adjustment: {e}", exc_info=True)
            
            # Apply bounds
            return max(0.1, min(5.0, adjustment))
            
        except Exception as e:
            logger.warning(f"Error in participation adjustment: {e}", exc_info=True)
            return 1.0  # Fallback to no adjustment
    
    async def _schedule_child_orders(
        self,
        order: Order,
        participation_rates: List[Dict[str, Any]],
        params: ExecutionParameters
    ) -> List[Order]:
        """
        Schedule child orders with enhanced features:
        1. Applies market-aware timing with jitter
        2. Handles order type selection and parameter optimization
        3. Optimizes order types and parameters
        4. Handles scheduling with jitter
        
        Args:
            order: Parent order
            participation_rates: List of participation rates and quantities
            params: Execution parameters
            
        Returns:
            List of scheduled child orders
        """
        if not participation_rates:
            logger.warning("No participation rates provided for scheduling")
            return []

        child_orders = []
        total_intervals = len(participation_rates)
        symbol = order.symbol
        
        try:
            # Get current market conditions
            market_conditions = await self._get_market_conditions(symbol)
            
            # Calculate total duration and interval timing
            total_duration = (params.end_time - params.start_time).total_seconds()
            base_interval = total_duration / total_intervals
            
            # Schedule each order with adaptive timing
            for i, rate in enumerate(participation_rates):
                try:
                    # Calculate timing with jitter
                    jitter = self._calculate_jitter(i, total_intervals, market_conditions)
                    interval_start = params.start_time + timedelta(seconds=i * base_interval)
                    interval_end = params.start_time + timedelta(seconds=(i + 1) * base_interval)
                    
                    # Apply jitter to start time (up to 20% of interval)
                    max_jitter = base_interval * 0.2
                    jitter_seconds = jitter * max_jitter
                    scheduled_time = interval_start + timedelta(seconds=jitter_seconds)
                    
                    # Ensure we don't schedule in the past
                    now = datetime.utcnow()
                    if scheduled_time < now:
                        scheduled_time = now + timedelta(seconds=random.uniform(0.1, 1.0))
                    
                    # Update rate with timing info
                    rate['scheduled_time'] = scheduled_time
                    rate['interval_start'] = interval_start
                    rate['interval_end'] = interval_end
                    
                    # Create child order with optimized parameters
                    child_order = await self._create_optimized_child_order(
                        order=order,
                        rate=rate,
                        params=params,
                        market_conditions=market_conditions
                    )
                    
                    if not child_order:
                        continue
                        
                    # Calculate delay and schedule
                    delay = (scheduled_time - now).total_seconds()
                    if delay <= 0:
                        # Execute immediately if already past scheduled time
                        asyncio.create_task(
                            self._execute_child_order(
                                child_order=child_order,
                                rate=rate,
                                params=params
                            )
                        )
                    else:
                        # Schedule for future execution
                        task = asyncio.create_task(
                            self._execute_child_order_with_delay(
                                child_order=child_order,
                                delay=delay,
                                rate=rate,
                                params=params
                            )
                        )
                        self._scheduled_tasks[child_order.id] = task
                    
                    # Track order
                    child_orders.append(child_order)
                    
                    # Update market conditions for next order
                    market_conditions = await self._update_market_conditions(
                        market_conditions,
                        child_order,
                        rate
                    )
                    
                except Exception as e:
                    logger.error(f"Error scheduling child order {i+1}/{total_intervals}: {e}", 
                               exc_info=True)
                    continue
            
            logger.info(f"Scheduled {len(child_orders)}/{total_intervals} child orders for {symbol}")
            return child_orders
            
        except Exception as e:
            logger.error(f"Error in schedule_child_orders: {e}", exc_info=True)
            # Cancel any scheduled tasks on error
            await self.cancel_all_orders()
            raise

    async def _execute_child_order_with_delay(
        self,
        child_order: Order,
        delay: float,
        rate: Dict[str, Any],
        params: ExecutionParameters
    ) -> None:
        """
        Execute a child order after a delay with enhanced error handling and monitoring.
        
        Args:
            child_order: The child order to execute
            delay: Delay in seconds before execution
            rate: Participation rate details
            params: Execution parameters
        """
        try:
            # Wait for delay with cancellation support
            try:
                await asyncio.wait_for(
                    self._cancellation_event.wait(),
                    timeout=delay
                )
                # If we get here, cancellation was requested
                logger.info(f"Order {child_order.id} execution cancelled during delay")
                child_order.status = OrderStatus.CANCELLED
                await self._notify_order_update(child_order)
                return
                
            except asyncio.TimeoutError:
                # This is expected - proceed with order execution
                pass
                
            # Check if order was explicitly cancelled
            if child_order.id in self._cancelled_orders:
                logger.info(f"Order {child_order.id} was cancelled, skipping execution")
                child_order.status = OrderStatus.CANCELLED
                await self._notify_order_update(child_order)
                return
                
            # Get current market conditions
            market_conditions = await self._get_market_conditions(child_order.symbol)
            
            # Validate market conditions
            if not self._validate_market_conditions(market_conditions, child_order):
                logger.warning(f"Market conditions not favorable for order {child_order.id}")
                
                # Apply adaptive response - could be delay, requeue, or adjust
                requeue_delay = self._calculate_requeue_delay(market_conditions)
                if requeue_delay > 0:
                    logger.info(f"Requeuing order {child_order.id} with {requeue_delay}s delay")
                    asyncio.create_task(
                        self._execute_child_order_with_delay(
                            child_order=child_order,
                            delay=requeue_delay,
                            rate=rate,
                            params=params
                        )
                    )
                    return
                
                # If not requeuing, mark as rejected
                child_order.status = OrderStatus.REJECTED
                child_order.error_message = "Unfavorable market conditions"
                await self._notify_order_update(child_order)
                return
            
            # Apply price improvement if enabled
            if self.vwap_params.enable_price_improvement:
                price_improvement = self._calculate_price_improvement(
                    child_order,
                    market_conditions
                )
                if price_improvement:
                    child_order = self._apply_price_improvement(
                        child_order,
                        price_improvement
                    )
            
            # Execute the order with retry logic
            max_retries = 3
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    # Execute the order
                    result = await self._execute_child_order(child_order, rate, params)
                    
                    # Update order with execution details
                    child_order.status = result.status
                    child_order.filled_quantity = result.filled_quantity
                    child_order.avg_fill_price = result.avg_fill_price
                    
                    # Prepare execution result for impact analysis
                    execution_result = {
                        'avg_fill_price': float(child_order.avg_fill_price) if child_order.avg_fill_price else None,
                        'filled_quantity': float(child_order.filled_quantity),
                        'status': child_order.status,
                        'participation_rate': rate.get('target_pct', 0),
                        'execution_time': time.time()
                    }
                    
                    # Calculate and log market impact
                    impact_metrics = self._calculate_market_impact(
                        child_order,
                        market_conditions,
                        execution_result
                    )
                    
                    # Update market impact model asynchronously
                    if self._market_impact_model:
                        asyncio.create_task(
                            self._update_market_impact_model(
                                child_order,
                                market_conditions,
                                execution_result
                            )
                        )
                    
                    # Log execution with impact metrics
                    self._log_order_execution(
                        child_order, 
                        {**execution_result, **impact_metrics}
                    )
                    
                    # Update execution stats
                    self._update_execution_stats(child_order, impact_metrics)
                    
                    return
                    
                except (asyncio.TimeoutError, ConnectionError) as e:
                    last_error = e
                    if attempt == max_retries - 1:
                        break
                        
                    # Exponential backoff
                    await asyncio.sleep(2 ** attempt)
            
            # If we get here, all retries failed
            raise last_error if last_error else Exception("Failed to execute order")
            
        except asyncio.CancelledError:
            logger.info(f"Order {child_order.id} execution was cancelled")
            child_order.status = OrderStatus.CANCELLED
            await self._notify_order_update(child_order)
            raise
            
        except Exception as e:
            logger.error(f"Error executing child order {child_order.id}: {e}", exc_info=True)
            child_order.status = OrderStatus.REJECTED
            child_order.error_message = str(e)
            await self._notify_order_update(child_order)
            
            # Update error analytics
            self._update_order_analytics(child_order, 'error')
            
            # Notify monitoring system
            await self._alert_order_failure(child_order, str(e))
    
    def _calculate_market_impact(
        self,
        order: Order,
        market_data: Dict[str, Any],
        execution_result: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Calculate market impact metrics for an executed order.
        
        Args:
            order: The executed order
            market_data: Market data at time of execution
            execution_result: Execution details including fills
            
        Returns:
            Dictionary containing market impact metrics
        """
        impact_metrics = {
            'pre_trade_impact': 0.0,
            'post_trade_impact': 0.0,
            'permanent_impact': 0.0,
            'temporary_impact': 0.0,
            'implementation_shortfall': 0.0
        }
        
        try:
            # Get reference price (e.g., mid-price at order submission)
            ref_price = (market_data.get('bid', 0) + market_data.get('ask', 0)) / 2
            if ref_price <= 0:
                return impact_metrics
                
            # Calculate pre-trade impact (price movement before execution)
            arrival_price = market_data.get('arrival_price', ref_price)
            impact_metrics['pre_trade_impact'] = (
                (arrival_price - ref_price) / ref_price * 10000  # in bps
            )
            
            # Calculate post-trade impact (immediate price movement)
            if execution_result.get('avg_fill_price'):
                impact_metrics['post_trade_impact'] = (
                    (execution_result['avg_fill_price'] - arrival_price) / arrival_price * 10000
                )
                
                # Calculate temporary impact (immediate liquidity cost)
                impact_metrics['temporary_impact'] = (
                    (execution_result['avg_fill_price'] - arrival_price) / arrival_price * 10000
                )
                
                # Calculate implementation shortfall (total cost of execution)
                if order.price and order.price > 0:
                    impact_metrics['implementation_shortfall'] = (
                        (execution_result['avg_fill_price'] - order.price) / order.price * 10000
                    )
            
            return impact_metrics
            
        except Exception as e:
            logger.error(f"Error calculating market impact: {e}", exc_info=True)
            return impact_metrics
    
    async def _update_market_impact_model(
        self, 
        order: Order,
        market_data: Dict[str, Any],
        execution_result: Dict[str, Any]
    ) -> None:
        """
        Update the market impact model with new execution data.
        
        Args:
            order: The executed order
            market_data: Market data at time of execution
            execution_result: Execution details including fills
        """
        try:
            if not self._market_impact_model:
                return
                
            # Calculate market impact metrics
            impact_metrics = self._calculate_market_impact(order, market_data, execution_result)
            
            # Prepare features for model update
            features = {
                'symbol': order.symbol,
                'side': order.side.value,
                'quantity': float(order.quantity),
                'participation_rate': execution_result.get('participation_rate', 0),
                'volatility': market_data.get('volatility', 0),
                'spread': market_data.get('spread', 0),
                'liquidity': market_data.get('liquidity_score', 1.0),
                'time_of_day': datetime.utcnow().time().strftime('%H%M'),
                **impact_metrics
            }
            
            # Update model with new data
            await self._market_impact_model.update(features)
            
        except Exception as e:
            logger.error(f"Error updating market impact model: {e}", exc_info=True)
    
    async def _execute_child_order(
        self,
        child_order: Order,
        rate: Dict[str, Any],
        params: ExecutionParameters
    ) -> Order:
        """
        Execute a child order with optimized execution parameters.
        
        This method:
        1. Gets current market data
        2. Calculates optimal execution parameters
        3. Submits the order
        4. Tracks market impact
        5. Updates the market impact model
        
        Args:
            child_order: The child order to execute
            rate: Participation rate details
            params: Execution parameters
            
        Returns:
            The executed order with updated status
        """
        try:
            # Get current market data
            market_data = await self._get_market_data(child_order.symbol)
            
            # Store arrival price for impact calculation
            market_data['arrival_price'] = (market_data.get('bid', 0) + market_data.get('ask', 0)) / 2
            
            # Calculate execution parameters
            exec_params = await self._calculate_execution_parameters(
                child_order,
                market_data,
                rate,
                params
            )
            
            # Apply execution parameters to order
            child_order = self._apply_execution_parameters(child_order, exec_params)
            
            # Execute with retry logic
            max_retries = 3
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    # Submit the order
                    result = await self._submit_order(child_order)
                    
                    # Update order with execution details
                    child_order.status = result.status
                    child_order.filled_quantity = result.filled_quantity
                    child_order.avg_fill_price = result.avg_fill_price
                    
                    # Prepare execution result for impact analysis
                    execution_result = {
                        'avg_fill_price': float(child_order.avg_fill_price) if child_order.avg_fill_price else None,
                        'filled_quantity': float(child_order.filled_quantity),
                        'status': child_order.status,
                        'participation_rate': rate.get('target_pct', 0),
                        'execution_time': time.time()
                    }
                    
                    # Calculate and log market impact
                    impact_metrics = self._calculate_market_impact(
                        child_order,
                        market_data,
                        execution_result
                    )
                    
                    # Update market impact model asynchronously
                    if self._market_impact_model:
                        asyncio.create_task(
                            self._update_market_impact_model(
                                child_order,
                                market_data,
                                execution_result
                            )
                        )
                    
                    # Log execution with impact metrics
                    self._log_order_execution(
                        child_order, 
                        {**exec_params, **impact_metrics}
                    )
                    
                    # Update execution stats
                    self._update_execution_stats(child_order, impact_metrics)
                    
                    return child_order
                    
                except (asyncio.TimeoutError, ConnectionError) as e:
                    last_error = e
                    if attempt == max_retries - 1:
                        break
                        
                    # Exponential backoff
                    await asyncio.sleep(2 ** attempt)
            
            # If we get here, all retries failed
            raise last_error if last_error else Exception("Failed to execute order")
            
        except Exception as e:
            logger.error(f"Error in _execute_child_order: {e}", exc_info=True)
            child_order.status = OrderStatus.REJECTED
            child_order.error_message = str(e)
            await self._notify_order_update(child_order)
            return child_order

    async def _calculate_execution_parameters(
        self,
        child_order: Order,
        market_data: Dict[str, Any],
        rate: Dict[str, Any],
        params: ExecutionParameters
    ) -> Dict[str, Any]:
        # ... (rest of the method remains the same)
            market_data: Current market data including order book
            rate: Participation rate details
            params: Execution parameters
            
        Returns:
            Dictionary of execution parameters
        """
        symbol = child_order.symbol
        side = child_order.side
        quantity = child_order.quantity
        
        # Get current market prices and depth
        best_bid = Decimal(str(market_data.get('bid', 0)))
        best_ask = Decimal(str(market_data.get('ask', 0)))
        mid_price = (best_bid + best_ask) / 2
        spread = (best_ask - best_bid) / mid_price if mid_price > 0 else Decimal('0')
        
        # Initialize execution parameters
        exec_params = {
            'order_type': OrderType.LIMIT,
            'time_in_force': f"GTC,{self.vwap_params.order_timeout_seconds}s",
            'exec_inst': [],
            'display_quantity': None,
            'price_improvement': Decimal('0'),
            'urgency': rate.get('urgency', 'normal'),
            'participation_rate': rate.get('target_pct', 0)
        }
        
        # Determine order type based on market conditions and order size
        if self.vwap_params.adaptive_order_types:
            # Analyze order book depth and liquidity
            order_book_depth = self._analyze_order_book(
                market_data.get('bids', []),
                market_data.get('asks', []),
                quantity
            )
            
            # Select order type based on market impact and urgency
            if order_book_depth['liquidity_imbalance'] > 0.7:
                # High imbalance - use more aggressive order types
                if spread < Decimal('0.001'):  # 0.1%
                    exec_params['order_type'] = OrderType.LIMIT
                    exec_params['price_improvement'] = Decimal('0.0001')  # 0.01%
                else:
                    exec_params['order_type'] = OrderType.MARKET
            else:
                # More balanced market - can use limit orders
                if spread < Decimal('0.0005'):  # 0.05%
                    exec_params['order_type'] = OrderType.PEGGED
                    exec_params['peg_offset'] = Decimal('-0.0001') if side == OrderSide.BUY else Decimal('0.0001')
                else:
                    exec_params['order_type'] = OrderType.LIMIT
                    exec_params['price_improvement'] = spread * Decimal('0.2')  # 20% of spread
        
        # Calculate limit price if needed
        if exec_params['order_type'] == OrderType.LIMIT:
            if side == OrderSide.BUY:
                reference_price = best_ask
                price_improvement = reference_price * exec_params['price_improvement']
                limit_price = reference_price - price_improvement
                limit_price = max(limit_price, best_bid)  # Don't cross the spread
            else:  # Sell
                reference_price = best_bid
                price_improvement = reference_price * exec_params['price_improvement']
                limit_price = reference_price + price_improvement
                limit_price = min(limit_price, best_ask)  # Don't cross the spread
            
            # Round to tick size
            tick_size = Decimal(str(market_data.get('tick_size', '0.01')))
            if tick_size > 0:
                limit_price = (limit_price / tick_size).quantize(
                    Decimal('1.'), 
                    rounding=ROUND_HALF_UP
                ) * tick_size
            
            exec_params['price'] = limit_price
        
        # Set display quantity for iceberg orders
        if self.vwap_params.use_iceberg_orders and quantity > self.vwap_params.min_iceberg_size:
            # Calculate display quantity based on average trade size and current volume
            avg_trade_size = self._calculate_average_trade_size(market_data.get('recent_trades', []))
            if avg_trade_size > 0:
                display_qty = min(
                    max(quantity * Decimal('0.1'), avg_trade_size * Decimal('2')),  # 10% or 2x avg trade size
                    quantity * Decimal('0.3')  # But no more than 30% of order size
                )
                exec_params['display_quantity'] = display_qty.quantize(Decimal('0.00000001'))
        
        # Add execution instructions
        if self.vwap_params.reduce_market_impact:
            exec_params['exec_inst'].extend(['REDUCE_ONLY', 'AVOID_MARKET_IMPACT'])
        
        if self.vwap_params.hide_liquidity:
            exec_params['exec_inst'].append('HIDDEN')
        
        return exec_params
    
    def _calculate_adaptive_order_size(
        self,
        order: Order,
        market_data: Dict[str, Any],
        target_participation: float,
        max_impact_bps: float = 10.0,
        min_size_ratio: float = 0.1,
        max_iterations: int = 10
    ) -> Decimal:
        """
        Calculate the optimal order size considering predicted market impact.
        
        This method uses the market impact model to predict the impact of different
        order sizes and adjusts the size to stay within the specified impact threshold.
        
        Args:
            order: The order being executed
            market_data: Current market data
            target_participation: Target participation rate (0-1)
            max_impact_bps: Maximum allowed market impact in basis points
            min_size_ratio: Minimum size as a ratio of the original order size
            max_iterations: Maximum number of iterations for binary search
            
        Returns:
            Decimal: The optimal order size considering market impact
        """
        if not self._market_impact_model:
            return order.quantity
            
        original_size = float(order.quantity)
        min_size = original_size * min_size_ratio
        current_size = original_size
        
        # Get current market metrics
        mid_price = (market_data.get('bid', 0) + market_data.get('ask', 0)) / 2
        if mid_price <= 0:
            return Decimal(str(original_size))
            
        # Prepare base features for prediction
        base_features = {
            'symbol': order.symbol,
            'side': order.side.value,
            'volatility': market_data.get('volatility', 0.02),
            'spread': market_data.get('spread', 0.0),
            'liquidity': market_data.get('liquidity_score', 1.0),
            'time_of_day': datetime.utcnow().time().strftime('%H%M'),
            'participation_rate': target_participation
        }
        
        # Binary search for optimal size
        low = min_size
        high = original_size
        best_size = high
        
        for _ in range(max_iterations):
            if high - low < min_size * 0.01:  # 1% tolerance
                break
                
            mid = (low + high) / 2
            
            # Predict impact for current size
            features = {
                **base_features,
                'quantity': mid,
                'participation_rate': target_participation * (mid / original_size)
            }
            
            # Get predicted impact in bps
            predicted_impact = asyncio.get_event_loop().run_until_complete(
                self._market_impact_model.predict(features)
            )
            
            if predicted_impact <= max_impact_bps:
                # Can potentially use a larger size
                best_size = mid
                low = mid
            else:
                # Need to reduce size
                high = mid
        
        # Apply config limits
        max_allowed = float(self.config.get('max_order_size', float('inf')))
        min_allowed = float(self.config.get('min_order_size', 0.0))
        
        final_size = max(min(best_size, max_allowed), min_allowed)
        
        # Round to appropriate decimal places
        step_size = float(self.config.get('step_size', 0.000001))
        if step_size > 0:
            final_size = round(final_size / step_size) * step_size
            
        return Decimal(str(final_size))

    async def _calculate_final_metrics(
        self,
        parent_order: Order,
        result: ExecutionResult,
        child_orders: List[Order]
    ) -> ExecutionResult:
        """
        Calculate comprehensive execution metrics and analytics.
        
        This enhanced version provides detailed performance analysis:
        - Execution quality metrics (VWAP, TWAP, implementation shortfall)
        - Market impact analysis
        - Cost analysis (explicit and implicit costs)
        - Performance attribution
        - Benchmark comparison
        - Risk metrics
        
        Args:
            parent_order: The original parent order
            result: Current execution result
            child_orders: List of all child orders
            
        Returns:
            Updated execution result with comprehensive metrics
        """
        try:
            # Initialize metrics
            metrics = {
                'execution': {},
                'costs': {},
                'performance': {},
                'risk': {},
                'benchmark': {}
            }
            
            # Calculate basic execution metrics
            filled_orders = [o for o in child_orders if o.status == OrderStatus.FILLED]
            filled_qty = sum(o.filled_quantity for o in filled_orders)
            total_qty = parent_order.quantity
            fill_rate = float(filled_qty / total_qty) if total_qty > 0 else 0.0
            
            # Calculate VWAP and TWAP
            vwap_numerator = Decimal('0')
            twap_numerator = Decimal('0')
            total_notional = Decimal('0')
            total_fees = Decimal('0')
            execution_times = []
            
            for order in filled_orders:
                if order.filled_quantity > 0 and order.avg_fill_price > 0:
                    vwap_numerator += order.filled_quantity * order.avg_fill_price
                    twap_numerator += order.filled_quantity * Decimal(str(order.metadata.get('mid_price', 0)))
                    total_notional += order.filled_quantity * order.avg_fill_price
                    total_fees += order.fees or Decimal('0')
                    
                    if 'executed_at' in order.metadata:
                        execution_times.append(order.metadata['executed_at'])
            
            vwap_price = vwap_numerator / filled_qty if filled_qty > 0 else Decimal('0')
            twap_price = twap_numerator / filled_qty if filled_qty > 0 else Decimal('0')
            
            # Calculate arrival and decision prices
            arrival_price = Decimal(str(self.market_data.get_arrival_price(
                parent_order.symbol, 
                parent_order.side
            )))
            
            decision_price = Decimal(str(self.market_data.get_decision_price(
                parent_order.symbol,
                parent_order.side,
                parent_order.created_at
            )))
            
            # Calculate implementation shortfall
            if parent_order.side == OrderSide.BUY and decision_price > 0:
                is_perf = (vwap_price / decision_price - 1) * Decimal('10000')
            elif parent_order.side == OrderSide.SELL and decision_price > 0:
                is_perf = (decision_price / vwap_price - 1) * Decimal('10000')
            else:
                is_perf = Decimal('0')
            
            # Calculate market impact and timing risk
            market_impact = self._calculate_market_impact_metrics(
                child_orders=filled_orders,
                arrival_price=arrival_price,
                vwap_price=vwap_price,
                twap_price=twap_price
            )
            
            # Calculate costs
            costs = self._calculate_execution_costs(
                child_orders=filled_orders,
                arrival_price=arrival_price,
                vwap_price=vwap_price,
                total_notional=total_notional
            )
            
            # Calculate performance metrics
            performance = self._calculate_performance_metrics(
                parent_order=parent_order,
                child_orders=filled_orders,
                vwap_price=vwap_price,
                arrival_price=arrival_price,
                decision_price=decision_price,
                total_fees=total_fees,
                total_notional=total_notional
            )
            
            # Calculate risk metrics
            risk_metrics = self._calculate_risk_metrics(
                child_orders=filled_orders,
                symbol=parent_order.symbol,
                side=parent_order.side
            )
            
            # Prepare final metrics
            metrics.update({
                'execution': {
                    'vwap': float(vwap_price),
                    'twap': float(twap_price),
                    'arrival_price': float(arrival_price),
                    'decision_price': float(decision_price),
                    'fill_rate': fill_rate,
                    'total_orders': len(child_orders),
                    'filled_orders': len(filled_orders),
                    'partially_filled_orders': len([o for o in child_orders 
                                                  if o.status == OrderStatus.PARTIALLY_FILLED]),
                    'cancelled_orders': len([o for o in child_orders 
                                           if o.status == OrderStatus.CANCELLED]),
                    'rejected_orders': len([o for o in child_orders 
                                          if o.status == OrderStatus.REJECTED]),
                    'total_quantity': float(total_qty),
                    'filled_quantity': float(filled_qty),
                    'remaining_quantity': float(max(Decimal('0'), total_qty - filled_qty)),
                    'total_notional': float(total_notional),
                    'avg_trade_size': float(filled_qty / len(filled_orders)) if filled_orders else 0.0,
                    'execution_start': min((o.metadata.get('created_at') for o in child_orders 
                                          if o.metadata.get('created_at')), default=None),
                    'execution_end': max((o.metadata.get('last_updated') for o in child_orders 
                                        if o.metadata.get('last_updated')), default=None),
                    'duration_seconds': (max((o.metadata.get('last_updated') for o in child_orders 
                                           if o.metadata.get('last_updated')), default=datetime.utcnow()) - 
                                       min((o.metadata.get('created_at') for o in child_orders 
                                           if o.metadata.get('created_at')), default=datetime.utcnow())).total_seconds()
                },
                'costs': costs,
                'market_impact': market_impact,
                'performance': performance,
                'risk': risk_metrics,
                'metadata': {
                    'strategy': 'VWAP',
                    'version': '1.0',
                    'calculated_at': datetime.utcnow().isoformat(),
                    'parameters': self.vwap_params.dict()
                }
            })
            
            # Update the execution result
            result = result.update(
                filled_quantity=filled_qty,
                avg_fill_price=vwap_price,
                fees=total_fees,
                status=ExecutionState.COMPLETED if fill_rate >= 0.99 else 
                      (ExecutionState.PARTIALLY_FILLED if fill_rate > 0 else ExecutionState.FAILED),
                metadata={
                    **result.metadata,
                    'metrics': metrics,
                    'analysis': self._generate_execution_analysis(metrics)
                }
            )
            
            # Log the execution summary
            self._log_execution_summary(parent_order, result)
            
            # Update ML models with execution results
            if self.model_registry and self.vwap_params.use_rl_optimization:
                await self._update_ml_models(parent_order, child_orders, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error calculating final metrics: {e}", exc_info=True)
            # Return basic metrics if detailed calculation fails
            return result.update(
                status=ExecutionState.FAILED,
                error_message=f"Error calculating metrics: {str(e)}",
                metadata={
                    **result.metadata,
                    'metrics_error': str(e)
                }
            )
    
    async def cancel_all_orders(self) -> None:
        """Cancel all pending child orders."""
        for task_name, task in list(self._scheduled_tasks.items()):
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    logger.info(f"Cancelled VWAP child order: {task_name}")
                except Exception as e:
                    logger.error(f"Error cancelling VWAP child order {task_name}: {e}")
        
        self._scheduled_tasks.clear()
    
    async def close(self) -> None:
        """Clean up resources."""
        await self.cancel_all_orders()
        await super().close()
