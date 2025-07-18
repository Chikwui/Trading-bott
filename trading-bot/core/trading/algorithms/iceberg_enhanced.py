"""
Enhanced Iceberg Order Execution Algorithm

This module implements an advanced Iceberg order execution strategy with:
- Sophisticated anti-detection mechanisms
- Advanced risk management integration
- Real-time PnL optimization
- Comprehensive monitoring and alerting
"""
import asyncio
import logging
import random
import time
from decimal import Decimal, ROUND_DOWN
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque, defaultdict
from datetime import datetime, timedelta, timezone
import numpy as np
from dataclasses import dataclass, field
from enum import Enum, auto

from ..order_types import (
    Order, OrderSide, OrderType, OrderStatus, TimeInForce, ExecutionReport
)
from .base import ExecutionAlgorithm, ExecutionResult
from ..risk_models import (
    RiskAssessment,
    RiskModelType,
    calculate_var,
    calculate_cvar,
    calculate_liquidity_impact
)
from ..monitoring import (
    ExecutionMetrics,
    LatencyMetrics,
    PerformanceMetrics,
    AlertManager
)

logger = logging.getLogger(__name__)

class OrderFlowPattern(Enum):
    """Different order flow patterns to avoid detection."""
    STEADY = auto()
    BURST = auto()
    DECREASING = auto()
    RANDOM = auto()
    MOMENTUM = auto()
    REVERSAL = auto()
    LIQUIDITY_ABSORPTION = auto()
    STOP_HUNTING = auto()

@dataclass
class StealthConfig:
    """Configuration for stealth and anti-detection features."""
    # Order size randomization
    size_variation_pct: float = 0.2  # ±20% size variation
    min_size_change: Decimal = Decimal('0.1')
    
    # Timing randomization
    base_refresh_interval: float = 5.0  # seconds
    refresh_jitter_pct: float = 0.4  # ±40% jitter
    
    # Order flow patterns
    pattern_rotation_interval: int = 30  # minutes
    current_pattern: OrderFlowPattern = OrderFlowPattern.STEADY
    
    # Anti-detection
    max_consecutive_same_side: int = 3
    max_same_price_level_orders: int = 2
    
    # Price improvement
    min_price_improvement_bps: int = 1  # 0.01%
    max_price_improvement_bps: int = 5  # 0.05%
    
    # New advanced anti-detection parameters
    order_split_count: int = 3  # Number of orders to split into
    max_consecutive_orders: int = 5  # Max orders in same direction before flipping
    min_order_interval: float = 0.5  # Min seconds between orders
    max_order_interval: float = 10.0  # Max seconds between orders
    spoofing_probability: float = 0.1  # 10% chance to place spoof orders
    spoofing_size_multiplier: float = 2.0  # Size of spoof orders relative to real ones
    
    # Volume profile matching
    volume_profile_lookback: int = 20  # Number of periods to analyze
    volume_profile_weight: float = 0.7  # Weight for volume profile matching
    
    # Anti-pattern detection
    anomaly_threshold: float = 3.0  # Standard deviations for anomaly detection
    anomaly_cooldown: int = 300  # Seconds to wait after detecting surveillance
    
    # Order type distribution (percentages should sum to 100)
    order_type_distribution: Dict[str, float] = field(
        default_factory=lambda: {
            'limit': 70.0,
            'post_only': 20.0,
            'ioc': 10.0,
            'fok': 0.0  # Use with caution
        }
    )

@dataclass
class RiskParameters:
    """Risk management parameters."""
    max_position_pct: Decimal = Decimal('0.05')  # 5% of portfolio
    max_daily_loss_pct: Decimal = Decimal('0.02')  # 2% daily loss limit
    var_confidence: float = 0.99  # 99% VaR
    cvar_alpha: float = 0.95  # 95% CVaR
    max_slippage_bps: int = 10  # 0.1%
    max_orderbook_impact_bps: int = 5  # 0.05%

class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    LIQUIDITY_CRISIS = "liquidity_crisis"

class MarketMicrostructure:
    """Market microstructure analysis."""
    def __init__(self):
        self.spread: float = 0.0
        self.order_book_imbalance: float = 0.0
        self.trade_flow_imbalance: float = 0.0
        self.volatility: float = 0.0
        self.volume_profile: Dict[float, float] = {}
        self.support_resistance_levels: List[float] = []
        self.market_regime: MarketRegime = MarketRegime.RANGING
        self.timestamp: datetime = datetime.now(timezone.utc)

@dataclass
class PatternRecognitionResult:
    pattern_type: OrderFlowPattern
    confidence: float
    expected_duration: timedelta
    expected_impact: float
    recommended_actions: List[str]
    features: Dict[str, float]

@dataclass
class MLModelConfig:
    model_path: str = "models/iceberg_ml"
    retrain_interval: int = 3600  # seconds
    features: List[str] = ['spread', 'order_book_imbalance', 'trade_flow_imbalance',
        'volatility', 'volume_imbalance', 'price_momentum',
        'liquidity_score', 'adverse_selection_risk']
    target_metric: str = 'execution_quality_score'
    model_params: Dict[str, Any] = {'n_estimators': 100,
        'max_depth': 5, 'learning_rate': 0.1, 'subsample': 0.8}

class AdvancedPatternRecognizer:
    """Advanced pattern recognition using ensemble of ML models and deep learning."""
    
    def __init__(self):
        self.models = {
            'lstm': self._initialize_lstm_model(),
            'transformer': self._initialize_transformer_model(),
            'xgboost': self._initialize_xgboost_model(),
            'graph_attention': self._initialize_graph_attention_model(),
            'temporal_conv': self._initialize_temporal_conv_model(),
            'isolation_forest': IsolationForest(contamination=0.1, random_state=42),
            'autoencoder': self._initialize_autoencoder()
        }
        self.scaler = RobustScaler()
        self.pca = IncrementalPCA(n_components=16, batch_size=100)
        self.feature_selector = SelectFromModel(estimator=XGBClassifier())
        self.attention_weights = {}
        self.last_training_time = 0
        self.model_performance = {}
        self.feature_importance = {}
        self.meta_learner = self._initialize_meta_learner()
        
    def _initialize_graph_attention_model(self):
        """Initialize Graph Attention Network for market structure analysis."""
        node_features = Input(shape=(None, 15))  # Market features per node
        edge_indices = Input(shape=(None, 2), dtype='int32')
        
        # Graph attention layers
        x = GATv2Conv(64, activation='relu', attention_heads=4)([node_features, edge_indices])
        x = BatchNormalization()(x)
        x = GATv2Conv(32, activation='relu', attention_heads=2)([x, edge_indices])
        x = GlobalMaxPooling1D()(x)
        
        # Output layer
        outputs = Dense(len(OrderFlowPattern), activation='softmax')(x)
        
        model = Model(inputs=[node_features, edge_indices], outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=0.0005),
                     loss='categorical_crossentropy',
                     metrics=['accuracy'])
        return model
        
    def _initialize_temporal_conv_model(self):
        """Initialize Temporal Convolutional Network for sequential pattern recognition."""
        model = Sequential([
            Conv1D(128, kernel_size=3, activation='relu', padding='causal', input_shape=(30, 15)),
            LayerNormalization(),
            Conv1D(128, kernel_size=3, activation='relu', padding='causal', dilation_rate=2),
            LayerNormalization(),
            Conv1D(128, kernel_size=3, activation='relu', padding='causal', dilation_rate=4),
            LayerNormalization(),
            GlobalAveragePooling1D(),
            Dense(64, activation='relu'),
            Dense(len(OrderFlowPattern), activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
        return model

class EnhancedExecutionMonitor:
    """Advanced monitoring system for execution quality and market impact."""
    
    def __init__(self):
        self.metrics = {
            'execution': defaultdict(list),
            'market_impact': defaultdict(list),
            'risk': defaultdict(list),
            'performance': defaultdict(list),
            'liquidity': defaultdict(list),
            'latency': defaultdict(list),
            'anomalies': [],
            'strategy_performance': defaultdict(dict)
        }
        self.alert_rules = self._initialize_alert_rules()
        self.performance_benchmarks = self._load_benchmarks()
        self.anomaly_detectors = self._initialize_anomaly_detectors()
        self.dashboard = self._initialize_dashboard()
        self.report_generator = ReportGenerator()
        
    def _initialize_anomaly_detectors(self):
        """Initialize multiple anomaly detection models."""
        return {
            'isolation_forest': IsolationForest(contamination=0.05, random_state=42),
            'one_class_svm': OneClassSVM(nu=0.05, kernel='rbf', gamma=0.1),
            'autoencoder': self._initialize_anomaly_autoencoder()
        }
        
    def _initialize_anomaly_autoencoder(self):
        """Initialize autoencoder for anomaly detection."""
        input_dim = 20  # Number of features
        encoding_dim = 8
        
        input_layer = Input(shape=(input_dim,))
        encoder = Dense(encoding_dim, activation='relu')(input_layer)
        decoder = Dense(input_dim, activation='sigmoid')(encoder)
        
        autoencoder = Model(inputs=input_layer, outputs=decoder)
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder

class AdvancedExecutionStrategies:
    """Advanced order execution strategies with reinforcement learning and adaptive logic."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.strategies = {
            'iceberg': self._initialize_iceberg_strategy(),
            'twap': self._initialize_twap_strategy(),
            'vwap': self._initialize_vwap_strategy(),
            'pov': self._initialize_pov_strategy(),
            'implementation_shortfall': self._initialize_is_strategy(),
            'sniper': self._initialize_sniper_strategy(),
            'adaptive_liquidity': self._initialize_adaptive_liquidity_strategy(),
            'market_making': self._initialize_market_making_strategy(),
            'liquidity_seeking': self._initialize_liquidity_seeking_strategy(),
            'dark_pool': self._initialize_dark_pool_strategy()
        }
        self.rl_agent = self._initialize_rl_agent()
        self.market_regime_classifier = self._initialize_market_regime_classifier()
        self.strategy_performance = {}
        self.risk_manager = RiskManager()
    
    def _initialize_liquidity_seeking_strategy(self) -> Dict:
        """Initialize liquidity-seeking execution strategy."""
        return {
            'min_liquidity_share': 0.05,  # 5% of average daily volume
            'max_impact': 0.0015,  # 15bps max market impact
            'aggression': 0.6,  # Aggression level (0-1)
            'liquidity_horizon': '1h',  # Time horizon for liquidity analysis
            'venue_priority': ['lit', 'dark_pool', 'rfq'],  # Execution venue priority
            'slippage_control': True,  # Enable slippage control
            'adaptive_sizing': True,  # Dynamic order sizing based on liquidity
            'volume_profile': {  # Volume profile analysis parameters
                'window': '1d',
                'bins': 20,
                'sensitivity': 0.8
            }
        }
    
    def _initialize_dark_pool_strategy(self) -> Dict:
        """Initialize dark pool execution strategy."""
        return {
            'min_size': 0.1,  # 10% of average daily volume
            'max_dark_pool_share': 0.3,  # Max 30% of order size in dark pools
            'price_improvement': 0.0002,  # 2bps minimum price improvement
            'venue_weights': {  # Weights for different dark pools
                'pool_a': 0.4,
                'pool_b': 0.3,
                'pool_c': 0.3
            },
            'minimum_resting_time': 30,  # seconds
            'maximum_resting_time': 300,  # seconds
            'leakage_control': True,  # Control information leakage
            'anti_gaming': True  # Anti-gaming measures
        }
    
    def select_strategy(self, order: Order, market_data: Dict) -> Dict:
        """Select and configure optimal execution strategy."""
        # Get market regime classification
        regime = self._classify_market_regime(market_data)
        
        # Calculate order characteristics
        order_size_pct = order.quantity / market_data.get('average_daily_volume', order.quantity)
        urgency = self._calculate_order_urgency(order, market_data)
        
        # Select strategy based on order and market conditions
        if order_size_pct > 0.3:
            # Large orders: Use implementation shortfall or VWAP
            if urgency > 0.7:
                strategy = 'implementation_shortfall'
            else:
                strategy = 'vwap'
        elif market_data.get('liquidity_score', 0) < 0.3:
            # Low liquidity: Use liquidity-seeking strategy
            strategy = 'liquidity_seeking'
        elif order_size_pct < 0.05 and market_data.get('spread', 0) < 0.0005:
            # Small orders in tight markets: Use market making
            strategy = 'market_making'
        elif order_size_pct > 0.1 and market_data.get('dark_pool_liquidity', 0) > order.quantity * 0.5:
            # Medium orders with dark pool liquidity
            strategy = 'dark_pool'
        else:
            # Default to iceberg for medium-sized orders
            strategy = 'iceberg'
        
        # Configure strategy parameters
        params = self._configure_strategy(strategy, order, market_data, regime)
        
        return {
            'strategy': strategy,
            'parameters': params,
            'market_regime': regime,
            'selection_metrics': {
                'order_size_pct': order_size_pct,
                'urgency': urgency,
                'liquidity_score': market_data.get('liquidity_score', 0),
                'market_impact': self._estimate_market_impact(order, market_data, strategy)
            }
        }
    
    def _configure_strategy(self, strategy: str, order: Order, market_data: Dict, regime: str) -> Dict:
        """Configure strategy parameters based on order and market conditions."""
        base_params = self.strategies[strategy].copy()
        
        if strategy == 'iceberg':
            # Adjust iceberg parameters based on market conditions
            base_params['display_size'] = self._calculate_display_size(order, market_data)
            base_params['refresh_rate'] = self._calculate_refresh_rate(market_data)
            base_params['aggression'] = self._calculate_aggression(order, market_data, regime)
            
        elif strategy == 'liquidity_seeking':
            # Configure liquidity-seeking parameters
            base_params['aggression'] = min(0.8, order.urgency * 1.2)
            base_params['max_impact'] = min(0.002, order.max_slippage * 0.8)
            
        elif strategy == 'dark_pool':
            # Adjust dark pool parameters
            base_params['min_size'] = max(
                base_params['min_size'],
                order.quantity * 0.05  # At least 5% of order size
            )
            
        return base_params
    
    def _calculate_display_size(self, order: Order, market_data: Dict) -> float:
        """Calculate optimal display size for iceberg orders."""
        # Base size as percentage of average trade size
        avg_trade_size = market_data.get('average_trade_size', order.quantity * 0.1)
        base_size = avg_trade_size * 0.5  # Start with 50% of average trade size
        
        # Adjust based on order book depth
        depth_ratio = market_data.get('bid_depth', 0) / max(market_data.get('ask_depth', 1), 1)
        size_multiplier = 0.5 + (depth_ratio * 0.5)  # 0.5-1.0x based on depth
        
        # Apply volatility adjustment
        volatility = market_data.get('volatility', 0.01)
        vol_adjustment = 1.0 / (1.0 + volatility * 10)  # Reduce size in high vol
        
        return max(order.min_size, base_size * size_multiplier * vol_adjustment)
    
    def _calculate_refresh_rate(self, market_data: Dict) -> float:
        """Calculate optimal refresh rate for iceberg orders."""
        base_rate = 5.0  # seconds
        
        # Adjust based on market volatility
        volatility = market_data.get('volatility', 0.01)
        vol_adjustment = 1.0 / (1.0 + volatility * 5)  # Faster updates in high vol
        
        # Adjust based on spread
        spread = market_data.get('spread', 0.0005)
        spread_adjustment = 1.0 + (spread * 1000)  # Slower updates in wide spreads
        
        return max(1.0, base_rate * vol_adjustment * spread_adjustment)
    
    def _calculate_aggression(self, order: Order, market_data: Dict, regime: str) -> float:
        """Calculate aggression level based on order and market conditions."""
        # Base aggression from order urgency
        aggression = order.urgency
        
        # Adjust for market regime
        regime_adjustments = {
            'trending_up': 1.2 if order.side == 'buy' else 0.8,
            'trending_down': 0.8 if order.side == 'buy' else 1.2,
            'ranging': 1.0,
            'volatile': 1.3
        }
        aggression *= regime_adjustments.get(regime, 1.0)
        
        # Adjust for time of day (more aggressive near market open/close)
        hour = datetime.now().hour
        if hour in [9, 10, 15, 16]:  # Market open/close hours
            aggression = min(1.0, aggression * 1.2)
            
        return max(0.1, min(1.0, aggression))  # Keep within bounds
    
    def _estimate_market_impact(self, order: Order, market_data: Dict, strategy: str) -> float:
        """Estimate market impact of the order."""
        # Base impact based on order size relative to average daily volume
        size_ratio = order.quantity / market_data.get('average_daily_volume', order.quantity)
        
        # Strategy-specific impact multipliers
        impact_multipliers = {
            'iceberg': 0.7,
            'twap': 0.8,
            'vwap': 0.9,
            'implementation_shortfall': 1.2,
            'liquidity_seeking': 0.6,
            'market_making': 0.3,
            'dark_pool': 0.4
        }
        
        # Adjust for market conditions
        volatility = market_data.get('volatility', 0.01)
        liquidity = market_data.get('liquidity_score', 0.5)
        
        # Calculate base impact (simplified model)
        base_impact = size_ratio * (1.0 / (liquidity + 0.1)) * (1.0 + volatility)
        
        # Apply strategy multiplier
        strategy_impact = base_impact * impact_multipliers.get(strategy, 1.0)
        
        # Apply time decay (less impact for longer execution horizons)
        if 'time_horizon' in order.parameters:
            time_decay = 1.0 / (1.0 + order.parameters['time_horizon'] / 3600)  # Hours
            strategy_impact *= time_decay
            
        return max(0.0, min(1.0, strategy_impact))  # Keep within 0-100%

class EnhancedIcebergExecutor(ExecutionAlgorithm):
    """Enhanced Iceberg Order Execution Algorithm with advanced features."""
    
    def __init__(self, exchange_adapter, position_manager, config: Dict = None):
        super().__init__(exchange_adapter, position_manager, config or {})
        
        # Initialize advanced components
        self.pattern_recognizer = AdvancedPatternRecognizer()
        self.execution_monitor = EnhancedExecutionMonitor()
        self.execution_strategies = AdvancedExecutionStrategies()
        
        # ML model state
        self.market_regime = None
        self.current_patterns = []
        self.last_optimization_time = 0
        
        # Execution state
        self.working_orders = {}
        self.fill_history = deque(maxlen=1000)
        self.order_history = deque(maxlen=1000)
        
        # Initialize with default strategy
        self.current_strategy = 'iceberg'
        self.strategy_parameters = {}
        self.strategy_performance = {}
        
        # Market making state
        self.quote_book = {}
        self.last_quote_time = 0
        self.inventory = 0
        
    async def execute_market_making_strategy(self, order: Order, market_data: Dict):
        """Execute market making strategy with inventory management."""
        strategy = self.execution_strategies.strategies['market_making']
        current_time = time.time()
        
        # Check if it's time to update quotes
        if current_time - self.last_quote_time < strategy['quote_refresh_time']:
            return
            
        # Cancel existing quotes
        await self._cancel_all_quotes()
        
        # Calculate fair value and spread
        fair_value = self._calculate_fair_value(market_data)
        spread = self._calculate_dynamic_spread(market_data, strategy)
        
        # Calculate bid/ask prices with inventory skew
        bid_price, ask_price = self._calculate_skewed_prices(
            fair_value, spread, strategy)
            
        # Calculate order sizes based on inventory and risk
        bid_size, ask_size = self._calculate_order_sizes(
            market_data, strategy)
            
        # Place new quotes
        if bid_size > 0 and bid_price > 0:
            bid_order = self._create_quote_order(
                'buy', bid_price, bid_size, order.symbol)
            await self._place_quote(bid_order)
            
        if ask_size > 0 and ask_price > 0:
            ask_order = self._create_quote_order(
                'sell', ask_price, ask_size, order.symbol)
            await self._place_quote(ask_order)
            
        self.last_quote_time = current_time
        
    async def execute_adaptive_liquidity_strategy(self, order: Order, market_data: Dict):
        """Execute adaptive liquidity seeking strategy."""
        strategy = self.execution_strategies.strategies['adaptive_liquidity']
        
        # Analyze liquidity profile
        liquidity_profile = self._analyze_liquidity_profile(market_data)
        params = strategy['liquidity_profiles'].get(
            liquidity_profile, 
            strategy['liquidity_profiles']['medium']
        )
        
        # Calculate order parameters
        participation_rate = params['participation_rate']
        aggression = params['aggression']
        
        # Calculate target price based on market impact
        target_price = self._calculate_target_price(
            order.side, market_data, aggression)
            
        # Calculate order size based on participation rate and market depth
        order_size = self._calculate_adaptive_order_size(
            order, market_data, participation_rate)
            
        # Place order
        if order_size > 0:
            await self._place_adaptive_order(
                order, order_size, target_price, market_data)
                
    def _calculate_dynamic_spread(self, market_data: Dict, strategy: Dict) -> float:
        """Calculate dynamic spread based on market conditions."""
        # Base spread from order book
        spread = market_data.get('spread', 0.0005)
        
        # Adjust for volatility
        volatility = market_data.get('volatility', 0.01)
        spread *= (1 + volatility * strategy['volatility_scale'])
        
        # Adjust for inventory
        inventory_skew = self.inventory / strategy['max_position']
        spread *= (1 + abs(inventory_skew) * strategy['inventory_skew'])
        
        # Apply minimum spread
        return max(spread, strategy['min_spread'])
        
    def _analyze_liquidity_profile(self, market_data: Dict) -> str:
        """Analyze market liquidity profile."""
        # Calculate liquidity score based on order book depth and spread
        spread_ratio = market_data.get('spread', 0.0005) / market_data.get('mid_price', 1.0)
        depth_ratio = market_data.get('bid_depth', 0) / (market_data.get('ask_depth', 0) + 1e-6)
        
        if spread_ratio < 0.0005 and depth_ratio > 0.8 and depth_ratio < 1.2:
            return 'high'
        elif spread_ratio < 0.001 and depth_ratio > 0.6 and depth_ratio < 1.5:
            return 'medium'
        return 'low'

class EnhancedExecutionDashboard:
    """Advanced monitoring dashboard for execution performance and strategy analytics."""
    
    def __init__(self):
        self.metrics = {
            'execution': {},
            'risk': {},
            'performance': {},
            'market_impact': {},
            'liquidity': {},
            'latency': {},
            'strategy_analytics': {}
        }
        self.plots = {}
        self.alerts = []
        self.performance_metrics = {}
        self.strategy_comparison = {}
        self.risk_metrics = {}
        self._initialize_ui_components()
    
    def _initialize_ui_components(self):
        """Initialize dashboard UI components."""
        self.components = {
            'performance_gauges': {},
            'strategy_heatmap': None,
            'order_flow_chart': None,
            'risk_metrics_panel': None,
            'market_depth_view': None,
            'latency_histogram': None,
            'alert_panel': None
        }
    
    def update_metrics(self, metric_type: str, name: str, value: float, metadata: Dict = None):
        """Update metric values with enhanced tracking."""
        if metric_type not in self.metrics:
            self.metrics[metric_type] = {}
            
        timestamp = datetime.now(timezone.utc)
        self.metrics[metric_type][name] = {
            'value': value,
            'timestamp': timestamp,
            'metadata': metadata or {}
        }
        
        # Update time series data for visualization
        self._update_time_series(metric_type, name, value, timestamp)
        
        # Check for anomalies and generate alerts
        self._check_anomalies(metric_type, name, value)
        
        # Update performance metrics
        self._update_performance_metrics(metric_type, name, value, metadata)
        
        # Update strategy analytics if applicable
        if 'strategy' in (metadata or {}):
            self._update_strategy_analytics(metadata['strategy'], name, value, timestamp)
    
    def _update_time_series(self, metric_type: str, name: str, value: float, timestamp: datetime):
        """Update time series data for visualization."""
        if metric_type not in self.plots:
            self.plots[metric_type] = {}
        if name not in self.plots[metric_type]:
            self.plots[metric_type][name] = []
            
        self.plots[metric_type][name].append({
            'x': timestamp,
            'y': value,
            'metadata': self.metrics[metric_type][name].get('metadata', {})
        })
        
        # Keep only the last 1000 data points
        self.plots[metric_type][name] = self.plots[metric_type][name][-1000:]
    
    def _check_anomalies(self, metric_type: str, name: str, value: float):
        """Check for anomalous metric values and generate alerts."""
        thresholds = {
            'latency': {'warning': 0.1, 'critical': 0.5},  # seconds
            'slippage': {'warning': 0.0005, 'critical': 0.002},  # 5-20 bps
            'fill_rate': {'warning': 0.7, 'critical': 0.5},  # 50-70%
            'market_impact': {'warning': 0.001, 'critical': 0.003},  # 10-30 bps
        }
        
        for metric, levels in thresholds.items():
            if metric in name.lower():
                if value > levels.get('critical', float('inf')):
                    self._trigger_alert(
                        level='CRITICAL',
                        message=f"{metric} exceeded critical threshold: {value:.6f} > {levels['critical']:.6f}",
                        metric=name,
                        value=value
                    )
                elif value > levels.get('warning', float('inf')):
                    self._trigger_alert(
                        level='WARNING',
                        message=f"{metric} exceeded warning threshold: {value:.6f} > {levels['warning']:.6f}",
                        metric=name,
                        value=value
                    )
    
    def _trigger_alert(self, level: str, message: str, metric: str, value: float):
        """Create and store a new alert."""
        alert = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.now(timezone.utc),
            'level': level,
            'message': message,
            'metric': metric,
            'value': value,
            'acknowledged': False
        }
        self.alerts.append(alert)
        
        # Send real-time notification (e.g., WebSocket, email, etc.)
        self._send_notification(alert)
    
    def _update_performance_metrics(self, metric_type: str, name: str, value: float, metadata: Dict = None):
        """Update aggregated performance metrics."""
        if 'strategy' in (metadata or {}):
            strategy = metadata['strategy']
            if strategy not in self.performance_metrics:
                self.performance_metrics[strategy] = {
                    'total_volume': 0,
                    'avg_fill_price': 0,
                    'slippage': 0,
                    'fill_rate': 0,
                    'count': 0
                }
            
            stats = self.performance_metrics[strategy]
            stats['count'] += 1
            
            if 'volume' in name.lower():
                stats['total_volume'] += value
            elif 'price' in name.lower() and 'fill' in name.lower():
                # Update running average
                stats['avg_fill_price'] = (
                    (stats['avg_fill_price'] * (stats['count'] - 1) + value) / 
                    stats['count']
                )
            elif 'slippage' in name.lower():
                stats['slippage'] = (
                    (stats['slippage'] * (stats['count'] - 1) + value) / 
                    stats['count']
                )
            elif 'fill_rate' in name.lower():
                stats['fill_rate'] = (
                    (stats['fill_rate'] * (stats['count'] - 1) + value) / 
                    stats['count']
                )
    
    def _update_strategy_analytics(self, strategy: str, metric: str, value: float, timestamp: datetime):
        """Update analytics for strategy performance comparison."""
        if strategy not in self.strategy_comparison:
            self.strategy_comparison[strategy] = {}
        
        if metric not in self.strategy_comparison[strategy]:
            self.strategy_comparison[strategy][metric] = []
            
        self.strategy_comparison[strategy][metric].append({
            'timestamp': timestamp,
            'value': value
        })
    
    def get_performance_summary(self) -> Dict:
        """Generate a performance summary across all strategies."""
        summary = {
            'total_volume': 0,
            'weighted_slippage': 0,
            'avg_fill_rate': 0,
            'strategy_performance': {}
        }
        
        total_volume = sum(
            stats['total_volume'] 
            for stats in self.performance_metrics.values()
        )
        
        for strategy, stats in self.performance_metrics.items():
            weight = stats['total_volume'] / total_volume if total_volume > 0 else 0
            
            summary['strategy_performance'][strategy] = {
                'volume': stats['total_volume'],
                'volume_pct': weight * 100,
                'avg_fill_price': stats['avg_fill_price'],
                'slippage_bps': stats['slippage'] * 10000,
                'fill_rate': stats['fill_rate'] * 100
            }
            
            summary['total_volume'] += stats['total_volume']
            summary['weighted_slippage'] += stats['slippage'] * weight
            summary['avg_fill_rate'] += stats['fill_rate'] * weight
        
        return summary
    
    def get_visualization_data(self, metric_type: str = None, name: str = None) -> Dict:
        """Get formatted data for visualization components."""
        if metric_type and name:
            return self.plots.get(metric_type, {}).get(name, [])
        elif metric_type:
            return self.plots.get(metric_type, {})
        return self.plots
    
    def get_active_alerts(self, level: str = None) -> List[Dict]:
        """Get active alerts, optionally filtered by level."""
        if level:
            return [a for a in self.alerts if a['level'] == level and not a['acknowledged']]
        return [a for a in self.alerts if not a['acknowledged']]
    
    def acknowledge_alert(self, alert_id: str):
        """Mark an alert as acknowledged."""
        for alert in self.alerts:
            if alert['id'] == alert_id:
                alert['acknowledged'] = True
                break
    
    def _send_notification(self, alert: Dict):
        """Send real-time notification for critical alerts."""
        if alert['level'] in ['CRITICAL', 'ERROR']:
            # Implementation for sending notifications (e.g., WebSocket, email, etc.)
            logger.warning(f"{alert['level']} ALERT: {alert['message']}")

class AdvancedRiskManager:
    """Advanced risk management for execution strategies."""
    
    def __init__(self):
        self.risk_models = {
            'var': self._initialize_var_model(),
            'cvar': self._initialize_cvar_model(),
            'liquidity_risk': self._initialize_liquidity_risk_model()
        }
        self.position_limits = {}
        self.risk_limits = {
            'max_drawdown': 0.05,  # 5% max drawdown
            'max_position': 10000,  # Max position size
            'max_risk_per_trade': 0.01,  # 1% of capital
            'max_daily_loss': 0.02  # 2% max daily loss
        }
        
    def calculate_position_risk(self, position: Dict, market_data: Dict) -> Dict:
        """Calculate risk metrics for a position."""
        var = self.risk_models['var'].calculate(position, market_data)
        cvar = self.risk_models['cvar'].calculate(position, market_data)
        liquidity_risk = self.risk_models['liquidity_risk'].assess(position, market_data)
        
        return {
            'var': var,
            'cvar': cvar,
            'liquidity_risk': liquidity_risk,
            'risk_limits': self._check_risk_limits(position, var, cvar)
        }
        
    def _check_risk_limits(self, position: Dict, var: float, cvar: float) -> Dict:
        """Check if position violates any risk limits."""
        return {
            'max_drawdown': position['unrealized_pnl'] / position['cost_basis'] < -self.risk_limits['max_drawdown'],
            'max_position': abs(position['size']) > self.risk_limits['max_position'],
            'max_risk': cvar > self.risk_limits['max_risk_per_trade'] * position['cost_basis'],
            'daily_loss': position['daily_pnl'] < -self.risk_limits['max_daily_loss'] * position['cost_basis']
        }

class SmartOrderRouter:
    """Intelligent order routing and execution management."""
    
    def __init__(self, exchange_adapters: Dict):
        self.exchange_adapters = exchange_adapters
        self.latency_monitor = LatencyMonitor()
        self.smart_order_books = {}
        
    async def route_order(self, order: Order, strategy_params: Dict) -> Dict:
        """Route order to optimal execution venue."""
        # Get best execution venues based on strategy
        venues = self._select_execution_venues(order, strategy_params)
        
        # Calculate smart order parameters
        order_params = self._calculate_order_parameters(order, venues, strategy_params)
        
        # Execute order across venues
        executions = await self._execute_across_venues(order, order_params, venues)
        
        return {
            'order_id': order.id,
            'executions': executions,
            'metrics': self._calculate_execution_metrics(executions, order)
        }
        
    def _select_execution_venues(self, order: Order, strategy_params: Dict) -> List[Dict]:
        """Select optimal execution venues based on order and strategy."""
        # Implementation for venue selection
        pass
        
    def _calculate_order_parameters(self, order: Order, venues: List[Dict], 
                                  strategy_params: Dict) -> Dict:
        """Calculate optimal order parameters."""
        # Implementation for order parameter calculation
        pass

# Add new utility functions
def _calculate_order_book_imbalance(market_data: Dict) -> float:
    """Calculate order book imbalance metric."""
    bids = market_data.get('bids', [])
    asks = market_data.get('asks', [])

    if not bids or not asks:
        return 0.0

    total_bid_volume = sum(price * qty for price, qty in bids[:5])  # Top 5 levels
    total_ask_volume = sum(price * qty for price, qty in asks[:5])

    if total_bid_volume + total_ask_volume == 0:
        return 0.0

    return (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)

def _calculate_trade_flow_imbalance(trades: List[Dict]) -> float:
    """Calculate trade flow imbalance."""
    if not trades:
        return 0.0

    buy_volume = sum(t['size'] for t in trades if t['side'] == 'buy')
    sell_volume = sum(t['size'] for t in trades if t['side'] == 'sell')

    if buy_volume + sell_volume == 0:
        return 0.0

    return (buy_volume - sell_volume) / (buy_volume + sell_volume)

# Add more utility functions as needed...
