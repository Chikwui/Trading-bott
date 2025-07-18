"""
Integration tests for Iceberg order execution strategy.

This module tests the Iceberg execution strategy with a mock exchange simulator
that can simulate various market conditions.
"""
import asyncio
import logging
import random
import time
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any, Deque, Set
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest
from dataclasses import dataclass, field

from core.execution.base import (
    ExecutionClient, ExecutionParameters, ExecutionResult, ExecutionState
)
from core.execution.models.market_impact import (
    MarketImpactModelType, MarketImpactParams, create_market_impact_model
)
from core.execution.strategies.iceberg import IcebergExecutionClient, IcebergParameters
from core.market.data import MarketDataService, BarData, TickerData
from core.risk.manager import RiskManager
from core.trading.order import (
    Order, OrderSide, OrderType, OrderStatus, TimeInForce
)

logger = logging.getLogger(__name__)

# Test configuration
TEST_SYMBOL = "BTC/USD"
TEST_QUANTITY = Decimal("10.0")
TEST_PRICE = Decimal("50000.0")
SPREAD = Decimal("10.0")  # $10 spread

@dataclass
class MockOrderBook:
    """Mock order book implementation for testing."""
    bids: List[Tuple[Decimal, Decimal]] = field(default_factory=list)
    asks: List[Tuple[Decimal, Decimal]] = field(default_factory=list)
    
    def add_bid(self, price: Decimal, size: Decimal) -> None:
        """Add a bid to the order book."""
        self.bids.append((price, size))
        self.bids.sort(reverse=True)  # Best bid first
    
    def add_ask(self, price: Decimal, size: Decimal) -> None:
        """Add an ask to the order book."""
        self.asks.append((price, size))
        self.asks.sort()  # Best ask first
    
    def get_bbo(self) -> Tuple[Optional[Decimal], Optional[Decimal]]:
        """Get best bid and offer."""
        best_bid = self.bids[0][0] if self.bids else None
        best_ask = self.asks[0][0] if self.asks else None
        return best_bid, best_ask
    
    def get_mid_price(self) -> Optional[Decimal]:
        """Get mid price from BBO."""
        bid, ask = self.get_bbo()
        if bid is not None and ask is not None:
            return (bid + ask) / 2
        return None
    
    def get_spread(self) -> Optional[Decimal]:
        """Get current spread."""
        bid, ask = self.get_bbo()
        if bid is not None and ask is not None:
            return ask - bid
        return None

class MockMarketDataService(MarketDataService):
    """Mock market data service for testing."""
    
    def __init__(self):
        self.order_books: Dict[str, MockOrderBook] = {}
        self.ticker_data: Dict[str, Dict[str, Any]] = {}
        self.bar_data: Dict[str, List[BarData]] = {}
        self.trade_history: Dict[str, List[Dict]] = {}
        self._subscribers = set()
        
        # Initialize test market data
        self._init_test_data()
    
    def _init_test_data(self) -> None:
        """Initialize test market data."""
        # Create order book for test symbol
        book = MockOrderBook()
        book.add_bid(TEST_PRICE - SPREAD/2, Decimal("5.0"))
        book.add_ask(TEST_PRICE + SPREAD/2, Decimal("5.0"))
        self.order_books[TEST_SYMBOL] = book
        
        # Initialize ticker data
        self.ticker_data[TEST_SYMBOL] = {
            'bid': float(TEST_PRICE - SPREAD/2),
            'ask': float(TEST_PRICE + SPREAD/2),
            'last': float(TEST_PRICE),
            'volume': 100.0,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        # Initialize some historical data
        now = datetime.utcnow()
        self.bar_data[TEST_SYMBOL] = [
            BarData(
                symbol=TEST_SYMBOL,
                timestamp=now - timedelta(minutes=i),
                open=float(TEST_PRICE) * (1 + random.uniform(-0.01, 0.01)),
                high=float(TEST_PRICE) * (1 + random.uniform(0, 0.02)),
                low=float(TEST_PRICE) * (1 - random.uniform(0, 0.02)),
                close=float(TEST_PRICE) * (1 + random.uniform(-0.01, 0.01)),
                volume=random.uniform(50, 150)
            )
            for i in range(100, 0, -1)
        ]
    
    async def get_symbol_data(self, symbol: str) -> Dict[str, Any]:
        """Get current market data for a symbol."""
        if symbol not in self.order_books:
            return {}
            
        book = self.order_books[symbol]
        bid, ask = book.get_bbo()
        mid = book.get_mid_price() or Decimal("0")
        
        return {
            'symbol': symbol,
            'bid': float(bid) if bid else None,
            'ask': float(ask) if ask else None,
            'mid': float(mid) if mid else None,
            'spread': float(book.get_spread() or Decimal("0")),
            'timestamp': datetime.utcnow().isoformat(),
            'volume_24h': 1000.0,
            'volume_5m': 100.0,
            'avg_volume_5m': 90.0,
            'volatility_5m': 0.005,
            'daily_volatility': 0.02,
            'avg_daily_volume': 10000.0,
            'vwap': float(mid) if mid else 0.0,
            'bids': [(float(p), float(s)) for p, s in book.bids[:10]],
            'asks': [(float(p), float(s)) for p, s in book.asks[:10]],
            'lot_size': 0.000001,
            'tick_size': 0.01
        }
    
    async def get_order_book(self, symbol: str, depth: int = 10) -> Dict[str, List[Tuple[float, float]]]:
        """Get order book data."""
        if symbol not in self.order_books:
            return {'bids': [], 'asks': []}
            
        book = self.order_books[symbol]
        return {
            'bids': [(float(p), float(s)) for p, s in book.bids[:depth]],
            'asks': [(float(p), float(s)) for p, s in book.asks[:depth]]
        }
    
    async def get_recent_trades(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Get recent trades."""
        if symbol not in self.trade_history:
            return []
        return self.trade_history[symbol][-limit:]
    
    async def get_recent_bars(
        self, 
        symbol: str, 
        timeframe: str = '1m', 
        limit: int = 100
    ) -> List[BarData]:
        """Get recent OHLCV bars."""
        if symbol not in self.bar_data:
            return []
        return self.bar_data[symbol][:limit]
    
    async def subscribe(self, symbols: List[str]) -> None:
        """Subscribe to market data updates."""
        self._subscribers.update(symbols)
    
    async def unsubscribe(self, symbols: List[str]) -> None:
        """Unsubscribe from market data updates."""
        self._subscribers.difference_update(symbols)
    
    # Methods to manipulate the mock market for testing
    def update_market_price(self, symbol: str, new_price: Decimal) -> None:
        """Update the market price for testing."""
        if symbol not in self.order_books:
            return
            
        book = self.order_books[symbol]
        spread = book.get_spread() or SPREAD
        
        # Clear existing orders
        book.bids.clear()
        book.asks.clear()
        
        # Add new orders around the new price
        book.add_bid(new_price - spread/2, Decimal("5.0"))
        book.add_ask(new_price + spread/2, Decimal("5.0"))
        
        # Update ticker data
        if symbol in self.ticker_data:
            self.ticker_data[symbol].update({
                'bid': float(new_price - spread/2),
                'ask': float(new_price + spread/2),
                'last': float(new_price),
                'timestamp': datetime.utcnow().isoformat()
            })
    
    def add_trade(
        self, 
        symbol: str, 
        price: Decimal, 
        size: Decimal, 
        side: str, 
        timestamp: Optional[datetime] = None
    ) -> None:
        """Add a trade to the trade history."""
        if symbol not in self.trade_history:
            self.trade_history[symbol] = []
            
        self.trade_history[symbol].append({
            'symbol': symbol,
            'price': float(price),
            'size': float(size),
            'side': side,
            'timestamp': timestamp or datetime.utcnow(),
            'trade_id': f"trade_{len(self.trade_history[symbol])}"
        })
        
        # Keep only the last 1000 trades per symbol
        if len(self.trade_history[symbol]) > 1000:
            self.trade_history[symbol] = self.trade_history[symbol][-1000:]

class MockRiskManager(RiskManager):
    """Mock risk manager for testing."""
    
    def __init__(self):
        self.positions: Dict[str, Decimal] = {}
        self.orders: Dict[str, Order] = {}
        self.max_position_size = Decimal("100.0")
    
    async def check_order(self, order: Order) -> bool:
        """Check if an order passes risk checks."""
        # Simple check: don't exceed max position size
        current_size = abs(self.positions.get(order.symbol, Decimal("0")))
        new_size = current_size + (order.quantity if order.side == OrderSide.BUY else -order.quantity)
        
        if abs(new_size) > self.max_position_size:
            return False
            
        return True
    
    async def update_position(self, symbol: str, delta: Decimal) -> None:
        """Update position for a symbol."""
        self.positions[symbol] = self.positions.get(symbol, Decimal("0")) + delta

# Fixtures
@pytest.fixture
def mock_market_data() -> MockMarketDataService:
    """Create a mock market data service."""
    return MockMarketDataService()

@pytest.fixture
def mock_risk_manager() -> MockRiskManager:
    """Create a mock risk manager."""
    return MockRiskManager()

@pytest.fixture
def iceberg_params() -> IcebergParameters:
    """Create default Iceberg parameters for testing."""
    return IcebergParameters(
        max_visible_pct_adv=0.05,
        min_visible_qty=Decimal("0.1"),
        max_visible_qty=Decimal("5.0"),
        refresh_interval=1,  # 1 second for faster tests
        max_duration=60,     # 1 minute max duration
        aggressiveness=0.5,
        dynamic_quantity=True,
        randomize_refresh=False,  # Disable for deterministic tests
        max_allowed_spread_bps=20,
        order_timeout=30,
        min_order_interval_ms=100,
        max_order_interval_ms=1000,
        size_randomization_pct=0.0,  # Disable for deterministic tests
        stealth_mode=False
    )

@pytest.fixture
def market_impact_params() -> MarketImpactParams:
    """Create market impact model parameters for testing."""
    return MarketImpactParams(
        model_type=MarketImpactModelType.ALMGREN_CHRISS,
        ac_permanent_impact=0.1,
        ac_temporary_impact=0.1,
        ac_volatility=0.2
    )

@pytest.fixture
async def iceberg_client(
    mock_market_data: MockMarketDataService,
    mock_risk_manager: MockRiskManager,
    iceberg_params: IcebergParameters,
    market_impact_params: MarketImpactParams
) -> IcebergExecutionClient:
    """Create an Iceberg execution client for testing."""
    # Create market impact model
    impact_model = create_market_impact_model(market_impact_params)
    
    # Create and return the client
    return IcebergExecutionClient(
        client_id="test_client",
        market_data=mock_market_data,
        risk_manager=mock_risk_manager,
        iceberg_params=iceberg_params,
        market_impact_model=impact_model
    )

# Test cases
@pytest.mark.asyncio
async def test_iceberg_basic_execution(iceberg_client: IcebergExecutionClient):
    """Test basic Iceberg order execution."""
    # Create a test order
    order = Order(
        symbol=TEST_SYMBOL,
        side=OrderSide.BUY,
        quantity=TEST_QUANTITY,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.GTC
    )
    
    # Create execution parameters
    params = ExecutionParameters(
        strategy="iceberg",
        urgency="normal",
        max_slippage_bps=50,
        start_time=datetime.utcnow(),
        end_time=datetime.utcnow() + timedelta(seconds=30)
    )
    
    # Execute the order
    result = await iceberg_client.execute(order, params)
    
    # Verify the result
    assert result is not None
    assert result.status == ExecutionState.COMPLETED
    assert result.filled_quantity == TEST_QUANTITY
    assert result.avg_fill_price is not None
    assert result.avg_fill_price > 0
    
    # Verify metrics
    assert 'execution_time_sec' in result.metrics
    assert 'avg_slippage_bps' in result.metrics
    assert 'market_impact_bps' in result.metrics
    assert 'participation_rate' in result.metrics

@pytest.mark.asyncio
async def test_iceberg_with_price_movement(iceberg_client: IcebergExecutionClient, mock_market_data: MockMarketDataService):
    """Test Iceberg execution with price movement during execution."""
    # Initial price movement function
    async def move_price():
        await asyncio.sleep(0.5)  # Wait for initial order placement
        mock_market_data.update_market_price(TEST_SYMBOL, TEST_PRICE * Decimal("1.01"))  # 1% price increase
        await asyncio.sleep(0.5)
        mock_market_data.update_market_price(TEST_SYMBOL, TEST_PRICE * Decimal("0.99"))  # 1% price decrease
    
    # Start price movement in background
    price_task = asyncio.create_task(move_price())
    
    try:
        # Create and execute order
        order = Order(
            symbol=TEST_SYMBOL,
            side=OrderSide.BUY,
            quantity=TEST_QUANTITY,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.GTC
        )
        
        params = ExecutionParameters(
            strategy="iceberg",
            urgency="normal",
            max_slippage_bps=100,  # Allow more slippage for this test
            start_time=datetime.utcnow()
        )
        
        result = await iceberg_client.execute(order, params)
        
        # Verify the result
        assert result.status == ExecutionState.COMPLETED
        assert result.filled_quantity == TEST_QUANTITY
        
        # Verify we got some price improvement or paid some slippage
        assert result.avg_fill_price is not None
        assert result.avg_fill_price > 0
        
        # Check that we have execution metrics
        assert 'slippage_bps' in result.metrics
        assert 'market_impact_bps' in result.metrics
        
    finally:
        # Clean up
        if not price_task.done():
            price_task.cancel()
            try:
                await price_task
            except asyncio.CancelledError:
                pass

@pytest.mark.asyncio
async def test_iceberg_with_high_volatility(iceberg_client: IcebergExecutionClient, mock_market_data: MockMarketDataService):
    """Test Iceberg execution during high volatility."""
    # Simulate high volatility by frequently changing prices
    async def simulate_volatility():
        while True:
            try:
                # Random price changes between -0.5% and +0.5%
                change = Decimal(str(random.uniform(-0.005, 0.005)))
                new_price = TEST_PRICE * (1 + change)
                mock_market_data.update_market_price(TEST_SYMBOL, new_price)
                await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                break
    
    # Start volatility simulation
    vol_task = asyncio.create_task(simulate_volatility())
    
    try:
        # Create and execute order
        order = Order(
            symbol=TEST_SYMBOL,
            side=OrderSide.BUY,
            quantity=TEST_QUANTITY,
            order_type=OrderType.MARKET,
            time_in_force=TimeInForce.GTC
        )
        
        params = ExecutionParameters(
            strategy="iceberg",
            urgency="normal",
            max_slippage_bps=200,  # Higher slippage tolerance for volatile markets
            start_time=datetime.utcnow()
        )
        
        result = await iceberg_client.execute(order, params)
        
        # Verify the result
        assert result.status == ExecutionState.COMPLETED
        assert result.filled_quantity == TEST_QUANTITY
        
        # Check execution metrics
        assert 'volatility_bps' in result.metrics
        assert 'max_drawdown_bps' in result.metrics
        
    finally:
        # Clean up
        if not vol_task.done():
            vol_task.cancel()
            try:
                await vol_task
            except asyncio.CancelledError:
                pass

@pytest.mark.asyncio
async def test_iceberg_with_low_liquidity(iceberg_client: IcebergExecutionClient, mock_market_data: MockMarketDataService):
    """Test Iceberg execution in low liquidity conditions."""
    # Reduce liquidity in the order book
    book = mock_market_data.order_books[TEST_SYMBOL]
    book.bids.clear()
    book.asks.clear()
    
    # Add thin order book
    for i in range(5):
        price = TEST_PRICE * (1 - Decimal("0.001") * (i + 1))
        book.add_bid(price, Decimal("0.1"))  # Small size
    
    for i in range(5):
        price = TEST_PRICE * (1 + Decimal("0.001") * (i + 1))
        book.add_ask(price, Decimal("0.1"))  # Small size
    
    # Create and execute order
    order = Order(
        symbol=TEST_SYMBOL,
        side=OrderSide.BUY,
        quantity=TEST_QUANTITY,
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.GTC
    )
    
    params = ExecutionParameters(
        strategy="iceberg",
        urgency="normal",
        max_slippage_bps=500,  # Higher slippage tolerance for illiquid markets
        start_time=datetime.utcnow()
    )
    
    result = await iceberg_client.execute(order, params)
    
    # Verify the result
    assert result.status == ExecutionState.COMPLETED
    assert result.filled_quantity == TEST_QUANTITY
    
    # Check execution metrics
    assert 'liquidity_impact_bps' in result.metrics
    assert 'slippage_bps' in result.metrics
    assert result.metrics.get('slippage_bps', 0) > 0  # Expect some slippage

@pytest.mark.asyncio
async def test_iceberg_cancellation(iceberg_client: IcebergExecutionClient):
    """Test cancellation of an in-progress Iceberg order."""
    # Create a large order that will take time to execute
    order = Order(
        symbol=TEST_SYMBOL,
        side=OrderSide.BUY,
        quantity=TEST_QUANTITY * 10,  # Larger order
        order_type=OrderType.MARKET,
        time_in_force=TimeInForce.GTC
    )
    
    # Execute the order in the background
    params = ExecutionParameters(
        strategy="iceberg",
        urgency="normal",
        start_time=datetime.utcnow()
    )
    
    # Start execution
    exec_task = asyncio.create_task(iceberg_client.execute(order, params))
    
    # Wait a bit for execution to start
    await asyncio.sleep(0.5)
    
    try:
        # Cancel the execution
        await iceberg_client.cancel(order.id)
        
        # Wait for the execution to complete
        result = await exec_task
        
        # Verify the result
        assert result.status == ExecutionState.CANCELLED
        assert result.filled_quantity < order.quantity  # Some fills may have happened
        
        # Check metrics
        assert 'cancellation_reason' in result.metrics
        assert 'filled_pct' in result.metrics
        
    except asyncio.CancelledError:
        # Task was cancelled, which is expected
        pass
    finally:
        # Ensure task is done
        if not exec_task.done():
            exec_task.cancel()
            try:
                await exec_task
            except asyncio.CancelledError:
                pass

# Run the tests
if __name__ == "__main__":
    import sys
    sys.exit(pytest.main(["-v", "-s", __file__]))
