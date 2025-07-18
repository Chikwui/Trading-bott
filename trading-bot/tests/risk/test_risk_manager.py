"""
Comprehensive test suite for RiskManager class.
"""
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
import pytest
from unittest.mock import MagicMock, patch

from core.risk.manager import (
    RiskManager, 
    Position, 
    RiskCheckResult, 
    RiskViolationType
)
from core.trading.order import Order, OrderSide, OrderType
from core.market.instrument import Instrument, AssetClass

# Fixtures
@pytest.fixture
def sample_instrument():
    return Instrument(
        symbol="AAPL",
        name="Apple Inc.",
        asset_class=AssetClass.STOCK,
        exchange="NASDAQ",
        min_price_increment=Decimal("0.01"),
        multiplier=Decimal("1.0")
    )

@pytest.fixture
def risk_manager():
    return RiskManager(
        account_balance=Decimal("100000"),
        max_position_size_pct=0.1,  # 10%
        max_daily_loss_pct=0.02,    # 2%
        max_drawdown_pct=0.1,       # 10%
        max_leverage=5.0,           # 5x
        volatility_window=5,
        circuit_breaker_pct=0.05    # 5%
    )

@pytest.fixture
def market_order(sample_instrument):
    return Order(
        order_id="test_order_1",
        symbol="AAPL",
        instrument=sample_instrument,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("100"),
        timestamp=datetime.utcnow()
    )

@pytest.fixture
def limit_order(sample_instrument):
    return Order(
        order_id="test_order_2",
        symbol="AAPL",
        instrument=sample_instrument,
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=Decimal("100"),
        limit_price=Decimal("150.00"),
        timestamp=datetime.utcnow()
    )

@pytest.mark.asyncio
async def test_initialization(risk_manager):
    """Test RiskManager initialization with default parameters."""
    assert risk_manager.account_balance == Decimal("100000")
    assert risk_manager.equity == Decimal("100000")
    assert risk_manager.max_position_size_pct == Decimal("0.1")
    assert risk_manager.max_daily_loss_pct == Decimal("0.02")
    assert risk_manager.max_drawdown_pct == Decimal("0.1")
    assert risk_manager.max_leverage == Decimal("5.0")
    assert not risk_manager.circuit_breaker_triggered

@pytest.mark.asyncio
async def test_position_management(risk_manager, sample_instrument, market_order):
    """Test position tracking and updates."""
    # Initial state
    assert "AAPL" not in risk_manager.positions
    
    # Add a position
    await risk_manager.on_order_fill(market_order, Decimal("50"), Decimal("150.00"))
    
    # Verify position was created
    assert "AAPL" in risk_manager.positions
    position = risk_manager.positions["AAPL"]
    assert position.quantity == Decimal("50")
    assert position.avg_price == Decimal("150.00")
    
    # Update price and verify P&L
    await risk_manager.update_market_data("AAPL", Decimal("155.00"))
    assert position.unrealized_pnl == Decimal("250.00")  # 50 * (155 - 150)
    
    # Add to position
    await risk_manager.on_order_fill(market_order, Decimal("50"), Decimal("160.00"))
    assert position.quantity == Decimal("100")
    assert position.avg_price == Decimal("155.00")  # (50*150 + 50*160) / 100
    
    # Update price and verify P&L after adding to position
    await risk_manager.update_market_data("AAPL", Decimal("165.00"))
    expected_pnl = 100 * (165 - 155)  # 1000
    assert position.unrealized_pnl == Decimal(str(expected_pnl))
    
    # Close position partially
    sell_order = market_order.copy()
    sell_order.side = OrderSide.SELL
    await risk_manager.on_order_fill(sell_order, Decimal("30"), Decimal("170.00"))
    
    # Verify realized and unrealized P&L
    realized_pnl = 30 * (170 - 155)  # 450
    assert position.realized_pnl == Decimal(str(realized_pnl))
    assert position.quantity == Decimal("70")
    
    # Update price and verify P&L
    await risk_manager.update_market_data("AAPL", Decimal("175.00"))
    expected_unrealized = 70 * (175 - 155)  # 1400
    assert position.unrealized_pnl == Decimal(str(expected_unrealized))
    
    # Close position completely
    await risk_manager.on_order_fill(sell_order, Decimal("70"), Decimal("175.00"))
    assert position.quantity == Decimal("0")
    assert position.realized_pnl == Decimal("1850")  # 450 + 70*(175-155)
    assert position.unrealized_pnl == Decimal("0")

@pytest.mark.asyncio
async def test_risk_checks(risk_manager, market_order, sample_instrument):
    """Test various risk checks."""
    # Test position size limit (10% of 100k = 10k)
    # 1000 shares at $150 = $150,000 > $10,000 limit
    large_order = market_order.copy()
    large_order.quantity = Decimal("1000")
    
    result = await risk_manager.check_order(large_order)
    assert not result.passed
    assert result.metadata["violation"] == RiskViolationType.POSITION_LIMIT
    
    # Test leverage limit (5x)
    # 50 shares at $150 = $7,500 exposure on $100k balance = 0.075x leverage - should pass
    small_order = market_order.copy()
    small_order.quantity = Decimal("50")
    
    result = await risk_manager.check_order(small_order)
    assert result.passed
    
    # Fill the order
    await risk_manager.on_order_fill(small_order, Decimal("50"), Decimal("150.00"))
    
    # Try another order that would exceed leverage
    # 4000 shares at $150 = $600k exposure, existing $7.5k = $607.5k / $100k = 6.075x > 5x
    large_order.quantity = Decimal("4000")
    result = await risk_manager.check_order(large_order)
    assert not result.passed
    assert result.metadata["violation"] == RiskViolationType.LEVERAGE_LIMIT

@pytest.mark.asyncio
async def test_daily_loss_limit(risk_manager, market_order):
    """Test daily loss limit functionality."""
    # Initial balance is 100k, max daily loss is 2% = 2k
    
    # First, take a loss of 1k
    loss_order = market_order.copy()
    loss_order.side = OrderSide.SELL
    
    # Buy high, sell low
    await risk_manager.on_order_fill(market_order, Decimal("100"), Decimal("150.00"))
    await risk_manager.on_order_fill(loss_order, Decimal("100"), Decimal("140.00"))
    
    # Should have lost 1k (100 * $10)
    assert risk_manager.daily_pnl == Decimal("-1000.00")
    
    # Try to take another 1.5k loss (total 2.5k > 2k limit)
    await risk_manager.on_order_fill(market_order, Decimal("100"), Decimal("150.00"))
    
    # Check if the order would be rejected
    loss_order.quantity = Decimal("150")
    result = await risk_manager.check_order(loss_order)
    assert not result.passed
    assert result.metadata["violation"] == RiskViolationType.LOSS_LIMIT

@pytest.mark.asyncio
async def test_drawdown_protection(risk_manager, market_order):
    """Test drawdown protection functionality."""
    # Initial balance is 100k, max drawdown is 10% = 10k
    
    # First, take a loss that would trigger the drawdown protection
    loss_order = market_order.copy()
    loss_order.side = OrderSide.SELL
    
    # Buy high, sell low
    await risk_manager.on_order_fill(market_order, Decimal("100"), Decimal("150.00"))
    
    # Update price down to cause a large unrealized loss
    await risk_manager.update_market_data("AAPL", Decimal("50.00"))
    
    # Check if circuit breaker is triggered
    assert risk_manager.circuit_breaker_triggered
    
    # Try to place a new order - should be rejected
    result = await risk_manager.check_order(market_order)
    assert not result.passed
    assert result.metadata["violation"] == RiskViolationType.CIRCUIT_BREAKER
    
    # Reset circuit breaker
    await risk_manager.reset_circuit_breaker()
    assert not risk_manager.circuit_breaker_triggered
    
    # Now orders should be allowed again
    result = await risk_manager.check_order(market_order)
    assert result.passed

@pytest.mark.asyncio
async def test_volatility_adjustment(risk_manager, market_order):
    """Test volatility-based position sizing."""
    # Add some price history to calculate volatility
    prices = [100.00, 101.00, 99.00, 102.00, 98.00, 103.00, 97.00, 104.00, 96.00, 105.00]
    
    for price in prices:
        await risk_manager.update_market_data("AAPL", Decimal(str(price)))
    
    # Check volatility - should be high given the price swings
    assert "AAPL" in risk_manager.volatility
    assert risk_manager.volatility["AAPL"] > Decimal("0.02")  # At least 2% volatility
    
    # Try to place a large order - should be rejected due to high volatility
    large_order = market_order.copy()
    large_order.quantity = Decimal("1000")
    
    result = await risk_manager.check_order(large_order)
    assert not result.passed
    assert result.metadata["violation"] == RiskViolationType.VOLATILITY_LIMIT
    
    # A smaller order should be allowed
    small_order = market_order.copy()
    small_order.quantity = Decimal("10")
    
    result = await risk_manager.check_order(small_order)
    assert result.passed

@pytest.mark.asyncio
async def test_daily_reset(risk_manager, market_order):
    """Test daily reset functionality."""
    # Take some trades
    await risk_manager.on_order_fill(market_order, Decimal("100"), Decimal("150.00"))
    await risk_manager.update_market_data("AAPL", Decimal("155.00"))
    
    # Verify state before reset
    assert risk_manager.daily_pnl != Decimal("0")
    assert risk_manager.daily_high_watermark > risk_manager.initial_balance
    
    # Reset
    await risk_manager.reset_daily()
    
    # Verify state after reset
    assert risk_manager.daily_pnl == Decimal("0")
    assert risk_manager.daily_high_watermark == risk_manager.equity
    assert risk_manager.daily_low_watermark == risk_manager.equity
    assert risk_manager.max_daily_drawdown == Decimal("0")
    assert not risk_manager.circuit_breaker_triggered
    assert risk_manager.circuit_breaker_time is None

@pytest.mark.asyncio
async def test_concurrent_access(risk_manager, market_order):
    """Test that the risk manager handles concurrent access safely."""
    # Create multiple tasks that will try to update the risk manager concurrently
    async def update_price(price):
        await risk_manager.update_market_data("AAPL", Decimal(str(price)))
        return True
    
    async def place_order(quantity, price):
        order = market_order.copy()
        order.quantity = Decimal(str(quantity))
        result = await risk_manager.check_order(order)
        if result.passed:
            await risk_manager.on_order_fill(order, Decimal(str(quantity)), Decimal(str(price)))
        return result.passed
    
    # Run multiple updates and orders concurrently
    tasks = []
    for i in range(10):
        tasks.append(update_price(150 + i))
        tasks.append(place_order(10, 150 + i))
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Verify no exceptions were raised
    assert not any(isinstance(r, Exception) for r in results)
    
    # Verify final state makes sense
    assert "AAPL" in risk_manager.positions
    position = risk_manager.positions["AAPL"]
    assert position.quantity > 0  # At least some orders should have been filled
    assert position.avg_price is not None

@pytest.mark.asyncio
async def test_get_risk_metrics(risk_manager, market_order):
    """Test the get_risk_metrics method."""
    # Initial state
    metrics = risk_manager.get_risk_metrics()
    assert metrics["equity"] == 100000.0
    assert metrics["daily_pnl"] == 0.0
    assert metrics["circuit_breaker_triggered"] is False
    assert metrics["circuit_breaker_time"] is None
    assert metrics["positions"] == {}
    
    # Take a position
    await risk_manager.on_order_fill(market_order, Decimal("100"), Decimal("150.00"))
    await risk_manager.update_market_data("AAPL", Decimal("155.00"))
    
    # Check metrics after position
    metrics = risk_manager.get_risk_metrics()
    assert "AAPL" in metrics["positions"]
    position = metrics["positions"]["AAPL"]
    assert position["quantity"] == 100.0
    assert position["avg_price"] == 150.0
    assert position["unrealized_pnl"] == 500.0  # 100 * (155 - 150)
    assert metrics["daily_pnl"] == 0.0  # No realized P&L yet
    
    # Check volatility metrics
    assert "AAPL" in metrics["volatility"]
    
    # Close position and check realized P&L
    sell_order = market_order.copy()
    sell_order.side = OrderSide.SELL
    await risk_manager.on_order_fill(sell_order, Decimal("100"), Decimal("160.00"))
    
    metrics = risk_manager.get_risk_metrics()
    position = metrics["positions"]["AAPL"]
    assert position["quantity"] == 0.0
    assert position["realized_pnl"] == 1000.0  # 100 * (160 - 150)
    assert metrics["daily_pnl"] == 1000.0

@pytest.mark.asyncio
async def test_negative_balance_protection(risk_manager, market_order):
    """Test that the risk manager prevents negative balance scenarios."""
    # Set a very small account balance
    risk_manager.account_balance = Decimal("1000.00")
    risk_manager.equity = Decimal("1000.00")
    
    # Try to place an order that would result in negative balance
    large_order = market_order.copy()
    large_order.quantity = Decimal("1000")  # 1000 shares at $150 = $150k > $1k balance
    
    result = await risk_manager.check_order(large_order)
    assert not result.passed
    assert result.metadata["violation"] in [
        RiskViolationType.LEVERAGE_LIMIT,
        RiskViolationType.POSITION_LIMIT
    ]
    
    # Test with margin requirements
    risk_manager.max_leverage = Decimal("100.0")  # Very high leverage for test
    
    # This should still be rejected due to position size limit
    result = await risk_manager.check_order(large_order)
    assert not result.passed
    assert result.metadata["violation"] == RiskViolationType.POSITION_LIMIT

@pytest.mark.asyncio
async def test_multiple_symbols(risk_manager, sample_instrument):
    """Test risk management with multiple symbols."""
    # Create instruments for different symbols
    msft_instrument = sample_instrument.copy()
    msft_instrument.symbol = "MSFT"
    
    googl_instrument = sample_instrument.copy()
    googl_instrument.symbol = "GOOGL"
    
    # Create orders for different symbols
    order_msft = Order(
        order_id="order_msft",
        symbol="MSFT",
        instrument=msft_instrument,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("50"),
        timestamp=datetime.utcnow()
    )
    
    order_googl = Order(
        order_id="order_googl",
        symbol="GOOGL",
        instrument=googl_instrument,
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("30"),
        timestamp=datetime.utcnow()
    )
    
    # Fill orders
    await risk_manager.on_order_fill(order_msft, Decimal("50"), Decimal("300.00"))  # $15k
    await risk_manager.on_order_fill(order_googl, Decimal("30"), Decimal("2800.00"))  # $84k
    
    # Update prices
    await risk_manager.update_market_data("MSFT", Decimal("310.00"))
    await risk_manager.update_market_data("GOOGL", Decimal("2750.00"))
    
    # Check positions
    assert "MSFT" in risk_manager.positions
    assert "GOOGL" in risk_manager.positions
    
    # Check P&L
    assert risk_manager.positions["MSFT"].unrealized_pnl == Decimal("500.00")  # 50 * (310 - 300)
    assert risk_manager.positions["GOOGL"].unrealized_pnl == Decimal("-1500.00")  # 30 * (2750 - 2800)
    
    # Check total equity
    expected_equity = (
        risk_manager.account_balance + 
        risk_manager.positions["MSFT"].unrealized_pnl + 
        risk_manager.positions["GOOGL"].unrealized_pnl
    )
    assert risk_manager.equity == expected_equity
    
    # Check risk metrics
    metrics = risk_manager.get_risk_metrics()
    assert set(metrics["positions"].keys()) == {"MSFT", "GOOGL"}
    assert set(metrics["volatility"].keys()) == {"MSFT", "GOOGL"}
