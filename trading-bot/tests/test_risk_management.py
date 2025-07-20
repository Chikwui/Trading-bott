"""
Tests for the risk management system.
"""
import pytest
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

from core.risk.position_manager import PositionManager, Position
from core.risk.exposure_calculator import ExposureCalculator, ExposureMetrics
from core.risk.position_sizing import PositionSizer, PositionSizingMethod

class TestPositionManager:
    """Test cases for PositionManager."""
    
    def test_position_creation(self):
        """Test creating and managing positions."""
        pm = PositionManager(account_balance=100000)
        
        # Test adding a position
        position = pm.add_position(
            symbol="AAPL", 
            quantity=100, 
            entry_price=150.0, 
            current_price=152.0,
            stop_loss=145.0,
            take_profit=160.0
        )
        
        assert position.symbol == "AAPL"
        assert position.quantity == 100
        assert position.entry_price == 150.0
        assert position.current_price == 152.0
        assert position.pnl == 200.0  # (152-150)*100
        
        # Test updating prices
        pm.update_prices({"AAPL": 155.0})
        assert pm.positions["AAPL"].current_price == 155.0
        assert pm.positions["AAPL"].pnl == 500.0
        
        # Test portfolio value calculation
        assert pm.get_portfolio_value() == 100500.0  # 100k + 500 P&L
        
    def test_position_sizing(self):
        """Test position size calculation."""
        pm = PositionManager(account_balance=100000)
        
        # Test basic position sizing
        size, risk = pm.calculate_position_size(
            symbol="AAPL",
            entry_price=150.0,
            stop_loss=145.0
        )
        
        # Expected: 2% of 100k = 2000 risk, 2000 / (150-145) = 400 shares
        assert abs(size - 400) < 1e-6
        assert abs(risk - 2000) < 1e-6


class TestExposureCalculator:
    """Test cases for ExposureCalculator."""
    
    def test_exposure_metrics(self):
        """Test exposure metric calculations."""
        pm = PositionManager(account_balance=100000)
        pm.add_position("AAPL", 100, 150.0, 152.0)
        pm.add_position("MSFT", -50, 300.0, 295.0)
        
        calculator = ExposureCalculator(pm)
        
        # Mock historical returns
        returns = {
            'AAPL': [0.01, -0.02, 0.015, -0.01, 0.02],
            'MSFT': [0.005, -0.01, 0.02, -0.015, 0.01]
        }
        calculator.historical_returns = pd.DataFrame(returns)
        
        # Calculate metrics
        metrics = calculator.calculate_exposure_metrics()
        
        # Basic exposure checks
        assert abs(metrics.total_exposure - 100000) < 1e-6
        assert abs(metrics.net_exposure - (15200 - 14750)) < 1e-6
        assert abs(metrics.gross_exposure - (15200 + 14750)) < 1e-6
        
        # Risk metrics should be calculated
        assert metrics.var_95 is not None
        assert metrics.cvar_95 is not None
        
        # Test liquidity metrics
        adv_data = {"AAPL": 1000000, "MSFT": 2000000}
        def impact_model(qty, adv):
            return abs(qty) / adv * 0.1  # Simple impact model
            
        liq_metrics = calculator.calculate_liquidity_metrics(adv_data, impact_model)
        assert "AAPL" in liq_metrics
        assert "MSFT" in liq_metrics
        assert liq_metrics["AAPL"]["adv_utilization"] == 100 / 1_000_000  # 100 shares / 1M ADV


class TestPositionSizer:
    """Test cases for PositionSizer."""
    
    def test_position_sizing_methods(self):
        """Test different position sizing methods."""
        sizer = PositionSizer(account_balance=100000)
        
        # Test fixed fractional sizing
        result = sizer.calculate_position_size(
            symbol="AAPL",
            entry_price=150.0,
            stop_loss=145.0,
            method=PositionSizingMethod.FIXED_FRACTIONAL
        )
        
        # 1% of 100k = 1000 risk, 1000 / (150-145) = 200 shares
        assert abs(result.size - 200) < 1e-6
        
        # Test volatility-adjusted sizing
        result = sizer.calculate_position_size(
            symbol="AAPL",
            entry_price=150.0,
            stop_loss=145.0,
            volatility=0.02,  # 2% daily vol
            method=PositionSizingMethod.VOLATILITY_ADJUSTED
        )
        
        # Should be smaller than fixed fractional due to volatility adjustment
        assert result.size <= 200
        
    def test_kelly_criterion(self):
        """Test Kelly Criterion position sizing."""
        sizer = PositionSizer(account_balance=100000)
        
        # High win probability should result in larger position
        result = sizer._apply_kelly_criterion(base_size=100, win_probability=0.7)
        assert result > 100
        
        # Low win probability should result in smaller position
        result = sizer._apply_kelly_criterion(base_size=100, win_probability=0.3)
        assert result < 100


if __name__ == "__main__":
    pytest.main(["-v", "test_risk_management.py"])
