"""
Comprehensive test suite for PositionManager.
"""
import pytest
from datetime import datetime, time, timedelta
from decimal import Decimal
import logging
from unittest.mock import patch, MagicMock

from core.risk.position_manager import Position, PositionManager
from core.risk.validation import ValidationError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestPositionManager:
    """Test cases for PositionManager."""
    
    @pytest.fixture
    def position_manager(self):
        """Create a PositionManager instance for testing."""
        return PositionManager(
            account_balance=100000,  # $100,000 account
            risk_per_trade=0.02,     # 2% risk per trade
            max_position_size=0.1,   # 10% of account per position
            max_position_value=20000, # $20k max position value
            allowed_instruments={"AAPL", "MSFT", "GOOGL"},
            trading_hours={
                "AAPL": (time(9, 30), time(16, 0)),  # 9:30 AM - 4:00 PM
                "MSFT": (time(9, 30), time(16, 0)),
                "GOOGL": (time(9, 30), time(16, 0)),
            }
        )
    
    @pytest.fixture
    def sample_position(self):
        """Create a sample position for testing."""
        return Position(
            symbol="AAPL",
            quantity=Decimal("10"),
            entry_price=Decimal("150.50"),
            current_price=Decimal("152.75"),
            stop_loss=Decimal("145.00"),
            take_profit=Decimal("160.00"),
            meta={"strategy": "mean_reversion"}
        )
    
    def test_initialization(self, position_manager):
        """Test PositionManager initialization."""
        assert position_manager.account_balance == Decimal('100000')
        assert position_manager.risk_per_trade == Decimal('0.02')
        assert position_manager.max_position_size == Decimal('0.1')
        assert position_manager.max_position_value == Decimal('20000')
        assert len(position_manager.get_positions()) == 0
    
    def test_add_position(self, position_manager, sample_position):
        """Test adding a new position."""
        # Add a position
        position = position_manager.add_position(
            symbol="AAPL",
            quantity=10,
            entry_price=150.50,
            current_price=152.75,
            stop_loss=145.00,
            take_profit=160.00,
            meta={"strategy": "mean_reversion"}
        )
        
        # Verify position was added
        assert position.symbol == "AAPL"
        assert position.quantity == Decimal('10')
        assert position.entry_price == Decimal('150.50')
        assert position.current_price == Decimal('152.75')
        assert position.stop_loss == Decimal('145.00')
        assert position.take_profit == Decimal('160.00')
        assert position.meta["strategy"] == "mean_reversion"
        
        # Verify position is in the manager
        assert len(position_manager.get_positions()) == 1
        assert "AAPL" in position_manager.get_positions()
    
    def test_position_validation(self, position_manager):
        """Test position validation."""
        # Test invalid symbol (not in allowed_instruments)
        with pytest.raises(ValidationError):
            position_manager.add_position(
                symbol="INVALID",
                quantity=10,
                entry_price=100,
                current_price=100
            )
        
        # Test position size exceeds max position value
        with pytest.raises(ValidationError):
            position_manager.add_position(
                symbol="AAPL",
                quantity=200,  # 200 * 150 = $30,000 > $20,000 max
                entry_price=150,
                current_price=150
            )
    
    def test_calculate_position_size(self, position_manager):
        """Test position size calculation."""
        # Test long position
        size, risk = position_manager.calculate_position_size(
            symbol="AAPL",
            entry_price=150,
            stop_loss=145  # $5 risk per share
        )
        # Expected: 2% of $100,000 = $2,000 risk / $5 = 400 shares
        # But limited by max position size (10% of $100,000 = $10,000 / $150 ≈ 66 shares)
        assert abs(float(size) - 66.6667) < 0.1
        assert risk == Decimal('2000')
        
        # Test short position
        size, _ = position_manager.calculate_position_size(
            symbol="AAPL",
            entry_price=145,
            stop_loss=150  # $5 risk per share (short)
        )
        assert size < 0  # Negative for short position
    
    def test_update_prices(self, position_manager):
        """Test updating position prices."""
        # Add a position
        position_manager.add_position(
            symbol="AAPL",
            quantity=10,
            entry_price=150,
            current_price=152,
            stop_loss=145,
            take_profit=160
        )
        
        # Update prices
        position_manager.update_prices({"AAPL": 153.25})
        
        # Verify price was updated
        position = position_manager.get_position("AAPL")
        assert position.current_price == Decimal('153.25')
        
        # Verify P&L calculation
        assert position.unrealized_pnl == Decimal('32.50')  # 10 * (153.25 - 150)
    
    def test_stop_loss_take_profit(self, position_manager):
        """Test stop loss and take profit triggers."""
        # Add a long position
        position_manager.add_position(
            symbol="AAPL",
            quantity=10,
            entry_price=150,
            current_price=152,
            stop_loss=145,
            take_profit=160
        )
        
        # Test stop loss trigger
        with patch.object(position_manager, 'close_position') as mock_close:
            position_manager.update_prices({"AAPL": 144})  # Below stop loss
            mock_close.assert_called_once_with("AAPL", None, "stop_loss")
        
        # Reset and test take profit
        position_manager = self.position_manager()
        position_manager.add_position(
            symbol="AAPL",
            quantity=10,
            entry_price=150,
            current_price=152,
            stop_loss=145,
            take_profit=160
        )
        
        with patch.object(position_manager, 'close_position') as mock_close:
            position_manager.update_prices({"AAPL": 161})  # Above take profit
            mock_close.assert_called_once_with("AAPL", None, "take_profit")
    
    def test_get_exposure(self, position_manager):
        """Test exposure calculation."""
        # Add some positions
        position_manager.add_position("AAPL", 10, 150, 152)
        position_manager.add_position("MSFT", -5, 300, 295)  # Short position
        
        # Get exposure
        exposure = position_manager.get_exposure()
        
        # Verify exposure metrics
        assert exposure['total_value'] > 100000  # Account + P&L
        assert exposure['total_exposure'] > 0
        assert exposure['net_exposure'] < exposure['total_exposure']  # Net < Gross (due to short)
        assert "AAPL" in exposure['exposure_per_symbol']
        assert "MSFT" in exposure['exposure_per_symbol']
        assert 'risk_metrics' in exposure
    
    def test_close_position(self, position_manager):
        """Test closing a position."""
        # Add a position
        position_manager.add_position("AAPL", 10, 150, 152)
        
        # Close the position
        closed_position = position_manager.close_position("AAPL", 153)
        
        # Verify position was closed
        assert closed_position is not None
        assert closed_position.current_price == Decimal('153')
        assert "AAPL" not in position_manager.get_positions()
        
        # Verify account balance was updated
        assert position_manager.account_balance > 100000  # Should have made a profit
    
    def test_trading_hours_validation(self, position_manager):
        """Test trading hours validation."""
        # Create a time outside trading hours (4:30 PM)
        after_hours = datetime.now().replace(hour=16, minute=30, second=0, microsecond=0)
        
        # Try to add a position outside trading hours
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.now.return_value = after_hours
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)
            
            with pytest.raises(ValidationError):
                position_manager.add_position(
                    symbol="AAPL",
                    quantity=10,
                    entry_price=150,
                    current_price=150,
                    current_time=after_hours
                )
    
    def test_position_update_validation(self, position_manager):
        """Test validation when updating positions."""
        # Add a position
        position_manager.add_position("AAPL", 10, 150, 152)
        
        # Try to update with invalid quantity (exceeds max position size)
        with pytest.raises(ValidationError):
            position_manager.update_position(
                symbol="AAPL",
                quantity=200,  # Would exceed max position value
                current_price=150
            )
        
        # Verify the position wasn't updated
        position = position_manager.get_position("AAPL")
        assert position.quantity == Decimal('10')

if __name__ == "__main__":
    pytest.main(["-v", "test_position_manager.py"])
