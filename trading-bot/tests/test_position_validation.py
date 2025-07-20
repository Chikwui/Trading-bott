"""
Tests for position validation.
"""
import pytest
from datetime import datetime, time
from decimal import Decimal

from core.risk.validation import PositionValidator, ValidationError

class TestPositionValidator:
    """Test cases for position validation."""
    
    @pytest.fixture
    def validator(self):
        """Create a validator with test parameters."""
        return PositionValidator(
            max_position_size=1000,  # Max 1000 units
            max_position_value=100000,  # Max $100k per position
            allowed_instruments={"AAPL", "MSFT", "GOOGL"},
            trading_hours={
                "AAPL": (time(9, 30), time(16, 0)),  # 9:30 AM - 4:00 PM
                "MSFT": (time(9, 30), time(16, 0)),
            }
        )
    
    def test_allowed_instrument(self, validator):
        """Test validation with allowed instruments."""
        # Valid symbol
        is_valid, reason = validator.validate_new_position("AAPL", 100, 150.0)
        assert is_valid is True
        assert reason == ""
        
        # Invalid symbol
        is_valid, reason = validator.validate_new_position("TSLA", 100, 150.0)
        assert is_valid is False
        assert "not allowed" in reason
    
    def test_position_size_limits(self, validator):
        """Test position size validation."""
        # At limit
        is_valid, _ = validator.validate_new_position("AAPL", 1000, 100.0)
        assert is_valid is True
        
        # Over limit
        is_valid, reason = validator.validate_new_position("AAPL", 1001, 100.0)
        assert is_valid is False
        assert "exceeds maximum" in reason
        
        # Negative position (short)
        is_valid, _ = validator.validate_new_position("AAPL", -1000, 100.0)
        assert is_valid is True
    
    def test_position_value_limits(self, validator):
        """Test position value validation."""
        # At value limit (1000 * 100 = 100,000)
        is_valid, _ = validator.validate_new_position("AAPL", 1000, 100.0)
        assert is_valid is True
        
        # Over value limit
        is_valid, reason = validator.validate_new_position("AAPL", 1000, 101.0)
        assert is_valid is False
        assert "exceeds maximum allowed value" in reason
    
    def test_account_balance_check(self, validator):
        """Test validation against account balance."""
        # 90% of 50,000 = 45,000 max position value
        account_balance = 50000
        
        # Valid (40,000 < 45,000)
        is_valid, _ = validator.validate_new_position(
            "AAPL", 400, 100.0, account_balance=account_balance
        )
        assert is_valid is True
        
        # Invalid (50,000 > 45,000)
        is_valid, reason = validator.validate_new_position(
            "AAPL", 500, 100.0, account_balance=account_balance
        )
        assert is_valid is False
        assert "exceeds 90% of account balance" in reason
    
    def test_trading_hours(self, validator, monkeypatch):
        """Test trading hours validation."""
        # Mock current time to 10:00 AM
        class MockDatetime:
            @classmethod
            def now(cls):
                return datetime(2023, 1, 1, 10, 0)  # Sunday, 10:00 AM
        
        monkeypatch.setattr('datetime.datetime', MockDatetime)
        
        # During trading hours
        is_valid, _ = validator.validate_new_position("AAPL", 100, 150.0)
        assert is_valid is True
        
        # Outside trading hours
        MockDatetime.now = lambda: datetime(2023, 1, 1, 8, 0)  # 8:00 AM
        is_valid, reason = validator.validate_new_position("AAPL", 100, 150.0)
        assert is_valid is False
        assert "Outside trading hours" in reason
        
        # No trading hours defined
        is_valid, _ = validator.validate_new_position("GOOGL", 100, 150.0)
        assert is_valid is True
    
    def test_aggregate_exposure(self, validator):
        """Test aggregate exposure validation."""
        # Current positions: AAPL $50k, MSFT $30k
        current_positions = {
            "AAPL": {"quantity": 500, "price": 100.0},
            "MSFT": {"quantity": 200, "price": 150.0},
        }
        
        # New position that would take total to $100k (under 5x $100k limit)
        is_valid, _ = validator.validate_new_position(
            "GOOGL", 100, 200.0, current_positions=current_positions
        )
        assert is_valid is True
        
        # New position that would exceed the limit
        is_valid, reason = validator.validate_new_position(
            "GOOGL", 2100, 200.0, current_positions=current_positions
        )
        assert is_valid is False
        assert "exceed aggregate exposure" in reason
    
    def test_invalid_inputs(self, validator):
        """Test validation with invalid inputs."""
        # Invalid quantity
        is_valid, reason = validator.validate_new_position("AAPL", "invalid", 150.0)
        assert is_valid is False
        assert "Invalid input parameters" in reason
        
        # Missing price when needed
        is_valid, reason = validator.validate_new_position("AAPL", 100)
        assert is_valid is True  # Should pass basic validation without price
        
        # But should fail when account balance check is requested
        is_valid, reason = validator.validate_new_position(
            "AAPL", 100, account_balance=10000
        )
        assert is_valid is False
        assert "price is required" in reason.lower()


if __name__ == "__main__":
    pytest.main(["-v", "test_position_validation.py"])
