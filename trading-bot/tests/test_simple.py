"""
A simple test file to verify the test environment is working correctly.
"""

def test_addition():
    """Test that 1 + 1 equals 2."""
    assert 1 + 1 == 2

class TestSimple:
    """A simple test class to verify test discovery."""
    
    def test_subtraction(self):
        """Test that 2 - 1 equals 1."""
        assert 2 - 1 == 1

    def test_multiplication(self):
        """Test that 2 * 2 equals 4."""
        assert 2 * 2 == 4
