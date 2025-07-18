"""
Configuration and fixtures for integration tests with real dependencies.
"""
import os
import pytest
import asyncio
from decimal import Decimal
from typing import Dict, Any, Optional

# Test configuration
class TestConfig:
    # Exchange configuration
    EXCHANGE_API_KEY: str = os.getenv('EXCHANGE_API_KEY', 'test_key')
    EXCHANGE_SECRET: str = os.getenv('EXCHANGE_SECRET', 'test_secret')
    EXCHANGE_PASSPHRASE: Optional[str] = os.getenv('EXCHANGE_PASSPHRASE')
    
    # Test symbols (use testnet symbols when available)
    TEST_SYMBOL: str = 'BTC/USDT'
    TEST_QUANTITY: Decimal = Decimal('0.001')  # Small quantity for testing
    
    # Test mode (paper trading or real trading - should be paper for tests)
    PAPER_TRADING: bool = True
    
    # Timeouts and retries
    ORDER_TIMEOUT: int = 30  # seconds
    MAX_RETRIES: int = 3
    
    # Test account settings
    TEST_ACCOUNT_ID: str = 'test_integration_account'

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    policy = asyncio.get_event_loop_policy()
    loop = policy.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="module")
def test_config() -> TestConfig:
    """Return test configuration."""
    return TestConfig()

# Helper functions for test data
def create_test_order(
    symbol: str = TestConfig.TEST_SYMBOL,
    side: str = 'BUY',
    order_type: str = 'MARKET',
    quantity: Decimal = TestConfig.TEST_QUANTITY,
    price: Optional[Decimal] = None,
    **kwargs
) -> Dict[str, Any]:
    """Create a test order dictionary."""
    from datetime import datetime, timezone
    from uuid import uuid4
    
    order = {
        'order_id': str(uuid4()),
        'client_order_id': f'test_{int(datetime.now(timezone.utc).timestamp())}',
        'symbol': symbol,
        'side': side,
        'order_type': order_type,
        'quantity': str(quantity),
        'status': 'NEW',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        **kwargs
    }
    
    if price is not None:
        order['price'] = str(price)
    
    return order
