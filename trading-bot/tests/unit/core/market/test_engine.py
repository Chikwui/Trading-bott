"""
Unit tests for the TradingEngine class.

This module contains comprehensive tests for the TradingEngine class,
ensuring all functionality works as expected in various scenarios.
"""
from __future__ import annotations

import json
import logging
import queue
import time
from datetime import datetime, time as dt_time, timezone, timedelta
from unittest.mock import MagicMock, patch, ANY, call

import pytest
from freezegun import freeze_time

from core.instruments.metadata import InstrumentMetadata, AssetClass, InstrumentType
from core.market.engine import TradingEngine, TradingStatus, ExecutionHandler, RiskManager, DataHandler, PortfolioManager
from core.calendar import MarketCalendar, CalendarError, CalendarFactory

# Configure test logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Test data
TEST_SYMBOL = "AAPL"
TEST_INSTRUMENT = InstrumentMetadata(
    symbol=TEST_SYMBOL,
    name="Apple Inc.",
    asset_class=AssetClass.STOCK,
    exchange="NASDAQ",
    base_currency="USD",
    quote_currency="USD",
    lot_size=1.0,
    min_lot_size=1.0,
    max_lot_size=10000.0,
    tick_size=0.01,
    tick_value=1.0,
    margin_required=0.0,
    leverage=1.0
)

# Fixtures
@pytest.fixture
def mock_calendar():
    """Create a mock market calendar for testing."""
    calendar = MagicMock(spec=MarketCalendar)
    calendar.is_session_open.return_value = True
    calendar.next_trading_day.return_value = datetime.now(timezone.utc).date() + timedelta(days=1)
    calendar.schedule = MagicMock()
    
    # Add the missing methods that are expected by the tests
    calendar.add_instrument = MagicMock()
    calendar.remove_instrument = MagicMock()
    
    # Mock the schedule DataFrame-like access
    mock_schedule = MagicMock()
    mock_loc = MagicMock()
    mock_loc.__getitem__.return_value = {
        'market_open': datetime.now(timezone.utc) + timedelta(hours=1)
    }
    calendar.schedule.loc = mock_loc
    
    return calendar

@pytest.fixture
def mock_data_handler():
    """Create a mock data handler for testing."""
    handler = MagicMock(spec=DataHandler)
    handler.get_latest_bar.return_value = {
        'open': 150.0,
        'high': 151.0,
        'low': 149.5,
        'close': 150.5,
        'volume': 1000,
        'timestamp': datetime.now(timezone.utc)
    }
    return handler

@pytest.fixture
def mock_execution_handler():
    """Create a mock execution handler for testing."""
    handler = MagicMock(spec=ExecutionHandler)
    handler.execute_order.return_value = True
    return handler

@pytest.fixture
def mock_risk_manager():
    """Create a mock risk manager for testing."""
    manager = MagicMock(spec=RiskManager)
    manager.check_order_risk.return_value = True
    return manager

@pytest.fixture
def mock_portfolio():
    """Create a mock portfolio for testing."""
    portfolio = MagicMock(spec=PortfolioManager)
    portfolio.positions = {}
    portfolio.update_portfolio = MagicMock()
    portfolio.update_order = MagicMock()
    return portfolio

@pytest.fixture
def trading_engine(mock_calendar, mock_data_handler, mock_execution_handler, 
                    mock_risk_manager, mock_portfolio):
    """Create a TradingEngine instance with mocked dependencies for testing."""
    # Patch the CalendarFactory to return our mock calendar
    with patch('core.calendar.CalendarFactory.get_calendar', return_value=mock_calendar):
        engine = TradingEngine(
            data_handler=mock_data_handler,
            execution_handler=mock_execution_handler,
            portfolio=mock_portfolio,
            risk_manager=mock_risk_manager,
            calendar=mock_calendar,  # Pass the mock directly for testing
            timezone="UTC"
        )
        return engine

# Test cases
class TestTradingEngineInitialization:
    """Tests for TradingEngine initialization and basic properties."""
    
    def test_init_with_defaults(self, mock_calendar):
        """Test initialization with default parameters."""
        with patch('core.calendar.CalendarFactory.get_calendar', return_value=mock_calendar):
            engine = TradingEngine()
            assert engine is not None
            assert engine.is_running is False
            assert engine.timezone == "UTC"
            assert engine.calendar is not None
            assert engine.data_handler is None
            assert engine.execution_handler is None
            assert engine.risk_manager is None
            assert engine.portfolio is None
    
    def test_init_with_custom_params(self, mock_calendar, mock_data_handler, 
                                  mock_execution_handler, mock_risk_manager, mock_portfolio):
        """Test initialization with custom parameters."""
        with patch('core.calendar.CalendarFactory.get_calendar', return_value=mock_calendar):
            engine = TradingEngine(
                data_handler=mock_data_handler,
                execution_handler=mock_execution_handler,
                risk_manager=mock_risk_manager,
                portfolio=mock_portfolio,
                calendar=mock_calendar,
                timezone="America/New_York"
            )
            
            assert engine.timezone == "America/New_York"
            assert engine.calendar == mock_calendar
            assert engine.data_handler == mock_data_handler
            assert engine.execution_handler == mock_execution_handler
            assert engine.risk_manager == mock_risk_manager
            assert engine.portfolio == mock_portfolio
    
    def test_init_with_invalid_calendar(self):
        """Test initialization with an invalid calendar raises an error."""
        with patch('core.calendar.CalendarFactory.get_calendar', side_effect=CalendarError("Test error")):
            with pytest.raises(CalendarError, match="Test error"):
                TradingEngine(calendar=None)
    
    def test_init_with_default_calendar(self, mock_calendar):
        """Test initialization with default calendar creation."""
        with patch('core.calendar.CalendarFactory.get_calendar', return_value=mock_calendar) as mock_get_calendar:
            engine = TradingEngine()
            mock_get_calendar.assert_called_once()
            assert engine.calendar == mock_calendar
    
    def test_init_with_explicit_none_calendar(self, mock_calendar):
        """Test initialization with explicit None calendar uses default."""
        with patch('core.calendar.CalendarFactory.get_calendar', return_value=mock_calendar) as mock_get_calendar:
            engine = TradingEngine(calendar=None)
            mock_get_calendar.assert_called_once()
            assert engine.calendar == mock_calendar

class TestInstrumentManagement:
    """Tests for instrument management functionality."""
    
    def test_add_instrument(self, trading_engine):
        """Test adding an instrument to the trading engine."""
        trading_engine.add_instrument(TEST_INSTRUMENT)
        
        # Verify the instrument was added
        assert TEST_SYMBOL in trading_engine.instruments
        assert trading_engine.instruments[TEST_SYMBOL] == TEST_INSTRUMENT
        
        # Verify the instrument was registered with the calendar
        trading_engine.calendar.add_instrument.assert_called_once_with(TEST_INSTRUMENT)
    
    def test_add_instrument_with_existing_symbol(self, trading_engine, caplog):
        """Test adding an instrument with an existing symbol logs a warning."""
        # Add the instrument once
        trading_engine.add_instrument(TEST_INSTRUMENT)
        
        # Reset the mock to track only the second call
        trading_engine.calendar.add_instrument.reset_mock()
        
        # Add the same instrument again
        with caplog.at_level(logging.WARNING):
            trading_engine.add_instrument(TEST_INSTRUMENT)
            
            # Verify the warning was logged
            assert f"Instrument with symbol {TEST_SYMBOL} is already registered" in caplog.text
            
            # Verify the calendar's add_instrument was not called again
            trading_engine.calendar.add_instrument.assert_not_called()
    
    def test_remove_instrument(self, trading_engine):
        """Test removing an instrument from the trading engine."""
        # First add the instrument
        trading_engine.add_instrument(TEST_INSTRUMENT)
        assert TEST_SYMBOL in trading_engine.instruments
        
        # Reset the mock to track only the remove call
        trading_engine.calendar.add_instrument.reset_mock()
        
        # Now remove it
        result = trading_engine.remove_instrument(TEST_SYMBOL)
        
        # Verify the result and state
        assert result is True
        assert TEST_SYMBOL not in trading_engine.instruments
        trading_engine.calendar.remove_instrument.assert_called_once_with(TEST_SYMBOL)
    
    def test_remove_nonexistent_instrument(self, trading_engine):
        """Test removing a non-existent instrument returns False."""
        # Try to remove a non-existent instrument
        result = trading_engine.remove_instrument("NONEXISTENT")
        
        # Verify the result and that no calendar operations were performed
        assert result is False
        trading_engine.calendar.remove_instrument.assert_not_called()
    
    def test_get_instrument(self, trading_engine):
        """Test getting an instrument by symbol."""
        # Add the instrument
        trading_engine.add_instrument(TEST_INSTRUMENT)
        
        # Get the instrument
        instrument = trading_engine.get_instrument(TEST_SYMBOL)
        
        # Verify the result
        assert instrument == TEST_INSTRUMENT
    
    def test_get_nonexistent_instrument(self, trading_engine):
        """Test getting a non-existent instrument returns None."""
        # Try to get a non-existent instrument
        instrument = trading_engine.get_instrument("NONEXISTENT")
        
        # Verify the result
        assert instrument is None
    
    def test_has_instrument(self, trading_engine):
        """Test checking if an instrument exists."""
        # Initially should not have the instrument
        assert not trading_engine.has_instrument(TEST_SYMBOL)
        
        # Add the instrument
        trading_engine.add_instrument(TEST_INSTRUMENT)
        
        # Now should have the instrument
        assert trading_engine.has_instrument(TEST_SYMBOL)
    
    def test_list_instruments(self, trading_engine):
        """Test listing all instruments."""
        # Initially should be empty
        assert trading_engine.list_instruments() == []
        
        # Add an instrument
        trading_engine.add_instrument(TEST_INSTRUMENT)
        
        # Now should contain the instrument
        assert trading_engine.list_instruments() == [TEST_SYMBOL]

class TestMarketHours:
    """Tests for market hours and session management."""
    
    def test_is_market_open(self, trading_engine):
        """Test checking if market is open for a symbol."""
        # Add the instrument
        trading_engine.add_instrument(TEST_INSTRUMENT)
        
        # Test when market is open
        trading_engine.calendar.is_session_open.return_value = True
        assert trading_engine.is_market_open(TEST_SYMBOL) is True
        trading_engine.calendar.is_session_open.assert_called_once_with(
            symbol=TEST_SYMBOL, 
            dt=ANY  # Should be a datetime, but we don't care about the exact value
        )
        
        # Reset the mock and test when market is closed
        trading_engine.calendar.is_session_open.reset_mock()
        trading_engine.calendar.is_session_open.return_value = False
        assert trading_engine.is_market_open(TEST_SYMBOL) is False
    
    def test_is_market_open_nonexistent_symbol(self, trading_engine):
        """Test checking market status for a non-existent symbol."""
        with pytest.raises(ValueError, match=f"No instrument found with symbol: NONEXISTENT"):
            trading_engine.is_market_open("NONEXISTENT")
    
    def test_time_to_next_session(self, trading_engine):
        """Test calculating time to next trading session."""
        # Add the instrument
        trading_engine.add_instrument(TEST_INSTRUMENT)
        
        # Set up test data
        test_time = datetime(2023, 1, 1, 12, 0, tzinfo=timezone.utc)
        next_session = test_time + timedelta(hours=1)
        
        # Configure the mock to return our test data
        trading_engine.calendar.next_session.return_value = next_session
        
        with freeze_time(test_time):
            # Call the method
            seconds = trading_engine.time_to_next_session(TEST_SYMBOL)
            
            # Verify the result is as expected (within 1 second of 1 hour)
            assert abs(seconds - 3600) < 1
            
            # Verify the calendar was called correctly
            trading_engine.calendar.next_session.assert_called_once_with(
                symbol=TEST_SYMBOL,
                dt=test_time
            )
    
    def test_time_to_next_session_nonexistent_symbol(self, trading_engine):
        """Test time to next session for a non-existent symbol."""
        with pytest.raises(ValueError, match=f"No instrument found with symbol: NONEXISTENT"):
            trading_engine.time_to_next_session("NONEXISTENT")
    
    def test_get_trading_status(self, trading_engine):
        """Test getting trading status for a symbol."""
        # Add the instrument
        trading_engine.add_instrument(TEST_INSTRUMENT)
        
        # Set up test data
        test_time = datetime(2023, 1, 1, 12, 0, tzinfo=timezone.utc)
        next_session = test_time + timedelta(hours=1)
        
        # Configure the mocks
        trading_engine.calendar.is_session_open.return_value = True
        trading_engine.calendar.next_trading_day.return_value = next_session.date()
        
        with freeze_time(test_time):
            # Call the method
            status = trading_engine.get_trading_status(TEST_SYMBOL)
            
            # Verify the mocks were called correctly
            trading_engine.calendar.is_session_open.assert_called_once()
            trading_engine.calendar.next_trading_day.assert_called_once()
            
            # Verify the result
            assert status.symbol == TEST_SYMBOL
            assert status.is_market_open is True
            assert status.time_to_next_session == 3600  # 1 hour in seconds
            assert status.next_session_start == next_session
            assert status.current_time == test_time.isoformat()
            assert status.timezone == "UTC"
            
            # Verify the mocks were called correctly
            trading_engine.calendar.is_session_open.assert_called_once_with(
                symbol=TEST_SYMBOL,
                dt=test_time
            )
            trading_engine.calendar.next_session.assert_called_once_with(
                symbol=TEST_SYMBOL,
                dt=test_time
            )
    
    def test_get_trading_status_nonexistent_symbol(self, trading_engine):
        """Test getting trading status for a non-existent symbol."""
        with pytest.raises(ValueError, match=f"No instrument found with symbol: NONEXISTENT"):
            trading_engine.get_trading_status("NONEXISTENT")
    
    def test_get_market_hours(self, trading_engine):
        """Test getting market hours for a symbol."""
        # Add the instrument
        trading_engine.add_instrument(TEST_INSTRUMENT)
        
        # Set up test data
        test_date = datetime(2023, 1, 1).date()
        market_open = datetime(2023, 1, 1, 9, 30, tzinfo=timezone.utc)
        market_close = datetime(2023, 1, 1, 16, 0, tzinfo=timezone.utc)
        
        # Configure the mock
        trading_engine.calendar.get_market_hours.return_value = (market_open, market_close)
        
        # Call the method
        open_time, close_time = trading_engine.get_market_hours(TEST_SYMBOL, test_date)
        
        # Verify the result
        assert open_time == market_open
        assert close_time == market_close
        
        # Verify the mock was called correctly
        trading_engine.calendar.get_market_hours.assert_called_once_with(
            symbol=TEST_SYMBOL,
            date=test_date
        )
    
    def test_get_market_hours_nonexistent_symbol(self, trading_engine):
        """Test getting market hours for a non-existent symbol."""
        with pytest.raises(ValueError, match=f"No instrument found with symbol: NONEXISTENT"):
            trading_engine.get_market_hours("NONEXISTENT", datetime.now().date())

class TestOrderManagement:
    """Tests for order submission and management."""
    
    def test_submit_order(self, trading_engine, mock_risk_manager, mock_execution_handler):
        """Test submitting a new order successfully."""
        # Set up test data
        order = {
            'symbol': TEST_SYMBOL,
            'type': 'limit',
            'side': 'buy',
            'quantity': 100,
            'price': 150.0
        }
        
        # Configure mocks
        mock_risk_manager.check_order_risk.return_value = True
        mock_execution_handler.execute_order.return_value = {
            'order_id': '12345',
            'status': 'submitted',
            'filled_quantity': 0,
            'remaining_quantity': 100,
            'avg_fill_price': 0.0
        }
        
        # Submit the order
        result = trading_engine.submit_order(order)
        
        # Verify the result
        assert result['order_id'] == '12345'
        assert result['status'] == 'submitted'
        
        # Verify the mocks were called correctly
        mock_risk_manager.check_order_risk.assert_called_once()
        
        # Check the call to execute_order includes our order with any additional fields
        call_args = mock_execution_handler.execute_order.call_args[0][0]
        assert call_args['symbol'] == TEST_SYMBOL
        assert call_args['quantity'] == 100
        assert call_args['price'] == 150.0
        assert 'timestamp' in call_args  # Should be added by submit_order
        assert 'order_id' in call_args   # Should be added by submit_order

    def test_submit_order_risk_check_fails(self, trading_engine, mock_risk_manager, mock_execution_handler):
        """Test order submission when risk check fails."""
        # Set up test data
        order = {
            'symbol': TEST_SYMBOL,
            'type': 'limit',
            'side': 'buy',
            'quantity': 1000000,  # Unusually large order that should fail risk check
            'price': 150.0
        }
        
        # Configure mocks
        mock_risk_manager.check_order_risk.return_value = False
        
        # Submit the order
        with pytest.raises(ValueError, match="Order failed risk check"):
            trading_engine.submit_order(order)
        
        # Verify the mocks were called correctly
        mock_risk_manager.check_order_risk.assert_called_once()
        mock_execution_handler.execute_order.assert_not_called()

    def test_submit_order_invalid_symbol(self, trading_engine, mock_risk_manager, mock_execution_handler):
        """Test submitting an order for a symbol that hasn't been registered."""
        # Set up test data with an unregistered symbol
        order = {
            'symbol': 'UNREGISTERED',
            'type': 'limit',
            'side': 'buy',
            'quantity': 100,
            'price': 150.0
        }
        
        # Submit the order - should raise an exception
        with pytest.raises(ValueError, match="No instrument found with symbol: UNREGISTERED"):
            trading_engine.submit_order(order)
        
        # Verify no risk check or execution was attempted
        mock_risk_manager.check_order_risk.assert_not_called()
        mock_execution_handler.execute_order.assert_not_called()

    def test_cancel_order(self, trading_engine, mock_execution_handler):
        """Test canceling an existing order."""
        # Set up test data
        order_id = "12345"
        
        # Configure mock
        mock_execution_handler.cancel_order.return_value = {
            'order_id': order_id,
            'status': 'cancelled'
        }
        
        # Cancel the order
        result = trading_engine.cancel_order(order_id)
        
        # Verify the result
        assert result['order_id'] == order_id
        assert result['status'] == 'cancelled'
        
        # Verify the mock was called correctly
        mock_execution_handler.cancel_order.assert_called_once_with(order_id)

    def test_cancel_nonexistent_order(self, trading_engine, mock_execution_handler):
        """Test canceling an order that doesn't exist."""
        # Set up test data
        order_id = "NONEXISTENT"
        
        # Configure mock to raise an exception
        mock_execution_handler.cancel_order.side_effect = ValueError("Order not found")
        
        # Cancel the order - should raise an exception
        with pytest.raises(ValueError, match="Order not found"):
            trading_engine.cancel_order(order_id)
        
        # Verify the mock was called correctly
        mock_execution_handler.cancel_order.assert_called_once_with(order_id)

    def test_modify_order(self, trading_engine, mock_risk_manager, mock_execution_handler):
        """Test modifying an existing order successfully."""
        # Set up test data
        order_id = "12345"
        updates = {
            'quantity': 50,
            'price': 155.0
        }
        
        # Configure mocks
        mock_risk_manager.check_order_risk.return_value = True
        mock_execution_handler.modify_order.return_value = {
            'order_id': order_id,
            'status': 'modified',
            'quantity': 50,
            'price': 155.0
        }
        
        # Modify the order
        result = trading_engine.modify_order(order_id, updates)
        
        # Verify the result
        assert result['order_id'] == order_id
        assert result['status'] == 'modified'
        assert result['quantity'] == 50
        assert result['price'] == 155.0
        
        # Verify the mocks were called correctly
        mock_risk_manager.check_order_risk.assert_called_once()
        mock_execution_handler.modify_order.assert_called_once_with(order_id, updates)

    def test_modify_order_risk_check_fails(self, trading_engine, mock_risk_manager, mock_execution_handler):
        """Test modifying an order when the risk check fails."""
        # Set up test data
        order_id = "12345"
        updates = {
            'quantity': 1000000,  # Unusually large quantity that should fail risk check
            'price': 155.0
        }
        
        # Configure mocks
        mock_risk_manager.check_order_risk.return_value = False
        
        # Modify the order - should raise an exception
        with pytest.raises(ValueError, match="Order modification failed risk check"):
            trading_engine.modify_order(order_id, updates)
        
        # Verify the mocks were called correctly
        mock_risk_manager.check_order_risk.assert_called_once()
        mock_execution_handler.modify_order.assert_not_called()

    def test_get_order_status(self, trading_engine, mock_execution_handler):
        """Test getting the status of an order."""
        # Set up test data
        order_id = "12345"
        expected_status = {
            'order_id': order_id,
            'symbol': TEST_SYMBOL,
            'status': 'filled',
            'filled_quantity': 100,
            'remaining_quantity': 0,
            'avg_fill_price': 150.0
        }
        
        # Configure mock
        mock_execution_handler.get_order_status.return_value = expected_status
        
        # Get the order status
        status = trading_engine.get_order_status(order_id)
        
        # Verify the result
        assert status == expected_status
        
        # Verify the mock was called correctly
        mock_execution_handler.get_order_status.assert_called_once_with(order_id)

    def test_list_orders(self, trading_engine, mock_execution_handler):
        """Test listing all orders."""
        # Set up test data
        expected_orders = [
            {'order_id': '12345', 'symbol': TEST_SYMBOL, 'status': 'filled'},
            {'order_id': '12346', 'symbol': TEST_SYMBOL, 'status': 'open'}
        ]
        
        # Configure mock
        mock_execution_handler.list_orders.return_value = expected_orders
        
        # List the orders
        orders = trading_engine.list_orders()
        
        # Verify the result
        assert len(orders) == 2
        assert orders[0]['order_id'] == '12345'
        assert orders[1]['symbol'] == TEST_SYMBOL
        
        # Verify the mock was called correctly
        mock_execution_handler.list_orders.assert_called_once()

class TestEventProcessing:
    """Tests for event processing functionality."""
    
    def test_process_events_empty(self, trading_engine):
        """Test processing events with an empty queue."""
        # Set up an empty event queue
        trading_engine.event_queue = queue.Queue()
        
        # Process events - should not raise any exceptions
        trading_engine._process_events()
        
        # Verify the queue is still empty
        assert trading_engine.event_queue.empty()
    
    def test_process_market_data_event(self, trading_engine, mock_portfolio, mock_data_handler):
        """Test processing a market data event."""
        # Add the test instrument
        trading_engine.add_instrument(TEST_INSTRUMENT)
        
        # Create a market data event
        event = {
            'type': 'MARKET_DATA',
            'symbol': TEST_SYMBOL,
            'data': {
                'open': 149.0,
                'high': 151.0,
                'low': 148.5,
                'close': 150.0,
                'volume': 1000,
                'timestamp': datetime.now(timezone.utc).isoformat()
            }
        }
        
        # Configure mocks
        mock_data_handler.process_market_data.return_value = {
            'indicators': {'sma_20': 148.75, 'rsi_14': 55.3},
            'signals': [{'type': 'BUY', 'strength': 0.7}]
        }
        
        # Add the event to the queue and process it
        trading_engine.event_queue.put(event)
        trading_engine._process_events()
        
        # Verify the portfolio was updated with the processed data
        mock_portfolio.update_portfolio.assert_called_once()
        call_args = mock_portfolio.update_portfolio.call_args[0][0]
        assert call_args['symbol'] == TEST_SYMBOL
        assert 'indicators' in call_args
        assert 'signals' in call_args
        
        # Verify the data handler processed the market data
        mock_data_handler.process_market_data.assert_called_once_with(event['data'])
    
    def test_process_order_event(self, trading_engine, mock_portfolio, mock_execution_handler):
        """Test processing an order event."""
        # Create an order event
        event = {
            'type': 'ORDER',
            'order_id': '12345',
            'status': 'FILLED',
            'symbol': TEST_SYMBOL,
            'side': 'BUY',
            'quantity': 100,
            'filled_quantity': 100,
            'remaining_quantity': 0,
            'avg_fill_price': 150.0,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Configure mocks
        mock_execution_handler.get_order_status.return_value = event
        
        # Add the event to the queue and process it
        trading_engine.event_queue.put(event)
        trading_engine._process_events()
        
        # Verify the portfolio was updated with the order
        mock_portfolio.update_order.assert_called_once_with(event)
        
        # Verify the execution handler was called to get the latest status
        mock_execution_handler.get_order_status.assert_called_once_with('12345')
    
    def test_process_signal_event(self, trading_engine, mock_portfolio, mock_risk_manager):
        """Test processing a trading signal event."""
        # Create a signal event
        event = {
            'type': 'SIGNAL',
            'symbol': TEST_SYMBOL,
            'signal': 'BUY',
            'strength': 0.8,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'source': 'strategy_ma_crossover',
            'confidence': 0.75
        }
        
        # Configure mocks
        mock_risk_manager.evaluate_signal.return_value = {
            'approved': True,
            'confidence': 0.8,
            'risk_score': 0.2
        }
        
        # Add the event to the queue and process it
        trading_engine.event_queue.put(event)
        trading_engine._process_events()
        
        # Verify the risk manager evaluated the signal
        mock_risk_manager.evaluate_signal.assert_called_once()
        
        # Verify the portfolio was updated with the signal
        mock_portfolio.process_signal.assert_called_once()
        call_args = mock_portfolio.process_signal.call_args[0][0]
        assert call_args['symbol'] == TEST_SYMBOL
        assert call_args['signal'] == 'BUY'
        assert 'timestamp' in call_args
    
    def test_process_error_event(self, trading_engine, caplog):
        """Test processing an error event."""
        # Create an error event
        error = ValueError("Test error")
        event = {
            'type': 'ERROR',
            'error': error,
            'source': 'data_handler',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Add the event to the queue and process it
        trading_engine.event_queue.put(event)
        
        with caplog.at_level(logging.ERROR):
            trading_engine._process_events()
            
            # Verify the error was logged
            assert "Error in data_handler" in caplog.text
            assert "Test error" in caplog.text
    
    def test_process_unknown_event_type(self, trading_engine, caplog):
        """Test processing an unknown event type logs a warning."""
        # Create an unknown event type
        event = {
            'type': 'UNKNOWN_EVENT',
            'data': {'key': 'value'},
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Add the event to the queue and process it
        trading_engine.event_queue.put(event)
        
        with caplog.at_level(logging.WARNING):
            trading_engine._process_events()
            
            # Verify the warning was logged
            assert "Unknown event type: UNKNOWN_EVENT" in caplog.text
    
    def test_process_event_with_exception(self, trading_engine, caplog):
        """Test that exceptions during event processing are caught and logged."""
        # Create a test event
        event = {
            'type': 'TEST_EVENT',
            'data': 'test',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Create a handler that raises an exception
        def failing_handler(event):
            raise ValueError("Handler failed")
            
        # Register the failing handler
        trading_engine._event_handlers['TEST_EVENT'] = [failing_handler]
        
        # Add the event to the queue and process it
        trading_engine.event_queue.put(event)
        
        with caplog.at_level(logging.ERROR):
            trading_engine._process_events()
            
            # Verify the error was logged
            assert "Error processing TEST_EVENT event" in caplog.text
            assert "Handler failed" in caplog.text

class TestEngineLifecycle:
    """Tests for engine startup and shutdown."""
    
    def test_start_engine(self, trading_engine, mock_data_handler, mock_execution_handler, 
                         mock_risk_manager, mock_portfolio):
        """Test starting the trading engine successfully."""
        # Configure mocks
        mock_data_handler.start.return_value = True
        mock_execution_handler.start.return_value = True
        mock_risk_manager.start.return_value = True
        mock_portfolio.start.return_value = True
        
        # Start the engine
        result = trading_engine.start()
        
        # Verify the result and state
        assert result is True
        assert trading_engine.is_running is True
        
        # Verify all components were started
        mock_data_handler.start.assert_called_once()
        mock_execution_handler.start.assert_called_once()
        mock_risk_manager.start.assert_called_once()
        mock_portfolio.start.assert_called_once()
        
        # Verify the engine status was logged
        # (This would typically be verified with a mock logger)
    
    def test_start_engine_component_failure(self, trading_engine, mock_data_handler, 
                                          mock_execution_handler, mock_risk_manager, 
                                          mock_portfolio, caplog):
        """Test starting the trading engine when a component fails to start."""
        # Configure mocks - data handler fails to start
        mock_data_handler.start.return_value = False
        mock_execution_handler.start.return_value = True
        mock_risk_manager.start.return_value = True
        mock_portfolio.start.return_value = True
        
        # Start the engine - should raise an exception
        with pytest.raises(RuntimeError, match="Failed to start all components"):
            trading_engine.start()
        
        # Verify the engine is not running
        assert trading_engine.is_running is False
        
        # Verify stop was called on any components that did start
        mock_execution_handler.stop.assert_called_once()
        mock_risk_manager.stop.assert_called_once()
        mock_portfolio.stop.assert_called_once()
    
    def test_stop_engine(self, trading_engine, mock_data_handler, mock_execution_handler, 
                        mock_risk_manager, mock_portfolio):
        """Test stopping the trading engine successfully."""
        # Start the engine first
        trading_engine.start()
        
        # Configure mocks for stop
        mock_data_handler.stop.return_value = True
        mock_execution_handler.stop.return_value = True
        mock_risk_manager.stop.return_value = True
        mock_portfolio.stop.return_value = True
        
        # Stop the engine
        result = trading_engine.stop()
        
        # Verify the result and state
        assert result is True
        assert trading_engine.is_running is False
        
        # Verify all components were stopped
        mock_data_handler.stop.assert_called_once()
        mock_execution_handler.stop.assert_called_once()
        mock_risk_manager.stop.assert_called_once()
        mock_portfolio.stop.assert_called_once()
    
    def test_stop_engine_not_running(self, trading_engine, caplog):
        """Test stopping an already stopped engine logs a warning."""
        with caplog.at_level(logging.WARNING):
            trading_engine.stop()
            assert "not running" in caplog.text
    
    def test_restart_engine(self, trading_engine, mock_data_handler, mock_execution_handler, 
                           mock_risk_manager, mock_portfolio):
        """Test restarting the trading engine."""
        # Configure mocks for first start
        mock_data_handler.start.return_value = True
        mock_execution_handler.start.return_value = True
        mock_risk_manager.start.return_value = True
        mock_portfolio.start.return_value = True
        
        # Start the engine
        trading_engine.start()
        
        # Configure mocks for stop
        mock_data_handler.stop.return_value = True
        mock_execution_handler.stop.return_value = True
        mock_risk_manager.stop.return_value = True
        mock_portfolio.stop.return_value = True
        
        # Stop the engine
        trading_engine.stop()
        
        # Reset mocks for restart
        mock_data_handler.start.reset_mock()
        mock_execution_handler.start.reset_mock()
        mock_risk_manager.start.reset_mock()
        mock_portfolio.start.reset_mock()
        
        # Configure mocks for second start
        mock_data_handler.start.return_value = True
        mock_execution_handler.start.return_value = True
        mock_risk_manager.start.return_value = True
        mock_portfolio.start.return_value = True
        
        # Restart the engine
        result = trading_engine.start()
        
        # Verify the result and state
        assert result is True
        assert trading_engine.is_running is True
        
        # Verify all components were started again
        mock_data_handler.start.assert_called_once()
        mock_execution_handler.start.assert_called_once()
        mock_risk_manager.start.assert_called_once()
        mock_portfolio.start.assert_called_once()
    
    def test_start_already_running(self, trading_engine, caplog):
        """Test starting an already running engine logs a warning."""
        # Start the engine
        trading_engine.start()
        
        # Try to start it again
        with caplog.at_level(logging.WARNING):
            trading_engine.start()
            assert "already running" in caplog.text
    
    def test_context_manager(self, trading_engine, mock_data_handler, mock_execution_handler, 
                           mock_risk_manager, mock_portfolio):
        """Test using the trading engine as a context manager."""
        # Configure mocks
        mock_data_handler.start.return_value = True
        mock_execution_handler.start.return_value = True
        mock_risk_manager.start.return_value = True
        mock_portfolio.start.return_value = True
        
        mock_data_handler.stop.return_value = True
        mock_execution_handler.stop.return_value = True
        mock_risk_manager.stop.return_value = True
        mock_portfolio.stop.return_value = True
        
        # Use the engine as a context manager
        with trading_engine:
            # Verify the engine is running
            assert trading_engine.is_running is True
            
            # Verify all components were started
            mock_data_handler.start.assert_called_once()
            mock_execution_handler.start.assert_called_once()
            mock_risk_manager.start.assert_called_once()
            mock_portfolio.start.assert_called_once()
        
        # Verify the engine is stopped
        assert trading_engine.is_running is False
        
        # Verify all components were stopped
        mock_data_handler.stop.assert_called_once()
        mock_execution_handler.stop.assert_called_once()
        mock_risk_manager.stop.assert_called_once()
        mock_portfolio.stop.assert_called_once()
    
    def test_engine_status(self, trading_engine):
        """Test getting the engine status."""
        # Get the status when not running
        status = trading_engine.get_status()
        assert status['status'] == 'stopped'
        assert 'uptime' in status
        assert 'components' in status
        
        # Start the engine and check status again
        trading_engine.start()
        status = trading_engine.get_status()
        assert status['status'] == 'running'
        assert 'uptime' in status
        assert 'components' in status
            
        # Verify the counters were reset
        assert trading_engine._events_processed == 0
        assert trading_engine._last_event_count_time <= time.time()

    def test_error_handling(self, trading_engine, caplog):
        """Test error handling in the trading engine."""
        # Test handling an error event
        error = ValueError("Test error")
        event = {
            'type': 'ERROR',
            'error': error,
            'source': 'test',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Process the error event
        with caplog.at_level(logging.ERROR):
            trading_engine._handle_error(event)
            assert "Error in test" in caplog.text
            assert "Test error" in caplog.text

class TestErrorHandling:
    """Tests for error handling and edge cases in the TradingEngine."""
    
    def test_handle_error_event(self, trading_engine, caplog):
        """Test handling an error event with a proper error object."""
        # Create a test error event
        error = ValueError("Test error")
        event = {
            'type': 'ERROR',
            'error': error,
            'source': 'test_component',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'details': {'key': 'value'}
        }
        
        # Process the error event
        with caplog.at_level(logging.ERROR):
            trading_engine._handle_error(event)
            
            # Verify the error was logged correctly
            assert "Error in test_component" in caplog.text
            assert "Test error" in caplog.text
            assert "key='value'" in caplog.text  # Check if details are included
    
    def test_handle_error_event_with_string_error(self, trading_engine, caplog):
        """Test handling an error event with a string error instead of an exception."""
        # Create an error event with a string error
        event = {
            'type': 'ERROR',
            'error': "Test error message",
            'source': 'test_component',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Process the error event
        with caplog.at_level(logging.ERROR):
            trading_engine._handle_error(event)
            assert "Error in test_component" in caplog.text
            assert "Test error message" in caplog.text
    
    def test_handle_error_event_missing_fields(self, trading_engine, caplog):
        """Test handling an error event with missing fields."""
        # Create a minimal error event
        event = {
            'type': 'ERROR',
            'error': "Test error",
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Process the error event
        with caplog.at_level(logging.ERROR):
            trading_engine._handle_error(event)
            assert "Error in unknown source" in caplog.text
    
    def test_handle_error_event_invalid_type(self, trading_engine, caplog):
        """Test handling an error event with an invalid type."""
        # Create an invalid error event
        event = "not a dictionary"
        
        # Process the error event
        with caplog.at_level(logging.ERROR):
            trading_engine._handle_error(event)
            assert "Invalid error event format" in caplog.text
    
    def test_handle_error_during_event_processing(self, trading_engine, caplog):
        """Test that exceptions during event processing are caught and logged."""
        # Create a test event that will cause an error
        event = {
            'type': 'TEST_ERROR_EVENT',
            'data': 'test',
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Create a handler that raises an exception
        def failing_handler(event):
            raise ValueError("Handler failed")
            
        # Register the failing handler
        trading_engine._event_handlers['TEST_ERROR_EVENT'] = [failing_handler]
        
        # Add the event to the queue and process it
        trading_engine.event_queue.put(event)
        
        with caplog.at_level(logging.ERROR):
            trading_engine._process_events()
            
            # Verify the error was logged
            assert "Error processing TEST_ERROR_EVENT event" in caplog.text
            assert "Handler failed" in caplog.text
    
    def test_handle_market_data_error(self, trading_engine, mock_data_handler, caplog):
        """Test handling an error during market data processing."""
        # Configure the data handler to raise an exception
        mock_data_handler.process_market_data.side_effect = Exception("Data processing failed")
        
        # Create a market data event
        event = {
            'type': 'MARKET_DATA',
            'symbol': TEST_SYMBOL,
            'data': {'price': 150.0, 'volume': 1000},
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Process the event
        with caplog.at_level(logging.ERROR):
            trading_engine._handle_market_data(event)
            assert "Error processing market data" in caplog.text
    
    def test_handle_order_event_error(self, trading_engine, mock_portfolio, caplog):
        """Test handling an error during order event processing."""
        # Configure the portfolio to raise an exception
        mock_portfolio.update_order.side_effect = Exception("Order update failed")
        
        # Create an order event
        event = {
            'type': 'ORDER',
            'order_id': '12345',
            'status': 'FILLED',
            'symbol': TEST_SYMBOL,
            'quantity': 100,
            'filled_quantity': 100,
            'price': 150.0,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Process the event
        with caplog.at_level(logging.ERROR):
            trading_engine._handle_order_event(event)
            assert "Error processing order event" in caplog.text
    
    def test_handle_signal_event_error(self, trading_engine, mock_risk_manager, caplog):
        """Test handling an error during signal processing."""
        # Configure the risk manager to raise an exception
        mock_risk_manager.evaluate_signal.side_effect = Exception("Signal evaluation failed")
        
        # Create a signal event
        event = {
            'type': 'SIGNAL',
            'symbol': TEST_SYMBOL,
            'signal': 'BUY',
            'strength': 0.8,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Process the event
        with caplog.at_level(logging.ERROR):
            trading_engine._handle_signal(event)
            assert "Error processing signal event" in caplog.text
    
    def test_handle_error_event(self, trading_engine, caplog):
        """Test handling an error event."""
        error_event = {
            'type': 'ERROR',
            'error_type': 'DATA_ERROR',
            'message': 'Test error',
            'source': 'DataHandler',
            'severity': 'error',
            'details': {'key': 'value'}
        }
        trading_engine.event_queue = queue.Queue()
        trading_engine.event_queue.put(error_event)
        
        with caplog.at_level(logging.ERROR):
            trading_engine._process_events()
            assert "DATA_ERROR" in caplog.text
            assert "Test error" in caplog.text
            
    def test_handle_critical_error(self, trading_engine, caplog):
        """Test handling a critical error event."""
        error_event = {
            'type': 'ERROR',
            'error_type': 'FATAL',
            'message': 'Critical error',
            'source': 'RiskManager',
            'severity': 'critical'
        }
        trading_engine.event_queue = queue.Queue()
        trading_engine.event_queue.put(error_event)
        
        with caplog.at_level(logging.CRITICAL):
            trading_engine._process_events()
            assert "Critical error detected" in caplog.text

if __name__ == "__main__":
    pytest.main(["-v", "--cov=core.market.engine", "--cov-report=term-missing", __file__])
