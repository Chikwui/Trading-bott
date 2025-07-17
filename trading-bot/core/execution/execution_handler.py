"""
Execution Handler

This module provides the main execution handler that manages order execution,
position tracking, and risk management for the trading system.
"""
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime
import logging

from .order import Order, OrderType, OrderSide, OrderStatus, OrderList
from .position import Position, PositionStatus, PositionManager, PositionUpdateResult
from .risk_manager import RiskManager
from .portfolio import Portfolio
from .broker import Broker, BrokerType
from .exceptions import (
    ExecutionError,
    InsufficientFundsError,
    InvalidOrderError,
    OrderNotFoundError,
    PositionNotFoundError,
    RiskCheckFailedError
)

# Configure logging
logger = logging.getLogger(__name__)


class ExecutionHandler:
    """
    Main execution handler for the trading system.
    
    This class is responsible for:
    1. Managing order lifecycle
    2. Tracking positions
    3. Managing portfolio and risk
    4. Interfacing with brokers
    5. Executing trades
    """
    
    def __init__(
        self, 
        broker: Broker,
        initial_capital: float = 100000.0,
        risk_free_rate: float = 0.0,
        max_position_size: float = 0.1,  # 10% of portfolio per position
        max_leverage: float = 1.0,      # No leverage by default
        max_drawdown: float = 0.2,      # 20% max drawdown
        max_orders_per_day: int = 10,   # Max number of orders per day
        enable_shorting: bool = True,
        enable_leverage: bool = False
    ):
        """
        Initialize the execution handler.
        
        Args:
            broker: Broker instance for order execution
            initial_capital: Initial trading capital
            risk_free_rate: Risk-free rate for performance calculations
            max_position_size: Maximum position size as a fraction of portfolio value
            max_leverage: Maximum allowed leverage (1.0 = no leverage)
            max_drawdown: Maximum allowed drawdown before risk limits are triggered
            max_orders_per_day: Maximum number of orders allowed per day
            enable_shorting: Whether short selling is allowed
            enable_leverage: Whether leverage is allowed
        """
        self.broker = broker
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        
        # Initialize components
        self.order_manager = OrderList()
        self.position_manager = PositionManager()
        self.portfolio = Portfolio(initial_capital, risk_free_rate)
        
        # Configure risk manager
        self.risk_manager = RiskManager(
            max_position_size=max_position_size,
            max_leverage=max_leverage,
            max_drawdown=max_drawdown,
            max_orders_per_day=max_orders_per_day,
            enable_shorting=enable_shorting,
            enable_leverage=enable_leverage
        )
        
        # Track order execution
        self.order_history = []
        self.trade_history = []
        self.daily_order_count = 0
        self.last_trade_day = datetime.now().date()
        
        # Performance metrics
        self.start_time = datetime.now()
        self.trades_executed = 0
        self.shares_traded = 0
        self.commissions_paid = 0.0
        
        logger.info(f"ExecutionHandler initialized with initial capital: ${initial_capital:,.2f}")
    
    def submit_order(self, order: Order) -> str:
        """
        Submit a new order for execution.
        
        Args:
            order: Order to submit
            
        Returns:
            Order ID of the submitted order
            
        Raises:
            InvalidOrderError: If the order is invalid
            RiskCheckFailedError: If the order fails risk checks
            InsufficientFundsError: If there are insufficient funds for the order
        """
        # Validate order
        self._validate_order(order)
        
        # Check risk limits
        self._check_risk_limits(order)
        
        # Add to order manager
        self.order_manager.add(order)
        
        # Update order status to PENDING_NEW
        order.status = OrderStatus.PENDING_NEW
        order.updated_at = datetime.utcnow()
        
        logger.info(f"Order submitted: {order.order_id} {order.side.name} {order.quantity} "
                   f"{order.symbol} @ {order.limit_price or 'MARKET'}")
        
        # Execute the order (in a real system, this would be asynchronous)
        self._execute_order(order)
        
        return order.order_id
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an existing order.
        
        Args:
            order_id: ID of the order to cancel
            
        Returns:
            True if the order was successfully canceled, False otherwise
            
        Raises:
            OrderNotFoundError: If the order is not found
        """
        order = self.order_manager.get(order_id)
        if not order:
            raise OrderNotFoundError(f"Order {order_id} not found")
        
        # Check if order can be canceled
        if not order.is_active():
            logger.warning(f"Cannot cancel order {order_id}: status is {order.status.name}")
            return False
        
        # Update order status
        order.status = OrderStatus.CANCELED
        order.updated_at = datetime.utcnow()
        
        # Notify broker to cancel the order
        try:
            self.broker.cancel_order(order_id)
            logger.info(f"Order canceled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to cancel order {order_id}: {str(e)}")
            return False
    
    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get the status of an order.
        
        Args:
            order_id: ID of the order to check
            
        Returns:
            Dictionary with order status information
            
        Raises:
            OrderNotFoundError: If the order is not found
        """
        order = self.order_manager.get(order_id)
        if not order:
            raise OrderNotFoundError(f"Order {order_id} not found")
        
        # In a real system, we might want to sync with the broker here
        # to get the latest status
        
        return order.to_dict()
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get the current position for a symbol.
        
        Args:
            symbol: Symbol to get position for
            
        Returns:
            Position information as a dictionary, or None if no position exists
        """
        positions = self.position_manager.get_positions_by_symbol(symbol)
        if not positions:
            return None
        
        # For simplicity, return the first position if there are multiple
        # In a real system, you might want to handle this differently
        return positions[0].to_dict()
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current portfolio.
        
        Returns:
            Dictionary with portfolio summary information
        """
        return {
            'total_value': self.portfolio.total_value,
            'cash': self.portfolio.cash,
            'unrealized_pnl': self.portfolio.unrealized_pnl,
            'realized_pnl': self.portfolio.realized_pnl,
            'leverage': self.portfolio.leverage,
            'positions': [
                pos.to_dict() 
                for pos in self.position_manager.get_open_positions()
            ]
        }
    
    def _validate_order(self, order: Order):
        """
        Validate an order before submission.
        
        Args:
            order: Order to validate
            
        Raises:
            InvalidOrderError: If the order is invalid
        """
        if not order.symbol:
            raise InvalidOrderError("Order symbol is required")
        
        if order.quantity <= 0:
            raise InvalidOrderError("Order quantity must be positive")
        
        if order.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and order.limit_price is None:
            raise InvalidOrderError(f"Limit price is required for {order.order_type.name} orders")
        
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and order.stop_price is None:
            raise InvalidOrderError(f"Stop price is required for {order.order_type.name} orders")
        
        # Additional validations can be added here
    
    def _check_risk_limits(self, order: Order):
        """
        Check if an order complies with risk limits.
        
        Args:
            order: Order to check
            
        Raises:
            RiskCheckFailedError: If the order fails risk checks
            InsufficientFundsError: If there are insufficient funds for the order
        """
        # Get current positions and portfolio state
        current_positions = self.position_manager.get_open_positions()
        portfolio_value = self.portfolio.total_value
        
        # Check position size limit
        position_size = order.quantity * (order.limit_price or self._get_current_price(order.symbol))
        if position_size > (portfolio_value * self.risk_manager.max_position_size):
            raise RiskCheckFailedError(
                f"Position size {position_size:,.2f} exceeds maximum allowed position size "
                f"of {portfolio_value * self.risk_manager.max_position_size:,.2f}"
            )
        
        # Check daily order limit
        current_date = datetime.now().date()
        if current_date != self.last_trade_day:
            self.daily_order_count = 0
            self.last_trade_day = current_date
        
        if self.daily_order_count >= self.risk_manager.max_orders_per_day:
            raise RiskCheckFailedError(
                f"Daily order limit of {self.risk_manager.max_orders_per_day} reached"
            )
        
        # Check short selling
        if order.side == OrderSide.SELL and not self.risk_manager.enable_shorting:
            # Check if we have a long position to sell
            has_position = any(
                p.symbol == order.symbol and p.quantity > 0 
                for p in current_positions
            )
            
            if not has_position:
                raise RiskCheckFailedError("Short selling is not enabled")
        
        # Check leverage
        if not self.risk_manager.enable_leverage and self.portfolio.leverage > 1.0:
            raise RiskCheckFailedError("Leverage is not enabled")
        
        # Additional risk checks can be added here
    
    def _execute_order(self, order: Order):
        """
        Execute an order through the broker.
        
        Args:
            order: Order to execute
            
        Raises:
            ExecutionError: If the order execution fails
        """
        try:
            # In a real system, this would be an async operation
            # For simplicity, we'll simulate immediate execution
            
            # Get current price for market orders
            if order.order_type == OrderType.MARKET:
                order.limit_price = self._get_current_price(order.symbol)
            
            # Simulate order execution
            filled_qty = order.quantity  # Assume full fill for simplicity
            fill_price = order.limit_price  # Assume limit price is filled at
            
            # Update order status
            order.status = OrderStatus.FILLED
            order.filled_quantity = filled_qty
            order.avg_fill_price = fill_price
            order.updated_at = datetime.utcnow()
            
            # Update position
            self._update_position_from_fill(order, fill_price)
            
            # Update portfolio
            self._update_portfolio(order, fill_price)
            
            # Update metrics
            self.trades_executed += 1
            self.shares_traded += filled_qty
            self.daily_order_count += 1
            
            logger.info(
                f"Order executed: {order.order_id} {order.side.name} {filled_qty} "
                f"{order.symbol} @ {fill_price:.2f}"
            )
            
        except Exception as e:
            logger.error(f"Failed to execute order {order.order_id}: {str(e)}")
            order.status = OrderStatus.REJECTED
            order.updated_at = datetime.utcnow()
            raise ExecutionError(f"Order execution failed: {str(e)}") from e
    
    def _update_position_from_fill(self, order: Order, fill_price: float):
        """
        Update positions based on a filled order.
        
        Args:
            order: Filled order
            fill_price: Price at which the order was filled
        """
        is_reducing = (
            order.side == OrderSide.SELL and 
            any(p.symbol == order.symbol and p.quantity > 0 
                for p in self.position_manager.get_open_positions())
        )
        
        try:
            # Update position
            result = self.position_manager.update_position(
                symbol=order.symbol,
                quantity=order.quantity * (1 if order.side == OrderSide.BUY else -1),
                price=fill_price,
                is_reducing=is_reducing,
                strategy_id=order.strategy_id
            )
            
            # Log position update
            if result.is_closed:
                logger.info(f"Position closed: {order.symbol} @ {fill_price:.2f}")
            elif result.side_flipped:
                logger.info(
                    f"Position flipped: {order.symbol} to {result.new_side.name} "
                    f"{result.new_quantity} @ {fill_price:.2f}"
                )
            
        except Exception as e:
            logger.error(f"Failed to update position for order {order.order_id}: {str(e)}")
            raise ExecutionError(f"Position update failed: {str(e)}") from e
    
    def _update_portfolio(self, order: Order, fill_price: float):
        """
        Update portfolio based on a filled order.
        
        Args:
            order: Filled order
            fill_price: Price at which the order was filled
        """
        try:
            # Calculate commission and fees
            commission = self.broker.calculate_commission(order, fill_price)
            self.commissions_paid += commission
            
            # Update portfolio
            if order.side == OrderSide.BUY:
                self.portfolio.update_from_buy(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    price=fill_price,
                    commission=commission
                )
            else:  # SELL
                self.portfolio.update_from_sell(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    price=fill_price,
                    commission=commission
                )
            
            # Update portfolio value based on current market prices
            self._update_portfolio_value()
            
        except Exception as e:
            logger.error(f"Failed to update portfolio for order {order.order_id}: {str(e)}")
            raise ExecutionError(f"Portfolio update failed: {str(e)}") from e
    
    def _update_portfolio_value(self):
        """Update portfolio value based on current market prices."""
        # Get all open positions
        positions = self.position_manager.get_open_positions()
        
        # Get current prices for all symbols
        symbols = list({p.symbol for p in positions})
        prices = {symbol: self._get_current_price(symbol) for symbol in symbols}
        
        # Update portfolio with current prices
        self.portfolio.update_positions(positions, prices)
    
    def _get_current_price(self, symbol: str) -> float:
        """
        Get the current market price for a symbol.
        
        Args:
            symbol: Symbol to get price for
            
        Returns:
            Current market price
            
        Raises:
            ExecutionError: If the price cannot be retrieved
        """
        try:
            # In a real system, this would fetch the latest market data
            # For now, we'll return a dummy price
            return 100.0  # Placeholder
        except Exception as e:
            logger.error(f"Failed to get price for {symbol}: {str(e)}")
            raise ExecutionError(f"Failed to get price for {symbol}: {str(e)}") from e
    
    def run(self):
        """Main execution loop (for backtesting or live trading)."""
        # In a real system, this would be an event loop that processes
        # market data, executes orders, and manages positions
        pass
    
    def shutdown(self):
        """Shut down the execution handler and clean up resources."""
        # Close all positions (if needed)
        # Save state
        # Disconnect from broker
        logger.info("Execution handler shutdown complete")
