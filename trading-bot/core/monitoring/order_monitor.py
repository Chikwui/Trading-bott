"""
Advanced order monitoring and metrics collection system.

This module provides real-time monitoring, metrics collection, and alerting
for order lifecycle events across the trading system.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Callable, Awaitable, TypeVar, Type
from uuid import UUID, uuid4

from prometheus_client import (Counter, Gauge, Histogram, Summary, 
                             start_http_server as start_prometheus_server)
import numpy as np
from pydantic import BaseModel, Field, validator

from core.trading.order_state import OrderStatus, StateTransitionEvent
from core.utils.metrics import timer, counter, histogram
from core.utils.retry import async_retry

logger = logging.getLogger(__name__)
T = TypeVar('T', bound='OrderMonitor')

class MetricType(str, Enum):
    """Types of metrics supported by the monitoring system."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"

@dataclass
class MetricDefinition:
    """Definition of a metric to be collected."""
    name: str
    metric_type: MetricType
    description: str
    labels: List[str] = field(default_factory=list)
    buckets: Optional[List[float]] = None  # For histogram
    objectives: Optional[Dict[float, float]] = None  # For summary

class OrderMetrics:
    """Container for all order-related metrics."""
    
    def __init__(self, namespace: str = 'trading'):
        self.namespace = namespace
        self._metrics: Dict[str, Any] = {}
        self._initialize_metrics()
    
    def _initialize_metrics(self) -> None:
        """Initialize all Prometheus metrics."""
        # Order lifecycle metrics
        self._metrics['order_events_total'] = Counter(
            f'{self.namespace}_order_events_total',
            'Total number of order events by type and status',
            ['event_type', 'status', 'symbol', 'order_type']
        )
        
        # Order execution metrics
        self._metrics['order_execution_time'] = Histogram(
            f'{self.namespace}_order_execution_time_seconds',
            'Time taken to execute orders',
            ['symbol', 'order_type', 'status'],
            buckets=[0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0]
        )
        
        # Order size metrics
        self._metrics['order_size'] = Histogram(
            f'{self.namespace}_order_size',
            'Size of orders by symbol and type',
            ['symbol', 'order_type'],
            buckets=[0.1, 1, 10, 50, 100, 500, 1000, 5000, 10000]
        )
        
        # Latency metrics
        self._metrics['order_latency'] = Histogram(
            f'{self.namespace}_order_latency_seconds',
            'Order processing latency by stage',
            ['stage', 'symbol', 'order_type'],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        
        # Error metrics
        self._metrics['order_errors_total'] = Counter(
            f'{self.namespace}_order_errors_total',
            'Total number of order processing errors',
            ['error_type', 'symbol', 'order_type']
        )
        
        # Position metrics
        self._metrics['position_size'] = Gauge(
            f'{self.namespace}_position_size',
            'Current position size by symbol',
            ['symbol', 'account_id']
        )
        
        # PnL metrics
        self._metrics['realized_pnl'] = Gauge(
            f'{self.namespace}_realized_pnl',
            'Realized profit and loss',
            ['symbol', 'account_id']
        )
        
        self._metrics['unrealized_pnl'] = Gauge(
            f'{self.namespace}_unrealized_pnl',
            'Unrealized profit and loss',
            ['symbol', 'account_id']
        )
    
    def record_order_event(
        self, 
        event_type: str, 
        status: str, 
        symbol: str, 
        order_type: str,
        quantity: Optional[float] = None,
        price: Optional[float] = None
    ) -> None:
        """Record an order lifecycle event."""
        labels = {
            'event_type': event_type,
            'status': status,
            'symbol': symbol,
            'order_type': order_type
        }
        
        self._metrics['order_events_total'].labels(**labels).inc()
        
        if quantity is not None:
            self._metrics['order_size'].labels(
                symbol=symbol,
                order_type=order_type
            ).observe(quantity)
    
    def record_latency(
        self, 
        stage: str, 
        symbol: str, 
        order_type: str, 
        latency_seconds: float
    ) -> None:
        """Record order processing latency."""
        self._metrics['order_latency'].labels(
            stage=stage,
            symbol=symbol,
            order_type=order_type
        ).observe(latency_seconds)
    
    def record_error(
        self, 
        error_type: str, 
        symbol: str, 
        order_type: str
    ) -> None:
        """Record an order processing error."""
        self._metrics['order_errors_total'].labels(
            error_type=error_type,
            symbol=symbol,
            order_type=order_type
        ).inc()
    
    def update_position(
        self,
        symbol: str,
        account_id: str,
        size: float,
        realized_pnl: Optional[float] = None,
        unrealized_pnl: Optional[float] = None
    ) -> None:
        """Update position metrics."""
        self._metrics['position_size'].labels(
            symbol=symbol,
            account_id=account_id
        ).set(size)
        
        if realized_pnl is not None:
            self._metrics['realized_pnl'].labels(
                symbol=symbol,
                account_id=account_id
            ).set(realized_pnl)
            
        if unrealized_pnl is not None:
            self._metrics['unrealized_pnl'].labels(
                symbol=symbol,
                account_id=account_id
            ).set(unrealized_pnl)

class AlertRule(BaseModel):
    """Rule for triggering alerts based on order metrics."""
    
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    condition: str  # Python expression that evaluates to a boolean
    severity: str  # 'info', 'warning', 'critical'
    message: str
    action: Optional[str] = None  # Optional action to take when alert triggers
    cooldown_seconds: int = 300  # Minimum time between alerts for the same rule
    active: bool = True
    
    class Config:
        json_encoders = {
            'condition': str,
            'action': str
        }

class OrderMonitor:
    """Advanced order monitoring and alerting system."""
    
    def __init__(
        self,
        metrics_port: int = 9090,
        namespace: str = 'trading',
        alert_rules: Optional[List[Dict]] = None
    ):
        self.metrics = OrderMetrics(namespace=namespace)
        self.metrics_port = metrics_port
        self.alert_rules: Dict[str, AlertRule] = {}
        self._alert_last_triggered: Dict[str, float] = {}
        self._alert_handlers: Dict[str, Callable[[AlertRule, Dict], Awaitable[None]]] = {}
        
        # Register default alert handlers
        self.register_alert_handler('log', self._log_alert)
        
        # Load alert rules if provided
        if alert_rules:
            self.load_alert_rules(alert_rules)
    
    async def start(self) -> None:
        """Start the monitoring server."""
        try:
            start_prometheus_server(self.metrics_port)
            logger.info(f"Started Prometheus metrics server on port {self.metrics_port}")
        except Exception as e:
            logger.error(f"Failed to start metrics server: {e}")
            raise
    
    def load_alert_rules(self, rules: List[Dict]) -> None:
        """Load alert rules from configuration."""
        for rule_data in rules:
            try:
                rule = AlertRule(**rule_data)
                self.alert_rules[rule.id] = rule
                logger.info(f"Loaded alert rule: {rule.name}")
            except Exception as e:
                logger.error(f"Failed to load alert rule {rule_data.get('name')}: {e}")
    
    def register_alert_handler(
        self, 
        name: str, 
        handler: Callable[[AlertRule, Dict], Awaitable[None]]
    ) -> None:
        """Register an alert handler function."""
        self._alert_handlers[name] = handler
    
    async def process_order_event(
        self, 
        event: StateTransitionEvent,
        order: 'Order',
        execution_time: Optional[float] = None
    ) -> None:
        """Process an order state transition event."""
        # Record metrics
        self.metrics.record_order_event(
            event_type='state_transition',
            status=event.to_state,
            symbol=order.symbol,
            order_type=order.order_type,
            quantity=float(order.quantity) if order.quantity else None,
            price=float(order.price) if order.price else None
        )
        
        # Record latency if available
        if execution_time is not None:
            self.metrics.record_latency(
                stage=f'{event.from_state}_to_{event.to_state}',
                symbol=order.symbol,
                order_type=order.order_type,
                latency_seconds=execution_time
            )
        
        # Check alert rules
        await self._check_alert_rules(event, order)
    
    async def _check_alert_rules(self, event: StateTransitionEvent, order: 'Order') -> None:
        """Check if any alert rules match the current event."""
        for rule_id, rule in self.alert_rules.items():
            if not rule.active:
                continue
                
            # Check cooldown
            last_triggered = self._alert_last_triggered.get(rule_id, 0)
            if time.time() - last_triggered < rule.cooldown_seconds:
                continue
            
            # Evaluate condition in a safe context
            try:
                # Create a safe context for evaluation
                context = {
                    'event': event,
                    'order': order,
                    'now': datetime.now(timezone.utc),
                    'metrics': self.metrics._metrics,
                    'np': np  # For numerical operations in conditions
                }
                
                # Evaluate condition
                condition_met = eval(  # nosec - This is a controlled eval with a restricted context
                    rule.condition,
                    {'__builtins__': {}},
                    context
                )
                
                if condition_met:
                    await self._trigger_alert(rule, {
                        'event': event,
                        'order': order,
                        'timestamp': datetime.now(timezone.utc)
                    })
                    self._alert_last_triggered[rule.id] = time.time()
                    
            except Exception as e:
                logger.error(f"Error evaluating alert rule {rule.name}: {e}", exc_info=True)
    
    async def _trigger_alert(self, rule: AlertRule, context: Dict) -> None:
        """Trigger an alert using all registered handlers."""
        alert_data = {
            'rule_id': rule.id,
            'name': rule.name,
            'severity': rule.severity,
            'message': rule.message,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'context': context
        }
        
        # Execute the alert action if defined
        if rule.action:
            try:
                # Execute action in a safe context
                action_context = {
                    'alert': alert_data,
                    'metrics': self.metrics._metrics,
                    'np': np
                }
                
                # Execute action
                exec(rule.action, {'__builtins__': {}}, action_context)  # nosec - This is a controlled exec
            except Exception as e:
                logger.error(f"Error executing alert action for {rule.name}: {e}", exc_info=True)
        
        # Call all registered alert handlers
        for handler in self._alert_handlers.values():
            try:
                await handler(rule, alert_data)
            except Exception as e:
                logger.error(f"Error in alert handler: {e}", exc_info=True)
    
    async def _log_alert(self, rule: AlertRule, alert_data: Dict) -> None:
        """Default alert handler that logs alerts."""
        log_level = {
            'info': logger.info,
            'warning': logger.warning,
            'critical': logger.critical
        }.get(rule.severity.lower(), logger.info)
        
        log_level(
            f"ALERT [{rule.severity.upper()}] {rule.name}: {rule.message}\n"
            f"Context: {alert_data.get('context', {})}"
        )

# Example usage
async def example_usage():
    """Example of using the OrderMonitor."""
    # Create a monitor with some alert rules
    monitor = OrderMonitor(
        metrics_port=9090,
        alert_rules=[
            {
                'name': 'High Order Latency',
                'condition': "event.to_state == 'FILLED' and metrics['order_latency']._sum.get() / metrics['order_latency']._count.get() > 0.5",
                'severity': 'warning',
                'message': 'High order execution latency detected',
                'action': "print('High latency alert triggered')",
                'cooldown_seconds': 60
            },
            {
                'name': 'Order Rejection',
                'condition': "event.to_state == 'REJECTED'",
                'severity': 'critical',
                'message': 'Order was rejected by the exchange',
                'cooldown_seconds': 0
            }
        ]
    )
    
    # Start the metrics server
    await monitor.start()
    
    # Example of processing an order event
    class MockOrder:
        def __init__(self):
            self.id = "order_123"
            self.symbol = "BTC/USD"
            self.order_type = "LIMIT"
            self.quantity = Decimal("1.0")
            self.price = Decimal("50000.00")
    
    event = StateTransitionEvent(
        from_state="NEW",
        to_state="FILLED",
        timestamp=datetime.now(timezone.utc),
        reason="Filled",
        metadata={"exchange": "binance"}
    )
    
    await monitor.process_order_event(event, MockOrder(), execution_time=0.75)

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_usage())
