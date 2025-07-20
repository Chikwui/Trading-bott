"""
Trade Monitoring Service

This module provides real-time monitoring, analysis, and alerting for trade execution logs.
It includes log rotation, pattern detection, and integration with monitoring systems.
"""
import json
import gzip
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set, Callable, Any, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
import re
import time
import os
import shutil
from collections import defaultdict, deque
from typing import Pattern

import pandas as pd
import numpy as np
from prometheus_client import start_http_server, Counter, Gauge, Histogram, Summary

from core.utils.helpers import get_logger
from core.utils.trade_logger import TradeEventType, TradeLogger, trade_logger

logger = get_logger(__name__)

# Prometheus metrics
METRICS_PORT = 8000
METRICS_PREFIX = 'trading_bot_'

# Define Prometheus metrics
METRICS = {
    'orders_total': Counter(
        f'{METRICS_PREFIX}orders_total',
        'Total number of orders',
        ['status', 'symbol', 'order_type']
    ),
    'order_execution_time': Histogram(
        f'{METRICS_PREFIX}order_execution_time_seconds',
        'Order execution time in seconds',
        ['symbol', 'order_type']
    ),
    'position_size': Gauge(
        f'{METRICS_PREFIX}position_size',
        'Current position size',
        ['symbol', 'side']
    ),
    'risk_check_failures': Counter(
        f'{METRICS_PREFIX}risk_check_failures_total',
        'Total number of failed risk checks',
        ['check_name', 'symbol']
    ),
    'log_entries_processed': Counter(
        f'{METRICS_PREFIX}log_entries_processed_total',
        'Total number of log entries processed',
        ['level']
    ),
    'alert_events': Counter(
        f'{METRICS_PREFIX}alert_events_total',
        'Total number of alert events',
        ['alert_type', 'severity']
    )
}

class LogRotationConfig:
    """Configuration for log rotation and retention."""
    def __init__(
        self,
        max_size_mb: int = 100,
        backup_count: int = 10,
        retention_days: int = 30,
        compress_backups: bool = True
    ):
        """
        Initialize log rotation configuration.
        
        Args:
            max_size_mb: Maximum log file size in MB before rotation
            backup_count: Number of backup files to keep
            retention_days: Number of days to keep log files
            compress_backups: Whether to compress rotated log files
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.backup_count = backup_count
        self.retention_days = retention_days
        self.compress_backups = compress_backups

class LogPattern:
    """Pattern to match in log entries for alerting."""
    def __init__(
        self,
        name: str,
        pattern: str,
        level: str = 'WARNING',
        alert_on_match: bool = True,
        alert_message: Optional[str] = None,
        condition: Optional[Callable[[Dict], bool]] = None
    ):
        """
        Initialize a log pattern matcher.
        
        Args:
            name: Name of the pattern
            pattern: Regex pattern to match in log messages
            level: Minimum log level to match
            alert_on_match: Whether to trigger an alert on match
            alert_message: Custom alert message (uses pattern name if None)
            condition: Optional function to evaluate additional conditions
        """
        self.name = name
        self.pattern = re.compile(pattern, re.IGNORECASE)
        self.level = level.upper()
        self.alert_on_match = alert_on_match
        self.alert_message = alert_message or f"Pattern matched: {name}"
        self.condition = condition or (lambda _: True)

@dataclass
class Alert:
    """Represents an alert condition and its configuration."""
    name: str
    condition: Callable[[Dict], bool]
    severity: str = 'WARNING'  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    message: Optional[str] = None
    cooldown_seconds: int = 300  # Minimum time between alerts of the same type
    notify_channels: List[str] = field(default_factory=lambda: ['log', 'email'])
    last_triggered: Optional[float] = None

class TradeMonitor:
    """Monitors trade logs and manages alerts and metrics."""
    
    def __init__(
        self,
        log_dir: str = 'logs/trades',
        log_rotation: Optional[LogRotationConfig] = None,
        patterns: Optional[List[LogPattern]] = None,
        metrics_port: int = METRICS_PORT
    ):
        """
        Initialize the trade monitor.
        
        Args:
            log_dir: Directory containing trade logs
            log_rotation: Log rotation configuration
            patterns: List of log patterns to monitor
            metrics_port: Port to expose Prometheus metrics on
        """
        self.log_dir = Path(log_dir)
        self.log_rotation = log_rotation or LogRotationConfig()
        self.patterns = patterns or self._get_default_patterns()
        self.metrics_port = metrics_port
        self.active_alerts: Dict[str, Alert] = {}
        self.running = False
        self._log_file_handles: Dict[Path, Any] = {}
        self._log_positions: Dict[Path, int] = {}
        self._last_check_time = time.time()
        self._buffer: List[Dict] = []
        self._buffer_lock = asyncio.Lock()
        self._metrics_initialized = False
        
        # Initialize metrics
        self._init_metrics()
    
    def _init_metrics(self) -> None:
        """Initialize Prometheus metrics."""
        if not self._metrics_initialized:
            try:
                start_http_server(self.metrics_port)
                logger.info(f"Started metrics server on port {self.metrics_port}")
                self._metrics_initialized = True
            except Exception as e:
                logger.error(f"Failed to start metrics server: {e}")
    
    def _get_default_patterns(self) -> List[LogPattern]:
        """Get default patterns for common issues."""
        return [
            LogPattern(
                name="order_rejected",
                pattern=r"order.*rejected",
                level="ERROR",
                alert_message="Order was rejected"
            ),
            LogPattern(
                name="risk_check_failed",
                pattern=r"risk check failed",
                level="WARNING",
                alert_message="Risk check failed"
            ),
            LogPattern(
                name="position_limit_exceeded",
                pattern=r"position.*limit.*exceeded",
                level="ERROR",
                alert_message="Position limit exceeded"
            ),
            LogPattern(
                name="high_slippage",
                pattern=r"slippage.*>\s*[0-9.]+%",
                level="WARNING",
                alert_message="High slippage detected"
            ),
            LogPattern(
                name="connection_error",
                pattern=r"connection.*error|timeout|disconnected",
                level="ERROR",
                alert_message="Connection issue detected"
            )
        ]
    
    async def start(self) -> None:
        """Start the monitoring service."""
        if self.running:
            logger.warning("Monitor is already running")
            return
            
        self.running = True
        logger.info("Starting trade monitoring service")
        
        # Start background tasks
        self._monitor_task = asyncio.create_task(self._monitor_logs())
        self._process_buffer_task = asyncio.create_task(self._process_buffer())
        self._cleanup_task = asyncio.create_task(self._cleanup_old_logs())
    
    async def stop(self) -> None:
        """Stop the monitoring service."""
        if not self.running:
            return
            
        logger.info("Stopping trade monitoring service")
        self.running = False
        
        # Cancel background tasks
        self._monitor_task.cancel()
        self._process_buffer_task.cancel()
        self._cleanup_task.cancel()
        
        # Close any open file handles
        for f in self._log_file_handles.values():
            try:
                f.close()
            except Exception as e:
                logger.error(f"Error closing log file: {e}")
        
        self._log_file_handles.clear()
        self._log_positions.clear()
    
    async def _monitor_logs(self) -> None:
        """Monitor log files for new entries."""
        while self.running:
            try:
                # Find all log files in the directory
                log_files = list(self.log_dir.glob("*.log"))
                
                for log_file in log_files:
                    if not self.running:
                        break
                        
                    try:
                        # Skip if file hasn't changed since last check
                        mtime = log_file.stat().st_mtime
                        if mtime <= self._last_check_time:
                            continue
                        
                        # Read new lines
                        lines = await self._read_new_lines(log_file)
                        
                        # Process each line
                        for line in lines:
                            try:
                                entry = json.loads(line)
                                await self._process_log_entry(entry, log_file.name)
                            except json.JSONDecodeError:
                                logger.warning(f"Invalid JSON in log file {log_file}: {line}")
                    except Exception as e:
                        logger.error(f"Error processing log file {log_file}: {e}")
                
                # Update last check time
                self._last_check_time = time.time()
                
                # Sleep before next check
                await asyncio.sleep(1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in log monitor: {e}")
                await asyncio.sleep(5)  # Back off on error
    
    async def _read_new_lines(self, file_path: Path) -> List[str]:
        """Read new lines from a log file since last read position."""
        try:
            # Get current file size
            current_size = file_path.stat().st_size
            
            # Initialize file position if needed
            if file_path not in self._log_positions:
                self._log_positions[file_path] = 0
            
            # If file was truncated, reset position
            if current_size < self._log_positions[file_path]:
                self._log_positions[file_path] = 0
            
            # No new data
            if current_size <= self._log_positions[file_path]:
                return []
            
            # Open file and seek to last position
            if file_path not in self._log_file_handles:
                self._log_file_handles[file_path] = open(file_path, 'r', encoding='utf-8')
            
            f = self._log_file_handles[file_path]
            f.seek(self._log_positions[file_path])
            
            # Read new lines
            lines = f.readlines()
            
            # Update position
            self._log_positions[file_path] = f.tell()
            
            return lines
            
        except Exception as e:
            logger.error(f"Error reading log file {file_path}: {e}")
            # Reset position on error
            if file_path in self._log_positions:
                self._log_positions[file_path] = 0
            return []
    
    async def _process_log_entry(self, entry: Dict, source: str) -> None:
        """Process a single log entry."""
        try:
            # Add source and timestamp if missing
            if 'timestamp' not in entry:
                entry['timestamp'] = datetime.utcnow().isoformat()
            if 'source' not in entry:
                entry['source'] = source
            
            # Update metrics
            self._update_metrics(entry)
            
            # Check for patterns
            await self._check_patterns(entry)
            
            # Add to buffer for batch processing
            async with self._buffer_lock:
                self._buffer.append(entry)
                
                # Process buffer if it reaches a certain size
                if len(self._buffer) >= 100:  # Configurable batch size
                    buffer = self._buffer.copy()
                    self._buffer = []
                    asyncio.create_task(self._process_buffer(buffer))
        
        except Exception as e:
            logger.error(f"Error processing log entry: {e}")
    
    async def _process_buffer(self, buffer: Optional[List[Dict]] = None) -> None:
        """Process a batch of log entries."""
        if buffer is None:
            async with self._buffer_lock:
                if not self._buffer:
                    return
                buffer = self._buffer
                self._buffer = []
        
        if not buffer:
            return
        
        try:
            # Example: Store in database or forward to external system
            # await self._store_entries(buffer)
            
            # Update metrics
            for entry in buffer:
                self.METRICS['log_entries_processed'].labels(
                    level=entry.get('level', 'UNKNOWN')
                ).inc()
                
        except Exception as e:
            logger.error(f"Error processing log buffer: {e}")
    
    def _update_metrics(self, entry: Dict) -> None:
        """Update Prometheus metrics based on log entry."""
        try:
            event_type = entry.get('event_type')
            
            # Update order metrics
            if event_type in ['order_submit', 'order_accepted', 'order_rejected', 'order_filled']:
                METRICS['orders_total'].labels(
                    status=event_type.split('_')[-1],
                    symbol=entry.get('symbol', 'unknown'),
                    order_type=entry.get('order_type', 'unknown')
                ).inc()
            
            # Update position metrics
            elif event_type in ['position_opened', 'position_modified', 'position_closed']:
                if 'size' in entry and 'symbol' in entry and 'side' in entry:
                    METRICS['position_size'].labels(
                        symbol=entry['symbol'],
                        side=entry['side']
                    ).set(entry['size'])
            
            # Update risk metrics
            elif event_type == 'risk_check_failed':
                METRICS['risk_check_failures'].labels(
                    check_name=entry.get('check_name', 'unknown'),
                    symbol=entry.get('symbol', 'unknown')
                ).inc()
        
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    async def _check_patterns(self, entry: Dict) -> None:
        """Check log entry against patterns and trigger alerts."""
        message = str(entry.get('message', ''))
        level = entry.get('level', 'INFO').upper()
        
        for pattern in self.patterns:
            try:
                # Skip if log level is below pattern level
                if self._get_level_value(level) < self._get_level_value(pattern.level):
                    continue
                
                # Check if pattern matches
                if pattern.pattern.search(message) and pattern.condition(entry):
                    # Check cooldown for this alert type
                    if pattern.name in self.active_alerts:
                        alert = self.active_alerts[pattern.name]
                        if time.time() - alert.last_triggered < alert.cooldown_seconds:
                            continue
                    
                    # Create alert
                    alert = Alert(
                        name=pattern.name,
                        condition=lambda e: True,  # Already matched
                        severity=pattern.level,
                        message=pattern.alert_message,
                        cooldown_seconds=300,
                        last_triggered=time.time()
                    )
                    
                    # Trigger alert
                    await self._trigger_alert(alert, entry)
                    
                    # Update active alerts
                    self.active_alerts[pattern.name] = alert
                    
            except Exception as e:
                logger.error(f"Error checking pattern {pattern.name}: {e}")
    
    async def _trigger_alert(self, alert: Alert, entry: Dict) -> None:
        """Trigger an alert."""
        try:
            # Update metrics
            METRICS['alert_events'].labels(
                alert_type=alert.name,
                severity=alert.severity
            ).inc()
            
            # Log the alert
            logger.log(
                getattr(logging, alert.severity, logging.WARNING),
                f"ALERT [{alert.name}]: {alert.message}\n{json.dumps(entry, indent=2)}"
            )
            
            # TODO: Send notifications to configured channels
            # e.g., email, Slack, PagerDuty, etc.
            
        except Exception as e:
            logger.error(f"Error triggering alert {alert.name}: {e}")
    
    async def _cleanup_old_logs(self) -> None:
        """Clean up old log files based on retention policy."""
        while self.running:
            try:
                now = time.time()
                cutoff_time = now - (self.log_rotation.retention_days * 86400)
                
                # Find and delete old log files
                for log_file in self.log_dir.glob("*.log*"):  # Matches .log and .log.gz
                    try:
                        if log_file.stat().st_mtime < cutoff_time:
                            log_file.unlink()
                            logger.info(f"Deleted old log file: {log_file}")
                    except Exception as e:
                        logger.error(f"Error deleting log file {log_file}: {e}")
                
                # Sleep for 1 hour between cleanup runs
                await asyncio.sleep(3600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in log cleanup: {e}")
                await asyncio.sleep(300)  # Back off on error
    
    @staticmethod
    def _get_level_value(level: str) -> int:
        """Get numeric value for a log level."""
        level = level.upper()
        if level == 'DEBUG':
            return 10
        elif level == 'INFO':
            return 20
        elif level == 'WARNING':
            return 30
        elif level == 'ERROR':
            return 40
        elif level == 'CRITICAL':
            return 50
        return 0  # Default to lowest priority

# Global instance
trade_monitor = TradeMonitor()

async def start_monitoring() -> None:
    """Start the global trade monitoring service."""
    await trade_monitor.start()

async def stop_monitoring() -> None:
    """Stop the global trade monitoring service."""
    await trade_monitor.stop()

def get_metrics() -> Dict[str, Any]:
    """Get current metrics."""
    # This would query Prometheus or return cached metrics
    return {}

def analyze_logs(
    start_time: datetime,
    end_time: Optional[datetime] = None,
    filters: Optional[Dict[str, Any]] = None
) -> pd.DataFrame:
    """
    Analyze logs within a time range with optional filters.
    
    Args:
        start_time: Start of time range
        end_time: End of time range (defaults to now)
        filters: Dictionary of field: value to filter by
        
    Returns:
        DataFrame containing matching log entries
    """
    end_time = end_time or datetime.utcnow()
    filters = filters or {}
    
    # This would query the log storage (e.g., database, Elasticsearch)
    # For now, return an empty DataFrame
    return pd.DataFrame()
