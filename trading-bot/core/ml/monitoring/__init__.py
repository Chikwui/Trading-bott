"""
Monitoring Module

This module provides functionality for monitoring ML models in production,
including performance metrics, drift detection, and alerting.
"""

import os
from typing import Optional

from .base import Monitor
from .file_monitor import FileMonitor

__all__ = ['Monitor', 'FileMonitor', 'get_monitor']

def get_monitor(
    monitor_type: str = 'file',
    log_dir: Optional[str] = None,
    **kwargs
) -> Monitor:
    """Get a monitoring instance.
    
    Args:
        monitor_type: Type of monitor to create ('file')
        log_dir: Directory to store logs (defaults to 'monitoring' in current directory)
        **kwargs: Additional arguments for the monitor constructor
        
    Returns:
        An instance of Monitor
    """
    if log_dir is None:
        log_dir = os.path.join(os.getcwd(), 'monitoring')
    
    if monitor_type == 'file':
        return FileMonitor(log_dir=log_dir, **kwargs)
    else:
        raise ValueError(f"Unsupported monitor type: {monitor_type}")
