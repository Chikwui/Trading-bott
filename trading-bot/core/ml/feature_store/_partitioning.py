""
Feature Store Partitioning Utilities

Provides utilities for time-based partitioning of feature data.
"""
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import pandas as pd

class TimePartitioning:
    """Handles time-based partitioning of feature data."""
    
    def __init__(self, partition_format: str = "%Y/%m/%d"):
        """Initialize with partition format.
        
        Args:
            partition_format: Format string for time-based partitioning
        """
        self.partition_format = partition_format
    
    def get_partition_path(self, timestamp: datetime) -> str:
        """Get partition path for a given timestamp.
        
        Args:
            timestamp: Timestamp to generate partition for
            
        Returns:
            Partition path string
        """
        return timestamp.strftime(self.partition_format)
    
    def get_partition_bounds(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Tuple[datetime, datetime]:
        """Get default time bounds if not provided.
        
        Args:
            start_date: Optional start date
            end_date: Optional end date
            
        Returns:
            Tuple of (start_date, end_date) with defaults if None
        """
        if start_date is None:
            start_date = datetime.utcnow().replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        if end_date is None:
            end_date = datetime.utcnow()
        return start_date, end_date
    
    def expand_partitions(
        self,
        start_date: datetime,
        end_date: datetime,
        base_path: str,
        file_extension: str = "parquet"
    ) -> Dict[str, str]:
        """Expand time range into partition paths.
        
        Args:
            start_date: Start of time range
            end_date: End of time range
            base_path: Base path for partitions
            file_extension: File extension for partition files
            
        Returns:
            Dictionary mapping partition paths to file paths
        """
        partitions = {}
        current = start_date
        while current <= end_date:
            partition = self.get_partition_path(current)
            file_name = f"data.{file_extension}"
            file_path = str(Path(base_path) / partition / file_name)
            partitions[partition] = file_path
            
            # Move to next day
            if "%d" in self.partition_format:
                current = current.replace(day=current.day+1)
            elif "%m" in self.partition_format:
                current = current.replace(month=current.month+1)
            else:  # Default to monthly
                current = current.replace(month=current.month+1)
                
        return partitions
