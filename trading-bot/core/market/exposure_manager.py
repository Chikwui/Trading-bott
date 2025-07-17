"""
Exposure management for tracking and controlling risk across multiple dimensions.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple, Any
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ExposureType(Enum):
    """Types of exposure to track."""
    ASSET_CLASS = auto()
    SECTOR = auto()
    CURRENCY = auto()
    STRATEGY = auto()
    INSTRUMENT = auto()

@dataclass
class ExposureLimit:
    """Defines exposure limits for a specific dimension."""
    max_notional: float  # Maximum notional exposure
    max_percent: float   # Maximum as % of portfolio
    max_leverage: float  # Maximum leverage allowed
    
    def __post_init__(self):
        """Validate the exposure limits."""
        if self.max_percent < 0 or self.max_percent > 1.0:
            raise ValueError("max_percent must be between 0 and 1.0")
        if self.max_leverage < 0:
            raise ValueError("max_leverage must be non-negative")

class ExposureManager:
    """Manages trading exposure across multiple dimensions."""
    
    def __init__(self, portfolio_value: float):
        """
        Initialize the exposure manager.
        
        Args:
            portfolio_value: Current total portfolio value
        """
        self.portfolio_value = portfolio_value
        self.exposure_limits: Dict[ExposureType, Dict[str, ExposureLimit]] = {
            exp_type: {} for exp_type in ExposureType
        }
        self.current_exposure: Dict[ExposureType, Dict[str, float]] = {
            exp_type: {} for exp_type in ExposureType
        }
        self.position_exposures: Dict[str, Dict[ExposureType, Set[str]]] = {}
        
    def set_exposure_limit(
        self,
        exposure_type: ExposureType,
        dimension: str,
        max_notional: float,
        max_percent: float,
        max_leverage: float = 3.0
    ) -> None:
        """
        Set exposure limits for a specific dimension.
        
        Args:
            exposure_type: Type of exposure (ASSET_CLASS, SECTOR, etc.)
            dimension: The specific dimension (e.g., 'equity', 'forex')
            max_notional: Maximum notional exposure
            max_percent: Maximum exposure as % of portfolio
            max_leverage: Maximum allowed leverage
        """
        self.exposure_limits[exposure_type][dimension] = ExposureLimit(
            max_notional=max_notional,
            max_percent=max_percent,
            max_leverage=max_leverage
        )
    
    def update_position(
        self,
        position_id: str,
        notional_value: float,
        exposures: Dict[ExposureType, List[str]]
    ) -> None:
        """
        Update exposure for a position.
        
        Args:
            position_id: Unique identifier for the position
            notional_value: Notional value of the position
            exposures: Dictionary mapping exposure types to dimensions
        """
        # Remove old exposures for this position
        self.remove_position(position_id)
        
        # Track which dimensions this position affects
        self.position_exposures[position_id] = {}
        
        # Update current exposures
        for exp_type, dimensions in exposures.items():
            for dim in dimensions:
                # Initialize if not exists
                if dim not in self.current_exposure[exp_type]:
                    self.current_exposure[exp_type][dim] = 0.0
                
                # Update exposure
                self.current_exposure[exp_type][dim] += notional_value
                
                # Track this exposure for the position
                if exp_type not in self.position_exposures[position_id]:
                    self.position_exposures[position_id][exp_type] = set()
                self.position_exposures[position_id][exp_type].add(dim)
    
    def remove_position(self, position_id: str) -> None:
        """
        Remove a position's exposure.
        
        Args:
            position_id: ID of the position to remove
        """
        if position_id not in self.position_exposures:
            return
            
        # Get the position's notional value from one of its exposures
        position_value = 0
        for exp_type, dimensions in self.position_exposures[position_id].items():
            for dim in dimensions:
                if dim in self.current_exposure[exp_type]:
                    position_value = max(position_value, self.current_exposure[exp_type][dim])
                    break
        
        # Remove the position's contribution to each exposure
        for exp_type, dimensions in self.position_exposures[position_id].items():
            for dim in list(dimensions):
                if dim in self.current_exposure[exp_type]:
                    self.current_exposure[exp_type][dim] -= position_value
                    if self.current_exposure[exp_type][dim] <= 0:
                        del self.current_exposure[exp_type][dim]
        
        # Remove the position from tracking
        del self.position_exposures[position_id]
    
    def check_exposure_limits(
        self,
        new_position: Optional[Dict[ExposureType, List[str]]] = None,
        new_notional: float = 0.0
    ) -> Tuple[bool, List[str]]:
        """
        Check if adding a new position would exceed any exposure limits.
        
        Args:
            new_position: New position's exposures (optional)
            new_notional: Notional value of the new position
            
        Returns:
            Tuple of (is_valid, list_of_violations)
        """
        violations = []
        
        # Check current exposures
        for exp_type, dimensions in self.current_exposure.items():
            for dim, current in dimensions.items():
                limit = self.exposure_limits[exp_type].get(dim)
                if not limit:
                    continue
                    
                # Calculate projected exposure
                projected = current
                if new_position and dim in new_position.get(exp_type, []):
                    projected += new_notional
                
                # Check limits
                if limit.max_notional > 0 and projected > limit.max_notional:
                    violations.append(
                        f"{exp_type.name} {dim} notional limit: "
                        f"{projected:.2f} > {limit.max_notional:.2f}"
                    )
                
                if limit.max_percent > 0 and projected > self.portfolio_value * limit.max_percent:
                    violations.append(
                        f"{exp_type.name} {dim} percent limit: "
                        f"{projected/self.portfolio_value*100:.1f}% > {limit.max_percent*100:.1f}%"
                    )
                
                leverage = projected / self.portfolio_value
                if limit.max_leverage > 0 and leverage > limit.max_leverage:
                    violations.append(
                        f"{exp_type.name} {dim} leverage: "
                        f"{leverage:.2f}x > {limit.max_leverage:.2f}x"
                    )
        
        return (len(violations) == 0, violations)
    
    def get_exposure_report(self) -> Dict[str, Any]:
        """
        Generate a report of current exposures.
        
        Returns:
            Dictionary with exposure details
        """
        report = {
            "portfolio_value": self.portfolio_value,
            "exposures": {},
            "warnings": []
        }
        
        for exp_type, dimensions in self.current_exposure.items():
            report["exposures"][exp_type.name] = {}
            
            for dim, value in dimensions.items():
                limit = self.exposure_limits[exp_type].get(dim)
                leverage = value / self.portfolio_value if self.portfolio_value > 0 else 0
                
                exposure_info = {
                    "notional": value,
                    "percent": value / self.portfolio_value * 100 if self.portfolio_value > 0 else 0,
                    "leverage": leverage,
                    "limit": {
                        "max_notional": limit.max_notional if limit else None,
                        "max_percent": limit.max_percent * 100 if limit else None,
                        "max_leverage": limit.max_leverage if limit else None,
                    } if limit else None,
                    "warning": None
                }
                
                # Check for warnings
                if limit:
                    warnings = []
                    if limit.max_notional > 0 and value > limit.max_notional:
                        warnings.append(f"Exceeds notional limit of {limit.max_notional:.2f}")
                    if limit.max_percent > 0 and value > self.portfolio_value * limit.max_percent:
                        warnings.append(f"Exceeds {limit.max_percent*100:.1f}% of portfolio")
                    if limit.max_leverage > 0 and leverage > limit.max_leverage:
                        warnings.append(f"Exceeds {limit.max_leverage:.1f}x leverage")
                    
                    if warnings:
                        exposure_info["warning"] = "; ".join(warnings)
                        report["warnings"].append(f"{exp_type.name} {dim}: {exposure_info['warning']}")
                
                report["exposures"][exp_type.name][dim] = exposure_info
        
        return report
