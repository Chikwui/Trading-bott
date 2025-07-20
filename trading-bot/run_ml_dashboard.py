"""
ML Model Monitoring Dashboard Runner

This script demonstrates the ML Model Monitoring Dashboard with mock data.
"""
import os
import sys
import logging
from datetime import datetime
import numpy as np
import pandas as pd
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock ModelVersion class
class MockModelVersion:
    def __init__(self, version: str, status: str, metrics: Dict[str, float]):
        self.version = version
        self.status = status
        self.metrics = metrics or {}
        self.created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Mock ModelVersionManager
class MockModelVersionManager:
    def __init__(self):
        self.versions = [
            MockModelVersion(
                version="1.0.0",
                status="production",
                metrics={
                    "accuracy": 0.92,
                    "precision": 0.89,
                    "recall": 0.85,
                    "f1": 0.87
                }
            ),
            MockModelVersion(
                version="1.1.0",
                status="staging",
                metrics={
                    "accuracy": 0.93,
                    "precision": 0.90,
                    "recall": 0.86,
                    "f1": 0.88
                }
            ),
            MockModelVersion(
                version="1.2.0",
                status="development",
                metrics={
                    "accuracy": 0.935,
                    "precision": 0.905,
                    "recall": 0.865,
                    "f1": 0.885
                }
            )
        ]
    
    def get_versions(self) -> List[MockModelVersion]:
        # Add some random variation to metrics to simulate changes over time
        for version in self.versions:
            for metric in version.metrics:
                # Add small random variation (±0.01)
                version.metrics[metric] += np.random.uniform(-0.01, 0.01)
                # Keep metrics in valid range [0, 1]
                version.metrics[metric] = max(0, min(1, version.metrics[metric]))
        return self.versions

# Import the dashboard after setting up the mock classes
from core.monitoring.dashboard import ModelMonitoringDashboard

def main():
    """Run the ML Model Monitoring Dashboard with mock data."""
    try:
        # Initialize the mock model version manager
        model_manager = MockModelVersionManager()
        
        # Create and run the dashboard
        dashboard = ModelMonitoringDashboard(
            model_version_manager=model_manager,
            port=8050,
            debug=True,
            update_interval=5,  # Update every 5 seconds for demo
            title="Trading Bot - ML Model Monitoring"
        )
        
        print("\n" + "="*80)
        print("Starting ML Model Monitoring Dashboard...")
        print("Open your web browser and navigate to: http://localhost:8050")
        print("="*80 + "\n")
        
        dashboard.run()
        
    except Exception as e:
        logger.error(f"Error running dashboard: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
