"""
Global pytest configuration and fixtures.
"""
import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.absolute())
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure environment variables for testing
os.environ["ENV"] = "test"
os.environ["PYTHONPATH"] = f"{project_root}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers",
        "stress: mark test as a stress test"
    )
    config.addinivalue_line(
        "markers",
        "benchmark: mark test as a performance benchmark"
    )
