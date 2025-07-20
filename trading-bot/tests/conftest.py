"""
Global pytest configuration and fixtures.
"""
import os
import sys
from pathlib import Path

# Add project root and src to Python path
project_root = str(Path(__file__).parent.absolute())
src_path = str(Path(project_root).parent.absolute())

# Add both project root and src to Python path
for path in [project_root, src_path]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Configure environment variables for testing
os.environ["ENV"] = "test"
os.environ["PYTHONPATH"] = f"{project_root}{os.pathsep}{src_path}{os.pathsep}{os.environ.get('PYTHONPATH', '')}"

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
