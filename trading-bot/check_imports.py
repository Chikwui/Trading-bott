"""Script to check for circular imports and slow imports in the application."""
import sys
import time
import importlib
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

def time_import(module_name: str) -> Tuple[float, bool, Optional[str]]:
    """Time the import of a module and check for errors."""
    start_time = time.time()
    error = None
    success = True
    
    try:
        importlib.import_module(module_name)
    except Exception as e:
        success = False
        error = f"{type(e).__name__}: {str(e)}"
    
    duration = time.time() - start_time
    return duration, success, error

def find_python_files(directory: Path) -> List[Path]:
    """Find all Python files in a directory recursively."""
    return list(directory.glob("**/*.py"))

def get_module_name(file_path: Path) -> str:
    """Convert a file path to a module name."""
    relative = file_path.relative_to(PROJECT_ROOT)
    return str(relative.with_suffix('')).replace("\\", ".")

def check_imports():
    """Check imports in all Python files in the project."""
    print(f"Checking imports in {PROJECT_ROOT}...")
    
    # Find all Python files
    python_files = find_python_files(PROJECT_ROOT)
    print(f"Found {len(python_files)} Python files.")
    
    # Test importing each module
    results = []
    slow_imports = []
    failed_imports = []
    
    for file_path in python_files:
        module_name = get_module_name(file_path)
        if module_name.endswith(".__init__"):
            module_name = module_name[:-9]  # Remove .__init__
        
        # Skip test files for now
        if "test" in module_name.lower() or "test" in file_path.parts:
            continue
            
        duration, success, error = time_import(module_name)
        
        if not success:
            failed_imports.append((module_name, error))
            print(f"❌ Failed to import {module_name}: {error}")
        elif duration > 1.0:  # More than 1 second is considered slow
            slow_imports.append((module_name, duration))
            print(f"🐌 Slow import: {module_name} took {duration:.2f}s")
        else:
            print(f"✅ Imported {module_name} in {duration:.3f}s")
        
        results.append((module_name, duration, success, error))
    
    # Print summary
    print("\n=== Import Check Summary ===")
    print(f"Total modules checked: {len(results)}")
    print(f"Failed imports: {len(failed_imports)}")
    print(f"Slow imports: {len(slow_imports)}")
    
    if failed_imports:
        print("\n=== Failed Imports ===")
        for module, error in failed_imports[:10]:  # Limit to first 10
            print(f"{module}: {error}")
    
    if slow_imports:
        print("\n=== Slow Imports (top 10) ===")
        for module, duration in sorted(slow_imports, key=lambda x: -x[1])[:10]:
            print(f"{module}: {duration:.2f}s")
    
    print("\n=== Done ===")

if __name__ == "__main__":
    check_imports()
