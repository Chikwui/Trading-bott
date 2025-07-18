"""
Cleanup Script for Python Packages

This script helps clean up and reinstall packages that may have
installation issues, particularly numpy and related packages.
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

def remove_directory(path: Path):
    """Safely remove a directory if it exists."""
    try:
        if path.exists() and path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            print(f"Removed directory: {path}")
    except Exception as e:
        print(f"Warning: Could not remove {path}: {e}")

def clean_numpy_installation():
    """Clean up numpy installation and related temporary files."""
    print("Cleaning up numpy installation...")
    
    # Common paths where numpy temporary files might be
    site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    user_site_packages = Path(os.environ.get('APPDATA', '')) / 'Python' / f'Python{sys.version_info.major}{sys.version_info.minor}' / 'site-packages'
    
    # Numpy related directories to clean
    numpy_dirs = [
        site_packages / 'numpy',
        site_packages / 'numpy-*.dist-info',
        site_packages / '~umpy',
        site_packages / '~umpy.libs',
        user_site_packages / 'numpy',
        user_site_packages / 'numpy-*.dist-info',
        user_site_packages / '~umpy',
        user_site_packages / '~umpy.libs',
    ]
    
    # Clean up numpy directories
    for dir_pattern in numpy_dirs:
        # Handle glob patterns
        if '*' in str(dir_pattern):
            for path in dir_pattern.parent.glob(dir_pattern.name):
                remove_directory(Path(path))
        else:
            remove_directory(dir_pattern)
    
    # Clean up any remaining temporary files
    temp_dir = Path(tempfile.gettempdir())
    for temp_file in temp_dir.glob('*numpy*'):
        try:
            if temp_file.is_file():
                temp_file.unlink()
                print(f"Removed temporary file: {temp_file}")
            elif temp_file.is_dir():
                shutil.rmtree(temp_file, ignore_errors=True)
                print(f"Removed temporary directory: {temp_file}")
        except Exception as e:
            print(f"Warning: Could not remove {temp_file}: {e}")
    
    print("Numpy cleanup completed.")

def install_package(package_name, version=None, extra_args=None):
    """Helper function to install a package with error handling."""
    if version:
        package_spec = f"{package_name}=={version}"
    else:
        package_spec = package_name
    
    cmd = [
        sys.executable, "-m", "pip", "install",
        "--only-binary=:all:",
        "--find-links", "https://download.lfd.uci.edu/pythonlibs/archived/",
        "--find-links", "https://pypi.anaconda.org/scientific-python-nightly-wheels/simple",
        "--prefer-binary",
        "--no-cache-dir"
    ]
    
    if extra_args:
        cmd.extend(extra_args)
    
    cmd.append(package_spec)
    
    try:
        print(f"Installing {package_spec}...")
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
            capture_output=True
        )
        print(result.stdout)
        if result.stderr:
            print(f"Warning during installation: {result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install {package_spec}:")
        print(f"Exit code: {e.returncode}")
        print(f"Output: {e.stdout}")
        print(f"Error: {e.stderr}")
        return False

def reinstall_packages():
    """Reinstall numpy and related packages with versions compatible with Python 3.13."""
    print("\nReinstalling packages with Python 3.13 compatible versions...")
    
    # First, ensure we have the latest pip and setuptools
    print("\nUpgrading pip and setuptools...")
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"], 
                  check=True)
    
    # Uninstall numpy and related packages
    packages_to_remove = [
        "numpy", "pandas", "pandas-ta", "scipy", "scikit-learn", "numba"
    ]
    
    print("\nUninstalling existing packages...")
    for pkg in packages_to_remove:
        try:
            subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", pkg], 
                         check=False, 
                         stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE)
        except Exception as e:
            print(f"Warning: Could not uninstall {pkg}: {e}")
    
    # Clear pip cache
    print("\nClearing pip cache...")
    subprocess.run([sys.executable, "-m", "pip", "cache", "purge"], 
                  check=False,
                  stdout=subprocess.PIPE,
                  stderr=subprocess.PIPE)
    
    # Install packages in the correct order with specific versions
    # that are known to work with Python 3.13
    packages = [
        ("numpy", "2.2.6"),      # Latest stable numpy for Python 3.13
        ("scipy", "1.14.1"),     # Latest scipy with Python 3.13 wheels
        ("pandas", "2.2.3"),     # Latest pandas with Python 3.13 support
        ("numba", "0.61.2"),     # Latest numba with Python 3.13 support
        ("scikit-learn", "1.7.0"),# Latest scikit-learn with Python 3.13 support
        ("pandas-ta", "0.3.14b0"),  # Specific version known to work with Python 3.13
        ("pandas-datareader", None),  # Often needed with pandas-ta
    ]
    
    print("\nInstalling required packages...")
    for package_spec in packages:
        if len(package_spec) == 3:
            package, version, extra_args = package_spec
        else:
            package, version = package_spec
            extra_args = None
        install_package(package, version, extra_args)
    
    print("\nPackage reinstallation completed. Some warnings may be present.")

def verify_installation():
    """Verify that packages were installed correctly."""
    print("\nVerifying installations...")
    
    try:
        import numpy as np
        import pandas as pd
        import scipy
        import sklearn
        import numba
        
        print("\nCore packages imported successfully!")
        print(f"Python version: {sys.version}")
        print(f"numpy version: {np.__version__}")
        print(f"pandas version: {pd.__version__}")
        print(f"scipy version: {scipy.__version__}")
        print(f"scikit-learn version: {sklearn.__version__}")
        print(f"numba version: {numba.__version__}")
        
        # Test basic numpy functionality
        arr = np.array([1, 2, 3])
        assert len(arr) == 3, "Basic numpy test failed"
        print("\nBasic numpy functionality test passed!")
        
        # Try to import pandas-ta, but don't fail if it doesn't work
        try:
            import pandas_ta as ta
            print(f"pandas-ta version: {ta.__version__}")
        except ImportError:
            print("\nWarning: pandas-ta could not be imported. Some functionality may be limited.")
            print("You can try installing it manually with: pip install pandas-ta")
        
        return True
    except Exception as e:
        print(f"\nError during verification: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("Python Package Cleanup and Reinstallation Tool")
    print("=" * 60)
    
    # Run as administrator check (Windows)
    if os.name == 'nt':
        import ctypes
        if not ctypes.windll.shell32.IsUserAnAdmin():
            print("\nWARNING: Please run this script as Administrator to ensure proper cleanup.")
            print("Right-click on the script and select 'Run as administrator'\n")
    
    # Clean up numpy installation
    clean_numpy_installation()
    
    # Reinstall packages
    reinstall_packages()
    
    # Verify installation
    success = verify_installation()
    
    if success:
        print("\n" + "=" * 60)
        print("Cleanup and reinstallation completed successfully!")
        print("=" * 60)
    else:
        print("\n" + "!" * 60)
        print("There were issues during the cleanup and reinstallation process.")
        print("Please review the output for any error messages.")
        print("!" * 60)

if __name__ == "__main__":
    main()
