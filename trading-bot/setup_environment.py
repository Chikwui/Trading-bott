#!/usr/bin/env python3
"""
Environment setup script for the AI Trading Bot project.
This script automates the setup of the development environment.
"""
import os
import sys
import subprocess
import platform
import shutil
import venv
from pathlib import Path
from typing import List, Optional, Tuple

# Project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# File paths - in order of installation priority
REQUIREMENTS_FILES = [
    "requirements.txt",        # Core dependencies first
    "requirements-ml.txt",     # Then ML-specific dependencies
    "requirements-test.txt",   # Then test dependencies
    "requirements-dev.txt"     # Finally, development tools
]

# Platform-specific configurations
IS_WINDOWS = platform.system() == 'Windows'
VENV_DIR = PROJECT_ROOT / 'venv'
PYTHON = sys.executable
PIP = 'pip'

# Colors for console output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

# Platform-specific commands
IS_WINDOWS = platform.system() == "Windows"
PYTHON = "python" + ("3" if not IS_WINDOWS else "")
PIP = "pip"
VENV_DIR = "venv"
VENV_ACTIVATE = os.path.join(VENV_DIR, "Scripts" if IS_WINDOWS else "bin", "activate")


def run_command(command: List[str], cwd: Optional[str] = None, shell: bool = False) -> bool:
    """Run a shell command and return True if successful."""
    try:
        print(f"Running: {' '.join(command)}")
        result = subprocess.run(
            command,
            cwd=cwd or PROJECT_ROOT,
            shell=shell,
            check=True,
            text=True,
            capture_output=True
        )
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False


def setup_venv() -> bool:
    """Set up a Python virtual environment."""
    if os.path.exists(VENV_DIR):
        print(f"Virtual environment already exists at {VENV_DIR}")
        return True
    
    print(f"Creating virtual environment in {VENV_DIR}...")
    return run_command([PYTHON, "-m", "venv", VENV_DIR])


def install_requirements() -> bool:
    """Install all requirements files."""
    for req_file in REQUIREMENTS_FILES:
        req_path = os.path.join(PROJECT_ROOT, req_file)
        if not os.path.exists(req_path):
            print(f"Warning: {req_file} not found, skipping...")
            continue
            
        print(f"\nInstalling {req_file}...")
        if not run_command([PIP, "install", "-r", req_path]):
            print(f"Failed to install {req_file}")
            return False
    return True


def setup_pre_commit() -> bool:
    """Set up pre-commit hooks."""
    print("\nSetting up pre-commit hooks...")
    return run_command(["pre-commit", "install"])


def main() -> int:
    """Main setup function."""
    print("=" * 50)
    print("AI Trading Bot - Environment Setup")
    print("=" * 50)
    
    # Setup virtual environment
    if not setup_venv():
        return 1
    
    # Activate virtual environment and update pip
    if not run_command([PIP, "install", "--upgrade", "pip"]):
        return 1
    
    # Install requirements
    if not install_requirements():
        return 1
    
    # Setup pre-commit hooks
    if not setup_pre_commit():
        print("Warning: Failed to set up pre-commit hooks")
    
    print("\n" + "=" * 50)
    print("Setup completed successfully!")
    print("To activate the virtual environment, run:")
    if IS_WINDOWS:
        print(f"  .\\{VENV_ACTIVATE}")
    else:
        print(f"  source {VENV_ACTIVATE}")
    print("=" * 50)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
