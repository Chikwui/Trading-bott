#!/usr/bin/env python3
"""
Run ML Pipeline integration tests with support for stress tests and benchmarks.
"""
import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from typing import List, Optional

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Test configuration
DEFAULT_STRESS_MULTIPLIER = 1
DEFAULT_ITERATIONS = 5
DEFAULT_WORKERS = 4

def run_tests(
    test_path: str = 'tests/ml/integration',
    pattern: str = 'test_*.py',
    markers: Optional[List[str]] = None,
    stress_multiplier: int = DEFAULT_STRESS_MULTIPLIER,
    verbose: bool = False
) -> bool:
    """Run tests with specified parameters.
    
    Args:
        test_path: Path to test directory
        pattern: Test file pattern
        markers: List of pytest markers to include
        stress_multiplier: Multiplier for stress test parameters
        verbose: Enable verbose output
    
    Returns:
        bool: True if all tests passed, False otherwise
    """
    # Set environment variables
    env = os.environ.copy()
    env['STRESS_TEST_MULTIPLIER'] = str(stress_multiplier)
    
    # Build pytest command
    cmd = [
        'pytest',
        '--cov=core',
        '--cov-report=term-missing',
        '--durations=10',
    ]
    
    if verbose:
        cmd.append('-v')
    
    # Add markers if specified
    if markers:
        mark_expr = ' or '.join(markers)
        cmd.extend(['-m', mark_expr])
    
    # Add test path and pattern
    cmd.extend([
        '--junitxml=test-results/junit.xml',
        '--cov-report=xml:test-results/coverage.xml',
        test_path,
        '-k', pattern
    ])
    
    # Run the tests
    print(f"Running tests with command: {' '.join(cmd)}")
    print(f"Stress test multiplier: {stress_multiplier}")
    
    start_time = time.time()
    result = subprocess.run(cmd, env=env)
    duration = time.time() - start_time
    
    print(f"\nTests completed in {duration:.2f} seconds")
    return result.returncode == 0

def run_benchmarks(
    iterations: int = DEFAULT_ITERATIONS,
    workers: int = DEFAULT_WORKERS,
    verbose: bool = False
) -> bool:
    """Run performance benchmarks.
    
    Args:
        iterations: Number of iterations per benchmark
        workers: Number of worker processes
        verbose: Enable verbose output
    
    Returns:
        bool: True if benchmarks completed successfully
    """
    cmd = [
        'pytest',
        'tests/ml/integration/test_edge_cases_perf.py::TestPerformance',
        '-m', 'not stress',
        '--benchmark-warmup=on',
        f'--benchmark-warmup-iterations={iterations // 2}',
        f'--benchmark-min-rounds={iterations}',
        f'--benchmark-max-time=60',
        f'--benchmark-disable-gc',
        f'--benchmark-json=test-results/benchmark.json',
        '-v' if verbose else ''
    ]
    
    print(f"Running benchmarks with {iterations} iterations and {workers} workers...")
    return subprocess.run(cmd).returncode == 0

def run_stress_tests(
    multiplier: int = 1,
    verbose: bool = False
) -> bool:
    """Run stress tests with the given multiplier.
    
    Args:
        multiplier: Stress test multiplier
        verbose: Enable verbose output
    
    Returns:
        bool: True if stress tests passed
    """
    print(f"Running stress tests with multiplier: {multiplier}")
    return run_tests(
        test_path='tests/ml/integration/test_edge_cases_perf.py::TestStress',
        markers=['stress'],
        stress_multiplier=multiplier,
        verbose=verbose
    )

def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(description='Run ML Pipeline tests')
    
    # Test selection
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--unit', action='store_true', help='Run only unit tests')
    group.add_argument('--integration', action='store_true', help='Run only integration tests')
    group.add_argument('--benchmark', action='store_true', help='Run only benchmarks')
    group.add_argument('--stress', action='store_true', help='Run only stress tests')
    
    # Test configuration
    parser.add_argument('--pattern', default='test_*.py', help='Test file pattern')
    parser.add_argument('--marker', action='append', help='Only run tests with this marker')
    parser.add_argument('--stress-multiplier', type=int, default=DEFAULT_STRESS_MULTIPLIER,
                      help='Stress test multiplier')
    parser.add_argument('--iterations', type=int, default=DEFAULT_ITERATIONS,
                      help='Number of benchmark iterations')
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS,
                      help='Number of worker processes')
    parser.add_argument('-v', '--verbose', action='store_true', help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Create test results directory
    (Path(project_root) / 'test-results').mkdir(exist_ok=True)
    
    # Determine which tests to run
    success = True
    
    if args.benchmark:
        success = run_benchmarks(
            iterations=args.iterations,
            workers=args.workers,
            verbose=args.verbose
        )
    elif args.stress:
        success = run_stress_tests(
            multiplier=args.stress_multiplier,
            verbose=args.verbose
        )
    else:
        # Default: run all tests
        test_path = 'tests/'
        if args.unit:
            test_path = 'tests/unit/'
        elif args.integration:
            test_path = 'tests/ml/integration/'
        
        success = run_tests(
            test_path=test_path,
            pattern=args.pattern,
            markers=args.marker,
            stress_multiplier=args.stress_multiplier,
            verbose=args.verbose
        )
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    print("ML Pipeline Test Runner")
    print("=" * 50)
    
    try:
        main()
    except KeyboardInterrupt:
        print("\nTest run interrupted by user.")
        sys.exit(1)
