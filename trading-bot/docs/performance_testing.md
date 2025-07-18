# Performance Testing Framework

This document provides comprehensive documentation for the ML Pipeline Performance Testing Framework, including setup, usage, and best practices for performance monitoring and regression detection.

## Table of Contents

1. [Overview](#overview)
2. [Getting Started](#getting-started)
3. [Running Tests](#running-tests)
4. [Performance Analysis](#performance-analysis)
5. [Visualization](#visualization)
6. [CI/CD Integration](#cicd-integration)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## Overview

The Performance Testing Framework provides tools for:

- **Performance Benchmarking**: Measure execution time, memory usage, and resource utilization
- **Stress Testing**: Test system behavior under extreme conditions
- **Regression Detection**: Identify performance regressions across code changes
- **Visualization**: Generate interactive reports and visualizations
- **CI/CD Integration**: Automated performance gates and reporting

## Getting Started

### Prerequisites

- Python 3.8+
- Poetry (for dependency management)
- Required system dependencies (for Linux):
  ```bash
  sudo apt-get update
  sudo apt-get install -y python3-dev python3-pip
  ```

### Installation

1. Install project dependencies:
   ```bash
   poetry install --with test,ml
   ```

2. Install additional test dependencies:
   ```bash
   poetry add --group dev pytest pytest-benchmark pytest-cov pytest-xdist
   ```

## Running Tests

### Unit Tests

```bash
# Run all unit tests
poetry run pytest tests/unit/

# Run with coverage report
poetry run pytest tests/unit/ --cov=core --cov-report=html
```

### Integration Tests

```bash
# Run all integration tests
poetry run pytest tests/ml/integration/

# Run specific test file
poetry run pytest tests/ml/integration/test_feature_store.py
```

### Performance Benchmarks

```bash
# Run all benchmarks
poetry run pytest tests/ml/integration/test_performance.py -v --benchmark-warmup=on

# Run specific benchmark with more iterations
poetry run pytest tests/ml/integration/test_performance.py::TestPerformance::test_feature_store_performance \
  --benchmark-warmup=on \
  --benchmark-warmup-iterations=5 \
  --benchmark-min-rounds=20
```

### Stress Tests

```bash
# Run all stress tests
poetry run pytest tests/ml/integration/test_stress.py -m stress

# Run with higher load
STRESS_MULTIPLIER=5 poetry run pytest tests/ml/integration/test_stress.py -m stress
```

## Performance Analysis

### Analyzing Results

```bash
# Generate performance report
poetry run python scripts/analyze_performance.py --results-dir=test-results

# Update baseline with current results
poetry run python scripts/analyze_performance.py --update-baseline

# Set custom regression threshold (default: 1.2x)
poetry run python scripts/analyze_performance.py --threshold 1.1
```

### Understanding the Output

The analysis script generates:

1. **Performance Report** (`performance-report.md`):
   - Summary of test executions
   - Detected regressions and improvements
   - Statistical significance of changes
   - Detailed metrics comparison

2. **Baseline Data** (`test-results/baseline/performance_baseline.json`):
   - Historical performance data
   - Statistical metrics for each test
   - Timestamp of last update

## Visualization

### Generating Visualizations

```bash
# Generate HTML report with visualizations
poetry run python scripts/visualize_performance.py

# Specify custom results directory
poetry run python scripts/visualize_performance.py --results-dir=test-results
```

The visualization script creates:

1. **HTML Report** (`test-results/reports/performance_report.html`):
   - Interactive performance trends
   - Metric distributions
   - Correlation heatmaps

2. **Image Files** (in `test-results/reports/`):
   - Trend plots for key metrics
   - Distribution plots
   - Correlation heatmaps

## CI/CD Integration

The framework includes GitHub Actions workflows for automated testing and performance monitoring:

### Workflows

1. **CI Pipeline** (`.github/workflows/ci-cd.yml`):
   - Runs on every push and PR
   - Executes unit and integration tests
   - Generates coverage reports
   - Uploads test artifacts

2. **Performance Regression Check**:
   - Runs on schedule and manual trigger
   - Compares current performance against baseline
   - Fails on significant regressions
   - Posts results as PR comments

### Configuration

Customize the CI pipeline by updating `.github/workflows/ci-cd.yml`:

```yaml
jobs:
  test:
    # Configure test matrix
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest]
        python-version: ['3.8', '3.9', '3.10']
    
  performance-regression:
    # Adjust regression thresholds
    env:
      PERFORMANCE_THRESHOLD: 1.15  # 15% regression threshold
      SIGNIFICANCE_LEVEL: 0.01     # 1% significance level
```

## Best Practices

### Writing Performance Tests

1. **Isolate Test Cases**:
   - Each test should measure one specific operation
   - Avoid external dependencies when possible
   - Use fixtures for common setup/teardown

2. **Use Appropriate Assertions**:
   ```python
   def test_feature_performance(benchmark):
       result = benchmark(my_function, arg1, arg2)
       assert result is not None
       assert benchmark.stats.stats['mean'] < 1.0  # Require < 1 second
   ```

3. **Handle Test Data**:
   - Use realistic test data sizes
   - Consider both best-case and worst-case scenarios
   - Document data assumptions

### Monitoring Production Performance

1. **Key Metrics to Track**:
   - Latency percentiles (p50, p90, p99)
   - Memory usage and garbage collection
   - CPU utilization
   - I/O operations

2. **Alerting Thresholds**:
   - Set thresholds based on historical data
   - Use statistical methods to detect anomalies
   - Implement gradual degradation detection

## Troubleshooting

### Common Issues

1. **Performance Variability**:
   - Run tests multiple times
   - Use warmup iterations
   - Ensure consistent test environment

2. **Memory Leaks**:
   - Check for unbounded data structures
   - Monitor object creation/destruction
   - Use memory profiler for detailed analysis

3. **Test Failures**:
   - Check for external dependencies
   - Verify test data consistency
   - Review test environment setup

### Debugging Performance Issues

1. **Profile CPU Usage**:
   ```bash
   python -m cProfile -o profile.cprof -m pytest tests/...
   snakeviz profile.cprof
   ```

2. **Analyze Memory Usage**:
   ```bash
   mprof run --include-children python -m pytest tests/...
   mprof plot
   ```

3. **Inspect System Resources**:
   ```bash
   # Monitor system resources
   htop  # or top/Activity Monitor
   ```

## Support

For issues and feature requests, please [open an issue](https://github.com/your-org/your-repo/issues).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
