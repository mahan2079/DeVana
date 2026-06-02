# DeVana Test Suite

This directory contains the comprehensive test suite for the DeVana framework.

## Running Tests

To run all tests, use the `run_all.py` script from the project root:

```bash
python tests/run_all.py
```

Or run individual test files:

```bash
python -m unittest tests/test_frf.py
```

## Test Structure

- `test_frf.py`: Tests the core physics engine and Frequency Response Function calculations.
- `test_workers.py`: Verifies the multi-threaded optimization workers (GA, PSO, SA, etc.).
- `test_integration.py`: Integration tests for the full system workflow.
- `test_metrics.py`: Tests the benchmarking and computational metrics system.
- `test_sobol.py`: Verifies global sensitivity analysis logic.
- `test_adavea.py` / `test_nsga2.py`: Tests for multi-objective optimization algorithms.

## Requirements

Ensure all dependencies are installed:

```bash
pip install -r requirements.txt
```

Note: Some GUI-related tests may require a display environment. If running in a CI/CD pipeline without a GUI, use a virtual framebuffer like `xvfb`.
