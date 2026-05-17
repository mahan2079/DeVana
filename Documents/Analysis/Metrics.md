# Computational Metrics

## Overview
DeVana includes a comprehensive benchmarking and metrics system (`computational_metrics_new.py`) to monitor the resource efficiency and numerical stability of the optimization algorithms.

## Monitored Metrics
- **System Resource Usage**:
  - **CPU Utilization**: Total and per-core usage.
  - **Memory Footprint**: Process RSS, VMS, and total system available memory.
  - **Thread Count**: Monitoring parallel execution overhead.
- **Optimization Performance**:
  - **Generation Timings**: Time taken per generation/iteration.
  - **Evaluation Latency**: Time spent specifically in FRF calculations.
  - **Convergence Rate**: Tracking fitness improvement over time.
- **Algorithm-Specific Metrics**:
  - **GA**: Diversity and mutation/crossover rate adaptations.
  - **PSO**: Swarm diversity and inertia weight decay.
  - **SA**: Temperature history and acceptance rates.
  - **RL**: Epsilon decay and reward history.

## Visualization Suite
The metrics system provides real-time feedback and post-optimization reports, including:
- **Resource Time-series**: Overlaying CPU/Memory usage with generation markers.
- **Hardware Profile**: Capturing system architecture (CPU freq, Cores, Total RAM).
- **Metric Breakdown**: Visualizing where time is spent (e.g., Evaluation vs. Algorithm Overhead).
