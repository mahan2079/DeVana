# Domain Glossary

This file serves as the canonical domain model and ubiquitous language for DeVana. Use these terms exactly as defined here in code, UI, and documentation.

## Core Optimization Concepts

- **AlgorithmEngine**: A headless, pure-Python implementation of a specific mathematical optimization algorithm (e.g., NSGA-II, PSO). It is a deep module entirely decoupled from UI or threading infrastructure.
- **OptimizationConfig**: The aggregate configuration required to execute an `AlgorithmEngine`. It acts as the primary interface boundary and is composed of more specific configuration groupings:
  - **PhysicsConfig**: The domain model representing the physical constraints, target weights, and FRF (Frequency Response Function) variables of the structural system. These are common across all algorithms.
  - **SparsityConfig**: The constraints and penalties applied to enforce sparsity in the DVA layout (e.g., `sparsity_tau`, `sparsity_alpha`). Common across all algorithms.
  - **AlgorithmHyperparameters**: The configuration values specific to the inner workings of a given algorithm (e.g., crossover probability for GA, inertia for PSO).
- **ExecutionController**: A state object passed into the engine that manages lifecycle events (pause, resume, abort). It acts as an adapter bridging the infrastructure (e.g., UI threads, REST requests) to the core engine.
