---
labels: ready-for-agent
---

# Specification: Headless Algorithm Engine Extraction

## Problem Statement

The DeVana mathematical optimization logic (specifically the DEAP loop for NSGA-II and other algorithms) is currently tightly coupled to the PyQt `QThread` infrastructure. The workers (like `NSGA2Worker`) inherit directly from `QThread`, manage Qt signals inline, and accept over 20 unstructured parameters. This makes the algorithms impossible to run natively in a Docker/REST API environment without booting a GUI application, violates the single responsibility principle, and creates shallow modules that are difficult to test in isolation.

## Solution

We will extract the pure mathematical optimization logic out of the `QThread` workers into headless, standalone `AlgorithmEngine` classes housed within `devana_lib/`. The engines will accept a unified `OptimizationConfig` interface and an `ExecutionController` for lifecycle management, and will communicate state via standard Python callbacks. The Qt workers will be refactored into thin adapters that translate the Qt UI commands into these pure domain models.

## User Stories

1. As a developer, I want to define `PhysicsConfig`, `SparsityConfig`, and `AlgorithmHyperparameters` so that I can group the 20+ loose parameters into a clean, strongly-typed `OptimizationConfig` interface.
2. As a developer, I want an `ExecutionController` object that handles `abort()`, `pause()`, and `resume()` so that the engine lifecycle can be managed without the engine knowing about PyQt threads.
3. As a developer, I want a pure Python `NSGA2Engine` class in `devana_lib/optimization/engines/` so that the DEAP logic can run completely independent of any UI event loop.
4. As a REST API consumer, I want to instantiate the `NSGA2Engine` directly and run it headlessly so that optimizations can be triggered via HTTP in a Docker container.
5. As an end-user, I want the `NSGA2Worker` to act as a thin adapter so that I can still run optimizations and see live progress via the PyQt GUI exactly as I did before.

## Implementation Decisions

- **Domain Glossary Enforcement**: We will adhere strictly to the terms defined in `CONTEXT.md` (`AlgorithmEngine`, `OptimizationConfig`, `ExecutionController`).
- **Engine Location**: The pure engines will live in `devana_lib/optimization/engines/`. This is the physical seam; they must never import from `codes/`.
- **Adapter Location**: The existing Qt workers (e.g., `codes/workers/NSGA2Worker.py`) will remain in place but will be gutted of DEAP logic, serving only to map `OptimizationConfig` to the engine and translating Python callbacks back to `pyqtSignal` events.
- **Callback Interface**: Progress and completion will be handled via simple `on_progress(metrics)` and `on_finished(results)` callbacks passed into the engine's `run()` method.

## Testing Decisions

Tests will focus on the external behavior at the highest possible seams without testing internal implementation details.

- **Engine Seam**: We will write headless unit tests that instantiate the `NSGA2Engine` with a dummy `OptimizationConfig`. We will assert that the `on_progress` and `on_finished` callbacks are executed correctly and yield expected data structures, completely ignoring the internal DEAP mechanics.
- **Lifecycle Seam**: We will test the `ExecutionController` to ensure that `abort()` correctly flips the state and halts a running engine.
- **Adapter Seam**: We will mock the `NSGA2Engine` to assert that the `NSGA2Worker` translates engine callbacks into the correct `pyqtSignal` emissions.

## Out of Scope

- Migrating every single algorithm immediately. This specification acts as the "tracer bullet" focused solely on `NSGA-II`. Once the pattern is proven, other algorithms (PSO, GA, AdaVEA) will be migrated in subsequent tickets.
- UI Redesign: The GUI layout and interaction flow will remain identical.
- Modifying the core DEAP mathematical constraints. The optimization results must remain identical.

## Further Notes

- By implementing this, we unlock the full potential of the headless REST API component that was requested for Docker environments.
- This serves as the blueprint for deepening all stochastic optimizers in the codebase.
