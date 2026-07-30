# Ticket 1: Core Configuration and ExecutionController

## Objective
Establish the foundational domain models and lifecycle adapters required for the headless optimization engines.

## Tasks
1. Create `devana_lib/optimization/config.py`.
2. Implement the `PhysicsConfig`, `SparsityConfig`, and `OptimizationConfig` dataclasses exactly as specified in the technical spec.
3. Create `devana_lib/optimization/controller.py`.
4. Implement the `ExecutionController` class with `should_abort()`, `pause()`, `resume()`, and `wait_if_paused()` logic using `threading.Event` to ensure it is thread-safe.

## Blockers
- None
