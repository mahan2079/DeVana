# Ticket 2: Extract NSGA2Engine

## Objective
Extract the DEAP mathematical logic out of the PyQt worker and into a pure Python engine.

## Tasks
1. Create `devana_lib/optimization/engines/nsga2.py`.
2. Implement `AlgorithmHyperparameters` for NSGA-II inside this file (or in `config.py`).
3. Implement `NSGA2Engine.run()`. It must take the config, the `ExecutionController`, and Python callbacks for `on_progress`, `on_finished`, and `on_error`.
4. Move the DEAP evaluation loop, initialization, and iteration logic from the current `NSGA2Worker.py` into this engine.
5. Ensure the loop checks `controller.should_abort()` and `controller.wait_if_paused()`.
6. Write a small, headless unit test in `tests/` that runs the engine natively (without Qt).

## Blockers
- Blocks on Ticket 1 (001-core-config.md)
