# Ticket 3: Refactor NSGA2Worker as an Adapter

## Objective
Update the existing Qt Worker to use the new headless engine, restoring functionality to the GUI.

## Tasks
1. Open `codes/workers/NSGA2Worker.py`.
2. Remove all DEAP logic.
3. Refactor the `__init__` to instantiate the `ExecutionController`.
4. Refactor the `run()` method to build the `OptimizationConfig` and `NSGA2Hyperparameters` from the raw inputs.
5. Instantiate `NSGA2Engine` and call its `run()` method, mapping `self.progress.emit` and `self.finished.emit` to the engine's callbacks.
6. Refactor the `stop()`, `pause()`, and `resume()` methods to simply call the corresponding methods on the `ExecutionController`.
7. Verify the UI still behaves identically when running the NSGA-II optimization.

## Blockers
- Blocks on Ticket 2 (002-extract-nsga2-engine.md)
