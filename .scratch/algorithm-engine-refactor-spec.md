# Technical Specification: Algorithm Engine Decoupling

## Objective
Extract the mathematical DEAP optimization logic out of the PyQt `QThread` workers into headless `AlgorithmEngine` classes to ensure they can run reliably in Docker/REST environments, and to deepen the architectural modules.

## 1. Directory Structure and Boundaries
- All pure mathematical logic and engine implementations will be moved to `devana_lib/optimization/`.
- `codes/workers/` will only contain the `QThread` **adapters** that bridge the PyQt UI to the engines.
- `codes/` must never be imported by `devana_lib/`.

## 2. The Configuration Interface
Instead of passing 20+ parameters directly to the engine, we will use a consolidated `OptimizationConfig` composed of logical models:

```python
from dataclasses import dataclass

@dataclass
class PhysicsConfig:
    main_system_parameters: list
    target_values_weights: list
    omega_start: float
    omega_end: float
    omega_points: int

@dataclass
class SparsityConfig:
    tau: float
    alpha: float
    beta: float

@dataclass
class AlgorithmHyperparameters:
    # Example for NSGA-II
    pop_size: int
    generations: int
    cxpb: float
    mutpb: float
    eta_c: float
    eta_m: float
    indpb: float
    num_runs: int
    # ...

@dataclass
class OptimizationConfig:
    physics: PhysicsConfig
    sparsity: SparsityConfig
    hyperparameters: AlgorithmHyperparameters
```

## 3. The Execution Lifecycle (ExecutionController)
To allow the UI or REST API to pause or stop a running optimization without polluting the engine with threading logic, we will pass an `ExecutionController`.

```python
class ExecutionController:
    def __init__(self):
        self._abort = False
        self._paused = False
    
    def should_abort(self) -> bool:
        return self._abort
        
    def abort(self):
        self._abort = True
        
    def pause(self):
        self._paused = True
        
    def resume(self):
        self._paused = False
        
    def wait_if_paused(self):
        # Implementation using time.sleep or threading.Event
        pass
```

## 4. The AlgorithmEngine Interface
The engine itself will expose a simple `run` method that accepts the config, the controller, and standard Python callbacks.

```python
class NSGA2Engine:
    def run(self, 
            config: OptimizationConfig, 
            controller: ExecutionController,
            on_progress: callable, 
            on_finished: callable, 
            on_error: callable):
        
        # Internal DEAP loop runs here...
        # At the start of each generation:
        if controller.should_abort():
            break
        controller.wait_if_paused()
        
        # At the end of each generation:
        on_progress(run_idx, current_gen, total_gens, metrics)
        
        # At the end of the optimization:
        on_finished(all_runs_data)
```

## 5. The QThread Adapter
The existing workers (e.g., `NSGA2Worker`) will be gutted. They will simply build the `OptimizationConfig`, instantiate the `ExecutionController`, and pass lambdas to wrap their `pyqtSignals`.

```python
class NSGA2WorkerAdapter(QThread):
    progress = pyqtSignal(int, int, int, dict)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def run(self):
        config = self._build_config()
        engine = NSGA2Engine()
        
        engine.run(
            config=config,
            controller=self.controller,
            on_progress=lambda r, c, t, m: self.progress.emit(r, c, t, m),
            on_finished=lambda d: self.finished.emit(d),
            on_error=lambda e: self.error.emit(e)
        )
```
