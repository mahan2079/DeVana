# Optimization Architecture Overview

The DeVana framework employs a **two-tiered architecture** for optimization routines, providing both rich GUI interactivity and lightweight headless execution capabilities. This architecture ensures high cohesion and low coupling between the evolutionary algorithms and the user interface.

## 1. Two-Tiered Optimization Architecture

### 1.1 Core Solvers (`devana_lib/optimization/engines/` & `devana_lib/optimize/`)
The foundation of the optimization suite is located in the domain layer, offering GUI-independent execution.

- **Headless Algorithm Engines (e.g., `NSGA2Engine`)**: Pure mathematical engines housed in `devana_lib/optimization/engines/`. They accept a strongly-typed `OptimizationConfig` interface and an `ExecutionController` for lifecycle management (`abort()`, `pause()`). 
- **Legacy Solvers (`base.py`)**: An abstract base class (`Solver`) that older algorithms implement, transitioning towards the new Headless Engine pattern.
- **Purpose**: Primarily used by the headless REST API (`codes/api/`) and automated testing scripts, ensuring the mathematical core runs in Docker or background workers without any PyQT dependencies. They communicate state changes via standard Python callbacks (`on_progress`, `on_finished`).

### 1.2 Worker Adapters (`codes/workers/`)
The GUI execution layer utilizes `QThread` subclasses located in `codes/workers/`, acting as thin adapters for the core engines.

- **Inheritance**: Subclasses `QThread` from `PyQt5.QtCore`.
- **Communication**: The worker translates Python callbacks emitted by the engine into PyQT signals (`finished`, `progress`, `error`) to interact safely with the UI thread.
- **Features**: Instantiates the `OptimizationConfig` and `ExecutionController` from unstructured UI data, invoking the engine's `run()` method in the background.

```mermaid
classDiagram
    class QThread {
        <<PyQt5>>
    }
    class OptimizationConfig {
        <<DataClass>>
    }
    class ExecutionController {
        +abort()
        +pause()
        +resume()
    }
    class NSGA2Engine {
        +run(config, controller, callbacks)
    }
    
    class NSGA2Worker {
        +run()
        -signals
    }
    
    QThread <|-- NSGA2Worker
    NSGA2Worker *-- OptimizationConfig : builds
    NSGA2Worker *-- ExecutionController : manages
    NSGA2Worker ..> NSGA2Engine : invokes
```

## 2. Universal Fitness and Penalty Formulation

Regardless of the selected metaheuristic, the evaluation heavily relies on the Frequency Response Function (FRF) and enforces structural constraints.

The generalized fitness scalar $F$ (for single-objective algorithms like GA, DE, PSO) is formulated as:

$$ F = O_{primary} + P_{sparsity} + P_{error} + P_{activation} + C_{cost} $$

Where:
- **Primary Objective ($O_{primary}$)**: Distance from the normalized ideal response.
  $$ O_{primary} = | R_{singular} - 1.0 | $$
- **Sparsity Penalty ($P_{sparsity}$)**: L1 regularization to favor simpler absorber topologies (Occam's Razor).
  $$ P_{sparsity} = \alpha \sum_{i} |x_i| $$
- **Percentage Error ($P_{error}$)**: Scaled sum of percentage deviations across all masses.
- **Activation Penalty ($P_{activation}$)**: Penalizes active variables above a certain threshold.
- **Cost Term ($C_{cost}$)**: Evaluates manufacturing/material costs (can be a standard normalized cost or an advanced Cost-Benefit Ratio formulation).

For **Multi-Objective Algorithms** (e.g., NSGA-II, MOGA), these elements are decoupled into orthogonal objectives:
1. $f_1(x) = R_{singular}$ (or distance)
2. $f_2(x) = P_{sparsity}$
3. $f_3(x) = \sum C_i x_i$

## 3. Supported Algorithms & Frameworks
- **DEAP-based**: GA, NSGA-II, MOGA.
- **Custom / NumPy implementations**: RL (DDPG-inspired Policy Gradient), AdaVEA.
- **Hybrid Integrations**: Algorithms are infused with ML surrogates, QMC initialization, and PINN accelerators.

```mermaid
graph TD
    A["Optimization Request"] --> B{"Environment"}
    B -- GUI --> C["Worker Threads<br>"codes/workers/\""]
    B -- API / CLI --> D["Core Solvers<br>"devana/optimize/\""]
    
    C --> E["Evaluation Loop"]
    D --> E
    
    E --> F{"Accelerator Engine"}
    F -- Surrogate Model --> G["Neural/KNN Screening"]
    F -- PINN --> H["Physics-Informed Neural Net"]
    F -- Exact --> I["FRF Analytical Solver"]
    
    G --> I
    H -.-> I
    
    I --> J["Fitness + Penalties"]
    J --> K["Algorithm Updates<br>"Selection/Mutation/Policy\""]
```

#### Pseudo-code
```text
BEGIN
  EXECUTE Optimization Request
  EXECUTE Environment
  EXECUTE Worker Threads<br>
  EXECUTE Core Solvers<br>
  EXECUTE Evaluation Loop
  EXECUTE Accelerator Engine
  EXECUTE Neural/KNN Screening
  EXECUTE Physics-Informed Neural Net
  EXECUTE FRF Analytical Solver
  EXECUTE Fitness + Penalties
  EXECUTE Algorithm Updates<br>
END
```