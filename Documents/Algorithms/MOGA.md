# Multi-Objective Genetic Algorithm (MOGA)

## Overview
The MOGA module (`MOGAWorker.py`) is designed to handle optimization problems with multiple conflicting objectives. While the implementation in this codebase is a placeholder, the architecture is designed to support the search for a Pareto-optimal front where no single objective can be improved without degrading another.

## Intended Features
- **Non-dominated Sorting**: Ranking individuals based on Pareto dominance.
- **Diversity Maintenance**: Using mechanisms like crowding distance or niching to ensure a well-spread Pareto front.
- **Conflicting Objectives**:
  - **Minimizing Vibrations**: Achieving a singular response close to 1.
  - **Minimizing Cost**: Reducing material and manufacturing expenses.
  - **Maximizing Sparsity**: Reducing the number of active absorbers.

## Theoretical Flowchart

```mermaid
flowchart TD
    Start(["Start MOGA"]) --> InitPop["Initialize Population"]
    InitPop --> Eval["Evaluate Multiple Objectives <br/> (FRF, Cost, Sparsity)"]
    
    Eval --> ParetoRank["Perform Non-dominated Sorting <br/> Assign Pareto Ranks"]
    ParetoRank --> Density["Calculate Crowding Distance / <br/> Niching Factor"]
    
    Density --> GenLoop{"Max Generations <br/> Reached?"}
    
    GenLoop -- No --> Select["Selection based on <br/> Rank and Density"]
    Select --> Crossover["Crossover"]
    Crossover --> Mutate["Mutation"]
    
    Mutate --> EvalNew["Evaluate New Population"]
    EvalNew --> Combine["Combine Parents & Offspring"]
    Combine --> ParetoRank
    
    GenLoop -- Yes --> OutputFront
    
    OutputFront["Output Pareto Front Set"] --> End(["End MOGA"])
```
