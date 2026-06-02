# Non-dominated Sorting Genetic Algorithm II (NSGA-II)

## Overview
The NSGA-II module (`NSGA2Worker.py`) implements one of the most popular and efficient multi-objective optimization algorithms. It uses the `DEAP` library to maintain a diverse set of solutions on the Pareto-optimal front, balancing vibration suppression, design sparsity, and cost.

## Objectives
1. **Objective 1 (Vibration)**: Minimize the difference between the FRF singular response and the target.
2. **Objective 2 (Sparsity)**: Minimize the number of active components and the total magnitude of parameters (L1 regularization).
3. **Objective 3 (Cost)**: Minimize the total cost calculated as a weighted sum of parameter values using `cost_coeffs`.

## Advanced Features
- **Elitist Preservation**: Ensures that the best individuals from the current generation are carried over to the next.
- **Fast Non-dominated Sorting**: Efficiently ranks the population into different fronts based on Pareto dominance.
- **Crowding Distance Assignment**: Maintains diversity along the Pareto front by favoring individuals in less dense regions.
- **Simulated Binary Crossover (SBX)**: A robust crossover operator for continuous search spaces.
- **Polynomial Mutation**: Effective mutation for real-valued parameters.

## Algorithm Flowchart

```mermaid
flowchart TD
    Start([Start NSGA-II]) --> InitPop[Initialize Population <br/> Attribute Generator]
    InitPop --> EvalInitial[Evaluate Objectives <br/> (f1: FRF, f2: Sparsity, f3: Cost)]
    
    EvalInitial --> GenLoop{Max Generations <br/> Reached?}
    
    GenLoop -- No --> Select[Tournament Selection <br/> based on Rank & Crowding]
    
    Select --> Mate[Simulated Binary Crossover]
    Mate --> Mutate[Polynomial Mutation]
    
    Mutate --> EvalOffspring[Evaluate Offspring Objectives]
    EvalOffspring --> Combine[Merge Parents N + Offspring N]
    
    Combine --> NDSort[Non-dominated Sorting <br/> Assign Fronts F1, F2, ...]
    NDSort --> Crowding[Calculate Crowding Distance <br/> for each front]
    
    Crowding --> Replace[Select Best N individuals <br/> for next generation]
    Replace --> GenLoop
    
    GenLoop -- Yes --> FinalPareto[Extract First Pareto Front F1]
    FinalPareto --> Save[Save Pareto Front to JSON]
    
    Save --> End([End NSGA-II])
```
