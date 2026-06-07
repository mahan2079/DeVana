# Differential Evolution (DE)

## Overview
The Differential Evolution module (`DEWorker.py`) optimizes DVA parameters by maintaining a population of candidate solutions and creating trial vectors through mutation and crossover operations. DE is particularly well-suited for multidimensional real-valued continuous optimization spaces.

## Advanced Features
- **Multiple Strategies**: Supports `rand/1`, `rand/2`, `best/1`, `best/2`, `current-to-best/1`, and `current-to-rand/1` mutation strategies.
- **Adaptive Parameter Control**:
  - *Jitter/Dither*: Random perturbations to the mutation factor (F).
  - *JADE*: Adaptive DE with an optional archive to guide parameter adjustments.
  - *SaDE / Success-History*: Adjusts crossover rate (CR) and F based on historical success.
- **Diversity Preservation**: Reinitializes parts of the population if diversity falls below a threshold.
- **Constraint Handling**: Strategies including "penalty", "reflection", and "projection".
- **ML/RL Controllers**: Similar to GA/PSO, adjusts F, CR, and population size dynamically based on a multi-armed bandit or reinforcement learning agent.
- **Multi-Run Statistics**: Capable of executing multiple independent runs and computing statistical distributions.

## Algorithm Flowchart

```mermaid
flowchart TD
    Start("[Start DE]") --> InitPop["Initialize Population <br/> (Latin Hypercube or Random)"]
    InitPop --> EvalInitial["Evaluate Initial Population"]
    EvalInitial --> GenLoop{"Max Generations <br/> Reached?"}
    
    GenLoop -- No --> AdaptParams["Adapt Control Parameters <br/> F and CR"]
    AdaptParams --> MLRL["Update ML/RL Controllers if enabled"]
    MLRL --> PopLoop["For each target vector x_i"]
    
    PopLoop --> Mutation["Apply DE Strategy <br/> Create Donor Vector v"]
    Mutation --> Crossover["Apply Binomial Crossover <br/> Create Trial Vector u"]
    Crossover --> Constraints["Apply Constraint Handling <br/> (Reflection, Projection)"]
    Constraints --> EvalTrial["Evaluate Trial Vector u"]
    
    EvalTrial --> Selection{"Is f("\\"u\\"") < f("\\"x_i\\"")?"}
    Selection -- Yes --> Replace["Replace x_i with u"]
    Selection -- No --> Keep["Keep x_i"]
    
    Replace --> CheckNext{"More targets?"}
    Keep --> CheckNext
    
    CheckNext -- Yes --> PopLoop
    CheckNext -- No --> Diversity["Calculate Diversity & <br/> Apply Preservation"]
    
    Diversity --> Metrics["Calculate Generation Metrics & Update Best"]
    Metrics --> EarlyStop{"Convergence Met?"}
    
    EarlyStop -- No --> GenLoop
    EarlyStop -- Yes --> OutputBest
    GenLoop -- Yes --> OutputBest
    
    OutputBest["Output Best Solution & Multi-run Stats"] --> End("[End DE]")
```

#### Pseudo-code
```text
BEGIN
  EXECUTE [Start DE]
  EXECUTE Initialize Population   (Latin Hypercube or Random)
  EXECUTE Evaluate Initial Population
  EXECUTE Max Generations   Reached?
  EXECUTE Adapt Control Parameters   F and CR
  EXECUTE Update ML/RL Controllers if enabled
  EXECUTE For each target vector x_i
  EXECUTE Apply DE Strategy   Create Donor Vector v
  EXECUTE Apply Binomial Crossover   Create Trial Vector u
  EXECUTE Apply Constraint Handling   (Reflection, Projection)
  EXECUTE Evaluate Trial Vector u
  EXECUTE Is f(\
  EXECUTE ) < f(\
  EXECUTE )?
  EXECUTE Replace x_i with u
  EXECUTE Keep x_i
  EXECUTE More targets?
  EXECUTE Calculate Diversity &   Apply Preservation
  EXECUTE Calculate Generation Metrics & Update Best
  EXECUTE Convergence Met?
  EXECUTE Output Best Solution & Multi-run Stats
  EXECUTE [End DE]
END
```
