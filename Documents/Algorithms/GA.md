# Genetic Algorithm (GA)

## Overview
The Genetic Algorithm (`GAWorker.py`) optimizes DVA parameters by simulating natural selection. It evaluates fitness based on a comprehensive [Objective Function](ObjectiveFunction.md) that considers FRF analysis, sparsity, and cost-benefit ratios.

## Standard GA Workflow
The algorithm follows a biologically inspired cycle of selection, crossover, and mutation to evolve a population of candidate solutions toward the global optimum.

```mermaid
flowchart TD
    Init["Initialize Population (Random, Sobol, LHS, Neural)"] --> Eval["Evaluate Fitness (Objective Function)"]
    Eval --> Select["Selection (Tournament, Roulette)"]
    Select --> Cross["Crossover (Blend, Simulated Binary)"]
    Select --> Mut["Mutation (Gaussian, Polynomial)"]
    Cross --> Offspring["Create New Population"]
    Mut --> Offspring
    Offspring --> Replace["Replacement (Elitism)"]
    Replace --> Term{"Termination Criteria Met?"}
    
    Term -- No --> Eval
    Term -- Yes --> Output["Output Best Solutions"]
```

## Enhanced Adaptive Rates
While fixed rates (e.g., static $p_{mut}=0.2$) are easy to implement, they often struggle to balance exploration (searching new areas) and exploitation (refining good solutions) across different phases of the optimization. 

DeVana implements an **Adaptive Rate Controller** that dynamically adjusts mutation rate ($p_{mut}$), crossover rate ($p_{cx}$), and mutation step size ($\eta$) based on:
1. **Smoothed Success Rate ($\hat{s}$)**: The ratio of offspring outperforming their parents.
2. **Genetic Diversity ($\hat{D}$)**: The dispersion of genes within the current population.

### Adaptive Logic Flowchart
```mermaid
flowchart TD
    Start["Start Parameter Adjustment"] --> CheckS1{"Is success rate < 0.9 * target?"}
    
    CheckS1 -- Yes --> MutUp["Increase mutation rate and step size (Explore)"]
    CheckS1 -- No --> CheckS2{"Is success rate > 1.1 * target?"}
    
    CheckS2 -- Yes --> MutDown["Decrease mutation rate (Exploit)"]
    CheckS2 -- No --> CheckD1{"Is diversity << target?"}
    
    CheckD1 -- Yes --> DivLow["Increase mutation, decrease crossover (Prevent Stagnation)"]
    CheckD1 -- No --> CheckD2{"Is diversity >> target?"}
    
    CheckD2 -- Yes --> DivHigh["Increase crossover, decrease mutation (Focus Search)"]
    CheckD2 -- No --> End["Keep parameters unchanged"]
    
    MutUp --> End
    MutDown --> End
    DivLow --> End
    DivHigh --> End
```

## Integrated Controllers
- **ML Bandit Controller**: Formulates operator selection as a Multi-Armed Bandit problem, using Upper Confidence Bound (UCB) to choose the best genetic operators dynamically.
- **RL Controller**: Utilizes Reinforcement Learning to adjust hyperparameters based on the historical reward (fitness improvement) signal.
