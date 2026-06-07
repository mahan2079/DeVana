# Simulated Annealing (SA)

## Overview
The Simulated Annealing module (`SAWorker.py`) is a probabilistic technique for approximating the global optimum of a given function. It mimics the process of annealing in metallurgy, where a material is heated and then cooled slowly to reach a low-energy state.

## Advanced Features
- **Adaptive Cooling**: Dynamically adjusts the cooling rate to focus search effort.
- **Dynamic Step Scaling**: Adjusts the scale of perturbations based on the current temperature and acceptance rate.
- **ML Bandit & RL Controllers**: Similar to other workers, adapts temperature, cooling rate, and step scale to optimize performance.
- **Probabilistic Acceptance**: Accepts worse solutions based on the Boltzmann distribution $P = \exp(-\Delta f / T)$, allowing the algorithm to escape local optima.
- **Boundary Constraint Management**: Ensures candidate solutions remain within the specified physical bounds of the DVA parameters.

## Algorithm Flowchart

```mermaid
flowchart TD
    Start("[Start SA]") --> InitState["Initialize Current Candidate <br/> (Random or Seeded)"]
    InitState --> EvalInitial["Evaluate Initial Fitness via FRF"]
    EvalInitial --> InitParams["Set Initial Temperature T, <br/> Cooling Rate, and Step Scale"]
    
    InitParams --> IterLoop{"Max Iterations <br/> Reached?"}
    
    IterLoop -- No --> CheckTermination["Check Termination Flags"]
    CheckTermination --> Adapt["Adapt T, Cooling, Step <br/> (ML or RL Controllers)"]
    
    Adapt --> Perturb["Generate New Candidate <br/> via Gaussian Perturbation"]
    Perturb --> Bounds["Apply Parameter Bounds"]
    Bounds --> EvalNew["Evaluate New Fitness via FRF"]
    
    EvalNew --> CalcDelta["Calculate Delta f"]
    
    CalcDelta --> Improvement{"Is Delta f < 0?"}
    Improvement -- Yes --> Accept["Accept New Candidate"]
    Improvement -- No --> ProbCheck{"Accept with <br/> Boltzmann Probability?"}
    
    ProbCheck -- Yes --> Accept
    ProbCheck -- No --> Reject["Keep Current Candidate"]
    
    Accept --> UpdateBest["Update Best Candidate Found"]
    Reject --> UpdateBest
    
    UpdateBest --> Cool["Update Temperature: <br/> T = T * CoolingRate"]
    Cool --> Metrics["Calculate Progress & <br/> Update Metrics"]
    
    Metrics --> Tolerance{"Best Fitness <br/> < Tolerance?"}
    Tolerance -- No --> IterLoop
    Tolerance -- Yes --> OutputBest
    
    IterLoop -- Yes --> OutputBest
    
    OutputBest["Output Best Candidate & Metrics"] --> End("[End SA]")
```

#### Pseudo-code
```text
BEGIN
  EXECUTE [Start SA]
  EXECUTE Initialize Current Candidate   (Random or Seeded)
  EXECUTE Evaluate Initial Fitness via FRF
  EXECUTE Set Initial Temperature T,   Cooling Rate, and Step Scale
  EXECUTE Max Iterations   Reached?
  EXECUTE Check Termination Flags
  EXECUTE Adapt T, Cooling, Step   (ML or RL Controllers)
  EXECUTE Generate New Candidate   via Gaussian Perturbation
  EXECUTE Apply Parameter Bounds
  EXECUTE Evaluate New Fitness via FRF
  EXECUTE Calculate Delta f
  EXECUTE Is Delta f < 0?
  EXECUTE Accept New Candidate
  EXECUTE Accept with   Boltzmann Probability?
  EXECUTE Keep Current Candidate
  EXECUTE Update Best Candidate Found
  EXECUTE Update Temperature:   T = T * CoolingRate
  EXECUTE Calculate Progress &   Update Metrics
  EXECUTE Best Fitness   < Tolerance?
  EXECUTE Output Best Candidate & Metrics
  EXECUTE [End SA]
END
```
