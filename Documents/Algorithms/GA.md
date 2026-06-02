# Genetic Algorithm (GA)

## Overview
The Genetic Algorithm (`GAWorker.py`) optimizes DVA parameters by simulating natural selection. It evaluates fitness based on a comprehensive [Objective Function](ObjectiveFunction.md) that considers FRF analysis, sparsity, and cost-benefit ratios.

## GA Workflow with Advanced Controllers
The core evolutionary loop is augmented by multiple intelligent controllers that dynamically tune hyperparameters to ensure robust convergence and prevent stagnation.

```mermaid
flowchart TD
    Start([Start GA Optimization]) --> Init["Initialize Population <br/> (Random, Sobol, LHS, Memory, or Neural)"]
    Init --> Eval["Evaluate Initial Fitness <br/> (FRF Analysis)"]
    
    Eval --> GenLoop{Max Generations <br/> Reached?}
    
    GenLoop -- No --> SelectController["Select Parameter Controller <br/> (Fixed, Adaptive, ML Bandit, or RL)"]
    
    SelectController --> AdaptLogic["Update Rates (cxpb, mutpb) <br/> & Population Size"]
    
    AdaptLogic --> Selection["Selection <br/> (Tournament Selection)"]
    Selection --> Crossover["Crossover <br/> (Blend or SBX)"]
    Crossover --> Mutation["Mutation <br/> (Gaussian or Polynomial)"]
    
    Mutation --> Resize{"Population <br/> Resize Needed?"}
    Resize -- Yes --> ResizeLogic["Adjust Population <br/> (Neural Seeding for growth)"]
    Resize -- No --> EvalOffspring["Evaluate Offspring Fitness"]
    ResizeLogic --> EvalOffspring
    
    EvalOffspring --> Replace["Form New Population <br/> (Elitism & Replacement)"]
    Replace --> Metrics["Track Resource Metrics <br/> (CPU, RAM, Diversity)"]
    
    Metrics --> GenLoop
    
    GenLoop -- Yes --> Output["Output Best Solutions & <br/> Statistical Reports"]
    Output --> End([End GA])
```

## Intelligent Parameter Controllers

### 1. ML Bandit Controller (MAB)
If enabled, DeVana uses a Multi-Armed Bandit strategy to select the optimal combination of crossover probability, mutation probability, and population size.
- **Algorithm:** Upper Confidence Bound (UCB).
- **Action Space:** A Cartesian product of deltas for rates ($\pm 15\%, \pm 30\%$) and population multipliers ($0.75x, 1.25x$).
- **Reward Function:** Blends fitness improvement, computational efficiency (gen time), and genetic diversity.

```mermaid
flowchart TD
    StartMAB[Start Generation] --> CalcReward[Calculate Reward from Previous Action <br/> (Improvement + Speed - Diversity Penalty)]
    CalcReward --> UpdateUCB[Update Action Counts & <br/> Cumulative Rewards]
    UpdateUCB --> SelectBest[Select Action with Highest UCB Score: <br/> Score = AvgReward + C * sqrt(log(t)/count)]
    SelectBest --> ApplyAction[Apply Deltas to cxpb, mutpb, pop_size]
    ApplyAction --> EndMAB[Next Generation]
```

### 2. RL Controller (Q-Learning)
An alternative adaptive method that uses a reinforcement learning agent to learn the best parameter adjustments.
- **Algorithm:** Discrete Q-Learning.
- **State Space:** Binary state (extended for stagnation/diversity).
- **Policy:** $\epsilon$-greedy (Exploration vs. Exploitation).

### 3. Adaptive Rate Controller
A heuristic-based controller that monitors:
- **Success Rate:** Ratio of offspring outperforming parents.
- **Genetic Diversity:** Standard deviation of parameters.
It increases mutation to explore when diversity is low and decreases it to exploit when success is high.

## Advanced Seeding & Population Management
DeVana supports a high-tier hierarchy of initialization methods:
- **Sobol & LHS:** Quasi-random sequences for optimal search space coverage.
- **Memory Seeding:** Reuses high-quality solutions from previous optimization sessions.
- **Neural Seeding:** Uses an online surrogate model (MLP ensemble) to generate promising candidates when the population grows.
