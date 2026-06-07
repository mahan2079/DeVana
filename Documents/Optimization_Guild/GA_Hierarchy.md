# GA Hierarchy & Variants Documentation

## Overview
DeVana's Genetic Algorithm (GA) framework acts as the cornerstone of the optimization suite. It has been extensively layered with intelligent sub-systems, giving rise to multiple **Variants** that fundamentally alter the execution logic.

This document traces how different options (ML Bandit, PINN Acceleration, Smart Mutation, Advanced Seeding) alter the GA control flow.

---

## 1. Advanced Seeding Mechanisms
The way the initial population is generated sets the foundation for the search space exploration. The flag `seeding_method` dictates the path.

### Execution Trace:
- **Random Uniform:** Standard pseudo-random number generator within `parameter_bounds`.
- **Sobol (QMC):** Generates low-discrepancy sequences using `scipy.stats.qmc.Sobol`. Avoids clustering in multidimensional spaces.
- **Latin Hypercube (LHS):** Uses `qmc.LatinHypercube`. Stratifies parameter bounds evenly.
- **Memory Seeder:** Parses a JSON database of historically excellent solutions, injecting them via a `MemorySeeder` agent that applies small jitters and exploration fractions.
- **Neural Seeder (MLP):** Uses a pre-trained internal surrogate to predict promising candidate zones.
- **Best-of-Pool:** Samples a pool 5x the population size, evaluates all, and selects the initial population based on a diversity stride.

---

## 2. Dynamic Rate Controllers (ML Bandit vs. RL vs. Adaptive)
Standard GAs use fixed Crossover (`cxpb`) and Mutation (`mutpb`) probabilities. DeVana introduces dynamic controllers that adjust these rates *per generation*.

### Execution Trace:
When `run()` executes, the evolution loop checks which controller is active before applying operators.

*   **Heuristic (Adaptive):**
    *   *Trigger:* `adaptive_rates=True`
    *   *Logic:* Uses an exponential moving average of "Success Rate" (fraction of children better than parents) and "Gene Diversity" (normalized standard deviation of genes).
    *   *Action:* If diversity is low, it boosts mutation and reduces crossover. If success rate plummets, it increases mutation step-size (exploration).
*   **ML Bandit Controller (UCB):**
    *   *Trigger:* `use_ml_adaptive=True`
    *   *Logic:* Implements a Multi-Armed Bandit using the Upper Confidence Bound (UCB). Actions are discrete deltas applied to `cxpb`, `mutpb`, and population multipliers.
    *   *Reward:* Formulated as $\frac{\text{Fitness Improvement}}{\text{Generation Time}} - \lambda_{div} | \text{Diversity} - \text{Target} |$.
    *   *Action:* Alters the rates immediately. Population resizing happens via truncation (if shrinking) or Neural Seeding (if growing).
*   **RL Controller:**
    *   *Trigger:* `use_rl_controller=True`
    *   *Logic:* Implements Q-Learning with $\epsilon$-greedy exploration. The state is binary (0 = No Improvement, 1 = Improvement).
    *   *Action:* Similar action space to ML Bandit, but updates via the Bellman equation.

---

## 3. Physics-Informed Neural Network (PINN) Acceleration
FRF evaluations are computationally expensive. The PINN acts as a surrogate forward solver.

### Execution Trace:
*   *Trigger:* `use_pinn_solver=True`
*   *Logic:* When invalid offspring are generated, instead of passing them to `multiprocessing.Pool` mapping `frf()`, they are passed directly to `pinn_solver.predict()`.
*   *Online Learning:* If `pinn_online_learning=True`, there is a ~5% chance that a subset of individuals will bypass the PINN, hit the exact `frf()`, and the results will be back-propagated into the PINN via `train_step()` to prevent model drift.

---

## 4. Surrogate Screening (K-Nearest/MLP Filter)
Instead of replacing the FRF engine entirely, the surrogate model acts as a highly selective filter.

### Execution Trace:
*   *Trigger:* `use_surrogate=True`
*   *Logic:*
    1. The GA generates a pool of invalid offspring that is *larger* than necessary (`surrogate_pool_factor`, e.g., 2x).
    2. The Neural Surrogate (or KNN) predicts the fitness of the entire expanded pool.
    3. The pool is sorted by predicted fitness.
    4. **Selection:** The GA explicitly selects the top $N$ individuals (Exploitation) and a fraction of the most *novel* individuals based on distance to the training set (Exploration).
    5. Only this filtered subset is passed to the exact FRF engine, saving ~50% evaluation time while maintaining solution quality.

---

## 5. Smart Mutation (Gradient-Guided)
Instead of purely random Gaussian mutation, the GA leverages the Neural Surrogate to find the direction of steepest descent.

### Execution Trace:
*   *Trigger:* `use_smart_mutation=True`
*   *Logic:*
    - Replaces `tools.mutPolynomialBounded`.
    - Normalizes the individual's parameter vector.
    - Queries the Neural Surrogate for the gradient: `grad = neural_surrogate.get_fitness_gradient(x)`.
    - Mutates the individual by taking a step in the negative gradient direction (`-grad * eta`) combined with a micro-Gaussian noise vector to prevent trapping.