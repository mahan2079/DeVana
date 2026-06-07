# Optimization Domain Master Index

This document serves as the ground truth index for the multi-faceted optimization capabilities within DeVana. It maps out all available algorithms, their configurations, and specific architectural implementations.

## Genetic Algorithm Family
*   **[Genetic Algorithm (GA)](GA.md):** The core optimizer featuring ML/RL adaptive controllers, PINN acceleration, multiple seeding strategies (QMC, LHS, Neural, Memory), and an enhanced cost-benefit analysis fitness function.
*   **[Non-dominated Sorting Genetic Algorithm II (NSGA-II)](NSGA2.md):** Multi-objective algorithm tracking Performance, Sparsity, and Cost using Pareto fronts and Hypervolume metrics.
*   **[Multi-Objective Genetic Algorithm (MOGA)](MOGA.md):** A streamlined multi-objective worker focused on efficient execution across multiple independent runs.
*   **[Adaptive Value-Encoded Algorithm (AdaVEA)](AdaVEA.md):** Specialized multi-objective algorithm with heuristic initialization (biasing towards lower bounds for sparsity) and time-decaying crossover/mutation rates.

## Swarm & Evolutionary Strategies
*   **[Particle Swarm Optimization (PSO)](PSO.md):** Simulates flocking behavior. Features adaptive inertia/acceleration coefficients, topological neighborhoods (Global, Ring, Von Neumann, Random), and multiple boundary handling strategies.
*   **[Covariance Matrix Adaptation Evolution Strategy (CMA-ES)](CMAES.md):** Advanced optimizer for non-linear problems. Updates a multivariate normal distribution based on success. Includes ML/RL sigma step-size controllers.
*   **[Differential Evolution (DE)](DE.md):** Population-based optimizer featuring strategies like `rand/1`, `best/1`, and `current-to-best/1`, with adaptive F and CR mechanisms (JADE, SaDE) and built-in Sobol sensitivity analysis capabilities.

## Probabilistic Search
*   **[Simulated Annealing (SA)](SA.md):** Single-candidate explorer utilizing a cooling schedule and Metropolis-Hastings acceptance criteria. Incorporates ML/RL controllers to dynamically adapt step-size, cooling rate, and temperature.

## Machine Learning & Reinforcement Learning
*   **[Reinforcement Learning (RL)](RL.md):** Continuous optimization using a Deep Deterministic Policy Gradient (DDPG) approach for direct policy generation. Supports Sobol parameter ranking prior to training and experience replay buffer management.

## Optimization Workers (Core Logic)
These files contain the exhaustive documentation for the backend worker threads that execute the algorithms.

### Evolutionary & Swarm Workers
- [GAWorker](Workers/GAWorker.md) - Genetic Algorithm backend.
- [PSOWorker](Workers/PSOWorker.md) - Particle Swarm Optimization backend.
- [DEWorker](Workers/DEWorker.md) - Differential Evolution backend.
- [SAWorker](Workers/SAWorker.md) - Simulated Annealing backend.
- [CMAESWorker](Workers/CMAESWorker.md) - CMA-ES backend.
- [NSGA2Worker](Workers/NSGA2Worker.md) - NSGA-II backend.
- [MOGAWorker](Workers/MOGAWorker.md) - Multi-Objective GA backend.
- [AdaVEAWorker](Workers/AdaVEAWorker.md) - Adaptive Vibration Evolutionary Algorithm backend.

### ML & Analysis Workers
- [AIWorker](Workers/AIWorker.md) - LLM and Engineering Assistant backend.
- [FRFWorker](Workers/FRFWorker.md) - Frequency Response Function backend.
- [SobolWorker](Workers/SobolWorker.md) - Sobol sensitivity analysis backend.
- [PINNWorker](Workers/PINNWorker.md) - PINN training and inference worker.
- [PINNSolver](Workers/PINNSolver.md) - High-speed physics-informed solver.
- [NeuralSeeder](Workers/NeuralSeeder.md) - Intelligent population seeding.
- [NeuralSurrogate](Workers/NeuralSurrogate.md) - Surrogate model acceleration.
- [MemorySeeder](Workers/MemorySeeder.md) - Historical result based seeding.

## Advanced Domain Concepts
*   **[GA Hierarchy & Variants](GA_Hierarchy.md):** Deep-dive trace of how various sub-systems (ML Bandit, RL, PINN, Surrogate Screening, Smart Mutation, Advanced Seeding) alter the execution logic and flow of the evolutionary algorithms.

## Common Architecture Mechanisms
All optimizers in this domain share the following architectural traits:
1.  **Multi-Threading:** Execution is wrapped in PyQt `QThread` or `QObject` to preserve GUI responsiveness.
2.  **Physics Acceleration:** Seamless routing to either the exact `FRF.py` engine or a `PINNSolver` for near-instant scalar prediction (with online learning hooks).
3.  **Sparsity Penalization:** Fitness functions are inherently regularized using $L_0$/$L_1$ norms (`sparsity_tau`, `sparsity_alpha`) to drive solutions toward manufacturability.
4.  **Signal Emission:** Comprehensive state logging via Qt signals (`progress`, `generation_metrics`, `finished`) for real-time dashboard plotting.
