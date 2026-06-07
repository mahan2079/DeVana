# Simulated Annealing (SA) Documentation

## Overview
Simulated Annealing (SA) is a probabilistic technique for approximating the global optimum of a given function. It is inspired by annealing in metallurgy, a technique involving heating and controlled cooling of a material to increase the size of its crystals and reduce their defects.

In DeVana, SA operates on a single candidate solution (rather than a population). It explores the parameter space by taking random steps; better solutions are always accepted, while worse solutions are accepted with a probability that decreases as the "temperature" cools down, allowing the algorithm to escape local minima early on.

## Class: `SAWorker` (inherits `QThread`)

### Purpose
Executes the Simulated Annealing optimization in a background thread. It provides integration with the FRF engine, PINN acceleration, and supports advanced ML/RL controllers to dynamically tune the cooling rate and step scale.

### Key Initialization Parameters
*   `sa_initial_temp`: The starting temperature ($T$). High values allow frequent acceptance of worse solutions.
*   `sa_cooling_rate`: The geometric cooling factor (e.g., 0.95), where $T_{new} = T \times \text{rate}$.
*   `sa_num_iterations`: Maximum iterations.
*   `step_scale`: The base scale for random perturbations relative to the parameter bounds.
*   **Controllers:** `use_ml_adaptive`, `use_rl_controller` (Adapts step scale, cooling rate, and temperature multiplier).
*   **Acceleration:** `use_pinn_solver`.

### Methods

#### 1. `evaluate_candidate(self, candidate)`
**Purpose:** Computes the scalar fitness of the candidate.
**Logic:**
- Same structure as other single-objective workers (FRF evaluation or PINN scalar prediction).
- Applies `alpha` sparsity penalty and `percentage_error_scale`.

#### 2. `run(self)`
**Purpose:** Main execution loop.
**Logic Flow:**
1.  **Initialization:**
    - Generates a random initial candidate vector.
    - Evaluates `current_fitness = evaluate_candidate(current_candidate)`.
    - Sets initial temperature $T$.
2.  **Iteration Loop:**
    - **Control Adjustment:** If active, uses ML Bandit or RL to adjust `base_step_scale`, `cooling_rate`, and a multiplier for $T$. The reward signal compares the acceptance rate to `ml_accept_target` (to maintain a healthy exploration/exploitation ratio).
    - **Perturbation:**
        - For each non-fixed parameter, generates a Gaussian perturbation: 
          $\text{perturbation} = \text{random.gauss}(0, \text{base\_scale}) \times (T / T_{initial})$.
        - Clamps the new candidate to parameter bounds.
    - **Evaluation:** Evaluates the new candidate to get `new_fitness`.
    - **Acceptance Criterion (Metropolis-Hastings):**
        - If `new_fitness < current_fitness`, accept immediately.
        - Else, calculate $P = \exp(-(\text{new\_fitness} - \text{current\_fitness}) / T)$.
        - If `random.random() < P`, accept.
        - Else, reject (keep current candidate).
    - **Cooling:** $T = T \times \text{cooling\_rate}$.
    - **Convergence Check:** Stops if `best_fitness <= sa_tol`.
3.  **Finalization:** Emits the best candidate found over all iterations and performs a final exact evaluation.

---

## Architectural Flowchart

```mermaid
flowchart TD
    Start["Start SAWorker"] --> Init["Init Random Candidate & Temperature T"]
    Init --> EvalInit["Evaluate Initial Fitness"]
    EvalInit --> IterLoop{"For Iteration in 1..Max"}
    IterLoop -- Yes --> Adapt["Adapt Step Scale, Cooling, T via ML/RL"]
    Adapt --> Perturb["Generate Neighbor via T-Scaled Gaussian Perturbation"]
    Perturb --> EvalNew["Evaluate Neighbor Fitness"]
    EvalNew --> CheckBetter{"New < Current?"}
    CheckBetter -- Yes --> Accept["Accept Neighbor"]
    CheckBetter -- No --> Metropolis["Calc P = exp("-Delta/T")"]
    Metropolis --> RandomCheck{"Rand < P?"}
    RandomCheck -- Yes --> Accept
    RandomCheck -- No --> Reject["Keep Current"]
    Accept --> UpdateBest["Update Best Solution"]
    Reject --> Cool["T = T * cooling_rate"]
    UpdateBest --> Cool
    Cool --> IterLoop
    IterLoop -- No --> Finish["Emit Results"]
```

### Flowchart Pseudo-code
```text
function run_sa():
    current_x = generate_random()
    current_f = evaluate(current_x)
    best_x = current_x
    T = initial_temp
    
    for iteration in 1..max_iterations:
        step, cooling, T = update_controllers(step, cooling, T)
        
        # Perturb
        new_x = current_x + gaussian_noise(scale=step * (T / initial_temp))
        new_x = clamp(new_x, bounds)
        
        new_f = evaluate(new_x)
        delta = new_f - current_f
        
        if delta < 0 or random() < exp(-delta / T):
            current_x = new_x
            current_f = new_f
            
            if current_f < best_f:
                best_x = current_x
                best_f = current_f
                
        T = T * cooling
        
    return best_x
```
