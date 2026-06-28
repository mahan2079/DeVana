# DeVana Python Library: User Guide

The `devana_lib` library is a standalone, headless framework for vibration analysis and metaheuristic optimization. It decouples the core engineering logic of DeVana from its PyQt5 GUI, allowing for automation, scripting, and integration into other Python applications.

## 📦 Installation & Setup

Ensure you are in the project root and have the dependencies installed:

```bash
# Using the project's virtual environment
source venv/bin/activate
pip install -r requirements.txt
```

To use the library in your scripts, simply import the `devana_lib` package:

```python
import devana_lib as devana
```

---

## 🏗️ 1. Physics & Modeling
Use the `physics` and `systems` modules to define your vibrational models.

### Discrete DVA System Modeling (Recommended)
The `DVASystem` class provides a high-level interface for configuring the main structure and its absorbers.

```python
from devana_lib import DVASystem

# 1. Initialize system with Characteristic Frequency and Damping
system = DVASystem(omega_dc=5000.0, zeta_dc=0.01)

# 2. Configure Main System
system.set_primary_mass_ratio(1.0)          # M2 / M1
system.set_stiffness_ratios([1.0, 1.0, 1.0]) # Lambda ratios
system.set_damping_ratios([0.05, 0.05])      # Nu ratios

# 3. Add Dynamic Vibration Absorbers (DVAs)
# Set Mass ratios (mu), Stiffness (lambda), and Damping (nu)
system.set_dva_parameters(
    mass_ratios=[0.1, 0.1, 0.1], 
    stiffness=[1.1, 1.2, 1.3],
    damping=[0.02, 0.02, 0.02]
)

# 4. Calculate Frequency Response
results = system.calculate_response(
    omega_start=0.1, 
    omega_end=2.0, 
    points=1000,
    target_masses={1: {"peak_value": 0.001}} # Monitor Mass 1
)

print(f"Singular Response (Fitness): {results['singular_response']}")
```

### Continuous Beam Modeling
```python
from devana_lib import BeamModel

# Define a beam: Length, Width, Thickness, Young's Modulus, Density
beam = BeamModel(
    length=1.0, 
    width=0.05, 
    thickness=0.01, 
    youngs_modulus=210e9, 
    density=7850
)

print(f"Bending Stiffness (EI): {beam.EI}")
print(f"Mass per unit length: {beam.m_line}")
```

---

## 🚀 2. Optimization Solvers
All solvers follow a common interface: `Solver(config, evaluate_fn, callback)`.

### Configuration Dictionary
Solvers expect a configuration dictionary with the following keys:
- `pop_size`: Number of individuals in the population.
- `num_generations`: Number of iterations.
- `parameter_data`: A list of tuples `(name, low, high, is_fixed)`.
- `random_seed`: (Optional) For reproducibility.

### Running a Genetic Algorithm (GA)
```python
from devana_lib import GASolver

# 1. Setup Configuration
config = {
    'pop_size': 50,
    'num_generations': 100,
    'parameter_data': [
        ('stiffness_1', 1e3, 1e6, False),
        ('mass_1', 0.1, 5.0, False),
    ],
    'cxpb': 0.8,      # Crossover probability
    'mutpb': 0.2,     # Mutation probability
    'tournsize': 3,   # Selection tournament size
    'indpb': 0.1      # Attribute-level mutation probability
}

# 2. Define an Objective Function
def my_objective(individual):
    return sum(individual)

# 3. Initialize and Solve
solver = GASolver(config, evaluate_fn=my_objective)
results = solver.solve()
```

> [!NOTE]
> The library's `GASolver` is a streamlined implementation. For a detailed reference on the advanced seeding, adaptive controllers, and machine learning accelerators available in the GUI/Worker version, see **[GA Deep Specification](../Documents/Algorithms/GA_Deep_Spec.md)**.

print(f"Optimization Finished!")
print(f"Best Solution: {results['best_individual']}")
```

### Available Solvers
- `GASolver`: Genetic Algorithm
- `PSOSolver`: Particle Swarm Optimization
- `CMAESSolver`: CMA-ES
- `NSGA2Solver`: Multi-objective NSGA-II
- `DESolver`: Differential Evolution
- `SASolver`: Simulated Annealing
- `AdaVEASolver`: Adaptive Variable Elastic Algorithm
- `MOGASolver`: Multi-Objective GA
- `RLSolver`: Reinforcement Learning (DDPG-based)

---

## ⚡ 3. Advanced AI & Physics
The library includes cutting-edge AI tools for system identification and acceleration.

### Physics-Informed Neural Networks (PINN)
Train a neural network to learn the "mechanical intuition" of your system.

```python
from devana_lib import PINNSolver

# Initialize solver for a system with 48 parameters
pinn = PINNSolver(param_dim=48)

# Train with data (params, frequency, target_amplitude)
loss = pinn.train_step(my_params, my_omega, my_target)

# Predict 1000x faster than traditional FRF
predicted_frf = pinn.predict(my_params, omega_range)
```

## 🧠 4. Machine Learning & Seeding
Accelerate optimization using intelligent seeding.

```python
import numpy as np
from devana_lib import MemorySeeder

lows = np.array([0.0, 0.0])
highs = np.array([1.0, 1.0])
fixed_mask = np.array([False, False])
fixed_values = np.array([0.0, 0.0])

# Initialize Memory Seeder
seeder = MemorySeeder(lows, highs, fixed_mask, fixed_values)

# Propose new seeds based on "remembered" good solutions
new_seeds = seeder.propose(n=10)
```

---

## 🔍 4. Sensitivity Analysis
Analyze which parameters impact your results the most.

```python
from devana_lib import perform_sobol_analysis

# Define parameter bounds
bounds = {
    'k1': [1000, 5000],
    'm1': [0.1, 2.0]
}

# Run Sobol Analysis
# Requires an evaluation function that takes a dict of parameters
results = perform_sobol_analysis(my_eval_function, bounds, calc_second_order=True)
print(results['S1']) # First-order indices
```
