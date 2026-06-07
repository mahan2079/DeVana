import sys
import os
import numpy as np

# Add the current directory to path to ensure devana is found
sys.path.append(os.getcwd())

import devana
from devana import BeamModel, GASolver, PSOSolver, MemorySeeder

def test_physics_layer():
    print("\n--- Testing Physics Layer ---")
    try:
        # Initialize a continuous beam model with correct arguments
        # length, width, thickness, youngs_modulus, density
        beam = BeamModel(
            length=1.0, 
            width=0.05, 
            thickness=0.01, 
            youngs_modulus=210e9, 
            density=7850
        )
        # Check property calculation
        print(f"Beam EI: {beam.EI:.2f}")
        print(f"Beam mass/line: {beam.m_line:.2f}")
        print("Physics Layer: SUCCESS")
    except Exception as e:
        print(f"Physics Layer: FAILED - {e}")

def test_optimization_layer():
    print("\n--- Testing Optimization Layer ---")
    try:
        # 1. Define a dummy objective function (sphere function)
        def sphere_objective(individual):
            return sum(x**2 for x in individual)

        # 2. Setup a basic configuration for solvers
        # Solvers expect config to be a dict
        config = {
            'pop_size': 20,
            'num_generations': 5,
            'parameter_data': [
                ('param1', 0.0, 1.0, False),
                ('param2', 0.0, 1.0, False)
            ],
            'cxpb': 0.5,
            'mutpb': 0.2,
            'random_seed': 42
        }
        
        # 3. Test GA Solver
        print("Running GA Solver...")
        # Note: Solver __init__ sets evaluate_fn and callback
        def ga_callback(gen, best_fit, best_ind, metrics):
            if gen % 2 == 0:
                print(f"  GA Gen {gen}: Best Fit = {best_fit:.4f}")

        ga_solver = GASolver(config, evaluate_fn=sphere_objective, callback=ga_callback)
        ga_results = ga_solver.solve()
        print(f"GA Solver Result: {ga_results['best_fitness']:.4f}")

        # 4. Test PSO Solver
        print("Running PSO Solver...")
        pso_solver = PSOSolver(config, evaluate_fn=sphere_objective)
        pso_results = pso_solver.solve()
        print(f"PSO Solver Result: {pso_results['best_fitness']:.4f}")
        
        print("Optimization Layer: SUCCESS")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Optimization Layer: FAILED - {e}")

def test_ml_layer():
    print("\n--- Testing ML Layer ---")
    try:
        # MemorySeeder requires lows, highs, fixed_mask, fixed_values
        lows = np.array([0.0, 0.0])
        highs = np.array([1.0, 1.0])
        fixed_mask = np.array([False, False])
        fixed_values = np.array([0.0, 0.0])
        
        seeder = MemorySeeder(lows, highs, fixed_mask, fixed_values)
        # Check the size property
        print(f"MemorySeeder initialized. Database size: {seeder.size}")
        print("ML Layer: SUCCESS")
    except Exception as e:
        print(f"ML Layer: FAILED - {e}")

if __name__ == "__main__":
    print(f"Testing DeVana Library v{devana.__version__}")
    test_physics_layer()
    test_ml_layer()
    test_optimization_layer()
    print("\n--- Library Verification Complete ---")
