import random
import math
from devana.optimize.base import Solver

class SASolver(Solver):
    """
    Simulated Annealing (SA) Solver.
    
    Ported from SAWorker.py, removing PyQt5 dependencies.
    """
    def __init__(self, config, evaluate_fn=None, callback=None):
        super().__init__(config, evaluate_fn, callback)
        
        # SA specific configuration
        self.initial_temp = config.get('initial_temp', 100.0)
        self.cooling_rate = config.get('cooling_rate', 0.95)
        self.step_scale = config.get('step_scale', 0.1)

    def solve(self):
        """Execute the SA optimization."""
        num_params = self.num_parameters
        
        # Initialize current candidate
        current_candidate = []
        for j in range(num_params):
            low, high = self.parameter_bounds[j]
            if j in self.fixed_parameters:
                current_candidate.append(self.fixed_parameters[j])
            else:
                current_candidate.append(random.uniform(low, high))
        
        current_fitness = self.evaluate(current_candidate)
        best_candidate = current_candidate[:]
        best_fitness = current_fitness
        
        T = self.initial_temp
        
        metrics = {
            'best_fitness_history': [],
            'temperature_history': []
        }
        
        for iteration in range(1, self.num_generations + 1):
            if self.stop_requested:
                break
                
            # Generate new candidate by perturbation
            new_candidate = []
            for j in range(num_params):
                if j in self.fixed_parameters:
                    new_candidate.append(self.fixed_parameters[j])
                else:
                    low, high = self.parameter_bounds[j]
                    # Scale perturbation by temperature
                    scale = (high - low) * self.step_scale * (T / self.initial_temp)
                    perturbation = random.gauss(0, scale)
                    new_val = current_candidate[j] + perturbation
                    new_val = max(low, min(new_val, high))
                    new_candidate.append(new_val)
            
            new_fitness = self.evaluate(new_candidate)
            delta_fitness = new_fitness - current_fitness
            
            # Acceptance criteria
            if delta_fitness < 0:
                current_candidate = new_candidate
                current_fitness = new_fitness
                
                # Update best
                if current_fitness < best_fitness:
                    best_fitness = current_fitness
                    best_candidate = current_candidate[:]
            else:
                # Accept worse solution with probability
                acceptance_prob = math.exp(-delta_fitness / max(1e-12, T))
                if random.random() < acceptance_prob:
                    current_candidate = new_candidate
                    current_fitness = new_fitness
            
            # Cooling
            T *= self.cooling_rate
            
            metrics['best_fitness_history'].append(best_fitness)
            metrics['temperature_history'].append(T)
            self._report_progress(iteration, best_fitness, best_candidate, metrics)
            
            if best_fitness <= self.tolerance:
                break
                
        return {
            'best_individual': best_candidate,
            'best_fitness': best_fitness,
            'metrics': metrics,
            'parameter_names': self.parameter_names
        }
