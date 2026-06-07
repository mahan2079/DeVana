import random
import numpy as np
from devana.optimize.base import Solver

class DESolver(Solver):
    """
    Differential Evolution (DE) Solver.
    
    Ported from DEWorker.py, removing PyQt5 dependencies.
    """
    def __init__(self, config, evaluate_fn=None, callback=None):
        super().__init__(config, evaluate_fn, callback)
        
        # DE specific configuration
        self.F = config.get('F', 0.5)      # Mutation factor
        self.CR = config.get('CR', 0.7)     # Crossover probability
        self.strategy = config.get('strategy', 'rand/1/bin')

    def solve(self):
        """Execute the DE optimization."""
        num_pop = self.pop_size
        num_params = self.num_parameters
        
        # Initialize population
        population = []
        for i in range(num_pop):
            ind = []
            for j in range(num_params):
                low, high = self.parameter_bounds[j]
                if j in self.fixed_parameters:
                    ind.append(self.fixed_parameters[j])
                else:
                    ind.append(random.uniform(low, high))
            population.append(np.array(ind))
            
        # Evaluate initial population
        fitnesses = [self.evaluate(ind.tolist()) for ind in population]
        
        # Global best
        best_idx = np.argmin(fitnesses)
        best_fitness = fitnesses[best_idx]
        best_ind = population[best_idx].copy()
        
        metrics = {
            'best_fitness_history': []
        }
        
        for gen in range(1, self.num_generations + 1):
            if self.stop_requested:
                break
                
            for i in range(num_pop):
                # Mutation
                idxs = [idx for idx in range(num_pop) if idx != i]
                r1, r2, r3 = random.sample(idxs, 3)
                
                # donor = x_r1 + F * (x_r2 - x_r3)
                donor = population[r1] + self.F * (population[r2] - population[r3])
                
                # Crossover
                trial = np.copy(population[i])
                j_rand = random.randrange(num_params)
                for j in range(num_params):
                    if j in self.fixed_parameters:
                        trial[j] = self.fixed_parameters[j]
                    elif random.random() <= self.CR or j == j_rand:
                        low, high = self.parameter_bounds[j]
                        trial[j] = max(low, min(high, donor[j]))
                
                # Selection
                trial_fitness = self.evaluate(trial.tolist())
                if trial_fitness < fitnesses[i]:
                    population[i] = trial
                    fitnesses[i] = trial_fitness
                    
                    if trial_fitness < best_fitness:
                        best_fitness = trial_fitness
                        best_ind = trial.copy()
            
            metrics['best_fitness_history'].append(best_fitness)
            self._report_progress(gen, best_fitness, best_ind.tolist(), metrics)
            
            if best_fitness <= self.tolerance:
                break
                
        return {
            'best_individual': best_ind.tolist(),
            'best_fitness': best_fitness,
            'metrics': metrics,
            'parameter_names': self.parameter_names
        }
