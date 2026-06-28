import random
import cma
from devana.optimize.base import Solver

class CMAESSolver(Solver):
    """
    Covariance Matrix Adaptation Evolution Strategy (CMA-ES) Solver.
    
    Ported from CMAESWorker.py, removing PyQt5 dependencies.
    """
    def __init__(self, config, evaluate_fn=None, callback=None):
        super().__init__(config, evaluate_fn, callback)
        
        # CMA-ES specific configuration
        self.initial_sigma = config.get('initial_sigma', 0.5)
        
    def solve(self):
        """Execute the CMA-ES optimization."""
        # Extract parameter names and bounds
        lower_bounds = [b[0] for b in self.parameter_bounds]
        upper_bounds = [b[1] for b in self.parameter_bounds]
        
        # Build initial candidate x0
        x0 = []
        for j in range(self.num_parameters):
            if j in self.fixed_parameters:
                x0.append(self.fixed_parameters[j])
            else:
                x0.append(random.uniform(lower_bounds[j], upper_bounds[j]))
        
        # Options for cma
        options = {
            'bounds': [lower_bounds, upper_bounds],
            'maxiter': self.num_generations,
            'verb_disp': 0,
            'tolx': self.tolerance
        }
        
        # Objective wrapper to handle fixed parameters and callback logic
        def objective(x):
            # Enforce fixed parameters
            for idx, val in self.fixed_parameters.items():
                x[idx] = val
            return self.evaluate(x.tolist())
            
        es = cma.CMAEvolutionStrategy(x0, self.initial_sigma, options)
        
        best_fitness = float('inf')
        best_ind = None
        metrics = {
            'best_fitness_history': []
        }
        
        it = 0
        while not es.stop():
            if self.stop_requested:
                break
                
            it += 1
            solutions = es.ask()
            fitnesses = [objective(x) for x in solutions]
            es.tell(solutions, fitnesses)
            
            # Find current best
            cur_min_fit = min(fitnesses)
            if cur_min_fit < best_fitness:
                best_fitness = cur_min_fit
                best_ind = solutions[fitnesses.index(cur_min_fit)].tolist()
                
            metrics['best_fitness_history'].append(best_fitness)
            self._report_progress(it, best_fitness, best_ind, metrics)
            
            if best_fitness <= self.tolerance:
                break
                
            if it >= self.num_generations:
                break
        
        res = es.result
        return {
            'best_individual': res.xbest.tolist() if res.xbest is not None else best_ind,
            'best_fitness': res.fbest if res.fbest is not None else best_fitness,
            'metrics': metrics,
            'parameter_names': self.parameter_names
        }
