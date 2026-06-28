import random
import numpy as np
import time
import psutil
import os
from deap import base, creator, tools, algorithms
from devana.optimize.base import Solver

class AdaVEASolver(Solver):
    """
    Adaptive Vibration Engineering Algorithm (AdaVEA) Solver.
    
    Ported from AdaVEAWorker.py, removing PyQt5 dependencies.
    """
    def __init__(self, config, evaluate_fn=None, callback=None):
        super().__init__(config, evaluate_fn, callback)
        
        # AdaVEA specific configuration
        self.initial_cxpb = config.get('cxpb', 0.8)
        self.initial_mutpb = config.get('mutpb', 0.2)
        self.eta_c = config.get('eta_c', 20.0)
        self.eta_m = config.get('eta_m', 20.0)
        self.num_runs = config.get('num_runs', 1)
        self.random_seed = config.get('random_seed', 42)
        self.convergence_epsilon = config.get('convergence_epsilon', 1e-5)
        self.convergence_window = config.get('convergence_window', 10)
        self.convergence_min_gen = config.get('convergence_min_gen', 20)
        self.hv_ref_point = config.get('hv_ref_point', [1.0, 100.0, 100.0])
        self.heuristic_init_ratio = config.get('heuristic_init_ratio', 0.2)
        self.cost_coeffs = config.get('cost_coeffs', [1.0] * self.num_parameters)
        
        self._setup_deap()
        self.toolbox = base.Toolbox()
        
        self.toolbox.register("attr_float", self._generate_attr)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, 
                             lambda: [self.toolbox.attr_float(i) for i in range(self.num_parameters)])
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", self.evaluate)
        
        # Bounds for operators
        self.low_bounds = [b[0] for b in self.parameter_bounds]
        self.high_bounds = [b[1] for b in self.parameter_bounds]
        
        self.toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=self.low_bounds, up=self.high_bounds, eta=self.eta_c)
        self.toolbox.register("mutate", tools.mutPolynomialBounded, low=self.low_bounds, up=self.high_bounds, eta=self.eta_m, indpb=1.0/self.num_parameters)
        self.toolbox.register("select", tools.selNSGA2)

    def _setup_deap(self):
        """Setup DEAP types safely."""
        if not hasattr(creator, "FitnessMulti"):
            creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMulti)

    def _generate_attr(self, i):
        """Generate a random attribute value within bounds."""
        if i in self.fixed_parameters:
            return self.fixed_parameters[i]
        low, high = self.parameter_bounds[i]
        return random.uniform(low, high)

    def _heuristic_initialization(self):
        """
        Generate a heuristic individual by biasing some parameters towards lower bound.
        """
        ind = self.toolbox.individual()
        for i in range(len(ind)):
            if i not in self.fixed_parameters and random.random() < 0.3:
                ind[i] = self.parameter_bounds[i][0]
        return ind

    def solve(self):
        """Execute the AdaVEA optimization."""
        all_runs_data = []
        process = psutil.Process(os.getpid())

        for run_idx in range(self.num_runs):
            if self.stop_requested:
                break

            random.seed(self.random_seed + run_idx)
            np.random.seed(self.random_seed + run_idx)

            pop = []
            num_heuristic = int(self.pop_size * self.heuristic_init_ratio)
            for _ in range(num_heuristic):
                pop.append(self._heuristic_initialization())
            for _ in range(self.pop_size - num_heuristic):
                pop.append(self.toolbox.individual())

            # Evaluate initial population
            fitnesses = list(map(self.toolbox.evaluate, pop))
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit

            generation_metrics = []
            start_time = time.time()

            for gen in range(1, self.num_generations + 1):
                if self.stop_requested:
                    break

                gen_start = time.time()

                # Adaptive Crossover Rate
                tau_crossover = self.num_generations / 4.0
                current_cxpb = 0.5 + 0.5 * np.exp(-gen / tau_crossover)

                # Adaptive Mutation Rate
                current_mutpb = self.initial_mutpb * (1.0 - gen / self.num_generations) + \
                               (1.0/self.num_parameters) * (gen / self.num_generations)
                
                offspring = algorithms.varAnd(pop, self.toolbox, current_cxpb, current_mutpb)
                
                # Evaluate invalid individuals
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(self.toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # Selection
                pop = self.toolbox.select(pop + offspring, self.pop_size)
                
                current_pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]

                # Calculate Hypervolume
                try:
                    from deap.tools._hypervolume import hv
                    objs = [ind.fitness.values for ind in current_pareto_front]
                    hypervolume_val = hv.hypervolume(objs, self.hv_ref_point)
                except Exception:
                    hypervolume_val = 0.0

                time_gen = time.time() - gen_start
                memory_peak = process.memory_info().rss / (1024 * 1024) 

                metrics = {
                    "gen": gen,
                    "hv": hypervolume_val,
                    "n_pareto": len(current_pareto_front),
                    "time_gen": time_gen,
                    "memory_peak": memory_peak
                }
                generation_metrics.append(metrics)

                self._report_progress(gen, None, current_pareto_front[0], metrics)
                
                # Convergence check
                if gen > self.convergence_min_gen and gen % self.convergence_window == 0:
                    recent_hvs = [m['hv'] for m in generation_metrics[-self.convergence_window:]]
                    if np.max(recent_hvs) - np.min(recent_hvs) < self.convergence_epsilon:
                        break

            final_pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
            
            run_results = {
                "run_id": run_idx + 1,
                "total_time_hours": (time.time() - start_time) / 3600,
                "generation_metrics": generation_metrics,
                "final_pareto_front_objectives": [list(ind.fitness.values) for ind in final_pareto_front],
                "final_population_parameters": [list(ind) for ind in final_pareto_front],
            }
            all_runs_data.append(run_results)

        return all_runs_data
