import random
import numpy as np
import psutil
from deap import base, creator, tools
from devana.optimize.base import Solver

def safe_deap_operation(func):
    """Decorator to handle DEAP creator issues."""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            if hasattr(creator, "FitnessMulti"):
                del creator.FitnessMulti
            if hasattr(creator, "Individual"):
                del creator.Individual
            raise
    return wrapper

class MOGASolver(Solver):
    """
    Multi-Objective Genetic Algorithm (MOGA) Solver.
    
    Ported from MOGAWorker.py, removing PyQt5 dependencies.
    """
    def __init__(self, config, evaluate_fn=None, callback=None):
        super().__init__(config, evaluate_fn, callback)
        
        # MOGA specific configuration
        self.cxpb = config.get('cxpb', 0.9)
        self.mutpb = config.get('mutpb', 0.1)
        self.eta_c = config.get('eta_c', 20.0)
        self.eta_m = config.get('eta_m', 20.0)
        self.indpb = config.get('indpb', 0.1)
        self.num_runs = config.get('num_runs', 1)
        self.random_seed = config.get('random_seed', None)
        
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
        self.toolbox.register("mutate", tools.mutPolynomialBounded, low=self.low_bounds, up=self.high_bounds, eta=self.eta_m, indpb=self.indpb)
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

    @safe_deap_operation
    def solve(self):
        """Execute the MOGA optimization."""
        all_runs_data = []
        process = psutil.Process()

        for run_idx in range(self.num_runs):
            if self.stop_requested:
                break
                
            seed = self.random_seed + run_idx if self.random_seed is not None else None
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)

            pop = self.toolbox.population(n=self.pop_size)
            
            # Evaluate initial population
            fitnesses = list(map(self.toolbox.evaluate, pop))
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit

            for gen in range(self.num_generations):
                if self.stop_requested:
                    break

                # Binary Tournament Selection with Crowding Distance
                offspring = tools.selTournamentDCD(pop, len(pop))
                offspring = [self.toolbox.clone(ind) for ind in offspring]

                # Crossover
                for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < self.cxpb:
                        self.toolbox.mate(ind1, ind2)
                        del ind1.fitness.values
                        del ind2.fitness.values

                # Mutation
                for ind in offspring:
                    if random.random() < self.mutpb:
                        self.toolbox.mutate(ind)
                        del ind.fitness.values

                # Evaluate invalid individuals
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(self.toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # Replacement using NSGA-II selection
                pop = self.toolbox.select(pop + offspring, self.pop_size)
                
                metrics = {
                    "gen": gen,
                    "min_f1": min(ind.fitness.values[0] for ind in pop),
                    "avg_f1": np.mean([ind.fitness.values[0] for ind in pop]),
                    "mem_mb": process.memory_info().rss / 1024 / 1024
                }
                
                self._report_progress(gen, metrics["min_f1"], pop[0], metrics)

            final_pop = [list(ind) for ind in tools.selNSGA2(pop, len(pop))]
            all_runs_data.append(final_pop)

        return all_runs_data
