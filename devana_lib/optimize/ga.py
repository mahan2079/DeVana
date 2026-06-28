import random
from deap import base, creator, tools
from devana.optimize.base import Solver

class GASolver(Solver):
    """
    Genetic Algorithm (GA) Solver.
    
    Ported from GAWorker.py, removing PyQt5 dependencies.
    """
    def __init__(self, config, evaluate_fn=None, callback=None):
        super().__init__(config, evaluate_fn, callback)
        
        # GA specific configuration
        self.cxpb = config.get('cxpb', 0.8)
        self.mutpb = config.get('mutpb', 0.2)
        self.tournsize = config.get('tournsize', 3)
        self.indpb = config.get('indpb', 0.1)  # mutation probability for each attribute
        
        # Setup DEAP
        self._setup_deap()
        self.toolbox = base.Toolbox()
        
        # Register operators
        self.toolbox.register("attr_float", self._generate_attr)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, 
                             lambda: [self._generate_attr(i) for i in range(self.num_parameters)])
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        self.toolbox.register("evaluate", self.evaluate)
        self.toolbox.register("mate", tools.cxBlend, alpha=0.5)
        self.toolbox.register("mutate", self._mutate)
        self.toolbox.register("select", tools.selTournament, tournsize=self.tournsize)

    def _setup_deap(self):
        """Setup DEAP types safely."""
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)

    def _generate_attr(self, i):
        """Generate a random attribute value within bounds."""
        if i in self.fixed_parameters:
            return self.fixed_parameters[i]
        low, high = self.parameter_bounds[i]
        return random.uniform(low, high)

    def _mutate(self, individual):
        """Custom mutation that respects bounds and fixed parameters."""
        for i in range(len(individual)):
            if i in self.fixed_parameters:
                continue
            if random.random() < self.indpb:
                low, high = self.parameter_bounds[i]
                # Gaussian mutation with sigma as 10% of range
                sigma = (high - low) * 0.1
                individual[i] += random.gauss(0, sigma)
                individual[i] = max(low, min(high, individual[i]))
        return (individual,)

    def solve(self):
        """Execute the GA optimization."""
        pop = self.toolbox.population(n=self.pop_size)
        
        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = (fit,) if not isinstance(fit, (tuple, list)) else fit

        best_ind = None
        best_fitness = float('inf')
        
        metrics = {
            'fitness_history': [],
            'best_fitness_history': []
        }

        for gen in range(1, self.num_generations + 1):
            if self.stop_requested:
                break
                
            # Selection
            offspring = self.toolbox.select(pop, len(pop))
            offspring = list(map(self.toolbox.clone, offspring))

            # Crossover
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.cxpb:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Mutation
            for mutant in offspring:
                if random.random() < self.mutpb:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Evaluate invalid individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = (fit,) if not isinstance(fit, (tuple, list)) else fit

            # Replacement
            pop[:] = offspring

            # Gather stats
            fits = [ind.fitness.values[0] for ind in pop]
            min_fit = min(fits)
            if min_fit < best_fitness:
                best_fitness = min_fit
                best_ind = self.toolbox.clone(tools.selBest(pop, 1)[0])

            metrics['fitness_history'].append(fits)
            metrics['best_fitness_history'].append(best_fitness)

            self._report_progress(gen, best_fitness, best_ind, metrics)

            if best_fitness <= self.tolerance:
                break

        return {
            'best_individual': best_ind,
            'best_fitness': best_fitness,
            'metrics': metrics,
            'parameter_names': self.parameter_names
        }
