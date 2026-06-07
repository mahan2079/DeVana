import random
from deap import base, creator, tools
from devana.optimize.base import Solver

class NSGA2Solver(Solver):
    """
    Non-dominated Sorting Genetic Algorithm II (NSGA-II) Solver.
    
    Ported from NSGA2Worker.py, removing PyQt5 dependencies.
    """
    def __init__(self, config, evaluate_fn=None, callback=None):
        super().__init__(config, evaluate_fn, callback)
        
        # NSGA-II specific configuration
        self.cxpb = config.get('cxpb', 0.9)
        self.mutpb = config.get('mutpb', 0.1)
        self.eta_c = config.get('eta_c', 20.0)
        self.eta_m = config.get('eta_m', 20.0)
        self.indpb = config.get('indpb', 0.05)
        
        # Objectives weights (default to 3 objectives minimizing)
        self.weights = config.get('weights', (-1.0, -1.0, -1.0))
        
        # Setup DEAP
        self._setup_deap()
        self.toolbox = base.Toolbox()
        
        # Register operators
        self.toolbox.register("attr_float", self._generate_attr)
        self.toolbox.register("individual", tools.initIterate, creator.Individual, 
                             lambda: [self._generate_attr(i) for i in range(self.num_parameters)])
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        
        self.toolbox.register("evaluate", self.evaluate)
        
        # Use NSGA-II recommended operators
        low_bounds = [b[0] for b in self.parameter_bounds]
        high_bounds = [b[1] for b in self.parameter_bounds]
        self.toolbox.register("mate", tools.cxSimulatedBinaryBounded, 
                             low=low_bounds, up=high_bounds, eta=self.eta_c)
        self.toolbox.register("mutate", tools.mutPolynomialBounded, 
                             low=low_bounds, up=high_bounds, eta=self.eta_m, indpb=self.indpb)
        self.toolbox.register("select", tools.selNSGA2)

    def _setup_deap(self):
        """Setup DEAP types safely for multi-objective optimization."""
        if not hasattr(creator, "FitnessMulti"):
            creator.create("FitnessMulti", base.Fitness, weights=self.weights)
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMulti)

    def _generate_attr(self, i):
        """Generate a random attribute value within bounds."""
        if i in self.fixed_parameters:
            return self.fixed_parameters[i]
        low, high = self.parameter_bounds[i]
        return random.uniform(low, high)

    def solve(self):
        """Execute the NSGA-II optimization."""
        pop = self.toolbox.population(n=self.pop_size)
        
        # Evaluate initial population
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # This is just to assign the crowding distance to the individuals
        # no selection happens
        pop = self.toolbox.select(pop, len(pop))

        metrics = {
            'pareto_front_size_history': []
        }

        for gen in range(1, self.num_generations + 1):
            if self.stop_requested:
                break
                
            # Selection of parents
            offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = [self.toolbox.clone(ind) for ind in offspring]

            # Crossover and Mutation
            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.cxpb:
                    self.toolbox.mate(ind1, ind2)
                
                if random.random() < self.mutpb:
                    self.toolbox.mutate(ind1)
                if random.random() < self.mutpb:
                    self.toolbox.mutate(ind2)
                
                # Enforce fixed parameters after operators
                for child in (ind1, ind2):
                    for idx, val in self.fixed_parameters.items():
                        child[idx] = val
                    del child.fitness.values

            # Evaluate invalid individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # Select the next generation population
            pop = self.toolbox.select(pop + offspring, self.pop_size)
            
            # Pareto front info
            pareto_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
            metrics['pareto_front_size_history'].append(len(pareto_front))
            
            # For reporting, we can use the best individual from the first front (e.g., min sum of objectives)
            best_ind = min(pareto_front, key=lambda ind: sum(ind.fitness.values))
            best_fitness = best_ind.fitness.values
            
            self._report_progress(gen, best_fitness, best_ind, metrics)

        final_pareto = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
        
        return {
            'best_individual': best_ind, # One representative
            'best_fitness': best_fitness,
            'pareto_front': [list(ind) for ind in final_pareto],
            'pareto_objectives': [ind.fitness.values for ind in final_pareto],
            'metrics': metrics,
            'parameter_names': self.parameter_names
        }
