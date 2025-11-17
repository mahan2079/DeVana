import numpy as np
from deap import base, creator, tools, algorithms
from PyQt5.QtCore import QThread, pyqtSignal
import time
import random

from modules.FRF import frf

def safe_deap_operation(func):
    def wrapper(*args, **kwargs):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt < max_retries - 1:
                    try:
                        if hasattr(creator, "FitnessMulti"):
                            delattr(creator, "FitnessMulti")
                        if hasattr(creator, "Individual"):
                            delattr(creator, "Individual")
                    except Exception:
                        pass
                else:
                    raise
    return wrapper

class NSGA2Worker(QThread):
    progress = pyqtSignal(int, dict)
    finished = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, main_params, dva_params, target_values_weights, omega_start, omega_end, omega_points,
                 pop_size, generations, cxpb, mutpb, eta_c, eta_m, indpb, sparsity_tau, sparsity_alpha, sparsity_beta, parent=None):
        super().__init__(parent)
        self.main_params = main_params
        # Parse dva_params
        self.parameter_names = [p[0] for p in dva_params]
        self.low_bounds = [p[1] for p in dva_params]
        self.high_bounds = [p[2] for p in dva_params]
        self.fixed_params = {i: p[4] for i, p in enumerate(dva_params) if p[3]} # {index: fixed_value}
        self.cost_coeffs = [p[5] for p in dva_params]

        self.target_values_weights = target_values_weights
        self.omega_start = omega_start
        self.omega_end = omega_end
        self.omega_points = omega_points
        self.pop_size = pop_size
        self.generations = generations
        self.cxpb = cxpb
        self.mutpb = mutpb
        self.eta_c = eta_c
        self.eta_m = eta_m
        self.indpb = indpb
        self.sparsity_tau = sparsity_tau
        self.sparsity_alpha = sparsity_alpha
        self.sparsity_beta = sparsity_beta
        
        self.abort = False

    def stop(self):
        self.abort = True

    def evaluate(self, individual):
        # Objective 1: FRF
        try:
            results = frf(
                main_system_parameters=self.main_params,
                dva_parameters=tuple(individual),
                omega_start=self.omega_start,
                omega_end=self.omega_end,
                omega_points=self.omega_points,
                target_values_mass1=self.target_values_weights[0][0],
                weights_mass1=self.target_values_weights[0][1],
                target_values_mass2=self.target_values_weights[1][0],
                weights_mass2=self.target_values_weights[1][1],
                target_values_mass3=self.target_values_weights[2][0],
                weights_mass3=self.target_values_weights[2][1],
                target_values_mass4=self.target_values_weights[3][0],
                weights_mass4=self.target_values_weights[3][1],
                target_values_mass5=self.target_values_weights[4][0],
                weights_mass5=self.target_values_weights[4][1],
                plot_figure=False,
                show_peaks=False,
                show_slopes=False
            )
            f1 = results.get('singular_response', 1e6)
            if not np.isfinite(f1):
                f1 = 1e6
        except Exception as e:
            f1 = 1e6

        # Objective 2: Sparsity
        n_active = np.sum(np.array(individual) > self.sparsity_tau)
        f2 = self.sparsity_alpha * n_active + self.sparsity_beta * np.sum(np.abs(individual))

        # Objective 3: Cost
        f3 = np.sum(np.array(individual) * np.array(self.cost_coeffs))

        return f1, f2, f3

    @safe_deap_operation
    def run(self):
        try:
            creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0))
            creator.create("Individual", list, fitness=creator.FitnessMulti)

            toolbox = base.Toolbox()
            
            # Attribute generator
            def attr_float(i):
                if i in self.fixed_params:
                    return self.fixed_params[i]
                return random.uniform(self.low_bounds[i], self.high_bounds[i])
            
            toolbox.register("attr_float", attr_float)
            toolbox.register("individual", tools.initIterate, creator.Individual, 
                             lambda: [toolbox.attr_float(i) for i in range(len(self.low_bounds))])
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            toolbox.register("evaluate", self.evaluate)
            toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=self.low_bounds, up=self.high_bounds, eta=self.eta_c)
            toolbox.register("mutate", tools.mutPolynomialBounded, low=self.low_bounds, up=self.high_bounds, eta=self.eta_m, indpb=self.indpb)
            toolbox.register("select", tools.selNSGA2)

            pop = toolbox.population(n=self.pop_size)
            
            # Evaluate the entire population
            fitnesses = list(map(toolbox.evaluate, pop))
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit

            for gen in range(self.generations):
                if self.abort:
                    break

                # Select the next generation individuals
                offspring = toolbox.select(pop, len(pop))
                # Clone the selected individuals
                offspring = list(map(toolbox.clone, offspring))

                # Apply crossover and mutation on the offspring
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < self.cxpb:
                        toolbox.mate(child1, child2)
                        del child1.fitness.values
                        del child2.fitness.values

                for mutant in offspring:
                    if random.random() < self.mutpb:
                        toolbox.mutate(mutant)
                        del mutant.fitness.values

                # Evaluate the individuals with an invalid fitness
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = map(toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # The population is entirely replaced by the offspring
                pop[:] = offspring
                
                # Gather all the fitnesses in one list and print the stats
                fits = [ind.fitness.values for ind in pop]
                
                # Calculate metrics
                # This is a placeholder for the actual metrics calculation
                metrics = {
                    "Gen": gen + 1,
                    "HV": 0, # Placeholder
                    "IGD+": 0, # Placeholder
                    "GD": 0, # Placeholder
                    "Spread": 0, # Placeholder
                    "N_Pareto": len(tools.sortNondominated(pop, len(pop), first_front_only=True)[0]),
                    "Diversity": 0, # Placeholder
                    "Time (s)": 0, # Placeholder
                    "Memory (MB)": 0, # Placeholder
                    "Rank Diversity": 0, # Placeholder
                    "Best Fitness (f1,f2,f3)": min(fits)
                }
                
                self.progress.emit(int(((gen + 1) / self.generations) * 100), metrics)

            self.finished.emit(tools.sortNondominated(pop, len(pop), first_front_only=True)[0])

        except Exception as e:
            self.error.emit(str(e))
        finally:
            if hasattr(creator, "FitnessMulti"):
                delattr(creator, "FitnessMulti")
            if hasattr(creator, "Individual"):
                delattr(creator, "Individual")

