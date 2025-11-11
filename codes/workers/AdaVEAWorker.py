import numpy as np
import random
import time
import psutil
import json
import os
from collections import defaultdict

from PyQt5.QtCore import QObject, pyqtSignal

from deap import base, creator, tools, algorithms

# Assuming FRF function is available in modules.FRF
from modules.FRF import frf

class AdaVEAWorker(QObject):
    progress = pyqtSignal(int, int, int, dict) # run_idx, current_gen, total_gens, metrics
    finished = pyqtSignal(list) # all_runs_data
    error = pyqtSignal(str) # error_message

    def __init__(self, main_system_parameters, dva_parameters, target_values_weights,
                 omega_start, omega_end, omega_points,
                 pop_size, generations, cxpb, mutpb, eta_c, eta_m,
                 num_runs, random_seed, convergence_epsilon, convergence_window, convergence_min_gen,
                 hv_ref_point, heuristic_init_ratio):
        super().__init__()
        
        self.main_system_parameters = main_system_parameters
        self.dva_parameters = dva_parameters
        self.target_values_weights = target_values_weights
        self.omega_start = omega_start
        self.omega_end = omega_end
        self.omega_points = omega_points
        
        self.pop_size = pop_size
        self.generations = generations
        self.initial_cxpb = cxpb # Store initial for adaptive
        self.initial_mutpb = mutpb # Store initial for adaptive
        self.eta_c = eta_c
        self.eta_m = eta_m
        
        self.num_runs = num_runs
        self.random_seed = random_seed
        self.convergence_epsilon = convergence_epsilon
        self.convergence_window = convergence_window
        self.convergence_min_gen = convergence_min_gen
        self.hv_ref_point = hv_ref_point
        self.heuristic_init_ratio = heuristic_init_ratio

        self.is_running = False
        self.is_paused = False
        self.stop_requested = False

        # DEAP setup
        creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0)) # Minimize all 3 objectives
        creator.create("Individual", list, fitness=creator.FitnessMulti)

        self.toolbox = base.Toolbox()
        # Attribute generator
        # Assuming 48 parameters, each between 0 and 1
        self.toolbox.register("attr_float", random.uniform, 0, 1)
        # Individual generator
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float, n=48)
        # Population generator
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", self._evaluate_objectives)
        self.toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0, up=1, eta=self.eta_c)
        self.toolbox.register("mutate", tools.mutPolynomialBounded, low=0, up=1, eta=self.eta_m, indpb=1.0/48.0) # indpb is probability of each attribute being mutated
        self.toolbox.register("select", tools.selNSGA2) # AdaVEA still uses NSGA-II selection for non-dominated sorting

    def _evaluate_objectives(self, individual):
        """
        Evaluate the three objectives for a given individual (parameter set).
        f1: FRF minimization
        f2: Sparsity
        f3: Cost
        """
        # Convert individual to DVA parameter tuple
        dva_tuple = tuple(individual)

        # Objective 1: FRF Minimization (max_omega |H(omega, x)|)
        try:
            frf_results = frf(
                main_system_parameters=self.main_system_parameters,
                dva_parameters=dva_tuple,
                omega_start=self.omega_start,
                omega_end=self.omega_end,
                omega_points=self.omega_points,
                target_values_mass1=self.target_values_weights[0]['mass_1'],
                weights_mass1=self.target_values_weights[1]['mass_1'],
                target_values_mass2=self.target_values_weights[0]['mass_2'],
                weights_mass2=self.target_values_weights[1]['mass_2'],
                target_values_mass3=self.target_values_weights[0]['mass_3'],
                weights_mass3=self.target_values_weights[1]['mass_3'],
                target_values_mass4=self.target_values_weights[0]['mass_4'],
                weights_mass4=self.target_values_weights[1]['mass_4'],
                target_values_mass5=self.target_values_weights[0]['mass_5'],
                weights_mass5=self.target_values_weights[1]['mass_5'],
                plot_figure=False,
                show_peaks=False,
                show_slopes=False,
            )
            
            max_frf_magnitude = 0.0
            for mass_key in ['mass_1', 'mass_2', 'mass_3', 'mass_4', 'mass_5']:
                if mass_key in frf_results and 'magnitude' in frf_results[mass_key]:
                    max_frf_magnitude = max(max_frf_magnitude, np.max(frf_results[mass_key]['magnitude']))
            f1 = max_frf_magnitude
        except Exception as e:
            self.error.emit(f"Error during FRF calculation: {e}")
            f1 = 1e9 # Assign a very high penalty for failed FRF calculation

        # Objective 2: Sparsity (alpha * N_active + beta * sum(|x_i|))
        tau = 0.1 # Sparsity threshold
        alpha = 1.0 # Weight for cardinality term
        beta = 0.5 # Weight for magnitude term
        
        n_active = sum(1 for x_i in individual if abs(x_i) > tau)
        sum_abs_xi = sum(abs(x_i) for x_i in individual)
        f2 = alpha * n_active + beta * sum_abs_xi

        # Objective 3: Cost (sum(c_i * x_i))
        cost_coefficients = np.linspace(0.1, 1.0, 48) # Example: varying costs
        f3 = sum(c_i * x_i for c_i, x_i in zip(cost_coefficients, individual))

        return f1, f2, f3

    def _heuristic_initialization(self):
        """
        Generate a heuristic individual.
        Placeholder for actual heuristic logic.
        """
        # For now, just return a random individual as a placeholder
        # In a real scenario, this would involve domain-specific knowledge
        # to generate individuals that are likely to be good solutions.
        return self.toolbox.individual()

    def run(self):
        self.is_running = True
        self.stop_requested = False
        
        all_runs_data = []

        for run_idx in range(self.num_runs):
            if self.stop_requested:
                break

            random.seed(self.random_seed + run_idx)
            np.random.seed(self.random_seed + run_idx)

            pop = []
            # Heuristic Initialization
            num_heuristic = int(self.pop_size * self.heuristic_init_ratio)
            for _ in range(num_heuristic):
                pop.append(self._heuristic_initialization())
            # Random Initialization for the rest
            for _ in range(self.pop_size - num_heuristic):
                pop.append(self.toolbox.individual())

            # Evaluate the initial population
            fitnesses = self.toolbox.map(self.toolbox.evaluate, pop)
            for ind, fit in zip(pop, fitnesses):
                ind.fitness.values = fit

            logbook = tools.Logbook()
            logbook.header = "gen", "evals", "min_f1", "min_f2", "min_f3", "avg_f1", "avg_f2", "avg_f3", "max_f1", "max_f2", "max_f3", "hv", "igd", "gd", "spread", "n_pareto", "diversity", "time_gen", "memory_peak"

            stats_obj = tools.Statistics(lambda ind: ind.fitness.values)
            stats_obj.register("min", np.min, axis=0)
            stats_obj.register("avg", np.mean, axis=0)
            stats_obj.register("max", np.max, axis=0)
            
            generation_metrics = []

            for gen in range(1, self.generations + 1):
                if self.stop_requested:
                    break
                while self.is_paused:
                    time.sleep(0.1) # Wait while paused

                gen_start_time = time.time()
                process = psutil.Process(os.getpid())

                # --- Adaptive Crossover Rate ---
                # p_c(g) = 0.5 + 0.5 * e^(-g/tau)
                # Let's define tau for crossover adaptation, e.g., generations / 4
                tau_crossover = self.generations / 4.0
                current_cxpb = 0.5 + 0.5 * np.exp(-gen / tau_crossover)
                self.toolbox.unregister("mate")
                self.toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=0, up=1, eta=self.eta_c)

                # --- Adaptive Mutation Rate ---
                # p_m(g) = f(diversity)
                # For now, a simplified adaptive mutation based on generation progress
                # In a real scenario, diversity would be calculated and used.
                current_mutpb = self.initial_mutpb * (1.0 - gen / self.generations) + (1.0/48.0) * (gen / self.generations)
                self.toolbox.unregister("mutate")
                self.toolbox.register("mutate", tools.mutPolynomialBounded, low=0, up=1, eta=self.eta_m, indpb=current_mutpb)

                # Select the next generation individuals
                offspring = algorithms.varAnd(pop, self.toolbox, current_cxpb, current_mutpb)
                
                # --- Local Search (Hybrid Lamarckian-Baldwinian) ---
                # Placeholder: In a real implementation, this would involve
                # applying a local search operator to some individuals in the offspring
                # and then deciding whether to update the individual (Lamarckian)
                # or just its fitness (Baldwinian).
                # For now, we'll skip this complex step.

                # Evaluate the individuals with an invalid fitness
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # Combine the current population and offspring
                pop = self.toolbox.select(pop + offspring, self.pop_size)
                
                current_pareto_front = tools.selNSGA2(pop, len(pop))

                # Calculate per-generation metrics
                record = stats_obj.compile(pop)
                
                hv = 0.0 # Placeholder
                igd = 0.0
                gd = 0.0
                spread = 0.0
                diversity = 0.0
                n_pareto = len(current_pareto_front)

                gen_end_time = time.time()
                time_gen = gen_end_time - gen_start_time
                memory_peak = process.memory_info().rss / (1024 * 1024) # in MB

                # Append custom metrics to the record
                record['hv'] = hv
                record['igd'] = igd
                record['gd'] = gd
                record['spread'] = spread
                record['n_pareto'] = n_pareto
                record['diversity'] = diversity
                record['time_gen'] = time_gen
                record['memory_peak'] = memory_peak
                
                logbook.record(gen=gen, evals=len(invalid_ind), **record)
                generation_metrics.append(logbook.chapters["gen"].current)

                self.progress.emit(run_idx, gen, self.generations, logbook.chapters["gen"].current)
                
                # Check for convergence (placeholder logic)
                if gen > self.convergence_min_gen and gen % self.convergence_window == 0:
                    if len(generation_metrics) >= self.convergence_window:
                        recent_hvs = [m['hv'] for m in generation_metrics[-self.convergence_window:]]
                        if np.max(recent_hvs) - np.min(recent_hvs) < self.convergence_epsilon:
                            print(f"Convergence detected at generation {gen}")
                            # break # Uncomment to enable early stopping on convergence

            # After all generations or convergence
            final_pareto_front = tools.selNSGA2(pop, len(pop))
            
            # Save results for this run
            run_results = {
                "run_id": run_idx,
                "settings": {
                    "pop_size": self.pop_size,
                    "generations": self.generations,
                    "cxpb": self.initial_cxpb,
                    "mutpb": self.initial_mutpb,
                    "eta_c": self.eta_c,
                    "eta_m": self.eta_m,
                    "random_seed": self.random_seed + run_idx,
                    "convergence_epsilon": self.convergence_epsilon,
                    "convergence_window": self.convergence_window,
                    "convergence_min_gen": self.convergence_min_gen,
                    "hv_ref_point": self.hv_ref_point,
                    "heuristic_init_ratio": self.heuristic_init_ratio,
                },
                "generation_metrics": generation_metrics,
                "final_pareto_front_objectives": [list(ind.fitness.values) for ind in final_pareto_front],
                "final_population_parameters": [list(ind) for ind in final_pareto_front],
            }
            all_runs_data.append(run_results)

        self.is_running = False
        self.finished.emit(all_runs_data)

    def pause(self):
        self.is_paused = True

    def resume(self):
        self.is_paused = False

    def stop(self):
        self.stop_requested = True
        self.is_running = False
        self.is_paused = False