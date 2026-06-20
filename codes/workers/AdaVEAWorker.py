import numpy as np
import random
import time
import psutil
import os

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
        # Parse dva_params
        self.parameter_names = [p[0] for p in dva_parameters]
        self.low_bounds = [p[1] for p in dva_parameters]
        self.high_bounds = [p[2] for p in dva_parameters]
        self.fixed_params = {i: p[4] for i, p in enumerate(dva_parameters) if p[3]} 
        self.cost_coeffs = [p[5] for p in dva_parameters]
        self.num_params = len(dva_parameters)

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
        self.hv_ref_point = hv_ref_point if hv_ref_point else [1.0, 100.0, 100.0]
        self.heuristic_init_ratio = heuristic_init_ratio

        self.is_running = False
        self.is_paused = False
        self.stop_requested = False

        # DEAP setup - Safely create classes
        if not hasattr(creator, "FitnessMulti"):
            creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -1.0, -1.0)) # Minimize all 3 objectives
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMulti)

        self.toolbox = base.Toolbox()
        # Attribute generator
        def attr_float(i):
            if i in self.fixed_params: return self.fixed_params[i]
            return random.uniform(self.low_bounds[i], self.high_bounds[i])
        
        self.toolbox.register("attr_float", attr_float)
        # Individual generator
        self.toolbox.register("individual", tools.initIterate, creator.Individual, 
                             lambda: [self.toolbox.attr_float(i) for i in range(self.num_params)])
        # Population generator
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("evaluate", self._evaluate_objectives)
        self.toolbox.register("mate", tools.cxSimulatedBinaryBounded, low=self.low_bounds, up=self.high_bounds, eta=self.eta_c)
        self.toolbox.register("mutate", tools.mutPolynomialBounded, low=self.low_bounds, up=self.high_bounds, eta=self.eta_m, indpb=1.0/self.num_params)
        self.toolbox.register("select", tools.selNSGA2) 

    def _evaluate_objectives(self, individual):
        try:
            # self.target_values_weights is expected to be a tuple (target_values_dict, weights_dict) 
            # as passed from adavea_mixin.py
            target_values_dict, weights_dict = self.target_values_weights
            
            results = frf(
                main_system_parameters=self.main_system_parameters,
                dva_parameters=tuple(individual),
                omega_start=self.omega_start,
                omega_end=self.omega_end,
                omega_points=self.omega_points,
                target_values_mass1=target_values_dict.get('mass_1', {}),
                weights_mass1=weights_dict.get('mass_1', {}),
                target_values_mass2=target_values_dict.get('mass_2', {}),
                weights_mass2=weights_dict.get('mass_2', {}),
                target_values_mass3=target_values_dict.get('mass_3', {}),
                weights_mass3=weights_dict.get('mass_3', {}),
                target_values_mass4=target_values_dict.get('mass_4', {}),
                weights_mass4=weights_dict.get('mass_4', {}),
                target_values_mass5=target_values_dict.get('mass_5', {}),
                weights_mass5=weights_dict.get('mass_5', {}),
                plot_figure=False,
                show_peaks=False,
                show_slopes=False,
            )
            
            f1 = results.get('singular_response', 1e9)
            if not np.isfinite(f1):
                f1 = 1e9
        except Exception:
            f1 = 1e9 

        # Objective 2: Sparsity 
        tau = 0.1 
        alpha = 1.0 
        beta = 0.5 
        
        n_active = sum(1 for x_i in individual if abs(x_i) > tau)
        sum_abs_xi = sum(abs(x_i) for x_i in individual)
        f2 = alpha * n_active + beta * sum_abs_xi

        # Objective 3: Cost 
        f3 = sum(c_i * x_i for c_i, x_i in zip(self.cost_coeffs, individual))

        return f1, f2, f3

    def _heuristic_initialization(self):
        """
        Generate a heuristic individual by biasing some parameters towards 0.
        """
        ind = self.toolbox.individual()
        for i in range(len(ind)):
            if random.random() < 0.3: # 30% chance to set to 0 (or lower bound)
                ind[i] = self.low_bounds[i]
        return ind

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

            generation_metrics = []
            start_time = time.time()

            for gen in range(1, self.generations + 1):
                if self.stop_requested:
                    break
                while self.is_paused:
                    time.sleep(0.1) 

                gen_start = time.time()
                process = psutil.Process(os.getpid())

                # --- Adaptive Crossover Rate ---
                tau_crossover = self.generations / 4.0
                current_cxpb = 0.5 + 0.5 * np.exp(-gen / tau_crossover)

                # --- Adaptive Mutation Rate ---
                current_mutpb = self.initial_mutpb * (1.0 - gen / self.generations) + (1.0/self.num_params) * (gen / self.generations)
                
                # Select the next generation individuals
                offspring = algorithms.varAnd(pop, self.toolbox, current_cxpb, current_mutpb)
                
                # Evaluate the individuals with an invalid fitness
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                fitnesses = self.toolbox.map(self.toolbox.evaluate, invalid_ind)
                for ind, fit in zip(invalid_ind, fitnesses):
                    ind.fitness.values = fit

                # Combine the current population and offspring
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
                    "igd": 0.0,
                    "n_pareto": len(current_pareto_front),
                    "time_gen": time_gen,
                    "memory_peak": memory_peak
                }
                generation_metrics.append(metrics)

                self.progress.emit(run_idx, gen, self.generations, metrics)
                
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