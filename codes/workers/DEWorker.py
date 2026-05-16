import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random
import time
import multiprocessing as mp
from enum import Enum
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional, Union, Callable

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
import platform
import psutil

# Local imports
from modules.FRF import frf
from modules.sobol_sensitivity import (
    perform_sobol_analysis,
    calculate_and_save_errors,
    format_parameter_name
)

# Define DE strategy options as enum
class DEStrategy(Enum):
    RAND_1 = "rand/1"  # DE/rand/1: v = x_r1 + F*(x_r2 - x_r3)
    RAND_2 = "rand/2"  # DE/rand/2: v = x_r1 + F*(x_r2 - x_r3 + x_r4 - x_r5)
    BEST_1 = "best/1"  # DE/best/1: v = x_best + F*(x_r1 - x_r2)
    BEST_2 = "best/2"  # DE/best/2: v = x_best + F*(x_r1 - x_r2 + x_r3 - x_r4)
    CURRENT_TO_BEST_1 = "current-to-best/1"  # DE/current-to-best/1: v = x_i + F*(x_best - x_i) + F*(x_r1 - x_r2)
    CURRENT_TO_RAND_1 = "current-to-rand/1"  # DE/current-to-rand/1: v = x_i + K*(x_r1 - x_i) + F*(x_r2 - x_r3)

# Define adaptation method options
class AdaptiveMethod(Enum):
    NONE = "none"
    JITTER = "jitter"  # Small random perturbation to F
    DITHER = "dither"  # Random F for each generation
    SaDE = "sade"      # Self-adaptive DE
    JADE = "jade"      # Adaptive DE with archive
    SUCCESS_HISTORY = "success-history"  # Adaptation based on success history

@dataclass
class DEStatistics:
    """Class for tracking DE algorithm statistics during optimization"""
    generations: List[int] = None
    best_fitness_history: List[float] = None
    mean_fitness_history: List[float] = None
    diversity_history: List[float] = None
    parameter_mean_history: List[List[float]] = None
    parameter_std_history: List[List[float]] = None
    success_rates: List[float] = None
    execution_times: List[float] = None
    f_values: List[float] = None
    cr_values: List[float] = None
    
    def __post_init__(self):
        self.generations = []
        self.best_fitness_history = []
        self.mean_fitness_history = []
        self.diversity_history = []
        self.parameter_mean_history = []
        self.parameter_std_history = []
        self.success_rates = []
        self.execution_times = []
        self.f_values = []
        self.cr_values = []

@dataclass
class MultiRunStatistics:
    """Class for tracking statistics across multiple DE runs"""
    run_best_fitnesses: List[float] = None
    run_best_solutions: List[List[float]] = None
    run_convergence_gens: List[int] = None
    run_execution_times: List[float] = None
    parameter_distributions: Dict[str, List[float]] = None
    
    def __post_init__(self):
        self.run_best_fitnesses = []
        self.run_best_solutions = []
        self.run_convergence_gens = []
        self.run_execution_times = []
        self.parameter_distributions = {}

class DEWorker(QThread):
    # Signals: finished(dict, best_individual, parameter_names, best_fitness, statistics), error(str), update(str)
    finished = pyqtSignal(dict, list, list, float, object)
    error = pyqtSignal(str)
    update = pyqtSignal(str)
    progress = pyqtSignal(int, float, float)  # generation, best_fitness, diversity
    multi_run_progress = pyqtSignal(int, int)  # current_run, total_runs
    benchmark_data = pyqtSignal(dict)
    generation_metrics = pyqtSignal(dict)

    def __init__(self, 
                 main_params,
                 target_values_dict,
                 weights_dict,
                 omega_start,
                 omega_end,
                 omega_points,
                 de_pop_size=50,
                 de_num_generations=100,
                 de_F=0.5,           # Mutation factor (typically 0.5-0.9)
                 de_CR=0.7,          # Crossover probability (typically 0.7-1.0)
                 de_tol=1e-6,        # Tolerance for convergence
                 de_parameter_data=None,  # List of tuples: (name, lower bound, upper bound, fixed flag)
                 alpha=0.01,         # Sparsity penalty factor
                 beta=0.0,           # Smoothness penalty factor
                 strategy=DEStrategy.RAND_1,
                 adaptive_method=AdaptiveMethod.NONE,
                 adaptive_params=None,
                 termination_criteria=None,
                 use_parallel=False,
                 n_processes=None,
                 seed=None,
                 record_statistics=True,
                 constraint_handling="penalty",
                 diversity_preservation=False,
                 num_runs=1,
                 # GA/PSO parity: metrics and controllers
                 track_metrics=True,
                 use_ml_adaptive=False,
                 pop_min=None,
                 pop_max=None,
                 ml_ucb_c=0.6,
                 ml_adapt_population=True,
                 ml_diversity_weight=0.02,
                 ml_diversity_target=0.2,
                 use_rl_controller=False,
                 rl_alpha=0.1,
                 rl_gamma=0.9,
                 rl_epsilon=0.2,
                 rl_epsilon_decay=0.95):        # Number of independent runs
        super().__init__()
        
        # Initialize base parameters
        self.main_params = main_params
        self.target_values_dict = target_values_dict
        self.weights_dict = weights_dict
        self.omega_start = omega_start
        self.omega_end = omega_end
        self.omega_points = omega_points
        
        # DE algorithm parameters
        self.de_pop_size = de_pop_size
        self.de_num_generations = de_num_generations
        self.de_F = de_F
        self.de_CR = de_CR
        self.de_tol = de_tol
        self.de_parameter_data = de_parameter_data or []
        
        # Objective function parameters
        self.alpha = alpha  # Sparsity penalty
        self.beta = beta    # Smoothness penalty
        
        # Advanced DE configurations
        self.strategy = strategy if isinstance(strategy, DEStrategy) else DEStrategy(strategy)
        self.adaptive_method = adaptive_method if isinstance(adaptive_method, AdaptiveMethod) else AdaptiveMethod(adaptive_method)
        self.adaptive_params = adaptive_params or {}
        self.termination_criteria = termination_criteria or {"max_generations": de_num_generations, "tol": de_tol}
        
        # Computational settings
        self.use_parallel = use_parallel
        self.n_processes = n_processes or max(1, mp.cpu_count() - 1)
        
        # Set random seed for reproducibility if provided
        self.base_seed = seed
        
        # Statistics tracking
        self.record_statistics = record_statistics
        self.statistics = DEStatistics() if record_statistics else None
        
        # Additional options
        self.constraint_handling = constraint_handling  # "penalty", "reflection", "projection", "repair"
        self.diversity_preservation = diversity_preservation
        
        # Multiple run settings
        self.num_runs = num_runs
        self.multi_run_stats = MultiRunStatistics() if num_runs > 1 else None
        
        # Runtime variables
        self.should_stop = False
        self.pool = None

        # Metrics and controllers (GA/PSO parity)
        self.track_metrics = bool(track_metrics)
        self.use_ml_adaptive = bool(use_ml_adaptive)
        self.ml_ucb_c = float(ml_ucb_c)
        self.ml_adapt_population = bool(ml_adapt_population)
        self.ml_diversity_weight = float(ml_diversity_weight)
        self.ml_diversity_target = float(ml_diversity_target)
        self.use_rl_controller = bool(use_rl_controller)
        self.rl_alpha = float(rl_alpha)
        self.rl_gamma = float(rl_gamma)
        self.rl_epsilon = float(rl_epsilon)
        self.rl_epsilon_decay = float(rl_epsilon_decay)
        # Guardrails for dynamic population (if used)
        self.pop_min = pop_min if pop_min is not None else max(10, int(0.5 * self.de_pop_size))
        self.pop_max = pop_max if pop_max is not None else int(2.0 * self.de_pop_size)
        
        # Metrics store
        self.metrics = {
            'start_time': None,
            'end_time': None,
            'total_duration': None,
            'cpu_usage': [],
            'memory_usage': [],
            'system_info': {},
            'generation_times': [],
            'time_per_generation_breakdown': [],
            'evaluation_times': [],
            'mutation_times': [],
            'crossover_times': [],
            'selection_times': [],
            'fitness_history': [],
            'mean_fitness_history': [],
            'std_fitness_history': [],
            'best_fitness_per_gen': [],
            'best_individual_per_gen': [],
            'convergence_rate': [],
            'evaluation_count': 0,
            'pop_size_history': [],
            'rates_history': [],  # Track F/CR per gen
            'controller': 'fixed',
            'cpu_per_core': [],
            'memory_details': [],
            'thread_count': [],
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.metrics_timer = QTimer()
        self.metrics_timer.setSingleShot(False)
        self.metrics_timer_interval = 500
        # Watchdog timer
        self._watchdog = QTimer()
        self._watchdog.setSingleShot(True)
        self._watchdog.timeout.connect(self._handle_timeout)

    def run(self):
        try:
            # Controller descriptor for metrics
            try:
                if self.use_rl_controller:
                    self.metrics['controller'] = 'rl'
                elif self.use_ml_adaptive:
                    self.metrics['controller'] = 'ml_bandit'
                elif self.adaptive_method != AdaptiveMethod.NONE:
                    self.metrics['controller'] = 'adaptive'
                else:
                    self.metrics['controller'] = 'fixed'
                self.update.emit(f"DEBUG: DE controller is set to: {self.metrics['controller']}")
                if self.use_ml_adaptive:
                    self.update.emit(f"DEBUG: ML params: UCB c={self.ml_ucb_c:.2f}, pop_adapt={self.ml_adapt_population}, div_weight={self.ml_diversity_weight:.3f}, div_target={self.ml_diversity_target:.2f}")
                if self.use_rl_controller:
                    self.update.emit(f"DEBUG: RL params: alpha={self.rl_alpha:.3f}, gamma={self.rl_gamma:.3f}, epsilon={self.rl_epsilon:.3f}, decay={self.rl_epsilon_decay:.3f}")
            except Exception:
                pass

            if self.track_metrics:
                self._start_metrics_tracking()
            try:
                self._watchdog.start(600000)  # 10 minutes
            except Exception:
                pass
            if self.num_runs > 1:
                self._run_multiple()
            else:
                self._run_single()
        except Exception as e:
            self.error.emit(f"Error in DE optimization: {str(e)}")
            if self.pool:
                self.pool.close()
                self.pool.join()
        finally:
            try:
                if self._watchdog.isActive():
                    self._watchdog.stop()
            except Exception:
                pass

    def _run_multiple(self):
        """Execute multiple independent runs of the DE algorithm"""
        overall_start_time = time.time()
        parameter_names = [name for name, _, _, _ in self.de_parameter_data]
        
        for run in range(self.num_runs):
            if self.should_stop:
                break
                
            self.update.emit(f"Starting DE run {run + 1}/{self.num_runs}")
            self.multi_run_progress.emit(run + 1, self.num_runs)
            
            # Set unique seed for each run if base_seed is provided
            if self.base_seed is not None:
                current_seed = self.base_seed + run
                random.seed(current_seed)
                np.random.seed(current_seed)
            
            # Run single optimization
            run_start_time = time.time()
            best_solution, best_fitness, convergence_gen = self._run_single(return_convergence=True)
            run_time = time.time() - run_start_time
            
            # Store results
            self.multi_run_stats.run_best_fitnesses.append(best_fitness)
            self.multi_run_stats.run_best_solutions.append(best_solution)
            self.multi_run_stats.run_convergence_gens.append(convergence_gen)
            self.multi_run_stats.run_execution_times.append(run_time)
            
            # Update parameter distributions
            for i, param_name in enumerate(parameter_names):
                if param_name not in self.multi_run_stats.parameter_distributions:
                    self.multi_run_stats.parameter_distributions[param_name] = []
                self.multi_run_stats.parameter_distributions[param_name].append(best_solution[i])
        
        # Create statistical visualizations
        if not self.should_stop:
            self._create_multi_run_plots(parameter_names)
        
        # Find best overall solution
        best_run_idx = np.argmin(self.multi_run_stats.run_best_fitnesses)
        best_overall_solution = self.multi_run_stats.run_best_solutions[best_run_idx]
        best_overall_fitness = self.multi_run_stats.run_best_fitnesses[best_run_idx]
        
        # Emit final results
        results_dict = self._evaluate_solution(best_overall_solution)
        self.finished.emit(results_dict, best_overall_solution, parameter_names, 
                         best_overall_fitness, self.multi_run_stats)

    def _run_single(self, return_convergence=False):
        """Execute a single run of the DE algorithm"""
        start_time = time.time()
        
        # Extract parameter information
        parameter_names = []
        parameter_bounds = []
        fixed_parameters = {}
        
        for idx, (name, low, high, fixed) in enumerate(self.de_parameter_data):
            parameter_names.append(name)
            if fixed:
                parameter_bounds.append((low, low))
                fixed_parameters[idx] = low
            else:
                parameter_bounds.append((low, high))
        num_params = len(parameter_bounds)
        
        # Setup parallel processing if enabled
        if self.use_parallel:
            self.pool = mp.Pool(processes=self.n_processes)
            self.update.emit(f"[INFO] Running with {self.n_processes} parallel processes")
        
        # Initialize adaptive parameters if using an adaptive method
        self._initialize_adaptive_parameters(num_params)
        
        # Initialize population
        population = self._initialize_population(parameter_bounds, fixed_parameters, num_params)
        
        # Evaluate initial population
        if self.use_parallel:
            fitnesses = self._evaluate_population_parallel(population)
        else:
            fitnesses = [self.evaluate_individual(ind) for ind in population]
        
        # Identify global best
        best_idx = np.argmin(fitnesses)
        global_best = population[best_idx].copy()
        best_fitness = fitnesses[best_idx]
        
        # Record initial statistics
        if self.record_statistics:
            self._record_statistics(0, population, fitnesses, global_best, best_fitness, start_time)
        
        # Optional ML bandit setup for F/CR/pop
        if self.use_ml_adaptive:
            deltas = [-0.25, -0.1, 0.0, 0.1, 0.25]
            pop_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5]
            ml_actions = [(dF, dCR, pm) for dF in deltas for dCR in deltas for pm in pop_multipliers]
            ml_counts = [0 for _ in ml_actions]
            ml_sums = [0.0 for _ in ml_actions]
            ml_t = 0
            def ml_select(cur_F, cur_CR, cur_pop):
                nonlocal ml_t
                ml_t += 1
                scores = []
                for i in range(len(ml_actions)):
                    if ml_counts[i] == 0:
                        scores.append((float('inf'), i))
                    else:
                        avg = ml_sums[i] / ml_counts[i]
                        bonus = self.ml_ucb_c * np.sqrt(np.log(max(ml_t, 1)) / ml_counts[i])
                        scores.append((avg + bonus, i))
                scores.sort(key=lambda t: t[0], reverse=True)
                _, idx = scores[0]
                dF, dCR, pm = ml_actions[idx]
                new_F = max(0.05, min(1.0, cur_F * (1.0 + dF)))
                new_CR = max(0.0, min(1.0, cur_CR * (1.0 + dCR)))
                new_pop = int(min(self.pop_max, max(self.pop_min, round(cur_pop * pm))))
                return idx, new_F, new_CR, new_pop
            def ml_update(idx, reward):
                ml_counts[idx] += 1
                ml_sums[idx] += float(reward)
            def resize_population(pop_list, new_size, parameter_bounds, fixed_parameters):
                # Shrink: keep best individuals
                if new_size < len(pop_list):
                    # sort by current fitnesses
                    idx_sorted = np.argsort(fitnesses)
                    selected = [pop_list[i] for i in idx_sorted[:new_size]]
                    return selected
                # Grow: add random individuals around global best
                extra = new_size - len(pop_list)
                for _ in range(extra):
                    ind = []
                    for j in range(num_params):
                        lo, hi = parameter_bounds[j]
                        if j in fixed_parameters:
                            ind.append(fixed_parameters[j])
                        else:
                            center = global_best[j]
                            radius = (hi - lo) * 0.1
                            ind.append(random.uniform(max(lo, center - radius), min(hi, center + radius)))
                    pop_list.append(ind)
                return pop_list

        # Optional RL controller setup for F/CR/pop
        if self.use_rl_controller:
            deltas = [-0.25, -0.1, -0.05, 0.0, 0.05, 0.1, 0.25]
            pop_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5]
            rl_actions = [(dF, dCR, pm) for dF in deltas for dCR in deltas for pm in pop_multipliers]
            rl_q = {0: [0.0 for _ in rl_actions], 1: [0.0 for _ in rl_actions]}
            rl_state = 0
            def rl_select(cur_F, cur_CR, cur_pop):
                if random.random() < self.rl_epsilon:
                    idx = random.randrange(len(rl_actions))
                else:
                    values = rl_q[rl_state]
                    best_idx = int(np.argmax(values)) if values else 0
                    idx = best_idx
                dF, dCR, pm = rl_actions[idx]
                new_F = max(0.05, min(1.0, cur_F * (1.0 + dF)))
                new_CR = max(0.0, min(1.0, cur_CR * (1.0 + dCR)))
                new_pop = int(min(self.pop_max, max(self.pop_min, round(cur_pop * pm))))
                return idx, new_F, new_CR, new_pop
            def rl_update(state, action_idx, reward, next_state):
                q_old = rl_q[state][action_idx]
                q_next = max(rl_q[next_state]) if rl_q[next_state] else 0.0
                rl_q[state][action_idx] = q_old + self.rl_alpha * (reward + self.rl_gamma * q_next - q_old)
            def rl_resize_population(pop_list, new_size, parameter_bounds, fixed_parameters):
                # Shrink: keep best by fitness
                if new_size < len(pop_list):
                    idx_sorted = np.argsort(fitnesses)
                    return [pop_list[i] for i in idx_sorted[:new_size]]
                extra = new_size - len(pop_list)
                for _ in range(extra):
                    ind = []
                    for j in range(num_params):
                        lo, hi = parameter_bounds[j]
                        if j in fixed_parameters:
                            ind.append(fixed_parameters[j])
                        else:
                            center = global_best[j]
                            radius = (hi - lo) * 0.1
                            ind.append(random.uniform(max(lo, center - radius), min(hi, center + radius)))
                    pop_list.append(ind)
                return pop_list

        # DE main loop
        no_improvement_count = 0
        convergence_gen = self.de_num_generations
        
        for gen in range(1, self.de_num_generations + 1):
            if self.should_stop:
                break
                
            gen_start_time = time.time()
            self.update.emit(f"-- Generation {gen} --")
            
            # Adapt control parameters if using adaptive method
            if self.adaptive_method != AdaptiveMethod.NONE:
                self._adapt_control_parameters(gen, population, fitnesses)
            
            # Optionally apply ML/RL controller to adjust F/CR and population size
            if self.use_ml_adaptive or self.use_rl_controller:
                # Compute diversity and mean fitness for reward shaping
                cur_mean_fit = float(np.mean(fitnesses)) if fitnesses else float('inf')
                diversity_val = self._calculate_diversity(population)
                if self.use_ml_adaptive:
                    ml_idx, newF, newCR, newPop = ml_select(self.de_F, self.de_CR, len(population))
                    self.de_F, self.de_CR = newF, newCR
                    if self.ml_adapt_population and newPop != len(population):
                        population = resize_population(population, newPop, parameter_bounds, fixed_parameters)
                        # Recompute fitnesses for new individuals
                        if len(fitnesses) != len(population):
                            fitnesses = [self.evaluate_individual(ind) for ind in population]
                elif self.use_rl_controller:
                    rl_idx, newF, newCR, newPop = rl_select(self.de_F, self.de_CR, len(population))
                    self.de_F, self.de_CR = newF, newCR
                    if newPop != len(population):
                        population = rl_resize_population(population, newPop, parameter_bounds, fixed_parameters)
                        if len(fitnesses) != len(population):
                            fitnesses = [self.evaluate_individual(ind) for ind in population]

            # Create new generation
            new_population = []
            new_fitnesses = []
            successful_mutations = 0
            
            mut_time_acc = 0.0
            cross_time_acc = 0.0
            eval_time_acc = 0.0

            for i in range(self.de_pop_size):
                target = population[i]
                
                # Apply DE strategy to create donor vector
                _t0 = time.time()
                trial = self._apply_de_strategy(i, population, global_best, fitnesses, parameter_bounds, fixed_parameters, num_params)
                mut_time_acc += (time.time() - _t0)
                
                # Evaluate trial vector
                _t1 = time.time()
                trial_fitness = self.evaluate_individual(trial)
                eval_time_acc += (time.time() - _t1)
                
                # Selection: if trial is better, it replaces target
                if trial_fitness < fitnesses[i]:
                    new_population.append(trial)
                    new_fitnesses.append(trial_fitness)
                    successful_mutations += 1
                    
                    # Update global best if necessary
                    if trial_fitness < best_fitness:
                        global_best = trial.copy()
                        best_fitness = trial_fitness
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                else:
                    new_population.append(target)
                    new_fitnesses.append(fitnesses[i])
            
            population = new_population
            fitnesses = new_fitnesses
            
            # Calculate success rate for this generation
            success_rate = successful_mutations / self.de_pop_size
            
            # Apply diversity preservation if enabled
            if self.diversity_preservation:
                population = self._apply_diversity_preservation(population, fitnesses, parameter_bounds, fixed_parameters)
            
            # Record statistics for this generation
            if self.record_statistics:
                self._record_statistics(gen, population, fitnesses, global_best, best_fitness, gen_start_time, success_rate)
            
            # Record metrics for this generation
            if self.track_metrics:
                self.metrics['evaluation_times'].append(float(eval_time_acc))
                # We approximate crossover_time as 0 since our crossover is inside strategy; attribute to mutation
                self.metrics['mutation_times'].append(float(mut_time_acc))
                self.metrics['crossover_times'].append(0.0)
                # Selection occurs inline; rough estimate
                self.metrics['selection_times'].append(0.0)
                self.metrics['rates_history'].append({'generation': gen, 'F': self.de_F, 'CR': self.de_CR})
                self.metrics['pop_size_history'].append(len(population))

            # Report progress
            diversity = self._calculate_diversity(population)
            self.update.emit(f"  Generation {gen}: Best={best_fitness:.6f}, Mean={np.mean(fitnesses):.6f}, Div={diversity:.6f}, SR={success_rate:.2f}")
            self.progress.emit(gen, best_fitness, diversity)
            
            # Check for convergence
            if self._check_termination(gen, best_fitness, no_improvement_count, diversity):
                convergence_gen = gen
                break
        
            # Metrics bookkeeping at end of generation
            if self.track_metrics:
                gen_time = time.time() - gen_start_time
                self.metrics['generation_times'].append(gen_time)
                self.metrics['time_per_generation_breakdown'].append({
                    'total': gen_time,
                    'mutation': float(mut_time_acc),
                    'evaluation': float(eval_time_acc)
                })
                fits = fitnesses[:]
                self.metrics['fitness_history'].append(fits)
                mean = float(np.mean(fits)) if fits else float('inf')
                std = float(np.std(fits)) if fits else 0.0
                self.metrics['mean_fitness_history'].append(mean)
                self.metrics['std_fitness_history'].append(std)
                self.metrics['best_fitness_per_gen'].append(best_fitness)
                self.metrics['best_individual_per_gen'].append(list(global_best))
                if len(self.metrics['best_fitness_per_gen']) > 1:
                    prev_best = self.metrics['best_fitness_per_gen'][-2]
                    cur_best = self.metrics['best_fitness_per_gen'][-1]
                    self.metrics['convergence_rate'].append(max(0.0, prev_best - cur_best))

                # Controller reward logging
                if self.use_ml_adaptive or self.use_rl_controller:
                    last_best = self.metrics['best_fitness_per_gen'][-2] if len(self.metrics['best_fitness_per_gen']) > 1 else None
                    imp = (last_best - best_fitness) if (last_best is not None and last_best > best_fitness) else 0.0
                    cv = (std / (abs(mean) + 1e-12)) if mean == mean else 0.0
                    effort = max(1.0, len(fitnesses))
                    reward = (imp / max(gen_time, 1e-6)) / effort - self.ml_diversity_weight * abs(cv - self.ml_diversity_target)
                    if self.use_ml_adaptive:
                        try:
                            ml_update(ml_idx, reward)
                        except Exception:
                            pass
                        self.metrics.setdefault('ml_controller_history', []).append({
                            'generation': gen,
                            'F': self.de_F,
                            'CR': self.de_CR,
                            'pop': len(population),
                            'best_fitness': best_fitness,
                            'mean_fitness': mean,
                            'std_fitness': std,
                            'reward': reward
                        })
                    elif self.use_rl_controller:
                        try:
                            next_state = 1 if imp > 0 else 0
                            rl_update(rl_state, rl_idx, reward, next_state)
                            rl_state = next_state
                            self.rl_epsilon *= self.rl_epsilon_decay
                        except Exception:
                            pass
                        self.metrics.setdefault('rl_controller_history', []).append({
                            'generation': gen,
                            'F': self.de_F,
                            'CR': self.de_CR,
                            'pop': len(population),
                            'best_fitness': best_fitness,
                            'mean_fitness': mean,
                            'std_fitness': std,
                            'reward': reward,
                            'epsilon': self.rl_epsilon
                        })

        # Cleanup
        if self.pool:
            self.pool.close()
            self.pool.join()
        
        if return_convergence:
            return global_best, best_fitness, convergence_gen
        else:
            # Emit results for single run
            results_dict = self._evaluate_solution(global_best)
            if self.track_metrics:
                self._stop_metrics_tracking()
                if not self.metrics.get('system_info'):
                    self.metrics['system_info'] = self._get_system_info()
                self.metrics['total_duration'] = time.time() - start_time
                results_dict['benchmark_metrics'] = self.metrics
                try:
                    self.benchmark_data.emit(self.metrics)
                except Exception:
                    pass
            self.finished.emit(results_dict, global_best, parameter_names, best_fitness, self.statistics)

    def _create_multi_run_plots(self, parameter_names):
        """Create statistical visualizations for multiple runs"""
        try:
            # Create figure for multiple subplots
            n_params = len(parameter_names)
            n_rows = (n_params + 2 + 1) // 2  # Parameters + fitness + convergence, 2 columns
            fig = plt.figure(figsize=(15, 5 * n_rows))
            
            # 1. Parameter distributions (violin plots)
            for i, param_name in enumerate(parameter_names):
                plt.subplot(n_rows, 2, i + 1)
                param_values = self.multi_run_stats.parameter_distributions[param_name]
                sns.violinplot(data=param_values)
                plt.title(f'Distribution of {param_name} across runs')
                plt.ylabel('Parameter Value')
            
            # 2. Best fitness distribution
            plt.subplot(n_rows, 2, n_params + 1)
            sns.violinplot(data=self.multi_run_stats.run_best_fitnesses)
            plt.title('Distribution of Best Fitness Values')
            plt.ylabel('Fitness Value')
            
            # 3. Convergence generation distribution
            plt.subplot(n_rows, 2, n_params + 2)
            sns.violinplot(data=self.multi_run_stats.run_convergence_gens)
            plt.title('Distribution of Convergence Generations')
            plt.ylabel('Generation')
            
            plt.tight_layout()
            
            # Save the plot
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            plot_dir = "decoupled_results"
            os.makedirs(plot_dir, exist_ok=True)
            plt.savefig(os.path.join(plot_dir, f'de_multi_run_stats_{timestamp}.png'))
            plt.close()
            
            # Create summary statistics DataFrame
            stats_data = {
                'Parameter': parameter_names + ['Fitness', 'Convergence Gen', 'Execution Time'],
                'Mean': [np.mean(self.multi_run_stats.parameter_distributions[param]) for param in parameter_names] + 
                       [np.mean(self.multi_run_stats.run_best_fitnesses),
                        np.mean(self.multi_run_stats.run_convergence_gens),
                        np.mean(self.multi_run_stats.run_execution_times)],
                'Std': [np.std(self.multi_run_stats.parameter_distributions[param]) for param in parameter_names] + 
                      [np.std(self.multi_run_stats.run_best_fitnesses),
                       np.std(self.multi_run_stats.run_convergence_gens),
                       np.std(self.multi_run_stats.run_execution_times)],
                'Min': [np.min(self.multi_run_stats.parameter_distributions[param]) for param in parameter_names] + 
                      [np.min(self.multi_run_stats.run_best_fitnesses),
                       np.min(self.multi_run_stats.run_convergence_gens),
                       np.min(self.multi_run_stats.run_execution_times)],
                'Max': [np.max(self.multi_run_stats.parameter_distributions[param]) for param in parameter_names] + 
                      [np.max(self.multi_run_stats.run_best_fitnesses),
                       np.max(self.multi_run_stats.run_convergence_gens),
                       np.max(self.multi_run_stats.run_execution_times)]
            }
            
            stats_df = pd.DataFrame(stats_data)
            stats_df.to_csv(os.path.join(plot_dir, f'de_multi_run_stats_{timestamp}.csv'), index=False)
            
        except Exception as e:
            self.error.emit(f"Error creating multi-run plots: {str(e)}")

    def _handle_timeout(self):
        try:
            self.should_stop = True
            self.update.emit("DE optimization timed out. The operation was taking too long.")
        except Exception:
            pass

    def _get_system_info(self):
        try:
            system_info = {
                'platform': platform.system(),
                'platform_release': platform.release(),
                'platform_version': platform.version(),
                'architecture': platform.machine(),
                'processor': platform.processor(),
                'physical_cores': psutil.cpu_count(logical=False),
                'total_cores': psutil.cpu_count(logical=True),
                'total_memory': round(psutil.virtual_memory().total / (1024.0 ** 3), 2),
                'python_version': platform.python_version(),
            }
            try:
                cpu_freq = psutil.cpu_freq()
                if cpu_freq:
                    system_info['cpu_max_freq'] = cpu_freq.max
                    system_info['cpu_min_freq'] = cpu_freq.min
                    system_info['cpu_current_freq'] = cpu_freq.current
            except Exception:
                pass
            return system_info
        except Exception as e:
            try:
                self.update.emit(f"Warning: Could not collect complete system info: {str(e)}")
            except Exception:
                pass
            return {'error': str(e)}

    def _update_resource_metrics(self):
        if not self.track_metrics:
            return
        try:
            cpu_percent = psutil.cpu_percent(interval=None)
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_usage_mb = memory_info.rss / (1024 * 1024)

            self.metrics['cpu_usage'].append(cpu_percent)
            self.metrics['memory_usage'].append(memory_usage_mb)

            per_core = psutil.cpu_percent(interval=None, percpu=True)
            self.metrics['cpu_per_core'].append(per_core)

            mem_details = {
                'rss': memory_info.rss / (1024 * 1024),
                'vms': memory_info.vms / (1024 * 1024),
                'shared': getattr(memory_info, 'shared', 0) / (1024 * 1024),
                'system_total': psutil.virtual_memory().total / (1024 * 1024),
                'system_available': psutil.virtual_memory().available / (1024 * 1024),
                'system_percent': psutil.virtual_memory().percent,
            }
            self.metrics['memory_details'].append(mem_details)
            self.metrics['thread_count'].append(process.num_threads())

            current_metrics = {
                'cpu': cpu_percent,
                'cpu_per_core': per_core,
                'memory': memory_usage_mb,
                'memory_details': mem_details,
                'thread_count': process.num_threads(),
                'time': time.time() - self.metrics['start_time'] if self.metrics.get('start_time') else 0
            }
            self.generation_metrics.emit(current_metrics)
        except Exception as e:
            try:
                self.update.emit(f"Warning: Failed to update resource metrics: {str(e)}")
            except Exception:
                pass

    def _start_metrics_tracking(self):
        if not self.track_metrics:
            return
        self.metrics['start_time'] = time.time()
        if not self.metrics.get('system_info'):
            self.metrics['system_info'] = self._get_system_info()
        self.metrics_timer.timeout.connect(self._update_resource_metrics)
        self.metrics_timer.start(self.metrics_timer_interval)
        try:
            self.update.emit(f"Started metrics tracking with interval: {self.metrics_timer_interval}ms")
        except Exception:
            pass

    def _stop_metrics_tracking(self):
        if not self.track_metrics:
            return
        try:
            self.metrics_timer.stop()
        except Exception:
            pass
        self.metrics['end_time'] = time.time()
        if self.metrics.get('start_time'):
            self.metrics['total_duration'] = self.metrics['end_time'] - self.metrics['start_time']
        try:
            if not self.metrics.get('cpu_usage'):
                self._update_resource_metrics()
        except Exception:
            pass

    def _evaluate_solution(self, solution):
        """Evaluate a solution and return the results dictionary"""
        # Create a dictionary of parameter values
        param_dict = {}
        for i, (name, _, _, _) in enumerate(self.de_parameter_data):
            param_dict[name] = solution[i]
        
        # Update main parameters with the solution
        updated_params = self.main_params.copy()
        updated_params.update(param_dict)
        
        # Perform FRF analysis
        frf_analyzer = frf.FRF(updated_params)
        results = frf_analyzer.analyze(self.omega_start, self.omega_end, self.omega_points)
        
        return results

    def evaluate_individual(self, individual):
        """
        Evaluate the fitness of an individual (candidate DVA parameters)
        using the FRF function. The fitness is defined as the absolute difference 
        between the singular response and 1 plus a sparsity penalty and smoothness penalty.
        """
        try:
            results = frf(
                main_system_parameters=self.main_params,
                dva_parameters=tuple(individual),
                omega_start=self.omega_start,
                omega_end=self.omega_end,
                omega_points=self.omega_points,
                target_values_mass1=self.target_values_dict['mass_1'],
                weights_mass1=self.weights_dict['mass_1'],
                target_values_mass2=self.target_values_dict['mass_2'],
                weights_mass2=self.weights_dict['mass_2'],
                target_values_mass3=self.target_values_dict['mass_3'],
                weights_mass3=self.weights_dict['mass_3'],
                target_values_mass4=self.target_values_dict['mass_4'],
                weights_mass4=self.weights_dict['mass_4'],
                target_values_mass5=self.target_values_dict['mass_5'],
                weights_mass5=self.weights_dict['mass_5'],
                plot_figure=False,
                show_peaks=False,
                show_slopes=False
            )
            singular_response = results.get('singular_response', None)
            if singular_response is None or not np.isfinite(singular_response):
                return 1e6
            else:
                primary_objective = abs(singular_response - 1)
                sparsity_penalty = self.alpha * sum(abs(x) for x in individual)
                
                # Smoothness penalty (parameter differences)
                smoothness_penalty = 0
                if self.beta > 0 and len(individual) > 1:
                    for i in range(len(individual) - 1):
                        smoothness_penalty += abs(individual[i+1] - individual[i])
                    smoothness_penalty *= self.beta
                
                return primary_objective + sparsity_penalty + smoothness_penalty
        except Exception as e:
            return 1e6

    def _initialize_population(self, parameter_bounds, fixed_parameters, num_params):
        """Initialize the population using various methods"""
        population = []
        
        # Use Latin Hypercube Sampling for better initial coverage if available
        try:
            from scipy.stats.qmc import LatinHypercube
            
            # Count non-fixed parameters
            num_free_params = num_params - len(fixed_parameters)
            
            if num_free_params > 0:
                # Generate Latin Hypercube samples for free parameters
                sampler = LatinHypercube(d=num_free_params)
                samples = sampler.random(n=self.de_pop_size)
                
                # Scale samples to parameter bounds
                free_bounds = [(parameter_bounds[j][0], parameter_bounds[j][1]) 
                              for j in range(num_params) if j not in fixed_parameters]
                
                for i in range(self.de_pop_size):
                    individual = []
                    free_idx = 0
                    
                    for j in range(num_params):
                        if j in fixed_parameters:
                            individual.append(fixed_parameters[j])
                        else:
                            low, high = free_bounds[free_idx]
                            value = low + samples[i, free_idx] * (high - low)
                            individual.append(value)
                            free_idx += 1
                    
                    population.append(individual)
            else:
                # All parameters are fixed
                individual = [fixed_parameters[j] for j in range(num_params)]
                for _ in range(self.de_pop_size):
                    population.append(individual.copy())
                
        except ImportError:
            # Fall back to random initialization if scipy is not available
            for i in range(self.de_pop_size):
                individual = []
                for j in range(num_params):
                    low, high = parameter_bounds[j]
                    if j in fixed_parameters:
                        value = fixed_parameters[j]
                    else:
                        value = random.uniform(low, high)
                    individual.append(value)
                population.append(individual)
                
        return population

    def _evaluate_population_parallel(self, population):
        """Evaluate population in parallel"""
        return self.pool.map(self.evaluate_individual, population)

    def _apply_de_strategy(self, i, population, global_best, fitnesses, parameter_bounds, fixed_parameters, num_params):
        """Apply the selected DE strategy to create a trial vector"""
        target = population[i]
        
        # Get current F and CR values (may be adapted)
        F = self._get_current_F(i)
        CR = self._get_current_CR(i)
        
        # Select indices for mutation (excluding current index i)
        idxs = list(range(self.de_pop_size))
        idxs.remove(i)
        
        # Apply different mutation strategies
        if self.strategy == DEStrategy.RAND_1:
            # DE/rand/1: v = x_r1 + F*(x_r2 - x_r3)
            r1, r2, r3 = random.sample(idxs, 3)
            x_r1, x_r2, x_r3 = population[r1], population[r2], population[r3]
            donor = self._create_donor_rand_1(x_r1, x_r2, x_r3, F, parameter_bounds, fixed_parameters, num_params)
            
        elif self.strategy == DEStrategy.RAND_2:
            # DE/rand/2: v = x_r1 + F*(x_r2 - x_r3 + x_r4 - x_r5)
            r1, r2, r3, r4, r5 = random.sample(idxs, 5)
            x_r1, x_r2, x_r3 = population[r1], population[r2], population[r3]
            x_r4, x_r5 = population[r4], population[r5]
            donor = self._create_donor_rand_2(x_r1, x_r2, x_r3, x_r4, x_r5, F, parameter_bounds, fixed_parameters, num_params)
            
        elif self.strategy == DEStrategy.BEST_1:
            # DE/best/1: v = x_best + F*(x_r1 - x_r2)
            r1, r2 = random.sample(idxs, 2)
            x_r1, x_r2 = population[r1], population[r2]
            donor = self._create_donor_best_1(global_best, x_r1, x_r2, F, parameter_bounds, fixed_parameters, num_params)
            
        elif self.strategy == DEStrategy.BEST_2:
            # DE/best/2: v = x_best + F*(x_r1 - x_r2 + x_r3 - x_r4)
            r1, r2, r3, r4 = random.sample(idxs, 4)
            x_r1, x_r2 = population[r1], population[r2]
            x_r3, x_r4 = population[r3], population[r4]
            donor = self._create_donor_best_2(global_best, x_r1, x_r2, x_r3, x_r4, F, parameter_bounds, fixed_parameters, num_params)
            
        elif self.strategy == DEStrategy.CURRENT_TO_BEST_1:
            # DE/current-to-best/1: v = x_i + F*(x_best - x_i) + F*(x_r1 - x_r2)
            r1, r2 = random.sample(idxs, 2)
            x_r1, x_r2 = population[r1], population[r2]
            donor = self._create_donor_current_to_best_1(target, global_best, x_r1, x_r2, F, parameter_bounds, fixed_parameters, num_params)
            
        elif self.strategy == DEStrategy.CURRENT_TO_RAND_1:
            # DE/current-to-rand/1: v = x_i + K*(x_r1 - x_i) + F*(x_r2 - x_r3)
            r1, r2, r3 = random.sample(idxs, 3)
            x_r1, x_r2, x_r3 = population[r1], population[r2], population[r3]
            K = random.random()  # Random value between 0 and 1
            donor = self._create_donor_current_to_rand_1(target, x_r1, x_r2, x_r3, K, F, parameter_bounds, fixed_parameters, num_params)
        
        else:
            # Default to DE/rand/1 if strategy not recognized
            r1, r2, r3 = random.sample(idxs, 3)
            x_r1, x_r2, x_r3 = population[r1], population[r2], population[r3]
            donor = self._create_donor_rand_1(x_r1, x_r2, x_r3, F, parameter_bounds, fixed_parameters, num_params)
        
        # Crossover: create trial vector
        trial = self._apply_crossover(target, donor, CR, fixed_parameters, num_params)
        
        # Handle constraints if needed
        trial = self._handle_constraints(trial, parameter_bounds)
        
        return trial

    def _create_donor_rand_1(self, x_r1, x_r2, x_r3, F, parameter_bounds, fixed_parameters, num_params):
        """Create donor vector using DE/rand/1 strategy"""
        donor = []
        for j in range(num_params):
            if j in fixed_parameters:
                donor.append(fixed_parameters[j])
            else:
                mutated_val = x_r1[j] + F * (x_r2[j] - x_r3[j])
                donor.append(mutated_val)
        return donor

    def _create_donor_rand_2(self, x_r1, x_r2, x_r3, x_r4, x_r5, F, parameter_bounds, fixed_parameters, num_params):
        """Create donor vector using DE/rand/2 strategy"""
        donor = []
        for j in range(num_params):
            if j in fixed_parameters:
                donor.append(fixed_parameters[j])
            else:
                mutated_val = x_r1[j] + F * (x_r2[j] - x_r3[j] + x_r4[j] - x_r5[j])
                donor.append(mutated_val)
        return donor

    def _create_donor_best_1(self, x_best, x_r1, x_r2, F, parameter_bounds, fixed_parameters, num_params):
        """Create donor vector using DE/best/1 strategy"""
        donor = []
        for j in range(num_params):
            if j in fixed_parameters:
                donor.append(fixed_parameters[j])
            else:
                mutated_val = x_best[j] + F * (x_r1[j] - x_r2[j])
                donor.append(mutated_val)
        return donor

    def _create_donor_best_2(self, x_best, x_r1, x_r2, x_r3, x_r4, F, parameter_bounds, fixed_parameters, num_params):
        """Create donor vector using DE/best/2 strategy"""
        donor = []
        for j in range(num_params):
            if j in fixed_parameters:
                donor.append(fixed_parameters[j])
            else:
                mutated_val = x_best[j] + F * (x_r1[j] - x_r2[j] + x_r3[j] - x_r4[j])
                donor.append(mutated_val)
        return donor

    def _create_donor_current_to_best_1(self, x_i, x_best, x_r1, x_r2, F, parameter_bounds, fixed_parameters, num_params):
        """Create donor vector using DE/current-to-best/1 strategy"""
        donor = []
        for j in range(num_params):
            if j in fixed_parameters:
                donor.append(fixed_parameters[j])
            else:
                mutated_val = x_i[j] + F * (x_best[j] - x_i[j]) + F * (x_r1[j] - x_r2[j])
                donor.append(mutated_val)
        return donor

    def _create_donor_current_to_rand_1(self, x_i, x_r1, x_r2, x_r3, K, F, parameter_bounds, fixed_parameters, num_params):
        """Create donor vector using DE/current-to-rand/1 strategy"""
        donor = []
        for j in range(num_params):
            if j in fixed_parameters:
                donor.append(fixed_parameters[j])
            else:
                mutated_val = x_i[j] + K * (x_r1[j] - x_i[j]) + F * (x_r2[j] - x_r3[j])
                donor.append(mutated_val)
        return donor

    def _apply_crossover(self, target, donor, CR, fixed_parameters, num_params):
        """Apply binomial crossover to create trial vector"""
        trial = []
        j_rand = random.randint(0, num_params - 1)
        
        for j in range(num_params):
            if j in fixed_parameters:
                trial.append(fixed_parameters[j])
            else:
                if random.random() <= CR or j == j_rand:
                    trial.append(donor[j])
                else:
                    trial.append(target[j])
        
        return trial

    def _handle_constraints(self, trial, parameter_bounds):
        """Handle constraints based on the selected method"""
        if self.constraint_handling == "penalty":
            # Already handled in fitness evaluation
            return trial
        
        elif self.constraint_handling == "reflection":
            # Reflect values that are out of bounds
            for j, (low, high) in enumerate(parameter_bounds):
                if trial[j] < low:
                    trial[j] = low + (low - trial[j])
                    # Make sure reflection doesn't go beyond upper bound
                    trial[j] = min(trial[j], high)
                elif trial[j] > high:
                    trial[j] = high - (trial[j] - high)
                    # Make sure reflection doesn't go below lower bound
                    trial[j] = max(trial[j], low)
            return trial
        
        elif self.constraint_handling == "projection":
            # Simply project values to boundaries
            for j, (low, high) in enumerate(parameter_bounds):
                trial[j] = max(low, min(trial[j], high))
            return trial
        
        else:  # Default to projection
            for j, (low, high) in enumerate(parameter_bounds):
                trial[j] = max(low, min(trial[j], high))
            return trial

    def _initialize_adaptive_parameters(self, num_params):
        """Initialize parameters for adaptive methods"""
        if self.adaptive_method == AdaptiveMethod.NONE:
            return
        
        # Setup JADE parameters
        if self.adaptive_method == AdaptiveMethod.JADE:
            self.adaptive_params["mu_F"] = 0.5  # Mean of successful mutation factors
            self.adaptive_params["mu_CR"] = 0.5  # Mean of successful crossover rates
            self.adaptive_params["archive"] = []  # Archive of successful solutions
            self.adaptive_params["archive_size"] = self.de_pop_size  # Size of archive
            self.adaptive_params["c"] = 0.1  # Adaptation rate
            
        # Setup SaDE parameters
        elif self.adaptive_method == AdaptiveMethod.SaDE:
            self.adaptive_params["CRm"] = 0.5  # Mean of normal distribution for CR
            self.adaptive_params["success_memory"] = []  # Memory of successful CRs
            self.adaptive_params["memory_size"] = 20  # Size of memory
            self.adaptive_params["LP"] = 50  # Learning period
            
        # Setup success history parameters
        elif self.adaptive_method == AdaptiveMethod.SUCCESS_HISTORY:
            self.adaptive_params["F_history"] = []  # History of successful F values
            self.adaptive_params["CR_history"] = []  # History of successful CR values
            self.adaptive_params["window_size"] = 10  # Size of history window
            
        # For jitter, we'll just use a small random perturbation in _get_current_F
        # For dither, we'll generate random F for each generation in _adapt_control_parameters

    def _adapt_control_parameters(self, gen, population, fitnesses):
        """Adapt control parameters based on the selected method"""
        if self.adaptive_method == AdaptiveMethod.NONE:
            return
            
        elif self.adaptive_method == AdaptiveMethod.DITHER:
            # Dither: set F to a random value in a specified range for each generation
            F_min = self.adaptive_params.get("F_min", 0.4)
            F_max = self.adaptive_params.get("F_max", 0.9)
            self.de_F = random.uniform(F_min, F_max)
            
        elif self.adaptive_method == AdaptiveMethod.JADE:
            # JADE method is handled in _get_current_F and _get_current_CR
            # We update the means of successful F and CR after selection
            pass
            
        elif self.adaptive_method == AdaptiveMethod.SaDE:
            # SaDE updates CRm based on success memory after the learning period
            if gen > self.adaptive_params["LP"] and len(self.adaptive_params["success_memory"]) > 0:
                self.adaptive_params["CRm"] = np.mean(self.adaptive_params["success_memory"])
                
        elif self.adaptive_method == AdaptiveMethod.SUCCESS_HISTORY:
            # Success history is updated during selection and used in _get_current_F and _get_current_CR
            pass

    def _get_current_F(self, i):
        """Get the current F value, possibly adapted"""
        if self.adaptive_method == AdaptiveMethod.NONE:
            return self.de_F
            
        elif self.adaptive_method == AdaptiveMethod.JITTER:
            # Add small random perturbation to F
            jitter_range = self.adaptive_params.get("jitter_range", 0.1)
            return max(0.1, min(0.9, self.de_F + random.uniform(-jitter_range, jitter_range)))
            
        elif self.adaptive_method == AdaptiveMethod.DITHER:
            # Dither sets a common F for all individuals in a generation
            return self.de_F
            
        elif self.adaptive_method == AdaptiveMethod.JADE:
            # JADE: generate F from Cauchy distribution with location parameter mu_F and scale 0.1
            F = np.random.standard_cauchy() * 0.1 + self.adaptive_params["mu_F"]
            return max(0.1, min(1.0, F))  # truncate to [0.1, 1.0]
            
        elif self.adaptive_method == AdaptiveMethod.SaDE:
            # SaDE uses fixed F (or could be randomized within a range)
            return self.de_F
            
        elif self.adaptive_method == AdaptiveMethod.SUCCESS_HISTORY:
            # Use history of successful F values if available
            history = self.adaptive_params["F_history"]
            if len(history) > 0:
                return random.choice(history)
            else:
                return self.de_F
                
        return self.de_F  # Default

    def _get_current_CR(self, i):
        """Get the current CR value, possibly adapted"""
        if self.adaptive_method == AdaptiveMethod.NONE:
            return self.de_CR
            
        elif self.adaptive_method == AdaptiveMethod.JADE:
            # JADE: generate CR from normal distribution with mean mu_CR and std 0.1
            CR = np.random.normal(self.adaptive_params["mu_CR"], 0.1)
            return max(0.0, min(1.0, CR))  # truncate to [0, 1]
            
        elif self.adaptive_method == AdaptiveMethod.SaDE:
            # SaDE: generate CR from normal distribution with mean CRm and std 0.1
            CR = np.random.normal(self.adaptive_params["CRm"], 0.1)
            return max(0.0, min(1.0, CR))  # truncate to [0, 1]
            
        elif self.adaptive_method == AdaptiveMethod.SUCCESS_HISTORY:
            # Use history of successful CR values if available
            history = self.adaptive_params["CR_history"]
            if len(history) > 0:
                return random.choice(history)
            else:
                return self.de_CR
                
        return self.de_CR  # Default

    def _calculate_diversity(self, population):
        """Calculate population diversity as the average Euclidean distance between individuals"""
        if len(population) <= 1:
            return 0.0
            
        # Convert to numpy array for easier calculation
        pop_array = np.array(population)
        n_individuals, n_dimensions = pop_array.shape
        
        # Calculate centroid
        centroid = np.mean(pop_array, axis=0)
        
        # Calculate average distance to centroid
        distances = np.sqrt(np.sum((pop_array - centroid)**2, axis=1))
        diversity = np.mean(distances)
        
        return diversity

    def _apply_diversity_preservation(self, population, fitnesses, parameter_bounds, fixed_parameters):
        """Apply diversity preservation techniques to maintain population diversity"""
        # Check if diversity is too low
        diversity = self._calculate_diversity(population)
        diversity_threshold = self.adaptive_params.get("diversity_threshold", 0.01)
        
        if diversity < diversity_threshold:
            # Get the best individuals (e.g., top 10%)
            num_best = max(1, int(0.1 * len(population)))
            indices = np.argsort(fitnesses)
            best_indices = indices[:num_best]
            
            # Keep the best individuals and reinitialize the rest
            new_population = []
            for i in range(len(population)):
                if i in best_indices:
                    new_population.append(population[i])
                else:
                    # Create a new random individual
                    individual = []
                    for j in range(len(population[0])):
                        low, high = parameter_bounds[j]
                        if j in fixed_parameters:
                            value = fixed_parameters[j]
                        else:
                            value = random.uniform(low, high)
                        individual.append(value)
                    new_population.append(individual)
            
            return new_population
        
        return population  # If diversity is acceptable, return unchanged

    def _check_termination(self, gen, best_fitness, no_improvement_count, diversity):
        """Check if any termination criteria are met"""
        # Check maximum generations
        if gen >= self.de_num_generations:
            return True
            
        # Check fitness tolerance
        if best_fitness <= self.de_tol:
            return True
            
        # Check for stagnation
        stagnation_limit = self.termination_criteria.get("stagnation_limit", self.de_num_generations // 4)
        if no_improvement_count >= stagnation_limit:
            return True
            
        # Check for minimum diversity
        min_diversity = self.termination_criteria.get("min_diversity", 1e-6)
        if diversity <= min_diversity:
            return True
            
        # Check for time limit
        # (This would require tracking elapsed time, which is not currently implemented)
        
        return False

    def _record_statistics(self, gen, population, fitnesses, best_individual, best_fitness, start_time, success_rate=None):
        """Record statistics for the current generation"""
        if not self.record_statistics:
            return
            
        # Convert population to numpy array for easier calculations
        pop_array = np.array(population)
        
        # Record basic statistics
        self.statistics.generations.append(gen)
        self.statistics.best_fitness_history.append(best_fitness)
        self.statistics.mean_fitness_history.append(np.mean(fitnesses))
        
        # Calculate and record diversity
        diversity = self._calculate_diversity(population)
        self.statistics.diversity_history.append(diversity)
        
        # Record parameter statistics
        if pop_array.size > 0:
            param_means = np.mean(pop_array, axis=0)
            param_stds = np.std(pop_array, axis=0)
            self.statistics.parameter_mean_history.append(param_means.tolist())
            self.statistics.parameter_std_history.append(param_stds.tolist())
        
        # Record success rate if provided
        if success_rate is not None:
            self.statistics.success_rates.append(success_rate)
        
        # Record execution time
        self.statistics.execution_times.append(time.time() - start_time)
        
        # Record current control parameters
        self.statistics.f_values.append(self.de_F)
        self.statistics.cr_values.append(self.de_CR)

    def _create_diagnostic_plots(self, parameter_names):
        """Create diagnostic plots from recorded statistics"""
        if not self.record_statistics:
            return
            
        try:
            # Create output directory if it doesn't exist
            os.makedirs('optimization_results', exist_ok=True)
            
            # Set style with a valid style name
            plt.style.use('seaborn-v0_8-darkgrid')  # For newer versions of matplotlib
            # If that fails, fallback to a default style
            try:
                plt.style.use('seaborn-v0_8-darkgrid')
            except:
                try:
                    plt.style.use('seaborn-darkgrid')  # For older versions
                except:
                    pass  # Default to matplotlib's default style
            
            # 1. Convergence plot
            plt.figure(figsize=(10, 6))
            plt.plot(self.statistics.generations, self.statistics.best_fitness_history, 'b-', label='Best Fitness')
            plt.plot(self.statistics.generations, self.statistics.mean_fitness_history, 'r--', label='Mean Fitness')
            plt.xlabel('Generation')
            plt.ylabel('Fitness Value')
            plt.title('Convergence History')
            plt.legend()
            plt.grid(True)
            plt.savefig('optimization_results/convergence_history.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Diversity plot
            plt.figure(figsize=(10, 6))
            plt.plot(self.statistics.generations, self.statistics.diversity_history, 'g-')
            plt.xlabel('Generation')
            plt.ylabel('Population Diversity')
            plt.title('Population Diversity History')
            plt.grid(True)
            plt.savefig('optimization_results/diversity_history.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Success rate plot if available
            if len(self.statistics.success_rates) > 0:
                plt.figure(figsize=(10, 6))
                plt.plot(self.statistics.generations[1:], self.statistics.success_rates, 'm-')
                plt.xlabel('Generation')
                plt.ylabel('Success Rate')
                plt.title('Mutation Success Rate History')
                plt.grid(True)
                plt.savefig('optimization_results/success_rate_history.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # 4. Control parameter adaptation if available
            if self.adaptive_method != AdaptiveMethod.NONE:
                plt.figure(figsize=(10, 6))
                plt.plot(self.statistics.generations, self.statistics.f_values, 'b-', label='F')
                plt.plot(self.statistics.generations, self.statistics.cr_values, 'r-', label='CR')
                plt.xlabel('Generation')
                plt.ylabel('Parameter Value')
                plt.title('Control Parameter Adaptation')
                plt.legend()
                plt.grid(True)
                plt.savefig('optimization_results/control_parameter_adaptation.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # 5. Parameter evolution
            if len(self.statistics.parameter_mean_history) > 0:
                param_means = np.array(self.statistics.parameter_mean_history)
                param_stds = np.array(self.statistics.parameter_std_history)
                
                for i in range(param_means.shape[1]):
                    plt.figure(figsize=(10, 6))
                    plt.plot(self.statistics.generations, param_means[:, i], 'b-')
                    plt.fill_between(
                        self.statistics.generations,
                        param_means[:, i] - param_stds[:, i],
                        param_means[:, i] + param_stds[:, i],
                        alpha=0.2
                    )
                    plt.xlabel('Generation')
                    plt.ylabel(f'Parameter Value: {parameter_names[i]}')
                    plt.title(f'Parameter Evolution: {parameter_names[i]}')
                    plt.grid(True)
                    plt.savefig(f'optimization_results/parameter_{i}_{parameter_names[i]}.png', dpi=300, bbox_inches='tight')
                    plt.close()
                
                # Create parameter correlation heatmap for final population
                if len(param_means) > 0:
                    final_params = param_means[-1]
                    num_params = len(final_params)
                    if num_params > 1:
                        corr_matrix = np.corrcoef(param_means.T)
                        plt.figure(figsize=(10, 8))
                        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                                   xticklabels=parameter_names, yticklabels=parameter_names)
                        plt.title('Parameter Correlation Heatmap')
                        plt.tight_layout()
                        plt.savefig('optimization_results/parameter_correlation.png', dpi=300, bbox_inches='tight')
                        plt.close()
            
            # 6. Save statistics to CSV
            stats_df = pd.DataFrame({
                'Generation': self.statistics.generations,
                'Best_Fitness': self.statistics.best_fitness_history,
                'Mean_Fitness': self.statistics.mean_fitness_history,
                'Diversity': self.statistics.diversity_history,
                'F_Value': self.statistics.f_values,
                'CR_Value': self.statistics.cr_values
            })
            
            if len(self.statistics.success_rates) > 0:
                # Add padding for first generation where success rate isn't calculated
                padded_success_rates = [np.nan] + self.statistics.success_rates
                if len(padded_success_rates) < len(self.statistics.generations):
                    padded_success_rates = padded_success_rates + [np.nan] * (len(self.statistics.generations) - len(padded_success_rates))
                stats_df['Success_Rate'] = padded_success_rates
                
            stats_df.to_csv('optimization_results/optimization_statistics.csv', index=False)
            
        except Exception as e:
            self.error.emit(f"Error creating diagnostic plots: {str(e)}")

    def perform_sensitivity_analysis(self, best_individual, parameter_names, n_samples=1000, plot_results=True):
        """
        Perform Sobol sensitivity analysis on the best solution to identify
        the most influential parameters
        
        Args:
            best_individual: The best individual (solution) found by DE
            parameter_names: List of parameter names
            n_samples: Number of samples for Sobol analysis
            plot_results: Whether to create sensitivity plots
            
        Returns:
            Dictionary with sensitivity indices
        """
        try:
            # Initialize parameter bounds for sensitivity analysis
            # Use a narrow range around the best solution (e.g., ±10%)
            param_bounds = []
            for i, val in enumerate(best_individual):
                lower = max(self.de_parameter_data[i][1], val * 0.9)
                upper = min(self.de_parameter_data[i][2], val * 1.1)
                param_bounds.append((lower, upper))
                
            # Perform Sobol analysis
            self.update.emit("[INFO] Running Sobol sensitivity analysis...")
            results = perform_sobol_analysis(
                evaluate_func=self.evaluate_individual,
                param_bounds=param_bounds,
                param_names=parameter_names,
                n_samples=n_samples
            )
            
            # Create sensitivity plots if requested
            if plot_results:
                self._create_sensitivity_plots(results, parameter_names)
                
            return results
            
        except Exception as e:
            self.error.emit(f"Error in sensitivity analysis: {str(e)}")
            return None

    def _create_sensitivity_plots(self, sensitivity_results, parameter_names):
        """Create plots visualizing sensitivity analysis results"""
        try:
            # Create output directory if it doesn't exist
            os.makedirs('sensitivity_results', exist_ok=True)
            
            # Extract first-order and total indices
            S1 = sensitivity_results['S1']
            ST = sensitivity_results['ST']
            
            # Create bar plot
            plt.figure(figsize=(12, 8))
            
            # Plot first-order indices
            bar_width = 0.35
            x_pos = np.arange(len(parameter_names))
            plt.bar(x_pos - bar_width/2, S1, bar_width, alpha=0.6, color='b', label='First Order')
            
            # Plot total indices
            plt.bar(x_pos + bar_width/2, ST, bar_width, alpha=0.6, color='r', label='Total Effect')
            
            plt.xlabel('Parameter')
            plt.ylabel('Sensitivity Index')
            plt.title('Parameter Sensitivity (Sobol Indices)')
            plt.xticks(x_pos, parameter_names, rotation=45)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            
            plt.savefig('sensitivity_results/sobol_indices.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create heatmap for parameter interactions
            if 'S2' in sensitivity_results and len(parameter_names) > 1:
                S2 = sensitivity_results['S2']
                
                # Create a matrix for the heatmap
                interaction_matrix = np.zeros((len(parameter_names), len(parameter_names)))
                for i in range(len(parameter_names)):
                    for j in range(len(parameter_names)):
                        if i != j and (i, j) in S2:
                            interaction_matrix[i, j] = S2[(i, j)]
                
                plt.figure(figsize=(10, 8))
                sns.heatmap(interaction_matrix, annot=True, fmt='.3f', cmap='viridis',
                           xticklabels=parameter_names, yticklabels=parameter_names)
                plt.title('Parameter Interaction Effects')
                plt.tight_layout()
                plt.savefig('sensitivity_results/parameter_interactions.png', dpi=300, bbox_inches='tight')
                plt.close()
                
            # Save results to CSV
            results_df = pd.DataFrame({
                'Parameter': parameter_names,
                'First_Order_Index': S1,
                'Total_Effect_Index': ST
            })
            results_df.to_csv('sensitivity_results/sensitivity_indices.csv', index=False)
            
        except Exception as e:
            self.error.emit(f"Error creating sensitivity plots: {str(e)}")

    @staticmethod
    def tune_hyperparameters(main_params, target_values_dict, weights_dict, omega_start, omega_end, omega_points, 
                           de_parameter_data, n_trials=20, parallel=True, n_processes=None):
        """
        Static method to find optimal DE hyperparameters using a grid search approach.
        This can be called separately before running the full optimization.
        
        Args:
            main_params: Main system parameters
            target_values_dict: Target values for optimization
            weights_dict: Weights for different objectives
            omega_start, omega_end, omega_points: Frequency range parameters
            de_parameter_data: Parameter data (name, lower bound, upper bound, fixed flag)
            n_trials: Number of tuning trials for each hyperparameter combination
            parallel: Whether to use parallel processing
            n_processes: Number of processes for parallel execution
            
        Returns:
            Best hyperparameter combination
        """
        # Define hyperparameter grid
        pop_sizes = [30, 50, 80]
        f_values = [0.3, 0.5, 0.7, 0.9]
        cr_values = [0.3, 0.5, 0.7, 0.9]
        strategies = [DEStrategy.RAND_1, DEStrategy.BEST_1, DEStrategy.CURRENT_TO_BEST_1]
        
        # Set up parallel processing if enabled
        if parallel:
            import multiprocessing as mp
            pool = mp.Pool(processes=n_processes or max(1, mp.cpu_count() - 1))
        
        best_combination = None
        best_fitness = float('inf')
        best_convergence_gen = float('inf')
        
        # Create all combinations of hyperparameters
        param_combinations = []
        for pop_size in pop_sizes:
            for f in f_values:
                for cr in cr_values:
                    for strategy in strategies:
                        param_combinations.append((pop_size, f, cr, strategy))
        
        print(f"Tuning DE hyperparameters with {len(param_combinations)} combinations...")
        
        # Function to evaluate a single hyperparameter combination
        def evaluate_params(params):
            pop_size, f, cr, strategy = params
            
            # Run multiple trials with this combination
            fitnesses = []
            convergence_gens = []
            
            for trial in range(n_trials):
                # Create a simplified DE worker for tuning
                de = DEWorker(
                    main_params=main_params,
                    target_values_dict=target_values_dict,
                    weights_dict=weights_dict,
                    omega_start=omega_start,
                    omega_end=omega_end,
                    omega_points=omega_points,
                    de_pop_size=pop_size,
                    de_num_generations=50,  # Limited generations for tuning
                    de_F=f,
                    de_CR=cr,
                    de_tol=1e-6,
                    de_parameter_data=de_parameter_data,
                    strategy=strategy,
                    record_statistics=True,
                    seed=trial  # Different seed for each trial
                )
                
                # Execute in current thread (no QThread for tuning)
                try:
                    # Extract parameter names, bounds, and fixed parameters
                    parameter_names = []
                    parameter_bounds = []
                    fixed_parameters = {}

                    for idx, (name, low, high, fixed) in enumerate(de.de_parameter_data):
                        parameter_names.append(name)
                        if fixed:
                            parameter_bounds.append((low, low))
                            fixed_parameters[idx] = low
                        else:
                            parameter_bounds.append((low, high))
                    num_params = len(parameter_bounds)
                    
                    # Initialize and evaluate population
                    population = de._initialize_population(parameter_bounds, fixed_parameters, num_params)
                    fitnesses_list = [de.evaluate_individual(ind) for ind in population]
                    
                    # Find initial best
                    best_idx = np.argmin(fitnesses_list)
                    global_best = population[best_idx].copy()
                    best_fitness_value = fitnesses_list[best_idx]
                    
                    # Simplified DE loop
                    converged_at = None
                    for gen in range(1, 51):  # Max 50 generations for tuning
                        new_population = []
                        new_fitnesses = []
                        
                        for i in range(pop_size):
                            target = population[i]
                            trial = de._apply_de_strategy(i, population, global_best, fitnesses_list, 
                                                       parameter_bounds, fixed_parameters, num_params)
                            trial_fitness = de.evaluate_individual(trial)
                            
                            if trial_fitness < fitnesses_list[i]:
                                new_population.append(trial)
                                new_fitnesses.append(trial_fitness)
                                if trial_fitness < best_fitness_value:
                                    global_best = trial.copy()
                                    best_fitness_value = trial_fitness
                            else:
                                new_population.append(target)
                                new_fitnesses.append(fitnesses_list[i])
                        
                        population = new_population
                        fitnesses_list = new_fitnesses
                        
                        # Check for convergence
                        if best_fitness_value <= 1e-6:
                            converged_at = gen
                            break
                    
                    fitnesses.append(best_fitness_value)
                    convergence_gens.append(converged_at if converged_at else 50)
                    
                except Exception as e:
                    print(f"Error in tuning trial: {e}")
                    fitnesses.append(float('inf'))
                    convergence_gens.append(50)
            
            # Calculate average performance across trials
            avg_fitness = np.mean(fitnesses)
            avg_convergence = np.mean(convergence_gens)
            
            return params, avg_fitness, avg_convergence
        
        # Run evaluations
        results = []
        if parallel:
            results = pool.map(evaluate_params, param_combinations)
            pool.close()
            pool.join()
        else:
            results = [evaluate_params(params) for params in param_combinations]
        
        # Find best combination
        for params, avg_fitness, avg_convergence in results:
            if avg_fitness < best_fitness or (avg_fitness == best_fitness and avg_convergence < best_convergence_gen):
                best_fitness = avg_fitness
                best_convergence_gen = avg_convergence
                best_combination = params
        
        pop_size, f, cr, strategy = best_combination
        print(f"Best hyperparameters found:")
        print(f"  Population Size: {pop_size}")
        print(f"  F (mutation factor): {f}")
        print(f"  CR (crossover rate): {cr}")
        print(f"  Strategy: {strategy.value}")
        print(f"  Average Fitness: {best_fitness}")
        print(f"  Average Convergence Generation: {best_convergence_gen}")
        
        return {
            "pop_size": pop_size,
            "F": f,
            "CR": cr,
            "strategy": strategy,
            "avg_fitness": best_fitness,
            "avg_convergence_gen": best_convergence_gen
        }

    @classmethod
    def restart_optimization(cls, previous_results, main_params, target_values_dict, weights_dict, 
                           omega_start, omega_end, omega_points, de_parameter_data, 
                           restart_options=None):
        """
        Restart optimization from a previous run with refined search space.
        
        Args:
            previous_results: Results from a previous optimization
            main_params, target_values_dict, etc.: Same parameters as in normal initialization
            restart_options: Dict with restart options (narrowing_factor, etc.)
            
        Returns:
            New DEWorker instance configured for restarting
        """
        # Default restart options
        options = {
            "narrowing_factor": 0.5,  # How much to narrow the search space around best solution
            "population_size_factor": 1.5,  # Increase population size for better exploration
            "include_previous_best": True,  # Include previous best solution in initial population
            "strategy": DEStrategy.CURRENT_TO_BEST_1  # Use different strategy for restart
        }
        
        # Update with user-provided options
        if restart_options:
            options.update(restart_options)
        
        # Extract best individual from previous results
        best_individual = previous_results.get("best_individual", None)
        if best_individual is None:
            raise ValueError("Previous results do not contain a best individual")
        
        # Create refined parameter bounds around the best solution
        refined_parameter_data = []
        for i, (name, low, high, fixed) in enumerate(de_parameter_data):
            if fixed:
                refined_parameter_data.append((name, low, high, True))
            else:
                # Calculate new bounds centered on best value
                best_val = best_individual[i]
                range_size = (high - low) * options["narrowing_factor"]
                new_low = max(low, best_val - range_size/2)
                new_high = min(high, best_val + range_size/2)
                refined_parameter_data.append((name, new_low, new_high, False))
        
        # Calculate new population size
        pop_size = int(len(previous_results.get("population", [])) * options["population_size_factor"])
        
        # Create new DEWorker with refined parameters
        de_worker = cls(
            main_params=main_params,
            target_values_dict=target_values_dict,
            weights_dict=weights_dict,
            omega_start=omega_start,
            omega_end=omega_end,
            omega_points=omega_points,
            de_pop_size=pop_size,
            de_num_generations=previous_results.get("de_num_generations", 100),
            de_F=previous_results.get("de_F", 0.5),
            de_CR=previous_results.get("de_CR", 0.7),
            de_tol=previous_results.get("de_tol", 1e-6) / 10,  # Tighter tolerance
            de_parameter_data=refined_parameter_data,
            alpha=previous_results.get("alpha", 0.01),
            beta=previous_results.get("beta", 0.0),
            strategy=options["strategy"],
            adaptive_method=previous_results.get("adaptive_method", AdaptiveMethod.NONE),
            adaptive_params=previous_results.get("adaptive_params", None),
            termination_criteria=previous_results.get("termination_criteria", None),
            use_parallel=previous_results.get("use_parallel", False),
            n_processes=previous_results.get("n_processes", None),
            record_statistics=True,
            constraint_handling=previous_results.get("constraint_handling", "penalty"),
            diversity_preservation=True
        )
        
        # Save previous best solution for initialization
        if options["include_previous_best"]:
            de_worker.previous_best = best_individual
        
        return de_worker
