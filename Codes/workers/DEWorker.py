import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import random

from PyQt5.QtCore import Qt, QThread, pyqtSignal

# Local imports (assuming these modules exist and are similar to the GAWorker case)
from modules.FRF import frf
from modules.sobol_sensitivity import (
    perform_sobol_analysis,
    calculate_and_save_errors,
    format_parameter_name
)

class DEWorker(QThread):
    # Signals: finished(dict, best_individual, parameter_names, best_fitness), error(str), update(str)
    finished = pyqtSignal(dict, list, list, float)
    error = pyqtSignal(str)
    update = pyqtSignal(str)

    def __init__(self, 
                 main_params,
                 target_values_dict,
                 weights_dict,
                 omega_start,
                 omega_end,
                 omega_points,
                 de_pop_size,
                 de_num_generations,
                 de_F,           # Mutation factor (typically 0.5-0.9)
                 de_CR,          # Crossover probability (typically 0.7-1.0)
                 de_tol,         # Tolerance for convergence
                 de_parameter_data,  # List of tuples: (name, lower bound, upper bound, fixed flag)
                 alpha=0.01):        # Sparsity penalty factor
        super().__init__()
        self.main_params = main_params
        self.target_values_dict = target_values_dict
        self.weights_dict = weights_dict
        self.omega_start = omega_start
        self.omega_end = omega_end
        self.omega_points = omega_points
        self.de_pop_size = de_pop_size
        self.de_num_generations = de_num_generations
        self.de_F = de_F
        self.de_CR = de_CR
        self.de_tol = de_tol
        self.de_parameter_data = de_parameter_data
        self.alpha = alpha

    def run(self):
        try:
            # Extract parameter names, bounds, and fixed parameters
            parameter_names = []
            parameter_bounds = []
            fixed_parameters = {}  # key: index, value: fixed value

            for idx, (name, low, high, fixed) in enumerate(self.de_parameter_data):
                parameter_names.append(name)
                if fixed:
                    parameter_bounds.append((low, low))
                    fixed_parameters[idx] = low
                else:
                    parameter_bounds.append((low, high))
            num_params = len(parameter_bounds)

            # Initialize population
            population = []
            fitnesses = []
            for i in range(self.de_pop_size):
                individual = []
                for j in range(num_params):
                    low, high = parameter_bounds[j]
                    if j in fixed_parameters:
                        value = fixed_parameters[j]
                    else:
                        value = random.uniform(low, high)
                    individual.append(value)
                fitness = self.evaluate_individual(individual)
                population.append(individual)
                fitnesses.append(fitness)

            # Identify global best
            best_idx = np.argmin(fitnesses)
            global_best = population[best_idx]
            best_fitness = fitnesses[best_idx]

            # DE main loop
            for gen in range(1, self.de_num_generations + 1):
                self.update.emit(f"-- Generation {gen} --")
                new_population = []
                new_fitnesses = []
                for i in range(self.de_pop_size):
                    target = population[i]

                    # Select three distinct individuals (r1, r2, r3) different from i
                    idxs = list(range(self.de_pop_size))
                    idxs.remove(i)
                    r1, r2, r3 = random.sample(idxs, 3)
                    x_r1 = population[r1]
                    x_r2 = population[r2]
                    x_r3 = population[r3]

                    # Mutation: create donor vector v = x_r1 + F*(x_r2 - x_r3)
                    donor = []
                    for j in range(num_params):
                        if j in fixed_parameters:
                            donor.append(fixed_parameters[j])
                        else:
                            mutated_val = x_r1[j] + self.de_F * (x_r2[j] - x_r3[j])
                            # Ensure donor value is within bounds
                            low, high = parameter_bounds[j]
                            donor.append(max(low, min(mutated_val, high)))
                    
                    # Crossover: create trial vector
                    trial = []
                    j_rand = random.randint(0, num_params - 1)
                    for j in range(num_params):
                        if j in fixed_parameters:
                            trial.append(fixed_parameters[j])
                        else:
                            if random.random() <= self.de_CR or j == j_rand:
                                trial.append(donor[j])
                            else:
                                trial.append(target[j])
                    
                    # Evaluate trial vector
                    trial_fitness = self.evaluate_individual(trial)
                    # Selection: if trial is better, it replaces target
                    if trial_fitness < fitnesses[i]:
                        new_population.append(trial)
                        new_fitnesses.append(trial_fitness)
                        # Update global best if necessary
                        if trial_fitness < best_fitness:
                            global_best = trial
                            best_fitness = trial_fitness
                    else:
                        new_population.append(target)
                        new_fitnesses.append(fitnesses[i])
                
                population = new_population
                fitnesses = new_fitnesses

                self.update.emit(f"  Best fitness at generation {gen}: {best_fitness:.6f}")
                if best_fitness <= self.de_tol:
                    self.update.emit(f"[INFO] Convergence reached at generation {gen}")
                    break

            # Run final FRF evaluation with the best candidate found
            best_individual = global_best
            try:
                final_results = frf(
                    main_system_parameters=self.main_params,
                    dva_parameters=tuple(best_individual),
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
            except Exception as e:
                final_results = {"Error": str(e)}

            self.finished.emit(final_results, best_individual, parameter_names, best_fitness)

        except Exception as e:
            self.error.emit(str(e))

    def evaluate_individual(self, individual):
        """
        Evaluate the fitness of an individual (candidate DVA parameters)
        using the FRF function. The fitness is defined as the absolute difference 
        between the singular response and 1 plus a sparsity penalty.
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
                return primary_objective + sparsity_penalty
        except Exception as e:
            return 1e6
