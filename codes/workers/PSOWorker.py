import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from PyQt5.QtCore import Qt, QThread, pyqtSignal
import random

# Local imports (assuming similar modules as in GAWorker)
from modules.FRF import frf
from modules.sobol_sensitivity import (
    perform_sobol_analysis,
    calculate_and_save_errors,
    format_parameter_name
)

class PSOWorker(QThread):
    # Signals: final_results, best_particle, parameter_names, best_fitness
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
                 pso_swarm_size,
                 pso_num_iterations,
                 pso_w,      # Inertia weight
                 pso_c1,     # Cognitive coefficient
                 pso_c2,     # Social coefficient
                 pso_tol,    # Tolerance for convergence
                 pso_parameter_data,  # List of tuples: (name, low, high, fixed)
                 alpha=0.01):         # Sparsity penalty factor
        super().__init__()
        self.main_params = main_params
        self.target_values_dict = target_values_dict
        self.weights_dict = weights_dict
        self.omega_start = omega_start
        self.omega_end = omega_end
        self.omega_points = omega_points
        self.pso_swarm_size = pso_swarm_size
        self.pso_num_iterations = pso_num_iterations
        self.pso_w = pso_w
        self.pso_c1 = pso_c1
        self.pso_c2 = pso_c2
        self.pso_tol = pso_tol
        self.pso_parameter_data = pso_parameter_data
        self.alpha = alpha

    def run(self):
        try:
            # Extract parameter names, bounds, and fixed parameters
            parameter_names = []
            parameter_bounds = []
            fixed_parameters = {}  # key: index, value: fixed value
            for idx, (name, low, high, fixed) in enumerate(self.pso_parameter_data):
                parameter_names.append(name)
                if fixed:
                    parameter_bounds.append((low, low))
                    fixed_parameters[idx] = low
                else:
                    parameter_bounds.append((low, high))
            num_params = len(parameter_bounds)

            # Initialize the swarm: Each particle is a dict with its current position, velocity,
            # personal best position, and personal best fitness.
            swarm = []
            for i in range(self.pso_swarm_size):
                position = []
                velocity = []
                for j in range(num_params):
                    low, high = parameter_bounds[j]
                    if j in fixed_parameters:
                        pos = fixed_parameters[j]
                    else:
                        pos = random.uniform(low, high)
                    position.append(pos)
                    # Initialize velocity with a random value relative to the bound range
                    delta = high - low
                    vel = random.uniform(-delta, delta) if delta != 0 else 0
                    velocity.append(vel)
                fitness = self.evaluate_particle(position)
                particle = {
                    'position': position,
                    'velocity': velocity,
                    'best_position': position[:],
                    'best_fitness': fitness
                }
                swarm.append(particle)

            # Determine the global best particle in the initial swarm.
            global_best = None
            for particle in swarm:
                if global_best is None or particle['best_fitness'] < global_best['best_fitness']:
                    global_best = particle

            # PSO main loop: update each particle's velocity and position.
            for iteration in range(1, self.pso_num_iterations + 1):
                self.update.emit(f"-- Iteration {iteration} --")
                for particle in swarm:
                    for j in range(num_params):
                        # For fixed parameters, ensure they remain constant.
                        if j in fixed_parameters:
                            particle['velocity'][j] = 0
                            particle['position'][j] = fixed_parameters[j]
                        else:
                            r1 = random.random()
                            r2 = random.random()
                            cognitive = self.pso_c1 * r1 * (particle['best_position'][j] - particle['position'][j])
                            social = self.pso_c2 * r2 * (global_best['best_position'][j] - particle['position'][j])
                            particle['velocity'][j] = (self.pso_w * particle['velocity'][j] + cognitive + social)
                            particle['position'][j] += particle['velocity'][j]
                            # Enforce bounds on the position.
                            low, high = parameter_bounds[j]
                            if particle['position'][j] < low:
                                particle['position'][j] = low
                                particle['velocity'][j] = 0
                            elif particle['position'][j] > high:
                                particle['position'][j] = high
                                particle['velocity'][j] = 0
                    # Evaluate the new position.
                    fitness = self.evaluate_particle(particle['position'])
                    if fitness < particle['best_fitness']:
                        particle['best_position'] = particle['position'][:]
                        particle['best_fitness'] = fitness
                        if fitness < global_best['best_fitness']:
                            global_best = particle

                self.update.emit(f"  Global best fitness at iteration {iteration}: {global_best['best_fitness']:.6f}")
                # Check if the solution meets the tolerance criteria.
                if global_best['best_fitness'] <= self.pso_tol:
                    self.update.emit(f"[INFO] Convergence reached at iteration {iteration}")
                    break

            best_particle = global_best['best_position']
            best_fitness = global_best['best_fitness']

            # Run the final FRF evaluation using the best found parameters.
            try:
                final_results = frf(
                    main_system_parameters=self.main_params,
                    dva_parameters=tuple(best_particle),
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

            self.finished.emit(final_results, best_particle, parameter_names, best_fitness)

        except Exception as e:
            self.error.emit(str(e))

    def evaluate_particle(self, position):
        """
        Evaluate the fitness of a particle based on its position.
        The fitness is defined as the absolute difference between the singular response
        from the FRF function and 1, plus a sparsity penalty.
        """
        try:
            results = frf(
                main_system_parameters=self.main_params,
                dva_parameters=tuple(position),
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
                sparsity_penalty = self.alpha * sum(abs(x) for x in position)
                fitness = primary_objective + sparsity_penalty
                return fitness
        except Exception as e:
            return 1e6
