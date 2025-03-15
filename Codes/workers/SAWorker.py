import sys
import numpy as np
import random
from PyQt5.QtCore import QThread, pyqtSignal
from modules.FRF import frf
from modules.sobol_sensitivity import format_parameter_name

class SAWorker(QThread):
    # Signals: finished(final_results, best_candidate, parameter_names, best_fitness), error(str), update(str)
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
                 sa_initial_temp,      # Initial temperature
                 sa_cooling_rate,      # Cooling factor (e.g., 0.95)
                 sa_num_iterations,    # Maximum number of iterations
                 sa_tol,               # Tolerance to stop (if best fitness is below this value)
                 sa_parameter_data,    # List of tuples: (name, lower bound, upper bound, fixed flag)
                 alpha=0.01):          # Sparsity penalty factor
        super().__init__()
        self.main_params = main_params
        self.target_values_dict = target_values_dict
        self.weights_dict = weights_dict
        self.omega_start = omega_start
        self.omega_end = omega_end
        self.omega_points = omega_points
        self.sa_initial_temp = sa_initial_temp
        self.sa_cooling_rate = sa_cooling_rate
        self.sa_num_iterations = sa_num_iterations
        self.sa_tol = sa_tol
        self.sa_parameter_data = sa_parameter_data
        self.alpha = alpha

    def run(self):
        try:
            # Extract parameter names, bounds, and fixed parameters
            parameter_names = []
            parameter_bounds = []
            fixed_parameters = {}  # key: index, value: fixed value
            for idx, (name, low, high, fixed) in enumerate(self.sa_parameter_data):
                parameter_names.append(name)
                if fixed:
                    parameter_bounds.append((low, low))
                    fixed_parameters[idx] = low
                else:
                    parameter_bounds.append((low, high))
            num_params = len(parameter_bounds)

            # Initialize candidate solution (current state)
            current_candidate = []
            for j in range(num_params):
                low, high = parameter_bounds[j]
                if j in fixed_parameters:
                    current_candidate.append(fixed_parameters[j])
                else:
                    current_candidate.append(random.uniform(low, high))
            current_fitness = self.evaluate_candidate(current_candidate)
            best_candidate = current_candidate[:]
            best_fitness = current_fitness

            # Initialize temperature
            T = self.sa_initial_temp
            initial_temp = self.sa_initial_temp

            # Simulated Annealing main loop
            for iteration in range(1, self.sa_num_iterations + 1):
                # Generate a new candidate by perturbing the current candidate
                new_candidate = []
                for j in range(num_params):
                    if j in fixed_parameters:
                        new_candidate.append(fixed_parameters[j])
                    else:
                        low, high = parameter_bounds[j]
                        # Perturbation scale: 10% of range, scaled by current temperature relative to initial
                        base_scale = (high - low) * 0.1
                        perturbation = random.gauss(0, base_scale) * (T / initial_temp)
                        new_val = current_candidate[j] + perturbation
                        # Ensure new value remains within bounds
                        new_val = max(low, min(new_val, high))
                        new_candidate.append(new_val)
                new_fitness = self.evaluate_candidate(new_candidate)
                delta_fitness = new_fitness - current_fitness

                # Accept new candidate if it is better or with a probability if worse
                if delta_fitness < 0:
                    current_candidate = new_candidate
                    current_fitness = new_fitness
                else:
                    acceptance_probability = np.exp(-delta_fitness / T)
                    if random.random() < acceptance_probability:
                        current_candidate = new_candidate
                        current_fitness = new_fitness

                # Update the best candidate found so far
                if current_fitness < best_fitness:
                    best_candidate = current_candidate[:]
                    best_fitness = current_fitness

                self.update.emit(f"Iteration {iteration}: Current Fitness = {current_fitness:.6f}, "
                                 f"Best Fitness = {best_fitness:.6f}, Temperature = {T:.6f}")

                # Update temperature
                T = T * self.sa_cooling_rate

                # Check convergence criterion
                if best_fitness <= self.sa_tol:
                    self.update.emit(f"[INFO] Convergence reached at iteration {iteration}")
                    break

            # Final evaluation using best candidate
            try:
                final_results = frf(
                    main_system_parameters=self.main_params,
                    dva_parameters=tuple(best_candidate),
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

            self.finished.emit(final_results, best_candidate, parameter_names, best_fitness)

        except Exception as e:
            self.error.emit(str(e))

    def evaluate_candidate(self, candidate):
        """
        Evaluate the fitness of a candidate solution using the FRF function.
        The fitness is defined as the absolute difference between the singular response
        and 1 plus a sparsity penalty.
        """
        try:
            results = frf(
                main_system_parameters=self.main_params,
                dva_parameters=tuple(candidate),
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
                sparsity_penalty = self.alpha * sum(abs(x) for x in candidate)
                return primary_objective + sparsity_penalty
        except Exception as e:
            return 1e6
