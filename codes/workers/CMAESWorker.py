import sys
import numpy as np
import random
import cma  # Make sure the cma package is installed
from PyQt5.QtCore import QThread, pyqtSignal
from modules.FRF import frf
from modules.sobol_sensitivity import format_parameter_name

class CMAESWorker(QThread):
    # Emits: finished(final_results, best_candidate, parameter_names, best_fitness)
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
                 cma_initial_sigma,   # Scalar: initial standard deviation for search
                 cma_max_iter,        # Maximum number of iterations/generations
                 cma_tol,             # Tolerance to stop the search
                 cma_parameter_data,  # List of tuples: (name, lower bound, upper bound, fixed flag)
                 alpha=0.01):         # Sparsity penalty factor
        super().__init__()
        self.main_params = main_params
        self.target_values_dict = target_values_dict
        self.weights_dict = weights_dict
        self.omega_start = omega_start
        self.omega_end = omega_end
        self.omega_points = omega_points
        self.cma_initial_sigma = cma_initial_sigma
        self.cma_max_iter = cma_max_iter
        self.cma_tol = cma_tol
        self.cma_parameter_data = cma_parameter_data
        self.alpha = alpha

    def run(self):
        try:
            # Extract parameter names, bounds, and fixed parameters.
            parameter_names = []
            parameter_bounds = []
            fixed_parameters = {}  # key: index, value: fixed value
            for idx, (name, low, high, fixed) in enumerate(self.cma_parameter_data):
                parameter_names.append(name)
                if fixed:
                    parameter_bounds.append((low, low))
                    fixed_parameters[idx] = low
                else:
                    parameter_bounds.append((low, high))
            num_params = len(parameter_bounds)

            # Build initial candidate x0.
            x0 = []
            for j in range(num_params):
                low, high = parameter_bounds[j]
                if j in fixed_parameters:
                    x0.append(fixed_parameters[j])
                else:
                    x0.append(random.uniform(low, high))

            # Use provided cma_initial_sigma as the initial standard deviation.
            sigma0 = self.cma_initial_sigma

            # Build lower and upper bound arrays.
            lower_bounds = [lb for lb, ub in parameter_bounds]
            upper_bounds = [ub for lb, ub in parameter_bounds]

            # Define the objective function.
            def objective(x):
                # For dimensions with fixed values, force them.
                for idx, fixed_val in fixed_parameters.items():
                    x[idx] = fixed_val
                try:
                    results = frf(
                        main_system_parameters=self.main_params,
                        dva_parameters=tuple(x),
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
                        sparsity_penalty = self.alpha * sum(abs(xi) for xi in x)
                        return primary_objective + sparsity_penalty
                except Exception as e:
                    return 1e6

            # Set up options for CMA-ES, including bounds.
            options = {
                'bounds': [lower_bounds, upper_bounds],
                'maxiter': self.cma_max_iter,
                'verb_disp': 0,  # We handle our own logging
                'tolx': self.cma_tol
            }
            es = cma.CMAEvolutionStrategy(x0, sigma0, options)

            iter_count = 0
            best_fitness = float('inf')
            best_candidate = None

            # CMA-ES main loop.
            while not es.stop():
                iter_count += 1
                solutions = es.ask()
                fitnesses = [objective(x) for x in solutions]
                es.tell(solutions, fitnesses)
                current_best = min(fitnesses)
                if current_best < best_fitness:
                    best_fitness = current_best
                    best_candidate = solutions[fitnesses.index(current_best)]
                self.update.emit(f"Iteration {iter_count}: Best fitness = {best_fitness:.6f}")
                if best_fitness <= self.cma_tol:
                    self.update.emit(f"[INFO] Convergence reached at iteration {iter_count}")
                    break

            if best_candidate is None:
                best_candidate = es.result.xbest

            # Final evaluation using the best candidate.
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
