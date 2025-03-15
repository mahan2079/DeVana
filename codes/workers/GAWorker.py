import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QDoubleSpinBox, QSpinBox,
    QVBoxLayout, QHBoxLayout, QPushButton, QTabWidget, QFormLayout, QGroupBox,
    QTextEdit, QCheckBox, QScrollArea, QFileDialog, QMessageBox, QDockWidget,
    QMenuBar, QMenu, QAction, QSplitter, QToolBar, QStatusBar, QLineEdit, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QSizePolicy, QActionGroup
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QIcon, QPalette, QColor, QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from modules.FRF import frf
from modules.sobol_sensitivity import (
    perform_sobol_analysis,
    calculate_and_save_errors,
    format_parameter_name
)

import random
from deap import base, creator, tools


class GAWorker(QThread):
    finished = pyqtSignal(dict, list, list, float)  # results, best_ind, parameter_names, best_fitness
    error = pyqtSignal(str)
    update = pyqtSignal(str)

    def __init__(self, 
                 main_params,
                 target_values_dict,
                 weights_dict,
                 omega_start,
                 omega_end,
                 omega_points,
                 ga_pop_size,
                 ga_num_generations,
                 ga_cxpb,
                 ga_mutpb,
                 ga_tol,
                 ga_parameter_data,
                 alpha=0.01):
        super().__init__()
        self.main_params = main_params
        self.target_values_dict = target_values_dict
        self.weights_dict = weights_dict
        self.omega_start = omega_start
        self.omega_end = omega_end
        self.omega_points = omega_points
        self.ga_pop_size = ga_pop_size
        self.ga_num_generations = ga_num_generations
        self.ga_cxpb = ga_cxpb
        self.ga_mutpb = ga_mutpb
        self.ga_tol = ga_tol
        self.ga_parameter_data = ga_parameter_data  
        self.alpha = alpha  

    def run(self):
        try:
            # Extract parameter names, bounds, and fixed parameters
            parameter_names = []
            parameter_bounds = []
            fixed_parameters = {}

            for idx, (name, low, high, fixed) in enumerate(self.ga_parameter_data):
                parameter_names.append(name)
                if fixed:
                    parameter_bounds.append((low, low))
                    fixed_parameters[idx] = low  
                else:
                    parameter_bounds.append((low, high))

            # Setup DEAP framework
            if not hasattr(creator, "FitnessMin"):
                creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            if not hasattr(creator, "Individual"):
                creator.create("Individual", list, fitness=creator.FitnessMin)

            toolbox = base.Toolbox()

            # Attribute generator
            def attr_float(i):
                if i in fixed_parameters:
                    return fixed_parameters[i]
                else:
                    return random.uniform(parameter_bounds[i][0], parameter_bounds[i][1])

            toolbox.register("attr_float", attr_float, i=None)

            # Structure initializers
            toolbox.register("individual", tools.initIterate, creator.Individual,
                             lambda: [attr_float(i) for i in range(len(parameter_bounds))])
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            # Evaluation function with sparsity penalty
            def evaluate_individual(individual):
                dva_parameters_tuple = tuple(individual)
                try:
                    results = frf(
                        main_system_parameters=self.main_params,
                        dva_parameters=dva_parameters_tuple,
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
                        return (1e6,)
                    else:
                        primary_objective = abs(singular_response - 1)
                        # Sparsity penalty
                        sparsity_penalty = self.alpha * sum(abs(param) for param in individual)
                        fitness = primary_objective + sparsity_penalty
                        return (fitness,)
                except Exception as e:
                    return (1e6,)

            toolbox.register("evaluate", evaluate_individual)
            toolbox.register("mate", tools.cxBlend, alpha=0.5)

            # Mutation function
            def mutate_individual(individual, indpb=0.1):
                for i in range(len(individual)):
                    if i in fixed_parameters:
                        continue 
                    if random.random() < indpb:
                        min_val, max_val = parameter_bounds[i]
                        perturb = random.uniform(-0.1 * (max_val - min_val), 0.1 * (max_val - min_val))
                        individual[i] += perturb
                        individual[i] = max(min_val, min(individual[i], max_val))
                return (individual,)

            toolbox.register("mutate", mutate_individual)
            toolbox.register("select", tools.selTournament, tournsize=3)

            # Initialize population
            population = toolbox.population(n=self.ga_pop_size)
            fitnesses = list(map(toolbox.evaluate, population))
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

            # Evolution loop
            for gen in range(1, self.ga_num_generations + 1):
                self.update.emit(f"-- Generation {gen} --")

                # Selection
                offspring = toolbox.select(population, len(population))
                offspring = list(map(toolbox.clone, offspring))

                # Crossover
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < self.ga_cxpb:
                        toolbox.mate(child1, child2)
                        for child in [child1, child2]:
                            for i in range(len(child)):
                                if i in fixed_parameters:
                                    child[i] = fixed_parameters[i]
                                else:
                                    min_val, max_val = parameter_bounds[i]
                                    child[i] = max(min_val, min(child[i], max_val))
                        del child1.fitness.values
                        del child2.fitness.values

                # Mutation
                for mutant in offspring:
                    if random.random() < self.ga_mutpb:
                        toolbox.mutate(mutant)
                        del mutant.fitness.values

                # Evaluate invalid individuals
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                if invalid_ind:
                    self.update.emit(f"  Evaluating {len(invalid_ind)} individuals...")
                    fitnesses = map(toolbox.evaluate, invalid_ind)
                    for ind, fit in zip(invalid_ind, fitnesses):
                        ind.fitness.values = fit

                # Replace population
                population[:] = offspring

                # Gather all fitnesses
                fits = [ind.fitness.values[0] for ind in population]

                # Statistics
                length = len(population)
                mean = sum(fits) / length
                sum2 = sum(f ** 2 for f in fits)
                std = abs(sum2 / length - mean ** 2) ** 0.5

                min_fit = min(fits)
                max_fit = max(fits)

                self.update.emit(f"  Min fitness: {min_fit:.6f}")
                self.update.emit(f"  Max fitness: {max_fit:.6f}")
                self.update.emit(f"  Avg fitness: {mean:.6f}")
                self.update.emit(f"  Std fitness: {std:.6f}")

                # Check for convergence
                if min_fit <= self.ga_tol:
                    self.update.emit(f"\n[INFO] Solution found within tolerance at generation {gen}")
                    break

            # Select the best individual
            best_ind = tools.selBest(population, 1)[0]
            best_fitness = best_ind.fitness.values[0]

            dva_parameters_tuple = tuple(best_ind)
            try:
                final_results = frf(
                    main_system_parameters=self.main_params,
                    dva_parameters=dva_parameters_tuple,
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

            self.finished.emit(final_results, best_ind, parameter_names, best_fitness)

        except Exception as e:
            self.error.emit(str(e))