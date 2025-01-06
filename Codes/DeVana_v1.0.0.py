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

from FRF import frf
from sobol_sensitivity import (
    perform_sobol_analysis,
    calculate_and_save_errors,
    format_parameter_name
)

sns.set(style="whitegrid")
plt.rc('text', usetex=True)

import random
from deap import base, creator, tools


############################################
# PlotWindow Class
############################################

class PlotWindow(QMainWindow):
    def __init__(self, fig, title="Plot"):
        super().__init__()
        self.setWindowTitle(title)
        self.setWindowIcon(QIcon.fromTheme("applications-graphics"))
        self.fig = fig
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)


############################################
# GA-Related Worker Classes
############################################

class FRFWorker(QThread):
    finished = pyqtSignal(dict, dict)  # Emitting two dicts: with DVA and without DVA
    error = pyqtSignal(str)
    
    def __init__(self, main_params, dva_params, omega_start, omega_end, omega_points,
                 target_values_dict, weights_dict, plot_figure, show_peaks, show_slopes):
        super().__init__()
        self.main_params = main_params
        self.dva_params = dva_params
        self.omega_start = omega_start
        self.omega_end = omega_end
        self.omega_points = omega_points
        self.target_values_dict = target_values_dict
        self.weights_dict = weights_dict
        self.plot_figure = plot_figure
        self.show_peaks = show_peaks
        self.show_slopes = show_slopes

    def run(self):
        try:
            # First FRF call: With DVAs
            results_with_dva = frf(
                main_system_parameters=self.main_params,
                dva_parameters=self.dva_params,
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
                plot_figure=False,    # Avoid plotting in worker
                show_peaks=self.show_peaks,
                show_slopes=self.show_slopes
            )
            
            # Second FRF call: Without DVAs for Mass 1 and Mass 2
            # Assuming Mass 1 and Mass 2 are main masses and their DVA parameters are not directly influencing them
            # Therefore, to remove DVAs, set all DVA parameters to zero
            dva_params_zero = list(self.dva_params)
            for i in range(len(dva_params_zero)):
                dva_params_zero[i] = 0.0
            results_without_dva = frf(
                main_system_parameters=self.main_params,
                dva_parameters=tuple(dva_params_zero),
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
                plot_figure=False,    # Avoid plotting in worker
                show_peaks=self.show_peaks,
                show_slopes=self.show_slopes
            )
            
            self.finished.emit(results_with_dva, results_without_dva)
        except Exception as e:
            self.error.emit(str(e))


class SobolWorker(QThread):
    finished = pyqtSignal(dict, list)
    error = pyqtSignal(str)
    
    def __init__(self, main_params, dva_bounds, dva_order,
                 omega_start, omega_end, omega_points, num_samples_list,
                 target_values_dict, weights_dict, n_jobs):
        super().__init__()
        self.main_params = main_params
        self.dva_bounds = dva_bounds
        self.dva_order = dva_order
        self.omega_start = omega_start
        self.omega_end = omega_end
        self.omega_points = omega_points
        self.num_samples_list = num_samples_list
        self.target_values_dict = target_values_dict
        self.weights_dict = weights_dict
        self.n_jobs = n_jobs

    def run(self):
        try:
            all_results, warnings = perform_sobol_analysis(
                main_system_parameters=self.main_params,
                dva_parameters_bounds=self.dva_bounds,
                dva_parameter_order=self.dva_order,
                omega_start=self.omega_start,
                omega_end=self.omega_end,
                omega_points=self.omega_points,
                num_samples_list=self.num_samples_list,
                target_values_dict=self.target_values_dict,
                weights_dict=self.weights_dict,
                visualize=False,  
                n_jobs=self.n_jobs
            )
            self.finished.emit(all_results, warnings)
        except Exception as e:
            self.error.emit(str(e))


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


############################################
# Main Window Class
############################################

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("VIBRAOPT - FRF, Sobol & GA Analysis")
        self.resize(1600, 900)

        # Initialize theme
        self.current_theme = 'Light'
        self.apply_light_theme()

        central_widget = QWidget()
        central_layout = QVBoxLayout(central_widget)
        self.setCentralWidget(central_widget)

        self.tabs = QTabWidget()
        self.create_main_system_tab()
        self.create_dva_parameters_tab()
        self.create_target_weights_tab()
        self.create_frequency_tab()
        self.create_sobol_analysis_tab()
        self.create_ga_tab()

        self.tabs.addTab(self.main_system_tab, "Main System")
        self.tabs.addTab(self.dva_tab, "DVA Parameters")
        self.tabs.addTab(self.tw_tab, "Targets & Weights")
        self.tabs.addTab(self.freq_tab, "Frequency & Plot")
        self.tabs.addTab(self.sobol_tab, "Sobol Analysis")
        self.tabs.addTab(self.ga_tab, "GA Optimization")

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setMinimumHeight(200)

        # FRF Dock
        self.frf_fig = Figure(figsize=(6,4))
        self.frf_canvas = FigureCanvas(self.frf_fig)
        self.frf_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.frf_combo = QComboBox()
        self.frf_combo.currentIndexChanged.connect(self.update_frf_plot)

        frf_widget = QWidget()
        frf_layout = QVBoxLayout(frf_widget)
        frf_layout.addWidget(QLabel("Select FRF Plot:"))
        frf_layout.addWidget(self.frf_combo)
        self.frf_toolbar = NavigationToolbar(self.frf_canvas, frf_widget)
        frf_layout.addWidget(self.frf_toolbar)
        frf_layout.addWidget(self.frf_canvas)
        self.frf_save_plot_button = QPushButton("Save FRF Plot")
        self.frf_save_plot_button.clicked.connect(lambda: self.save_plot(self.frf_fig, "FRF"))
        frf_layout.addWidget(self.frf_save_plot_button)
        frf_layout.setStretchFactor(self.frf_canvas, 1)
        frf_widget.setLayout(frf_layout)

        self.frf_dock = QDockWidget("FRF Plots", self)
        self.frf_dock.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.frf_dock.setWidget(frf_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, self.frf_dock)
        self.add_maximize_action(self.frf_dock, self.frf_canvas, is_sobol=False)

        # Sobol Dock
        self.sobol_fig = Figure(figsize=(6,4))
        self.sobol_canvas = FigureCanvas(self.sobol_fig)
        self.sobol_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.sobol_combo = QComboBox()
        self.sobol_combo.currentIndexChanged.connect(self.update_sobol_plot)

        sobol_widget = QWidget()
        sobol_layout = QVBoxLayout(sobol_widget)
        sobol_layout.addWidget(QLabel("Select Sobol Plot:"))
        sobol_layout.addWidget(self.sobol_combo)
        self.sobol_results_text = QTextEdit()
        self.sobol_results_text.setReadOnly(True)
        sobol_layout.addWidget(QLabel("Sobol Analysis Results:"))
        sobol_layout.addWidget(self.sobol_results_text)
        self.sobol_save_results_button = QPushButton("Save Sobol Results")
        self.sobol_save_results_button.clicked.connect(self.save_sobol_results)
        sobol_layout.addWidget(self.sobol_save_results_button)
        self.sobol_toolbar = NavigationToolbar(self.sobol_canvas, sobol_widget)
        sobol_layout.addWidget(self.sobol_toolbar)
        sobol_layout.addWidget(self.sobol_canvas)
        self.sobol_save_plot_button = QPushButton("Save Sobol Plot")
        self.sobol_save_plot_button.clicked.connect(lambda: self.save_plot(self.sobol_fig, "Sobol"))
        sobol_layout.addWidget(self.sobol_save_plot_button)
        sobol_layout.setStretchFactor(self.sobol_canvas, 1)
        sobol_widget.setLayout(sobol_layout)

        self.sobol_dock = QDockWidget("Sobol Plots \& Results", self)
        self.sobol_dock.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.sobol_dock.setWidget(sobol_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.sobol_dock)
        self.add_maximize_action(self.sobol_dock, self.sobol_canvas, is_sobol=True)

        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.tabs)
        splitter.addWidget(self.results_text)
        splitter.setStretchFactor(0,5)
        splitter.setStretchFactor(1,1)

        central_layout.addWidget(splitter)

        self.create_menubar()
        self.create_toolbar()
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        run_layout = QHBoxLayout()
        run_layout.addStretch()
        self.run_frf_button = QPushButton("Run FRF")
        self.run_frf_button.setFixedWidth(150)
        self.run_frf_button.clicked.connect(self.run_frf)
        run_layout.addWidget(self.run_frf_button)

        self.run_sobol_button = QPushButton("Run Sobol")
        self.run_sobol_button.setFixedWidth(150)
        self.run_sobol_button.clicked.connect(self.run_sobol)
        run_layout.addWidget(self.run_sobol_button)

        self.run_ga_button = QPushButton("Run GA")
        self.run_ga_button.setFixedWidth(150)
        self.run_ga_button.clicked.connect(self.run_ga)
        run_layout.addWidget(self.run_ga_button)

        central_layout.addLayout(run_layout)

        self.frf_results = None
        self.sobol_results = None
        self.sobol_warnings = None
        self.frf_plots = {}
        self.sobol_plots = {}
        self.ga_plots = {}
        self.ga_results = None

        self.set_default_values()

        # Redraw canvases to ensure toolbar interactivity
        self.frf_canvas.draw()
        self.sobol_canvas.draw()

    ##################################
    # MainWindow Setup Methods
    ##################################
    
    def add_maximize_action(self, dock, canvas, is_sobol=False):
        toolbar = QToolBar("Plot Controls", dock)
        dock_layout = dock.widget().layout()
        dock_layout.insertWidget(0, toolbar)

        maximize_action = QAction("Maximize", dock)
        maximize_action.setIcon(QIcon.fromTheme("zoom-in"))
        if is_sobol:
            # For Sobol Dock, maximize only the plot by opening it in a new window
            maximize_action.triggered.connect(lambda: self.maximize_plot(dock, canvas, is_sobol=True))
        else:
            # For other docks, retain existing behavior
            maximize_action.triggered.connect(lambda: self.maximize_dock(dock))
        toolbar.addAction(maximize_action)

    def maximize_plot(self, dock, canvas, is_sobol=False):
        current_plot_key = ""
        if is_sobol:
            current_plot_key = self.sobol_combo.currentText()
            fig = self.sobol_plots.get(current_plot_key, None)
            title = f"Sobol Plot: {current_plot_key}"
        else:
            current_plot_key = self.frf_combo.currentText()
            fig = self.frf_plots.get(current_plot_key, None)
            title = f"FRF Plot: {current_plot_key}"

        if fig:
            self.plot_window = PlotWindow(fig, title=title)
            self.plot_window.show()
        else:
            QMessageBox.warning(self, "Maximize Plot", "No plot available to maximize.")

    def maximize_dock(self, dock):
        if not dock.isFloating():
            dock.setFloating(True)
        screen_geometry = QApplication.desktop().screenGeometry()
        dock.resize(screen_geometry.width(), screen_geometry.height())
        dock.show()

    def create_menubar(self):
        menubar = QMenuBar(self)
        file_menu = menubar.addMenu("File")
        view_menu = menubar.addMenu("View")
        theme_menu = menubar.addMenu("Theme")
        run_menu = menubar.addMenu("Run")
        help_menu = menubar.addMenu("Help")

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        self.show_frf_dock_act = QAction("Show FRF Dock", self, checkable=True, checked=True)
        self.show_frf_dock_act.triggered.connect(lambda checked: self.frf_dock.setVisible(checked))
        view_menu.addAction(self.show_frf_dock_act)

        self.show_sobol_dock_act = QAction("Show Sobol Dock", self, checkable=True, checked=True)
        self.show_sobol_dock_act.triggered.connect(lambda checked: self.sobol_dock.setVisible(checked))
        view_menu.addAction(self.show_sobol_dock_act)

        light_theme_act = QAction("Light Theme", self, checkable=True, checked=True)
        dark_theme_act = QAction("Dark Theme", self, checkable=True)
        light_theme_act.triggered.connect(lambda: self.switch_theme('Light'))
        dark_theme_act.triggered.connect(lambda: self.switch_theme('Dark'))
        theme_menu.addAction(light_theme_act)
        theme_menu.addAction(dark_theme_act)
        theme_group = QActionGroup(self)
        theme_group.addAction(light_theme_act)
        theme_group.addAction(dark_theme_act)

        run_frf_act = QAction("Run FRF Analysis", self)
        run_frf_act.triggered.connect(self.run_frf)
        run_sobol_act = QAction("Run Sobol Analysis", self)
        run_sobol_act.triggered.connect(self.run_sobol)
        run_ga_act = QAction("Run GA Optimization", self)
        run_ga_act.triggered.connect(self.run_ga)
        run_menu.addAction(run_frf_act)
        run_menu.addAction(run_sobol_act)
        run_menu.addAction(run_ga_act)

        about_act = QAction("About", self)
        about_act.triggered.connect(lambda: QMessageBox.information(self, "About", "VIBRAOPT Program\nVersion 1.0"))
        help_menu.addAction(about_act)

        self.setMenuBar(menubar)

    def create_toolbar(self):
        toolbar = QToolBar("Main Toolbar", self)
        toolbar.setMovable(True)
        run_frf_act = QAction(QIcon.fromTheme("system-run"), "Run FRF", self)
        run_frf_act.triggered.connect(self.run_frf)
        run_sobol_act = QAction(QIcon.fromTheme("system-run"), "Run Sobol", self)
        run_sobol_act.triggered.connect(self.run_sobol)
        run_ga_act = QAction(QIcon.fromTheme("system-run"), "Run GA", self)
        run_ga_act.triggered.connect(self.run_ga)
        toolbar.addAction(run_frf_act)
        toolbar.addAction(run_sobol_act)
        toolbar.addAction(run_ga_act)
        self.addToolBar(Qt.TopToolBarArea, toolbar)

    def switch_theme(self, theme):
        if theme == 'Dark':
            self.apply_dark_theme()
            self.current_theme = 'Dark'
        else:
            self.apply_light_theme()
            self.current_theme = 'Light'

    def apply_dark_theme(self):
        dark_palette = QPalette()
        dark_palette.setColor(QPalette.Window, QColor(53,53,53))
        dark_palette.setColor(QPalette.WindowText, Qt.white)
        dark_palette.setColor(QPalette.Base, QColor(25,25,25))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53,53,53))
        dark_palette.setColor(QPalette.ToolTipBase, Qt.white)
        dark_palette.setColor(QPalette.ToolTipText, Qt.white)
        dark_palette.setColor(QPalette.Text, Qt.white)
        dark_palette.setColor(QPalette.Button, QColor(53,53,53))
        dark_palette.setColor(QPalette.ButtonText, Qt.white)
        dark_palette.setColor(QPalette.BrightText, Qt.red)
        dark_palette.setColor(QPalette.Highlight, QColor(142,45,197).lighter())
        dark_palette.setColor(QPalette.HighlightedText, Qt.black)
        QApplication.instance().setPalette(dark_palette)

        dark_stylesheet = """
            QWidget {
                background-color: #1E1E1E;
                color: #FFFFFF;
                font-family: "Roboto", "Segoe UI", "Helvetica", sans-serif;
                font-size: 10pt;
            }
            QTabWidget::pane {
                border: 1px solid #444;
                background-color: #1E1E1E;
            }
            QTabBar::tab {
                background: #555;
                color: white;
                padding: 10px;
                border: 1px solid #444;
                border-bottom: none;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                margin-right: 2px;
            }
            QTabBar::tab:selected, QTabBar::tab:hover {
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0,
                    stop:0 #6A0DAD, stop:1 #FF8C00);
                color: #FFFFFF;
            }
            QGroupBox {
                border: 2px solid #444;
                border-radius: 10px;
                margin-top: 15px;
                background-color: #2E2E2E;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
                color: #FF8C00;
                font-weight: bold;
                font-size: 12pt;
            }
            QPushButton {
                background-color: #1E90FF;
                border: none;
                color: white;
                padding: 10px 20px;
                font-size: 11pt;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #63B8FF;
            }
            QPushButton:pressed {
                background-color: #104E8B;
            }
            QDoubleSpinBox, QSpinBox {
                background-color: #2E2E2E;
                border: 1px solid #555555;
                border-radius: 5px;
                padding: 5px;
                color: #FFFFFF;
                height: 25px;
            }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button,
            QSpinBox::up-button, QSpinBox::down-button {
                background-color: #444444;
                border: none;
                width: 16px;
                height: 16px;
                border-radius: 3px;
            }
            QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover,
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background-color: #555555;
            }
            QTextEdit {
                background-color: #2E2E2E;
                border: 1px solid #555555;
                border-radius: 5px;
                color: #FFFFFF;
                padding: 10px;
                font-family: "Consolas", monospace;
                font-size: 10pt;
            }
            QLabel {
                color: #FFFFFF;
                font-weight: normal;
                font-size: 10pt;
            }
            QCheckBox {
                spacing: 5px;
                font-size: 10pt;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QCheckBox::indicator:unchecked {
                background-color: #3C3C3C;
                border: 1px solid #555555;
                border-radius: 3px;
            }
            QCheckBox::indicator:checked {
                background-color: #FF8C00;
                border: 1px solid #555555;
                border-radius: 3px;
            }
            QCheckBox::indicator:hover {
                border: 1px solid #1E90FF;
            }
            QScrollArea {
                border: none;
            }
            QSpacerItem {
                height: 20px;
            }
            QGroupBox:hover {
                border: 2px solid #FF8C00;
            }
            QLabel#GroupBoxTitle {
                color: #FF8C00;
                font-size: 12pt;
                font-weight: bold;
            }
        """
        QApplication.instance().setStyleSheet(dark_stylesheet)

    def apply_light_theme(self):
        light_palette = QPalette()
        light_palette.setColor(QPalette.Window, Qt.white)
        light_palette.setColor(QPalette.WindowText, Qt.black)
        light_palette.setColor(QPalette.Base, QColor(240,240,240))
        light_palette.setColor(QPalette.AlternateBase, QColor(225,225,225))
        light_palette.setColor(QPalette.ToolTipBase, Qt.white)
        light_palette.setColor(QPalette.ToolTipText, Qt.black)
        light_palette.setColor(QPalette.Text, Qt.black)
        light_palette.setColor(QPalette.Button, QColor(240,240,240))
        light_palette.setColor(QPalette.ButtonText, Qt.black)
        light_palette.setColor(QPalette.BrightText, Qt.red)
        light_palette.setColor(QPalette.Highlight, QColor(0,120,215))
        light_palette.setColor(QPalette.HighlightedText, Qt.white)
        QApplication.instance().setPalette(light_palette)

        light_stylesheet = """
            QWidget {
                background-color: #F0F0F0;
                color: #000000;
                font-family: "Roboto", "Segoe UI", "Helvetica", sans-serif;
                font-size: 10pt;
            }
            QTabWidget::pane {
                border: 1px solid #CCCCCC;
                background-color: #F0F0F0;
            }
            QTabBar::tab {
                background: #E0E0E0;
                color: #000000;
                padding: 10px;
                border: 1px solid #CCCCCC;
                border-bottom: none;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
                margin-right: 2px;
            }
            QTabBar::tab:selected, QTabBar::tab:hover {
                background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0,
                    stop:0 #6A0DAD, stop:1 #FF8C00);
                color: #FFFFFF;
            }
            QGroupBox {
                border: 2px solid #CCCCCC;
                border-radius: 10px;
                margin-top: 15px;
                background-color: #FFFFFF;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
                color: #6A0DAD;
                font-weight: bold;
                font-size: 12pt;
            }
            QPushButton {
                background-color: #1E90FF;
                border: none;
                color: white;
                padding: 10px 20px;
                font-size: 11pt;
                border-radius: 8px;
            }
            QPushButton:hover {
                background-color: #63B8FF;
            }
            QPushButton:pressed {
                background-color: #104E8B;
            }
            QDoubleSpinBox, QSpinBox {
                background-color: #FFFFFF;
                border: 1px solid #CCCCCC;
                border-radius: 5px;
                padding: 5px;
                color: #000000;
                height: 25px;
            }
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button,
            QSpinBox::up-button, QSpinBox::down-button {
                background-color: #E0E0E0;
                border: none;
                width: 16px;
                height: 16px;
                border-radius: 3px;
            }
            QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover,
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background-color: #CCCCCC;
            }
            QTextEdit {
                background-color: #FFFFFF;
                border: 1px solid #CCCCCC;
                border-radius: 5px;
                color: #000000;
                padding: 10px;
                font-family: "Consolas", monospace;
                font-size: 10pt;
            }
            QLabel {
                color: #000000;
                font-weight: normal;
                font-size: 10pt;
            }
            QCheckBox {
                spacing: 5px;
                font-size: 10pt;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QCheckBox::indicator:unchecked {
                background-color: #E0E0E0;
                border: 1px solid #CCCCCC;
                border-radius: 3px;
            }
            QCheckBox::indicator:checked {
                background-color: #FF8C00;
                border: 1px solid #CCCCCC;
                border-radius: 3px;
            }
            QCheckBox::indicator:hover {
                border: 1px solid #6A0DAD;
            }
            QScrollArea {
                border: none;
            }
            QSpacerItem {
                height: 20px;
            }
            QGroupBox:hover {
                border: 2px solid #FF8C00;
            }
            QLabel#GroupBoxTitle {
                color: #FF8C00;
                font-size: 12pt;
                font-weight: bold;
            }
        """
        QApplication.instance().setStyleSheet(light_stylesheet)

    ##################################
    # Tab Creation
    ##################################

    def create_main_system_tab(self):
        self.main_system_tab = QWidget()
        layout = QVBoxLayout(self.main_system_tab)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        sc_widget = QWidget()
        sc_layout = QVBoxLayout(sc_widget)

        group = QGroupBox("Main System Parameters")
        form = QFormLayout(group)

        self.mu_box = QDoubleSpinBox(); self.mu_box.setRange(-1e6,1e6); self.mu_box.setDecimals(6)
        self.landa_boxes = []
        for i in range(5):
            box = QDoubleSpinBox(); box.setRange(-1e6,1e6); box.setDecimals(6)
            self.landa_boxes.append(box)
        self.nu_boxes = []
        for i in range(5):
            box = QDoubleSpinBox(); box.setRange(-1e6,1e6); box.setDecimals(6)
            self.nu_boxes.append(box)

        self.a_low_box = QDoubleSpinBox(); self.a_low_box.setRange(0,1e10); self.a_low_box.setDecimals(6)
        self.a_up_box = QDoubleSpinBox(); self.a_up_box.setRange(0,1e10); self.a_up_box.setDecimals(6)
        self.f_1_box = QDoubleSpinBox(); self.f_1_box.setRange(0,1e10); self.f_1_box.setDecimals(6)
        self.f_2_box = QDoubleSpinBox(); self.f_2_box.setRange(0,1e10); self.f_2_box.setDecimals(6)
        self.omega_dc_box = QDoubleSpinBox(); self.omega_dc_box.setRange(0,1e10); self.omega_dc_box.setDecimals(6)
        self.zeta_dc_box = QDoubleSpinBox(); self.zeta_dc_box.setRange(0,1e10); self.zeta_dc_box.setDecimals(6)

        form.addRow("μ (MU):", self.mu_box)
        for i in range(5):
            form.addRow(f"Λ_{i+1}:", self.landa_boxes[i])
        for i in range(5):
            form.addRow(f"Ν_{i+1}:", self.nu_boxes[i])
        form.addRow("A_LOW:", self.a_low_box)
        form.addRow("A_UPP:", self.a_up_box)
        form.addRow("F_1:", self.f_1_box)
        form.addRow("F_2:", self.f_2_box)
        form.addRow("Ω_DC:", self.omega_dc_box)
        form.addRow("ζ_DC:", self.zeta_dc_box)

        sc_layout.addWidget(group)
        sc_layout.addStretch()
        sc_widget.setLayout(sc_layout)
        scroll.setWidget(sc_widget)
        layout.addWidget(scroll)

    def create_dva_parameters_tab(self):
        self.dva_tab = QWidget()
        layout = QVBoxLayout(self.dva_tab)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        sc_widget = QWidget()
        sc_layout = QVBoxLayout(sc_widget)

        beta_group = QGroupBox("β (beta) Parameters")
        beta_form = QFormLayout(beta_group)
        self.beta_boxes = []
        for i in range(15):
            b = QDoubleSpinBox(); b.setRange(-1e6,1e6); b.setDecimals(6)
            self.beta_boxes.append(b)
            beta_form.addRow(f"β_{i+1}:", b)
        sc_layout.addWidget(beta_group)

        lambda_group = QGroupBox("λ (lambda) Parameters")
        lambda_form = QFormLayout(lambda_group)
        self.lambda_boxes = []
        for i in range(15):
            l = QDoubleSpinBox(); l.setRange(-1e6,1e6); l.setDecimals(6)
            self.lambda_boxes.append(l)
            lambda_form.addRow(f"λ_{i+1}:", l)
        sc_layout.addWidget(lambda_group)

        mu_group = QGroupBox("μ (mu) Parameters")
        mu_form = QFormLayout(mu_group)
        self.mu_dva_boxes = []
        for i in range(3):
            m = QDoubleSpinBox(); m.setRange(-1e6,1e6); m.setDecimals(6)
            self.mu_dva_boxes.append(m)
            mu_form.addRow(f"μ_{i+1}:", m)
        sc_layout.addWidget(mu_group)

        nu_group = QGroupBox("ν (nu) Parameters")
        nu_form = QFormLayout(nu_group)
        self.nu_dva_boxes = []
        for i in range(15):
            n = QDoubleSpinBox(); n.setRange(-1e6,1e6); n.setDecimals(6)
            self.nu_dva_boxes.append(n)
            nu_form.addRow(f"ν_{i+1}:", n)
        sc_layout.addWidget(nu_group)

        sc_layout.addStretch()
        sc_widget.setLayout(sc_layout)
        scroll.setWidget(sc_widget)
        layout.addWidget(scroll)

    def create_target_weights_tab(self):
        self.tw_tab = QWidget()
        layout = QVBoxLayout(self.tw_tab)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        sc_widget = QWidget()
        sc_layout = QVBoxLayout(sc_widget)

        self.mass_target_spins = {}
        self.mass_weight_spins = {}

        for mass_num in range(1,6):
            mass_group = QGroupBox(f"Mass {mass_num} Targets \& Weights")
            mg_layout = QVBoxLayout(mass_group)

            peak_group = QGroupBox("Peak Values \& Weights")
            peak_form = QFormLayout(peak_group)
            for peak_num in range(1,5):
                pv = QDoubleSpinBox(); pv.setRange(0,1e6); pv.setDecimals(6)
                wv = QDoubleSpinBox(); wv.setRange(0,1e3); wv.setDecimals(6)
                peak_form.addRow(f"Peak Value {peak_num}:", pv)
                peak_form.addRow(f"Weight Peak Value {peak_num}:", wv)
                self.mass_target_spins[f"peak_value_{peak_num}_m{mass_num}"] = pv
                self.mass_weight_spins[f"peak_value_{peak_num}_m{mass_num}"] = wv
            mg_layout.addWidget(peak_group)

            bw_group = QGroupBox("Bandwidth Targets \& Weights")
            bw_form = QFormLayout(bw_group)
            for i in range(1,5):
                for j in range(i+1,5):
                    bw = QDoubleSpinBox(); bw.setRange(0,1e6); bw.setDecimals(6)
                    wbw = QDoubleSpinBox(); wbw.setRange(0,1e3); wbw.setDecimals(6)
                    bw_form.addRow(f"Bandwidth {i}-{j}:", bw)
                    bw_form.addRow(f"Weight Bandwidth {i}-{j}:", wbw)
                    self.mass_target_spins[f"bandwidth_{i}_{j}_m{mass_num}"] = bw
                    self.mass_weight_spins[f"bandwidth_{i}_{j}_m{mass_num}"] = wbw
            mg_layout.addWidget(bw_group)

            slope_group = QGroupBox("Slope Targets \& Weights")
            slope_form = QFormLayout(slope_group)
            for i in range(1,5):
                for j in range(i+1,5):
                    s = QDoubleSpinBox(); s.setRange(-1e6,1e6); s.setDecimals(6)
                    ws = QDoubleSpinBox(); ws.setRange(0,1e3); ws.setDecimals(6)
                    slope_form.addRow(f"Slope {i}-{j}:", s)
                    slope_form.addRow(f"Weight Slope {i}-{j}:", ws)
                    self.mass_target_spins[f"slope_{i}_{j}_m{mass_num}"] = s
                    self.mass_weight_spins[f"slope_{i}_{j}_m{mass_num}"] = ws
            mg_layout.addWidget(slope_group)

            auc_group = QGroupBox("Area Under Curve \& Weight")
            auc_form = QFormLayout(auc_group)
            auc = QDoubleSpinBox(); auc.setRange(0,1e6); auc.setDecimals(6)
            wauc = QDoubleSpinBox(); wauc.setRange(0,1e3); wauc.setDecimals(6)
            auc_form.addRow("Area Under Curve:", auc)
            auc_form.addRow("Weight Area Under Curve:", wauc)
            self.mass_target_spins[f"area_under_curve_m{mass_num}"] = auc
            self.mass_weight_spins[f"area_under_curve_m{mass_num}"] = wauc
            mg_layout.addWidget(auc_group)

            mg_layout.addStretch()
            sc_layout.addWidget(mass_group)

        sc_layout.addStretch()
        sc_widget.setLayout(sc_layout)
        scroll.setWidget(sc_widget)
        layout.addWidget(scroll)

    def create_frequency_tab(self):
        self.freq_tab = QWidget()
        layout = QVBoxLayout(self.freq_tab)

        freq_group = QGroupBox("Frequency Range \& Plot Options")
        freq_form = QFormLayout(freq_group)

        self.omega_start_box = QDoubleSpinBox(); self.omega_start_box.setRange(0,1e6); self.omega_start_box.setDecimals(6)
        self.omega_end_box = QDoubleSpinBox(); self.omega_end_box.setRange(0,1e6); self.omega_end_box.setDecimals(6)
        self.omega_points_box = QSpinBox(); self.omega_points_box.setRange(1,1024)

        freq_form.addRow("Ω Start:", self.omega_start_box)
        freq_form.addRow("Ω End:", self.omega_end_box)
        freq_form.addRow("Ω Points:", self.omega_points_box)

        self.plot_figure_chk = QCheckBox("Plot Figure")
        self.show_peaks_chk = QCheckBox("Show Peaks")
        self.show_slopes_chk = QCheckBox("Show Slopes")

        freq_form.addRow(self.plot_figure_chk)
        freq_form.addRow(self.show_peaks_chk)
        freq_form.addRow(self.show_slopes_chk)

        layout.addWidget(freq_group)

    def create_sobol_analysis_tab(self):
        self.sobol_tab = QWidget()
        layout = QVBoxLayout(self.sobol_tab)

        splitter = QSplitter(Qt.Horizontal)

        sobol_group = QGroupBox("Sobol Analysis Settings")
        sobol_form = QFormLayout(sobol_group)

        self.num_samples_line = QLineEdit()
        self.num_samples_line.setPlaceholderText("e.g. 32,64,128")
        sobol_form.addRow("Num Samples List:", self.num_samples_line)

        self.n_jobs_spin = QSpinBox()
        self.n_jobs_spin.setRange(1,64)
        sobol_form.addRow("Number of Jobs (n_jobs):", self.n_jobs_spin)

        sobol_group.setLayout(sobol_form)
        splitter.addWidget(sobol_group)

        dva_parameters = [
            *[f"beta_{i}" for i in range(1,16)],
            *[f"lambda_{i}" for i in range(1,16)],
            *[f"mu_{i}" for i in range(1,4)],
            *[f"nu_{i}" for i in range(1,16)]
        ]

        dva_param_group = QGroupBox("DVA Parameter Settings")
        dva_param_layout = QVBoxLayout(dva_param_group)

        self.dva_param_table = QTableWidget()
        self.dva_param_table.setRowCount(len(dva_parameters))
        self.dva_param_table.setColumnCount(5)
        self.dva_param_table.setHorizontalHeaderLabels(["Parameter", "Fixed", "Fixed Value", "Lower Bound", "Upper Bound"])
        self.dva_param_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.dva_param_table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        for row, param in enumerate(dva_parameters):
            param_item = QTableWidgetItem(param)
            param_item.setFlags(Qt.ItemIsEnabled)
            self.dva_param_table.setItem(row, 0, param_item)

            fixed_checkbox = QCheckBox()
            fixed_checkbox.stateChanged.connect(lambda state, r=row: self.toggle_fixed(state, r))
            self.dva_param_table.setCellWidget(row, 1, fixed_checkbox)

            fixed_value_spin = QDoubleSpinBox()
            fixed_value_spin.setRange(-1e6,1e6)
            fixed_value_spin.setDecimals(6)
            fixed_value_spin.setEnabled(False)
            self.dva_param_table.setCellWidget(row, 2, fixed_value_spin)

            lower_bound_spin = QDoubleSpinBox()
            lower_bound_spin.setRange(-1e6,1e6)
            lower_bound_spin.setDecimals(6)
            lower_bound_spin.setEnabled(True)
            self.dva_param_table.setCellWidget(row, 3, lower_bound_spin)

            upper_bound_spin = QDoubleSpinBox()
            upper_bound_spin.setRange(-1e6,1e6)
            upper_bound_spin.setDecimals(6)
            upper_bound_spin.setEnabled(True)
            self.dva_param_table.setCellWidget(row, 4, upper_bound_spin)

            if param.startswith("beta_") or param.startswith("lambda_") or param.startswith("nu_"):
                lower_bound_spin.setValue(0.0001)
                upper_bound_spin.setValue(2.5)
            elif param.startswith("mu_"):
                lower_bound_spin.setValue(0.0001)
                upper_bound_spin.setValue(0.75)
            else:
                lower_bound_spin.setValue(0.0)
                upper_bound_spin.setValue(1.0)

        dva_param_layout.addWidget(self.dva_param_table)
        dva_param_group.setLayout(dva_param_layout)

        splitter.addWidget(dva_param_group)
        splitter.setSizes([300, 800])

        layout.addWidget(splitter)
        layout.addStretch()
        self.sobol_tab.setLayout(layout)

    def create_ga_tab(self):
        self.ga_tab = QWidget()
        layout = QVBoxLayout(self.ga_tab)

        splitter = QSplitter(Qt.Horizontal)

        ga_group = QGroupBox("GA Settings")
        ga_form = QFormLayout(ga_group)
        self.ga_pop_size_box = QSpinBox(); self.ga_pop_size_box.setRange(1,10000); self.ga_pop_size_box.setValue(800)
        self.ga_num_generations_box = QSpinBox(); self.ga_num_generations_box.setRange(1,10000); self.ga_num_generations_box.setValue(100)
        self.ga_cxpb_box = QDoubleSpinBox(); self.ga_cxpb_box.setRange(0,1); self.ga_cxpb_box.setValue(0.7); self.ga_cxpb_box.setDecimals(3)
        self.ga_mutpb_box = QDoubleSpinBox(); self.ga_mutpb_box.setRange(0,1); self.ga_mutpb_box.setValue(0.2); self.ga_mutpb_box.setDecimals(3)
        self.ga_tol_box = QDoubleSpinBox(); self.ga_tol_box.setRange(0,1e6); self.ga_tol_box.setValue(1e-3); self.ga_tol_box.setDecimals(6)
        self.ga_alpha_box = QDoubleSpinBox(); self.ga_alpha_box.setRange(0.0,10.0); self.ga_alpha_box.setDecimals(4); self.ga_alpha_box.setSingleStep(0.01); self.ga_alpha_box.setValue(0.01)

        ga_form.addRow("Population Size:", self.ga_pop_size_box)
        ga_form.addRow("Number of Generations:", self.ga_num_generations_box)
        ga_form.addRow("Crossover Probability (cxpb):", self.ga_cxpb_box)
        ga_form.addRow("Mutation Probability (mutpb):", self.ga_mutpb_box)
        ga_form.addRow("Tolerance (tol):", self.ga_tol_box)
        ga_form.addRow("Sparsity Penalty (alpha):", self.ga_alpha_box)

        ga_param_group = QGroupBox("DVA Parameter Settings for GA")
        ga_param_layout = QVBoxLayout(ga_param_group)

        self.ga_param_table = QTableWidget()
        dva_parameters = [
            *[f"beta_{i}" for i in range(1,16)],
            *[f"lambda_{i}" for i in range(1,16)],
            *[f"mu_{i}" for i in range(1,4)],
            *[f"nu_{i}" for i in range(1,16)]
        ]

        self.ga_param_table.setRowCount(len(dva_parameters))
        self.ga_param_table.setColumnCount(5)
        self.ga_param_table.setHorizontalHeaderLabels(["Parameter", "Fixed", "Fixed Value", "Lower Bound", "Upper Bound"])
        self.ga_param_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ga_param_table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        for row, param in enumerate(dva_parameters):
            param_item = QTableWidgetItem(param)
            param_item.setFlags(Qt.ItemIsEnabled)
            self.ga_param_table.setItem(row, 0, param_item)

            fixed_checkbox = QCheckBox()
            fixed_checkbox.stateChanged.connect(lambda state, r=row: self.toggle_ga_fixed(state, r))
            self.ga_param_table.setCellWidget(row, 1, fixed_checkbox)

            fixed_value_spin = QDoubleSpinBox()
            fixed_value_spin.setRange(-1e6,1e6)
            fixed_value_spin.setDecimals(6)
            fixed_value_spin.setEnabled(False)
            self.ga_param_table.setCellWidget(row, 2, fixed_value_spin)

            lower_bound_spin = QDoubleSpinBox()
            lower_bound_spin.setRange(-1e6,1e6)
            lower_bound_spin.setDecimals(6)
            lower_bound_spin.setEnabled(True)
            self.ga_param_table.setCellWidget(row, 3, lower_bound_spin)

            upper_bound_spin = QDoubleSpinBox()
            upper_bound_spin.setRange(-1e6,1e6)
            upper_bound_spin.setDecimals(6)
            upper_bound_spin.setEnabled(True)
            self.ga_param_table.setCellWidget(row, 4, upper_bound_spin)

            if param.startswith("beta_") or param.startswith("lambda_") or param.startswith("nu_"):
                lower_bound_spin.setValue(0.0001)
                upper_bound_spin.setValue(2.5)
            elif param.startswith("mu_"):
                lower_bound_spin.setValue(0.0001)
                upper_bound_spin.setValue(0.75)
            else:
                lower_bound_spin.setValue(0.0)
                upper_bound_spin.setValue(1.0)

        ga_param_layout.addWidget(self.ga_param_table)
        ga_param_group.setLayout(ga_param_layout)

        splitter.addWidget(ga_group)
        splitter.addWidget(ga_param_group)
        splitter.setSizes([400, 800])

        layout.addWidget(splitter)
        self.ga_results_text = QTextEdit()
        self.ga_results_text.setReadOnly(True)
        layout.addWidget(QLabel("GA Optimization Results:"))
        layout.addWidget(self.ga_results_text)

        self.run_ga_button_tab = QPushButton("Run GA")
        self.run_ga_button_tab.setFixedWidth(150)
        self.run_ga_button_tab.clicked.connect(self.run_ga)
        run_ga_layout = QHBoxLayout()
        run_ga_layout.addStretch()
        run_ga_layout.addWidget(self.run_ga_button_tab)
        layout.addLayout(run_ga_layout)

        layout.addStretch()
        self.ga_tab.setLayout(layout)

    ##################################
    # Handling GA "Fixed" toggling
    ##################################
    
    def toggle_ga_fixed(self, state, row):
        fixed = (state == Qt.Checked)
        fixed_value_spin = self.ga_param_table.cellWidget(row, 2)
        lower_bound_spin = self.ga_param_table.cellWidget(row, 3)
        upper_bound_spin = self.ga_param_table.cellWidget(row, 4)

        fixed_value_spin.setEnabled(fixed)
        lower_bound_spin.setEnabled(not fixed)
        upper_bound_spin.setEnabled(not fixed)

    ##################################
    # Running FRF / Sobol / GA
    ##################################

    def run_frf(self):
        if self.omega_start_box.value() >= self.omega_end_box.value():
            QMessageBox.warning(self, "Input Error", "Ω Start must be less than Ω End.")
            return

        target_values, weights = self.get_target_values_weights()

        self.run_frf_button.setEnabled(False)
        self.run_sobol_button.setEnabled(False)
        self.run_ga_button.setEnabled(False)
        self.results_text.append("\n--- Running FRF Analysis ---\n")
        self.status_bar.showMessage("Running FRF Analysis...")

        main_params = (
            self.mu_box.value(),
            *[b.value() for b in self.landa_boxes],
            *[b.value() for b in self.nu_boxes],
            self.a_low_box.value(),
            self.a_up_box.value(),
            self.f_1_box.value(),
            self.f_2_box.value(),
            self.omega_dc_box.value(),
            self.zeta_dc_box.value()
        )

        beta_vals = [b.value() for b in self.beta_boxes]
        lambda_vals = [l.value() for l in self.lambda_boxes]
        mu_vals = [m.value() for m in self.mu_dva_boxes]
        nu_vals = [n.value() for n in self.nu_dva_boxes]
        dva_params = tuple(beta_vals + lambda_vals + mu_vals + nu_vals)

        omega_start_val = self.omega_start_box.value()
        omega_end_val = self.omega_end_box.value()
        omega_points_val = self.omega_points_box.value()

        plot_figure = self.plot_figure_chk.isChecked()
        show_peaks = self.show_peaks_chk.isChecked()
        show_slopes = self.show_slopes_chk.isChecked()

        self.frf_worker = FRFWorker(
            main_params=main_params,
            dva_params=dva_params,
            omega_start=omega_start_val,
            omega_end=omega_end_val,
            omega_points=omega_points_val,
            target_values_dict=target_values,
            weights_dict=weights,
            plot_figure=plot_figure,
            show_peaks=show_peaks,
            show_slopes=show_slopes
        )
        self.frf_worker.finished.connect(self.display_frf_results)
        self.frf_worker.error.connect(self.handle_frf_error)
        self.frf_worker.start()

    def run_sobol(self):
        if self.omega_start_box.value() >= self.omega_end_box.value():
            QMessageBox.warning(self, "Input Error", "Ω Start must be less than Ω End.")
            return

        target_values, weights = self.get_target_values_weights()
        num_samples_list = self.get_num_samples_list()
        n_jobs = self.n_jobs_spin.value()

        self.run_frf_button.setEnabled(False)
        self.run_sobol_button.setEnabled(False)
        self.run_ga_button.setEnabled(False)
        self.sobol_results_text.append("\n--- Running Sobol Sensitivity Analysis ---\n")
        self.status_bar.showMessage("Running Sobol Analysis...")

        main_params = (
            self.mu_box.value(),
            *[b.value() for b in self.landa_boxes],
            *[b.value() for b in self.nu_boxes],
            self.a_low_box.value(),
            self.a_up_box.value(),
            self.f_1_box.value(),
            self.f_2_box.value(),
            self.omega_dc_box.value(),
            self.zeta_dc_box.value()
        )

        dva_bounds = {}
        EPSILON = 1e-6
        for row in range(self.dva_param_table.rowCount()):
            param_item = self.dva_param_table.item(row, 0)
            param_name = param_item.text()

            fixed_widget = self.dva_param_table.cellWidget(row, 1)
            fixed = fixed_widget.isChecked()

            if fixed:
                fixed_value_widget = self.dva_param_table.cellWidget(row, 2)
                fixed_value = fixed_value_widget.value()
                dva_bounds[param_name] = (fixed_value, fixed_value + EPSILON)
            else:
                lower_bound_widget = self.dva_param_table.cellWidget(row, 3)
                upper_bound_widget = self.dva_param_table.cellWidget(row, 4)
                lower = lower_bound_widget.value()
                upper = upper_bound_widget.value()
                if lower > upper:
                    QMessageBox.warning(self, "Input Error",
                                        f"For parameter {param_name}, lower bound is greater than upper bound.")
                    self.run_frf_button.setEnabled(True)
                    self.run_sobol_button.setEnabled(True)
                    self.run_ga_button.setEnabled(True)
                    return
                dva_bounds[param_name] = (lower, upper)

        original_dva_parameter_order = [
            'beta_1','beta_2','beta_3','beta_4','beta_5','beta_6',
            'beta_7','beta_8','beta_9','beta_10','beta_11','beta_12',
            'beta_13','beta_14','beta_15',
            'lambda_1','lambda_2','lambda_3','lambda_4','lambda_5',
            'lambda_6','lambda_7','lambda_8','lambda_9','lambda_10',
            'lambda_11','lambda_12','lambda_13','lambda_14','lambda_15',
            'mu_1','mu_2','mu_3',
            'nu_1','nu_2','nu_3','nu_4','nu_5','nu_6',
            'nu_7','nu_8','nu_9','nu_10','nu_11','nu_12',
            'nu_13','nu_14','nu_15'
        ]

        self.sobol_worker = SobolWorker(
            main_params=main_params,
            dva_bounds=dva_bounds,
            dva_order=original_dva_parameter_order,
            omega_start=self.omega_start_box.value(),
            omega_end=self.omega_end_box.value(),
            omega_points=self.omega_points_box.value(),
            num_samples_list=num_samples_list,
            target_values_dict=target_values,
            weights_dict=weights,
            n_jobs=n_jobs
        )
        self.sobol_worker.finished.connect(lambda res,warn: self.display_sobol_results(res,warn))
        self.sobol_worker.error.connect(self.handle_sobol_error)
        self.sobol_worker.start()

    def run_ga(self):
        pop_size = self.ga_pop_size_box.value()
        num_gens = self.ga_num_generations_box.value()
        cxpb = self.ga_cxpb_box.value()
        mutpb = self.ga_mutpb_box.value()
        tol = self.ga_tol_box.value()
        alpha = self.ga_alpha_box.value()

        ga_dva_parameters = []
        row_count = self.ga_param_table.rowCount()
        for row in range(row_count):
            param_name = self.ga_param_table.item(row, 0).text()
            fixed_widget = self.ga_param_table.cellWidget(row, 1)
            fixed = fixed_widget.isChecked()
            if fixed:
                fixed_value_widget = self.ga_param_table.cellWidget(row, 2)
                fv = fixed_value_widget.value()
                ga_dva_parameters.append((param_name, fv, fv, True))
            else:
                lower_bound_widget = self.ga_param_table.cellWidget(row, 3)
                upper_bound_widget = self.ga_param_table.cellWidget(row, 4)
                lb = lower_bound_widget.value()
                ub = upper_bound_widget.value()
                if lb > ub:
                    QMessageBox.warning(self, "Input Error",
                                        f"For parameter {param_name}, lower bound is greater than upper bound.")
                    return
                ga_dva_parameters.append((param_name, lb, ub, False))

        main_params = (
            self.mu_box.value(),
            *[b.value() for b in self.landa_boxes],
            *[b.value() for b in self.nu_boxes],
            self.a_low_box.value(),
            self.a_up_box.value(),
            self.f_1_box.value(),
            self.f_2_box.value(),
            self.omega_dc_box.value(),
            self.zeta_dc_box.value()
        )

        target_values, weights = self.get_target_values_weights()

        omega_start_val = self.omega_start_box.value()
        omega_end_val = self.omega_end_box.value()
        omega_points_val = self.omega_points_box.value()

        self.ga_worker = GAWorker(
            main_params=main_params,
            target_values_dict=target_values,
            weights_dict=weights,
            omega_start=omega_start_val,
            omega_end=omega_end_val,
            omega_points=omega_points_val,
            ga_pop_size=pop_size,
            ga_num_generations=num_gens,
            ga_cxpb=cxpb,
            ga_mutpb=mutpb,
            ga_tol=tol,
            ga_parameter_data=ga_dva_parameters,
            alpha=alpha
        )
        self.ga_worker.finished.connect(self.handle_ga_finished)
        self.ga_worker.error.connect(self.handle_ga_error)
        self.ga_worker.update.connect(self.handle_ga_update)
        self.run_ga_button.setEnabled(False)
        self.ga_results_text.append("Running GA...")
        self.ga_worker.start()

    ##################################
    # Display FRF / Sobol / GA Results
    ##################################

    def display_frf_results(self, results_with_dva, results_without_dva):
        self.frf_results = results_with_dva
        self.results_text.append("\n--- FRF Analysis Results (With DVA) ---")

        required_masses = [f'mass_{m}' for m in range(1,6)]
        def format_float(val):
            if isinstance(val,(np.float64,float,int)):
                return f"{val:.6e}"
            return str(val)

        for mass in required_masses:
            self.results_text.append(f"\nRaw results for {mass}:")
            if mass in self.frf_results:
                for key, value in self.frf_results[mass].items():
                    if isinstance(value, dict):
                        formatted_dict = {k: format_float(v) for k,v in value.items()}
                        self.results_text.append(f"{key}: {formatted_dict}")
                    else:
                        self.results_text.append(f"{key}: {format_float(value)}")
            else:
                self.results_text.append(f"No results for {mass}")

        self.results_text.append("\nComposite Measures:")
        if 'composite_measures' in self.frf_results:
            for mass, comp in self.frf_results['composite_measures'].items():
                self.results_text.append(f"{mass}: {format_float(comp)}")
        else:
            self.results_text.append("No composite measures found.")

        self.results_text.append("\nSingular Response:")
        if 'singular_response' in self.frf_results:
            self.results_text.append(f"{format_float(self.frf_results['singular_response'])}")
        else:
            self.results_text.append("No singular response found.")

        # Handle results without DVA
        self.results_text.append("\n--- FRF Analysis Results (Without DVAs for Mass 1 and Mass 2) ---")
        required_masses_without_dva = ['mass_1', 'mass_2']
        for mass in required_masses_without_dva:
            self.results_text.append(f"\nRaw results for {mass}:")
            if mass in results_without_dva:
                for key, value in results_without_dva[mass].items():
                    if isinstance(value, dict):
                        formatted_dict = {k: format_float(v) for k,v in value.items()}
                        self.results_text.append(f"{key}: {formatted_dict}")
                    else:
                        self.results_text.append(f"{key}: {format_float(value)}")
            else:
                self.results_text.append(f"No results for {mass}")

        self.results_text.append("\nComposite Measures (Without DVAs for Mass 1 and Mass 2):")
        if 'composite_measures' in results_without_dva:
            for mass, comp in results_without_dva['composite_measures'].items():
                if mass in ['mass_1', 'mass_2']:
                    self.results_text.append(f"{mass}: {format_float(comp)}")
        else:
            self.results_text.append("No composite measures found.")

        self.results_text.append("\nSingular Response (Without DVAs for Mass 1 and Mass 2):")
        if 'singular_response' in results_without_dva:
            self.results_text.append(f"{format_float(results_without_dva['singular_response'])}")
        else:
            self.results_text.append("No singular response found.")

        self.status_bar.showMessage("FRF Analysis Completed.")
        self.run_frf_button.setEnabled(True)
        self.run_sobol_button.setEnabled(True)
        self.run_ga_button.setEnabled(True)

        self.frf_combo.clear()
        self.frf_plots.clear()

        omega = np.linspace(self.omega_start_box.value(), self.omega_end_box.value(), self.omega_points_box.value())
        mass_labels = []
        for m_label in required_masses:
            if m_label in self.frf_results and 'magnitude' in self.frf_results[m_label]:
                mass_labels.append(m_label)

        # Plot with DVAs
        for m_label in mass_labels:
            fig = Figure(figsize=(6,4))
            ax = fig.add_subplot(111)
            mag = self.frf_results[m_label]['magnitude']
            if len(mag) == len(omega):
                ax.plot(omega, mag, label=m_label)
                ax.set_xlabel('Frequency (rad/s)')
                ax.set_ylabel('Amplitude')
                ax.set_title(f'FRF of {m_label} (With DVA)')
                ax.legend()
                ax.grid(True)
                self.frf_combo.addItem(f"{m_label} (With DVA)")
                self.frf_plots[f"{m_label} (With DVA)"] = fig
            else:
                QMessageBox.warning(self, "Plot Error", f"{m_label}: magnitude length does not match omega length.")

        # Plot combined with DVAs
        if mass_labels:
            fig_combined = Figure(figsize=(6,4))
            ax_combined = fig_combined.add_subplot(111)
            for m_label in mass_labels:
                mag = self.frf_results[m_label]['magnitude']
                if len(mag) == len(omega):
                    ax_combined.plot(omega, mag, label=m_label)
            ax_combined.set_xlabel('Frequency (rad/s)')
            ax_combined.set_ylabel('Amplitude')
            ax_combined.set_title('Combined FRF of All Masses (With DVAs)')
            ax_combined.grid(True)
            ax_combined.legend()
            self.frf_combo.addItem("All Masses Combined (With DVAs)")
            self.frf_plots["All Masses Combined (With DVAs)"] = fig_combined

        # Plot without DVAs for Mass1 and Mass2
        for m_label in ['mass_1', 'mass_2']:
            if m_label in results_without_dva and 'magnitude' in results_without_dva[m_label]:
                fig = Figure(figsize=(6,4))
                ax = fig.add_subplot(111)
                mag = results_without_dva[m_label]['magnitude']
                if len(mag) == len(omega):
                    ax.plot(omega, mag, label=f"{m_label} (Without DVA)", color='green')
                    ax.set_xlabel('Frequency (rad/s)')
                    ax.set_ylabel('Amplitude')
                    ax.set_title(f'FRF of {m_label} (Without DVA)')
                    ax.legend()
                    ax.grid(True)
                    self.frf_combo.addItem(f"{m_label} (Without DVA)")
                    self.frf_plots[f"{m_label} (Without DVA)"] = fig
                else:
                    QMessageBox.warning(self, "Plot Error", f"{m_label} (Without DVA): magnitude length does not match omega length.")

        # Plot combined with and without DVAs for Mass1 and Mass2
        fig_combined_with_without = Figure(figsize=(6,4))
        ax_combined_with_without = fig_combined_with_without.add_subplot(111)
        # Plot with DVA
        for m_label in ['mass_1', 'mass_2']:
            if m_label in self.frf_results and 'magnitude' in self.frf_results[m_label]:
                mag = self.frf_results[m_label]['magnitude']
                if len(mag) == len(omega):
                    ax_combined_with_without.plot(omega, mag, label=f"{m_label} (With DVA)")
        # Plot without DVA
        for m_label in ['mass_1', 'mass_2']:
            if m_label in results_without_dva and 'magnitude' in results_without_dva[m_label]:
                mag = results_without_dva[m_label]['magnitude']
                if len(mag) == len(omega):
                    ax_combined_with_without.plot(omega, mag, label=f"{m_label} (Without DVA)", linestyle='--')
        ax_combined_with_without.set_xlabel('Frequency (rad/s)')
        ax_combined_with_without.set_ylabel('Amplitude')
        ax_combined_with_without.set_title('FRF of Mass 1 \& 2: With and Without DVAs')
        ax_combined_with_without.grid(True)
        ax_combined_with_without.legend()
        self.frf_combo.addItem("Mass 1 \& 2: With and Without DVAs")
        self.frf_plots["Mass 1 \& 2: With and Without DVAs"] = fig_combined_with_without

        # Plot all masses combined with and without DVAs
        fig_all_combined = Figure(figsize=(6,4))
        ax_all_combined = fig_all_combined.add_subplot(111)
        # With DVAs
        for m_label in mass_labels:
            mag = self.frf_results[m_label]['magnitude']
            if len(mag) == len(omega):
                ax_all_combined.plot(omega, mag, label=f"{m_label} (With DVA)")
        # Without DVAs for Mass1 and Mass2
        for m_label in ['mass_1', 'mass_2']:
            if m_label in results_without_dva and 'magnitude' in results_without_dva[m_label]:
                mag = results_without_dva[m_label]['magnitude']
                if len(mag) == len(omega):
                    ax_all_combined.plot(omega, mag, label=f"{m_label} (Without DVA)", linestyle='--')
        ax_all_combined.set_xlabel('Frequency (rad/s)')
        ax_all_combined.set_ylabel('Amplitude')
        ax_all_combined.set_title('Combined FRF of All Masses: With and Without DVAs for Mass 1 \& 2')
        ax_all_combined.grid(True)
        ax_all_combined.legend()
        self.frf_combo.addItem("All Masses Combined: With and Without DVAs for Mass 1 \& 2")
        self.frf_plots["All Masses Combined: With and Without DVAs for Mass 1 \& 2"] = fig_all_combined

        self.update_frf_plot()
        self.frf_canvas.draw()

    def handle_frf_error(self, err):
        self.results_text.append(f"Error running FRF: {err}")
        self.status_bar.showMessage("FRF Error encountered.")
        self.run_frf_button.setEnabled(True)
        self.run_sobol_button.setEnabled(True)
        self.run_ga_button.setEnabled(True)

    def display_sobol_results(self, all_results, warnings):
        self.sobol_results = all_results
        self.sobol_warnings = warnings
        self.sobol_results_text.append("\n--- Sobol Sensitivity Analysis Results ---")

        original_dva_parameter_order = [
            'beta_1','beta_2','beta_3','beta_4','beta_5','beta_6',
            'beta_7','beta_8','beta_9','beta_10','beta_11','beta_12',
            'beta_13','beta_14','beta_15',
            'lambda_1','lambda_2','lambda_3','lambda_4','lambda_5',
            'lambda_6','lambda_7','lambda_8','lambda_9','lambda_10',
            'lambda_11','lambda_12','lambda_13','lambda_14','lambda_15',
            'mu_1','mu_2','mu_3',
            'nu_1','nu_2','nu_3','nu_4','nu_5','nu_6',
            'nu_7','nu_8','nu_9','nu_10','nu_11','nu_12',
            'nu_13','nu_14','nu_15'
        ]
        param_names = original_dva_parameter_order

        def format_float(val):
            if isinstance(val,(np.float64,float,int)):
                return f"{val:.6f}"
            return str(val)

        for run_idx, num_samples in enumerate(all_results['samples']):
            self.sobol_results_text.append(f"\nSample Size: {num_samples}")
            S1 = all_results['S1'][run_idx]
            ST = all_results['ST'][run_idx]
            self.sobol_results_text.append(f"  Length of S1: {len(S1)}, Length of ST: {len(ST)}")
            for param_index, param_name in enumerate(param_names):
                if param_index < len(S1) and param_index < len(ST):
                    s1_val = S1[param_index]
                    st_val = ST[param_index]
                    self.sobol_results_text.append(f"Parameter {param_name}: S1 = {s1_val:.6f}, ST = {st_val:.6f}")
                else:
                    self.sobol_results_text.append(f"IndexError: Parameter {param_name} out of range")

        if warnings:
            self.sobol_results_text.append("\nWarnings:")
            for w in warnings:
                self.sobol_results_text.append(w)
        else:
            self.sobol_results_text.append("\nNo warnings encountered.")

        self.status_bar.showMessage("Sobol Analysis Completed.")
        self.run_frf_button.setEnabled(True)
        self.run_sobol_button.setEnabled(True)
        self.run_ga_button.setEnabled(True)

        self.sobol_combo.clear()
        self.sobol_plots.clear()

        self.generate_sobol_plots(all_results, param_names)
        self.update_sobol_plot()
        self.sobol_canvas.draw()

    def handle_sobol_error(self, err):
        self.sobol_results_text.append(f"Error running Sobol Analysis: {err}")
        self.status_bar.showMessage("Sobol Error encountered.")
        self.run_frf_button.setEnabled(True)
        self.run_sobol_button.setEnabled(True)
        self.run_ga_button.setEnabled(True)

    def handle_ga_finished(self, results, best_ind, parameter_names, best_fitness):
        self.run_ga_button.setEnabled(True)
        self.ga_results_text.append("\nGA Completed.\n")
        self.ga_results_text.append("Best individual parameters:")
        for name, val in zip(parameter_names, best_ind):
            self.ga_results_text.append(f"{name}: {val}")
        self.ga_results_text.append(f"\nBest fitness: {best_fitness:.6f}")

        singular_response = results.get('singular_response', None)
        if singular_response is not None:
            self.ga_results_text.append(f"\nSingular response of best individual: {singular_response}")

        self.ga_results_text.append("\nFull Results:")
        for section, data in results.items():
            self.ga_results_text.append(f"{section}: {data}")

    def handle_ga_error(self, err):
        self.run_ga_button.setEnabled(True)
        QMessageBox.warning(self, "GA Error", f"Error during GA: {err}")
        self.ga_results_text.append(f"\nError running GA: {err}")

    def handle_ga_update(self, msg):
        self.ga_results_text.append(msg)

    ##################################
    # Plotting Helpers
    ##################################

    def update_frf_plot(self):
        key = self.frf_combo.currentText()
        if key in self.frf_plots:
            fig = self.frf_plots[key]
            self.frf_canvas.figure = fig
            self.frf_canvas.draw()
        else:
            self.frf_canvas.figure.clear()
            self.frf_canvas.draw()

    def update_sobol_plot(self):
        key = self.sobol_combo.currentText()
        if key in self.sobol_plots:
            fig = self.sobol_plots[key]
            self.sobol_canvas.figure = fig
            self.sobol_canvas.draw()
        else:
            self.sobol_canvas.figure.clear()
            self.sobol_canvas.draw()

    def save_plot(self, fig, plot_type):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, f"Save {plot_type} Plot", "",
            "PNG Files (*.png);;JPEG Files (*.jpg);;PDF Files (*.pdf);;All Files (*)",
            options=options
        )
        if file_path:
            try:
                fig.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Success", f"{plot_type} plot saved to {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to save plot: {e}")

    def save_sobol_results(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Sobol Results", "",
                                                   "Text Files (*.txt);;All Files (*)", options=options)
        if file_path:
            try:
                with open(file_path,'w') as f:
                    f.write(self.sobol_results_text.toPlainText())
                QMessageBox.information(self,"Success",f"Sobol results saved to {file_path}")
            except Exception as e:
                QMessageBox.warning(self,"Error",f"Failed to save results: {e}")

    ##################################
    # Default Values
    ##################################

    def set_default_values(self):
        self.mu_box.setValue(1.0)
        for i in range(5):
            if i < 2:
                self.landa_boxes[i].setValue(1.0)
            else:
                self.landa_boxes[i].setValue(0.5)
        for i in range(5):
            self.nu_boxes[i].setValue(0.75)

        self.a_low_box.setValue(0.05)
        self.a_up_box.setValue(0.05)
        self.f_1_box.setValue(100.0)
        self.f_2_box.setValue(100.0)
        self.omega_dc_box.setValue(5000.0)
        self.zeta_dc_box.setValue(0.01)

        self.omega_start_box.setValue(0.0)
        self.omega_end_box.setValue(10000.0)
        self.omega_points_box.setValue(1200)

        self.plot_figure_chk.setChecked(True)
        self.show_peaks_chk.setChecked(False)
        self.show_slopes_chk.setChecked(False)

        self.num_samples_line.setText("32,64,128")
        self.n_jobs_spin.setValue(5)

        for row in range(self.dva_param_table.rowCount()):
            param_name = self.dva_param_table.item(row, 0).text()
            if param_name.startswith("beta_") or param_name.startswith("lambda_") or param_name.startswith("nu_"):
                lower = 0.0001
                upper = 2.5
            elif param_name.startswith("mu_"):
                lower = 0.0001
                upper = 0.75
            else:
                lower = 0.0
                upper = 1.0
            lower_bound_spin = self.dva_param_table.cellWidget(row, 3)
            upper_bound_spin = self.dva_param_table.cellWidget(row, 4)
            lower_bound_spin.setValue(lower)
            upper_bound_spin.setValue(upper)
            fixed_checkbox = self.dva_param_table.cellWidget(row, 1)
            fixed_checkbox.setChecked(False)

        for row in range(self.ga_param_table.rowCount()):
            param_name = self.ga_param_table.item(row, 0).text()
            if param_name.startswith("beta_") or param_name.startswith("lambda_") or param_name.startswith("nu_"):
                lower = 0.0001
                upper = 2.5
            elif param_name.startswith("mu_"):
                lower = 0.0001
                upper = 0.75
            else:
                lower = 0.0
                upper = 1.0
            lower_bound_spin = self.ga_param_table.cellWidget(row, 3)
            upper_bound_spin = self.ga_param_table.cellWidget(row, 4)
            lower_bound_spin.setValue(lower)
            upper_bound_spin.setValue(upper)
            fixed_checkbox = self.ga_param_table.cellWidget(row, 1)
            fixed_checkbox.setChecked(False)

    ##################################
    # Sobol Plots Generation
    ##################################

    def generate_sobol_plots(self, all_results, param_names):
        fig_last_run = self.visualize_last_run(all_results, param_names)
        self.sobol_combo.addItem("Last Run Results")
        self.sobol_plots["Last Run Results"] = fig_last_run

        fig_grouped_ST = self.visualize_grouped_bar_plot_sorted_on_ST(all_results, param_names)
        self.sobol_combo.addItem("Grouped Bar (Sorted by ST)")
        self.sobol_plots["Grouped Bar (Sorted by ST)"] = fig_grouped_ST

        conv_figs = self.visualize_convergence_plots(all_results, param_names)
        for i, cf in enumerate(conv_figs, start=1):
            name = f"Convergence Plots Fig {i}"
            self.sobol_combo.addItem(name)
            self.sobol_plots[name] = cf

        fig_heat = self.visualize_combined_heatmap(all_results, param_names)
        self.sobol_combo.addItem("Combined Heatmap")
        self.sobol_plots["Combined Heatmap"] = fig_heat

        fig_comp_radar = self.visualize_comprehensive_radar_plots(all_results, param_names)
        self.sobol_combo.addItem("Comprehensive Radar Plot")
        self.sobol_plots["Comprehensive Radar Plot"] = fig_comp_radar

        fig_s1_radar, fig_st_radar = self.visualize_separate_radar_plots(all_results, param_names)
        self.sobol_combo.addItem("Radar Plot S1")
        self.sobol_plots["Radar Plot S1"] = fig_s1_radar
        self.sobol_combo.addItem("Radar Plot ST")
        self.sobol_plots["Radar Plot ST"] = fig_st_radar

        fig_box = self.visualize_box_plots(all_results)
        self.sobol_combo.addItem("Box Plots")
        self.sobol_plots["Box Plots"] = fig_box

        fig_violin = self.visualize_violin_plots(all_results)
        self.sobol_combo.addItem("Violin Plots")
        self.sobol_plots["Violin Plots"] = fig_violin

        fig_scatter = self.visualize_scatter_S1_ST(all_results, param_names)
        self.sobol_combo.addItem("Scatter S1 vs ST")
        self.sobol_plots["Scatter S1 vs ST"] = fig_scatter

        fig_parallel = self.visualize_parallel_coordinates(all_results, param_names)
        self.sobol_combo.addItem("Parallel Coordinates")
        self.sobol_plots["Parallel Coordinates"] = fig_parallel

        fig_s1_hist, fig_st_hist = self.visualize_histograms(all_results)
        self.sobol_combo.addItem("S1 Histogram")
        self.sobol_plots["S1 Histogram"] = fig_s1_hist
        self.sobol_combo.addItem("ST Histogram")
        self.sobol_plots["ST Histogram"] = fig_st_hist

    ##################################
    # Sobol Visualization Methods
    ##################################

    def visualize_last_run(self, all_results, param_names):
        last_run_idx = -1
        S1_last_run = np.array(all_results['S1'][last_run_idx])
        ST_last_run = np.array(all_results['ST'][last_run_idx])
        sorted_indices_S1 = np.argsort(S1_last_run)[::-1]
        sorted_param_names_S1 = [param_names[i] for i in sorted_indices_S1]
        S1_sorted = S1_last_run[sorted_indices_S1]
        ST_sorted = ST_last_run[sorted_indices_S1]

        fig = Figure(figsize=(6,4))
        ax = fig.add_subplot(111)
        # Escape & with \&
        ax.bar(np.arange(len(sorted_param_names_S1)) - 0.175, S1_sorted, 0.35,
               label=r'$S_1$', color='skyblue')
        ax.bar(np.arange(len(sorted_param_names_S1)) + 0.175, ST_sorted, 0.35,
               label=r'$S_T$', color='salmon')
        ax.set_xlabel('Parameters', fontsize=20)
        ax.set_ylabel('Sensitivity Index', fontsize=20)
        # Fix the LaTeX & here:
        ax.set_title('First-order ($S_1$) \\& Total-order ($S_T$)', fontsize=24)
        ax.set_xticks(np.arange(len(sorted_param_names_S1)))
        ax.set_xticklabels([format_parameter_name(p) for p in sorted_param_names_S1],
                           rotation=90, fontsize=12)
        ax.legend(fontsize=16)
        fig.tight_layout()
        return fig

    def visualize_grouped_bar_plot_sorted_on_ST(self, all_results, param_names):
        last_run_idx = -1
        S1_last_run = np.array(all_results['S1'][last_run_idx])
        ST_last_run = np.array(all_results['ST'][last_run_idx])
        sorted_indices_ST = np.argsort(ST_last_run)[::-1]
        sorted_param_names_ST = [param_names[i] for i in sorted_indices_ST]
        S1_sorted = S1_last_run[sorted_indices_ST]
        ST_sorted = ST_last_run[sorted_indices_ST]

        fig = Figure(figsize=(6,4))
        ax = fig.add_subplot(111)
        ax.bar(np.arange(len(sorted_param_names_ST)) - 0.175, S1_sorted, 0.35,
               label=r'$S_1$', color='skyblue')
        ax.bar(np.arange(len(sorted_param_names_ST)) + 0.175, ST_sorted, 0.35,
               label=r'$S_T$', color='salmon')
        ax.set_xlabel('Parameters', fontsize=20)
        ax.set_ylabel('Sensitivity Index', fontsize=20)
        # Fix the LaTeX & here:
        ax.set_title('First-order ($S_1$) \\& Total-order ($S_T$) - Sorted by $S_T$', fontsize=24)
        ax.set_xticks(np.arange(len(sorted_param_names_ST)))
        ax.set_xticklabels([format_parameter_name(p) for p in sorted_param_names_ST],
                           rotation=90, fontsize=12)
        ax.legend(fontsize=16)
        fig.tight_layout()
        return fig

    def visualize_convergence_plots(self, all_results, param_names):
        sample_sizes = all_results['samples']
        S1_matrix = np.array(all_results['S1'])
        ST_matrix = np.array(all_results['ST'])

        plots_per_fig = 12
        total_params = len(param_names)
        num_figs = int(np.ceil(total_params / plots_per_fig))
        figs = []
        for fig_idx in range(num_figs):
            fig = Figure(figsize=(25,20))
            start_idx = fig_idx*plots_per_fig
            end_idx = min(start_idx+plots_per_fig, total_params)
            for subplot_idx, param_idx in enumerate(range(start_idx,end_idx)):
                param = param_names[param_idx]
                ax = fig.add_subplot(3,4,subplot_idx+1)
                S1_values = S1_matrix[:, param_idx]
                ST_values = ST_matrix[:, param_idx]

                ax.plot(sample_sizes, S1_values, marker='o', linestyle='-', color='tab:blue', label=r'$S_1$')
                ax.plot(sample_sizes, ST_values, marker='s', linestyle='-', color='tab:red', label=r'$S_T$')
                ax.set_xlabel('Sample Size', fontsize=12)
                ax.set_ylabel('Sensitivity Index', fontsize=12)
                ax.set_title(f'Convergence {format_parameter_name(param)}', fontsize=14)
                ax.legend(fontsize=10)
                ax.grid(True, ls="--", linewidth=0.5)
            fig.tight_layout()
            figs.append(fig)
        return figs

    def visualize_combined_heatmap(self, all_results, param_names):
        last_run_idx = -1
        S1_last = np.array(all_results['S1'][last_run_idx])
        ST_last = np.array(all_results['ST'][last_run_idx])
        df = pd.DataFrame({'Parameter': param_names, 'S1': S1_last, 'ST': ST_last}).set_index('Parameter')
        df_sorted = df.sort_values('S1', ascending=False)
        fig = Figure(figsize=(20,max(8,len(param_names)*0.3)))
        ax = fig.add_subplot(111)
        sns.heatmap(df_sorted, annot=True, cmap='coolwarm',
                    cbar_kws={'label': 'Sensitivity Index'}, linewidths=.5,
                    linecolor='gray', ax=ax)
        ax.set_title('Combined Heatmap (S1 \\& ST)', fontsize=24)
        ax.set_xlabel('Sensitivity Indices', fontsize=20)
        ax.set_ylabel('Parameters', fontsize=20)
        return fig

    def visualize_comprehensive_radar_plots(self, all_results, param_names):
        last_run_idx = -1
        S1 = np.array(all_results['S1'][last_run_idx])
        ST = np.array(all_results['ST'][last_run_idx])
        num_vars = len(param_names)
        angles = np.linspace(0,2*np.pi,num_vars,endpoint=False).tolist()
        angles += angles[:1]

        fig = Figure(figsize=(30,30))
        ax = fig.add_subplot(111, polar=True)
        max_val = max(np.max(S1), np.max(ST))*1.1
        ax.set_ylim(0, max_val)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([format_parameter_name(p) for p in param_names], fontsize=14)
        values_S1 = list(S1)+[S1[0]]
        values_ST = list(ST)+[ST[0]]
        ax.plot(angles, values_S1, linewidth=3, label=r'$S_1$')
        ax.fill(angles, values_S1, alpha=0.25, color='skyblue')
        ax.plot(angles, values_ST, linewidth=3, label=r'$S_T$')
        ax.fill(angles, values_ST, alpha=0.25, color='salmon')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3,1.1), fontsize=20)
        ax.set_title('Comprehensive Radar Plot', fontsize=28, y=1.05)
        fig.tight_layout()
        return fig

    def visualize_separate_radar_plots(self, all_results, param_names):
        last_run_idx = -1
        S1 = np.array(all_results['S1'][last_run_idx])
        ST = np.array(all_results['ST'][last_run_idx])
        num_vars = len(param_names)
        angles = np.linspace(0,2*np.pi,num_vars,endpoint=False).tolist()
        angles += angles[:1]

        fig_s1 = Figure(figsize=(30,30))
        ax_s1 = fig_s1.add_subplot(111, polar=True)
        values_S1 = list(S1)+[S1[0]]
        max_val_s1 = np.max(S1)*1.1
        ax_s1.set_ylim(0, max_val_s1)
        ax_s1.set_xticks(angles[:-1])
        ax_s1.set_xticklabels([format_parameter_name(p) for p in param_names], fontsize=14)
        ax_s1.plot(angles, values_S1, linewidth=3, label=r'$S_1$')
        ax_s1.fill(angles, values_S1, alpha=0.25, color='skyblue')
        ax_s1.set_title('Radar Plot of First-order Sensitivity Indices (S1)', fontsize=28, y=1.05)
        ax_s1.legend(loc='upper right', bbox_to_anchor=(1.3,1.1), fontsize=20)
        fig_s1.tight_layout()

        fig_st = Figure(figsize=(30,30))
        ax_st = fig_st.add_subplot(111, polar=True)
        values_ST = list(ST)+[ST[0]]
        max_val_st = np.max(ST)*1.1
        ax_st.set_ylim(0, max_val_st)
        ax_st.set_xticks(angles[:-1])
        ax_st.set_xticklabels([format_parameter_name(p) for p in param_names], fontsize=14)
        ax_st.plot(angles, values_ST, linewidth=3, label=r'$S_T$')
        ax_st.fill(angles, values_ST, alpha=0.25, color='salmon')
        ax_st.set_title('Radar Plot of Total-order Sensitivity Indices (ST)', fontsize=28, y=1.05)
        ax_st.legend(loc='upper right', bbox_to_anchor=(1.3,1.1), fontsize=20)
        fig_st.tight_layout()

        return fig_s1, fig_st

    def visualize_box_plots(self, all_results):
        data = {
            'S1': np.concatenate(all_results['S1']),
            'ST': np.concatenate(all_results['ST'])
        }
        df = pd.DataFrame(data)
        fig = Figure(figsize=(16,12))
        ax = fig.add_subplot(111)
        sns.boxplot(data=df, palette=['skyblue','salmon'], ax=ax)
        ax.set_xlabel('Sensitivity Indices', fontsize=22)
        ax.set_ylabel('Values', fontsize=22)
        # Fix the LaTeX & here:
        ax.set_title('Box Plots of S1 \\& ST', fontsize=26)
        fig.tight_layout()
        return fig

    def visualize_violin_plots(self, all_results):
        data = {
            'S1': np.concatenate(all_results['S1']),
            'ST': np.concatenate(all_results['ST'])
        }
        df = pd.DataFrame(data)
        fig = Figure(figsize=(16,12))
        ax = fig.add_subplot(111)
        sns.violinplot(data=df, palette=['skyblue','salmon'], inner='quartile', ax=ax)
        ax.set_xlabel('Sensitivity Indices', fontsize=22)
        ax.set_ylabel('Values', fontsize=22)
        # Fix the LaTeX & here:
        ax.set_title('Violin Plots of S1 \\& ST', fontsize=26)
        fig.tight_layout()
        return fig

    def visualize_scatter_S1_ST(self, all_results, param_names):
        last_run_idx = -1
        S1_last_run = np.array(all_results['S1'][last_run_idx])
        ST_last_run = np.array(all_results['ST'][last_run_idx])
        fig = Figure(figsize=(18,14))
        ax = fig.add_subplot(111)
        scatter = ax.scatter(S1_last_run, ST_last_run, c=np.arange(len(param_names)),
                             cmap='tab20', edgecolor='k', s=200)
        for i, param in enumerate(param_names):
            ax.text(S1_last_run[i]+0.001, ST_last_run[i]+0.001,
                    format_parameter_name(param), fontsize=12)
        ax.set_xlabel(r'$S_1$', fontsize=22)
        ax.set_ylabel(r'$S_T$', fontsize=22)
        ax.set_title('Scatter Plot of S1 vs ST', fontsize=26)
        ax.grid(True)
        fig.tight_layout()
        return fig

    def visualize_parallel_coordinates(self, all_results, param_names):
        data = []
        for run_idx, num_samples in enumerate(all_results['samples']):
            row = {'Sample Size': num_samples}
            for param_idx, param in enumerate(param_names):
                row[f'S1_{param}'] = all_results['S1'][run_idx][param_idx]
                row[f'ST_{param}'] = all_results['ST'][run_idx][param_idx]
            data.append(row)
        df = pd.DataFrame(data)
        fig = Figure(figsize=(25,20))
        ax = fig.add_subplot(111)
        for param in param_names:
            ax.plot(df['Sample Size'], df[f'S1_{param}'],
                    label=f'S1 {format_parameter_name(param)}',
                    linestyle='-', marker='o', alpha=0.6)
            ax.plot(df['Sample Size'], df[f'ST_{param}'],
                    label=f'ST {format_parameter_name(param)}',
                    linestyle='--', marker='s', alpha=0.6)
        ax.set_xlabel('Sample Size', fontsize=22)
        ax.set_ylabel('Sensitivity Index', fontsize=22)
        # Fix the LaTeX & here:
        ax.set_title('Parallel Coordinates (S1 \\& ST)', fontsize=28)
        ax.grid(True)
        fig.tight_layout()
        return fig

    def visualize_histograms(self, all_results):
        last_run_idx = -1
        S1_last_run = np.array(all_results['S1'][last_run_idx])
        ST_last_run = np.array(all_results['ST'][last_run_idx])

        fig_s1 = Figure(figsize=(18,12))
        ax_s1 = fig_s1.add_subplot(111)
        sns.histplot(S1_last_run, bins=30, kde=True, color='skyblue', ax=ax_s1)
        ax_s1.set_xlabel(r'$S_1$', fontsize=22)
        ax_s1.set_ylabel('Frequency', fontsize=22)
        ax_s1.set_title('Histogram of S1', fontsize=26)
        fig_s1.tight_layout()

        fig_st = Figure(figsize=(18,12))
        ax_st = fig_st.add_subplot(111)
        sns.histplot(ST_last_run, bins=30, kde=True, color='salmon', ax=ax_st)
        ax_st.set_xlabel(r'$S_T$', fontsize=22)
        ax_st.set_ylabel('Frequency', fontsize=22)
        ax_st.set_title('Histogram of $S_T$', fontsize=26)
        fig_st.tight_layout()

        return fig_s1, fig_st

    ##################################
    # Toggles and Inputs
    ##################################

    def toggle_fixed(self, state, row):
        fixed = (state == Qt.Checked)
        fixed_value_spin = self.dva_param_table.cellWidget(row, 2)
        lower_bound_spin = self.dva_param_table.cellWidget(row, 3)
        upper_bound_spin = self.dva_param_table.cellWidget(row, 4)

        fixed_value_spin.setEnabled(fixed)
        lower_bound_spin.setEnabled(not fixed)
        upper_bound_spin.setEnabled(not fixed)

    def get_target_values_weights(self):
        target_values_dict = {}
        weights_dict = {}

        for mass_num in range(1,6):
            t_dict = {}
            w_dict = {}
            for peak_num in range(1,5):
                pv_key = f"peak_value_{peak_num}_m{mass_num}"
                if pv_key in self.mass_target_spins:
                    t_dict[f"peak_value_{peak_num}"] = self.mass_target_spins[pv_key].value()
                    w_dict[f"peak_value_{peak_num}"] = self.mass_weight_spins[pv_key].value()

            for i in range(1,5):
                for j in range(i+1,5):
                    bw_key = f"bandwidth_{i}_{j}_m{mass_num}"
                    if bw_key in self.mass_target_spins:
                        t_dict[f"bandwidth_{i}_{j}"] = self.mass_target_spins[bw_key].value()
                        w_dict[f"bandwidth_{i}_{j}"] = self.mass_weight_spins[bw_key].value()

            for i in range(1,5):
                for j in range(i+1,5):
                    slope_key = f"slope_{i}_{j}_m{mass_num}"
                    if slope_key in self.mass_target_spins:
                        t_dict[f"slope_{i}_{j}"] = self.mass_target_spins[slope_key].value()
                        w_dict[f"slope_{i}_{j}"] = self.mass_weight_spins[slope_key].value()

            auc_key = f"area_under_curve_m{mass_num}"
            if auc_key in self.mass_target_spins:
                t_dict["area_under_curve"] = self.mass_target_spins[auc_key].value()
                w_dict["area_under_curve"] = self.mass_weight_spins[auc_key].value()

            target_values_dict[f"mass_{mass_num}"] = t_dict
            weights_dict[f"mass_{mass_num}"] = w_dict

        return target_values_dict, weights_dict

    def get_num_samples_list(self):
        text = self.num_samples_line.text().strip()
        if not text:
            return [32,64,128]
        try:
            parts = text.split(',')
            nums = [int(p.strip()) for p in parts if p.strip().isdigit()]
            if not nums:
                raise ValueError("No valid integers found.")
            return nums
        except Exception:
            QMessageBox.warning(self, "Input Error", "Invalid Sample Sizes Input. Using default [32,64,128].")
            return [32,64,128]

##################################
# FRF Display and Plotting
##################################

    def display_frf_results(self, results_with_dva, results_without_dva):
        self.frf_results = results_with_dva
        self.results_text.append("\n--- FRF Analysis Results (With DVA) ---")

        required_masses = [f'mass_{m}' for m in range(1,6)]
        def format_float(val):
            if isinstance(val,(np.float64,float,int)):
                return f"{val:.6e}"
            return str(val)

        for mass in required_masses:
            self.results_text.append(f"\nRaw results for {mass}:")
            if mass in self.frf_results:
                for key, value in self.frf_results[mass].items():
                    if isinstance(value, dict):
                        formatted_dict = {k: format_float(v) for k,v in value.items()}
                        self.results_text.append(f"{key}: {formatted_dict}")
                    else:
                        self.results_text.append(f"{key}: {format_float(value)}")
            else:
                self.results_text.append(f"No results for {mass}")

        self.results_text.append("\nComposite Measures:")
        if 'composite_measures' in self.frf_results:
            for mass, comp in self.frf_results['composite_measures'].items():
                self.results_text.append(f"{mass}: {format_float(comp)}")
        else:
            self.results_text.append("No composite measures found.")

        self.results_text.append("\nSingular Response:")
        if 'singular_response' in self.frf_results:
            self.results_text.append(f"{format_float(self.frf_results['singular_response'])}")
        else:
            self.results_text.append("No singular response found.")

        # Handle results without DVA
        self.results_text.append("\n--- FRF Analysis Results (Without DVAs for Mass 1 and Mass 2) ---")
        required_masses_without_dva = ['mass_1', 'mass_2']
        for mass in required_masses_without_dva:
            self.results_text.append(f"\nRaw results for {mass}:")
            if mass in results_without_dva:
                for key, value in results_without_dva[mass].items():
                    if isinstance(value, dict):
                        formatted_dict = {k: format_float(v) for k,v in value.items()}
                        self.results_text.append(f"{key}: {formatted_dict}")
                    else:
                        self.results_text.append(f"{key}: {format_float(value)}")
            else:
                self.results_text.append(f"No results for {mass}")

        self.results_text.append("\nComposite Measures (Without DVAs for Mass 1 and Mass 2):")
        if 'composite_measures' in results_without_dva:
            for mass, comp in results_without_dva['composite_measures'].items():
                if mass in ['mass_1', 'mass_2']:
                    self.results_text.append(f"{mass}: {format_float(comp)}")
        else:
            self.results_text.append("No composite measures found.")

        self.results_text.append("\nSingular Response (Without DVAs for Mass 1 and Mass 2):")
        if 'singular_response' in results_without_dva:
            self.results_text.append(f"{format_float(results_without_dva['singular_response'])}")
        else:
            self.results_text.append("No singular response found.")

        self.status_bar.showMessage("FRF Analysis Completed.")
        self.run_frf_button.setEnabled(True)
        self.run_sobol_button.setEnabled(True)
        self.run_ga_button.setEnabled(True)

        self.frf_combo.clear()
        self.frf_plots.clear()

        omega = np.linspace(self.omega_start_box.value(), self.omega_end_box.value(), self.omega_points_box.value())
        mass_labels = []
        for m_label in required_masses:
            if m_label in self.frf_results and 'magnitude' in self.frf_results[m_label]:
                mass_labels.append(m_label)

        # Plot with DVAs
        for m_label in mass_labels:
            fig = Figure(figsize=(6,4))
            ax = fig.add_subplot(111)
            mag = self.frf_results[m_label]['magnitude']
            if len(mag) == len(omega):
                ax.plot(omega, mag, label=m_label)
                ax.set_xlabel('Frequency (rad/s)')
                ax.set_ylabel('Amplitude')
                ax.set_title(f'FRF of {m_label} (With DVA)')
                ax.legend()
                ax.grid(True)
                self.frf_combo.addItem(f"{m_label} (With DVA)")
                self.frf_plots[f"{m_label} (With DVA)"] = fig
            else:
                QMessageBox.warning(self, "Plot Error", f"{m_label}: magnitude length does not match omega length.")

        # Plot combined with DVAs
        if mass_labels:
            fig_combined = Figure(figsize=(6,4))
            ax_combined = fig_combined.add_subplot(111)
            for m_label in mass_labels:
                mag = self.frf_results[m_label]['magnitude']
                if len(mag) == len(omega):
                    ax_combined.plot(omega, mag, label=m_label)
            ax_combined.set_xlabel('Frequency (rad/s)')
            ax_combined.set_ylabel('Amplitude')
            ax_combined.set_title('Combined FRF of All Masses (With DVAs)')
            ax_combined.grid(True)
            ax_combined.legend()
            self.frf_combo.addItem("All Masses Combined (With DVAs)")
            self.frf_plots["All Masses Combined (With DVAs)"] = fig_combined

        # Plot without DVAs for Mass1 and Mass2
        for m_label in ['mass_1', 'mass_2']:
            if m_label in results_without_dva and 'magnitude' in results_without_dva[m_label]:
                fig = Figure(figsize=(6,4))
                ax = fig.add_subplot(111)
                mag = results_without_dva[m_label]['magnitude']
                if len(mag) == len(omega):
                    ax.plot(omega, mag, label=f"{m_label} (Without DVA)", color='green')
                    ax.set_xlabel('Frequency (rad/s)')
                    ax.set_ylabel('Amplitude')
                    ax.set_title(f'FRF of {m_label} (Without DVA)')
                    ax.legend()
                    ax.grid(True)
                    self.frf_combo.addItem(f"{m_label} (Without DVA)")
                    self.frf_plots[f"{m_label} (Without DVA)"] = fig
                else:
                    QMessageBox.warning(self, "Plot Error", f"{m_label} (Without DVA): magnitude length does not match omega length.")

        # Plot combined with and without DVAs for Mass1 and Mass2
        fig_combined_with_without = Figure(figsize=(6,4))
        ax_combined_with_without = fig_combined_with_without.add_subplot(111)
        # Plot with DVA
        for m_label in ['mass_1', 'mass_2']:
            if m_label in self.frf_results and 'magnitude' in self.frf_results[m_label]:
                mag = self.frf_results[m_label]['magnitude']
                if len(mag) == len(omega):
                    ax_combined_with_without.plot(omega, mag, label=f"{m_label} (With DVA)")
        # Plot without DVA
        for m_label in ['mass_1', 'mass_2']:
            if m_label in results_without_dva and 'magnitude' in results_without_dva[m_label]:
                mag = results_without_dva[m_label]['magnitude']
                if len(mag) == len(omega):
                    ax_combined_with_without.plot(omega, mag, label=f"{m_label} (Without DVA)", linestyle='--')
        ax_combined_with_without.set_xlabel('Frequency (rad/s)')
        ax_combined_with_without.set_ylabel('Amplitude')
        ax_combined_with_without.set_title('FRF of Mass 1 \& 2: With and Without DVAs')
        ax_combined_with_without.grid(True)
        ax_combined_with_without.legend()
        self.frf_combo.addItem("Mass 1 \& 2: With and Without DVAs")
        self.frf_plots["Mass 1 \& 2: With and Without DVAs"] = fig_combined_with_without

        # Plot all masses combined with and without DVAs for Mass1 and Mass2
        fig_all_combined = Figure(figsize=(6,4))
        ax_all_combined = fig_all_combined.add_subplot(111)
        # With DVAs
        for m_label in mass_labels:
            mag = self.frf_results[m_label]['magnitude']
            if len(mag) == len(omega):
                ax_all_combined.plot(omega, mag, label=f"{m_label} (With DVA)")
        # Without DVAs for Mass1 and Mass2
        for m_label in ['mass_1', 'mass_2']:
            if m_label in results_without_dva and 'magnitude' in results_without_dva[m_label]:
                mag = results_without_dva[m_label]['magnitude']
                if len(mag) == len(omega):
                    ax_all_combined.plot(omega, mag, label=f"{m_label} (Without DVA)", linestyle='--')
        ax_all_combined.set_xlabel('Frequency (rad/s)')
        ax_all_combined.set_ylabel('Amplitude')
        ax_all_combined.set_title('Combined FRF of All Masses: With and Without DVAs for Mass 1 \& 2')
        ax_all_combined.grid(True)
        ax_all_combined.legend()
        self.frf_combo.addItem("All Masses Combined: With and Without DVAs for Mass 1 \& 2")
        self.frf_plots["All Masses Combined: With and Without DVAs for Mass 1 \& 2"] = fig_all_combined

        self.update_frf_plot()
        self.frf_canvas.draw()

##################################
# Sobol Results Handling
##################################

    def handle_sobol_error(self, err):
        self.sobol_results_text.append(f"Error running Sobol Analysis: {err}")
        self.status_bar.showMessage("Sobol Error encountered.")
        self.run_frf_button.setEnabled(True)
        self.run_sobol_button.setEnabled(True)
        self.run_ga_button.setEnabled(True)

    def handle_ga_finished(self, results, best_ind, parameter_names, best_fitness):
        self.run_ga_button.setEnabled(True)
        self.ga_results_text.append("\nGA Completed.\n")
        self.ga_results_text.append("Best individual parameters:")
        for name, val in zip(parameter_names, best_ind):
            self.ga_results_text.append(f"{name}: {val}")
        self.ga_results_text.append(f"\nBest fitness: {best_fitness:.6f}")

        singular_response = results.get('singular_response', None)
        if singular_response is not None:
            self.ga_results_text.append(f"\nSingular response of best individual: {singular_response}")

        self.ga_results_text.append("\nFull Results:")
        for section, data in results.items():
            self.ga_results_text.append(f"{section}: {data}")

    def handle_ga_error(self, err):
        self.run_ga_button.setEnabled(True)
        QMessageBox.warning(self, "GA Error", f"Error during GA: {err}")
        self.ga_results_text.append(f"\nError running GA: {err}")

    def handle_ga_update(self, msg):
        self.ga_results_text.append(msg)

    ##################################
    # Plotting Helpers
    ##################################

    def update_frf_plot(self):
        key = self.frf_combo.currentText()
        if key in self.frf_plots:
            fig = self.frf_plots[key]
            self.frf_canvas.figure = fig
            self.frf_canvas.draw()
        else:
            self.frf_canvas.figure.clear()
            self.frf_canvas.draw()

    def update_sobol_plot(self):
        key = self.sobol_combo.currentText()
        if key in self.sobol_plots:
            fig = self.sobol_plots[key]
            self.sobol_canvas.figure = fig
            self.sobol_canvas.draw()
        else:
            self.sobol_canvas.figure.clear()
            self.sobol_canvas.draw()

    def save_plot(self, fig, plot_type):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, f"Save {plot_type} Plot", "",
            "PNG Files (*.png);;JPEG Files (*.jpg);;PDF Files (*.pdf);;All Files (*)",
            options=options
        )
        if file_path:
            try:
                fig.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Success", f"{plot_type} plot saved to {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to save plot: {e}")

    def save_sobol_results(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Sobol Results", "",
                                                   "Text Files (*.txt);;All Files (*)", options=options)
        if file_path:
            try:
                with open(file_path,'w') as f:
                    f.write(self.sobol_results_text.toPlainText())
                QMessageBox.information(self,"Success",f"Sobol results saved to {file_path}")
            except Exception as e:
                QMessageBox.warning(self,"Error",f"Failed to save results: {e}")

    ##################################
    # Sobol Plots Generation
    ##################################

    def generate_sobol_plots(self, all_results, param_names):
        fig_last_run = self.visualize_last_run(all_results, param_names)
        self.sobol_combo.addItem("Last Run Results")
        self.sobol_plots["Last Run Results"] = fig_last_run

        fig_grouped_ST = self.visualize_grouped_bar_plot_sorted_on_ST(all_results, param_names)
        self.sobol_combo.addItem("Grouped Bar (Sorted by ST)")
        self.sobol_plots["Grouped Bar (Sorted by ST)"] = fig_grouped_ST

        conv_figs = self.visualize_convergence_plots(all_results, param_names)
        for i, cf in enumerate(conv_figs, start=1):
            name = f"Convergence Plots Fig {i}"
            self.sobol_combo.addItem(name)
            self.sobol_plots[name] = cf

        fig_heat = self.visualize_combined_heatmap(all_results, param_names)
        self.sobol_combo.addItem("Combined Heatmap")
        self.sobol_plots["Combined Heatmap"] = fig_heat

        fig_comp_radar = self.visualize_comprehensive_radar_plots(all_results, param_names)
        self.sobol_combo.addItem("Comprehensive Radar Plot")
        self.sobol_plots["Comprehensive Radar Plot"] = fig_comp_radar

        fig_s1_radar, fig_st_radar = self.visualize_separate_radar_plots(all_results, param_names)
        self.sobol_combo.addItem("Radar Plot S1")
        self.sobol_plots["Radar Plot S1"] = fig_s1_radar
        self.sobol_combo.addItem("Radar Plot ST")
        self.sobol_plots["Radar Plot ST"] = fig_st_radar

        fig_box = self.visualize_box_plots(all_results)
        self.sobol_combo.addItem("Box Plots")
        self.sobol_plots["Box Plots"] = fig_box

        fig_violin = self.visualize_violin_plots(all_results)
        self.sobol_combo.addItem("Violin Plots")
        self.sobol_plots["Violin Plots"] = fig_violin

        fig_scatter = self.visualize_scatter_S1_ST(all_results, param_names)
        self.sobol_combo.addItem("Scatter S1 vs ST")
        self.sobol_plots["Scatter S1 vs ST"] = fig_scatter

        fig_parallel = self.visualize_parallel_coordinates(all_results, param_names)
        self.sobol_combo.addItem("Parallel Coordinates")
        self.sobol_plots["Parallel Coordinates"] = fig_parallel

        fig_s1_hist, fig_st_hist = self.visualize_histograms(all_results)
        self.sobol_combo.addItem("S1 Histogram")
        self.sobol_plots["S1 Histogram"] = fig_s1_hist
        self.sobol_combo.addItem("ST Histogram")
        self.sobol_plots["ST Histogram"] = fig_st_hist

    ##################################
    # Sobol Visualization Methods
    ##################################

    def visualize_last_run(self, all_results, param_names):
        last_run_idx = -1
        S1_last_run = np.array(all_results['S1'][last_run_idx])
        ST_last_run = np.array(all_results['ST'][last_run_idx])
        sorted_indices_S1 = np.argsort(S1_last_run)[::-1]
        sorted_param_names_S1 = [param_names[i] for i in sorted_indices_S1]
        S1_sorted = S1_last_run[sorted_indices_S1]
        ST_sorted = ST_last_run[sorted_indices_S1]

        fig = Figure(figsize=(6,4))
        ax = fig.add_subplot(111)
        # Escape & with \&
        ax.bar(np.arange(len(sorted_param_names_S1)) - 0.175, S1_sorted, 0.35,
               label=r'$S_1$', color='skyblue')
        ax.bar(np.arange(len(sorted_param_names_S1)) + 0.175, ST_sorted, 0.35,
               label=r'$S_T$', color='salmon')
        ax.set_xlabel('Parameters', fontsize=20)
        ax.set_ylabel('Sensitivity Index', fontsize=20)
        # Fix the LaTeX & here:
        ax.set_title('First-order ($S_1$) \\& Total-order ($S_T$)', fontsize=24)
        ax.set_xticks(np.arange(len(sorted_param_names_S1)))
        ax.set_xticklabels([format_parameter_name(p) for p in sorted_param_names_S1],
                           rotation=90, fontsize=12)
        ax.legend(fontsize=16)
        fig.tight_layout()
        return fig

    def visualize_grouped_bar_plot_sorted_on_ST(self, all_results, param_names):
        last_run_idx = -1
        S1_last_run = np.array(all_results['S1'][last_run_idx])
        ST_last_run = np.array(all_results['ST'][last_run_idx])
        sorted_indices_ST = np.argsort(ST_last_run)[::-1]
        sorted_param_names_ST = [param_names[i] for i in sorted_indices_ST]
        S1_sorted = S1_last_run[sorted_indices_ST]
        ST_sorted = ST_last_run[sorted_indices_ST]

        fig = Figure(figsize=(6,4))
        ax = fig.add_subplot(111)
        ax.bar(np.arange(len(sorted_param_names_ST)) - 0.175, S1_sorted, 0.35,
               label=r'$S_1$', color='skyblue')
        ax.bar(np.arange(len(sorted_param_names_ST)) + 0.175, ST_sorted, 0.35,
               label=r'$S_T$', color='salmon')
        ax.set_xlabel('Parameters', fontsize=20)
        ax.set_ylabel('Sensitivity Index', fontsize=20)
        # Fix the LaTeX & here:
        ax.set_title('First-order ($S_1$) \\& Total-order ($S_T$) - Sorted by $S_T$', fontsize=24)
        ax.set_xticks(np.arange(len(sorted_param_names_ST)))
        ax.set_xticklabels([format_parameter_name(p) for p in sorted_param_names_ST],
                           rotation=90, fontsize=12)
        ax.legend(fontsize=16)
        fig.tight_layout()
        return fig

    def visualize_convergence_plots(self, all_results, param_names):
        sample_sizes = all_results['samples']
        S1_matrix = np.array(all_results['S1'])
        ST_matrix = np.array(all_results['ST'])

        plots_per_fig = 12
        total_params = len(param_names)
        num_figs = int(np.ceil(total_params / plots_per_fig))
        figs = []
        for fig_idx in range(num_figs):
            fig = Figure(figsize=(25,20))
            start_idx = fig_idx*plots_per_fig
            end_idx = min(start_idx+plots_per_fig, total_params)
            for subplot_idx, param_idx in enumerate(range(start_idx,end_idx)):
                param = param_names[param_idx]
                ax = fig.add_subplot(3,4,subplot_idx+1)
                S1_values = S1_matrix[:, param_idx]
                ST_values = ST_matrix[:, param_idx]

                ax.plot(sample_sizes, S1_values, marker='o', linestyle='-', color='tab:blue', label=r'$S_1$')
                ax.plot(sample_sizes, ST_values, marker='s', linestyle='-', color='tab:red', label=r'$S_T$')
                ax.set_xlabel('Sample Size', fontsize=12)
                ax.set_ylabel('Sensitivity Index', fontsize=12)
                ax.set_title(f'Convergence {format_parameter_name(param)}', fontsize=14)
                ax.legend(fontsize=10)
                ax.grid(True, ls="--", linewidth=0.5)
            fig.tight_layout()
            figs.append(fig)
        return figs

    def visualize_combined_heatmap(self, all_results, param_names):
        last_run_idx = -1
        S1_last = np.array(all_results['S1'][last_run_idx])
        ST_last = np.array(all_results['ST'][last_run_idx])
        df = pd.DataFrame({'Parameter': param_names, 'S1': S1_last, 'ST': ST_last}).set_index('Parameter')
        df_sorted = df.sort_values('S1', ascending=False)
        fig = Figure(figsize=(20,max(8,len(param_names)*0.3)))
        ax = fig.add_subplot(111)
        sns.heatmap(df_sorted, annot=True, cmap='coolwarm',
                    cbar_kws={'label': 'Sensitivity Index'}, linewidths=.5,
                    linecolor='gray', ax=ax)
        ax.set_title('Combined Heatmap (S1 \\& ST)', fontsize=24)
        ax.set_xlabel('Sensitivity Indices', fontsize=20)
        ax.set_ylabel('Parameters', fontsize=20)
        return fig

    def visualize_comprehensive_radar_plots(self, all_results, param_names):
        last_run_idx = -1
        S1 = np.array(all_results['S1'][last_run_idx])
        ST = np.array(all_results['ST'][last_run_idx])
        num_vars = len(param_names)
        angles = np.linspace(0,2*np.pi,num_vars,endpoint=False).tolist()
        angles += angles[:1]

        fig = Figure(figsize=(30,30))
        ax = fig.add_subplot(111, polar=True)
        max_val = max(np.max(S1), np.max(ST))*1.1
        ax.set_ylim(0, max_val)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([format_parameter_name(p) for p in param_names], fontsize=14)
        values_S1 = list(S1)+[S1[0]]
        values_ST = list(ST)+[ST[0]]
        ax.plot(angles, values_S1, linewidth=3, label=r'$S_1$')
        ax.fill(angles, values_S1, alpha=0.25, color='skyblue')
        ax.plot(angles, values_ST, linewidth=3, label=r'$S_T$')
        ax.fill(angles, values_ST, alpha=0.25, color='salmon')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3,1.1), fontsize=20)
        ax.set_title('Comprehensive Radar Plot', fontsize=28, y=1.05)
        fig.tight_layout()
        return fig

    def visualize_separate_radar_plots(self, all_results, param_names):
        last_run_idx = -1
        S1 = np.array(all_results['S1'][last_run_idx])
        ST = np.array(all_results['ST'][last_run_idx])
        num_vars = len(param_names)
        angles = np.linspace(0,2*np.pi,num_vars,endpoint=False).tolist()
        angles += angles[:1]

        fig_s1 = Figure(figsize=(30,30))
        ax_s1 = fig_s1.add_subplot(111, polar=True)
        values_S1 = list(S1)+[S1[0]]
        max_val_s1 = np.max(S1)*1.1
        ax_s1.set_ylim(0, max_val_s1)
        ax_s1.set_xticks(angles[:-1])
        ax_s1.set_xticklabels([format_parameter_name(p) for p in param_names], fontsize=14)
        ax_s1.plot(angles, values_S1, linewidth=3, label=r'$S_1$')
        ax_s1.fill(angles, values_S1, alpha=0.25, color='skyblue')
        ax_s1.set_title('Radar Plot of First-order Sensitivity Indices (S1)', fontsize=28, y=1.05)
        ax_s1.legend(loc='upper right', bbox_to_anchor=(1.3,1.1), fontsize=20)
        fig_s1.tight_layout()

        fig_st = Figure(figsize=(30,30))
        ax_st = fig_st.add_subplot(111, polar=True)
        values_ST = list(ST)+[ST[0]]
        max_val_st = np.max(ST)*1.1
        ax_st.set_ylim(0, max_val_st)
        ax_st.set_xticks(angles[:-1])
        ax_st.set_xticklabels([format_parameter_name(p) for p in param_names], fontsize=14)
        ax_st.plot(angles, values_ST, linewidth=3, label=r'$S_T$')
        ax_st.fill(angles, values_ST, alpha=0.25, color='salmon')
        ax_st.set_title('Radar Plot of Total-order Sensitivity Indices (ST)', fontsize=28, y=1.05)
        ax_st.legend(loc='upper right', bbox_to_anchor=(1.3,1.1), fontsize=20)
        fig_st.tight_layout()

        return fig_s1, fig_st

    def visualize_box_plots(self, all_results):
        data = {
            'S1': np.concatenate(all_results['S1']),
            'ST': np.concatenate(all_results['ST'])
        }
        df = pd.DataFrame(data)
        fig = Figure(figsize=(16,12))
        ax = fig.add_subplot(111)
        sns.boxplot(data=df, palette=['skyblue','salmon'], ax=ax)
        ax.set_xlabel('Sensitivity Indices', fontsize=22)
        ax.set_ylabel('Values', fontsize=22)
        # Fix the LaTeX & here:
        ax.set_title('Box Plots of S1 \\& ST', fontsize=26)
        fig.tight_layout()
        return fig

    def visualize_violin_plots(self, all_results):
        data = {
            'S1': np.concatenate(all_results['S1']),
            'ST': np.concatenate(all_results['ST'])
        }
        df = pd.DataFrame(data)
        fig = Figure(figsize=(16,12))
        ax = fig.add_subplot(111)
        sns.violinplot(data=df, palette=['skyblue','salmon'], inner='quartile', ax=ax)
        ax.set_xlabel('Sensitivity Indices', fontsize=22)
        ax.set_ylabel('Values', fontsize=22)
        # Fix the LaTeX & here:
        ax.set_title('Violin Plots of S1 \\& ST', fontsize=26)
        fig.tight_layout()
        return fig

    def visualize_scatter_S1_ST(self, all_results, param_names):
        last_run_idx = -1
        S1_last_run = np.array(all_results['S1'][last_run_idx])
        ST_last_run = np.array(all_results['ST'][last_run_idx])
        fig = Figure(figsize=(18,14))
        ax = fig.add_subplot(111)
        scatter = ax.scatter(S1_last_run, ST_last_run, c=np.arange(len(param_names)),
                             cmap='tab20', edgecolor='k', s=200)
        for i, param in enumerate(param_names):
            ax.text(S1_last_run[i]+0.001, ST_last_run[i]+0.001,
                    format_parameter_name(param), fontsize=12)
        ax.set_xlabel(r'$S_1$', fontsize=22)
        ax.set_ylabel(r'$S_T$', fontsize=22)
        ax.set_title('Scatter Plot of S1 vs ST', fontsize=26)
        ax.grid(True)
        fig.tight_layout()
        return fig

    def visualize_parallel_coordinates(self, all_results, param_names):
        data = []
        for run_idx, num_samples in enumerate(all_results['samples']):
            row = {'Sample Size': num_samples}
            for param_idx, param in enumerate(param_names):
                row[f'S1_{param}'] = all_results['S1'][run_idx][param_idx]
                row[f'ST_{param}'] = all_results['ST'][run_idx][param_idx]
            data.append(row)
        df = pd.DataFrame(data)
        fig = Figure(figsize=(25,20))
        ax = fig.add_subplot(111)
        for param in param_names:
            ax.plot(df['Sample Size'], df[f'S1_{param}'],
                    label=f'S1 {format_parameter_name(param)}',
                    linestyle='-', marker='o', alpha=0.6)
            ax.plot(df['Sample Size'], df[f'ST_{param}'],
                    label=f'ST {format_parameter_name(param)}',
                    linestyle='--', marker='s', alpha=0.6)
        ax.set_xlabel('Sample Size', fontsize=22)
        ax.set_ylabel('Sensitivity Index', fontsize=22)
        # Fix the LaTeX & here:
        ax.set_title('Parallel Coordinates (S1 \\& ST)', fontsize=28)
        ax.grid(True)
        fig.tight_layout()
        return fig

    def visualize_histograms(self, all_results):
        last_run_idx = -1
        S1_last_run = np.array(all_results['S1'][last_run_idx])
        ST_last_run = np.array(all_results['ST'][last_run_idx])

        fig_s1 = Figure(figsize=(18,12))
        ax_s1 = fig_s1.add_subplot(111)
        sns.histplot(S1_last_run, bins=30, kde=True, color='skyblue', ax=ax_s1)
        ax_s1.set_xlabel(r'$S_1$', fontsize=22)
        ax_s1.set_ylabel('Frequency', fontsize=22)
        ax_s1.set_title('Histogram of S1', fontsize=26)
        fig_s1.tight_layout()

        fig_st = Figure(figsize=(18,12))
        ax_st = fig_st.add_subplot(111)
        sns.histplot(ST_last_run, bins=30, kde=True, color='salmon', ax=ax_st)
        ax_st.set_xlabel(r'$S_T$', fontsize=22)
        ax_st.set_ylabel('Frequency', fontsize=22)
        ax_st.set_title('Histogram of $S_T$', fontsize=26)
        fig_st.tight_layout()

        return fig_s1, fig_st

    ##################################
    # Toggles and Inputs
    ##################################

    def toggle_fixed(self, state, row):
        fixed = (state == Qt.Checked)
        fixed_value_spin = self.dva_param_table.cellWidget(row, 2)
        lower_bound_spin = self.dva_param_table.cellWidget(row, 3)
        upper_bound_spin = self.dva_param_table.cellWidget(row, 4)

        fixed_value_spin.setEnabled(fixed)
        lower_bound_spin.setEnabled(not fixed)
        upper_bound_spin.setEnabled(not fixed)

    def get_target_values_weights(self):
        target_values_dict = {}
        weights_dict = {}

        for mass_num in range(1,6):
            t_dict = {}
            w_dict = {}
            for peak_num in range(1,5):
                pv_key = f"peak_value_{peak_num}_m{mass_num}"
                if pv_key in self.mass_target_spins:
                    t_dict[f"peak_value_{peak_num}"] = self.mass_target_spins[pv_key].value()
                    w_dict[f"peak_value_{peak_num}"] = self.mass_weight_spins[pv_key].value()

            for i in range(1,5):
                for j in range(i+1,5):
                    bw_key = f"bandwidth_{i}_{j}_m{mass_num}"
                    if bw_key in self.mass_target_spins:
                        t_dict[f"bandwidth_{i}_{j}"] = self.mass_target_spins[bw_key].value()
                        w_dict[f"bandwidth_{i}_{j}"] = self.mass_weight_spins[bw_key].value()

            for i in range(1,5):
                for j in range(i+1,5):
                    slope_key = f"slope_{i}_{j}_m{mass_num}"
                    if slope_key in self.mass_target_spins:
                        t_dict[f"slope_{i}_{j}"] = self.mass_target_spins[slope_key].value()
                        w_dict[f"slope_{i}_{j}"] = self.mass_weight_spins[slope_key].value()

            auc_key = f"area_under_curve_m{mass_num}"
            if auc_key in self.mass_target_spins:
                t_dict["area_under_curve"] = self.mass_target_spins[auc_key].value()
                w_dict["area_under_curve"] = self.mass_weight_spins[auc_key].value()

            target_values_dict[f"mass_{mass_num}"] = t_dict
            weights_dict[f"mass_{mass_num}"] = w_dict

        return target_values_dict, weights_dict

    def get_num_samples_list(self):
        text = self.num_samples_line.text().strip()
        if not text:
            return [32,64,128]
        try:
            parts = text.split(',')
            nums = [int(p.strip()) for p in parts if p.strip().isdigit()]
            if not nums:
                raise ValueError("No valid integers found.")
            return nums
        except Exception:
            QMessageBox.warning(self, "Input Error", "Invalid Sample Sizes Input. Using default [32,64,128].")
            return [32,64,128]

##################################
# Sobol Results Handling
##################################

    def display_sobol_results(self, all_results, warnings):
        self.sobol_results = all_results
        self.sobol_warnings = warnings
        self.sobol_results_text.append("\n--- Sobol Sensitivity Analysis Results ---")

        original_dva_parameter_order = [
            'beta_1','beta_2','beta_3','beta_4','beta_5','beta_6',
            'beta_7','beta_8','beta_9','beta_10','beta_11','beta_12',
            'beta_13','beta_14','beta_15',
            'lambda_1','lambda_2','lambda_3','lambda_4','lambda_5',
            'lambda_6','lambda_7','lambda_8','lambda_9','lambda_10',
            'lambda_11','lambda_12','lambda_13','lambda_14','lambda_15',
            'mu_1','mu_2','mu_3',
            'nu_1','nu_2','nu_3','nu_4','nu_5','nu_6',
            'nu_7','nu_8','nu_9','nu_10','nu_11','nu_12',
            'nu_13','nu_14','nu_15'
        ]
        param_names = original_dva_parameter_order

        def format_float(val):
            if isinstance(val,(np.float64,float,int)):
                return f"{val:.6f}"
            return str(val)

        for run_idx, num_samples in enumerate(all_results['samples']):
            self.sobol_results_text.append(f"\nSample Size: {num_samples}")
            S1 = all_results['S1'][run_idx]
            ST = all_results['ST'][run_idx]
            self.sobol_results_text.append(f"  Length of S1: {len(S1)}, Length of ST: {len(ST)}")
            for param_index, param_name in enumerate(param_names):
                if param_index < len(S1) and param_index < len(ST):
                    s1_val = S1[param_index]
                    st_val = ST[param_index]
                    self.sobol_results_text.append(f"Parameter {param_name}: S1 = {s1_val:.6f}, ST = {st_val:.6f}")
                else:
                    self.sobol_results_text.append(f"IndexError: Parameter {param_name} out of range")

        if warnings:
            self.sobol_results_text.append("\nWarnings:")
            for w in warnings:
                self.sobol_results_text.append(w)
        else:
            self.sobol_results_text.append("\nNo warnings encountered.")

        self.status_bar.showMessage("Sobol Analysis Completed.")
        self.run_frf_button.setEnabled(True)
        self.run_sobol_button.setEnabled(True)
        self.run_ga_button.setEnabled(True)

        self.sobol_combo.clear()
        self.sobol_plots.clear()

        self.generate_sobol_plots(all_results, param_names)
        self.update_sobol_plot()
        self.sobol_canvas.draw()

    ##################################
    # GA Results Handling
    ##################################

    def handle_ga_finished(self, results, best_ind, parameter_names, best_fitness):
        self.run_ga_button.setEnabled(True)
        self.ga_results_text.append("\nGA Completed.\n")
        self.ga_results_text.append("Best individual parameters:")
        for name, val in zip(parameter_names, best_ind):
            self.ga_results_text.append(f"{name}: {val}")
        self.ga_results_text.append(f"\nBest fitness: {best_fitness:.6f}")

        singular_response = results.get('singular_response', None)
        if singular_response is not None:
            self.ga_results_text.append(f"\nSingular response of best individual: {singular_response}")

        self.ga_results_text.append("\nFull Results:")
        for section, data in results.items():
            self.ga_results_text.append(f"{section}: {data}")

    def handle_ga_error(self, err):
        self.run_ga_button.setEnabled(True)
        QMessageBox.warning(self, "GA Error", f"Error during GA: {err}")
        self.ga_results_text.append(f"\nError running GA: {err}")

    def handle_ga_update(self, msg):
        self.ga_results_text.append(msg)

    ##################################
    # Plotting Helpers
    ##################################

    def update_frf_plot(self):
        key = self.frf_combo.currentText()
        if key in self.frf_plots:
            fig = self.frf_plots[key]
            self.frf_canvas.figure = fig
            self.frf_canvas.draw()
        else:
            self.frf_canvas.figure.clear()
            self.frf_canvas.draw()

    def update_sobol_plot(self):
        key = self.sobol_combo.currentText()
        if key in self.sobol_plots:
            fig = self.sobol_plots[key]
            self.sobol_canvas.figure = fig
            self.sobol_canvas.draw()
        else:
            self.sobol_canvas.figure.clear()
            self.sobol_canvas.draw()

    def save_plot(self, fig, plot_type):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(
            self, f"Save {plot_type} Plot", "",
            "PNG Files (*.png);;JPEG Files (*.jpg);;PDF Files (*.pdf);;All Files (*)",
            options=options
        )
        if file_path:
            try:
                fig.savefig(file_path, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Success", f"{plot_type} plot saved to {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to save plot: {e}")

    def save_sobol_results(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Sobol Results", "",
                                                   "Text Files (*.txt);;All Files (*)", options=options)
        if file_path:
            try:
                with open(file_path,'w') as f:
                    f.write(self.sobol_results_text.toPlainText())
                QMessageBox.information(self,"Success",f"Sobol results saved to {file_path}")
            except Exception as e:
                QMessageBox.warning(self,"Error",f"Failed to save results: {e}")

    ##################################
    # Sobol Plots Generation
    ##################################

    def generate_sobol_plots(self, all_results, param_names):
        fig_last_run = self.visualize_last_run(all_results, param_names)
        self.sobol_combo.addItem("Last Run Results")
        self.sobol_plots["Last Run Results"] = fig_last_run

        fig_grouped_ST = self.visualize_grouped_bar_plot_sorted_on_ST(all_results, param_names)
        self.sobol_combo.addItem("Grouped Bar (Sorted by ST)")
        self.sobol_plots["Grouped Bar (Sorted by ST)"] = fig_grouped_ST

        conv_figs = self.visualize_convergence_plots(all_results, param_names)
        for i, cf in enumerate(conv_figs, start=1):
            name = f"Convergence Plots Fig {i}"
            self.sobol_combo.addItem(name)
            self.sobol_plots[name] = cf

        fig_heat = self.visualize_combined_heatmap(all_results, param_names)
        self.sobol_combo.addItem("Combined Heatmap")
        self.sobol_plots["Combined Heatmap"] = fig_heat

        fig_comp_radar = self.visualize_comprehensive_radar_plots(all_results, param_names)
        self.sobol_combo.addItem("Comprehensive Radar Plot")
        self.sobol_plots["Comprehensive Radar Plot"] = fig_comp_radar

        fig_s1_radar, fig_st_radar = self.visualize_separate_radar_plots(all_results, param_names)
        self.sobol_combo.addItem("Radar Plot S1")
        self.sobol_plots["Radar Plot S1"] = fig_s1_radar
        self.sobol_combo.addItem("Radar Plot ST")
        self.sobol_plots["Radar Plot ST"] = fig_st_radar

        fig_box = self.visualize_box_plots(all_results)
        self.sobol_combo.addItem("Box Plots")
        self.sobol_plots["Box Plots"] = fig_box

        fig_violin = self.visualize_violin_plots(all_results)
        self.sobol_combo.addItem("Violin Plots")
        self.sobol_plots["Violin Plots"] = fig_violin

        fig_scatter = self.visualize_scatter_S1_ST(all_results, param_names)
        self.sobol_combo.addItem("Scatter S1 vs ST")
        self.sobol_plots["Scatter S1 vs ST"] = fig_scatter

        fig_parallel = self.visualize_parallel_coordinates(all_results, param_names)
        self.sobol_combo.addItem("Parallel Coordinates")
        self.sobol_plots["Parallel Coordinates"] = fig_parallel

        fig_s1_hist, fig_st_hist = self.visualize_histograms(all_results)
        self.sobol_combo.addItem("S1 Histogram")
        self.sobol_plots["S1 Histogram"] = fig_s1_hist
        self.sobol_combo.addItem("ST Histogram")
        self.sobol_plots["ST Histogram"] = fig_st_hist

    ##################################
    # Sobol Visualization Methods
    ##################################

    def visualize_last_run(self, all_results, param_names):
        last_run_idx = -1
        S1_last_run = np.array(all_results['S1'][last_run_idx])
        ST_last_run = np.array(all_results['ST'][last_run_idx])
        sorted_indices_S1 = np.argsort(S1_last_run)[::-1]
        sorted_param_names_S1 = [param_names[i] for i in sorted_indices_S1]
        S1_sorted = S1_last_run[sorted_indices_S1]
        ST_sorted = ST_last_run[sorted_indices_S1]

        fig = Figure(figsize=(6,4))
        ax = fig.add_subplot(111)
        # Escape & with \&
        ax.bar(np.arange(len(sorted_param_names_S1)) - 0.175, S1_sorted, 0.35,
               label=r'$S_1$', color='skyblue')
        ax.bar(np.arange(len(sorted_param_names_S1)) + 0.175, ST_sorted, 0.35,
               label=r'$S_T$', color='salmon')
        ax.set_xlabel('Parameters', fontsize=20)
        ax.set_ylabel('Sensitivity Index', fontsize=20)
        # Fix the LaTeX & here:
        ax.set_title('First-order ($S_1$) \\& Total-order ($S_T$)', fontsize=24)
        ax.set_xticks(np.arange(len(sorted_param_names_S1)))
        ax.set_xticklabels([format_parameter_name(p) for p in sorted_param_names_S1],
                           rotation=90, fontsize=12)
        ax.legend(fontsize=16)
        fig.tight_layout()
        return fig

    def visualize_grouped_bar_plot_sorted_on_ST(self, all_results, param_names):
        last_run_idx = -1
        S1_last_run = np.array(all_results['S1'][last_run_idx])
        ST_last_run = np.array(all_results['ST'][last_run_idx])
        sorted_indices_ST = np.argsort(ST_last_run)[::-1]
        sorted_param_names_ST = [param_names[i] for i in sorted_indices_ST]
        S1_sorted = S1_last_run[sorted_indices_ST]
        ST_sorted = ST_last_run[sorted_indices_ST]

        fig = Figure(figsize=(6,4))
        ax = fig.add_subplot(111)
        ax.bar(np.arange(len(sorted_param_names_ST)) - 0.175, S1_sorted, 0.35,
               label=r'$S_1$', color='skyblue')
        ax.bar(np.arange(len(sorted_param_names_ST)) + 0.175, ST_sorted, 0.35,
               label=r'$S_T$', color='salmon')
        ax.set_xlabel('Parameters', fontsize=20)
        ax.set_ylabel('Sensitivity Index', fontsize=20)
        # Fix the LaTeX & here:
        ax.set_title('First-order ($S_1$) \\& Total-order ($S_T$) - Sorted by $S_T$', fontsize=24)
        ax.set_xticks(np.arange(len(sorted_param_names_ST)))
        ax.set_xticklabels([format_parameter_name(p) for p in sorted_param_names_ST],
                           rotation=90, fontsize=12)
        ax.legend(fontsize=16)
        fig.tight_layout()
        return fig

    def visualize_convergence_plots(self, all_results, param_names):
        sample_sizes = all_results['samples']
        S1_matrix = np.array(all_results['S1'])
        ST_matrix = np.array(all_results['ST'])

        plots_per_fig = 12
        total_params = len(param_names)
        num_figs = int(np.ceil(total_params / plots_per_fig))
        figs = []
        for fig_idx in range(num_figs):
            fig = Figure(figsize=(25,20))
            start_idx = fig_idx*plots_per_fig
            end_idx = min(start_idx+plots_per_fig, total_params)
            for subplot_idx, param_idx in enumerate(range(start_idx,end_idx)):
                param = param_names[param_idx]
                ax = fig.add_subplot(3,4,subplot_idx+1)
                S1_values = S1_matrix[:, param_idx]
                ST_values = ST_matrix[:, param_idx]

                ax.plot(sample_sizes, S1_values, marker='o', linestyle='-', color='tab:blue', label=r'$S_1$')
                ax.plot(sample_sizes, ST_values, marker='s', linestyle='-', color='tab:red', label=r'$S_T$')
                ax.set_xlabel('Sample Size', fontsize=12)
                ax.set_ylabel('Sensitivity Index', fontsize=12)
                ax.set_title(f'Convergence {format_parameter_name(param)}', fontsize=14)
                ax.legend(fontsize=10)
                ax.grid(True, ls="--", linewidth=0.5)
            fig.tight_layout()
            figs.append(fig)
        return figs

    def visualize_combined_heatmap(self, all_results, param_names):
        last_run_idx = -1
        S1_last = np.array(all_results['S1'][last_run_idx])
        ST_last = np.array(all_results['ST'][last_run_idx])
        df = pd.DataFrame({'Parameter': param_names, 'S1': S1_last, 'ST': ST_last}).set_index('Parameter')
        df_sorted = df.sort_values('S1', ascending=False)
        fig = Figure(figsize=(20,max(8,len(param_names)*0.3)))
        ax = fig.add_subplot(111)
        sns.heatmap(df_sorted, annot=True, cmap='coolwarm',
                    cbar_kws={'label': 'Sensitivity Index'}, linewidths=.5,
                    linecolor='gray', ax=ax)
        ax.set_title('Combined Heatmap (S1 \\& ST)', fontsize=24)
        ax.set_xlabel('Sensitivity Indices', fontsize=20)
        ax.set_ylabel('Parameters', fontsize=20)
        return fig

    def visualize_comprehensive_radar_plots(self, all_results, param_names):
        last_run_idx = -1
        S1 = np.array(all_results['S1'][last_run_idx])
        ST = np.array(all_results['ST'][last_run_idx])
        num_vars = len(param_names)
        angles = np.linspace(0,2*np.pi,num_vars,endpoint=False).tolist()
        angles += angles[:1]

        fig = Figure(figsize=(30,30))
        ax = fig.add_subplot(111, polar=True)
        max_val = max(np.max(S1), np.max(ST))*1.1
        ax.set_ylim(0, max_val)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([format_parameter_name(p) for p in param_names], fontsize=14)
        values_S1 = list(S1)+[S1[0]]
        values_ST = list(ST)+[ST[0]]
        ax.plot(angles, values_S1, linewidth=3, label=r'$S_1$')
        ax.fill(angles, values_S1, alpha=0.25, color='skyblue')
        ax.plot(angles, values_ST, linewidth=3, label=r'$S_T$')
        ax.fill(angles, values_ST, alpha=0.25, color='salmon')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3,1.1), fontsize=20)
        ax.set_title('Comprehensive Radar Plot', fontsize=28, y=1.05)
        fig.tight_layout()
        return fig

    def visualize_separate_radar_plots(self, all_results, param_names):
        last_run_idx = -1
        S1 = np.array(all_results['S1'][last_run_idx])
        ST = np.array(all_results['ST'][last_run_idx])
        num_vars = len(param_names)
        angles = np.linspace(0,2*np.pi,num_vars,endpoint=False).tolist()
        angles += angles[:1]

        fig_s1 = Figure(figsize=(30,30))
        ax_s1 = fig_s1.add_subplot(111, polar=True)
        values_S1 = list(S1)+[S1[0]]
        max_val_s1 = np.max(S1)*1.1
        ax_s1.set_ylim(0, max_val_s1)
        ax_s1.set_xticks(angles[:-1])
        ax_s1.set_xticklabels([format_parameter_name(p) for p in param_names], fontsize=14)
        ax_s1.plot(angles, values_S1, linewidth=3, label=r'$S_1$')
        ax_s1.fill(angles, values_S1, alpha=0.25, color='skyblue')
        ax_s1.set_title('Radar Plot of First-order Sensitivity Indices (S1)', fontsize=28, y=1.05)
        ax_s1.legend(loc='upper right', bbox_to_anchor=(1.3,1.1), fontsize=20)
        fig_s1.tight_layout()

        fig_st = Figure(figsize=(30,30))
        ax_st = fig_st.add_subplot(111, polar=True)
        values_ST = list(ST)+[ST[0]]
        max_val_st = np.max(ST)*1.1
        ax_st.set_ylim(0, max_val_st)
        ax_st.set_xticks(angles[:-1])
        ax_st.set_xticklabels([format_parameter_name(p) for p in param_names], fontsize=14)
        ax_st.plot(angles, values_ST, linewidth=3, label=r'$S_T$')
        ax_st.fill(angles, values_ST, alpha=0.25, color='salmon')
        ax_st.set_title('Radar Plot of Total-order Sensitivity Indices (ST)', fontsize=28, y=1.05)
        ax_st.legend(loc='upper right', bbox_to_anchor=(1.3,1.1), fontsize=20)
        fig_st.tight_layout()

        return fig_s1, fig_st

    def visualize_box_plots(self, all_results):
        data = {
            'S1': np.concatenate(all_results['S1']),
            'ST': np.concatenate(all_results['ST'])
        }
        df = pd.DataFrame(data)
        fig = Figure(figsize=(16,12))
        ax = fig.add_subplot(111)
        sns.boxplot(data=df, palette=['skyblue','salmon'], ax=ax)
        ax.set_xlabel('Sensitivity Indices', fontsize=22)
        ax.set_ylabel('Values', fontsize=22)
        # Fix the LaTeX & here:
        ax.set_title('Box Plots of S1 \\& ST', fontsize=26)
        fig.tight_layout()
        return fig

    def visualize_violin_plots(self, all_results):
        data = {
            'S1': np.concatenate(all_results['S1']),
            'ST': np.concatenate(all_results['ST'])
        }
        df = pd.DataFrame(data)
        fig = Figure(figsize=(16,12))
        ax = fig.add_subplot(111)
        sns.violinplot(data=df, palette=['skyblue','salmon'], inner='quartile', ax=ax)
        ax.set_xlabel('Sensitivity Indices', fontsize=22)
        ax.set_ylabel('Values', fontsize=22)
        # Fix the LaTeX & here:
        ax.set_title('Violin Plots of S1 \\& ST', fontsize=26)
        fig.tight_layout()
        return fig

    def visualize_scatter_S1_ST(self, all_results, param_names):
        last_run_idx = -1
        S1_last_run = np.array(all_results['S1'][last_run_idx])
        ST_last_run = np.array(all_results['ST'][last_run_idx])
        fig = Figure(figsize=(18,14))
        ax = fig.add_subplot(111)
        scatter = ax.scatter(S1_last_run, ST_last_run, c=np.arange(len(param_names)),
                             cmap='tab20', edgecolor='k', s=200)
        for i, param in enumerate(param_names):
            ax.text(S1_last_run[i]+0.001, ST_last_run[i]+0.001,
                    format_parameter_name(param), fontsize=12)
        ax.set_xlabel(r'$S_1$', fontsize=22)
        ax.set_ylabel(r'$S_T$', fontsize=22)
        ax.set_title('Scatter Plot of S1 vs ST', fontsize=26)
        ax.grid(True)
        fig.tight_layout()
        return fig

    def visualize_parallel_coordinates(self, all_results, param_names):
        data = []
        for run_idx, num_samples in enumerate(all_results['samples']):
            row = {'Sample Size': num_samples}
            for param_idx, param in enumerate(param_names):
                row[f'S1_{param}'] = all_results['S1'][run_idx][param_idx]
                row[f'ST_{param}'] = all_results['ST'][run_idx][param_idx]
            data.append(row)
        df = pd.DataFrame(data)
        fig = Figure(figsize=(25,20))
        ax = fig.add_subplot(111)
        for param in param_names:
            ax.plot(df['Sample Size'], df[f'S1_{param}'],
                    label=f'S1 {format_parameter_name(param)}',
                    linestyle='-', marker='o', alpha=0.6)
            ax.plot(df['Sample Size'], df[f'ST_{param}'],
                    label=f'ST {format_parameter_name(param)}',
                    linestyle='--', marker='s', alpha=0.6)
        ax.set_xlabel('Sample Size', fontsize=22)
        ax.set_ylabel('Sensitivity Index', fontsize=22)
        # Fix the LaTeX & here:
        ax.set_title('Parallel Coordinates (S1 \\& ST)', fontsize=28)
        ax.grid(True)
        fig.tight_layout()
        return fig

    def visualize_histograms(self, all_results):
        last_run_idx = -1
        S1_last_run = np.array(all_results['S1'][last_run_idx])
        ST_last_run = np.array(all_results['ST'][last_run_idx])

        fig_s1 = Figure(figsize=(18,12))
        ax_s1 = fig_s1.add_subplot(111)
        sns.histplot(S1_last_run, bins=30, kde=True, color='skyblue', ax=ax_s1)
        ax_s1.set_xlabel(r'$S_1$', fontsize=22)
        ax_s1.set_ylabel('Frequency', fontsize=22)
        ax_s1.set_title('Histogram of S1', fontsize=26)
        fig_s1.tight_layout()

        fig_st = Figure(figsize=(18,12))
        ax_st = fig_st.add_subplot(111)
        sns.histplot(ST_last_run, bins=30, kde=True, color='salmon', ax=ax_st)
        ax_st.set_xlabel(r'$S_T$', fontsize=22)
        ax_st.set_ylabel('Frequency', fontsize=22)
        ax_st.set_title('Histogram of $S_T$', fontsize=26)
        fig_st.tight_layout()

        return fig_s1, fig_st

    ##################################
    # Running FRF / Sobol / GA
    ##################################

    def run_frf(self):
        if self.omega_start_box.value() >= self.omega_end_box.value():
            QMessageBox.warning(self, "Input Error", "Ω Start must be less than Ω End.")
            return

        target_values, weights = self.get_target_values_weights()

        self.run_frf_button.setEnabled(False)
        self.run_sobol_button.setEnabled(False)
        self.run_ga_button.setEnabled(False)
        self.results_text.append("\n--- Running FRF Analysis ---\n")
        self.status_bar.showMessage("Running FRF Analysis...")

        main_params = (
            self.mu_box.value(),
            *[b.value() for b in self.landa_boxes],
            *[b.value() for b in self.nu_boxes],
            self.a_low_box.value(),
            self.a_up_box.value(),
            self.f_1_box.value(),
            self.f_2_box.value(),
            self.omega_dc_box.value(),
            self.zeta_dc_box.value()
        )

        beta_vals = [b.value() for b in self.beta_boxes]
        lambda_vals = [l.value() for l in self.lambda_boxes]
        mu_vals = [m.value() for m in self.mu_dva_boxes]
        nu_vals = [n.value() for n in self.nu_dva_boxes]
        dva_params = tuple(beta_vals + lambda_vals + mu_vals + nu_vals)

        omega_start_val = self.omega_start_box.value()
        omega_end_val = self.omega_end_box.value()
        omega_points_val = self.omega_points_box.value()

        plot_figure = self.plot_figure_chk.isChecked()
        show_peaks = self.show_peaks_chk.isChecked()
        show_slopes = self.show_slopes_chk.isChecked()

        self.frf_worker = FRFWorker(
            main_params=main_params,
            dva_params=dva_params,
            omega_start=omega_start_val,
            omega_end=omega_end_val,
            omega_points=omega_points_val,
            target_values_dict=target_values,
            weights_dict=weights,
            plot_figure=plot_figure,
            show_peaks=show_peaks,
            show_slopes=show_slopes
        )
        self.frf_worker.finished.connect(self.display_frf_results)
        self.frf_worker.error.connect(self.handle_frf_error)
        self.frf_worker.start()

    def run_sobol(self):
        if self.omega_start_box.value() >= self.omega_end_box.value():
            QMessageBox.warning(self, "Input Error", "Ω Start must be less than Ω End.")
            return

        target_values, weights = self.get_target_values_weights()
        num_samples_list = self.get_num_samples_list()
        n_jobs = self.n_jobs_spin.value()

        self.run_frf_button.setEnabled(False)
        self.run_sobol_button.setEnabled(False)
        self.run_ga_button.setEnabled(False)
        self.sobol_results_text.append("\n--- Running Sobol Sensitivity Analysis ---\n")
        self.status_bar.showMessage("Running Sobol Analysis...")

        main_params = (
            self.mu_box.value(),
            *[b.value() for b in self.landa_boxes],
            *[b.value() for b in self.nu_boxes],
            self.a_low_box.value(),
            self.a_up_box.value(),
            self.f_1_box.value(),
            self.f_2_box.value(),
            self.omega_dc_box.value(),
            self.zeta_dc_box.value()
        )

        dva_bounds = {}
        EPSILON = 1e-6
        for row in range(self.dva_param_table.rowCount()):
            param_item = self.dva_param_table.item(row, 0)
            param_name = param_item.text()

            fixed_widget = self.dva_param_table.cellWidget(row, 1)
            fixed = fixed_widget.isChecked()

            if fixed:
                fixed_value_widget = self.dva_param_table.cellWidget(row, 2)
                fixed_value = fixed_value_widget.value()
                dva_bounds[param_name] = (fixed_value, fixed_value + EPSILON)
            else:
                lower_bound_widget = self.dva_param_table.cellWidget(row, 3)
                upper_bound_widget = self.dva_param_table.cellWidget(row, 4)
                lower = lower_bound_widget.value()
                upper = upper_bound_widget.value()
                if lower > upper:
                    QMessageBox.warning(self, "Input Error",
                                        f"For parameter {param_name}, lower bound is greater than upper bound.")
                    self.run_frf_button.setEnabled(True)
                    self.run_sobol_button.setEnabled(True)
                    self.run_ga_button.setEnabled(True)
                    return
                dva_bounds[param_name] = (lower, upper)

        original_dva_parameter_order = [
            'beta_1','beta_2','beta_3','beta_4','beta_5','beta_6',
            'beta_7','beta_8','beta_9','beta_10','beta_11','beta_12',
            'beta_13','beta_14','beta_15',
            'lambda_1','lambda_2','lambda_3','lambda_4','lambda_5',
            'lambda_6','lambda_7','lambda_8','lambda_9','lambda_10',
            'lambda_11','lambda_12','lambda_13','lambda_14','lambda_15',
            'mu_1','mu_2','mu_3',
            'nu_1','nu_2','nu_3','nu_4','nu_5','nu_6',
            'nu_7','nu_8','nu_9','nu_10','nu_11','nu_12',
            'nu_13','nu_14','nu_15'
        ]

        self.sobol_worker = SobolWorker(
            main_params=main_params,
            dva_bounds=dva_bounds,
            dva_order=original_dva_parameter_order,
            omega_start=self.omega_start_box.value(),
            omega_end=self.omega_end_box.value(),
            omega_points=self.omega_points_box.value(),
            num_samples_list=num_samples_list,
            target_values_dict=target_values,
            weights_dict=weights,
            n_jobs=n_jobs
        )
        self.sobol_worker.finished.connect(lambda res,warn: self.display_sobol_results(res,warn))
        self.sobol_worker.error.connect(self.handle_sobol_error)
        self.sobol_worker.start()

    def run_ga(self):
        pop_size = self.ga_pop_size_box.value()
        num_gens = self.ga_num_generations_box.value()
        cxpb = self.ga_cxpb_box.value()
        mutpb = self.ga_mutpb_box.value()
        tol = self.ga_tol_box.value()
        alpha = self.ga_alpha_box.value()

        ga_dva_parameters = []
        row_count = self.ga_param_table.rowCount()
        for row in range(row_count):
            param_name = self.ga_param_table.item(row, 0).text()
            fixed_widget = self.ga_param_table.cellWidget(row, 1)
            fixed = fixed_widget.isChecked()
            if fixed:
                fixed_value_widget = self.ga_param_table.cellWidget(row, 2)
                fv = fixed_value_widget.value()
                ga_dva_parameters.append((param_name, fv, fv, True))
            else:
                lower_bound_widget = self.ga_param_table.cellWidget(row, 3)
                upper_bound_widget = self.ga_param_table.cellWidget(row, 4)
                lb = lower_bound_widget.value()
                ub = upper_bound_widget.value()
                if lb > ub:
                    QMessageBox.warning(self, "Input Error",
                                        f"For parameter {param_name}, lower bound is greater than upper bound.")
                    return
                ga_dva_parameters.append((param_name, lb, ub, False))

        main_params = (
            self.mu_box.value(),
            *[b.value() for b in self.landa_boxes],
            *[b.value() for b in self.nu_boxes],
            self.a_low_box.value(),
            self.a_up_box.value(),
            self.f_1_box.value(),
            self.f_2_box.value(),
            self.omega_dc_box.value(),
            self.zeta_dc_box.value()
        )

        target_values, weights = self.get_target_values_weights()

        omega_start_val = self.omega_start_box.value()
        omega_end_val = self.omega_end_box.value()
        omega_points_val = self.omega_points_box.value()

        self.ga_worker = GAWorker(
            main_params=main_params,
            target_values_dict=target_values,
            weights_dict=weights,
            omega_start=omega_start_val,
            omega_end=omega_end_val,
            omega_points=omega_points_val,
            ga_pop_size=pop_size,
            ga_num_generations=num_gens,
            ga_cxpb=cxpb,
            ga_mutpb=mutpb,
            ga_tol=tol,
            ga_parameter_data=ga_dva_parameters,
            alpha=alpha
        )
        self.ga_worker.finished.connect(self.handle_ga_finished)
        self.ga_worker.error.connect(self.handle_ga_error)
        self.ga_worker.update.connect(self.handle_ga_update)
        self.run_ga_button.setEnabled(False)
        self.ga_results_text.append("Running GA...")
        self.ga_worker.start()

##################################
# Run the Application
##################################

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
