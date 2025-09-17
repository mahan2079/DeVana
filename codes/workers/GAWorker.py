"""
Comprehensive Flow Chart & Feature Overview of GAWorker.py

[1] Imports & Dependencies
    |
    |-- Standard Libraries: sys, os, time, platform, json, traceback, math, datetime, random
    |-- Data Science Libraries: numpy, pandas, matplotlib, seaborn
    |-- System Monitoring: psutil
    |-- GUI: PyQt5.QtWidgets, PyQt5.QtCore, PyQt5.QtGui
    |-- Evolutionary Algorithms: deap (base, creator, tools)
    |-- Advanced Sampling: scipy.stats.qmc (Sobol, LatinHypercube)
    |-- Custom Modules: modules.FRF (frf), .NeuralSeeder (NeuralSeeder)
    |
    V

[2] Main Class: GAWorker
    |
    |-- __init__():
    |     |-- Initialize all parameters, options, and data structures
    |     |-- Set up system info, metrics tracking, and GUI signals
    |     |-- Support for advanced seeding (QMC, Sobol, LHS, NeuralSeeder)
    |
    |-- run():
    |     |-- Main execution loop for the genetic algorithm
    |     |-- Handles all evolutionary operations and feature toggles
    |     |
    |     |-- Population Initialization:
    |     |      |-- Supports random, Sobol, LHS, and neural seeding
    |     |      |-- Handles fixed parameters and parameter bounds
    |     |
    |     |-- Fitness Evaluation:
    |     |      |-- Uses FRF analysis for multi-mass system
    |     |      |-- Tracks evaluation count and timing
    |     |
    |     |-- Selection, Crossover, Mutation:
    |     |      |-- Tournament selection
    |     |      |-- Custom mutation with fixed parameter support
    |     |      |-- Crossover and mutation rates can be adaptive
    |     |
    |     |-- Advanced Features:
    |     |      |-- Adaptive Rates: Dynamically adjust crossover/mutation based on stagnation and diversity
    |     |      |-- ML Bandit Controller: UCB-based controller for rates and population size (exploration/exploitation)
    |     |      |-- RL Controller: Reinforcement learning-based control of rates and population size
    |     |      |-- NeuralSeeder: Neural network-based population seeding and data collection
    |     |      |-- Diversity Tracking: Monitors population diversity for adaptation
    |     |      |-- Multi-objective Fitness: Tracks and logs primary objective, sparsity penalty, and percentage error
    |     |      |-- Metrics Tracking: Real-time and historical tracking of CPU, memory, network, thread count, and evaluation times
    |     |      |-- Robust Exception Handling: Retries and cleans up DEAP state on errors
    |     |
    |     |-- Logging & Visualization:
    |     |      |-- Emits progress, statistics, and adaptation events to GUI
    |     |      |-- Tabular display of fitness components and rates
    |     |      |-- Plots and saves progress if enabled
    |     |
    |     |-- Finalization:
    |     |      |-- Evaluates and emits best solution and metrics
    |     |      |-- Handles cleanup and error reporting
    |
    |-- save_results(): Store results to file or display
    |-- plot_progress(): Visualize algorithm progress and metrics
    |-- handle_exceptions(): Error handling and logging
    |-- _get_system_info(): Collects detailed system info for benchmarking
    |-- _update_resource_metrics(): Tracks CPU, memory, disk, network, and thread usage
    |-- _start_metrics_tracking() / _stop_metrics_tracking(): Manage periodic metrics collection
    |-- cleanup(): Ensures proper resource and state cleanup

[3] Algorithmic Flow (Genetic Algorithm)
    |
    |-- [Start]
    |-- Initialize population (with advanced seeding options)
    |-- For each generation:
    |      |-- Evaluate fitness (FRF analysis, multi-objective)
    |      |-- Select parents (tournament)
    |      |-- Apply crossover (with adaptive/ML/RL rates)
    |      |-- Apply mutation (with adaptive/ML/RL rates)
    |      |-- Form new population (replacement, resizing if needed)
    |      |-- Log/visualize progress (tabular, plots, GUI)
    |      |-- Track and adapt rates/population (adaptive, ML bandit, RL)
    |      |-- Track metrics (CPU, memory, evaluation times, diversity)
    |-- [End]
    |-- Output best solution(s), statistics, metrics, and plots

[4] Advanced Features & Integrations
    |
    |-- Adaptive Rates: Detects stagnation/diversity and adapts crossover/mutation rates
    |-- ML Bandit Controller: UCB-based action selection for rates and population size
    |-- RL Controller: Reinforcement learning for dynamic control of GA parameters
    |-- NeuralSeeder: Neural network-driven population initialization and data collection
    |-- Multi-objective Fitness: Tracks and displays multiple fitness components
    |-- Real-time System Metrics: CPU, memory, disk, network, thread count
    |-- Robust Exception Handling: Retries, cleans up DEAP state, and emits errors to GUI
    |-- GUI Integration: Real-time updates, progress, and results via PyQt signals

[5] Utilities & Helpers
    |
    |-- Data loading/saving (JSON, CSV, etc.)
    |-- System resource monitoring (psutil)
    |-- Exception handling and retry logic (for DEAP and FRF)
    |-- Parameter bounds and fixed parameter management
    |-- Population resizing and diversity calculation
    |-- Metrics aggregation and export

[6] GUI Integration
    |
    |-- Set up main window and widgets (PyQt5)
    |-- Display progress, results, and controls for user interaction
    |-- Emit signals for updates, errors, and metrics to the GUI

"""




# Import the sys module, which provides access to system-specific parameters and functions.
import sys

# Import numpy, a powerful library for numerical computations and working with arrays.
import numpy as np

# Import os, which allows interaction with the operating system (like file and directory management).
import os

# Import matplotlib's pyplot module for creating static, interactive, and animated plots.
import matplotlib.pyplot as plt

# Import seaborn, a statistical data visualization library built on top of matplotlib.
import seaborn as sns

# Import pandas, a library for data manipulation and analysis, especially with tabular data.
import pandas as pd

# Import traceback, which helps in printing or retrieving stack traces (useful for debugging errors).
import traceback

# Import psutil, a library for retrieving information on running processes and system utilization (CPU, memory, etc.).
import psutil

# Import time, which provides time-related functions (like sleeping, measuring time, etc.).
import time

# Import platform, which allows you to access underlying platform data (OS, architecture, etc.).
import platform

# Import json, a module for parsing and creating JSON (JavaScript Object Notation) data.
import json

# Import datetime from the datetime module, for working with dates and times.
from datetime import datetime

# Import sqrt, log, and ceil functions from the math module for mathematical operations.
from math import sqrt, log, ceil

# Import a large set of widgets and GUI components from PyQt5.QtWidgets.
# These are used to build the application's graphical user interface.
from PyQt5.QtWidgets import (
    QApplication,      # The main application object
    QMainWindow,       # The main window class
    QWidget,           # Base class for all UI objects
    QLabel,            # Display text or images
    QDoubleSpinBox,    # Spin box for floating-point numbers
    QSpinBox,          # Spin box for integers
    QVBoxLayout,       # Vertical layout manager
    QHBoxLayout,       # Horizontal layout manager
    QPushButton,       # Button widget
    QTabWidget,        # Tabbed widget
    QFormLayout,       # Form layout manager
    QGroupBox,         # Group box for grouping widgets
    QTextEdit,         # Multi-line text editor
    QCheckBox,         # Checkbox widget
    QScrollArea,       # Scrollable area
    QFileDialog,       # File dialog for opening/saving files
    QMessageBox,       # Message box for dialogs
    QDockWidget,       # Dockable widget
    QMenuBar,          # Menu bar
    QMenu,             # Menu
    QAction,           # Action for menu/toolbars
    QSplitter,         # Splitter for resizing widgets
    QToolBar,          # Toolbar
    QStatusBar,        # Status bar
    QLineEdit,         # Single-line text editor
    QComboBox,         # Drop-down list
    QTableWidget,      # Table widget
    QTableWidgetItem,  # Item for table widget
    QHeaderView,       # Header for tables
    QAbstractItemView, # Abstract base class for item views
    QSizePolicy,       # Size policy for widgets
    QActionGroup       # Grouping actions together
)

# Import core classes from PyQt5.QtCore for threading, signals, and other core functionality.
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, QWaitCondition, QTimer

# Import GUI-related classes from PyQt5.QtGui for icons, colors, and fonts.
from PyQt5.QtGui import QIcon, QPalette, QColor, QFont

# Import a custom function 'frf' from the modules.FRF module.
# This is likely a user-defined module for a specific purpose (e.g., Frequency Response Function).
from modules.FRF import frf
from .MemorySeeder import MemorySeeder

# Import the random module for generating random numbers (used in algorithms like genetic algorithms).
import random

# Import base, creator, and tools from the deap library, which is used for evolutionary algorithms (like genetic algorithms).
from deap import base, creator, tools

# Import qmc (quasi-Monte Carlo) from scipy.stats for advanced sampling methods.
from scipy.stats import qmc

# Import NeuralSeeder from a local module in the same package.
# This is likely a custom class for initializing neural networks or populations.
from .NeuralSeeder import NeuralSeeder
from concurrent.futures import ThreadPoolExecutor
import threading

# -----------------------------------------------------------------------------------------------

# Imagine you have a function that does something with the DEAP library (which is used for evolutionary algorithms).
# Sometimes, DEAP can get into a weird state (for example, if you try to register the same class twice), and your function might crash.
# This helper function is a "decorator" â€“ a special kind of function in Python that wraps around another function to add extra behavior.
# In this case, the decorator tries to run your function, and if it fails, it will try up to 3 times.
# If it fails, it also tries to "clean up" DEAP's global state by deleting certain attributes, so the next attempt has a better chance of working.
# If it still fails after 3 tries, it gives up and raises the error.

# The science/logic behind this:
# - DEAP uses global state to register things like "FitnessMin" and "Individual" classes.
# - If you try to register them again, or if something goes wrong, DEAP can throw errors.
# - By deleting these attributes, you can "reset" DEAP's state and try again.
# - The retry loop is a common pattern for handling flaky or stateful errors.


def safe_deap_operation(func):
    """
    Decorator to safely execute functions that use DEAP, with error recovery and retries.

    This decorator wraps a function so that if it fails due to DEAP's global state issues,
    it will attempt to clean up and retry the operation a few times before giving up.

    Python concepts used here:
    - Decorators: Functions that modify the behavior of other functions.
    - Nested functions: Defining a function (wrapper) inside another function.
    - Exception handling: Using try/except to catch and handle errors.
    - Loops: Using a for loop to retry the operation multiple times.
    - Attribute manipulation: Using hasattr and delattr to modify objects at runtime.
    """
    def wrapper(*args, **kwargs):
        max_retries = 3  # Set the maximum number of times to try running the function

        # Try to run the function up to max_retries times
        for attempt in range(max_retries):
            try:
                # Attempt to run the original function with all its arguments
                return func(*args, **kwargs)
            except Exception as e:
                # If an error occurs, check if we have more retries left
                if attempt < max_retries - 1:
                    # Suppress console output on retry; silently attempt cleanup and retry
                    # Try to clean up DEAP's global state, which might be causing the error
                    try:
                        # If the 'FitnessMin' class is registered in DEAP's creator, remove it
                        if hasattr(creator, "FitnessMin"):
                            delattr(creator, "FitnessMin")
                        # If the 'Individual' class is registered in DEAP's creator, remove it
                        if hasattr(creator, "Individual"):
                            delattr(creator, "Individual")
                    except Exception:
                        # If cleanup itself fails, ignore the error and continue to the next retry
                        pass
                else:
                    # If we've used up all our retries, re-raise the exception without printing to console
                    raise  # Re-raise the last exception so the caller knows something went wrong
        # (No explicit return here; if all retries fail, the exception is raised)

    # Return the wrapper function so it can be used as a decorator
    return wrapper

# Python concepts to understand here:
# - Decorators: Used to add retry/error-handling logic to any function that might have DEAP issues.
# - Exception handling: try/except blocks let you catch and respond to errors.
# - Attribute manipulation: 
#   - The functions hasattr() and delattr() are used here to interact with Python objects at runtime.
#   - hasattr(object, name) checks if the given object has an attribute with the specified name, returning True or False.
#   - delattr(object, name) removes the specified attribute from the object if it exists.
#   - In this context, we use these functions to check if DEAP's global 'creator' object has certain classes (like 'FitnessMin' or 'Individual') registered as attributes, and if so, remove them to clean up the global state before retrying the operation.
# - Loops: The for loop is used to retry the operation multiple times.

# Suggestions for improvement:
# 1. You could make the list of attributes to clean up configurable, in case you want to handle other DEAP classes.
# 2. Instead of printing errors, you could use Python's logging module for better control over error reporting.
# 3. You could add a small delay (e.g., time.sleep(0.1)) between retries to avoid hammering the system if something is wrong.

# -----------------------------------------------------------------------------------------------


# -----------------------------------------------------------------------------------------------


def build_random_validation_payload(df, omega_vector, frf_curves):
    """Bundle random validation results, frequency vector, and FRF curves."""
    payload = {
        "dataframe": df,
        "omega_vector": None,
        "frf_curves": {}
    }

    try:
        if omega_vector is not None:
            payload["omega_vector"] = np.asarray(omega_vector, dtype=float)
    except Exception:
        payload["omega_vector"] = None

    safe_curves = {}
    if isinstance(frf_curves, dict):
        for key, curves in frf_curves.items():
            try:
                idx = int(key)
            except Exception:
                continue
            if not isinstance(curves, dict):
                continue
            mass_map = {}
            for mass_key, magnitudes in curves.items():
                try:
                    mass_map[str(mass_key)] = np.asarray(magnitudes, dtype=float)
                except Exception:
                    continue
            safe_curves[idx] = mass_map
    payload["frf_curves"] = safe_curves
    return payload


class GAWorker(QThread):
    """Background worker thread that executes the Genetic Algorithm (GA).

    The heavy optimisation work runs in this thread so the GUI remains responsive.
    Progress and results are communicated back to the GUI via Qt signals.
    """

    # Qt Signals --------------------------------------------------------------
    finished = pyqtSignal(dict, list, list, float)
    error = pyqtSignal(str)
    update = pyqtSignal(str)
    progress = pyqtSignal(int)
    benchmark_data = pyqtSignal(dict)
    generation_metrics = pyqtSignal(dict)

    # ---------------------------------------------------------------------------
    # Python Concepts to Understand Here:
    # - Classes and Inheritance: GAWorker is a class that inherits from QThread,
    #   which means it gets all the threading capabilities of QThread.
    # - PyQt Signals: These are special variables that allow communication between
    #   threads in PyQt applications. They are declared as class variables using
    #   pyqtSignal.
    # - Thread Safety: Signals ensure that data passed between threads is handled
    #   safely, preventing crashes or data corruption.
    #
    # ---------------------------------------------------------------------------

    # Overall Purpose:
    # This __init__ method initializes a GAWorker object, configuring all parameters needed to run a genetic algorithm (GA) optimization.
    # It supports advanced features such as adaptive rates, machine learning (ML) and reinforcement learning (RL) controllers, surrogate models, and neural network-based seeding.
    #
    # Underlying Python Concepts:
    # - This is a class constructor (__init__), which is a special method in Python that is automatically called when a new instance of the class is created. It is used to initialize the object's attributes and set up any necessary state.
    # - The __init__ method can take both required and optional arguments. Required arguments must be provided by the caller, while optional arguments have default values and can be omitted.
    # - Default arguments (e.g., alpha=0.01) allow the user to create a GAWorker object with minimal parameters for simple use cases, or to override defaults for advanced customization. This makes the class flexible and user-friendly.
    # - The method signature shows how Python supports both positional and keyword arguments. Positional arguments (like main_params) must be given in order, while keyword arguments (like alpha=0.01) can be specified by name, improving code readability.
    # - The use of keyword arguments and default values also makes the API more robust to future changes, as new optional parameters can be added without breaking existing code that uses the class.
    # - This design pattern is common in Python for classes that may have many configuration options, as it balances ease of use with extensibility.
    # - The docstring and comments within the method provide documentation, helping users and developers understand what each parameter does and how to use the class effectively.
    #
    # Key Python Principles:
    # - Object-Oriented Programming (OOP): This is a method of a class, encapsulating state and behavior.
    # - Function Arguments: Shows positional, keyword, and default arguments.
    # - Documentation: Docstrings and comments clarify usage and intent.
    # ---------------------------------------------------------------------------
    def __init__(
        self,
        main_params,                # Core system parameters
        target_values_dict,         # Target values for objectives (dict: objective -> target)
        weights_dict,               # Weights for each objective (dict: objective -> weight)
        omega_start,                # Start of frequency range for analysis
        omega_end,                  # End of frequency range for analysis
        omega_points,               # Number of frequency points to analyze
        ga_pop_size,                # Population size for the genetic algorithm
        ga_num_generations,         # Number of generations to run the GA
        ga_cxpb,                    # Crossover probability (float, 0-1)
        ga_mutpb,                   # Mutation probability (float, 0-1)
        ga_tol,                     # Convergence tolerance
        ga_parameter_data,          # Parameter metadata for optimization
        alpha=0.01,                 # Learning rate or step size (default: 0.01)
        percentage_error_scale=1000.0, # Scaling factor for percentage error in fitness calculation
        cost_scale_factor=30.0, # Scaling factor for cost term in fitness calculation (higher = less influence)
        track_metrics=False,        # Enable tracking of computational metrics
        adaptive_rates=False,       # Enable adaptive crossover/mutation rates
        stagnation_limit=5,         # Generations without improvement before adapting rates
        cxpb_min=0.1,               # Minimum crossover probability
        cxpb_max=0.9,               # Maximum crossover probability
        mutpb_min=0.05,             # Minimum mutation probability
        mutpb_max=0.5,              # Maximum mutation probability
        # ML/Bandit-based adaptive controller for rates and population
        use_ml_adaptive=False,      # Enable ML-based adaptation of rates/population
        pop_min=None,               # Minimum allowed population size (if adaptive)
        pop_max=None,               # Maximum allowed population size (if adaptive)
        ml_ucb_c=0.6,               # UCB exploration parameter for ML controller
        ml_adapt_population=True,   # Allow ML controller to resize population
        ml_diversity_weight=0.02,   # Penalty weight for diversity deviation
        ml_diversity_target=0.2,    # Target diversity for ML controller
        ml_historical_weight=0.7,   # Weight for historical average in reward blending
        ml_current_weight=0.3,      # Weight for current reward in reward blending
        # ML stagnation controls
        ml_stag_enabled=False,      # Enable exploration boost when stagnation persists
        ml_stag_limit=None,         # Generations without improvement before ML boost
        # Reinforcement Learning controller parameters
        use_rl_controller=False,    # Enable RL-based controller
        rl_alpha=0.1,               # RL learning rate
        rl_gamma=0.9,               # RL discount factor
        rl_epsilon=0.2,             # RL exploration rate
        rl_epsilon_decay=0.95,      # RL epsilon decay per episode
        # RL stagnation controls
        rl_stag_enabled=False,      # Enable exploration boost when stagnation persists
        rl_stag_limit=None,         # Generations without improvement before RL boost
        # RL reward shaping weights (optional; RL-only)
        rl_w1=1.0,                  # Weight for normalized objective improvement term
        rl_w2=0.0,                  # Weight for normalized diversity change term
        rl_w3=0.0,                  # Weight for inverse generation time term
        rl_w4=None,                 # Weight for |cv - cv*| penalty (defaults to ml_diversity_weight)
        rl_cv_target=None,          # Target coefficient of variation for RL (defaults to ml_diversity_target)
        # Surrogate-assisted screening parameters
        use_surrogate=False,        # Enable surrogate model for screening
        surrogate_pool_factor=2.0,  # Pool size multiplier for surrogate screening
        surrogate_k=5,              # Number of nearest neighbors for surrogate
        surrogate_explore_frac=0.15,# Fraction of pool for exploration (not exploitation)
        # Seeding method for initial population and injections
        seeding_method="random",    # Method for seeding ("random", "sobol", "lhs", "neural", "memory", "best")
        seeding_seed=None,          # Random seed for reproducibility
        # Neural seeding options
        use_neural_seeding=False,   # Enable neural network-based seeding
        neural_acq_type="ucb",      # Acquisition function type ("ucb", "ei")
        neural_beta_min=1.0,        # Minimum beta for UCB acquisition
        neural_beta_max=2.5,        # Maximum beta for UCB acquisition
        neural_epsilon=0.1,         # Epsilon for exploration in neural seeding
        neural_pool_mult=3.0,       # Pool size multiplier for neural seeding
        neural_epochs=8,            # Number of training epochs for neural network
        neural_time_cap_ms=750,     # Time cap per neural training (milliseconds)
        neural_ensemble_n=3,        # Number of models in neural ensemble
        neural_hidden=96,           # Hidden layer size for neural network
        neural_layers=2,            # Number of hidden layers
        neural_dropout=0.1,         # Dropout rate for neural network
        neural_weight_decay=1e-4,   # Weight decay (L2 regularization) for neural network
        neural_enable_grad_refine=False, # Enable gradient-based refinement
        neural_grad_steps=0,        # Number of gradient refinement steps
        neural_device="cpu",        # Device for neural computation ("cpu" or "cuda")
        # Optional: adaptive epsilon linkage for neural seeding
        neural_adapt_epsilon=False, # Enable adaptive epsilon for neural seeding
        neural_eps_min=0.05,        # Minimum epsilon value
        neural_eps_max=0.30,        # Maximum epsilon value
        # Best-of-Pool seeding options
        best_pool_mult=5.0,         # Candidate pool multiplier relative to population
        best_diversity_frac=0.20,   # Diversity stride fraction for selection
        # DVA usage/cost extensions
        dva_activation_threshold=0.0,  # Threshold for considering a DVA parameter as active
        dva_activation_penalty=0.0,    # Penalty added per active DVA parameter
        dva_costs=None,                # Dict mapping parameter name -> user-defined cost (legacy)
        dva_costs_by_category=None,    # Dict of dicts: {category: {param: cost}}
        # Enhanced cost-benefit controls
        use_enhanced_cost=True,        # Enable enhanced cost-benefit analysis
        benefit_w_primary=0.6,         # Weight for primary objective in benefit calculation
        benefit_w_accuracy=0.3,        # Weight for accuracy (percentage error) in benefit calculation
        benefit_w_sparsity=0.1,        # Weight for sparsity in benefit calculation
        cat_w_material=0.4,            # Category weight: material
        cat_w_manufacturing=0.35,      # Category weight: manufacturing
        cat_w_maintenance=0.15,        # Category weight: maintenance
        cat_w_operational=0.10,        # Category weight: operational
        benefit_weight_start=0.3,      # Start value for benefit weight over generations (0..1)
        benefit_weight_end=0.7,        # End value for benefit weight over generations (0..1)
        generation_ratio=0.5,          # Fixed generation ratio for adaptive cost-benefit weighting (0..1)
        dva_category_map=None,         # Optional explicit mapping param_name -> category name
        # Performance/monitoring knobs
        metrics_timer_interval=2000,   # ms between resource metric polls (higher = less overhead)
        metrics_verbose=False          # emit verbose metrics updates to log
    ):
        # ------------------------------------------------------------------------
        # Genetic Algorithm Worker Initialization
        #
        # This section documents the initialization of a Genetic Algorithm (GA)
        # worker class. The GA worker is responsible for optimizing a set of
        # parameters to achieve specified target objectives, using evolutionary
        # strategies such as selection, crossover, and mutation.
        #
        # The constructor accepts a comprehensive set of parameters that define:
        #   - The system to be optimized (main_params, ga_parameter_data)
        #   - The optimization objectives and their relative importance (target_values_dict, weights_dict)
        #   - The search/analysis space (omega_start, omega_end, omega_points)
        #   - Genetic algorithm configuration (population size, generations, rates, tolerance)
        #   - Optional adaptive and tracking features
        #
        # Underlying Python Concepts:
        #   - The use of a class constructor (__init__) to encapsulate state and configuration.
        #   - Parameter passing and default arguments for flexible instantiation.
        #   - Use of dictionaries and lists for structured data representation.
        #   - Object-oriented design for modularity and reusability.
        #   - The principle of encapsulation: all configuration is stored as instance attributes.
        # ------------------------------------------------------------------------

        """
        Initialize the Genetic Algorithm Worker.

        Parameters
        ----------
        main_params : dict
            Core configuration of the system to be optimized.
        target_values_dict : dict
            Target values for each optimization objective.
        weights_dict : dict
            Relative importance (weight) of each objective.
        omega_start : float
            Start of frequency range for analysis.
        omega_end : float
            End of frequency range for analysis.
        omega_points : int
            Number of frequency points to evaluate.
        ga_pop_size : int
            Number of candidate solutions in the population.
        ga_num_generations : int
            Number of generations (iterations) to run the GA.
        ga_cxpb : float
            Probability of crossover (combining two solutions).
        ga_mutpb : float
            Probability of mutation (randomly altering a solution).
        ga_tol : float
            Acceptable error threshold for convergence.
        ga_parameter_data : dict or list
            Parameters subject to optimization.
        alpha : float, optional
            Step size for parameter updates (default: 0.01).
        percentage_error_scale : float, optional
            Scaling factor for percentage error component in fitness calculation (default: 1000.0).
        cost_scale_factor : float, optional
            Scaling factor for cost term component in fitness calculation (higher = less influence, default: 1.0).
        track_metrics : bool, optional
            If True, collect and report computational metrics.
        adaptive_rates : bool, optional
            If True, enable automatic adjustment of crossover/mutation rates.
        stagnation_limit : int, optional
            Number of generations without improvement before adapting rates.
        cxpb_min : float, optional
            Minimum allowed crossover probability (if adaptive).
        cxpb_max : float, optional
            Maximum allowed crossover probability (if adaptive).
        mutpb_min : float, optional
            Minimum allowed mutation probability (if adaptive).
        mutpb_max : float, optional
            Maximum allowed mutation probability (if adaptive).

        Notes
        -----
        - This constructor sets up all configuration for the GA worker.
        - Parameters are stored as instance attributes for use throughout the optimization process.
        - Adaptive rates allow the GA to dynamically balance exploration and exploitation.
        - The design leverages Python's object-oriented features for maintainability and extensibility.
        """
        # Call the constructor of the parent class (QThread) to ensure that all
        # thread-related initialization is properly set up. This is necessary so that
        # our GAWorker class inherits all the threading capabilities and can be run
        # as a separate thread in the Qt event loop. Without this call, the thread
        # would not be correctly initialized, which could lead to unexpected behavior.
        super().__init__()
        

        
        # Instance variables are variables that are bound to a specific object (instance) of a class.
        # They are defined using 'self.' and store data unique to each instance.
        # Here, we will store all the input parameters as instance variables so that
        # each GAWorker object keeps track of its own configuration and state.
        # These are like the "settings" for our genetic algorithm
        # Think of it like setting up a recipe - we need to know all the ingredients and steps
        
        # ---------------------------------------------------------------------------
        # ---------------------------------------------------------------------------

        # The concept of 'self' in Python is fundamental to understanding how classes and objects work.
        # 'self' is a reference to the current instance of the class. It allows each object created from a class
        # to keep track of its own data. When you define a method inside a class, the first parameter is always 'self',
        # which means "the object that is calling this method."
        #
        # For example, if you have:
        #   class Dog:
        #       def bark(self):
        #           print("Woof!")
        #   my_dog = Dog()
        #   my_dog.bark()
        # Here, inside the 'bark' method, 'self' refers to 'my_dog'.
        #
        # Why do we need 'self'? Because each object (instance) of a class can have its own data.
        # For example:
        #   class Dog:
        #       def __init__(self, name):
        #           self.name = name
        #   dog1 = Dog("Fido")
        #   dog2 = Dog("Rex")
        #   print(dog1)        # prints something like "<__main__.Dog object at 0x7f8b2c3e0>"
        #   print(dog1.name)   # prints "Fido"
        #   print(dog2.name)   # prints "Rex"
        # By default, printing an object like 'dog1' shows its type and memory address,
        # unless the class defines a __str__ or __repr__ method for a custom string representation.
        # Here, 'self.name' means "the name belonging to this particular dog."
        #
        # In summary:
        # - 'self' is how an object refers to itself.
        # - It lets each object keep track of its own data and methods.
        # - When you see 'self.x', it means "the x that belongs to this object."
        # - You must always include 'self' as the first parameter in instance methods.

        # ---------------------------------------------------------------------------
        # ---------------------------------------------------------------------------

        self.main_params = main_params          # Main system parameters (like engine specifications)
        self.target_values_dict = target_values_dict  # What we're trying to achieve (like target speed)
        self.weights_dict = weights_dict        # How important each goal is (like prioritizing fuel efficiency over speed)
        self.omega_start = omega_start          # Starting frequency (like the lowest radio station frequency)
        self.omega_end = omega_end              # Ending frequency (like the highest radio station frequency)
        self.omega_points = omega_points        # How many frequencies to check (like how many stations to scan)
        self.ga_pop_size = ga_pop_size          # How many solutions to try at once (like having multiple car designs)
        self.ga_num_generations = ga_num_generations  # How many times to improve solutions (like breeding plants for multiple generations)
        self.ga_cxpb = ga_cxpb                  # Initial chance of combining solutions (like breeding two good plants)
        self.ga_mutpb = ga_mutpb                # Initial chance of random changes (like mutations in DNA)
        self.ga_tol = ga_tol                    # How close we need to get to the target (like acceptable error margin)
        self.ga_parameter_data = ga_parameter_data  # What we can change (like adjustable car parts)
        # Sanitize alpha to avoid NaNs propagating into sparsity penalty
        try:
            self.alpha = float(alpha)
        except Exception:
            self.alpha = 0.0
        if not np.isfinite(self.alpha):
            self.alpha = 0.0
        self.percentage_error_scale = percentage_error_scale if percentage_error_scale is not None else 1000.0  # Scaling factor for percentage error in fitness calculation
        self.cost_scale_factor = max(0.1, float(cost_scale_factor)) if cost_scale_factor is not None else 30.0  # Scaling factor for cost term in fitness calculation
        # DVA activation/cost configuration
        try:
            self.dva_activation_threshold = float(dva_activation_threshold)
        except Exception:
            self.dva_activation_threshold = 0.0
        try:
            self.dva_activation_penalty = float(dva_activation_penalty)
        except Exception:
            self.dva_activation_penalty = 0.0
        self.dva_costs = dict(dva_costs) if isinstance(dva_costs, dict) else {}
        # Normalize category cost structure
        if isinstance(dva_costs_by_category, dict):
            self.dva_costs_by_category = {
                'material': dict(dva_costs_by_category.get('material', {})) if isinstance(dva_costs_by_category.get('material', {}), dict) else {},
                'manufacturing': dict(dva_costs_by_category.get('manufacturing', {})) if isinstance(dva_costs_by_category.get('manufacturing', {}), dict) else {},
                'maintenance': dict(dva_costs_by_category.get('maintenance', {})) if isinstance(dva_costs_by_category.get('maintenance', {}), dict) else {},
                'operational': dict(dva_costs_by_category.get('operational', {})) if isinstance(dva_costs_by_category.get('operational', {}), dict) else {},
            }
        else:
            self.dva_costs_by_category = None
        # Precompute denominator for cost normalization (avoid division by zero)
        try:
            self._dva_cost_denominator = float(sum(float(v) for v in self.dva_costs.values() if np.isfinite(float(v))))
        except Exception:
            self._dva_cost_denominator = 0.0
        # Enhanced cost-benefit controls
        self.use_enhanced_cost = bool(use_enhanced_cost)
        # Benefit weights (normalize to sum 1)
        try:
            bw = [float(benefit_w_primary), float(benefit_w_accuracy), float(benefit_w_sparsity)]
            bw_sum = sum(v for v in bw if np.isfinite(v))
            if bw_sum <= 0:
                self.benefit_weights = (0.6, 0.3, 0.1)
            else:
                self.benefit_weights = tuple(v / bw_sum for v in bw)
        except Exception:
            self.benefit_weights = (0.6, 0.3, 0.1)
        # Category weights (normalize to sum 1)
        try:
            cw = [float(cat_w_material), float(cat_w_manufacturing), float(cat_w_maintenance), float(cat_w_operational)]
            cw_sum = sum(v for v in cw if np.isfinite(v))
            if cw_sum <= 0:
                cw_norm = [0.25, 0.25, 0.25, 0.25]
            else:
                cw_norm = [v / cw_sum for v in cw]
        except Exception:
            cw_norm = [0.25, 0.25, 0.25, 0.25]
        self.category_weights = {
            'material': cw_norm[0],
            'manufacturing': cw_norm[1],
            'maintenance': cw_norm[2],
            'operational': cw_norm[3],
        }
        try:
            self.benefit_weight_start = min(1.0, max(0.0, float(benefit_weight_start)))
        except Exception:
            self.benefit_weight_start = 0.3
        try:
            self.benefit_weight_end = min(1.0, max(0.0, float(benefit_weight_end)))
        except Exception:
            self.benefit_weight_end = 0.7
        try:
            self.generation_ratio = min(1.0, max(0.0, float(generation_ratio)))
        except Exception:
            self.generation_ratio = 0.5
        self.dva_category_map = dict(dva_category_map) if isinstance(dva_category_map, dict) else {}
        self.track_metrics = track_metrics      # Whether to track computational metrics
        
        # Adaptive rate parameters
        self.adaptive_rates = adaptive_rates        # Whether to use adaptive rates
        self.stagnation_limit = stagnation_limit    # How many generations without improvement before adapting
        self.stagnation_counter = 0                 # Counter for generations without improvement
        self.cxpb_min = cxpb_min                    # Minimum crossover probability
        self.cxpb_max = cxpb_max                    # Maximum crossover probability
        self.mutpb_min = mutpb_min                  # Minimum mutation probability
        self.mutpb_max = mutpb_max                  # Maximum mutation probability
        self.current_cxpb = ga_cxpb                 # Current crossover probability (starts with initial value)
        self.current_mutpb = ga_mutpb               # Current mutation probability (starts with initial value)
        self.rate_adaptation_history = []           # Track how rates change over time
        
        # Legacy adaptive control state
        self.mutation_scale = 0.10                  # Relative perturbation of parameter range for mutations
        self.mutation_scale_min = 0.01              # Lower bound for mutation scale
        self.mutation_scale_max = 0.30              # Upper bound for mutation scale
        self.target_success_rate = 0.20             # 1/5th success rule target
        self.adapt_smoothing = 0.20                 # EMA smoothing factor for noisy signals
        self._ema_success_rate = None               # Exponential moving average of success rate
        self._ema_gene_diversity = None             # Exponential moving average of normalized gene diversity
        
        # ML/Bandit controller configuration
        self.use_ml_adaptive = use_ml_adaptive
        # Guardrail population bounds
        self.pop_min = pop_min if pop_min is not None else max(10, int(0.5 * self.ga_pop_size))
        self.pop_max = pop_max if pop_max is not None else int(2.0 * self.ga_pop_size)
        self.ml_ucb_c = ml_ucb_c
        self.ml_adapt_population = ml_adapt_population
        self.ml_diversity_weight = ml_diversity_weight
        self.ml_diversity_target = ml_diversity_target
        self.ml_historical_weight = ml_historical_weight
        self.ml_current_weight = ml_current_weight
        # ML bandit performance controls
        self.ml_update_interval = 3  # adapt rates every N generations
        self.ml_pop_update_interval = 5  # resize population every M generations
        # ML stagnation config
        self.ml_stag_enabled = bool(ml_stag_enabled)
        try:
            self.ml_stag_limit = int(ml_stag_limit) if ml_stag_limit is not None else int(self.stagnation_limit)
        except Exception:
            self.ml_stag_limit = int(self.stagnation_limit)
        self.ml_stag_counter = 0

        # Reinforcement learning controller configuration
        self.use_rl_controller = use_rl_controller
        self.rl_alpha = float(rl_alpha)
        self.rl_gamma = float(rl_gamma)
        self.rl_epsilon = float(rl_epsilon)
        self.rl_epsilon_decay = float(rl_epsilon_decay)
        # RL stagnation config
        self.rl_stag_enabled = bool(rl_stag_enabled)
        try:
            self.rl_stag_limit = int(rl_stag_limit) if rl_stag_limit is not None else int(self.stagnation_limit)
        except Exception:
            self.rl_stag_limit = int(self.stagnation_limit)
        self.rl_stag_counter = 0
        # RL reward weights and targets (fallback to ML counterparts where appropriate)
        try:
            self.rl_w1 = float(rl_w1)
        except Exception:
            self.rl_w1 = 1.0
        try:
            self.rl_w2 = float(rl_w2)
        except Exception:
            self.rl_w2 = 0.0
        try:
            self.rl_w3 = float(rl_w3)
        except Exception:
            self.rl_w3 = 0.0
        try:
            self.rl_w4 = float(rl_w4) if rl_w4 is not None else float(self.ml_diversity_weight)
        except Exception:
            self.rl_w4 = float(self.ml_diversity_weight)
        try:
            self.rl_cv_target = float(rl_cv_target) if rl_cv_target is not None else float(self.ml_diversity_target)
        except Exception:
            self.rl_cv_target = float(self.ml_diversity_target)

        # Surrogate configuration
        self.use_surrogate = use_surrogate
        self.surrogate_pool_factor = max(1.0, float(surrogate_pool_factor))
        self.surrogate_k = max(1, int(surrogate_k))
        self.surrogate_explore_frac = max(0.0, min(0.5, float(surrogate_explore_frac)))
        self._surrogate_X = []  # raw parameter vectors
        self._surrogate_y = []  # corresponding fitness values
        # Maintain normalized cache to avoid re-normalizing entire history each generation
        self._surrogate_Xn = []  # normalized parameter vectors (using current run's bounds)
        self._surrogate_norm_lows = None
        self._surrogate_norm_spans = None
        
        # Seeding configuration
        # Seeding config (allow legacy seeding_method but also a boolean flag)
        self.seeding_method = (seeding_method or "random").lower()
        if self.seeding_method not in ("random", "sobol", "lhs", "neural", "memory", "best"):
            self.seeding_method = "random"
        if use_neural_seeding:
            self.seeding_method = "neural"
        self.seeding_seed = seeding_seed if (seeding_seed is None or isinstance(seeding_seed, (int, np.integer))) else None
        self._qmc_engine = None
        # Best-of-Pool config
        self.best_pool_mult = float(best_pool_mult)
        self.best_diversity_frac = float(best_diversity_frac)

        # Neural seeding settings
        self.use_neural_seeding = (self.seeding_method == "neural")
        self.neural_acq_type = str(neural_acq_type or "ucb").lower()
        self.neural_beta_min = float(neural_beta_min)
        self.neural_beta_max = float(neural_beta_max)
        self.neural_epsilon = float(neural_epsilon)
        self.neural_pool_mult = float(neural_pool_mult)
        self.neural_epochs = int(neural_epochs)
        self.neural_time_cap_ms = int(neural_time_cap_ms)
        self.neural_ensemble_n = int(neural_ensemble_n)
        self.neural_hidden = int(neural_hidden)
        self.neural_layers = int(neural_layers)
        self.neural_dropout = float(neural_dropout)
        self.neural_weight_decay = float(neural_weight_decay)
        self.neural_enable_grad_refine = bool(neural_enable_grad_refine)
        self.neural_grad_steps = int(neural_grad_steps)
        self.neural_device = str(neural_device)
        # Adaptive epsilon controls
        self.neural_adapt_epsilon = bool(neural_adapt_epsilon)
        self.neural_eps_min = float(neural_eps_min)
        self.neural_eps_max = float(neural_eps_max)
        self.current_epsilon = float(self.neural_epsilon)
        # Neural seeding state helpers
        self._neural_last_diversity = 0.2
        
        # Thread safety mechanisms - these prevent crashes when multiple parts of the program try to use the same data
        # Think of it like traffic lights controlling access to a busy intersection
        self.mutex = QMutex()                   # A lock that only one part of the program can hold at a time
        self.condition = QWaitCondition()       # A way for different parts to signal each other
        self.abort = False                      # A flag to safely stop the program if needed
        self.paused = False                     # Flag used to pause/resume the algorithm
        
        # ---------------------------------------------------------------------------

        # --- Plain English Explanation ---
        # This block sets up a watchdog timer using Qt's QTimer class. The timer is configured to fire only once
        # (single-shot) and, when triggered, will call the 'handle_timeout' method. This mechanism is used to
        # automatically handle situations where the program runs too long without progress, acting as a safety net.
        # The 'last_progress_update' variable tracks the last time progress was made, which can be used to reset
        # or check the timer as needed.

        # --- Python Concepts ---
        # - QTimer is a Qt class for event-driven programming, allowing you to schedule code execution after a delay.
        # - The 'setSingleShot' method ensures the timer only triggers once per start.
        # - The 'timeout.connect' method uses Qt's signal-slot mechanism to bind the timer's event to a handler.
        # - Instance variables (self.*) are used to maintain state within the class.

        # --- Production-Quality Comments and Code ---
        self.watchdog_timer = QTimer()  # Instantiate a QTimer for monitoring long-running operations
        self.watchdog_timer.setSingleShot(True)  # Configure timer to trigger only once per activation
        self.watchdog_timer.timeout.connect(self.handle_timeout)  # Connect timer expiration to timeout handler
        self.last_progress_update = 0  # Timestamp or counter for the last recorded progress update

        # --- Python Principles Highlighted ---
        # - Object-oriented programming: using instance variables and methods within a class.
        # - Event-driven programming: leveraging Qt's signal-slot system for asynchronous event handling.

        # --- Suggestions for Learning ---
        # 1. Try making the watchdog timer interval configurable via a class parameter, so you can experiment with different timeout durations.
        # 2. Add logging inside 'handle_timeout' to record when and why the timeout occurs, helping you learn about debugging and monitoring long-running processes.
        
        # Initialize benchmark metrics tracking
        self.metrics = {
            'start_time': None,
            'end_time': None,
            'total_duration': None,
            'proc_cpu_time_start': None,
            'proc_cpu_time_end': None,
            'proc_cpu_time_used': None,
            'cpu_usage': [],
            'memory_usage': [],
            'fitness_history': [],
            'mean_fitness_history': [],
            'std_fitness_history': [],
            'convergence_rate': [],
            'system_info': self._get_system_info(),
            'generation_times': [],
            'best_fitness_per_gen': [],
            'best_individual_per_gen': [],
            'evaluation_count': 0,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            # New detailed computational metrics
            'cpu_per_core': [],
            'memory_details': [],
            'io_counters': [],
            'disk_usage': [],
            'network_usage': [],
            'gpu_usage': [],
            'thread_count': [],
            'evaluation_times': [],
            'crossover_times': [],
            'mutation_times': [],
            'selection_times': [],
            'time_per_generation_breakdown': [],
            'adaptive_rates_history': [],  # Track how rates change if adaptive rates are used
            # ML/Bandit controller histories
            'ml_controller_history': [],   # Per-generation records of decisions and rewards
            'rl_controller_history': [],   # RL controller decisions and rewards
            'pop_size_history': [],        # Track population size across generations
            'rates_history': [],           # Track cxpb/mutpb chosen each generation
            'controller': None,            # Which controller was used: 'fixed' | 'adaptive' | 'ml_bandit' | 'rl'
            'ml_blending_weights': {       # Store the blending weights used
                'historical': float(self.ml_historical_weight),
                'current': float(self.ml_current_weight)
            },
            'ml_stagnation_enabled': bool(self.ml_stag_enabled),
            'rl_stagnation_enabled': bool(self.rl_stag_enabled),
            'ml_stagnation_counter_history': [],
            'rl_stagnation_counter_history': [],
            # Surrogate metrics
            'surrogate_enabled': bool(use_surrogate),
            'surrogate_pool_factor': float(self.surrogate_pool_factor),
            'surrogate_k': int(self.surrogate_k),
            'surrogate_explore_frac': float(self.surrogate_explore_frac),
            'surrogate_info': [],          # List of dicts per generation with pool/eval counts and error
            # Legacy adaptive v2 metrics
            'success_rate_history': [],
            'gene_diversity_history': [],
            'mutation_scale_history': []
        }
        # Record seeding method in metrics for transparency
        self.metrics['seeding_method'] = self.seeding_method
        self.metrics['neural_seeding'] = {
            'enabled': bool(self.use_neural_seeding),
            'backend': 'torch',
            'ensemble_n': int(self.neural_ensemble_n),
            'epochs': int(self.neural_epochs),
            'pool_mult': float(self.neural_pool_mult),
            'epsilon': float(self.neural_epsilon),
            'acq_type': self.neural_acq_type,
            'beta_min': float(self.neural_beta_min),
            'beta_max': float(self.neural_beta_max),
            # Architecture & regularization details for UI
            'hidden': int(self.neural_hidden),
            'layers': int(self.neural_layers),
            'dropout': float(self.neural_dropout),
            'weight_decay': float(self.neural_weight_decay),
            'device': self.neural_device,
            'adapt_epsilon': bool(self.neural_adapt_epsilon),
            'eps_min': float(self.neural_eps_min),
            'eps_max': float(self.neural_eps_max),
        }
        self.metrics['neural_history'] = []
        
        # Create metrics tracking timer (throttled to reduce CPU overhead)
        self.metrics_timer = QTimer()
        self.metrics_timer_interval = int(metrics_timer_interval) if metrics_timer_interval else 2000
        self.metrics_verbose = bool(metrics_verbose)
        self._metrics_tick = 0
        # Heavy metrics (per-core cpu, disk, net) every ~5s by default
        try:
            self._metrics_heavy_every = max(1, int(5000 / self.metrics_timer_interval))
        except Exception:
            self._metrics_heavy_every = 5
        self.log_component_table_every_n = 5
        # Persistent executor for parallel fitness evaluation (reduces per-call overhead)
        self._executor = None
        self._eval_workers = None
        # History/window controls
        self.metrics_window = 200  # keep last N generations of metrics/history
        self.surrogate_dataset_max = 3000  # cap KNN memory to avoid O(N) blowup
        
    def __del__(self):
        """
        Cleanup method that runs when the object is destroyed
        Like cleaning up after a party - making sure everything is turned off and put away
        """
        self.mutex.lock()                       # Get exclusive access to prevent other parts from interfering
        self.abort = True                       # Tell the program to stop
        self.paused = False                     # Ensure we are not left in a paused state
        self.condition.wakeAll()                # Wake up any waiting parts of the program
        self.mutex.unlock()                     # Release the lock
        self.wait()                             # Wait for everything to finish

    def handle_timeout(self):
        """
        What to do if the program runs too long
        Like having a backup plan if the microwave timer goes off
        """
        self.mutex.lock()                       # Get exclusive access
        if not self.abort:                      # If we haven't already stopped
            self.abort = True                   # Tell the program to stop
            self.paused = False                 # Clear paused state on timeout
            self.mutex.unlock()                 # Release the lock
            self.error.emit("Genetic algorithm optimization timed out. The operation was taking too long.")
        else:
            self.mutex.unlock()                 # Release the lock

    # ------------------------------------------------------------------
    # Control methods for pause/resume/stop functionality
    # ------------------------------------------------------------------
    def pause(self):
        """Pause the optimization thread"""
        self.mutex.lock()
        self.paused = True
        self.mutex.unlock()

    def resume(self):
        """Resume the optimization thread if paused"""
        self.mutex.lock()
        self.paused = False
        self.condition.wakeAll()
        self.mutex.unlock()

    def stop(self):
        """Request termination of the optimization thread"""
        self.mutex.lock()
        self.abort = True
        self.paused = False
        self.condition.wakeAll()
        self.mutex.unlock()

    def _check_pause_abort(self):
        """Helper to block while paused and report abort state"""
        self.mutex.lock()
        while self.paused and not self.abort:
            self.condition.wait(self.mutex)
        aborted = self.abort
        self.mutex.unlock()
        return aborted
            
    def cleanup(self):
        """
        Clean up resources to prevent memory leaks
        Like properly closing files and turning off equipment
        """
        # Remove DEAP framework types to prevent memory leaks
        # DEAP is the genetic algorithm framework we're using
        if hasattr(creator, "FitnessMin"):      # If we have a fitness type defined
            try:
                delattr(creator, "FitnessMin")  # Remove it
            except Exception:
                pass                            # Ignore errors if it's already gone
        if hasattr(creator, "Individual"):      # If we have an individual type defined
            try:
                delattr(creator, "Individual")  # Remove it
            except Exception:
                pass                            # Ignore errors if it's already gone
        
        # Stop the watchdog timer if it's running
        # Like turning off the microwave timer
        if self.watchdog_timer.isActive():
            self.watchdog_timer.stop()
        # Shutdown persistent executor if it exists
        try:
            if self._executor is not None:
                self._executor.shutdown(wait=False, cancel_futures=True)
                self._executor = None
        except Exception:
            pass
 
    # The 'run' method is the main entry point for executing the genetic algorithm (GA) optimization.
    # It is decorated with @safe_deap_operation to ensure that any exceptions or errors during DEAP (the GA framework) operations
    # are handled gracefully, preventing crashes and resource leaks.
    # This method is typically called when the GAWorker thread is started, and it manages the entire optimization process.

    @safe_deap_operation  # Ensures robust error handling for DEAP-related operations
    def run(self):
        pass  # The actual implementation is provided elsewhere in the class

    # --- Python Concepts Highlighted ---
    # - Decorators: @safe_deap_operation wraps the 'run' method to add error handling logic.
    # - Methods: 'run' is an instance method, using 'self' to access instance attributes and methods.
    # - Classes: This method is part of a class (likely a QThread subclass), enabling concurrent execution.
    # - Threading: The method is designed to be run in a separate thread, allowing the GA to operate asynchronously.

        """
        Main execution method for the Genetic Algorithm (GA) optimization.
        
        Scientific Context:
        - This is a Genetic Algorithm implementation for optimizing Dynamic Vibration Absorber (DVA) parameters
        - The algorithm mimics natural selection to find optimal solutions
        - Uses fitness evaluation based on Frequency Response Function (FRF) analysis
        - Incorporates sparsity penalties to encourage simpler solutions
        
        Coding Context:
        - Uses DEAP (Distributed Evolutionary Algorithms in Python) framework
        - Implements thread-safe operations with mutex locks
        - Includes watchdog timer for safety
        - Handles parameter bounds and fixed parameters
        """
        
        # Start watchdog timer (10 minutes timeout)
        # This is like having a safety net - if the algorithm runs too long, it will stop
        self.watchdog_timer.start(600000)  # 600,000 milliseconds = 10 minutes
        
        # ---------------------------------------------------------------------------
        # --- Plain English Explanation ---
        # This block emits debug messages to the UI or log, reporting which optimization controllers
        # (adaptive rates, ML bandit, RL) are enabled and what the initial genetic algorithm (GA)
        # crossover and mutation probabilities are. This helps track the configuration for each run.

        # --- Python Concepts and Science ---
        # - Method calls: self.update.emit(...) sends signals/messages, likely to a Qt slot for UI/logging.
        # - f-strings: Used for readable, formatted output.
        # - Instance attributes: Accesses self.adaptive_rates, self.use_ml_adaptive, etc., to report current settings.
        # - This is useful for reproducibility and debugging, as it records the algorithm's configuration.

        # --- Rewritten with Professional Comments ---
        # Emit current controller configuration for debugging and reproducibility
        self.update.emit(f"DEBUG: adaptive_rates parameter is set to: {self.adaptive_rates}")  # Adaptive rate control enabled/disabled
        self.update.emit(f"DEBUG: ML bandit controller is set to: {self.use_ml_adaptive}")     # ML bandit controller enabled/disabled
        self.update.emit(f"DEBUG: RL controller is set to: {self.use_rl_controller}")          # RL controller enabled/disabled

        # Emit initial GA operator probabilities for traceability
        self.update.emit(
            f"DEBUG: GA parameters: crossover={self.ga_cxpb:.4f}, mutation={self.ga_mutpb:.4f}"
        )

        # --- Python Principles Highlighted ---
        # - Object-oriented programming: Uses instance attributes and methods.
        # - String formatting: f-strings for readable, precise output.
        # - Event-driven programming: Signals/slots (Qt) for asynchronous UI/logging updates.


        try:
            if self.use_rl_controller:
                self.metrics['controller'] = 'rl'
            elif self.use_ml_adaptive:
                self.metrics['controller'] = 'ml_bandit'
            elif self.adaptive_rates:
                self.metrics['controller'] = 'adaptive'
            else:
                self.metrics['controller'] = 'fixed'
        except Exception:
            pass

        if self.adaptive_rates:
            self.update.emit("DEBUG: Adaptive rate parameters:")
            self.update.emit(f"DEBUG: - Stagnation limit: {self.stagnation_limit}")
            self.update.emit(f"DEBUG: - Crossover range: {self.cxpb_min:.2f} - {self.cxpb_max:.2f}")
            self.update.emit(f"DEBUG: - Mutation range: {self.mutpb_min:.2f} - {self.mutpb_max:.2f}")
        if self.use_ml_adaptive:
            self.update.emit(f"DEBUG: ML params: UCB c={self.ml_ucb_c:.2f}, pop_adapt={self.ml_adapt_population}, div_weight={self.ml_diversity_weight:.3f}, div_target={self.ml_diversity_target:.2f}, blending=[{self.ml_historical_weight:.2f}, {self.ml_current_weight:.2f}]")
        if self.use_surrogate:
            self.update.emit(f"DEBUG: Surrogate screening enabled â†’ pool_factor={self.surrogate_pool_factor:.2f}, k={self.surrogate_k}, explore_frac={self.surrogate_explore_frac:.2f}")
        
        # Start metrics tracking if enabled
        if self.track_metrics:
            self._start_metrics_tracking()
        # Record process CPU time start for per-run CPU usage estimation
        try:
            proc = psutil.Process(os.getpid())
            cpu_times = proc.cpu_times()
            # Sum user + system for process-level CPU time
            self.metrics['proc_cpu_time_start'] = float(getattr(cpu_times, 'user', 0.0) + getattr(cpu_times, 'system', 0.0))
        except Exception:
            self.metrics['proc_cpu_time_start'] = None
        
        try:
            # Initialize parameter tracking lists/dictionaries
            # These will store information about what parameters we're optimizing
            parameter_names = []      # Names of parameters (e.g., "mass", "stiffness")
            parameter_bounds = []     # Valid ranges for each parameter
            fixed_parameters = {}     # Parameters that won't change during optimization

            # Process each parameter's configuration
            # This is like setting up the rules for our optimization game
            for idx, (name, low, high, fixed) in enumerate(self.ga_parameter_data):
                parameter_names.append(name)
                if fixed:
                    # If parameter is fixed, set both bounds to the same value
                    parameter_bounds.append((low, low))
                    fixed_parameters[idx] = low  
                else:
                    # If parameter is variable, set its valid range
                    parameter_bounds.append((low, high))

            # Safely reset DEAP framework types
            # This is like clearing the board before starting a new game
            self.mutex.lock()  # Get exclusive access to prevent other threads from interfering
            if hasattr(creator, "FitnessMin"):
                delattr(creator, "FitnessMin")
            if hasattr(creator, "Individual"):
                delattr(creator, "Individual")
                
            # Create new DEAP types for this run
            # These define how we'll measure success and structure our solutions
            
            # SCIENTIFIC EXPLANATION:
            # In genetic algorithms, we need two fundamental components:
            # 1. A way to measure how "good" a solution is (fitness)
            # 2. A way to represent potential solutions (individuals)
            
            # CODING EXPLANATION:
            # The DEAP library uses a special system called "creator" to define custom types
            # Think of it like creating blueprints for our genetic algorithm
            
            # Create a fitness type that aims to minimize the objective function
            # weights=(-1.0,) means we want to minimize the fitness value
            # The negative sign is important - in DEAP, lower fitness is better
            # The comma after -1.0 makes it a tuple, which DEAP requires
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            
            # Create a type to represent a single solution (called an "individual")
            # Each individual is a list of parameters (like mass, stiffness, etc.)
            # The fitness attribute will store how good this solution is
            # Think of it like a person's DNA and their fitness score
            creator.create("Individual", list, fitness=creator.FitnessMin)
            
            # Release the mutex lock we acquired earlier
            # This is like unlocking a door after we're done using a room
            # It allows other parts of the program to access shared resources
            self.mutex.unlock()

            # Initialize the DEAP toolbox
            # This is like setting up our workshop with all the tools we'll need
            toolbox = base.Toolbox()

            # Thread-safe FRF evaluation cache to avoid recomputing identical individuals
            frf_cache = {}
            cache_lock = threading.Lock()

            # Precompute arrays and maps used repeatedly (avoid per-evaluation overhead)
            # Bounds arrays
            lows_np = np.array([b[0] for b in parameter_bounds], dtype=float)
            highs_np = np.array([b[1] for b in parameter_bounds], dtype=float)
            spans_np = highs_np - lows_np
            spans_np[~np.isfinite(spans_np)] = 0.0
            # Fixed mask
            fixed_mask_np = np.zeros(len(parameter_bounds), dtype=bool)
            for _i in range(len(parameter_bounds)):
                if _i in fixed_parameters:
                    fixed_mask_np[_i] = True
            # Name index map for cost/category lookup
            name_to_idx = {n: i for i, n in enumerate(parameter_names)}

            # Precompute category cost arrays if provided
            cost_arrays = None
            total_cost_all = 0.0
            if self.dva_costs_by_category:
                cost_arrays = {
                    'material': np.zeros(len(parameter_names), dtype=float),
                    'manufacturing': np.zeros(len(parameter_names), dtype=float),
                    'maintenance': np.zeros(len(parameter_names), dtype=float),
                    'operational': np.zeros(len(parameter_names), dtype=float),
                }
                for cat, cmap in self.dva_costs_by_category.items():
                    if not isinstance(cmap, dict) or cat not in cost_arrays:
                        continue
                    arr = cost_arrays[cat]
                    for pname, cval in cmap.items():
                        j = name_to_idx.get(pname)
                        if j is None or j >= arr.size:
                            continue
                        try:
                            v = float(cval)
                            if np.isfinite(v):
                                arr[j] = v
                        except Exception:
                            pass
                # Precompute total across all categories
                total_cost_all = float(sum(a.sum() for a in cost_arrays.values()))

            # Fallback flat cost array (legacy)
            fallback_cost_arr = None
            if not self.dva_costs_by_category and getattr(self, 'dva_costs', None):
                fallback_cost_arr = np.zeros(len(parameter_names), dtype=float)
                for pname, cval in self.dva_costs.items():
                    j = name_to_idx.get(pname)
                    if j is None or j >= fallback_cost_arr.size:
                        continue
                    try:
                        v = float(cval)
                        if np.isfinite(v):
                            fallback_cost_arr[j] = v
                    except Exception:
                        pass

            # Prepare surrogate normalization parameters for this run
            self._surrogate_norm_lows = lows_np
            # Avoid zeros in spans for normalization
            _sp = spans_np.copy()
            _sp[_sp == 0.0] = 1.0
            self._surrogate_norm_spans = _sp

            # Define how to generate random parameter values
            # This is like having a recipe for creating new potential solutions
            def attr_float(i):
                """
                Generate a random parameter value within its bounds
                or return fixed value if parameter is fixed
                """
                if i in fixed_parameters:
                    return fixed_parameters[i]  # Return fixed value
                else:
                    # Generate random value within bounds
                    return random.uniform(parameter_bounds[i][0], parameter_bounds[i][1])

            # ============================================================================
            # SCIENTIFIC EXPLANATION:
            # In genetic algorithms, we need to set up three main components:
            # 1. How to generate random parameters (like DNA building blocks)
            # 2. How to create complete solutions (like creating organisms)
            # 3. How to evaluate how good each solution is (like testing survival fitness)
            # ============================================================================

            # Register our parameter generator with DEAP's toolbox
            # Think of this like registering a recipe for creating DNA building blocks
            # attr_float is our function that generates random numbers within bounds
            # i=None means we'll specify which parameter to generate when we use it
            toolbox.register("attr_float", attr_float, i=None)

            # ============================================================================
            # CODING EXPLANATION FOR BEGINNERS:
            # The next two lines set up how we create complete solutions:
            # 1. First, we define how to create a single solution (an "individual")
            # 2. Then, we define how to create a group of solutions (a "population")
            # ============================================================================

            # Create a single solution (individual)
            # tools.initIterate is like a factory that creates solutions
            # creator.Individual is our blueprint for what a solution looks like
            # The lambda function creates a list of random parameters using our attr_float recipe
            toolbox.register("individual", tools.initIterate, creator.Individual,
                             lambda: [attr_float(i) for i in range(len(parameter_bounds))])

            # Create a group of solutions (population)
            # tools.initRepeat is like a factory that creates multiple solutions
            # list is the container type (like a box to hold our solutions)
            # toolbox.individual is our recipe for creating each solution
            toolbox.register("population", tools.initRepeat, list, toolbox.individual)

            # ============================================================================
            # SCIENTIFIC EXPLANATION:
            # The evaluate_individual function is like a fitness test for our solutions.
            # It:
            # 1. Checks if we should stop the process
            # 2. Converts our solution into a format the FRF analysis can understand
            # 3. Runs the FRF analysis to see how well our solution performs
            # 4. Returns a score (fitness) that tells us how good the solution is
            # ============================================================================

            def evaluate_individual(individual):
                """
                ============================================================================
                SCIENTIFIC EXPLANATION:
                This function evaluates how well a potential solution performs in our system.
                Think of it like a fitness test for a robot:
                1. We check if the robot should stop (abort signal)
                2. We convert the robot's parameters into a format our analysis can understand
                3. We run a test (FRF analysis) to see how well the robot performs
                4. We calculate a score based on:
                   - How close the performance is to our target (primary objective)
                   - How complex the solution is (sparsity penalty)
                The lower the score, the better the solution!
                ============================================================================

                CODING EXPLANATION FOR BEGINNERS:
                This function is like a judge at a robot competition:
                1. It takes one robot (individual) as input
                2. It checks if the competition should stop
                3. It prepares the robot for testing
                4. It runs the test and calculates a score
                5. It returns the score as a tuple (a special type of list that can't be changed)
                """
                # Check for pause or abort requests
                if self._check_pause_abort():
                    return (1e6,)  # Return a very bad score (1 million) to signal we should stop

                # Track evaluation count for benchmark metrics
                if self.track_metrics:
                    self.metrics['evaluation_count'] += 1

                # Convert our solution into a tuple (immutable list) for the FRF analysis
                # This is like preparing the robot for testing
                dva_parameters_tuple = tuple(float(v) for v in individual)

                # Fast path: check cache for identical parameter vector
                try:
                    with cache_lock:
                        cached = frf_cache.get(dva_parameters_tuple)
                    if cached is not None:
                        # Support both legacy (4-tuple) and extended (6-tuple) cache formats
                        if isinstance(cached, (list, tuple)):
                            if len(cached) >= 6:
                                primary_objective, sparsity_penalty, percentage_error_component, activation_penalty_term, cost_term, fitness_value = cached[:6]
                                individual.activation_penalty = activation_penalty_term
                                individual.cost_term = cost_term / self.cost_scale_factor
                            else:
                                primary_objective, sparsity_penalty, percentage_error_component, fitness_value = cached[:4]
                                individual.activation_penalty = 0.0
                                individual.cost_term = 0.0
                        elif isinstance(cached, dict):
                            primary_objective = float(cached.get('primary_objective', 0.0))
                            sparsity_penalty = float(cached.get('sparsity_penalty', 0.0))
                            percentage_error_component = float(cached.get('percentage_error', 0.0))
                            fitness_value = float(cached.get('fitness', 1e6))
                            individual.activation_penalty = float(cached.get('activation_penalty', 0.0))
                            individual.cost_term = float(cached.get('cost_term', 0.0)) / self.cost_scale_factor
                        else:
                            primary_objective = sparsity_penalty = percentage_error_component = 0.0
                            fitness_value = 1e6
                            individual.activation_penalty = 0.0
                            individual.cost_term = 0.0
                        # Restore component attributes for reporting
                        individual.primary_objective = primary_objective
                        individual.sparsity_penalty = sparsity_penalty
                        individual.percentage_error = percentage_error_component
                        return (fitness_value,)
                except Exception:
                    pass

                try:
                    
                    # ============================================================================
                    # SCIENTIFIC EXPLANATION:
                    # The FRF (Frequency Response Function) analysis is like a comprehensive test
                    # that measures how our system responds to different frequencies of vibration.
                    # We're testing:
                    # - How well our solution (DVA parameters) works with the main system
                    # - How the system behaves across a range of frequencies
                    # - How well it matches our target performance for each mass
                    # ============================================================================

                    # Run the FRF analysis with all necessary parameters
                    results = frf(
                        # Main system parameters (like the base structure of our robot)
                        main_system_parameters=self.main_params,
                        # Our solution parameters (like the robot's settings)
                        dva_parameters=dva_parameters_tuple,
                        # Frequency range to analyze (like testing different speeds)
                        omega_start=self.omega_start,
                        omega_end=self.omega_end,
                        omega_points=self.omega_points,
                        # Target values and weights for each mass (like performance goals)
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
                        # Disable visualization for speed (like turning off the camera during testing)
                        plot_figure=False,
                        show_peaks=False,
                        show_slopes=False
                    )
                    
                    # Check if results are in the correct format
                    if not isinstance(results, dict):
                        self.update.emit("Warning: FRF returned non-dictionary result")
                        return (1e6,)  # Return bad score if results are invalid
                    
                    # ============================================================================
                    # SCIENTIFIC EXPLANATION:
                    # This section validates the results from our FRF (Frequency Response Function) analysis.
                    # Think of it like checking if a medical test result is valid before using it:
                    # 1. First, we try to get the main result (singular_response)
                    # 2. If that's not available or invalid, we try to calculate it from other measurements
                    # 3. If all else fails, we return a very high score (1e6) to indicate failure
                    # ============================================================================

                    # Try to get the main performance measure from our results
                    # This is like getting the main test result from a medical report
                    singular_response = results.get('singular_response', None)

                    # Check if we got a valid result
                    # This is like checking if the test result is a real number and not "error" or "invalid"
                    if singular_response is None or not np.isfinite(singular_response):
                        # If the main result is missing, try to calculate it from other measurements
                        # This is like calculating an overall health score from individual test results
                        if 'composite_measures' in results:
                            # Get all the individual measurements
                            composite_measures = results['composite_measures']
                            # Add them up to get a total score
                            singular_response = sum(composite_measures.values())
                            # Save this calculated result back to our results
                            results['singular_response'] = singular_response
                        else:
                            # If we can't even calculate a result, send a warning message
                            self.update.emit("Warning: Could not compute singular response")
                            # Return a very high score (1e6) to indicate this solution is bad
                            return (1e6,)
                    
                    # One final check to make sure our result is a valid number
                    # This is like double-checking the test result is a real number
                    if not np.isfinite(singular_response):
                        # If it's still not valid, return the bad score
                        return (1e6,)
                    
                    # ============================================================================
                    # SCIENTIFIC EXPLANATION:
                    # This section calculates the "fitness" (quality) of our solution using two parts:
                    # 1. Primary Objective: How close our solution is to the ideal target (1.0)
                    #    - We use abs() because being too high or too low is equally bad
                    #    - Think of it like trying to hit exactly 1.0 on a dartboard
                    # 2. Sparsity Penalty: A penalty for making the solution too complex
                    #    - We multiply each parameter by self.alpha (a weight factor)
                    #    - This encourages simpler solutions (like Occam's Razor)
                    # ============================================================================

                    # Calculate how far we are from our target value of 1.0
                    # Example: If singular_response is 1.2, primary_objective will be 0.2
                    primary_objective = abs(singular_response - 1.0)
                    
                    # Calculate sparsity penalty quickly with NumPy
                    try:
                        alpha_eff = float(self.alpha) if np.isfinite(self.alpha) else 0.0
                    except Exception:
                        alpha_eff = 0.0
                    ind_np = np.asarray(individual, dtype=float)
                    ind_np[~np.isfinite(ind_np)] = 0.0
                    sparsity_penalty = float(alpha_eff * np.sum(np.abs(ind_np)))
                    
                    # Calculate sum of percentage differences
                    # percent_diff is defined as the value in the nested dictionary structure:
                    # results["percentage_differences"][mass_key][criterion]
                    # It represents the percentage difference for a given mass_key and criterion.
                    # It is defined in the FRF.py 
                    
                    percentage_error_sum = 0.0
                    if "percentage_differences" in results:
                        for mass_key, pdiffs in results["percentage_differences"].items():
                            for criterion, percent_diff in pdiffs.items():
                                # percent_diff is the value for this criterion under this mass_key
                                # Use absolute value to prevent positive and negative errors from cancelling
                                try:
                                    v = float(percent_diff)
                                    if np.isfinite(v):
                                        percentage_error_sum += abs(v)
                                except Exception:
                                    continue
                    
                    # Activation-based penalty and cost term
                    try:
                        thr = float(self.dva_activation_threshold)
                    except Exception:
                        thr = 0.0
                    try:
                        pen_per = float(self.dva_activation_penalty)
                    except Exception:
                        pen_per = 0.0
                    # Vectorized activation mask and penalty
                    try:
                        active_mask = np.abs(ind_np) >= thr
                        activation_count = int(np.count_nonzero(active_mask))
                    except Exception:
                        active_mask = np.zeros_like(ind_np, dtype=bool)
                        activation_count = 0
                    activation_penalty_term = float(activation_count) * pen_per if np.isfinite(pen_per) else 0.0

                    # ENHANCED COST-BENEFIT ANALYSIS WITH HIERARCHICAL STRUCTURE
                    cost_term = 0.0
                    benefit_cost_ratio = 0.0
                    # Fast normalized active-cost baseline
                    if 'cost_arrays' in locals() and cost_arrays is not None:
                        try:
                            active_all = float(sum(np.sum(arr[active_mask]) for arr in cost_arrays.values()))
                        except Exception:
                            active_all = 0.0
                        cost_sum_active = active_all
                        denom = total_cost_all if total_cost_all > 0 else 1.0
                    else:
                        if 'fallback_cost_arr' in locals() and fallback_cost_arr is not None:
                            try:
                                cost_sum_active = float(np.sum(fallback_cost_arr[active_mask]))
                            except Exception:
                                cost_sum_active = 0.0
                            denom = float(self._dva_cost_denominator) if getattr(self, '_dva_cost_denominator', 0.0) else 1.0
                        else:
                            cost_sum_active = 0.0
                            denom = 1.0
                    try:
                        # If enhanced cost is disabled, fallback to normalized active cost
                        if not self.use_enhanced_cost:
                            cost_term = cost_sum_active / denom if denom > 0 else 0.0
                        else:
                            # 1. Calculate benefit metrics
                            benefit_primary = 1- abs(singular_response - 1.0)
                            benefit_accuracy = 1- percentage_error_sum / 100.0
                            benefit_sparsity = 1- sparsity_penalty
                            bw_p, bw_a, bw_s = self.benefit_weights
                            total_benefit = (bw_p * benefit_primary + bw_a * benefit_accuracy + bw_s * benefit_sparsity)

                        # 2. Hierarchical cost categorization (vectorized)
                        cost_categories = {
                            'material': {'params': [], 'weight': 0.4, 'total': 0.0, 'active': 0.0},
                            'manufacturing': {'params': [], 'weight': 0.35, 'total': 0.0, 'active': 0.0},
                            'maintenance': {'params': [], 'weight': 0.15, 'total': 0.0, 'active': 0.0},
                            'operational': {'params': [], 'weight': 0.1, 'total': 0.0, 'active': 0.0}
                        }
                        if 'cost_arrays' in locals() and cost_arrays is not None:
                            for cat, arr in cost_arrays.items():
                                try:
                                    cost_categories[cat]['total'] = float(np.sum(arr))
                                    cost_categories[cat]['active'] = float(np.sum(arr[active_mask]))
                                except Exception:
                                    pass
                        else:
                            if 'fallback_cost_arr' in locals() and fallback_cost_arr is not None:
                                cat = 'operational'
                                try:
                                    cost_categories[cat]['total'] = float(np.sum(fallback_cost_arr))
                                    cost_categories[cat]['active'] = float(np.sum(fallback_cost_arr[active_mask]))
                                except Exception:
                                    pass

                        # 4. Calculate category-specific benefit-cost ratios
                        category_ratios = {}
                        total_weighted_cost = 0.0
                        for cat_name, cat_data in cost_categories.items():
                            if cat_data['total'] > 0:
                                cat_benefit = total_benefit * cat_data['weight']
                                cat_ratio = cat_benefit / cat_data['total'] if cat_data['active'] > 0 else float('inf')
                                category_ratios[cat_name] = cat_ratio
                                normalized_cat_cost = cat_data['active'] / cat_data['total']
                                total_weighted_cost += cat_data['weight'] * normalized_cat_cost
                            else:
                                category_ratios[cat_name] = float('inf')
                                total_weighted_cost += cat_data['weight'] * (1.0 if cat_data['active'] > 0 else 0.0)

                        # 5. Adaptive weighting based on user-defined generation ratio
                        # Use the user-defined generation ratio instead of dynamic calculation
                        generation_ratio = self.generation_ratio
                        benefit_weight = float(self.benefit_weight_start) + (float(self.benefit_weight_end) - float(self.benefit_weight_start)) * generation_ratio; benefit_weight = min(1.0, max(0.0, benefit_weight))
                        cost_weight = 1.0 - benefit_weight

                        # 6. Combined cost-benefit term
                        if total_benefit > 1e-10:
                            overall_bcr = total_benefit / max(total_weighted_cost, 1e-10)
                            benefit_cost_ratio = 1.0 / overall_bcr
                        else:
                            benefit_cost_ratio = 1e6
                        cost_term = cost_weight * total_weighted_cost + benefit_weight * benefit_cost_ratio

                        # Store detailed cost information
                        individual.cost_categories = cost_categories
                        individual.category_ratios = category_ratios
                        individual.total_benefit = total_benefit
                        individual.benefit_cost_ratio = benefit_cost_ratio
                    except Exception as e:
                        # Fallback to original normalized cost
                        self.update.emit(f"Warning: Enhanced cost calculation failed: {str(e)}")
                        cost_term = cost_sum_active / denom if denom > 0 else 0.0
                        individual.cost_categories = {}
                        individual.category_ratios = {}
                        individual.total_benefit = 0.0
                        individual.benefit_cost_ratio = 0.0

                    # Store the fitness components in the individual's attributes
                    # This allows us to access them later for detailed reporting
                    individual.primary_objective = primary_objective
                    individual.sparsity_penalty = sparsity_penalty
                    individual.percentage_error = percentage_error_sum/100.0
                    individual.activation_penalty = activation_penalty_term
                    # Store scaled cost term for consistency with fitness calculation
                    individual.cost_term = cost_term / self.cost_scale_factor

                    # Combine all components to get final score:
                    # 1. Primary objective: Distance from target value of 1.0
                    # 2. Sparsity penalty: Encourages simpler solutions
                    # 3. Percentage error sum: Sum of all percentage differences from target values (scaled by percentage_error_scale)
                    # 4. Activation penalty term: Penalty per active parameter above threshold
                    # 5. Cost term: Normalized cost of active parameters (scaled by cost_scale_factor)
                    # Lower score = better solution (like golf scoring)

                    # Debug output suppressed to avoid console noise during optimization

                    fitness = (
                        primary_objective
                        + sparsity_penalty
                        + percentage_error_sum/self.percentage_error_scale
                        + activation_penalty_term
                        + individual.cost_term
                    )
                    # Memoize results for identical parameter vectors
                    try:
                        with cache_lock:
                            frf_cache[dva_parameters_tuple] = (
                                primary_objective,
                                sparsity_penalty,
                                individual.percentage_error,
                                activation_penalty_term,
                                cost_term,
                                float(fitness)
                            )
                    except Exception:
                        pass
                    return (fitness,)
                except Exception as e:
                    # If anything goes wrong (like  math error or invalid input)
                    # We log the error and return a very high score (1e6) to indicate failure
                    # This is like getting disqualified in a competition
                    self.update.emit(f"Warning: FRF evaluation failed: {str(e)}")
                    return (1e6,)

            # ============================================================================
            # GENETIC ALGORITHM SETUP AND EXECUTION
            # ============================================================================
            
            # Register our evaluation function with DEAP's toolbox
            # Think of this like setting up the rules for a competition:
            # - evaluate: How we score each solution
            # - mate: How we combine two good solutions to create new ones
            # - mutate: How we randomly tweak solutions to explore new possibilities
            # - select: How we choose which solutions get to reproduce
            toolbox.register("evaluate", evaluate_individual)  # Our scoring function

            # Parallel evaluator to utilize multiple CPU cores for fitness evaluations
            def parallel_evaluate(individuals):
                if not individuals:
                    return []
                # Determine worker count once; retain at least 1
                workers = getattr(self, '_eval_workers', None)
                if workers is None:
                    try:
                        cpu_n = os.cpu_count() or 1
                    except Exception:
                        cpu_n = 1
                    # Leave one core free for UI/OS
                    workers = max(1, cpu_n - 1)
                    self._eval_workers = workers
                # Avoid oversubscription when evaluating small batches
                eff_workers = min(workers, max(1, len(individuals)))
                if eff_workers <= 1:
                    return [toolbox.evaluate(ind) for ind in individuals]
                # Reuse a persistent executor to avoid per-call creation overhead
                if self._executor is None:
                    try:
                        self._executor = ThreadPoolExecutor(max_workers=eff_workers, thread_name_prefix="gaeval")
                    except Exception:
                        self._executor = ThreadPoolExecutor(max_workers=eff_workers)
                return list(self._executor.map(toolbox.evaluate, individuals))
            toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Blend two solutions together

            # ============================================================================
            # MUTATION FUNCTION
            # ============================================================================
            # This function randomly changes some parameters of a solution
            # Think of it like making small random adjustments to a recipe
            def mutate_individual(individual, indpb=0.1):
                # First, check if we should stop or pause (like if someone called timeout)
                if self._check_pause_abort():
                    return (individual,)
                    
                # Go through each parameter in our solution
                for i in range(len(individual)):
                    # Skip parameters that are fixed (like ingredients you can't change)
                    if i in fixed_parameters:
                        continue 
                    # 10% chance (indpb=0.1) to mutate each parameter
                    if random.random() < indpb:
                        # Get the allowed range for this parameter
                        min_val, max_val = parameter_bounds[i]
                        # Make a small random change using dynamic mutation scale of the parameter range
                        scale = float(self.mutation_scale) if hasattr(self, 'mutation_scale') else 0.10
                        perturb = random.uniform(-scale * (max_val - min_val), scale * (max_val - min_val))
                        individual[i] += perturb
                        # Make sure we stay within allowed bounds
                        individual[i] = max(min_val, min(individual[i], max_val))
                return (individual,)

            # Register our mutation function and selection method
            toolbox.register("mutate", mutate_individual)
            toolbox.register("select", tools.selTournament, tournsize=3)  # Tournament selection with 3 competitors

            # ============================================================================
            # INITIAL POPULATION SETUP WITH FLEXIBLE SEEDING STRATEGY
            # ============================================================================
            # This section initializes the first generation of candidate solutions for the genetic algorithm (GA).
            # The seeding method (random, Sobol, or Latin Hypercube Sampling) determines how the initial population is distributed in parameter space.
            # The code uses a helper function to lazily initialize a Quasi-Monte Carlo (QMC) engine for low-discrepancy sampling if required.

            # --- Plain English Explanation ---
            # The code emits a message to indicate that the population initialization is starting.
            # It defines a helper function to set up a QMC engine (for Sobol or LHS seeding) only when needed.
            # The QMC engine is used to generate well-distributed initial samples in the parameter space, improving the diversity of the initial population.
            # If the QMC engine cannot be initialized (e.g., due to an error or unsupported method), the code falls back to random seeding.

            # --- Python Concepts and Science ---
            # - Uses function definition for modularity and lazy initialization (function is only called if needed).
            # - Uses exception handling (try/except) to ensure robustness and fallback behavior.
            # - Uses object attributes (self._qmc_engine) to store state across function calls.
            # - Applies conditional logic (if/elif/else) to select the appropriate seeding method.
            # - Utilizes external libraries (e.g., scipy.stats.qmc) for advanced sampling techniques.

            # --- How Population Changes After Each Generation ---
            # After the initial population is created, the genetic algorithm evolves the population over generations.
            # In each generation, individuals are selected, mated, and mutated to produce new candidate solutions.
            # The population thus gradually shifts toward regions of the parameter space with better fitness, guided by the evaluation function and genetic operators.

            self.update.emit("Initializing population...")  # Notify that population initialization is starting

            def _ensure_qmc_engine():
                """
                Lazily initialize the QMC (Quasi-Monte Carlo) engine for low-discrepancy sampling if required by the seeding method.
                Sets self._qmc_engine to an appropriate QMC sampler or None if not applicable.

                This function is called only when a QMC-based seeding method (Sobol or LHS) is requested.
                It ensures that the QMC engine is only created once and reused for subsequent calls.
                """
                # Check if the QMC engine has already been initialized
                if self._qmc_engine is not None:
                    # If already initialized, do nothing and return immediately
                    return

                try:
                    # Determine the number of parameters (dimensions) in the optimization problem
                    dim = len(parameter_bounds)  # parameter_bounds is a list of (low, high) tuples for each parameter

                    # If there are no parameters to sample (dim <= 0), set engine to None and return
                    if dim <= 0:
                        self._qmc_engine = None  # No parameters to sample; QMC engine not needed
                        return
                    # - qmc.Sobol: Implements the Sobol sequence.
                    # - qmc.LatinHypercube: Implements Latin Hypercube Sampling.

                    # Check which seeding method is requested
                    if self.seeding_method == "sobol":
                        # If Sobol is selected, use the Sobol sequence for quasi-random, low-discrepancy sampling
                        # scramble=True randomizes the sequence for better uniformity
                        # self.seeding_seed is used for reproducibility
                        self._qmc_engine = qmc.Sobol(d=dim, scramble=True, seed=self.seeding_seed)
                        # The Sobol engine will be used to generate initial population samples

                    elif self.seeding_method == "lhs":
                        # If Latin Hypercube Sampling (LHS) is selected, use the LHS engine
                        # LHS ensures stratified sampling across all dimensions
                        self._qmc_engine = qmc.LatinHypercube(d=dim, seed=self.seeding_seed)
                        # The LHS engine will be used to generate initial population samples

                    else:
                        # If the seeding method is not supported (e.g., "random" or unknown), do not use a QMC engine
                        # This ensures that only supported methods use QMC, and others fall back to random sampling
                        self._qmc_engine = None

                except Exception as qe:
                    # If any error occurs during QMC engine initialization (e.g., invalid parameters, library issues)
                    # Emit a warning message to the user interface or log, including the method and error details
                    self.update.emit(
                        f"Warning: Failed to initialize QMC engine ({self.seeding_method}): {str(qe)}. Falling back to random seeding."
                    )
                    # Set the QMC engine to None to indicate that random seeding should be used as a fallback
                    self._qmc_engine = None


            # Helper function to generate initial individuals for the GA population
            # This function chooses the seeding strategy based on user selection and parameter configuration.
            # Science: The way the initial population is seeded can have a significant impact on the convergence speed and diversity of a genetic algorithm.
            def generate_seed_individuals(count):
                # Check if all parameters are fixed (i.e., no variables to optimize)
                # Science: If all parameters are fixed, the search space collapses to a single point, so every individual is identical.
                all_fixed = len(fixed_parameters) == len(parameter_bounds)
                if all_fixed:
                    # Build the fixed vector in the correct order
                    fixed_vec = [fixed_parameters[i] for i in range(len(parameter_bounds))]
                    # Return a list of identical individuals, each with the fixed parameter values
                    return [creator.Individual(list(fixed_vec)) for _ in range(count)]

                # If random seeding is selected or count is zero or negative, use random individuals
                # Science: Random seeding provides a uniform, unbiased sampling of the search space, but may not cover it efficiently in high dimensions.
                if self.seeding_method == "random" or count <= 0:
                    return [toolbox.individual() for _ in range(count)]

                # For QMC-based seeding (Sobol or LHS), ensure the QMC engine is initialized
                # Science: Quasi-Monte Carlo (QMC) methods like Sobol and LHS generate low-discrepancy sequences, which fill the space more uniformly than pure random sampling.
                _ensure_qmc_engine()
                if self._qmc_engine is None:
                    # If QMC engine failed to initialize, fall back to random seeding
                    return [toolbox.individual() for _ in range(count)]

                # Use QMC engine to generate low-discrepancy samples and scale them to parameter bounds
                try:
                    # If using memory seeding directly, short-circuit to it
                    if self.seeding_method == "memory" and 'memory_seeder' in locals() and memory_seeder is not None:
                        seeds = memory_seeder.propose(count)
                        return [creator.Individual([float(row[i]) for i in range(len(parameter_bounds))]) for row in seeds]
                    # Best-of-Pool: sample a large pool then evaluate and choose the best pop
                    if self.seeding_method == "best":
                        pool_n = int(max(count, ceil(getattr(self, 'best_pool_mult', 5.0) * count)))
                        pool = []
                        # draw QMC
                        m = int(np.ceil(np.log2(max(1, pool_n))))
                        samples = self._qmc_engine.random_base2(m=m)[:pool_n]
                        lows = np.array([b[0] for b in parameter_bounds], dtype=float)
                        highs = np.array([b[1] for b in parameter_bounds], dtype=float)
                        span = (highs - lows)
                        span[span == 0.0] = 0.0
                        scaled = lows + samples * span
                        for idx, val in fixed_parameters.items():
                            scaled[:, idx] = val
                        for row in scaled:
                            pool.append(creator.Individual([float(row[i]) for i in range(len(parameter_bounds))]))
                        # Evaluate pool
                        fits = parallel_evaluate(pool)
                        for ind, fit in zip(pool, fits):
                            ind.fitness.values = fit
                        pool_sorted = sorted(pool, key=lambda ind: ind.fitness.values[0])
                        # diversity stride selection
                        k = max(1, count)
                        step = max(1, int(1.0 / max(1e-6, getattr(self, 'best_diversity_frac', 0.2))))
                        out = []
                        i = 0
                        while len(out) < k and i < len(pool_sorted):
                            out.append(pool_sorted[i])
                            i += step
                        j = 0
                        while len(out) < k and j < len(pool_sorted):
                            cand = pool_sorted[j]
                            if cand not in out:
                                out.append(cand)
                            j += 1
                        return out[:k]
                    # Generate 'count' samples in [0,1)^d, where d is the number of parameters (default QMC path)
                    samples = self._qmc_engine.random(count)  # shape: (count, num_parameters)
                    # Extract lower and upper bounds for each parameter
                    lows = np.array([b[0] for b in parameter_bounds], dtype=float)
                    highs = np.array([b[1] for b in parameter_bounds], dtype=float)
                    span = (highs - lows)
                    # Science: The span is the range for each parameter; multiplying by the QMC sample stretches the [0,1) sample to the parameter's domain.
                    # Avoid NaNs if any parameter is fixed (span == 0)
                    span[span == 0.0] = 0.0
                    # Scale samples from [0,1) to [low, high) for each parameter
                    scaled = lows + samples * span
                    # Enforce fixed parameters exactly (overwriting the sampled value)
                    for idx, val in fixed_parameters.items():
                        scaled[:, idx] = val
                    # Convert each row of scaled samples into a DEAP Individual
                    individuals = []
                    for row in scaled:
                        # Each individual is a list of parameter values (floats)
                        ind = creator.Individual([float(row[i]) for i in range(len(parameter_bounds))])
                        individuals.append(ind)
                    return individuals
                except Exception as gen_err:
                    # If any error occurs during QMC sampling, emit a warning and fall back to random seeding
                    self.update.emit(f"Warning: QMC sampling error: {str(gen_err)}. Falling back to random seeding.")
                    return [toolbox.individual() for _ in range(count)]

            # ---------------------------------------------------------------------------
            # ---------------------------------------------------------------------------

            # --- Plain English Explanation ---
            # This block initializes the population for the genetic algorithm (GA), using either a neural network-based seeder
            # or a classical seeding method (random, Sobol, or LHS). If neural seeding is enabled, it prepares the neural seeder
            # with parameter bounds and fixed values, then emits a message about the chosen method. The initial population is generated,
            # evaluated, and, if neural seeding is used, the results are fed back to the neural seeder for future use. Metrics and
            # history are updated for tracking and visualization.

            # --- Python Concepts and Science ---
            # - Conditional logic (if/else) is used to select the seeding strategy.
            # - Numpy arrays are used for efficient numerical operations on parameter bounds.
            # - Loops and array indexing set up masks and values for fixed/variable parameters.
            # - Object instantiation (NeuralSeeder) demonstrates OOP principles.
            # - The population is generated via a helper function, and fitness is evaluated using map and a toolbox function.
            # - Data is collected and stored for metrics and history, supporting reproducibility and analysis.
            # - Exception handling ensures robustness if neural seeder data feeding fails.

            # --- Rewritten with Professional Comments ---

            # Decide which seeding strategy to use for the initial population.
            # The code chooses the neural seeder if self.use_neural_seeding is True.
            # This is set during initialization based on the seeding_method argument or use_neural_seeding flag.
            # If neural seeding is not enabled, it falls back to the method specified by self.seeding_method
            # (which can be "random", "sobol", or "lhs").
            #
            # The logic is:
            # - If self.use_neural_seeding is True, use the neural seeder.
            # - Else, use the classical seeding method as specified.
            #
            # This is determined by the following lines in __init__:
            #   self.seeding_method = (seeding_method or "random").lower()
            #   if self.seeding_method not in ("random", "sobol", "lhs", "neural"):
            #       self.seeding_method = "random"
            #   if use_neural_seeding:
            #       self.seeding_method = "neural"
            #   self.use_neural_seeding = (self.seeding_method == "neural")
            #
            # So, if the user requests "neural" or sets use_neural_seeding=True, neural seeder is used.
            # Otherwise, the code uses the method in self.seeding_method ("random", "sobol", or "lhs").

        
            # ---------------------------------------------------------------------------
            # ---------------------------------------------------------------------------

            # This block is responsible for generating the initial population for the genetic algorithm (GA).
            # The initial population can be seeded using either a neural network-based approach (neural seeder)
            # or a classical statistical method (random, Sobol, or Latin Hypercube Sampling).
            # The choice of seeding method is determined by the configuration set during initialization.

            # 1. Initialize the neural seeder reference to None.
            #    This will be used only if neural seeding is enabled.
            neural_seeder = None

            # 2. Decide which seeding strategy to use.
            #    If neural or memory seeding is enabled, prepare required data and instantiate.
            if self.use_neural_seeding:
                # --- Neural Seeder Preparation ---

                # a. Convert parameter bounds to numpy arrays for efficient numerical operations.
                #    - 'lows' contains the lower bounds for each parameter.
                #    - 'highs' contains the upper bounds for each parameter.
                lows = np.array([b[0] for b in parameter_bounds], dtype=float)
                highs = np.array([b[1] for b in parameter_bounds], dtype=float)

                # b. Prepare arrays to indicate which parameters are fixed and their fixed values.
                #    - 'fixed_mask' is a boolean array: True if the parameter is fixed, False otherwise.
                #    - 'fixed_values' contains the fixed value for each parameter if fixed, or a default (midpoint) otherwise.
                fixed_mask = np.zeros(len(parameter_bounds), dtype=bool)
                fixed_values = np.zeros(len(parameter_bounds), dtype=float)
                for i in range(len(parameter_bounds)):
                    if i in fixed_parameters:
                        # If the parameter is fixed, mark it and set its value.
                        fixed_mask[i] = True
                        fixed_values[i] = float(fixed_parameters[i])
                    else:
                        # If not fixed, use the midpoint of the bounds as a placeholder.
                        fixed_values[i] = float((lows[i] + highs[i]) * 0.5)

                # c. Instantiate the NeuralSeeder object with all relevant hyperparameters.
                #    The neural seeder is a model that learns to generate diverse and promising initial solutions.
                #    It is configured with architecture, training, and acquisition parameters.
                neural_seeder = NeuralSeeder(
                    lows=lows,                                 # Lower bounds for each parameter
                    highs=highs,                               # Upper bounds for each parameter
                    fixed_mask=fixed_mask,                     # Boolean mask for fixed parameters
                    fixed_values=fixed_values,                 # Values for fixed parameters
                    ensemble_n=self.neural_ensemble_n,         # Number of models in the ensemble
                    hidden=self.neural_hidden,                 # Hidden layer size
                    layers=self.neural_layers,                 # Number of layers
                    dropout=self.neural_dropout,               # Dropout rate for regularization
                    weight_decay=self.neural_weight_decay,     # Weight decay for regularization
                    epochs=self.neural_epochs,                 # Number of training epochs
                    time_cap_ms=self.neural_time_cap_ms,       # Time cap for training (ms)
                    pool_mult=self.neural_pool_mult,           # Pool multiplier for candidate generation
                    epsilon=self.neural_epsilon,               # Exploration parameter for acquisition
                    acq_type=self.neural_acq_type,             # Acquisition function type (e.g., UCB, EI)
                    device=self.neural_device,                 # Device for computation (e.g., 'cpu', 'cuda')
                    seed=self.seeding_seed if isinstance(self.seeding_seed, (int, np.integer)) else None,  # Random seed
                    diversity_min_dist=0.03,                   # Minimum diversity distance for generated samples
                    enable_grad_refine=self.neural_enable_grad_refine,  # Whether to use gradient refinement
                    grad_steps=self.neural_grad_steps,         # Number of gradient refinement steps
                )

                # d. Inform the user (via UI or log) that neural surrogate seeding is being used.
                self.update.emit("Seeding method: Neural surrogate (UCB/EI)")

            else:
                # --- Classical Seeding Methods ---
                # If neural seeding is not enabled, emit a message about the classical seeding method in use.
                if self.seeding_method == "random":
                    self.update.emit("Seeding method: Random uniform within bounds")
                elif self.seeding_method == "sobol":
                    self.update.emit("Seeding method: Sobol low-discrepancy sequence")
                elif self.seeding_method == "lhs":
                    self.update.emit("Seeding method: Latin Hypercube Sampling (LHS)")
                elif self.seeding_method == "memory":
                    try:
                        lows = np.array([b[0] for b in parameter_bounds], dtype=float)
                        highs = np.array([b[1] for b in parameter_bounds], dtype=float)
                        fixed_mask = np.zeros(len(parameter_bounds), dtype=bool)
                        fixed_values = np.zeros(len(parameter_bounds), dtype=float)
                        for i in range(len(parameter_bounds)):
                            if i in fixed_parameters:
                                fixed_mask[i] = True
                                fixed_values[i] = float(fixed_parameters[i])
                            else:
                                fixed_values[i] = float((lows[i] + highs[i]) * 0.5)
                        # Create memory seeder with a persistent file near project root
                        mem_file = os.path.join(os.getcwd(), 'seeding_memory.json')
                        memory_seeder = MemorySeeder(
                            lows=lows,
                            highs=highs,
                            fixed_mask=fixed_mask,
                            fixed_values=fixed_values,
                            max_size=2000,
                            top_k=100,
                            sigma_scale=0.05,
                            exploration_frac=0.2,
                            replay_frac=0.2,
                            file_path=mem_file,
                            seed=int(self.seeding_seed) if isinstance(self.seeding_seed, (int, np.integer)) else None,
                        )
                        self.update.emit("Seeding method: Memory (replay + jitter + explore)")
                    except Exception as _:
                        memory_seeder = None

            # 3. Generate the initial population using the selected seeding strategy.
            #    - The function 'generate_seed_individuals' encapsulates the logic for both neural and classical seeding.
            #    - The population size is determined by self.ga_pop_size.
            population = generate_seed_individuals(self.ga_pop_size)
            # If memory seeding active, mix in memory proposals to improve initial pool
            try:
                if self.seeding_method == "memory" and memory_seeder is not None and len(population) > 0:
                    need = max(0, self.ga_pop_size - len(population))
                    mem_extra = memory_seeder.propose(need if need > 0 else len(population))
                    if mem_extra:
                        # Replace worst by memory suggestions for stronger start
                        # Convert to individuals
                        mem_inds = [creator.Individual(list(x)) for x in mem_extra]
                        # Keep size consistent
                        if need > 0:
                            population.extend(mem_inds[:need])
                        else:
                            population = mem_inds[:len(population)]
            except Exception:
                pass

            # 4. Record the initial population size for metrics tracking, if enabled.
            #    - This helps track how the population size evolves over generations.
            if self.track_metrics:
                self.metrics['pop_size_history'].append(len(population))

            # 5. Evaluate the fitness of each individual in the initial population.
            #    - The fitness function is applied to each individual using the DEAP toolbox.
            #    - The results are assigned to the individual's fitness attribute.
            self.update.emit("Evaluating initial population...")
            fitnesses = parallel_evaluate(population)
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit  # Assign the computed fitness to the individual

            # Seed surrogate dataset with initial evaluations so screening can engage
            if self.use_surrogate:
                try:
                    xs = [list(ind) for ind in population]
                    ys = [float(f[0]) for f in fitnesses]
                    self._surrogate_X.extend(xs)
                    self._surrogate_y.extend(ys)
                    # Maintain normalized cache
                    if self._surrogate_norm_lows is not None and self._surrogate_norm_spans is not None:
                        Xn = (np.array(xs, dtype=float) - self._surrogate_norm_lows) / self._surrogate_norm_spans
                        self._surrogate_Xn.extend([row.tolist() for row in Xn])
                    # Trim to cap
                    if len(self._surrogate_X) > self.surrogate_dataset_max:
                        keep = self.surrogate_dataset_max
                        self._surrogate_X = self._surrogate_X[-keep:]
                        self._surrogate_y = self._surrogate_y[-keep:]
                        if self._surrogate_Xn:
                            self._surrogate_Xn = self._surrogate_Xn[-keep:]
                except Exception:
                    pass

            # 6. If neural seeding was used, feed the evaluated data back to the neural seeder for training.
            #    - This allows the neural seeder to learn from the initial population and improve future suggestions.
            #    - The data consists of parameter vectors (X_data) and their corresponding fitness values (y_data).
            #    - If metrics tracking is enabled, log the initial state of the neural seeder for generation 0.
            if self.use_neural_seeding and neural_seeder is not None:
                try:
                    # a. Extract parameter vectors and fitness values from the population.
                    X_data = [list(ind) for ind in population]  # Each individual's parameter vector
                    y_data = [float(ind.fitness.values[0]) for ind in population]  # Each individual's fitness

                    # b. Add the data to the neural seeder for training.
                    neural_seeder.add_data(X_data, y_data)

                    # c. Log the initial neural seeder state for metrics/history (generation 0).
                    if self.track_metrics:
                        init_beta = self.neural_beta_min
                        init_eps = self.neural_epsilon
                        self.metrics['neural_history'].append({
                            'generation': 0,
                            'train_time_ms': 0.0,         # No training yet at generation 0
                            'epochs': 0,                  # No epochs yet
                            'beta': init_beta,            # Initial beta value for acquisition
                            'pool_mult': self.neural_pool_mult,
                            'epsilon': init_eps,          # Initial epsilon value
                            'acq': self.neural_acq_type   # Acquisition function type
                        })
                except Exception:
                    # Silently ignore any errors in neural seeder data feeding.
                    # This ensures that a failure in the neural seeder does not crash the optimization.
                    pass
            # Feed memory with initial evaluations
            try:
                if self.seeding_method == "memory" and memory_seeder is not None:
                    X_data = [list(ind) for ind in population]
                    y_data = [float(ind.fitness.values[0]) for ind in population]
                    memory_seeder.add_data(X_data, y_data)
            except Exception:
                pass

            # --- Summary of Key Concepts ---
            # - This block demonstrates conditional logic for feature toggling (neural vs. classical seeding).
            # - It uses numpy for efficient numerical operations and array management.
            # - Object-oriented programming is shown via NeuralSeeder instantiation and method calls.
            # - List comprehensions and mapping are used for data extraction and fitness evaluation.
            # - Exception handling ensures robustness and fault tolerance.
            # - Emitting signals (self.update.emit) supports UI/logging for transparency and debugging.
            # - Metrics tracking enables reproducibility and analysis of the optimization process.


            # ============================================================================
            # EVOLUTION LOOP
            # ============================================================================
            # This is the main loop where the genetic algorithm evolves the population
            # over multiple generations to search for optimal solutions.
            # The following variables and logic set up the tracking and adaptive control
            # mechanisms for the evolutionary process.

            self.update.emit("Starting evolution...")  # Notify UI/log that evolution is starting

            # Initialize variables to track the best solution found so far.
            best_fitness_overall = float('inf')  # Best (lowest) fitness value seen across all generations
            best_ind_overall = None              # The individual (solution) with the best fitness

            # --- ML Bandit Controller setup ---
            # If enabled, use a multi-armed bandit (MAB) controller to adaptively tune
            # genetic algorithm hyperparameters (crossover/mutation probabilities, population size)
            # using an Upper Confidence Bound (UCB) strategy.
            if self.use_ml_adaptive:
                # Define the discrete action space for the bandit controller:
                # - deltas: relative changes to crossover (cxpb) and mutation (mutpb) probabilities
                # - pop_multipliers: relative changes to population size
                # Each action is a tuple: (delta_cxpb, delta_mutpb, pop_multiplier)
                # Make the action space for deltas and population multipliers much more extensive and comprehensive
                # - deltas: even finer granularity and much wider range for crossover/mutation probability changes
                # Keep the action space compact to ensure fast controller decisions each generation
                deltas = [-0.30, -0.15, 0.0, 0.15, 0.30]

                # - pop_multipliers: much broader and finer set of possible population size multipliers, including more small and large values
                pop_multipliers = [0.75, 1.0, 1.25]

                # Create the full action space as a Cartesian product of deltas and multipliers
                ml_actions = [(dcx, dmu, pm) for dcx in deltas for dmu in deltas for pm in pop_multipliers]

                # Initialize tracking for each action:
                # - ml_counts: how many times each action has been selected
                # - ml_sums: cumulative reward (e.g., improvement in fitness) for each action
                # - ml_t: total number of action selections (time step)
                ml_counts = [0 for _ in ml_actions]
                ml_sums = [0.0 for _ in ml_actions]
                ml_t = 0

                def ml_select_action(current_cx, current_mu, current_pop):
                    """
                    Select an action (i.e., a set of hyperparameter adjustments) using the UCB algorithm.

                    Args:
                        current_cx (float): Current crossover probability.
                        current_mu (float): Current mutation probability.
                        current_pop (int): Current population size.

                    Returns:
                        idx (int): Index of the selected action in ml_actions.
                        new_cx (float): New crossover probability after applying the action.
                        new_mu (float): New mutation probability after applying the action.
                        new_pop (int): New population size after applying the action.
                    """
                    nonlocal ml_t
                    ml_t += 1  # Increment the time step (number of action selections)

                    # Compute UCB scores for each action:
                    # - If an action has never been tried, assign it infinite score to ensure exploration.
                    # - Otherwise, compute a comprehensive reward that considers improvement, speed, diversity, and effort,
                    #   similar to the RL reward system, plus the UCB exploration bonus.
                    # Compute the best action in a single pass to avoid per-generation list building and sorting
                    best_score = -float('inf')
                    best_idx = 0
                    for i, _ in enumerate(ml_actions):
                        if ml_counts[i] == 0:
                            score = float('inf')
                        else:
                            avg = ml_sums[i] / ml_counts[i]
                            try:
                                last_best = self.metrics['best_fitness_per_gen'][-2] if len(self.metrics['best_fitness_per_gen']) > 1 else None
                                imp = (last_best - min_fit) if (last_best is not None and last_best > min_fit) else 0.0
                                cv = std / (abs(mean) + 1e-12)
                                effort = max(1.0, evals_this_gen)
                                reward = (imp / max(gen_time, 1e-6)) / effort - self.ml_diversity_weight * abs(cv - self.ml_diversity_target)
                                blended = self.ml_historical_weight * avg + self.ml_current_weight * reward
                            except Exception:
                                blended = avg
                            bonus = self.ml_ucb_c * sqrt(log(max(ml_t, 1)) / ml_counts[i])
                            score = blended + bonus
                        if score > best_score:
                            best_score = score
                            best_idx = i
                    idx = best_idx

                    # --- Retrieve the Action Parameters ---
                    # Each action is a tuple (dcx, dmu, pm):
                    #   - dcx: relative change to crossover probability (e.g., +0.05 means increase by 5%)
                    #   - dmu: relative change to mutation probability
                    #   - pm:  population size multiplier (e.g., 1.2 means increase pop size by 20%)
                    dcx, dmu, pm = ml_actions[idx]

                    # --- Apply the Action to Current Hyperparameters ---
                    # Update the hyperparameters by applying the selected action, ensuring they remain within allowed bounds:
                    #   - Crossover probability (cxpb): must be between self.cxpb_min and self.cxpb_max
                    #   - Mutation probability (mutpb): must be between self.mutpb_min and self.mutpb_max
                    #   - Population size (pop): must be between self.pop_min and self.pop_max, and is rounded to int
                    # The new values are calculated as:
                    #   - new_cx = current_cx * (1 + dcx)
                    #   - new_mu = current_mu * (1 + dmu)
                    #   - new_pop = current_pop * pm
                    new_cx = min(self.cxpb_max, max(self.cxpb_min, current_cx * (1.0 + dcx)))
                    new_mu = min(self.mutpb_max, max(self.mutpb_min, current_mu * (1.0 + dmu)))
                    new_cx = min(self.cxpb_max, max(self.cxpb_min, current_cx * (1.0 + dcx)))
                    new_mu = min(self.mutpb_max, max(self.mutpb_min, current_mu * (1.0 + dmu)))
                    new_pop = int(min(self.pop_max, max(self.pop_min, round(current_pop * pm))))

                    # --- Return the Selected Action and New Hyperparameter Values ---
                    # idx: index of the selected action (for updating statistics later)
                    # new_cx: new crossover probability
                    # new_mu: new mutation probability
                    # new_pop: new population size
                    return idx, new_cx, new_mu, new_pop

                # --- Extensive Comments for ml_update and resize_population ---

                def ml_update(idx, reward):
                    """
                    Update the statistics for a given ML bandit action after observing its reward.

                    Args:
                        idx (int): The index of the action that was taken.
                        reward (float): The reward received for this action (e.g., improvement in fitness).

                    This function implements the standard update for a multi-armed bandit algorithm:
                    - It increments the count of times this action has been selected.
                    - It adds the observed reward to the running sum for this action.
                    These statistics are later used to compute the average reward and guide future action selection.
                    """
                    ml_counts[idx] += 1  # Increment the count for this action
                    ml_sums[idx] += float(reward)  # Add the reward to the running sum for this action

                def resize_population(pop, new_size):
                    """
                    Adjust the population size to match new_size, either by shrinking or growing.

                    Args:
                        pop (list): The current population (list of individuals).
                        new_size (int): The desired population size.

                    Returns:
                        list: The resized population.

                    This function handles both shrinking and growing the population:
                    - If shrinking, it selects the best individuals to keep (elitism).
                    - If growing, it generates new individuals using the current seeding strategy.
                      - If neural seeding is enabled and the neural seeder is sufficiently trained, it uses the neural seeder.
                      - Otherwise, it falls back to the default seeding method.

                    The function also adapts neural seeding parameters (beta and epsilon) based on stagnation and diversity,
                    and ensures that all new individuals are evaluated immediately to maintain a valid population.
                    """

                    # --- Shrinking the population (reduce size) ---
                    if new_size < len(pop):
                        # Use DEAP's selBest to keep the top-performing individuals
                        return tools.selBest(pop, new_size)

                    # --- Growing the population (increase size) ---
                    extra = new_size - len(pop)  # Number of new individuals needed
                    if extra > 0:
                        # --- Use neural seeding if enabled and neural seeder is ready ---
                        if (
                            self.use_neural_seeding and
                            neural_seeder is not None and
                            neural_seeder.size >= max(50, 5 * len(parameter_bounds))
                        ):
                            # --- Adapt beta parameter for neural seeding based on stagnation ---
                            # beta controls the exploration/exploitation tradeoff in neural seeding
                            beta = self.neural_beta_min
                            try:
                                if self.adaptive_rates:
                                    # Linearly interpolate beta between min and max based on stagnation
                                    # More stagnation â†’ higher beta (more exploration)
                                    beta = self.neural_beta_min + (
                                        (self.neural_beta_max - self.neural_beta_min) *
                                        min(1.0, max(0.0, self.stagnation_counter / max(1, self.stagnation_limit)))
                                    )
                            except Exception:
                                # If anything goes wrong, just use the minimum beta
                                pass

                            # --- Find the best fitness in the current population (for neural seeder guidance) ---
                            best_y = None
                            try:
                                # Only consider individuals with valid fitness
                                best_y = min(ind.fitness.values[0] for ind in pop if ind.fitness.valid)
                            except Exception:
                                # If no valid fitness values, leave as None
                                best_y = None

                            # --- Optionally adapt epsilon (exploration fraction) for neural seeding ---
                            # Epsilon controls how much randomness/exploration is used in neural seeding
                            eps = self.neural_epsilon
                            try:
                                if self.neural_adapt_epsilon and self.adaptive_rates:
                                    # Map stagnation into [0,1] (0 = no stagnation, 1 = max stagnation)
                                    stag_ratio = min(1.0, max(0.0, self.stagnation_counter / max(1, self.stagnation_limit)))
                                    # As stagnation increases, move epsilon toward eps_max (more exploration)
                                    eps = (1.0 - stag_ratio) * self.neural_eps_min + stag_ratio * self.neural_eps_max
                                    # Clamp epsilon to allowed range
                                    eps = max(self.neural_eps_min, min(eps, self.neural_eps_max))
                                    # Store the current epsilon for reference/metrics
                                    self.current_epsilon = eps
                            except Exception:
                                # If adaptation fails, just use the default epsilon
                                pass

                            # --- Generate new individuals using the neural seeder ---
                            # The neural seeder proposes new parameter vectors based on learned model
                            seeds = neural_seeder.propose(
                                count=extra,
                                beta=beta,
                                best_y=best_y,
                                exploration_fraction=eps
                            )
                            # Convert each seed (parameter vector) into a DEAP Individual and add to population
                            for s in seeds:
                                pop.append(creator.Individual(list(s)))

                            # --- Evaluate new individuals immediately ---
                            # This ensures that all individuals in the population have valid fitness values,
                            # which is important for selection and statistics.
                            need_eval = [ind for ind in pop if not ind.fitness.valid]
                            if need_eval:
                                fits_new = parallel_evaluate(need_eval)
                                for ind, fit in zip(need_eval, fits_new):
                                    ind.fitness.values = fit
                                if self.use_surrogate:
                                    try:
                                        xs_add = [list(ind) for ind in need_eval]
                                        ys_add = [float(f[0]) for f in fits_new]
                                        self._surrogate_X.extend(xs_add)
                                        self._surrogate_y.extend(ys_add)
                                        if self._surrogate_norm_lows is not None and self._surrogate_norm_spans is not None:
                                            Xn_add = (np.array(xs_add, dtype=float) - self._surrogate_norm_lows) / self._surrogate_norm_spans
                                            self._surrogate_Xn.extend([row.tolist() for row in Xn_add])
                                        if len(self._surrogate_X) > self.surrogate_dataset_max:
                                            keep = self.surrogate_dataset_max
                                            self._surrogate_X = self._surrogate_X[-keep:]
                                            self._surrogate_y = self._surrogate_y[-keep:]
                                            if self._surrogate_Xn:
                                                self._surrogate_Xn = self._surrogate_Xn[-keep:]
                                    except Exception:
                                        pass
                        else:
                            # --- Fallback: Use default seeding method if neural seeding is not available ---
                            new_inds = generate_seed_individuals(extra)
                            pop.extend(new_inds)
                    # Return the resized population (either shrunk or grown)
                    return pop
            
            
            # --- Reinforcement Learning (RL) Controller Setup ---
            #
            # This section configures and implements a Reinforcement Learning controller for adaptive
            # adjustment of genetic algorithm (GA) parameters: crossover probability, mutation probability,
            # and population size. The RL controller learns to select parameter adjustments that improve
            # optimization performance over time, using a simple Q-learning approach.
            #
            # The RL controller is only activated if self.use_rl_controller is True.

            elif self.use_rl_controller:
                # --- Define the action space for the RL agent ---
                # deltas: Much finer granularity and wider range for crossover/mutation probability changes
                #   - For example, a delta of 0.1 means "increase by 10%", -0.25 means "decrease by 25%", etc.
                # Keep RL action space compact for responsiveness
                deltas = [-0.30, -0.15, 0.0, 0.15, 0.30]

                # pop_multipliers: Much broader and finer set of possible population size multipliers, including more small and large values
                #   - For example, 0.5 means "halve the population", 1.5 means "increase by 50%", etc.
                pop_multipliers = [0.75, 1.0, 1.25]

                # rl_actions: All possible combinations of (crossover delta, mutation delta, pop multiplier).
                #   - This forms the discrete action space for the RL agent.
                rl_actions = [(dcx, dmu, pm) for dcx in deltas for dmu in deltas for pm in pop_multipliers]

                # --- Q-table Initialization ---
                # The Q-table stores the expected reward for each action in each state.
                # Here, we use a very simple state space: just two states (0 and 1).
                #   - In a more advanced implementation, the state could encode more information (e.g., stagnation, diversity).
                # Each state maps to a list of Q-values, one for each action.
                rl_q = {0: [0.0 for _ in rl_actions], 1: [0.0 for _ in rl_actions]}
                rl_state = 0  # Start in state 0 (could be extended for more complex state tracking)

                # --- RL Action Selection Function ---
                def rl_select_action(current_cx, current_mu, current_pop):
                    """
                    Select an action (parameter adjustment) using an epsilon-greedy policy:
                    - With probability epsilon, pick a random action (exploration).
                    - Otherwise, pick the action with the highest Q-value for the current state (exploitation).
                    The selected action determines how to adjust crossover, mutation, and population size.
                    """
                    if random.random() < self.rl_epsilon:
                        # Exploration: choose a random action index
                        idx = random.randrange(len(rl_actions))
                    else:
                        # Exploitation: choose the best-known action for the current state
                        idx = int(np.argmax(rl_q[rl_state]))
                    # Unpack the action: (delta_cx, delta_mu, pop_multiplier)
                    dcx, dmu, pm = rl_actions[idx]
                    # Compute new crossover probability, clamped to allowed range
                    new_cx = min(self.cxpb_max, max(self.cxpb_min, current_cx * (1.0 + dcx)))
                    # Compute new mutation probability, clamped to allowed range
                    new_mu = min(self.mutpb_max, max(self.mutpb_min, current_mu * (1.0 + dmu)))
                    # Compute new population size, clamped to allowed range and rounded to integer
                    new_pop = int(min(self.pop_max, max(self.pop_min, round(current_pop * pm))))
                    return idx, new_cx, new_mu, new_pop

                # --- RL Q-table Update Function (Q-learning) ---
                def rl_update(state, action, reward, next_state):
                    """
                    Update the Q-table using the Q-learning rule:
                    Q(s, a) â† Q(s, a) + Î± * [reward + Î³ * max_a' Q(s', a') - Q(s, a)]
                    - state: current state (int)
                    - action: action index taken (int)
                    - reward: observed reward (float)
                    - next_state: state after action (int)
                    """
                    q_old = rl_q[state][action]
                    q_next = max(rl_q[next_state])
                    rl_q[state][action] = q_old + self.rl_alpha * (reward + self.rl_gamma * q_next - q_old)

                # --- Population Resizing Function (with Neural Seeding Support) ---
                def resize_population(pop, new_size):
                    """
                    Adjust the population to the desired new_size.
                    - If shrinking, select the best individuals to keep.
                    - If growing, generate new individuals using neural seeding (if enabled and available),
                      otherwise use the default seeding method.
                    - All new individuals are evaluated immediately to ensure valid fitness values.
                    """
                    if new_size < len(pop):
                        # Shrink: select the best individuals to keep
                        return tools.selBest(pop, new_size)
                    extra = new_size - len(pop)
                    if extra > 0:
                        # Grow: need to add 'extra' new individuals
                        # --- Use neural seeding if enabled and neural_seeder is sufficiently trained ---
                        if (
                            self.use_neural_seeding
                            and neural_seeder is not None
                            and neural_seeder.size >= max(50, 5 * len(parameter_bounds))
                        ):
                            # Set the beta parameter for neural seeding (controls exploration/exploitation)
                            beta = self.neural_beta_min
                            try:
                                # If adaptive rates are enabled, interpolate beta based on stagnation
                                if self.adaptive_rates:
                                    beta = self.neural_beta_min + (
                                        self.neural_beta_max - self.neural_beta_min
                                    ) * min(1.0, max(0.0, self.stagnation_counter / max(1, self.stagnation_limit)))
                            except Exception:
                                # If anything goes wrong, fall back to minimum beta
                                pass
                            # Find the best fitness value in the current population (for neural seeder guidance)
                            best_y = None
                            try:
                                best_y = min(ind.fitness.values[0] for ind in pop if ind.fitness.valid)
                            except Exception:
                                best_y = None
                            # Set the exploration fraction (epsilon) for neural seeding
                            eps = self.neural_epsilon
                            try:
                                # If adaptive epsilon is enabled, interpolate based on stagnation
                                if self.neural_adapt_epsilon and self.adaptive_rates:
                                    stag_ratio = min(
                                        1.0, max(0.0, self.stagnation_counter / max(1, self.stagnation_limit))
                                    )
                                    eps = (1.0 - stag_ratio) * self.neural_eps_min + stag_ratio * self.neural_eps_max
                                    eps = max(self.neural_eps_min, min(eps, self.neural_eps_max))
                                    self.current_epsilon = eps
                            except Exception:
                                # If adaptation fails, use the default epsilon
                                pass
                            # --- Generate new individuals using the neural seeder ---
                            seeds = neural_seeder.propose(
                                count=extra, beta=beta, best_y=best_y, exploration_fraction=eps
                            )
                            # Convert each seed (parameter vector) into a DEAP Individual and add to population
                            for s in seeds:
                                pop.append(creator.Individual(list(s)))
                            # Evaluate all new individuals immediately to ensure valid fitness
                            need_eval = [ind for ind in pop if not ind.fitness.valid]
                            if need_eval:
                                fits_new = parallel_evaluate(need_eval)
                                for ind, fit in zip(need_eval, fits_new):
                                    ind.fitness.values = fit
                                if self.use_surrogate:
                                    try:
                                        xs_add = [list(ind) for ind in need_eval]
                                        ys_add = [float(f[0]) for f in fits_new]
                                        self._surrogate_X.extend(xs_add)
                                        self._surrogate_y.extend(ys_add)
                                        if self._surrogate_norm_lows is not None and self._surrogate_norm_spans is not None:
                                            Xn_add = (np.array(xs_add, dtype=float) - self._surrogate_norm_lows) / self._surrogate_norm_spans
                                            self._surrogate_Xn.extend([row.tolist() for row in Xn_add])
                                        if len(self._surrogate_X) > self.surrogate_dataset_max:
                                            keep = self.surrogate_dataset_max
                                            self._surrogate_X = self._surrogate_X[-keep:]
                                            self._surrogate_y = self._surrogate_y[-keep:]
                                            if self._surrogate_Xn:
                                                self._surrogate_Xn = self._surrogate_Xn[-keep:]
                                    except Exception:
                                        pass
                        else:
                            # --- Fallback: Use default seeding method if neural seeding is not available ---
                            new_inds = generate_seed_individuals(extra)
                            pop.extend(new_inds)
                    # Return the resized population (either shrunk or grown)
                    return pop
            # =========================
            # Main Evolutionary Loop
            # =========================
            # This loop runs the genetic algorithm for a specified number of generations.
            # Each generation consists of selection, crossover, mutation, and evaluation steps.
            for gen in range(1, self.ga_num_generations + 1):
                # Track current generation index for adaptive components
                try:
                    self._current_generation = int(gen)
                except Exception:
                    pass
                # ----------------------------------------
                # Early Exit or Pause Handling
                # ----------------------------------------
                # Check if the user has paused or aborted the optimization.
                if self._check_pause_abort():
                    self.update.emit("Optimization aborted by user")
                    break

                # ----------------------------------------
                # Metrics Tracking: Generation Timing
                # ----------------------------------------
                # If metrics tracking is enabled, record the start time of this generation for benchmarking.
                if self.track_metrics:
                    gen_start_time = time.time()
                    # Dictionary to store timing breakdown for each evolutionary step in this generation.
                    generation_time_breakdown = {}

                    # Record the start time of the selection step.
                    selection_start = time.time()

                # ----------------------------------------
                # Progress Reporting
                # ----------------------------------------
                # Emit a message to update the user interface with the current generation number.
                self.update.emit(f"-- Generation {gen} / {self.ga_num_generations} --")

                # Calculate and emit the progress percentage for a progress bar.
                progress_percent = int((gen / self.ga_num_generations) * 100)
                self.progress.emit(progress_percent)
                self.last_progress_update = progress_percent

                # ----------------------------------------
                # Watchdog Timer Reset
                # ----------------------------------------
                # The watchdog timer is a safety feature to prevent infinite loops or hangs.
                # It is reset at the start of each generation.
                if self.watchdog_timer.isActive():
                    self.watchdog_timer.stop()
                self.watchdog_timer.start(600000)  # 10 minutes in milliseconds

                # ----------------------------------------
                # Adaptive Rate Control
                # ----------------------------------------
                # The algorithm can adapt its crossover and mutation rates (and population size)
                # using either a reinforcement learning (RL) controller, a machine learning (ML) bandit,
                # or a legacy heuristic. This section determines which method to use and updates the rates accordingly.
                evals_this_gen = 0  # Counter for the number of evaluations performed this generation

                # --- RL Controller: Use a reinforcement learning agent to select rates and population size
                if self.use_rl_controller:
                    old_pop = len(population)  # Store the current population size
                    # Stagnation-driven exploration boost: temporarily raise epsilon when stagnated
                    eps_eff = float(self.rl_epsilon)
                    if self.rl_stag_enabled and self.rl_stag_counter >= max(1, self.rl_stag_limit):
                        # Simple boost: double epsilon, capped at 1.0
                        eps_eff = min(1.0, self.rl_epsilon * 2.0)
                    _saved_eps = self.rl_epsilon
                    try:
                        self.rl_epsilon = eps_eff
                        # The RL controller selects new crossover/mutation rates and possibly a new population size.
                        rl_idx, new_cxpb, new_mutpb, new_pop = rl_select_action(self.current_cxpb, self.current_mutpb, old_pop)
                    finally:
                        self.rl_epsilon = _saved_eps
                    self.current_cxpb = new_cxpb
                    self.current_mutpb = new_mutpb
                    # If the RL controller changed the population size, resize and evaluate new individuals.
                    if new_pop != old_pop:
                        population = resize_population(population, new_pop)
                        # Find individuals that need evaluation (i.e., new or unevaluated)
                        need_eval = [ind for ind in population if not ind.fitness.valid]
                        if need_eval:
                            self.update.emit(f"  RL ctrl: evaluating {len(need_eval)} new individuals after resize...")
                            eval_start = time.time()
                            fits_new = parallel_evaluate(need_eval)
                            for ind, fit in zip(need_eval, fits_new):
                                ind.fitness.values = fit
                            if self.track_metrics:
                                self.metrics['evaluation_times'].append(time.time() - eval_start)
                            evals_this_gen += len(need_eval)
                    # Set the current rates for this generation
                    current_cxpb = min(self.cxpb_max, max(self.cxpb_min, self.current_cxpb))
                    current_mutpb = min(self.mutpb_max, max(self.mutpb_min, self.current_mutpb))
                    # Log the chosen rates and population size
                    self.update.emit("  Rates type: RL-Controller")
                    self.update.emit(f"  - Crossover: {current_cxpb:.4f}")
                    self.update.emit(f"  - Mutation: {current_mutpb:.4f}")
                    self.update.emit(f"  - Population: {len(population)}")

                # --- ML Bandit: Use a machine learning bandit to adapt rates and (optionally) population size
                elif self.use_ml_adaptive:
                    # Store the current rates and population size for reference
                    old_cxpb = self.current_cxpb
                    old_mutpb = self.current_mutpb
                    old_pop_size = len(population)
                    # Throttle ML bandit updates to reduce overhead
                    did_bandit_update = (gen % max(1, getattr(self, 'ml_update_interval', 1)) == 0)
                    idx = None
                    if did_bandit_update:
                        # Stagnation-driven exploration boost: temporarily raise UCB c when stagnated
                        ucb_c_eff = float(self.ml_ucb_c)
                        if self.ml_stag_enabled and self.ml_stag_counter >= max(1, self.ml_stag_limit):
                            # Simple boost: double exploration coefficient
                            ucb_c_eff = float(self.ml_ucb_c) * 2.0
                        _saved_c = self.ml_ucb_c
                        try:
                            self.ml_ucb_c = ucb_c_eff
                            # The ML bandit selects new rates and possibly a new population size
                            idx, new_cx, new_mu, new_pop = ml_select_action(self.current_cxpb, self.current_mutpb, len(population))
                        finally:
                            self.ml_ucb_c = _saved_c
                        self.current_cxpb = new_cx
                        self.current_mutpb = new_mu
                        # Resize population less frequently to avoid heavy re-evaluations
                        do_pop_update = (self.ml_adapt_population and new_pop != len(population) and (gen % max(1, getattr(self, 'ml_pop_update_interval', 1)) == 0))
                        if do_pop_update:
                            population = resize_population(population, new_pop)
                            # Evaluate any new or unevaluated individuals
                            need_eval = [ind for ind in population if not ind.fitness.valid]
                            if need_eval:
                                # Batch evaluate with persistent executor
                                self.update.emit(f"  ML ctrl: evaluating {len(need_eval)} new individuals after resize...")
                                eval_start = time.time()
                                fits_new = parallel_evaluate(need_eval)
                                for ind, fit in zip(need_eval, fits_new):
                                    ind.fitness.values = fit
                                if self.track_metrics:
                                    self.metrics['evaluation_times'].append(time.time() - eval_start)
                                evals_this_gen += len(need_eval)
                    # Set the current rates for this generation
                    current_cxpb = min(self.cxpb_max, max(self.cxpb_min, self.current_cxpb))
                    current_mutpb = min(self.mutpb_max, max(self.mutpb_min, self.current_mutpb))
                    # Log the chosen rates and population size (throttled)
                    if gen == 1 or gen == self.ga_num_generations or (gen % max(1, getattr(self, 'log_component_table_every_n', 5)) == 0):
                        self.update.emit("  Rates type: ML-Bandit")
                        self.update.emit(f"  - Crossover: {current_cxpb:.4f}")
                        self.update.emit(f"  - Mutation: {current_mutpb:.4f}")
                        self.update.emit(f"  - Population: {len(population)}")

                # --- Legacy/Heuristic: Use fixed or legacy adaptive rates
                else:
                    # If adaptive rates are enabled, use the current adaptive values; otherwise, use fixed defaults.
                    current_cxpb = (min(self.cxpb_max, max(self.cxpb_min, self.current_cxpb)) if self.adaptive_rates else min(self.cxpb_max, max(self.cxpb_min, self.ga_cxpb)))
                    current_mutpb = (min(self.mutpb_max, max(self.mutpb_min, self.current_mutpb)) if self.adaptive_rates else min(self.mutpb_max, max(self.mutpb_min, self.ga_mutpb)))
                    self.update.emit(f"  Rates type: {'Adaptive' if self.adaptive_rates else 'Fixed'}")
                    self.update.emit(f"  - Crossover: {current_cxpb:.4f}")
                    self.update.emit(f"  - Mutation: {current_mutpb:.4f}")

                # ----------------------------------------
                # Adaptive Rate Change Reporting
                # ----------------------------------------
                # If adaptive rates are enabled, report any changes in rates from the previous generation.
                if self.adaptive_rates:
                    # Only report changes if this is not the first generation and previous rates are available.
                    if gen > 1 and hasattr(self, 'prev_cxpb') and hasattr(self, 'prev_mutpb'):
                        cxpb_change = current_cxpb - self.prev_cxpb
                        mutpb_change = current_mutpb - self.prev_mutpb
                        # Emit a message if there was any change in crossover or mutation rates.
                        if cxpb_change != 0 or mutpb_change != 0:
                            self.update.emit(
                                f"  - Changes: cx {'+' if cxpb_change > 0 else ''}{cxpb_change:.4f}, "
                                f"mut {'+' if mutpb_change > 0 else ''}{mutpb_change:.4f}"
                            )
                    # Report the current stagnation counter (used for adaptive logic)
                    self.update.emit(f"  - Stagnation counter: {self.stagnation_counter}/{self.stagnation_limit}")

                    # Store the current rates for comparison in the next generation
                    self.prev_cxpb = current_cxpb
                    self.prev_mutpb = current_mutpb

                # ============================================================================
                # EVOLUTION STEPS
                # ============================================================================
                # 1. SELECTION: Choose which solutions get to reproduce
                offspring = toolbox.select(population, len(population))
                offspring = list(map(toolbox.clone, offspring))
                
                if self.track_metrics:
                    selection_time = time.time() - selection_start
                    self.metrics['selection_times'].append(selection_time)
                    generation_time_breakdown['selection'] = selection_time
                    
                    # Track crossover time
                    crossover_start = time.time()
                
                # 2. CROSSOVER: Combine pairs of solutions to create new ones
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if random.random() < current_cxpb:  # Use current crossover probability
                        toolbox.mate(child1, child2)  # Blend the two solutions
                        # Make sure the new solutions are valid
                        for child in [child1, child2]:
                            for i in range(len(child)):
                                if i in fixed_parameters:
                                    child[i] = fixed_parameters[i]
                                else:
                                    min_val, max_val = parameter_bounds[i]
                                    child[i] = max(min_val, min(child[i], max_val))
                        # Clear their fitness scores (they need to be re-evaluated)
                        del child1.fitness.values
                        del child2.fitness.values
                
                if self.track_metrics:
                    crossover_time = time.time() - crossover_start
                    self.metrics['crossover_times'].append(crossover_time)
                    generation_time_breakdown['crossover'] = crossover_time
                    
                    # Track mutation time
                    mutation_start = time.time()
                
                # 3. MUTATION: Randomly tweak some solutions
                for mutant in offspring:
                    if random.random() < current_mutpb:  # Use current mutation probability
                        toolbox.mutate(mutant)
                        del mutant.fitness.values
                
                if self.track_metrics:
                    mutation_time = time.time() - mutation_start
                    self.metrics['mutation_times'].append(mutation_time)
                    generation_time_breakdown['mutation'] = mutation_time
                    
                    # Track evaluation time
                    evaluation_start = time.time()
                
                # 4. EVALUATION: Score the new solutions (with optional surrogate screening)
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                if invalid_ind:
                    if self.use_surrogate and len(self._surrogate_X) >= max(20, self.surrogate_k * 3):
                        # Build candidate pool by cloning invalid_ind to a larger pool for screening
                        target_eval_count = len(invalid_ind)
                        pool_size = int(max(target_eval_count, self.surrogate_pool_factor * target_eval_count))
                        # Generate additional candidates from current population (pairwise ops) to reach pool size
                        pool = list(map(toolbox.clone, invalid_ind))
                        # Simple pool generation: crossover/mutate random pairs
                        while len(pool) < pool_size:
                            p1, p2 = random.sample(population, 2)
                            c1, c2 = toolbox.clone(p1), toolbox.clone(p2)
                            if random.random() < current_cxpb:
                                toolbox.mate(c1, c2)
                            if random.random() < current_mutpb:
                                toolbox.mutate(c1)
                            if random.random() < current_mutpb:
                                toolbox.mutate(c2)
                            for ch in (c1, c2):
                                for i in range(len(ch)):
                                    if i in fixed_parameters:
                                        ch[i] = fixed_parameters[i]
                                    else:
                                        lo, hi = parameter_bounds[i]
                                        ch[i] = max(lo, min(ch[i], hi))
                                if len(pool) < pool_size:
                                    pool.append(ch)
                                else:
                                    break

                        # Vectorized surrogate normalization and KNN prediction
                        # Reuse precomputed normalization if available
                        lows = self._surrogate_norm_lows if self._surrogate_norm_lows is not None else np.array([b[0] for b in parameter_bounds], dtype=float)
                        spans = self._surrogate_norm_spans if self._surrogate_norm_spans is not None else (np.array([b[1] for b in parameter_bounds], dtype=float) - lows)
                        if isinstance(spans, np.ndarray):
                            spans = spans.copy()
                            spans[spans == 0.0] = 1.0
                        if self._surrogate_Xn:
                            try:
                                Xn_np = np.array(self._surrogate_Xn, dtype=float)
                            except Exception:
                                Xn_np = np.empty((0, len(parameter_bounds)), dtype=float)
                        elif self._surrogate_X:
                            try:
                                Xn_np = (np.array(self._surrogate_X, dtype=float) - lows) / spans
                            except Exception:
                                Xn_np = np.empty((0, len(parameter_bounds)), dtype=float)
                        else:
                            Xn_np = np.empty((0, len(parameter_bounds)), dtype=float)
                        y_np = np.array(self._surrogate_y, dtype=float) if self._surrogate_y else np.empty((0,), dtype=float)

                        def _predict_fitness(v):
                            if Xn_np.size == 0:
                                return float('inf')
                            vz = (np.array(v, dtype=float) - lows) / spans
                            dists = np.linalg.norm(Xn_np - vz, axis=1)
                            k = min(self.surrogate_k, len(dists))
                            idxs = np.argsort(dists, kind='mergesort')[:k]
                            return float(np.sum(y_np[idxs]) / max(1, k))

                        # Score pool by surrogate (lower is better)
                        scored = [(_predict_fitness(list(ind)), ind) for ind in pool]
                        scored.sort(key=lambda t: t[0])
                        # Exploit top-q and explore a fraction with highest distance (novel)
                        q = target_eval_count
                        exploit_n = max(1, int((1.0 - self.surrogate_explore_frac) * q))
                        explore_n = max(0, q - exploit_n)
                        chosen = [ind for _, ind in scored[:exploit_n]]

                        if explore_n > 0:
                            # pick explore_n most novel relative to training set (by distance)
                            def _novelty(ind):
                                if Xn_np.size == 0:
                                    return 0.0
                                vz = (np.array(list(ind), dtype=float) - lows) / spans
                                dists = np.linalg.norm(Xn_np - vz, axis=1)
                                return float(np.min(dists))
                            remain = [ind for _, ind in scored[exploit_n:]]
                            remain.sort(key=lambda ind: _novelty(ind), reverse=True)
                            chosen.extend(remain[:explore_n])

                        # Evaluate chosen only
                        self.update.emit(f"  Surrogate: pool={len(pool)} eval={len(chosen)} (exploit={exploit_n}, explore={len(chosen)-exploit_n})")
                        evaluation_start = time.time()
                        fits = parallel_evaluate(chosen)
                        for ind, fit in zip(chosen, fits):
                            ind.fitness.values = fit
                        if self.track_metrics:
                            self.metrics['evaluation_times'].append(time.time() - evaluation_start)
                        evals_this_gen += len(chosen)
                        # Update surrogate dataset with evaluated chosen
                        if self.use_surrogate:
                            try:
                                xs_add = [list(ind) for ind in chosen]
                                ys_add = [float(ind.fitness.values[0]) for ind in chosen]
                                self._surrogate_X.extend(xs_add)
                                self._surrogate_y.extend(ys_add)
                                if self._surrogate_norm_lows is not None and self._surrogate_norm_spans is not None:
                                    Xn_add = (np.array(xs_add, dtype=float) - self._surrogate_norm_lows) / self._surrogate_norm_spans
                                    self._surrogate_Xn.extend([row.tolist() for row in Xn_add])
                                if len(self._surrogate_X) > self.surrogate_dataset_max:
                                    keep = self.surrogate_dataset_max
                                    self._surrogate_X = self._surrogate_X[-keep:]
                                    self._surrogate_y = self._surrogate_y[-keep:]
                                    if self._surrogate_Xn:
                                        self._surrogate_Xn = self._surrogate_Xn[-keep:]
                            except Exception:
                                pass

                        # Replace offspring invalids by chosen (truncate if needed)
                        # Ensure all offspring are valid by filling from chosen first, then best others
                        new_offspring = []
                        # keep already valid
                        new_offspring.extend([ind for ind in offspring if ind.fitness.valid])
                        # add chosen evaluated
                        new_offspring.extend(chosen)
                        # if size mismatch, trim or pad with best evaluated
                        if len(new_offspring) > len(offspring):
                            new_offspring = new_offspring[:len(offspring)]
                        elif len(new_offspring) < len(offspring):
                            # pad with best among chosen by fitness
                            chosen_sorted = sorted(chosen, key=lambda ind: ind.fitness.values[0])
                            while len(new_offspring) < len(offspring) and chosen_sorted:
                                new_offspring.append(chosen_sorted.pop(0))
                        offspring = new_offspring
                    else:
                        # Fallback: evaluate all invalids
                        self.update.emit(f"  Evaluating {len(invalid_ind)} individuals...")
                        fitnesses = parallel_evaluate(invalid_ind)
                        for ind, fit in zip(invalid_ind, fitnesses):
                            ind.fitness.values = fit
                        # Count evaluations once per individual
                        evals_this_gen += len(invalid_ind)
                        # Update surrogate dataset
                        if self.use_surrogate:
                            try:
                                xs_add = [list(ind) for ind in invalid_ind]
                                ys_add = [float(f[0]) for f in fitnesses]
                                self._surrogate_X.extend(xs_add)
                                self._surrogate_y.extend(ys_add)
                                if self._surrogate_norm_lows is not None and self._surrogate_norm_spans is not None:
                                    Xn_add = (np.array(xs_add, dtype=float) - self._surrogate_norm_lows) / self._surrogate_norm_spans
                                    self._surrogate_Xn.extend([row.tolist() for row in Xn_add])
                                if len(self._surrogate_X) > self.surrogate_dataset_max:
                                    keep = self.surrogate_dataset_max
                                    self._surrogate_X = self._surrogate_X[-keep:]
                                    self._surrogate_y = self._surrogate_y[-keep:]
                                    if self._surrogate_Xn:
                                        self._surrogate_Xn = self._surrogate_Xn[-keep:]
                            except Exception:
                                pass

                # After evaluation, update NeuralSeeder with fresh data and optionally reseed on stagnation
                if (self.use_neural_seeding or self.seeding_method == "memory") and 'invalid_ind' in locals():
                    try:
                        if neural_seeder is not None:
                            # add data
                            X_batch = [list(ind) for ind in population]
                            y_batch = [float(ind.fitness.values[0]) for ind in population]
                            neural_seeder.add_data(X_batch, y_batch)
                            # adaptive beta via stagnation/diversity
                            beta = self.neural_beta_min
                            if self.adaptive_rates:
                                beta = self.neural_beta_min + (self.neural_beta_max - self.neural_beta_min) * min(1.0, max(0.0, self.stagnation_counter / max(1, self.stagnation_limit)))
                            # optionally adapt epsilon here too (mirrors resize logic)
                            eps = self.neural_epsilon
                            if self.neural_adapt_epsilon and self.adaptive_rates:
                                stag_ratio = min(1.0, max(0.0, self.stagnation_counter / max(1, self.stagnation_limit)))
                                eps = (1.0 - stag_ratio) * self.neural_eps_min + stag_ratio * self.neural_eps_max
                                eps = max(self.neural_eps_min, min(eps, self.neural_eps_max))
                                self.current_epsilon = eps
                            # train
                            train_time, epochs_done = neural_seeder.train()
                            # log
                            if self.track_metrics:
                                self.metrics['neural_history'].append({
                                    'generation': gen,
                                    'train_time_ms': train_time,
                                    'epochs': epochs_done,
                                    'beta': beta,
                                    'pool_mult': self.neural_pool_mult,
                                    'epsilon': self.current_epsilon if self.neural_adapt_epsilon else self.neural_epsilon,
                                    'acq': self.neural_acq_type
                                })
                        # Update memory seeder as well
                        if self.seeding_method == "memory" and 'memory_seeder' in locals() and memory_seeder is not None:
                            X_batch = [list(ind) for ind in population]
                            y_batch = [float(ind.fitness.values[0]) for ind in population]
                            memory_seeder.add_data(X_batch, y_batch)
                    except Exception:
                        pass
                
                if self.track_metrics:
                    evaluation_time = time.time() - evaluation_start
                    self.metrics['evaluation_times'].append(evaluation_time)
                    generation_time_breakdown['evaluation'] = evaluation_time
                
                # 5. REPLACEMENT: Replace old population with new one
                population[:] = offspring

                # Compute success rate: fraction of offspring that are better than their closest parent
                success_rate = None
                try:
                    parent_fits = [ind.fitness.values[0] for ind in population]
                    off_fits = [ind.fitness.values[0] for ind in offspring]
                    improved = 0
                    total_considered = 0
                    for o in offspring:
                        # compare to random parent sample as proxy (parents not stored one-to-one)
                        p = random.choice(population)
                        if hasattr(o.fitness, 'values') and hasattr(p.fitness, 'values'):
                            total_considered += 1
                            if o.fitness.values[0] < p.fitness.values[0]:
                                improved += 1
                    if total_considered > 0:
                        success_rate = improved / total_considered
                except Exception:
                    success_rate = None

                # ============================================================================
                # STATISTICS AND MONITORING
                # ============================================================================
                # Calculate statistics for this generation
                fits = [ind.fitness.values[0] for ind in population]
                length = len(population)
                mean = sum(fits) / length
                sum2 = sum(f ** 2 for f in fits)
                std = abs(sum2 / length - mean ** 2) ** 0.5
                min_fit = min(fits)
                max_fit = max(fits)

                # Gene diversity: average normalized std across genes (vectorized)
                try:
                    if length > 0 and len(population[0]) > 0:
                        pop_mat = np.asarray(population, dtype=float)
                        # Avoid div-by-zero on fixed genes
                        spans_nonzero = spans_np.copy()
                        spans_nonzero[spans_nonzero == 0.0] = 1.0
                        col_std = np.std(pop_mat, axis=0)
                        norm_std = np.clip(col_std / spans_nonzero, 0.0, 1.0)
                        if norm_std.size > 0:
                            gene_diversity = float(np.mean(norm_std))
                        else:
                            gene_diversity = 0.0
                    else:
                        gene_diversity = 0.0
                except Exception:
                    gene_diversity = 0.0

                # Create a table for fitness components
                best_idx = fits.index(min_fit)
                best_individual = population[best_idx]

                # Check if the best individual has the fitness components
                has_components = hasattr(best_individual, 'primary_objective')
                
                # If adaptive rates are enabled, check if we need to adjust rates (legacy improved)
                if self.adaptive_rates:
                    # Track if we found an improvement
                    improved = False

                    # Track best solution found
                    if min_fit < best_fitness_overall:
                        improved = True
                        best_fitness_overall = min_fit
                        best_ind_overall = tools.selBest(population, 1)[0]
                        self.update.emit(f"  New best solution found! Fitness: {best_fitness_overall:.6f}")
                        # Nudge stagnation down but keep history
                        self.stagnation_counter = max(0, self.stagnation_counter - 1)
                    else:
                        self.stagnation_counter += 1

                    # Smooth success rate and gene diversity
                    if success_rate is not None:
                        if self._ema_success_rate is None:
                            self._ema_success_rate = success_rate
                        else:
                            a = self.adapt_smoothing
                            self._ema_success_rate = (1 - a) * self._ema_success_rate + a * success_rate
                    if 'gene_diversity' in locals():
                        if self._ema_gene_diversity is None:
                            self._ema_gene_diversity = gene_diversity
                        else:
                            a = self.adapt_smoothing
                            self._ema_gene_diversity = (1 - a) * self._ema_gene_diversity + a * gene_diversity

                    # Decide if we adapt this generation: upon stagnation or periodic heartbeat
                    heartbeat = (gen % 2 == 0)
                    if self.stagnation_counter >= self.stagnation_limit or heartbeat:
                        # Begin with current rates
                        old_cx = current_cxpb
                        old_mu = current_mutpb

                        # 1) Success rule for mutation rate (1/5th rule like):
                        if self._ema_success_rate is not None:
                            if self._ema_success_rate < self.target_success_rate * 0.9:
                                # Too few improvements: increase mutation
                                self.current_mutpb = min(self.mutpb_max, self.current_mutpb * (1.0 + 0.25))
                            elif self._ema_success_rate > self.target_success_rate * 1.1:
                                # Too many easy improvements: reduce mutation
                                self.current_mutpb = max(self.mutpb_min, self.current_mutpb * (1.0 - 0.20))

                        # 2) Diversity-guided crossover vs mutation tradeoff
                        div = self._ema_gene_diversity if (self._ema_gene_diversity is not None) else (std / (abs(mean) + 1e-12))
                        # Normalize diversity to [0,1]
                        if mean > 0:
                            norm_div = min(1.0, max(0.0, div))
                        else:
                            norm_div = 0.5
                        if norm_div < 0.08:
                            # Very low diversity: push exploration strongly
                            self.current_mutpb = min(self.mutpb_max, self.current_mutpb * 1.35)
                            self.current_cxpb = max(self.cxpb_min, self.current_cxpb * 0.90)
                            # Also widen mutation step size
                            self.mutation_scale = min(self.mutation_scale_max, self.mutation_scale * 1.25)
                            adaptation_type = "Exploration boost: â†‘mutation prob/scale, â†“crossover"
                        elif norm_div > 0.35:
                            # High diversity: push exploitation
                            self.current_cxpb = min(self.cxpb_max, self.current_cxpb * 1.20)
                            self.current_mutpb = max(self.mutpb_min, self.current_mutpb * 0.90)
                            # Narrow mutation step size
                            self.mutation_scale = max(self.mutation_scale_min, self.mutation_scale * 0.90)
                            adaptation_type = "Exploitation tilt: â†‘crossover, â†“mutation prob/scale"
                        else:
                            # Moderate diversity: gentle balancing with small moves
                            if gen % 2 == 0:
                                self.current_cxpb = min(self.cxpb_max, self.current_cxpb * 1.08)
                                self.current_mutpb = max(self.mutpb_min, self.current_mutpb * 0.97)
                                adaptation_type = "Balanced: slight â†‘crossover, slight â†“mutation"
                            else:
                                self.current_mutpb = min(self.mutpb_max, self.current_mutpb * 1.08)
                                self.current_cxpb = max(self.cxpb_min, self.current_cxpb * 0.97)
                                adaptation_type = "Balanced: slight â†‘mutation, slight â†“crossover"

                        # Clamp after changes
                        self.current_cxpb = min(self.cxpb_max, max(self.cxpb_min, self.current_cxpb))
                        self.current_mutpb = min(self.mutpb_max, max(self.mutpb_min, self.current_mutpb))
                        self.mutation_scale = min(self.mutation_scale_max, max(self.mutation_scale_min, self.mutation_scale))

                        # Log the adaptation
                        self.update.emit(
                            f"  Adaptive (legacy+): sr={self._ema_success_rate if self._ema_success_rate is not None else 'n/a':} div={self._ema_gene_diversity if self._ema_gene_diversity is not None else 'n/a':}"
                        )
                        self.update.emit(
                            f"  New rates: crossover={self.current_cxpb:.3f}, mutation={self.current_mutpb:.3f}, mut_scale={self.mutation_scale:.3f} - {adaptation_type}"
                        )
                        cxpb_change = self.current_cxpb - old_cx
                        mutpb_change = self.current_mutpb - old_mu
                        self.update.emit(
                            f"  â†³ Î”cx={cxpb_change:+.4f}, Î”mut={mutpb_change:+.4f}"
                        )

                        # Reset stagnation gently only if changed
                        if self.stagnation_counter >= self.stagnation_limit:
                            self.stagnation_counter = max(0, self.stagnation_counter - 2)

                        # Record adaptation in history and metrics
                        adaptation_record = {
                            'generation': gen,
                            'old_cxpb': old_cx,
                            'old_mutpb': old_mu,
                            'new_cxpb': self.current_cxpb,
                            'new_mutpb': self.current_mutpb,
                            'mutation_scale': self.mutation_scale,
                            'success_rate': self._ema_success_rate,
                            'gene_diversity': self._ema_gene_diversity,
                            'adaptation_type': adaptation_type
                        }
                        self.rate_adaptation_history.append(adaptation_record)
                        if self.track_metrics:
                            self.metrics['adaptive_rates_history'].append(adaptation_record)
                            if success_rate is not None:
                                self.metrics['success_rate_history'].append(success_rate)
                            self.metrics['gene_diversity_history'].append(self._ema_gene_diversity if self._ema_gene_diversity is not None else gene_diversity)
                            self.metrics['mutation_scale_history'].append(self.mutation_scale)
                else:
                    # If adaptive rates are not enabled, just track best solution
                    if min_fit < best_fitness_overall:
                        best_fitness_overall = min_fit
                        best_ind_overall = tools.selBest(population, 1)[0]
                        self.update.emit(f"  New best solution found! Fitness: {best_fitness_overall:.6f}")

                if has_components and (gen == 1 or gen == self.ga_num_generations or getattr(self, 'log_component_table_every_n', 1) <= 1 or (gen % getattr(self, 'log_component_table_every_n', 1) == 0)):
                    table_header = "  â”Œâ”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”¬â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”"
                    table_format = "  â”‚ {0:<21} â”‚ {1:>13} â”‚ {2:>13} â”‚ {3:>13} â”‚ {4:>13} â”‚"
                    table_footer = "  â””â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”´â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”˜"
                    
                    self.update.emit(table_header)
                    self.update.emit(table_format.format("Fitness Components", "Min", "Max", "Average", "Best"))
                    self.update.emit(table_format.format("â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€", "â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€", "â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€", "â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€", "â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€"))
                    
                    # Calculate component statistics
                    primary_objectives = [ind.primary_objective if hasattr(ind, 'primary_objective') else 0 for ind in population]
                    sparsity_penalties = [ind.sparsity_penalty if hasattr(ind, 'sparsity_penalty') else 0 for ind in population]
                    percentage_errors = [ind.percentage_error if hasattr(ind, 'percentage_error') else 0 for ind in population]
                    activation_penalties = [getattr(ind, 'activation_penalty', 0.0) for ind in population]
                    cost_terms = [getattr(ind, 'cost_term', 0.0) for ind in population]
                    
                    min_primary = min(primary_objectives) if primary_objectives else 0
                    max_primary = max(primary_objectives) if primary_objectives else 0
                    avg_primary = sum(primary_objectives) / len(primary_objectives) if primary_objectives else 0
                    best_primary = best_individual.primary_objective if hasattr(best_individual, 'primary_objective') else 0
                    
                    min_sparsity = min(sparsity_penalties) if sparsity_penalties else 0
                    max_sparsity = max(sparsity_penalties) if sparsity_penalties else 0
                    avg_sparsity = sum(sparsity_penalties) / len(sparsity_penalties) if sparsity_penalties else 0
                    best_sparsity = best_individual.sparsity_penalty if hasattr(best_individual, 'sparsity_penalty') else 0
                    
                    min_percentage = min(percentage_errors) if percentage_errors else 0
                    max_percentage = max(percentage_errors) if percentage_errors else 0
                    avg_percentage = sum(percentage_errors) / len(percentage_errors) if percentage_errors else 0
                    best_percentage = best_individual.percentage_error if hasattr(best_individual, 'percentage_error') else 0
                    
                    # Display component values in table
                    self.update.emit(table_format.format("Primary Objective", f"{min_primary:.6f}", f"{max_primary:.6f}", f"{avg_primary:.6f}", f"{best_primary:.6f}"))
                    self.update.emit(table_format.format("Sparsity Penalty", f"{min_sparsity:.6f}", f"{max_sparsity:.6f}", f"{avg_sparsity:.6f}", f"{best_sparsity:.6f}"))
                    self.update.emit(table_format.format("Percentage Error", f"{min_percentage:.6f}", f"{max_percentage:.6f}", f"{avg_percentage:.6f}", f"{best_percentage:.6f}"))
                    if activation_penalties:
                        self.update.emit(table_format.format("Activation Penalty", f"{min(activation_penalties):.6f}", f"{max(activation_penalties):.6f}", f"{(sum(activation_penalties)/len(activation_penalties)):.6f}", f"{getattr(best_individual,'activation_penalty',0.0):.6f}"))
                    if cost_terms:
                        self.update.emit(table_format.format("Cost Term", f"{min(cost_terms):.6f}", f"{max(cost_terms):.6f}", f"{(sum(cost_terms)/len(cost_terms)):.6f}", f"{getattr(best_individual,'cost_term',0.0):.6f}"))
                    self.update.emit(table_format.format("Total Fitness", f"{min_fit:.6f}", f"{max_fit:.6f}", f"{mean:.6f}", f"{min_fit:.6f}"))
                    self.update.emit(table_footer)
                    
                    # If adaptive rates are enabled, display current rates
                    if self.adaptive_rates:
                        # Instead of showing rates again, show an indicator of whether rates will be adapted
                        if self.stagnation_counter >= self.stagnation_limit - 1:
                            self.update.emit(f"  âš ï¸ Rates will adapt next generation due to stagnation ({self.stagnation_counter}/{self.stagnation_limit})")
                        else:
                            self.update.emit(f"  Stagnation counter: {self.stagnation_counter}/{self.stagnation_limit}")
                else:
                    # If components are not available, use the traditional display (throttled)
                    if (gen == 1 or gen == self.ga_num_generations or getattr(self, 'log_component_table_every_n', 1) <= 1 or (gen % getattr(self, 'log_component_table_every_n', 1) == 0)):
                        self.update.emit(f"  Min fitness: {min_fit:.6f}")
                        self.update.emit(f"  Max fitness: {max_fit:.6f}")
                        self.update.emit(f"  Avg fitness: {mean:.6f}")
                        self.update.emit(f"  Std fitness: {std:.6f}")
                        if self.adaptive_rates:
                            if self.stagnation_counter >= self.stagnation_limit - 1:
                                self.update.emit(f"  âš ï¸ Rates will adapt next generation due to stagnation ({self.stagnation_counter}/{self.stagnation_limit})")
                            else:
                                self.update.emit(f"  Stagnation counter: {self.stagnation_counter}/{self.stagnation_limit}")

                # Track metrics for this generation if enabled
                if self.track_metrics:
                    # Record time for this generation
                    gen_time = time.time() - gen_start_time
                    self.metrics['generation_times'].append(gen_time)
                    
                    # Record the time breakdown for this generation
                    generation_time_breakdown['total'] = gen_time
                    self.metrics['time_per_generation_breakdown'].append(generation_time_breakdown)
                    
                    # Record fitness statistics
                    self.metrics['fitness_history'].append(fits)
                    self.metrics['mean_fitness_history'].append(mean)
                    self.metrics['std_fitness_history'].append(std)
                    # Record population size and rates for this generation
                    self.metrics['pop_size_history'].append(len(population))
                    self.metrics['rates_history'].append({'generation': gen, 'cxpb': current_cxpb, 'mutpb': current_mutpb})
                    
                    # Track best individual in this generation
                    best_gen_idx = fits.index(min_fit)
                    best_gen_ind = population[best_gen_idx]
                    self.metrics['best_fitness_per_gen'].append(min_fit)
                    self.metrics['best_individual_per_gen'].append(list(best_gen_ind))
                    
                    # Calculate instantaneous convergence rate
                    if len(self.metrics['best_fitness_per_gen']) > 1:
                        prev_best = self.metrics['best_fitness_per_gen'][-2]
                        if prev_best > min_fit:  # If we improved
                            improvement = prev_best - min_fit
                            self.metrics['convergence_rate'].append(improvement)
                        else:
                            self.metrics['convergence_rate'].append(0.0)

                    # Controller-specific reward logging
                    if self.use_ml_adaptive or self.use_rl_controller:
                        last_best = self.metrics['best_fitness_per_gen'][-2] if len(self.metrics['best_fitness_per_gen']) > 1 else None
                        imp = (last_best - min_fit) if (last_best is not None and last_best > min_fit) else 0.0
                        cv = std / (abs(mean) + 1e-12)
                        effort = max(1.0, evals_this_gen)
                        if self.use_rl_controller:
                            # RL reward per LaTeX spec:
                            # R_t = w1 * max(0, f*_{t-1}-f*_t)/(1+|f*_t|) + w2 * (d_t - d_{t-1})/(1+d_{t-1})
                            #       + w3 * (1/Î”t) - w4 * |cv_t - cv*|
                            f_t = min_fit
                            f_prev = last_best if last_best is not None else f_t
                            term1 = self.rl_w1 * (max(0.0, (f_prev - f_t)) / (1.0 + abs(f_t)))
                            # diversity terms
                            d_t = None
                            d_prev = None
                            try:
                                # reuse computed gene_diversity if available in locals(); else approximate via cv
                                d_t = gene_diversity if 'gene_diversity' in locals() else min(1.0, max(0.0, std / (abs(mean) + 1e-12)))
                            except Exception:
                                d_t = 0.0
                            try:
                                d_prev = self._ema_gene_diversity if (self._ema_gene_diversity is not None) else d_t
                            except Exception:
                                d_prev = d_t
                            term2 = self.rl_w2 * ((d_t - d_prev) / (1.0 + max(0.0, d_prev)))
                            term3 = self.rl_w3 * (1.0 / max(gen_time, 1e-6))
                            term4 = self.rl_w4 * abs(cv - (self.rl_cv_target if hasattr(self, 'rl_cv_target') else self.ml_diversity_target))
                            reward = term1 + term2 + term3 - term4
                        else:
                            # ML bandit reward (unchanged)
                            reward = (imp / max(gen_time, 1e-6)) / effort - self.ml_diversity_weight * abs(cv - self.ml_diversity_target)

                        if self.use_ml_adaptive:
                            try:
                                ml_update(idx, reward)
                            except Exception:
                                pass
                            self.metrics['ml_controller_history'].append({
                                'generation': gen,
                                'cxpb': current_cxpb,
                                'mutpb': current_mutpb,
                                'pop': len(population),
                                'best_fitness': min_fit,
                                'mean_fitness': mean,
                                'std_fitness': std,
                                'reward': reward,
                                'blending_weights': {
                                    'historical': self.ml_historical_weight,
                                    'current': self.ml_current_weight
                                }
                            })
                        elif self.use_rl_controller:
                            next_state = 1 if imp > 0 else 0
                            try:
                                rl_update(rl_state, rl_idx, reward, next_state)
                                rl_state = next_state
                                self.rl_epsilon *= self.rl_epsilon_decay
                            except Exception:
                                pass
                            self.metrics['rl_controller_history'].append({
                                'generation': gen,
                                'cxpb': current_cxpb,
                                'mutpb': current_mutpb,
                                'pop': len(population),
                                'best_fitness': min_fit,
                                'mean_fitness': mean,
                                'std_fitness': std,
                                'reward': reward,
                                'epsilon': self.rl_epsilon
                            })

                    # Update controller stagnation counters at end of generation
                    try:
                        if self.use_ml_adaptive and self.ml_stag_enabled:
                            last_best = self.metrics['best_fitness_per_gen'][-2] if len(self.metrics['best_fitness_per_gen']) > 1 else None
                            if last_best is not None and min_fit < last_best:
                                self.ml_stag_counter = 0
                            else:
                                self.ml_stag_counter += 1
                            if self.track_metrics:
                                self.metrics['ml_stagnation_counter_history'].append(self.ml_stag_counter)
                    except Exception:
                        pass
                    try:
                        if self.use_rl_controller and self.rl_stag_enabled:
                            last_best = self.metrics['best_fitness_per_gen'][-2] if len(self.metrics['best_fitness_per_gen']) > 1 else None
                            if last_best is not None and min_fit < last_best:
                                self.rl_stag_counter = 0
                            else:
                                self.rl_stag_counter += 1
                            if self.track_metrics:
                                self.metrics['rl_stagnation_counter_history'].append(self.rl_stag_counter)
                    except Exception:
                        pass

                    # Record surrogate info
                    if self.use_surrogate:
                        self.metrics['surrogate_info'].append({
                            'generation': gen,
                            'pool_factor': self.surrogate_pool_factor,
                            'pool_size': int(self.surrogate_pool_factor * len(invalid_ind)) if 'invalid_ind' in locals() else 0,
                            'evaluated_count': evals_this_gen
                        })

                    # Prune histories to sliding window to avoid unbounded growth
                    try:
                        W = int(getattr(self, 'metrics_window', 0))
                        if W and W > 0:
                            def _trim(lst):
                                if isinstance(lst, list) and len(lst) > W:
                                    del lst[:-W]
                            _trim(self.metrics['generation_times'])
                            _trim(self.metrics['time_per_generation_breakdown'])
                            _trim(self.metrics['fitness_history'])
                            _trim(self.metrics['mean_fitness_history'])
                            _trim(self.metrics['std_fitness_history'])
                            _trim(self.metrics['pop_size_history'])
                            _trim(self.metrics['rates_history'])
                            _trim(self.metrics['best_fitness_per_gen'])
                            _trim(self.metrics['best_individual_per_gen'])
                            _trim(self.metrics['convergence_rate'])
                            _trim(self.metrics['ml_controller_history'])
                            _trim(self.metrics['rl_controller_history'])
                            _trim(self.metrics['surrogate_info'])
                            _trim(self.metrics['adaptive_rates_history'])
                            _trim(self.metrics['success_rate_history'])
                            _trim(self.metrics['gene_diversity_history'])
                            _trim(self.metrics['mutation_scale_history'])
                    except Exception:
                        pass

                # Check if we've found a good enough solution
                if min_fit <= self.ga_tol:
                    self.update.emit(f"\n[INFO] Solution found within tolerance at generation {gen}")
                    break

            # ============================================================================
            # FINAL RESULTS
            # ============================================================================
            # Show we're done
            self.progress.emit(100)
            
            # Stop metrics tracking
            if self.track_metrics:
                self._stop_metrics_tracking()
            # Capture process CPU time end and compute used
            try:
                proc = psutil.Process(os.getpid())
                cpu_times = proc.cpu_times()
                self.metrics['proc_cpu_time_end'] = float(getattr(cpu_times, 'user', 0.0) + getattr(cpu_times, 'system', 0.0))
                if self.metrics.get('proc_cpu_time_start') is not None:
                    self.metrics['proc_cpu_time_used'] = float(self.metrics['proc_cpu_time_end'] - self.metrics['proc_cpu_time_start'])
            except Exception:
                self.metrics['proc_cpu_time_end'] = None
                self.metrics['proc_cpu_time_used'] = None
            
            # Process the best solution we found
            if not self.abort:
                # Get the best solution
                best_ind = best_ind_overall if best_ind_overall is not None else tools.selBest(population, 1)[0]
                best_fitness = best_ind.fitness.values[0]

                # Convert to the format needed for final evaluation
                dva_parameters_tuple = tuple(best_ind)
                try:
                    # Do one final evaluation with the best solution
                    self.update.emit("Computing final results...")
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
                    
                    # Make sure we have a singular response value
                    if 'singular_response' not in final_results and 'composite_measures' in final_results:
                        composite_measures = final_results['composite_measures']
                        final_results['singular_response'] = sum(composite_measures.values())
                        self.update.emit("Calculated missing singular response from composite measures")
                    
                    # Add benchmark metrics to final results if tracking was enabled
                    if self.track_metrics:
                        final_results['benchmark_metrics'] = self.metrics
                        # Convenience flat fields for All Runs ranking
                        try:
                            final_results['elapsed_time'] = float(self.metrics.get('total_duration')) if self.metrics.get('total_duration') is not None else None
                        except Exception:
                            pass
                        try:
                            # Convergence generation = index of first occurrence of global min best_fitness_per_gen
                            if self.metrics.get('best_fitness_per_gen'):
                                series = self.metrics['best_fitness_per_gen']
                                best = min(series)
                                # 1-based generation index
                                final_results['convergence_generation'] = int(series.index(best) + 1)
                        except Exception:
                            pass
                        try:
                            if self.metrics.get('proc_cpu_time_used') is not None:
                                final_results['cpu_time_used'] = float(self.metrics['proc_cpu_time_used'])
                        except Exception:
                            pass
                    
                    # Attach run settings for front-end plots/components breakdown
                    try:
                        final_results['dva_activation_threshold'] = float(self.dva_activation_threshold)
                    except Exception:
                        pass
                    try:
                        final_results['dva_activation_penalty'] = float(self.dva_activation_penalty)
                    except Exception:
                        pass
                    try:
                        final_results['percentage_error_scale'] = float(self.percentage_error_scale)
                    except Exception:
                        pass
                    try:
                        # Provide cost mapping and denominator for normalization
                        final_results['dva_costs'] = dict(self.dva_costs)
                        final_results['dva_costs_total'] = sum(self.dva_costs.values()) if self.dva_costs else 0.0
                        final_results['use_enhanced_cost'] = bool(self.use_enhanced_cost)
                        # Add fitness component information
                        final_results['cost_scale_factor'] = float(self.cost_scale_factor)
                        final_results['primary_objective'] = float(getattr(best_ind, 'primary_objective', 0.0))
                        final_results['sparsity_penalty'] = float(getattr(best_ind, 'sparsity_penalty', 0.0))
                        final_results['percentage_error'] = float(getattr(best_ind, 'percentage_error', 0.0))
                        final_results['activation_penalty'] = float(getattr(best_ind, 'activation_penalty', 0.0))
                        # Calculate raw cost from scaled cost
                        scaled_cost = float(getattr(best_ind, 'cost_term', 0.0))
                        final_results['cost_term_raw'] = scaled_cost * self.cost_scale_factor
                        final_results['cost_term'] = scaled_cost  # Scaled cost used in fitness
                        final_results['fitness'] = float(best_fitness)
                        try:
                            bw_p, bw_a, bw_s = self.benefit_weights
                            final_results['benefit_w_primary'] = float(bw_p)
                            final_results['benefit_w_accuracy'] = float(bw_a)
                            final_results['benefit_w_sparsity'] = float(bw_s)
                        except Exception:
                            pass
                        try:
                            final_results['category_w_material'] = float(self.category_weights.get('material', 0.25))
                            final_results['category_w_manufacturing'] = float(self.category_weights.get('manufacturing', 0.25))
                            final_results['category_w_maintenance'] = float(self.category_weights.get('maintenance', 0.25))
                            final_results['category_w_operational'] = float(self.category_weights.get('operational', 0.25))
                        except Exception:
                            pass
                        try:
                            final_results['benefit_weight_start'] = float(self.benefit_weight_start)
                            final_results['benefit_weight_end'] = float(self.benefit_weight_end)
                        except Exception:
                            pass
                        try:
                            final_results['dva_category_map'] = dict(self.dva_category_map)
                        except Exception:
                            pass
                        final_results['_dva_cost_denominator'] = float(self._dva_cost_denominator)
                    except Exception:
                        pass

                    # Make sure to clean up after a successful run
                    self.cleanup()
                    self.finished.emit(final_results, best_ind, parameter_names, best_fitness)
                except Exception as e:
                    error_msg = f"Error during final FRF evaluation: {str(e)}"
                    self.update.emit(error_msg)
                    # Still try to return the best individual found
                    final_results = {"Error": error_msg, "Warning": "Using best individual without final evaluation"}
                    
                    # Try one more time to calculate singular response using a simplified method
                    try:
                        # Create a simplified version of the target_values and weights
                        # to calculate a basic singular response value
                        final_results["singular_response"] = best_fitness  # Use the fitness as a fallback
                        self.update.emit("Added estimated singular response based on fitness value")
                    except Exception as calc_err:
                        self.update.emit(f"Could not estimate singular response: {str(calc_err)}")
                    
                    # Add benchmark metrics to final results if tracking was enabled
                    if self.track_metrics:
                        final_results['benchmark_metrics'] = self.metrics
                        try:
                            final_results['elapsed_time'] = float(self.metrics.get('total_duration')) if self.metrics.get('total_duration') is not None else None
                        except Exception:
                            pass
                        try:
                            if self.metrics.get('best_fitness_per_gen'):
                                series = self.metrics['best_fitness_per_gen']
                                best = min(series)
                                final_results['convergence_generation'] = int(series.index(best) + 1)
                        except Exception:
                            pass
                        try:
                            if self.metrics.get('proc_cpu_time_used') is not None:
                                final_results['cpu_time_used'] = float(self.metrics['proc_cpu_time_used'])
                        except Exception:
                            pass
                    
                    # Make sure to clean up after an error
                    self.cleanup()
                    self.finished.emit(final_results, best_ind, parameter_names, best_fitness)
            else:
                # If aborted, still try to return the best solution found so far
                if best_ind_overall is not None:
                    self.update.emit("Optimization was aborted, returning best solution found so far.")
                    final_results = {"Warning": "Optimization was aborted before completion"}
                    
                    # Include a singular response estimate based on the best fitness found
                    if best_fitness_overall < 1e6:  # Only if we found a reasonable solution
                        final_results["singular_response"] = best_fitness_overall  # Approximate with fitness
                        self.update.emit("Added estimated singular response based on best fitness value")
                    
                    # Add benchmark metrics to final results if tracking was enabled
                    if self.track_metrics:
                        final_results['benchmark_metrics'] = self.metrics
                        try:
                            final_results['elapsed_time'] = float(self.metrics.get('total_duration')) if self.metrics.get('total_duration') is not None else None
                        except Exception:
                            pass
                        try:
                            if self.metrics.get('best_fitness_per_gen') is not None and len(self.metrics['best_fitness_per_gen']) > 0:
                                series = self.metrics['best_fitness_per_gen']
                                best = min(series)
                                final_results['convergence_generation'] = int(series.index(best) + 1)
                        except Exception:
                            pass
                        try:
                            if self.metrics.get('proc_cpu_time_used') is not None:
                                final_results['cpu_time_used'] = float(self.metrics['proc_cpu_time_used'])
                        except Exception:
                            pass
                    
                    self.cleanup()
                    self.finished.emit(final_results, best_ind_overall, parameter_names, best_fitness_overall)
                else:
                    error_msg = "Optimization was aborted before finding any valid solutions"
                    self.update.emit(error_msg)
                    self.cleanup()
                    self.error.emit(error_msg)

        except Exception as e:
            # Stop metrics tracking if it was enabled
            if self.track_metrics:
                self._stop_metrics_tracking()
                
            error_msg = f"GA optimization error: {str(e)}\n{traceback.format_exc()}"
            self.update.emit(error_msg)
            # Clean up before emitting error signal
            self.cleanup()
            self.error.emit(error_msg)

    def _get_system_info(self):
        """Collect system information for benchmarking"""
        try:
            system_info = {
                'platform': platform.system(),
                'platform_release': platform.release(),
                'platform_version': platform.version(),
                'architecture': platform.machine(),
                'processor': platform.processor(),
                'physical_cores': psutil.cpu_count(logical=False),
                'total_cores': psutil.cpu_count(logical=True),
                'total_memory': round(psutil.virtual_memory().total / (1024.0 ** 3), 2),  # GB
                'python_version': platform.python_version(),
            }
            
            # Get CPU frequency if available
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
            self.update.emit(f"Warning: Could not collect complete system info: {str(e)}")
            return {'error': str(e)}
    
    def _update_resource_metrics(self):
        """Update CPU and memory usage metrics with more detailed information"""
        if not self.track_metrics:
            return
            
        try:
            # tick counter to throttle heavy metrics
            try:
                self._metrics_tick += 1
            except Exception:
                self._metrics_tick = 1
            heavy = (self._metrics_tick % getattr(self, '_metrics_heavy_every', 5) == 0)
            # Log that metrics collection is happening (throttled)
            if getattr(self, 'metrics_verbose', False):
                self.update.emit("Collecting resource metrics...")
            
            # Get basic CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=None)
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_usage_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
            
            # Add basic metrics
            self.metrics['cpu_usage'].append(cpu_percent)
            self.metrics['memory_usage'].append(memory_usage_mb)
            
            # Log basic metrics for debugging (throttled)
            if getattr(self, 'metrics_verbose', False):
                self.update.emit(f"CPU: {cpu_percent}%, Memory: {memory_usage_mb:.2f} MB")
            
            # Get per-core CPU usage (heavy)
            if heavy:
                per_core_cpu = psutil.cpu_percent(interval=None, percpu=True)
                self.metrics['cpu_per_core'].append(per_core_cpu)
            
            # Get detailed memory information
            memory_details = {
                'rss': memory_info.rss / (1024 * 1024),  # Resident Set Size in MB
                'vms': memory_info.vms / (1024 * 1024),  # Virtual Memory Size in MB
                'shared': getattr(memory_info, 'shared', 0) / (1024 * 1024),  # Shared memory in MB
                'system_total': psutil.virtual_memory().total / (1024 * 1024) if heavy else None,
                'system_available': psutil.virtual_memory().available / (1024 * 1024) if heavy else None,
                'system_percent': psutil.virtual_memory().percent if heavy else None,
            }
            self.metrics['memory_details'].append(memory_details)
            
            # Get I/O counters
            if heavy:
                try:
                    io_counters = process.io_counters()
                    io_data = {
                        'read_count': io_counters.read_count,
                        'write_count': io_counters.write_count,
                        'read_bytes': io_counters.read_bytes,
                        'write_bytes': io_counters.write_bytes,
                    }
                    self.metrics['io_counters'].append(io_data)
                    
                    # Log I/O metrics for debugging (throttled)
                    if getattr(self, 'metrics_verbose', False):
                        self.update.emit(f"I/O - Read: {io_counters.read_count}, Write: {io_counters.write_count}")
                except (AttributeError, psutil.AccessDenied) as e:
                    # Some platforms might not support this
                    self.update.emit(f"Warning: Unable to collect I/O metrics: {str(e)}")
                    pass
            
            # Get disk usage
            if heavy:
                try:
                    disk_usage = {
                        'total': psutil.disk_usage('/').total / (1024 * 1024 * 1024),  # GB
                        'used': psutil.disk_usage('/').used / (1024 * 1024 * 1024),    # GB
                        'free': psutil.disk_usage('/').free / (1024 * 1024 * 1024),    # GB
                        'percent': psutil.disk_usage('/').percent
                    }
                    self.metrics['disk_usage'].append(disk_usage)
                except Exception as disk_err:
                    self.update.emit(f"Warning: Unable to collect disk metrics: {str(disk_err)}")
                    pass
            
            # Get network usage (bytes sent/received)
            if heavy:
                try:
                    net_io = psutil.net_io_counters()
                    net_data = {
                        'bytes_sent': net_io.bytes_sent,
                        'bytes_recv': net_io.bytes_recv,
                        'packets_sent': net_io.packets_sent,
                        'packets_recv': net_io.packets_recv,
                    }
                    self.metrics['network_usage'].append(net_data)
                except Exception as net_err:
                    self.update.emit(f"Warning: Unable to collect network metrics: {str(net_err)}")
                    pass
            
            # Get thread count
            self.metrics['thread_count'].append(process.num_threads())
            
            # Emit current metrics for real-time monitoring with enhanced data
            current_metrics = {
                'cpu': cpu_percent,
                'cpu_per_core': per_core_cpu if heavy else None,
                'memory': memory_usage_mb,
                'memory_details': memory_details,
                'thread_count': process.num_threads(),
                'time': time.time() - self.metrics['start_time'] if self.metrics['start_time'] else 0
            }
            self.generation_metrics.emit(current_metrics)
            
            # Log that metrics collection completed (throttled)
            if getattr(self, 'metrics_verbose', False):
                self.update.emit("Resource metrics collection completed")
        except Exception as e:
            self.update.emit(f"Warning: Failed to update resource metrics: {str(e)}")
            
    def _start_metrics_tracking(self):
        """Start tracking computational metrics"""
        if not self.track_metrics:
            return
            
        self.metrics['start_time'] = time.time()
        # Set up the metrics timer to collect data regularly
        self.metrics_timer.timeout.connect(self._update_resource_metrics)
        try:
            self._metrics_tick = 0
            self._metrics_heavy_every = max(1, int(5000 / self.metrics_timer_interval))
        except Exception:
            self._metrics_heavy_every = 5
            self._metrics_tick = 0
        self.metrics_timer.start(self.metrics_timer_interval)
        self.update.emit(f"Started metrics tracking with interval: {self.metrics_timer_interval}ms")
        
    def _stop_metrics_tracking(self):
        """Stop tracking computational metrics and calculate final values"""
        if not self.track_metrics:
            return
            
        self.metrics_timer.stop()
        self.metrics['end_time'] = time.time()
        self.metrics['total_duration'] = self.metrics['end_time'] - self.metrics['start_time']
        
        # Calculate convergence metrics if we have enough data
        if len(self.metrics['best_fitness_per_gen']) > 1:
            # Calculate convergence rate as improvement per generation
            fitness_improvements = []
            for i in range(1, len(self.metrics['best_fitness_per_gen'])):
                improvement = self.metrics['best_fitness_per_gen'][i-1] - self.metrics['best_fitness_per_gen'][i]
                if improvement > 0:  # Only count actual improvements
                    fitness_improvements.append(improvement)
            
            if fitness_improvements:
                self.metrics['avg_improvement_per_gen'] = sum(fitness_improvements) / len(fitness_improvements)
                self.metrics['max_improvement'] = max(fitness_improvements)
            
            # Calculate convergence rate as percentage of max improvement achieved per generation
            total_improvement = self.metrics['best_fitness_per_gen'][0] - min(self.metrics['best_fitness_per_gen'])
            if total_improvement > 0:
                self.metrics['convergence_percentage'] = [(self.metrics['best_fitness_per_gen'][0] - fitness) / 
                                                        total_improvement * 100 
                                                        for fitness in self.metrics['best_fitness_per_gen']]
            
        # Emit the complete metrics data
        self.benchmark_data.emit(self.metrics)
