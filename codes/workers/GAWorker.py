"""
Genetic Algorithm Worker (GAWorker) Pseudocode

CLASS: GAWorker
    INHERITS: QThread
    
    SIGNALS:
        finished(results_dict: dict, best_individual: list, param_names: list, best_fitness: float)
        error(error_message: str)
        update(status_message: str)
        progress(percentage: int)
        benchmark_data(dict)
        generation_metrics(dict)

    METHODS:
        __init__(self, main_params: dict, target_values_dict: dict, weights_dict: dict, 
                omega_start: float, omega_end: float, omega_points: int,
                ga_pop_size: int, ga_num_generations: int, ga_cxpb: float, 
                ga_mutpb: float, ga_tol: float, ga_parameter_data: dict, alpha: float = 0.01, track_metrics=False,
                adaptive_rates=False, stagnation_limit=5, cxpb_min=0.1, cxpb_max=0.9, mutpb_min=0.05, mutpb_max=0.5)
            - Initialize GA parameters and thread safety mechanisms
            - Set up watchdog timer for safety
            - Store all input parameters as instance variables

        __del__(self)
            - Cleanup method when object is destroyed
            - Stop thread and release resources

        handle_timeout(self)
            - Handle watchdog timer timeout
            - Abort operation if taking too long

        cleanup(self)
            - Clean up DEAP framework types
            - Stop watchdog timer
            - Prevent memory leaks

        run(self)
            - Main execution method for GA optimization
            - Sets up DEAP toolbox and genetic operators
            - Implements evolution loop
            - Handles results processing and error recovery

    HELPER FUNCTIONS:
        safe_deap_operation(func: callable) -> callable
            - Decorator for safe DEAP operations
            - Implements retry logic with error recovery
            - Maximum 3 retry attempts

    GENETIC ALGORITHM COMPONENTS:
        1. Population Initialization
            - create_initial_population(self) -> list
            - Each solution is a set of parameters within bounds

        2. Fitness Evaluation
            - evaluate_fitness(self, individual: list) -> float
            - Uses frf.analyze() for FRF analysis
            - calculate_fitness() based on target values and weights
            - apply_sparsity_penalty() for simpler solutions
            
            FITNESS FUNCTION DETAILS:
            The fitness function evaluates solutions using three components:
            1. Primary Objective (Distance from Target):
               - Measures how close the solution is to target value of 1.0
               - Formula: abs(singular_response - 1.0)
               - Example: If response is 1.2, objective = 0.2
            
            2. Sparsity Penalty:
               - Encourages simpler solutions by penalizing complexity
               - Formula: alpha * sum(abs(param) for param in individual)
               - alpha is a weight factor (default 0.01)
               - Higher parameter values increase penalty
            
            3. Percentage Error Sum:
               - Sums absolute percentage differences from target values
               - Formula: sum(abs(percent_diff) for all criteria)
               - Prevents positive and negative errors from cancelling
            
            Final Fitness = Primary Objective + Sparsity Penalty + Percentage Error Sum
            - Lower fitness values indicate better solutions
            - Invalid solutions return high penalty (1e6)

        3. Selection
            - tournament_selection(self, population: list, k: int = 3) -> list
            - Choose best solutions for reproduction

        4. Crossover
            - blend_crossover(self, ind1: list, ind2: list, alpha: float = 0.5) -> tuple
            - Combine pairs of solutions to create new ones
            - Respect parameter bounds and fixed parameters

        5. Mutation
            - mutate_parameters(self, individual: list, mutpb: float) -> list
            - Random parameter changes within bounds
            - Skip fixed parameters
            - Maintain solution validity

        6. Evolution Loop
            - evolve_population(self) -> list
            - Iterate for specified number of generations
            - Track best solution found
            - Check for convergence
            - Update progress and statistics

    ERROR HANDLING:
        - retry_deap_operation() for DEAP operations
        - handle_frf_evaluation_failure() for FRF evaluation failures
        - timeout_protection() for timeout protection
        - cleanup_resources() for resource cleanup

    THREAD SAFETY:
        - mutex_lock: QMutex for critical sections
        - wait_condition: QWaitCondition for thread coordination
        - abort_flag: bool for safe abort mechanism

    OUTPUT PROCESSING:
        - evaluate_best_solution(self, best_individual: list) -> dict
        - calculate_composite_measures(self, results: dict) -> dict
        - generate_response_value(self, measures: dict) -> float
        - report_errors(self, error: Exception) -> None
"""



import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import traceback
import psutil
import time
import platform
import json
from datetime import datetime
from math import sqrt, log
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QDoubleSpinBox, QSpinBox,
    QVBoxLayout, QHBoxLayout, QPushButton, QTabWidget, QFormLayout, QGroupBox,
    QTextEdit, QCheckBox, QScrollArea, QFileDialog, QMessageBox, QDockWidget,
    QMenuBar, QMenu, QAction, QSplitter, QToolBar, QStatusBar, QLineEdit, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QSizePolicy, QActionGroup
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, QWaitCondition, QTimer
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


# Helper function to safely perform operations involving DEAP
def safe_deap_operation(func):
    """Decorator for safely executing DEAP operations with error recovery"""
    def wrapper(*args, **kwargs):
        # Set maximum number of retry attempts for failed operations
        max_retries = 3
        
        # Loop through retry attempts
        for attempt in range(max_retries):
            try:
                # Try to execute the decorated function with its arguments
                return func(*args, **kwargs)
            except Exception as e:
                # If we haven't reached max retries, attempt recovery
                if attempt < max_retries - 1:
                    # Log the failure and retry attempt
                    print(f"DEAP operation failed, retrying ({attempt+1}/{max_retries}): {str(e)}")
                    
                    # Attempt to clean up DEAP global attributes that might be corrupted
                    try:
                        # Remove FitnessMin attribute if it exists
                        if hasattr(creator, "FitnessMin"):
                            delattr(creator, "FitnessMin")
                        # Remove Individual attribute if it exists
                        if hasattr(creator, "Individual"):
                            delattr(creator, "Individual")
                    except:
                        # If cleanup fails, continue to next retry attempt
                        pass
                else:
                    # If we've exhausted all retries, log final failure and raise the exception
                    print(f"DEAP operation failed after {max_retries} attempts: {str(e)}")
                    raise
    # Return the wrapper function that will handle the retry logic
    return wrapper


class GAWorker(QThread):
    """
    Genetic Algorithm Worker Class
    
    This class implements a genetic algorithm (GA) optimization process in a separate thread.
    A genetic algorithm is a search heuristic inspired by natural selection and genetics.
    It's used to find optimal solutions to complex problems by simulating the process of
    natural selection.
    
    Scientific Explanation:
    - Genetic algorithms mimic biological evolution by:
      1. Creating a population of potential solutions
      2. Evaluating their fitness (how good they solve the problem)
      3. Selecting the best solutions
      4. Creating new solutions through crossover (combining good solutions)
      5. Introducing random mutations to maintain diversity
      6. Repeating until an optimal solution is found
    
    Technical Explanation:
    - This class inherits from QThread to run the GA in a separate thread
    - Uses PyQt signals to communicate results back to the main application
    - Implements thread safety mechanisms to prevent crashes
    - Includes a watchdog timer to prevent infinite loops
    """
    
    # Define signals that this class can emit to communicate with the main application
    finished = pyqtSignal(dict, list, list, float)  # Emits: results, best individual, parameter names, best fitness
    error = pyqtSignal(str)                         # Emits: error messages
    update = pyqtSignal(str)                        # Emits: status updates
    progress = pyqtSignal(int)                      # Emits: progress percentage (0-100)
    # New signals for benchmarking
    benchmark_data = pyqtSignal(dict)               # Emits: benchmark metrics (CPU, memory, convergence)
    generation_metrics = pyqtSignal(dict)           # Emits: per-generation metrics for real-time tracking

    def __init__(self, 
                 main_params,           # Main parameters for the system
                 target_values_dict,    # Dictionary of target values to optimize towards
                 weights_dict,          # Dictionary of weights for different objectives
                 omega_start,           # Starting frequency for analysis
                 omega_end,             # Ending frequency for analysis
                 omega_points,          # Number of frequency points to analyze
                 ga_pop_size,           # Size of the genetic algorithm population
                 ga_num_generations,    # Number of generations to run the GA
                 ga_cxpb,               # Crossover probability (chance of combining solutions)
                 ga_mutpb,              # Mutation probability (chance of random changes)
                 ga_tol,                # Tolerance for convergence
                 ga_parameter_data,     # Data about parameters to optimize
                 alpha=0.01,            # Learning rate or step size
                 track_metrics=False,   # Whether to track and report computational metrics
                 adaptive_rates=False,  # Whether to use adaptive crossover and mutation rates
                 stagnation_limit=5,    # Number of generations without improvement before adapting rates
                 cxpb_min=0.1,          # Minimum crossover probability
                 cxpb_max=0.9,          # Maximum crossover probability
                 mutpb_min=0.05,        # Minimum mutation probability
                 mutpb_max=0.5,
                 # ML/Bandit-based adaptive controller for rates + population
                 use_ml_adaptive=False,
                 pop_min=None,
                 pop_max=None,
                 ml_ucb_c=0.6,         # Exploration strength for UCB
                 ml_adapt_population=True,  # Whether ML controller can resize population
                 ml_diversity_weight=0.02,  # Penalty weight for diversity deviation
                 ml_diversity_target=0.2,
                 # Surrogate-assisted screening
                 use_surrogate=False,
                 surrogate_pool_factor=2.0,
                 surrogate_k=5,
                 surrogate_explore_frac=0.15):  # fraction evaluated for exploration
        """
        Initialize the Genetic Algorithm Worker
        
        Parameters:
        - main_params: Core parameters of the system being optimized
            # This is like the DNA of your system - all the basic settings that define how it works
            # For example, if optimizing a car engine, this would include things like cylinder size, fuel type, etc.
            # In code terms, this is usually a dictionary or object containing all the initial settings

        - target_values_dict: What we're trying to achieve
            # Think of this as your "goal" - what you want the system to do
            # Like setting a target speed for a car or a target temperature for a heater
            # In code, this is a dictionary where keys are what you're measuring and values are your goals

        - weights_dict: How important each objective is
            # This tells the algorithm which goals are more important than others
            # Like saying "fuel efficiency is twice as important as top speed"
            # In code, this is a dictionary where higher numbers mean more important objectives

        - omega_start/end/points: Frequency range for analysis
            # These define the range of frequencies we're analyzing
            # Like tuning a radio across different stations (frequencies)
            # In code:
            #   omega_start: The lowest frequency to check
            #   omega_end: The highest frequency to check
            #   omega_points: How many frequencies to check in between

        - ga_pop_size: How many potential solutions to maintain
            # This is like having multiple different designs to try
            # More solutions = better chance of finding the best one, but slower
            # In code, this is just a number (like 100) representing how many solutions to keep

        - ga_num_generations: How long to run the optimization
            # How many times the algorithm will try to improve the solutions
            # Like breeding plants for multiple generations to get better crops
            # In code, this is a number representing how many improvement cycles to run

        - ga_cxpb: Probability of combining solutions (like genetic crossover)
            # Chance of mixing two good solutions to create a new one
            # Like breeding two good plants to get a better one
            # In code, this is a number between 0 and 1 (like 0.7 for 70% chance)

        - ga_mutpb: Probability of random changes (like genetic mutations)
            # Chance of making random changes to a solution
            # Like random mutations in DNA that might lead to improvements
            # In code, this is a number between 0 and 1 (like 0.1 for 10% chance)

        - ga_tol: How close we need to get to consider it solved
            # How close to the target we need to be to say "good enough"
            # Like saying "if we're within 1 degree of the target temperature, that's fine"
            # In code, this is a small number (like 0.001) representing acceptable error

        - ga_parameter_data: What parameters we're trying to optimize
            # Which parts of the system we're allowed to change
            # Like saying "we can adjust the engine size but not the fuel type"
            # In code, this is a dictionary or list of parameters that can be modified

        - alpha: Step size for optimization (default 0.01)
            # How big of changes to make when trying to improve
            # Like taking small steps when climbing a mountain to avoid missing the path
            # In code, this is a small number controlling how much to change things each time
        - track_metrics: Whether to collect and report computational metrics for benchmarking
        - adaptive_rates: Whether to automatically adjust crossover and mutation rates
            # This allows the algorithm to adapt its exploration vs. exploitation balance
            # When progress stagnates, it will adjust rates to try new search strategies
            
        - stagnation_limit: Number of generations without improvement before adapting rates
            # After this many generations without finding a better solution,
            # the algorithm will adjust its crossover and mutation rates
            
        - cxpb_min/max: Minimum/maximum crossover probability when using adaptive rates
            # Sets bounds for how much the algorithm can adjust crossover probability
            
        - mutpb_min/max: Minimum/maximum mutation probability when using adaptive rates
            # Sets bounds for how much the algorithm can adjust mutation probability
        """
        # Call the parent class (QThread) constructor
        super().__init__()
        

        
        # Store all the input parameters as instance variables
        # These are like the "settings" for our genetic algorithm
        # Think of it like setting up a recipe - we need to know all the ingredients and steps
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
        self.alpha = alpha                      # How big of steps to take (like how much to adjust the engine)
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
        
        # ML/Bandit controller configuration
        self.use_ml_adaptive = use_ml_adaptive
        # Guardrail population bounds
        self.pop_min = pop_min if pop_min is not None else max(10, int(0.5 * self.ga_pop_size))
        self.pop_max = pop_max if pop_max is not None else int(2.0 * self.ga_pop_size)
        self.ml_ucb_c = ml_ucb_c
        self.ml_adapt_population = ml_adapt_population
        self.ml_diversity_weight = ml_diversity_weight
        self.ml_diversity_target = ml_diversity_target

        # Surrogate configuration
        self.use_surrogate = use_surrogate
        self.surrogate_pool_factor = max(1.0, float(surrogate_pool_factor))
        self.surrogate_k = max(1, int(surrogate_k))
        self.surrogate_explore_frac = max(0.0, min(0.5, float(surrogate_explore_frac)))
        self._surrogate_X = []  # raw parameter vectors
        self._surrogate_y = []  # corresponding fitness values
        
        # Thread safety mechanisms - these prevent crashes when multiple parts of the program try to use the same data
        # Think of it like traffic lights controlling access to a busy intersection
        self.mutex = QMutex()                   # A lock that only one part of the program can hold at a time
        self.condition = QWaitCondition()       # A way for different parts to signal each other
        self.abort = False                      # A flag to safely stop the program if needed
        
        # Watchdog timer - like a safety timer that stops the program if it runs too long
        # Similar to how a microwave stops if it runs too long to prevent overheating
        self.watchdog_timer = QTimer()          # Create a timer object
        self.watchdog_timer.setSingleShot(True) # Timer only goes off once (like a one-time alarm)
        self.watchdog_timer.timeout.connect(self.handle_timeout)  # What to do when timer goes off
        self.last_progress_update = 0           # Keep track of when we last updated progress
        
        # Initialize benchmark metrics tracking
        self.metrics = {
            'start_time': None,
            'end_time': None,
            'total_duration': None,
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
            'pop_size_history': [],        # Track population size across generations
            'rates_history': [],           # Track cxpb/mutpb chosen each generation
            'controller': None,            # Which controller was used: 'fixed' | 'adaptive' | 'ml_bandit'
            # Surrogate metrics
            'surrogate_enabled': bool(use_surrogate),
            'surrogate_pool_factor': float(self.surrogate_pool_factor),
            'surrogate_k': int(self.surrogate_k),
            'surrogate_explore_frac': float(self.surrogate_explore_frac),
            'surrogate_info': []           # List of dicts per generation with pool/eval counts and error
        }
        
        # Create metrics tracking timer
        self.metrics_timer = QTimer()
        self.metrics_timer_interval = 500  # milliseconds
        
    def __del__(self):
        """
        Cleanup method that runs when the object is destroyed
        Like cleaning up after a party - making sure everything is turned off and put away
        """
        self.mutex.lock()                       # Get exclusive access to prevent other parts from interfering
        self.abort = True                       # Tell the program to stop
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
            self.mutex.unlock()                 # Release the lock
            self.error.emit("Genetic algorithm optimization timed out. The operation was taking too long.")
        else:
            self.mutex.unlock()                 # Release the lock
            
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
            except:
                pass                            # Ignore errors if it's already gone
        if hasattr(creator, "Individual"):      # If we have an individual type defined
            try:
                delattr(creator, "Individual")  # Remove it
            except:
                pass                            # Ignore errors if it's already gone
        
        # Stop the watchdog timer if it's running
        # Like turning off the microwave timer
        if self.watchdog_timer.isActive():
            self.watchdog_timer.stop()
 
    @safe_deap_operation  # Decorator that ensures safe execution of DEAP operations
    def run(self):
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
        
        # Debug output for adaptive rates / ML controller settings
        self.update.emit(f"DEBUG: adaptive_rates parameter is set to: {self.adaptive_rates}")
        self.update.emit(f"DEBUG: ML bandit controller is set to: {self.use_ml_adaptive}")
        self.update.emit(f"DEBUG: GA parameters: crossover={self.ga_cxpb:.4f}, mutation={self.ga_mutpb:.4f}")
        # Record which controller is active for this run
        try:
            if self.use_ml_adaptive:
                self.metrics['controller'] = 'ml_bandit'
            elif self.adaptive_rates:
                self.metrics['controller'] = 'adaptive'
            else:
                self.metrics['controller'] = 'fixed'
        except Exception:
            pass

        if self.adaptive_rates:
            self.update.emit(f"DEBUG: Adaptive rate parameters:")
            self.update.emit(f"DEBUG: - Stagnation limit: {self.stagnation_limit}")
            self.update.emit(f"DEBUG: - Crossover range: {self.cxpb_min:.2f} - {self.cxpb_max:.2f}")
            self.update.emit(f"DEBUG: - Mutation range: {self.mutpb_min:.2f} - {self.mutpb_max:.2f}")
        if self.use_ml_adaptive:
            self.update.emit(f"DEBUG: ML params: UCB c={self.ml_ucb_c:.2f}, pop_adapt={self.ml_adapt_population}, div_weight={self.ml_diversity_weight:.3f}, div_target={self.ml_diversity_target:.2f}")
        if self.use_surrogate:
            self.update.emit(f"DEBUG: Surrogate screening enabled → pool_factor={self.surrogate_pool_factor:.2f}, k={self.surrogate_k}, explore_frac={self.surrogate_explore_frac:.2f}")
        
        # Start metrics tracking if enabled
        if self.track_metrics:
            self._start_metrics_tracking()
        
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
                # Check if we should stop the evaluation
                if self.abort:
                    return (1e6,)  # Return a very bad score (1 million) to signal we should stop

                # Track evaluation count for benchmark metrics
                if self.track_metrics:
                    self.metrics['evaluation_count'] += 1

                # Convert our solution into a tuple (immutable list) for the FRF analysis
                # This is like preparing the robot for testing
                dva_parameters_tuple = tuple(individual)

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
                    
                    # Calculate how complex our solution is
                    # We sum up all parameter values and multiply by a weight (self.alpha)
                    # This penalizes solutions that use too many parameters
                    sparsity_penalty = self.alpha * sum(abs(param) for param in individual)
                    
                    # Calculate sum of percentage differences
                    percentage_error_sum = 0.0
                    if "percentage_differences" in results:
                        for mass_key, pdiffs in results["percentage_differences"].items():
                            for criterion, percent_diff in pdiffs.items():
                                # Use absolute value to prevent positive and negative errors from cancelling
                                percentage_error_sum += abs(percent_diff)
                    
                    # Store the fitness components in the individual's attributes
                    # This allows us to access them later for detailed reporting
                    individual.primary_objective = primary_objective
                    individual.sparsity_penalty = sparsity_penalty
                    individual.percentage_error = percentage_error_sum/100
                    
                    # Combine all three components to get final score:
                    # 1. Primary objective: Distance from target value of 1.0
                    # 2. Sparsity penalty: Encourages simpler solutions
                    # 3. Percentage error sum: Sum of all percentage differences from target values
                    # Lower score = better solution (like golf scoring)
                    fitness = primary_objective + sparsity_penalty + percentage_error_sum/1000
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
            toolbox.register("mate", tools.cxBlend, alpha=0.5)  # Blend two solutions together

            # ============================================================================
            # MUTATION FUNCTION
            # ============================================================================
            # This function randomly changes some parameters of a solution
            # Think of it like making small random adjustments to a recipe
            def mutate_individual(individual, indpb=0.1):
                # First, check if we should stop (like if someone called timeout)
                if self.abort:
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
                        # Make a small random change (up to ±10% of the parameter range)
                        perturb = random.uniform(-0.1 * (max_val - min_val), 0.1 * (max_val - min_val))
                        individual[i] += perturb
                        # Make sure we stay within allowed bounds
                        individual[i] = max(min_val, min(individual[i], max_val))
                return (individual,)

            # Register our mutation function and selection method
            toolbox.register("mutate", mutate_individual)
            toolbox.register("select", tools.selTournament, tournsize=3)  # Tournament selection with 3 competitors

            # ============================================================================
            # INITIAL POPULATION
            # ============================================================================
            # Create our first generation of solutions
            self.update.emit("Initializing population...")
            population = toolbox.population(n=self.ga_pop_size)  # Create population of specified size
            # Track initial population size
            if self.track_metrics:
                self.metrics['pop_size_history'].append(len(population))
            
            # Score each solution in our initial population
            self.update.emit("Evaluating initial population...")
            fitnesses = list(map(toolbox.evaluate, population))
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

            # ============================================================================
            # EVOLUTION LOOP
            # ============================================================================
            # This is where the magic happens! We'll evolve our solutions over multiple generations
            self.update.emit("Starting evolution...")
            best_fitness_overall = float('inf')  # Track the best solution we've found
            best_ind_overall = None  # Store the best solution
            
            # --- ML Bandit Controller setup ---
            if self.use_ml_adaptive:
                # Define a simple UCB-based controller over discrete action space
                # Actions are relative multipliers for cxpb/mutpb and a population multiplier
                deltas = [-0.25, -0.1, 0.0, 0.1, 0.25]
                pop_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
                ml_actions = [(dcx, dmu, pm) for dcx in deltas for dmu in deltas for pm in pop_multipliers]
                ml_counts = [0 for _ in ml_actions]
                ml_sums = [0.0 for _ in ml_actions]
                ml_t = 0

                def ml_select_action(current_cx, current_mu, current_pop):
                    nonlocal ml_t
                    ml_t += 1
                    # UCB score
                    scores = []
                    for i, _ in enumerate(ml_actions):
                        if ml_counts[i] == 0:
                            scores.append((float('inf'), i))
                        else:
                            avg = ml_sums[i] / ml_counts[i]
                            bonus = self.ml_ucb_c * sqrt(log(max(ml_t, 1)) / ml_counts[i])
                            scores.append((avg + bonus, i))
                    scores.sort(key=lambda t: t[0], reverse=True)
                    _, idx = scores[0]
                    dcx, dmu, pm = ml_actions[idx]
                    new_cx = min(self.cxpb_max, max(self.cxpb_min, current_cx * (1.0 + dcx)))
                    new_mu = min(self.mutpb_max, max(self.mutpb_min, current_mu * (1.0 + dmu)))
                    new_pop = int(min(self.pop_max, max(self.pop_min, round(current_pop * pm))))
                    return idx, new_cx, new_mu, new_pop

                def ml_update(idx, reward):
                    ml_counts[idx] += 1
                    ml_sums[idx] += float(reward)

                def resize_population(pop, new_size):
                    # Shrink: keep best
                    if new_size < len(pop):
                        return tools.selBest(pop, new_size)
                    # Grow: add random individuals
                    extra = new_size - len(pop)
                    for _ in range(extra):
                        pop.append(toolbox.individual())
                    return pop

            # Run for the specified number of generations
            for gen in range(1, self.ga_num_generations + 1):
                # Check if we should stop
                if self.abort:
                    self.update.emit("Optimization aborted by user")
                    break
                
                # Track generation start time for benchmarking
                if self.track_metrics:
                    gen_start_time = time.time()
                    generation_time_breakdown = {}
                    
                    # Track selection time
                    selection_start = time.time()
                
                # Show progress
                self.update.emit(f"-- Generation {gen} / {self.ga_num_generations} --")
                
                # Update progress bar
                progress_percent = int((gen / self.ga_num_generations) * 100)
                self.progress.emit(progress_percent)
                self.last_progress_update = progress_percent
                
                # Reset watchdog timer (safety feature to prevent infinite loops)
                if self.watchdog_timer.isActive():
                    self.watchdog_timer.stop()
                self.watchdog_timer.start(600000)  # 10 minutes
                
                # Determine current rates and optionally adjust using ML bandit
                evals_this_gen = 0
                if self.use_ml_adaptive:
                    # Initial defaults are the current values
                    old_cxpb = self.current_cxpb
                    old_mutpb = self.current_mutpb
                    old_pop_size = len(population)
                    idx, new_cx, new_mu, new_pop = ml_select_action(self.current_cxpb, self.current_mutpb, len(population))
                    self.current_cxpb = new_cx
                    self.current_mutpb = new_mu
                    if self.ml_adapt_population and new_pop != len(population):
                        population = resize_population(population, new_pop)
                        # Evaluate any individuals missing fitness (new ones)
                        need_eval = [ind for ind in population if not ind.fitness.valid]
                        if need_eval:
                            self.update.emit(f"  ML ctrl: evaluating {len(need_eval)} new individuals after resize...")
                            eval_start = time.time()
                            fits_new = list(map(toolbox.evaluate, need_eval))
                            for ind, fit in zip(need_eval, fits_new):
                                ind.fitness.values = fit
                            if self.track_metrics:
                                self.metrics['evaluation_times'].append(time.time() - eval_start)
                            evals_this_gen += len(need_eval)
                    current_cxpb = self.current_cxpb
                    current_mutpb = self.current_mutpb
                    # Log
                    self.update.emit("  Rates type: ML-Bandit")
                    self.update.emit(f"  - Crossover: {current_cxpb:.4f}")
                    self.update.emit(f"  - Mutation: {current_mutpb:.4f}")
                    self.update.emit(f"  - Population: {len(population)}")
                else:
                    # Use current adaptive rates if enabled (legacy heuristic)
                    current_cxpb = self.current_cxpb if self.adaptive_rates else self.ga_cxpb
                    current_mutpb = self.current_mutpb if self.adaptive_rates else self.ga_mutpb
                    self.update.emit(f"  Rates type: {'Adaptive' if self.adaptive_rates else 'Fixed'}")
                    self.update.emit(f"  - Crossover: {current_cxpb:.4f}")
                    self.update.emit(f"  - Mutation: {current_mutpb:.4f}")
                
                if self.adaptive_rates:
                    # Calculate change from previous rates if not first generation
                    if gen > 1 and hasattr(self, 'prev_cxpb') and hasattr(self, 'prev_mutpb'):
                        cxpb_change = current_cxpb - self.prev_cxpb
                        mutpb_change = current_mutpb - self.prev_mutpb
                        if cxpb_change != 0 or mutpb_change != 0:
                            self.update.emit(f"  - Changes: cx {'+' if cxpb_change > 0 else ''}{cxpb_change:.4f}, mut {'+' if mutpb_change > 0 else ''}{mutpb_change:.4f}")
                    self.update.emit(f"  - Stagnation counter: {self.stagnation_counter}/{self.stagnation_limit}")
                    
                    # Store current rates for next generation comparison
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

                        # Normalize helper
                        def _norm_vec(vec):
                            out = []
                            for i, val in enumerate(vec):
                                lo, hi = parameter_bounds[i]
                                if hi == lo:
                                    out.append(0.0)
                                else:
                                    out.append((val - lo) / (hi - lo))
                            return out

                        # KNN surrogate prediction
                        Xn = [_norm_vec(x) for x in self._surrogate_X]
                        def _predict_fitness(v):
                            vz = _norm_vec(v)
                            dists = []
                            for Xrow, y in zip(Xn, self._surrogate_y):
                                d = 0.0
                                for a, b in zip(vz, Xrow):
                                    d += (a - b) * (a - b)
                                d = d ** 0.5
                                dists.append((d, y))
                            dists.sort(key=lambda t: t[0])
                            k = min(self.surrogate_k, len(dists))
                            return sum(y for _, y in dists[:k]) / max(1, k)

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
                                vz = _norm_vec(list(ind))
                                # min distance to seen
                                mind = float('inf')
                                for Xrow in Xn:
                                    d = 0.0
                                    for a, b in zip(vz, Xrow):
                                        d += (a - b) * (a - b)
                                    d = d ** 0.5
                                    if d < mind:
                                        mind = d
                                return mind
                            remain = [ind for _, ind in scored[exploit_n:]]
                            remain.sort(key=lambda ind: _novelty(ind), reverse=True)
                            chosen.extend(remain[:explore_n])

                        # Evaluate chosen only
                        self.update.emit(f"  Surrogate: pool={len(pool)} eval={len(chosen)} (exploit={exploit_n}, explore={len(chosen)-exploit_n})")
                        evaluation_start = time.time()
                        fits = list(map(toolbox.evaluate, chosen))
                        for ind, fit in zip(chosen, fits):
                            ind.fitness.values = fit
                        if self.track_metrics:
                            self.metrics['evaluation_times'].append(time.time() - evaluation_start)
                        evals_this_gen += len(chosen)

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
                        fitnesses = map(toolbox.evaluate, invalid_ind)
                        for ind, fit in zip(invalid_ind, fitnesses):
                            ind.fitness.values = fit
                        evals_this_gen += len(invalid_ind)
                
                if self.track_metrics:
                    evaluation_time = time.time() - evaluation_start
                    self.metrics['evaluation_times'].append(evaluation_time)
                    generation_time_breakdown['evaluation'] = evaluation_time
                
                # 5. REPLACEMENT: Replace old population with new one
                population[:] = offspring

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

                # Create a table for fitness components
                best_idx = fits.index(min_fit)
                best_individual = population[best_idx]

                # Check if the best individual has the fitness components
                has_components = hasattr(best_individual, 'primary_objective')
                
                # If adaptive rates are enabled, check if we need to adjust rates
                if self.adaptive_rates:
                    # Track if we found an improvement
                    improved = False
                    
                    # Track best solution found
                    if min_fit < best_fitness_overall:
                        improved = True
                        best_fitness_overall = min_fit
                        best_ind_overall = tools.selBest(population, 1)[0]
                        self.update.emit(f"  New best solution found! Fitness: {best_fitness_overall:.6f}")
                        
                        # Reduce stagnation counter but don't reset completely when improvement is found
                        # This ensures rates will still adapt periodically even during successful runs
                        self.stagnation_counter = max(0, self.stagnation_counter - 1)
                    else:
                        # No improvement, increment stagnation counter
                        self.stagnation_counter += 1
                        
                    # If we've reached the stagnation limit or it's an even-numbered generation (to ensure periodic adaptation)
                    # Force adaptation at least every 3 generations to ensure rates change during short runs
                    if self.stagnation_counter >= self.stagnation_limit or gen % 3 == 0:
                        # We'll adjust rates based on current convergence state:
                        # - If population diversity is low (low std), increase mutation to explore more
                        # - If diversity is high (high std), increase crossover to exploit more
                        
                        # Calculate normalized diversity (0 to 1)
                        if mean > 0:
                            normalized_diversity = min(1.0, std / mean)
                        else:
                            normalized_diversity = 0.5  # Default middle value
                        
                        # Adjust crossover and mutation rates based on diversity
                        if normalized_diversity < 0.1:  # Low diversity
                            # Increase mutation, decrease crossover to explore more
                            self.current_mutpb = min(self.mutpb_max, self.current_mutpb * 1.5)  # Larger multiplier for more dramatic change
                            self.current_cxpb = max(self.cxpb_min, self.current_cxpb * 0.8)     # Smaller multiplier for more dramatic change
                            adaptation_type = "Increasing exploration (↑mutation, ↓crossover)"
                        elif normalized_diversity > 0.3:  # High diversity
                            # Increase crossover, decrease mutation to exploit more
                            self.current_cxpb = min(self.cxpb_max, self.current_cxpb * 1.5)     # Larger multiplier for more dramatic change
                            self.current_mutpb = max(self.mutpb_min, self.current_mutpb * 0.8)  # Smaller multiplier for more dramatic change
                            adaptation_type = "Increasing exploitation (↑crossover, ↓mutation)"
                        else:
                            # Alternate strategy: swing in opposite direction
                            if gen % 2 == 0:
                                self.current_cxpb = min(self.cxpb_max, self.current_cxpb * 1.3)
                                self.current_mutpb = max(self.mutpb_min, self.current_mutpb * 0.9)
                                adaptation_type = "Balanced adjustment (↑crossover, ↓mutation)"
                            else:
                                self.current_mutpb = min(self.mutpb_max, self.current_mutpb * 1.3)
                                self.current_cxpb = max(self.cxpb_min, self.current_cxpb * 0.9)
                                adaptation_type = "Balanced adjustment (↓crossover, ↑mutation)"
                        
                        # Log the adaptation
                        self.update.emit(f"  Adapting rates due to {self.stagnation_counter} generations without improvement")
                        self.update.emit(f"  New rates: crossover={self.current_cxpb:.3f}, mutation={self.current_mutpb:.3f} - {adaptation_type}")
                        
                        # Add visual indicators of rate changes
                        cxpb_change = self.current_cxpb - current_cxpb
                        mutpb_change = self.current_mutpb - current_mutpb
                        self.update.emit(f"  ↳ Crossover: {current_cxpb:.4f} → {self.current_cxpb:.4f} ({'+' if cxpb_change > 0 else ''}{cxpb_change:.4f})")
                        self.update.emit(f"  ↳ Mutation:  {current_mutpb:.4f} → {self.current_mutpb:.4f} ({'+' if mutpb_change > 0 else ''}{mutpb_change:.4f})")
                        
                        # Reset stagnation counter
                        self.stagnation_counter = 0
                        
                        # Record adaptation in history
                        adaptation_record = {
                            'generation': gen,
                            'old_cxpb': current_cxpb,
                            'old_mutpb': current_mutpb,
                            'new_cxpb': self.current_cxpb,
                            'new_mutpb': self.current_mutpb,
                            'normalized_diversity': normalized_diversity,
                            'adaptation_type': adaptation_type
                        }
                        self.rate_adaptation_history.append(adaptation_record)
                        
                        if self.track_metrics:
                            self.metrics['adaptive_rates_history'].append(adaptation_record)
                else:
                    # If adaptive rates are not enabled, just track best solution
                    if min_fit < best_fitness_overall:
                        best_fitness_overall = min_fit
                        best_ind_overall = tools.selBest(population, 1)[0]
                        self.update.emit(f"  New best solution found! Fitness: {best_fitness_overall:.6f}")

                if has_components:
                    table_header = "  ┌───────────────────────┬───────────────┬───────────────┬───────────────┬───────────────┐"
                    table_format = "  │ {0:<21} │ {1:>13} │ {2:>13} │ {3:>13} │ {4:>13} │"
                    table_footer = "  └───────────────────────┴───────────────┴───────────────┴───────────────┴───────────────┘"
                    
                    self.update.emit(table_header)
                    self.update.emit(table_format.format("Fitness Components", "Min", "Max", "Average", "Best"))
                    self.update.emit(table_format.format("───────────────────────", "───────────────", "───────────────", "───────────────", "───────────────"))
                    
                    # Calculate component statistics
                    primary_objectives = [ind.primary_objective if hasattr(ind, 'primary_objective') else 0 for ind in population]
                    sparsity_penalties = [ind.sparsity_penalty if hasattr(ind, 'sparsity_penalty') else 0 for ind in population]
                    percentage_errors = [ind.percentage_error if hasattr(ind, 'percentage_error') else 0 for ind in population]
                    
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
                    self.update.emit(table_format.format("Total Fitness", f"{min_fit:.6f}", f"{max_fit:.6f}", f"{mean:.6f}", f"{min_fit:.6f}"))
                    self.update.emit(table_footer)
                    
                    # If adaptive rates are enabled, display current rates
                    if self.adaptive_rates:
                        # Instead of showing rates again, show an indicator of whether rates will be adapted
                        if self.stagnation_counter >= self.stagnation_limit - 1:
                            self.update.emit(f"  ⚠️ Rates will adapt next generation due to stagnation ({self.stagnation_counter}/{self.stagnation_limit})")
                        else:
                            self.update.emit(f"  Stagnation counter: {self.stagnation_counter}/{self.stagnation_limit}")
                else:
                    # If components are not available, use the traditional display
                    self.update.emit(f"  Min fitness: {min_fit:.6f}")
                    self.update.emit(f"  Max fitness: {max_fit:.6f}")
                    self.update.emit(f"  Avg fitness: {mean:.6f}")
                    self.update.emit(f"  Std fitness: {std:.6f}")
                    
                    # If adaptive rates are enabled, display current rates
                    if self.adaptive_rates:
                        # Instead of showing rates again, show an indicator of whether rates will be adapted
                        if self.stagnation_counter >= self.stagnation_limit - 1:
                            self.update.emit(f"  ⚠️ Rates will adapt next generation due to stagnation ({self.stagnation_counter}/{self.stagnation_limit})")
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

                    # If ML controller is active, compute and record reward and controller choice
                    if self.use_ml_adaptive:
                        # Reward: improvement per second with diversity shaping
                        last_best = self.metrics['best_fitness_per_gen'][-2] if len(self.metrics['best_fitness_per_gen']) > 1 else None
                        imp = (last_best - min_fit) if (last_best is not None and last_best > min_fit) else 0.0
                        cv = std / (abs(mean) + 1e-12)
                        effort = max(1.0, evals_this_gen)
                        reward = (imp / max(gen_time, 1e-6)) / effort - self.ml_diversity_weight * abs(cv - self.ml_diversity_target)
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
                            'reward': reward
                        })

                    # Record surrogate info
                    if self.use_surrogate:
                        self.metrics['surrogate_info'].append({
                            'generation': gen,
                            'pool_factor': self.surrogate_pool_factor,
                            'pool_size': int(self.surrogate_pool_factor * len(invalid_ind)) if 'invalid_ind' in locals() else 0,
                            'evaluated_count': evals_this_gen
                        })

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
            except:
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
            # Log that metrics collection is happening
            self.update.emit("Collecting resource metrics...")
            
            # Get basic CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=None)
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_usage_mb = memory_info.rss / (1024 * 1024)  # Convert to MB
            
            # Add basic metrics
            self.metrics['cpu_usage'].append(cpu_percent)
            self.metrics['memory_usage'].append(memory_usage_mb)
            
            # Log basic metrics for debugging
            self.update.emit(f"CPU: {cpu_percent}%, Memory: {memory_usage_mb:.2f} MB")
            
            # Get per-core CPU usage
            per_core_cpu = psutil.cpu_percent(interval=None, percpu=True)
            self.metrics['cpu_per_core'].append(per_core_cpu)
            
            # Get detailed memory information
            memory_details = {
                'rss': memory_info.rss / (1024 * 1024),  # Resident Set Size in MB
                'vms': memory_info.vms / (1024 * 1024),  # Virtual Memory Size in MB
                'shared': getattr(memory_info, 'shared', 0) / (1024 * 1024),  # Shared memory in MB
                'system_total': psutil.virtual_memory().total / (1024 * 1024),  # Total system memory in MB
                'system_available': psutil.virtual_memory().available / (1024 * 1024),  # Available system memory in MB
                'system_percent': psutil.virtual_memory().percent,  # System memory usage percentage
            }
            self.metrics['memory_details'].append(memory_details)
            
            # Get I/O counters
            try:
                io_counters = process.io_counters()
                io_data = {
                    'read_count': io_counters.read_count,
                    'write_count': io_counters.write_count,
                    'read_bytes': io_counters.read_bytes,
                    'write_bytes': io_counters.write_bytes,
                }
                self.metrics['io_counters'].append(io_data)
                
                # Log I/O metrics for debugging
                self.update.emit(f"I/O - Read: {io_counters.read_count}, Write: {io_counters.write_count}")
            except (AttributeError, psutil.AccessDenied) as e:
                # Some platforms might not support this
                self.update.emit(f"Warning: Unable to collect I/O metrics: {str(e)}")
                pass
            
            # Get disk usage
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
                'cpu_per_core': per_core_cpu,
                'memory': memory_usage_mb,
                'memory_details': memory_details,
                'thread_count': process.num_threads(),
                'time': time.time() - self.metrics['start_time'] if self.metrics['start_time'] else 0
            }
            self.generation_metrics.emit(current_metrics)
            
            # Log that metrics collection completed
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