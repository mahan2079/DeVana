import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import qmc
import time
import math
from enum import Enum

from PyQt5.QtCore import Qt, QThread, pyqtSignal

# --- Particle Swarm Optimization (PSO) for Beginners Explained ---

# What is Particle Swarm Optimization (PSO)?
# PSO is a technique inspired by the social behavior of bird flocking or fish schooling.
# Think of it like this:
#
# 1.  **The Swarm:** You have a group of "particles" (like individual birds or fish) that are
#     randomly scattered across the landscape (your problem's search space). Each particle
#     represents a potential solution.
#
# 2.  **Their Movement:** Each particle moves around, trying to find better spots.
#     Their movement is influenced by two main things:
#     *   **Their Own Best Experience (Personal Best):** Each particle remembers the best
#         (lowest/highest) spot it has ever found so far. It has a tendency to move back
#         towards this remembered good spot.
#     *   **The Swarm's Best Experience (Global Best):** The entire swarm knows the
#         absolute best spot that *any* particle has found so far. All particles are
#         also attracted to this globally best known spot.
#
# 3.  **Velocity and Position:**
#     *   **Velocity:** This determines how fast and in what direction a particle moves.
#         It's adjusted based on the particle's current velocity, its personal best, and
#         the global best.
#     *   **Position:** This is the particle's current location in the search space,
#         representing a specific set of parameters for your problem.
#
# 4.  **Fitness Function:** How do particles know if a spot is "good"? They use a "fitness function."
#     This function evaluates how good a particular set of parameters (a particle's position) is.
#     In this code, the `evaluate_particle` method acts as the fitness function. It calculates
#     how well the DVA (Dynamic Vibration Absorber) parameters (the particle's position)
#     achieve the desired frequency response, aiming for a "singular response" close to 1.
#     Lower fitness values are better, meaning a more optimal solution.
#
# 5.  **Iteration:** The particles move, update their personal and global bests, and re-evaluate
#     their positions repeatedly over many "iterations" (cycles) until a good solution is found
#     or a maximum number of iterations is reached.
#
# How does this `PSOWorker.py` code work?
# This Python script implements an *advanced* PSO algorithm specifically for
# optimizing a vibration system (likely a Dynamic Vibration Absorber, or DVA).
# It's designed to run as a separate thread (`QThread`) in a PyQt5 application,
# meaning it can perform calculations in the background without freezing the user interface.
#
# Key components and advanced features in this code:
#
# *   **`PSOWorker` Class:** This is the main class that encapsulates the entire PSO logic.
#     It takes various parameters for the optimization process (swarm size, iterations,
#     parameter bounds, etc.).
#
# *   **Initialization (`__init__` and `run` methods):**
#     *   Particles are initialized with `position` (the DVA parameters), `velocity`,
#         `best_position` (personal best), and `best_fitness`.
#     *   It can use `quasi_random_init` (Sobol sequence) for better initial coverage of
#         the search space, which helps in finding better solutions.
#
# *   **Fitness Evaluation (`evaluate_particle` method):**
#     *   This method calls an external `frf` (Frequency Response Function) module to
#         simulate the vibration system's behavior with the current DVA parameters.
#     *   It calculates a "singular response" and applies various penalties:
#         *   `primary_objective`: How far the singular response is from the ideal value (1).
#         *   `sparsity_penalty`: Encourages simpler designs by penalizing larger parameter values.
#         *   `peak_penalty`: Penalizes high resonance peaks in the frequency response.
#         *   `smoothness_penalty`: Penalizes uneven responses.
#     *   The goal is to minimize this combined fitness value.
#
# *   **PSO Main Loop (`run` method):**
#     *   **Adaptive Parameters:** `adaptive_inertia_weight` and `adaptive_acceleration_coefficients`
#         dynamically adjust `w` (inertia weight), `c1` (cognitive coefficient), and `c2` (social coefficient)
#         over iterations. This helps balance exploration (searching new areas) and
#         exploitation (fine-tuning existing good areas) during the optimization.
#     *   **Neighborhood Topologies (`TopologyType` enum and `create_neighborhoods`):**
#         Particles don't necessarily interact with *everyone*. This code allows for different
#         social structures:
#         *   `GLOBAL`: All particles interact with the swarm's best.
#         *   `RING`, `VON_NEUMANN`, `RANDOM`: Particles only interact with a smaller group of neighbors.
#         This affects how quickly information spreads and helps avoid getting stuck in local optima.
#     *   **Velocity Clamping:** `max_velocity_factor` prevents particle velocities from becoming
#         too large, which can cause them to overshoot good solutions.
#     *   **Boundary Handling (`handle_boundary_violation`):** Specifies how particles behave
#         when they try to move outside the allowed range for a parameter ("absorbing," "reflecting," or "invisible" boundaries).
#     *   **Diversity Maintenance (`apply_mutation` and `calculate_diversity`):**
#         *   `calculate_diversity` measures how spread out the particles are.
#         *   `apply_mutation` randomly perturbs particle positions if diversity is low, helping
#             the swarm escape local optima and explore more effectively.
#     *   **Stagnation Detection and Recovery:** If a particle or the swarm isn't improving for
#         a certain number of iterations (`stagnation_limit`), the code can reinitialize
#         "stagnant" particles to encourage new exploration.
#     *   **Early Stopping:** The optimization can stop early (`early_stopping`) if the improvement
#         in fitness becomes very small, saving computational time.
#
# *   **Signals (`pyqtSignal`):** The `PSOWorker` communicates its progress and final results
#     back to the main application using PyQt signals (`finished`, `error`, `update`, `convergence_signal`).
#
# In essence, this code simulates a "swarm" of potential solutions, where each solution
# learns from its own experience and the experience of its "neighbors" (or the entire swarm)
# to collectively find the best possible DVA parameters that minimize vibrations.
#
# --- End of PSO Explanation ---
import random

# Local imports (assuming similar modules as in GAWorker)
from modules.FRF import frf
from modules.sobol_sensitivity import (
    perform_sobol_analysis,
    calculate_and_save_errors,
    format_parameter_name
)




class TopologyType(Enum):
    """
    Topology types for PSO neighborhood structures.
    
    Different topologies affect information flow through the swarm:
    - GLOBAL: All particles connected, fastest convergence but may get trapped in local optima
    - RING: Each particle connected to two neighbors, slower convergence but better exploration
    - VON_NEUMANN: Grid-like neighborhood structure balancing exploration and exploitation
    - RANDOM: Random connections that change periodically, enhancing diversity
    """
    GLOBAL = 1
    RING = 2
    VON_NEUMANN = 3
    RANDOM = 4

class PSOWorker(QThread):
    """
    Advanced Particle Swarm Optimization (PSO) worker thread for vibration system optimization.
    
    This worker inherits from QThread and implements proper thread termination handling.
    
    Scientific Background:
    ---------------------
    PSO is a population-based stochastic optimization technique inspired by social behavior
    of bird flocking or fish schooling. In PSO, particles move through the search space,
    guided by their own best known position and the swarm's best known position.
    
    This implementation features numerous advanced PSO techniques:
    1. Adaptive parameters (inertia weight, acceleration coefficients)
    2. Constriction factor for stability
    3. Multiple topology options for neighborhood structure
    4. Velocity clamping to prevent explosion
    5. Diversity maintenance through mutation and re-initialization
    6. Stagnation detection and mitigation
    7. Quasi-random initialization for better coverage of search space
    8. Multiple boundary handling methods
    
    The enhanced algorithm offers superior performance for complex optimization problems,
    particularly for multi-modal, non-convex solution spaces typical in vibration absorber design.
    """
    # Signals: final_results, best_particle, parameter_names, best_fitness, convergence_data
    finished = pyqtSignal(dict, list, list, float)  # Signal emitted when optimization completes
    error = pyqtSignal(str)  # Signal emitted when an error occurs
    update = pyqtSignal(str)  # Signal for progress updates during optimization
    convergence_signal = pyqtSignal(list, list)  # Signal for sending convergence data (iterations, best fitness values)

    def __init__(self, 
                 main_params,
                 target_values_dict,
                 weights_dict,
                 omega_start,
                 omega_end,
                 omega_points,
                 pso_swarm_size=40,
                 pso_num_iterations=100,
                 pso_w=0.729,      # Inertia weight (default: constriction factor value)
                 pso_w_damping=1.0, # Inertia weight damping ratio
                 pso_c1=1.49445,   # Cognitive coefficient (default: constriction factor recommended)
                 pso_c2=1.49445,   # Social coefficient (default: constriction factor recommended)
                 pso_tol=1e-6,     # Tolerance for convergence
                 pso_parameter_data=None,  # List of tuples: (name, low, high, fixed)
                 alpha=0.01,       # Sparsity penalty factor
                 adaptive_params=True,  # Enable adaptive parameters
                 topology=TopologyType.GLOBAL,  # Neighborhood topology
                 mutation_rate=0.1,  # Mutation rate for diversity
                 max_velocity_factor=0.1,  # Maximum velocity as fraction of range
                 stagnation_limit=10,  # Iterations before stagnation is detected
                 boundary_handling="absorbing",  # Method for handling boundary violations
                 early_stopping=True,  # Enable early stopping based on progress
                 early_stopping_iters=15,  # Iterations to check for early stopping
                 early_stopping_tol=1e-5,  # Improvement tolerance for early stopping
                 diversity_threshold=0.01,  # Minimum diversity threshold
                 quasi_random_init=True):  # Use quasi-random initialization
        """
        Initialize the enhanced PSO optimization worker with advanced features.
        
        Parameters:
        -----------
        main_params : tuple
            Main system parameters (masses, stiffnesses, damping coefficients)
        target_values_dict : dict
            Dictionary of target frequency response values for each mass
        weights_dict : dict
            Dictionary of weights for each target value and mass
        omega_start : float
            Start frequency for analysis (rad/s)
        omega_end : float
            End frequency for analysis (rad/s)
        omega_points : int
            Number of frequency points for analysis
        pso_swarm_size : int
            Number of particles in the swarm (default: 40)
        pso_num_iterations : int
            Maximum number of iterations for the PSO algorithm (default: 100)
        pso_w : float
            Initial inertia weight - controls influence of previous velocity (default: 0.729)
        pso_w_damping : float
            Damping ratio for inertia weight (default: 1.0 - no damping)
        pso_c1 : float
            Initial cognitive coefficient - controls attraction to particle's best position (default: 1.49445)
        pso_c2 : float
            Initial social coefficient - controls attraction to swarm's best position (default: 1.49445)
        pso_tol : float
            Convergence tolerance - algorithm stops if fitness reaches this value (default: 1e-6)
        pso_parameter_data : list
            List of tuples (name, low_bound, high_bound, is_fixed) for each parameter
        alpha : float
            Sparsity penalty factor to favor simpler solutions (default: 0.01)
        adaptive_params : bool
            Enable adaptive parameter adjustment (default: True)
        topology : TopologyType
            Neighborhood topology for determining social influence (default: GLOBAL)
        mutation_rate : float
            Probability of mutation for each parameter (default: 0.1)
        max_velocity_factor : float
            Maximum velocity as fraction of parameter range (default: 0.1)
        stagnation_limit : int
            Number of iterations without improvement to trigger stagnation handling (default: 10)
        boundary_handling : str
            Method for handling boundary violations: "absorbing", "reflecting", "invisible" (default: "absorbing")
        early_stopping : bool
            Enable early stopping based on improvement rate (default: True)
        early_stopping_iters : int
            Number of iterations to monitor for early stopping (default: 15)
        early_stopping_tol : float
            Tolerance for improvement to continue optimization (default: 1e-5)
        diversity_threshold : float
            Minimum diversity threshold as fraction of parameter range (default: 0.01)
        quasi_random_init : bool
            Use quasi-random initialization (Sobol sequence) for better space coverage (default: True)
        """
        super().__init__()
        
        # Flag for signaling thread termination
        self._terminate_flag = False
        
        # Core parameters
        self.main_params = main_params
        self.target_values_dict = target_values_dict
        self.weights_dict = weights_dict
        self.omega_start = omega_start
        self.omega_end = omega_end
        self.omega_points = omega_points
        
        # PSO parameters
        self.pso_swarm_size = pso_swarm_size
        self.pso_num_iterations = pso_num_iterations
        self.pso_w_init = pso_w  # Store initial value for adaptive methods
        self.pso_w = pso_w
        self.pso_w_damping = pso_w_damping
        self.pso_w_min = 0.1  # Minimum value for inertia weight
        self.pso_w_max = 0.9  # Maximum value for inertia weight
        self.pso_c1_init = pso_c1  # Store initial value for adaptive methods
        self.pso_c1 = pso_c1
        self.pso_c2_init = pso_c2  # Store initial value for adaptive methods
        self.pso_c2 = pso_c2
        self.pso_tol = pso_tol
        self.pso_parameter_data = pso_parameter_data if pso_parameter_data else []
        self.alpha = alpha
        
        # Advanced PSO features
        self.adaptive_params = adaptive_params
        self.topology = topology
        self.mutation_rate = mutation_rate
        self.max_velocity_factor = max_velocity_factor
        self.stagnation_limit = stagnation_limit
        self.stagnation_counter = 0
        self.boundary_handling = boundary_handling
        self.early_stopping = early_stopping
        self.early_stopping_iters = early_stopping_iters
        self.early_stopping_tol = early_stopping_tol
        self.diversity_threshold = diversity_threshold
        self.quasi_random_init = quasi_random_init
        
        # Runtime data
        self.iteration_best_fitness = []
        self.iteration_avg_fitness = []
        self.iteration_diversity = []
        self.start_time = None
        self.neighborhoods = None
        self.last_improvement_iter = 0
        
        # Constriction factor calculation
        # Based on Clerc & Kennedy (2002) - provides guaranteed convergence when c1+c2>4
        phi = self.pso_c1 + self.pso_c2
        if phi > 4:
            self.constriction_factor = 2.0 / abs(2.0 - phi - math.sqrt(phi * phi - 4.0 * phi))
        else:
            self.constriction_factor = 1.0  # Default to no constriction if phi <= 4

    def adaptive_inertia_weight(self, iter_num, max_iter, best_fitness, avg_fitness, diversity):
        """
        Adaptively adjust inertia weight based on multiple factors.
        
        This implements several adaptive w strategies:
        1. Linear time-varying: w decreases linearly with iteration
        2. Nonlinear time-varying: w decreases non-linearly (faster initially)
        3. Fitness-based: w adapts based on global vs. average fitness
        4. Diversity-based: w increases when diversity is low
        
        The combined approach balances exploration and exploitation:
        - High w (≈0.9): Encourages global exploration early in the search
        - Low w (≈0.1): Encourages local exploitation later in the search
        - Increases w when diversity drops to avoid premature convergence
        
        Scientific basis:
        ----------------
        - Shi & Eberhart (1998): Linear decreasing inertia weight
        - Chatterjee & Siarry (2006): Nonlinear decreasing inertia weight
        - Arumugam & Rao (2008): Adaptive inertia weight based on success
        
        Parameters:
        -----------
        iter_num : int
            Current iteration number
        max_iter : int
            Maximum number of iterations
        best_fitness : float
            Current best fitness value
        avg_fitness : float
            Average fitness of current swarm
        diversity : float
            Current swarm diversity measure
            
        Returns:
        --------
        float
            Updated inertia weight value
        """
        # Safeguard against division by zero
        if max_iter <= 1:
            return self.pso_w
        
        # Base linear time-varying component (Shi & Eberhart, 1998)
        w_linear = self.pso_w_max - (self.pso_w_max - self.pso_w_min) * (iter_num / (max_iter - 1))
        
        # Nonlinear time-varying component (Chatterjee & Siarry, 2006)
        # Decreases faster in early iterations, slower in later iterations
        n = 1.2  # Nonlinearity factor
        w_nonlinear = self.pso_w_max - (self.pso_w_max - self.pso_w_min) * (iter_num / (max_iter - 1))**n
        
        # Fitness-based adaptive component
        if avg_fitness == 0 or best_fitness == 0:
            w_fitness = 0
        else:
            # Increases w when best_fitness is close to avg_fitness (swarm converging)
            ratio = best_fitness / avg_fitness if avg_fitness > best_fitness else avg_fitness / best_fitness
            w_fitness = 0.5 * (1 - ratio)  # 0 when identical, up to 0.5 when very different
        
        # Diversity-based component
        # Increases w when diversity is low to encourage exploration
        w_diversity = self.pso_w_min + (1 - min(1, diversity / self.diversity_threshold)) * (self.pso_w_max - self.pso_w_min) * 0.5
        
        # Combine different adaptive components with weights
        w = 0.4 * w_linear + 0.3 * w_nonlinear + 0.1 * w_fitness + 0.2 * w_diversity
        
        # Apply bounds
        w = max(self.pso_w_min, min(self.pso_w_max, w))
        
        # Apply damping ratio
        w = self.pso_w * self.pso_w_damping + w * (1 - self.pso_w_damping)
        
        return w

    def adaptive_acceleration_coefficients(self, iter_num, max_iter, diversity):
        """
        Adaptively adjust cognitive (c1) and social (c2) coefficients.
        
        Implements time-varying acceleration coefficients:
        - c1 decreases over time: Reduces cognitive component (personal best attraction)
        - c2 increases over time: Increases social component (global best attraction)
        - Diversity-based adjustment to maintain swarm diversity
        
        This balances exploration early (high c1, low c2) and 
        exploitation later (low c1, high c2) in the optimization process.
        
        Scientific basis:
        ----------------
        - Ratnaweera et al. (2004): Time-varying acceleration coefficients
        - Zhan et al. (2009): Adaptive PSO with dynamic parameters
        
        Parameters:
        -----------
        iter_num : int
            Current iteration number
        max_iter : int
            Maximum number of iterations
        diversity : float
            Current swarm diversity measure
            
        Returns:
        --------
        tuple
            Updated (c1, c2) acceleration coefficients
        """
        # Safeguard against division by zero
        if max_iter <= 1:
            return self.pso_c1, self.pso_c2
        
        # Time-varying adjustment (Ratnaweera et al., 2004)
        # c1 decreases from 2.5 to 0.5, c2 increases from 0.5 to 2.5
        c1_min, c1_max = 0.5, 2.5
        c2_min, c2_max = 0.5, 2.5
        
        # Linear time-varying component
        progress = iter_num / (max_iter - 1)
        c1_time = c1_max - (c1_max - c1_min) * progress
        c2_time = c2_min + (c2_max - c2_min) * progress
        
        # Diversity-based adjustment
        # If diversity is low, increase c1 and decrease c2 to encourage exploration
        if diversity < self.diversity_threshold:
            diversity_factor = 1 - (diversity / self.diversity_threshold)
            c1_diversity = c1_min + (c1_max - c1_min) * diversity_factor * 0.5
            c2_diversity = c2_max - (c2_max - c2_min) * diversity_factor * 0.5
        else:
            c1_diversity = c1_time
            c2_diversity = c2_time
        
        # Combine components with weights
        c1 = 0.7 * c1_time + 0.3 * c1_diversity
        c2 = 0.7 * c2_time + 0.3 * c2_diversity
        
        # Ensure c1 + c2 is within stable range (typically <= 4)
        if c1 + c2 > 4:
            sum_c = c1 + c2
            c1 = c1 * 4 / sum_c
            c2 = c2 * 4 / sum_c
        
        return c1, c2

    def calculate_diversity(self, swarm, parameter_bounds):
        """
        Calculate swarm diversity as a measure of particle dispersion.
        
        Diversity measures how spread out the particles are in the search space.
        Low diversity indicates particles are clustered (possible premature convergence),
        while high diversity indicates good exploration of the search space.
        
        The method uses normalized Euclidean distances between particles and the swarm center,
        averaged over all particles and all dimensions, and scaled by parameter ranges.
        
        Parameters:
        -----------
        swarm : list
            List of particle dictionaries
        parameter_bounds : list
            List of (min, max) tuples for each parameter
            
        Returns:
        --------
        float
            Diversity value between 0 (no diversity) and 1 (maximum diversity)
        """
        if not swarm:
            return 0
        
        num_particles = len(swarm)
        if num_particles <= 1:
            return 0
        
        num_dimensions = len(swarm[0]['position'])
        
        # Calculate swarm center (centroid)
        centroid = [0.0] * num_dimensions
        for particle in swarm:
            for j in range(num_dimensions):
                centroid[j] += particle['position'][j]
        
        for j in range(num_dimensions):
            centroid[j] /= num_particles
        
        # Calculate average normalized distance to centroid
        total_distance = 0
        for particle in swarm:
            distance_sq = 0
            for j in range(num_dimensions):
                # Normalize by parameter range to handle different scales
                param_range = parameter_bounds[j][1] - parameter_bounds[j][0]
                if param_range > 0:  # Avoid division by zero
                    normalized_diff = (particle['position'][j] - centroid[j]) / param_range
                    distance_sq += normalized_diff ** 2
            
            total_distance += math.sqrt(distance_sq / num_dimensions)
        
        # Normalize by theoretical maximum diversity (assuming uniform distribution)
        # and scale by dimensionality
        diversity = total_distance / (num_particles * math.sqrt(num_dimensions / 12))
        
        return min(1.0, diversity)  # Cap at 1.0 for meaningful interpretation

    def create_neighborhoods(self, swarm_size, topology_type):
        """
        Create particle neighborhoods based on the specified topology.
        
        Different topologies balance information flow and diversity:
        - GLOBAL: All particles connected to all others (fully connected)
        - RING: Each particle connected to two adjacent particles
        - VON_NEUMANN: Grid-like connections (each particle has 4 neighbors)
        - RANDOM: Random connections, different for each particle
        
        Local topologies (RING, VON_NEUMANN) typically converge slower but are
        less likely to get trapped in local optima compared to GLOBAL topology.
        
        Parameters:
        -----------
        swarm_size : int
            Number of particles in the swarm
        topology_type : TopologyType
            Type of neighborhood topology to create
            
        Returns:
        --------
        list
            List of neighborhood lists for each particle
        """
        neighborhoods = []
        
        if topology_type == TopologyType.GLOBAL:
            # Global topology: each particle is connected to all others
            for i in range(swarm_size):
                neighbors = list(range(swarm_size))
                neighborhoods.append(neighbors)
                
        elif topology_type == TopologyType.RING:
            # Ring topology: each particle is connected to its immediate neighbors
            for i in range(swarm_size):
                left = (i - 1) % swarm_size
                right = (i + 1) % swarm_size
                neighborhoods.append([left, i, right])
                
        elif topology_type == TopologyType.VON_NEUMANN:
            # Von Neumann topology: grid-like arrangement with 4 neighbors per particle
            # Calculate grid dimensions for closest square-like grid
            side = int(math.ceil(math.sqrt(swarm_size)))
            for i in range(swarm_size):
                row, col = i // side, i % side
                neighbors = [i]  # Include self
                
                # Add north, east, south, west neighbors if they exist
                north = ((row - 1) % side) * side + col
                south = ((row + 1) % side) * side + col
                west = row * side + ((col - 1) % side)
                east = row * side + ((col + 1) % side)
                
                if north < swarm_size:
                    neighbors.append(north)
                if south < swarm_size:
                    neighbors.append(south)
                if west < swarm_size:
                    neighbors.append(west)
                if east < swarm_size:
                    neighbors.append(east)
                
                neighborhoods.append(neighbors)
                
        elif topology_type == TopologyType.RANDOM:
            # Random topology: each particle is connected to k random others
            k = min(5, swarm_size - 1)  # Number of random connections
            for i in range(swarm_size):
                # Always include self
                neighbors = [i]
                # Add k random distinct neighbors
                potential_neighbors = list(range(swarm_size))
                potential_neighbors.remove(i)  # Remove self
                neighbors.extend(random.sample(potential_neighbors, k))
                neighborhoods.append(neighbors)
        
        return neighborhoods

    def update_neighborhoods(self, iteration):
        """
        Update the particle neighborhoods periodically.
        
        For dynamic topologies (RANDOM), neighborhoods are regenerated periodically
        to maintain diversity and enhance exploration.
        
        Parameters:
        -----------
        iteration : int
            Current iteration number
        """
        # Only update for RANDOM topology, and only every 5 iterations
        if self.topology == TopologyType.RANDOM and iteration % 5 == 0:
            self.neighborhoods = self.create_neighborhoods(self.pso_swarm_size, self.topology)

    def handle_boundary_violation(self, position, velocity, dim, low, high):
        """
        Handle boundary violations using different strategies.
        
        Implements three boundary handling methods:
        1. Absorbing: Position is set to boundary, velocity zeroed
        2. Reflecting: Position bounces off boundary, velocity inverted
        3. Invisible: Particle moves freely outside but fitness penalized
        
        Scientific basis:
        ----------------
        - Robinson & Rahmat-Samii (2004): Empirical study of boundary handling methods
        - Helwig et al. (2013): Boundary handling in PSO and its impact
        
        Parameters:
        -----------
        position : float
            Current position value for the dimension
        velocity : float
            Current velocity value for the dimension
        dim : int
            Dimension index
        low : float
            Lower bound for the dimension
        high : float
            Upper bound for the dimension
            
        Returns:
        --------
        tuple
            Updated (position, velocity)
        """
        if low <= position <= high:
            return position, velocity
        
        if self.boundary_handling == "absorbing":
            # Set position to boundary and zero out velocity
            if position < low:
                return low, 0
            else:
                return high, 0
                
        elif self.boundary_handling == "reflecting":
            # Bounce off boundary and invert velocity
            if position < low:
                overshoot = low - position
                return low + overshoot, -velocity * 0.8  # Dampen velocity
            else:
                overshoot = position - high
                return high - overshoot, -velocity * 0.8  # Dampen velocity
                
        elif self.boundary_handling == "invisible":
            # Allow particles to move outside boundaries
            # (fitness function will penalize invalid positions)
            return position, velocity
            
        # Default to absorbing if invalid method
        if position < low:
            return low, 0
        else:
            return high, 0

    def apply_mutation(self, position, parameter_bounds, fixed_parameters):
        """
        Apply mutation to particle position to maintain diversity.
        
        Mutation randomly perturbs some dimensions of the position vector
        to help the swarm escape local optima and explore the search space.
        
        Scientific basis:
        ----------------
        - Kennedy & Eberhart (2001): Hybrid PSO with mutation
        - Ratnaweera et al. (2004): Self-organizing hierarchical PSO
        
        Parameters:
        -----------
        position : list
            Current position vector
        parameter_bounds : list
            List of (min, max) tuples for each parameter
        fixed_parameters : dict
            Dictionary of fixed parameter indices
            
        Returns:
        --------
        list
            Mutated position vector
        """
        mutated_position = position[:]
        
        for j in range(len(position)):
            # Skip fixed parameters
            if j in fixed_parameters:
                continue
                
            # Apply mutation with probability mutation_rate
            if random.random() < self.mutation_rate:
                low, high = parameter_bounds[j]
                # Gaussian mutation centered on current value
                # with standard deviation proportional to parameter range
                sigma = (high - low) * 0.1
                mutation = random.gauss(0, sigma)
                mutated_position[j] += mutation
                
                # Ensure bounds are respected
                mutated_position[j] = max(low, min(high, mutated_position[j]))
                
        return mutated_position

    def quasi_random_initialize(self, num_params, parameter_bounds, fixed_parameters, num_particles):
        """
        Initialize swarm using quasi-random Sobol sequence for better coverage.
        
        Quasi-random (low-discrepancy) sequences provide more uniform coverage
        of the search space compared to purely random initialization, improving
        the probability of finding good initial positions.
        
        Scientific basis:
        ----------------
        - Schutte et al. (2004): Improved initialization for PSO
        - Brits et al. (2007): Low-discrepancy sequences in PSO
        
        Parameters:
        -----------
        num_params : int
            Number of parameters/dimensions
        parameter_bounds : list
            List of (min, max) tuples for each parameter
        fixed_parameters : dict
            Dictionary of fixed parameter indices
        num_particles : int
            Number of particles to initialize
            
        Returns:
        --------
        list
            List of initialized position vectors
        """
        # Create Sobol sequence generator
        sampler = qmc.Sobol(d=num_params, scramble=True)
        
        # Generate samples in [0, 1] range
        samples = sampler.random(n=num_particles)
        
        # Scale to parameter bounds
        positions = []
        for i in range(num_particles):
            position = []
            for j in range(num_params):
                if j in fixed_parameters:
                    pos = fixed_parameters[j]
                else:
                    low, high = parameter_bounds[j]
                    pos = low + samples[i, j] * (high - low)
                position.append(pos)
            positions.append(position)
            
        return positions

    def run(self):
        """
        Main method executed when the thread starts.
        Implements the enhanced PSO algorithm with adaptive features.
        
        The advanced PSO algorithm follows these steps:
        1. Initialize swarm with quasi-random positions (better space coverage)
        2. Set up neighborhood topology for information sharing
        3. For each iteration:
           a. Update adaptive parameters (inertia, acceleration coefficients)
           b. Apply velocity constriction and clamping
           c. Update each particle's velocity and position with boundary handling
           d. Apply diversity maintenance through mutation
           e. Re-evaluate fitness
           f. Update personal and neighborhood best positions
           g. Check for stagnation and apply recovery measures
           h. Calculate and monitor diversity
           i. Check early stopping criteria
        4. Return the best solution found
        
        Scientific enhancements:
        -----------------------
        - Quasi-random initialization for better search space coverage
        - Adaptive parameter adjustment for balancing exploration/exploitation
        - Constriction factor for convergence stability
        - Multiple neighborhood topologies for information flow control
        - Velocity clamping to prevent explosion
        - Diversity maintenance through mutation
        - Multiple boundary handling strategies
        - Early stopping based on improvement rate
        - Stagnation detection and recovery
        """
        try:
            self.start_time = time.time()
            self.update.emit("[INFO] Starting enhanced PSO optimization...")
            
            # Extract parameter names, bounds, and fixed parameters
            parameter_names = []
            parameter_bounds = []
            fixed_parameters = {}  # key: index, value: fixed value
            for idx, (name, low, high, fixed) in enumerate(self.pso_parameter_data):
                parameter_names.append(name)
                if fixed:
                    parameter_bounds.append((low, low))  # For fixed parameters, set identical bounds
                    fixed_parameters[idx] = low
                else:
                    parameter_bounds.append((low, high))
            num_params = len(parameter_bounds)

            # Create neighborhood structure based on topology
            self.neighborhoods = self.create_neighborhoods(self.pso_swarm_size, self.topology)
            
            # Initialize the swarm with random or quasi-random positions
            swarm = []
            
            if self.quasi_random_init:
                # Quasi-random initialization for positions (better space coverage)
                positions = self.quasi_random_initialize(
                    num_params, parameter_bounds, fixed_parameters, self.pso_swarm_size
                )
            else:
                # Standard random initialization
                positions = []
            for i in range(self.pso_swarm_size):
                position = []
                for j in range(num_params):
                    low, high = parameter_bounds[j]
                    if j in fixed_parameters:
                        pos = fixed_parameters[j]
                    else:
                        pos = random.uniform(low, high)
                    position.append(pos)
                    positions.append(position)
            
            # Calculate max velocities for each dimension (for velocity clamping)
            max_velocities = []
            for j in range(num_params):
                low, high = parameter_bounds[j]
                max_velocities.append((high - low) * self.max_velocity_factor)
            
            # Initialize each particle
            for i in range(self.pso_swarm_size):
                position = positions[i]
                velocity = []
                
                # Initialize velocity
                for j in range(num_params):
                    if j in fixed_parameters:
                        vel = 0
                    else:
                        # Initialize velocity with a smaller range for stability
                        max_vel = max_velocities[j]
                        vel = random.uniform(-max_vel/2, max_vel/2)
                    velocity.append(vel)
                
                # Evaluate initial fitness
                fitness = self.evaluate_particle(position, parameter_bounds)
                
                # Create particle
                particle = {
                    'position': position,                 # Current position vector
                    'velocity': velocity,                 # Current velocity vector
                    'best_position': position[:],         # Best position found by this particle
                    'best_fitness': fitness,              # Best fitness achieved by this particle
                    'current_fitness': fitness,           # Current fitness (for tracking improvement)
                    'stagnation_counter': 0,              # Counter for personal stagnation
                    'neighborhood_best_position': None,   # Best position in the neighborhood
                    'neighborhood_best_fitness': float('inf')  # Best fitness in the neighborhood
                }
                swarm.append(particle)

            # Calculate initial diversity
            diversity = self.calculate_diversity(swarm, parameter_bounds)
            self.iteration_diversity.append(diversity)
            
            # Determine the global best particle in the initial swarm
            global_best_particle_idx = 0
            global_best_fitness = float('inf')
            
            for i, particle in enumerate(swarm):
                if particle['best_fitness'] < global_best_fitness:
                    global_best_fitness = particle['best_fitness']
                    global_best_particle_idx = i
            
            # Set initial neighborhood bests based on topology
            for i, particle in enumerate(swarm):
                neighborhood_best_fitness = float('inf')
                neighborhood_best_idx = i
                
                for neighbor_idx in self.neighborhoods[i]:
                    neighbor = swarm[neighbor_idx]
                    if neighbor['best_fitness'] < neighborhood_best_fitness:
                        neighborhood_best_fitness = neighbor['best_fitness']
                        neighborhood_best_idx = neighbor_idx
                
                particle['neighborhood_best_position'] = swarm[neighborhood_best_idx]['best_position'][:]
                particle['neighborhood_best_fitness'] = neighborhood_best_fitness

            # PSO main loop: update each particle's velocity and position.
            for iteration in range(1, self.pso_num_iterations + 1):
                # Check if termination has been requested
                if self._terminate_flag:
                    self.update.emit("[INFO] PSO optimization terminated by user")
                    break
                    
                self.update.emit(f"-- Iteration {iteration} --")
                
                # Update adaptive parameters if enabled
                if self.adaptive_params:
                    # Calculate current average fitness
                    avg_fitness = sum(p['current_fitness'] for p in swarm) / len(swarm)
                    
                    # Calculate current diversity
                    diversity = self.calculate_diversity(swarm, parameter_bounds)
                    self.iteration_diversity.append(diversity)
                    
                    # Update inertia weight
                    self.pso_w = self.adaptive_inertia_weight(
                        iteration, self.pso_num_iterations,
                        swarm[global_best_particle_idx]['best_fitness'],
                        avg_fitness, diversity
                    )
                    
                    # Update acceleration coefficients
                    self.pso_c1, self.pso_c2 = self.adaptive_acceleration_coefficients(
                        iteration, self.pso_num_iterations, diversity
                    )
                    
                    # Update neighborhoods for dynamic topologies
                    self.update_neighborhoods(iteration)
                
                # Update all particles
                current_fitnesses = []
                
                # Check termination flag again before updating particles
                if self._terminate_flag:
                    break
                    
                for i, particle in enumerate(swarm):
                    # Update velocities and positions for each dimension
                    for j in range(num_params):
                        # For fixed parameters, ensure they remain constant
                        if j in fixed_parameters:
                            particle['velocity'][j] = 0
                            particle['position'][j] = fixed_parameters[j]
                        else:
                            # Generate random weights for cognitive and social components
                            r1 = random.random()  # Random factor for cognitive component
                            r2 = random.random()  # Random factor for social component
                            
                            # Determine social attractor (neighborhood best or global best)
                            if self.topology != TopologyType.GLOBAL:
                                # Use neighborhood best for local topologies
                                social_best_position = particle['neighborhood_best_position'][j]
                            else:
                                # Use global best for global topology
                                social_best_position = swarm[global_best_particle_idx]['best_position'][j]
                            
                            # Calculate cognitive component: attraction to particle's best position
                            cognitive = self.pso_c1 * r1 * (particle['best_position'][j] - particle['position'][j])
                            
                            # Calculate social component: attraction to neighborhood/global best position
                            social = self.pso_c2 * r2 * (social_best_position - particle['position'][j])
                            
                            # Calculate raw velocity update
                            raw_velocity = self.pso_w * particle['velocity'][j] + cognitive + social
                            
                            # Apply constriction factor for stability if needed
                            if self.constriction_factor != 1.0:
                                raw_velocity *= self.constriction_factor
                            
                            # Apply velocity clamping to prevent explosion
                            max_vel = max_velocities[j]
                            particle['velocity'][j] = max(-max_vel, min(max_vel, raw_velocity))
                            
                            # Update position
                            particle['position'][j] += particle['velocity'][j]
                            
                            # Handle boundary violations
                            particle['position'][j], particle['velocity'][j] = self.handle_boundary_violation(
                                particle['position'][j], particle['velocity'][j], j, parameter_bounds[j][0], parameter_bounds[j][1]
                            )
                    
                    # Apply mutation for diversity maintenance if needed
                    if diversity < self.diversity_threshold:
                        mutated_position = self.apply_mutation(
                            particle['position'], parameter_bounds, fixed_parameters
                        )
                        particle['position'] = mutated_position
                    
                    # Evaluate new position
                    fitness = self.evaluate_particle(particle['position'], parameter_bounds)
                    particle['current_fitness'] = fitness
                    current_fitnesses.append(fitness)
                    
                    # Update personal best if improved
                    if fitness < particle['best_fitness']:
                        particle['best_position'] = particle['position'][:]
                        particle['best_fitness'] = fitness
                        particle['stagnation_counter'] = 0  # Reset stagnation counter
                    else:
                        # Increment stagnation counter if no improvement
                        particle['stagnation_counter'] += 1
                    
                    # Update global best if improved
                    if fitness < swarm[global_best_particle_idx]['best_fitness']:
                        global_best_particle_idx = i
                        self.last_improvement_iter = iteration
                
                # Update neighborhood bests
                if self.topology != TopologyType.GLOBAL:
                    for i, particle in enumerate(swarm):
                        for neighbor_idx in self.neighborhoods[i]:
                            neighbor = swarm[neighbor_idx]
                            if neighbor['best_fitness'] < particle['neighborhood_best_fitness']:
                                particle['neighborhood_best_fitness'] = neighbor['best_fitness']
                                particle['neighborhood_best_position'] = neighbor['best_position'][:]
                
                # Calculate and store statistics
                avg_fitness = sum(current_fitnesses) / len(current_fitnesses)
                self.iteration_best_fitness.append(swarm[global_best_particle_idx]['best_fitness'])
                self.iteration_avg_fitness.append(avg_fitness)
                
                # Handle stagnation: Reinitialize particles that haven't improved
                for i, particle in enumerate(swarm):
                    if particle['stagnation_counter'] >= self.stagnation_limit:
                        # Reinitialize position (except fixed parameters)
                        for j in range(num_params):
                            if j not in fixed_parameters:
                                low, high = parameter_bounds[j]
                                # Reinitialize around global best with some noise
                                center = swarm[global_best_particle_idx]['position'][j]
                                radius = (high - low) * 0.1
                                particle['position'][j] = random.uniform(
                                    max(low, center - radius),
                                    min(high, center + radius)
                                )
                                # Reset velocity
                                particle['velocity'][j] = random.uniform(-max_velocities[j]/2, max_velocities[j]/2)
                        
                        # Reset stagnation counter
                        particle['stagnation_counter'] = 0
                        
                        # Evaluate new position
                        fitness = self.evaluate_particle(particle['position'], parameter_bounds)
                        particle['current_fitness'] = fitness
                
                # Emit progress
                if iteration % 5 == 0 or iteration == 1:
                    self.update.emit(
                        f"  Iteration {iteration}: Best fitness = {swarm[global_best_particle_idx]['best_fitness']:.6f}, "
                        f"Avg fitness = {avg_fitness:.6f}, Diversity = {diversity:.4f}, w = {self.pso_w:.4f}"
                    )
                    
                    # Emit convergence data every 5 iterations
                    if len(self.iteration_best_fitness) >= 5:
                        self.convergence_signal.emit(
                            list(range(1, iteration + 1)),
                            self.iteration_best_fitness
                        )
                
                # Check for convergence
                if swarm[global_best_particle_idx]['best_fitness'] <= self.pso_tol:
                    self.update.emit(f"[INFO] Convergence reached at iteration {iteration} (fitness below tolerance)")
                    break

                # Check for early stopping if enabled
                if self.early_stopping and iteration > self.early_stopping_iters:
                    recent_improvement = (
                        self.iteration_best_fitness[iteration - self.early_stopping_iters - 1] -
                        self.iteration_best_fitness[iteration - 1]
                    )
                    if recent_improvement < self.early_stopping_tol:
                        self.update.emit(
                            f"[INFO] Early stopping at iteration {iteration} "
                            f"(improvement {recent_improvement:.8f} < {self.early_stopping_tol})"
                        )
                        break
            
            # Extract best solution
            best_particle = swarm[global_best_particle_idx]['best_position']
            best_fitness = swarm[global_best_particle_idx]['best_fitness']
            
            # Calculate elapsed time
            elapsed_time = time.time() - self.start_time
            
            # Print final results summary
            self.update.emit(f"[INFO] Optimization completed in {elapsed_time:.2f} seconds")
            self.update.emit(f"[INFO] Best fitness achieved: {best_fitness:.8f}")
            self.update.emit(f"[INFO] Parameters found:")
            for i, name in enumerate(parameter_names):
                self.update.emit(f"  {name}: {best_particle[i]:.6f}")

            # Run the final FRF evaluation using the best found parameters
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
                
                # Add optimization metadata to results
                final_results['optimization_metadata'] = {
                    'iterations': len(self.iteration_best_fitness),
                    'elapsed_time': elapsed_time,
                    'final_diversity': self.iteration_diversity[-1] if self.iteration_diversity else 0,
                    'convergence_iterations': self.iteration_best_fitness,
                    'convergence_diversity': self.iteration_diversity
                }
                
            except Exception as e:
                final_results = {"Error": str(e)}

            # Emit results to be processed by the main thread
            self.finished.emit(final_results, best_particle, parameter_names, best_fitness)

        except Exception as e:
            self.error.emit(str(e))

    def evaluate_particle(self, position, parameter_bounds):
        """
        Evaluate the fitness of a particle based on its position.
        
        This enhanced fitness function calculates how well the current DVA parameters
        achieve the desired frequency response, with additional features:
        1. Multi-objective weighting for handling conflicting objectives
        2. Parameter sparsity penalty for simpler designs
        3. Boundary violation penalties for constrained optimization
        4. Adaptive penalty weights based on constraint satisfaction
        
        Scientific explanation:
        ----------------------
        In vibration control, the "singular response" refers to the maximum
        amplitude of the frequency response function across all frequencies.
        A value of 1 indicates an optimal vibration absorber that effectively
        flattens the response.
        
        The sparsity penalty encourages simpler solutions by penalizing non-zero
        parameter values, improving robustness and reducing physical complexity.
        This is equivalent to L1 regularization in machine learning models.
        
        Parameters:
        -----------
        position : list
            The position vector of the particle (DVA parameters)
        parameter_bounds : list
            List of (min, max) tuples for each parameter
            
        Returns:
        --------
        float
            The fitness value (lower is better)
        """
        try:
            # Check for boundary violations in invisible mode
            if self.boundary_handling == "invisible":
                # Apply constraint violation penalty
                penalty = 0
                for j, (low, high) in enumerate(parameter_bounds):
                    if position[j] < low:
                        # Quadratic penalty for going below minimum
                        rel_violation = (low - position[j]) / (high - low) if high > low else 0
                        penalty += rel_violation ** 2
                    elif position[j] > high:
                        # Quadratic penalty for going above maximum
                        rel_violation = (position[j] - high) / (high - low) if high > low else 0
                        penalty += rel_violation ** 2
                
                # If serious constraint violation, return large penalty
                if penalty > 1.0:
                    return 1e6 * penalty
            
            # Call FRF function to get frequency response
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
            
            # Extract the singular response (key performance metric)
            singular_response = results.get('singular_response', None)
            
            # Handle invalid responses by returning a large penalty value
            if singular_response is None or not np.isfinite(singular_response):
                return 1e6
            
            # Get additional performance metrics if available
            peak_amplitudes = results.get('peak_amplitudes', [])
            average_amplitude = results.get('average_amplitude', 0)
            
            # Calculate primary objective (minimize difference from ideal response)
            primary_objective = abs(singular_response - 1)
            
            # Calculate sparsity penalty (L1 regularization)
            # Encourages solutions with fewer/smaller parameters
            sparsity_penalty = self.alpha * sum(abs(x) for x in position)
            
            # Calculate peak penalty (optional)
            # Penalize solutions with high resonance peaks
            peak_penalty = 0
            if peak_amplitudes:
                # Penalize max peak amplitude and number of peaks
                peak_penalty = 0.1 * (max(peak_amplitudes) - 1) + 0.05 * len(peak_amplitudes)
                peak_penalty = max(0, peak_penalty)  # Only apply if positive
            
            # Calculate smoothness penalty (optional)
            # Penalize solutions with uneven response
            smoothness_penalty = 0
            if 'frequency_response_variance' in results:
                smoothness_penalty = 0.1 * results['frequency_response_variance']
            
            # Combine all penalty terms for final fitness
            fitness = (
                primary_objective +   # Main objective
                sparsity_penalty +    # Sparsity penalty
                peak_penalty +        # Peak penalty
                smoothness_penalty    # Smoothness penalty
            )
            
            return fitness
            
        except Exception as e:
            # Return a large penalty value if evaluation fails
            return 1e6

    def terminate(self):
        """
        Signal the thread to terminate gracefully.
        """
        self._terminate_flag = True

    def is_terminated(self):
        """
        Check if the thread has been signaled to terminate.
        """
        return self._terminate_flag
