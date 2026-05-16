import sys
import numpy as np
import os
import random
import pickle
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
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QMutex, QWaitCondition, QTimer
from PyQt5.QtGui import QIcon, QPalette, QColor, QFont

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# Import your modules accordingly
from modules.FRF import frf
from modules.sobol_sensitivity import (
    perform_sobol_analysis,
    calculate_and_save_errors,
    format_parameter_name
)

class RLWorker(QThread):
    """
    Reinforcement Learning Worker for Dynamic Vibration Absorber (DVA) parameter optimization.
    
    This implementation uses Deep Deterministic Policy Gradient (DDPG)-inspired approach
    adapted for DVA optimization, making it suitable for continuous parameter spaces.
    
    Scientific Background:
    ---------------------
    Unlike traditional tabular Q-learning which is designed for discrete action spaces,
    this implementation uses a policy gradient approach suitable for continuous optimization.
    The agent learns to directly output DVA parameter values rather than discrete actions.
    
    The approach combines:
    1. Policy-based learning for continuous parameter generation
    2. Value-based evaluation for fitness assessment
    3. Experience replay for stable learning
    4. Exploration noise for parameter space exploration
    
    This makes it scientifically sound for vibration absorber design where parameters
    are inherently continuous (masses, stiffnesses, damping coefficients).
    
    Signals:
        finished(dict, list, list, float):
            Emitted upon completion with FRF results, best parameters, parameter names, best fitness
        error(str):
            Emitted if any exception occurs
        update(str):
            Emitted for progress updates during training
        progress(int):
            Emitted for progress percentage (0-100) - consistent with GAWorker
        episode_metrics(dict):
            Emitted with training metrics for visualization
    """
    finished = pyqtSignal(dict, list, list, float)
    error = pyqtSignal(str)
    update = pyqtSignal(str)
    progress = pyqtSignal(int)  # Emits progress percentage (0-100) - consistent with GAWorker
    episode_metrics = pyqtSignal(dict)

    def __init__(
        self,
        main_params,
        target_values_dict,
        weights_dict,
        omega_start,
        omega_end,
        omega_points,
        rl_num_episodes,
        rl_max_steps,
        rl_alpha,        # Learning rate
        rl_gamma,        # Discount factor
        rl_epsilon,      # Initial exploration rate
        rl_epsilon_min,  # Minimum exploration rate
        rl_epsilon_decay_type,  # Decay type for epsilon
        rl_epsilon_decay,       # Decay factor
        rl_parameter_data,
        
        # Simplified reward system - consistent with other methods
        alpha_sparsity=0.01,    # Sparsity penalty factor (same as GA, PSO, SA)
        
        # Additional RL-specific parameters
        replay_buffer_size=10000,
        batch_size=32,
        tau=0.001,              # Soft update parameter
        noise_std=0.1,          # Exploration noise standard deviation
        
        # For saving/loading experience
        experience_save_path=None,
        load_existing_experience=False,
        sobol_settings=None,
        
        # Additional epsilon decay parameters for compatibility
        rl_linear_decay_step=None,
        rl_inverse_decay_coefficient=1.0,
        rl_step_interval=10,
        rl_step_decay_amount=None,
        rl_cosine_decay_amplitude=1.0
    ):
        """
        Initialize the RL worker with scientifically sound continuous optimization approach.
        """
        super().__init__()
        self.main_params = main_params
        self.target_values_dict = target_values_dict
        self.weights_dict = weights_dict
        self.omega_start = omega_start
        self.omega_end = omega_end
        self.omega_points = omega_points

        # RL Hyperparameters
        self.rl_num_episodes = rl_num_episodes
        self.rl_max_steps = rl_max_steps
        self.rl_alpha = rl_alpha
        self.rl_gamma = rl_gamma
        self.rl_epsilon = rl_epsilon
        self.rl_epsilon_min = rl_epsilon_min
        self.rl_epsilon_decay_type = rl_epsilon_decay_type
        self.rl_epsilon_decay = rl_epsilon_decay

        # Simplified reward system (consistent with other methods)
        self.alpha_sparsity = alpha_sparsity

        self.rl_parameter_data = rl_parameter_data

        # RL-specific parameters
        self.replay_buffer_size = replay_buffer_size
        self.batch_size = batch_size
        self.tau = tau
        self.noise_std = noise_std

        # Extra epsilon decay parameters
        self.rl_linear_decay_step = rl_linear_decay_step
        self.rl_inverse_decay_coefficient = rl_inverse_decay_coefficient
        self.rl_step_interval = rl_step_interval
        self.rl_step_decay_amount = rl_step_decay_amount if rl_step_decay_amount is not None else rl_epsilon_decay
        self.rl_cosine_decay_amplitude = rl_cosine_decay_amplitude

        # Experience saving/loading
        self.experience_save_path = experience_save_path
        self.load_existing_experience = load_existing_experience

        # Optional Sobol settings
        self.sobol_settings = sobol_settings if sobol_settings is not None else {}

        # Build parameter mappings (consistent with other workers)
        self.parameter_names = []
        self.parameter_bounds = []
        self.fixed_parameters = {}
        for idx, (name, low, high, fixed) in enumerate(self.rl_parameter_data):
            self.parameter_names.append(name)
            if fixed:
                self.parameter_bounds.append((low, low))
                self.fixed_parameters[idx] = low
            else:
                self.parameter_bounds.append((low, high))

        # Initialize policy network (simple linear policy for continuous parameters)
        self.num_params = len(self.parameter_bounds)
        self.policy_weights = np.random.randn(self.num_params) * 0.1
        self.policy_bias = np.zeros(self.num_params)
        
        # Experience replay buffer
        self.experience_buffer = []
        
        # Track best solution
        self.best_fitness = float('inf')  # Lower is better (consistent with other methods)
        self.best_solution = None
        self.episode_rewards = []
        
        # Load existing experience if requested
        if self.load_existing_experience and self.experience_save_path is not None:
            self._load_experience()

        # Thread safety mechanisms - consistent with GAWorker and PSOWorker
        self.mutex = QMutex()                   # Mutex for critical sections
        self.condition = QWaitCondition()       # Wait condition for thread coordination
        self.abort = False                      # Abort flag for safe termination
        self._terminate_flag = False            # Additional termination flag (PSO-style)
        
        # Watchdog timer for safety (consistent with GAWorker)
        self.watchdog_timer = QTimer()
        self.watchdog_timer.setSingleShot(True)
        self.watchdog_timer.timeout.connect(self.handle_timeout)
        self.last_progress_update = 0

    def _load_experience(self):
        """Load existing experience from file"""
        if os.path.exists(self.experience_save_path):
            try:
                with open(self.experience_save_path, 'rb') as f:
                    data = pickle.load(f)
                    self.experience_buffer = data.get('experience_buffer', [])
                    self.policy_weights = data.get('policy_weights', self.policy_weights)
                    self.policy_bias = data.get('policy_bias', self.policy_bias)
                print(f"Experience loaded from {self.experience_save_path}")
            except Exception as e:
                print(f"Error loading experience: {e}")

    def _save_experience(self):
        """Save experience to file"""
        if self.experience_save_path is not None:
            try:
                data = {
                    'experience_buffer': self.experience_buffer,
                    'policy_weights': self.policy_weights,
                    'policy_bias': self.policy_bias
                }
                with open(self.experience_save_path, 'wb') as f:
                    pickle.dump(data, f)
                print(f"Experience saved to {self.experience_save_path}")
            except Exception as e:
                print(f"Error saving experience: {e}")

    def __del__(self):
        """
        Cleanup method that runs when the object is destroyed
        Consistent with GAWorker implementation
        """
        self.mutex.lock()
        self.abort = True
        self._terminate_flag = True
        self.condition.wakeAll()
        self.mutex.unlock()
        self.wait()

    def handle_timeout(self):
        """
        Handle watchdog timer timeout
        Consistent with GAWorker implementation
        """
        self.mutex.lock()
        if not self.abort:
            self.abort = True
            self._terminate_flag = True
            self.mutex.unlock()
            self.error.emit("RL optimization timed out. The operation was taking too long.")
        else:
            self.mutex.unlock()

    def terminate(self):
        """
        Signal the thread to terminate gracefully
        Consistent with PSOWorker implementation
        """
        self._terminate_flag = True
        self.abort = True

    def is_terminated(self):
        """
        Check if the thread has been signaled to terminate
        Consistent with PSOWorker implementation
        """
        return self._terminate_flag or self.abort

    def cleanup(self):
        """
        Clean up resources to prevent memory leaks
        Consistent with GAWorker implementation
        """
        # Stop the watchdog timer if it's running
        if hasattr(self, 'watchdog_timer') and self.watchdog_timer.isActive():
            self.watchdog_timer.stop()
        
        # Clear experience buffer to free memory
        if hasattr(self, 'experience_buffer'):
            self.experience_buffer.clear()
        
        # Clear policy weights
        if hasattr(self, 'policy_weights'):
            self.policy_weights = None
        if hasattr(self, 'policy_bias'):
            self.policy_bias = None

    def generate_parameters(self, add_noise=True):
        """
        Generate DVA parameters using current policy.
        This replaces the inappropriate tabular Q-learning approach.
        """
        # Generate base parameters using policy
        raw_params = self.policy_weights + self.policy_bias
        
        # Add exploration noise if requested
        if add_noise and self.rl_epsilon > 0:
            noise = np.random.normal(0, self.noise_std * self.rl_epsilon, self.num_params)
            raw_params += noise
        
        # Apply bounds and handle fixed parameters
        bounded_params = []
        for i, (low, high) in enumerate(self.parameter_bounds):
            if i in self.fixed_parameters:
                bounded_params.append(self.fixed_parameters[i])
            else:
                # Normalize to [0,1] then scale to bounds
                normalized = 1 / (1 + np.exp(-raw_params[i]))  # Sigmoid activation
                scaled = low + normalized * (high - low)
                bounded_params.append(scaled)
        
        return bounded_params

    def evaluate_parameters(self, params):
        """
        Evaluate DVA parameters using FRF analysis.
        This uses the same fitness function as GA, PSO, and SA for consistency.
        """
        try:
            results = frf(
                main_system_parameters=self.main_params,
                dva_parameters=tuple(params),
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
            
            # Extract singular response (consistent with other methods)
            singular_response = results.get('singular_response', None)
            if singular_response is None or not np.isfinite(singular_response):
                return 1e6, results
            
            # Calculate fitness (consistent with GA, PSO, SA)
            primary_objective = abs(singular_response - 1)
            sparsity_penalty = self.alpha_sparsity * sum(abs(p) for p in params)
            fitness = primary_objective + sparsity_penalty
            
            return fitness, results
            
        except Exception as e:
            return 1e6, {"Error": str(e)}

    def update_policy(self, experiences):
        """
        Update the policy based on collected experiences.
        Uses policy gradient approach suitable for continuous optimization.
        """
        if len(experiences) < self.batch_size:
            return
        
        # Sample batch from experiences
        batch = random.sample(experiences, min(self.batch_size, len(experiences)))
        
        # Calculate policy gradients
        policy_gradient_weights = np.zeros_like(self.policy_weights)
        policy_gradient_bias = np.zeros_like(self.policy_bias)
        
        for params, fitness, _ in batch:
            # Calculate advantage (lower fitness is better)
            advantage = -fitness  # Convert to reward (higher is better)
            
            # Calculate gradients (simplified policy gradient)
            for i in range(self.num_params):
                if i not in self.fixed_parameters:
                    # Gradient with respect to policy parameters
                    policy_gradient_weights[i] += advantage * params[i] * self.rl_alpha
                    policy_gradient_bias[i] += advantage * self.rl_alpha
        
        # Update policy parameters
        self.policy_weights += policy_gradient_weights / len(batch)
        self.policy_bias += policy_gradient_bias / len(batch)
        
        # Apply weight decay for regularization
        self.policy_weights *= 0.999
        self.policy_bias *= 0.999

    def update_epsilon(self, episode):
        """Update exploration rate (consistent with other methods)"""
        if self.rl_epsilon_decay_type == 'exponential':
            self.rl_epsilon = max(self.rl_epsilon_min, self.rl_epsilon * self.rl_epsilon_decay)
        elif self.rl_epsilon_decay_type == 'linear':
            if self.rl_linear_decay_step is None:
                step = (self.rl_epsilon - self.rl_epsilon_min) / self.rl_num_episodes
            else:
                step = self.rl_linear_decay_step
            self.rl_epsilon = max(self.rl_epsilon_min, self.rl_epsilon - step)
        elif self.rl_epsilon_decay_type == 'inverse':
            self.rl_epsilon = self.rl_epsilon_min + (self.rl_epsilon - self.rl_epsilon_min) / (1 + self.rl_inverse_decay_coefficient * episode)
        elif self.rl_epsilon_decay_type == 'step':
            if episode % self.rl_step_interval == 0:
                self.rl_epsilon = max(self.rl_epsilon_min, self.rl_epsilon - self.rl_step_decay_amount)
        elif self.rl_epsilon_decay_type == 'cosine':
            cosine_term = (1 + np.cos(np.pi * episode / self.rl_num_episodes)) / 2
            self.rl_epsilon = self.rl_epsilon_min + (self.rl_epsilon - self.rl_epsilon_min) * (1 + self.rl_cosine_decay_amplitude * cosine_term) / 2

    def run(self):
        """
        Main RL training loop with scientifically sound continuous optimization.
        """
        # Start watchdog timer (10 minutes timeout) - consistent with GAWorker
        self.watchdog_timer.start(600000)  # 600,000 milliseconds = 10 minutes
        
        try:
            # Check for early termination
            if self.abort or self._terminate_flag:
                self.update.emit("RL optimization aborted before starting")
                return

            # Perform Sobol analysis for parameter hierarchy (consistent with other methods)
            self.update.emit("Performing Sobol Analysis for parameter hierarchy...")

            parameter_order = [item[0] for item in self.rl_parameter_data]
            sample_size = self.sobol_settings.get("sample_size", 32)
            num_samples_list = [sample_size]

            sobol_all_results, sobol_warnings = perform_sobol_analysis(
                main_system_parameters=self.main_params,
                dva_parameters_bounds=self.rl_parameter_data,
                dva_parameter_order=parameter_order,
                omega_start=self.omega_start,
                omega_end=self.omega_end,
                omega_points=self.omega_points,
                num_samples_list=num_samples_list,
                target_values_dict=self.target_values_dict,
                weights_dict=self.weights_dict,
                visualize=False,
                n_jobs=1
            )

            # Reorder parameters by Sobol sensitivity (consistent with other methods)
            last_ST = np.array(sobol_all_results['ST'][-1])
            sorted_indices = np.argsort(last_ST)[::-1]
            ranking = [self.parameter_names[i] for i in sorted_indices]
            self.update.emit("Sobol Analysis completed. Parameter ranking: " + ", ".join(ranking))

            # Reorder parameter data based on sensitivity
            new_rl_parameter_data = []
            for param in ranking:
                for item in self.rl_parameter_data:
                    if item[0] == param:
                        new_rl_parameter_data.append(item)
                        break
            self.rl_parameter_data = new_rl_parameter_data

            # Rebuild parameter mappings with new order
            self.parameter_names = []
            self.parameter_bounds = []
            self.fixed_parameters = {}
            for idx, (name, low, high, fixed) in enumerate(self.rl_parameter_data):
                self.parameter_names.append(name)
                if fixed:
                    self.parameter_bounds.append((low, low))
                    self.fixed_parameters[idx] = low
                else:
                    self.parameter_bounds.append((low, high))

            # Reset tracking variables
            self.best_fitness = float('inf')
            self.best_solution = None
            self.episode_rewards = []

            # Main RL training loop
            for episode in range(1, self.rl_num_episodes + 1):
                # Check for termination - consistent with GAWorker
                if self.abort or self._terminate_flag:
                    self.update.emit("RL optimization aborted by user")
                    break
                    
                # Reset watchdog timer
                if self.watchdog_timer.isActive():
                    self.watchdog_timer.stop()
                self.watchdog_timer.start(600000)  # 10 minutes
                
                self.update.emit(f"--- RL Episode {episode}/{self.rl_num_episodes} ---")
                episode_best_fitness = float('inf')
                episode_experiences = []

                # Update progress bar - consistent with GAWorker
                progress_percent = int((episode / self.rl_num_episodes) * 100)
                self.progress.emit(progress_percent)

                for step in range(self.rl_max_steps):
                    # Check for termination within episode steps
                    if self.abort or self._terminate_flag:
                        break
                        
                    # Generate parameters using current policy
                    params = self.generate_parameters(add_noise=True)
                    
                    # Evaluate parameters
                    fitness, results = self.evaluate_parameters(params)
                    
                    # Store experience
                    experience = (params, fitness, results)
                    episode_experiences.append(experience)
                    
                    # Update best solution
                    if fitness < self.best_fitness:
                        self.best_fitness = fitness
                        self.best_solution = params.copy()
                    
                    if fitness < episode_best_fitness:
                        episode_best_fitness = fitness

                # Check for termination before processing experiences
                if self.abort or self._terminate_flag:
                    break

                # Add experiences to replay buffer
                self.experience_buffer.extend(episode_experiences)
                
                # Limit buffer size
                if len(self.experience_buffer) > self.replay_buffer_size:
                    self.experience_buffer = self.experience_buffer[-self.replay_buffer_size:]
                
                # Update policy based on experiences
                self.update_policy(self.experience_buffer)
                
                # Update exploration rate
                self.update_epsilon(episode)
                
                # Track episode metrics
                self.episode_rewards.append(episode_best_fitness)
                self.update.emit(f"End of episode {episode}, episode best fitness: {episode_best_fitness:.6f}")
                self.update.emit(f"Overall best fitness: {self.best_fitness:.6f}")
                self.update.emit(f"Epsilon: {self.rl_epsilon:.4f}")
                
                # Emit metrics for visualization
                self.episode_metrics.emit({
                    'episode': episode, 
                    'best_reward': -episode_best_fitness,  # Convert to reward for plotting
                    'epsilon': self.rl_epsilon
                })

            # Final progress update - consistent with GAWorker
            self.progress.emit(100)

            # Final evaluation with best solution
            if self.best_solution is not None:
                try:
                    final_results = frf(
                        main_system_parameters=self.main_params,
                        dva_parameters=tuple(self.best_solution),
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
            else:
                final_results = {"Warning": "No valid solution found."}

            # Save experience
            self._save_experience()
            
            # Process results based on termination status
            if not (self.abort or self._terminate_flag):
                # Normal completion - consistent with GAWorker
                self.cleanup()
                self.finished.emit(
                    final_results,
                    list(self.best_solution) if self.best_solution else [],
                    self.parameter_names,
                    float(self.best_fitness)
                )
            else:
                # Aborted - return best solution found so far (consistent with GAWorker)
                if self.best_solution is not None:
                    self.update.emit("RL optimization was aborted, returning best solution found so far.")
                    abort_results = {"Warning": "RL optimization was aborted before completion"}
                    
                    # Include singular response estimate if we have a reasonable solution
                    if self.best_fitness < 1e6:
                        abort_results["singular_response"] = self.best_fitness
                        self.update.emit("Added estimated singular response based on best fitness value")
                    
                    self.cleanup()
                    self.finished.emit(
                        abort_results,
                        list(self.best_solution),
                        self.parameter_names,
                        float(self.best_fitness)
                    )
                else:
                    error_msg = "RL optimization was aborted before finding any valid solutions"
                    self.update.emit(error_msg)
                    self.cleanup()
                    self.error.emit(error_msg)
            
        except Exception as e:
            error_msg = f"RL optimization error: {str(e)}"
            self.update.emit(error_msg)
            self.cleanup()
            self.error.emit(error_msg)
