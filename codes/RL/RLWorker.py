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
from PyQt5.QtCore import Qt, QThread, pyqtSignal
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
    A Worker thread for running a Reinforcement Learning (RL) optimization
    of Dynamic Vibration Absorber (DVA) parameters, analogous to GAWorker.

    This version adds the option of selecting one of five different reward systems:
      1) Basic Absolute Error
      2) Absolute Error + Simple Cost
      3) Absolute Error + Parameter Sparsity
      4) Original Code's Reward (Abs Error + alpha_sparsity * sum of parameter magnitudes)
      5) Absolute Error + Extended Cost/Time

    In systems 2 and 5, the code now expects that the user provides a list 
    (cost_values or extended_cost_values) of costs—one per DVA parameter—so you can 
    manually specify the cost for each individual parameter.

    Each reward system aims for zero as the ideal reward when singular_response is 1 
    and any additional penalty (cost, sparsity, time) is minimized.

    Signals:
        finished(dict, list, list, float):
            Emitted upon completion, carrying:
              - A dictionary of final FRF results
              - The best parameter list found
              - The names of all parameters
              - The best (maximum) reward achieved
        error(str):
            Emitted if any exception or error is raised.
        update(str):
            Emitted for progress/status updates during training.
    """
    finished = pyqtSignal(dict, list, list, float)
    error = pyqtSignal(str)
    update = pyqtSignal(str)

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
        rl_epsilon_decay_type,  # Decay type for epsilon (e.g., 'exponential', 'linear', 'inverse', 'step', 'cosine')
        rl_epsilon_decay,       # Generic decay factor (used by exponential decay by default)
        rl_parameter_data,

        # ------------------- Reward System Selection -------------------
        rl_reward_system,  # Integer (1 to 5) selecting which reward system to use

        # For system #1 (Basic Absolute Error): no extra parameter.
        # For system #2 (Absolute Error + Simple Cost):
        cost_values=None,  # Now a list of costs for each DVA parameter (manual cost per parameter)
        # For system #3 (Absolute Error + Parameter Sparsity):
        alpha_sparsity_simplified=0.01,  # Penalty factor for each active parameter
        # For system #4 (Original Reward):
        alpha_sparsity=0.01,  # Penalty factor multiplied by the sum of absolute parameter values
        # For system #5 (Absolute Error + Extended Cost):
        extended_cost_values=None,  # A list of extended cost values for each parameter (manual per-parameter cost)
        time_penalty_weight=0.0,    # Additional penalty weight for time (if desired)

        # For saving/loading Q-table
        q_table_save_path=None,
        load_existing_qtable=False,
        sobol_settings=None,  # Dictionary of additional Sobol parameters (e.g., sample_size)

        # ----------------- Additional Epsilon Decay Parameters -----------------
        rl_linear_decay_step=None,         # For linear decay: manual step value; if None, computed as default.
        rl_inverse_decay_coefficient=1.0,    # For inverse decay: scaling coefficient.
        rl_step_interval=10,               # For step decay: episode interval for decay.
        rl_step_decay_amount=None,         # For step decay: decay amount; if None, defaults to rl_epsilon_decay.
        rl_cosine_decay_amplitude=1.0       # For cosine decay: amplitude factor.
    ):
        """
        Constructor for the RL worker, with extra parameters to handle
        multiple reward systems (including per-parameter cost input) and
        extra cost/sparsity/time penalty terms.
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

        # Reward System selection and associated parameters
        self.rl_reward_system = rl_reward_system  # 1..5
        self.cost_values = cost_values              # List of cost per DVA parameter (for system 2)
        self.alpha_sparsity_simplified = alpha_sparsity_simplified
        self.alpha_sparsity = alpha_sparsity
        self.extended_cost_values = extended_cost_values  # List of extended cost per parameter (for system 5)
        self.time_penalty_weight = time_penalty_weight

        self.rl_parameter_data = rl_parameter_data

        # Extra epsilon decay parameters
        self.rl_linear_decay_step = rl_linear_decay_step
        self.rl_inverse_decay_coefficient = rl_inverse_decay_coefficient
        self.rl_step_interval = rl_step_interval
        self.rl_step_decay_amount = rl_step_decay_amount if rl_step_decay_amount is not None else rl_epsilon_decay
        self.rl_cosine_decay_amplitude = rl_cosine_decay_amplitude

        # Q-table saving/loading
        self.q_table_save_path = q_table_save_path
        self.load_existing_qtable = load_existing_qtable

        # Optional Sobol settings provided from the GUI.
        self.sobol_settings = sobol_settings if sobol_settings is not None else {}

        # --------------------------
        # Build Parameter Mappings & Q-Table
        # --------------------------
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

        # Discretization of state space (tabular)
        self.num_bins = 5  
        self.param_discretizations = []
        for (low, high) in self.parameter_bounds:
            if np.isclose(low, high, atol=1e-12):
                self.param_discretizations.append([low])
            else:
                self.param_discretizations.append(np.linspace(low, high, self.num_bins).tolist())

        # Initialize Q-table (load if available)
        self.q_table = {}
        if self.load_existing_qtable and self.q_table_save_path is not None:
            self._load_q_table()

        # Build action space: each tunable parameter gets an increment and decrement action.
        self.actions = self._build_action_space()

        # Track best solution
        self.best_reward = -np.inf
        self.best_solution = None

    # ---------------------------------------------------------------------
    # Build Actions, Load & Save Q-Table
    # ---------------------------------------------------------------------
    def _build_action_space(self):
        actions = []
        for i, (low, high) in enumerate(self.parameter_bounds):
            if i in self.fixed_parameters:
                continue
            actions.append((i, +1))
            actions.append((i, -1))
        return actions

    def _load_q_table(self):
        if os.path.exists(self.q_table_save_path):
            try:
                with open(self.q_table_save_path, 'rb') as f:
                    self.q_table = pickle.load(f)
                print("Q-table loaded from", self.q_table_save_path)
            except Exception as e:
                print("Error loading Q-table:", e)

    def _save_q_table(self):
        if self.q_table_save_path is not None:
            try:
                with open(self.q_table_save_path, 'wb') as f:
                    pickle.dump(self.q_table, f)
                print("Q-table saved to", self.q_table_save_path)
            except Exception as e:
                print("Error saving Q-table:", e)

    # ---------------------------------------------------------------------
    # Q-Learning Helper Methods
    # ---------------------------------------------------------------------
    def _get_state_key(self, state_indices):
        return tuple(state_indices)

    def _initialize_q_values(self, state_key):
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(len(self.actions), dtype=float)

    def _select_action(self, state_key):
        self._initialize_q_values(state_key)
        if random.random() < self.rl_epsilon:
            return random.randrange(len(self.actions))
        else:
            return int(np.argmax(self.q_table[state_key]))

    def _indices_to_parameters(self, state_indices):
        params = []
        for i, idx in enumerate(state_indices):
            if i in self.fixed_parameters:
                params.append(self.fixed_parameters[i])
            else:
                params.append(self.param_discretizations[i][idx])
        return tuple(params)

    def _random_initial_state(self):
        init_indices = []
        for i, discretized_vals in enumerate(self.param_discretizations):
            if i in self.fixed_parameters:
                init_indices.append(0)
            else:
                init_indices.append(random.randrange(len(discretized_vals)))
        return init_indices

    # ---------------------------------------------------------------------
    # Epsilon Decay
    # ---------------------------------------------------------------------
    def update_epsilon(self, episode):
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

    # ---------------------------------------------------------------------
    # Reward Calculation: 5 Different Systems
    # ---------------------------------------------------------------------
    def compute_reward(self, singular_response, params):
        if singular_response is None or not np.isfinite(singular_response):
            return -1e6

        # Always measure the absolute error
        perf_error = abs(singular_response - 1)

        # 1) Basic Absolute Error
        if self.rl_reward_system == 1:
            return -perf_error

        # 2) Absolute Error + Simple Cost
        elif self.rl_reward_system == 2:
            total_cost = 0.0
            if self.cost_values is not None:
                # Expect cost_values to be a list of costs (one per DVA parameter)
                for i, val in enumerate(params):
                    c = self.cost_values[i] if i < len(self.cost_values) else 0.0
                    total_cost += c * abs(val)
            return - (perf_error + total_cost)

        # 3) Absolute Error + Parameter Sparsity
        elif self.rl_reward_system == 3:
            active_count = sum(1 for p in params if abs(p) > 1e-9)
            return - (perf_error + self.alpha_sparsity_simplified * active_count)

        # 4) Original Reward: Abs Error + alpha_sparsity * sum(abs(params))
        elif self.rl_reward_system == 4:
            sparsity_penalty = self.alpha_sparsity * sum(abs(p) for p in params)
            return - (perf_error + sparsity_penalty)

        # 5) Absolute Error + Extended Cost
        elif self.rl_reward_system == 5:
            extended_cost = 0.0
            if self.extended_cost_values is not None:
                # Expect extended_cost_values to be a list of costs (one per DVA parameter)
                for i, val in enumerate(params):
                    c = self.extended_cost_values[i] if i < len(self.extended_cost_values) else 0.0
                    extended_cost += c * abs(val)
            total_penalty = extended_cost + self.time_penalty_weight
            return - (perf_error + total_penalty)

        return -perf_error

    # ---------------------------------------------------------------------
    # Step Through Environment
    # ---------------------------------------------------------------------
    def _step_environment(self, state_indices, action_idx):
        param_index, delta = self.actions[action_idx]
        new_state_indices = list(state_indices)
        old_val = new_state_indices[param_index]
        new_val = old_val + delta
        new_val = max(0, min(new_val, self.num_bins - 1))
        new_state_indices[param_index] = new_val

        current_params = self._indices_to_parameters(new_state_indices)

        try:
            results = frf(
                main_system_parameters=self.main_params,
                dva_parameters=current_params,
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
            reward = self.compute_reward(singular_response, current_params)
        except Exception as e:
            reward = -1e6
            results = {"Error": str(e)}

        done_flag = False
        return new_state_indices, reward, done_flag, results

    # ---------------------------------------------------------------------
    # Main RL Loop
    # ---------------------------------------------------------------------
    def run(self):
        try:
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

            last_ST = np.array(sobol_all_results['ST'][-1])
            sorted_indices = np.argsort(last_ST)[::-1]
            ranking = [self.parameter_names[i] for i in sorted_indices]
            self.update.emit("Sobol Analysis completed. Parameter ranking: " + ", ".join(ranking))

            new_rl_parameter_data = []
            for param in ranking:
                for item in self.rl_parameter_data:
                    if item[0] == param:
                        new_rl_parameter_data.append(item)
                        break
            self.rl_parameter_data = new_rl_parameter_data

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

            self.param_discretizations = []
            for (low, high) in self.parameter_bounds:
                if np.isclose(low, high, atol=1e-12):
                    self.param_discretizations.append([low])
                else:
                    self.param_discretizations.append(np.linspace(low, high, self.num_bins).tolist())

            self.best_reward = -np.inf
            self.best_solution = None

            for episode in range(1, self.rl_num_episodes + 1):
                self.update.emit(f"--- RL Episode {episode}/{self.rl_num_episodes} ---")
                state_indices = self._random_initial_state()
                state_key = self._get_state_key(state_indices)

                for step in range(self.rl_max_steps):
                    action_idx = self._select_action(state_key)
                    new_state_indices, reward, done, results = self._step_environment(state_indices, action_idx)
                    new_state_key = self._get_state_key(new_state_indices)
                    self._initialize_q_values(new_state_key)

                    old_q = self.q_table[state_key][action_idx]
                    max_future_q = np.max(self.q_table[new_state_key])
                    new_q = old_q + self.rl_alpha * (reward + self.rl_gamma * max_future_q - old_q)
                    self.q_table[state_key][action_idx] = new_q

                    if reward > self.best_reward:
                        self.best_reward = reward
                        self.best_solution = self._indices_to_parameters(new_state_indices)

                    state_indices = new_state_indices
                    state_key = new_state_key

                    if done:
                        self.update.emit(f"Episode {episode} ended early at step {step} with reward={reward:.6f}")
                        break

                self.update_epsilon(episode)
                self.update.emit(f"End of episode {episode}, current best reward: {self.best_reward:.6f}")
                self.update.emit(f"Epsilon after decay: {self.rl_epsilon:.4f}")

            if self.best_solution is not None:
                try:
                    final_results = frf(
                        main_system_parameters=self.main_params,
                        dva_parameters=self.best_solution,
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

            self._save_q_table()
            self.finished.emit(final_results,
                               list(self.best_solution) if self.best_solution else [],
                               self.parameter_names,
                               float(self.best_reward))
        except Exception as e:
            self.error.emit(str(e))
