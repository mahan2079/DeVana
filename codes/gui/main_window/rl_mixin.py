from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from RL.RLWorker import RLWorker
import numpy as np
import time
import json


class RLOptimizationMixin:
    def create_rl_tab(self):
        """Create the reinforcement learning optimization tab with subtabs"""
        self.rl_tab = QWidget()
        layout = QVBoxLayout(self.rl_tab)

        self.rl_sub_tabs = QTabWidget()
        layout.addWidget(self.rl_sub_tabs)

        # ------------------- Sub-tab 1: Hyperparameters -------------------
        rl_hyper_tab = QWidget()
        hyper_layout = QVBoxLayout(rl_hyper_tab)

        # Basic RL Parameters Group
        hyper_group = QGroupBox("RL Hyperparameters")
        hyper_form = QFormLayout(hyper_group)

        self.rl_num_episodes_box = QSpinBox()
        self.rl_num_episodes_box.setRange(1, 10000)
        self.rl_num_episodes_box.setValue(100)
        self.rl_num_episodes_box.setToolTip("Number of training episodes")

        self.rl_max_steps_box = QSpinBox()
        self.rl_max_steps_box.setRange(1, 10000)
        self.rl_max_steps_box.setValue(50)
        self.rl_max_steps_box.setToolTip("Maximum steps per episode")

        self.rl_alpha_box = QDoubleSpinBox()
        self.rl_alpha_box.setRange(0.0001, 1.0)
        self.rl_alpha_box.setDecimals(4)
        self.rl_alpha_box.setValue(0.001)
        self.rl_alpha_box.setToolTip("Learning rate for policy updates")

        self.rl_gamma_box = QDoubleSpinBox()
        self.rl_gamma_box.setRange(0.0, 1.0)
        self.rl_gamma_box.setDecimals(4)
        self.rl_gamma_box.setValue(0.95)
        self.rl_gamma_box.setToolTip("Discount factor for future rewards")

        self.rl_epsilon_box = QDoubleSpinBox()
        self.rl_epsilon_box.setRange(0.0, 1.0)
        self.rl_epsilon_box.setDecimals(4)
        self.rl_epsilon_box.setValue(1.0)
        self.rl_epsilon_box.setToolTip("Initial exploration rate")

        self.rl_epsilon_min_box = QDoubleSpinBox()
        self.rl_epsilon_min_box.setRange(0.0, 1.0)
        self.rl_epsilon_min_box.setDecimals(4)
        self.rl_epsilon_min_box.setValue(0.05)
        self.rl_epsilon_min_box.setToolTip("Minimum exploration rate")

        self.rl_epsilon_decay_box = QDoubleSpinBox()
        self.rl_epsilon_decay_box.setRange(0.0, 1.0)
        self.rl_epsilon_decay_box.setDecimals(4)
        self.rl_epsilon_decay_box.setValue(0.99)
        self.rl_epsilon_decay_box.setToolTip("Exploration decay factor")

        self.rl_epsilon_decay_type_combo = QComboBox()
        self.rl_epsilon_decay_type_combo.addItems([
            "exponential", "linear", "inverse", "step", "cosine"
        ])
        self.rl_epsilon_decay_type_combo.setToolTip("Type of exploration decay")

        hyper_form.addRow("Episodes:", self.rl_num_episodes_box)
        hyper_form.addRow("Max Steps per Episode:", self.rl_max_steps_box)
        hyper_form.addRow("Learning Rate (α):", self.rl_alpha_box)
        hyper_form.addRow("Discount Factor (γ):", self.rl_gamma_box)
        hyper_form.addRow("Initial Exploration (ε):", self.rl_epsilon_box)
        hyper_form.addRow("Min Exploration:", self.rl_epsilon_min_box)
        hyper_form.addRow("Exploration Decay:", self.rl_epsilon_decay_box)
        hyper_form.addRow("Decay Type:", self.rl_epsilon_decay_type_combo)

        hyper_layout.addWidget(hyper_group)

        # Advanced Parameters Group
        advanced_group = QGroupBox("Advanced Parameters")
        advanced_form = QFormLayout(advanced_group)

        # Sparsity penalty (consistent with GA, PSO, SA)
        self.rl_alpha_sparsity_box = QDoubleSpinBox()
        self.rl_alpha_sparsity_box.setRange(0.0, 1.0)
        self.rl_alpha_sparsity_box.setDecimals(4)
        self.rl_alpha_sparsity_box.setValue(0.01)
        self.rl_alpha_sparsity_box.setToolTip("Sparsity penalty factor (same as GA/PSO/SA)")

        # Experience replay parameters
        self.rl_replay_buffer_size_box = QSpinBox()
        self.rl_replay_buffer_size_box.setRange(1000, 100000)
        self.rl_replay_buffer_size_box.setValue(10000)
        self.rl_replay_buffer_size_box.setToolTip("Size of experience replay buffer")

        self.rl_batch_size_box = QSpinBox()
        self.rl_batch_size_box.setRange(8, 256)
        self.rl_batch_size_box.setValue(32)
        self.rl_batch_size_box.setToolTip("Batch size for policy updates")

        # Exploration noise
        self.rl_noise_std_box = QDoubleSpinBox()
        self.rl_noise_std_box.setRange(0.01, 1.0)
        self.rl_noise_std_box.setDecimals(3)
        self.rl_noise_std_box.setValue(0.1)
        self.rl_noise_std_box.setToolTip("Standard deviation of exploration noise")

        advanced_form.addRow("Sparsity Penalty (α):", self.rl_alpha_sparsity_box)
        advanced_form.addRow("Replay Buffer Size:", self.rl_replay_buffer_size_box)
        advanced_form.addRow("Batch Size:", self.rl_batch_size_box)
        advanced_form.addRow("Exploration Noise σ:", self.rl_noise_std_box)

        hyper_layout.addWidget(advanced_group)

        # Run button
        self.run_rl_button = QPushButton("Run RL Optimization")
        self.run_rl_button.clicked.connect(self.run_rl_optimization)
        hyper_layout.addWidget(self.run_rl_button)

        self.rl_sub_tabs.addTab(rl_hyper_tab, "Hyperparameters")

        # ------------------- Sub-tab 2: Parameters -------------------
        param_tab = QWidget()
        param_layout = QVBoxLayout(param_tab)

        # Parameter table (consistent with GA, PSO, SA)
        self.rl_param_table = QTableWidget()
        dva_parameters = [
            *[f"beta_{i}" for i in range(1, 16)],
            *[f"lambda_{i}" for i in range(1, 16)],
            *[f"mu_{i}" for i in range(1, 4)],
            *[f"nu_{i}" for i in range(1, 16)],
        ]

        self.rl_param_table.setRowCount(len(dva_parameters))
        self.rl_param_table.setColumnCount(5)  # Removed cost column - using unified sparsity penalty
        self.rl_param_table.setHorizontalHeaderLabels([
            "Parameter",
            "Fixed",
            "Fixed Value",
            "Lower Bound",
            "Upper Bound"
        ])
        self.rl_param_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.rl_param_table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        for row, param in enumerate(dva_parameters):
            param_item = QTableWidgetItem(param)
            param_item.setFlags(Qt.ItemIsEnabled)
            self.rl_param_table.setItem(row, 0, param_item)

            fixed_checkbox = QCheckBox()
            fixed_checkbox.setChecked(True)
            fixed_checkbox.stateChanged.connect(lambda state, r=row: self.toggle_rl_fixed(state, r))
            self.rl_param_table.setCellWidget(row, 1, fixed_checkbox)

            fixed_value_spin = QDoubleSpinBox()
            fixed_value_spin.setRange(0, 1e10)
            fixed_value_spin.setDecimals(6)
            fixed_value_spin.setValue(0.0)
            fixed_value_spin.setEnabled(True)
            self.rl_param_table.setCellWidget(row, 2, fixed_value_spin)

            lower_spin = QDoubleSpinBox()
            lower_spin.setRange(0, 1e10)
            lower_spin.setDecimals(6)
            lower_spin.setValue(0.0)
            lower_spin.setEnabled(False)
            self.rl_param_table.setCellWidget(row, 3, lower_spin)

            upper_spin = QDoubleSpinBox()
            upper_spin.setRange(0, 1e10)
            upper_spin.setDecimals(6)
            upper_spin.setValue(1.0)
            upper_spin.setEnabled(False)
            self.rl_param_table.setCellWidget(row, 4, upper_spin)

        param_layout.addWidget(self.rl_param_table)

        self.rl_sub_tabs.addTab(param_tab, "Parameters")

        # ------------------- Sub-tab 3: Results -------------------
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)

        self.rl_results_text = QTextEdit()
        self.rl_results_text.setReadOnly(True)
        results_layout.addWidget(self.rl_results_text)

        # Plot for episode rewards/fitness
        self.rl_reward_fig = Figure(figsize=(5, 3))
        self.rl_reward_canvas = FigureCanvas(self.rl_reward_fig)
        results_layout.addWidget(self.rl_reward_canvas)

        self.rl_sub_tabs.addTab(results_tab, "Results")

        # ------------------- Sub-tab 4: Benchmark -------------------
        bench_tab = QWidget()
        bench_layout = QVBoxLayout(bench_tab)

        # Controls
        ctrl_group = QGroupBox("Benchmark Controls")
        ctrl_form = QFormLayout(ctrl_group)
        self.rl_bench_runs_box = QSpinBox()
        self.rl_bench_runs_box.setRange(1, 200)
        self.rl_bench_runs_box.setValue(5)
        self.rl_bench_seed_box = QSpinBox()
        self.rl_bench_seed_box.setRange(0, 10_000_000)
        self.rl_bench_seed_box.setValue(0)
        self.rl_bench_seed_box.setToolTip("Optional base seed; 0 = random")
        btn_row = QWidget()
        btn_row_layout = QHBoxLayout(btn_row)
        btn_row_layout.setContentsMargins(0,0,0,0)
        self.rl_bench_start_button = QPushButton("Start Benchmark")
        self.rl_bench_export_button = QPushButton("Export Results")
        self.rl_bench_import_button = QPushButton("Import Results")
        btn_row_layout.addWidget(self.rl_bench_start_button)
        btn_row_layout.addWidget(self.rl_bench_export_button)
        btn_row_layout.addWidget(self.rl_bench_import_button)
        ctrl_form.addRow("Number of Runs:", self.rl_bench_runs_box)
        ctrl_form.addRow("Base Seed:", self.rl_bench_seed_box)
        ctrl_form.addRow(btn_row)
        bench_layout.addWidget(ctrl_group)

        # Runs table
        self.rl_benchmark_table = QTableWidget()
        self.rl_benchmark_table.setColumnCount(6)
        self.rl_benchmark_table.setHorizontalHeaderLabels([
            "Run #", "Best Fitness", "Duration (s)", "Episodes", "Epsilon Final", "Details"
        ])
        self.rl_benchmark_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        bench_layout.addWidget(self.rl_benchmark_table)

        # Parameter Ranges Recommendation
        rec_group = QGroupBox("Parameter Ranges Recommendation")
        rec_layout = QVBoxLayout(rec_group)

        rec_row = QWidget()
        rec_row_layout = QHBoxLayout(rec_row)
        rec_row_layout.setContentsMargins(0,0,0,0)
        self.rl_rec_method_combo = QComboBox()
        self.rl_rec_method_combo.addItems(["IQR", "P05-P95"])
        self.rl_compute_rec_button = QPushButton("Compute Recommendations")
        self.rl_apply_rec_button = QPushButton("Apply to Parameter Table")
        rec_row_layout.addWidget(QLabel("Method:"))
        rec_row_layout.addWidget(self.rl_rec_method_combo)
        rec_row_layout.addWidget(self.rl_compute_rec_button)
        rec_row_layout.addWidget(self.rl_apply_rec_button)
        rec_layout.addWidget(rec_row)

        self.rl_rec_table = QTableWidget()
        self.rl_rec_table.setColumnCount(4)
        self.rl_rec_table.setHorizontalHeaderLabels(["Parameter", "Lower", "Upper", "N"])
        self.rl_rec_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        rec_layout.addWidget(self.rl_rec_table)
        bench_layout.addWidget(rec_group)

        self.rl_sub_tabs.addTab(bench_tab, "Benchmark")

        # State
        self.rl_benchmark_data = []
        self._rl_bench_active = False
        self._rl_bench_total = 0
        self._rl_bench_index = 0
        self._rl_current_episode_history = []
        self._rl_run_start_time = None

        # Wire actions
        self.rl_bench_start_button.clicked.connect(self.start_rl_benchmark)
        self.rl_bench_export_button.clicked.connect(self.export_rl_benchmark_data)
        self.rl_bench_import_button.clicked.connect(self.import_rl_benchmark_data)
        self.rl_compute_rec_button.clicked.connect(self.compute_rl_parameter_recommendations)
        self.rl_apply_rec_button.clicked.connect(self.apply_rl_recommended_ranges_to_table)

    def run_rl_optimization(self):
        """Run RL optimization with improved parameter handling"""
        try:
            # Clean up any previous RL worker that might still exist
            if hasattr(self, 'rl_worker'):
                try:
                    # First use our custom terminate method if available
                    if hasattr(self.rl_worker, 'terminate'):
                        self.rl_worker.terminate()
                    
                    # Disconnect signals
                    self.rl_worker.finished.disconnect()
                    self.rl_worker.error.disconnect()
                    self.rl_worker.update.disconnect()
                    self.rl_worker.progress.disconnect()
                    self.rl_worker.episode_metrics.disconnect()
                except Exception as e:
                    print(f"Error disconnecting RL worker signals: {str(e)}")
                
                # Wait for thread to finish if it's still running
                if self.rl_worker.isRunning():
                    if not self.rl_worker.wait(1000):  # Wait up to 1 second for graceful termination
                        print("RL worker didn't terminate gracefully, forcing termination...")
                        # Force termination as a last resort
                        self.rl_worker.terminate()
                        self.rl_worker.wait()
            
            # Get main system parameters (consistent with other methods)
            main_params = self.get_main_system_params()
            target_values, weights = self.get_target_values_weights()
            
            # Initialize reward history for plotting
            self.rl_reward_history = []
            
            # Build parameter data (consistent with GA, PSO, SA)
            dva_bounds = {}
            EPSILON = 1e-6
            
            for row in range(self.rl_param_table.rowCount()):
                param_name = self.rl_param_table.item(row, 0).text()
                fixed = self.rl_param_table.cellWidget(row, 1).isChecked()
                
                if fixed:
                    fixed_value = self.rl_param_table.cellWidget(row, 2).value()
                    dva_bounds[param_name] = (fixed_value, fixed_value + EPSILON)
                else:
                    lower = self.rl_param_table.cellWidget(row, 3).value()
                    upper = self.rl_param_table.cellWidget(row, 4).value()
                    if lower > upper:
                        QMessageBox.warning(self, "Input Error",
                            f"For parameter {param_name}, lower bound ({lower}) is greater than upper bound ({upper}).")
                        return
                    dva_bounds[param_name] = (lower, upper)
            
            # Parameter order (consistent with other methods)
            param_order = [
                'beta_1','beta_2','beta_3','beta_4','beta_5','beta_6','beta_7','beta_8','beta_9','beta_10','beta_11','beta_12','beta_13','beta_14','beta_15',
                'lambda_1','lambda_2','lambda_3','lambda_4','lambda_5','lambda_6','lambda_7','lambda_8','lambda_9','lambda_10','lambda_11','lambda_12','lambda_13','lambda_14','lambda_15',
                'mu_1','mu_2','mu_3',
                'nu_1','nu_2','nu_3','nu_4','nu_5','nu_6','nu_7','nu_8','nu_9','nu_10','nu_11','nu_12','nu_13','nu_14','nu_15'
            ]
            
            # Build parameter data list (consistent format)
            rl_parameter_data = []
            for name in param_order:
                if name in dva_bounds:
                    low, high = dva_bounds[name]
                    fixed = abs(low - high) < EPSILON
                    rl_parameter_data.append((name, low, high, fixed))

            # Create and configure RL worker
            self.rl_worker = RLWorker(
                main_params=main_params,
                target_values_dict=target_values,
                weights_dict=weights,
                omega_start=self.omega_start_box.value(),
                omega_end=self.omega_end_box.value(),
                omega_points=self.omega_points_box.value(),
                rl_num_episodes=self.rl_num_episodes_box.value(),
                rl_max_steps=self.rl_max_steps_box.value(),
                rl_alpha=self.rl_alpha_box.value(),
                rl_gamma=self.rl_gamma_box.value(),
                rl_epsilon=self.rl_epsilon_box.value(),
                rl_epsilon_min=self.rl_epsilon_min_box.value(),
                rl_epsilon_decay_type=self.rl_epsilon_decay_type_combo.currentText(),
                rl_epsilon_decay=self.rl_epsilon_decay_box.value(),
                rl_parameter_data=rl_parameter_data,
                
                # Simplified reward system (consistent with other methods)
                alpha_sparsity=self.rl_alpha_sparsity_box.value(),
                
                # Advanced parameters
                replay_buffer_size=self.rl_replay_buffer_size_box.value(),
                batch_size=self.rl_batch_size_box.value(),
                noise_std=self.rl_noise_std_box.value(),
                
                # Sobol settings (consistent with other methods)
                sobol_settings={"sample_size": 32}
            )
            
            # Connect signals (consistent with other methods)
            self.rl_worker.finished.connect(self.handle_rl_finished)
            self.rl_worker.error.connect(self.handle_rl_error)
            self.rl_worker.update.connect(self.handle_rl_update)
            self.rl_worker.progress.connect(lambda p: self.status_bar.showMessage(f"RL optimization progress: {p}%"))
            self.rl_worker.episode_metrics.connect(self.handle_rl_metrics)

            # Start optimization
            self.run_rl_button.setEnabled(False)
            self.rl_results_text.clear()
            self.rl_results_text.append("Starting RL optimization...")
            self.rl_results_text.append("Performing Sobol sensitivity analysis...")
            # For benchmark capturing
            if self._rl_bench_active:
                self._rl_current_episode_history = []
                self._rl_run_start_time = time.time()
            self.rl_worker.start()
            
        except Exception as e:
            self.handle_rl_error(str(e))

    def handle_rl_finished(self, results, best_params, param_names, best_fitness):
        """Handle completion of RL optimization (consistent with other methods)"""
        self.run_rl_button.setEnabled(True)

        # Store results
        best_dict = {n: v for n, v in zip(param_names, best_params)}
        self.current_rl_best_params = best_dict
        self.current_rl_best_fitness = best_fitness

        # If benchmarking, record run and chain next
        if self._rl_bench_active:
            duration_s = 0.0
            if self._rl_run_start_time is not None:
                duration_s = max(0.0, time.time() - self._rl_run_start_time)

            # Determine final epsilon if present in metrics history
            eps_final = None
            if self._rl_current_episode_history:
                try:
                    eps_final = float(self._rl_current_episode_history[-1].get('epsilon', None))
                except Exception:
                    eps_final = None

            run_record = {
                'run_number': int(self._rl_bench_index + 1),
                'best_fitness': float(best_fitness),
                'best_solution': [float(best_dict[n]) for n in param_names],
                'parameter_names': list(param_names),
                'episode_history': list(self._rl_current_episode_history),
                'episodes': int(len(self._rl_current_episode_history)),
                'epsilon_final': eps_final,
                'duration_s': float(duration_s),
            }
            self.rl_benchmark_data.append(run_record)
            self._append_rl_benchmark_row(run_record)

            # Next run or finish
            self._rl_bench_index += 1
            if self._rl_bench_index < self._rl_bench_total:
                self.run_next_rl_benchmark()
            else:
                self._rl_bench_active = False
                self.status_bar.showMessage("RL benchmark completed")
        else:
            # Single-run display
            self.rl_results_text.append("\n" + "="*50)
            self.rl_results_text.append("OPTIMIZATION COMPLETED")
            self.rl_results_text.append("="*50)
            self.rl_results_text.append(f"\nBest Fitness: {best_fitness:.6f}")
            if isinstance(results, dict) and 'singular_response' in results:
                self.rl_results_text.append(f"Singular Response: {results['singular_response']:.6f}")
            self.rl_results_text.append("\nBest Parameters:")
            for name, val in best_dict.items():
                self.rl_results_text.append(f"  {name}: {val:.6f}")
            self.status_bar.showMessage("RL optimization completed successfully")

    def handle_rl_error(self, err):
        """Handle RL optimization errors (consistent with other methods)"""
        self.run_rl_button.setEnabled(True)
        QMessageBox.critical(self, "RL Optimization Error", 
                           f"An error occurred during RL optimization:\n\n{err}")
        self.rl_results_text.append(f"\nERROR: {err}")
        self.status_bar.showMessage("RL optimization failed")

    def handle_rl_update(self, msg):
        """Handle progress updates from RL worker"""
        self.rl_results_text.append(msg)
        # Auto-scroll to bottom
        cursor = self.rl_results_text.textCursor()
        cursor.movePosition(cursor.End)
        self.rl_results_text.setTextCursor(cursor)

    def handle_rl_metrics(self, metrics):
        """Update reward plot based on episode metrics"""
        episode = metrics.get('episode')
        reward = metrics.get('best_reward')  # This is -fitness for plotting
        
        if not hasattr(self, 'rl_reward_history'):
            self.rl_reward_history = []
        self.rl_reward_history.append((episode, reward))

        # If benchmarking, capture the raw metrics per episode
        if self._rl_bench_active:
            self._rl_current_episode_history.append({
                'episode': int(metrics.get('episode', 0)),
                'best_reward': float(metrics.get('best_reward', 0.0)),
                'epsilon': float(metrics.get('epsilon', 0.0))
            })

        # Update plot
        self.rl_reward_fig.clear()
        ax = self.rl_reward_fig.add_subplot(111)
        
        if len(self.rl_reward_history) > 1:
            episodes = [e for e, _ in self.rl_reward_history]
            rewards = [r for _, r in self.rl_reward_history]
            
            ax.plot(episodes, rewards, 'b-', marker='o', markersize=3, linewidth=1)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Best Reward (Higher is Better)')
            ax.set_title('RL Training Progress')
            ax.grid(True, alpha=0.3)
            
            # Add trend line if enough data points
            if len(episodes) > 10:
                z = np.polyfit(episodes, rewards, 1)
                p = np.poly1d(z)
                ax.plot(episodes, p(episodes), 'r--', alpha=0.7, linewidth=1, label='Trend')
                ax.legend()
        
        self.rl_reward_canvas.draw()

    def toggle_rl_fixed(self, state, row):
        """Toggle parameter fixed state (consistent with other methods)"""
        fixed = (state == Qt.Checked)
        fixed_value_spin = self.rl_param_table.cellWidget(row, 2)
        lower_spin = self.rl_param_table.cellWidget(row, 3)
        upper_spin = self.rl_param_table.cellWidget(row, 4)

        fixed_value_spin.setEnabled(fixed)
        lower_spin.setEnabled(not fixed)
        upper_spin.setEnabled(not fixed)

        if fixed:
            # When fixing a parameter, set fixed value to current lower bound
            fixed_value_spin.setValue(lower_spin.value())
        else:
            # When unfixing, ensure bounds are valid
            if lower_spin.value() > upper_spin.value():
                upper_spin.setValue(lower_spin.value() + 1.0)

    # ------------------- Benchmark orchestration -------------------
    def start_rl_benchmark(self):
        try:
            self.rl_benchmark_table.setRowCount(0)
            self.rl_benchmark_data = []
            self._rl_bench_total = self.rl_bench_runs_box.value()
            self._rl_bench_index = 0
            self._rl_bench_active = True
            self.status_bar.showMessage("Starting RL benchmark...")
            self.run_next_rl_benchmark()
        except Exception as e:
            QMessageBox.critical(self, "RL Benchmark Error", str(e))

    def run_next_rl_benchmark(self):
        # For each run, just reuse current settings and start RL
        self._rl_current_episode_history = []
        self._rl_run_start_time = time.time()
        self.run_rl_optimization()

    def _append_rl_benchmark_row(self, run_record):
        row = self.rl_benchmark_table.rowCount()
        self.rl_benchmark_table.insertRow(row)
        self.rl_benchmark_table.setItem(row, 0, QTableWidgetItem(str(run_record.get('run_number', ''))))
        self.rl_benchmark_table.setItem(row, 1, QTableWidgetItem(f"{run_record.get('best_fitness', float('nan')):.6f}"))
        self.rl_benchmark_table.setItem(row, 2, QTableWidgetItem(f"{run_record.get('duration_s', 0.0):.2f}"))
        self.rl_benchmark_table.setItem(row, 3, QTableWidgetItem(str(run_record.get('episodes', 0))))
        eps_final = run_record.get('epsilon_final', None)
        self.rl_benchmark_table.setItem(row, 4, QTableWidgetItem("" if eps_final is None else f"{eps_final:.4f}"))
        btn = QPushButton("Details")
        btn.clicked.connect(lambda _=False, rr=run_record: self.show_rl_run_details(rr))
        self.rl_benchmark_table.setCellWidget(row, 5, btn)

    def show_rl_run_details(self, run_data):
        try:
            # Create a simple window with training plots and best params
            dlg = QDialog(self)
            dlg.setWindowTitle(f"RL Run Details - Run #{run_data.get('run_number', '')}")
            vbox = QVBoxLayout(dlg)

            # Plot
            fig = Figure(figsize=(8, 5), tight_layout=True)
            canvas = FigureCanvas(fig)
            ax1 = fig.add_subplot(2,1,1)
            ax2 = fig.add_subplot(2,1,2)
            hist = run_data.get('episode_history', [])
            if hist:
                episodes = [h.get('episode', i+1) for i, h in enumerate(hist)]
                rewards = [h.get('best_reward', 0.0) for h in hist]
                eps = [h.get('epsilon', 0.0) for h in hist]
                ax1.plot(episodes, rewards, 'b-', marker='o', linewidth=1)
                ax1.set_title('Best Reward per Episode')
                ax1.set_xlabel('Episode')
                ax1.set_ylabel('Reward')
                ax1.grid(True, alpha=0.3)
                ax2.plot(episodes, eps, 'g-', marker='s', linewidth=1)
                ax2.set_title('Epsilon per Episode')
                ax2.set_xlabel('Episode')
                ax2.set_ylabel('Epsilon')
                ax2.grid(True, alpha=0.3)
            else:
                ax1.text(0.5, 0.5, 'No episode history', ha='center', va='center', transform=ax1.transAxes)
                ax2.axis('off')
            vbox.addWidget(canvas)

            # Best parameters
            best_params = run_data.get('best_solution', [])
            param_names = run_data.get('parameter_names', [])
            if best_params and param_names:
                text = QTextEdit()
                text.setReadOnly(True)
                text.append(f"Best Fitness: {run_data.get('best_fitness', float('nan')):.6f}")
                text.append("")
                text.append("Best Parameters:")
                for n, v in zip(param_names, best_params):
                    try:
                        text.append(f"  {n}: {float(v):.6f}")
                    except Exception:
                        text.append(f"  {n}: {v}")
                vbox.addWidget(text)

            dlg.resize(900, 700)
            dlg.exec_()
        except Exception as e:
            QMessageBox.warning(self, "Details Error", str(e))

    # ------------------- Export / Import -------------------
    def export_rl_benchmark_data(self):
        if not self.rl_benchmark_data:
            QMessageBox.information(self, "Export RL Benchmark", "No benchmark data to export.")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export RL Benchmark Data", "rl_benchmark.json", "JSON Files (*.json)")
        if not path:
            return
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                try:
                    if isinstance(obj, (np.integer,)):
                        return int(obj)
                    if isinstance(obj, (np.floating,)):
                        return float(obj)
                    if isinstance(obj, (np.ndarray,)):
                        return obj.tolist()
                except Exception:
                    pass
                return json.JSONEncoder.default(self, obj)
        try:
            with open(path, 'w') as f:
                json.dump(self.rl_benchmark_data, f, cls=NumpyEncoder, indent=2)
            QMessageBox.information(self, "Export RL Benchmark", f"Exported {len(self.rl_benchmark_data)} runs.")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def import_rl_benchmark_data(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import RL Benchmark Data", "", "JSON Files (*.json)")
        if not path:
            return
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("Invalid file format: expected a list of runs")
            self.rl_benchmark_data = data
            self.rl_benchmark_table.setRowCount(0)
            for run in self.rl_benchmark_data:
                self._append_rl_benchmark_row(run)
            QMessageBox.information(self, "Import RL Benchmark", f"Imported {len(self.rl_benchmark_data)} runs.")
        except Exception as e:
            QMessageBox.critical(self, "Import Error", str(e))

    # ------------------- Parameter range recommendations -------------------
    def compute_rl_parameter_recommendations(self):
        if not self.rl_benchmark_data:
            QMessageBox.information(self, "Parameter Ranges", "Run a benchmark or import data first.")
            return
        # Collect best parameter values across runs
        param_to_values = {}
        param_names = None
        for run in self.rl_benchmark_data:
            names = run.get('parameter_names', [])
            vals = run.get('best_solution', [])
            if not names or not vals or len(names) != len(vals):
                continue
            if param_names is None:
                param_names = names
            for n, v in zip(names, vals):
                param_to_values.setdefault(n, []).append(float(v))

        if not param_to_values:
            QMessageBox.information(self, "Parameter Ranges", "No parameter data found in runs.")
            return

        method = self.rl_rec_method_combo.currentText()
        recs = []
        for n, vals in param_to_values.items():
            arr = np.array(vals, dtype=float)
            arr = arr[np.isfinite(arr)]
            if arr.size == 0:
                continue
            if method == "IQR":
                q1, q3 = np.percentile(arr, [25, 75])
                iqr = q3 - q1
                low = q1 - 0.1 * iqr
                high = q3 + 0.1 * iqr
            else:  # P05-P95
                low, high = np.percentile(arr, [5, 95])
            recs.append((n, float(low), float(high), int(arr.size)))

        # Populate table
        self.rl_rec_table.setRowCount(0)
        for r, (n, low, high, cnt) in enumerate(sorted(recs)):
            self.rl_rec_table.insertRow(r)
            self.rl_rec_table.setItem(r, 0, QTableWidgetItem(n))
            self.rl_rec_table.setItem(r, 1, QTableWidgetItem(f"{low:.6g}"))
            self.rl_rec_table.setItem(r, 2, QTableWidgetItem(f"{high:.6g}"))
            self.rl_rec_table.setItem(r, 3, QTableWidgetItem(str(cnt)))

    def apply_rl_recommended_ranges_to_table(self):
        rows = self.rl_rec_table.rowCount()
        if rows == 0:
            QMessageBox.information(self, "Apply Ranges", "No recommendations to apply.")
            return
        # Build a map from name to (low, high)
        rec_map = {}
        for r in range(rows):
            name = self.rl_rec_table.item(r, 0).text()
            try:
                low = float(self.rl_rec_table.item(r, 1).text())
                high = float(self.rl_rec_table.item(r, 2).text())
            except Exception:
                continue
            rec_map[name] = (low, high)
        # Apply to parameter table
        for row in range(self.rl_param_table.rowCount()):
            name = self.rl_param_table.item(row, 0).text()
            if name in rec_map:
                low, high = rec_map[name]
                fixed_checkbox = self.rl_param_table.cellWidget(row, 1)
                # Unfix to allow editing bounds
                if fixed_checkbox.isChecked():
                    fixed_checkbox.setChecked(False)
                self.rl_param_table.cellWidget(row, 3).setValue(low)
                self.rl_param_table.cellWidget(row, 4).setValue(high)

