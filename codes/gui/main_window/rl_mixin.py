from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from RL.RLWorker import RLWorker


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

        hyper_group = QGroupBox("RL Hyperparameters")
        hyper_form = QFormLayout(hyper_group)

        self.rl_num_episodes_box = QSpinBox()
        self.rl_num_episodes_box.setRange(1, 10000)
        self.rl_num_episodes_box.setValue(100)

        self.rl_max_steps_box = QSpinBox()
        self.rl_max_steps_box.setRange(1, 10000)
        self.rl_max_steps_box.setValue(50)

        self.rl_alpha_box = QDoubleSpinBox()
        self.rl_alpha_box.setRange(0.0, 1.0)
        self.rl_alpha_box.setDecimals(4)
        self.rl_alpha_box.setValue(0.1)

        self.rl_gamma_box = QDoubleSpinBox()
        self.rl_gamma_box.setRange(0.0, 1.0)
        self.rl_gamma_box.setDecimals(4)
        self.rl_gamma_box.setValue(0.95)

        self.rl_epsilon_box = QDoubleSpinBox()
        self.rl_epsilon_box.setRange(0.0, 1.0)
        self.rl_epsilon_box.setDecimals(4)
        self.rl_epsilon_box.setValue(1.0)

        self.rl_epsilon_min_box = QDoubleSpinBox()
        self.rl_epsilon_min_box.setRange(0.0, 1.0)
        self.rl_epsilon_min_box.setDecimals(4)
        self.rl_epsilon_min_box.setValue(0.05)

        self.rl_epsilon_decay_box = QDoubleSpinBox()
        self.rl_epsilon_decay_box.setRange(0.0, 1.0)
        self.rl_epsilon_decay_box.setDecimals(4)
        self.rl_epsilon_decay_box.setValue(0.99)

        self.rl_epsilon_decay_type_combo = QComboBox()
        self.rl_epsilon_decay_type_combo.addItems(["exponential", "linear"])

        hyper_form.addRow("Episodes:", self.rl_num_episodes_box)
        hyper_form.addRow("Max Steps:", self.rl_max_steps_box)
        hyper_form.addRow("Learning Rate (alpha):", self.rl_alpha_box)
        hyper_form.addRow("Discount (gamma):", self.rl_gamma_box)
        hyper_form.addRow("Epsilon:", self.rl_epsilon_box)
        hyper_form.addRow("Min Epsilon:", self.rl_epsilon_min_box)
        hyper_form.addRow("Epsilon Decay:", self.rl_epsilon_decay_box)
        hyper_form.addRow("Decay Type:", self.rl_epsilon_decay_type_combo)

        hyper_layout.addWidget(hyper_group)

        reward_group = QGroupBox("Reward Settings")
        reward_form = QFormLayout(reward_group)

        self.rl_reward_system_combo = QComboBox()
        self.rl_reward_system_combo.addItems([
            "1 - Absolute Error",
            "2 - Error + Cost",
            "3 - Error + Sparsity Count",
            "4 - Error + Sparsity Sum",
            "5 - Error + Extended Cost",
        ])

        self.rl_cost_box = QDoubleSpinBox()
        self.rl_cost_box.setRange(0.0, 1000.0)
        self.rl_cost_box.setDecimals(4)
        self.rl_cost_box.setValue(0.0)

        self.rl_sparsity_box = QDoubleSpinBox()
        self.rl_sparsity_box.setRange(0.0, 1.0)
        self.rl_sparsity_box.setDecimals(4)
        self.rl_sparsity_box.setValue(0.01)

        self.rl_time_penalty_box = QDoubleSpinBox()
        self.rl_time_penalty_box.setRange(0.0, 10.0)
        self.rl_time_penalty_box.setDecimals(4)
        self.rl_time_penalty_box.setValue(0.0)

        reward_form.addRow("Reward System:", self.rl_reward_system_combo)
        reward_form.addRow("Cost Coefficient:", self.rl_cost_box)
        reward_form.addRow("Sparsity Penalty:", self.rl_sparsity_box)
        reward_form.addRow("Time Penalty:", self.rl_time_penalty_box)

        hyper_layout.addWidget(reward_group)

        self.run_rl_button = QPushButton("Run RL")
        self.run_rl_button.clicked.connect(self.run_rl_optimization)
        hyper_layout.addWidget(self.run_rl_button)

        self.rl_sub_tabs.addTab(rl_hyper_tab, "Hyperparameters")

        # ------------------- Sub-tab 2: Parameters -------------------
        param_tab = QWidget()
        param_layout = QVBoxLayout(param_tab)

        self.rl_param_table = QTableWidget()
        dva_parameters = [
            *[f"beta_{i}" for i in range(1, 16)],
            *[f"lambda_{i}" for i in range(1, 16)],
            *[f"mu_{i}" for i in range(1, 4)],
            *[f"nu_{i}" for i in range(1, 16)],
        ]

        self.rl_param_table.setRowCount(len(dva_parameters))
        self.rl_param_table.setColumnCount(6)
        self.rl_param_table.setHorizontalHeaderLabels([
            "Parameter",
            "Fixed",
            "Fixed Value",
            "Lower Bound",
            "Upper Bound",
            "Cost",
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

            cost_spin = QDoubleSpinBox()
            cost_spin.setRange(0.0, 1000.0)
            cost_spin.setDecimals(4)
            cost_spin.setValue(1.0)
            self.rl_param_table.setCellWidget(row, 5, cost_spin)

        param_layout.addWidget(self.rl_param_table)

        self.rl_sub_tabs.addTab(param_tab, "Parameters")

        # ------------------- Sub-tab 3: Results -------------------
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)

        self.rl_results_text = QTextEdit()
        self.rl_results_text.setReadOnly(True)
        results_layout.addWidget(self.rl_results_text)

        # Plot for episode rewards
        self.rl_reward_fig = Figure(figsize=(5, 3))
        self.rl_reward_canvas = FigureCanvas(self.rl_reward_fig)
        results_layout.addWidget(self.rl_reward_canvas)

        self.rl_sub_tabs.addTab(results_tab, "Results")

    def run_rl_optimization(self):
        try:
            main_params = self.get_main_system_params()
            target_values, weights = self.get_target_values_weights()
            self.rl_reward_history = []
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
                            f"For parameter {param_name}, lower bound is greater than upper bound.")
                        return
                    dva_bounds[param_name] = (lower, upper)
            param_order = [
                'beta_1','beta_2','beta_3','beta_4','beta_5','beta_6','beta_7','beta_8','beta_9','beta_10','beta_11','beta_12','beta_13','beta_14','beta_15',
                'lambda_1','lambda_2','lambda_3','lambda_4','lambda_5','lambda_6','lambda_7','lambda_8','lambda_9','lambda_10','lambda_11','lambda_12','lambda_13','lambda_14','lambda_15',
                'mu_1','mu_2','mu_3',
                'nu_1','nu_2','nu_3','nu_4','nu_5','nu_6','nu_7','nu_8','nu_9','nu_10','nu_11','nu_12','nu_13','nu_14','nu_15'
            ]
            rl_parameter_data = []
            for name in param_order:
                if name in dva_bounds:
                    low, high = dva_bounds[name]
                    fixed = abs(low - high) < EPSILON
                    rl_parameter_data.append((name, low, high, fixed))

            cost_coeff = self.rl_cost_box.value()
            cost_values = []
            for row in range(self.rl_param_table.rowCount()):
                cost_spin = self.rl_param_table.cellWidget(row, 5)
                if cost_spin is not None:
                    cost_values.append(cost_coeff * cost_spin.value())
                else:
                    cost_values.append(cost_coeff)

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
                rl_reward_system=self.rl_reward_system_combo.currentIndex() + 1,
                cost_values=cost_values,
                alpha_sparsity_simplified=self.rl_sparsity_box.value(),
                alpha_sparsity=self.rl_sparsity_box.value(),
                extended_cost_values=cost_values,
                time_penalty_weight=self.rl_time_penalty_box.value()
            )
            self.rl_worker.finished.connect(self.handle_rl_finished)
            self.rl_worker.error.connect(self.handle_rl_error)
            self.rl_worker.update.connect(self.handle_rl_update)
            self.rl_worker.episode_metrics.connect(self.handle_rl_metrics)

            self.run_rl_button.setEnabled(False)
            self.rl_results_text.clear()
            self.rl_results_text.append("Running RL optimization...")
            self.rl_worker.start()
        except Exception as e:
            self.handle_rl_error(str(e))

    def handle_rl_finished(self, results, best_params, param_names, best_reward):
        self.run_rl_button.setEnabled(True)
        best_dict = {n: v for n, v in zip(param_names, best_params)}
        self.current_rl_best_params = best_dict
        self.current_rl_best_reward = best_reward
        for name, val in best_dict.items():
            self.rl_results_text.append(f"{name}: {val:.6f}")
        self.rl_results_text.append(f"Best Reward: {best_reward:.6f}")
        if isinstance(results, dict) and 'singular_response' in results:
            self.rl_results_text.append(f"Singular Response: {results['singular_response']:.6f}")
        self.status_bar.showMessage("RL optimization completed")

    def handle_rl_error(self, err):
        self.run_rl_button.setEnabled(True)
        QMessageBox.warning(self, "RL Error", f"Error during RL optimization: {err}")
        self.rl_results_text.append(f"Error: {err}")
        self.status_bar.showMessage("RL optimization failed")

    def handle_rl_update(self, msg):
        self.rl_results_text.append(msg)

    def handle_rl_metrics(self, metrics):
        """Update reward plot based on episode metrics"""
        episode = metrics.get('episode')
        reward = metrics.get('best_reward')
        if not hasattr(self, 'rl_reward_history'):
            self.rl_reward_history = []
        self.rl_reward_history.append((episode, reward))

        self.rl_reward_fig.clear()
        ax = self.rl_reward_fig.add_subplot(111)
        episodes = [e for e, _ in self.rl_reward_history]
        rewards = [r for _, r in self.rl_reward_history]
        ax.plot(episodes, rewards, marker='o')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Best Reward')
        ax.grid(True)
        self.rl_reward_canvas.draw()

    def toggle_rl_fixed(self, state, row):
        fixed = (state == Qt.Checked)
        fixed_value_spin = self.rl_param_table.cellWidget(row, 2)
        lower_spin = self.rl_param_table.cellWidget(row, 3)
        upper_spin = self.rl_param_table.cellWidget(row, 4)

        fixed_value_spin.setEnabled(fixed)
        lower_spin.setEnabled(not fixed)
        upper_spin.setEnabled(not fixed)

        if fixed:
            fixed_value_spin.setValue(lower_spin.value())
        else:
            if lower_spin.value() > upper_spin.value():
                upper_spin.setValue(lower_spin.value())

