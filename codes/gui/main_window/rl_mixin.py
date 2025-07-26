from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from RL.RLWorker import RLWorker

class RLOptimizationMixin:
    def create_rl_tab(self):
        """Create the reinforcement learning optimization tab"""
        self.rl_tab = QWidget()
        layout = QVBoxLayout(self.rl_tab)

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
        layout.addWidget(hyper_group)

        reward_group = QGroupBox("Reward Settings")
        reward_form = QFormLayout(reward_group)
        self.rl_reward_system_combo = QComboBox()
        self.rl_reward_system_combo.addItems([
            "1 - Absolute Error",
            "2 - Error + Cost",
            "3 - Error + Sparsity Count",
            "4 - Error + Sparsity Sum",
            "5 - Error + Extended Cost"
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
        layout.addWidget(reward_group)

        self.run_rl_button = QPushButton("Run RL Optimization")
        self.run_rl_button.clicked.connect(self.run_rl_optimization)
        layout.addWidget(self.run_rl_button)

        self.rl_results_text = QTextEdit()
        self.rl_results_text.setReadOnly(True)
        layout.addWidget(self.rl_results_text)

    def run_rl_optimization(self):
        try:
            main_params = self.get_main_system_params()
            target_values, weights = self.get_target_values_weights()
            dva_bounds = {}
            EPSILON = 1e-6
            for row in range(self.ga_param_table.rowCount()):
                param_name = self.ga_param_table.item(row, 0).text()
                fixed = self.ga_param_table.cellWidget(row, 1).isChecked()
                if fixed:
                    fixed_value = self.ga_param_table.cellWidget(row, 2).value()
                    dva_bounds[param_name] = (fixed_value, fixed_value + EPSILON)
                else:
                    lower = self.ga_param_table.cellWidget(row, 3).value()
                    upper = self.ga_param_table.cellWidget(row, 4).value()
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
            cost_values = [cost_coeff] * len(rl_parameter_data)

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
