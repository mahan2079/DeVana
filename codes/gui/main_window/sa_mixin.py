from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from workers.SAWorker import SAWorker

class SAOptimizationMixin:
    def create_sa_tab(self):
        """Create the simulated annealing optimization tab"""
        self.sa_tab = QWidget()
        layout = QVBoxLayout(self.sa_tab)
        
        # Create sub-tabs widget
        self.sa_sub_tabs = QTabWidget()

        # -------------------- Sub-tab 1: SA Hyperparameters --------------------
        sa_hyper_tab = QWidget()
        sa_hyper_layout = QFormLayout(sa_hyper_tab)

        self.sa_initial_temp_box = QDoubleSpinBox()
        self.sa_initial_temp_box.setRange(0, 1e6)
        self.sa_initial_temp_box.setValue(1000)
        self.sa_initial_temp_box.setDecimals(2)

        self.sa_cooling_rate_box = QDoubleSpinBox()
        self.sa_cooling_rate_box.setRange(0, 1)
        self.sa_cooling_rate_box.setValue(0.95)
        self.sa_cooling_rate_box.setDecimals(3)

        self.sa_num_iterations_box = QSpinBox()
        self.sa_num_iterations_box.setRange(1, 10000)
        self.sa_num_iterations_box.setValue(1000)

        self.sa_tol_box = QDoubleSpinBox()
        self.sa_tol_box.setRange(0, 1e6)
        self.sa_tol_box.setValue(1e-3)
        self.sa_tol_box.setDecimals(6)

        self.sa_alpha_box = QDoubleSpinBox()
        self.sa_alpha_box.setRange(0.0, 10.0)
        self.sa_alpha_box.setDecimals(4)
        self.sa_alpha_box.setSingleStep(0.01)
        self.sa_alpha_box.setValue(0.01)

        # Controller Mode
        self.sa_controller_group = QGroupBox("Controller Mode")
        sa_ctrl_layout = QHBoxLayout(self.sa_controller_group)
        self.sa_controller_fixed_radio = QRadioButton("Fixed")
        self.sa_controller_ml_radio = QRadioButton("ML Bandit")
        self.sa_controller_rl_radio = QRadioButton("RL Controller")
        self.sa_controller_fixed_radio.setChecked(True)
        sa_ctrl_layout.addWidget(self.sa_controller_fixed_radio)
        sa_ctrl_layout.addWidget(self.sa_controller_ml_radio)
        sa_ctrl_layout.addWidget(self.sa_controller_rl_radio)

        # ML options
        self.sa_ml_group = QGroupBox("ML Bandit Options")
        ml_layout = QFormLayout(self.sa_ml_group)
        self.sa_ml_ucb_c = QDoubleSpinBox(); self.sa_ml_ucb_c.setRange(0.1, 3.0); self.sa_ml_ucb_c.setDecimals(2); self.sa_ml_ucb_c.setSingleStep(0.05); self.sa_ml_ucb_c.setValue(0.60)
        self.sa_ml_accept_target = QDoubleSpinBox(); self.sa_ml_accept_target.setRange(0.0, 1.0); self.sa_ml_accept_target.setDecimals(2); self.sa_ml_accept_target.setSingleStep(0.05); self.sa_ml_accept_target.setValue(0.30)
        self.sa_step_scale_box = QDoubleSpinBox(); self.sa_step_scale_box.setRange(1e-6, 1.0); self.sa_step_scale_box.setDecimals(6); self.sa_step_scale_box.setSingleStep(0.01); self.sa_step_scale_box.setValue(0.10)
        ml_layout.addRow("ML UCB c:", self.sa_ml_ucb_c)
        ml_layout.addRow("Accept Target:", self.sa_ml_accept_target)
        ml_layout.addRow("Base Step Scale:", self.sa_step_scale_box)

        # RL options
        self.sa_rl_group = QGroupBox("RL Options")
        rl_layout = QFormLayout(self.sa_rl_group)
        self.sa_rl_alpha = QDoubleSpinBox(); self.sa_rl_alpha.setRange(0.0, 1.0); self.sa_rl_alpha.setDecimals(3); self.sa_rl_alpha.setValue(0.1)
        self.sa_rl_gamma = QDoubleSpinBox(); self.sa_rl_gamma.setRange(0.0, 1.0); self.sa_rl_gamma.setDecimals(3); self.sa_rl_gamma.setValue(0.9)
        self.sa_rl_epsilon = QDoubleSpinBox(); self.sa_rl_epsilon.setRange(0.0, 1.0); self.sa_rl_epsilon.setDecimals(3); self.sa_rl_epsilon.setValue(0.2)
        self.sa_rl_decay = QDoubleSpinBox(); self.sa_rl_decay.setRange(0.0, 1.0); self.sa_rl_decay.setDecimals(3); self.sa_rl_decay.setValue(0.95)
        rl_layout.addRow("RL α (learning rate):", self.sa_rl_alpha)
        rl_layout.addRow("RL γ (discount):", self.sa_rl_gamma)
        rl_layout.addRow("RL ε (explore):", self.sa_rl_epsilon)
        rl_layout.addRow("RL ε decay:", self.sa_rl_decay)
        self.sa_ml_group.setVisible(False)
        self.sa_rl_group.setVisible(False)
        self.sa_controller_ml_radio.toggled.connect(lambda _: self.sa_ml_group.setVisible(self.sa_controller_ml_radio.isChecked()))
        self.sa_controller_rl_radio.toggled.connect(lambda _: self.sa_rl_group.setVisible(self.sa_controller_rl_radio.isChecked()))

        sa_hyper_layout.addRow("Initial Temperature:", self.sa_initial_temp_box)
        sa_hyper_layout.addRow("Cooling Rate:", self.sa_cooling_rate_box)
        sa_hyper_layout.addRow("Number of Iterations:", self.sa_num_iterations_box)
        sa_hyper_layout.addRow("Tolerance (tol):", self.sa_tol_box)
        sa_hyper_layout.addRow("Sparsity Penalty (alpha):", self.sa_alpha_box)
        sa_hyper_layout.addRow(self.sa_controller_group)
        sa_hyper_layout.addRow(self.sa_ml_group)
        sa_hyper_layout.addRow(self.sa_rl_group)

        # Add a small Run SA button in the hyperparameters sub-tab
        self.hyper_run_sa_button = QPushButton("Run SA")
        self.hyper_run_sa_button.setFixedWidth(100)
        self.hyper_run_sa_button.clicked.connect(self.run_sa)
        sa_hyper_layout.addRow("Run SA:", self.hyper_run_sa_button)

        # -------------------- Sub-tab 2: DVA Parameters --------------------
        sa_param_tab = QWidget()
        sa_param_layout = QVBoxLayout(sa_param_tab)

        self.sa_param_table = QTableWidget()
        dva_parameters = [
            *[f"beta_{i}" for i in range(1,16)],
            *[f"lambda_{i}" for i in range(1,16)],
            *[f"mu_{i}" for i in range(1,4)],
            *[f"nu_{i}" for i in range(1,16)]
        ]
        self.sa_param_table.setRowCount(len(dva_parameters))
        self.sa_param_table.setColumnCount(5)
        self.sa_param_table.setHorizontalHeaderLabels(
            ["Parameter", "Fixed", "Fixed Value", "Lower Bound", "Upper Bound"]
        )
        self.sa_param_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.sa_param_table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        # Set up table rows
        for row, param in enumerate(dva_parameters):
            param_item = QTableWidgetItem(param)
            param_item.setFlags(Qt.ItemIsEnabled)
            self.sa_param_table.setItem(row, 0, param_item)

            fixed_checkbox = QCheckBox()
            fixed_checkbox.stateChanged.connect(lambda state, r=row: self.toggle_sa_fixed(state, r))
            self.sa_param_table.setCellWidget(row, 1, fixed_checkbox)

            fixed_value_spin = QDoubleSpinBox()
            fixed_value_spin.setRange(-1e6, 1e6)
            fixed_value_spin.setDecimals(6)
            fixed_value_spin.setEnabled(False)
            self.sa_param_table.setCellWidget(row, 2, fixed_value_spin)

            lower_bound_spin = QDoubleSpinBox()
            lower_bound_spin.setRange(-1e6, 1e6)
            lower_bound_spin.setDecimals(6)
            lower_bound_spin.setEnabled(True)
            self.sa_param_table.setCellWidget(row, 3, lower_bound_spin)

            upper_bound_spin = QDoubleSpinBox()
            upper_bound_spin.setRange(-1e6, 1e6)
            upper_bound_spin.setDecimals(6)
            upper_bound_spin.setEnabled(True)
            self.sa_param_table.setCellWidget(row, 4, upper_bound_spin)

            # Default ranges
            if param.startswith("beta_") or param.startswith("lambda_") or param.startswith("nu_"):
                lower_bound_spin.setValue(0.0001)
                upper_bound_spin.setValue(2.5)
            elif param.startswith("mu_"):
                lower_bound_spin.setValue(0.0001)
                upper_bound_spin.setValue(0.75)
            else:
                lower_bound_spin.setValue(0.0)
                upper_bound_spin.setValue(1.0)

        sa_param_layout.addWidget(self.sa_param_table)

        # -------------------- Sub-tab 3: Results --------------------
        sa_results_tab = QWidget()
        sa_results_layout = QVBoxLayout(sa_results_tab)
        
        self.sa_results_text = QTextEdit()
        self.sa_results_text.setReadOnly(True)
        sa_results_layout.addWidget(QLabel("SA Optimization Results:"))
        sa_results_layout.addWidget(self.sa_results_text)

        # Add all sub-tabs to the SA tab widget
        self.sa_sub_tabs.addTab(sa_hyper_tab, "SA Settings")
        self.sa_sub_tabs.addTab(sa_param_tab, "DVA Parameters")
        self.sa_sub_tabs.addTab(sa_results_tab, "Results")

        # Add the SA sub-tabs widget to the main SA tab layout
        layout.addWidget(self.sa_sub_tabs)
        self.sa_tab.setLayout(layout)
        
    def toggle_sa_fixed(self, state, row, table=None):
        """Toggle the fixed state of a SA parameter row"""
        if table is None:
            table = self.sa_param_table
            
        fixed = (state == Qt.Checked)
        fixed_value_spin = table.cellWidget(row, 2)
        lower_bound_spin = table.cellWidget(row, 3)
        upper_bound_spin = table.cellWidget(row, 4)

        fixed_value_spin.setEnabled(fixed)
        lower_bound_spin.setEnabled(not fixed)
        upper_bound_spin.setEnabled(not fixed)
        
    def run_sa(self):
        """Run the simulated annealing optimization"""
        try:
            if hasattr(self, 'sa_worker') and self.sa_worker.isRunning():
                QMessageBox.warning(self, "Process Running", "A Simulated Annealing run is already in progress.")
                return
            # Gather parameter data
            sa_parameter_data = []
            row_count = self.sa_param_table.rowCount()
            for row in range(row_count):
                name = self.sa_param_table.item(row, 0).text()
                fixed = self.sa_param_table.cellWidget(row, 1).isChecked()
                if fixed:
                    val = self.sa_param_table.cellWidget(row, 2).value()
                    sa_parameter_data.append((name, val, val, True))
                else:
                    low = self.sa_param_table.cellWidget(row, 3).value()
                    high = self.sa_param_table.cellWidget(row, 4).value()
                    sa_parameter_data.append((name, low, high, False))

            main_params = self.get_main_system_params()
            target_values, weights = self.get_target_values_weights()
            omega_start_val = self.omega_start_box.value()
            omega_end_val = self.omega_end_box.value()
            omega_points_val = self.omega_points_box.value()

            # Controller selection
            use_ml = self.sa_controller_ml_radio.isChecked()
            use_rl = self.sa_controller_rl_radio.isChecked()

            self.sa_results_text.clear()
            self.sa_results_text.append("Starting SA optimization...")

            self.sa_worker = SAWorker(
                main_params=main_params,
                target_values_dict=target_values,
                weights_dict=weights,
                omega_start=omega_start_val,
                omega_end=omega_end_val,
                omega_points=omega_points_val,
                sa_initial_temp=self.sa_initial_temp_box.value(),
                sa_cooling_rate=self.sa_cooling_rate_box.value(),
                sa_num_iterations=self.sa_num_iterations_box.value(),
                sa_tol=self.sa_tol_box.value(),
                sa_parameter_data=sa_parameter_data,
                alpha=self.sa_alpha_box.value(),
                track_metrics=True,
                use_ml_adaptive=bool(use_ml and not use_rl),
                ml_ucb_c=self.sa_ml_ucb_c.value() if use_ml else 0.6,
                ml_accept_target=self.sa_ml_accept_target.value() if use_ml else 0.3,
                use_rl_controller=bool(use_rl and not use_ml),
                rl_alpha=self.sa_rl_alpha.value() if use_rl else 0.1,
                rl_gamma=self.sa_rl_gamma.value() if use_rl else 0.9,
                rl_epsilon=self.sa_rl_epsilon.value() if use_rl else 0.2,
                rl_epsilon_decay=self.sa_rl_decay.value() if use_rl else 0.95,
                step_scale=self.sa_step_scale_box.value()
            )
            self.sa_worker.update.connect(lambda msg: self.sa_results_text.append(msg))
            self.sa_worker.error.connect(lambda err: QMessageBox.critical(self, "SA Error", err))
            self.sa_worker.finished.connect(self._handle_sa_finished)
            self.sa_worker.progress.connect(lambda p: None)
            try:
                self.sa_worker.benchmark_data.connect(lambda m: None)
                self.sa_worker.generation_metrics.connect(lambda m: None)
            except Exception:
                pass
            self.sa_worker.start()
        except Exception as e:
            QMessageBox.critical(self, "SA Error", str(e))
        
    def run_cmaes(self):
        """Run the CMA-ES optimization"""
        # Implementation already exists at line 2840
        pass

    def _handle_sa_finished(self, results, best_candidate, parameter_names, best_fitness):
        try:
            self.sa_results_text.append("\n=== SA Optimization Complete ===")
            self.sa_results_text.append(f"Best fitness: {best_fitness:.6f}")
            self.sa_results_text.append("Best parameters:")
            for n, v in zip(parameter_names, best_candidate):
                self.sa_results_text.append(f"  {n}: {float(v):.6f}")
            metrics = results.get('benchmark_metrics', {}) if isinstance(results, dict) else {}
            if metrics:
                if metrics.get('best_fitness_per_gen'):
                    self.sa_results_text.append(f"Final best fitness (metrics): {metrics['best_fitness_per_gen'][-1]:.6f}")
                if metrics.get('total_duration'):
                    self.sa_results_text.append(f"Duration: {float(metrics['total_duration']):.2f}s")
        except Exception as e:
            QMessageBox.warning(self, "SA Results", f"Error processing SA results: {str(e)}")
        
