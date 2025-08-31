from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from workers.CMAESWorker import CMAESWorker

class CMAESOptimizationMixin:
    def create_cmaes_tab(self):
        """Create the CMA-ES optimization tab"""
        self.cmaes_tab = QWidget()
        layout = QVBoxLayout(self.cmaes_tab)

        # Create sub-tabs widget
        self.cmaes_sub_tabs = QTabWidget()

        # -------------------- Sub-tab 1: CMA-ES Hyperparameters --------------------
        cmaes_hyper_tab = QWidget()
        cmaes_hyper_layout = QFormLayout(cmaes_hyper_tab)

        self.cmaes_sigma_box = QDoubleSpinBox()
        self.cmaes_sigma_box.setRange(0, 1e6)
        self.cmaes_sigma_box.setValue(0.5)
        self.cmaes_sigma_box.setDecimals(2)

        self.cmaes_max_iter_box = QSpinBox()
        self.cmaes_max_iter_box.setRange(1, 10000)
        self.cmaes_max_iter_box.setValue(500)

        self.cmaes_tol_box = QDoubleSpinBox()
        self.cmaes_tol_box.setRange(0, 1e6)
        self.cmaes_tol_box.setValue(1e-3)
        self.cmaes_tol_box.setDecimals(6)

        self.cmaes_alpha_box = QDoubleSpinBox()
        self.cmaes_alpha_box.setRange(0.0, 10.0)
        self.cmaes_alpha_box.setDecimals(4)
        self.cmaes_alpha_box.setSingleStep(0.01)
        self.cmaes_alpha_box.setValue(0.01)

        # Controller Mode (parity)
        self.cmaes_controller_group = QGroupBox("Controller Mode")
        ctrl_layout = QHBoxLayout(self.cmaes_controller_group)
        self.cmaes_controller_fixed_radio = QRadioButton("Fixed")
        self.cmaes_controller_ml_radio = QRadioButton("ML Bandit")
        self.cmaes_controller_rl_radio = QRadioButton("RL Controller")
        self.cmaes_controller_fixed_radio.setChecked(True)
        ctrl_layout.addWidget(self.cmaes_controller_fixed_radio)
        ctrl_layout.addWidget(self.cmaes_controller_ml_radio)
        ctrl_layout.addWidget(self.cmaes_controller_rl_radio)

        # ML options
        self.cmaes_ml_group = QGroupBox("ML Bandit Options")
        ml_layout = QFormLayout(self.cmaes_ml_group)
        self.cmaes_ml_ucb_c = QDoubleSpinBox(); self.cmaes_ml_ucb_c.setRange(0.1, 3.0); self.cmaes_ml_ucb_c.setDecimals(2); self.cmaes_ml_ucb_c.setSingleStep(0.05); self.cmaes_ml_ucb_c.setValue(0.60)
        self.cmaes_sigma_scale = QDoubleSpinBox(); self.cmaes_sigma_scale.setRange(1e-6, 10.0); self.cmaes_sigma_scale.setDecimals(6); self.cmaes_sigma_scale.setSingleStep(0.01); self.cmaes_sigma_scale.setValue(1.0)
        ml_layout.addRow("ML UCB c:", self.cmaes_ml_ucb_c)
        ml_layout.addRow("Base Sigma Scale:", self.cmaes_sigma_scale)

        # RL options
        self.cmaes_rl_group = QGroupBox("RL Options")
        rl_layout = QFormLayout(self.cmaes_rl_group)
        self.cmaes_rl_alpha = QDoubleSpinBox(); self.cmaes_rl_alpha.setRange(0.0, 1.0); self.cmaes_rl_alpha.setDecimals(3); self.cmaes_rl_alpha.setValue(0.1)
        self.cmaes_rl_gamma = QDoubleSpinBox(); self.cmaes_rl_gamma.setRange(0.0, 1.0); self.cmaes_rl_gamma.setDecimals(3); self.cmaes_rl_gamma.setValue(0.9)
        self.cmaes_rl_epsilon = QDoubleSpinBox(); self.cmaes_rl_epsilon.setRange(0.0, 1.0); self.cmaes_rl_epsilon.setDecimals(3); self.cmaes_rl_epsilon.setValue(0.2)
        self.cmaes_rl_decay = QDoubleSpinBox(); self.cmaes_rl_decay.setRange(0.0, 1.0); self.cmaes_rl_decay.setDecimals(3); self.cmaes_rl_decay.setValue(0.95)
        rl_layout.addRow("RL α (learning rate):", self.cmaes_rl_alpha)
        rl_layout.addRow("RL γ (discount):", self.cmaes_rl_gamma)
        rl_layout.addRow("RL ε (explore):", self.cmaes_rl_epsilon)
        rl_layout.addRow("RL ε decay:", self.cmaes_rl_decay)
        self.cmaes_ml_group.setVisible(False)
        self.cmaes_rl_group.setVisible(False)
        self.cmaes_controller_ml_radio.toggled.connect(lambda _: self.cmaes_ml_group.setVisible(self.cmaes_controller_ml_radio.isChecked()))
        self.cmaes_controller_rl_radio.toggled.connect(lambda _: self.cmaes_rl_group.setVisible(self.cmaes_controller_rl_radio.isChecked()))

        cmaes_hyper_layout.addRow("Initial Sigma:", self.cmaes_sigma_box)
        cmaes_hyper_layout.addRow("Max Iterations:", self.cmaes_max_iter_box)
        cmaes_hyper_layout.addRow("Tolerance (tol):", self.cmaes_tol_box)
        cmaes_hyper_layout.addRow("Sparsity Penalty (alpha):", self.cmaes_alpha_box)
        cmaes_hyper_layout.addRow(self.cmaes_controller_group)
        cmaes_hyper_layout.addRow(self.cmaes_ml_group)
        cmaes_hyper_layout.addRow(self.cmaes_rl_group)

        # Add a small Run CMA-ES button in the hyperparameters sub-tab
        self.hyper_run_cmaes_button = QPushButton("Run CMA-ES")
        self.hyper_run_cmaes_button.setFixedWidth(100)
        self.hyper_run_cmaes_button.clicked.connect(self.run_cmaes)
        cmaes_hyper_layout.addRow("Run CMA-ES:", self.hyper_run_cmaes_button)

        # -------------------- Sub-tab 2: DVA Parameters --------------------
        cmaes_param_tab = QWidget()
        cmaes_param_layout = QVBoxLayout(cmaes_param_tab)

        self.cmaes_param_table = QTableWidget()
        dva_parameters = [
            *[f"beta_{i}" for i in range(1,16)],
            *[f"lambda_{i}" for i in range(1,16)],
            *[f"mu_{i}" for i in range(1,4)],
            *[f"nu_{i}" for i in range(1,16)]
        ]
        self.cmaes_param_table.setRowCount(len(dva_parameters))
        self.cmaes_param_table.setColumnCount(5)
        self.cmaes_param_table.setHorizontalHeaderLabels(
            ["Parameter", "Fixed", "Fixed Value", "Lower Bound", "Upper Bound"]
        )
        self.cmaes_param_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.cmaes_param_table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        # Set up table rows
        for row, param in enumerate(dva_parameters):
            param_item = QTableWidgetItem(param)
            param_item.setFlags(Qt.ItemIsEnabled)
            self.cmaes_param_table.setItem(row, 0, param_item)

            fixed_checkbox = QCheckBox()
            fixed_checkbox.stateChanged.connect(lambda state, r=row: self.toggle_cmaes_fixed(state, r))
            self.cmaes_param_table.setCellWidget(row, 1, fixed_checkbox)

            fixed_value_spin = QDoubleSpinBox()
            fixed_value_spin.setRange(-1e6, 1e6)
            fixed_value_spin.setDecimals(6)
            fixed_value_spin.setEnabled(False)
            self.cmaes_param_table.setCellWidget(row, 2, fixed_value_spin)

            lower_bound_spin = QDoubleSpinBox()
            lower_bound_spin.setRange(-1e6, 1e6)
            lower_bound_spin.setDecimals(6)
            lower_bound_spin.setEnabled(True)
            self.cmaes_param_table.setCellWidget(row, 3, lower_bound_spin)

            upper_bound_spin = QDoubleSpinBox()
            upper_bound_spin.setRange(-1e6, 1e6)
            upper_bound_spin.setDecimals(6)
            upper_bound_spin.setEnabled(True)
            self.cmaes_param_table.setCellWidget(row, 4, upper_bound_spin)

            # Set default ranges based on parameter name
            if param.startswith("beta_") or param.startswith("lambda_") or param.startswith("nu_"):
                lower_bound_spin.setValue(0.0001)
                upper_bound_spin.setValue(2.5)
            elif param.startswith("mu_"):
                lower_bound_spin.setValue(0.0001)
                upper_bound_spin.setValue(0.75)
            else:
                lower_bound_spin.setValue(0.0)
                upper_bound_spin.setValue(1.0)

        cmaes_param_layout.addWidget(self.cmaes_param_table)

        # -------------------- Sub-tab 3: Results --------------------
        cmaes_results_tab = QWidget()
        cmaes_results_layout = QVBoxLayout(cmaes_results_tab)
        
        self.cmaes_results_text = QTextEdit()
        self.cmaes_results_text.setReadOnly(True)
        cmaes_results_layout.addWidget(QLabel("CMA-ES Optimization Results:"))
        cmaes_results_layout.addWidget(self.cmaes_results_text)

        # Add all sub-tabs to the CMA-ES tab widget
        self.cmaes_sub_tabs.addTab(cmaes_hyper_tab, "CMA-ES Settings")
        self.cmaes_sub_tabs.addTab(cmaes_param_tab, "DVA Parameters")
        self.cmaes_sub_tabs.addTab(cmaes_results_tab, "Results")

        # Add the CMA-ES sub-tabs widget to the main CMA-ES tab layout
        layout.addWidget(self.cmaes_sub_tabs)
        self.cmaes_tab.setLayout(layout)

    def toggle_cmaes_fixed(self, state, row, table=None):
        """Toggle the fixed state of a CMA-ES parameter row"""
        if table is None:
            table = self.cmaes_param_table
        
        fixed = (state == Qt.Checked)
        fixed_value_spin = table.cellWidget(row, 2)
        lower_bound_spin = table.cellWidget(row, 3)
        upper_bound_spin = table.cellWidget(row, 4)

        fixed_value_spin.setEnabled(fixed)
        lower_bound_spin.setEnabled(not fixed)
        upper_bound_spin.setEnabled(not fixed)

    def run_cmaes(self):
        """Run the CMA-ES optimization"""
        try:
            if hasattr(self, 'cmaes_worker') and self.cmaes_worker.isRunning():
                QMessageBox.warning(self, "Process Running", "CMA-ES is already running.")
                return
            # Gather parameter data
            cma_parameter_data = []
            row_count = self.cmaes_param_table.rowCount()
            for row in range(row_count):
                name = self.cmaes_param_table.item(row, 0).text()
                fixed = self.cmaes_param_table.cellWidget(row, 1).isChecked()
                if fixed:
                    val = self.cmaes_param_table.cellWidget(row, 2).value()
                    cma_parameter_data.append((name, val, val, True))
                else:
                    low = self.cmaes_param_table.cellWidget(row, 3).value()
                    high = self.cmaes_param_table.cellWidget(row, 4).value()
                    cma_parameter_data.append((name, low, high, False))

            main_params = self.get_main_system_params()
            target_values, weights = self.get_target_values_weights()
            omega_start_val = self.omega_start_box.value()
            omega_end_val = self.omega_end_box.value()
            omega_points_val = self.omega_points_box.value()

            use_ml = self.cmaes_controller_ml_radio.isChecked()
            use_rl = self.cmaes_controller_rl_radio.isChecked()

            self.cmaes_results_text.clear()
            self.cmaes_results_text.append("Starting CMA-ES optimization...")

            self.cmaes_worker = CMAESWorker(
                main_params=main_params,
                target_values_dict=target_values,
                weights_dict=weights,
                omega_start=omega_start_val,
                omega_end=omega_end_val,
                omega_points=omega_points_val,
                cma_initial_sigma=self.cmaes_sigma_box.value(),
                cma_max_iter=self.cmaes_max_iter_box.value(),
                cma_tol=self.cmaes_tol_box.value(),
                cma_parameter_data=cma_parameter_data,
                alpha=self.cmaes_alpha_box.value(),
                track_metrics=True,
                use_ml_adaptive=bool(use_ml and not use_rl),
                ml_ucb_c=self.cmaes_ml_ucb_c.value() if use_ml else 0.6,
                use_rl_controller=bool(use_rl and not use_ml),
                rl_alpha=self.cmaes_rl_alpha.value() if use_rl else 0.1,
                rl_gamma=self.cmaes_rl_gamma.value() if use_rl else 0.9,
                rl_epsilon=self.cmaes_rl_epsilon.value() if use_rl else 0.2,
                rl_epsilon_decay=self.cmaes_rl_decay.value() if use_rl else 0.95,
                sigma_scale=self.cmaes_sigma_scale.value()
            )
            self.cmaes_worker.update.connect(lambda msg: self.cmaes_results_text.append(msg))
            self.cmaes_worker.error.connect(lambda err: QMessageBox.critical(self, "CMA-ES Error", err))
            self.cmaes_worker.finished.connect(self.handle_cmaes_finished)
            self.cmaes_worker.progress.connect(lambda p: None)
            try:
                self.cmaes_worker.benchmark_data.connect(lambda m: None)
                self.cmaes_worker.generation_metrics.connect(lambda m: None)
            except Exception:
                pass
            self.cmaes_worker.start()
        except Exception as e:
            QMessageBox.critical(self, "CMA-ES Error", str(e))

    def handle_cmaes_finished(self, results, best_candidate, parameter_names, best_fitness):
        """Handle the completion of CMA-ES optimization"""
        # Re-enable both run CMA-ES buttons
        self.hyper_run_cmaes_button.setEnabled(True)
        self.run_cmaes_button.setEnabled(True)
        
        self.cmaes_results_text.append("\nCMA-ES Completed.\n")
        self.cmaes_results_text.append("Best candidate parameters:")

        for name, val in zip(parameter_names, best_candidate):
            self.cmaes_results_text.append(f"{name}: {val}")
        self.cmaes_results_text.append(f"\nBest fitness: {best_fitness:.6f}")

        singular_response = results.get('singular_response', None)
        if singular_response is not None:
            self.cmaes_results_text.append(f"\nSingular response of best candidate: {singular_response}")

        self.cmaes_results_text.append("\nFull Results:")
        for section, data in results.items():
            self.cmaes_results_text.append(f"{section}: {data}")
            
        self.status_bar.showMessage("CMA-ES optimization completed")

    def handle_cmaes_error(self, err):
        """Handle errors during CMA-ES optimization"""
        # Re-enable both run CMA-ES buttons
        self.hyper_run_cmaes_button.setEnabled(True)
        self.run_cmaes_button.setEnabled(True)
        
        QMessageBox.warning(self, "CMA-ES Error", f"Error during CMA-ES optimization: {err}")
        self.cmaes_results_text.append(f"\nError running CMA-ES: {err}")
        self.status_bar.showMessage("CMA-ES optimization failed")

    def handle_cmaes_update(self, msg):
        """Handle progress updates from CMA-ES worker"""
        self.cmaes_results_text.append(msg)

    # Comprehensive Analysis create_rl_tab method has been removed

    # RL-related method removed

    # RL-related method removed

    # RL-related method removed

    # RL-related method removed
    
