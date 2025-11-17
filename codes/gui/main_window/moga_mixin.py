from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QDoubleSpinBox,
    QFormLayout, QGroupBox, QPushButton, QTabWidget, QTextEdit, QProgressBar,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QAbstractItemView, QCheckBox
)
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from workers.NSGA2Worker import NSGA2Worker
import numpy as np

class MOGAOptimizationMixin:
    def create_moga_tab(self):
        """Create the MOGA optimization tab, which will contain sub-tabs for different MOGA algorithms."""
        self.moga_main_tab = QWidget()
        main_layout = QVBoxLayout(self.moga_main_tab)

        # Create a tab widget to hold different MOGA methods
        self.moga_method_tabs = QTabWidget()
        main_layout.addWidget(self.moga_method_tabs)

        # Create the NSGA-II tab
        self.create_nsga2_sub_tab()
        self.moga_method_tabs.addTab(self.nsga2_tab, "NSGA-II")

        # Other MOGA methods can be added here as new tabs in the future

        return self.moga_main_tab

    def create_nsga2_sub_tab(self):
        """Create the NSGA-II optimization tab with Settings, Live Calculations, and Results sub-tabs."""
        self.nsga2_tab = QWidget()
        main_layout = QVBoxLayout(self.nsga2_tab)

        self.nsga2_sub_tabs = QTabWidget()
        main_layout.addWidget(self.nsga2_sub_tabs)

        # --- Settings Sub-tab ---
        self.create_nsga2_settings_tab()
        self.nsga2_sub_tabs.addTab(self.nsga2_settings_tab, "Settings")

        # --- Live Calculations Sub-tab ---
        self.create_nsga2_live_calc_tab()
        self.nsga2_sub_tabs.addTab(self.nsga2_live_calc_tab, "Live Calculations")

        # --- Results Sub-tab ---
        self.create_nsga2_results_tab()
        self.nsga2_sub_tabs.addTab(self.nsga2_results_tab, "Results")

        # --- Statistics Sub-tab ---
        self.create_nsga2_statistics_tab()
        self.nsga2_sub_tabs.addTab(self.nsga2_statistics_tab, "Statistics")

    def create_nsga2_settings_tab(self):
        """Creates the 'Settings' sub-tab for NSGA-II."""
        self.nsga2_settings_tab = QWidget()
        layout = QVBoxLayout(self.nsga2_settings_tab) # Changed to QVBoxLayout to stack groups

        # NSGA-II Parameters Group
        nsga2_params_group = QGroupBox("NSGA-II Algorithm Parameters")
        nsga2_params_layout = QFormLayout(nsga2_params_group)

        self.nsga2_pop_size_box = QSpinBox()
        self.nsga2_pop_size_box.setRange(10, 1000)
        self.nsga2_pop_size_box.setValue(100)
        nsga2_params_layout.addRow("Population Size (N):", self.nsga2_pop_size_box)

        self.nsga2_generations_box = QSpinBox()
        self.nsga2_generations_box.setRange(10, 10000)
        self.nsga2_generations_box.setValue(2000)
        nsga2_params_layout.addRow("Generations (G_max):", self.nsga2_generations_box)

        self.nsga2_cxpb_box = QDoubleSpinBox()
        self.nsga2_cxpb_box.setRange(0.0, 1.0)
        self.nsga2_cxpb_box.setSingleStep(0.01)
        self.nsga2_cxpb_box.setValue(0.9)
        nsga2_params_layout.addRow("Crossover Probability (p_c):", self.nsga2_cxpb_box)

        self.nsga2_mutpb_box = QDoubleSpinBox()
        self.nsga2_mutpb_box.setRange(0.0, 1.0)
        self.nsga2_mutpb_box.setSingleStep(0.01)
        self.nsga2_mutpb_box.setValue(1/48)
        nsga2_params_layout.addRow("Mutation Probability (p_m):", self.nsga2_mutpb_box)
        
        self.nsga2_eta_c_box = QSpinBox()
        self.nsga2_eta_c_box.setRange(1, 100)
        self.nsga2_eta_c_box.setValue(20)
        nsga2_params_layout.addRow("SBX Crossover (eta_c):", self.nsga2_eta_c_box)

        self.nsga2_eta_m_box = QSpinBox()
        self.nsga2_eta_m_box.setRange(1, 100)
        self.nsga2_eta_m_box.setValue(20)
        nsga2_params_layout.addRow("Polynomial Mutation (eta_m):", self.nsga2_eta_m_box)

        self.nsga2_indpb_box = QDoubleSpinBox()
        self.nsga2_indpb_box.setRange(0.0, 1.0)
        self.nsga2_indpb_box.setSingleStep(0.01)
        self.nsga2_indpb_box.setValue(0.1)
        nsga2_params_layout.addRow("Individual Mutation Probability (indpb):", self.nsga2_indpb_box)
        layout.addWidget(nsga2_params_group)

        # Sparsity Parameters Group
        sparsity_group = QGroupBox("Sparsity Parameters (f2)")
        sparsity_layout = QFormLayout(sparsity_group)
        self.nsga2_sparsity_tau = QDoubleSpinBox()
        self.nsga2_sparsity_tau.setRange(0.0, 1.0)
        self.nsga2_sparsity_tau.setSingleStep(0.01)
        self.nsga2_sparsity_tau.setValue(0.1)
        sparsity_layout.addRow("Threshold (tau):", self.nsga2_sparsity_tau)
        self.nsga2_sparsity_alpha = QDoubleSpinBox()
        self.nsga2_sparsity_alpha.setRange(0.0, 10.0)
        self.nsga2_sparsity_alpha.setSingleStep(0.1)
        self.nsga2_sparsity_alpha.setValue(1.0)
        sparsity_layout.addRow("Cardinality Weight (alpha):", self.nsga2_sparsity_alpha)
        self.nsga2_sparsity_beta = QDoubleSpinBox()
        self.nsga2_sparsity_beta.setRange(0.0, 10.0)
        self.nsga2_sparsity_beta.setSingleStep(0.1)
        self.nsga2_sparsity_beta.setValue(0.5)
        sparsity_layout.addRow("Magnitude Weight (beta):", self.nsga2_sparsity_beta)
        layout.addWidget(sparsity_group)

        # Multi-Run Settings
        multi_run_group = QGroupBox("Multi-Run Settings")
        multi_run_layout = QFormLayout(multi_run_group)
        self.nsga2_multi_run_checkbox = QCheckBox("Enable Multi-Run")
        self.nsga2_multi_run_checkbox.setChecked(False)
        multi_run_layout.addRow(self.nsga2_multi_run_checkbox)
        self.nsga2_num_runs_box = QSpinBox()
        self.nsga2_num_runs_box.setRange(1, 100)
        self.nsga2_num_runs_box.setValue(10)
        multi_run_layout.addRow("Number of Runs:", self.nsga2_num_runs_box)
        layout.addWidget(multi_run_group)

        # DVA Parameters Table
        dva_params_group = QGroupBox("DVA Parameters (x) and Costs (f3)")
        dva_params_layout = QVBoxLayout(dva_params_group)

        self.nsga2_param_table = QTableWidget()
        dva_parameters = [
            *[f"beta_{i}" for i in range(1,16)],
            *[f"lambda_{i}" for i in range(1,16)],
            *[f"mu_{i}" for i in range(1,4)],
            *[f"nu_{i}" for i in range(1,16)]
        ]
        self.nsga2_param_table.setRowCount(len(dva_parameters))
        self.nsga2_param_table.setColumnCount(6) # Name, Fixed, Fixed Value, Lower Bound, Upper Bound, Cost
        self.nsga2_param_table.setHorizontalHeaderLabels(
            ["Parameter", "Fixed", "Fixed Value", "Lower Bound", "Upper Bound", "Cost (c_i)"]
        )
        self.nsga2_param_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.nsga2_param_table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        for row, param in enumerate(dva_parameters):
            param_item = QTableWidgetItem(param)
            param_item.setFlags(Qt.ItemIsEnabled)
            self.nsga2_param_table.setItem(row, 0, param_item)

            fixed_checkbox = QCheckBox()
            fixed_checkbox.setChecked(True)  # Set fixed to true by default
            fixed_checkbox.stateChanged.connect(lambda state, r=row: self.toggle_nsga2_fixed(state, r))
            self.nsga2_param_table.setCellWidget(row, 1, fixed_checkbox)

            fixed_value_spin = QDoubleSpinBox()
            fixed_value_spin.setRange(0, 10e9)
            fixed_value_spin.setDecimals(6)
            fixed_value_spin.setValue(0.0)
            fixed_value_spin.setEnabled(True)
            self.nsga2_param_table.setCellWidget(row, 2, fixed_value_spin)

            lower_bound_spin = QDoubleSpinBox()
            lower_bound_spin.setRange(0, 10e9)
            lower_bound_spin.setDecimals(6)
            lower_bound_spin.setValue(0.0)
            lower_bound_spin.setEnabled(False)
            self.nsga2_param_table.setCellWidget(row, 3, lower_bound_spin)

            upper_bound_spin = QDoubleSpinBox()
            upper_bound_spin.setRange(0, 10e9)
            upper_bound_spin.setDecimals(6)
            upper_bound_spin.setValue(1.0)
            upper_bound_spin.setEnabled(False)
            self.nsga2_param_table.setCellWidget(row, 4, upper_bound_spin)

            cost_spin = QDoubleSpinBox()
            cost_spin.setRange(0.0, 10.0) # Cost coefficients between 0.0 and 10.0
            cost_spin.setDecimals(3)
            cost_spin.setValue(0.5) # Default cost
            self.nsga2_param_table.setCellWidget(row, 5, cost_spin)
        
        dva_params_layout.addWidget(self.nsga2_param_table)
        layout.addWidget(dva_params_group)

        # Control Buttons
        control_buttons_layout = QHBoxLayout()
        self.nsga2_run_button = QPushButton("Run NSGA-II")
        self.nsga2_run_button.clicked.connect(self.run_nsga2)
        control_buttons_layout.addWidget(self.nsga2_run_button)
        self.nsga2_stop_button = QPushButton("Stop")
        self.nsga2_stop_button.clicked.connect(self.stop_nsga2)
        self.nsga2_stop_button.setEnabled(False)
        control_buttons_layout.addWidget(self.nsga2_stop_button)
        layout.addLayout(control_buttons_layout)

        self.nsga2_progress_bar = QProgressBar()
        layout.addWidget(self.nsga2_progress_bar)

    def toggle_nsga2_fixed(self, state, row):
        fixed = bool(state == Qt.Checked)
        self.nsga2_param_table.cellWidget(row, 2).setEnabled(fixed) # Fixed Value
        self.nsga2_param_table.cellWidget(row, 3).setEnabled(not fixed) # Lower Bound
        self.nsga2_param_table.cellWidget(row, 4).setEnabled(not fixed) # Upper Bound

    def get_nsga2_parameter_data(self):
        """Extracts parameter data from the NSGA-II DVA parameters table."""
        parameters = []
        for r in range(self.nsga2_param_table.rowCount()):
            name = self.nsga2_param_table.item(r, 0).text()
            fixed = self.nsga2_param_table.cellWidget(r, 1).isChecked()
            fixed_value = self.nsga2_param_table.cellWidget(r, 2).value()
            low = self.nsga2_param_table.cellWidget(r, 3).value()
            high = self.nsga2_param_table.cellWidget(r, 4).value()
            cost = self.nsga2_param_table.cellWidget(r, 5).value()
            parameters.append((name, low, high, fixed, fixed_value, cost))
        return parameters

    def create_nsga2_live_calc_tab(self):
        """Creates the 'Live Calculations' sub-tab for NSGA-II."""
        self.nsga2_live_calc_tab = QWidget()
        layout = QVBoxLayout(self.nsga2_live_calc_tab)

        self.nsga2_live_table = QTableWidget()
        self.nsga2_live_table.setColumnCount(11)
        self.nsga2_live_table.setHorizontalHeaderLabels([
            "Gen", "HV", "IGD+", "GD", "Spread", "N_Pareto", 
            "Diversity", "Time (s)", "Memory (MB)", "Rank Diversity", "Best Fitness (f1,f2,f3)"
        ])
        self.nsga2_live_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        layout.addWidget(self.nsga2_live_table)

    def create_nsga2_statistics_tab(self):
        """Creates the 'Statistics' sub-tab for NSGA-II."""
        self.nsga2_statistics_tab = QWidget()
        layout = QVBoxLayout(self.nsga2_statistics_tab)

        self.nsga2_stats_table = QTableWidget()
        self.nsga2_stats_table.setColumnCount(9)
        self.nsga2_stats_table.setHorizontalHeaderLabels([
            "Run ID", "Final HV", "Mean HV (30 runs)", "Std Dev HV (30 runs)",
            "Final IGD+", "Pareto Size", "Time (hours)", "Convergence Gen", "Robustness"
        ])
        self.nsga2_stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.nsga2_stats_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.nsga2_stats_table.itemSelectionChanged.connect(self.display_selected_run_results)
        layout.addWidget(self.nsga2_stats_table)

        self.calculate_stats_button = QPushButton("Calculate Statistics")
        self.calculate_stats_button.clicked.connect(self.calculate_and_display_statistics)
        layout.addWidget(self.calculate_stats_button)

    def calculate_and_display_statistics(self):
        # This method will be implemented later
        pass

    def display_selected_run_results(self):
        # This method will be implemented later
        pass

    def create_nsga2_results_tab(self):
        """Creates the 'Results' sub-tab for NSGA-II."""
        self.nsga2_results_tab = QWidget()
        layout = QVBoxLayout(self.nsga2_results_tab)

        # This will hold the plots
        self.nsga2_results_plot_tabs = QTabWidget()
        layout.addWidget(self.nsga2_results_plot_tabs)

        # Pareto Front Plot
        self.nsga2_pareto_plot_widget = QWidget()
        pareto_layout = QVBoxLayout(self.nsga2_pareto_plot_widget)
        self.nsga2_pareto_fig = Figure()
        self.nsga2_pareto_canvas = FigureCanvas(self.nsga2_pareto_fig)
        pareto_layout.addWidget(NavigationToolbar(self.nsga2_pareto_canvas, self.nsga2_pareto_plot_widget))
        pareto_layout.addWidget(self.nsga2_pareto_canvas)
        self.nsga2_results_plot_tabs.addTab(self.nsga2_pareto_plot_widget, "Pareto Front")

        # Convergence Plot
        self.nsga2_convergence_plot_widget = QWidget()
        convergence_layout = QVBoxLayout(self.nsga2_convergence_plot_widget)
        self.nsga2_convergence_fig = Figure()
        self.nsga2_convergence_canvas = FigureCanvas(self.nsga2_convergence_fig)
        convergence_layout.addWidget(NavigationToolbar(self.nsga2_convergence_canvas, self.nsga2_convergence_plot_widget))
        convergence_layout.addWidget(self.nsga2_convergence_canvas)
        self.nsga2_results_plot_tabs.addTab(self.nsga2_convergence_plot_widget, "Convergence Metrics")
        
        # Final Results Summary
        self.nsga2_results_summary = QTextEdit()
        self.nsga2_results_summary.setReadOnly(True)
        layout.addWidget(self.nsga2_results_summary)

    def run_nsga2(self):
        try:
            if self.nsga2_multi_run_checkbox.isChecked():
                num_runs = self.nsga2_num_runs_box.value()
                self.nsga2_run_button.setEnabled(False)
                self.nsga2_stop_button.setEnabled(True)
                self.nsga2_progress_bar.setValue(0)
                self.nsga2_live_table.setRowCount(0)
                self.nsga2_stats_table.setRowCount(0)
                
                for i in range(num_runs):
                    main_params = self.get_main_system_params()
                    nsga2_parameter_data = self.get_nsga2_parameter_data()
                    target_values_dict, weights_dict = self.get_target_values_weights()
                    target_values_weights = []
                    for mass_idx in range(1, 6):
                        target_values = target_values_dict.get(f"mass_{mass_idx}", {})
                        weights = weights_dict.get(f"mass_{mass_idx}", {})
                        target_values_weights.append((target_values, weights))
                    omega_start = self.omega_start_box.value()
                    omega_end = self.omega_end_box.value()
                    omega_points = self.omega_points_box.value()

                    worker = NSGA2Worker(
                        main_params=main_params,
                        dva_params=nsga2_parameter_data,
                        target_values_weights=target_values_weights,
                        omega_start=omega_start,
                        omega_end=omega_end,
                        omega_points=omega_points,
                        pop_size=self.nsga2_pop_size_box.value(),
                        generations=self.nsga2_generations_box.value(),
                        cxpb=self.nsga2_cxpb_box.value(),
                        mutpb=self.nsga2_mutpb_box.value(),
                        eta_c=self.nsga2_eta_c_box.value(),
                        eta_m=self.nsga2_eta_m_box.value(),
                        indpb=self.nsga2_indpb_box.value(),
                        sparsity_tau=self.nsga2_sparsity_tau.value(),
                        sparsity_alpha=self.nsga2_sparsity_alpha.value(),
                        sparsity_beta=self.nsga2_sparsity_beta.value(),
                        run_id=i + 1,
                        random_seed=i + 1
                    )
                    worker.progress.connect(self.update_nsga2_progress)
                    worker.finished.connect(self.nsga2_finished)
                    worker.error.connect(self.nsga2_error)
                    worker.start()
                    # This will run workers sequentially. For parallel execution, more complex management is needed.
                    worker.wait() 
            else:
                # Single run
                main_params = self.get_main_system_params()
                nsga2_parameter_data = self.get_nsga2_parameter_data()
                target_values_dict, weights_dict = self.get_target_values_weights()
                target_values_weights = []
                for mass_idx in range(1, 6):
                    target_values = target_values_dict.get(f"mass_{mass_idx}", {})
                    weights = weights_dict.get(f"mass_{mass_idx}", {})
                    target_values_weights.append((target_values, weights))
                omega_start = self.omega_start_box.value()
                omega_end = self.omega_end_box.value()
                omega_points = self.omega_points_box.value()

                self.nsga2_worker = NSGA2Worker(
                    main_params=main_params,
                    dva_params=nsga2_parameter_data,
                    target_values_weights=target_values_weights,
                    omega_start=omega_start,
                    omega_end=omega_end,
                    omega_points=omega_points,
                    pop_size=self.nsga2_pop_size_box.value(),
                    generations=self.nsga2_generations_box.value(),
                    cxpb=self.nsga2_cxpb_box.value(),
                    mutpb=self.nsga2_mutpb_box.value(),
                    eta_c=self.nsga2_eta_c_box.value(),
                    eta_m=self.nsga2_eta_m_box.value(),
                    indpb=self.nsga2_indpb_box.value(),
                    sparsity_tau=self.nsga2_sparsity_tau.value(),
                    sparsity_alpha=self.nsga2_sparsity_alpha.value(),
                    sparsity_beta=self.nsga2_sparsity_beta.value(),
                    run_id=1,
                    random_seed=None
                )
                self.nsga2_worker.progress.connect(self.update_nsga2_progress)
                self.nsga2_worker.finished.connect(self.nsga2_finished)
                self.nsga2_worker.error.connect(self.nsga2_error)
                self.nsga2_worker.start()
                self.nsga2_run_button.setEnabled(False)
                self.nsga2_stop_button.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start NSGA-II: {str(e)}")

    def stop_nsga2(self):
        if hasattr(self, 'nsga2_worker') and self.nsga2_worker.isRunning():
            self.nsga2_worker.stop()
            self.nsga2_run_button.setEnabled(True)
            self.nsga2_stop_button.setEnabled(False)

    def update_nsga2_progress(self, progress, metrics):
        self.nsga2_progress_bar.setValue(progress)
        row_position = self.nsga2_live_table.rowCount()
        self.nsga2_live_table.insertRow(row_position)
        
        self.nsga2_live_table.setItem(row_position, 0, QTableWidgetItem(str(metrics["Gen"])))
        self.nsga2_live_table.setItem(row_position, 1, QTableWidgetItem(f"{metrics['HV']:.4f}"))
        self.nsga2_live_table.setItem(row_position, 2, QTableWidgetItem(f"{metrics['IGD+']:.4f}"))
        self.nsga2_live_table.setItem(row_position, 3, QTableWidgetItem(f"{metrics['GD']:.4f}"))
        self.nsga2_live_table.setItem(row_position, 4, QTableWidgetItem(f"{metrics['Spread']:.4f}"))
        self.nsga2_live_table.setItem(row_position, 5, QTableWidgetItem(str(metrics["N_Pareto"])))
        self.nsga2_live_table.setItem(row_position, 6, QTableWidgetItem(f"{metrics['Diversity']:.4f}"))
        self.nsga2_live_table.setItem(row_position, 7, QTableWidgetItem(f"{metrics['Time (s)']:.2f}"))
        self.nsga2_live_table.setItem(row_position, 8, QTableWidgetItem(f"{metrics['Memory (MB)']:.2f}"))
        self.nsga2_live_table.setItem(row_position, 9, QTableWidgetItem(f"{metrics['Rank Diversity']:.4f}"))
        self.nsga2_live_table.setItem(row_position, 10, QTableWidgetItem(str(metrics["Best Fitness (f1,f2,f3)"])))

    def nsga2_finished(self, pareto_front):
        self.nsga2_run_button.setEnabled(True)
        self.nsga2_stop_button.setEnabled(False)
        QMessageBox.information(self, "NSGA-II Finished", "The NSGA-II optimization has completed.")
        
        # Display results
        self.nsga2_results_summary.clear()
        self.nsga2_results_summary.append("<h3>Final Pareto Front:</h3>")
        for ind in pareto_front:
            self.nsga2_results_summary.append(f"Fitness: {ind.fitness.values}, Individual: {ind}")

        # Plot Pareto Front
        self.nsga2_pareto_fig.clear()
        ax = self.nsga2_pareto_fig.add_subplot(111, projection='3d')
        fitnesses = np.array([ind.fitness.values for ind in pareto_front])
        ax.scatter(fitnesses[:, 0], fitnesses[:, 1], fitnesses[:, 2])
        ax.set_xlabel("f1 (FRF)")
        ax.set_ylabel("f2 (Sparsity)")
        ax.set_zlabel("f3 (Cost)")
        self.nsga2_pareto_canvas.draw()

    def nsga2_error(self, error_message):
        self.nsga2_run_button.setEnabled(True)
        self.nsga2_stop_button.setEnabled(False)
        QMessageBox.critical(self, "NSGA-II Error", error_message)