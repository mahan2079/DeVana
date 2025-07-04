from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT

class ExtraOptimizationMixin:
    def create_de_tab(self):
        """Create the differential evolution optimization tab"""
        self.de_tab = QWidget()
        layout = QVBoxLayout(self.de_tab)
        
        # Create sub-tabs widget
        self.de_sub_tabs = QTabWidget()

        # -------------------- Sub-tab 1: DE Hyperparameters --------------------
        de_hyper_tab = QWidget()
        de_hyper_layout = QFormLayout(de_hyper_tab)

        self.de_pop_size_box = QSpinBox()
        self.de_pop_size_box.setRange(1, 10000)
        self.de_pop_size_box.setValue(50)

        self.de_num_generations_box = QSpinBox()
        self.de_num_generations_box.setRange(1, 10000)
        self.de_num_generations_box.setValue(100)

        self.de_F_box = QDoubleSpinBox()
        self.de_F_box.setRange(0, 2)
        self.de_F_box.setValue(0.8)
        self.de_F_box.setDecimals(3)

        self.de_CR_box = QDoubleSpinBox()
        self.de_CR_box.setRange(0, 1)
        self.de_CR_box.setValue(0.7)
        self.de_CR_box.setDecimals(3)

        self.de_tol_box = QDoubleSpinBox()
        self.de_tol_box.setRange(0, 1e6)
        self.de_tol_box.setValue(1e-3)
        self.de_tol_box.setDecimals(6)

        self.de_alpha_box = QDoubleSpinBox()
        self.de_alpha_box.setRange(0.0, 10.0)
        self.de_alpha_box.setDecimals(4)
        self.de_alpha_box.setSingleStep(0.01)
        self.de_alpha_box.setValue(0.01)
        
        # New smoothness penalty parameter
        self.de_beta_box = QDoubleSpinBox()
        self.de_beta_box.setRange(0.0, 10.0)
        self.de_beta_box.setDecimals(4)
        self.de_beta_box.setSingleStep(0.01)
        self.de_beta_box.setValue(0.0)
        self.de_beta_box.setToolTip("Parameter for smoothness penalty (0 = no smoothness penalty)")

        de_hyper_layout.addRow("Population Size:", self.de_pop_size_box)
        de_hyper_layout.addRow("Number of Generations:", self.de_num_generations_box)
        de_hyper_layout.addRow("Mutation Factor (F):", self.de_F_box)
        de_hyper_layout.addRow("Crossover Rate (CR):", self.de_CR_box)
        de_hyper_layout.addRow("Tolerance (tol):", self.de_tol_box)
        de_hyper_layout.addRow("Sparsity Penalty (alpha):", self.de_alpha_box)
        de_hyper_layout.addRow("Smoothness Penalty (beta):", self.de_beta_box)

        # Add a small Run DE button in the hyperparameters sub-tab
        self.hyper_run_de_button = QPushButton("Run DE")
        self.hyper_run_de_button.setFixedWidth(100)
        self.hyper_run_de_button.clicked.connect(self.run_de)
        de_hyper_layout.addRow("Run DE:", self.hyper_run_de_button)

        # -------------------- Sub-tab 2: DVA Parameters --------------------
        de_param_tab = QWidget()
        de_param_layout = QVBoxLayout(de_param_tab)

        self.de_param_table = QTableWidget()
        dva_parameters = [
            *[f"beta_{i}" for i in range(1,16)],
            *[f"lambda_{i}" for i in range(1,16)],
            *[f"mu_{i}" for i in range(1,4)],
            *[f"nu_{i}" for i in range(1,16)]
        ]
        self.de_param_table.setRowCount(len(dva_parameters))
        self.de_param_table.setColumnCount(5)
        self.de_param_table.setHorizontalHeaderLabels(
            ["Parameter", "Fixed", "Fixed Value", "Lower Bound", "Upper Bound"]
        )
        self.de_param_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.de_param_table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        for row, param in enumerate(dva_parameters):
            param_item = QTableWidgetItem(param)
            param_item.setFlags(Qt.ItemIsEnabled)
            self.de_param_table.setItem(row, 0, param_item)

            fixed_checkbox = QCheckBox()
            fixed_checkbox.stateChanged.connect(lambda state, r=row: self.toggle_de_fixed(state, r))
            self.de_param_table.setCellWidget(row, 1, fixed_checkbox)

            fixed_value_spin = QDoubleSpinBox()
            fixed_value_spin.setRange(-1e6, 1e6)
            fixed_value_spin.setDecimals(6)
            fixed_value_spin.setEnabled(False)
            self.de_param_table.setCellWidget(row, 2, fixed_value_spin)

            lower_bound_spin = QDoubleSpinBox()
            lower_bound_spin.setRange(-1e6, 1e6)
            lower_bound_spin.setDecimals(6)
            lower_bound_spin.setEnabled(True)
            self.de_param_table.setCellWidget(row, 3, lower_bound_spin)

            upper_bound_spin = QDoubleSpinBox()
            upper_bound_spin.setRange(-1e6, 1e6)
            upper_bound_spin.setDecimals(6)
            upper_bound_spin.setEnabled(True)
            self.de_param_table.setCellWidget(row, 4, upper_bound_spin)

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

        de_param_layout.addWidget(self.de_param_table)

        # -------------------- Sub-tab 3: Results --------------------
        de_results_tab = QWidget()
        de_results_layout = QVBoxLayout(de_results_tab)
        
        self.de_results_text = QTextEdit()
        self.de_results_text.setReadOnly(True)
        de_results_layout.addWidget(QLabel("DE Optimization Results:"))
        de_results_layout.addWidget(self.de_results_text)

        # -------------------- Sub-tab 4: Advanced Settings --------------------
        de_advanced_tab = QWidget()
        de_advanced_layout = QVBoxLayout(de_advanced_tab)
        
        # Create scrollable area for advanced settings
        advanced_scroll = QScrollArea()
        advanced_scroll.setWidgetResizable(True)
        advanced_scroll_content = QWidget()
        advanced_scroll_layout = QVBoxLayout(advanced_scroll_content)
        advanced_scroll.setWidget(advanced_scroll_content)
        
        # DE strategy selection
        strategy_group = QGroupBox("DE Strategy")
        strategy_layout = QVBoxLayout(strategy_group)
        
        self.de_strategy_combo = QComboBox()
        self.de_strategy_combo.addItems([
            "rand/1 (Standard DE)", 
            "rand/2", 
            "best/1", 
            "best/2",
            "current-to-best/1", 
            "current-to-rand/1"
        ])
        self.de_strategy_combo.setToolTip("Different mutation strategies for creating donor vectors")
        strategy_layout.addWidget(self.de_strategy_combo)
        
        advanced_scroll_layout.addWidget(strategy_group)
        
        # Adaptive methods
        adaptive_group = QGroupBox("Adaptive Parameter Control")
        adaptive_layout = QFormLayout(adaptive_group)
        
        self.de_adaptive_method_combo = QComboBox()
        self.de_adaptive_method_combo.addItems([
            "none (Fixed Parameters)",
            "jitter (Small Random Variation)",
            "dither (Random F per Generation)",
            "sade (Self-adaptive DE)",
            "jade (Adaptive DE with Archive)",
            "success-history (Success-based Adaptation)"
        ])
        self.de_adaptive_method_combo.setToolTip("Methods for automatically adapting control parameters during optimization")
        adaptive_layout.addRow("Adaptation Method:", self.de_adaptive_method_combo)
        
        # JADE parameters
        jade_frame = QGroupBox("JADE Parameters")
        jade_layout = QFormLayout(jade_frame)
        
        self.de_jade_c_box = QDoubleSpinBox()
        self.de_jade_c_box.setRange(0.01, 1.0)
        self.de_jade_c_box.setValue(0.1)
        self.de_jade_c_box.setDecimals(2)
        jade_layout.addRow("Adaptation Rate (c):", self.de_jade_c_box)
        
        adaptive_layout.addRow("", jade_frame)
        
        # SaDE parameters
        sade_frame = QGroupBox("SaDE Parameters")
        sade_layout = QFormLayout(sade_frame)
        
        self.de_sade_lp_box = QSpinBox()
        self.de_sade_lp_box.setRange(1, 1000)
        self.de_sade_lp_box.setValue(50)
        sade_layout.addRow("Learning Period:", self.de_sade_lp_box)
        
        self.de_sade_memory_box = QSpinBox()
        self.de_sade_memory_box.setRange(1, 100)
        self.de_sade_memory_box.setValue(20)
        sade_layout.addRow("Memory Size:", self.de_sade_memory_box)
        
        adaptive_layout.addRow("", sade_frame)
        
        # Dither parameters
        dither_frame = QGroupBox("Dither Parameters")
        dither_layout = QFormLayout(dither_frame)
        
        self.de_dither_f_min_box = QDoubleSpinBox()
        self.de_dither_f_min_box.setRange(0.1, 0.9)
        self.de_dither_f_min_box.setValue(0.4)
        self.de_dither_f_min_box.setDecimals(2)
        dither_layout.addRow("F Minimum:", self.de_dither_f_min_box)
        
        self.de_dither_f_max_box = QDoubleSpinBox()
        self.de_dither_f_max_box.setRange(0.1, 2.0)
        self.de_dither_f_max_box.setValue(0.9)
        self.de_dither_f_max_box.setDecimals(2)
        dither_layout.addRow("F Maximum:", self.de_dither_f_max_box)
        
        adaptive_layout.addRow("", dither_frame)
        
        advanced_scroll_layout.addWidget(adaptive_group)
        
        # Constraint handling
        constraint_group = QGroupBox("Constraint Handling")
        constraint_layout = QVBoxLayout(constraint_group)
        
        self.de_constraint_handling_combo = QComboBox()
        self.de_constraint_handling_combo.addItems([
            "penalty (Apply Penalty)",
            "reflection (Reflect at Bounds)",
            "projection (Project to Bounds)"
        ])
        self.de_constraint_handling_combo.setToolTip("Method for handling parameter constraints")
        constraint_layout.addWidget(self.de_constraint_handling_combo)
        
        advanced_scroll_layout.addWidget(constraint_group)
        
        # Termination criteria
        termination_group = QGroupBox("Termination Criteria")
        termination_layout = QFormLayout(termination_group)
        
        self.de_stagnation_box = QSpinBox()
        self.de_stagnation_box.setRange(10, 10000)
        self.de_stagnation_box.setValue(100)
        termination_layout.addRow("Max Generations without Improvement:", self.de_stagnation_box)
        
        self.de_min_diversity_box = QDoubleSpinBox()
        self.de_min_diversity_box.setRange(1e-10, 1.0)
        self.de_min_diversity_box.setValue(1e-6)
        self.de_min_diversity_box.setDecimals(10)
        termination_layout.addRow("Minimum Population Diversity:", self.de_min_diversity_box)
        
        advanced_scroll_layout.addWidget(termination_group)
        
        # Diversity preservation
        diversity_group = QGroupBox("Diversity Preservation")
        diversity_layout = QFormLayout(diversity_group)
        
        self.de_diversity_checkbox = QCheckBox()
        self.de_diversity_checkbox.setChecked(False)
        diversity_layout.addRow("Enable Diversity Preservation:", self.de_diversity_checkbox)
        
        self.de_diversity_threshold_box = QDoubleSpinBox()
        self.de_diversity_threshold_box.setRange(1e-6, 1.0)
        self.de_diversity_threshold_box.setValue(0.01)
        self.de_diversity_threshold_box.setDecimals(4)
        diversity_layout.addRow("Diversity Threshold:", self.de_diversity_threshold_box)
        
        advanced_scroll_layout.addWidget(diversity_group)
        
        # Add multiple run settings
        multi_run_group = QGroupBox("Multiple Runs")
        multi_run_layout = QFormLayout(multi_run_group)
        
        self.de_num_runs_box = QSpinBox()
        self.de_num_runs_box.setRange(1, 100)
        self.de_num_runs_box.setValue(1)
        self.de_num_runs_box.setToolTip("Number of independent optimization runs to perform")
        multi_run_layout.addRow("Number of Runs:", self.de_num_runs_box)
        
        advanced_scroll_layout.addWidget(multi_run_group)
        
        # Add multi-run progress bar
        self.de_multi_run_progress_bar = QProgressBar()
        self.de_multi_run_progress_bar.setFormat("Run %v/%m")
        self.de_multi_run_progress_bar.hide()
        advanced_scroll_layout.addWidget(self.de_multi_run_progress_bar)
        
        # Parallel processing
        parallel_group = QGroupBox("Parallel Processing")
        parallel_layout = QFormLayout(parallel_group)
        
        self.de_parallel_checkbox = QCheckBox()
        self.de_parallel_checkbox.setChecked(False)
        parallel_layout.addRow("Use Parallel Processing:", self.de_parallel_checkbox)
        
        self.de_processes_box = QSpinBox()
        self.de_processes_box.setRange(1, 64)
        # Use multiprocessing properly
        import multiprocessing
        self.de_processes_box.setValue(max(1, multiprocessing.cpu_count() - 1))
        self.de_processes_box.setEnabled(False)
        parallel_layout.addRow("Number of Processes:", self.de_processes_box)
        
        # Connect parallel checkbox to enable/disable processes box
        self.de_parallel_checkbox.stateChanged.connect(
            lambda state: self.de_processes_box.setEnabled(state == Qt.Checked)
        )
        
        advanced_scroll_layout.addWidget(parallel_group)
        
        # Random seed
        seed_group = QGroupBox("Random Seed")
        seed_layout = QFormLayout(seed_group)
        
        self.de_seed_checkbox = QCheckBox()
        self.de_seed_checkbox.setChecked(False)
        seed_layout.addRow("Use Fixed Seed:", self.de_seed_checkbox)
        
        self.de_seed_box = QSpinBox()
        self.de_seed_box.setRange(0, 1000000)
        self.de_seed_box.setValue(42)
        self.de_seed_box.setEnabled(False)
        seed_layout.addRow("Random Seed:", self.de_seed_box)
        
        # Connect seed checkbox to enable/disable seed box
        self.de_seed_checkbox.stateChanged.connect(
            lambda state: self.de_seed_box.setEnabled(state == Qt.Checked)
        )
        
        advanced_scroll_layout.addWidget(seed_group)
        
        # Hyperparameter tuning
        tuning_group = QGroupBox("Hyperparameter Tuning")
        tuning_layout = QVBoxLayout(tuning_group)
        
        self.de_tune_button = QPushButton("Tune DE Hyperparameters")
        self.de_tune_button.setToolTip("Run automatic hyperparameter tuning to find optimal settings")
        self.de_tune_button.clicked.connect(self.tune_de_hyperparameters)
        tuning_layout.addWidget(self.de_tune_button)
        
        advanced_scroll_layout.addWidget(tuning_group)
        
        # Add stretch to push widgets to the top
        advanced_scroll_layout.addStretch()
        
        # Add scroll area to advanced tab
        de_advanced_layout.addWidget(advanced_scroll)
        
        # -------------------- Sub-tab 5: Visualization --------------------
        de_viz_tab = QWidget()
        de_viz_layout = QVBoxLayout(de_viz_tab)
        
        # Add save button
        save_container = QWidget()
        save_layout = QHBoxLayout(save_container)
        save_layout.setContentsMargins(0, 0, 0, 0)
        
        self.de_viz_save_button = QPushButton("Save Plot")
        self.de_viz_save_button.clicked.connect(self.save_de_visualization)
        save_layout.addWidget(self.de_viz_save_button)
        save_layout.addStretch()
        
        de_viz_layout.addWidget(save_container)
        
        # Create tabs for different visualizations
        self.de_viz_tabs = QTabWidget()
        
        # Create tabs for different visualizations
        de_violin_tab = QWidget()
        de_violin_layout = QVBoxLayout(de_violin_tab)
        self.de_violin_plot_widget = QWidget()
        de_violin_layout.addWidget(self.de_violin_plot_widget)
        
        de_convergence_tab = QWidget()
        de_convergence_layout = QVBoxLayout(de_convergence_tab)
        self.de_convergence_plot_widget = QWidget()
        de_convergence_layout.addWidget(self.de_convergence_plot_widget)
        
        de_diversity_tab = QWidget()
        de_diversity_layout = QVBoxLayout(de_diversity_tab)
        self.de_diversity_plot_widget = QWidget()
        de_diversity_layout.addWidget(self.de_diversity_plot_widget)
        
        de_adaptation_tab = QWidget()
        de_adaptation_layout = QVBoxLayout(de_adaptation_tab)
        self.de_adaptation_plot_widget = QWidget()
        de_adaptation_layout.addWidget(self.de_adaptation_plot_widget)
        
        de_param_evolution_tab = QWidget()
        de_param_evolution_layout = QVBoxLayout(de_param_evolution_tab)
        self.de_param_evolution_plot_widget = QWidget()
        de_param_evolution_layout.addWidget(self.de_param_evolution_plot_widget)
        
        de_correlation_tab = QWidget()
        de_correlation_layout = QVBoxLayout(de_correlation_tab)
        self.de_correlation_plot_widget = QWidget()
        de_correlation_layout.addWidget(self.de_correlation_plot_widget)
        
        # Add all visualization tabs
        self.de_viz_tabs.addTab(de_violin_tab, "Multi-Run Statistics")
        self.de_viz_tabs.addTab(de_convergence_tab, "Convergence History")
        self.de_viz_tabs.addTab(de_diversity_tab, "Population Diversity")
        self.de_viz_tabs.addTab(de_adaptation_tab, "Control Parameter Adaptation")
        self.de_viz_tabs.addTab(de_param_evolution_tab, "Parameter Evolution")
        self.de_viz_tabs.addTab(de_correlation_tab, "Parameter Correlation")
        
        # Add the visualization tabs to the layout
        de_viz_layout.addWidget(self.de_viz_tabs)
        
        # Connect tab change to update function
        self.de_viz_tabs.currentChanged.connect(self.update_de_visualization)
        
        # Add all sub-tabs to the DE tab widget
        self.de_sub_tabs.addTab(de_hyper_tab, "DE Settings")
        self.de_sub_tabs.addTab(de_param_tab, "DVA Parameters")
        self.de_sub_tabs.addTab(de_results_tab, "Results")
        self.de_sub_tabs.addTab(de_advanced_tab, "Advanced Settings")
        self.de_sub_tabs.addTab(de_viz_tab, "Visualization")

        # Add the DE sub-tabs widget to the main DE tab layout
        layout.addWidget(self.de_sub_tabs)
        self.de_tab.setLayout(layout)
        
    def toggle_de_fixed(self, state, row, table=None):
        """Toggle the fixed state of a DE parameter row"""
        if table is None:
            table = self.de_param_table
            
        fixed = (state == Qt.Checked)
        fixed_value_spin = table.cellWidget(row, 2)
        lower_bound_spin = table.cellWidget(row, 3)
        upper_bound_spin = table.cellWidget(row, 4)

        fixed_value_spin.setEnabled(fixed)
        lower_bound_spin.setEnabled(not fixed)
        upper_bound_spin.setEnabled(not fixed)
        
    def run_de(self):
        """Run the differential evolution optimization"""
        self.status_bar.showMessage("Running DE optimization...")
        self.results_text.append("DE optimization started...")
        
        try:
            # Retrieve DE parameters from the GUI
            pop_size = self.de_pop_size_box.value()
            num_generations = self.de_num_generations_box.value()
            mutation_factor = self.de_F_box.value()
            crossover_rate = self.de_CR_box.value()
            tol = self.de_tol_box.value()
            alpha = self.de_alpha_box.value()
            beta = self.de_beta_box.value()
            
            # Get advanced DE options
            strategy_idx = self.de_strategy_combo.currentIndex()
            strategy_names = ["rand/1", "rand/2", "best/1", "best/2", "current-to-best/1", "current-to-rand/1"]
            strategy = strategy_names[strategy_idx]
            
            adaptive_idx = self.de_adaptive_method_combo.currentIndex()
            adaptive_names = ["none", "jitter", "dither", "sade", "jade", "success-history"]
            adaptive_method = adaptive_names[adaptive_idx]
            
            constraint_idx = self.de_constraint_handling_combo.currentIndex()
            constraint_names = ["penalty", "reflection", "projection"]
            constraint_handling = constraint_names[constraint_idx]
            
            use_parallel = self.de_parallel_checkbox.isChecked()
            n_processes = self.de_processes_box.value() if use_parallel else None
            
            use_seed = self.de_seed_checkbox.isChecked()
            seed = self.de_seed_box.value() if use_seed else None
            
            diversity_preservation = self.de_diversity_checkbox.isChecked()
            
            # Get number of runs
            num_runs = self.de_num_runs_box.value()
            
            # Show multi-run progress bar if doing multiple runs
            if num_runs > 1:
                self.de_multi_run_progress_bar.setRange(0, num_runs)
                self.de_multi_run_progress_bar.setValue(0)
                self.de_multi_run_progress_bar.show()
            else:
                self.de_multi_run_progress_bar.hide()
            
            # Prepare adaptive parameters based on selected method
            adaptive_params = {}
            if adaptive_method == "jade":
                adaptive_params["c"] = self.de_jade_c_box.value()
            elif adaptive_method == "sade":
                adaptive_params["LP"] = self.de_sade_lp_box.value()
                adaptive_params["memory_size"] = self.de_sade_memory_box.value()
            elif adaptive_method == "dither":
                adaptive_params["F_min"] = self.de_dither_f_min_box.value()
                adaptive_params["F_max"] = self.de_dither_f_max_box.value()
            
            # Diversity preservation parameters
            if diversity_preservation:
                adaptive_params["diversity_threshold"] = self.de_diversity_threshold_box.value()
            
            # Prepare termination criteria
            termination_criteria = {
                "max_generations": num_generations,
                "tol": tol,
                "stagnation_limit": self.de_stagnation_box.value(),
                "min_diversity": self.de_min_diversity_box.value()
            }

            de_dva_parameters = []
            row_count = self.de_param_table.rowCount()
            for row in range(row_count):
                param_name = self.de_param_table.item(row, 0).text()
                fixed_widget = self.de_param_table.cellWidget(row, 1)
                fixed = fixed_widget.isChecked()
                if fixed:
                    fixed_value_widget = self.de_param_table.cellWidget(row, 2)
                    fv = fixed_value_widget.value()
                    de_dva_parameters.append((param_name, fv, fv, True))
                else:
                    lower_bound_widget = self.de_param_table.cellWidget(row, 3)
                    upper_bound_widget = self.de_param_table.cellWidget(row, 4)
                    lb = lower_bound_widget.value()
                    ub = upper_bound_widget.value()
                    if lb > ub:
                        QMessageBox.warning(self, "Input Error",
                                            f"For parameter {param_name}, lower bound is greater than upper bound.")
                        return
                    de_dva_parameters.append((param_name, lb, ub, False))

            # Get main system parameters
            main_params = self.get_main_system_params()

            # Get target values and weights
            target_values, weights = self.get_target_values_weights()

            # Get frequency range values
            omega_start_val = self.omega_start_box.value()
            omega_end_val = self.omega_end_box.value()
            omega_points_val = self.omega_points_box.value()

            # Create and start DEWorker with enhanced parameters
            self.de_worker = DEWorker(
                main_params=main_params,
                target_values_dict=target_values,
                weights_dict=weights,
                omega_start=omega_start_val,
                omega_end=omega_end_val,
                omega_points=omega_points_val,
                de_pop_size=pop_size,
                de_num_generations=num_generations,
                de_F=mutation_factor,
                de_CR=crossover_rate,
                de_tol=tol,
                de_parameter_data=de_dva_parameters,
                alpha=alpha,
                beta=beta,
                strategy=strategy,
                adaptive_method=adaptive_method,
                adaptive_params=adaptive_params,
                termination_criteria=termination_criteria,
                use_parallel=use_parallel,
                n_processes=n_processes,
                seed=seed,
                record_statistics=True,
                constraint_handling=constraint_handling,
                diversity_preservation=diversity_preservation,
                num_runs=num_runs  # Add number of runs parameter
            )
            
            self.de_worker.finished.connect(self.handle_de_finished)
            self.de_worker.error.connect(self.handle_de_error)
            self.de_worker.update.connect(self.handle_de_update)
            self.de_worker.progress.connect(self.handle_de_progress)
            self.de_worker.multi_run_progress.connect(self.handle_de_multi_run_progress)  # Connect multi-run progress signal
            
            # Disable both run DE buttons to prevent multiple runs
            self.hyper_run_de_button.setEnabled(False)
            self.run_de_button.setEnabled(False)
            
            self.de_results_text.clear()
            self.de_results_text.append("Running DE optimization...")
            
            # Initialize progress bar if not exists
            if not hasattr(self, 'de_progress_bar'):
                self.de_progress_bar = QProgressBar()
                self.de_progress_bar.setRange(0, num_generations)
                self.de_progress_bar.setFormat("%v/%m gen - Best fitness: %p%")
                self.de_sub_tabs.widget(3).layout().addWidget(self.de_progress_bar)
            else:
                self.de_progress_bar.setRange(0, num_generations)
                self.de_progress_bar.setValue(0)
            
            self.de_worker.start()
            
        except Exception as e:
            self.handle_de_error(str(e))
        
    
    def handle_de_finished(self, results, best_individual, parameter_names, best_fitness, statistics):
        """Handle the completion of DE optimization"""
        # Re-enable both run DE buttons
        self.hyper_run_de_button.setEnabled(True)
        self.run_de_button.setEnabled(True)
        
        self.de_results_text.append("\nDE Completed.\n")
        self.de_results_text.append("Best individual parameters:")

        for name, val in zip(parameter_names, best_individual):
            self.de_results_text.append(f"{name}: {val}")
        self.de_results_text.append(f"\nBest fitness: {best_fitness:.6f}")

        singular_response = results.get('singular_response', None)
        if singular_response is not None:
            self.de_results_text.append(f"\nSingular response of best individual: {singular_response}")

        self.de_results_text.append("\nFull Results:")
        for section, data in results.items():
            self.de_results_text.append(f"{section}: {data}")
            
        # Store optimization statistics for visualization
        self.de_statistics = statistics
        
        # Update visualization tab
        self.update_de_visualization()
            
        self.status_bar.showMessage("DE optimization completed")
        
        # Store results for later use in comparative visualization
        if "singular_response" in results and results["singular_response"] is not None:
            # Instead of using the missing method, just store the results for potential comparison
            if not hasattr(self, 'frf_comparison_results'):
                self.frf_comparison_results = {}
            self.frf_comparison_results["DE"] = results
            
        # Store best parameters for potential application
        self.current_de_best_params = best_individual
        self.current_de_parameter_names = parameter_names

    def handle_de_error(self, err):
        """Handle errors during DE optimization"""
        # Re-enable both run DE buttons
        self.hyper_run_de_button.setEnabled(True)
        self.run_de_button.setEnabled(True)
        
        QMessageBox.warning(self, "DE Error", f"Error during DE optimization: {err}")
        self.de_results_text.append(f"\nError running DE: {err}")
        self.status_bar.showMessage("DE optimization failed")

    def handle_de_update(self, msg):
        """Handle progress updates from DE worker"""
        self.de_results_text.append(msg)
        
    def handle_de_progress(self, generation, best_fitness, diversity):
        """Handle progress updates from DE worker"""
        if hasattr(self, 'de_progress_bar'):
            self.de_progress_bar.setValue(generation)
            self.de_progress_bar.setFormat(f"{generation}/{self.de_progress_bar.maximum()} gen - Best: {best_fitness:.6f}")
            
    def handle_de_multi_run_progress(self, current_run, total_runs):
        """Handle progress updates for multiple runs"""
        self.de_multi_run_progress_bar.setValue(current_run)
        self.status_bar.showMessage(f"Running DE optimization - Run {current_run}/{total_runs}")
            
    def update_de_visualization(self):
        """Update the DE visualization based on selected tab"""
        if not hasattr(self, 'de_statistics'):
            return
            
        # Get current tab
        current_tab = self.de_viz_tabs.currentWidget()
        
        # Clear the current widget's layout
        if current_tab.layout():
            for i in reversed(range(current_tab.layout().count())): 
                widget = current_tab.layout().itemAt(i).widget()
                if widget:
                    widget.setParent(None)
        
        # Create figure and canvas for the current visualization
        fig = Figure(figsize=(8, 6))
        canvas = FigureCanvas(fig)
        toolbar = NavigationToolbar(canvas, current_tab)
        
        # Add toolbar and canvas to layout
        current_tab.layout().addWidget(toolbar)
        current_tab.layout().addWidget(canvas)
        
        # Get tab name
        tab_name = self.de_viz_tabs.tabText(self.de_viz_tabs.currentIndex())
        
        if tab_name == "Multi-Run Statistics" and hasattr(self.de_statistics, 'run_best_fitnesses'):
            # Create violin plots for multi-run statistics
            n_params = len(self.current_de_parameter_names)
            n_rows = (n_params + 2 + 1) // 2  # Parameters + fitness + convergence, 2 columns
            
            # Create subplots
            for i, param_name in enumerate(self.current_de_parameter_names):
                ax = fig.add_subplot(n_rows, 2, i + 1)
                param_values = self.de_statistics.parameter_distributions[param_name]
                sns.violinplot(data=param_values, ax=ax)
                ax.set_title(f'Distribution of {param_name}')
                ax.set_ylabel('Parameter Value')
            
            # Add fitness distribution plot
            ax = fig.add_subplot(n_rows, 2, n_params + 1)
            sns.violinplot(data=self.de_statistics.run_best_fitnesses, ax=ax)
            ax.set_title('Distribution of Best Fitness Values')
            ax.set_ylabel('Fitness Value')
            
            # Add convergence generation distribution plot
            ax = fig.add_subplot(n_rows, 2, n_params + 2)
            sns.violinplot(data=self.de_statistics.run_convergence_gens, ax=ax)
            ax.set_title('Distribution of Convergence Generations')
            ax.set_ylabel('Generation')
            
        elif tab_name == "Convergence History":
            ax = fig.add_subplot(111)
            generations = self.de_statistics.generations
            best_fitness = self.de_statistics.best_fitness_history
            mean_fitness = self.de_statistics.mean_fitness_history
            
            ax.plot(generations, best_fitness, 'b-', label='Best Fitness')
            ax.plot(generations, mean_fitness, 'r--', label='Mean Fitness')
            ax.set_xlabel('Generation')
            ax.set_ylabel('Fitness Value')
            ax.set_title('Convergence History')
            ax.legend()
            ax.grid(True)
            
        elif tab_name == "Population Diversity":
            ax = fig.add_subplot(111)
            generations = self.de_statistics.generations
            diversity = self.de_statistics.diversity_history
            
            ax.plot(generations, diversity, 'g-')
            ax.set_xlabel('Generation')
            ax.set_ylabel('Population Diversity')
            ax.set_title('Population Diversity Over Time')
            ax.grid(True)
            
        elif tab_name == "Control Parameter Adaptation":
            if hasattr(self.de_statistics, 'f_values') and hasattr(self.de_statistics, 'cr_values'):
                ax1 = fig.add_subplot(211)
                ax2 = fig.add_subplot(212)
                
                generations = self.de_statistics.generations
                f_values = self.de_statistics.f_values
                cr_values = self.de_statistics.cr_values
                
                ax1.plot(generations, f_values, 'b-')
                ax1.set_xlabel('Generation')
                ax1.set_ylabel('F Value')
                ax1.set_title('Mutation Factor (F) Adaptation')
                ax1.grid(True)
                
                ax2.plot(generations, cr_values, 'r-')
                ax2.set_xlabel('Generation')
                ax2.set_ylabel('CR Value')
                ax2.set_title('Crossover Rate (CR) Adaptation')
                ax2.grid(True)
            
        elif tab_name == "Parameter Evolution":
            ax = fig.add_subplot(111)
            generations = self.de_statistics.generations
            param_means = np.array(self.de_statistics.parameter_mean_history)
            param_stds = np.array(self.de_statistics.parameter_std_history)
            
            for i, param_name in enumerate(self.current_de_parameter_names):
                mean = param_means[:, i]
                std = param_stds[:, i]
                ax.plot(generations, mean, label=param_name)
                ax.fill_between(generations, mean - std, mean + std, alpha=0.2)
            
            ax.set_xlabel('Generation')
            ax.set_ylabel('Parameter Value')
            ax.set_title('Parameter Evolution Over Time')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True)
            
        elif tab_name == "Parameter Correlation":
            if len(self.current_de_parameter_names) > 1:
                param_data = {}
                for i, param_name in enumerate(self.current_de_parameter_names):
                    param_data[param_name] = np.array(self.de_statistics.parameter_mean_history)[:, i]
                
                df = pd.DataFrame(param_data)
                sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=fig.add_subplot(111))
                plt.title('Parameter Correlation Matrix')
        
        # Adjust layout and draw
        fig.tight_layout()
        canvas.draw()
        if not hasattr(self, 'de_statistics') or self.de_statistics is None:
            return
            
        viz_type = self.de_viz_combo.currentText()
        
        # Clear the figure
        self.de_viz_fig.clear()
        
        if viz_type == "Convergence History":
            ax = self.de_viz_fig.add_subplot(111)
            ax.plot(self.de_statistics.generations, self.de_statistics.best_fitness_history, 'b-', label='Best Fitness')
            ax.plot(self.de_statistics.generations, self.de_statistics.mean_fitness_history, 'r--', label='Mean Fitness')
            ax.set_xlabel('Generation')
            ax.set_ylabel('Fitness Value')
            ax.set_title('Convergence History')
            ax.legend()
            ax.grid(True)
            
        elif viz_type == "Population Diversity":
            ax = self.de_viz_fig.add_subplot(111)
            ax.plot(self.de_statistics.generations, self.de_statistics.diversity_history, 'g-')
            ax.set_xlabel('Generation')
            ax.set_ylabel('Population Diversity')
            ax.set_title('Population Diversity History')
            ax.grid(True)
            
        elif viz_type == "Control Parameter Adaptation":
            ax = self.de_viz_fig.add_subplot(111)
            ax.plot(self.de_statistics.generations, self.de_statistics.f_values, 'b-', label='F')
            ax.plot(self.de_statistics.generations, self.de_statistics.cr_values, 'r-', label='CR')
            ax.set_xlabel('Generation')
            ax.set_ylabel('Parameter Value')
            ax.set_title('Control Parameter Adaptation')
            ax.legend()
            ax.grid(True)
            
        elif viz_type == "Parameter Evolution":
            if len(self.de_statistics.parameter_mean_history) > 0:
                param_means = np.array(self.de_statistics.parameter_mean_history)
                param_stds = np.array(self.de_statistics.parameter_std_history)
                
                n_params = param_means.shape[1]
                if n_params > 10:  # Too many parameters to show all at once
                    # Show just a sample of parameters
                    param_indices = np.linspace(0, n_params-1, 6, dtype=int)
                    
                    fig = self.de_viz_fig
                    fig.subplots_adjust(hspace=0.4, wspace=0.4)
                    
                    for i, idx in enumerate(param_indices):
                        ax = fig.add_subplot(2, 3, i+1)
                        ax.plot(self.de_statistics.generations, param_means[:, idx], 'b-')
                        ax.fill_between(
                            self.de_statistics.generations,
                            param_means[:, idx] - param_stds[:, idx],
                            param_means[:, idx] + param_stds[:, idx],
                            alpha=0.2
                        )
                        param_name = self.current_de_parameter_names[idx] if hasattr(self, 'current_de_parameter_names') else f"Param {idx}"
                        ax.set_title(f"{param_name}")
                        ax.grid(True)
                    
                    fig.suptitle("Parameter Evolution (Sample)", fontsize=12)
                else:
                    # Show all parameters
                    ax = self.de_viz_fig.add_subplot(111)
                    for i in range(n_params):
                        ax.plot(self.de_statistics.generations, param_means[:, i], label=f"Param {i}")
                    ax.set_xlabel('Generation')
                    ax.set_ylabel('Parameter Value')
                    ax.set_title('Parameter Evolution')
                    if n_params <= 5:  # Only show legend if not too many parameters
                        ax.legend()
                    ax.grid(True)
            
        elif viz_type == "Parameter Correlation":
            if len(self.de_statistics.parameter_mean_history) > 0:
                param_means = np.array(self.de_statistics.parameter_mean_history)
                
                # Create correlation matrix
                corr_matrix = np.corrcoef(param_means.T)
                
                # Create heatmap
                ax = self.de_viz_fig.add_subplot(111)
                cax = ax.matshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                self.de_viz_fig.colorbar(cax)
                
                # Set axis labels with parameter names if available
                if hasattr(self, 'current_de_parameter_names'):
                    # Use shorter parameter names for readability
                    short_names = [name.split('_')[-1] for name in self.current_de_parameter_names]
                    if len(short_names) > 15:
                        # Too many parameters, show indices instead
                        ax.set_title("Parameter Correlation Matrix")
                    else:
                        ax.set_xticks(np.arange(len(short_names)))
                        ax.set_yticks(np.arange(len(short_names)))
                        ax.set_xticklabels(short_names, rotation=90)
                        ax.set_yticklabels(short_names)
                        ax.set_title("Parameter Correlation Matrix")
        
        # Refresh the canvas
        self.de_viz_canvas.draw()
        
    def save_de_visualization(self):
        """Save the current DE visualization plot"""
        if not hasattr(self, 'de_statistics') or self.de_statistics is None:
            QMessageBox.warning(self, "No Data", "No visualization data available to save.")
            return
            
        viz_type = self.de_viz_combo.currentText()
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Visualization", "", "PNG Files (*.png);;PDF Files (*.pdf);;All Files (*)"
        )
        
        if file_path:
            self.de_viz_fig.savefig(file_path, dpi=300, bbox_inches='tight')
            self.status_bar.showMessage(f"Visualization saved to {file_path}")
            
    def tune_de_hyperparameters(self):
        """Run hyperparameter tuning for DE"""
        reply = QMessageBox.question(
            self, 'Hyperparameter Tuning',
            'Hyperparameter tuning will evaluate multiple combinations of DE parameters '
            'which may take a significant amount of time. Continue?',
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        
        if reply == QMessageBox.No:
            return
            
        try:
            # Get necessary parameters for tuning
            main_params = self.get_main_system_params()
            target_values, weights = self.get_target_values_weights()
            omega_start_val = self.omega_start_box.value()
            omega_end_val = self.omega_end_box.value()
            omega_points_val = self.omega_points_box.value()
            
            # Get DVA parameters
            de_dva_parameters = []
            row_count = self.de_param_table.rowCount()
            for row in range(row_count):
                param_name = self.de_param_table.item(row, 0).text()
                fixed_widget = self.de_param_table.cellWidget(row, 1)
                fixed = fixed_widget.isChecked()
                if fixed:
                    fixed_value_widget = self.de_param_table.cellWidget(row, 2)
                    fv = fixed_value_widget.value()
                    de_dva_parameters.append((param_name, fv, fv, True))
                else:
                    lower_bound_widget = self.de_param_table.cellWidget(row, 3)
                    upper_bound_widget = self.de_param_table.cellWidget(row, 4)
                    lb = lower_bound_widget.value()
                    ub = upper_bound_widget.value()
                    de_dva_parameters.append((param_name, lb, ub, False))
            
            # Create a progress dialog
            progress_dialog = QDialog(self)
            progress_dialog.setWindowTitle("DE Hyperparameter Tuning")
            progress_dialog.setMinimumWidth(400)
            
            dialog_layout = QVBoxLayout(progress_dialog)
            dialog_layout.addWidget(QLabel("Running hyperparameter tuning. This may take some time..."))
            
            tuning_progress_text = QTextEdit()
            tuning_progress_text.setReadOnly(True)
            dialog_layout.addWidget(tuning_progress_text)
            
            progress_bar = QProgressBar()
            progress_bar.setRange(0, 0)  # Indeterminate progress
            dialog_layout.addWidget(progress_bar)
            
            cancel_button = QPushButton("Cancel")
            cancel_button.clicked.connect(progress_dialog.reject)
            dialog_layout.addWidget(cancel_button)
            
            # Create a worker thread for tuning
            class TuningThread(QThread):
                finished_signal = pyqtSignal(dict)
                update_signal = pyqtSignal(str)
                
                def __init__(self, main_params, target_values, weights, omega_start, 
                             omega_end, omega_points, de_parameter_data):
                    super().__init__()
                    self.main_params = main_params
                    self.target_values = target_values
                    self.weights = weights
                    self.omega_start = omega_start
                    self.omega_end = omega_end
                    self.omega_points = omega_points
                    self.de_parameter_data = de_parameter_data
                    
                def run(self):
                    try:
                        # Redirect print output to emit as signal
                        original_print = print
                        def print_override(*args, **kwargs):
                            message = " ".join(map(str, args))
                            self.update_signal.emit(message)
                            original_print(*args, **kwargs)
                        
                        __builtins__['print'] = print_override
                        
                        # Run tuning with fewer trials for UI responsiveness
                        best_params = DEWorker.tune_hyperparameters(
                            main_params=self.main_params,
                            target_values_dict=self.target_values,
                            weights_dict=self.weights,
                            omega_start=self.omega_start,
                            omega_end=self.omega_end,
                            omega_points=self.omega_points,
                            de_parameter_data=self.de_parameter_data,
                            n_trials=5,  # Reduced for UI
                            parallel=True
                        )
                        
                        # Restore original print
                        __builtins__['print'] = original_print
                        
                        self.finished_signal.emit(best_params)
                    except Exception as e:
                        self.update_signal.emit(f"Error in tuning: {str(e)}")
                        self.finished_signal.emit({})
            
            # Create and start the tuning thread
            tuning_thread = TuningThread(
                main_params, target_values, weights, 
                omega_start_val, omega_end_val, omega_points_val,
                de_dva_parameters
            )
            
            # Connect signals
            tuning_thread.update_signal.connect(lambda msg: tuning_progress_text.append(msg))
            tuning_thread.finished_signal.connect(lambda result: self._apply_tuning_results(result, progress_dialog))
            
            # Start the thread and show dialog
            tuning_thread.start()
            progress_dialog.exec_()
            
            # Ensure thread is terminated if dialog is closed
            if tuning_thread.isRunning():
                tuning_thread.terminate()
                tuning_thread.wait()
                
        except Exception as e:
            QMessageBox.warning(self, "Tuning Error", f"Error during hyperparameter tuning: {str(e)}")
    
    def _apply_tuning_results(self, best_params, dialog):
        """Apply the results of hyperparameter tuning to the UI"""
        if not best_params:
            QMessageBox.warning(self, "Tuning Error", "Hyperparameter tuning failed or was cancelled.")
            dialog.accept()
            return
            
        # Update UI controls with best parameters
        self.de_pop_size_box.setValue(best_params.get("pop_size", 50))
        self.de_F_box.setValue(best_params.get("F", 0.5))
        self.de_CR_box.setValue(best_params.get("CR", 0.7))
        
        # Update strategy combo box
        if "strategy" in best_params:
            strategy_map = {
                "rand/1": 0,
                "rand/2": 1,
                "best/1": 2,
                "best/2": 3,
                "current-to-best/1": 4,
                "current-to-rand/1": 5
            }
            strategy_value = best_params["strategy"].value if hasattr(best_params["strategy"], "value") else best_params["strategy"]
            if strategy_value in strategy_map:
                self.de_strategy_combo.setCurrentIndex(strategy_map[strategy_value])
        
        # Show summary of results
        QMessageBox.information(
            self, "Tuning Complete",
            f"Hyperparameter tuning completed.\n\n"
            f"Best parameters:\n"
            f"Population Size: {best_params.get('pop_size', 'N/A')}\n"
            f"Mutation Factor (F): {best_params.get('F', 'N/A')}\n"
            f"Crossover Rate (CR): {best_params.get('CR', 'N/A')}\n"
            f"Strategy: {best_params.get('strategy', 'N/A')}\n\n"
            f"Average Fitness: {best_params.get('avg_fitness', 'N/A')}\n"
            f"Average Convergence Generation: {best_params.get('avg_convergence_gen', 'N/A')}"
        )
        
        dialog.accept()
        
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

        sa_hyper_layout.addRow("Initial Temperature:", self.sa_initial_temp_box)
        sa_hyper_layout.addRow("Cooling Rate:", self.sa_cooling_rate_box)
        sa_hyper_layout.addRow("Number of Iterations:", self.sa_num_iterations_box)
        sa_hyper_layout.addRow("Tolerance (tol):", self.sa_tol_box)
        sa_hyper_layout.addRow("Sparsity Penalty (alpha):", self.sa_alpha_box)

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
        # Implementation already exists at line 2591
        pass
        
    def run_cmaes(self):
        """Run the CMA-ES optimization"""
        # Implementation already exists at line 2840
        pass
        
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

        cmaes_hyper_layout.addRow("Initial Sigma:", self.cmaes_sigma_box)
        cmaes_hyper_layout.addRow("Max Iterations:", self.cmaes_max_iter_box)
        cmaes_hyper_layout.addRow("Tolerance (tol):", self.cmaes_tol_box)
        cmaes_hyper_layout.addRow("Sparsity Penalty (alpha):", self.cmaes_alpha_box)

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
        # Implementation already exists at line 2840
        pass

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
    
    def run_omega_sensitivity(self):
        """Run the Omega points sensitivity analysis"""
        # Get main system parameters
        main_params = self.get_main_system_params()
        
        # Get DVA parameters - ensure we have 48 values (15 betas, 15 lambdas, 3 mus, 15 nus)
        dva_params = []
        
        # Add beta parameters (15)
        for i in range(15):
            if i < len(self.beta_boxes):
                dva_params.append(self.beta_boxes[i].value())
            else:
                dva_params.append(0.0)
                
        # Add lambda parameters (15)
        for i in range(15):
            if i < len(self.lambda_boxes):
                dva_params.append(self.lambda_boxes[i].value())
            else:
                dva_params.append(0.0)
                
        # Add mu parameters (3)
        for i in range(3):
            if i < len(self.mu_dva_boxes):
                dva_params.append(self.mu_dva_boxes[i].value())
            else:
                dva_params.append(0.0)
                
        # Add nu parameters (15)
        for i in range(15):
            if i < len(self.nu_dva_boxes):
                dva_params.append(self.nu_dva_boxes[i].value())
            else:
                dva_params.append(0.0)
        
        # Get the omega range from the frequency tab
        omega_start = self.omega_start_box.value()
        omega_end = self.omega_end_box.value()
        
        # Check if start is less than end
        if omega_start >= omega_end:
            QMessageBox.warning(self, "Input Error", "Ω Start must be less than Ω End.")
            return
        
        # Get sensitivity analysis parameters
        initial_points = self.sensitivity_initial_points.value()
        max_points = self.sensitivity_max_points.value()
        step_size = self.sensitivity_step_size.value()
        convergence_threshold = self.sensitivity_threshold.value()
        max_iterations = self.sensitivity_max_iterations.value()
        mass_of_interest = self.sensitivity_mass.currentText()
        plot_results = self.sensitivity_plot_results.isChecked()
        
        # Update UI
        self.sensitivity_results_text.clear()
        self.sensitivity_results_text.append("Running Omega points sensitivity analysis...\n")
        self.status_bar.showMessage("Running Omega points sensitivity analysis...")
        
        # Disable run button during analysis
        self.run_sensitivity_btn.setEnabled(False)
        
        # Create worker for background processing
        class SensitivityWorker(QThread):
            finished = pyqtSignal(dict)
            error = pyqtSignal(str)
            
            def __init__(self, main_params, dva_params, omega_start, omega_end, 
                         initial_points, max_points, step_size, convergence_threshold,
                         max_iterations, mass_of_interest, plot_results):
                super().__init__()
                self.main_params = main_params
                self.dva_params = dva_params
                self.omega_start = omega_start
                self.omega_end = omega_end
                self.initial_points = initial_points
                self.max_points = max_points
                self.step_size = step_size
                self.convergence_threshold = convergence_threshold
                self.max_iterations = max_iterations
                self.mass_of_interest = mass_of_interest
                self.plot_results = plot_results
            
            def run(self):
                try:
                    # Import the function from FRF module
                    from modules.FRF import perform_omega_points_sensitivity_analysis
                    
                    # Run sensitivity analysis
                    results = perform_omega_points_sensitivity_analysis(
                        main_system_parameters=self.main_params,
                        dva_parameters=self.dva_params,
                        omega_start=self.omega_start,
                        omega_end=self.omega_end,
                        initial_points=self.initial_points,
                        max_points=self.max_points,
                        step_size=self.step_size,
                        convergence_threshold=self.convergence_threshold,
                        max_iterations=self.max_iterations,
                        mass_of_interest=self.mass_of_interest,
                        plot_results=self.plot_results
                    )
                    
                    self.finished.emit(results)
                except Exception as e:
                    import traceback
                    self.error.emit(f"Error in sensitivity analysis: {str(e)}\n{traceback.format_exc()}")

        # Create and start the worker
        self.sensitivity_worker = SensitivityWorker(
            main_params, dva_params, omega_start, omega_end, 
            initial_points, max_points, step_size, convergence_threshold,
            max_iterations, mass_of_interest, plot_results
        )
        
        # Connect signals
        self.sensitivity_worker.finished.connect(self.handle_sensitivity_finished)
        self.sensitivity_worker.error.connect(self.handle_sensitivity_error)
        
        # Start worker
        self.sensitivity_worker.start()
    
    def handle_sensitivity_finished(self, results):
        """Handle the completion of the Omega points sensitivity analysis"""
        # Re-enable run button
        self.run_sensitivity_btn.setEnabled(True)
        
        # Update status
        self.status_bar.showMessage("Omega points sensitivity analysis completed")
        
        # Store results for later use in plotting
        self.sensitivity_results = results
        
        # Display results
        self.sensitivity_results_text.append("\n--- Analysis Results ---\n")
        
        # Show analysis outcome with detailed information
        optimal_points = results["optimal_points"]
        converged = results["converged"]
        convergence_point = results.get("convergence_point")
        all_analyzed = results.get("all_points_analyzed", False)
        requested_max = results.get("requested_max_points", optimal_points)
        highest_analyzed = results.get("highest_analyzed_point", optimal_points)
        hit_iter_limit = results.get("iteration_limit_reached", False)
        
        # No automatic step size adjustment as per user request
        
        # Did the analysis reach the requested maximum points?
        if requested_max > highest_analyzed:
            # No, it stopped early
            self.sensitivity_results_text.append(f"⚠️ WARNING: Analysis stopped at {highest_analyzed} points (requested maximum: {requested_max})\n")
            
            if hit_iter_limit:
                self.sensitivity_results_text.append(f"   Reason: Maximum number of iterations reached ({self.sensitivity_max_iterations.value()})\n")
                self.sensitivity_results_text.append(f"   Solution: Increase 'Maximum Iterations' parameter to analyze more points\n")
            else:
                self.sensitivity_results_text.append(f"   Possible reasons: calculation constraints or memory limits\n")
                self.sensitivity_results_text.append(f"   Try using an even larger step size for higher point values\n")
        
        # Show convergence status
        if converged:
            if convergence_point == optimal_points:
                # Converged right at the last point
                self.sensitivity_results_text.append(f"✅ Analysis converged at {convergence_point} omega points\n")
            else:
                # Converged earlier but continued as requested
                self.sensitivity_results_text.append(f"✅ Analysis converged at {convergence_point} omega points, continued to {highest_analyzed}\n")
                
            # Report explicitly about whether we made it to max_points
            if all_analyzed:
                self.sensitivity_results_text.append(f"   Successfully analyzed all requested points up to maximum: {requested_max}\n")
        else:
            # Did not converge anywhere
            self.sensitivity_results_text.append(f"⚠️ Analysis did not converge at any point up to {highest_analyzed} omega points\n")
        
        # Show result details in a formatted table
        self.sensitivity_results_text.append("--- Detailed Results ---")
        self.sensitivity_results_text.append("Points | Max Slope | Relative Change")
        self.sensitivity_results_text.append("-------|-----------|----------------")
        
        for i in range(len(results["omega_points"])):
            points = results["omega_points"][i]
            slope = results["max_slopes"][i]
            change = results["relative_changes"][i] if i < len(results["relative_changes"]) else float('nan')
            
            if not np.isnan(change):
                change_str = f"{change:.6f}"
            else:
                change_str = "N/A"
                
            self.sensitivity_results_text.append(f"{points:6d} | {slope:10.6f} | {change_str}")
                
        # If user selected to use optimal points, update the FRF omega points setting
        if self.sensitivity_use_optimal.isChecked():
            # Use the highest points value we calculated, or the requested max if we reached it
            points_to_use = requested_max if all_analyzed else highest_analyzed
            
            # Update UI
            self.omega_points_box.setValue(points_to_use)
            self.sensitivity_results_text.append(f"\nAutomatically updated Frequency tab's Ω Points to {points_to_use}")
            
        # Create visualization using our improved dual-plot system
        self.refresh_sensitivity_plot()
            
        # Enable the buttons for plot interaction
        self.sensitivity_save_plot_btn.setEnabled(True)
        self.sensitivity_refresh_plot_btn.setEnabled(True)

    def handle_sensitivity_error(self, error_msg):
        """Handle errors in the Omega points sensitivity analysis"""
        # Re-enable run button
        self.run_sensitivity_btn.setEnabled(True)
        
        # Update status
        self.status_bar.showMessage("Omega points sensitivity analysis failed")
        
        # Display error message
        self.sensitivity_results_text.append(f"\n❌ ERROR: {error_msg}")
        
        # Also show a message box
        QMessageBox.critical(self, "Sensitivity Analysis Error", error_msg)
        
    def save_sensitivity_plot(self):
        """Save the current sensitivity analysis plot to a file"""
        # Determine which tab is active and save that plot
        current_tab_idx = self.vis_tabs.currentIndex()
        
        if current_tab_idx == 0:  # Convergence plot
            if not hasattr(self, 'convergence_fig') or self.convergence_fig is None:
                QMessageBox.warning(self, "Error", "No convergence plot to save.")
                return
                
            self.save_plot(self.convergence_fig, "Slope_Convergence_Analysis")
            
        elif current_tab_idx == 1:  # Relative change plot
            if not hasattr(self, 'rel_change_fig') or self.rel_change_fig is None:
                QMessageBox.warning(self, "Error", "No relative change plot to save.")
                return
                
            self.save_plot(self.rel_change_fig, "Relative_Change_Analysis")
    

