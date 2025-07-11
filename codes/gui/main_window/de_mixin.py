from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from workers.DEWorker import DEWorker
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class DEOptimizationMixin:
    def create_de_tab(self):
        """Create the differential evolution optimization tab"""
        # Create the main widget
        self.de_tab = QWidget()
        
        # Create and set the main layout
        layout = QVBoxLayout()
        self.de_tab.setLayout(layout)
        
        # Create sub-tabs widget
        self.de_sub_tabs = QTabWidget()

        # -------------------- Sub-tab 1: DE Settings --------------------
        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)

        # DE Algorithm Parameters Group
        de_params_group = QGroupBox("DE Algorithm Parameters")
        de_params_layout = QFormLayout(de_params_group)

        # Population Size
        self.de_pop_size_spinbox = QSpinBox()
        self.de_pop_size_spinbox.setRange(10, 1000)
        self.de_pop_size_spinbox.setValue(50)
        de_params_layout.addRow("Population Size:", self.de_pop_size_spinbox)

        # Number of Generations
        self.de_num_generations_spinbox = QSpinBox()
        self.de_num_generations_spinbox.setRange(10, 10000)
        self.de_num_generations_spinbox.setValue(100)
        de_params_layout.addRow("Number of Generations:", self.de_num_generations_spinbox)

        # Mutation Factor (F)
        self.de_F_spinbox = QDoubleSpinBox()
        self.de_F_spinbox.setRange(0.1, 2.0)
        self.de_F_spinbox.setValue(0.5)
        self.de_F_spinbox.setSingleStep(0.1)
        de_params_layout.addRow("Mutation Factor (F):", self.de_F_spinbox)

        # Crossover Rate (CR)
        self.de_CR_spinbox = QDoubleSpinBox()
        self.de_CR_spinbox.setRange(0.0, 1.0)
        self.de_CR_spinbox.setValue(0.7)
        self.de_CR_spinbox.setSingleStep(0.1)
        de_params_layout.addRow("Crossover Rate (CR):", self.de_CR_spinbox)

        # Tolerance
        self.de_tol_spinbox = QDoubleSpinBox()
        self.de_tol_spinbox.setRange(1e-10, 1.0)
        self.de_tol_spinbox.setValue(1e-6)
        self.de_tol_spinbox.setDecimals(10)
        de_params_layout.addRow("Convergence Tolerance:", self.de_tol_spinbox)

        # DE Strategy Selection
        self.de_strategy_combo = QComboBox()
        self.de_strategy_combo.addItems([
            "rand/1", "rand/2", "best/1", "best/2",
            "current-to-best/1", "current-to-rand/1"
        ])
        de_params_layout.addRow("DE Strategy:", self.de_strategy_combo)

        # Adaptive Method Selection
        self.de_adaptive_combo = QComboBox()
        self.de_adaptive_combo.addItems([
            "none", "jitter", "dither", "sade", "jade", "success-history"
        ])
        de_params_layout.addRow("Adaptive Method:", self.de_adaptive_combo)

        # Number of independent runs for benchmarking
        self.de_num_runs_spinbox = QSpinBox()
        self.de_num_runs_spinbox.setRange(1, 100)
        self.de_num_runs_spinbox.setValue(1)
        self.de_num_runs_spinbox.setToolTip("Number of DE runs (1 = single run)")
        de_params_layout.addRow("Benchmark Runs:", self.de_num_runs_spinbox)

        # Add the parameters group to the layout
        settings_layout.addWidget(de_params_group)
        
        # Add a small Run DE button in the settings sub-tab
        self.hyper_run_de_button = QPushButton("Run DE")
        self.hyper_run_de_button.setFixedWidth(100)
        self.hyper_run_de_button.clicked.connect(self.run_de)
        run_button_layout = QHBoxLayout()
        run_button_layout.addWidget(self.hyper_run_de_button)
        run_button_layout.addStretch()
        settings_layout.addLayout(run_button_layout)

        # -------------------- Sub-tab 2: DVA Parameters --------------------
        dva_params_tab = QWidget()
        dva_params_layout = QVBoxLayout(dva_params_tab)

        self.de_dva_params_table = QTableWidget()
        dva_parameters = [
            *[f"beta_{i}" for i in range(1,16)],
            *[f"lambda_{i}" for i in range(1,16)],
            *[f"mu_{i}" for i in range(1,4)],
            *[f"nu_{i}" for i in range(1,16)]
        ]
        self.de_dva_params_table.setRowCount(len(dva_parameters))
        self.de_dva_params_table.setColumnCount(5)
        self.de_dva_params_table.setHorizontalHeaderLabels(
            ["Parameter", "Fixed", "Fixed Value", "Lower Bound", "Upper Bound"]
        )
        self.de_dva_params_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.de_dva_params_table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        # Set up table rows
        for row, param in enumerate(dva_parameters):
            param_item = QTableWidgetItem(param)
            param_item.setFlags(Qt.ItemIsEnabled)
            self.de_dva_params_table.setItem(row, 0, param_item)

            fixed_checkbox = QCheckBox()
            fixed_checkbox.stateChanged.connect(lambda state, r=row: self.toggle_de_dva_fixed(state, r))
            self.de_dva_params_table.setCellWidget(row, 1, fixed_checkbox)

            fixed_value_spin = QDoubleSpinBox()
            fixed_value_spin.setRange(-1e6, 1e6)
            fixed_value_spin.setDecimals(6)
            fixed_value_spin.setEnabled(False)
            self.de_dva_params_table.setCellWidget(row, 2, fixed_value_spin)

            lower_bound_spin = QDoubleSpinBox()
            lower_bound_spin.setRange(-1e6, 1e6)
            lower_bound_spin.setDecimals(6)
            lower_bound_spin.setEnabled(True)
            self.de_dva_params_table.setCellWidget(row, 3, lower_bound_spin)

            upper_bound_spin = QDoubleSpinBox()
            upper_bound_spin.setRange(-1e6, 1e6)
            upper_bound_spin.setDecimals(6)
            upper_bound_spin.setEnabled(True)
            self.de_dva_params_table.setCellWidget(row, 4, upper_bound_spin)

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

        dva_params_layout.addWidget(self.de_dva_params_table)

        # Control Buttons
        button_layout = QHBoxLayout()
        
        self.run_de_button = QPushButton("Run DE Optimization")
        self.run_de_button.clicked.connect(self.run_de)
        button_layout.addWidget(self.run_de_button)
        
        self.tune_de_button = QPushButton("Tune DE Parameters")
        self.tune_de_button.clicked.connect(self.tune_de_hyperparameters)
        button_layout.addWidget(self.tune_de_button)
        
        dva_params_layout.addLayout(button_layout)

        # -------------------- Sub-tab 3: Results --------------------
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)

        # Header with export button
        header_container = QWidget()
        header_layout = QHBoxLayout(header_container)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.addWidget(QLabel("DE Optimization Results:"))
        header_layout.addStretch()
        self.export_de_results_button = QPushButton("Export DE Results")
        self.export_de_results_button.setToolTip("Export the DE optimization results to a JSON file")
        self.export_de_results_button.setEnabled(False)
        self.export_de_results_button.clicked.connect(self.export_de_results_to_file)
        header_layout.addWidget(self.export_de_results_button)
        results_layout.addWidget(header_container)

        # Progress bar (hidden until run starts)
        self.de_progress_bar = QProgressBar()
        self.de_progress_bar.setRange(0, 100)
        self.de_progress_bar.setValue(0)
        self.de_progress_bar.setTextVisible(True)
        self.de_progress_bar.setFormat("DE Progress: %p%")
        self.de_progress_bar.hide()
        results_layout.addWidget(self.de_progress_bar)

        # Add Results visualization split
        results_splitter = QSplitter(Qt.Vertical)

        # Text results area
        text_results_widget = QWidget()
        text_results_layout = QVBoxLayout(text_results_widget)
        self.de_results_text = QTextEdit()
        self.de_results_text.setReadOnly(True)
        text_results_layout.addWidget(self.de_results_text)

        results_splitter.addWidget(text_results_widget)
        
        # Visualization area
        viz_widget = QWidget()
        viz_layout = QVBoxLayout(viz_widget)
        
        viz_layout.addWidget(QLabel("Optimization Progress:"))
        
        # Create plot canvas for real-time visualization
        self.de_fig = Figure(figsize=(8, 6))
        self.de_canvas = FigureCanvasQTAgg(self.de_fig)
        viz_layout.addWidget(self.de_canvas)

        # Add toolbar for plot interaction
        toolbar = NavigationToolbar2QT(self.de_canvas, viz_widget)
        viz_layout.addWidget(toolbar)

        # Save visualization button
        save_button = QPushButton("Save Visualization")
        save_button.clicked.connect(self.save_de_visualization)
        viz_layout.addWidget(save_button)
        
        results_splitter.addWidget(viz_widget)
        
        # Set reasonable splitter sizes
        results_splitter.setSizes([200, 400])
        
        results_layout.addWidget(results_splitter)

        # -------------------- Sub-tab 4: Benchmarking --------------------
        benchmark_tab = QWidget()
        benchmark_layout = QVBoxLayout(benchmark_tab)

        self.de_benchmark_table = QTableWidget()
        self.de_benchmark_table.setColumnCount(4)
        self.de_benchmark_table.setHorizontalHeaderLabels([
            "Run #", "Best Fitness", "Conv. Gen", "Exec Time (s)"
        ])
        self.de_benchmark_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        benchmark_layout.addWidget(self.de_benchmark_table)

        self.de_benchmark_fig = Figure(figsize=(6, 4))
        self.de_benchmark_canvas = FigureCanvasQTAgg(self.de_benchmark_fig)
        benchmark_layout.addWidget(self.de_benchmark_canvas)

        self.de_sub_tabs.addTab(settings_tab, "DE Settings")
        self.de_sub_tabs.addTab(dva_params_tab, "DVA Parameters")
        self.de_sub_tabs.addTab(results_tab, "Results")
        self.de_sub_tabs.addTab(benchmark_tab, "DE Benchmarking")

        # Add sub-tabs to main layout
        layout.addWidget(self.de_sub_tabs)

        # Initialize the old parameter table for backward compatibility
        self.de_params_table = QTableWidget()  # Create a dummy table for compatibility
        
        # Make sure the tab is properly set up before returning
        self.de_tab.setLayout(layout)
        
        # Return the tab
        return self.de_tab

    def initialize_de_parameter_table(self):
        """Initialize the DE parameter table with default values - kept for backward compatibility"""
        # This method is kept for backward compatibility
        # The DVA parameter table is now initialized directly in create_de_tab
        pass

    def get_parameter_data(self):
        """Get parameter data for optimization algorithms
        
        Returns:
            List of tuples: (name, lower_bound, upper_bound, fixed_flag)
        """
        # Create a list of parameter names
        parameter_data = []
        
        # Beta parameters (15)
        for i in range(1, 16):
            parameter_data.append((f"beta_{i}", 0.0001, 2.5, False))
            
        # Lambda parameters (15)
        for i in range(1, 16):
            parameter_data.append((f"lambda_{i}", 0.0001, 2.5, False))
            
        # Mu parameters (3)
        for i in range(1, 4):
            parameter_data.append((f"mu_{i}", 0.0001, 0.75, False))
            
        # Nu parameters (15)
        for i in range(1, 16):
            parameter_data.append((f"nu_{i}", 0.0001, 2.5, False))
            
        return parameter_data

    def toggle_de_fixed(self, state, row, table=None):
        """Toggle the fixed state of a DE parameter row"""
        if table is None:
            table = self.de_params_table
            
        fixed = (state == Qt.Checked)
        
        # Get the widgets from the correct columns
        # Column 0: Parameter name
        # Column 1: Lower bound
        # Column 2: Upper bound
        # Column 3: Fixed checkbox
        
        # For fixed parameters, we'll use the lower bound value as the fixed value
        lower_bound_item = table.item(row, 1)
        upper_bound_item = table.item(row, 2)
        
        if fixed:
            # When fixed, make both bounds the same
            upper_bound_item.setText(lower_bound_item.text())
        else:
            # When not fixed, set a reasonable range
            param_name = table.item(row, 0).text()
            if param_name.startswith(("beta_", "lambda_", "nu_")):
                lower_bound_item.setText("0.0001")
                upper_bound_item.setText("2.5")
            elif param_name.startswith("mu_"):
                lower_bound_item.setText("0.0001")
                upper_bound_item.setText("0.75")
            else:
                lower_bound_item.setText("0.0")
                upper_bound_item.setText("1.0")

    def toggle_de_dva_fixed(self, state, row, table=None):
        """Toggle the fixed state of a DE DVA parameter row"""
        if table is None:
            table = self.de_dva_params_table
            
        fixed = (state == Qt.Checked)
        fixed_value_spin = table.cellWidget(row, 2)
        lower_bound_spin = table.cellWidget(row, 3)
        upper_bound_spin = table.cellWidget(row, 4)

        fixed_value_spin.setEnabled(fixed)
        lower_bound_spin.setEnabled(not fixed)
        upper_bound_spin.setEnabled(not fixed)
        
        if fixed:
            # When fixed, set a default value
            param_name = table.item(row, 0).text()
            if param_name.startswith(("beta_", "lambda_", "nu_")):
                fixed_value_spin.setValue(0.5)  # Default value for these parameters
            elif param_name.startswith("mu_"):
                fixed_value_spin.setValue(0.1)  # Default value for mu parameters
            else:
                fixed_value_spin.setValue(0.0)  # Default for others

    def run_de(self):
        """Run the differential evolution optimization"""
        try:
            if hasattr(self, 'de_worker') and self.de_worker.isRunning():
                QMessageBox.warning(self, "Process Running",
                                   "A Differential Evolution optimization is already running. Please wait for it to complete.")
                return

            # Get parameter data from the DVA table
            parameter_data = []
            for row in range(self.de_dva_params_table.rowCount()):
                name = self.de_dva_params_table.item(row, 0).text()
                is_fixed = self.de_dva_params_table.cellWidget(row, 1).isChecked()
                
                if is_fixed:
                    # For fixed parameters, use the fixed value for both lower and upper bounds
                    value = self.de_dva_params_table.cellWidget(row, 2).value()
                    parameter_data.append((name, value, value, True))
                else:
                    # For non-fixed parameters, use the lower and upper bounds
                    lower = self.de_dva_params_table.cellWidget(row, 3).value()
                    upper = self.de_dva_params_table.cellWidget(row, 4).value()
                    parameter_data.append((name, lower, upper, False))

            # Get DE parameters from the GUI
            de_params = {
                'de_pop_size': self.de_pop_size_spinbox.value(),
                'de_num_generations': self.de_num_generations_spinbox.value(),
                'de_F': self.de_F_spinbox.value(),
                'de_CR': self.de_CR_spinbox.value(),
                'de_tol': self.de_tol_spinbox.value(),
                'strategy': self.de_strategy_combo.currentText(),
                'adaptive': self.de_adaptive_combo.currentText(),
                'num_runs': self.de_num_runs_spinbox.value()
            }

            # Store run information for progress tracking
            self.total_de_runs = de_params['num_runs']
            self.current_de_run = 1
            
            # Show a message in the results area
            self.de_results_text.clear()
            self.de_results_text.append("Starting Differential Evolution optimization...")
            self.de_results_text.append(f"Population size: {de_params['de_pop_size']}")
            self.de_results_text.append(f"Generations: {de_params['de_num_generations']}")
            self.de_results_text.append(f"Mutation factor (F): {de_params['de_F']}")
            self.de_results_text.append(f"Crossover rate (CR): {de_params['de_CR']}")
            self.de_results_text.append(f"Strategy: {de_params['strategy']}")
            self.de_results_text.append(f"Adaptive method: {de_params['adaptive']}")
            self.de_results_text.append(f"Benchmark runs: {de_params['num_runs']}")
            self.de_results_text.append("-------------------")

            # Get main parameters
            try:
                main_params = self.get_main_system_params()
                target_values, weights = self.get_target_values_weights()

                omega_start = self.omega_start_box.value()
                omega_end = self.omega_end_box.value()
                omega_points = self.omega_points_box.value()
                
                self.de_results_text.append(f"Frequency range: {omega_start} - {omega_end}")
                self.de_results_text.append(f"Number of points: {omega_points}")
                self.de_results_text.append("-------------------")
                
                # Show the worker is running
                self.de_results_text.append("Optimization in progress...")
                
                # Prepare the figure for visualization
                self.de_fig.clear()
                self.ax1 = self.de_fig.add_subplot(211)
                self.ax2 = self.de_fig.add_subplot(212)
                self.ax1.set_title("Best Fitness")
                self.ax2.set_title("Population Diversity")
                self.ax1.set_xlabel("Generation")
                self.ax1.set_ylabel("Fitness")
                self.ax2.set_xlabel("Generation")
                self.ax2.set_ylabel("Diversity")
                self.fitness_values = []
                self.diversity_values = []
                self.generations = []
                self.de_canvas.draw()

                # Show progress bar
                if hasattr(self, 'de_progress_bar'):
                    self.de_progress_bar.setValue(0)
                    self.de_progress_bar.show()

                if hasattr(self, 'run_de_button'):
                    self.run_de_button.setEnabled(False)

                # Create and start worker thread with correct parameters
                self.de_worker = DEWorker(
                    main_params=main_params,
                    target_values_dict=target_values,
                    weights_dict=weights,
                    omega_start=omega_start,
                    omega_end=omega_end,
                    omega_points=omega_points,
                    de_pop_size=de_params['de_pop_size'],
                    de_num_generations=de_params['de_num_generations'],
                    de_F=de_params['de_F'],
                    de_CR=de_params['de_CR'],
                    de_tol=de_params['de_tol'],
                    de_parameter_data=parameter_data,
                    strategy=de_params['strategy'],
                    adaptive_method=de_params['adaptive'],
                    record_statistics=True,
                    num_runs=de_params['num_runs']
                )
                self.de_worker.progress.connect(self.handle_de_progress)
                self.de_worker.multi_run_progress.connect(self.handle_de_multi_run_progress)
                self.de_worker.finished.connect(self.handle_de_finished)
                self.de_worker.error.connect(self.handle_de_error)
                self.de_worker.update.connect(self.handle_de_update)
                
                # Show the DE tab with Results subtab
                self.de_sub_tabs.setCurrentIndex(2)  # Switch to Results tab
                
                self.de_worker.start()
                
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Error",
                    f"Failed to start optimization: {str(e)}"
                )
                self.de_results_text.append(f"ERROR: {str(e)}")

        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to run DE optimization: {str(e)}"
            )
            self.de_results_text.append(f"ERROR: {str(e)}")

    def handle_de_progress(self, generation, best_fitness, diversity):
        """Handle progress updates from the DE optimization"""
        try:
            # Add the data points
            self.generations.append(generation)
            self.fitness_values.append(best_fitness)
            self.diversity_values.append(diversity)

            if hasattr(self, 'de_progress_bar') and self.de_worker:
                total_gen = self.de_worker.de_num_generations
                run_fraction = (generation / total_gen)
                overall = ((getattr(self, 'current_de_run', 1) - 1) + run_fraction) / getattr(self, 'total_de_runs', 1)
                self.de_progress_bar.setValue(int(overall * 100))
            
            # Update the results text
            if generation % 5 == 0:  # Update every 5 generations to avoid too many updates
                self.de_results_text.append(f"Generation {generation}: Fitness={best_fitness:.6f}, Diversity={diversity:.6f}")
                
                # Make sure the text area scrolls to show the latest update
                scrollbar = self.de_results_text.verticalScrollBar()
                scrollbar.setValue(scrollbar.maximum())
                
            # Update the visualization
            self.update_de_visualization()
            
        except Exception as e:
            print(f"Error updating DE progress: {str(e)}")
    
    def update_de_visualization(self):
        """Update the DE visualization with current data"""
        try:
            # Clear the plots
            self.ax1.clear()
            self.ax2.clear()
            
            # Set titles and labels
            self.ax1.set_title("Best Fitness")
            self.ax2.set_title("Population Diversity")
            self.ax1.set_xlabel("Generation")
            self.ax1.set_ylabel("Fitness")
            self.ax2.set_xlabel("Generation")
            self.ax2.set_ylabel("Diversity")
            
            # Plot the data
            if self.generations and self.fitness_values:
                self.ax1.plot(self.generations, self.fitness_values, 'b-')
                self.ax2.plot(self.generations, self.diversity_values, 'r-')
                
                # Set the y-axis limits with some padding
                min_fitness = min(self.fitness_values)
                max_fitness = max(self.fitness_values)
                padding = 0.1 * (max_fitness - min_fitness) if max_fitness > min_fitness else 0.1
                self.ax1.set_ylim([min_fitness - padding, max_fitness + padding])
                
                min_diversity = min(self.diversity_values)
                max_diversity = max(self.diversity_values)
                padding = 0.1 * (max_diversity - min_diversity) if max_diversity > min_diversity else 0.1
                self.ax2.set_ylim([min_diversity - padding, max_diversity + padding])
                
            # Redraw the canvas
            self.de_fig.tight_layout()
            self.de_canvas.draw()
            
            # Process pending events to update the UI
            QApplication.processEvents()
            
        except Exception as e:
            print(f"Error updating DE visualization: {str(e)}")

    def handle_de_multi_run_progress(self, current_run, total_runs):
        """Handle progress between multiple DE runs"""
        try:
            self.current_de_run = current_run
            self.total_de_runs = total_runs
            self.de_results_text.append(f"\n--- Starting run {current_run}/{total_runs} ---")
            if hasattr(self, 'statusBar'):
                self.statusBar().showMessage(f"DE optimization run {current_run} of {total_runs}")
        except Exception as e:
            print(f"Error handling multi-run progress: {str(e)}")

    def handle_de_finished(self, results, best_individual, parameter_names, best_fitness, statistics):
        """Handle the completion of the DE optimization"""
        try:
            # Store the best parameters and fitness
            self.current_de_best_params = best_individual
            self.current_de_best_fitness = best_fitness
            if hasattr(statistics, 'run_best_fitnesses'):
                # Multi-run statistics
                self.current_de_full_results = {
                    'results': results,
                    'multi_run_stats': statistics.__dict__
                }
                self.visualize_de_benchmark_results(statistics)
            else:
                self.current_de_full_results = results
            
            # Display results in the text area
            self.de_results_text.append("\n=== Optimization Complete ===")
            self.de_results_text.append(f"Best fitness: {best_fitness:.6f}")
            self.de_results_text.append("\nBest parameter values:")
            
            # Create a formatted table of best parameters
            for i, (name, value) in enumerate(zip(parameter_names, best_individual)):
                self.de_results_text.append(f"{name:<10}: {value:.6f}")
                
            # Create the final visualization
            self.create_de_final_visualization(statistics, parameter_names)

            # Switch to appropriate tab to show final results
            if getattr(self, 'total_de_runs', 1) > 1:
                self.de_sub_tabs.setCurrentIndex(3)
            else:
                self.de_sub_tabs.setCurrentIndex(2)

            if hasattr(self, 'de_progress_bar'):
                self.de_progress_bar.setValue(100)
                self.de_progress_bar.hide()

            if hasattr(self, 'export_de_results_button'):
                self.export_de_results_button.setEnabled(True)
            
            # Enable the run button again
            if hasattr(self, 'run_de_button'):
                self.run_de_button.setEnabled(True)
                
            # Update status
            if hasattr(self, 'statusBar'):
                self.statusBar().showMessage("DE optimization completed", 5000)
                
            # Show completion message
            QMessageBox.information(
                self,
                "Optimization Complete",
                f"Differential Evolution optimization completed successfully.\nBest fitness: {best_fitness:.6f}"
            )
            
        except Exception as e:
            self.de_results_text.append(f"Error processing results: {str(e)}")
            QMessageBox.warning(
                self,
                "Error Processing Results",
                f"An error occurred while processing the optimization results: {str(e)}"
            )

    def create_de_final_visualization(self, statistics, parameter_names):
        """Create the final visualization after DE optimization completes"""
        try:
            # Clear the figure
            self.de_fig.clear()
            
            # Create a 2x2 subplot layout
            gs = self.de_fig.add_gridspec(2, 2)
            
            # 1. Fitness Evolution
            ax1 = self.de_fig.add_subplot(gs[0, 0])
            ax1.plot(self.generations, self.fitness_values, 'b-', label='Best Fitness')
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Fitness')
            ax1.set_title('Fitness Evolution')
            ax1.grid(True)
            
            # 2. Population Diversity
            ax2 = self.de_fig.add_subplot(gs[0, 1])
            ax2.plot(self.generations, self.diversity_values, 'r-')
            ax2.set_xlabel('Generation')
            ax2.set_ylabel('Diversity')
            ax2.set_title('Population Diversity')
            ax2.grid(True)
            
            # 3. Parameter Distribution (top parameters)
            ax3 = self.de_fig.add_subplot(gs[1, 0])
            
            # Use the best individual values for this plot
            if hasattr(self, 'current_de_best_params') and parameter_names:
                # Select a subset of parameters for visualization (e.g., top 5)
                num_params = min(5, len(parameter_names))
                selected_params = parameter_names[:num_params]
                selected_values = self.current_de_best_params[:num_params]
                
                # Create a bar chart of the top parameters
                y_pos = range(num_params)
                ax3.barh(y_pos, selected_values, align='center')
                ax3.set_yticks(y_pos)
                ax3.set_yticklabels(selected_params)
                ax3.set_xlabel('Value')
                ax3.set_title('Top Parameters')
                ax3.grid(True, axis='x')
            
            # 4. Convergence Rate
            ax4 = self.de_fig.add_subplot(gs[1, 1])
            
            # Calculate convergence rate (improvement between generations)
            if len(self.fitness_values) > 1:
                convergence = []
                for i in range(1, len(self.fitness_values)):
                    improvement = abs(self.fitness_values[i] - self.fitness_values[i-1])
                    convergence.append(improvement)
                
                # Plot convergence rate
                ax4.semilogy(self.generations[1:], convergence, 'g-')
                ax4.set_xlabel('Generation')
                ax4.set_ylabel('Improvement')
                ax4.set_title('Convergence Rate')
                ax4.grid(True)
            
            # Adjust layout and display
            self.de_fig.tight_layout()
            self.de_canvas.draw()
            
            # Add a note to the results text
            self.de_results_text.append("\nVisualization updated with final results.")
            
        except Exception as e:
            self.de_results_text.append(f"Error creating final visualization: {str(e)}")
            print(f"Error creating DE final visualization: {str(e)}")

    def save_de_visualization(self):
        """Save the DE visualization to a file"""
        try:
            # Ask for file path using dialog
            file_path, _ = QFileDialog.getSaveFileName(
                None, "Save DE Visualization", "", "PNG Files (*.png);;PDF Files (*.pdf);;All Files (*)"
            )
            
            if file_path:
                # Save the figure to the file
                self.de_fig.savefig(file_path, dpi=300, bbox_inches='tight')
                
                # Also offer to save the results as text
                text_path = file_path.rsplit('.', 1)[0] + ".txt"
                reply = QMessageBox.question(
                    None, 
                    "Save Text Results",
                    "Do you also want to save the text results?",
                    QMessageBox.Yes | QMessageBox.No
                )
                
                if reply == QMessageBox.Yes:
                    with open(text_path, 'w') as f:
                        f.write(self.de_results_text.toPlainText())
                    
                QMessageBox.information(None, "File Saved", f"Visualization saved to {file_path}")
                self.de_results_text.append(f"\nVisualization saved to {file_path}")
                
        except Exception as e:
            QMessageBox.critical(None, "Save Error", f"Failed to save visualization: {str(e)}")
            self.de_results_text.append(f"Error saving visualization: {str(e)}")

    def export_de_results_to_file(self):
        """Export DE optimization results to a JSON file"""
        try:
            if not hasattr(self, 'current_de_full_results'):
                QMessageBox.warning(self, "No Results", "No DE optimization results available to export.")
                return

            file_path, _ = QFileDialog.getSaveFileName(self, "Export DE Results", "", "JSON Files (*.json)")
            if file_path:
                import json
                with open(file_path, 'w') as f:
                    json.dump(self.current_de_full_results, f, indent=4)
                QMessageBox.information(self, "Export Complete", f"Results saved to {file_path}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Failed to export results: {str(e)}")

    def visualize_de_benchmark_results(self, stats):
        """Visualize results from multiple DE runs"""
        try:
            if not stats or not stats.run_best_fitnesses:
                return

            df = pd.DataFrame({
                'Run': list(range(1, len(stats.run_best_fitnesses) + 1)),
                'BestFitness': stats.run_best_fitnesses,
                'ConvergenceGen': stats.run_convergence_gens,
                'ExecTime': stats.run_execution_times
            })

            self.de_benchmark_table.setRowCount(len(df))
            for i, row in df.iterrows():
                self.de_benchmark_table.setItem(i, 0, QTableWidgetItem(str(row['Run'])))
                self.de_benchmark_table.setItem(i, 1, QTableWidgetItem(f"{row['BestFitness']:.6f}"))
                self.de_benchmark_table.setItem(i, 2, QTableWidgetItem(str(row['ConvergenceGen'])))
                self.de_benchmark_table.setItem(i, 3, QTableWidgetItem(f"{row['ExecTime']:.2f}"))

            self.de_benchmark_fig.clear()
            ax1 = self.de_benchmark_fig.add_subplot(211)
            sns.violinplot(y=df['BestFitness'], ax=ax1, color='skyblue')
            ax1.set_title('Distribution of Best Fitness')
            ax1.set_ylabel('Fitness')

            ax2 = self.de_benchmark_fig.add_subplot(212)
            sns.violinplot(y=df['ConvergenceGen'], ax=ax2, color='lightgreen')
            ax2.set_title('Distribution of Convergence Generation')
            ax2.set_ylabel('Generation')

            self.de_benchmark_fig.tight_layout()
            self.de_benchmark_canvas.draw()
        except Exception as e:
            print(f"Error visualizing DE benchmark results: {str(e)}")

    def handle_de_error(self, error_msg):
        """Handle errors from the DE optimization worker"""
        try:
            # Add error to results text
            self.de_results_text.append(f"\nERROR: {error_msg}")
            
            # Make sure the text area scrolls to show the error
            scrollbar = self.de_results_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
            
            # Show error message box
            QMessageBox.critical(None, "DE Optimization Error", error_msg)

            if hasattr(self, 'de_progress_bar'):
                self.de_progress_bar.hide()
                self.de_progress_bar.setValue(0)
            
        except Exception as e:
            print(f"Error handling DE error: {str(e)}")

    def handle_de_update(self, msg):
        """Handle update messages from the DE optimization worker"""
        try:
            # Add update to results text
            self.de_results_text.append(f"{msg}")
            
            # Make sure the text area scrolls to show the update
            scrollbar = self.de_results_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
            
        except Exception as e:
            print(f"Error handling DE update: {str(e)}")

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
            row_count = self.de_dva_params_table.rowCount()
            for row in range(row_count):
                param_name = self.de_dva_params_table.item(row, 0).text()
                fixed_widget = self.de_dva_params_table.cellWidget(row, 1)
                fixed = fixed_widget.isChecked()
                if fixed:
                    fixed_value_widget = self.de_dva_params_table.cellWidget(row, 2)
                    fv = fixed_value_widget.value()
                    de_dva_parameters.append((param_name, fv, fv, True))
                else:
                    lower_bound_widget = self.de_dva_params_table.cellWidget(row, 3)
                    upper_bound_widget = self.de_dva_params_table.cellWidget(row, 4)
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
        self.de_pop_size_spinbox.setValue(best_params.get("pop_size", 50))
        self.de_F_spinbox.setValue(best_params.get("F", 0.5))
        self.de_CR_spinbox.setValue(best_params.get("CR", 0.7))
        
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
        
