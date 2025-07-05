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

class ExtraOptimizationMixin:
    def create_de_tab(self):
        """Create the differential evolution optimization tab"""
        # Create the main widget
        self.de_tab = QWidget()
        
        # Create and set the main layout
        layout = QVBoxLayout()
        self.de_tab.setLayout(layout)
        
        # Create sub-tabs widget
        self.de_sub_tabs = QTabWidget()

        # -------------------- Sub-tab 1: DE Parameters --------------------
        params_tab = QWidget()
        params_layout = QVBoxLayout(params_tab)

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

        # Add the parameters group to the layout
        params_layout.addWidget(de_params_group)

        # Parameter Bounds Table
        bounds_group = QGroupBox("Parameter Bounds")
        bounds_layout = QVBoxLayout(bounds_group)
        
        self.de_params_table = QTableWidget()
        self.de_params_table.setColumnCount(4)
        self.de_params_table.setHorizontalHeaderLabels(["Parameter", "Lower Bound", "Upper Bound", "Fixed"])
        bounds_layout.addWidget(self.de_params_table)
        
        params_layout.addWidget(bounds_group)

        # Control Buttons
        button_layout = QHBoxLayout()
        
        self.run_de_button = QPushButton("Run DE Optimization")
        self.run_de_button.clicked.connect(self.run_de)
        button_layout.addWidget(self.run_de_button)
        
        self.tune_de_button = QPushButton("Tune DE Parameters")
        self.tune_de_button.clicked.connect(self.tune_de_hyperparameters)
        button_layout.addWidget(self.tune_de_button)
        
        params_layout.addLayout(button_layout)

        # -------------------- Sub-tab 2: Visualization --------------------
        viz_tab = QWidget()
        viz_layout = QVBoxLayout(viz_tab)

        # Create plot canvas for real-time visualization
        self.de_fig = Figure(figsize=(8, 6))
        self.de_canvas = FigureCanvasQTAgg(self.de_fig)
        viz_layout.addWidget(self.de_canvas)

        # Add toolbar for plot interaction
        toolbar = NavigationToolbar2QT(self.de_canvas, viz_tab)
        viz_layout.addWidget(toolbar)

        # Save visualization button
        save_button = QPushButton("Save Visualization")
        save_button.clicked.connect(self.save_de_visualization)
        viz_layout.addWidget(save_button)

        # Add tabs to sub-tabs widget
        self.de_sub_tabs.addTab(params_tab, "Parameters")
        self.de_sub_tabs.addTab(viz_tab, "Visualization")

        # Add sub-tabs to main layout
        layout.addWidget(self.de_sub_tabs)

        # Initialize the parameter table
        self.initialize_de_parameter_table()
        
        # Make sure the tab is properly set up before returning
        self.de_tab.setLayout(layout)

    def initialize_de_parameter_table(self):
        """Initialize the DE parameter table with default values"""
        # Get the parameter data from the main window
        parameter_data = self.get_parameter_data()
        
        # Set the number of rows based on parameters
        self.de_params_table.setRowCount(len(parameter_data))
        
        # Fill the table with parameter data
        for row, (name, lower, upper, fixed) in enumerate(parameter_data):
            # Parameter name
            name_item = QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.de_params_table.setItem(row, 0, name_item)
            
            # Lower bound
            lower_item = QTableWidgetItem(str(lower))
            self.de_params_table.setItem(row, 1, lower_item)
            
            # Upper bound
            upper_item = QTableWidgetItem(str(upper))
            self.de_params_table.setItem(row, 2, upper_item)
            
            # Fixed checkbox
            checkbox = QCheckBox()
            checkbox.setChecked(fixed)
            checkbox.stateChanged.connect(lambda state, r=row: self.toggle_de_fixed(state, r))
            self.de_params_table.setCellWidget(row, 3, checkbox)
        
        # Adjust column widths
        self.de_params_table.resizeColumnsToContents()

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

    def run_de(self):
        """Run the differential evolution optimization"""
        try:
            # Get parameter data from the table
            parameter_data = []
            for row in range(self.de_params_table.rowCount()):
                name = self.de_params_table.item(row, 0).text()
                lower = float(self.de_params_table.item(row, 1).text())
                upper = float(self.de_params_table.item(row, 2).text())
                fixed = self.de_params_table.cellWidget(row, 3).isChecked()
                parameter_data.append((name, lower, upper, fixed))

            # Get DE parameters from the GUI
            de_params = {
                'de_pop_size': self.de_pop_size_spinbox.value(),
                'de_num_generations': self.de_num_generations_spinbox.value(),
                'de_F': self.de_F_spinbox.value(),
                'de_CR': self.de_CR_spinbox.value(),
                'de_tol': self.de_tol_spinbox.value(),
                'strategy': self.de_strategy_combo.currentText(),
                'adaptive_method': self.de_adaptive_combo.currentText()
            }

            # Create and configure the DE worker
            self.de_worker = DEWorker(
                main_params=self.main_params,
                target_values_dict=self.target_values,
                weights_dict=self.weights,
                omega_start=self.omega_start,
                omega_end=self.omega_end,
                omega_points=self.omega_points,
                de_pop_size=de_params['de_pop_size'],
                de_num_generations=de_params['de_num_generations'],
                de_F=de_params['de_F'],
                de_CR=de_params['de_CR'],
                de_tol=de_params['de_tol'],
                de_parameter_data=parameter_data,
                strategy=de_params['strategy'],
                adaptive_method=de_params['adaptive_method'],
                record_statistics=True
            )

            # Connect signals
            self.de_worker.finished.connect(self.handle_de_finished)
            self.de_worker.error.connect(self.handle_de_error)
            self.de_worker.update.connect(self.handle_de_update)
            self.de_worker.progress.connect(self.handle_de_progress)

            # Disable the run button and update status
            self.run_de_button.setEnabled(False)
            self.statusBar().showMessage("Running DE optimization...")

            # Start the worker
            self.de_worker.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start DE optimization: {str(e)}")
            self.run_de_button.setEnabled(True)

    def handle_de_progress(self, generation, best_fitness, diversity):
        """Handle progress updates from the DE worker"""
        # Update progress bar (assuming 100 generations)
        progress = int((generation / self.de_num_generations_spinbox.value()) * 100)
        self.statusBar().showMessage(f"Generation {generation}: Best Fitness = {best_fitness:.6f}, Diversity = {diversity:.6f}")
        
        # Update visualization
        self.update_de_visualization()

    def update_de_visualization(self):
        """Update the DE visualization during optimization"""
        try:
            # Check if we have a worker with statistics
            if not hasattr(self, 'de_worker') or not hasattr(self.de_worker, 'statistics'):
                return
                
            # Get statistics from the worker
            stats = self.de_worker.statistics
            
            # Clear the figure
            self.de_fig.clear()
            
            # Create subplots
            gs = self.de_fig.add_gridspec(2, 2)
            
            # 1. Fitness Evolution
            ax1 = self.de_fig.add_subplot(gs[0, 0])
            if stats.generations and stats.best_fitness_history:
                ax1.plot(stats.generations, stats.best_fitness_history, label='Best Fitness')
                if stats.mean_fitness_history:
                    ax1.plot(stats.generations, stats.mean_fitness_history, label='Mean Fitness')
                ax1.set_xlabel('Generation')
                ax1.set_ylabel('Fitness')
                ax1.set_title('Fitness Evolution')
                ax1.legend()
                ax1.grid(True)
            
            # 2. Population Diversity
            ax2 = self.de_fig.add_subplot(gs[0, 1])
            if stats.generations and stats.diversity_history:
                ax2.plot(stats.generations, stats.diversity_history)
                ax2.set_xlabel('Generation')
                ax2.set_ylabel('Diversity')
                ax2.set_title('Population Diversity')
                ax2.grid(True)
            
            # 3. Control Parameters (F and CR values if adaptive)
            ax3 = self.de_fig.add_subplot(gs[1, 0])
            if stats.generations and (stats.f_values or stats.cr_values):
                if stats.f_values:
                    ax3.plot(stats.generations, stats.f_values, label='F')
                if stats.cr_values:
                    ax3.plot(stats.generations, stats.cr_values, label='CR')
                ax3.set_xlabel('Generation')
                ax3.set_ylabel('Value')
                ax3.set_title('Control Parameters')
                ax3.legend()
                ax3.grid(True)
            
            # 4. Success Rates
            ax4 = self.de_fig.add_subplot(gs[1, 1])
            if stats.generations and stats.success_rates:
                ax4.plot(stats.generations, stats.success_rates)
                ax4.set_xlabel('Generation')
                ax4.set_ylabel('Success Rate')
                ax4.set_title('Success Rate')
                ax4.grid(True)
            
            # Adjust layout and display
            self.de_fig.tight_layout()
            self.de_canvas.draw()
            
        except Exception as e:
            # Silently fail - we don't want to interrupt the optimization
            print(f"Error updating DE visualization: {str(e)}")
            pass

    def handle_de_finished(self, results, best_individual, parameter_names, best_fitness, statistics):
        """Handle completion of the DE optimization"""
        try:
            # Re-enable the run button
            self.run_de_button.setEnabled(True)
            
            # Update status
            self.statusBar().showMessage(f"DE optimization completed. Best fitness: {best_fitness:.6f}")
            
            # Store results
            self.de_results = results
            self.de_best_individual = best_individual
            self.de_statistics = statistics
            
            # Create final visualization
            self.create_de_final_visualization(statistics, parameter_names)
            
            # Show results in a message box
            result_text = "Optimization completed successfully!\n\n"
            result_text += f"Best Fitness: {best_fitness:.6f}\n\n"
            result_text += "Best Parameters:\n"
            for name, value in zip(parameter_names, best_individual):
                result_text += f"{name}: {value:.6f}\n"
            
            QMessageBox.information(self, "DE Optimization Complete", result_text)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error handling DE results: {str(e)}")

    def create_de_final_visualization(self, statistics, parameter_names):
        """Create the final visualization for DE results"""
        try:
            # Clear the figure
            self.de_fig.clear()
            
            # Create subplots
            gs = self.de_fig.add_gridspec(2, 2)
            
            # 1. Fitness Evolution
            ax1 = self.de_fig.add_subplot(gs[0, 0])
            ax1.plot(statistics.generations, statistics.best_fitness_history, label='Best Fitness')
            ax1.plot(statistics.generations, statistics.mean_fitness_history, label='Mean Fitness')
            ax1.set_xlabel('Generation')
            ax1.set_ylabel('Fitness')
            ax1.set_title('Fitness Evolution')
            ax1.legend()
            ax1.grid(True)
            
            # 2. Population Diversity
            ax2 = self.de_fig.add_subplot(gs[0, 1])
            ax2.plot(statistics.generations, statistics.diversity_history)
            ax2.set_xlabel('Generation')
            ax2.set_ylabel('Diversity')
            ax2.set_title('Population Diversity')
            ax2.grid(True)
            
            # 3. Parameter Convergence
            ax3 = self.de_fig.add_subplot(gs[1, 0])
            for i, param in enumerate(parameter_names):
                ax3.plot(statistics.generations, 
                        [means[i] for means in statistics.parameter_mean_history],
                        label=param)
            ax3.set_xlabel('Generation')
            ax3.set_ylabel('Parameter Value')
            ax3.set_title('Parameter Convergence')
            ax3.legend()
            ax3.grid(True)
            
            # 4. Control Parameters (F and CR values if adaptive)
            ax4 = self.de_fig.add_subplot(gs[1, 1])
            if statistics.f_values and statistics.cr_values:
                ax4.plot(statistics.generations, statistics.f_values, label='F')
                ax4.plot(statistics.generations, statistics.cr_values, label='CR')
                ax4.set_xlabel('Generation')
                ax4.set_ylabel('Value')
                ax4.set_title('Control Parameters')
                ax4.legend()
                ax4.grid(True)
            
            # Adjust layout and display
            self.de_fig.tight_layout()
            self.de_canvas.draw()
            
        except Exception as e:
            QMessageBox.warning(self, "Visualization Error", f"Error creating visualization: {str(e)}")

    def save_de_visualization(self):
        """Save the current DE visualization"""
        try:
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Visualization", "", "PNG Files (*.png);;All Files (*)"
            )
            if filename:
                self.de_fig.savefig(filename, dpi=300, bbox_inches='tight')
                QMessageBox.information(self, "Success", "Visualization saved successfully!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save visualization: {str(e)}")

    def handle_de_error(self, error_msg):
        """Handle errors from the DE worker"""
        self.run_de_button.setEnabled(True)
        QMessageBox.critical(self, "DE Optimization Error", str(error_msg))
        self.statusBar().showMessage("DE optimization failed!")

    def handle_de_update(self, msg):
        """Handle status updates from the DE worker"""
        self.statusBar().showMessage(msg)

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
            row_count = self.de_params_table.rowCount()
            for row in range(row_count):
                param_name = self.de_params_table.item(row, 0).text()
                fixed_widget = self.de_params_table.cellWidget(row, 3)
                fixed = fixed_widget.isChecked()
                if fixed:
                    fixed_value_widget = self.de_params_table.cellWidget(row, 2)
                    fv = fixed_value_widget.value()
                    de_dva_parameters.append((param_name, fv, fv, True))
                else:
                    lower_bound_widget = self.de_params_table.cellWidget(row, 1)
                    upper_bound_widget = self.de_params_table.cellWidget(row, 2)
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
    

