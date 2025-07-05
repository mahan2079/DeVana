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
        
