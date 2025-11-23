"""
This module provides the NSGAIIOptimizationMixin for the main window,
which adds the NSGA-II multi-objective optimization tab.
"""
import numpy as np
import pandas as pd
import seaborn as sns
import traceback # Added for detailed error logging
from mpl_toolkits.mplot3d import Axes3D
from pandas.plotting import parallel_coordinates
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSpinBox, QDoubleSpinBox,
    QTabWidget, QGroupBox, QFormLayout, QMessageBox, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView, QSplitter, QComboBox, QProgressBar, QCheckBox
)
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

from workers.NSGAIIWorker import NSGAIIWorker

class NSGAIIOptimizationMixin:
    """
    Mixin class to add the NSGA-II optimization tab to the main window.
    """
    def create_nsgaii_tab(self):
        """Creates the NSGA-II optimization tab."""
        self.nsgaii_tab = QWidget()
        layout = QVBoxLayout(self.nsgaii_tab)

        # --- Main Splitter ---
        splitter = QSplitter(Qt.Horizontal)
        
        # --- Left side: Controls ---
        controls_widget = QWidget()
        controls_layout = QVBoxLayout(controls_widget)

        # Hyperparameters
        hyper_group = QGroupBox("NSGA-II Hyperparameters")
        hyper_layout = QFormLayout(hyper_group)

        self.nsgaii_pop_size_box = QSpinBox()
        self.nsgaii_pop_size_box.setRange(10, 10000)
        self.nsgaii_pop_size_box.setValue(100)
        hyper_layout.addRow("Population Size:", self.nsgaii_pop_size_box)

        self.nsgaii_num_generations_box = QSpinBox()
        self.nsgaii_num_generations_box.setRange(1, 10000)
        self.nsgaii_num_generations_box.setValue(50)
        hyper_layout.addRow("Number of Generations:", self.nsgaii_num_generations_box)

        self.nsgaii_cxpb_box = QDoubleSpinBox()
        self.nsgaii_cxpb_box.setRange(0, 1)
        self.nsgaii_cxpb_box.setValue(0.9)
        hyper_layout.addRow("Crossover Probability:", self.nsgaii_cxpb_box)

        self.nsgaii_mutpb_box = QDoubleSpinBox()
        self.nsgaii_mutpb_box.setRange(0, 1)
        self.nsgaii_mutpb_box.setValue(0.1)
        hyper_layout.addRow("Mutation Probability:", self.nsgaii_mutpb_box)

        self.nsgaii_cost_threshold_box = QDoubleSpinBox()
        self.nsgaii_cost_threshold_box.setDecimals(4)
        self.nsgaii_cost_threshold_box.setRange(0, 1)
        self.nsgaii_cost_threshold_box.setValue(0.001)
        hyper_layout.addRow("Cost Threshold:", self.nsgaii_cost_threshold_box)
        
        controls_layout.addWidget(hyper_group)

        # DVA Parameters
        param_group = QGroupBox("DVA Parameters")
        param_layout = QVBoxLayout(param_group)
        self.nsgaii_param_table = self.create_parameter_table()
        param_layout.addWidget(self.nsgaii_param_table)
        controls_layout.addWidget(param_group)

        self.run_nsgaii_button = QPushButton("Run NSGA-II")
        self.run_nsgaii_button.clicked.connect(self.run_nsgaii)
        controls_layout.addWidget(self.run_nsgaii_button)

        self.nsgaii_progress_bar = QProgressBar()
        self.nsgaii_progress_bar.setVisible(False)
        controls_layout.addWidget(self.nsgaii_progress_bar)


        splitter.addWidget(controls_widget)

        # --- Right side: Results ---
        results_widget = QWidget()
        results_layout = QVBoxLayout(results_widget)
        self.nsgaii_results_tabs = QTabWidget()

        # [NEW] Create and add visualization tabs
        self.nsgaii_plot_tabs = QTabWidget()
        scatter_2d_tab = self._create_2d_scatter_tab()
        self.nsgaii_plot_tabs.addTab(scatter_2d_tab, "2D Scatter")
        
        scatter_3d_tab = self._create_3d_scatter_tab()
        self.nsgaii_plot_tabs.addTab(scatter_3d_tab, "3D Scatter")

        parallel_coord_tab = self._create_parallel_coordinates_tab()
        self.nsgaii_plot_tabs.addTab(parallel_coord_tab, "Parallel Coordinates")

        # Results Table Tab
        table_tab = QWidget()
        table_layout = QVBoxLayout(table_tab)

        # --- Analysis Controls ---
        analysis_group = QGroupBox("Solutions Analysis")
        analysis_layout = QFormLayout(analysis_group)

        self.nsgaii_weight_perf = QDoubleSpinBox()
        self.nsgaii_weight_sparse = QDoubleSpinBox()
        self.nsgaii_weight_cost = QDoubleSpinBox()
        self.nsgaii_weight_error = QDoubleSpinBox()
        
        self.weight_spinboxes = [self.nsgaii_weight_perf, self.nsgaii_weight_sparse, self.nsgaii_weight_cost, self.nsgaii_weight_error]
        
        weights_container = QWidget()
        weights_layout = QHBoxLayout(weights_container)
        weights_layout.setContentsMargins(0, 0, 0, 0)
        for w, name in zip(self.weight_spinboxes, ["Perf.", "Sparsity", "Cost", "Error"]):
            w.setRange(0.0, 1.0)
            w.setDecimals(2)
            w.setSingleStep(0.05)
            w.setValue(0.25)
            weights_layout.addWidget(QLabel(name))
            weights_layout.addWidget(w)
            w.valueChanged.connect(self.update_nsgaii_results_display)
        
        self.nsgaii_normalize_weights_btn = QPushButton("Normalize")
        self.nsgaii_normalize_weights_btn.clicked.connect(self.normalize_nsgaii_weights)
        weights_layout.addWidget(self.nsgaii_normalize_weights_btn)
        analysis_layout.addRow("Objective Weights:", weights_container)
        
        table_layout.addWidget(analysis_group)
        
        self.nsgaii_results_table = QTableWidget()
        self.nsgaii_results_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.nsgaii_results_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.nsgaii_results_table.setSortingEnabled(True)
        self.nsgaii_results_table.horizontalHeader().sectionClicked.connect(self.update_nsgaii_results_display)
        
        table_layout.addWidget(self.nsgaii_results_table)
        table_tab.setLayout(table_layout)

        self.nsgaii_results_tabs.addTab(self.nsgaii_plot_tabs, "Pareto Front Visuals")
        self.nsgaii_results_tabs.addTab(table_tab, "Solutions")
        
        results_layout.addWidget(self.nsgaii_results_tabs)
        splitter.addWidget(results_widget)

        layout.addWidget(splitter)
        return self.nsgaii_tab

    def run_nsgaii(self):
        """Starts the NSGA-II optimization worker."""
        self.update_log("Starting NSGA-II optimization...")
        self.run_nsgaii_button.setEnabled(False)
        self.nsgaii_progress_bar.setVisible(True)
        self.nsgaii_progress_bar.setValue(0)

        try:
            ga_parameter_data = []
            for row in range(self.nsgaii_param_table.rowCount()):
                name_item = self.nsgaii_param_table.item(row, 0)
                low_spin = self.nsgaii_param_table.cellWidget(row, 1)
                high_spin = self.nsgaii_param_table.cellWidget(row, 2)
                cost_spin = self.nsgaii_param_table.cellWidget(row, 3)
                
                ga_parameter_data.append(
                    (
                        name_item.text(),
                        low_spin.value(),
                        high_spin.value(),
                        cost_spin.value()
                    )
                )

            main_params, target_values, weights, omega_start, omega_end, omega_points = self._get_main_params_targets_weights()

            self.nsgaii_worker = NSGAIIWorker(
                main_params=main_params,
                target_values_dict=target_values,
                weights_dict=weights,
                omega_start=omega_start,
                omega_end=omega_end,
                omega_points=omega_points,
                pop_size=self.nsgaii_pop_size_box.value(),
                num_generations=self.nsgaii_num_generations_box.value(),
                cxpb=self.nsgaii_cxpb_box.value(),
                mutpb=self.nsgaii_mutpb_box.value(),
                parameter_data=ga_parameter_data,
                cost_threshold=self.nsgaii_cost_threshold_box.value()
            )

            self.nsgaii_worker.finished.connect(self.handle_nsgaii_finished)
            self.nsgaii_worker.error.connect(self.handle_nsgaii_error)
            self.nsgaii_worker.update.connect(self.update_log)
            self.nsgaii_worker.progress.connect(self.update_progress)
            self.nsgaii_worker.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start NSGA-II worker: {e}")
            self.run_nsgaii_button.setEnabled(True)
            self.nsgaii_progress_bar.setVisible(False)
        
    def handle_nsgaii_finished(self, pareto_front, param_names):
        """Handles the results when NSGA-II optimization is finished."""
        self.update_log("NSGA-II optimization finished.")
        self.run_nsgaii_button.setEnabled(True)
        self.nsgaii_progress_bar.setVisible(False)
        
        if not pareto_front:
            self.update_log("NSGA-II returned an empty Pareto front.")
            return

        # Convert results to a DataFrame for easier manipulation
        obj_cols = [f'Obj {i+1}' for i in range(len(pareto_front[0]['objectives']))]
        self.last_param_names = param_names
        
        records = []
        for sol in pareto_front:
            record = {}
            for i, obj_val in enumerate(sol['objectives']):
                record[obj_cols[i]] = obj_val
            for i, param_val in enumerate(sol['parameters']):
                record[param_names[i]] = param_val
            records.append(record)
            
        self.nsgaii_results_df = pd.DataFrame(records)
        
        # Initial display update
        self.update_nsgaii_results_display()

        # [NEW] Update the new visualizations
        self._update_2d_scatter_plot()
        self._update_3d_scatter_plot()
        self._update_parallel_coordinates_plot()

    def handle_nsgaii_error(self, error_msg):
        """Handles errors from the NSGA-II worker."""
        QMessageBox.critical(self, "NSGA-II Error", error_msg)
        self.run_nsgaii_button.setEnabled(True)
        self.nsgaii_progress_bar.setVisible(False)

    def update_nsgaii_results_display(self):
        """Calculates scores, sorts, and updates the results table."""
        if not hasattr(self, 'nsgaii_results_df') or self.nsgaii_results_df.empty:
            return

        df = self.nsgaii_results_df.copy() # Work with a copy to avoid altering the original df
        obj_cols = [f'Obj {i+1}' for i in range(4)] # Assuming 4 objectives

        # --- 1. Normalization ---
        # Normalize each objective column from 0 (best) to 1 (worst)
        for col in obj_cols:
            min_val = df[col].min()
            max_val = df[col].max()
            if (max_val - min_val) > 1e-9:
                df[f'{col}_norm'] = (df[col] - min_val) / (max_val - min_val)
            else:
                df[f'{col}_norm'] = 0.5 # All values are the same

        # --- 2. Weighted Score Calculation ---
        weights = [
            self.nsgaii_weight_perf.value(),
            self.nsgaii_weight_sparse.value(),
            self.nsgaii_weight_cost.value(),
            self.nsgaii_weight_error.value()
        ]
        
        df['Balanced Score'] = (
            df['Obj 1_norm'] * weights[0] +
            df['Obj 2_norm'] * weights[1] +
            df['Obj 3_norm'] * weights[2] +
            df['Obj 4_norm'] * weights[3]
        )

        # Store the processed DataFrame for plotting and table display
        self.nsgaii_results_df_processed = df.copy()

        # --- 3. Sorting ---
        # Default sort by Balanced Score, ascending
        sort_col_name = 'Balanced Score'
        sort_ascending = True

        # If user has sorted by clicking a column, use that preference
        sort_col_idx = self.nsgaii_results_table.horizontalHeader().sortIndicatorSection()
        header_item = self.nsgaii_results_table.horizontalHeaderItem(sort_col_idx)

        if sort_col_idx > -1 and header_item is not None:
            sort_col_name = header_item.text()
            sort_order = self.nsgaii_results_table.horizontalHeader().sortIndicatorOrder()
            sort_ascending = (sort_order == Qt.AscendingOrder)

        # Sort the DataFrame
        if sort_col_name in df.columns:
            df.sort_values(by=sort_col_name, ascending=sort_ascending, inplace=True)

        self.update_results_table(df)

    def _create_2d_scatter_tab(self):
        """Creates the 2D scatter plot tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # --- Controls ---
        controls_widget = QWidget()
        controls_layout = QHBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 10)
        
        self.nsgaii_2d_x_combo = QComboBox()
        self.nsgaii_2d_y_combo = QComboBox()
        
        objective_names = ["Obj 1 (Performance)", "Obj 2 (Sparsity)", "Obj 3 (Cost)", "Obj 4 (Error)"]
        self.nsgaii_2d_x_combo.addItems(objective_names)
        self.nsgaii_2d_y_combo.addItems(objective_names)
        self.nsgaii_2d_x_combo.setCurrentIndex(0)
        self.nsgaii_2d_y_combo.setCurrentIndex(1)
        
        controls_layout.addWidget(QLabel("X-Axis:"))
        controls_layout.addWidget(self.nsgaii_2d_x_combo)
        controls_layout.addStretch()
        controls_layout.addWidget(QLabel("Y-Axis:"))
        controls_layout.addWidget(self.nsgaii_2d_y_combo)
        
        layout.addWidget(controls_widget)

        # --- Canvas ---
        self.nsgaii_2d_fig = Figure(figsize=(7, 5))
        self.nsgaii_2d_canvas = FigureCanvas(self.nsgaii_2d_fig)
        toolbar = NavigationToolbar(self.nsgaii_2d_canvas, self)
        
        layout.addWidget(toolbar)
        layout.addWidget(self.nsgaii_2d_canvas)

        # --- Connections ---
        self.nsgaii_2d_x_combo.currentIndexChanged.connect(self._update_2d_scatter_plot)
        self.nsgaii_2d_y_combo.currentIndexChanged.connect(self._update_2d_scatter_plot)

        return tab

    def _update_2d_scatter_plot(self):
        """Updates the 2D scatter plot with the current data and settings."""
        if not hasattr(self, 'nsgaii_results_df_processed') or self.nsgaii_results_df_processed.empty:
            return

        try:
            x_idx = self.nsgaii_2d_x_combo.currentIndex()
            y_idx = self.nsgaii_2d_y_combo.currentIndex()
            x_label = self.nsgaii_2d_x_combo.currentText()
            y_label = self.nsgaii_2d_y_combo.currentText()
            
            self.nsgaii_2d_fig.clear()
            ax = self.nsgaii_2d_fig.add_subplot(111)

            obj_cols = [f'Obj {i+1}' for i in range(4)]
            x_col = obj_cols[x_idx]
            y_col = obj_cols[y_idx]

            if x_col in self.nsgaii_results_df_processed.columns and y_col in self.nsgaii_results_df_processed.columns:
                sns.scatterplot(
                    data=self.nsgaii_results_df_processed,
                    x=x_col,
                    y=y_col,
                    hue='Balanced Score',
                    palette='viridis',
                    ax=ax,
                    s=50,
                    alpha=0.8
                )
            
            ax.legend(title='Balanced Score')
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_title("2D Pareto Front Scatter Plot")
            ax.grid(True)
            
            self.nsgaii_2d_fig.tight_layout()
            self.nsgaii_2d_canvas.draw()
        except Exception as e:
            self.update_log(f"Error updating 2D scatter plot: {e}")
            QMessageBox.critical(self, "Plotting Error", f"Failed to update 2D scatter plot: {e}")

    def _create_3d_scatter_tab(self):
        """Creates the 3D scatter plot tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # --- Controls ---
        controls_widget = QWidget()
        controls_layout = QHBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 10)
        
        self.nsgaii_3d_x_combo = QComboBox()
        self.nsgaii_3d_y_combo = QComboBox()
        self.nsgaii_3d_z_combo = QComboBox()
        
        objective_names = ["Obj 1 (Performance)", "Obj 2 (Sparsity)", "Obj 3 (Cost)", "Obj 4 (Error)"]
        self.nsgaii_3d_x_combo.addItems(objective_names)
        self.nsgaii_3d_y_combo.addItems(objective_names)
        self.nsgaii_3d_z_combo.addItems(objective_names)
        self.nsgaii_3d_x_combo.setCurrentIndex(0)
        self.nsgaii_3d_y_combo.setCurrentIndex(1)
        self.nsgaii_3d_z_combo.setCurrentIndex(2)
        
        controls_layout.addWidget(QLabel("X:"))
        controls_layout.addWidget(self.nsgaii_3d_x_combo)
        controls_layout.addWidget(QLabel("Y:"))
        controls_layout.addWidget(self.nsgaii_3d_y_combo)
        controls_layout.addWidget(QLabel("Z:"))
        controls_layout.addWidget(self.nsgaii_3d_z_combo)
        
        layout.addWidget(controls_widget)

        # --- Canvas ---
        self.nsgaii_3d_fig = Figure(figsize=(7, 5))
        self.nsgaii_3d_canvas = FigureCanvas(self.nsgaii_3d_fig)
        toolbar = NavigationToolbar(self.nsgaii_3d_canvas, self)
        
        layout.addWidget(toolbar)
        layout.addWidget(self.nsgaii_3d_canvas)

        # --- Connections ---
        self.nsgaii_3d_x_combo.currentIndexChanged.connect(self._update_3d_scatter_plot)
        self.nsgaii_3d_y_combo.currentIndexChanged.connect(self._update_3d_scatter_plot)
        self.nsgaii_3d_z_combo.currentIndexChanged.connect(self._update_3d_scatter_plot)

        return tab

    def _update_3d_scatter_plot(self):
        """Updates the 3D scatter plot with the current data and settings."""
        if not hasattr(self, 'nsgaii_results_df_processed') or self.nsgaii_results_df_processed.empty:
            return

        try:
            x_idx = self.nsgaii_3d_x_combo.currentIndex()
            y_idx = self.nsgaii_3d_y_combo.currentIndex()
            z_idx = self.nsgaii_3d_z_combo.currentIndex()
            x_label = self.nsgaii_3d_x_combo.currentText()
            y_label = self.nsgaii_3d_y_combo.currentText()
            z_label = self.nsgaii_3d_z_combo.currentText()
            
            self.nsgaii_3d_fig.clear()
            ax = self.nsgaii_3d_fig.add_subplot(111, projection='3d')

            obj_cols = [f'Obj {i+1}' for i in range(4)]
            x_col = obj_cols[x_idx]
            y_col = obj_cols[y_idx]
            z_col = obj_cols[z_idx]

            if x_col in self.nsgaii_results_df_processed.columns and \
               y_col in self.nsgaii_results_df_processed.columns and \
               z_col in self.nsgaii_results_df_processed.columns:
                
                sc = ax.scatter(
                    self.nsgaii_results_df_processed[x_col],
                    self.nsgaii_results_df_processed[y_col],
                    self.nsgaii_results_df_processed[z_col],
                    c=self.nsgaii_results_df_processed['Balanced Score'],
                    cmap='viridis',
                    s=50,
                    alpha=0.8
                )
                cbar = self.nsgaii_3d_fig.colorbar(sc)
                cbar.set_label('Balanced Score')
            
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_zlabel(z_label)
            ax.set_title("3D Pareto Front Scatter Plot")
            
            self.nsgaii_3d_fig.tight_layout()
            self.nsgaii_3d_canvas.draw()
        except Exception as e:
            self.update_log(f"Error updating 3D scatter plot: {e}")
            QMessageBox.critical(self, "Plotting Error", f"Failed to update 3D scatter plot: {e}")

    def _create_parallel_coordinates_tab(self):
        """Creates the parallel coordinates plot tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # --- Canvas ---
        self.nsgaii_pc_fig = Figure(figsize=(8, 4))
        self.nsgaii_pc_canvas = FigureCanvas(self.nsgaii_pc_fig)
        toolbar = NavigationToolbar(self.nsgaii_pc_canvas, self)
        
        layout.addWidget(toolbar)
        layout.addWidget(self.nsgaii_pc_canvas)

        return tab

    def _update_parallel_coordinates_plot(self):
        """Updates the parallel coordinates plot."""
        if not hasattr(self, 'nsgaii_results_df_processed') or self.nsgaii_results_df_processed.empty:
            return

        try:
            self.nsgaii_pc_fig.clear()
            ax = self.nsgaii_pc_fig.add_subplot(111)

            # Prepare data for parallel coordinates
            df = self.nsgaii_results_df_processed.copy()
            obj_cols = ['Obj 1', 'Obj 2', 'Obj 3', 'Obj 4']
            
            # Normalize the objective columns for better visualization
            for col in obj_cols:
                min_val = df[col].min()
                max_val = df[col].max()
                if (max_val - min_val) > 1e-9:
                    df[col] = (df[col] - min_val) / (max_val - min_val)
                else:
                    df[col] = 0.5
            
            parallel_coordinates(df, 'Balanced Score', cols=obj_cols, ax=ax, colormap='viridis', alpha=0.5)
            
            ax.set_title("Parallel Coordinates Plot of Pareto Front")
            ax.grid(True)
            ax.legend().remove()
            
            self.nsgaii_pc_fig.tight_layout()
            self.nsgaii_pc_canvas.draw()
        except Exception as e:
            self.update_log(f"Error updating parallel coordinates plot: {e}")
            QMessageBox.critical(self, "Plotting Error", f"Failed to update parallel coordinates plot: {e}")

    def update_results_table(self, df):
        """Fills the results table with the solutions from the DataFrame."""
        try:
            self.nsgaii_results_table.setSortingEnabled(False) # Disable sorting during population
            
            obj_cols = [f'Obj {i+1}' for i in range(4)]
            if not hasattr(self, 'last_param_names'): self.last_param_names = []
            display_cols = ['Balanced Score'] + obj_cols + self.last_param_names
            
            self.nsgaii_results_table.setRowCount(len(df))
            self.nsgaii_results_table.setColumnCount(len(display_cols))
            self.nsgaii_results_table.setHorizontalHeaderLabels(display_cols)

            for row_idx, table_row in enumerate(df[display_cols].itertuples(index=False)):
                for col_idx, value in enumerate(table_row):
                    item = QTableWidgetItem(f"{value:.4f}")
                    item.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled) # Make item selectable but not editable
                    self.nsgaii_results_table.setItem(row_idx, col_idx, item)

            self.nsgaii_results_table.resizeColumnsToContents()
        except Exception as e:
            import traceback
            print(f"Error updating results table: {e}\n{traceback.format_exc()}")
        finally:
            self.nsgaii_results_table.setSortingEnabled(True)

    def create_parameter_table(self):
        """Creates a simplified parameter table for NSGA-II."""
        table = QTableWidget()
        dva_parameters = [
            *[f"beta_{i}" for i in range(1,16)],
            *[f"lambda_{i}" for i in range(1,16)],
            *[f"mu_{i}" for i in range(1,4)],
            *[f"nu_{i}" for i in range(1,16)]
        ]
        table.setRowCount(len(dva_parameters))
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["Parameter", "Lower Bound", "Upper Bound", "Cost"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        for row, param in enumerate(dva_parameters):
            param_item = QTableWidgetItem(param)
            param_item.setFlags(Qt.ItemIsEnabled)
            table.setItem(row, 0, param_item)

            lower_bound_spin = QDoubleSpinBox()
            lower_bound_spin.setRange(0, 10e9)
            lower_bound_spin.setDecimals(6)
            lower_bound_spin.setValue(0.0)
            table.setCellWidget(row, 1, lower_bound_spin)

            upper_bound_spin = QDoubleSpinBox()
            upper_bound_spin.setRange(0, 10e9)
            upper_bound_spin.setDecimals(6)
            upper_bound_spin.setValue(1.0)
            table.setCellWidget(row, 2, upper_bound_spin)

            cost_spin = QDoubleSpinBox()
            cost_spin.setRange(0, 1000)
            cost_spin.setDecimals(2)
            cost_spin.setValue(1.0)
            table.setCellWidget(row, 3, cost_spin)
        return table
        
    def update_log(self, message):
        """Appends a message to the results text area."""
        if hasattr(self, 'results_text'):
            self.results_text.append(message)
        else:
            print(message)

    def update_progress(self, value):
        """Updates the progress bar."""
        if hasattr(self, 'nsgaii_progress_bar'):
            self.nsgaii_progress_bar.setValue(value)

    def _get_main_params_targets_weights(self):
        """Fetch or construct main system parameters, target values, and weights from current UI."""
        try:
            target_values, weights = self.get_target_values_weights()
            main_params = (
                self.mu_box.value(),
                *[b.value() for b in self.landa_boxes],
                *[b.value() for b in self.nu_boxes],
                self.a_low_box.value(),
                self.a_up_box.value(),
                self.f_1_box.value(),
                self.f_2_box.value(),
                self.omega_dc_box.value(),
                self.zeta_dc_box.value(),
            )
            omega_start = self.omega_start_box.value()
            omega_end = self.omega_end_box.value()
            omega_points = self.omega_points_box.value()
            return main_params, target_values, weights, float(omega_start), float(omega_end), int(omega_points)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not retrieve all parameters from UI: {e}")
            return None, None, None, 0.0, 1.0, 100

    def normalize_nsgaii_weights(self):
        """Normalizes the objective weights in the UI to sum to 1.0."""
        if not hasattr(self, 'weight_spinboxes'): return
        
        weights = [w.value() for w in self.weight_spinboxes]
        total_weight = sum(weights)
        
        if total_weight < 1e-6:
            # If sum is zero, reset to equal weights
            for w in self.weight_spinboxes:
                w.blockSignals(True)
                w.setValue(1.0 / len(self.weight_spinboxes))
                w.blockSignals(False)
        else:
            # Normalize to sum to 1
            for w in self.weight_spinboxes:
                w.blockSignals(True)
                w.setValue(w.value() / total_weight)
                w.blockSignals(False)
        
        # Manually trigger a single update after all values are set
        self.update_nsgaii_results_display()
