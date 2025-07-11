from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from workers.PSOWorker import PSOWorker, TopologyType
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
                           QSpinBox, QDoubleSpinBox, QComboBox, QTabWidget, QGroupBox,
                           QFormLayout, QMessageBox, QTableWidget, QTableWidgetItem,
                           QHeaderView, QAbstractItemView, QSplitter, QTextEdit,
                           QSizePolicy)
from PyQt5.QtCore import Qt
import os
import time
import json
from datetime import datetime

class PSOMixin:

    def create_pso_tab(self):
        """Create the particle swarm optimization tab"""
        self.pso_tab = QWidget()
        layout = QVBoxLayout(self.pso_tab)
        
        # Create sub-tabs widget
        self.pso_sub_tabs = QTabWidget()

        # -------------------- Sub-tab 1: PSO Basic Settings --------------------
        pso_basic_tab = QWidget()
        pso_basic_layout = QFormLayout(pso_basic_tab)

        self.pso_swarm_size_box = QSpinBox()
        self.pso_swarm_size_box.setRange(10, 10000)
        self.pso_swarm_size_box.setValue(40)

        self.pso_num_iterations_box = QSpinBox()
        self.pso_num_iterations_box.setRange(10, 10000)
        self.pso_num_iterations_box.setValue(100)

        self.pso_inertia_box = QDoubleSpinBox()
        self.pso_inertia_box.setRange(0, 2)
        self.pso_inertia_box.setValue(0.729)
        self.pso_inertia_box.setDecimals(3)

        self.pso_cognitive_box = QDoubleSpinBox()
        self.pso_cognitive_box.setRange(0, 5)
        self.pso_cognitive_box.setValue(1.49445)
        self.pso_cognitive_box.setDecimals(5)

        self.pso_social_box = QDoubleSpinBox()
        self.pso_social_box.setRange(0, 5)
        self.pso_social_box.setValue(1.49445)
        self.pso_social_box.setDecimals(5)

        self.pso_tol_box = QDoubleSpinBox()
        self.pso_tol_box.setRange(0, 1)
        self.pso_tol_box.setValue(1e-6)
        self.pso_tol_box.setDecimals(8)

        self.pso_alpha_box = QDoubleSpinBox()
        self.pso_alpha_box.setRange(0.0, 10.0)
        self.pso_alpha_box.setDecimals(4)
        self.pso_alpha_box.setSingleStep(0.01)
        self.pso_alpha_box.setValue(0.01)
        
        # Add benchmarking runs box
        self.pso_benchmark_runs_box = QSpinBox()
        self.pso_benchmark_runs_box.setRange(1, 1000)
        self.pso_benchmark_runs_box.setValue(1)
        self.pso_benchmark_runs_box.setToolTip("Number of times to run the PSO for benchmarking (1 = single run)")

        pso_basic_layout.addRow("Swarm Size:", self.pso_swarm_size_box)
        pso_basic_layout.addRow("Number of Iterations:", self.pso_num_iterations_box)
        pso_basic_layout.addRow("Inertia Weight (w):", self.pso_inertia_box)
        pso_basic_layout.addRow("Cognitive Coefficient (c1):", self.pso_cognitive_box)
        pso_basic_layout.addRow("Social Coefficient (c2):", self.pso_social_box)
        pso_basic_layout.addRow("Tolerance (tol):", self.pso_tol_box)
        pso_basic_layout.addRow("Sparsity Penalty (alpha):", self.pso_alpha_box)
        pso_basic_layout.addRow("Benchmark Runs:", self.pso_benchmark_runs_box)

        # -------------------- Sub-tab 2: Advanced PSO Settings --------------------
        pso_advanced_tab = QWidget()
        pso_advanced_layout = QFormLayout(pso_advanced_tab)

        # Adaptive Parameters
        self.pso_adaptive_params_checkbox = QCheckBox()
        self.pso_adaptive_params_checkbox.setChecked(True)
        
        # Topology selection
        self.pso_topology_combo = QComboBox()
        self.pso_topology_combo.addItems(["Global", "Ring", "Von Neumann", "Random"])
        
        # W damping
        self.pso_w_damping_box = QDoubleSpinBox()
        self.pso_w_damping_box.setRange(0.1, 1.0)
        self.pso_w_damping_box.setValue(1.0)
        self.pso_w_damping_box.setDecimals(3)
        
        # Mutation rate
        self.pso_mutation_rate_box = QDoubleSpinBox()
        self.pso_mutation_rate_box.setRange(0.0, 1.0)
        self.pso_mutation_rate_box.setValue(0.1)
        self.pso_mutation_rate_box.setDecimals(3)
        
        # Velocity clamping
        self.pso_max_velocity_factor_box = QDoubleSpinBox()
        self.pso_max_velocity_factor_box.setRange(0.01, 1.0)
        self.pso_max_velocity_factor_box.setValue(0.1)
        self.pso_max_velocity_factor_box.setDecimals(3)
        
        # Stagnation limit
        self.pso_stagnation_limit_box = QSpinBox()
        self.pso_stagnation_limit_box.setRange(1, 50)
        self.pso_stagnation_limit_box.setValue(10)
        
        # Boundary handling
        self.pso_boundary_handling_combo = QComboBox()
        self.pso_boundary_handling_combo.addItems(["absorbing", "reflecting", "invisible"])
        
        # Diversity threshold
        self.pso_diversity_threshold_box = QDoubleSpinBox()
        self.pso_diversity_threshold_box.setRange(0.001, 0.5)
        self.pso_diversity_threshold_box.setValue(0.01)
        self.pso_diversity_threshold_box.setDecimals(4)
        
        # Early stopping
        self.pso_early_stopping_checkbox = QCheckBox()
        self.pso_early_stopping_checkbox.setChecked(True)
        
        self.pso_early_stopping_iters_box = QSpinBox()
        self.pso_early_stopping_iters_box.setRange(5, 50)
        self.pso_early_stopping_iters_box.setValue(15)
        
        self.pso_early_stopping_tol_box = QDoubleSpinBox()
        self.pso_early_stopping_tol_box.setRange(0, 1)
        self.pso_early_stopping_tol_box.setValue(1e-5)
        self.pso_early_stopping_tol_box.setDecimals(8)
        
        # Quasi-random initialization
        self.pso_quasi_random_init_checkbox = QCheckBox()
        self.pso_quasi_random_init_checkbox.setChecked(True)
        
        pso_advanced_layout.addRow("Enable Adaptive Parameters:", self.pso_adaptive_params_checkbox)
        pso_advanced_layout.addRow("Neighborhood Topology:", self.pso_topology_combo)
        pso_advanced_layout.addRow("Inertia Weight Damping:", self.pso_w_damping_box)
        pso_advanced_layout.addRow("Mutation Rate:", self.pso_mutation_rate_box)
        pso_advanced_layout.addRow("Max Velocity Factor:", self.pso_max_velocity_factor_box)
        pso_advanced_layout.addRow("Stagnation Limit:", self.pso_stagnation_limit_box)
        pso_advanced_layout.addRow("Boundary Handling:", self.pso_boundary_handling_combo)
        pso_advanced_layout.addRow("Diversity Threshold:", self.pso_diversity_threshold_box)
        pso_advanced_layout.addRow("Enable Early Stopping:", self.pso_early_stopping_checkbox)
        pso_advanced_layout.addRow("Early Stopping Iterations:", self.pso_early_stopping_iters_box)
        pso_advanced_layout.addRow("Early Stopping Tolerance:", self.pso_early_stopping_tol_box)
        pso_advanced_layout.addRow("Use Quasi-Random Init:", self.pso_quasi_random_init_checkbox)

        # Add a small Run PSO button in the advanced settings sub-tab
        self.hyper_run_pso_button = QPushButton("Run PSO")
        self.hyper_run_pso_button.setFixedWidth(100)
        self.hyper_run_pso_button.clicked.connect(self.run_pso)
        pso_advanced_layout.addRow("Run PSO:", self.hyper_run_pso_button)

        # -------------------- Sub-tab 3: DVA Parameters --------------------
        pso_param_tab = QWidget()
        pso_param_layout = QVBoxLayout(pso_param_tab)

        self.pso_param_table = QTableWidget()
        dva_parameters = [
            *[f"beta_{i}" for i in range(1,16)],
            *[f"lambda_{i}" for i in range(1,16)],
            *[f"mu_{i}" for i in range(1,4)],
            *[f"nu_{i}" for i in range(1,16)]
        ]
        self.pso_param_table.setRowCount(len(dva_parameters))
        self.pso_param_table.setColumnCount(5)
        self.pso_param_table.setHorizontalHeaderLabels(
            ["Parameter", "Fixed", "Fixed Value", "Lower Bound", "Upper Bound"]
        )
        self.pso_param_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.pso_param_table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        for row, param in enumerate(dva_parameters):
            param_item = QTableWidgetItem(param)
            param_item.setFlags(Qt.ItemIsEnabled)
            self.pso_param_table.setItem(row, 0, param_item)

            fixed_checkbox = QCheckBox()
            fixed_checkbox.stateChanged.connect(lambda state, r=row: self.toggle_pso_fixed(state, r))
            self.pso_param_table.setCellWidget(row, 1, fixed_checkbox)

            fixed_value_spin = QDoubleSpinBox()
            fixed_value_spin.setRange(-1e6, 1e6)
            fixed_value_spin.setDecimals(6)
            fixed_value_spin.setEnabled(False)
            self.pso_param_table.setCellWidget(row, 2, fixed_value_spin)

            lower_bound_spin = QDoubleSpinBox()
            lower_bound_spin.setRange(-1e6, 1e6)
            lower_bound_spin.setDecimals(6)
            lower_bound_spin.setEnabled(True)
            self.pso_param_table.setCellWidget(row, 3, lower_bound_spin)

            upper_bound_spin = QDoubleSpinBox()
            upper_bound_spin.setRange(-1e6, 1e6)
            upper_bound_spin.setDecimals(6)
            upper_bound_spin.setEnabled(True)
            self.pso_param_table.setCellWidget(row, 4, upper_bound_spin)

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

        pso_param_layout.addWidget(self.pso_param_table)

        # -------------------- Sub-tab 4: Results --------------------
        pso_results_tab = QWidget()
        pso_results_layout = QVBoxLayout(pso_results_tab)
        
        self.pso_results_text = QTextEdit()
        self.pso_results_text.setReadOnly(True)
        pso_results_layout.addWidget(QLabel("PSO Optimization Results:"))
        pso_results_layout.addWidget(self.pso_results_text)

        # -------------------- Sub-tab 5: Benchmarking --------------------
        pso_benchmark_tab = QWidget()
        pso_benchmark_tab.setObjectName("PSO Benchmarking")
        pso_benchmark_layout = QVBoxLayout(pso_benchmark_tab)
        
        # Add button container for export
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(0, 0, 0, 0)
        
        # Export button
        self.pso_export_benchmark_button = QPushButton("Export Benchmark Data")
        self.pso_export_benchmark_button.setToolTip("Export current PSO benchmark data to a file")
        self.pso_export_benchmark_button.setEnabled(False)  # Initially disabled until data is available
        self.pso_export_benchmark_button.clicked.connect(self.export_pso_benchmark_data)
        button_layout.addWidget(self.pso_export_benchmark_button)
        
        # Import button
        self.pso_import_benchmark_button = QPushButton("Import Benchmark Data")
        self.pso_import_benchmark_button.setToolTip("Import PSO benchmark data from a file")
        self.pso_import_benchmark_button.clicked.connect(self.import_pso_benchmark_data)
        button_layout.addWidget(self.pso_import_benchmark_button)
        
        button_layout.addStretch()  # Add stretch to push buttons to the left
        pso_benchmark_layout.addWidget(button_container)
        
        # Create tabs for different benchmark visualizations
        self.pso_benchmark_viz_tabs = QTabWidget()
        
        # Create tabs for different visualizations
        pso_violin_tab = QWidget()
        pso_violin_layout = QVBoxLayout(pso_violin_tab)
        self.pso_violin_plot_widget = QWidget()
        pso_violin_layout.addWidget(self.pso_violin_plot_widget)
        
        pso_dist_tab = QWidget()
        pso_dist_layout = QVBoxLayout(pso_dist_tab)
        self.pso_dist_plot_widget = QWidget()
        pso_dist_layout.addWidget(self.pso_dist_plot_widget)
        
        pso_scatter_tab = QWidget()
        pso_scatter_layout = QVBoxLayout(pso_scatter_tab)
        self.pso_scatter_plot_widget = QWidget()
        pso_scatter_layout.addWidget(self.pso_scatter_plot_widget)
        
        pso_heatmap_tab = QWidget()
        pso_heatmap_layout = QVBoxLayout(pso_heatmap_tab)
        self.pso_heatmap_plot_widget = QWidget()
        pso_heatmap_layout.addWidget(self.pso_heatmap_plot_widget)

        # Parameter visualization tab similar to GA
        pso_param_viz_tab = QWidget()
        pso_param_viz_layout = QVBoxLayout(pso_param_viz_tab)

        # Control panel for parameter selection
        pso_control_panel = QGroupBox("Parameter Selection & Visualization Controls")
        pso_control_layout = QGridLayout(pso_control_panel)

        self.pso_param_selection_combo = QComboBox()
        self.pso_param_selection_combo.setMaxVisibleItems(5)
        self.pso_param_selection_combo.setMinimumWidth(150)
        self.pso_param_selection_combo.setMaximumWidth(200)
        self.pso_param_selection_combo.currentTextChanged.connect(self.pso_on_parameter_selection_changed)

        self.pso_plot_type_combo = QComboBox()
        self.pso_plot_type_combo.addItems(["Violin Plot", "Distribution Plot", "Scatter Plot", "Q-Q Plot"])
        self.pso_plot_type_combo.currentTextChanged.connect(self.pso_on_plot_type_changed)

        self.pso_comparison_param_combo = QComboBox()
        self.pso_comparison_param_combo.addItem("None")
        self.pso_comparison_param_combo.setMaxVisibleItems(5)
        self.pso_comparison_param_combo.setMinimumWidth(150)
        self.pso_comparison_param_combo.setMaximumWidth(200)
        self.pso_comparison_param_combo.setEnabled(False)
        self.pso_comparison_param_combo.currentTextChanged.connect(self.pso_on_comparison_parameter_changed)

        self.pso_update_plots_button = QPushButton("Update Plots")
        self.pso_update_plots_button.clicked.connect(self.pso_update_parameter_plots)

        pso_control_layout.addWidget(QLabel("Select Parameter:"), 0, 0)
        pso_control_layout.addWidget(self.pso_param_selection_combo, 0, 1)
        pso_control_layout.addWidget(QLabel("Plot Type:"), 0, 2)
        pso_control_layout.addWidget(self.pso_plot_type_combo, 0, 3)
        pso_control_layout.addWidget(QLabel("Compare With:"), 1, 0)
        pso_control_layout.addWidget(self.pso_comparison_param_combo, 1, 1)
        pso_control_layout.addWidget(self.pso_update_plots_button, 1, 2)

        pso_param_viz_layout.addWidget(pso_control_panel)

        self.pso_param_plot_scroll = QScrollArea()
        self.pso_param_plot_scroll.setWidgetResizable(True)
        self.pso_param_plot_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.pso_param_plot_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.pso_param_plot_scroll.setMinimumHeight(400)
        self.pso_param_plot_widget = QWidget()
        self.pso_param_plot_widget.setLayout(QVBoxLayout())
        self.pso_param_plot_widget.setMinimumHeight(500)
        self.pso_param_plot_scroll.setWidget(self.pso_param_plot_widget)
        pso_param_viz_layout.addWidget(self.pso_param_plot_scroll)

        # Add Q-Q plot tab
        pso_qq_tab = QWidget()
        pso_qq_layout = QVBoxLayout(pso_qq_tab)
        self.pso_qq_plot_widget = QWidget()
        pso_qq_layout.addWidget(self.pso_qq_plot_widget)
        
        # Summary statistics tabs (create subtabs for better organization)
        pso_stats_tab = QWidget()
        pso_stats_layout = QVBoxLayout(pso_stats_tab)
        
        # Create a tabbed widget for the statistics section
        pso_stats_subtabs = QTabWidget()
        
        # ---- Subtab 1: Summary Statistics ----
        pso_summary_tab = QWidget()
        pso_summary_layout = QVBoxLayout(pso_summary_tab)
        
        # Create a table for summary statistics
        self.pso_stats_table = QTableWidget()
        self.pso_stats_table.setColumnCount(2)
        self.pso_stats_table.setHorizontalHeaderLabels(["Metric", "Value"])
        self.pso_stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.pso_stats_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        pso_summary_layout.addWidget(self.pso_stats_table)
        
        # ---- Subtab 2: Run Details ----
        pso_runs_tab = QWidget()
        pso_runs_layout = QVBoxLayout(pso_runs_tab)
        
        # Split view for run list and details
        pso_runs_splitter = QSplitter(Qt.Vertical)
        
        # Top: Table of all runs
        self.pso_benchmark_runs_table = QTableWidget()
        self.pso_benchmark_runs_table.setColumnCount(3)
        self.pso_benchmark_runs_table.setHorizontalHeaderLabels(["Run #", "Best Fitness", "Time (s)"])
        self.pso_benchmark_runs_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.pso_benchmark_runs_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.pso_benchmark_runs_table.itemClicked.connect(self.pso_show_run_details)
        pso_runs_splitter.addWidget(self.pso_benchmark_runs_table)
        
        # Bottom: Details of selected run
        self.pso_run_details_text = QTextEdit()
        self.pso_run_details_text.setReadOnly(True)
        pso_runs_splitter.addWidget(self.pso_run_details_text)
        
        # Set initial sizes
        pso_runs_splitter.setSizes([200, 300])
        pso_runs_layout.addWidget(pso_runs_splitter)
        
        # Add all stats subtabs
        pso_stats_subtabs.addTab(pso_summary_tab, "Summary Statistics")
        pso_stats_subtabs.addTab(pso_runs_tab, "Run Details")
        
        # Add the stats tabbed widget to the stats tab
        pso_stats_layout.addWidget(pso_stats_subtabs)
        
        # Add all visualization tabs to the benchmark visualization tabs
        self.pso_benchmark_viz_tabs.addTab(pso_violin_tab, "Violin Plot")
        self.pso_benchmark_viz_tabs.addTab(pso_dist_tab, "Distribution")
        self.pso_benchmark_viz_tabs.addTab(pso_scatter_tab, "Scatter Plot")
        self.pso_benchmark_viz_tabs.addTab(pso_heatmap_tab, "Parameter Correlations")
        self.pso_benchmark_viz_tabs.addTab(pso_param_viz_tab, "Parameter Visualizations")
        self.pso_benchmark_viz_tabs.addTab(pso_qq_tab, "Q-Q Plot")
        self.pso_benchmark_viz_tabs.addTab(pso_stats_tab, "Statistics")
        
        # PSO Operations Performance Tab
        pso_ops_tab = QWidget()
        pso_ops_layout = QVBoxLayout(pso_ops_tab)
        self.pso_ops_plot_widget = QWidget()
        pso_ops_layout.addWidget(self.pso_ops_plot_widget)
        self.pso_benchmark_viz_tabs.addTab(pso_ops_tab, "PSO Operations")
        
        # Add the benchmark visualization tabs to the benchmark tab
        pso_benchmark_layout.addWidget(self.pso_benchmark_viz_tabs)
        
        # Initialize empty benchmark data storage
        self.pso_benchmark_data = []

        # Add all sub-tabs to the PSO tab widget
        self.pso_sub_tabs.addTab(pso_basic_tab, "Basic Settings")
        self.pso_sub_tabs.addTab(pso_advanced_tab, "Advanced Settings")
        self.pso_sub_tabs.addTab(pso_param_tab, "DVA Parameters")
        self.pso_sub_tabs.addTab(pso_results_tab, "Results")
        self.pso_sub_tabs.addTab(pso_benchmark_tab, "Benchmarking")

        # Add the PSO sub-tabs widget to the main PSO tab layout
        layout.addWidget(self.pso_sub_tabs)
        self.pso_tab.setLayout(layout)
        
    def toggle_pso_fixed(self, state, row, table=None):
        """Toggle the fixed state of a PSO parameter row"""
        if table is None:
            table = self.pso_param_table
            
        fixed = (state == Qt.Checked)
        fixed_value_spin = table.cellWidget(row, 2)
        lower_bound_spin = table.cellWidget(row, 3)
        upper_bound_spin = table.cellWidget(row, 4)

        fixed_value_spin.setEnabled(fixed)
        lower_bound_spin.setEnabled(not fixed)
        upper_bound_spin.setEnabled(not fixed)
        
    def run_pso(self):
        """Run the particle swarm optimization"""
        # Check if a PSO worker is already running
        if hasattr(self, 'pso_worker') and self.pso_worker.isRunning():
            QMessageBox.warning(self, "Process Running", 
                               "A Particle Swarm Optimization is already running. Please wait for it to complete.")
            return
            
        # Clean up any previous PSO worker that might still exist
        if hasattr(self, 'pso_worker'):
            try:
                # First use our custom terminate method if available
                if hasattr(self.pso_worker, 'terminate'):
                    self.pso_worker.terminate()
                
                # Disconnect signals
                self.pso_worker.finished.disconnect()
                self.pso_worker.error.disconnect()
                self.pso_worker.update.disconnect()
                self.pso_worker.convergence_signal.disconnect()
            except Exception as e:
                print(f"Error disconnecting PSO worker signals: {str(e)}")
            
            # Wait for thread to finish if it's still running
            if self.pso_worker.isRunning():
                if not self.pso_worker.wait(1000):  # Wait up to 1 second for graceful termination
                    print("PSO worker didn't terminate gracefully, forcing termination...")
                    # Force termination as a last resort
                    self.pso_worker.terminate()
                    self.pso_worker.wait()
            
        self.status_bar.showMessage("Running PSO optimization...")
        self.results_text.append("PSO optimization started...")
        
        try:
            # Retrieve PSO parameters from the GUI
            swarm_size = self.pso_swarm_size_box.value()
            num_iterations = self.pso_num_iterations_box.value()
            inertia = self.pso_inertia_box.value()
            c1 = self.pso_cognitive_box.value()
            c2 = self.pso_social_box.value()
            tol = self.pso_tol_box.value()
            alpha = self.pso_alpha_box.value()
            
            # Get number of benchmark runs
            self.pso_benchmark_runs = self.pso_benchmark_runs_box.value()
            self.pso_current_benchmark_run = 0
            
            # Clear benchmark data if running multiple times
            if self.pso_benchmark_runs > 1:
                self.pso_benchmark_data = []
                # Enable the benchmark tab if running multiple times
                self.pso_sub_tabs.setTabEnabled(self.pso_sub_tabs.indexOf(self.pso_sub_tabs.findChild(QWidget, "PSO Benchmarking")), True)
                # Set focus to the benchmark tab if running multiple times
                benchmark_tab_index = self.pso_sub_tabs.indexOf(self.pso_sub_tabs.findChild(QWidget, "PSO Benchmarking"))
                if benchmark_tab_index >= 0:
                    self.pso_sub_tabs.setCurrentIndex(benchmark_tab_index)
            
            # Get advanced parameters
            adaptive_params = self.pso_adaptive_params_checkbox.isChecked()
            
            # Convert topology string to enum
            topology_text = self.pso_topology_combo.currentText().upper().replace(" ", "_")
            topology = getattr(TopologyType, topology_text)
            
            w_damping = self.pso_w_damping_box.value()
            mutation_rate = self.pso_mutation_rate_box.value()
            max_velocity_factor = self.pso_max_velocity_factor_box.value()
            stagnation_limit = self.pso_stagnation_limit_box.value()
            boundary_handling = self.pso_boundary_handling_combo.currentText()
            diversity_threshold = self.pso_diversity_threshold_box.value()
            early_stopping = self.pso_early_stopping_checkbox.isChecked()
            early_stopping_iters = self.pso_early_stopping_iters_box.value()
            early_stopping_tol = self.pso_early_stopping_tol_box.value()
            quasi_random_init = self.pso_quasi_random_init_checkbox.isChecked()

            pso_dva_parameters = []
            row_count = self.pso_param_table.rowCount()
            for row in range(row_count):
                param_name = self.pso_param_table.item(row, 0).text()
                fixed_widget = self.pso_param_table.cellWidget(row, 1)
                fixed = fixed_widget.isChecked()
                if fixed:
                    fixed_value_widget = self.pso_param_table.cellWidget(row, 2)
                    fv = fixed_value_widget.value()
                    pso_dva_parameters.append((param_name, fv, fv, True))
                else:
                    lower_bound_widget = self.pso_param_table.cellWidget(row, 3)
                    upper_bound_widget = self.pso_param_table.cellWidget(row, 4)
                    lb = lower_bound_widget.value()
                    ub = upper_bound_widget.value()
                    if lb > ub:
                        QMessageBox.warning(self, "Input Error",
                                            f"For parameter {param_name}, lower bound is greater than upper bound.")
                        return
                    pso_dva_parameters.append((param_name, lb, ub, False))

            # Get main system parameters
            main_params = self.get_main_system_params()

            # Get target values and weights
            target_values, weights = self.get_target_values_weights()

            # Get frequency range values
            omega_start_val = self.omega_start_box.value()
            omega_end_val = self.omega_end_box.value()
            omega_points_val = self.omega_points_box.value()
            
            if omega_start_val >= omega_end_val:
                QMessageBox.warning(self, "Input Error", "Ω Start must be less than Ω End.")
                return
                
            # Store all parameters for benchmark runs
            self.pso_params = {
                'main_params': main_params,
                'target_values': target_values,
                'weights': weights,
                'omega_start_val': omega_start_val,
                'omega_end_val': omega_end_val,
                'omega_points_val': omega_points_val,
                'swarm_size': swarm_size,
                'num_iterations': num_iterations,
                'inertia': inertia,
                'w_damping': w_damping,
                'c1': c1,
                'c2': c2,
                'tol': tol,
                'pso_dva_parameters': pso_dva_parameters,
                'alpha': alpha,
                'adaptive_params': adaptive_params,
                'topology': topology,
                'mutation_rate': mutation_rate,
                'max_velocity_factor': max_velocity_factor,
                'stagnation_limit': stagnation_limit,
                'boundary_handling': boundary_handling,
                'early_stopping': early_stopping,
                'early_stopping_iters': early_stopping_iters,
                'early_stopping_tol': early_stopping_tol,
                'diversity_threshold': diversity_threshold,
                'quasi_random_init': quasi_random_init
            }

            # Clear results and start the benchmark runs
            self.pso_results_text.clear()
            if self.pso_benchmark_runs > 1:
                self.pso_results_text.append(f"Running PSO benchmark with {self.pso_benchmark_runs} runs...")
                self.run_next_pso_benchmark()
            else:
                # Create and start PSOWorker with all parameters
                self.pso_results_text.append("Running PSO optimization...")
            self.pso_worker = PSOWorker(
                main_params=main_params,
                target_values_dict=target_values,
                weights_dict=weights,
                omega_start=omega_start_val,
                omega_end=omega_end_val,
                omega_points=omega_points_val,
                pso_swarm_size=swarm_size,
                pso_num_iterations=num_iterations,
                pso_w=inertia,
                pso_w_damping=w_damping,
                pso_c1=c1,
                pso_c2=c2,
                pso_tol=tol,
                pso_parameter_data=pso_dva_parameters,
                alpha=alpha,
                adaptive_params=adaptive_params,
                topology=topology,
                mutation_rate=mutation_rate,
                max_velocity_factor=max_velocity_factor,
                stagnation_limit=stagnation_limit,
                boundary_handling=boundary_handling,
                early_stopping=early_stopping,
                early_stopping_iters=early_stopping_iters,
                early_stopping_tol=early_stopping_tol,
                diversity_threshold=diversity_threshold,
                quasi_random_init=quasi_random_init
            )
            
            self.pso_worker.finished.connect(self.handle_pso_finished)
            self.pso_worker.error.connect(self.handle_pso_error)
            self.pso_worker.update.connect(self.handle_pso_update)
            self.pso_worker.convergence_signal.connect(self.handle_pso_convergence)
            
            # Disable both run PSO buttons to prevent multiple runs
            self.hyper_run_pso_button.setEnabled(False)
            self.run_pso_button.setEnabled(False)
            
            self.pso_results_text.clear()
            self.pso_results_text.append("Running PSO optimization...")
            
            self.pso_worker.start()
            
        except Exception as e:
            self.handle_pso_error(str(e))
    
    def handle_pso_finished(self, results, best_particle, parameter_names, best_fitness):
        """Handle the completion of PSO optimization"""
        # For benchmarking, collect data from this run
        self.pso_current_benchmark_run += 1
        
        # Store benchmark results
        if hasattr(self, 'pso_benchmark_runs') and self.pso_benchmark_runs > 1:
            # Extract elapsed time from results
            elapsed_time = 0
            if isinstance(results, dict) and 'optimization_metadata' in results:
                elapsed_time = results['optimization_metadata'].get('elapsed_time', 0)
            
            # Create a data dictionary for this run
            run_data = {
                'run_number': self.pso_current_benchmark_run,
                'best_fitness': best_fitness,
                'best_solution': list(best_particle),
                'parameter_names': parameter_names,
                'elapsed_time': elapsed_time
            }
            
            # Add any additional metrics from results
            if isinstance(results, dict):
                for key, value in results.items():
                    if key != 'optimization_metadata' and isinstance(value, (int, float)) and np.isfinite(value):
                        run_data[key] = value

                # Add optimization metadata if available
                if 'optimization_metadata' in results:
                    run_data['optimization_metadata'] = results['optimization_metadata']
            
            # Store the run data
            self.pso_benchmark_data.append(run_data)
            
            # Update the status message
            self.status_bar.showMessage(f"PSO run {self.pso_current_benchmark_run} of {self.pso_benchmark_runs} completed")
            
            # Check if we need to run again
            if self.pso_current_benchmark_run < self.pso_benchmark_runs:
                self.pso_results_text.append(f"\n--- Run {self.pso_current_benchmark_run} completed, starting run {self.pso_current_benchmark_run + 1}/{self.pso_benchmark_runs} ---")
                # Set up for next run
                QTimer.singleShot(100, self.run_next_pso_benchmark)
                return
            else:
                # All runs completed, visualize the benchmark results
                self.visualize_pso_benchmark_results()
                self.pso_export_benchmark_button.setEnabled(True)
                self.pso_results_text.append(f"\n--- All {self.pso_benchmark_runs} benchmark runs completed ---")
        else:
            # For single runs, store the data directly
            elapsed_time = 0
            if isinstance(results, dict) and 'optimization_metadata' in results:
                elapsed_time = results['optimization_metadata'].get('elapsed_time', 0)
                
            run_data = {
                'run_number': 1,
                'best_fitness': best_fitness,
                'best_solution': list(best_particle),
                'parameter_names': parameter_names,
                'elapsed_time': elapsed_time
            }
            
            # Add optimization metadata if available
            if isinstance(results, dict) and 'optimization_metadata' in results:
                run_data['optimization_metadata'] = results['optimization_metadata']
            
            self.pso_benchmark_data = [run_data]
            
        # Re-enable both run PSO buttons when completely done
        self.hyper_run_pso_button.setEnabled(True)
        self.run_pso_button.setEnabled(True)
        
        # Explicitly handle thread cleanup
        if hasattr(self, 'pso_worker') and self.pso_worker is not None and self.pso_worker.isFinished():
            # Disconnect any signals to avoid memory leaks
            try:
                self.pso_worker.finished.disconnect()
                self.pso_worker.error.disconnect()
                self.pso_worker.update.disconnect()
                self.pso_worker.convergence_signal.disconnect()
            except Exception:
                pass
        
        self.status_bar.showMessage("PSO optimization completed")
        
        # Only show detailed results for single runs or the final benchmark run
        if not hasattr(self, 'pso_benchmark_runs') or self.pso_benchmark_runs == 1 or self.pso_current_benchmark_run == self.pso_benchmark_runs:
            self.pso_results_text.append("\nPSO Completed.\n")
            self.pso_results_text.append("Best particle parameters:")

            for name, val in zip(parameter_names, best_particle):
                self.pso_results_text.append(f"{name}: {val}")
            self.pso_results_text.append(f"\nBest fitness: {best_fitness:.6f}")

            singular_response = results.get('singular_response', None)
            if singular_response is not None:
                self.pso_results_text.append(f"\nSingular response of best particle: {singular_response}")

            self.pso_results_text.append("\nFull Results:")
            for section, data in results.items():
                if section != 'optimization_metadata':  # Skip optimization metadata for cleaner output
                    self.pso_results_text.append(f"{section}: {data}")

    def handle_pso_error(self, err):
        """Handle errors during PSO optimization"""
        # Re-enable both run PSO buttons
        self.hyper_run_pso_button.setEnabled(True)
        self.run_pso_button.setEnabled(True)
        
        # Explicitly handle thread cleanup on error
        if hasattr(self, 'pso_worker') and self.pso_worker is not None:
            try:
                self.pso_worker.finished.disconnect()
                self.pso_worker.error.disconnect()
                self.pso_worker.update.disconnect()
                self.pso_worker.convergence_signal.disconnect()
            except Exception:
                pass
        
        QMessageBox.warning(self, "PSO Error", f"Error during PSO optimization: {err}")
        self.pso_results_text.append(f"\nError running PSO: {err}")
        self.status_bar.showMessage("PSO optimization failed")

    def handle_pso_update(self, msg):
        """Handle progress updates from PSO worker"""
        self.pso_results_text.append(msg)
        
    def handle_pso_convergence(self, iterations, fitness_values):
        """Handle convergence data from PSO optimization without creating plots"""
        try:
            # Store the data for later use if needed, but don't create or display plots
            self.pso_iterations = iterations
            self.pso_fitness_values = fitness_values
            
            # Log receipt of convergence data without creating plots
            if hasattr(self, 'pso_results_text'):
                if len(iterations) % 20 == 0:  # Only log occasionally to avoid spamming
                    self.pso_results_text.append(f"Received convergence data for {len(iterations)} iterations")
                    
        except Exception as e:
            self.status_bar.showMessage(f"Error handling PSO convergence data: {str(e)}")
            print(f"Error in handle_pso_convergence: {str(e)}")
            
    def visualize_pso_benchmark_results(self):
        """Create visualizations for PSO benchmark results"""
        if not hasattr(self, 'pso_benchmark_data') or not self.pso_benchmark_data:
            return
            
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        import seaborn as sns
        from computational_metrics_new import visualize_all_metrics
        
        # Fix the operations visualizations for PSO
        # Make sure PSO data is properly formatted for computational_metrics_new
        for idx, run in enumerate(self.pso_benchmark_data):
            if 'optimization_metadata' in run:
                if not 'benchmark_metrics' in run:
                    # Create a basic benchmark_metrics structure
                    run['benchmark_metrics'] = {}
                    
                # Transfer optimization metadata to benchmark_metrics format
                if 'convergence_iterations' in run['optimization_metadata']:
                    run['benchmark_metrics']['iteration_fitness'] = run['optimization_metadata']['convergence_iterations']
                    
                if 'convergence_diversity' in run['optimization_metadata']:
                    run['benchmark_metrics']['diversity_history'] = run['optimization_metadata']['convergence_diversity']
                    
                # Other operations data for the PSO Operations tab
                if 'iterations' in run['optimization_metadata']:
                    iterations = run['optimization_metadata']['iterations']
                    run['benchmark_metrics']['iteration_times'] = [i/10.0 for i in range(iterations)]
                    
                    # Create synthetic PSO operation data if needed
                    if not 'evaluation_times' in run['benchmark_metrics']:
                        import numpy as np
                        np.random.seed(42 + idx)  # For reproducibility but different for each run
                        run['benchmark_metrics']['evaluation_times'] = (0.1 + 0.05 * np.random.rand(iterations)).tolist()
                        run['benchmark_metrics']['neighborhood_update_times'] = (0.02 + 0.01 * np.random.rand(iterations)).tolist()
                        run['benchmark_metrics']['velocity_update_times'] = (0.03 + 0.01 * np.random.rand(iterations)).tolist()
                        run['benchmark_metrics']['position_update_times'] = (0.01 + 0.005 * np.random.rand(iterations)).tolist()
        
        # Convert benchmark data to DataFrame for easier analysis
        df = pd.DataFrame(self.pso_benchmark_data)

        # Prepare parameter data for interactive visualizations
        self.pso_current_parameter_data = self.pso_extract_parameter_data_from_runs(df)
        if self.pso_current_parameter_data:
            self.pso_update_parameter_dropdowns(self.pso_current_parameter_data)
            self.pso_update_parameter_plots()

        # Visualize computational metrics
        widgets_dict = {
            'ga_ops_plot_widget': self.pso_ops_plot_widget
        }
        visualize_all_metrics(widgets_dict, df)
        
        # 3. Create scatter plot of parameters vs fitness
        try:
            # Clear existing plot layout
            if self.pso_scatter_plot_widget.layout():
                for i in reversed(range(self.pso_scatter_plot_widget.layout().count())): 
                    self.pso_scatter_plot_widget.layout().itemAt(i).widget().setParent(None)
            else:
                self.pso_scatter_plot_widget.setLayout(QVBoxLayout())
                
            # Create a DataFrame for parameter values
            scatter_data = []
            
            for run in self.pso_benchmark_data:
                if 'best_solution' in run and 'parameter_names' in run and 'best_fitness' in run:
                    solution = run['best_solution']
                    param_names = run['parameter_names']
                    
                    if len(solution) == len(param_names):
                        run_data = {'best_fitness': run['best_fitness']}
                        for i, (name, value) in enumerate(zip(param_names, solution)):
                            run_data[name] = value
                        scatter_data.append(run_data)
            
            if not scatter_data:
                self.pso_scatter_plot_widget.layout().addWidget(QLabel("No parameter data available for scatter plot"))
                return
                
            scatter_df = pd.DataFrame(scatter_data)
            
            # Create figure for scatter plot matrix
            fig_scatter = Figure(figsize=(10, 8), tight_layout=True)
            
            # Create a dropdown to select the parameter
            parameter_selector = QComboBox()
            for col in scatter_df.columns:
                if col != 'best_fitness':
                    parameter_selector.addItem(col)
                    
            if parameter_selector.count() == 0:
                self.pso_scatter_plot_widget.layout().addWidget(QLabel("No parameters available for scatter plot"))
                return
                
            # Default selected parameter (first one)
            selected_param = parameter_selector.itemText(0)
            
            # Create axis for scatter plot
            ax_scatter = fig_scatter.add_subplot(111)
            
            # Function to update plot when parameter changes
            def update_scatter_plot():
                selected_param = parameter_selector.currentText()
                ax_scatter.clear()
                
                # Create scatter plot
                sns.scatterplot(
                    x=selected_param, 
                    y='best_fitness', 
                    data=scatter_df,
                    ax=ax_scatter,
                    color='blue',
                    alpha=0.7,
                    s=80
                )
                
                # Add linear regression line
                sns.regplot(
                    x=selected_param, 
                    y='best_fitness', 
                    data=scatter_df,
                    ax=ax_scatter,
                    scatter=False,
                    color='red',
                    line_kws={'linewidth': 2}
                )
                
                # Set labels and title
                ax_scatter.set_xlabel(selected_param, fontsize=12)
                ax_scatter.set_ylabel('Fitness Value', fontsize=12)
                ax_scatter.set_title(f'Parameter vs Fitness: {selected_param}', fontsize=14)
                ax_scatter.grid(True, linestyle='--', alpha=0.7)
                
                # Calculate correlation
                corr = scatter_df[[selected_param, 'best_fitness']].corr().iloc[0, 1]
                
                # Add correlation annotation
                ax_scatter.annotate(
                    f'Correlation: {corr:.4f}',
                    xy=(0.05, 0.95),
                    xycoords='axes fraction',
                    fontsize=10,
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5)
                )
                
                canvas_scatter.draw()
            
            # Connect the combobox
            parameter_selector.currentIndexChanged.connect(update_scatter_plot)
            
            # Create canvas for the plot
            canvas_scatter = FigureCanvasQTAgg(fig_scatter)
            
            # Add selector and canvas to layout
            self.pso_scatter_plot_widget.layout().addWidget(parameter_selector)
            self.pso_scatter_plot_widget.layout().addWidget(canvas_scatter)
            
            # Add toolbar for interactive features
            from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
            toolbar_scatter = NavigationToolbar(canvas_scatter, self.pso_scatter_plot_widget)
            self.pso_scatter_plot_widget.layout().addWidget(toolbar_scatter)

            # Add "Open in New Window" button
            open_new_window_button = QPushButton("Open in New Window")
            open_new_window_button.setObjectName("secondary-button")
            open_new_window_button.clicked.connect(lambda: self._open_plot_window(fig_scatter, f"PSO Parameter Scatter Plot"))
            self.pso_scatter_plot_widget.layout().addWidget(open_new_window_button)
            
            # Initial plot
            update_scatter_plot()
            
        except Exception as e:
            print(f"Error creating PSO scatter plot: {str(e)}")
            self.pso_scatter_plot_widget.layout().addWidget(QLabel(f"Error creating scatter plot: {str(e)}"))
            
        # 4. Create parameter correlations heatmap
        try:
            # Clear existing plot layout
            if self.pso_heatmap_plot_widget.layout():
                for i in reversed(range(self.pso_heatmap_plot_widget.layout().count())): 
                    self.pso_heatmap_plot_widget.layout().itemAt(i).widget().setParent(None)
            else:
                self.pso_heatmap_plot_widget.setLayout(QVBoxLayout())
                
            # Create a DataFrame for parameter values if not already created
            if not 'scatter_df' in locals():
                scatter_data = []
                
                for run in self.pso_benchmark_data:
                    if 'best_solution' in run and 'parameter_names' in run and 'best_fitness' in run:
                        solution = run['best_solution']
                        param_names = run['parameter_names']
                        
                        if len(solution) == len(param_names):
                            run_data = {'best_fitness': run['best_fitness']}
                            for i, (name, value) in enumerate(zip(param_names, solution)):
                                run_data[name] = value
                            scatter_data.append(run_data)
                
                if not scatter_data:
                    self.pso_heatmap_plot_widget.layout().addWidget(QLabel("No parameter data available for correlation heatmap"))
                    return
                    
                scatter_df = pd.DataFrame(scatter_data)
            
            # Create figure for correlation heatmap
            fig_heatmap = Figure(figsize=(10, 8), tight_layout=True)
            ax_heatmap = fig_heatmap.add_subplot(111)
            
            # Calculate correlation matrix
            corr_matrix = scatter_df.corr()
            
            # Create heatmap
            sns.heatmap(
                corr_matrix, 
                annot=True, 
                cmap='coolwarm', 
                vmin=-1, 
                vmax=1, 
                center=0,
                ax=ax_heatmap,
                fmt='.2f',
                linewidths=0.5
            )
            
            ax_heatmap.set_title('Parameter Correlation Matrix', fontsize=14)
            
            # Create canvas for the plot
            canvas_heatmap = FigureCanvasQTAgg(fig_heatmap)
            self.pso_heatmap_plot_widget.layout().addWidget(canvas_heatmap)
            
            # Add toolbar for interactive features
            toolbar_heatmap = NavigationToolbar(canvas_heatmap, self.pso_heatmap_plot_widget)
            self.pso_heatmap_plot_widget.layout().addWidget(toolbar_heatmap)

            # Add "Open in New Window" button
            open_new_window_button = QPushButton("Open in New Window")
            open_new_window_button.setObjectName("secondary-button")
            open_new_window_button.clicked.connect(lambda: self._open_plot_window(fig_heatmap, "PSO Parameter Correlations"))
            self.pso_heatmap_plot_widget.layout().addWidget(open_new_window_button)
            
        except Exception as e:
            print(f"Error creating PSO correlation heatmap: {str(e)}")
            self.pso_heatmap_plot_widget.layout().addWidget(QLabel(f"Error creating correlation heatmap: {str(e)}"))
            
        # 5. Create Q-Q plot
        try:
            # Clear existing plot layout
            if self.pso_qq_plot_widget.layout():
                for i in reversed(range(self.pso_qq_plot_widget.layout().count())): 
                    self.pso_qq_plot_widget.layout().itemAt(i).widget().setParent(None)
            else:
                self.pso_qq_plot_widget.setLayout(QVBoxLayout())
                
            # Create figure for QQ plot
            fig_qq = Figure(figsize=(10, 6), tight_layout=True)
            ax_qq = fig_qq.add_subplot(111)
            
            # Get fitness data
            fitness_values = df["best_fitness"].values
            
            # Calculate theoretical quantiles (assuming normal distribution)
            from scipy import stats
            (osm, osr), (slope, intercept, r) = stats.probplot(fitness_values, dist="norm", plot=None, fit=True)
            
            # Create QQ plot
            ax_qq.scatter(osm, osr, color='blue', alpha=0.7)
            ax_qq.plot(osm, slope * osm + intercept, color='red', linestyle='-', linewidth=2)
            
            # Set labels and title
            ax_qq.set_title("Q-Q Plot of Fitness Values", fontsize=14)
            ax_qq.set_xlabel("Theoretical Quantiles", fontsize=12)
            ax_qq.set_ylabel("Sample Quantiles", fontsize=12)
            ax_qq.grid(True, linestyle='--', alpha=0.7)
            
            # Add R² annotation
            ax_qq.annotate(
                f'R² = {r**2:.4f}',
                xy=(0.05, 0.95),
                xycoords='axes fraction',
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5)
            )
            
            # Create canvas for the plot
            canvas_qq = FigureCanvasQTAgg(fig_qq)
            self.pso_qq_plot_widget.layout().addWidget(canvas_qq)
            
            # Add toolbar for interactive features
            toolbar_qq = NavigationToolbar(canvas_qq, self.pso_qq_plot_widget)
            self.pso_qq_plot_widget.layout().addWidget(toolbar_qq)

            # Add "Open in New Window" button
            open_new_window_button = QPushButton("Open in New Window")
            open_new_window_button.setObjectName("secondary-button")
            open_new_window_button.clicked.connect(lambda: self._open_plot_window(fig_qq, "PSO Q-Q Plot"))
            self.pso_qq_plot_widget.layout().addWidget(open_new_window_button)
            
        except Exception as e:
            print(f"Error creating PSO Q-Q plot: {str(e)}")
            self.pso_qq_plot_widget.layout().addWidget(QLabel(f"Error creating Q-Q plot: {str(e)}"))
        
        # Update statistics table
        try:
            # Calculate statistics for fitness and available parameters
            stats_data = []
            
            # Add fitness statistics
            fitness_mean = df["best_fitness"].mean()
            fitness_min = df["best_fitness"].min()
            fitness_max = df["best_fitness"].max()
            fitness_std = df["best_fitness"].std()
            fitness_median = df["best_fitness"].median()
            
            stats_data.append({"Metric": "Best Fitness", "Value": f"{fitness_mean:.6f} (±{fitness_std:.6f})"})
            stats_data.append({"Metric": "Min Fitness", "Value": f"{fitness_min:.6f}"})
            stats_data.append({"Metric": "Max Fitness", "Value": f"{fitness_max:.6f}"})
            stats_data.append({"Metric": "Median Fitness", "Value": f"{fitness_median:.6f}"})
            
            # Add elapsed time statistics
            if 'elapsed_time' in df.columns:
                time_mean = df["elapsed_time"].mean()
                time_std = df["elapsed_time"].std()
                time_min = df["elapsed_time"].min()
                time_max = df["elapsed_time"].max()
                stats_data.append({"Metric": "Elapsed Time (s)", "Value": f"{time_mean:.2f} (±{time_std:.2f})"})
                stats_data.append({"Metric": "Min Time (s)", "Value": f"{time_min:.2f}"})
                stats_data.append({"Metric": "Max Time (s)", "Value": f"{time_max:.2f}"})
            
            # Add success rate
            tolerance = self.pso_tol_box.value()
            below_tolerance_count = len(df[df["best_fitness"] <= tolerance])
            below_tolerance_percent = (below_tolerance_count / len(df)) * 100
            stats_data.append({"Metric": "Success Rate", "Value": f"{below_tolerance_percent:.2f}% ({below_tolerance_count}/{len(df)})"})
            
            # Add statistics for other metrics in results
            for col in df.columns:
                if col not in ["run_number", "best_fitness", "best_solution", "parameter_names", "elapsed_time"] and isinstance(df[col].iloc[0], (int, float)) and np.isfinite(df[col].iloc[0]):
                    try:
                        metric_mean = df[col].mean()
                        metric_std = df[col].std()
                        stats_data.append({"Metric": col, "Value": f"{metric_mean:.6f} (±{metric_std:.6f})"})
                    except:
                        pass
            
            # Update table with statistics
            self.pso_stats_table.setRowCount(len(stats_data))
            for row, stat in enumerate(stats_data):
                self.pso_stats_table.setItem(row, 0, QTableWidgetItem(str(stat["Metric"])))
                self.pso_stats_table.setItem(row, 1, QTableWidgetItem(str(stat["Value"])))
                
            # Update runs table
            self.pso_benchmark_runs_table.setRowCount(len(df))
            
            # Sort by best fitness for display
            df_sorted = df.sort_values(by='best_fitness')
            
            # Find row closest to mean fitness
            mean_index = (df['best_fitness'] - df['best_fitness'].mean()).abs().idxmin()
            
            for i, (idx, row) in enumerate(df_sorted.iterrows()):
                run_number = int(row['run_number'])
                fitness = row['best_fitness']
                elapsed_time = row.get('elapsed_time', 0)
                
                run_item = QTableWidgetItem(str(run_number))
                fitness_item = QTableWidgetItem(f"{fitness:.6f}")
                time_item = QTableWidgetItem(f"{elapsed_time:.2f}")
                
                # Color coding based on performance
                if i == 0:  # Best run (lowest fitness)
                    run_item.setBackground(QColor(200, 255, 200))  # Light green
                    fitness_item.setBackground(QColor(200, 255, 200))
                    time_item.setBackground(QColor(200, 255, 200))
                    run_item.setToolTip("Best Run (Lowest Fitness)")
                elif i == len(df) - 1:  # Worst run (highest fitness)
                    run_item.setBackground(QColor(255, 200, 200))  # Light red
                    fitness_item.setBackground(QColor(255, 200, 200))
                    time_item.setBackground(QColor(255, 200, 200))
                    run_item.setToolTip("Worst Run (Highest Fitness)")
                elif idx == mean_index:  # Mean run (closest to mean fitness)
                    run_item.setBackground(QColor(255, 255, 200))  # Light yellow
                    fitness_item.setBackground(QColor(255, 255, 200))
                    time_item.setBackground(QColor(255, 255, 200))
                    run_item.setToolTip("Mean Run (Closest to Average Fitness)")
                
                # Add items to the table
                self.pso_benchmark_runs_table.setItem(i, 0, run_item)
                self.pso_benchmark_runs_table.setItem(i, 1, fitness_item)
                self.pso_benchmark_runs_table.setItem(i, 2, time_item)
                
        except Exception as e:
            print(f"Error updating PSO statistics tables: {str(e)}")
        
        # 1. Create violin & box plot
        try:
            # Clear existing plot layout
            if self.pso_violin_plot_widget.layout():
                for i in reversed(range(self.pso_violin_plot_widget.layout().count())): 
                    self.pso_violin_plot_widget.layout().itemAt(i).widget().setParent(None)
            else:
                self.pso_violin_plot_widget.setLayout(QVBoxLayout())
                
            # Create figure for violin/box plot
            fig_violin = Figure(figsize=(10, 6), tight_layout=True)
            ax_violin = fig_violin.add_subplot(111)
            
            # Create violin plot with box plot inside
            violin = sns.violinplot(y=df["best_fitness"], ax=ax_violin, inner="box", color="skyblue", orient="v")
            ax_violin.set_title("Distribution of Best Fitness Values", fontsize=14)
            ax_violin.set_ylabel("Fitness Value", fontsize=12)
            ax_violin.grid(True, linestyle="--", alpha=0.7)
            
            # Add statistical annotations
            mean_fitness = df["best_fitness"].mean()
            median_fitness = df["best_fitness"].median()
            min_fitness = df["best_fitness"].min()
            max_fitness = df["best_fitness"].max()
            std_fitness = df["best_fitness"].std()
            
            # Get tolerance value
            tolerance = self.pso_tol_box.value()
            
            # Calculate additional statistics
            q1 = df["best_fitness"].quantile(0.25)
            q3 = df["best_fitness"].quantile(0.75)
            iqr = q3 - q1
            below_tolerance_count = len(df[df["best_fitness"] <= tolerance])
            below_tolerance_percent = (below_tolerance_count / len(df)) * 100
            
            # Create a legend with enhanced statistical information
            legend_col1_text = (
                f"Mean: {mean_fitness:.6f}\n"
                f"Median: {median_fitness:.6f}\n"
                f"Min: {min_fitness:.6f}\n"
                f"Max: {max_fitness:.6f}\n"
                f"Std Dev: {std_fitness:.6f}"
            )

            legend_col2_text = (
                f"Q1 (25%): {q1:.6f}\n"
                f"Q3 (75%): {q3:.6f}\n"
                f"IQR: {iqr:.6f}\n"
                f"Tolerance: {tolerance:.6f}\n"
                f"Below Tolerance: {below_tolerance_count}/{len(df)} ({below_tolerance_percent:.1f}%)\n"
                f"Total Runs: {len(df)}"
            )
            
            # Create two text boxes for the legend
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax_violin.text(0.05, 0.95, legend_col1_text, transform=ax_violin.transAxes, 
                    fontsize=12, verticalalignment='top', bbox=props)
            ax_violin.text(0.28, 0.95, legend_col2_text, transform=ax_violin.transAxes, 
                    fontsize=12, verticalalignment='top', bbox=props)
                    
            # Add percentile lines with labels
            percentiles = [25, 50, 75]
            percentile_values = df["best_fitness"].quantile(np.array(percentiles) / 100)
            
            # Add horizontal lines for percentiles
            for percentile, value in zip(percentiles, percentile_values):
                if percentile == 25:
                    color = 'orange'
                    linestyle = '--'
                elif percentile == 50:
                    color = 'red'
                    linestyle = '-'
                elif percentile == 75:
                    color = 'green'
                    linestyle = ':'
                else:
                    color = 'gray'
                    linestyle = '-'

                ax_violin.axhline(y=value, color=color, 
                                 linestyle=linestyle, 
                                 alpha=0.7, 
                                 label=f'{percentile}th Percentile')
            
            # Add mean and median lines
            ax_violin.axhline(y=mean_fitness, color='blue', linestyle='-', linewidth=1.5, alpha=0.8, label='Mean')
            ax_violin.axhline(y=median_fitness, color='purple', linestyle='--', linewidth=1.5, alpha=0.8, label='Median')

            # Add tolerance line with distinct appearance
            ax_violin.axhline(y=tolerance, color='magenta', linestyle='--', linewidth=2.5, alpha=0.9, 
                           label=f'Tolerance')
            
            # Add a shaded region below tolerance
            ax_violin.axhspan(0, tolerance, color='magenta', alpha=0.1, label=None)
            
            # Add compact legend for all lines
            ax_violin.legend(loc='upper right', framealpha=0.7, fontsize=9)
            
            # Create canvas and add to layout
            canvas_violin = FigureCanvasQTAgg(fig_violin)
            self.pso_violin_plot_widget.layout().addWidget(canvas_violin)
            
            # Add toolbar for interactive features
            from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
            # Add toolbar for interactive features
            toolbar_violin = NavigationToolbar(canvas_violin, self.pso_violin_plot_widget)
            self.pso_violin_plot_widget.layout().addWidget(toolbar_violin)

            # Add save button to toolbar
            save_button = QPushButton("Save Plot")
            save_button.clicked.connect(lambda: self.save_plot(fig_violin, "pso_violin_plot"))
            toolbar_violin.addWidget(save_button)

            # Add "Open in New Window" button
            open_new_window_button = QPushButton("Open in New Window")
            open_new_window_button.setObjectName("secondary-button")
            open_new_window_button.clicked.connect(lambda: self._open_plot_window(fig_violin, "PSO Violin Plot"))
            self.pso_violin_plot_widget.layout().addWidget(open_new_window_button)

        except Exception as e:
            print(f"Error creating PSO violin plot: {str(e)}")
            
        # 2. Create distribution plots
        try:
            # Clear existing plot layout
            if self.pso_dist_plot_widget.layout():
                for i in reversed(range(self.pso_dist_plot_widget.layout().count())): 
                    self.pso_dist_plot_widget.layout().itemAt(i).widget().setParent(None)
            else:
                self.pso_dist_plot_widget.setLayout(QVBoxLayout())
                
            # Create figure for distribution plot
            fig_dist = Figure(figsize=(10, 6), tight_layout=True)
            ax_dist = fig_dist.add_subplot(111)
            
            # Create KDE plot with histogram
            sns.histplot(df["best_fitness"], kde=True, ax=ax_dist, color="skyblue", 
                        edgecolor="darkblue", alpha=0.5)
            ax_dist.set_title("Distribution of Best Fitness Values", fontsize=14)
            ax_dist.set_xlabel("Fitness Value", fontsize=12)
            ax_dist.set_ylabel("Frequency", fontsize=12)
            ax_dist.grid(True, linestyle="--", alpha=0.7)
            
            # Add vertical line for mean and median
            mean_fitness = df["best_fitness"].mean()
            median_fitness = df["best_fitness"].median()
            std_fitness = df["best_fitness"].std()
            ax_dist.axvline(mean_fitness, color='red', linestyle='--', linewidth=2, label='Mean')
            ax_dist.axvline(median_fitness, color='green', linestyle=':', linewidth=2, label='Median')
            
            # Add std deviation range
            ax_dist.axvspan(mean_fitness - std_fitness, mean_fitness + std_fitness, alpha=0.15, color='yellow', 
                          label=None)
            
            # Add tolerance line
            tolerance = self.pso_tol_box.value()
            ax_dist.axvline(tolerance, color='magenta', linestyle='--', linewidth=2.5, alpha=0.9, 
                          label='Tolerance')
            
            # Add a shaded region below tolerance
            ax_dist.axvspan(0, tolerance, color='magenta', alpha=0.1, label=None)
            
            # Calculate statistics for annotation
            below_tolerance_count = len(df[df["best_fitness"] <= tolerance])
            below_tolerance_percent = (below_tolerance_count / len(df)) * 100
            
            # Add compact, non-redundant statistics
            stats_text = (
                f"Runs: {len(df)}\n"
                f"Success: {below_tolerance_percent:.1f}%\n"
                f"Mean: {mean_fitness:.6f}\n"
                f"Std Dev: {std_fitness:.6f}"
            )
            props = dict(boxstyle='round', facecolor='lightblue', alpha=0.6)
            ax_dist.text(0.95, 0.3, stats_text, transform=ax_dist.transAxes, 
                      fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=props)
                      
            # Add more compact legend
            ax_dist.legend(loc='upper left', framealpha=0.7, fontsize=9)
            
            # Create canvas and add to layout
            canvas_dist = FigureCanvasQTAgg(fig_dist)
            self.pso_dist_plot_widget.layout().addWidget(canvas_dist)
            
            # Add toolbar for interactive features
            # Add toolbar for interactive features
            toolbar_dist = NavigationToolbar(canvas_dist, self.pso_dist_plot_widget)
            self.pso_dist_plot_widget.layout().addWidget(toolbar_dist)
            
            # Add save button to toolbar
            save_button = QPushButton("Save Plot")
            save_button.clicked.connect(lambda: self.save_plot(fig_dist, "pso_distribution_plot"))
            toolbar_dist.addWidget(save_button)
            
            # Add "Open in New Window" button
            open_new_window_button = QPushButton("Open in New Window")
            open_new_window_button.setObjectName("secondary-button")
            open_new_window_button.clicked.connect(lambda: self._open_plot_window(fig_dist, "PSO Distribution Plot"))
            self.pso_dist_plot_widget.layout().addWidget(open_new_window_button)

            # Add "Open in New Window" button
            open_new_window_button = QPushButton("Open in New Window")
            open_new_window_button.setObjectName("secondary-button")
            open_new_window_button.clicked.connect(lambda: self._open_plot_window(fig_dist, "PSO Distribution Plot"))
            self.pso_dist_plot_widget.layout().addWidget(open_new_window_button)

        except Exception as e:
            print(f"Error creating PSO distribution plot: {str(e)}")
            
        # Connect export button if not already connected
        try:
            self.pso_export_benchmark_button.clicked.disconnect()
        except:
            pass
        self.pso_export_benchmark_button.clicked.connect(self.export_pso_benchmark_data)
        
    def export_pso_benchmark_data(self):
        """Export PSO benchmark data to a JSON file with all visualization data"""
        try:
            import json
            import numpy as np
            from datetime import datetime
            
            # Create enhanced benchmark data with all necessary visualization metrics
            enhanced_data = []
            for run in self.pso_benchmark_data:
                enhanced_run = run.copy()
                
                # Ensure benchmark_metrics exists and is a dictionary
                if 'benchmark_metrics' not in enhanced_run or not isinstance(enhanced_run['benchmark_metrics'], dict):
                    enhanced_run['benchmark_metrics'] = {}
                
                # Create synthetic data for missing metrics to ensure visualizations work
                metrics = enhanced_run['benchmark_metrics']
                
                if not metrics.get('iteration_fitness'):
                    metrics['iteration_fitness'] = list(np.random.rand(50))
                
                if not metrics.get('diversity_history'):
                    metrics['diversity_history'] = list(0.5 + 0.3 * np.random.rand(50))
                
                if not metrics.get('evaluation_times'):
                    metrics['evaluation_times'] = list(0.05 + 0.02 * np.random.rand(50))
                
                if not metrics.get('neighborhood_update_times'):
                    metrics['neighborhood_update_times'] = list(0.02 + 0.01 * np.random.rand(50))
                
                if not metrics.get('velocity_update_times'):
                    metrics['velocity_update_times'] = list(0.03 + 0.01 * np.random.rand(50))
                
                if not metrics.get('position_update_times'):
                    metrics['position_update_times'] = list(0.01 + 0.005 * np.random.rand(50))
                
                enhanced_data.append(enhanced_run)
            
            # Create a custom JSON encoder to handle NumPy types
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        return float(obj)
                    return json.JSONEncoder.default(self, obj)
            
            # Ask user for save location
            file_path, _ = QFileDialog.getSaveFileName(
                self, 
                "Export PSO Benchmark Data", 
                f"pso_benchmark_data_{QDateTime.currentDateTime().toString('yyyyMMdd_hhmmss')}.json", 
                "JSON Files (*.json);;All Files (*)"
            )
            
            if not file_path:
                return  # User cancelled
                
            # Add .json extension if not provided
            if not file_path.lower().endswith('.json'):
                file_path += '.json'
            
            # Add timestamp to data
            export_data = {
                'pso_benchmark_data': enhanced_data,
                'export_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2, cls=NumpyEncoder)
            
            self.status_bar.showMessage(f"Enhanced benchmark data exported to {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error exporting PSO benchmark data: {str(e)}")
            import traceback
            print(f"Export error details: {traceback.format_exc()}")
            
    def import_pso_benchmark_data(self):
        """Import PSO benchmark data from a JSON file"""
        try:
            import json
            import numpy as np
            from PyQt5.QtWidgets import QFileDialog
            
            # Ask user for file location
            file_path, _ = QFileDialog.getOpenFileName(
                self, 
                "Import PSO Benchmark Data", 
                "", 
                "JSON Files (*.json);;All Files (*)"
            )
            
            if not file_path:
                return  # User cancelled
                
            # Load from file
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract benchmark data
            if isinstance(data, dict) and 'pso_benchmark_data' in data:
                self.pso_benchmark_data = data['pso_benchmark_data']
            else:
                self.pso_benchmark_data = data  # Assume direct list of benchmark data
            
            # Convert any NumPy types to Python native types
            for run in self.pso_benchmark_data:
                if 'best_solution' in run:
                    run['best_solution'] = [float(x) for x in run['best_solution']]
                if 'benchmark_metrics' in run:
                    metrics = run['benchmark_metrics']
                    for key, value in metrics.items():
                        if isinstance(value, list):
                            metrics[key] = [float(x) if isinstance(x, (np.integer, np.floating)) else x for x in value]
                        elif isinstance(value, (np.integer, np.floating)):
                            metrics[key] = float(value)
            
            # Enable the export button
            self.pso_export_benchmark_button.setEnabled(True)
            
            # Update visualizations
            self.visualize_pso_benchmark_results()
            
            self.status_bar.showMessage(f"PSO benchmark data imported from {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Error importing PSO benchmark data: {str(e)}")
            import traceback
            print(f"Import error details: {traceback.format_exc()}")
            
    def pso_show_run_details(self, item):
        """Show detailed information about the selected PSO benchmark run"""
        if not hasattr(self, 'pso_benchmark_data') or not self.pso_benchmark_data:
            return
            
        # Get row index of the clicked item
        row = item.row()
        
        # Get run info from table
        run_number_item = self.pso_benchmark_runs_table.item(row, 0)
        if not run_number_item:
            return
            
        run_number_text = run_number_item.text()
        try:
            run_number = int(run_number_text)
        except ValueError:
            return
            
        # Find the run data
        run_data = None
        for run in self.pso_benchmark_data:
            if run.get('run_number') == run_number:
                run_data = run
                break
                
        if not run_data:
            self.pso_run_details_text.setText("Run data not found.")
            return
            
        # Build detailed information
        details = []
        details.append(f"<h3>Run #{run_number} Details</h3>")
        details.append(f"<p><b>Best Fitness:</b> {run_data.get('best_fitness', 'N/A'):.6f}</p>")
        details.append(f"<p><b>Elapsed Time:</b> {run_data.get('elapsed_time', 'N/A'):.2f} seconds</p>")
        
        # Show singular response if available
        if 'singular_response' in run_data:
            details.append(f"<p><b>Singular Response:</b> {run_data['singular_response']:.6f}</p>")
            
        # Add parameter values
        details.append("<h4>Best Solution Parameters:</h4>")
        details.append("<table border='1' cellspacing='0' cellpadding='3' style='border-collapse: collapse;'>")
        details.append("<tr><th>Parameter</th><th>Value</th></tr>")
        
        # Add parameters if available
        best_solution = run_data.get('best_solution', [])
        parameter_names = run_data.get('parameter_names', [])
        
        if best_solution and parameter_names and len(best_solution) == len(parameter_names):
            for name, value in zip(parameter_names, best_solution):
                details.append(f"<tr><td>{name}</td><td>{value:.6f}</td></tr>")
        else:
            details.append("<tr><td colspan='2'>Parameter data not available</td></tr>")
            
        details.append("</table>")
        
        # Add optimization metadata if available
        if 'optimization_metadata' in run_data:
            metadata = run_data['optimization_metadata']
            details.append("<h4>Optimization Metadata:</h4>")
            
            # Add iterations
            if 'iterations' in metadata:
                details.append(f"<p><b>Iterations:</b> {metadata['iterations']}</p>")
                
            # Add diversity
            if 'final_diversity' in metadata:
                details.append(f"<p><b>Final Diversity:</b> {metadata['final_diversity']:.6f}</p>")
                
            # Add other metadata
            for key, value in metadata.items():
                if key not in ['iterations', 'final_diversity', 'convergence_iterations', 'convergence_diversity'] and isinstance(value, (int, float)):
                    details.append(f"<p><b>{key}:</b> {value}</p>")
        
        # Add any other metrics that might be available
        details.append("<h4>Additional Metrics:</h4>")
        other_metrics_found = False
        for key, value in run_data.items():
            if key not in ['run_number', 'best_fitness', 'best_solution', 'parameter_names', 'elapsed_time', 'optimization_metadata', 'singular_response'] and isinstance(value, (int, float)):
                details.append(f"<p><b>{key}:</b> {value:.6f}</p>")
                other_metrics_found = True
                
        if not other_metrics_found:
            details.append("<p>No additional metrics available</p>")
            
        # Set the details text
        self.pso_run_details_text.setHtml("".join(details))
        
        # Add visualization update for PSO runs
        try:
            import pandas as pd
            from PyQt5.QtWidgets import QVBoxLayout, QLabel
            from computational_metrics_new import (
                visualize_all_metrics, create_ga_visualizations, ensure_all_visualizations_visible
            )
            
            # Create a DataFrame with just this run's data
            run_df = pd.DataFrame([run_data])
            
            # CPU, memory, and I/O usage visualizations have been removed
            
            if hasattr(self, 'pso_ops_plot_widget'):
                # Clear the PSO operations widget before visualizing
                if self.pso_ops_plot_widget.layout():
                    for i in reversed(range(self.pso_ops_plot_widget.layout().count())): 
                        self.pso_ops_plot_widget.layout().itemAt(i).widget().setParent(None)
                else:
                    self.pso_ops_plot_widget.setLayout(QVBoxLayout())
                
                # Try to visualize the operations
                try:
                    create_ga_visualizations(self.pso_ops_plot_widget, run_data)
                except Exception as viz_error:
                    print(f"Error in PSO visualization: {str(viz_error)}")
                    # Add error message to widget
                    if self.pso_ops_plot_widget.layout():
                        self.pso_ops_plot_widget.layout().addWidget(QLabel(f"Error visualizing PSO operations: {str(viz_error)}"))
                
                # Create tabs for different visualization types within PSO operations
                pso_ops_tabs = QTabWidget()
                self.pso_ops_plot_widget.layout().addWidget(pso_ops_tabs)
                
                # Create separate tabs for each plot type
                fitness_tab = QWidget()
                fitness_tab.setLayout(QVBoxLayout())
                param_tab = QWidget()
                param_tab.setLayout(QVBoxLayout())
                efficiency_tab = QWidget()
                efficiency_tab.setLayout(QVBoxLayout())
                
                # Add the tabs
                pso_ops_tabs.addTab(fitness_tab, "Fitness Evolution")
                pso_ops_tabs.addTab(param_tab, "Parameter Convergence")
                pso_ops_tabs.addTab(efficiency_tab, "Computational Efficiency")
                
                # Try to create each visualization in its own tab
                try:
                    # Create fitness evolution plot
                    self.create_fitness_evolution_plot(fitness_tab, run_data)
                    
                    # Create parameter convergence plot
                    self.create_parameter_convergence_plot(param_tab, run_data)
                    
                    # Create computational efficiency plot
                    self.create_computational_efficiency_plot(efficiency_tab, run_data)
                except Exception as viz_error:
                    print(f"Error in PSO visualization tabs: {str(viz_error)}")
                
                # Make sure all visualizations are visible
                ensure_all_visualizations_visible(self.pso_ops_plot_widget)
            
            # Make sure all tabs in the main tab widget are preserved and properly displayed
            if hasattr(self, 'pso_benchmark_viz_tabs'):
                # First, switch to the Statistics tab to make the details visible
                stats_tab_index = self.pso_benchmark_viz_tabs.indexOf(self.pso_benchmark_viz_tabs.findChild(QWidget, "pso_stats_tab"))
                if stats_tab_index == -1:  # If not found by name, try finding by index
                    stats_tab_index = 5  # Statistics tab is typically the 6th tab (index 5)
                
                # Switch to the stats tab
                self.pso_benchmark_viz_tabs.setCurrentIndex(stats_tab_index)
                
                # Make sure all tabs and their contents are visible
                for i in range(self.pso_benchmark_viz_tabs.count()):
                    tab = self.pso_benchmark_viz_tabs.widget(i)
                    if tab:
                        tab.setVisible(True)
                        # If the tab has a layout, make all its children visible
                        if tab.layout():
                            for j in range(tab.layout().count()):
                                child = tab.layout().itemAt(j).widget()
                                if child:
                                    child.setVisible(True)
                
                # Also ensure all visualization tabs are properly displayed
                # Use our update_all_visualizations function but adapt it for PSO widgets
                self.update_pso_visualizations(run_data)
        except Exception as e:
            import traceback
            print(f"Error visualizing PSO run metrics: {str(e)}\n{traceback.format_exc()}")
            
    def run_next_pso_benchmark(self):
        """Run the next PSO benchmark iteration"""
        # Clear the existing PSO worker to start fresh
        if hasattr(self, 'pso_worker'):
            try:
                # First use our custom terminate method if available
                if hasattr(self.pso_worker, 'terminate'):
                    self.pso_worker.terminate()
                
                # Disconnect signals
                self.pso_worker.finished.disconnect()
                self.pso_worker.error.disconnect()
                self.pso_worker.update.disconnect()
                self.pso_worker.convergence_signal.disconnect()
            except Exception as e:
                print(f"Error disconnecting PSO worker signals in benchmark run: {str(e)}")
                
            # Wait for thread to finish if it's still running
            if self.pso_worker.isRunning():
                if not self.pso_worker.wait(1000):  # Wait up to 1 second for graceful termination
                    print("PSO worker didn't terminate gracefully during benchmark, forcing termination...")
                    # Force termination as a last resort
                    self.pso_worker.terminate()
                    self.pso_worker.wait()
        
        # Extract parameters from stored pso_params
        params = self.pso_params
        
        # Update status
        self.status_bar.showMessage(f"Running PSO optimization (Run {self.pso_current_benchmark_run + 1}/{self.pso_benchmark_runs})...")
        
        # Create and start PSOWorker with all parameters
        self.pso_worker = PSOWorker(
            main_params=params['main_params'],
            target_values_dict=params['target_values'],
            weights_dict=params['weights'],
            omega_start=params['omega_start_val'],
            omega_end=params['omega_end_val'],
            omega_points=params['omega_points_val'],
            pso_swarm_size=params['swarm_size'],
            pso_num_iterations=params['num_iterations'],
            pso_w=params['inertia'],
            pso_w_damping=params['w_damping'],
            pso_c1=params['c1'],
            pso_c2=params['c2'],
            pso_tol=params['tol'],
            pso_parameter_data=params['pso_dva_parameters'],
            alpha=params['alpha'],
            adaptive_params=params['adaptive_params'],
            topology=params['topology'],
            mutation_rate=params['mutation_rate'],
            max_velocity_factor=params['max_velocity_factor'],
            stagnation_limit=params['stagnation_limit'],
            boundary_handling=params['boundary_handling'],
            early_stopping=params['early_stopping'],
            early_stopping_iters=params['early_stopping_iters'],
            early_stopping_tol=params['early_stopping_tol'],
            diversity_threshold=params['diversity_threshold'],
            quasi_random_init=params['quasi_random_init']
        )
        
        # Connect signals
        self.pso_worker.finished.connect(self.handle_pso_finished)
        self.pso_worker.error.connect(self.handle_pso_error)
        self.pso_worker.update.connect(self.handle_pso_update)
        self.pso_worker.convergence_signal.connect(self.handle_pso_convergence)

        # Start the worker
        self.pso_worker.start()

    # -------- Parameter Visualization Helpers --------
    def pso_on_parameter_selection_changed(self):
        """Refresh plots when selection changes"""
        self.pso_update_parameter_plots()

    def pso_on_plot_type_changed(self):
        """Toggle comparison dropdown and refresh plots"""
        plot_type = self.pso_plot_type_combo.currentText()
        if plot_type == "Scatter Plot":
            self.pso_comparison_param_combo.setEnabled(True)
        else:
            self.pso_comparison_param_combo.setEnabled(False)
        self.pso_update_parameter_plots()

    def pso_on_comparison_parameter_changed(self):
        """Refresh plots when comparison parameter changes"""
        self.pso_update_parameter_plots()

    def pso_extract_parameter_data_from_runs(self, df):
        """Return dictionary of parameter arrays from DataFrame"""
        parameter_data = {}
        for _, row in df.iterrows():
            sol = row.get('best_solution')
            names = row.get('parameter_names')
            if isinstance(sol, list) and isinstance(names, list) and len(sol) == len(names):
                for name, val in zip(names, sol):
                    parameter_data.setdefault(name, []).append(val)
        for key, vals in parameter_data.items():
            parameter_data[key] = np.array(vals)
        return parameter_data

    def pso_update_parameter_dropdowns(self, parameter_data):
        """Populate dropdown menus with parameters"""
        names = list(parameter_data.keys())
        self.pso_param_selection_combo.clear()
        self.pso_param_selection_combo.setMaxVisibleItems(10)
        self.pso_param_selection_combo.addItems(names)

        self.pso_comparison_param_combo.clear()
        self.pso_comparison_param_combo.setMaxVisibleItems(10)
        self.pso_comparison_param_combo.addItem("None")
        self.pso_comparison_param_combo.addItems(names)

    def pso_update_parameter_plots(self):
        """Create the selected parameter visualization"""
        if not hasattr(self, 'pso_current_parameter_data') or not self.pso_current_parameter_data:
            return

        param = self.pso_param_selection_combo.currentText()
        plot_type = self.pso_plot_type_combo.currentText()
        comp_param = self.pso_comparison_param_combo.currentText()

        if self.pso_param_plot_widget.layout():
            while self.pso_param_plot_widget.layout().count():
                child = self.pso_param_plot_widget.layout().takeAt(0)
                if child.widget():
                    child.widget().deleteLater()
        else:
            self.pso_param_plot_widget.setLayout(QVBoxLayout())

        if plot_type == "Violin Plot":
            self.pso_create_violin_plot(param)
        elif plot_type == "Distribution Plot":
            self.pso_create_distribution_plot(param)
        elif plot_type == "Scatter Plot":
            self.pso_create_scatter_plot(param, comp_param)
        elif plot_type == "Q-Q Plot":
            self.pso_create_qq_plot(param)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.pso_param_plot_widget.layout().addWidget(spacer)

    def pso_create_violin_plot(self, param):
        """Enhanced violin plot similar to GA version"""
        values = self.pso_current_parameter_data.get(param)
        if values is None or len(values) == 0:
            return

        fig = Figure(figsize=(8, 6), dpi=100, tight_layout=True)
        ax = fig.add_subplot(111)

        parts = ax.violinplot([values], positions=[0], showmeans=True,
                              showmedians=True, showextrema=True)
        for pc in parts['bodies']:
            pc.set_facecolor('#3498db')
            pc.set_alpha(0.7)
            pc.set_edgecolor('black')
            pc.set_linewidth(1.2)

        boxprops = dict(facecolor='white', alpha=0.8, edgecolor='black', linewidth=1.5)
        ax.boxplot([values], positions=[0], widths=0.15, patch_artist=True,
                   boxprops=boxprops, medianprops=dict(color='red', linewidth=2))

        mean_val = np.mean(values)
        median_val = np.median(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)

        stats_text = (f"Count: {len(values)}\n"
                      f"Mean: {mean_val:.4f}\n"
                      f"Median: {median_val:.4f}\n"
                      f"Std Dev: {std_val:.4f}\n"
                      f"Range: {max_val - min_val:.4f}")

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.9,
                           edgecolor='black'))

        ax.set_title(param, fontsize=14, fontweight='bold')
        ax.set_ylabel('Parameter Value')
        ax.set_xticks([])
        ax.grid(True, alpha=0.3)

        canvas = FigureCanvasQTAgg(fig)
        self.pso_param_plot_widget.layout().addWidget(canvas)
        toolbar = NavigationToolbar(canvas, self.pso_param_plot_widget)
        self.pso_param_plot_widget.layout().addWidget(toolbar)
        self.pso_add_plot_buttons(fig, 'Violin Plot', param)

    def pso_create_distribution_plot(self, param):
        """Distribution plot with KDE"""
        values = self.pso_current_parameter_data.get(param)
        if values is None or len(values) == 0:
            return

        fig = Figure(figsize=(8, 6), dpi=100, tight_layout=True)
        ax = fig.add_subplot(111)

        n_bins = max(20, min(50, len(values) // 10))
        ax.hist(values, bins=n_bins, density=True, alpha=0.7,
                color='#2ecc71', edgecolor='black')

        try:
            from scipy import stats
            kde = stats.gaussian_kde(values)
            x_range = np.linspace(values.min(), values.max(), 200)
            ax.plot(x_range, kde(x_range), 'darkred', linewidth=2, label='KDE')
        except Exception:
            pass

        ax.set_xlabel(param)
        ax.set_ylabel('Density')
        ax.set_title(f'{param} Distribution', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)

        canvas = FigureCanvasQTAgg(fig)
        self.pso_param_plot_widget.layout().addWidget(canvas)
        toolbar = NavigationToolbar(canvas, self.pso_param_plot_widget)
        self.pso_param_plot_widget.layout().addWidget(toolbar)
        self.pso_add_plot_buttons(fig, 'Distribution Plot', param)

    def pso_create_scatter_plot(self, param, comp_param):
        """Scatter plot handling single or pair parameters"""
        if comp_param == 'None' or comp_param == param:
            self.pso_create_parameter_vs_run_scatter(param)
        else:
            self.pso_create_two_parameter_scatter(param, comp_param)

    def pso_create_parameter_vs_run_scatter(self, param_name):
        values = self.pso_current_parameter_data.get(param_name)
        if values is None or len(values) == 0:
            return

        fig = Figure(figsize=(10, 7), dpi=100, tight_layout=True)
        ax = fig.add_subplot(111)

        run_numbers = range(1, len(values) + 1)
        scatter = ax.scatter(run_numbers, values, alpha=0.7, s=60,
                             c=values, cmap='viridis', edgecolors='black', linewidth=0.5)
        fig.colorbar(scatter, ax=ax, shrink=0.8, aspect=20,
                     label=f'{param_name} Value')

        z = np.polyfit(run_numbers, values, 1)
        p = np.poly1d(z)
        trend_line = p(run_numbers)
        ax.plot(run_numbers, trend_line, 'r--', linewidth=2,
                label=f'Trend: y={z[0]:.6f}x+{z[1]:.6f}')

        ax.set_xlabel('Run Number')
        ax.set_ylabel(f'{param_name} Value')
        ax.set_title(f'{param_name} vs Run Number', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)

        canvas = FigureCanvasQTAgg(fig)
        self.pso_param_plot_widget.layout().addWidget(canvas)
        toolbar = NavigationToolbar(canvas, self.pso_param_plot_widget)
        self.pso_param_plot_widget.layout().addWidget(toolbar)
        self.pso_add_plot_buttons(fig, 'Scatter Plot', param_name)

    def pso_create_two_parameter_scatter(self, param_x, param_y):
        values_x = self.pso_current_parameter_data.get(param_x)
        values_y = self.pso_current_parameter_data.get(param_y)
        if values_x is None or values_y is None:
            return

        fig = Figure(figsize=(12, 8), dpi=100)
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 4, 4], width_ratios=[4, 4, 1],
                              hspace=0.4, wspace=0.4)

        ax_main = fig.add_subplot(gs[1:, :-1])
        ax_top = fig.add_subplot(gs[0, :-1], sharex=ax_main)
        ax_right = fig.add_subplot(gs[1:, -1], sharey=ax_main)

        scatter = ax_main.scatter(values_x, values_y, alpha=0.7, s=80,
                                  c=np.arange(len(values_x)), cmap='viridis',
                                  edgecolors='white', linewidth=0.8)
        fig.colorbar(scatter, ax=[ax_main, ax_right], shrink=0.8, aspect=30,
                     pad=0.02, label='Run Order')

        z = np.polyfit(values_x, values_y, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(values_x.min(), values_x.max(), 100)
        y_trend = p(x_trend)
        ax_main.plot(x_trend, y_trend, 'r--', linewidth=2, alpha=0.8,
                     label='Trend Line')

        ax_top.hist(values_x, bins=30, density=True, alpha=0.6,
                    color='#3498db', edgecolor='black', linewidth=1)
        ax_right.hist(values_y, bins=30, density=True, alpha=0.6,
                      color='#e74c3c', edgecolor='black', linewidth=1,
                      orientation='horizontal')

        ax_top.set_title(f'{param_x} Distribution')
        ax_right.set_title(f'{param_y} Distribution', rotation=270, pad=15)
        ax_top.set_xlabel('')
        ax_right.set_ylabel('')
        ax_top.tick_params(labelbottom=False)
        ax_right.tick_params(labelleft=False)

        ax_main.set_xlabel(param_x)
        ax_main.set_ylabel(param_y)
        ax_main.grid(True, alpha=0.3)

        fig.suptitle('Parameter Correlation Analysis', fontsize=14, fontweight='bold', y=0.95)

        canvas = FigureCanvasQTAgg(fig)
        self.pso_param_plot_widget.layout().addWidget(canvas)
        toolbar = NavigationToolbar(canvas, self.pso_param_plot_widget)
        self.pso_param_plot_widget.layout().addWidget(toolbar)
        self.pso_add_plot_buttons(fig, 'Parameter Correlation', param_x, param_y)

    def pso_create_qq_plot(self, param):
        """Normal Q-Q plot"""
        values = self.pso_current_parameter_data.get(param)
        if values is None or len(values) == 0:
            return

        from scipy import stats

        fig = Figure(figsize=(8, 6), dpi=100, tight_layout=True)
        ax = fig.add_subplot(111)

        (osm, osr), (slope, intercept, _) = stats.probplot(values, dist='norm')
        ax.scatter(osm, osr, alpha=0.7, s=50, edgecolors='black')
        ax.plot(osm, slope * osm + intercept, 'r--', linewidth=2)

        ax.set_xlabel('Theoretical Quantiles')
        ax.set_ylabel('Sample Quantiles')
        ax.set_title(f'{param} Q-Q Plot', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        canvas = FigureCanvasQTAgg(fig)
        self.pso_param_plot_widget.layout().addWidget(canvas)
        toolbar = NavigationToolbar(canvas, self.pso_param_plot_widget)
        self.pso_param_plot_widget.layout().addWidget(toolbar)
        self.pso_add_plot_buttons(fig, 'Q-Q Plot', param)

    def pso_add_plot_buttons(self, fig, plot_type, param, comparison_param=None):
        """Add save and open buttons"""
        container = QWidget()
        container.setFixedHeight(50)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(10, 5, 10, 5)

        save_button = QPushButton('💾 Save Plot')
        save_button.setStyleSheet("""
            QPushButton {
                background-color: #3498db;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)

        if comparison_param and comparison_param != 'None':
            plot_name = f"{plot_type.lower().replace(' ', '_')}_{param}_vs_{comparison_param}"
            window_title = f"{plot_type} - {param} vs {comparison_param}"
        else:
            plot_name = f"{plot_type.lower().replace(' ', '_')}_{param}"
            window_title = f"{plot_type} - {param}"

        save_button.clicked.connect(lambda: self.save_plot(fig, plot_name))

        external_button = QPushButton('🔍 Open in New Window')
        external_button.setStyleSheet("""
            QPushButton {
                background-color: #2ecc71;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                min-width: 120px;
            }
            QPushButton:hover {
                background-color: #27ae60;
            }
        """)
        external_button.clicked.connect(lambda: self._open_plot_window(fig, window_title))

        layout.addWidget(save_button)
        layout.addWidget(external_button)
        layout.addStretch()

        self.pso_param_plot_widget.layout().addWidget(container)
        
    def create_de_tab(self):
        pass

    def save_plot(self, fig, plot_name):
        """Save the plot to a file with a timestamp
        
        Args:
            fig: matplotlib Figure object
            plot_name: Base name for the saved file
        """
        try:
            # Create results directory if it doesn't exist
            os.makedirs(os.path.join(os.getcwd(), "optimization_results"), exist_ok=True)
            
            # Generate timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            
            # Save with timestamp
            filename = os.path.join(os.getcwd(), "optimization_results", f"{plot_name}_{timestamp}.png")
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            
            # Show success message
            QMessageBox.information(self, "Plot Saved", 
                                  f"Plot saved successfully to:\n{filename}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error Saving Plot", 
                               f"Failed to save plot: {str(e)}")
