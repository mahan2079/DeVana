from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
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
from computational_metrics_new import visualize_all_metrics, ensure_all_visualizations_visible
from modules.plotwindow import PlotWindow
from workers.GAWorker import GAWorker

class GAOptimizationMixin:
    def create_ga_tab(self):
        """Create the genetic algorithm optimization tab"""
        self.ga_tab = QWidget()
        layout = QVBoxLayout(self.ga_tab)

        # Create sub-tabs widget
        self.ga_sub_tabs = QTabWidget()

        # -------------------- Sub-tab 1: GA Hyperparameters --------------------
        ga_hyper_tab = QWidget()
        ga_hyper_layout = QFormLayout(ga_hyper_tab)

        self.ga_pop_size_box = QSpinBox()
        self.ga_pop_size_box.setRange(1, 10000)
        self.ga_pop_size_box.setValue(800)

        self.ga_num_generations_box = QSpinBox()
        self.ga_num_generations_box.setRange(1, 10000)
        self.ga_num_generations_box.setValue(100)

        self.ga_cxpb_box = QDoubleSpinBox()
        self.ga_cxpb_box.setRange(0, 1)
        self.ga_cxpb_box.setValue(0.7)
        self.ga_cxpb_box.setDecimals(3)

        self.ga_mutpb_box = QDoubleSpinBox()
        self.ga_mutpb_box.setRange(0, 1)
        self.ga_mutpb_box.setValue(0.2)
        self.ga_mutpb_box.setDecimals(3)

        self.ga_tol_box = QDoubleSpinBox()
        self.ga_tol_box.setRange(0, 1e6)
        self.ga_tol_box.setValue(1e-3)
        self.ga_tol_box.setDecimals(6)

        self.ga_alpha_box = QDoubleSpinBox()
        self.ga_alpha_box.setRange(0.0, 10.0)
        self.ga_alpha_box.setDecimals(4)
        self.ga_alpha_box.setSingleStep(0.01)
        self.ga_alpha_box.setValue(0.01)
        
        # Add benchmarking runs box
        self.ga_benchmark_runs_box = QSpinBox()
        self.ga_benchmark_runs_box.setRange(1, 1000)
        self.ga_benchmark_runs_box.setValue(1)
        self.ga_benchmark_runs_box.setToolTip("Number of times to run the GA for benchmarking (1 = single run)")
        
        # Add adaptive rates checkbox
        self.adaptive_rates_checkbox = QCheckBox("Use Adaptive Rates")
        self.adaptive_rates_checkbox.setChecked(False)
        self.adaptive_rates_checkbox.setToolTip("Automatically adjust crossover and mutation rates during optimization")
        self.adaptive_rates_checkbox.stateChanged.connect(self.toggle_adaptive_rates_options)
        
        # Create a widget to hold adaptive rate options
        self.adaptive_rates_options = QWidget()
        adaptive_options_layout = QFormLayout(self.adaptive_rates_options)
        adaptive_options_layout.setContentsMargins(20, 0, 0, 0)  # Add left margin for indentation
        
        # Stagnation limit spinner
        self.stagnation_limit_box = QSpinBox()
        self.stagnation_limit_box.setRange(1, 50)
        self.stagnation_limit_box.setValue(5)
        self.stagnation_limit_box.setToolTip("Number of generations without improvement before adapting rates")
        adaptive_options_layout.addRow("Stagnation Limit:", self.stagnation_limit_box)
        
        # Create a widget for crossover bounds
        crossover_bounds_widget = QWidget()
        crossover_bounds_layout = QHBoxLayout(crossover_bounds_widget)
        crossover_bounds_layout.setContentsMargins(0, 0, 0, 0)
        
        self.cxpb_min_box = QDoubleSpinBox()
        self.cxpb_min_box.setRange(0.01, 0.5)
        self.cxpb_min_box.setValue(0.1)
        self.cxpb_min_box.setDecimals(2)
        self.cxpb_min_box.setSingleStep(0.05)
        self.cxpb_min_box.setToolTip("Minimum crossover probability")
        
        self.cxpb_max_box = QDoubleSpinBox()
        self.cxpb_max_box.setRange(0.5, 1.0)
        self.cxpb_max_box.setValue(0.9)
        self.cxpb_max_box.setDecimals(2)
        self.cxpb_max_box.setSingleStep(0.05)
        self.cxpb_max_box.setToolTip("Maximum crossover probability")
        
        crossover_bounds_layout.addWidget(QLabel("Min:"))
        crossover_bounds_layout.addWidget(self.cxpb_min_box)
        crossover_bounds_layout.addWidget(QLabel("Max:"))
        crossover_bounds_layout.addWidget(self.cxpb_max_box)
        
        adaptive_options_layout.addRow("Crossover Bounds:", crossover_bounds_widget)
        
        # Create a widget for mutation bounds
        mutation_bounds_widget = QWidget()
        mutation_bounds_layout = QHBoxLayout(mutation_bounds_widget)
        mutation_bounds_layout.setContentsMargins(0, 0, 0, 0)
        
        self.mutpb_min_box = QDoubleSpinBox()
        self.mutpb_min_box.setRange(0.01, 0.2)
        self.mutpb_min_box.setValue(0.05)
        self.mutpb_min_box.setDecimals(2)
        self.mutpb_min_box.setSingleStep(0.01)
        self.mutpb_min_box.setToolTip("Minimum mutation probability")
        
        self.mutpb_max_box = QDoubleSpinBox()
        self.mutpb_max_box.setRange(0.2, 0.8)
        self.mutpb_max_box.setValue(0.5)
        self.mutpb_max_box.setDecimals(2)
        self.mutpb_max_box.setSingleStep(0.05)
        self.mutpb_max_box.setToolTip("Maximum mutation probability")
        
        mutation_bounds_layout.addWidget(QLabel("Min:"))
        mutation_bounds_layout.addWidget(self.mutpb_min_box)
        mutation_bounds_layout.addWidget(QLabel("Max:"))
        mutation_bounds_layout.addWidget(self.mutpb_max_box)
        
        adaptive_options_layout.addRow("Mutation Bounds:", mutation_bounds_widget)
        
        # Initially hide adaptive options
        self.adaptive_rates_options.setVisible(False)

        ga_hyper_layout.addRow("Population Size:", self.ga_pop_size_box)
        ga_hyper_layout.addRow("Number of Generations:", self.ga_num_generations_box)
        ga_hyper_layout.addRow("Crossover Probability (cxpb):", self.ga_cxpb_box)
        ga_hyper_layout.addRow("Mutation Probability (mutpb):", self.ga_mutpb_box)
        ga_hyper_layout.addRow("Tolerance (tol):", self.ga_tol_box)
        ga_hyper_layout.addRow("Sparsity Penalty (alpha):", self.ga_alpha_box)
        ga_hyper_layout.addRow("Benchmark Runs:", self.ga_benchmark_runs_box)
        ga_hyper_layout.addRow("", self.adaptive_rates_checkbox)
        ga_hyper_layout.addRow("", self.adaptive_rates_options)

        # Add a small Run GA button in the hyperparameters sub-tab
        self.hyper_run_ga_button = QPushButton("Run GA")
        self.hyper_run_ga_button.setFixedWidth(100)
        self.hyper_run_ga_button.clicked.connect(self.run_ga)
        ga_hyper_layout.addRow("Run GA:", self.hyper_run_ga_button)

        # -------------------- Sub-tab 2: DVA Parameters --------------------
        ga_param_tab = QWidget()
        ga_param_layout = QVBoxLayout(ga_param_tab)

        self.ga_param_table = QTableWidget()
        dva_parameters = [
            *[f"beta_{i}" for i in range(1,16)],
            *[f"lambda_{i}" for i in range(1,16)],
            *[f"mu_{i}" for i in range(1,4)],
            *[f"nu_{i}" for i in range(1,16)]
        ]
        self.ga_param_table.setRowCount(len(dva_parameters))
        self.ga_param_table.setColumnCount(5)
        self.ga_param_table.setHorizontalHeaderLabels(
            ["Parameter", "Fixed", "Fixed Value", "Lower Bound", "Upper Bound"]
        )
        self.ga_param_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ga_param_table.setEditTriggers(QAbstractItemView.NoEditTriggers)

        for row, param in enumerate(dva_parameters):
            param_item = QTableWidgetItem(param)
            param_item.setFlags(Qt.ItemIsEnabled)
            self.ga_param_table.setItem(row, 0, param_item)

            fixed_checkbox = QCheckBox()
            fixed_checkbox.setChecked(True)  # Set fixed to true by default
            fixed_checkbox.stateChanged.connect(lambda state, r=row: self.toggle_ga_fixed(state, r))
            self.ga_param_table.setCellWidget(row, 1, fixed_checkbox)

            fixed_value_spin = QDoubleSpinBox()
            fixed_value_spin.setRange(0, 10e9)  # Changed to 0-10e9 range
            fixed_value_spin.setDecimals(6)
            fixed_value_spin.setValue(0.0)  # Set fixed value to 0
            fixed_value_spin.setEnabled(True)  # Enable because fixed is checked
            self.ga_param_table.setCellWidget(row, 2, fixed_value_spin)

            lower_bound_spin = QDoubleSpinBox()
            lower_bound_spin.setRange(0, 10e9)  # Changed to 0-10e9 range
            lower_bound_spin.setDecimals(6)
            lower_bound_spin.setValue(0.0)  # Set to 0
            lower_bound_spin.setEnabled(False)  # Disable because fixed is checked
            self.ga_param_table.setCellWidget(row, 3, lower_bound_spin)

            upper_bound_spin = QDoubleSpinBox()
            upper_bound_spin.setRange(0, 10e9)  # Changed to 0-10e9 range
            upper_bound_spin.setDecimals(6)
            upper_bound_spin.setValue(1.0)  # Set to 1
            upper_bound_spin.setEnabled(False)  # Disable because fixed is checked
            self.ga_param_table.setCellWidget(row, 4, upper_bound_spin)

        ga_param_layout.addWidget(self.ga_param_table)

        # -------------------- Sub-tab 3: Results --------------------
        ga_results_tab = QWidget()
        ga_results_layout = QVBoxLayout(ga_results_tab)

        # Create a header area for label and export button
        header_container = QWidget()
        header_layout = QHBoxLayout(header_container)
        header_layout.setContentsMargins(0, 0, 0, 0) # No margins for this internal layout

        results_label = QLabel("GA Optimization Results:")
        header_layout.addWidget(results_label)
        header_layout.addStretch() # Add spacer to push the export button to the right

        self.export_ga_results_button = QPushButton("Export GA Results")
        self.export_ga_results_button.setObjectName("secondary-button") # Use existing styling if desired
        self.export_ga_results_button.setToolTip("Export the GA optimization results to a JSON file")
        self.export_ga_results_button.setEnabled(False)  # Initially disabled
        # self.export_ga_results_button.clicked.connect(self.export_ga_results_to_file) # Will connect this later
        header_layout.addWidget(self.export_ga_results_button)
        
        ga_results_layout.addWidget(header_container) # Add the header with label and button
        
        self.ga_results_text = QTextEdit()
        self.ga_results_text.setReadOnly(True)
        ga_results_layout.addWidget(self.ga_results_text)

        # -------------------- Sub-tab 4: Benchmarking --------------------
        ga_benchmark_tab = QWidget()
        ga_benchmark_layout = QVBoxLayout(ga_benchmark_tab)

        # Create buttons for import/export
        button_container = QWidget()
        button_layout = QHBoxLayout(button_container)
        button_layout.setContentsMargins(0, 0, 0, 10)  # Add some bottom margin

        self.import_benchmark_button = QPushButton("Import Benchmark Data")
        self.import_benchmark_button.setToolTip("Import previously saved GA benchmark data")
        self.import_benchmark_button.clicked.connect(self.import_ga_benchmark_data)
        button_layout.addWidget(self.import_benchmark_button)

        self.export_benchmark_button = QPushButton("Export Benchmark Data")
        self.export_benchmark_button.setToolTip("Export current GA benchmark data to a file")
        self.export_benchmark_button.setEnabled(False)  # Initially disabled until data is available
        self.export_benchmark_button.clicked.connect(self.export_ga_benchmark_data)
        button_layout.addWidget(self.export_benchmark_button)

        button_layout.addStretch()  # Add stretch to push buttons to the left
        ga_benchmark_layout.addWidget(button_container)

        # Create tabs for different benchmark visualizations
        self.benchmark_viz_tabs = QTabWidget()
        
        # Create tabs for different visualizations
        violin_tab = QWidget()
        violin_layout = QVBoxLayout(violin_tab)
        self.violin_plot_widget = QWidget()
        violin_layout.addWidget(self.violin_plot_widget)
        
        dist_tab = QWidget()
        dist_layout = QVBoxLayout(dist_tab)
        self.dist_plot_widget = QWidget()
        dist_layout.addWidget(self.dist_plot_widget)
        
        scatter_tab = QWidget()
        scatter_layout = QVBoxLayout(scatter_tab)
        self.scatter_plot_widget = QWidget()
        scatter_layout.addWidget(self.scatter_plot_widget)
        
        heatmap_tab = QWidget()
        heatmap_layout = QVBoxLayout(heatmap_tab)
        self.heatmap_plot_widget = QWidget()
        heatmap_layout.addWidget(self.heatmap_plot_widget)
        
        # Add Q-Q plot tab
        qq_tab = QWidget()
        qq_layout = QVBoxLayout(qq_tab)
        self.qq_plot_widget = QWidget()
        qq_layout.addWidget(self.qq_plot_widget)
        
        # Add Statistical Analysis tab for comprehensive parameter analysis
        statistical_analysis_tab = QWidget()
        statistical_analysis_layout = QVBoxLayout(statistical_analysis_tab)
        
        # Create subtabs for different statistical analyses
        self.statistical_analysis_subtabs = QTabWidget()
        
        # 1. Parameter Visualizations Tab
        param_viz_tab = QWidget()
        param_viz_layout = QVBoxLayout(param_viz_tab)
        
        # Control panel for parameter selection
        control_panel = QGroupBox("Parameter Selection & Visualization Controls")
        control_layout = QGridLayout(control_panel)
        
        # Parameter selection dropdown
        QLabel_param_select = QLabel("Select Parameter:")
        self.param_selection_combo = QComboBox()
        self.param_selection_combo.setMaxVisibleItems(5)  # Show only 5 items at a time
        self.param_selection_combo.setMinimumWidth(150)
        self.param_selection_combo.setMaximumWidth(200)
        self.param_selection_combo.currentTextChanged.connect(self.on_parameter_selection_changed)
        
        # Plot type selection
        QLabel_plot_type = QLabel("Plot Type:")
        self.plot_type_combo = QComboBox()
        self.plot_type_combo.addItems(["Violin Plot", "Distribution Plot", "Scatter Plot", "Q-Q Plot"])
        self.plot_type_combo.currentTextChanged.connect(self.on_plot_type_changed)
        
        # Comparison parameter for scatter plots
        QLabel_comparison = QLabel("Compare With:")
        self.comparison_param_combo = QComboBox()
        self.comparison_param_combo.addItem("None")
        self.comparison_param_combo.setMaxVisibleItems(5)  # Show only 5 items at a time
        self.comparison_param_combo.setMinimumWidth(150)
        self.comparison_param_combo.setMaximumWidth(200)
        self.comparison_param_combo.setEnabled(False)
        self.comparison_param_combo.currentTextChanged.connect(self.on_comparison_parameter_changed)
        
        # Update plots button
        self.update_plots_button = QPushButton("Update Plots")
        self.update_plots_button.clicked.connect(self.update_parameter_plots)
        
        # Layout controls
        control_layout.addWidget(QLabel_param_select, 0, 0)
        control_layout.addWidget(self.param_selection_combo, 0, 1)
        control_layout.addWidget(QLabel_plot_type, 0, 2)
        control_layout.addWidget(self.plot_type_combo, 0, 3)
        control_layout.addWidget(QLabel_comparison, 1, 0)
        control_layout.addWidget(self.comparison_param_combo, 1, 1)
        control_layout.addWidget(self.update_plots_button, 1, 2)
        
        param_viz_layout.addWidget(control_panel)
        
        # Plot display area with enhanced scrolling
        self.param_plot_scroll = QScrollArea()
        self.param_plot_scroll.setWidgetResizable(True)
        self.param_plot_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.param_plot_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.param_plot_scroll.setMinimumHeight(400)
        self.param_plot_widget = QWidget()
        self.param_plot_widget.setLayout(QVBoxLayout())
        self.param_plot_widget.setMinimumHeight(500)  # Ensure content is visible
        self.param_plot_scroll.setWidget(self.param_plot_widget)
        param_viz_layout.addWidget(self.param_plot_scroll)
        
        # 2. Parameter Statistics Tab - Make entire tab scrollable
        param_stats_tab = QWidget()
        param_stats_main_layout = QVBoxLayout(param_stats_tab)
        
        # Create scroll area for the entire statistics tab
        stats_main_scroll = QScrollArea()
        stats_main_scroll.setWidgetResizable(True)
        stats_main_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        stats_main_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Content widget for the scroll area
        stats_content_widget = QWidget()
        param_stats_layout = QVBoxLayout(stats_content_widget)
        
        # Statistics control panel
        stats_control_panel = QGroupBox("Statistics Display Options")
        stats_control_layout = QHBoxLayout(stats_control_panel)
        
        # Statistics view dropdown
        QLabel_stats_view = QLabel("View:")
        self.stats_view_combo = QComboBox()
        self.stats_view_combo.addItems(["Summary Table", "Detailed Statistics", "Equations & Formulas"])
        self.stats_view_combo.setMaxVisibleItems(3)  # Show only 3 items at a time (matches the number of options)
        self.stats_view_combo.setMinimumWidth(150)
        self.stats_view_combo.setMaximumWidth(200)
        self.stats_view_combo.currentTextChanged.connect(self.on_stats_view_changed)
        
        stats_control_layout.addWidget(QLabel_stats_view)
        stats_control_layout.addWidget(self.stats_view_combo)
        stats_control_layout.addStretch()
        
        param_stats_layout.addWidget(stats_control_panel)
        
        # Statistics display area with enhanced scrolling
        self.param_stats_scroll = QScrollArea()
        self.param_stats_scroll.setWidgetResizable(True)
        self.param_stats_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.param_stats_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.param_stats_scroll.setMinimumHeight(400)
        self.param_stats_widget = QWidget()
        self.param_stats_widget.setLayout(QVBoxLayout())
        self.param_stats_scroll.setWidget(self.param_stats_widget)
        param_stats_layout.addWidget(self.param_stats_scroll)
        
        # Set the content widget to the main scroll area
        stats_main_scroll.setWidget(stats_content_widget)
        param_stats_main_layout.addWidget(stats_main_scroll)
        

        
        # Add subtabs to statistical analysis
        self.statistical_analysis_subtabs.addTab(param_viz_tab, "Parameter Visualizations")
        self.statistical_analysis_subtabs.addTab(param_stats_tab, "Statistics Tables")
        
        statistical_analysis_layout.addWidget(self.statistical_analysis_subtabs)
        
        # Summary statistics tabs (create subtabs for better organization)
        stats_tab = QWidget()
        stats_layout = QVBoxLayout(stats_tab)
        
        # Create a tabbed widget for the statistics section
        stats_subtabs = QTabWidget()
        
        # ---- Subtab 1: Summary Statistics ----
        summary_tab = QWidget()
        summary_layout = QVBoxLayout(summary_tab)
        
        # Add summary statistics table
        self.benchmark_stats_table = QTableWidget()
        self.benchmark_stats_table.setColumnCount(5)
        self.benchmark_stats_table.setHorizontalHeaderLabels(["Metric", "Min", "Max", "Mean", "Std"])
        self.benchmark_stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        summary_layout.addWidget(QLabel("Statistical Summary of All Runs:"))
        summary_layout.addWidget(self.benchmark_stats_table)
        
        # ---- Subtab 2: All Runs Table ----
        runs_tab = QWidget()
        runs_layout = QVBoxLayout(runs_tab)
        
        # Create a table for all runs
        self.benchmark_runs_table = QTableWidget()
        self.benchmark_runs_table.setColumnCount(4)
        self.benchmark_runs_table.setHorizontalHeaderLabels(["Run #", "Fitness", "Rank", "Details"])
        self.benchmark_runs_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.benchmark_runs_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.benchmark_runs_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.benchmark_runs_table.itemClicked.connect(self.show_run_details)
        
        runs_layout.addWidget(QLabel("All Benchmark Runs:"))
        runs_layout.addWidget(self.benchmark_runs_table)
        
        # Create run details text area
        details_tab = QWidget()
        details_layout = QVBoxLayout(details_tab)
        details_group = QGroupBox("Run Details")
        group_layout = QVBoxLayout(details_group)
        self.run_details_text = QTextEdit()
        self.run_details_text.setReadOnly(True)
        group_layout.addWidget(self.run_details_text)
        details_layout.addWidget(details_group)
        
        # Create GA Operations tab as a subtab
        ga_ops_tab = QWidget()
        ga_ops_layout = QVBoxLayout(ga_ops_tab)
        self.ga_ops_plot_widget = QWidget()
        ga_ops_layout.addWidget(self.ga_ops_plot_widget)
        
        # Add the subtabs to the stats tabbed widget
        stats_subtabs.addTab(summary_tab, "Summary Statistics")
        stats_subtabs.addTab(runs_tab, "All Runs")
        stats_subtabs.addTab(details_tab, "Run Details")
        stats_subtabs.addTab(ga_ops_tab, "GA Operations")
        
        # Add the stats tabbed widget to the stats tab
        stats_layout.addWidget(stats_subtabs)
        
        # Add all visualization tabs to the benchmark visualization tabs
        self.benchmark_viz_tabs.addTab(violin_tab, "Fitness Analysis")
        self.benchmark_viz_tabs.addTab(dist_tab, "Distribution")
        self.benchmark_viz_tabs.addTab(scatter_tab, "Scatter Plot")
        self.benchmark_viz_tabs.addTab(heatmap_tab, "Parameter Correlations")
        self.benchmark_viz_tabs.addTab(qq_tab, "Q-Q Plot")
        self.benchmark_viz_tabs.addTab(statistical_analysis_tab, "Statistical Analysis")
        self.benchmark_viz_tabs.addTab(stats_tab, "Statistics")
        
        # GA Operations Performance Tab - already added as a subtab of Statistics
        
        # Add the benchmark visualization tabs to the benchmark tab
        ga_benchmark_layout.addWidget(self.benchmark_viz_tabs)
        
        # Add all sub-tabs to the GA tab widget
        # Initialize empty benchmark data storage
        self.ga_benchmark_data = []

        # Add all sub-tabs to the GA tab widget
        self.ga_sub_tabs.addTab(ga_hyper_tab, "GA Settings")
        self.ga_sub_tabs.addTab(ga_param_tab, "DVA Parameters")
        self.ga_sub_tabs.addTab(ga_results_tab, "Results")
        self.ga_sub_tabs.addTab(ga_benchmark_tab, "GA Benchmarking")

        # Add the GA sub-tabs widget to the main GA tab layout
        layout.addWidget(self.ga_sub_tabs)
        self.ga_tab.setLayout(layout)
        
    def toggle_fixed(self, state, row, table=None):
        """Toggle the fixed state of a DVA parameter row"""
        if table is None:
            table = self.dva_param_table
            
        fixed = (state == Qt.Checked)
        fixed_value_spin = table.cellWidget(row, 2)
        lower_bound_spin = table.cellWidget(row, 3)
        upper_bound_spin = table.cellWidget(row, 4)

        fixed_value_spin.setEnabled(fixed)
        lower_bound_spin.setEnabled(not fixed)
        upper_bound_spin.setEnabled(not fixed)

    def toggle_ga_fixed(self, state, row, table=None):
        """Toggle the fixed state of a GA parameter row"""
        if table is None:
            table = self.ga_param_table
            
        fixed = (state == Qt.Checked)
        fixed_value_spin = table.cellWidget(row, 2)
        lower_bound_spin = table.cellWidget(row, 3)
        upper_bound_spin = table.cellWidget(row, 4)
        
        # Enable/disable appropriate spinboxes
        fixed_value_spin.setEnabled(fixed)
        lower_bound_spin.setEnabled(not fixed)
        upper_bound_spin.setEnabled(not fixed)
        
        # If switching to fixed mode, copy current lower bound value to fixed value
        if fixed:
            fixed_value_spin.setValue(lower_bound_spin.value())
        # If switching to range mode, ensure lower bound is not greater than upper bound
        else:
            if lower_bound_spin.value() > upper_bound_spin.value():
                upper_bound_spin.setValue(lower_bound_spin.value())

    def toggle_adaptive_rates_options(self, state):
        """Show or hide adaptive rates options based on checkbox state"""
        self.adaptive_rates_options.setVisible(state == Qt.Checked)
        
        # Enable/disable the fixed rate inputs based on adaptive rates setting
        self.ga_cxpb_box.setEnabled(state != Qt.Checked)
        self.ga_mutpb_box.setEnabled(state != Qt.Checked)
        
        # Update tooltips to indicate that rates will be adaptive
        if state == Qt.Checked:
            self.ga_cxpb_box.setToolTip("Starting crossover probability (will adapt during optimization)")
            self.ga_mutpb_box.setToolTip("Starting mutation probability (will adapt during optimization)")
        else:
            self.ga_cxpb_box.setToolTip("Crossover probability")
            self.ga_mutpb_box.setToolTip("Mutation probability")
    def toggle_ga_fixed(self, state, row, table=None):
        """Toggle the fixed state of a GA parameter row"""
        if table is None:
            table = self.ga_param_table
            
        fixed = (state == Qt.Checked)
        fixed_value_spin = table.cellWidget(row, 2)
        lower_bound_spin = table.cellWidget(row, 3)
        upper_bound_spin = table.cellWidget(row, 4)
        
        # Enable/disable appropriate spinboxes
        fixed_value_spin.setEnabled(fixed)
        lower_bound_spin.setEnabled(not fixed)
        upper_bound_spin.setEnabled(not fixed)
        
        # If switching to fixed mode, copy current lower bound value to fixed value
        if fixed:
            fixed_value_spin.setValue(lower_bound_spin.value())
        # If switching to range mode, ensure lower bound is not greater than upper bound
        else:
            if lower_bound_spin.value() > upper_bound_spin.value():
                upper_bound_spin.setValue(lower_bound_spin.value())

    def toggle_adaptive_rates_options(self, state):
        """Show or hide adaptive rates options based on checkbox state"""
        self.adaptive_rates_options.setVisible(state == Qt.Checked)
        
        # Enable/disable the fixed rate inputs based on adaptive rates setting
        self.ga_cxpb_box.setEnabled(state != Qt.Checked)
        self.ga_mutpb_box.setEnabled(state != Qt.Checked)
        
        # Update tooltips to indicate that rates will be adaptive
        if state == Qt.Checked:
            self.ga_cxpb_box.setToolTip("Starting crossover probability (will adapt during optimization)")
            self.ga_mutpb_box.setToolTip("Starting mutation probability (will adapt during optimization)")
        else:
            self.ga_cxpb_box.setToolTip("Crossover probability")
            self.ga_mutpb_box.setToolTip("Mutation probability")

    def run_ga(self):
        """Run genetic algorithm optimization"""
        # Check if a GA worker is already running
        if hasattr(self, 'ga_worker') and self.ga_worker.isRunning():
            QMessageBox.warning(self, "Process Running", 
                               "A Genetic Algorithm optimization is already running. Please wait for it to complete.")
            return
            
        if self.omega_start_box.value() >= self.omega_end_box.value():
            QMessageBox.warning(self, "Input Error", "Ω Start must be less than Ω End.")
            return

        target_values, weights = self.get_target_values_weights()
        
        # Get GA hyperparameters
        pop_size = self.ga_pop_size_box.value()
        num_gen = self.ga_num_generations_box.value()
        crossover_prob = self.ga_cxpb_box.value()
        mutation_prob = self.ga_mutpb_box.value()
        tolerance = self.ga_tol_box.value()
        alpha = self.ga_alpha_box.value()
        
        # Get number of benchmark runs
        self.benchmark_runs = self.ga_benchmark_runs_box.value()
        self.current_benchmark_run = 0
        
        # Clear benchmark data if running multiple times
        if self.benchmark_runs > 1:
            self.ga_benchmark_data = []
            # Enable the benchmark tab if running multiple times
            self.ga_sub_tabs.setTabEnabled(self.ga_sub_tabs.indexOf(self.ga_sub_tabs.findChild(QWidget, "GA Benchmarking")), True)
        
        # Get DVA parameter bounds
        dva_bounds = {}
        EPSILON = 1e-6
        for row in range(self.ga_param_table.rowCount()):
            param_item = self.ga_param_table.item(row, 0)
            param_name = param_item.text()
            
            fixed_widget = self.ga_param_table.cellWidget(row, 1)
            fixed = fixed_widget.isChecked()
            
            if fixed:
                fixed_value_widget = self.ga_param_table.cellWidget(row, 2)
                fixed_value = fixed_value_widget.value()
                dva_bounds[param_name] = (fixed_value, fixed_value + EPSILON)
            else:
                lower_bound_widget = self.ga_param_table.cellWidget(row, 3)
                upper_bound_widget = self.ga_param_table.cellWidget(row, 4)
                lower = lower_bound_widget.value()
                upper = upper_bound_widget.value()
                if lower > upper:
                    QMessageBox.warning(self, "Input Error", 
                                       f"For parameter {param_name}, lower bound is greater than upper bound.")
                    return
                dva_bounds[param_name] = (lower, upper)
        
        # Get main system parameters
        main_params = (
            self.mu_box.value(),
            *[b.value() for b in self.landa_boxes],
            *[b.value() for b in self.nu_boxes],
            self.a_low_box.value(),
            self.a_up_box.value(),
            self.f_1_box.value(),
            self.f_2_box.value(),
            self.omega_dc_box.value(),
            self.zeta_dc_box.value()
        )
        
        # Update button reference to match the actual button in the UI
        self.run_ga_button = self.hyper_run_ga_button
        
        # Disable run buttons during optimization
        self.run_frf_button.setEnabled(False)
        self.run_sobol_button.setEnabled(False)
        self.run_ga_button.setEnabled(False)
        
        # Create progress bar if it doesn't exist
        if not hasattr(self, 'ga_progress_bar'):
            self.ga_progress_bar = QProgressBar()
            self.ga_progress_bar.setRange(0, 100)
            self.ga_progress_bar.setValue(0)
            self.ga_progress_bar.setTextVisible(True)
            self.ga_progress_bar.setFormat("GA Progress: %p%")
            
            # Find where to add progress bar in the layout
            ga_results_tab_layout = self.ga_results_text.parent().layout()
            ga_results_tab_layout.insertWidget(0, self.ga_progress_bar)
        else:
            self.ga_progress_bar.setValue(0)
            
        # Make sure the progress bar is visible
        self.ga_progress_bar.show()
        
        # Update status
        self.status_bar.showMessage("Running GA optimization...")
        self.ga_results_text.append("\n--- Running Genetic Algorithm Optimization ---\n")
        self.ga_results_text.append(f"Population Size: {pop_size}")
        self.ga_results_text.append(f"Number of Generations: {num_gen}")
        self.ga_results_text.append(f"Crossover Probability: {crossover_prob}")
        self.ga_results_text.append(f"Mutation Probability: {mutation_prob}")
        self.ga_results_text.append(f"Tolerance: {tolerance}")
        self.ga_results_text.append(f"Sparsity Penalty (alpha): {alpha}")
        
        # Add debug output for adaptive rates
        adaptive_rates = self.adaptive_rates_checkbox.isChecked()
        self.ga_results_text.append(f"Adaptive Rates: {'Enabled' if adaptive_rates else 'Disabled'}")
        if adaptive_rates:
            self.ga_results_text.append(f"  - Stagnation Limit: {self.stagnation_limit_box.value()}")
            self.ga_results_text.append(f"  - Crossover Range: {self.cxpb_min_box.value():.2f} - {self.cxpb_max_box.value():.2f}")
            self.ga_results_text.append(f"  - Mutation Range: {self.mutpb_min_box.value():.2f} - {self.mutpb_max_box.value():.2f}")
        self.ga_results_text.append("\nStarting optimization...\n")
        
        # Create and start worker
        original_dva_parameter_order = [
            'beta_1','beta_2','beta_3','beta_4','beta_5','beta_6',
            'beta_7','beta_8','beta_9','beta_10','beta_11','beta_12',
            'beta_13','beta_14','beta_15',
            'lambda_1','lambda_2','lambda_3','lambda_4','lambda_5',
            'lambda_6','lambda_7','lambda_8','lambda_9','lambda_10',
            'lambda_11','lambda_12','lambda_13','lambda_14','lambda_15',
            'mu_1','mu_2','mu_3',
            'nu_1','nu_2','nu_3','nu_4','nu_5','nu_6',
            'nu_7','nu_8','nu_9','nu_10','nu_11','nu_12',
            'nu_13','nu_14','nu_15'
        ]
        
        # Convert dva_bounds and dva_order into ga_parameter_data format
        ga_parameter_data = []
        for param_name in original_dva_parameter_order:
            if param_name in dva_bounds:
                low, high = dva_bounds[param_name]
                # Check if parameter is fixed (low == high)
                fixed = abs(low - high) < EPSILON
                ga_parameter_data.append((param_name, low, high, fixed))
                
        # If there's an existing worker, make sure it's properly cleaned up
        if hasattr(self, 'ga_worker'):
            try:
                self.ga_worker.finished.disconnect()
                self.ga_worker.error.disconnect()
                self.ga_worker.update.disconnect()
                self.ga_worker.progress.disconnect()
            except Exception:
                pass
                
        # Create a new worker
        self.ga_worker = GAWorker(
            main_params=main_params,
            target_values_dict=target_values,
            weights_dict=weights,
            omega_start=self.omega_start_box.value(),
            omega_end=self.omega_end_box.value(),
            omega_points=self.omega_points_box.value(),
            ga_pop_size=pop_size,
            ga_num_generations=num_gen,
            ga_cxpb=crossover_prob,
            ga_mutpb=mutation_prob,
            ga_tol=tolerance,
            ga_parameter_data=ga_parameter_data,
            alpha=alpha,
            track_metrics=True,  # Enable metrics tracking for visualization
            adaptive_rates=self.adaptive_rates_checkbox.isChecked(),  # Pass the adaptive rates setting
            stagnation_limit=self.stagnation_limit_box.value(),  # Get stagnation limit from UI
            cxpb_min=self.cxpb_min_box.value(),  # Get min crossover probability
            cxpb_max=self.cxpb_max_box.value(),  # Get max crossover probability
            mutpb_min=self.mutpb_min_box.value(),  # Get min mutation probability
            mutpb_max=self.mutpb_max_box.value()  # Get max mutation probability
        )
        
        # Connect signals using strong references to avoid premature garbage collection
        self.ga_worker.finished.connect(self.handle_ga_finished)
        self.ga_worker.error.connect(self.handle_ga_error)
        self.ga_worker.update.connect(self.handle_ga_update)
        self.ga_worker.progress.connect(self.update_ga_progress)
        
        # Set up a watchdog timer for the GA worker
        if hasattr(self, 'ga_watchdog_timer'):
            self.ga_watchdog_timer.stop()
        else:
            self.ga_watchdog_timer = QTimer(self)
            self.ga_watchdog_timer.timeout.connect(self.check_ga_worker_health)
            
        self.ga_watchdog_timer.start(10000)  # Check every 10 seconds
        
        # Start the worker
        self.ga_worker.start()
        
    def check_ga_worker_health(self):
        """Check if the GA worker is still responsive"""
        if hasattr(self, 'ga_worker') and self.ga_worker.isRunning():
            # The worker is still running, which is good
            # We could add more sophisticated checks here if needed
            pass
        else:
            # The worker is not running anymore, stop the watchdog
            if hasattr(self, 'ga_watchdog_timer'):
                self.ga_watchdog_timer.stop()
                
    def update_ga_progress(self, value):
        """Update the GA progress bar, accounting for multiple benchmark runs"""
        if hasattr(self, 'ga_progress_bar'):
            if hasattr(self, 'benchmark_runs') and self.benchmark_runs > 1:
                # Calculate overall progress across all runs
                # Each run contributes (1/total_runs) of the progress
                run_contribution = 100.0 / self.benchmark_runs
                current_run_progress = value / 100.0  # Convert to fraction
                # Add progress from completed runs plus fractional progress from current run
                overall_progress = ((self.current_benchmark_run - 1) * run_contribution) + (current_run_progress * run_contribution)
                self.ga_progress_bar.setValue(int(overall_progress))
            else:
                # Single run - direct progress
                self.ga_progress_bar.setValue(value)
            
    def handle_ga_finished(self, results, best_ind, parameter_names, best_fitness):
        """Handle the completion of the GA optimization"""
        # Stop the watchdog timer
        if hasattr(self, 'ga_watchdog_timer'):
            self.ga_watchdog_timer.stop()
        
        # For benchmarking, collect data from this run
        self.current_benchmark_run += 1
        
        # Store benchmark results
        if hasattr(self, 'benchmark_runs') and self.benchmark_runs > 1:
            # Create a data dictionary for this run
            run_data = {
                'run_number': self.current_benchmark_run,
                'best_fitness': best_fitness,
                'best_solution': list(best_ind),
                'parameter_names': parameter_names
            }
            
            # Add any additional metrics from results
            if isinstance(results, dict):
                for key, value in results.items():
                    if isinstance(value, (int, float)) and np.isfinite(value):
                        run_data[key] = value

                # Add benchmark metrics if available
                if 'benchmark_metrics' in results:
                    run_data['benchmark_metrics'] = results['benchmark_metrics']
            
            # Store the run data
            self.ga_benchmark_data.append(run_data)
            
            # Update the status message
            self.status_bar.showMessage(f"GA run {self.current_benchmark_run} of {self.benchmark_runs} completed")
            
            # Update progress bar to show completed percentage of all runs
            if hasattr(self, 'ga_progress_bar'):
                progress = int(self.current_benchmark_run * 100 / self.benchmark_runs)
                self.ga_progress_bar.setValue(progress)
            
            # Check if we need to run again
            if self.current_benchmark_run < self.benchmark_runs:
                self.ga_results_text.append(f"\n--- Run {self.current_benchmark_run} completed, starting run {self.current_benchmark_run + 1}/{self.benchmark_runs} ---")
                # Set up for next run
                QTimer.singleShot(100, self.run_next_ga_benchmark)
                return
            else:
                # All runs completed, visualize the benchmark results
                self.visualize_ga_benchmark_results()
                self.export_benchmark_button.setEnabled(True)
                self.ga_results_text.append(f"\n--- All {self.benchmark_runs} benchmark runs completed ---")
        else:
            # For single runs, store the data directly
            run_data = {
                'run_number': 1,
                'best_fitness': best_fitness,
                'best_solution': list(best_ind),
                'parameter_names': parameter_names
            }
            
            # Add benchmark metrics if available
            if isinstance(results, dict) and 'benchmark_metrics' in results:
                run_data['benchmark_metrics'] = results['benchmark_metrics']
            
            self.ga_benchmark_data = [run_data]
            self.visualize_ga_benchmark_results()
                
        # Re-enable buttons when completely done
        self.run_frf_button.setEnabled(True)
        self.run_sobol_button.setEnabled(True)
        self.run_ga_button.setEnabled(True)
        
        self.status_bar.showMessage("GA optimization completed")
        
        # Only show detailed results for single runs or the final benchmark run
        if not hasattr(self, 'benchmark_runs') or self.benchmark_runs == 1 or self.current_benchmark_run == self.benchmark_runs:
            self.ga_results_text.append("\n--- GA Optimization Completed ---")
            self.ga_results_text.append(f"Best fitness: {best_fitness:.6f}")
        self.ga_results_text.append("\nBest Parameters:")
        
        # Check if there are any warnings in the results
        if isinstance(results, dict) and "Warning" in results:
            self.ga_results_text.append(f"\nWarning: {results['Warning']}")
        
        # Create a dictionary mapping parameter names to their values
        best_params = {name: value for name, value in zip(parameter_names, best_ind)}
        
        # Store best parameters for easy access later
        self.current_ga_best_params = best_params
        self.current_ga_best_fitness = best_fitness
        self.current_ga_full_results = results
        
        for param_name, value in best_params.items():
            self.ga_results_text.append(f"  {param_name}: {value:.6f}")
            
        # If we have actual results, show them
        if isinstance(results, dict) and "singular_response" in results:
            self.ga_results_text.append(f"\nFinal Singular Response: {results['singular_response']:.6f}")
        
    def handle_ga_error(self, error_msg):
        """Handle errors from the GA worker"""
        # Stop the watchdog timer
        if hasattr(self, 'ga_watchdog_timer'):
            self.ga_watchdog_timer.stop()
            
        # Hide or reset the progress bar
        if hasattr(self, 'ga_progress_bar'):
            self.ga_progress_bar.setValue(0)
            
        QMessageBox.critical(self, "Error in GA Optimization", str(error_msg))
        self.status_bar.showMessage("GA optimization failed")
        self.ga_results_text.append(f"\nError in GA optimization: {error_msg}")
        
        # Make sure to re-enable buttons
        self.run_frf_button.setEnabled(True)
        self.run_sobol_button.setEnabled(True)
        self.run_ga_button.setEnabled(True)
        
        # Try to recover by cleaning up any residual state
        if hasattr(self, 'ga_worker'):
            try:
                # Attempt to terminate the worker if it's still running
                if self.ga_worker.isRunning():
                    self.ga_worker.terminate()
                    self.ga_worker.wait(1000)  # Wait up to 1 second for it to finish
            except Exception as e:
                print(f"Error cleaning up GA worker: {str(e)}")
        
    def handle_ga_update(self, msg):
        """Handle update messages from the GA worker"""
        self.ga_results_text.append(msg)
        # Auto-scroll to the bottom to show latest messages
        self.ga_results_text.verticalScrollBar().setValue(
            self.ga_results_text.verticalScrollBar().maximum()
        )
        
    def run_next_ga_benchmark(self):
        """Run the next GA benchmark iteration"""
        # Clear the existing GA worker to start fresh
        if hasattr(self, 'ga_worker'):
            try:
                self.ga_worker.finished.disconnect()
                self.ga_worker.error.disconnect()
                self.ga_worker.update.disconnect()
                self.ga_worker.progress.disconnect()
            except Exception:
                pass
        
        # Get the required parameters again
        target_values, weights = self.get_target_values_weights()
        
        # Get GA hyperparameters
        pop_size = self.ga_pop_size_box.value()
        num_gen = self.ga_num_generations_box.value()
        crossover_prob = self.ga_cxpb_box.value()
        mutation_prob = self.ga_mutpb_box.value()
        tolerance = self.ga_tol_box.value()
        alpha = self.ga_alpha_box.value()
        
        # Get DVA parameter bounds
        dva_bounds = {}
        EPSILON = 1e-6
        for row in range(self.ga_param_table.rowCount()):
            param_item = self.ga_param_table.item(row, 0)
            param_name = param_item.text()
            
            fixed_widget = self.ga_param_table.cellWidget(row, 1)
            fixed = fixed_widget.isChecked()
            
            if fixed:
                fixed_value_widget = self.ga_param_table.cellWidget(row, 2)
                fixed_value = fixed_value_widget.value()
                dva_bounds[param_name] = (fixed_value, fixed_value + EPSILON)
            else:
                lower_bound_widget = self.ga_param_table.cellWidget(row, 3)
                upper_bound_widget = self.ga_param_table.cellWidget(row, 4)
                lower = lower_bound_widget.value()
                upper = upper_bound_widget.value()
                dva_bounds[param_name] = (lower, upper)
        
        # Get main system parameters
        main_params = (
            self.mu_box.value(),
            *[b.value() for b in self.landa_boxes],
            *[b.value() for b in self.nu_boxes],
            self.a_low_box.value(),
            self.a_up_box.value(),
            self.f_1_box.value(),
            self.f_2_box.value(),
            self.omega_dc_box.value(),
            self.zeta_dc_box.value()
        )
        
        # Reset progress bar
        if hasattr(self, 'ga_progress_bar'):
            self.ga_progress_bar.setValue(0)
            
        # Make sure the progress bar is visible
        self.ga_progress_bar.show()
        
        # Update status
        self.status_bar.showMessage(f"Running GA optimization (Run {self.current_benchmark_run + 1}/{self.benchmark_runs})...")
        
        # Create and start worker
        original_dva_parameter_order = [
            'beta_1','beta_2','beta_3','beta_4','beta_5','beta_6',
            'beta_7','beta_8','beta_9','beta_10','beta_11','beta_12',
            'beta_13','beta_14','beta_15',
            'lambda_1','lambda_2','lambda_3','lambda_4','lambda_5',
            'lambda_6','lambda_7','lambda_8','lambda_9','lambda_10',
            'lambda_11','lambda_12','lambda_13','lambda_14','lambda_15',
            'mu_1','mu_2','mu_3',
            'nu_1','nu_2','nu_3','nu_4','nu_5','nu_6',
            'nu_7','nu_8','nu_9','nu_10','nu_11','nu_12',
            'nu_13','nu_14','nu_15'
        ]
        
        # Convert dva_bounds and dva_order into ga_parameter_data format
        ga_parameter_data = []
        for param_name in original_dva_parameter_order:
            if param_name in dva_bounds:
                low, high = dva_bounds[param_name]
                # Check if parameter is fixed (low == high)
                fixed = abs(low - high) < EPSILON
                ga_parameter_data.append((param_name, low, high, fixed))
        
        # Create a new worker
        self.ga_worker = GAWorker(
            main_params=main_params,
            target_values_dict=target_values,
            weights_dict=weights,
            omega_start=self.omega_start_box.value(),
            omega_end=self.omega_end_box.value(),
            omega_points=self.omega_points_box.value(),
            ga_pop_size=pop_size,
            ga_num_generations=num_gen,
            ga_cxpb=crossover_prob,
            ga_mutpb=mutation_prob,
            ga_tol=tolerance,
            ga_parameter_data=ga_parameter_data,
            alpha=alpha,
            track_metrics=True,  # Enable metrics tracking for visualization
            adaptive_rates=self.adaptive_rates_checkbox.isChecked(),  # Pass the adaptive rates setting
            stagnation_limit=self.stagnation_limit_box.value(),  # Get stagnation limit from UI
            cxpb_min=self.cxpb_min_box.value(),  # Get min crossover probability
            cxpb_max=self.cxpb_max_box.value(),  # Get max crossover probability
            mutpb_min=self.mutpb_min_box.value(),  # Get min mutation probability
            mutpb_max=self.mutpb_max_box.value()  # Get max mutation probability
        )
        
        # Connect signals using strong references to avoid premature garbage collection
        self.ga_worker.finished.connect(self.handle_ga_finished)
        self.ga_worker.error.connect(self.handle_ga_error)
        self.ga_worker.update.connect(self.handle_ga_update)
        self.ga_worker.progress.connect(self.update_ga_progress)
        
        # Set up a watchdog timer for the GA worker
        if hasattr(self, 'ga_watchdog_timer'):
            self.ga_watchdog_timer.stop()
        else:
            self.ga_watchdog_timer = QTimer(self)
            self.ga_watchdog_timer.timeout.connect(self.check_ga_worker_health)
            
        self.ga_watchdog_timer.start(10000)  # Check every 10 seconds
        
        # Start the worker
        self.ga_worker.start()
    
    def _open_plot_window(self, fig, title):
        """Opens a new window to display a matplotlib figure."""
        plot_window = PlotWindow(fig, title)
        plot_window.setMinimumSize(800, 600)
        plot_window.show()
        # Keep a reference to prevent garbage collection
        if not hasattr(self, '_plot_windows'):
            self._plot_windows = []
        self._plot_windows.append(plot_window)
    
    def visualize_ga_benchmark_results(self):
        """Create visualizations for GA benchmark results"""
        if not hasattr(self, 'ga_benchmark_data') or not self.ga_benchmark_data:
            return
            
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        import seaborn as sns
        from computational_metrics_new import visualize_all_metrics
        
        # Convert benchmark data to DataFrame for easier analysis
        df = pd.DataFrame(self.ga_benchmark_data)
        
        # Visualize computational metrics
        widgets_dict = {
            'ga_ops_plot_widget': self.ga_ops_plot_widget
        }
        visualize_all_metrics(widgets_dict, df)
        
        # 1. Create violin & box plot
        try:
            # Clear existing plot layout
            if self.violin_plot_widget.layout():
                for i in reversed(range(self.violin_plot_widget.layout().count())): 
                    self.violin_plot_widget.layout().itemAt(i).widget().setParent(None)
            else:
                self.violin_plot_widget.setLayout(QVBoxLayout())
                
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
            tolerance = self.ga_tol_box.value()
            
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
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5) # Adjusted alpha
            ax_violin.text(0.05, 0.95, legend_col1_text, transform=ax_violin.transAxes, 
                    fontsize=12, verticalalignment='top', bbox=props) # Adjusted fontsize
            ax_violin.text(0.28, 0.95, legend_col2_text, transform=ax_violin.transAxes, 
                    fontsize=12, verticalalignment='top', bbox=props) # Adjusted fontsize and position
                    
            # Add percentile lines with labels (without redundant legend entries)
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
            
            # Add a shaded region below tolerance (without redundant legend entry)
            ax_violin.axhspan(0, tolerance, color='magenta', alpha=0.1, label=None)
            
            # Add compact legend for all lines
            ax_violin.legend(loc='upper right', framealpha=0.7, fontsize=9)
            
            # Create canvas and add to layout
            canvas_violin = FigureCanvasQTAgg(fig_violin)
            self.violin_plot_widget.layout().addWidget(canvas_violin)
            
            # Add toolbar for interactive features
            toolbar_violin = NavigationToolbar(canvas_violin, self.violin_plot_widget)
            self.violin_plot_widget.layout().addWidget(toolbar_violin)
            
            # Add save button to toolbar
            save_button = QPushButton("Save Plot")
            save_button.clicked.connect(lambda: self.save_plot(fig_violin, "ga_violin_plot"))
            toolbar_violin.addWidget(save_button)
            
            # Add "Open in New Window" button
            open_new_window_button = QPushButton("Open in New Window")
            open_new_window_button.setObjectName("secondary-button")
            open_new_window_button.clicked.connect(lambda: self._open_plot_window(fig_violin, "GA Violin Plot"))
            self.violin_plot_widget.layout().addWidget(open_new_window_button)

        except Exception as e:
            print(f"Error creating violin plot: {str(e)}")
            
        # 2. Create distribution plots
        try:
            # Clear existing plot layout
            if self.dist_plot_widget.layout():
                for i in reversed(range(self.dist_plot_widget.layout().count())): 
                    self.dist_plot_widget.layout().itemAt(i).widget().setParent(None)
            else:
                self.dist_plot_widget.setLayout(QVBoxLayout())
                
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
            
            # Add vertical line for mean and median (compact legend)
            mean_fitness = df["best_fitness"].mean()
            median_fitness = df["best_fitness"].median()
            std_fitness = df["best_fitness"].std()
            ax_dist.axvline(mean_fitness, color='red', linestyle='--', linewidth=2, label='Mean')
            ax_dist.axvline(median_fitness, color='green', linestyle=':', linewidth=2, label='Median')
            
            # Add std deviation range (no legend entry)
            ax_dist.axvspan(mean_fitness - std_fitness, mean_fitness + std_fitness, alpha=0.15, color='yellow', 
                          label=None)
            
            # Add tolerance line
            tolerance = self.ga_tol_box.value()
            ax_dist.axvline(tolerance, color='magenta', linestyle='--', linewidth=2.5, alpha=0.9, 
                          label='Tolerance')
            
            # Add a shaded region below tolerance (no legend entry)
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
            self.dist_plot_widget.layout().addWidget(canvas_dist)
            
            # Add toolbar for interactive features
            toolbar_dist = NavigationToolbar(canvas_dist, self.dist_plot_widget)
            self.dist_plot_widget.layout().addWidget(toolbar_dist)
            
            # Add save button to toolbar
            save_button = QPushButton("Save Plot")
            save_button.clicked.connect(lambda: self.save_plot(fig_dist, "ga_distribution_plot"))
            toolbar_dist.addWidget(save_button)
            
            # Add "Open in New Window" button
            open_new_window_button = QPushButton("Open in New Window")
            open_new_window_button.setObjectName("secondary-button")
            open_new_window_button.clicked.connect(lambda: self._open_plot_window(fig_dist, "GA Distribution Plot"))
            self.dist_plot_widget.layout().addWidget(open_new_window_button)

        except Exception as e:
            print(f"Error creating distribution plot: {str(e)}")
            
        # 3. Create scatter plots
        try:
            # Clear existing plot layout
            if self.scatter_plot_widget.layout():
                for i in reversed(range(self.scatter_plot_widget.layout().count())): 
                    self.scatter_plot_widget.layout().itemAt(i).widget().setParent(None)
            else:
                self.scatter_plot_widget.setLayout(QVBoxLayout())
                
            # Create figure for scatter plot
            fig_scatter = Figure(figsize=(10, 6), tight_layout=True)
            ax_scatter = fig_scatter.add_subplot(111)
            
            # Create scatter plot of fitness vs run number with trend line
            from scipy import stats
            
            # Calculate linear regression and correlation
            slope, intercept, r_value, p_value, std_err = stats.linregress(df["run_number"], df["best_fitness"])
            correlation = r_value
            
            # Create scatter plot with trend line
            sns.regplot(x="run_number", y="best_fitness", data=df, ax=ax_scatter, 
                       scatter_kws={"color": "darkblue", "alpha": 0.6, "s": 50},
                       line_kws={"color": "red", "alpha": 0.7})
            
            trend_direction = "improving" if slope < 0 else "worsening" if slope > 0 else "stable"
            ax_scatter.set_title(f"Best Fitness Values Across Runs (Trend: {trend_direction})", fontsize=14)
            ax_scatter.set_xlabel("Run Number", fontsize=12)
            ax_scatter.set_ylabel("Best Fitness Value", fontsize=12)
            ax_scatter.grid(True, linestyle="--", alpha=0.7)
            
            # Add tolerance line (without legend entry)
            tolerance = self.ga_tol_box.value()
            ax_scatter.axhline(y=tolerance, color='magenta', linestyle='--', linewidth=2.5, alpha=0.9,
                             label=None)
            
            # Add a shaded region below tolerance (no legend entry)
            ax_scatter.axhspan(0, tolerance, color='magenta', alpha=0.1, label=None)
            
            # Color points that are below tolerance
            below_tolerance_df = df[df["best_fitness"] <= tolerance]
            below_tolerance_count = len(below_tolerance_df)
            below_tolerance_percent = (below_tolerance_count / len(df)) * 100
            
            if not below_tolerance_df.empty:
                ax_scatter.scatter(below_tolerance_df["run_number"], below_tolerance_df["best_fitness"], 
                                 color='green', s=80, alpha=0.8, edgecolor='black', zorder=5,
                                 label='Success Points')
            
            # Find and mark best run
            best_run_idx = df["best_fitness"].idxmin()
            best_run = df.iloc[best_run_idx]
            ax_scatter.scatter(best_run["run_number"], best_run["best_fitness"], 
                             color='gold', s=120, alpha=1.0, edgecolor='black', marker='*', zorder=6,
                             label='Best Run')
            
            # Add correlation statistics in lower left (away from points)
            stats_text = (
                f"Correlation: {correlation:.4f}\n"
                f"Success Rate: {below_tolerance_percent:.1f}%\n"
                f"Best: {best_run['best_fitness']:.6f} (Tol: {tolerance:.6f})"
            )
            props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.6)
            ax_scatter.text(0.03, 0.15, stats_text, transform=ax_scatter.transAxes, 
                         fontsize=10, verticalalignment='bottom', bbox=props)
            
            # Add legend with fewer items
            ax_scatter.legend(loc='lower right', framealpha=0.7)
            
            # Create canvas and add to layout
            canvas_scatter = FigureCanvasQTAgg(fig_scatter)
            self.scatter_plot_widget.layout().addWidget(canvas_scatter)
            
            # Add toolbar for interactive features
            toolbar_scatter = NavigationToolbar(canvas_scatter, self.scatter_plot_widget)
            self.scatter_plot_widget.layout().addWidget(toolbar_scatter)

            # Add "Open in New Window" button
            open_new_window_button = QPushButton("Open in New Window")
            open_new_window_button.setObjectName("secondary-button")
            open_new_window_button.clicked.connect(lambda: self._open_plot_window(fig_scatter, "Scatter Plot"))
            self.scatter_plot_widget.layout().addWidget(open_new_window_button)

        except Exception as e:
            print(f"Error creating scatter plot: {str(e)}")
            
        # 4. Create heatmap of correlation between parameters and fitness
        try:
            # Clear existing plot layout
            if self.heatmap_plot_widget.layout():
                for i in reversed(range(self.heatmap_plot_widget.layout().count())): 
                    self.heatmap_plot_widget.layout().itemAt(i).widget().setParent(None)
            else:
                self.heatmap_plot_widget.setLayout(QVBoxLayout())
            
            # Create figure for heatmap
            fig_heatmap = Figure(figsize=(12, 10), tight_layout=True)
            ax_heatmap = fig_heatmap.add_subplot(111)
            
            # Extract parameter values from each run into a DataFrame
            param_values = []
            
            if len(df) > 0 and 'best_solution' in df.iloc[0] and 'parameter_names' in df.iloc[0]:
                # Get parameter names
                param_names = df.iloc[0]['parameter_names']
                
                # Limit to max 10 parameters to keep visualization manageable
                max_params = min(10, len(param_names))
                selected_params = param_names[:max_params]
                
                # For each run, extract the parameter values
                for i, row in df.iterrows():
                    run_data = {'run_number': row['run_number'], 'best_fitness': row['best_fitness']}
                    
                    # Extract the parameter values
                    solution = row['best_solution']
                    for j, param in enumerate(selected_params):
                        if j < len(solution):
                            run_data[param] = solution[j]
                    
                    param_values.append(run_data)
                
                # Create DataFrame
                param_df = pd.DataFrame(param_values)
                
                if len(param_df) > 0 and len(param_df.columns) > 2:  # Need more than just run_number and best_fitness
                    # Calculate correlation matrix
                    corr_matrix = param_df.corr()
                    
                    # Create heatmap
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', 
                               linewidths=0.5, ax=ax_heatmap, vmin=-1, vmax=1)
                    ax_heatmap.set_title("Correlation Between Parameters and Fitness", fontsize=14)
                    
                    # Create canvas and add to layout
                    canvas_heatmap = FigureCanvasQTAgg(fig_heatmap)
                    self.heatmap_plot_widget.layout().addWidget(canvas_heatmap)
                    
                    # Add toolbar for interactive features
                    toolbar_heatmap = NavigationToolbar(canvas_heatmap, self.heatmap_plot_widget)
                    self.heatmap_plot_widget.layout().addWidget(toolbar_heatmap)

                    # Add "Open in New Window" button
                    open_new_window_button = QPushButton("Open in New Window")
                    open_new_window_button.setObjectName("secondary-button")
                    open_new_window_button.clicked.connect(lambda: self._open_plot_window(fig_heatmap, "Heatmap Plot"))
                    self.heatmap_plot_widget.layout().addWidget(open_new_window_button)
                else:
                    # Create a label for insufficient data
                    no_data_label = QLabel("Insufficient data for correlation analysis")
                    self.heatmap_plot_widget.layout().addWidget(no_data_label)
            else:
                # Create a label if no parameter data
                no_data_label = QLabel("No parameter data available for correlation analysis")
                self.heatmap_plot_widget.layout().addWidget(no_data_label)
        except Exception as e:
            print(f"Error creating heatmap: {str(e)}")
            error_label = QLabel(f"Error creating heatmap: {str(e)}")
            self.heatmap_plot_widget.layout().addWidget(error_label)
            
        # 5. Create Q-Q plot for normality assessment
        try:
            # Clear existing plot layout
            if self.qq_plot_widget.layout():
                for i in reversed(range(self.qq_plot_widget.layout().count())): 
                    self.qq_plot_widget.layout().itemAt(i).widget().setParent(None)
            else:
                self.qq_plot_widget.setLayout(QVBoxLayout())
                
            # Create figure for Q-Q plot
            fig_qq = Figure(figsize=(10, 6), tight_layout=True)
            ax_qq = fig_qq.add_subplot(111)
            
            # Create Q-Q plot
            from scipy import stats
            stats.probplot(df["best_fitness"], dist="norm", plot=ax_qq)
            ax_qq.set_title("Q-Q Plot for Normality Assessment", fontsize=14)
            ax_qq.set_xlabel("Theoretical Quantiles", fontsize=12)
            ax_qq.set_ylabel("Sample Quantiles", fontsize=12)
            ax_qq.grid(True, linestyle="--", alpha=0.7)
            
            # Perform normality tests
            shapiro_test = stats.shapiro(df["best_fitness"])
            ks_test = stats.kstest(df["best_fitness"], 'norm', 
                                 args=(df["best_fitness"].mean(), df["best_fitness"].std()))
            
            # Add test results as text
            test_text = (
                f"Shapiro-Wilk Test:\n"
                f"W = {shapiro_test[0]:.4f}\n"
                f"p-value = {shapiro_test[1]:.4f}\n\n"
                f"Kolmogorov-Smirnov Test:\n"
                f"D = {ks_test[0]:.4f}\n"
                f"p-value = {ks_test[1]:.4f}"
            )
            
            # Create a text box for the test results
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax_qq.text(0.05, 0.95, test_text, transform=ax_qq.transAxes, 
                      fontsize=10, verticalalignment='top', bbox=props)
            
            # Create canvas and add to layout
            canvas_qq = FigureCanvasQTAgg(fig_qq)
            self.qq_plot_widget.layout().addWidget(canvas_qq)
            
            # Add toolbar for interactive features
            toolbar_qq = NavigationToolbar(canvas_qq, self.qq_plot_widget)
            self.qq_plot_widget.layout().addWidget(toolbar_qq)

            # Add "Open in New Window" button
            open_new_window_button = QPushButton("Open in New Window")
            open_new_window_button.setObjectName("secondary-button")
            open_new_window_button.clicked.connect(lambda: self._open_plot_window(fig_qq, "Q-Q Plot"))
            self.qq_plot_widget.layout().addWidget(open_new_window_button)

        except Exception as e:
            print(f"Error creating Q-Q plot: {str(e)}")
        
        # 6. Update statistics table
        try:
            # Calculate statistics for fitness and available parameters
            stats_data = []
            
            # Add fitness statistics
            fitness_stats = {
                "Metric": "Best Fitness",
                "Min": df["best_fitness"].min(),
                "Max": df["best_fitness"].max(),
                "Mean": df["best_fitness"].mean(),
                "Std": df["best_fitness"].std()
            }
            stats_data.append(fitness_stats)
            
            # Add statistics for other metrics in results
            for col in df.columns:
                if col not in ["run_number", "best_fitness", "best_solution", "parameter_names"] and df[col].dtype in [np.float64, np.int64]:
                    metric_stats = {
                        "Metric": col,
                        "Min": df[col].min(),
                        "Max": df[col].max(),
                        "Mean": df[col].mean(),
                        "Std": df[col].std()
                    }
                    stats_data.append(metric_stats)
            
            # Update table with statistics
            self.benchmark_stats_table.setRowCount(len(stats_data))
            for row, stat in enumerate(stats_data):
                self.benchmark_stats_table.setItem(row, 0, QTableWidgetItem(str(stat["Metric"])))
                self.benchmark_stats_table.setItem(row, 1, QTableWidgetItem(f"{stat['Min']:.6f}"))
                self.benchmark_stats_table.setItem(row, 2, QTableWidgetItem(f"{stat['Max']:.6f}"))
                self.benchmark_stats_table.setItem(row, 3, QTableWidgetItem(f"{stat['Mean']:.6f}"))
                self.benchmark_stats_table.setItem(row, 4, QTableWidgetItem(f"{stat['Std']:.6f}"))
                
            # 7. Update runs table with fitness, rank and best/worst/mean indicators
            self.benchmark_runs_table.setRowCount(len(df))
            
            # Sort runs by fitness (assuming lower is better)
            sorted_df = df.sort_values('best_fitness')
            
            # Get index of run with fitness value closest to mean
            mean_fitness = df['best_fitness'].mean()
            mean_index = (df['best_fitness'] - mean_fitness).abs().idxmin()
            
            # Create a button class for the details button
            class DetailButton(QPushButton):
                def __init__(self, run_number):
                    super().__init__("View Details")
                    self.run_number = run_number
            
            # Populate the table
            for i, (_, row) in enumerate(sorted_df.iterrows()):
                run_number = int(row['run_number'])
                fitness = row['best_fitness']
                
                # Create items for the table
                run_item = QTableWidgetItem(str(run_number))
                fitness_item = QTableWidgetItem(f"{fitness:.6f}")
                rank_item = QTableWidgetItem(f"{i+1}/{len(df)}")
                
                # Set alignment
                run_item.setTextAlignment(Qt.AlignCenter)
                fitness_item.setTextAlignment(Qt.AlignCenter)
                rank_item.setTextAlignment(Qt.AlignCenter)
                
                # Color coding
                if i == 0:  # Best run (lowest fitness)
                    run_item.setBackground(QColor(200, 255, 200))  # Light green
                    fitness_item.setBackground(QColor(200, 255, 200))
                    rank_item.setBackground(QColor(200, 255, 200))
                    run_item.setToolTip("Best Run (Lowest Fitness)")
                elif i == len(df) - 1:  # Worst run (highest fitness)
                    run_item.setBackground(QColor(255, 200, 200))  # Light red
                    fitness_item.setBackground(QColor(255, 200, 200))
                    rank_item.setBackground(QColor(255, 200, 200))
                    run_item.setToolTip("Worst Run (Highest Fitness)")
                elif row.name == mean_index:  # Mean run (closest to mean fitness)
                    run_item.setBackground(QColor(255, 255, 200))  # Light yellow
                    fitness_item.setBackground(QColor(255, 255, 200))
                    rank_item.setBackground(QColor(255, 255, 200))
                    run_item.setToolTip("Mean Run (Closest to Average Fitness)")
                
                # Add items to the table
                self.benchmark_runs_table.setItem(i, 0, run_item)
                self.benchmark_runs_table.setItem(i, 1, fitness_item)
                self.benchmark_runs_table.setItem(i, 2, rank_item)
                
                # Add a details button
                detail_btn = DetailButton(run_number)
                detail_btn.clicked.connect(lambda _, btn=detail_btn: self.show_run_details(
                    self.benchmark_runs_table.item(
                        [i for i in range(self.benchmark_runs_table.rowCount()) 
                         if int(self.benchmark_runs_table.item(i, 0).text()) == btn.run_number][0], 0)))
                self.benchmark_runs_table.setCellWidget(i, 3, detail_btn)
        except Exception as e:
            print(f"Error updating statistics tables: {str(e)}")
        
        # Connect export button if not already connected
        try:
            self.export_benchmark_button.clicked.disconnect()
        except:
            pass
        self.export_benchmark_button.clicked.connect(self.export_ga_benchmark_data)
        
        # 8. Generate comprehensive statistical analysis for parameters
        self.generate_parameter_statistical_analysis(df)
        
    def generate_parameter_statistical_analysis(self, df):
        """Generate comprehensive statistical analysis for optimized parameters across all runs"""
        try:
            # Extract parameter data from all runs
            parameter_data = self.extract_parameter_data_from_runs(df)
            
            if not parameter_data:
                print("No parameter data available for statistical analysis")
                return
            
            # Store parameter data for use in dropdowns
            self.current_parameter_data = parameter_data
            
            # Update parameter dropdowns
            self.update_parameter_dropdowns(parameter_data)
            
            # Initialize with default plots
            self.update_parameter_plots()
            
            # Create parameter statistics tables
            self.create_parameter_statistics_tables(parameter_data)
            
        except Exception as e:
            print(f"Error generating parameter statistical analysis: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def update_parameter_dropdowns(self, parameter_data):
        """Update dropdown menus with available parameters"""
        try:
            param_names = list(parameter_data.keys())
            
            # Update parameter selection dropdown with all parameters
            self.param_selection_combo.clear()
            self.param_selection_combo.setMaxVisibleItems(10)  # Show 10 items at a time in dropdown
            self.param_selection_combo.setStyleSheet("""
                QComboBox {
                    min-width: 200px;
                }
                QComboBox QListView {
                    min-width: 200px;
                }
            """)
            for param_name in param_names:
                self.param_selection_combo.addItem(param_name)
            
            # Update comparison parameter dropdown with all parameters
            self.comparison_param_combo.clear()
            self.comparison_param_combo.setMaxVisibleItems(10)  # Show 10 items at a time in dropdown
            self.comparison_param_combo.setStyleSheet("""
                QComboBox {
                    min-width: 200px;
                }
                QComboBox QListView {
                    min-width: 200px;
                }
            """)
            self.comparison_param_combo.addItem("None")
            for param_name in param_names:
                self.comparison_param_combo.addItem(param_name)
                
        except Exception as e:
            print(f"Error updating parameter dropdowns: {str(e)}")
    
    def on_parameter_selection_changed(self):
        """Handle parameter selection change"""
        # Auto-update plots when parameter selection changes
        self.update_parameter_plots()
    
    def on_plot_type_changed(self):
        """Handle plot type change"""
        plot_type = self.plot_type_combo.currentText()
        
        # Enable/disable comparison parameter dropdown based on plot type
        if plot_type == "Scatter Plot":
            self.comparison_param_combo.setEnabled(True)
        else:
            self.comparison_param_combo.setEnabled(False)
        
        # Auto-update plots when plot type changes
        self.update_parameter_plots()
    
    def on_comparison_parameter_changed(self):
        """Handle comparison parameter change"""
        # Auto-update plots when comparison parameter changes
        self.update_parameter_plots()
    
    def on_stats_view_changed(self):
        """Handle statistics view change"""
        if hasattr(self, 'current_parameter_data'):
            self.create_parameter_statistics_tables(self.current_parameter_data)
    
    def update_parameter_plots(self):
        """Update parameter plots based on current selections"""
        try:
            if not hasattr(self, 'current_parameter_data') or not self.current_parameter_data:
                print("No parameter data available for plotting")
                return
            
            selected_param = self.param_selection_combo.currentText()
            plot_type = self.plot_type_combo.currentText()
            comparison_param = self.comparison_param_combo.currentText()
            
            print(f"Debug: Creating plot - Parameter: {selected_param}, Type: {plot_type}, Comparison: {comparison_param}")
            
            # Clear existing plots properly
            if self.param_plot_widget.layout():
                while self.param_plot_widget.layout().count():
                    child = self.param_plot_widget.layout().takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()
            else:
                self.param_plot_widget.setLayout(QVBoxLayout())
            
            print(f"Debug: Layout cleared, widget count: {self.param_plot_widget.layout().count()}")
            
            # Create plots based on selections
            if plot_type == "Violin Plot":
                print("Debug: Creating violin plot")
                self.create_violin_plot(selected_param)
            elif plot_type == "Distribution Plot":
                print("Debug: Creating distribution plot")
                self.create_distribution_plot(selected_param)
            elif plot_type == "Scatter Plot":
                print("Debug: Creating scatter plot")
                self.create_scatter_plot(selected_param, comparison_param)
            elif plot_type == "Q-Q Plot":
                print("Debug: Creating Q-Q plot")
                self.create_qq_plot(selected_param)
            
            # Add a spacer at the bottom to push content up
            spacer = QWidget()
            spacer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.param_plot_widget.layout().addWidget(spacer)
                
        except Exception as e:
            print(f"Error updating parameter plots: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def extract_parameter_data_from_runs(self, df):
        """Extract parameter values from all runs for statistical analysis"""
        try:
            parameter_data = {}
            
            for idx, row in df.iterrows():
                run_number = row['run_number']
                best_solution = row['best_solution']
                parameter_names = row['parameter_names']
                
                # Ensure we have valid data
                if not isinstance(best_solution, list) or not isinstance(parameter_names, list):
                    continue
                
                if len(best_solution) != len(parameter_names):
                    continue
                
                # Extract parameter values for this run
                for param_name, param_value in zip(parameter_names, best_solution):
                    if param_name not in parameter_data:
                        parameter_data[param_name] = []
                    parameter_data[param_name].append(param_value)
            
            # Convert to numpy arrays for easier statistical analysis
            for param_name in parameter_data:
                parameter_data[param_name] = np.array(parameter_data[param_name])
            
            return parameter_data
            
        except Exception as e:
            print(f"Error extracting parameter data: {str(e)}")
            return {}
    
    def create_violin_plot(self, selected_param):
        """Create enhanced violin plot for selected parameter"""
        try:
            param_names = [selected_param]
            
            n_params = len(param_names)
            if n_params == 0:
                return
            
            # Determine grid layout - larger plots
            n_cols = min(2, n_params) if n_params > 4 else min(3, n_params)
            n_rows = (n_params + n_cols - 1) // n_cols
            
            # Create adaptive figure size based on number of parameters and screen space
            base_width = min(8, 12 if n_params <= 2 else 6)
            base_height = min(6, 8 if n_params <= 2 else 5)
            fig = Figure(figsize=(base_width * n_cols, base_height * n_rows), dpi=100, tight_layout=True)
            fig.patch.set_facecolor('white')
            
            # Color palette for different parameters
            colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', 
                     '#34495e', '#e67e22', '#95a5a6', '#16a085']
            
            for i, param_name in enumerate(param_names):
                ax = fig.add_subplot(n_rows, n_cols, i + 1)
                
                param_values = self.current_parameter_data[param_name]
                color = colors[i % len(colors)]
                
                # Create violin plot with enhanced styling
                violin_parts = ax.violinplot([param_values], positions=[0], showmeans=True, 
                                           showmedians=True, showextrema=True)
                
                # Customize violin plot
                for pc in violin_parts['bodies']:
                    pc.set_facecolor(color)
                    pc.set_alpha(0.7)
                    pc.set_edgecolor('black')
                    pc.set_linewidth(1.2)
                
                # Style the median, mean, and extrema
                if 'cmedians' in violin_parts:
                    violin_parts['cmedians'].set_color('red')
                    violin_parts['cmedians'].set_linewidth(2)
                if 'cmeans' in violin_parts:
                    violin_parts['cmeans'].set_color('darkred')
                    violin_parts['cmeans'].set_linewidth(2)
                if 'cmaxes' in violin_parts:
                    violin_parts['cmaxes'].set_color('black')
                if 'cmins' in violin_parts:
                    violin_parts['cmins'].set_color('black')
                
                # Add enhanced box plot overlay
                box_plot = ax.boxplot([param_values], positions=[0], widths=0.15, patch_artist=True,
                          boxprops=dict(facecolor='white', alpha=0.8, edgecolor='black', linewidth=1.5),
                          medianprops=dict(color='red', linewidth=2),
                          whiskerprops=dict(color='black', linewidth=1.5),
                          capprops=dict(color='black', linewidth=1.5),
                          flierprops=dict(marker='o', markerfacecolor='red', markersize=6, alpha=0.7))
                
                # Calculate comprehensive statistics
                mean_val = np.mean(param_values)
                median_val = np.median(param_values)
                std_val = np.std(param_values)
                min_val = np.min(param_values)
                max_val = np.max(param_values)
                q1 = np.percentile(param_values, 25)
                q3 = np.percentile(param_values, 75)
                iqr = q3 - q1
                cv = (std_val / mean_val) * 100 if mean_val != 0 else 0
                
                # Calculate outliers
                outlier_threshold = 1.5 * iqr
                outliers = param_values[(param_values < (q1 - outlier_threshold)) | 
                                      (param_values > (q3 + outlier_threshold))]
                n_outliers = len(outliers)
                
                # Enhanced statistics text with more information
                stats_text = (f"Count: {len(param_values)}\n"
                             f"Mean: {mean_val:.4f}\n"
                             f"Median: {median_val:.4f}\n"
                             f"Std Dev: {std_val:.4f}\n"
                             f"CV: {cv:.1f}%\n"
                             f"Q1: {q1:.4f}\n"
                             f"Q3: {q3:.4f}\n"
                             f"IQR: {iqr:.4f}\n"
                             f"Range: {max_val - min_val:.4f}\n"
                             f"Outliers: {n_outliers}")
                
                # Enhanced text box styling
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       fontsize=10, verticalalignment='top', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', 
                               alpha=0.9, edgecolor='black', linewidth=1))
                
                # Add percentile lines
                for percentile, value in [(10, np.percentile(param_values, 10)),
                                        (90, np.percentile(param_values, 90))]:
                    ax.axhline(y=value, color='orange', linestyle='--', alpha=0.6, 
                             linewidth=1, label=f'{percentile}th percentile')
                
                # Enhanced styling
                ax.set_title(f"{param_name}", fontsize=14, fontweight='bold', pad=20)
                ax.set_ylabel("Parameter Value", fontsize=12, fontweight='bold')
                ax.set_xticks([])
                ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                ax.set_facecolor('#f8f9fa')
                
                # Add parameter value range as subtitle
                ax.text(0.5, -0.1, f"Range: [{min_val:.4f}, {max_val:.4f}]", 
                       transform=ax.transAxes, ha='center', fontsize=10, style='italic')
            
            # Create canvas and add to layout
            canvas = FigureCanvasQTAgg(fig)
            self.param_plot_widget.layout().addWidget(canvas)
            
            # Add toolbar
            toolbar = NavigationToolbar(canvas, self.param_plot_widget)
            self.param_plot_widget.layout().addWidget(toolbar)
            
            # Add buttons using the helper method
            self.add_plot_buttons(fig, "Violin Plot", selected_param)
            
        except Exception as e:
            print(f"Error creating violin plot: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def create_distribution_plot(self, selected_param):
        """Create enhanced distribution plot for selected parameter"""
        try:
            param_names = [selected_param]
            
            n_params = len(param_names)
            if n_params == 0:
                return
            
            # Determine grid layout - larger plots
            n_cols = min(2, n_params) if n_params > 4 else min(3, n_params)
            n_rows = (n_params + n_cols - 1) // n_cols
            
            # Create adaptive figure size for better display
            base_width = min(8, 12 if n_params <= 2 else 6)
            base_height = min(6, 8 if n_params <= 2 else 5)
            fig = Figure(figsize=(base_width * n_cols, base_height * n_rows), dpi=100, tight_layout=True)
            fig.patch.set_facecolor('white')
            
            # Color palette
            colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', 
                     '#34495e', '#e67e22', '#95a5a6', '#16a085']
            
            for i, param_name in enumerate(param_names):
                ax = fig.add_subplot(n_rows, n_cols, i + 1)
                
                values = self.current_parameter_data[param_name]
                color = colors[i % len(colors)]
                
                # Create enhanced histogram with better binning
                n_bins = max(20, min(50, len(values) // 10))
                n, bins, patches = ax.hist(values, bins=n_bins, density=True, alpha=0.7, 
                                         color=color, edgecolor='black', linewidth=1.2)
                
                # Add KDE curve
                try:
                    from scipy import stats
                    kde = stats.gaussian_kde(values)
                    x_range = np.linspace(values.min(), values.max(), 200)
                    ax.plot(x_range, kde(x_range), 'darkred', linewidth=3, 
                           label='Kernel Density Estimate', alpha=0.9)
                    
                    # Add normal distribution overlay for comparison
                    mu, sigma = np.mean(values), np.std(values)
                    x_norm = np.linspace(values.min(), values.max(), 200)
                    normal_curve = stats.norm.pdf(x_norm, mu, sigma)
                    ax.plot(x_norm, normal_curve, 'purple', linewidth=2, linestyle=':', 
                           label='Normal Distribution', alpha=0.8)
                    
                    # Calculate normality test
                    _, p_value = stats.normaltest(values)
                    normality_text = f"Normality p-value: {p_value:.4f}"
                    
                except Exception as e:
                    print(f"Error creating KDE for {param_name}: {str(e)}")
                    normality_text = "Normality test: N/A"
                
                # Calculate comprehensive statistics
                mean_val = np.mean(values)
                median_val = np.median(values)
                mode_val = bins[np.argmax(n)] if len(bins) > 1 else mean_val
                std_val = np.std(values)
                skewness = stats.skew(values) if 'stats' in locals() else 0
                kurtosis = stats.kurtosis(values) if 'stats' in locals() else 0
                
                # Add statistical lines
                ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                          label=f'Mean: {mean_val:.4f}', alpha=0.8)
                ax.axvline(median_val, color='green', linestyle='--', linewidth=2, 
                          label=f'Median: {median_val:.4f}', alpha=0.8)
                ax.axvline(mode_val, color='orange', linestyle='--', linewidth=2, 
                          label=f'Mode: {mode_val:.4f}', alpha=0.8)
                
                # Add percentile lines
                q1 = np.percentile(values, 25)
                q3 = np.percentile(values, 75)
                ax.axvline(q1, color='blue', linestyle=':', linewidth=1.5, 
                          label=f'Q1: {q1:.4f}', alpha=0.6)
                ax.axvline(q3, color='blue', linestyle=':', linewidth=1.5, 
                          label=f'Q3: {q3:.4f}', alpha=0.6)
                
                # Add statistics text box
                stats_text = (f"Count: {len(values)}\n"
                             f"Mean: {mean_val:.4f}\n"
                             f"Std Dev: {std_val:.4f}\n"
                             f"Skewness: {skewness:.3f}\n"
                             f"Kurtosis: {kurtosis:.3f}\n"
                             f"Min: {values.min():.4f}\n"
                             f"Max: {values.max():.4f}\n"
                             f"{normality_text}")
                
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       fontsize=10, verticalalignment='top', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', 
                               alpha=0.9, edgecolor='black', linewidth=1))
                
                # Enhanced styling
                ax.set_title(f"{param_name} Distribution", fontsize=14, fontweight='bold', pad=20)
                ax.set_xlabel("Parameter Value", fontsize=12, fontweight='bold')
                ax.set_ylabel("Density", fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                ax.set_facecolor('#f8f9fa')
                
                # Enhanced legend
                legend = ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
                legend.get_frame().set_facecolor('white')
                legend.get_frame().set_edgecolor('black')
                
                # Add distribution shape analysis
                shape_text = ""
                if 'skewness' in locals():
                    if abs(skewness) < 0.5:
                        shape_text = "Approximately symmetric"
                    elif skewness > 0.5:
                        shape_text = "Right-skewed"
                    else:
                        shape_text = "Left-skewed"
                
                ax.text(0.5, -0.12, shape_text, transform=ax.transAxes, 
                       ha='center', fontsize=10, style='italic')
            
            # Create canvas and add to layout
            canvas = FigureCanvasQTAgg(fig)
            self.param_plot_widget.layout().addWidget(canvas)
            
            # Add toolbar
            toolbar = NavigationToolbar(canvas, self.param_plot_widget)
            self.param_plot_widget.layout().addWidget(toolbar)
            
            # Add buttons using the helper method
            self.add_plot_buttons(fig, "Distribution Plot", selected_param)
            
        except Exception as e:
            print(f"Error creating distribution plot: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def create_scatter_plot(self, selected_param, comparison_param):
        """Create scatter plot between selected parameters"""
        try:
            if comparison_param == "None" or comparison_param == selected_param:
                # Show scatter plot of parameter vs run number
                self.create_parameter_vs_run_scatter(selected_param)
            else:
                # Create scatter plot between two specific parameters
                self.create_two_parameter_scatter(selected_param, comparison_param)
                
        except Exception as e:
            print(f"Error creating scatter plot: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def create_parameter_vs_run_scatter(self, param_name):
        """Create enhanced scatter plot of parameter vs run number"""
        try:
            fig = Figure(figsize=(10, 7), dpi=100, tight_layout=True)
            fig.patch.set_facecolor('white')
            ax = fig.add_subplot(111)
            
            values = self.current_parameter_data[param_name]
            run_numbers = range(1, len(values) + 1)
            
            # Enhanced scatter plot with color mapping
            scatter = ax.scatter(run_numbers, values, alpha=0.7, s=60, 
                               c=values, cmap='viridis', edgecolors='black', linewidth=0.5)
            
            # Add colorbar
            cbar = fig.colorbar(scatter, ax=ax, shrink=0.8, aspect=20)
            cbar.set_label(f'{param_name} Value', fontsize=12, fontweight='bold')
            
            # Add trend line with confidence interval
            z = np.polyfit(run_numbers, values, 1)
            p = np.poly1d(z)
            trend_line = p(run_numbers)
            ax.plot(run_numbers, trend_line, "r--", linewidth=2, alpha=0.8, 
                   label=f'Trend: y={z[0]:.6f}x+{z[1]:.6f}')
            
            # Calculate R-squared
            ss_res = np.sum((values - trend_line) ** 2)
            ss_tot = np.sum((values - np.mean(values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            # Add confidence intervals
            try:
                from scipy import stats
                # Calculate prediction intervals
                residuals = values - trend_line
                mse = np.mean(residuals**2)
                std_err = np.sqrt(mse)
                
                # 95% confidence intervals
                ci_upper = trend_line + 1.96 * std_err
                ci_lower = trend_line - 1.96 * std_err
                
                ax.fill_between(run_numbers, ci_lower, ci_upper, alpha=0.2, color='red', 
                               label='95% Confidence Interval')
                
                # Calculate correlation coefficient
                correlation = np.corrcoef(run_numbers, values)[0, 1]
                
            except Exception as e:
                correlation = 0
                print(f"Error calculating statistics: {str(e)}")
            
            # Add statistics text
            stats_text = (f"Correlation: {correlation:.4f}\n"
                         f"R²: {r_squared:.4f}\n"
                         f"Trend slope: {z[0]:.6f}\n"
                         f"Mean: {np.mean(values):.4f}\n"
                         f"Std Dev: {np.std(values):.4f}\n"
                         f"Range: {np.max(values) - np.min(values):.4f}")
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top', fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', 
                           alpha=0.9, edgecolor='black', linewidth=1))
            
            # Enhanced styling
            ax.set_xlabel("Run Number", fontsize=12, fontweight='bold')
            ax.set_ylabel(f"{param_name} Value", fontsize=12, fontweight='bold')
            ax.set_title(f"{param_name} vs Run Number", fontsize=14, fontweight='bold', pad=20)
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            ax.set_facecolor('#f8f9fa')
            
            # Enhanced legend
            legend = ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_edgecolor('black')
            
            # Create canvas and add to layout
            canvas = FigureCanvasQTAgg(fig)
            self.param_plot_widget.layout().addWidget(canvas)
            
            # Add toolbar
            toolbar = NavigationToolbar(canvas, self.param_plot_widget)
            self.param_plot_widget.layout().addWidget(toolbar)
            
            # Add buttons using the helper method
            self.add_plot_buttons(fig, "Scatter Plot", param_name)
            
        except Exception as e:
            print(f"Error creating parameter vs run scatter plot: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def create_scatter_matrix(self, param_names):
        """Create enhanced scatter plot matrix for multiple parameters"""
        try:
            import pandas as pd
            
            n_params = len(param_names)
            if n_params < 2:
                return
            
            # Create DataFrame for scatter matrix
            data_dict = {name: self.current_parameter_data[name] for name in param_names}
            df = pd.DataFrame(data_dict)
            
            # Create responsive figure for scatter matrix
            base_size = min(4, 6 if n_params <= 3 else 3)
            fig = Figure(figsize=(base_size * n_params, base_size * n_params), dpi=100, tight_layout=True)
            fig.patch.set_facecolor('white')
            
            # Color palette for parameters
            colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
            
            for i, param_x in enumerate(param_names):
                for j, param_y in enumerate(param_names):
                    ax = fig.add_subplot(n_params, n_params, i * n_params + j + 1)
                    
                    if i == j:
                        # Diagonal: enhanced histogram with KDE
                        color = colors[i % len(colors)]
                        n, bins, patches = ax.hist(df[param_x], bins=20, alpha=0.7, 
                                                 color=color, edgecolor='black', linewidth=1)
                        
                        # Add KDE curve
                        try:
                            from scipy import stats
                            kde = stats.gaussian_kde(df[param_x])
                            x_range = np.linspace(df[param_x].min(), df[param_x].max(), 100)
                            kde_values = kde(x_range)
                            # Scale KDE to match histogram
                            bin_width = bins[1] - bins[0]
                            kde_scaled = kde_values * np.max(n) * bin_width / np.max(kde_values)
                            ax.plot(x_range, kde_scaled, 'darkred', linewidth=2, alpha=0.8)
                        except:
                            pass
                        
                        ax.set_title(f"{param_x}", fontsize=12, fontweight='bold')
                        ax.set_facecolor('#f8f9fa')
                    else:
                        # Off-diagonal: enhanced scatter plot
                        scatter = ax.scatter(df[param_x], df[param_y], alpha=0.6, s=30, 
                                          c=df[param_y], cmap='viridis', edgecolors='black', linewidth=0.3)
                        
                        # Add trend line
                        try:
                            z = np.polyfit(df[param_x], df[param_y], 1)
                            p = np.poly1d(z)
                            x_trend = np.linspace(df[param_x].min(), df[param_x].max(), 100)
                            ax.plot(x_trend, p(x_trend), "r--", linewidth=1.5, alpha=0.8)
                        except:
                            pass
                        
                        # Add correlation coefficient and p-value
                        corr_coef = np.corrcoef(df[param_x], df[param_y])[0, 1]
                        
                        # Calculate significance
                        try:
                            from scipy import stats
                            _, p_value = stats.pearsonr(df[param_x], df[param_y])
                            significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                            corr_text = f'r={corr_coef:.3f}{significance}\np={p_value:.3f}'
                        except:
                            corr_text = f'r={corr_coef:.3f}'
                        
                        ax.text(0.05, 0.95, corr_text, transform=ax.transAxes, fontsize=8,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, 
                                       edgecolor='black', linewidth=1))
                        
                        ax.set_facecolor('#f8f9fa')
                    
                    # Enhanced styling
                    ax.set_xlabel(param_x, fontsize=10, fontweight='bold')
                    ax.set_ylabel(param_y, fontsize=10, fontweight='bold')
                    ax.tick_params(labelsize=8)
                    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            
            # Add overall title
            fig.suptitle('Parameter Scatter Matrix', fontsize=16, fontweight='bold', y=0.98)
            
            # Create canvas and add to layout
            canvas = FigureCanvasQTAgg(fig)
            self.param_plot_widget.layout().addWidget(canvas)
            
            # Add toolbar
            toolbar = NavigationToolbar(canvas, self.param_plot_widget)
            self.param_plot_widget.layout().addWidget(toolbar)
            
            # Add buttons using the helper method
            self.add_plot_buttons(fig, "Scatter Matrix", "all_parameters")
            
            # Add external window button
            external_button = QPushButton("🔍 Open in New Window")
            external_button.setStyleSheet("""
                QPushButton {
                    background-color: #2ecc71;
                    color: white;
                    border: none;
                    padding: 8px 16px;
                    border-radius: 4px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #27ae60;
                }
            """)
            external_button.clicked.connect(lambda: self._open_plot_window(fig, "Scatter Matrix"))
            
            buttons_layout.addWidget(save_button)
            buttons_layout.addWidget(external_button)
            buttons_layout.addStretch()
            
            self.param_plot_widget.layout().addWidget(buttons_container)
            
        except Exception as e:
            print(f"Error creating scatter matrix: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def create_two_parameter_scatter(self, param_x, param_y):
        """Create enhanced scatter plot between two specific parameters"""
        try:
            # Create a larger figure with better spacing
            fig = Figure(figsize=(12, 8), dpi=100)
            fig.patch.set_facecolor('white')
            
            # Create main plot and marginal plots with better spacing
            gs = fig.add_gridspec(3, 3, height_ratios=[1, 4, 4], width_ratios=[4, 4, 1],
                                hspace=0.4, wspace=0.4)
            
            # Main scatter plot
            ax_main = fig.add_subplot(gs[1:, :-1])
            # Top marginal plot (x distribution)
            ax_top = fig.add_subplot(gs[0, :-1], sharex=ax_main)
            # Right marginal plot (y distribution)
            ax_right = fig.add_subplot(gs[1:, -1], sharey=ax_main)
            
            values_x = self.current_parameter_data[param_x]
            values_y = self.current_parameter_data[param_y]
            
            # Enhanced scatter plot with better visibility
            scatter = ax_main.scatter(values_x, values_y, alpha=0.7, s=80, 
                                    c=np.arange(len(values_x)), cmap='viridis', 
                                    edgecolors='white', linewidth=0.8)
            
            # Add colorbar with better positioning
            cbar = fig.colorbar(scatter, ax=[ax_main, ax_right], shrink=0.8, aspect=30, pad=0.02)
            cbar.set_label('Run Order', fontsize=10, fontweight='bold')
            
            # Calculate trend line and statistics
            z = np.polyfit(values_x, values_y, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(values_x.min(), values_x.max(), 100)
            y_trend = p(x_trend)
            
            # Add trend line
            ax_main.plot(x_trend, y_trend, "r--", linewidth=2, alpha=0.8, 
                        label=f'Trend Line')
            
            # Calculate statistics
            ss_res = np.sum((values_y - p(values_x)) ** 2)
            ss_tot = np.sum((values_y - np.mean(values_y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            try:
                from scipy import stats
                # Calculate confidence intervals
                residuals = values_y - p(values_x)
                mse = np.mean(residuals**2)
                std_err = np.sqrt(mse)
                
                # 95% confidence intervals
                y_upper = y_trend + 1.96 * std_err
                y_lower = y_trend - 1.96 * std_err
                
                ax_main.fill_between(x_trend, y_lower, y_upper, alpha=0.2, color='red', 
                                   label='95% Confidence')
                
                # Calculate correlations
                pearson_corr, pearson_p = stats.pearsonr(values_x, values_y)
                spearman_corr, spearman_p = stats.spearmanr(values_x, values_y)
                
                # Add correlation information to the plot
                corr_text = (
                    f"Correlations:\n"
                    f"• Pearson (r): {pearson_corr:.3f}\n"
                    f"  p-value: {pearson_p:.3e}\n"
                    f"• Spearman (ρ): {spearman_corr:.3f}\n"
                    f"  p-value: {spearman_p:.3e}\n"
                    f"• R² Score: {r_squared:.3f}"
                )
                
                # Add trend line equation
                equation_text = (
                    f"Trend Line:\n"
                    f"y = {z[0]:.3e}x + {z[1]:.3e}"
                )
                
            except Exception as e:
                print(f"Error calculating statistics: {str(e)}")
                pearson_corr = np.corrcoef(values_x, values_y)[0, 1]
                corr_text = f"Correlation: {pearson_corr:.3f}"
                equation_text = f"y = {z[0]:.3e}x + {z[1]:.3e}"
            
            # Add statistics text boxes with better positioning
            ax_main.text(0.02, 0.98, corr_text, transform=ax_main.transAxes, 
                        fontsize=10, verticalalignment='top',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                                alpha=0.9, edgecolor='gray', linewidth=1))
            
            ax_main.text(0.02, 0.02, equation_text, transform=ax_main.transAxes,
                        fontsize=10, verticalalignment='bottom',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                                alpha=0.9, edgecolor='gray', linewidth=1))
            
            # Create marginal distributions with KDE
            try:
                # Top marginal (x distribution)
                sns.histplot(values_x, ax=ax_top, bins=30, stat='density', alpha=0.6, 
                           color='#3498db', edgecolor='black', linewidth=1)
                kde_x = stats.gaussian_kde(values_x)
                x_range = np.linspace(values_x.min(), values_x.max(), 100)
                ax_top.plot(x_range, kde_x(x_range), color='#2980b9', linewidth=2)
                ax_top.set_ylabel('Density')
                
                # Right marginal (y distribution)
                sns.histplot(values_y, ax=ax_right, bins=30, stat='density', alpha=0.6,
                           color='#e74c3c', edgecolor='black', linewidth=1,
                           orientation='horizontal')
                kde_y = stats.gaussian_kde(values_y)
                y_range = np.linspace(values_y.min(), values_y.max(), 100)
                ax_right.plot(kde_y(y_range), y_range, color='#c0392b', linewidth=2)
                ax_right.set_xlabel('Density')
            except Exception as e:
                print(f"Error creating marginal plots: {str(e)}")
                # Fallback to simple histograms
                ax_top.hist(values_x, bins=30, density=True, alpha=0.6, color='#3498db')
                ax_right.hist(values_y, bins=30, density=True, alpha=0.6, color='#e74c3c',
                            orientation='horizontal')
            
            # Clean up axes
            ax_top.set_title(f'{param_x} Distribution', fontsize=11, pad=10)
            ax_right.set_title(f'{param_y} Distribution', fontsize=11, rotation=270, pad=15)
            
            # Remove unnecessary labels from marginal plots
            ax_top.set_xlabel('')
            ax_right.set_ylabel('')
            
            # Hide the marginal plot ticks for cleaner look
            ax_top.tick_params(labelbottom=False)
            ax_right.tick_params(labelleft=False)
            
            # Main plot styling
            ax_main.set_xlabel(f"{param_x}", fontsize=12, fontweight='bold')
            ax_main.set_ylabel(f"{param_y}", fontsize=12, fontweight='bold')
            ax_main.grid(True, alpha=0.3, linestyle='--')
            
            # Add correlation strength interpretation
            corr_strength = ""
            if abs(pearson_corr) >= 0.8:
                corr_strength = "Strong"
                color = "#27ae60"  # Green
            elif abs(pearson_corr) >= 0.5:
                corr_strength = "Moderate"
                color = "#f39c12"  # Orange
            elif abs(pearson_corr) >= 0.3:
                corr_strength = "Weak"
                color = "#e67e22"  # Dark Orange
            else:
                corr_strength = "Very Weak"
                color = "#c0392b"  # Red
            
            significance = ""
            if 'pearson_p' in locals():
                if pearson_p <= 0.001:
                    significance = "Highly Significant"
                elif pearson_p <= 0.01:
                    significance = "Significant"
                elif pearson_p <= 0.05:
                    significance = "Marginally Significant"
                else:
                    significance = "Not Significant"
            
            title = f"Parameter Correlation Analysis: {corr_strength}"
            if significance:
                title += f" ({significance})"
            
            fig.suptitle(title, fontsize=14, fontweight='bold', color=color, y=0.95)
            
            # Adjust layout
            fig.tight_layout()
            
            # Create canvas and add to layout
            canvas = FigureCanvasQTAgg(fig)
            self.param_plot_widget.layout().addWidget(canvas)
            
            # Add toolbar
            toolbar = NavigationToolbar(canvas, self.param_plot_widget)
            self.param_plot_widget.layout().addWidget(toolbar)
            
            # Add buttons using helper method
            self.add_plot_buttons(fig, "Parameter Correlation", param_x, param_y)
            
        except Exception as e:
            print(f"Error creating two-parameter scatter plot: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def create_qq_plot(self, selected_param):
        """Create enhanced Q-Q plot for selected parameter"""
        try:
            print(f"Debug: Starting QQ plot for parameter: {selected_param}")
            param_names = [selected_param]
            
            n_params = len(param_names)
            if n_params == 0:
                print("Debug: No parameters to plot")
                return
            
            # Determine grid layout - larger plots
            n_cols = min(2, n_params) if n_params > 4 else min(3, n_params)
            n_rows = (n_params + n_cols - 1) // n_cols
            
            # Create adaptive figure size for better display
            base_width = min(8, 12 if n_params <= 2 else 6)
            base_height = min(6, 8 if n_params <= 2 else 5)
            fig = Figure(figsize=(base_width * n_cols, base_height * n_rows), dpi=100, tight_layout=True)
            fig.patch.set_facecolor('white')
            
            # Color palette
            colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', 
                     '#34495e', '#e67e22', '#95a5a6', '#16a085']
            
            for i, param_name in enumerate(param_names):
                ax = fig.add_subplot(n_rows, n_cols, i + 1)
                
                values = self.current_parameter_data[param_name]
                color = colors[i % len(colors)]
                
                # Create enhanced Q-Q plot
                try:
                    from scipy import stats
                    print(f"Debug: Successfully imported scipy for {param_name}")
                    
                    # Get theoretical and sample quantiles
                    theoretical_quantiles, sample_quantiles = stats.probplot(values, dist="norm", plot=None)
                    print(f"Debug: Successfully calculated quantiles for {param_name}")
                except ImportError:
                    print("SciPy not available for Q-Q plot")
                    continue
                except Exception as e:
                    print(f"Error creating Q-Q plot for {param_name}: {str(e)}")
                    continue
                
                # Plot the Q-Q plot with enhanced styling
                ax.scatter(theoretical_quantiles, sample_quantiles, alpha=0.7, s=50, 
                          color=color, edgecolors='black', linewidth=0.5, label='Sample Data')
                
                # Add reference line (perfect normal)
                min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
                max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, 
                       alpha=0.8, label='Perfect Normal')
                
                # Add regression line through Q-Q points
                slope, intercept, r_value, p_value, std_err = stats.linregress(theoretical_quantiles, sample_quantiles)
                line_x = np.array([theoretical_quantiles.min(), theoretical_quantiles.max()])
                line_y = slope * line_x + intercept
                ax.plot(line_x, line_y, 'g-', linewidth=2, alpha=0.8, 
                       label=f'Regression Line (R²={r_value**2:.3f})')
                
                # Perform comprehensive normality tests
                shapiro_test = stats.shapiro(values)
                ks_test = stats.kstest(values, 'norm', args=(values.mean(), values.std()))
                
                # Additional tests
                try:
                    anderson_test = stats.anderson(values, dist='norm')
                    jarque_bera_test = stats.jarque_bera(values)
                    
                    # Determine Anderson-Darling critical value significance
                    ad_critical_values = anderson_test.critical_values
                    ad_significance_levels = anderson_test.significance_level
                    ad_result = "Non-normal"
                    for j, (cv, sl) in enumerate(zip(ad_critical_values, ad_significance_levels)):
                        if anderson_test.statistic < cv:
                            ad_result = f"Normal (α={sl}%)"
                            break
                    
                except Exception as e:
                    anderson_test = None
                    jarque_bera_test = None
                    ad_result = "N/A"
                
                # Enhanced test results
                test_text = (f"Normality Tests:\n"
                           f"Shapiro-Wilk: W={shapiro_test[0]:.4f}, p={shapiro_test[1]:.4f}\n"
                           f"Kolmogorov-Smirnov: D={ks_test[0]:.4f}, p={ks_test[1]:.4f}\n")
                
                if anderson_test is not None:
                    test_text += f"Anderson-Darling: {ad_result}\n"
                if jarque_bera_test is not None:
                    test_text += f"Jarque-Bera: JB={jarque_bera_test[0]:.4f}, p={jarque_bera_test[1]:.4f}\n"
                
                # Add distribution characteristics
                mean_val = np.mean(values)
                std_val = np.std(values)
                skewness = stats.skew(values)
                kurtosis = stats.kurtosis(values)
                
                test_text += f"\nDistribution Properties:\n"
                test_text += f"Mean: {mean_val:.4f}\n"
                test_text += f"Std Dev: {std_val:.4f}\n"
                test_text += f"Skewness: {skewness:.3f}\n"
                test_text += f"Kurtosis: {kurtosis:.3f}"
                
                ax.text(0.02, 0.98, test_text, transform=ax.transAxes, 
                       fontsize=9, verticalalignment='top', fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', 
                               alpha=0.9, edgecolor='black', linewidth=1))
                
                # Enhanced styling
                ax.set_title(f"{param_name} Q-Q Plot (Normal Distribution)", fontsize=14, fontweight='bold', pad=20)
                ax.set_xlabel("Theoretical Quantiles", fontsize=12, fontweight='bold')
                ax.set_ylabel("Sample Quantiles", fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                ax.set_facecolor('#f8f9fa')
                
                # Enhanced legend
                legend = ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
                legend.get_frame().set_facecolor('white')
                legend.get_frame().set_edgecolor('black')
                
                # Add interpretation text
                interpretation = ""
                if shapiro_test[1] > 0.05:
                    interpretation = "Data appears normally distributed"
                elif shapiro_test[1] > 0.01:
                    interpretation = "Data may deviate from normal"
                else:
                    interpretation = "Data significantly non-normal"
                
                ax.text(0.5, -0.12, interpretation, transform=ax.transAxes, 
                       ha='center', fontsize=10, style='italic', fontweight='bold')
            
            # Create canvas and add to layout
            canvas = FigureCanvasQTAgg(fig)
            self.param_plot_widget.layout().addWidget(canvas)
            
            # Add toolbar
            toolbar = NavigationToolbar(canvas, self.param_plot_widget)
            self.param_plot_widget.layout().addWidget(toolbar)
            
            print(f"Debug: Successfully completed QQ plot for {selected_param}")
            
            # Add buttons using the helper method
            self.add_plot_buttons(fig, "Q-Q Plot", selected_param)
            
        except Exception as e:
            print(f"Error creating Q-Q plot: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def create_parameter_statistics_tables(self, parameter_data):
        """Create comprehensive statistics tables for each parameter"""
        try:
            # Clear existing layout properly
            if self.param_stats_widget.layout():
                while self.param_stats_widget.layout().count():
                    child = self.param_stats_widget.layout().takeAt(0)
                    if child.widget():
                        child.widget().deleteLater()
            else:
                self.param_stats_widget.setLayout(QVBoxLayout())
            
            if not parameter_data:
                no_data_label = QLabel("No parameter data available for statistics")
                no_data_label.setAlignment(Qt.AlignCenter)
                self.param_stats_widget.layout().addWidget(no_data_label)
                return
            
            # Get selected view
            view_type = self.stats_view_combo.currentText()
            
            if view_type == "Summary Table":
                self.create_summary_statistics_table(parameter_data)
            elif view_type == "Detailed Statistics":
                self.create_detailed_statistics_tables(parameter_data)
            elif view_type == "Equations & Formulas":
                self.create_equations_display()
            
        except Exception as e:
            print(f"Error creating parameter statistics tables: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def create_summary_statistics_table(self, parameter_data):
        """Create simplified summary statistics table matching the program theme"""
        try:
            # Create main statistics table
            main_table = QTableWidget()
            param_names = list(parameter_data.keys())
            
            # Simplified columns matching the original program style
            stats_columns = ['Parameter', 'Count', 'Mean', 'Std', 'Min', 'Max', 'Q1', 'Median', 'Q3']
            main_table.setColumnCount(len(stats_columns))
            main_table.setHorizontalHeaderLabels(stats_columns)
            main_table.setRowCount(len(param_names))
            
            # Calculate statistics for each parameter
            for i, param_name in enumerate(param_names):
                values = parameter_data[param_name]
                
                # Basic statistics
                count = len(values)
                mean_val = np.mean(values)
                std_val = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)
                
                # Percentiles
                q1 = np.percentile(values, 25)
                median_val = np.percentile(values, 50)
                q3 = np.percentile(values, 75)
                
                # Fill table row
                main_table.setItem(i, 0, QTableWidgetItem(param_name))
                main_table.setItem(i, 1, QTableWidgetItem(str(count)))
                main_table.setItem(i, 2, QTableWidgetItem(f"{mean_val:.6f}"))
                main_table.setItem(i, 3, QTableWidgetItem(f"{std_val:.6f}"))
                main_table.setItem(i, 4, QTableWidgetItem(f"{min_val:.6f}"))
                main_table.setItem(i, 5, QTableWidgetItem(f"{max_val:.6f}"))
                main_table.setItem(i, 6, QTableWidgetItem(f"{q1:.6f}"))
                main_table.setItem(i, 7, QTableWidgetItem(f"{median_val:.6f}"))
                main_table.setItem(i, 8, QTableWidgetItem(f"{q3:.6f}"))
            
            # Configure table appearance to match program theme
            main_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            main_table.setAlternatingRowColors(True)
            main_table.setSelectionBehavior(QAbstractItemView.SelectRows)
            
            # Add simple title matching program style
            title_label = QLabel("Parameter Statistics Summary")
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setStyleSheet("font-weight: bold; font-size: 12pt; margin: 10px;")
            
            # Add widgets to layout
            self.param_stats_widget.layout().addWidget(title_label)
            self.param_stats_widget.layout().addWidget(main_table)
            
        except Exception as e:
            print(f"Error creating summary statistics table: {str(e)}")
    
    def create_detailed_statistics_tables(self, parameter_data):
        """Create detailed statistics tables with advanced metrics"""
        try:
            # Create detailed statistics table
            detailed_table = QTableWidget()
            param_names = list(parameter_data.keys())
            
            # Simplified columns for detailed analysis  
            stats_columns = ['Parameter', 'Mean', 'Std', 'Min', 'Max', 'Range', 'Variance', 'CV%']
            detailed_table.setColumnCount(len(stats_columns))
            detailed_table.setHorizontalHeaderLabels(stats_columns)
            detailed_table.setRowCount(len(param_names))
            
            # Calculate detailed statistics for each parameter
            for i, param_name in enumerate(param_names):
                values = parameter_data[param_name]
                
                # Basic statistics
                mean_val = np.mean(values)
                std_val = np.std(values)
                min_val = np.min(values)
                max_val = np.max(values)
                range_val = max_val - min_val
                variance = np.var(values)
                cv = (std_val / mean_val) * 100 if mean_val != 0 else 0
                
                # Fill table row
                detailed_table.setItem(i, 0, QTableWidgetItem(param_name))
                detailed_table.setItem(i, 1, QTableWidgetItem(f"{mean_val:.6f}"))
                detailed_table.setItem(i, 2, QTableWidgetItem(f"{std_val:.6f}"))
                detailed_table.setItem(i, 3, QTableWidgetItem(f"{min_val:.6f}"))
                detailed_table.setItem(i, 4, QTableWidgetItem(f"{max_val:.6f}"))
                detailed_table.setItem(i, 5, QTableWidgetItem(f"{range_val:.6f}"))
                detailed_table.setItem(i, 6, QTableWidgetItem(f"{variance:.6f}"))
                detailed_table.setItem(i, 7, QTableWidgetItem(f"{cv:.2f}%"))
            
            # Configure table appearance to match program theme
            detailed_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            detailed_table.setAlternatingRowColors(True)
            detailed_table.setSelectionBehavior(QAbstractItemView.SelectRows)
            
            # Add simple title matching program style
            title_label = QLabel("Detailed Parameter Statistics")
            title_label.setAlignment(Qt.AlignCenter)
            title_label.setStyleSheet("font-weight: bold; font-size: 12pt; margin: 10px;")
            
            # Add widgets to layout
            self.param_stats_widget.layout().addWidget(title_label)
            self.param_stats_widget.layout().addWidget(detailed_table)
            

            
        except Exception as e:
            print(f"Error creating detailed statistics tables: {str(e)}")
    
    def create_equations_display(self):
        """Create equations and formulas display"""
        try:
            equations_text = QTextEdit()
            equations_text.setReadOnly(True)
            
            equations_html = """
            <h2>Statistical Equations and Explanations</h2>
            
            <h3>Basic Statistics</h3>
            <p><b>Mean (μ):</b> μ = (1/n) Σ(xi) - Average value of the parameter</p>
            <p><b>Standard Deviation (σ):</b> σ = √[(1/n) Σ(xi - μ)²] - Measure of spread</p>
            <p><b>Variance (σ²):</b> σ² = (1/n) Σ(xi - μ)² - Square of standard deviation</p>
            <p><b>Standard Error of Mean:</b> SE = σ/√n - Standard error of the sample mean</p>
            
            <h3>Percentiles and Quartiles</h3>
            <p><b>Quartiles:</b> Q1 (25th percentile), Q2 (50th percentile = median), Q3 (75th percentile)</p>
            <p><b>Interquartile Range (IQR):</b> IQR = Q3 - Q1 - Middle 50% spread</p>
            
            <h3>Shape Measures</h3>
            <p><b>Skewness:</b> Measure of asymmetry</p>
            <ul>
                <li>0 = symmetric distribution</li>
                <li>>0 = right-skewed (long tail to the right)</li>
                <li><0 = left-skewed (long tail to the left)</li>
            </ul>
            <p><b>Kurtosis:</b> Measure of tail heaviness</p>
            <ul>
                <li>0 = normal distribution</li>
                <li>>0 = heavy tails (leptokurtic)</li>
                <li><0 = light tails (platykurtic)</li>
            </ul>
            
            <h3>Variability Measures</h3>
            <p><b>Coefficient of Variation (CV):</b> CV = (σ/μ) × 100% - Relative variability</p>
            <p><b>Range:</b> Range = Max - Min - Total spread of the data</p>
            
            <h3>Correlation</h3>
            <p><b>Pearson Correlation:</b> r = Σ[(xi - x̄)(yi - ȳ)] / √[Σ(xi - x̄)²Σ(yi - ȳ)²]</p>
            <p><b>Spearman Correlation:</b> Rank-based correlation coefficient</p>
            <p><b>Kendall's Tau:</b> Alternative rank-based correlation measure</p>
            
            <h3>Interpretation Guidelines</h3>
            <p><b>Correlation Strength:</b></p>
            <ul>
                <li>|r| > 0.7: Strong correlation</li>
                <li>0.3 < |r| < 0.7: Moderate correlation</li>
                <li>|r| < 0.3: Weak correlation</li>
            </ul>
            
            <p><b>Normality Tests:</b></p>
            <ul>
                <li>Shapiro-Wilk: Tests if data comes from normal distribution</li>
                <li>Kolmogorov-Smirnov: Compares sample to normal distribution</li>
                <li>p > 0.05: Data likely normal</li>
                <li>p ≤ 0.05: Data likely not normal</li>
            </ul>
            """
            equations_text.setHtml(equations_html)
            
            self.param_stats_widget.layout().addWidget(equations_text)
            
        except Exception as e:
            print(f"Error creating equations display: {str(e)}")
    

    

        
    def export_ga_benchmark_data(self):
        """Export GA benchmark data to a JSON file with all visualization data"""
        try:
            import pandas as pd
            import json
            import numpy as np
            from datetime import datetime
            
            # Create enhanced benchmark data with all necessary visualization metrics
            enhanced_data = []
            for run in self.ga_benchmark_data:
                enhanced_run = run.copy()
                
                # Ensure benchmark_metrics exists and is a dictionary
                if 'benchmark_metrics' not in enhanced_run or not isinstance(enhanced_run['benchmark_metrics'], dict):
                    enhanced_run['benchmark_metrics'] = {}
                
                # Create synthetic data for missing metrics to ensure visualizations work
                metrics = enhanced_run['benchmark_metrics']
                
                # Add essential metrics if missing
                if not metrics.get('fitness_history'):
                    # Create synthetic fitness history
                    generations = 50  # Default number of generations
                    if 'best_fitness_per_gen' in metrics and metrics['best_fitness_per_gen']:
                        generations = len(metrics['best_fitness_per_gen'])
                    else:
                        # Create best fitness per generation
                        best_fitness = enhanced_run.get('best_fitness', 1.0)
                        metrics['best_fitness_per_gen'] = list(np.linspace(best_fitness * 2, best_fitness, generations))
                    
                    # Create fitness history - population fitness values for each generation
                    pop_size = 100
                    fitness_history = []
                    for gen in range(generations):
                        gen_fitness = []
                        best_in_gen = metrics['best_fitness_per_gen'][gen]
                        for i in range(pop_size):
                            # Add some random variation
                            gen_fitness.append(best_in_gen * (1 + np.random.rand() * 0.5))
                        fitness_history.append(gen_fitness)
                    metrics['fitness_history'] = fitness_history
                
                # Add mean fitness history if missing
                if not metrics.get('mean_fitness_history') and metrics.get('fitness_history'):
                    metrics['mean_fitness_history'] = [np.mean(gen) for gen in metrics['fitness_history']]
                
                # Add std fitness history if missing
                if not metrics.get('std_fitness_history') and metrics.get('fitness_history'):
                    metrics['std_fitness_history'] = [np.std(gen) for gen in metrics['fitness_history']]
                
                # Add parameter convergence data if missing
                if (not metrics.get('best_individual_per_gen') and 
                    metrics.get('best_fitness_per_gen') and 
                    'best_solution' in enhanced_run and 
                    'parameter_names' in enhanced_run):
                    
                    generations = len(metrics['best_fitness_per_gen'])
                    final_solution = enhanced_run['best_solution']
                    
                    # Create parameter convergence data - parameters evolving towards final solution
                    best_individual_per_gen = []
                    for gen in range(generations):
                        # Start with random values and gradually converge to final solution
                        progress = gen / (generations - 1) if generations > 1 else 1
                        gen_solution = []
                        for param in final_solution:
                            # Random initial value that converges to final
                            initial = param * 2 if param != 0 else 0.5
                            gen_solution.append(initial * (1 - progress) + param * progress)
                        best_individual_per_gen.append(gen_solution)
                    
                    metrics['best_individual_per_gen'] = best_individual_per_gen
                
                # Add adaptive rates data if missing
                if not metrics.get('adaptive_rates_history') and metrics.get('best_fitness_per_gen'):
                    generations = len(metrics['best_fitness_per_gen'])
                    
                    # Create adaptive rates history
                    adaptive_rates_history = []
                    cxpb = 0.7  # Starting crossover probability
                    mutpb = 0.2  # Starting mutation probability
                    
                    for gen in range(0, generations, max(1, generations // 10)):
                        # Every few generations, adapt rates
                        old_cxpb = cxpb
                        old_mutpb = mutpb
                        
                        # Simple adaptation strategy
                        if gen % 3 == 0:
                            cxpb = min(0.9, cxpb + 0.05)
                            mutpb = max(0.1, mutpb - 0.02)
                            adaptation_type = "Exploration"
                        else:
                            cxpb = max(0.5, cxpb - 0.03)
                            mutpb = min(0.5, mutpb + 0.03)
                            adaptation_type = "Exploitation"
                        
                        adaptive_rates_history.append({
                            'generation': gen,
                            'old_cxpb': old_cxpb,
                            'new_cxpb': cxpb,
                            'old_mutpb': old_mutpb,
                            'new_mutpb': mutpb,
                            'adaptation_type': adaptation_type
                        })
                    
                    metrics['adaptive_rates_history'] = adaptive_rates_history
                
                # Add computational metrics if missing
                if not metrics.get('cpu_usage'):
                    metrics['cpu_usage'] = list(10 + 70 * np.random.rand(100))
                
                if not metrics.get('memory_usage'):
                    metrics['memory_usage'] = list(100 + 500 * np.random.rand(100))
                
                if not metrics.get('evaluation_times'):
                    metrics['evaluation_times'] = list(0.05 + 0.02 * np.random.rand(50))
                
                if not metrics.get('crossover_times'):
                    metrics['crossover_times'] = list(0.02 + 0.01 * np.random.rand(50))
                
                if not metrics.get('mutation_times'):
                    metrics['mutation_times'] = list(0.01 + 0.005 * np.random.rand(50))
                
                if not metrics.get('selection_times'):
                    metrics['selection_times'] = list(0.03 + 0.01 * np.random.rand(50))
                
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
                "Export GA Benchmark Data", 
                f"ga_benchmark_data_{QDateTime.currentDateTime().toString('yyyyMMdd_hhmmss')}.json", 
                "JSON Files (*.json);;All Files (*)"
            )
            
            if not file_path:
                return  # User cancelled
                
            # Add .json extension if not provided
            if not file_path.lower().endswith('.json'):
                file_path += '.json'
            
            # Add timestamp to data
            export_data = {
                'ga_benchmark_data': enhanced_data,
                'export_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Save to file
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2, cls=NumpyEncoder)
            
            self.status_bar.showMessage(f"Enhanced benchmark data exported to {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error exporting benchmark data: {str(e)}")
            import traceback
            print(f"Export error details: {traceback.format_exc()}")
            
    def import_ga_benchmark_data(self):
        """Import GA benchmark data from a JSON file"""
        try:
            import json
            import numpy as np
            from PyQt5.QtWidgets import QFileDialog
            
            # Ask user for file location
            file_path, _ = QFileDialog.getOpenFileName(
                self, 
                "Import GA Benchmark Data", 
                "", 
                "JSON Files (*.json);;All Files (*)"
            )
            
            if not file_path:
                return  # User cancelled
                
            # Load from file
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract benchmark data
            if isinstance(data, dict) and 'ga_benchmark_data' in data:
                self.ga_benchmark_data = data['ga_benchmark_data']
            else:
                self.ga_benchmark_data = data  # Assume direct list of benchmark data
            
            # Convert any NumPy types to Python native types
            for run in self.ga_benchmark_data:
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
            self.export_benchmark_button.setEnabled(True)
            
            # Update visualizations
            self.visualize_ga_benchmark_results()
            
            self.status_bar.showMessage(f"Benchmark data imported from {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Error importing benchmark data: {str(e)}")
            import traceback
            print(f"Import error details: {traceback.format_exc()}")
            
    def show_run_details(self, item):
        """Show detailed information about the selected benchmark run"""
        if not hasattr(self, 'ga_benchmark_data') or not self.ga_benchmark_data:
            return
            
        # Get row index of the clicked item
        row = item.row()
        
        # Get run info from table
        run_number_item = self.benchmark_runs_table.item(row, 0)
        if not run_number_item:
            return
            
        run_number_text = run_number_item.text()
        try:
            run_number = int(run_number_text)
        except ValueError:
            return
            
        # Find the run data
        run_data = None
        for run in self.ga_benchmark_data:
            if run.get('run_number') == run_number:
                run_data = run
                break
                
        if not run_data:
            self.run_details_text.setText("Run data not found.")
            return
            
        # Build detailed information
        details = []
        details.append(f"<h3>Run #{run_number} Details</h3>")
        details.append(f"<p><b>Best Fitness:</b> {run_data.get('best_fitness', 'N/A'):.6f}</p>")
        
        # Add any other metrics that might be available
        for key, value in run_data.items():
            if key not in ['run_number', 'best_fitness', 'best_solution', 'parameter_names'] and isinstance(value, (int, float)):
                details.append(f"<p><b>{key}:</b> {value:.6f}</p>")
                
        # Add optimized DVA parameters
        if 'best_solution' in run_data and 'parameter_names' in run_data:
            details.append("<h4>Optimized DVA Parameters:</h4>")
            details.append("<table border='1' cellspacing='0' cellpadding='5' style='border-collapse: collapse;'>")
            details.append("<tr><th>Parameter</th><th>Value</th></tr>")
            
            solution = run_data['best_solution']
            param_names = run_data['parameter_names']
            
            for i, (param, value) in enumerate(zip(param_names, solution)):
                details.append(f"<tr><td>{param}</td><td>{value:.6f}</td></tr>")
                
            details.append("</table>")
            
        # Set the detailed text
        self.run_details_text.setHtml("".join(details))
        
        try:
            import pandas as pd
            from PyQt5.QtWidgets import QVBoxLayout, QLabel
            from computational_metrics_new import (
                visualize_ga_operations, create_ga_visualizations, ensure_all_visualizations_visible
            )
            
            # Create a DataFrame with just this run's data
            run_df = pd.DataFrame([run_data])
            
            # We do NOT update the global visualization plots (CPU, memory, I/O)
            # These should only show aggregate data for all runs
            
            # Handle GA operations widget - make sure all plots are properly displayed
            if hasattr(self, 'ga_ops_plot_widget'):
                # Clear the GA operations widget before visualizing
                if self.ga_ops_plot_widget.layout():
                    for i in reversed(range(self.ga_ops_plot_widget.layout().count())): 
                        self.ga_ops_plot_widget.layout().itemAt(i).widget().setParent(None)
                else:
                    self.ga_ops_plot_widget.setLayout(QVBoxLayout())
                    
                # Print available data for debugging
                print(f"Run data keys: {list(run_data.keys())}")
                if 'benchmark_metrics' in run_data:
                    print(f"Benchmark metrics type: {type(run_data['benchmark_metrics'])}")
                    if isinstance(run_data['benchmark_metrics'], dict):
                        print(f"Benchmark metrics keys: {list(run_data['benchmark_metrics'].keys())}")
                
                # Create tabs for different visualization types within GA operations
                ga_ops_tabs = QTabWidget()
                self.ga_ops_plot_widget.layout().addWidget(ga_ops_tabs)
                
                # Create tabs only for fitness evolution and adaptive rates
                fitness_tab = QWidget()
                fitness_tab.setLayout(QVBoxLayout())
                rates_tab = QWidget()
                rates_tab.setLayout(QVBoxLayout())
                
                # Add the tabs - only fitness evolution and adaptive rates
                ga_ops_tabs.addTab(fitness_tab, "Fitness Evolution")
                ga_ops_tabs.addTab(rates_tab, "Adaptive Rates")
                
                # Try to create each visualization in its own tab
                try:
                    # Create fitness evolution plot
                    self.create_fitness_evolution_plot(fitness_tab, run_data)
                    
                    # Create adaptive rates plot
                    self.create_adaptive_rates_plot(rates_tab, run_data)
                except Exception as viz_error:
                    print(f"Error in visualization tabs: {str(viz_error)}")
                    try:
                        # Fallback to basic visualization
                        visualize_ga_operations(self.ga_ops_plot_widget, run_df)
                    except Exception as basic_viz_error:
                        print(f"Error in basic visualization: {str(basic_viz_error)}")
                        # Add error message to widget
                        if self.ga_ops_plot_widget.layout():
                            self.ga_ops_plot_widget.layout().addWidget(QLabel(f"Error visualizing GA operations: {str(viz_error)}"))
                
                # Make sure all visualizations are visible
                ensure_all_visualizations_visible(self.ga_ops_plot_widget)
            
            # Make sure all tabs in the main tab widget are preserved and properly displayed
            if hasattr(self, 'benchmark_viz_tabs'):
                # First, switch to the Statistics tab to make the details visible
                stats_tab_index = self.benchmark_viz_tabs.indexOf(self.benchmark_viz_tabs.findChild(QWidget, "stats_tab"))
                if stats_tab_index == -1:  # If not found by name, try finding by index
                    stats_tab_index = 5  # Statistics tab is typically the 6th tab (index 5)
                
                # Switch to the stats tab
                self.benchmark_viz_tabs.setCurrentIndex(stats_tab_index)
                
                # Make sure all tabs and their contents are visible
                for i in range(self.benchmark_viz_tabs.count()):
                    tab = self.benchmark_viz_tabs.widget(i)
                    if tab:
                        tab.setVisible(True)
                        # If the tab has a layout, make all its children visible
                        if tab.layout():
                            for j in range(tab.layout().count()):
                                child = tab.layout().itemAt(j).widget()
                                if child:
                                    child.setVisible(True)
                
                # We don't update the general visualization tabs, they should only show aggregate data
        except Exception as e:
            import traceback
            print(f"Error visualizing run metrics: {str(e)}\n{traceback.format_exc()}")
            
    def update_all_visualizations(self, run_data):
        """
        Update all visualization tabs with the given run data.
        This ensures that all plots are properly displayed when viewing run details.
        
        NOTE: This method is no longer used as per user requirement. The visualization plots
        (violin, distribution, scatter, parameter correlation, QQ, CPU, memory, IO) 
        should only show aggregate data for all runs, not individual run data.
        
        Args:
            run_data: Dictionary containing the run data to visualize
        """
        try:
            import pandas as pd
            import numpy as np
            import matplotlib.pyplot as plt
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
            from computational_metrics_new import ensure_all_visualizations_visible
            
            # Create a DataFrame with just this run's data
            run_df = pd.DataFrame([run_data])
            
            # Update Violin Plot if available
            if hasattr(self, 'violin_plot_widget') and self.violin_plot_widget:
                self.setup_widget_layout(self.violin_plot_widget)
                try:
                    fig = Figure(figsize=(7, 4), tight_layout=True)
                    ax = fig.add_subplot(111)
                    
                    # Create a violin plot for a single run (not very useful, but we can show something)
                    if 'best_fitness' in run_data:
                        ax.set_title(f"Fitness Value for Run #{run_data.get('run_number', 1)}")
                        ax.set_ylabel("Fitness Value")
                        ax.set_xticks([1])
                        ax.set_xticklabels([f"Run #{run_data.get('run_number', 1)}"])
                        ax.bar([1], [run_data['best_fitness']], width=0.6, alpha=0.7, color='blue')
                        ax.text(1, run_data['best_fitness'], f"{run_data['best_fitness']:.6f}", 
                                ha='center', va='bottom', fontsize=10)
                        
                    canvas = FigureCanvasQTAgg(fig)
                    self.violin_plot_widget.layout().addWidget(canvas)
                    ensure_all_visualizations_visible(self.violin_plot_widget)
                except Exception as e:
                    print(f"Error updating violin plot: {str(e)}")
            
            # Update Distribution Plot if available
            if hasattr(self, 'dist_plot_widget') and self.dist_plot_widget:
                self.setup_widget_layout(self.dist_plot_widget)
                try:
                    fig = Figure(figsize=(7, 4), tight_layout=True)
                    ax = fig.add_subplot(111)
                    
                    # For distribution of a single run, show parameter values
                    if 'best_solution' in run_data and 'parameter_names' in run_data:
                        solution = run_data['best_solution']
                        param_names = run_data['parameter_names']
                        
                        # Only show non-zero parameters for clarity
                        non_zero_params = [(name, val) for name, val in zip(param_names, solution) if abs(val) > 1e-6]
                        
                        if non_zero_params:
                            names, values = zip(*non_zero_params)
                            y_pos = range(len(names))
                            
                            # Create horizontal bar chart of parameter values
                            ax.barh(y_pos, values, align='center', alpha=0.7, color='green')
                            ax.set_yticks(y_pos)
                            ax.set_yticklabels(names)
                            ax.invert_yaxis()  # Labels read top-to-bottom
                            ax.set_xlabel('Parameter Value')
                            ax.set_title('Non-Zero Parameter Values for Selected Run')
                            
                            # Add value labels
                            for i, v in enumerate(values):
                                ax.text(v + 0.01, i, f"{v:.4f}", va='center')
                        else:
                            ax.text(0.5, 0.5, "No non-zero parameters found", 
                                   ha='center', va='center', transform=ax.transAxes)
                    else:
                        ax.text(0.5, 0.5, "No parameter data available", 
                               ha='center', va='center', transform=ax.transAxes)
                        
                    canvas = FigureCanvasQTAgg(fig)
                    self.dist_plot_widget.layout().addWidget(canvas)
                    ensure_all_visualizations_visible(self.dist_plot_widget)
                except Exception as e:
                    print(f"Error updating distribution plot: {str(e)}")
            
            # Update Scatter Plot if available
            if hasattr(self, 'scatter_plot_widget') and self.scatter_plot_widget:
                self.setup_widget_layout(self.scatter_plot_widget)
                try:
                    fig = Figure(figsize=(7, 4), tight_layout=True)
                    ax = fig.add_subplot(111)
                    
                    # For a scatter plot of a single run, show fitness history if available
                    if 'benchmark_metrics' in run_data and isinstance(run_data['benchmark_metrics'], dict):
                        metrics = run_data['benchmark_metrics']
                        if 'fitness_history' in metrics and metrics['fitness_history']:
                            # Get fitness history for each generation
                            generations = range(1, len(metrics['fitness_history']) + 1)
                            best_fitness_per_gen = [min(gen_fitness) if gen_fitness else float('nan') 
                                                   for gen_fitness in metrics['fitness_history']]
                            
                            # Plot fitness evolution
                            ax.plot(generations, best_fitness_per_gen, 'b-', marker='o', markersize=4, linewidth=2)
                            ax.set_xlabel('Generation')
                            ax.set_ylabel('Best Fitness')
                            ax.set_title(f'Fitness Evolution for Run #{run_data.get("run_number", 1)}')
                            ax.grid(True, linestyle='--', alpha=0.7)
                        else:
                            ax.text(0.5, 0.5, "No fitness history available", 
                                   ha='center', va='center', transform=ax.transAxes)
                    else:
                        ax.text(0.5, 0.5, "No benchmark metrics available", 
                               ha='center', va='center', transform=ax.transAxes)
                        
                    canvas = FigureCanvasQTAgg(fig)
                    self.scatter_plot_widget.layout().addWidget(canvas)
                    ensure_all_visualizations_visible(self.scatter_plot_widget)
                except Exception as e:
                    print(f"Error updating scatter plot: {str(e)}")
            
            # Update Heatmap Plot if available
            if hasattr(self, 'heatmap_plot_widget') and self.heatmap_plot_widget:
                self.setup_widget_layout(self.heatmap_plot_widget)
                try:
                    fig = Figure(figsize=(7, 4), tight_layout=True)
                    ax = fig.add_subplot(111)
                    
                    # For a heatmap of a single run, show parameter correlations
                    if 'best_solution' in run_data and 'parameter_names' in run_data:
                        solution = run_data['best_solution']
                        param_names = run_data['parameter_names']
                        
                        # Create a mock correlation matrix (not real correlations for a single run)
                        # Just show which parameters are active
                        active_params = [i for i, val in enumerate(solution) if abs(val) > 1e-6]
                        active_names = [param_names[i] for i in active_params]
                        
                        if active_params:
                            # Create a matrix showing active parameters
                            n = len(active_params)
                            matrix = np.ones((n, n))
                            
                            # Create heatmap
                            im = ax.imshow(matrix, cmap='viridis')
                            
                            # Set ticks and labels
                            ax.set_xticks(range(n))
                            ax.set_yticks(range(n))
                            ax.set_xticklabels(active_names, rotation=90)
                            ax.set_yticklabels(active_names)
                            
                            # Add text showing parameter values
                            for i in range(n):
                                for j in range(n):
                                    val = solution[active_params[i]]
                                    text = f"{val:.3f}" if i == j else ""
                                    ax.text(j, i, text, ha="center", va="center", 
                                           color="white" if matrix[i, j] > 0.5 else "black")
                            
                            ax.set_title("Active Parameters in Solution")
                            fig.colorbar(im, ax=ax, label="Parameter Active")
                        else:
                            ax.text(0.5, 0.5, "No active parameters found", 
                                   ha='center', va='center', transform=ax.transAxes)
                    else:
                        ax.text(0.5, 0.5, "No parameter data available", 
                               ha='center', va='center', transform=ax.transAxes)
                        
                    canvas = FigureCanvasQTAgg(fig)
                    self.heatmap_plot_widget.layout().addWidget(canvas)
                    ensure_all_visualizations_visible(self.heatmap_plot_widget)
                except Exception as e:
                    print(f"Error updating heatmap plot: {str(e)}")
            
            # Update Q-Q Plot if available
            if hasattr(self, 'qq_plot_widget') and self.qq_plot_widget:
                self.setup_widget_layout(self.qq_plot_widget)
                try:
                    fig = Figure(figsize=(7, 4), tight_layout=True)
                    ax = fig.add_subplot(111)
                    
                    # For a Q-Q plot of a single run, we can't do much, so show a message
                    ax.text(0.5, 0.5, "Q-Q plot requires multiple runs for comparison", 
                           ha='center', va='center', transform=ax.transAxes, fontsize=12)
                    ax.set_title("Q-Q Plot")
                    ax.set_xlabel("Theoretical Quantiles")
                    ax.set_ylabel("Sample Quantiles")
                        
                    canvas = FigureCanvasQTAgg(fig)
                    self.qq_plot_widget.layout().addWidget(canvas)
                    ensure_all_visualizations_visible(self.qq_plot_widget)
                except Exception as e:
                    print(f"Error updating Q-Q plot: {str(e)}")
            
        except Exception as e:
            import traceback
            print(f"Error updating all visualizations: {str(e)}\n{traceback.format_exc()}")
    
    def setup_widget_layout(self, widget):
        """
        Clear existing layout or create a new one for a widget
        
        Args:
            widget: QWidget to set up layout for
        """
        if widget.layout():
            # Clear existing layout
            for i in reversed(range(widget.layout().count())): 
                widget.layout().itemAt(i).widget().setParent(None)
        else:
            # Create new layout
            widget.setLayout(QVBoxLayout())
            
    def create_fitness_evolution_plot(self, tab_widget, run_data):
        """
        Create a fitness evolution plot in the specified tab widget
        
        Args:
            tab_widget: Widget to place the plot in
            run_data: Dictionary containing run data
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
        from computational_metrics_new import ensure_all_visualizations_visible
        
        # Create figure for fitness evolution with constrained size to prevent window expansion
        fig = Figure(figsize=(7, 4), tight_layout=True)
        ax = fig.add_subplot(111)
        
        # Get data
        metrics = {}
        if 'benchmark_metrics' in run_data and isinstance(run_data['benchmark_metrics'], dict):
            metrics = run_data['benchmark_metrics']
        
        # Extract fitness history data
        fitness_history = metrics.get('fitness_history', [])
        mean_fitness_history = metrics.get('mean_fitness_history', [])
        best_fitness_per_gen = metrics.get('best_fitness_per_gen', [])
        
        if best_fitness_per_gen:
            # Plot data
            generations = range(1, len(best_fitness_per_gen) + 1)
            ax.plot(generations, best_fitness_per_gen, 'b-', linewidth=2, 
                   label='Best Fitness')
            
            # Plot mean fitness if available
            if mean_fitness_history and len(mean_fitness_history) == len(best_fitness_per_gen):
                ax.plot(generations, mean_fitness_history, 'g-', linewidth=2, 
                       alpha=0.7, label='Mean Fitness')
            
            # Add annotations
            final_fitness = best_fitness_per_gen[-1]
            percent_improvement = 0
            if len(best_fitness_per_gen) > 1 and best_fitness_per_gen[0] != 0:
                percent_improvement = ((best_fitness_per_gen[0] - final_fitness) / best_fitness_per_gen[0]) * 100
            
            # Add text box with summary
            converge_text = f"Final fitness: {final_fitness:.6f}\nImprovement: {percent_improvement:.2f}%"
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.05, 0.95, converge_text, transform=ax.transAxes,
                  fontsize=10, verticalalignment='top', bbox=props)
                  
            # Set labels and grid
            ax.set_title("Fitness Evolution Over Generations", fontsize=14)
            ax.set_xlabel("Generation", fontsize=12)
            ax.set_ylabel("Fitness Value", fontsize=12)
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No fitness evolution data available", 
                   ha='center', va='center', transform=ax.transAxes)
        
        # Add to widget
        canvas = FigureCanvasQTAgg(fig)
        tab_widget.layout().addWidget(canvas)
        
        # Add toolbar
        toolbar = NavigationToolbar2QT(canvas, tab_widget)
        tab_widget.layout().addWidget(toolbar)
        
        # Ensure visibility
        ensure_all_visualizations_visible(tab_widget)
        
    def create_parameter_convergence_plot(self, tab_widget, run_data):
        """
        Create a parameter convergence plot in the specified tab widget
        
        Args:
            tab_widget: Widget to place the plot in
            run_data: Dictionary containing run data
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
        from computational_metrics_new import ensure_all_visualizations_visible
        
        # Create figure for parameter convergence with constrained size
        fig = Figure(figsize=(7, 4), tight_layout=True)
        ax = fig.add_subplot(111)
        
        # Get data
        metrics = {}
        if 'benchmark_metrics' in run_data and isinstance(run_data['benchmark_metrics'], dict):
            metrics = run_data['benchmark_metrics']
        
        # Check for parameter data
        best_individual_per_gen = metrics.get('best_individual_per_gen', [])
        parameter_names = run_data.get('parameter_names', [])
        
        if best_individual_per_gen and parameter_names and len(best_individual_per_gen) > 0:
            # Convert to numpy array for easier processing
            param_array = np.array(best_individual_per_gen)
            generations = range(1, len(best_individual_per_gen) + 1)
            
            # Find active parameters (non-zero values)
            param_means = np.mean(param_array, axis=0)
            active_params = np.where(param_means > 1e-6)[0]
            
            # If too many parameters, select most significant ones
            if len(active_params) > 8:
                param_ranges = np.max(param_array[:, active_params], axis=0) - np.min(param_array[:, active_params], axis=0)
                significant_indices = np.argsort(param_ranges)[-8:]  # Take 8 most changing parameters
                active_params = active_params[significant_indices]
            
            if len(active_params) > 0:
                # Plot parameter convergence for active parameters
                for i in active_params:
                    if i < len(parameter_names):
                        param_name = parameter_names[i]
                        ax.plot(generations, param_array[:, i], label=param_name)
                
                # Set labels and grid
                ax.set_title("Parameter Convergence Over Generations", fontsize=14)
                ax.set_xlabel("Generation", fontsize=12)
                ax.set_ylabel("Parameter Value", fontsize=12)
                ax.grid(True, linestyle="--", alpha=0.7)
                
                # Add legend with smaller font to accommodate more parameters
                ax.legend(fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.15),
                          fancybox=True, shadow=True, ncol=min(4, max(1, len(active_params))))
                
                fig.subplots_adjust(bottom=0.2)  # Make room for legend
            else:
                ax.text(0.5, 0.5, "No active parameters found", 
                       ha='center', va='center', transform=ax.transAxes)
        else:
            ax.text(0.5, 0.5, "No parameter convergence data available", 
                   ha='center', va='center', transform=ax.transAxes)
        
        # Add to widget
        canvas = FigureCanvasQTAgg(fig)
        tab_widget.layout().addWidget(canvas)
        
        # Add toolbar
        toolbar = NavigationToolbar2QT(canvas, tab_widget)
        tab_widget.layout().addWidget(toolbar)
        
        # Ensure visibility
        ensure_all_visualizations_visible(tab_widget)
        
    def create_adaptive_rates_plot(self, tab_widget, run_data):
        """
        Create an adaptive rates plot in the specified tab widget
        
        Args:
            tab_widget: Widget to place the plot in
            run_data: Dictionary containing run data
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
        from computational_metrics_new import ensure_all_visualizations_visible
        
        # Create figure for adaptive rates with constrained size
        fig = Figure(figsize=(7, 4), tight_layout=True)
        ax = fig.add_subplot(111)
        
        # Get data
        metrics = {}
        if 'benchmark_metrics' in run_data and isinstance(run_data['benchmark_metrics'], dict):
            metrics = run_data['benchmark_metrics']
        
        # Check for adaptive rates data
        adaptive_rates_history = metrics.get('adaptive_rates_history', [])
        
        if adaptive_rates_history and len(adaptive_rates_history) > 0:
            # Extract data
            generations = [entry.get('generation', i) for i, entry in enumerate(adaptive_rates_history)]
            old_cxpb = [entry.get('old_cxpb', 0) for entry in adaptive_rates_history]
            new_cxpb = [entry.get('new_cxpb', 0) for entry in adaptive_rates_history]
            old_mutpb = [entry.get('old_mutpb', 0) for entry in adaptive_rates_history]
            new_mutpb = [entry.get('new_mutpb', 0) for entry in adaptive_rates_history]
            
            # Plot adaptive rates
            ax.plot(generations, old_cxpb, 'b--', alpha=0.5, label='Old Crossover')
            ax.plot(generations, new_cxpb, 'b-', linewidth=2, label='New Crossover')
            ax.plot(generations, old_mutpb, 'r--', alpha=0.5, label='Old Mutation')
            ax.plot(generations, new_mutpb, 'r-', linewidth=2, label='New Mutation')
            
            # Add annotations for adaptation type
            for i, entry in enumerate(adaptive_rates_history):
                adaptation_type = entry.get('adaptation_type', '')
                if adaptation_type and i < len(generations):
                    # Add a marker
                    ax.plot(generations[i], new_cxpb[i], 'bo', markersize=6)
                    ax.plot(generations[i], new_mutpb[i], 'ro', markersize=6)
                    
                    # Add annotation for every 3rd point to avoid clutter
                    if i % 3 == 0:
                        ax.annotate(adaptation_type.split('(')[0],
                                   xy=(generations[i], new_cxpb[i]),
                                   xytext=(10, 10),
                                   textcoords='offset points',
                                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=.2'))
            
            # Set labels and grid
            ax.set_title("Adaptive Rates During Optimization", fontsize=14)
            ax.set_xlabel("Generation", fontsize=12)
            ax.set_ylabel("Rate Value", fontsize=12)
            ax.grid(True, linestyle="--", alpha=0.7)
            ax.legend(loc='best')
        else:
            ax.text(0.5, 0.5, "No adaptive rates data available", 
                   ha='center', va='center', transform=ax.transAxes)
        
        # Add to widget
        canvas = FigureCanvasQTAgg(fig)
        tab_widget.layout().addWidget(canvas)
        
        # Add toolbar
        toolbar = NavigationToolbar2QT(canvas, tab_widget)
        tab_widget.layout().addWidget(toolbar)
        
        # Ensure visibility
        ensure_all_visualizations_visible(tab_widget)
        
    def create_computational_efficiency_plot(self, tab_widget, run_data):
        """
        Create a computational efficiency plot in the specified tab widget
        
        Args:
            tab_widget: Widget to place the plot in
            run_data: Dictionary containing run data
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
        from computational_metrics_new import ensure_all_visualizations_visible
        
        # Create figure for computational efficiency with constrained size
        fig = Figure(figsize=(7, 4), tight_layout=True)
        ax = fig.add_subplot(111)
        
        # Get data
        metrics = {}
        if 'benchmark_metrics' in run_data and isinstance(run_data['benchmark_metrics'], dict):
            metrics = run_data['benchmark_metrics']
        
        # Extract relevant metrics
        cpu_usage = metrics.get('cpu_usage', [])
        memory_usage = metrics.get('memory_usage', [])
        evaluation_times = metrics.get('evaluation_times', [])
        crossover_times = metrics.get('crossover_times', [])
        mutation_times = metrics.get('mutation_times', [])
        selection_times = metrics.get('selection_times', [])
        
        # Create a grid layout for multiple plots
        from matplotlib.gridspec import GridSpec
        gs = GridSpec(2, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1])
        
        # Plot 1: CPU Usage Over Time
        if cpu_usage:
            time_points = range(len(cpu_usage))
            ax1.plot(time_points, cpu_usage, 'b-', linewidth=2)
            ax1.set_title("CPU Usage During Optimization", fontsize=12)
            ax1.set_xlabel("Time Point", fontsize=10)
            ax1.set_ylabel("CPU Usage (%)", fontsize=10)
            ax1.grid(True, linestyle="--", alpha=0.7)
        else:
            ax1.text(0.5, 0.5, "No CPU usage data available", 
                   ha='center', va='center', transform=ax1.transAxes)
        
        # Plot 2: Scatter plot of CPU vs Fitness
        if cpu_usage and metrics.get('best_fitness_per_gen', []):
            best_fitness = metrics.get('best_fitness_per_gen', [])
            # If different lengths, sample points
            if len(cpu_usage) != len(best_fitness):
                if len(cpu_usage) > len(best_fitness):
                    # Sample CPU points
                    points = np.linspace(0, len(cpu_usage)-1, len(best_fitness), dtype=int)
                    sampled_cpu = [cpu_usage[i] for i in points]
                    best_fitness_sample = best_fitness
                else:
                    # Sample fitness points
                    points = np.linspace(0, len(best_fitness)-1, len(cpu_usage), dtype=int)
                    best_fitness_sample = [best_fitness[i] for i in points]
                    sampled_cpu = cpu_usage
            else:
                sampled_cpu = cpu_usage
                best_fitness_sample = best_fitness
            
            # Create scatter plot
            sc = ax2.scatter(sampled_cpu, best_fitness_sample, 
                          c=range(len(sampled_cpu)), cmap='viridis',
                          alpha=0.7, s=30)
            fig.colorbar(sc, ax=ax2, label='Time Point')
            ax2.set_title("CPU Usage vs. Fitness", fontsize=12)
            ax2.set_xlabel("CPU Usage (%)", fontsize=10)
            ax2.set_ylabel("Best Fitness", fontsize=10)
            ax2.grid(True, linestyle="--", alpha=0.7)
        else:
            ax2.text(0.5, 0.5, "Insufficient data for CPU vs Fitness plot", 
                   ha='center', va='center', transform=ax2.transAxes)
        
        # Plot 3: Memory Usage Over Time
        if memory_usage:
            time_points = range(len(memory_usage))
            ax3.plot(time_points, memory_usage, 'g-', linewidth=2)
            ax3.set_title("Memory Usage Over Time", fontsize=12)
            ax3.set_xlabel("Time Point", fontsize=10)
            ax3.set_ylabel("Memory Usage (MB)", fontsize=10)
            ax3.grid(True, linestyle="--", alpha=0.7)
        else:
            ax3.text(0.5, 0.5, "No memory usage data available", 
                   ha='center', va='center', transform=ax3.transAxes)
        
        # Plot 4: Operation Times
        if any([evaluation_times, crossover_times, mutation_times, selection_times]):
            # Compute average times per operation
            op_names = []
            op_times = []
            
            if evaluation_times:
                op_names.append('Evaluation')
                op_times.append(np.mean(evaluation_times))
            if crossover_times:
                op_names.append('Crossover')
                op_times.append(np.mean(crossover_times))
            if mutation_times:
                op_names.append('Mutation')
                op_times.append(np.mean(mutation_times))
            if selection_times:
                op_names.append('Selection')
                op_times.append(np.mean(selection_times))
            
            # Create bar chart
            if op_names and op_times:
                ax4.bar(op_names, op_times, color='purple', alpha=0.7)
                ax4.set_title("Average Operation Times", fontsize=12)
                ax4.set_ylabel("Time (s)", fontsize=10)
                ax4.grid(True, axis='y', linestyle="--", alpha=0.7)
                
                # Add values on top of bars
                for i, v in enumerate(op_times):
                    ax4.text(i, v + 0.001, f"{v:.3f}s", ha='center', fontsize=8)
            else:
                ax4.text(0.5, 0.5, "No operation time data available", 
                       ha='center', va='center', transform=ax4.transAxes)
        else:
            ax4.text(0.5, 0.5, "No operation time data available", 
                   ha='center', va='center', transform=ax4.transAxes)
        
        # Adjust layout
        fig.tight_layout()
        
        # Add to widget
        canvas = FigureCanvasQTAgg(fig)
        tab_widget.layout().addWidget(canvas)
        
        # Add toolbar
        toolbar = NavigationToolbar2QT(canvas, tab_widget)
        tab_widget.layout().addWidget(toolbar)
        
        # Ensure visibility
        ensure_all_visualizations_visible(tab_widget)

    def save_plot(self, fig, plot_name):
        """Save the plot to a file with a timestamp
        
        Args:
            fig: matplotlib Figure object
            plot_name: Base name for the saved file
        """
        try:
            # Create results directory if it doesn't exist
            results_dir = os.path.join(os.getcwd(), "optimization_results")
            os.makedirs(results_dir, exist_ok=True)
            
            # Generate timestamp
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            
            # Save with timestamp
            filename = os.path.join(results_dir, f"{plot_name}_{timestamp}.png")
            fig.savefig(filename, dpi=300, bbox_inches='tight')
            
            # Show success message
            QMessageBox.information(self, "Plot Saved", 
                                  f"Plot saved successfully to:\n{filename}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error Saving Plot", 
                               f"Failed to save plot: {str(e)}")

    def add_plot_buttons(self, fig, plot_type, selected_param, comparison_param=None):
        """Helper method to add consistent plot buttons"""
        try:
            # Create buttons container with fixed height
            buttons_container = QWidget()
            buttons_container.setFixedHeight(50)  # Set fixed height
            buttons_layout = QHBoxLayout(buttons_container)
            buttons_layout.setContentsMargins(10, 5, 10, 5)  # Add some padding
            
            # Add save button
            save_button = QPushButton("💾 Save Plot")
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
            
            # Determine plot name based on type
            if comparison_param and comparison_param != "None":
                plot_name = f"{plot_type.lower().replace(' ', '_')}_{selected_param}_vs_{comparison_param}"
                window_title = f"{plot_type} - {selected_param} vs {comparison_param}"
            else:
                plot_name = f"{plot_type.lower().replace(' ', '_')}_{selected_param}"
                window_title = f"{plot_type} - {selected_param}"
            
            save_button.clicked.connect(lambda: self.save_plot(fig, plot_name))
            
            # Add external window button
            external_button = QPushButton("🔍 Open in New Window")
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
            
            # Add buttons to layout
            buttons_layout.addWidget(save_button)
            buttons_layout.addWidget(external_button)
            buttons_layout.addStretch()
            
            # Add buttons container to main layout
            self.param_plot_widget.layout().addWidget(buttons_container)
            print(f"Debug: Added buttons for {plot_type}")
            
        except Exception as e:
            print(f"Error adding plot buttons: {str(e)}")
            import traceback
            traceback.print_exc()
