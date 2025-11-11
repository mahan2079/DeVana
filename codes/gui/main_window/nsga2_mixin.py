from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QDoubleSpinBox,
    QFormLayout, QGroupBox, QPushButton, QTabWidget, QTextEdit, QProgressBar,
    QMessageBox, QScrollArea, QFileDialog, QTableWidget, QTableWidgetItem,
    QHeaderView, QSplitter
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import numpy as np
import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from workers.NSGA2Worker import NSGA2Worker

class NSGA2OptimizationMixin:
    def create_nsga2_tab(self):
        """Create the NSGA-II optimization tab with settings and results sub-tabs."""
        self.nsga2_tab = QWidget()
        main_layout = QVBoxLayout(self.nsga2_tab)

        self.nsga2_sub_tabs = QTabWidget()
        main_layout.addWidget(self.nsga2_sub_tabs)

        # --- Settings Sub-tab ---
        settings_tab = QWidget()
        settings_layout = QVBoxLayout(settings_tab)
        self.nsga2_sub_tabs.addTab(settings_tab, "Settings")

        # Scroll area for settings
        settings_scroll_area = QScrollArea()
        settings_scroll_area.setWidgetResizable(True)
        settings_container = QWidget()
        settings_layout.addWidget(settings_scroll_area)
        settings_scroll_area.setWidget(settings_container)
        settings_form_layout = QFormLayout(settings_container)

        # Population Size
        self.nsga2_pop_size_box = QSpinBox()
        self.nsga2_pop_size_box.setRange(10, 1000)
        self.nsga2_pop_size_box.setValue(100)
        settings_form_layout.addRow("Population Size (N):", self.nsga2_pop_size_box)

        # Generations
        self.nsga2_generations_box = QSpinBox()
        self.nsga2_generations_box.setRange(10, 10000)
        self.nsga2_generations_box.setValue(2000)
        settings_form_layout.addRow("Generations (G_max):", self.nsga2_generations_box)

        # Crossover Probability
        self.nsga2_cxpb_box = QDoubleSpinBox()
        self.nsga2_cxpb_box.setRange(0.0, 1.0)
        self.nsga2_cxpb_box.setSingleStep(0.01)
        self.nsga2_cxpb_box.setValue(0.9)
        settings_form_layout.addRow("Crossover Probability (p_c):", self.nsga2_cxpb_box)

        # Mutation Probability
        self.nsga2_mutpb_box = QDoubleSpinBox()
        self.nsga2_mutpb_box.setRange(0.0, 1.0)
        self.nsga2_mutpb_box.setSingleStep(0.001)
        self.nsga2_mutpb_box.setValue(1.0/48.0) # Approx 0.021
        settings_form_layout.addRow("Mutation Probability (p_m):", self.nsga2_mutpb_box)

        # SBX eta_c
        self.nsga2_eta_c_box = QSpinBox()
        self.nsga2_eta_c_box.setRange(1, 100)
        self.nsga2_eta_c_box.setValue(20)
        settings_form_layout.addRow("SBX eta_c:", self.nsga2_eta_c_box)

        # Polynomial Mutation eta_m
        self.nsga2_eta_m_box = QSpinBox()
        self.nsga2_eta_m_box.setRange(1, 100)
        self.nsga2_eta_m_box.setValue(20)
        settings_form_layout.addRow("Polynomial Mutation eta_m:", self.nsga2_eta_m_box)

        # Number of Runs (for statistical validity)
        self.nsga2_num_runs_box = QSpinBox()
        self.nsga2_num_runs_box.setRange(1, 100)
        self.nsga2_num_runs_box.setValue(1) # Default to 1 for single run
        settings_form_layout.addRow("Number of Runs:", self.nsga2_num_runs_box)

        # Random Seed
        self.nsga2_random_seed_box = QSpinBox()
        self.nsga2_random_seed_box.setRange(0, 99999)
        self.nsga2_random_seed_box.setValue(42)
        settings_form_layout.addRow("Initial Random Seed:", self.nsga2_random_seed_box)

        # Convergence Criteria Group
        convergence_group = QGroupBox("Convergence Criteria")
        convergence_layout = QFormLayout(convergence_group)
        
        self.nsga2_conv_epsilon_box = QDoubleSpinBox()
        self.nsga2_conv_epsilon_box.setRange(0.0, 1.0)
        self.nsga2_conv_epsilon_box.setSingleStep(0.0001)
        self.nsga2_conv_epsilon_box.setDecimals(5)
        self.nsga2_conv_epsilon_box.setValue(0.001)
        convergence_layout.addRow("Epsilon (HV change):", self.nsga2_conv_epsilon_box)

        self.nsga2_conv_window_box = QSpinBox()
        self.nsga2_conv_window_box.setRange(1, 200)
        self.nsga2_conv_window_box.setValue(50)
        convergence_layout.addRow("Window Size (generations):", self.nsga2_conv_window_box)

        self.nsga2_conv_min_gen_box = QSpinBox()
        self.nsga2_conv_min_gen_box.setRange(0, 1000)
        self.nsga2_conv_min_gen_box.setValue(500)
        convergence_layout.addRow("Min Generations before check:", self.nsga2_conv_min_gen_box)
        
        settings_form_layout.addRow(convergence_group)

        # HV Reference Point Group
        hv_ref_group = QGroupBox("Hypervolume Reference Point (f1, f2, f3)")
        hv_ref_layout = QFormLayout(hv_ref_group)
        
        self.nsga2_hv_ref_f1_box = QDoubleSpinBox()
        self.nsga2_hv_ref_f1_box.setRange(0.0, 1000.0) # Adjust range as needed for FRF
        self.nsga2_hv_ref_f1_box.setValue(1.0) # Assuming normalized FRF
        hv_ref_layout.addRow("f1 (FRF):", self.nsga2_hv_ref_f1_box)

        self.nsga2_hv_ref_f2_box = QDoubleSpinBox()
        self.nsga2_hv_ref_f2_box.setRange(0.0, 100.0) # Sparsity max 48*1.0 + 0.5*48 = 72
        self.nsga2_hv_ref_f2_box.setValue(72.0)
        hv_ref_layout.addRow("f2 (Sparsity):", self.nsga2_hv_ref_f2_box)

        self.nsga2_hv_ref_f3_box = QDoubleSpinBox()
        self.nsga2_hv_ref_f3_box.setRange(0.0, 100.0) # Cost max 48*1.0 = 48
        self.nsga2_hv_ref_f3_box.setValue(48.0)
        hv_ref_layout.addRow("f3 (Cost):", self.nsga2_hv_ref_f3_box)
        
        settings_form_layout.addRow(hv_ref_group)

        # Control Buttons
        control_buttons_layout = QHBoxLayout()
        self.nsga2_run_button = QPushButton("Run NSGA-II")
        self.nsga2_run_button.clicked.connect(self.run_nsga2)
        control_buttons_layout.addWidget(self.nsga2_run_button)

        self.nsga2_pause_button = QPushButton("Pause")
        self.nsga2_pause_button.clicked.connect(self.pause_nsga2)
        self.nsga2_pause_button.setEnabled(False)
        control_buttons_layout.addWidget(self.nsga2_pause_button)

        self.nsga2_resume_button = QPushButton("Resume")
        self.nsga2_resume_button.clicked.connect(self.resume_nsga2)
        self.nsga2_resume_button.setEnabled(False)
        control_buttons_layout.addWidget(self.nsga2_resume_button)

        self.nsga2_stop_button = QPushButton("Stop")
        self.nsga2_stop_button.clicked.connect(self.stop_nsga2)
        self.nsga2_stop_button.setEnabled(False)
        control_buttons_layout.addWidget(self.nsga2_stop_button)
        
        settings_form_layout.addRow(control_buttons_layout)

        # Progress Bar
        self.nsga2_progress_bar = QProgressBar()
        self.nsga2_progress_bar.setTextVisible(True)
        settings_form_layout.addRow("Progress:", self.nsga2_progress_bar)

        settings_layout.addStretch(1) # Push everything to the top

        # --- Results Sub-tab ---
        results_tab = QWidget()
        results_layout = QVBoxLayout(results_tab)
        self.nsga2_sub_tabs.addTab(results_tab, "Results")

        # Splitter for text results and plots
        results_splitter = QSplitter(Qt.Vertical)
        results_layout.addWidget(results_splitter)

        # Text Results Area
        text_results_group = QGroupBox("Summary Statistics")
        text_results_layout = QVBoxLayout(text_results_group)
        self.nsga2_results_text = QTextEdit()
        self.nsga2_results_text.setReadOnly(True)
        text_results_layout.addWidget(self.nsga2_results_text)
        results_splitter.addWidget(text_results_group)

        # Plots Area
        plots_group = QGroupBox("Visualizations")
        plots_layout = QVBoxLayout(plots_group)
        results_splitter.addWidget(plots_group)

        self.nsga2_plot_tabs = QTabWidget()
        plots_layout.addWidget(self.nsga2_plot_tabs)

        # Convergence Plot
        convergence_plot_widget = QWidget()
        convergence_plot_layout = QVBoxLayout(convergence_plot_widget)
        self.nsga2_convergence_fig = Figure(figsize=(10, 6))
        self.nsga2_convergence_canvas = FigureCanvas(self.nsga2_convergence_fig)
        self.nsga2_convergence_toolbar = NavigationToolbar(self.nsga2_convergence_canvas, convergence_plot_widget)
        convergence_plot_layout.addWidget(self.nsga2_convergence_toolbar)
        convergence_plot_layout.addWidget(self.nsga2_convergence_canvas)
        self.nsga2_plot_tabs.addTab(convergence_plot_widget, "Convergence")

        # 3D Pareto Front Plot
        pareto_3d_plot_widget = QWidget()
        pareto_3d_plot_layout = QVBoxLayout(pareto_3d_plot_widget)
        self.nsga2_pareto_3d_fig = Figure(figsize=(10, 6))
        self.nsga2_pareto_3d_canvas = FigureCanvas(self.nsga2_pareto_3d_fig)
        self.nsga2_pareto_3d_toolbar = NavigationToolbar(self.nsga2_pareto_3d_canvas, pareto_3d_plot_widget)
        pareto_3d_plot_layout.addWidget(self.nsga2_pareto_3d_toolbar)
        pareto_3d_plot_layout.addWidget(self.nsga2_pareto_3d_canvas)
        self.nsga2_plot_tabs.addTab(pareto_3d_plot_widget, "3D Pareto Front")

        # Box Plot/Violin Plot
        boxplot_widget = QWidget()
        boxplot_layout = QVBoxLayout(boxplot_widget)
        self.nsga2_boxplot_fig = Figure(figsize=(10, 6))
        self.nsga2_boxplot_canvas = FigureCanvas(self.nsga2_boxplot_fig)
        self.nsga2_boxplot_toolbar = NavigationToolbar(self.nsga2_boxplot_canvas, boxplot_widget)
        boxplot_layout.addWidget(self.nsga2_boxplot_toolbar)
        boxplot_layout.addWidget(self.nsga2_boxplot_canvas)
        self.nsga2_plot_tabs.addTab(boxplot_widget, "Metrics Distribution")

        # 2D Pareto Projections
        pareto_2d_widget = QWidget()
        pareto_2d_layout = QVBoxLayout(pareto_2d_widget)
        self.nsga2_pareto_2d_fig = Figure(figsize=(10, 6))
        self.nsga2_pareto_2d_canvas = FigureCanvas(self.nsga2_pareto_2d_fig)
        self.nsga2_pareto_2d_toolbar = NavigationToolbar(self.nsga2_pareto_2d_canvas, pareto_2d_widget)
        pareto_2d_layout.addWidget(self.nsga2_pareto_2d_toolbar)
        pareto_2d_layout.addWidget(self.nsga2_pareto_2d_canvas)
        self.nsga2_plot_tabs.addTab(pareto_2d_widget, "2D Pareto Projections")

        # Robustness Analysis
        robustness_widget = QWidget()
        robustness_layout = QVBoxLayout(robustness_widget)
        self.nsga2_robustness_fig = Figure(figsize=(10, 6))
        self.nsga2_robustness_canvas = FigureCanvas(self.nsga2_robustness_fig)
        self.nsga2_robustness_toolbar = NavigationToolbar(self.nsga2_robustness_canvas, robustness_widget)
        robustness_layout.addWidget(self.nsga2_robustness_toolbar)
        robustness_layout.addWidget(self.nsga2_robustness_canvas)
        self.nsga2_plot_tabs.addTab(robustness_widget, "Robustness Analysis")

        # Data Export Buttons
        export_buttons_layout = QHBoxLayout()
        self.nsga2_export_metrics_button = QPushButton("Export Metrics (CSV)")
        self.nsga2_export_metrics_button.clicked.connect(self.export_nsga2_metrics)
        export_buttons_layout.addWidget(self.nsga2_export_metrics_button)

        self.nsga2_export_pareto_button = QPushButton("Export Final Pareto (CSV)")
        self.nsga2_export_pareto_button.clicked.connect(self.export_nsga2_pareto)
        export_buttons_layout.addWidget(self.nsga2_export_pareto_button)
        
        plots_layout.addLayout(export_buttons_layout)

        self.nsga2_worker_thread = None
        self.nsga2_worker = None
        self.nsga2_all_runs_results = [] # Store results from all runs

        return self.nsga2_tab

    def run_nsga2(self):
        if self.nsga2_worker_thread and self.nsga2_worker_thread.isRunning():
            QMessageBox.warning(self, "NSGA-II", "NSGA-II is already running or paused.")
            return

        # Clear previous results
        self.nsga2_all_runs_results = []
        self.nsga2_results_text.clear()
        self.nsga2_progress_bar.setValue(0)
        self.nsga2_run_button.setEnabled(False)
        self.nsga2_pause_button.setEnabled(True)
        self.nsga2_stop_button.setEnabled(True)

        # Get parameters from UI
        pop_size = self.nsga2_pop_size_box.value()
        generations = self.nsga2_generations_box.value()
        cxpb = self.nsga2_cxpb_box.value()
        mutpb = self.nsga2_mutpb_box.value()
        eta_c = self.nsga2_eta_c_box.value()
        eta_m = self.nsga2_eta_m_box.value()
        num_runs = self.nsga2_num_runs_box.value()
        random_seed = self.nsga2_random_seed_box.value()
        convergence_epsilon = self.nsga2_conv_epsilon_box.value()
        convergence_window = self.nsga2_conv_window_box.value()
        convergence_min_gen = self.nsga2_conv_min_gen_box.value()
        hv_ref_point = [
            self.nsga2_hv_ref_f1_box.value(),
            self.nsga2_hv_ref_f2_box.value(),
            self.nsga2_hv_ref_f3_box.value()
        ]

        # Get system parameters from main window (assuming they are available)
        # This part needs to be carefully integrated with how MainWindow manages these inputs
        try:
            main_system_parameters = self.get_main_system_params()
            dva_parameters = self.get_dva_params()
            target_values, weights = self.get_target_values_weights()
            omega_start = self.omega_start_box.value()
            omega_end = self.omega_end_box.value()
            omega_points = self.omega_points_box.value()
        except AttributeError:
            QMessageBox.critical(self, "Error", "System parameters (main, DVA, targets, frequency) are not accessible. Ensure they are initialized.")
            self.reset_nsga2_buttons()
            return
        
        # Combine target_values and weights into a single structure for the worker
        target_values_weights = (target_values, weights)

        self.nsga2_worker_thread = QThread()
        self.nsga2_worker = NSGA2Worker(
            main_system_parameters=main_system_parameters,
            dva_parameters=dva_parameters,
            target_values_weights=target_values_weights,
            omega_start=omega_start,
            omega_end=omega_end,
            omega_points=omega_points,
            pop_size=pop_size,
            generations=generations,
            cxpb=cxpb,
            mutpb=mutpb,
            eta_c=eta_c,
            eta_m=eta_m,
            num_runs=num_runs,
            random_seed=random_seed,
            convergence_epsilon=convergence_epsilon,
            convergence_window=convergence_window,
            convergence_min_gen=convergence_min_gen,
            hv_ref_point=hv_ref_point,
            progress_callback=self.update_nsga2_progress,
            finished_callback=self.nsga2_finished,
            error_callback=self.nsga2_error
        )
        self.nsga2_worker.moveToThread(self.nsga2_worker_thread)
        self.nsga2_worker_thread.started.connect(self.nsga2_worker.run)
        self.nsga2_worker_thread.start()

    def pause_nsga2(self):
        if self.nsga2_worker:
            self.nsga2_worker.pause()
            self.nsga2_pause_button.setEnabled(False)
            self.nsga2_resume_button.setEnabled(True)

    def resume_nsga2(self):
        if self.nsga2_worker:
            self.nsga2_worker.resume()
            self.nsga2_pause_button.setEnabled(True)
            self.nsga2_resume_button.setEnabled(False)

    def stop_nsga2(self):
        if self.nsga2_worker:
            self.nsga2_worker.stop()
            self.nsga2_worker_thread.quit()
            self.nsga2_worker_thread.wait()
            self.reset_nsga2_buttons()
            QMessageBox.information(self, "NSGA-II", "NSGA-II optimization stopped.")

    def update_nsga2_progress(self, run_idx, current_gen, total_gens, metrics):
        total_progress = int(((run_idx * total_gens + current_gen) / (self.nsga2_num_runs_box.value() * total_gens)) * 100)
        self.nsga2_progress_bar.setValue(total_progress)
        
        # Update text results with latest generation metrics
        self.nsga2_results_text.append(f"Run {run_idx+1}, Gen {current_gen}/{total_gens}: HV={metrics.get('hv', 0.0):.4f}, IGD={metrics.get('igd', 0.0):.4f}, N_Pareto={metrics.get('n_pareto', 0)}")
        # Optionally, update plots in real-time or after certain intervals

    def nsga2_finished(self, all_runs_data):
        self.nsga2_all_runs_results = all_runs_data
        self.reset_nsga2_buttons()
        self.nsga2_progress_bar.setValue(100)
        QMessageBox.information(self, "NSGA-II", "NSGA-II optimization finished.")
        self.display_nsga2_results()

    def nsga2_error(self, message):
        self.reset_nsga2_buttons()
        QMessageBox.critical(self, "NSGA-II Error", message)

    def reset_nsga2_buttons(self):
        self.nsga2_run_button.setEnabled(True)
        self.nsga2_pause_button.setEnabled(False)
        self.nsga2_resume_button.setEnabled(False)
        self.nsga2_stop_button.setEnabled(False)

    def display_nsga2_results(self):
        if not self.nsga2_all_runs_results:
            self.nsga2_results_text.setText("No results to display.")
            return

        # --- Summary Statistics ---
        summary_text = "<h2>NSGA-II Optimization Results Summary</h2>"
        
        # Collect final metrics from all runs
        final_hvs = []
        final_igds = []
        final_gds = []
        final_spreads = []
        final_pareto_sizes = []
        final_times = []
        final_conv_gens = [] # Need to implement convergence detection in worker
        
        for run_data in self.nsga2_all_runs_results:
            if run_data["generation_metrics"]:
                last_gen_metrics = run_data["generation_metrics"][-1]
                final_hvs.append(last_gen_metrics.get('hv', 0.0))
                final_igds.append(last_gen_metrics.get('igd', 0.0))
                final_gds.append(last_gen_metrics.get('gd', 0.0))
                final_spreads.append(last_gen_metrics.get('spread', 0.0))
                final_pareto_sizes.append(last_gen_metrics.get('n_pareto', 0))
                final_times.append(sum(m.get('time_gen', 0.0) for m in run_data["generation_metrics"]) / 3600.0) # Convert to hours
                # Placeholder for convergence gen
                final_conv_gens.append(self.nsga2_generations_box.value()) # Assume full generations for now

        if final_hvs:
            summary_text += "<h3>Hypervolume (HV)</h3>"
            summary_text += f"Mean: {np.mean(final_hvs):.4f} ± {np.std(final_hvs):.4f}<br>"
            summary_text += f"Range: [{np.min(final_hvs):.4f}, {np.max(final_hvs):.4f}]<br>"
            
            summary_text += "<h3>IGD+</h3>"
            summary_text += f"Mean: {np.mean(final_igds):.4f} ± {np.std(final_igds):.4f}<br>"
            summary_text += f"Range: [{np.min(final_igds):.4f}, {np.max(final_igds):.4f}]<br>"

            summary_text += "<h3>Pareto Front Size</h3>"
            summary_text += f"Mean: {np.mean(final_pareto_sizes):.0f} ± {np.std(final_pareto_sizes):.0f}<br>"
            summary_text += f"Range: [{np.min(final_pareto_sizes):.0f}, {np.max(final_pareto_sizes):.0f}]<br>"

            summary_text += "<h3>Computational Time (hours)</h3>"
            summary_text += f"Mean: {np.mean(final_times):.2f} ± {np.std(final_times):.2f}<br>"
            summary_text += f"Range: [{np.min(final_times):.2f}, {np.max(final_times):.2f}]<br>"

        self.nsga2_results_text.setText(summary_text)

        # --- Visualizations ---
        self.plot_nsga2_convergence()
        self.plot_nsga2_pareto_3d()
        self.plot_nsga2_boxplot()
        self.plot_nsga2_pareto_2d()
        self.plot_nsga2_robustness()

    def plot_nsga2_convergence(self):
        self.nsga2_convergence_fig.clear()
        ax = self.nsga2_convergence_fig.add_subplot(111)
        
        if not self.nsga2_all_runs_results:
            ax.text(0.5, 0.5, "No data for convergence plot", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            self.nsga2_convergence_canvas.draw()
            return

        # Collect HV data for all runs
        hv_data_per_run = []
        for run_data in self.nsga2_all_runs_results:
            hv_data_per_run.append([m.get('hv', 0.0) for m in run_data["generation_metrics"]])
        
        if not hv_data_per_run or not hv_data_per_run[0]:
            ax.text(0.5, 0.5, "No HV data for convergence plot", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            self.nsga2_convergence_canvas.draw()
            return

        # Convert to numpy array for easier calculation
        hv_data_array = np.array(hv_data_per_run)
        
        # Calculate mean and std dev across runs for each generation
        mean_hv = np.mean(hv_data_array, axis=0)
        std_hv = np.std(hv_data_array, axis=0)
        generations_axis = np.arange(1, len(mean_hv) + 1)

        ax.plot(generations_axis, mean_hv, label='Mean Hypervolume', color='blue')
        ax.fill_between(generations_axis, mean_hv - std_hv, mean_hv + std_hv, color='blue', alpha=0.2, label='±1 Std Dev')
        
        ax.set_xlabel("Generation")
        ax.set_ylabel("Hypervolume")
        ax.set_title("Hypervolume Convergence")
        ax.legend()
        ax.grid(True)
        self.nsga2_convergence_canvas.draw()

    def plot_nsga2_pareto_3d(self):
        self.nsga2_pareto_3d_fig.clear()
        ax = self.nsga2_pareto_3d_fig.add_subplot(111, projection='3d')

        if not self.nsga2_all_runs_results:
            ax.text2D(0.5, 0.5, "No data for 3D Pareto front plot", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            self.nsga2_pareto_3d_canvas.draw()
            return
        
        # Use the final Pareto front from the first run for simplicity, or average/aggregate
        final_pareto_objectives = []
        if self.nsga2_all_runs_results:
            final_pareto_objectives = self.nsga2_all_runs_results[0].get("final_pareto_front_objectives", [])

        if not final_pareto_objectives:
            ax.text2D(0.5, 0.5, "No final Pareto objectives to plot", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            self.nsga2_pareto_3d_canvas.draw()
            return

        objectives_array = np.array(final_pareto_objectives)
        
        ax.scatter(objectives_array[:, 0], objectives_array[:, 1], objectives_array[:, 2], c='blue', marker='o')
        ax.set_xlabel("f1 (FRF)")
        ax.set_ylabel("f2 (Sparsity)")
        ax.set_zlabel("f3 (Cost)")
        ax.set_title("3D Pareto Front")
        self.nsga2_pareto_3d_canvas.draw()

    def plot_nsga2_boxplot(self):
        self.nsga2_boxplot_fig.clear()
        
        if not self.nsga2_all_runs_results or self.nsga2_num_runs_box.value() < 2:
            ax = self.nsga2_boxplot_fig.add_subplot(111)
            ax.text(0.5, 0.5, "Run multiple times for metrics distribution plots", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            self.nsga2_boxplot_canvas.draw()
            return

        # Collect final metrics for box plots
        final_hvs = [run_data["generation_metrics"][-1].get('hv', 0.0) for run_data in self.nsga2_all_runs_results if run_data["generation_metrics"]]
        final_igds = [run_data["generation_metrics"][-1].get('igd', 0.0) for run_data in self.nsga2_all_runs_results if run_data["generation_metrics"]]
        final_pareto_sizes = [run_data["generation_metrics"][-1].get('n_pareto', 0) for run_data in self.nsga2_all_runs_results if run_data["generation_metrics"]]
        final_times = [sum(m.get('time_gen', 0.0) for m in run_data["generation_metrics"]) / 3600.0 for run_data in self.nsga2_all_runs_results]

        metrics_data = {
            'HV': final_hvs,
            'IGD+': final_igds,
            'Pareto Size': final_pareto_sizes,
            'Time (hrs)': final_times
        }
        
        # Create 2x2 subplots
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, num=self.nsga2_boxplot_fig.number)
        axes = axes.flatten() # Flatten to easily iterate

        for i, (metric_name, data) in enumerate(metrics_data.items()):
            if data:
                sns.boxplot(y=data, ax=axes[i], color='skyblue')
                sns.swarmplot(y=data, ax=axes[i], color='blue', alpha=0.7) # Show individual points
                axes[i].set_title(f"{metric_name} Distribution")
                axes[i].set_ylabel(metric_name)
            else:
                axes[i].text(0.5, 0.5, "No data", horizontalalignment='center', verticalalignment='center', transform=axes[i].transAxes)
        
        plt.tight_layout()
        self.nsga2_boxplot_canvas.draw()

    def plot_nsga2_pareto_2d(self):
        self.nsga2_pareto_2d_fig.clear()
        
        if not self.nsga2_all_runs_results:
            ax = self.nsga2_pareto_2d_fig.add_subplot(111)
            ax.text(0.5, 0.5, "No data for 2D Pareto projections", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            self.nsga2_pareto_2d_canvas.draw()
            return

        final_pareto_objectives = []
        if self.nsga2_all_runs_results:
            final_pareto_objectives = self.nsga2_all_runs_results[0].get("final_pareto_front_objectives", [])

        if not final_pareto_objectives:
            ax = self.nsga2_pareto_2d_fig.add_subplot(111)
            ax.text(0.5, 0.5, "No final Pareto objectives to plot", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            self.nsga2_pareto_2d_canvas.draw()
            return

        objectives_array = np.array(final_pareto_objectives)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5), num=self.nsga2_pareto_2d_fig.number)

        # f1 vs f2
        axes[0].scatter(objectives_array[:, 0], objectives_array[:, 1], c='blue', marker='o', alpha=0.7)
        axes[0].set_xlabel("f1 (FRF)")
        axes[0].set_ylabel("f2 (Sparsity)")
        axes[0].set_title("Trade-off: FRF vs Sparsity")
        axes[0].grid(True)

        # f1 vs f3
        axes[1].scatter(objectives_array[:, 0], objectives_array[:, 2], c='blue', marker='o', alpha=0.7)
        axes[1].set_xlabel("f1 (FRF)")
        axes[1].set_ylabel("f3 (Cost)")
        axes[1].set_title("Trade-off: FRF vs Cost")
        axes[1].grid(True)

        # f2 vs f3
        axes[2].scatter(objectives_array[:, 1], objectives_array[:, 2], c='blue', marker='o', alpha=0.7)
        axes[2].set_xlabel("f2 (Sparsity)")
        axes[2].set_ylabel("f3 (Cost)")
        axes[2].set_title("Trade-off: Sparsity vs Cost")
        axes[2].grid(True)

        plt.tight_layout()
        self.nsga2_pareto_2d_canvas.draw()

    def plot_nsga2_robustness(self):
        self.nsga2_robustness_fig.clear()
        ax = self.nsga2_robustness_fig.add_subplot(111)

        if not self.nsga2_all_runs_results or self.nsga2_num_runs_box.value() < 2:
            ax.text(0.5, 0.5, "Run multiple times for robustness analysis", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            self.nsga2_robustness_canvas.draw()
            return

        for i, run_data in enumerate(self.nsga2_all_runs_results):
            hvs = [m.get('hv', 0.0) for m in run_data["generation_metrics"]]
            generations_axis = np.arange(1, len(hvs) + 1)
            ax.plot(generations_axis, hvs, color='blue', alpha=0.3, linewidth=0.8)
        
        # Plot mean HV
        hv_data_per_run = []
        for run_data in self.nsga2_all_runs_results:
            hv_data_per_run.append([m.get('hv', 0.0) for m in run_data["generation_metrics"]])
        hv_data_array = np.array(hv_data_per_run)
        mean_hv = np.mean(hv_data_array, axis=0)
        generations_axis = np.arange(1, len(mean_hv) + 1)
        ax.plot(generations_axis, mean_hv, label='Mean Hypervolume', color='darkblue', linewidth=2)

        ax.set_xlabel("Generation")
        ax.set_ylabel("Hypervolume")
        ax.set_title("Robustness Analysis: HV Progression Across Runs")
        ax.legend()
        ax.grid(True)
        self.nsga2_robustness_canvas.draw()

    def export_nsga2_metrics(self):
        if not self.nsga2_all_runs_results:
            QMessageBox.warning(self, "Export Data", "No metrics data to export.")
            return
        
        path, _ = QFileDialog.getSaveFileName(self, "Export NSGA-II Metrics", "nsga2_metrics.csv", "CSV Files (*.csv)")
        if path:
            all_metrics_df = pd.DataFrame()
            for run_data in self.nsga2_all_runs_results:
                run_df = pd.DataFrame(run_data["generation_metrics"])
                run_df['run_id'] = run_data['run_id']
                all_metrics_df = pd.concat([all_metrics_df, run_df], ignore_index=True)
            
            all_metrics_df.to_csv(path, index=False)
            QMessageBox.information(self, "Export Data", f"Metrics exported to {path}")

    def export_nsga2_pareto(self):
        if not self.nsga2_all_runs_results:
            QMessageBox.warning(self, "Export Data", "No Pareto front data to export.")
            return
        
        path, _ = QFileDialog.getSaveFileName(self, "Export NSGA-II Final Pareto Front", "nsga2_final_pareto.csv", "CSV Files (*.csv)")
        if path:
            all_pareto_df = pd.DataFrame()
            for run_data in self.nsga2_all_runs_results:
                objectives = run_data.get("final_pareto_front_objectives", [])
                parameters = run_data.get("final_population_parameters", [])
                
                if objectives and parameters:
                    df_obj = pd.DataFrame(objectives, columns=[f'f{i+1}' for i in range(len(objectives[0]))])
                    df_param = pd.DataFrame(parameters, columns=[f'param_{i+1}' for i in range(len(parameters[0]))])
                    
                    run_df = pd.concat([df_obj, df_param], axis=1)
                    run_df['run_id'] = run_data['run_id']
                    all_pareto_df = pd.concat([all_pareto_df, run_df], ignore_index=True)
            
            all_pareto_df.to_csv(path, index=False)
            QMessageBox.information(self, "Export Data", f"Final Pareto fronts exported to {path}")