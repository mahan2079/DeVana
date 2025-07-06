from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QTabWidget, QTextEdit,
    QPushButton, QComboBox, QLabel, QSpinBox, QLineEdit, QFileDialog,
    QMessageBox, QScrollArea, QGroupBox, QTableWidget, QTableWidgetItem, QCheckBox,
    QDoubleSpinBox, QHeaderView, QAbstractItemView
)
from PyQt5.QtCore import Qt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

from modules.sobol_sensitivity import (
    perform_sobol_analysis,
    calculate_and_save_errors,
    format_parameter_name,
    save_results,
)
from workers.SobolWorker import SobolWorker

# Set Seaborn style for better aesthetics
sns.set(style="whitegrid")
plt.rc('text', usetex=True)  # Use LaTeX for rendering text in plots

class SobolAnalysisMixin:

    def _run_sobol_implementation(self):
        """Run Sobol sensitivity analysis - main implementation"""
        if self.omega_start_box.value() >= self.omega_end_box.value():
            QMessageBox.warning(self, "Input Error", "Ω Start must be less than Ω End.")
            return

        # Get required parameters
        target_values, weights = self.get_target_values_weights()
        num_samples_list = self.get_num_samples_list()
        n_jobs = self.n_jobs_spin.value()
        main_params = self.get_main_system_params()

        # Update UI to show analysis is running
        self.run_frf_button.setEnabled(False)
        self.run_sobol_button.setEnabled(False)
        self.run_ga_button.setEnabled(False)
        self.hyper_run_sobol_button.setEnabled(False)
        
        # Clear and update results text area
        self.sobol_results_text.clear()
        self.sobol_results_text.append("--- Running Sobol Sensitivity Analysis ---\n")
        self.status_bar.showMessage("Running Sobol Analysis...")

        # Get DVA parameter bounds from table
        dva_bounds = {}
        for row in range(self.sobol_param_table.rowCount()):
            param_name = self.sobol_param_table.item(row, 0).text()
            fixed_checkbox = self.sobol_param_table.cellWidget(row, 1)
            fixed_value = self.sobol_param_table.cellWidget(row, 2)
            lower_bound = self.sobol_param_table.cellWidget(row, 3)
            upper_bound = self.sobol_param_table.cellWidget(row, 4)
            
            if fixed_checkbox.isChecked():
                dva_bounds[param_name] = fixed_value.value()
            else:
                if lower_bound.value() < upper_bound.value():
                    dva_bounds[param_name] = (lower_bound.value(), upper_bound.value())

        # Define parameter order
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

        # Print sample size
        self.sobol_results_text.append(f"Sample sizes: {num_samples_list}")
        
        # Create and start worker
        try:
            self.sobol_worker = SobolWorker(
                main_params=main_params,
                dva_bounds=dva_bounds,
                dva_order=original_dva_parameter_order,
                omega_start=self.omega_start_box.value(),
                omega_end=self.omega_end_box.value(),
                omega_points=self.omega_points_box.value(),
                num_samples_list=num_samples_list,
                target_values_dict=target_values,
                weights_dict=weights,
                n_jobs=n_jobs
            )
            
            # Connect signals
            self.sobol_worker.finished.connect(self.display_sobol_results)
            self.sobol_worker.error.connect(self.handle_sobol_error)
            
            # Start the worker thread
            self.sobol_worker.start()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start Sobol analysis: {str(e)}")
            self.run_frf_button.setEnabled(True)
            self.run_sobol_button.setEnabled(True)
            self.run_ga_button.setEnabled(True)
            self.hyper_run_sobol_button.setEnabled(True)
            self.status_bar.showMessage("Sobol analysis failed to start")

    def run_sobol(self):
        """Run Sobol sensitivity analysis"""
        try:
            # Clear previous results
            self.sobol_results_text.clear()
            self.sobol_plots = {}
            self.sobol_combo.clear()
            
            # Run the implementation
            self._run_sobol_implementation()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to run Sobol analysis: {str(e)}")
            self.status_bar.showMessage("Failed to run Sobol analysis")
        
    def get_num_samples_list(self):
        """Parse sample sizes from input line"""
        try:
            # Get the sample sizes string and split by comma
            samples_str = self.num_samples_line.text().strip()
            samples = [int(s.strip()) for s in samples_str.split(',')]
            
            # Validate sample sizes
            if not samples:
                raise ValueError("No sample sizes provided")
            if any(s <= 0 for s in samples):
                raise ValueError("Sample sizes must be positive integers")
            if len(samples) > 10:
                raise ValueError("Too many sample sizes (maximum 10)")
            
            # Sort sample sizes in ascending order
            samples.sort()
            return samples
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Invalid sample sizes: {str(e)}")
            return [32, 64, 128]  # Default values
            
    def handle_sobol_error(self, err):
        """Handle errors from the Sobol analysis worker thread"""
        # Re-enable buttons
        self.run_frf_button.setEnabled(True)
        self.run_sobol_button.setEnabled(True)
        self.run_ga_button.setEnabled(True)
        self.hyper_run_sobol_button.setEnabled(True)
        
        # Show error message
        QMessageBox.critical(self, "Error", f"Sobol analysis failed: {str(err)}")
        self.status_bar.showMessage("Sobol analysis failed")

    def display_sobol_results(self, all_results, warnings):
        """Display the results of the Sobol sensitivity analysis"""
        try:
            # Re-enable buttons
            self.run_frf_button.setEnabled(True)
            self.run_sobol_button.setEnabled(True)
            self.run_ga_button.setEnabled(True)
            self.hyper_run_sobol_button.setEnabled(True)
            
            # Update status
            self.status_bar.showMessage("Sobol analysis completed")
            
            # Display warnings if any
            if warnings:
                self.sobol_results_text.append("\nWarnings:")
                for warning in warnings:
                    self.sobol_results_text.append(f"- {warning}")
            
            # Get parameter names
            param_names = [
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
            
            # Display results for each sample size
            self.sobol_results_text.append("\nResults:")
            for i, N in enumerate(all_results['samples']):
                self.sobol_results_text.append(f"\nSample size N = {N}:")
                
                # Get S1 and ST indices for this sample size
                S1 = all_results['S1'][i]
                ST = all_results['ST'][i]
                
                # Sort parameters by total effect (ST)
                sorted_indices = np.argsort(ST)[::-1]
                sorted_params = [param_names[i] for i in sorted_indices]
                sorted_S1 = [S1[i] for i in sorted_indices]
                sorted_ST = [ST[i] for i in sorted_indices]
                
                # Display sorted results
                for param, s1, st in zip(sorted_params, sorted_S1, sorted_ST):
                    self.sobol_results_text.append(
                        f"{param:>12}: S1 = {s1:>8.4f}, ST = {st:>8.4f}"
                    )
            
            # Generate and display plots
            self.generate_sobol_plots(all_results, param_names)
            self.update_sobol_plot()
            self.sobol_canvas.draw()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to display Sobol results: {str(e)}")
            self.status_bar.showMessage("Failed to display Sobol results")

    def generate_sobol_plots(self, all_results, param_names):
        """
        This method prepares all the standard plots
        and adds them to self.sobol_plots so the user can pick them in the combo box.
        """
        fig_last_run = self.visualize_last_run(all_results, param_names)
        self.sobol_combo.addItem("Last Run Results")
        self.sobol_plots["Last Run Results"] = fig_last_run

        fig_grouped_ST = self.visualize_grouped_bar_plot_sorted_on_ST(all_results, param_names)
        self.sobol_combo.addItem("Grouped Bar (Sorted by ST)")
        self.sobol_plots["Grouped Bar (Sorted by ST)"] = fig_grouped_ST

        conv_figs = self.visualize_convergence_plots(all_results, param_names)
        for i, cf in enumerate(conv_figs, start=1):
            name = f"Convergence Plots Fig {i}"
            self.sobol_combo.addItem(name)
            self.sobol_plots[name] = cf

        fig_heat = self.visualize_combined_heatmap(all_results, param_names)
        self.sobol_combo.addItem("Combined Heatmap")
        self.sobol_plots["Combined Heatmap"] = fig_heat

        fig_comp_radar = self.visualize_comprehensive_radar_plots(all_results, param_names)
        self.sobol_combo.addItem("Comprehensive Radar Plot")
        self.sobol_plots["Comprehensive Radar Plot"] = fig_comp_radar

        fig_s1_radar, fig_st_radar = self.visualize_separate_radar_plots(all_results, param_names)
        self.sobol_combo.addItem("Radar Plot S1")
        self.sobol_plots["Radar Plot S1"] = fig_s1_radar
        self.sobol_combo.addItem("Radar Plot ST")
        self.sobol_plots["Radar Plot ST"] = fig_st_radar

        fig_box = self.visualize_box_plots(all_results)
        self.sobol_combo.addItem("Box Plots")
        self.sobol_plots["Box Plots"] = fig_box

        fig_violin = self.visualize_violin_plots(all_results)
        self.sobol_combo.addItem("Violin Plots")
        self.sobol_plots["Violin Plots"] = fig_violin

        fig_scatter = self.visualize_scatter_S1_ST(all_results, param_names)
        self.sobol_combo.addItem("Scatter S1 vs ST")
        self.sobol_plots["Scatter S1 vs ST"] = fig_scatter

        fig_parallel = self.visualize_parallel_coordinates(all_results, param_names)
        self.sobol_combo.addItem("Parallel Coordinates")
        self.sobol_plots["Parallel Coordinates"] = fig_parallel

        fig_s1_hist, fig_st_hist = self.visualize_histograms(all_results)
        self.sobol_combo.addItem("S1 Histogram")
        self.sobol_plots["S1 Histogram"] = fig_s1_hist
        self.sobol_combo.addItem("ST Histogram")
        self.sobol_plots["ST Histogram"] = fig_st_hist

    ########################################################################
    # -------------- Sobol Visualization Methods --------------
    ########################################################################

    def visualize_last_run(self, all_results, param_names):
        """Basic bar chart sorted by S1"""
        last_run_idx = -1
        S1_last_run = np.array(all_results['S1'][last_run_idx])
        ST_last_run = np.array(all_results['ST'][last_run_idx])

        sorted_indices_S1 = np.argsort(S1_last_run)[::-1]
        sorted_param_names_S1 = [param_names[i] for i in sorted_indices_S1]
        S1_sorted = S1_last_run[sorted_indices_S1]
        ST_sorted = ST_last_run[sorted_indices_S1]

        fig = Figure(figsize=(6,4))
        ax = fig.add_subplot(111)
        ax.bar(np.arange(len(sorted_param_names_S1)) - 0.175, S1_sorted, 0.35, label=r'$S_1$', color='skyblue')
        ax.bar(np.arange(len(sorted_param_names_S1)) + 0.175, ST_sorted, 0.35, label=r'$S_T$', color='salmon')
        ax.set_xlabel('Parameters', fontsize=20)
        ax.set_ylabel('Sensitivity Index', fontsize=20)
        ax.set_title('First-order ($S_1$) & Total-order ($S_T$)', fontsize=16)
        ax.set_xticks(np.arange(len(sorted_param_names_S1)))
        ax.set_xticklabels([format_parameter_name(p) for p in sorted_param_names_S1], rotation=90, fontsize=8)
        ax.legend(fontsize=10)
        fig.tight_layout()
        return fig

    def visualize_grouped_bar_plot_sorted_on_ST(self, all_results, param_names):
        """Bar chart sorted by ST"""
        last_run_idx = -1
        S1_last_run = np.array(all_results['S1'][last_run_idx])
        ST_last_run = np.array(all_results['ST'][last_run_idx])

        sorted_indices_ST = np.argsort(ST_last_run)[::-1]
        sorted_param_names_ST = [param_names[i] for i in sorted_indices_ST]
        S1_sorted = S1_last_run[sorted_indices_ST]
        ST_sorted = ST_last_run[sorted_indices_ST]

        fig = Figure(figsize=(6,4))
        ax = fig.add_subplot(111)
        ax.bar(np.arange(len(sorted_param_names_ST)) - 0.175, S1_sorted, 0.35, label=r'$S_1$', color='skyblue')
        ax.bar(np.arange(len(sorted_param_names_ST)) + 0.175, ST_sorted, 0.35, label=r'$S_T$', color='salmon')
        ax.set_xlabel('Parameters', fontsize=20)
        ax.set_ylabel('Sensitivity Index', fontsize=20)
        ax.set_title('Sorted by $S_T$', fontsize=16)
        ax.set_xticks(np.arange(len(sorted_param_names_ST)))
        ax.set_xticklabels([format_parameter_name(p) for p in sorted_param_names_ST], rotation=90, fontsize=8)
        ax.legend(fontsize=10)
        fig.tight_layout()
        return fig

    def visualize_convergence_plots(self, all_results, param_names):
        """Plot S1 and ST vs sample size for each parameter"""
        sample_sizes = all_results['samples']
        S1_matrix = np.array(all_results['S1'])
        ST_matrix = np.array(all_results['ST'])

        plots_per_fig = 12
        total_params = len(param_names)
        num_figs = int(np.ceil(total_params / plots_per_fig))
        figs = []

        for fig_idx in range(num_figs):
            fig = Figure(figsize=(20,15))
            start_idx = fig_idx * plots_per_fig
            end_idx = min(start_idx + plots_per_fig, total_params)
            for subplot_idx, param_idx in enumerate(range(start_idx, end_idx)):
                param = param_names[param_idx]
                ax = fig.add_subplot(3,4,subplot_idx+1)
                S1_values = S1_matrix[:, param_idx]
                ST_values = ST_matrix[:, param_idx]
                ax.plot(sample_sizes, S1_values, 'o-', color='blue', label=r'$S_1$')
                ax.plot(sample_sizes, ST_values, 's-', color='red', label=r'$S_T$')
                ax.set_title(f"Convergence: {format_parameter_name(param)}", fontsize=10)
                ax.set_xlabel("Sample Size", fontsize=8)
                ax.set_ylabel("Index", fontsize=8)
                ax.legend(fontsize=8)
                ax.grid(True)
            fig.tight_layout()
            figs.append(fig)
        return figs

    def visualize_combined_heatmap(self, all_results, param_names):
        """2D Heatmap (S1, ST) for the last run"""
        last_run_idx = -1
        S1_last = np.array(all_results['S1'][last_run_idx])
        ST_last = np.array(all_results['ST'][last_run_idx])

        df = pd.DataFrame({'Parameter': param_names, 'S1': S1_last, 'ST': ST_last})
        df = df.set_index('Parameter')

        fig = Figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        sns.heatmap(df, annot=True, cmap='coolwarm', cbar_kws={'label': 'Sensitivity'}, ax=ax)
        ax.set_title("Combined Heatmap (S1 & ST)")
        return fig

    def visualize_comprehensive_radar_plots(self, all_results, param_names):
        """Radar plot combining S1 and ST in single chart"""
        last_run_idx = -1
        S1 = np.array(all_results['S1'][last_run_idx])
        ST = np.array(all_results['ST'][last_run_idx])
        num_vars = len(param_names)
        angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        fig = Figure(figsize=(10,10))
        ax = fig.add_subplot(111, polar=True)
        max_val = max(np.max(S1), np.max(ST)) * 1.1
        ax.set_ylim(0, max_val)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([format_parameter_name(p) for p in param_names], fontsize=8)

        S1_vals = list(S1) + [S1[0]]
        ST_vals = list(ST) + [ST[0]]
        ax.plot(angles, S1_vals, label=r"$S_1$", color='blue', linewidth=2)
        ax.fill(angles, S1_vals, color='blue', alpha=0.2)
        ax.plot(angles, ST_vals, label=r"$S_T$", color='red', linewidth=2)
        ax.fill(angles, ST_vals, color='red', alpha=0.2)

        ax.legend(loc='best')
        ax.set_title("Comprehensive Radar Plot")
        return fig

    def visualize_separate_radar_plots(self, all_results, param_names):
        """One radar for S1, one for ST"""
        last_run_idx = -1
        S1 = np.array(all_results['S1'][last_run_idx])
        ST = np.array(all_results['ST'][last_run_idx])
        num_vars = len(param_names)
        angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        # Radar for S1
        fig_s1 = Figure(figsize=(10,10))
        ax_s1 = fig_s1.add_subplot(111, polar=True)
        max_val_s1 = np.max(S1)*1.1
        ax_s1.set_ylim(0, max_val_s1)
        ax_s1.set_xticks(angles[:-1])
        ax_s1.set_xticklabels([format_parameter_name(p) for p in param_names], fontsize=8)
        s1_vals = list(S1) + [S1[0]]
        ax_s1.plot(angles, s1_vals, color='blue', linewidth=2, label=r"$S_1$")
        ax_s1.fill(angles, s1_vals, color='blue', alpha=0.2)
        ax_s1.set_title("Radar - First-order S1")
        ax_s1.legend()

        # Radar for ST
        fig_st = Figure(figsize=(10,10))
        ax_st = fig_st.add_subplot(111, polar=True)
        max_val_st = np.max(ST)*1.1
        ax_st.set_ylim(0, max_val_st)
        ax_st.set_xticks(angles[:-1])
        ax_st.set_xticklabels([format_parameter_name(p) for p in param_names], fontsize=8)
        st_vals = list(ST) + [ST[0]]
        ax_st.plot(angles, st_vals, color='red', linewidth=2, label=r"$S_T$")
        ax_st.fill(angles, st_vals, color='red', alpha=0.2)
        ax_st.set_title("Radar - Total-order ST")
        ax_st.legend()

        return fig_s1, fig_st

    def visualize_box_plots(self, all_results):
        """Box plot of all S1 and ST from all runs"""
        data = {
            'S1': np.concatenate(all_results['S1']),
            'ST': np.concatenate(all_results['ST'])
        }
        df = pd.DataFrame(data)
        fig = Figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        sns.boxplot(data=df, palette=['skyblue', 'salmon'], ax=ax)
        ax.set_xlabel('Sensitivity Type', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title("Box Plots of S1 & ST")
        return fig

    def visualize_violin_plots(self, all_results):
        """Violin plot of all S1 and ST from all runs"""
        data = {
            'S1': np.concatenate(all_results['S1']),
            'ST': np.concatenate(all_results['ST'])
        }
        df = pd.DataFrame(data)
        fig = Figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        sns.violinplot(data=df, palette=['skyblue','salmon'], inner='quartile', ax=ax)
        ax.set_title("Violin Plots of S1 & ST")
        return fig

    def visualize_scatter_S1_ST(self, all_results, param_names):
        """Scatter plot of S1 vs ST"""
        last_run_idx = -1
        S1_last_run = np.array(all_results['S1'][last_run_idx])
        ST_last_run = np.array(all_results['ST'][last_run_idx])

        fig = Figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        scatter = ax.scatter(S1_last_run, ST_last_run, c=np.arange(len(param_names)), cmap='tab20', edgecolor='k')
        for i, param in enumerate(param_names):
            ax.text(S1_last_run[i]+0.001, ST_last_run[i]+0.001, format_parameter_name(param), fontsize=8)

        ax.set_xlabel("S1")
        ax.set_ylabel("ST")
        ax.set_title("Scatter: S1 vs ST")
        return fig

    def visualize_parallel_coordinates(self, all_results, param_names):
        """Parallel coordinates plot of S1 and ST vs sample size"""
        data = []
        for run_idx, num_samples in enumerate(all_results['samples']):
            row = {"Sample Size": num_samples}
            for param_idx, param in enumerate(param_names):
                row[f"S1_{param}"] = all_results['S1'][run_idx][param_idx]
                row[f"ST_{param}"] = all_results['ST'][run_idx][param_idx]
            data.append(row)
        df = pd.DataFrame(data)

        fig = Figure(figsize=(10,6))
        ax = fig.add_subplot(111)
        for param in param_names:
            ax.plot(df["Sample Size"], df[f"S1_{param}"], marker='o', label=f"S1 {param}", alpha=0.4)
            ax.plot(df["Sample Size"], df[f"ST_{param}"], marker='s', label=f"ST {param}", alpha=0.4)
        ax.set_xlabel("Sample Size")
        ax.set_ylabel("Sensitivity Index")
        ax.set_title("Parallel Coordinates of S1 & ST vs Sample Size")
        ax.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize=6)
        fig.tight_layout()
        return fig

    def visualize_histograms(self, all_results):
        """Histograms of S1 and ST"""
        last_run_idx = -1
        S1_last_run = np.array(all_results['S1'][last_run_idx])
        ST_last_run = np.array(all_results['ST'][last_run_idx])

        fig_s1 = Figure(figsize=(6,4))
        ax_s1 = fig_s1.add_subplot(111)
        sns.histplot(S1_last_run, bins=20, kde=True, color='skyblue', ax=ax_s1)
        ax_s1.set_title("Histogram of S1")

        fig_st = Figure(figsize=(6,4))
        ax_st = fig_st.add_subplot(111)
        sns.histplot(ST_last_run, bins=20, kde=True, color='salmon', ax=ax_st)
        ax_st.set_title("Histogram of ST")

        return fig_s1, fig_st
        
    def get_main_system_params(self):
        """Get the main system parameters from the UI"""
        # Return the same parameters as the main input_mixin
        return (
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
        
        
    def save_sobol_results(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Sobol Results", "",
                                                  "Text Files (*.txt);;All Files (*)", options=options)
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    f.write(self.sobol_results_text.toPlainText())
                QMessageBox.information(self, "Success", f"Sobol results saved to {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to save results: {e}")
            
    def update_sobol_plot(self):
        """Update the displayed Sobol plot"""
        try:
            plot_type = self.sobol_combo.currentText()
            
            # Clear the current figure
            self.sobol_figure.clear()
            
            # Get the selected plot figure
            selected_fig = self.sobol_plots[plot_type]
            
            # Copy the selected figure to the display figure
            for ax in selected_fig.get_axes():
                new_ax = self.sobol_figure.add_subplot(111)
                new_ax.clear()
                
                # Copy the plot data and properties
                for line in ax.get_lines():
                    new_ax.plot(line.get_xdata(), line.get_ydata(), 
                              color=line.get_color(), 
                              linestyle=line.get_linestyle(),
                              marker=line.get_marker(),
                              label=line.get_label())
                
                # Copy other plot elements
                new_ax.set_title(ax.get_title())
                new_ax.set_xlabel(ax.get_xlabel())
                new_ax.set_ylabel(ax.get_ylabel())
                if ax.get_legend():
                    new_ax.legend()
                
                # Copy axis limits
                new_ax.set_xlim(ax.get_xlim())
                new_ax.set_ylim(ax.get_ylim())
                
                # Copy tick labels and rotation
                new_ax.set_xticks(ax.get_xticks())
                new_ax.set_xticklabels(ax.get_xticklabels(), rotation=ax.get_xticklabels()[0].get_rotation())
                new_ax.set_yticks(ax.get_yticks())
                new_ax.set_yticklabels(ax.get_yticklabels())
            
            # Update the layout
            self.sobol_figure.tight_layout()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to update plot: {str(e)}")

    def create_sobol_analysis_tab(self):
        """Create the Sobol sensitivity analysis tab"""
        try:
            self.sobol_tab = QWidget()
            layout = QVBoxLayout(self.sobol_tab)  # Create layout with parent
            
            # Create sub-tabs for Sobol analysis
            self.sobol_sub_tabs = QTabWidget()
            
            # -------------------- Sub-tab 1: Analysis Settings --------------------
            sobol_settings_tab = QWidget()
            settings_layout = QVBoxLayout(sobol_settings_tab)
            
            # Sample size and parallel jobs settings
            sample_settings = QWidget()
            sample_form = QFormLayout(sample_settings)
            
            # Number of samples input
            self.num_samples_line = QLineEdit("32, 64, 128")
            self.num_samples_line.setToolTip("Comma-separated list of sample sizes")
            sample_form.addRow("Sample Sizes:", self.num_samples_line)
            
            # Number of parallel jobs
            self.n_jobs_spin = QSpinBox()
            self.n_jobs_spin.setRange(1, 32)
            self.n_jobs_spin.setValue(4)
            self.n_jobs_spin.setToolTip("Number of parallel processes to use")
            sample_form.addRow("Parallel Jobs:", self.n_jobs_spin)
            
            settings_layout.addWidget(sample_settings)
            
            # DVA Parameters table
            dva_group = QGroupBox("DVA Parameter Bounds")
            dva_layout = QVBoxLayout(dva_group)
            
            self.sobol_param_table = QTableWidget()
            dva_parameters = [
                *[f"beta_{i}" for i in range(1,16)],
                *[f"lambda_{i}" for i in range(1,16)],
                *[f"mu_{i}" for i in range(1,4)],
                *[f"nu_{i}" for i in range(1,16)]
            ]
            self.sobol_param_table.setRowCount(len(dva_parameters))
            self.sobol_param_table.setColumnCount(5)
            self.sobol_param_table.setHorizontalHeaderLabels(
                ["Parameter", "Fixed", "Fixed Value", "Lower Bound", "Upper Bound"]
            )
            self.sobol_param_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.sobol_param_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
            
            # Set up table rows
            for row, param in enumerate(dva_parameters):
                param_item = QTableWidgetItem(param)
                param_item.setFlags(Qt.ItemIsEnabled)
                self.sobol_param_table.setItem(row, 0, param_item)
                
                fixed_checkbox = QCheckBox()
                fixed_checkbox.stateChanged.connect(lambda state, r=row: self.toggle_sobol_fixed(state, r))
                self.sobol_param_table.setCellWidget(row, 1, fixed_checkbox)
                
                fixed_value_spin = QDoubleSpinBox()
                fixed_value_spin.setRange(-1e6, 1e6)
                fixed_value_spin.setDecimals(6)
                fixed_value_spin.setEnabled(False)
                self.sobol_param_table.setCellWidget(row, 2, fixed_value_spin)
                
                lower_bound_spin = QDoubleSpinBox()
                lower_bound_spin.setRange(-1e6, 1e6)
                lower_bound_spin.setDecimals(6)
                lower_bound_spin.setEnabled(True)
                self.sobol_param_table.setCellWidget(row, 3, lower_bound_spin)
                
                upper_bound_spin = QDoubleSpinBox()
                upper_bound_spin.setRange(-1e6, 1e6)
                upper_bound_spin.setDecimals(6)
                upper_bound_spin.setEnabled(True)
                self.sobol_param_table.setCellWidget(row, 4, upper_bound_spin)
                
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
            
            dva_layout.addWidget(self.sobol_param_table)
            settings_layout.addWidget(dva_group)
            
            # Run button
            self.hyper_run_sobol_button = QPushButton("Run Sobol Analysis")
            self.hyper_run_sobol_button.clicked.connect(self.run_sobol)
            settings_layout.addWidget(self.hyper_run_sobol_button)
            
            # -------------------- Sub-tab 2: Results --------------------
            results_tab = QWidget()
            results_layout = QVBoxLayout(results_tab)
            
            # Results text area
            self.sobol_results_text = QTextEdit()
            self.sobol_results_text.setReadOnly(True)
            self.sobol_results_text.setStyleSheet("font-family: monospace;")
            results_layout.addWidget(self.sobol_results_text)
            
            # -------------------- Sub-tab 3: Visualization --------------------
            viz_tab = QWidget()
            viz_layout = QVBoxLayout(viz_tab)
            
            # Plot type selection
            plot_controls = QWidget()
            plot_controls_layout = QHBoxLayout(plot_controls)
            
            # Plot selection combo box
            self.sobol_combo = QComboBox()
            plot_controls_layout.addWidget(QLabel("Plot Type:"))
            plot_controls_layout.addWidget(self.sobol_combo)
            
            # Save plot button
            self.save_plot_button = QPushButton("Save Plot")
            self.save_plot_button.clicked.connect(self.save_sobol_plot)
            plot_controls_layout.addWidget(self.save_plot_button)
            
            viz_layout.addWidget(plot_controls)
            
            # Figure canvas
            self.sobol_figure = Figure(figsize=(8, 6))
            self.sobol_canvas = FigureCanvas(self.sobol_figure)
            self.sobol_canvas.setMinimumHeight(400)
            self.sobol_toolbar = NavigationToolbar(self.sobol_canvas, viz_tab)
            
            viz_layout.addWidget(self.sobol_toolbar)
            viz_layout.addWidget(self.sobol_canvas)
            
            # Dictionary to store plot figures
            self.sobol_plots = {}
            
            # Connect combo box signal
            self.sobol_combo.currentTextChanged.connect(self.update_sobol_plot)
            
            # Add sub-tabs
            self.sobol_sub_tabs.addTab(sobol_settings_tab, "Settings")
            self.sobol_sub_tabs.addTab(results_tab, "Results")
            self.sobol_sub_tabs.addTab(viz_tab, "Visualization")
            
            # Add the sub-tabs widget to the main layout
            layout.addWidget(self.sobol_sub_tabs)
            
            return self.sobol_tab
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create Sobol analysis tab: {str(e)}")
            raise

    def save_sobol_plot(self):
        """Save the current Sobol plot to a file"""
        try:
            # Get the current plot type
            plot_type = self.sobol_combo.currentText()
            if not plot_type:
                QMessageBox.warning(self, "Warning", "No plot selected to save.")
                return
                
            # Get file name from user
            file_name, _ = QFileDialog.getSaveFileName(
                self,
                "Save Plot",
                "",
                "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg)"
            )
            
            if file_name:
                # Get the current figure
                fig = self.sobol_plots[plot_type]
                
                # Save the figure
                fig.savefig(file_name, bbox_inches='tight', dpi=300)
                self.status_bar.showMessage(f"Plot saved to {file_name}")
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save plot: {str(e)}")
            
    def toggle_sobol_fixed(self, state, row):
        """Toggle the fixed state of a Sobol DVA parameter row"""
        fixed = (state == Qt.Checked)
        fixed_value_spin = self.sobol_param_table.cellWidget(row, 2)
        lower_bound_spin = self.sobol_param_table.cellWidget(row, 3)
        upper_bound_spin = self.sobol_param_table.cellWidget(row, 4)
        
        fixed_value_spin.setEnabled(fixed)
        lower_bound_spin.setEnabled(not fixed)
        upper_bound_spin.setEnabled(not fixed)
        
        if fixed:
            # When fixed, set a default value based on parameter type
            param_name = self.sobol_param_table.item(row, 0).text()
            if param_name.startswith(("beta_", "lambda_", "nu_")):
                fixed_value_spin.setValue(0.5)  # Default value for these parameters
            elif param_name.startswith("mu_"):
                fixed_value_spin.setValue(0.1)  # Default value for mu parameters
            else:
                fixed_value_spin.setValue(0.0)  # Default for others

    def get_target_values_weights(self):
        """Get target values and weights for Sobol analysis"""
        target_values = {}
        weights = {}
        
        # Get target values and weights for each mass
        for i in range(1, 6):  # 5 masses
            target_box = getattr(self, f'target_m{i}_box')
            weight_box = getattr(self, f'weight_m{i}_box')
            
            target_values[f'm{i}'] = target_box.value()
            weights[f'm{i}'] = weight_box.value()
        
        return target_values, weights

            