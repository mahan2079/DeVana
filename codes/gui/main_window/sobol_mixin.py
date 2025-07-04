from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtCore import QDateTime
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from modules.sobol_sensitivity import (
    perform_sobol_analysis,
    calculate_and_save_errors,
    format_parameter_name,
    save_results,
)
from workers.SobolWorker import SobolWorker


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

        # Update UI to show analysis is running
        self.run_frf_button.setEnabled(False)
        self.run_sobol_button.setEnabled(False)
        self.run_ga_button.setEnabled(False)
        self.hyper_run_sobol_button.setEnabled(False)
        
        # Clear and update results text area
        self.sobol_results_text.clear()
        self.sobol_results_text.append("--- Running Sobol Sensitivity Analysis ---\n")
        self.status_bar.showMessage("Running Sobol Analysis...")

        # Get main system parameters
        main_params = self.get_main_system_params()

        # Get DVA bounds from parameter table
        dva_bounds = {}
        EPSILON = 1e-6
        
        for row in range(self.dva_param_table.rowCount()):
            param_item = self.dva_param_table.item(row, 0)
            param_name = param_item.text()

            fixed_widget = self.dva_param_table.cellWidget(row, 1)
            fixed = fixed_widget.isChecked()

            if fixed:
                fixed_value_widget = self.dva_param_table.cellWidget(row, 2)
                fixed_value = fixed_value_widget.value()
                dva_bounds[param_name] = (fixed_value, fixed_value + EPSILON)
            else:
                lower_bound_widget = self.dva_param_table.cellWidget(row, 3)
                upper_bound_widget = self.dva_param_table.cellWidget(row, 4)
                lower = lower_bound_widget.value()
                upper = upper_bound_widget.value()
                
                if lower > upper:
                    QMessageBox.warning(self, "Input Error", 
                                        f"For parameter {param_name}, lower bound is greater than upper bound.")
                    self.run_frf_button.setEnabled(True)
                    self.run_sobol_button.setEnabled(True)
                    self.run_ga_button.setEnabled(True)
                    self.hyper_run_sobol_button.setEnabled(True)
                    return
                    
                dva_bounds[param_name] = (lower, upper)

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
        """Run the Sobol sensitivity analysis - delegate to main implementation"""
        # Call the main implementation with a different name to avoid recursion
        self._run_sobol_implementation()
        
    def get_num_samples_list(self):
        """Get the list of sample sizes for Sobol analysis"""
        num_samples_text = self.num_samples_line.text().strip()
        if not num_samples_text:
            return [32, 64, 128]  # Default values
        
        try:
            # Parse comma-separated values
            samples = [int(n.strip()) for n in num_samples_text.split(',') if n.strip()]
            if not samples:
                return [32, 64, 128]  # Default if parsing yields empty list
            return samples
        except ValueError:
            QMessageBox.warning(self, "Input Error", 
                               "Invalid num_samples format. Using default values: 32, 64, 128")
            return [32, 64, 128]
            
    def handle_sobol_error(self, err):
        """Handle errors from the Sobol worker"""
        QMessageBox.critical(self, "Error in Sobol Analysis", str(err))
        self.sobol_results_text.append(f"\nError running Sobol analysis: {err}")
        self.status_bar.showMessage("Sobol analysis failed")
        
        # Re-enable buttons
        self.run_frf_button.setEnabled(True)
        self.run_sobol_button.setEnabled(True)
        self.run_ga_button.setEnabled(True)
        self.hyper_run_sobol_button.setEnabled(True)

    def display_sobol_results(self, all_results, warnings=None):
        """
        Called when SobolWorker finishes. This includes the full
        Sobol analysis results in `all_results` and any warnings in `warnings`.
        """
        self.sobol_results = all_results
        self.sobol_warnings = warnings
        self.sobol_results_text.append("\n--- Sobol Sensitivity Analysis Results ---")

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
        param_names = original_dva_parameter_order

        def format_float(val):
            if isinstance(val,(np.float64,float,int)):
                return f"{val:.6f}"
            return str(val)

        for run_idx, num_samples in enumerate(all_results['samples']):
            self.sobol_results_text.append(f"\nSample Size: {num_samples}")
            S1 = all_results['S1'][run_idx]
            ST = all_results['ST'][run_idx]
            self.sobol_results_text.append(f"  Length of S1: {len(S1)}, Length of ST: {len(ST)}")

            for param_index, param_name in enumerate(param_names):
                if param_index < len(S1) and param_index < len(ST):
                    s1_val = S1[param_index]
                    st_val = ST[param_index]
                    self.sobol_results_text.append(f"Parameter {param_name}: S1 = {s1_val:.6f}, ST = {st_val:.6f}")
                else:
                    self.sobol_results_text.append(f"IndexError: Parameter {param_name} out of range")

        if warnings:
            self.sobol_results_text.append("\nWarnings:")
            for w in warnings:
                self.sobol_results_text.append(w)
        else:
            self.sobol_results_text.append("\nNo warnings encountered.")

        self.status_bar.showMessage("Sobol Analysis Completed.")
        self.run_frf_button.setEnabled(True)
        self.run_sobol_button.setEnabled(True)
        self.run_ga_button.setEnabled(True)

        self.sobol_combo.clear()
        self.sobol_plots.clear()

        # Build the various Sobol plots
        self.generate_sobol_plots(all_results, param_names)
        self.update_sobol_plot()
        self.sobol_canvas.draw()

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
        # Basic example: bar chart sorted by S1
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
        ax.set_xticklabels([self.format_parameter_name(p) for p in sorted_param_names_S1], rotation=90, fontsize=8)
        ax.legend(fontsize=10)
        fig.tight_layout()
        return fig

    def visualize_grouped_bar_plot_sorted_on_ST(self, all_results, param_names):
        # Similar bar chart, sorted by ST
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
        ax.set_xticklabels([self.format_parameter_name(p) for p in sorted_param_names_ST], rotation=90, fontsize=8)
        ax.legend(fontsize=10)
        fig.tight_layout()
        return fig

    def visualize_convergence_plots(self, all_results, param_names):
        # Plot S1 and ST vs sample size, for each parameter
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
                ax.set_title(f"Convergence: {self.format_parameter_name(param)}", fontsize=10)
                ax.set_xlabel("Sample Size", fontsize=8)
                ax.set_ylabel("Index", fontsize=8)
                ax.legend(fontsize=8)
                ax.grid(True)
            fig.tight_layout()
            figs.append(fig)
        return figs

    def visualize_combined_heatmap(self, all_results, param_names):
        # 2D Heatmap (S1, ST) for the last run
        last_run_idx = -1
        S1_last = np.array(all_results['S1'][last_run_idx])
        ST_last = np.array(all_results['ST'][last_run_idx])

        import pandas as pd
        df = pd.DataFrame({'Parameter': param_names, 'S1': S1_last, 'ST': ST_last})
        df = df.set_index('Parameter')

        fig = Figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        import seaborn as sns
        sns.heatmap(df, annot=True, cmap='coolwarm', cbar_kws={'label': 'Sensitivity'}, ax=ax)
        ax.set_title("Combined Heatmap (S1 & ST)")
        return fig

    def visualize_comprehensive_radar_plots(self, all_results, param_names):
        # Radar plot combining S1 and ST in single chart
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
        ax.set_xticklabels([self.format_parameter_name(p) for p in param_names], fontsize=8)

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
        # One radar for S1, one for ST
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
        ax_s1.set_xticklabels([self.format_parameter_name(p) for p in param_names], fontsize=8)
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
        ax_st.set_xticklabels([self.format_parameter_name(p) for p in param_names], fontsize=8)
        st_vals = list(ST) + [ST[0]]
        ax_st.plot(angles, st_vals, color='red', linewidth=2, label=r"$S_T$")
        ax_st.fill(angles, st_vals, color='red', alpha=0.2)
        ax_st.set_title("Radar - Total-order ST")
        ax_st.legend()

        return fig_s1, fig_st

    def visualize_box_plots(self, all_results):
        # Box plot of all S1 and ST from all runs
        data = {
            'S1': np.concatenate(all_results['S1']),
            'ST': np.concatenate(all_results['ST'])
        }
        df = pd.DataFrame(data)
        fig = Figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        import seaborn as sns
        sns.boxplot(data=df, palette=['skyblue', 'salmon'], ax=ax)
        ax.set_xlabel('Sensitivity Index', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title("Box Plots of S1 & ST")
        return fig

    def visualize_violin_plots(self, all_results):
        data = {
            'S1': np.concatenate(all_results['S1']),
            'ST': np.concatenate(all_results['ST'])
        }
        df = pd.DataFrame(data)
        fig = Figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        import seaborn as sns
        sns.violinplot(data=df, palette=['skyblue','salmon'], inner='quartile', ax=ax)
        ax.set_title("Violin Plots of S1 & ST")
        return fig

    def visualize_scatter_S1_ST(self, all_results, param_names):
        last_run_idx = -1
        S1_last_run = np.array(all_results['S1'][last_run_idx])
        ST_last_run = np.array(all_results['ST'][last_run_idx])

        fig = Figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        scatter = ax.scatter(S1_last_run, ST_last_run, c=np.arange(len(param_names)), cmap='tab20', edgecolor='k')
        for i, param in enumerate(param_names):
            ax.text(S1_last_run[i]+0.001, ST_last_run[i]+0.001, self.format_parameter_name(param), fontsize=8)

        ax.set_xlabel("S1")
        ax.set_ylabel("ST")
        ax.set_title("Scatter: S1 vs ST")
        return fig

    def visualize_parallel_coordinates(self, all_results, param_names):
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
        last_run_idx = -1
        S1_last_run = np.array(all_results['S1'][last_run_idx])
        ST_last_run = np.array(all_results['ST'][last_run_idx])

        fig_s1 = Figure(figsize=(6,4))
        ax_s1 = fig_s1.add_subplot(111)
        import seaborn as sns
        sns.histplot(S1_last_run, bins=20, kde=True, color='skyblue', ax=ax_s1)
        ax_s1.set_title("Histogram of S1")

        fig_st = Figure(figsize=(6,4))
        ax_st = fig_st.add_subplot(111)
        sns.histplot(ST_last_run, bins=20, kde=True, color='salmon', ax=ax_st)
        ax_st.set_title("Histogram of ST")

        return fig_s1, fig_st
        
    def get_main_system_params(self):
        """Get the main system parameters in a tuple format"""
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
            
