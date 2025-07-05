from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import numpy as np

class OmegaSensitivityMixin:
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
    

