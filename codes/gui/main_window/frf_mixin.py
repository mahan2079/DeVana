from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from workers.FRFWorker import FRFWorker

class FRFMixin:

    def __init__(self):
        """Initialize FRF mixin"""
        self.plot_window = None
        self.comp_canvas = None
        self.comp_toolbar = None
        self.comp_fig = None

    def create_comparative_visualization_options(self, parent_layout):
        """Create options for comparative visualization of multiple FRF inputs"""
        # Initialize figures that will be used across the class
        self.rel_change_fig = Figure(figsize=(10, 6))
        self.rel_change_canvas = FigureCanvas(self.rel_change_fig)
        
        # Initialize no data label
        self.rel_change_no_data_label = QLabel("No data available")
        self.rel_change_no_data_label.setAlignment(Qt.AlignCenter)
        self.rel_change_no_data_label.setStyleSheet("color: gray; font-size: 14px;")
        self.rel_change_no_data_label.setVisible(False)
        
        # Create the comparative group box
        self.comp_group = QGroupBox("Comparative Visualization")
        self.comp_group.setObjectName("comparative-group")
        comp_layout = QVBoxLayout(self.comp_group)
        
        # Create plot container with fixed canvas and toolbar
        self.comp_plot_container = QWidget()
        self.comp_plot_layout = QVBoxLayout(self.comp_plot_container)
        
        # Initialize the figure, canvas and toolbar once
        self.comp_fig = Figure(figsize=(10, 6))
        self.comp_canvas = FigureCanvas(self.comp_fig)
        self.comp_toolbar = NavigationToolbar(self.comp_canvas, self.comp_plot_container)
        
        # Add them to the layout
        self.comp_plot_layout.addWidget(self.comp_toolbar)
        self.comp_plot_layout.addWidget(self.comp_canvas)
        
        # Hide the plot container by default
        self.comp_plot_container.hide()
        
        comp_layout.addWidget(self.comp_plot_container)
        
        # Introduction text
        intro_label = QLabel("This section allows you to create custom comparative plots by selecting multiple FRF results and customizing legends and title.")
        intro_label.setWordWrap(True)
        comp_layout.addWidget(intro_label)
        
        # Available plots section
        available_plots_group = QGroupBox("Available Plots")
        available_plots_layout = QVBoxLayout(available_plots_group)
        
        self.available_plots_list = QListWidget()
        self.available_plots_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.available_plots_list.itemSelectionChanged.connect(self._update_legend_table_from_selection)
        available_plots_layout.addWidget(self.available_plots_list)
        
        # Create button layout for management
        plots_btn_layout = QHBoxLayout()
        
        # Add clear button for plot history
        clear_plots_button = QPushButton("Clear All FRF Plots")
        clear_plots_button.clicked.connect(self.clear_all_frf_plots)
        plots_btn_layout.addWidget(clear_plots_button)
        
        # Add export/import buttons
        export_button = QPushButton("Export FRF Data")
        export_button.clicked.connect(self.export_frf_data)
        plots_btn_layout.addWidget(export_button)
        
        import_button = QPushButton("Import FRF Data")
        import_button.clicked.connect(self.import_frf_data)
        plots_btn_layout.addWidget(import_button)
        
        available_plots_layout.addLayout(plots_btn_layout)
        comp_layout.addWidget(available_plots_group)
        
        # Legend customization
        legend_group = QGroupBox("Legend & Style Customization")
        legend_layout = QVBoxLayout(legend_group)
        
        # Map of original plot names to custom properties
        self.legend_map = {}
        self.legend_table = QTableWidget()
        self.legend_table.setColumnCount(5)
        self.legend_table.setHorizontalHeaderLabels([
            "Original Name", 
            "Custom Legend", 
            "Line Style", 
            "Marker", 
            "Color"
        ])
        self.legend_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        legend_layout.addWidget(self.legend_table)
        
        comp_layout.addWidget(legend_group)
        
        # Plot title customization
        title_group = QGroupBox("Plot Title")
        title_layout = QFormLayout(title_group)
        
        self.plot_title_edit = QLineEdit()
        self.plot_title_edit.setPlaceholderText("Enter custom plot title here")
        title_layout.addRow("Custom Title:", self.plot_title_edit)
        
        # Font size for title
        self.title_font_size = QSpinBox()
        self.title_font_size.setRange(8, 24)
        self.title_font_size.setValue(14)
        title_layout.addRow("Title Font Size:", self.title_font_size)
        
        comp_layout.addWidget(title_group)
        
        # Plot customization options
        plot_options_group = QGroupBox("Plot Options")
        plot_options_layout = QFormLayout(plot_options_group)
        
        # Figure size
        fig_size_container = QWidget()
        fig_size_layout = QHBoxLayout(fig_size_container)
        fig_size_layout.setContentsMargins(0, 0, 0, 0)
        
        self.fig_width_spin = QSpinBox()
        self.fig_width_spin.setRange(4, 20)
        self.fig_width_spin.setValue(10)
        fig_size_layout.addWidget(QLabel("Width:"))
        fig_size_layout.addWidget(self.fig_width_spin)
        
        self.fig_height_spin = QSpinBox()
        self.fig_height_spin.setRange(3, 15)
        self.fig_height_spin.setValue(6)
        fig_size_layout.addWidget(QLabel("Height:"))
        fig_size_layout.addWidget(self.fig_height_spin)
        
        plot_options_layout.addRow("Figure Size:", fig_size_container)
        
        # Add normalization options
        norm_container = QWidget()
        norm_layout = QHBoxLayout(norm_container)
        norm_layout.setContentsMargins(0, 0, 0, 0)
        
        # X axis normalization
        self.x_norm_check = QCheckBox("X-Axis")
        norm_layout.addWidget(self.x_norm_check)
        
        self.x_norm_value = QDoubleSpinBox()
        self.x_norm_value.setRange(0.001, 1000000)
        self.x_norm_value.setValue(1.0)
        self.x_norm_value.setDecimals(3)
        self.x_norm_value.setSingleStep(0.1)
        self.x_norm_value.setEnabled(False)
        norm_layout.addWidget(self.x_norm_value)
        
        self.x_norm_check.toggled.connect(self.x_norm_value.setEnabled)
        
        norm_layout.addSpacing(20)
        
        # Y axis normalization
        self.y_norm_check = QCheckBox("Y-Axis")
        norm_layout.addWidget(self.y_norm_check)
        
        self.y_norm_value = QDoubleSpinBox()
        self.y_norm_value.setRange(0.001, 1000000)
        self.y_norm_value.setValue(1.0)
        self.y_norm_value.setDecimals(3)
        self.y_norm_value.setSingleStep(0.1)
        self.y_norm_value.setEnabled(False)
        norm_layout.addWidget(self.y_norm_value)
        
        self.y_norm_check.toggled.connect(self.y_norm_value.setEnabled)
        
        plot_options_layout.addRow("Normalize by:", norm_container)
        
        # Grid options
        self.show_grid_check = QCheckBox()
        self.show_grid_check.setChecked(True)
        plot_options_layout.addRow("Show Grid:", self.show_grid_check)
        
        # Legend position
        self.legend_position_combo = QComboBox()
        for pos in ["best", "upper right", "upper left", "lower left", "lower right", 
                   "right", "center left", "center right", "lower center", "upper center", "center"]:
            self.legend_position_combo.addItem(pos)
        plot_options_layout.addRow("Legend Position:", self.legend_position_combo)
        
        comp_layout.addWidget(plot_options_group)
        
        # Visualization actions
        actions_container = QWidget()
        actions_layout = QHBoxLayout(actions_container)
        
        self.create_comp_plot_btn = QPushButton("Create Comparative Plot")
        self.create_comp_plot_btn.setObjectName("primary-button")
        self.create_comp_plot_btn.clicked.connect(self.create_comparative_plot)
        actions_layout.addWidget(self.create_comp_plot_btn)
        
        self.save_comp_plot_btn = QPushButton("Save Plot")
        self.save_comp_plot_btn.setObjectName("secondary-button")
        self.save_comp_plot_btn.clicked.connect(lambda: self.save_plot(self.comp_fig, "Comparative FRF"))
        actions_layout.addWidget(self.save_comp_plot_btn)
        
        comp_layout.addWidget(actions_container)
        
        # Add the comparative group to the parent layout
        parent_layout.addWidget(self.comp_group)

    def _update_legend_table_from_selection(self):
        """Update the legend table based on the selected plots in the list widget"""
        # Clear current table contents
        self.legend_table.setRowCount(0)
        
        # Get selected items
        selected_items = self.available_plots_list.selectedItems()
        
        if not selected_items:
            return
            
        # Create a row for each selected plot
        self.legend_table.setRowCount(len(selected_items))
        
        for row, item in enumerate(selected_items):
            plot_name = item.text()
            
            # Original name column (non-editable)
            name_item = QTableWidgetItem(plot_name)
            name_item.setFlags(Qt.ItemIsEnabled)  # Make it non-editable
            self.legend_table.setItem(row, 0, name_item)
            
            # Custom legend name column
            if plot_name in self.legend_map and 'custom_name' in self.legend_map[plot_name]:
                legend_name = self.legend_map[plot_name]['custom_name']
            else:
                legend_name = plot_name
                
            legend_item = QTableWidgetItem(legend_name)
            self.legend_table.setItem(row, 1, legend_item)
            
            # Line style combo box
            line_style_combo = QComboBox()
            line_styles = ['-', '--', '-.', ':']
            for style in line_styles:
                line_style_combo.addItem(style)
                
            # Set previously selected style if available
            if plot_name in self.legend_map and 'line_style' in self.legend_map[plot_name]:
                line_style_value = self.legend_map[plot_name]['line_style']
                # Convert empty string to "None" for the combobox
                if line_style_value == "":
                    line_style_value = "None"
                try:
                    index = line_styles.index(line_style_value)
                    if index >= 0:
                        line_style_combo.setCurrentIndex(index)
                except ValueError:
                    # If line style value not in the list, use default
                    line_style_combo.setCurrentIndex(0)  # First style
                    
            self.legend_table.setCellWidget(row, 2, line_style_combo)
            
            # Marker combo box
            marker_combo = QComboBox()
            markers = ['None', '.', 'o', 'v', '^', '<', '>', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']
            for marker in markers:
                marker_combo.addItem(marker)
                
            # Set previously selected marker if available
            if plot_name in self.legend_map and 'marker' in self.legend_map[plot_name]:
                marker_value = self.legend_map[plot_name]['marker']
                # Convert empty string to "None" for the combobox
                if marker_value == "":
                    marker_value = "None"
                try:
                    index = markers.index(marker_value)
                    if index >= 0:
                        marker_combo.setCurrentIndex(index)
                except ValueError:
                    # If marker value not in the list, use default
                    marker_combo.setCurrentIndex(0)  # "None"
                    
            self.legend_table.setCellWidget(row, 3, marker_combo)
            
            # Color button
            color_button = QPushButton()
            color_button.setAutoFillBackground(True)
            
            # Set previously selected color if available
            if plot_name in self.legend_map and 'color' in self.legend_map[plot_name]:
                color = self.legend_map[plot_name]['color']
                color_button.setStyleSheet(f"background-color: {color};")
            else:
                # Generate random color if no previous color exists
                import random
                r, g, b = [random.randint(0, 255) for _ in range(3)]
                color = f"rgb({r},{g},{b})"
                color_button.setStyleSheet(f"background-color: {color};")
            
            # Connect button to color picker
            color_button.clicked.connect(lambda checked, row=row: self._choose_color(row))
            
            self.legend_table.setCellWidget(row, 4, color_button)
            
            # Store initial values in the legend map
            if plot_name not in self.legend_map:
                self.legend_map[plot_name] = {
                    'custom_name': legend_name,
                    'line_style': line_styles[line_style_combo.currentIndex()],
                    'marker': markers[marker_combo.currentIndex()],
                    'color': color
                }
    
    def _choose_color(self, row):
        """Open a color dialog when a color button is clicked"""
        from PyQt5.QtWidgets import QColorDialog
        
        color_button = self.legend_table.cellWidget(row, 4)
        color_dialog = QColorDialog(self)
        
        if color_dialog.exec_():
            color = color_dialog.selectedColor()
            if color.isValid():
                color_name = color.name()
                color_button.setStyleSheet(f"background-color: {color_name};")
                
                # Update the legend map with the new color
                plot_name = self.legend_table.item(row, 0).text()
                if plot_name in self.legend_map:
                    self.legend_map[plot_name]['color'] = color_name
                else:
                    self.legend_map[plot_name] = {'color': color_name}
    
    def clear_all_frf_plots(self):
        """Clear all FRF plots from the list and reset the legend map"""
        # Clear the list widget
        self.available_plots_list.clear()
        
        # Clear the legend table
        self.legend_table.setRowCount(0)
        
        # Reset the legend map
        self.legend_map = {}
        
        # Clear any existing plot
        try:
            if hasattr(self, 'comp_fig') and self.comp_fig:
                import matplotlib.pyplot as plt
                plt.close(self.comp_fig)
                self.comp_fig = None
        except Exception as e:
            print(f"Error clearing figures: {str(e)}")
    
    def export_frf_data(self):
        """Export FRF data to a file"""
        from PyQt5.QtWidgets import QFileDialog
        import json
        import os
        
        # Check if there are any plots to export
        if self.available_plots_list.count() == 0:
            QMessageBox.warning(self, "Export Error", "No FRF data available to export.")
            return
        
        # Get export filename
        filename, _ = QFileDialog.getSaveFileName(
            self, "Export FRF Data", os.path.expanduser("~"), "JSON Files (*.json)"
        )
        
        if not filename:
            return  # User cancelled
            
        # Gather data to export
        export_data = {
            'plots': {},
            'legend_map': self.legend_map
        }
        
        # Add individual plot data
        for i in range(self.available_plots_list.count()):
            plot_name = self.available_plots_list.item(i).text()
            if hasattr(self, f"frf_data_{plot_name}"):
                plot_data = getattr(self, f"frf_data_{plot_name}")
                export_data['plots'][plot_name] = plot_data
        
        # Write to file
        try:
            with open(filename, 'w') as f:
                json.dump(export_data, f)
            QMessageBox.information(self, "Export Successful", f"FRF data has been exported to {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Error exporting data: {str(e)}")
    
    def import_frf_data(self):
        """Import FRF data from a file"""
        from PyQt5.QtWidgets import QFileDialog
        import json
        import os
        
        # Get import filename
        filename, _ = QFileDialog.getOpenFileName(
            self, "Import FRF Data", os.path.expanduser("~"), "JSON Files (*.json)"
        )
        
        if not filename:
            return  # User cancelled
            
        # Read the file
        try:
            with open(filename, 'r') as f:
                import_data = json.load(f)
                
            # Validate imported data
            if 'plots' not in import_data or 'legend_map' not in import_data:
                raise ValueError("Invalid file format: missing required data")
                
            # Update legend map
            self.legend_map.update(import_data['legend_map'])
            
            # Import plot data
            for plot_name, plot_data in import_data['plots'].items():
                # Store the data
                setattr(self, f"frf_data_{plot_name}", plot_data)
                
                # Add to list if not already there
                found = False
                for i in range(self.available_plots_list.count()):
                    if self.available_plots_list.item(i).text() == plot_name:
                        found = True
                        break
                        
                if not found:
                    self.available_plots_list.addItem(plot_name)
                    
            QMessageBox.information(self, "Import Successful", 
                                   f"Imported {len(import_data['plots'])} FRF datasets from {filename}")
                                   
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Error importing data: {str(e)}")
    
    def create_sobol_analysis_tab(self):
        self.sobol_tab = QWidget()
        layout = QVBoxLayout(self.sobol_tab)

        # Create sub-tabs widget
        # No relative change data available
        ax = self.rel_change_fig.add_subplot(111)
        ax.text(0.5, 0.5, "Insufficient data for relative change calculation", 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=14, fontstyle='italic', color='#888')
        
        # Update the figure
        self.rel_change_fig.tight_layout()
        self.rel_change_canvas.draw()
        
        # Hide "no data" message
        self.rel_change_no_data_label.setVisible(False)
        
    def run_frf(self):
        """Run the FRF analysis"""
        if self.omega_start_box.value() >= self.omega_end_box.value():
            QMessageBox.warning(self, "Input Error", "Ω Start must be less than Ω End.")
            return
            
        self.status_bar.showMessage("Running FRF analysis...")
        self.results_text.append("\n--- Running FRF Analysis ---\n")
        
        # Get main system parameters
        main_params = self.get_main_system_params()
        
        # Get DVA parameters
        dva_params = []
        for i in range(15):
            if i < len(self.beta_boxes):
                dva_params.append(self.beta_boxes[i].value())
        
        for i in range(15):
            if i < len(self.lambda_boxes):
                dva_params.append(self.lambda_boxes[i].value())
        
        for i in range(3):
            if i < len(self.mu_dva_boxes):
                dva_params.append(self.mu_dva_boxes[i].value())
        
        for i in range(15):
            if i < len(self.nu_dva_boxes):
                dva_params.append(self.nu_dva_boxes[i].value())
        
        # Get target values and weights
        target_values_dict, weights_dict = self.get_target_values_weights()
        
        # Create and start FRFWorker
        self.frf_worker = FRFWorker(
            main_params=main_params,
            dva_params=tuple(dva_params),
            omega_start=self.omega_start_box.value(),
            omega_end=self.omega_end_box.value(),
            omega_points=self.omega_points_box.value(),
            target_values_dict=target_values_dict,
            weights_dict=weights_dict,
            plot_figure=self.plot_figure_chk.isChecked(),
            show_peaks=self.show_peaks_chk.isChecked(),
            show_slopes=self.show_slopes_chk.isChecked(),
            interpolation_method=self.interp_method_combo.currentText(),
            interpolation_points=self.interp_points_box.value()
        )
        
        # Disable run buttons during analysis
        self.run_frf_button.setEnabled(False)
        self.run_sobol_button.setEnabled(False)
        self.run_ga_button.setEnabled(False)
        
        # Connect signals
        self.frf_worker.finished.connect(self.handle_frf_finished)
        self.frf_worker.error.connect(self.handle_frf_error)
        
        # Start worker
        self.frf_worker.start()
    
    def handle_frf_finished(self, results_with_dva, results_without_dva):
        """Handle the completion of FRF analysis"""
        # Re-enable run buttons
        self.run_frf_button.setEnabled(True)
        self.run_sobol_button.setEnabled(True)
        self.run_ga_button.setEnabled(True)
        
        self.status_bar.showMessage("FRF analysis completed")
        
        # Store results for reference
        self.frf_results = results_with_dva
        
        # Get run name from user
        # Extract key parameters for default name
        key_params = []
        
        # Add main system info
        main_params = self.get_main_system_params()
        if len(main_params) >= 2:  # At least m1 and k1 exist
            key_params.append(f"m1={main_params[0]:.2f}")
            key_params.append(f"k1={main_params[1]:.2f}")
        
        # Add DVA info (first beta and first mu)
        dva_params = []
        if len(self.beta_boxes) > 0 and self.beta_boxes[0].value() > 0:
            key_params.append(f"β1={self.beta_boxes[0].value():.2f}")
        if len(self.mu_dva_boxes) > 0 and self.mu_dva_boxes[0].value() > 0:
            key_params.append(f"μ1={self.mu_dva_boxes[0].value():.2f}")
            
        # Create default name with parameters
        default_name = " ".join(key_params) if key_params else "Default"
        
        # Ask user to name this run
        run_name, ok = QInputDialog.getText(
            self, 
            "Name this FRF Run", 
            "Enter a name for this FRF analysis run:",
            QLineEdit.Normal, 
            default_name
        )
        
        if not ok or not run_name:
            # If user cancels or enters empty name, use default with timestamp
            timestamp = QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss")
            run_name = f"{default_name} ({timestamp})"
        
        # Generate timestamp for internal use
        timestamp = QDateTime.currentDateTime().toString("yyyy-MM-dd HH:mm:ss")
        
        # Initialize plots dictionary if needed
        if not hasattr(self, 'frf_plots'):
            self.frf_plots = {}
            
        # Store raw data for possible export/import
        self.frf_raw_data = {} if not hasattr(self, 'frf_raw_data') else self.frf_raw_data
        omega = np.linspace(self.omega_start_box.value(), self.omega_end_box.value(), self.omega_points_box.value())
        self.frf_raw_data[f"{run_name} ({timestamp})"] = {
            'omega': omega,
            'results_with_dva': results_with_dva,
            'results_without_dva': results_without_dva,
            'main_params': main_params,
            'omega_start': self.omega_start_box.value(),
            'omega_end': self.omega_end_box.value(),
            'omega_points': self.omega_points_box.value(),
            'interpolation_method': self.interp_method_combo.currentText(),
            'interpolation_points': self.interp_points_box.value(),
            'timestamp': timestamp,
            'run_name': run_name
        }
        
        # Get frequency range
        omega = np.linspace(self.omega_start_box.value(), self.omega_end_box.value(), self.omega_points_box.value())
        
        # For formatted output
        def format_float(val):
            if isinstance(val, (np.float64, float, int)):
                return f"{val:.6e}"
            return str(val)
        
        # Disable LaTeX rendering in matplotlib to prevent Unicode errors
        import matplotlib as mpl
        mpl.rcParams['text.usetex'] = False
        
        # Get list of masses with data
        required_masses = [f'mass_{m}' for m in range(1, 6)]
        mass_labels = []
        for m_label in required_masses:
            if m_label in results_with_dva and 'magnitude' in results_with_dva[m_label]:
                mass_labels.append(m_label)
        
        # Plot with DVAs, individually
        for m_label in mass_labels:
            fig = Figure(figsize=(6, 4))
            ax = fig.add_subplot(111)
            mag = results_with_dva[m_label]['magnitude']
            
            if len(mag) == len(omega):
                # Apply interpolation if requested
                interpolation_method = self.interp_method_combo.currentText()
                interpolation_points = self.interp_points_box.value()
                
                if interpolation_method != 'none':
                    from modules.FRF import apply_interpolation
                    omega_smooth, mag_smooth = apply_interpolation(
                        omega, mag, 
                        method=interpolation_method,
                        num_points=interpolation_points
                    )
                    # Plot smoothed interpolated line
                    ax.plot(omega_smooth, mag_smooth, label=m_label, linewidth=2)
                    # Also plot original points with small markers
                    ax.plot(omega, mag, 'o', markersize=1, alpha=0.3, color='gray')
                else:
                    # No interpolation, plot raw data
                    ax.plot(omega, mag, label=m_label)
                
                ax.set_xlabel('Frequency (rad/s)')
                ax.set_ylabel('Amplitude')
                ax.set_title(f'FRF of {m_label} (With DVA) - {run_name}')
                ax.legend()
                ax.grid(True)
                
                # Add to combo and plot dict with run name
                plot_name = f"{m_label} (With DVA) - {run_name}"
                self.frf_combo.addItem(plot_name)
                self.frf_plots[plot_name] = fig
                # Add to available plots list for comparative visualization
                self.available_plots_list.addItem(plot_name)
            else:
                QMessageBox.warning(self, "Plot Error", f"{m_label}: magnitude length != omega length.")
        
        # Combined plot with DVAs
        if mass_labels:
            fig_combined = Figure(figsize=(6, 4))
            ax_combined = fig_combined.add_subplot(111)
            
            for m_label in mass_labels:
                mag = results_with_dva[m_label]['magnitude']
                if len(mag) == len(omega):
                    # Apply interpolation if requested
                    interpolation_method = self.interp_method_combo.currentText()
                    interpolation_points = self.interp_points_box.value()
                    
                    if interpolation_method != 'none':
                        from modules.FRF import apply_interpolation
                        omega_smooth, mag_smooth = apply_interpolation(
                            omega, mag, 
                            method=interpolation_method,
                            num_points=interpolation_points
                        )
                        # Plot smoothed interpolated line
                        ax_combined.plot(omega_smooth, mag_smooth, label=m_label, linewidth=2)
                    else:
                        # No interpolation, plot raw data
                        ax_combined.plot(omega, mag, label=m_label)
            
            ax_combined.set_xlabel('Frequency (rad/s)')
            ax_combined.set_ylabel('Amplitude')
            ax_combined.set_title(f'Combined FRF of All Masses (With DVAs) - {run_name}')
            ax_combined.grid(True)
            ax_combined.legend()
            
            plot_name = f"All Masses Combined (With DVAs) - {run_name}"
            self.frf_combo.addItem(plot_name)
            self.frf_plots[plot_name] = fig_combined
            # Add to available plots list for comparative visualization
            self.available_plots_list.addItem(plot_name)
        
        # Plot without DVAs for Mass1 and Mass2
        for m_label in ['mass_1', 'mass_2']:
            if m_label in results_without_dva and 'magnitude' in results_without_dva[m_label]:
                fig = Figure(figsize=(6, 4))
                ax = fig.add_subplot(111)
                mag = results_without_dva[m_label]['magnitude']
                
                if len(mag) == len(omega):
                    # Apply interpolation if requested
                    interpolation_method = self.interp_method_combo.currentText()
                    interpolation_points = self.interp_points_box.value()
                    
                    if interpolation_method != 'none':
                        from modules.FRF import apply_interpolation
                        omega_smooth, mag_smooth = apply_interpolation(
                            omega, mag, 
                            method=interpolation_method,
                            num_points=interpolation_points
                        )
                        # Plot smoothed interpolated line
                        ax.plot(omega_smooth, mag_smooth, label=f"{m_label} (Without DVA)", color='green', linewidth=2)
                        # Also plot original points with small markers
                        ax.plot(omega, mag, 'o', markersize=1, alpha=0.3, color='gray')
                    else:
                        # No interpolation, plot raw data
                        ax.plot(omega, mag, label=f"{m_label} (Without DVA)", color='green')
                    
                    ax.set_xlabel('Frequency (rad/s)')
                    ax.set_ylabel('Amplitude')
                    ax.set_title(f'FRF of {m_label} (Without DVA) - {run_name}')
                    ax.legend()
                    ax.grid(True)
                    
                    plot_name = f"{m_label} (Without DVA) - {run_name}"
                    self.frf_combo.addItem(plot_name)
                    self.frf_plots[plot_name] = fig
                    # Add to available plots list for comparative visualization
                    self.available_plots_list.addItem(plot_name)
                else:
                    QMessageBox.warning(self, "Plot Error", f"{m_label} (Without DVA): magnitude length mismatch.")
        
        # Combined plot with and without DVAs for Mass1 & Mass2
        fig_combined_with_without = Figure(figsize=(6, 4))
        ax_combined_with_without = fig_combined_with_without.add_subplot(111)
        
        # Get interpolation settings
        interpolation_method = self.interp_method_combo.currentText()
        interpolation_points = self.interp_points_box.value()
        use_interpolation = interpolation_method != 'none'
        
        # With DVA lines
        for m_label in ['mass_1', 'mass_2']:
            if m_label in results_with_dva and 'magnitude' in results_with_dva[m_label]:
                mag = results_with_dva[m_label]['magnitude']
                if len(mag) == len(omega):
                    if use_interpolation:
                        from modules.FRF import apply_interpolation
                        omega_smooth, mag_smooth = apply_interpolation(
                            omega, mag, 
                            method=interpolation_method,
                            num_points=interpolation_points
                        )
                        ax_combined_with_without.plot(omega_smooth, mag_smooth, label=f"{m_label} (With DVA)", linewidth=2)
                    else:
                        ax_combined_with_without.plot(omega, mag, label=f"{m_label} (With DVA)")
        
        # Without DVA lines
        for m_label in ['mass_1', 'mass_2']:
            if m_label in results_without_dva and 'magnitude' in results_without_dva[m_label]:
                mag = results_without_dva[m_label]['magnitude']
                if len(mag) == len(omega):
                    if use_interpolation:
                        from modules.FRF import apply_interpolation
                        omega_smooth, mag_smooth = apply_interpolation(
                            omega, mag, 
                            method=interpolation_method,
                            num_points=interpolation_points
                        )
                        ax_combined_with_without.plot(
                            omega_smooth, mag_smooth, 
                            label=f"{m_label} (Without DVA)", 
                            linestyle='--',
                            linewidth=2
                        )
                    else:
                        ax_combined_with_without.plot(omega, mag, label=f"{m_label} (Without DVA)", linestyle='--')
        
        ax_combined_with_without.set_xlabel('Frequency (rad/s)')
        ax_combined_with_without.set_ylabel('Amplitude')
        ax_combined_with_without.set_title(f'FRF of Mass 1 & 2: With and Without DVAs - {run_name}')
        ax_combined_with_without.grid(True)
        ax_combined_with_without.legend()
        
        plot_name = f"Mass 1 & 2: With and Without DVAs - {run_name}"
        self.frf_combo.addItem(plot_name)
        self.frf_plots[plot_name] = fig_combined_with_without
        # Add to available plots list for comparative visualization
        self.available_plots_list.addItem(plot_name)
        
        # Plot all masses combined with and without DVAs for mass1 & mass2
        fig_all_combined = Figure(figsize=(6, 4))
        ax_all_combined = fig_all_combined.add_subplot(111)
        
        # Get interpolation settings if not already defined
        if not 'use_interpolation' in locals():
            interpolation_method = self.interp_method_combo.currentText()
            interpolation_points = self.interp_points_box.value()
            use_interpolation = interpolation_method != 'none'
        
        # With DVAs (all masses)
        for m_label in mass_labels:
            mag = results_with_dva[m_label]['magnitude']
            if len(mag) == len(omega):
                if use_interpolation:
                    from modules.FRF import apply_interpolation
                    omega_smooth, mag_smooth = apply_interpolation(
                        omega, mag, 
                        method=interpolation_method,
                        num_points=interpolation_points
                    )
                    ax_all_combined.plot(omega_smooth, mag_smooth, label=f"{m_label} (With DVA)", linewidth=2)
                else:
                    ax_all_combined.plot(omega, mag, label=f"{m_label} (With DVA)")
        
        # Without DVAs for mass1 & mass2
        for m_label in ['mass_1', 'mass_2']:
            if m_label in results_without_dva and 'magnitude' in results_without_dva[m_label]:
                mag = results_without_dva[m_label]['magnitude']
                if len(mag) == len(omega):
                    if use_interpolation:
                        from modules.FRF import apply_interpolation
                        omega_smooth, mag_smooth = apply_interpolation(
                            omega, mag, 
                            method=interpolation_method,
                            num_points=interpolation_points
                        )
                        ax_all_combined.plot(
                            omega_smooth, mag_smooth, 
                            label=f"{m_label} (Without DVA)", 
                            linestyle='--',
                            linewidth=2
                        )
                    else:
                        ax_all_combined.plot(omega, mag, label=f"{m_label} (Without DVA)", linestyle='--')
        
        ax_all_combined.set_xlabel('Frequency (rad/s)')
        ax_all_combined.set_ylabel('Amplitude')
        ax_all_combined.set_title(f'Combined FRF (All Masses), \nMass1 & 2 with/without DVAs - {run_name}')
        ax_all_combined.grid(True)
        ax_all_combined.legend()
        
        plot_name = f"All Masses Combined: With and Without DVAs for Mass 1 & 2 - {run_name}"
        self.frf_combo.addItem(plot_name)
        self.frf_plots[plot_name] = fig_all_combined
        # Add to available plots list for comparative visualization
        self.available_plots_list.addItem(plot_name)
        
        # Update the plot if we have data
        if self.frf_plots:
            self.update_frf_plot()
        
        # Display text results
        self.results_text.append(f"\n--- FRF Analysis Completed: {run_name} ({timestamp}) ---\n")
        
        # Results with DVA
        self.results_text.append("\nResults with DVA:")
        
        # Print "with DVA" results
        for mass in required_masses:
            self.results_text.append(f"\nRaw results for {mass}:")
            if mass in self.frf_results:
                for key, value in self.frf_results[mass].items():
                    if isinstance(value, dict):
                        formatted_dict = {k: format_float(v) for k, v in value.items()}
                        self.results_text.append(f"{key}: {formatted_dict}")
                    else:
                        self.results_text.append(f"{key}: {format_float(value)}")
            else:
                self.results_text.append(f"No results for {mass}")

        self.results_text.append("\nComposite Measures:")
        if 'composite_measures' in self.frf_results:
            for mass, comp in self.frf_results['composite_measures'].items():
                self.results_text.append(f"{mass}: {format_float(comp)}")
        else:
            self.results_text.append("No composite measures found.")

        self.results_text.append("\nPercentage Differences:")
        if 'percentage_differences' in self.frf_results:
            for mass, pdiffs in self.frf_results['percentage_differences'].items():
                self.results_text.append(f"\n{mass}:")
                for key, value in pdiffs.items():
                    self.results_text.append(f"  {key}: {format_float(value)}%")
        else:
            self.results_text.append("No percentage differences found.")

        self.results_text.append("\nSingular Response:")
        if 'singular_response' in self.frf_results:
            self.results_text.append(f"{format_float(self.frf_results['singular_response'])}")
        else:
            self.results_text.append("No singular response found.")
        
        # Results without DVA
        self.results_text.append("\n--- FRF Analysis Results (Without DVAs for Mass 1 and Mass 2) ---")
        required_masses_without_dva = ['mass_1', 'mass_2']
        
        for mass in required_masses_without_dva:
            self.results_text.append(f"\nRaw results for {mass}:")
            if mass in results_without_dva:
                for key, value in results_without_dva[mass].items():
                    if isinstance(value, dict):
                        formatted_dict = {k: format_float(v) for k, v in value.items()}
                        self.results_text.append(f"{key}: {formatted_dict}")
                    else:
                        self.results_text.append(f"{key}: {format_float(value)}")
            else:
                self.results_text.append(f"No results for {mass}")

        self.results_text.append("\nComposite Measures (Without DVAs for Mass 1 and Mass 2):")
        if 'composite_measures' in results_without_dva:
            for mass, comp in results_without_dva['composite_measures'].items():
                if mass in ['mass_1', 'mass_2']:
                    self.results_text.append(f"{mass}: {format_float(comp)}")
        else:
            self.results_text.append("No composite measures found.")

        self.results_text.append("\nSingular Response (Without DVAs for Mass 1 and Mass 2):")
        if 'singular_response' in results_without_dva:
            self.results_text.append(f"{format_float(results_without_dva['singular_response'])}")
        else:
            self.results_text.append("No singular response found.")

    def handle_frf_error(self, err):
        """Handle errors from the FRF worker"""
        # Re-enable run buttons
        self.run_frf_button.setEnabled(True)
        self.run_sobol_button.setEnabled(True)
        self.run_ga_button.setEnabled(True)
        
        QMessageBox.critical(self, "Error in FRF Analysis", str(err))
        self.results_text.append(f"\nError running FRF Analysis: {err}")
        self.status_bar.showMessage("FRF analysis failed")

    def run_sobol(self):
        """Run the Sobol sensitivity analysis"""
        self.status_bar.showMessage("Running Sobol analysis...")
        self.results_text.append("Sobol analysis started...")
        
    def run_sa(self):
        """Run the simulated annealing optimization"""
        # Implementation already exists at line 2591
        pass
        
    def update_frf_plot(self):
        """Update the FRF plot based on the selected option"""
        key = self.frf_combo.currentText()
        if key in self.frf_plots:
            fig = self.frf_plots[key]
            self.frf_canvas.figure = fig
            self.frf_canvas.draw()
        else:
            self.frf_canvas.figure.clear()
            self.frf_canvas.draw()
        
    def save_plot(self, figure, plot_type):
        """Save the current plot to a file"""
        if figure is None:
            QMessageBox.warning(self, "Error", "No plot to save.")
            return
            
        options = QFileDialog.Options()
        file_types = "PNG Files (*.png);;PDF Files (*.pdf);;SVG Files (*.svg);;JPEG Files (*.jpg);;All Files (*)"
        default_name = f"{plot_type.replace(' ', '_')}_{QDateTime.currentDateTime().toString('yyyyMMdd_HHmmss')}"
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, 
            f"Save {plot_type} Plot", 
            default_name,
            file_types, 
            options=options
        )
        
        if file_path:
            try:
                # Make sure file has correct extension based on selected filter
                if "PNG" in selected_filter and not file_path.lower().endswith(".png"):
                    file_path += ".png"
                elif "PDF" in selected_filter and not file_path.lower().endswith(".pdf"):
                    file_path += ".pdf"
                elif "SVG" in selected_filter and not file_path.lower().endswith(".svg"):
                    file_path += ".svg"
                elif "JPEG" in selected_filter and not file_path.lower().endswith((".jpg", ".jpeg")):
                    file_path += ".jpg"
                
                # Save with different formats
                if file_path.lower().endswith(".pdf"):
                    figure.savefig(file_path, format="pdf", bbox_inches="tight")
                elif file_path.lower().endswith(".svg"):
                    figure.savefig(file_path, format="svg", bbox_inches="tight")
                elif file_path.lower().endswith((".jpg", ".jpeg")):
                    figure.savefig(file_path, format="jpg", dpi=1200, bbox_inches="tight")
                else:  # Default to PNG
                    figure.savefig(file_path, format="png", dpi=1200, bbox_inches="tight")
                
                self.status_bar.showMessage(f"Plot saved to {file_path}")
                QMessageBox.information(self, "Plot Saved", f"Plot successfully saved to {file_path}")
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to save plot: {str(e)}")
        else:
            self.status_bar.showMessage("Plot save canceled")
        
    def save_sobol_results(self):
        pass

    def create_comparative_plot(self):
        """Create a comparative plot of multiple FRF results with customizable legends and styling"""
        selected_items = self.available_plots_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Plots Selected", 
                              "Please select at least one plot from the available plots list.")
            return

        # Keep plot container hidden
        self.comp_plot_container.hide()

        # Clear the current figure
        self.comp_fig.clear()
        
        # Create new axes with user-specified size
        self.comp_fig.set_size_inches(self.fig_width_spin.value(), self.fig_height_spin.value())
        ax = self.comp_fig.add_subplot(111)

        # Plot each selected FRF curve
        for row in range(self.legend_table.rowCount()):
            plot_name = self.legend_table.item(row, 0).text()
            if plot_name not in self.frf_plots:
                continue

            frf_data = self.frf_plots[plot_name]
            if isinstance(frf_data, Figure):
                # Extract data from the figure - handle all lines
                ax_data = frf_data.axes[0]
                lines = ax_data.get_lines()
                
                # Get custom legend name and style
                base_legend_name = self.legend_table.item(row, 1).text() or plot_name
                line_style = self.legend_table.cellWidget(row, 2).currentText()
                marker = self.legend_table.cellWidget(row, 3).currentText()
                color_btn = self.legend_table.cellWidget(row, 4)
                color = color_btn.property('color') if color_btn else None
                
                # Plot each line from the figure
                for i, line in enumerate(lines):
                    x_data = line.get_xdata()
                    y_data = line.get_ydata()
                    
                    # Apply normalization if enabled
                    if self.x_norm_check.isChecked():
                        x_data = x_data / self.x_norm_value.value()
                    if self.y_norm_check.isChecked():
                        y_data = y_data / self.y_norm_value.value()
                    
                    # For combined plots, use the original line's label
                    if "Combined" in plot_name:
                        legend_name = line.get_label()
                        # Use the line's original color if not overridden
                        line_color = color if color else line.get_color()
                    else:
                        legend_name = base_legend_name
                        line_color = color
                        
                    # Skip small marker points used for interpolation visualization
                    if line.get_markersize() == 1 and line.get_alpha() == 0.3:
                        continue
                        
                    # Plot with custom styling
                    ax.plot(x_data, y_data, 
                           linestyle=line_style if line_style != 'None' else '',
                           marker=marker if marker != 'None' else '',
                           color=line_color,
                           label=legend_name)
            else:
                # Handle dictionary data format
                x_data = frf_data['frequency']
                y_data = frf_data['magnitude']
                
                # Apply normalization if enabled
                if self.x_norm_check.isChecked():
                    x_data = x_data / self.x_norm_value.value()
                if self.y_norm_check.isChecked():
                    y_data = y_data / self.y_norm_value.value()

                # Get custom legend name and style
                legend_name = self.legend_table.item(row, 1).text() or plot_name
                line_style = self.legend_table.cellWidget(row, 2).currentText()
                marker = self.legend_table.cellWidget(row, 3).currentText()
                color_btn = self.legend_table.cellWidget(row, 4)
                color = color_btn.property('color') if color_btn else None

                # Plot with custom styling
                ax.plot(x_data, y_data, 
                       linestyle=line_style if line_style != 'None' else '',
                       marker=marker if marker != 'None' else '',
                       color=color,
                       label=legend_name)

        # Set title with custom font size
        title = self.plot_title_edit.text() or "Comparative FRF Plot"
        ax.set_title(title, fontsize=self.title_font_size.value())

        # Set labels
        x_label = "Normalized Frequency" if self.x_norm_check.isChecked() else "Frequency (Hz)"
        y_label = "Normalized Magnitude" if self.y_norm_check.isChecked() else "Magnitude"
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)

        # Configure grid
        ax.grid(self.show_grid_check.isChecked())

        # Configure legend
        if ax.get_legend_handles_labels()[0]:  # Only show legend if there are labeled plots
            ax.legend(loc=self.legend_position_combo.currentText())

        # Use linear scale for y-axis
        ax.set_yscale('linear')

        # Adjust layout and draw
        self.comp_fig.tight_layout()
        self.comp_canvas.draw()
        
        # Reset the view history in the toolbar
        self.comp_toolbar.update()
        
        # Show success message
        QMessageBox.information(self, "Plot Created", 
                              "Comparative plot has been created successfully.\nUse the Save Plot button to save it.")
