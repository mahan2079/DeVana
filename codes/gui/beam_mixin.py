from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import os
import pandas as pd
import numpy as np

from gui.widgets import ModernQTabWidget

# Check if beam module imports are successful
BEAM_IMPORTS_SUCCESSFUL = True
try:
    from Continues_beam import create_beam_optimization_interface
except ImportError as e:
    print(f"Beam imports failed: {e}")
    BEAM_IMPORTS_SUCCESSFUL = False

class ContinuousBeamMixin:
    def create_continuous_beam_page(self):
        """Create the continuous beam optimization page."""
        if not BEAM_IMPORTS_SUCCESSFUL:
            # Create placeholder page if imports failed
            beam_page = QWidget()
            layout = QVBoxLayout(beam_page)
            
            # Centered content
            center_widget = QWidget()
            center_layout = QVBoxLayout(center_widget)
            center_layout.setAlignment(Qt.AlignCenter)
            
            # Error message
            error_label = QLabel("Continuous Beam Module Not Available")
            error_label.setFont(QFont("Segoe UI", 24, QFont.Bold))
            error_label.setAlignment(Qt.AlignCenter)
            error_label.setStyleSheet("color: #D32F2F;")
            center_layout.addWidget(error_label)
            
            description = QLabel("Please make sure the 'Continues_beam' module is correctly installed.")
            description.setFont(QFont("Segoe UI", 12))
            description.setAlignment(Qt.AlignCenter)
            description.setStyleSheet("color: #666666;")
            center_layout.addWidget(description)
            
            # Add some helpful information
            help_text = QLabel("""
            The Continuous Beam Optimization module provides:
            • Two optimization modes:
              - Values-only at user-selected locations
              - Placement + values
            • Targets on displacement, velocity, or acceleration
            • Frequency-domain evaluation and constraints
            """)
            help_text.setFont(QFont("Segoe UI", 10))
            help_text.setAlignment(Qt.AlignCenter)
            help_text.setStyleSheet("color: #888888; margin-top: 20px;")
            center_layout.addWidget(help_text)
            
            layout.addWidget(center_widget)
            self.content_stack.addWidget(beam_page)
            return
        
        # Create the continuous beam optimization interface
        try:
            beam_page = QWidget()
            main_layout = QVBoxLayout(beam_page)
            main_layout.setContentsMargins(0, 0, 0, 0)
            main_layout.setSpacing(0)
            
            # Create header with title and quick info
            header = self.create_beam_header()
            main_layout.addWidget(header)
            
            # Create the main optimization interface
            theme = getattr(self, 'current_theme', 'Dark')
            self.beam_optimization_interface = create_beam_optimization_interface()
            self.beam_optimization_interface.set_theme(theme)

            # Connect signals
            self.beam_optimization_interface.analysis_completed.connect(self.on_beam_analysis_completed)

            # Store reference for theme updates
            self.beam_interface = self.beam_optimization_interface

            # Add to main layout
            main_layout.addWidget(self.beam_optimization_interface)
            
            # Add the page to the stack
            self.content_stack.addWidget(beam_page)
            print("Continuous beam optimization page created successfully")
            
        except Exception as e:
            print(f"Error creating continuous beam page: {str(e)}")
            import traceback
            traceback.print_exc()
            # Create a fallback page
            self.create_fallback_beam_page(str(e))
    
    def create_beam_header(self):
        """Create a professional header for the beam analysis page."""
        header = QWidget()
        header.setFixedHeight(80)
        header.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1976D2, stop:1 #1565C0);
                border-bottom: 2px solid #0D47A1;
            }
        """)
        
        layout = QHBoxLayout(header)
        layout.setContentsMargins(20, 10, 20, 10)
        
        # Title section
        title_layout = QVBoxLayout()
        
        title = QLabel("Continuous Beam Optimization")
        title.setFont(QFont("Segoe UI", 18, QFont.Bold))
        title.setStyleSheet("color: white;")
        title_layout.addWidget(title)
        
        subtitle = QLabel("Optimize spring/damper placement and values for vibration targets")
        subtitle.setFont(QFont("Segoe UI", 10))
        subtitle.setStyleSheet("color: rgba(255, 255, 255, 0.8);")
        title_layout.addWidget(subtitle)
        
        layout.addLayout(title_layout)
        layout.addStretch()
        
        # Quick stats section (will be updated by the interface)
        stats_layout = QVBoxLayout()
        stats_layout.setAlignment(Qt.AlignRight)
        
        self.layers_count_label = QLabel("Layers: 0")
        self.layers_count_label.setFont(QFont("Segoe UI", 10, QFont.Bold))
        self.layers_count_label.setStyleSheet("color: white;")
        stats_layout.addWidget(self.layers_count_label)
        
        self.analysis_status_label = QLabel("Ready for analysis")
        self.analysis_status_label.setFont(QFont("Segoe UI", 9))
        self.analysis_status_label.setStyleSheet("color: rgba(255, 255, 255, 0.8);")
        stats_layout.addWidget(self.analysis_status_label)
        
        layout.addLayout(stats_layout)
        
        return header
    
    def create_fallback_beam_page(self, error_message):
        """Create a fallback page when the enhanced interface fails."""
        beam_page = QWidget()
        layout = QVBoxLayout(beam_page)
        
        # Error display
        error_widget = QWidget()
        error_layout = QVBoxLayout(error_widget)
        error_layout.setAlignment(Qt.AlignCenter)
        
        error_label = QLabel("Continuous Beam Module Error")
        error_label.setFont(QFont("Segoe UI", 20, QFont.Bold))
        error_label.setAlignment(Qt.AlignCenter)
        error_label.setStyleSheet("color: #D32F2F; margin-bottom: 10px;")
        error_layout.addWidget(error_label)
        
        error_detail = QLabel(f"Error initializing continuous beam module:\n{error_message}")
        error_detail.setFont(QFont("Segoe UI", 10))
        error_detail.setAlignment(Qt.AlignCenter)
        error_detail.setWordWrap(True)
        error_detail.setStyleSheet("color: #666666; margin-bottom: 20px;")
        error_layout.addWidget(error_detail)
        
        # Fallback to basic beam interface
        fallback_label = QLabel("Falling back to basic beam analysis...")
        fallback_label.setFont(QFont("Segoe UI", 12))
        fallback_label.setAlignment(Qt.AlignCenter)
        fallback_label.setStyleSheet("color: #1976D2; margin-bottom: 20px;")
        error_layout.addWidget(fallback_label)
        
        layout.addWidget(error_widget)
        
        # Create basic beam interface
        try:
            basic_interface = self.create_basic_beam_interface()
            layout.addWidget(basic_interface)
        except Exception as e:
            basic_error_label = QLabel(f"Basic interface also failed: {str(e)}")
            basic_error_label.setStyleSheet("color: #D32F2F;")
            layout.addWidget(basic_error_label)
        
        self.content_stack.addWidget(beam_page)
    
    def create_basic_beam_interface(self):
        """Create a basic beam interface as fallback."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        
        # Create tab widget for different sections
        beam_tabs = QTabWidget()
        
        # 1. Input Parameters Tab
        input_tab = self.create_basic_input_tab()
        beam_tabs.addTab(input_tab, "Basic Input")
        
        # 2. Results Tab
        results_tab = self.create_basic_results_tab()
        beam_tabs.addTab(results_tab, "Results")
        
        layout.addWidget(beam_tabs)
        return widget
    
    def create_basic_input_tab(self):
        """Create a basic input tab for simple beam analysis."""
        input_widget = QWidget()
        layout = QHBoxLayout(input_widget)
        
        # Left side - Beam Parameters
        left_panel = QGroupBox("Basic Beam Parameters")
        left_layout = QVBoxLayout(left_panel)
        
        # Beam geometry
        geom_group = QGroupBox("Geometry")
        geom_layout = QFormLayout(geom_group)
        
        self.beam_length_input = QDoubleSpinBox()
        self.beam_length_input.setRange(0.1, 100.0)
        self.beam_length_input.setValue(1.0)
        self.beam_length_input.setSuffix(" m")
        self.beam_length_input.setDecimals(3)
        geom_layout.addRow("Length:", self.beam_length_input)
        
        self.beam_width_input = QDoubleSpinBox()
        self.beam_width_input.setRange(0.001, 1.0)
        self.beam_width_input.setValue(0.05)
        self.beam_width_input.setSuffix(" m")
        self.beam_width_input.setDecimals(4)
        geom_layout.addRow("Width:", self.beam_width_input)
        
        self.num_elements_input = QSpinBox()
        self.num_elements_input.setRange(5, 200)
        self.num_elements_input.setValue(20)
        geom_layout.addRow("Number of Elements:", self.num_elements_input)
        
        left_layout.addWidget(geom_group)
        
        # Single layer material properties
        material_group = QGroupBox("Material Properties (Single Layer)")
        material_layout = QFormLayout(material_group)
        
        self.youngs_modulus_input = QDoubleSpinBox()
        self.youngs_modulus_input.setRange(1e6, 1e12)
        self.youngs_modulus_input.setValue(210e9)
        self.youngs_modulus_input.setSuffix(" Pa")
        self.youngs_modulus_input.setDecimals(0)
        material_layout.addRow("Young's Modulus:", self.youngs_modulus_input)
        
        self.density_input = QDoubleSpinBox()
        self.density_input.setRange(100, 20000)
        self.density_input.setValue(7800)
        self.density_input.setSuffix(" kg/m³")
        self.density_input.setDecimals(0)
        material_layout.addRow("Density:", self.density_input)
        
        self.thickness_input = QDoubleSpinBox()
        self.thickness_input.setRange(0.001, 0.1)
        self.thickness_input.setValue(0.01)
        self.thickness_input.setSuffix(" m")
        self.thickness_input.setDecimals(4)
        material_layout.addRow("Thickness:", self.thickness_input)
        
        left_layout.addWidget(material_group)
        
        layout.addWidget(left_panel)
        
        # Right side - Analysis Settings
        right_panel = QGroupBox("Analysis Settings")
        right_layout = QVBoxLayout(right_panel)
        
        # Analysis parameters
        analysis_group = QGroupBox("Analysis Parameters")
        analysis_layout = QFormLayout(analysis_group)
        
        self.time_span_input = QDoubleSpinBox()
        self.time_span_input.setRange(0.1, 100)
        self.time_span_input.setValue(2.0)
        self.time_span_input.setSuffix(" s")
        analysis_layout.addRow("Time Duration:", self.time_span_input)
        
        self.time_points_input = QSpinBox()
        self.time_points_input.setRange(50, 2000)
        self.time_points_input.setValue(200)
        analysis_layout.addRow("Time Points:", self.time_points_input)
        
        right_layout.addWidget(analysis_group)
        
        # Run analysis button
        self.run_basic_analysis_btn = QPushButton("Run Basic Analysis")
        self.run_basic_analysis_btn.setStyleSheet("""
            QPushButton {
                background-color: #1976D2;
                color: white;
                border: none;
                padding: 12px 24px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 6px;
            }
            QPushButton:hover {
                background-color: #1565C0;
            }
            QPushButton:pressed {
                background-color: #0D47A1;
            }
        """)
        self.run_basic_analysis_btn.clicked.connect(self.run_basic_beam_analysis)
        right_layout.addWidget(self.run_basic_analysis_btn)
        
        layout.addWidget(right_panel)
        
        return input_widget
    
    def create_basic_results_tab(self):
        """Create a basic results tab."""
        results_widget = QWidget()
        layout = QVBoxLayout(results_widget)
        
        # Results text area
        self.basic_results_text = QTextEdit()
        self.basic_results_text.setReadOnly(True)
        self.basic_results_text.setPlainText("Run analysis to see results here...")
        layout.addWidget(self.basic_results_text)
        
        return results_widget
    
    def run_basic_beam_analysis(self):
        """Run a basic beam analysis."""
        try:
            self.run_basic_analysis_btn.setText("Running Analysis...")
            self.run_basic_analysis_btn.setEnabled(False)
            
            # Create single layer
            layers = [{
                'height': self.thickness_input.value(),
                'E': lambda: self.youngs_modulus_input.value(),
                'rho': lambda: self.density_input.value()
            }]
            
            # Legacy composite solver removed in rewrite; provide placeholder results
            results = {
                'natural_frequencies_hz': np.array([]),
                'tip_displacement': np.array([]),
            }
            
            # Display basic results
            results_text = f"""
Basic Beam Analysis Results:

Geometry:
- Length: {self.beam_length_input.value():.3f} m
- Width: {self.beam_width_input.value():.4f} m
- Thickness: {self.thickness_input.value():.4f} m

Material Properties:
- Young's Modulus: {self.youngs_modulus_input.value()/1e9:.1f} GPa
- Density: {self.density_input.value():.0f} kg/m³

Natural Frequencies:
"""
            
            if 'natural_frequencies_hz' in results and len(results['natural_frequencies_hz']):
                freqs = results['natural_frequencies_hz'][:5]
                for i, freq in enumerate(freqs):
                    results_text += f"- Mode {i+1}: {freq:.2f} Hz\n"
            else:
                results_text += "(natural frequencies unavailable in basic placeholder)\n"
            
            if 'tip_displacement' in results and len(results['tip_displacement']):
                results_text += f"\nMaximum tip displacement: {np.max(np.abs(results['tip_displacement']))*1000:.2f} mm"
            else:
                results_text += "\nTip displacement unavailable in basic placeholder"
            
            self.basic_results_text.setPlainText(results_text)
            
            QMessageBox.information(self, "Analysis Complete", "Basic beam analysis completed!")
            
        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", f"Error running analysis:\n{str(e)}")
            
        finally:
            self.run_basic_analysis_btn.setText("Run Basic Analysis")
            self.run_basic_analysis_btn.setEnabled(True)
    
    def on_beam_analysis_completed(self, results):
        """Handle completion of beam analysis from the composite interface."""
        try:
            # Update header status
            self.analysis_status_label.setText("Analysis completed successfully")
            
            # Update visualization tabs if they exist
            if hasattr(self, 'beam_animation_adapter'):
                self.beam_animation_adapter.update_animation(results)
                
            if hasattr(self, 'mode_shape_adapter'):
                self.mode_shape_adapter.update_results(results)
            
            print("Composite beam analysis completed and visualizations updated")
            
        except Exception as e:
            print(f"Error handling analysis completion: {str(e)}")
            self.analysis_status_label.setText("Analysis completed with errors")
    
    def update_force_parameters(self, force_type):
        """Update force parameters based on selection (for basic interface)."""
        if hasattr(self, 'force_params_widget'):
            if force_type == "No Force":
                self.force_params_widget.hide()
            else:
                self.force_params_widget.show()
    
    # Legacy methods for backward compatibility
    def create_beam_input_tab(self):
        """Legacy method - redirects to basic input tab."""
        return self.create_basic_input_tab()
    
    def create_beam_analysis_tab(self):
        """Legacy method - redirects to basic results tab."""
        return self.create_basic_results_tab()
    
    def create_beam_visualization_tab(self):
        """Legacy method - creates a basic visualization tab."""
        viz_widget = QWidget()
        layout = QVBoxLayout(viz_widget)
        
        info_label = QLabel("Enhanced visualizations available in the main composite interface")
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setStyleSheet("color: #666666; font-style: italic; padding: 50px;")
        layout.addWidget(info_label)
        
        return viz_widget
    
    def run_beam_analysis(self):
        """Legacy method - redirects to basic analysis."""
        self.run_basic_beam_analysis()
        
    def update_beam_interface_theme(self, theme):
        """Update the beam interface theme."""
        if hasattr(self, 'beam_interface') and self.beam_interface:
            self.beam_interface.set_theme(theme)
