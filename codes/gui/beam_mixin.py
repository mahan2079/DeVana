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
    from Continues_beam.utils import ForceRegionManager
    from Continues_beam.ui.results_dashboard import ResultsDashboard
    from Continues_beam.ui.cross_section_visualizer import CrossSectionVisualizer
    from Continues_beam.ui.force_regions_panel import ForceRegionsPanel
    from Continues_beam.ui.scrollable_form_widget import ScrollableFormWidget
    from Continues_beam.beam.solver import solve_beam_vibration
    from Continues_beam.beam_animation_adapter import BeamAnimationAdapter
    from Continues_beam.mode_shape_adapter import ModeShapeAdapter
except ImportError as e:
    print(f"Beam imports failed: {e}")
    BEAM_IMPORTS_SUCCESSFUL = False

class ContinuousBeamMixin:
    def create_continuous_beam_page(self):
        """Create the continuous beam analysis page"""
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
            center_layout.addWidget(error_label)
            
            description = QLabel("Please make sure the 'Continues beam' module is correctly installed.")
            description.setFont(QFont("Segoe UI", 12))
            description.setAlignment(Qt.AlignCenter)
            center_layout.addWidget(description)
            
            layout.addWidget(center_widget)
            self.content_stack.addWidget(beam_page)
            return
        
        # Create the comprehensive beam analysis interface
        try:
            beam_page = QWidget()
            main_layout = QVBoxLayout(beam_page)
            main_layout.setContentsMargins(0, 0, 0, 0)
            
            # Create tab widget for different sections
            beam_tabs = QTabWidget()
            
            # 1. Input Parameters Tab
            input_tab = self.create_beam_input_tab()
            beam_tabs.addTab(input_tab, "Input Parameters")
            
            # 2. Analysis Tab
            analysis_tab = self.create_beam_analysis_tab()
            beam_tabs.addTab(analysis_tab, "Analysis & Results")
            
            # 3. Visualization Tab
            viz_tab = self.create_beam_visualization_tab()
            beam_tabs.addTab(viz_tab, "Visualization")
            
            main_layout.addWidget(beam_tabs)
            self.content_stack.addWidget(beam_page)
            print("Comprehensive beam analysis page created successfully")
            
        except Exception as e:
            print(f"Error creating beam analysis page: {str(e)}")
            import traceback
            traceback.print_exc()
            # Create a fallback page
            beam_page = QWidget()
            layout = QVBoxLayout(beam_page)
            
            # Centered content
            center_widget = QWidget()
            center_layout = QVBoxLayout(center_widget)
            center_layout.setAlignment(Qt.AlignCenter)
            
            # Error message
            error_label = QLabel("Continuous Beam Module Error")
            error_label.setFont(QFont("Segoe UI", 24, QFont.Bold))
            error_label.setAlignment(Qt.AlignCenter)
            center_layout.addWidget(error_label)
            
            description = QLabel(f"Error initializing continuous beam module: {str(e)}")
            description.setFont(QFont("Segoe UI", 12))
            description.setAlignment(Qt.AlignCenter)
            center_layout.addWidget(description)
            
            layout.addWidget(center_widget)
            self.content_stack.addWidget(beam_page)
    
    def create_beam_input_tab(self):
        """Create the input parameters tab"""
        input_widget = QWidget()
        layout = QHBoxLayout(input_widget)
        
        # Left side - Beam Parameters
        left_panel = QGroupBox("Beam Parameters")
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
        
        # Material properties
        material_group = QGroupBox("Material Properties")
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
        
        # Boundary conditions
        boundary_group = QGroupBox("Boundary Conditions")
        boundary_layout = QFormLayout(boundary_group)
        
        self.spring_stiffness_input = QDoubleSpinBox()
        self.spring_stiffness_input.setRange(0, 1e8)
        self.spring_stiffness_input.setValue(0)
        self.spring_stiffness_input.setSuffix(" N/m")
        self.spring_stiffness_input.setDecimals(0)
        boundary_layout.addRow("Tip Spring Stiffness:", self.spring_stiffness_input)
        
        left_layout.addWidget(boundary_group)
        
        layout.addWidget(left_panel)
        
        # Right side - Loading and Analysis Settings
        right_panel = QGroupBox("Loading & Analysis Settings")
        right_layout = QVBoxLayout(right_panel)
        
        # Loading conditions
        loading_group = QGroupBox("Loading Conditions")
        loading_layout = QVBoxLayout(loading_group)
        
        # Force type selection
        force_type_layout = QHBoxLayout()
        force_type_layout.addWidget(QLabel("Force Type:"))
        self.force_type_combo = QComboBox()
        self.force_type_combo.addItems(["No Force", "Harmonic Force", "Impulse Force", "Custom Force"])
        self.force_type_combo.currentTextChanged.connect(self.update_force_parameters)
        force_type_layout.addWidget(self.force_type_combo)
        loading_layout.addLayout(force_type_layout)
        
        # Force parameters (initially hidden)
        self.force_params_widget = QWidget()
        self.force_params_layout = QFormLayout(self.force_params_widget)
        
        self.force_magnitude_input = QDoubleSpinBox()
        self.force_magnitude_input.setRange(0, 1e6)
        self.force_magnitude_input.setValue(1000)
        self.force_magnitude_input.setSuffix(" N")
        self.force_params_layout.addRow("Magnitude:", self.force_magnitude_input)
        
        self.force_frequency_input = QDoubleSpinBox()
        self.force_frequency_input.setRange(0.1, 1000)
        self.force_frequency_input.setValue(20)
        self.force_frequency_input.setSuffix(" Hz")
        self.force_params_layout.addRow("Frequency:", self.force_frequency_input)
        
        self.force_position_input = QDoubleSpinBox()
        self.force_position_input.setRange(0, 1)
        self.force_position_input.setValue(0.9)
        self.force_position_input.setSuffix(" (ratio)")
        self.force_params_layout.addRow("Position:", self.force_position_input)
        
        loading_layout.addWidget(self.force_params_widget)
        self.force_params_widget.hide()
        
        right_layout.addWidget(loading_group)
        
        # Analysis settings
        analysis_group = QGroupBox("Analysis Settings")
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
        self.run_analysis_btn = QPushButton("Run Analysis")
        self.run_analysis_btn.setStyleSheet("""
            QPushButton {
                background-color: #1E3A8A;
                color: white;
                border: none;
                padding: 10px 20px;
                font-size: 14px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1E40AF;
            }
            QPushButton:pressed {
                background-color: #1E3A8A;
            }
        """)
        self.run_analysis_btn.clicked.connect(self.run_beam_analysis)
        right_layout.addWidget(self.run_analysis_btn)
        
        layout.addWidget(right_panel)
        
        return input_widget
    
    def create_beam_analysis_tab(self):
        """Create the analysis and results tab"""
        try:
            # Use the existing ResultsDashboard
            self.results_dashboard = ResultsDashboard()
            return self.results_dashboard
        except Exception as e:
            print(f"Error creating results dashboard: {e}")
            # Create a simple fallback
            analysis_widget = QWidget()
            layout = QVBoxLayout(analysis_widget)
            layout.addWidget(QLabel("Results will appear here after running analysis"))
            return analysis_widget
    
    def create_beam_visualization_tab(self):
        """Create the visualization tab"""
        viz_widget = QWidget()
        layout = QVBoxLayout(viz_widget)
        
        # Create tab widget for different visualizations
        viz_tabs = QTabWidget()
        
        try:
            # Beam Animation
            self.beam_animation_adapter = BeamAnimationAdapter()
            viz_tabs.addTab(self.beam_animation_adapter, "Beam Animation")
            
            # Mode Shape Animation
            self.mode_shape_adapter = ModeShapeAdapter()
            viz_tabs.addTab(self.mode_shape_adapter, "Mode Shapes")
            
        except Exception as e:
            print(f"Error creating visualization adapters: {e}")
            # Create simple placeholders
            beam_anim_tab = QWidget()
            QVBoxLayout(beam_anim_tab).addWidget(QLabel("Beam animation will appear here"))
            viz_tabs.addTab(beam_anim_tab, "Beam Animation")
            
            mode_tab = QWidget()
            QVBoxLayout(mode_tab).addWidget(QLabel("Mode shapes will appear here"))
            viz_tabs.addTab(mode_tab, "Mode Shapes")
        
        layout.addWidget(viz_tabs)
        return viz_widget
    
    def update_force_parameters(self, force_type):
        """Update force parameters based on selected type"""
        if force_type == "No Force":
            self.force_params_widget.hide()
        else:
            self.force_params_widget.show()
            
            if force_type == "Harmonic Force":
                self.force_frequency_input.show()
                self.force_params_layout.labelForField(self.force_frequency_input).show()
            elif force_type == "Impulse Force":
                self.force_frequency_input.hide()
                self.force_params_layout.labelForField(self.force_frequency_input).hide()
    
    def run_beam_analysis(self):
        """Run the beam analysis with current parameters"""
        try:
            self.run_analysis_btn.setText("Running Analysis...")
            self.run_analysis_btn.setEnabled(False)
            
            # Get parameters from UI
            width = self.beam_width_input.value()
            length = self.beam_length_input.value()
            thickness = self.thickness_input.value()
            E = self.youngs_modulus_input.value()
            rho = self.density_input.value()
            k_spring = self.spring_stiffness_input.value()
            num_elems = self.num_elements_input.value()
            
            # Create layer definition
            layers = [
                {
                    'height': thickness,
                    'E': lambda T=0: E,
                    'rho': lambda T=0: rho
                }
            ]
            
            # Create force function based on selection
            force_type = self.force_type_combo.currentText()
            if force_type == "No Force":
                force_func = lambda x, t: 0.0
            elif force_type == "Harmonic Force":
                magnitude = self.force_magnitude_input.value()
                frequency = self.force_frequency_input.value()
                position_ratio = self.force_position_input.value()
                force_position = length * position_ratio
                
                def force_func(x, t):
                    if abs(x - force_position) < length / (2 * num_elems):
                        return magnitude * np.sin(2 * np.pi * frequency * t)
                    return 0.0
            elif force_type == "Impulse Force":
                magnitude = self.force_magnitude_input.value()
                position_ratio = self.force_position_input.value()
                force_position = length * position_ratio
                
                def force_func(x, t):
                    if abs(x - force_position) < length / (2 * num_elems) and t < 0.01:
                        return magnitude
                    return 0.0
            else:
                force_func = lambda x, t: 0.0
            
            # Run analysis
            time_span = self.time_span_input.value()
            time_points = self.time_points_input.value()
            
            results = solve_beam_vibration(
                width=width,
                layers=layers,
                L=length,
                k_spring=k_spring,
                num_elems=num_elems,
                f_profile=force_func,
                t_span=(0, time_span),
                num_time_points=time_points
            )
            
            # Update results dashboard
            if hasattr(self, 'results_dashboard'):
                self.results_dashboard.update_results(results)
            
            # Update visualizations
            if hasattr(self, 'beam_animation_adapter'):
                self.beam_animation_adapter.update_animation(results)
            
            if hasattr(self, 'mode_shape_adapter'):
                self.mode_shape_adapter.update_mode_shapes(results)
            
            print("Beam analysis completed successfully")
            
        except Exception as e:
            print(f"Error running beam analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Show error message
            QMessageBox.critical(self, "Analysis Error", 
                               f"Error running beam analysis:\n{str(e)}")
        
        finally:
            self.run_analysis_btn.setText("Run Analysis")
            self.run_analysis_btn.setEnabled(True)
