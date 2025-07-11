"""
Composite Beam Analysis Integration Example for DeVana

This script demonstrates how to:
1. Define multi-layer composite beam structures
2. Set up temperature-dependent material properties
3. Run comprehensive beam analysis
4. Visualize results with animation and mode shapes
5. Access effective properties and natural frequencies

To use this in DeVana:
1. Import the required classes and functions
2. Define your composite layer structure
3. Run the analysis
4. Integrate results into your UI
"""

import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
    QWidget, QPushButton, QLabel, QGroupBox, QTextEdit,
    QSplitter, QTabWidget
)
from PyQt5.QtCore import Qt

# Import composite beam analysis components
from beam_animation_adapter import BeamAnimationAdapter
from mode_shape_adapter import ModeShapeAdapter
from beam.solver import solve_beam_vibration, BeamVibrationSolver
from beam.properties import calc_composite_properties
from utils import ForceRegionManager, ForceRegion, get_force_generators


class CompositeBeamAnalysisExample(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Composite Beam Analysis Example - DeVana")
        self.resize(1200, 800)
        
        # Create main widget and layout
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        
        # Create splitter for controls and visualization
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left panel for controls and results
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Example selection
        examples_group = QGroupBox("Analysis Examples")
        examples_layout = QVBoxLayout(examples_group)
        
        # Example buttons
        self.steel_button = QPushButton("Simple Steel Beam")
        self.steel_button.clicked.connect(self.run_steel_example)
        examples_layout.addWidget(self.steel_button)
        
        self.composite_button = QPushButton("Steel-Aluminum Composite")
        self.composite_button.clicked.connect(self.run_composite_example)
        examples_layout.addWidget(self.composite_button)
        
        self.sandwich_button = QPushButton("Carbon Fiber Sandwich")
        self.sandwich_button.clicked.connect(self.run_sandwich_example)
        examples_layout.addWidget(self.sandwich_button)
        
        self.temperature_button = QPushButton("Temperature-Dependent Analysis")
        self.temperature_button.clicked.connect(self.run_temperature_example)
        examples_layout.addWidget(self.temperature_button)
        
        left_layout.addWidget(examples_group)
        
        # Results display
        results_group = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout(results_group)
        
        self.results_text = QTextEdit()
        self.results_text.setMaximumHeight(200)
        self.results_text.setReadOnly(True)
        results_layout.addWidget(self.results_text)
        
        left_layout.addWidget(results_group)
        
        # Animation controls
        controls_group = QGroupBox("Animation Controls")
        controls_layout = QVBoxLayout(controls_group)
        
        self.reset_button = QPushButton("Reset Animation")
        self.reset_button.clicked.connect(self.reset_animation)
        controls_layout.addWidget(self.reset_button)
        
        left_layout.addWidget(controls_group)
        
        # Add left panel to splitter
        splitter.addWidget(left_panel)
        
        # Right panel for visualization
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Create tab widget for different visualizations
        self.tab_widget = QTabWidget()
        
        # Animation tab
        self.animation_adapter = BeamAnimationAdapter()
        self.tab_widget.addTab(self.animation_adapter, "Time Animation")
        
        # Mode shape tab
        self.mode_shape_adapter = ModeShapeAdapter()
        self.tab_widget.addTab(self.mode_shape_adapter, "Mode Shapes")
        
        right_layout.addWidget(self.tab_widget)
        
        # Add right panel to splitter
        splitter.addWidget(right_panel)
        
        # Set splitter proportions
        splitter.setSizes([400, 800])
        
        # Initialize with simple example
        self.run_steel_example()
    
    def run_steel_example(self):
        """Run analysis for a simple steel beam"""
        self.results_text.append("Running Steel Beam Analysis...")
        
        # Define single steel layer
        layers = [
            {
                'height': 0.01,  # 10mm thickness
                'E': lambda: 210e9,  # Steel Young's modulus (Pa)
                'rho': lambda: 7800  # Steel density (kg/m³)
            }
        ]
        
        # Run analysis
        results = self.run_analysis(
            layers=layers,
            description="Simple Steel Beam (10mm thick)",
            force_frequency=10.0  # 10 Hz harmonic force
        )
        
        # Update visualizations
        self.update_visualizations(results)
    
    def run_composite_example(self):
        """Run analysis for steel-aluminum composite beam"""
        self.results_text.append("Running Steel-Aluminum Composite Analysis...")
        
        # Define composite layers (steel-aluminum-steel)
        layers = [
            {
                'height': 0.003,  # 3mm steel layer
                'E': lambda: 210e9,  # Steel
                'rho': lambda: 7800
            },
            {
                'height': 0.005,  # 5mm aluminum layer
                'E': lambda: 70e9,   # Aluminum
                'rho': lambda: 2700
            },
            {
                'height': 0.003,  # 3mm steel layer
                'E': lambda: 210e9,  # Steel
                'rho': lambda: 7800
            }
        ]
        
        # Run analysis
        results = self.run_analysis(
            layers=layers,
            description="Steel-Aluminum-Steel Composite (3+5+3mm)",
            force_frequency=15.0  # 15 Hz harmonic force
        )
        
        # Update visualizations
        self.update_visualizations(results)
    
    def run_sandwich_example(self):
        """Run analysis for carbon fiber sandwich beam"""
        self.results_text.append("Running Carbon Fiber Sandwich Analysis...")
        
        # Define sandwich structure (carbon fiber - foam - carbon fiber)
        layers = [
            {
                'height': 0.001,  # 1mm carbon fiber
                'E': lambda: 150e9,  # Carbon fiber
                'rho': lambda: 1600
            },
            {
                'height': 0.015,  # 15mm foam core
                'E': lambda: 0.1e9,  # Foam
                'rho': lambda: 100
            },
            {
                'height': 0.001,  # 1mm carbon fiber
                'E': lambda: 150e9,  # Carbon fiber
                'rho': lambda: 1600
            }
        ]
        
        # Run analysis
        results = self.run_analysis(
            layers=layers,
            description="Carbon Fiber Sandwich (1+15+1mm)",
            force_frequency=25.0  # 25 Hz harmonic force
        )
        
        # Update visualizations
        self.update_visualizations(results)
    
    def run_temperature_example(self):
        """Run analysis with temperature-dependent properties"""
        self.results_text.append("Running Temperature-Dependent Analysis...")
        
        # Define temperature-dependent steel layer
        layers = [
            {
                'height': 0.008,  # 8mm thickness
                'E': lambda T: 210e9 * (1 - 0.0001 * T),  # E decreases with temperature
                'rho': lambda T: 7800 * (1 + 0.00001 * T)  # ρ increases with temperature
            }
        ]
        
        # Run analysis
        results = self.run_analysis(
            layers=layers,
            description="Temperature-Dependent Steel (8mm, T-effects)",
            force_frequency=12.0  # 12 Hz harmonic force
        )
        
        # Update visualizations
        self.update_visualizations(results)
    
    def run_analysis(self, layers, description, force_frequency=10.0):
        """
        Run beam analysis with given parameters
        
        Parameters:
        -----------
        layers : list
            List of layer definitions
        description : str
            Description of the analysis
        force_frequency : float
            Frequency of harmonic excitation (Hz)
            
        Returns:
        --------
        dict : Analysis results
        """
        # Beam geometry
        width = 0.05  # 50mm width
        L = 1.0       # 1m length
        k_spring = 0.0  # No tip spring
        num_elems = 50  # 50 elements for good accuracy
        
        # Create force function (harmonic force at tip)
        def force_function(x, t):
            if x > 0.9:  # Apply near tip
                return 1000 * np.sin(2 * np.pi * force_frequency * t)
            return 0
        
        # Run the analysis
        results = solve_beam_vibration(
            width=width,
            layers=layers,
            L=L,
            k_spring=k_spring,
            num_elems=num_elems,
            f_profile=force_function,
            t_span=(0, 3),  # 3 second analysis
            num_time_points=300
        )
        
        # Calculate effective properties for display
        # Convert layer format
        converted_layers = []
        for layer in layers:
            converted_layers.append({
                'thickness': layer['height'],
                'E_func': layer['E'],
                'rho_func': layer['rho']
            })
        
        EI_eff, rhoA_eff = calc_composite_properties(width, converted_layers)
        
        # Display results
        self.display_results(results, EI_eff, rhoA_eff, description, force_frequency)
        
        return results
    
    def display_results(self, results, EI_eff, rhoA_eff, description, force_frequency):
        """Display analysis results in the text area"""
        self.results_text.append(f"\n--- {description} ---")
        self.results_text.append(f"Excitation Frequency: {force_frequency:.1f} Hz")
        self.results_text.append(f"Effective EI: {EI_eff:.2e} N·m²")
        self.results_text.append(f"Effective ρA: {rhoA_eff:.2f} kg/m")
        
        # Natural frequencies
        freqs = results['natural_frequencies_hz'][:5]  # First 5 modes
        self.results_text.append("Natural Frequencies (Hz):")
        for i, freq in enumerate(freqs):
            self.results_text.append(f"  Mode {i+1}: {freq:.2f} Hz")
        
        # Maximum tip displacement
        max_tip = np.max(np.abs(results['tip_displacement']))
        self.results_text.append(f"Max Tip Displacement: {max_tip*1000:.2f} mm")
        
        self.results_text.append("-" * 40)
    
    def update_visualizations(self, results):
        """Update the animation and mode shape visualizations"""
        # Update time animation
        self.animation_adapter.update_animation(results)
        
        # Update mode shapes
        self.mode_shape_adapter.update_results(results)
    
    def reset_animation(self):
        """Reset both animations"""
        self.animation_adapter.reset()
        self.mode_shape_adapter.reset()
        self.results_text.clear()
        self.results_text.append("Animations reset. Select an example to run analysis.")


def main():
    """Main function to run the example"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle("Fusion")
    
    # Create and show the example window
    window = CompositeBeamAnalysisExample()
    window.show()
    
    # Run the application
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
