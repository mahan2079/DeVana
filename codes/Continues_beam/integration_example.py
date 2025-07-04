"""
Example of how to integrate the beam animation into DeVana

This script demonstrates how to:
1. Import the beam animation adapter
2. Create an instance of the adapter
3. Update the animation with simulation results
4. Reset the animation

To use this in DeVana:
1. Import the BeamAnimationAdapter class
2. Create an instance in your UI
3. Call update_animation() after running a simulation
"""

import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton
from beam_animation_adapter import BeamAnimationAdapter
from beam.solver import solve_beam_vibration

class ExampleWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Beam Animation Example")
        self.resize(800, 600)
        
        # Create central widget
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        
        # Create animation adapter
        self.animation_adapter = BeamAnimationAdapter()
        layout.addWidget(self.animation_adapter)
        
        # Create run button
        run_button = QPushButton("Run Example Simulation")
        run_button.clicked.connect(self.run_example)
        layout.addWidget(run_button)
        
        # Create reset button
        reset_button = QPushButton("Reset Animation")
        reset_button.clicked.connect(self.reset_animation)
        layout.addWidget(reset_button)
    
    def run_example(self):
        """Run an example simulation and update the animation"""
        # Define example beam parameters
        width = 0.05  # meters
        layers = [
            {
                'height': 0.01,  # meters
                'E': lambda: 210e9,  # Pa
                'rho': lambda: 7800  # kg/m³
            }
        ]
        L = 1.0  # meters
        k_spring = 0.0  # N/m
        num_elems = 20
        
        # Define example force function
        def example_force(x, t):
            # Simple harmonic force at the tip
            if x > 0.9:
                return 1000 * np.sin(20 * t)
            return 0
        
        # Run simulation
        results = solve_beam_vibration(
            width=width,
            layers=layers,
            L=L,
            k_spring=k_spring,
            num_elems=num_elems,
            f_profile=example_force,
            t_span=(0, 2),
            num_time_points=200
        )
        
        # Update animation
        self.animation_adapter.update_animation(results)
    
    def reset_animation(self):
        """Reset the animation"""
        self.animation_adapter.reset()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ExampleWindow()
    window.show()
    sys.exit(app.exec_())
