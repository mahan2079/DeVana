import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QSlider, QGroupBox, QFormLayout, QComboBox, QGridLayout,
    QDoubleSpinBox, QSplitter, QSpacerItem, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer

class ModeShapeAnimationWidget(QWidget):
    """
    Widget for animating beam mode shapes
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        
        # Initialize data
        self.coords = None
        self.mode_shapes = None
        self.original_mode_shapes = None
        self.natural_frequencies = None
        self.current_mode = 0
        self.animation_amplitude = 1.0
        self.scale_factor = 1.0
        self.auto_scale_factors = None
        self.animation_time = 0.0
        self.animation_running = False
        self.animation_frames = 60  # Number of frames per animation cycle
        self.current_frame = 0
        self.animation_speed = 1.0
        
    def initUI(self):
        """Initialize the UI components"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Create main splitter for resizable sections
        main_splitter = QSplitter(Qt.Vertical)
        main_splitter.setChildrenCollapsible(False)
        
        # Top section - Visualization
        viz_widget = QWidget()
        viz_layout = QVBoxLayout(viz_widget)
        viz_layout.setContentsMargins(0, 0, 0, 0)
        viz_layout.setSpacing(5)
        
        # Add title
        title = QLabel("Mode Shape Animation")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        viz_layout.addWidget(title)
        
        # Create canvas for animation
        self.figure = Figure(figsize=(8, 5), dpi=100)  # Increased height
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(350)  # Increased minimum height
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel('Position (m)')
        self.ax.set_ylabel('Mode Shape Amplitude')
        self.ax.set_title('Mode Shape Animation')
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        # Create initial empty plot
        self.mode_line, = self.ax.plot([], [], 'b-', lw=2, marker='o', markersize=4)
        self.undeformed_line, = self.ax.plot([], [], 'k--', lw=1)
        
        # Tight layout for better use of space
        self.figure.tight_layout(pad=2.0)
        
        viz_layout.addWidget(self.canvas, 1)
        main_splitter.addWidget(viz_widget)
        
        # Bottom section - Controls
        controls_widget = QWidget()
        controls_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(5)
        
        # Create controls group
        controls_group = QGroupBox("Mode Shape Controls")
        controls_group_layout = QVBoxLayout(controls_group)
        controls_group_layout.setContentsMargins(10, 15, 10, 10)
        controls_group_layout.setSpacing(10)
        
        # Create a horizontal layout for all controls to make them more compact
        compact_controls = QHBoxLayout()
        
        # Mode selection and frequency display
        mode_section = QVBoxLayout()
        
        # Mode selection
        mode_selector = QHBoxLayout()
        mode_selector.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.setMinimumWidth(150)
        self.mode_combo.currentIndexChanged.connect(self.change_mode)
        mode_selector.addWidget(self.mode_combo)
        mode_section.addLayout(mode_selector)
        
        # Frequency display
        self.freq_label = QLabel("Frequency: 0.00 Hz")
        self.freq_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        mode_section.addWidget(self.freq_label)
        
        compact_controls.addLayout(mode_section)
        compact_controls.addSpacing(20)
        
        # Animation controls - playback buttons
        button_section = QVBoxLayout()
        button_layout = QHBoxLayout()
        
        # Play button with styled appearance
        self.play_button = QPushButton("▶ Play")
        self.play_button.clicked.connect(self.toggle_animation)
        self.play_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-weight: bold;
                border-radius: 4px;
                min-width: 80px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        button_layout.addWidget(self.play_button)
        
        # Reset button with styled appearance
        self.reset_button = QPushButton("⟳ Reset")
        self.reset_button.clicked.connect(self.reset_animation)
        self.reset_button.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                font-weight: bold;
                border-radius: 4px;
                min-width: 80px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #0b7dda;
            }
        """)
        button_layout.addWidget(self.reset_button)
        
        button_section.addLayout(button_layout)
        button_section.addStretch()
        
        compact_controls.addLayout(button_section)
        compact_controls.addSpacing(20)
        
        # Sliders section in a vertical layout
        sliders_section = QVBoxLayout()
        
        # Animation speed slider
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(10, 10000)  # Increased to 10x to 1000x
        self.speed_slider.setValue(100)
        self.speed_slider.setTickPosition(QSlider.TicksBelow)
        self.speed_slider.setTickInterval(100)
        self.speed_slider.valueChanged.connect(self.update_speed)
        speed_layout.addWidget(self.speed_slider)
        
        self.speed_value = QLabel("1.0x")
        self.speed_value.setMinimumWidth(40)
        speed_layout.addWidget(self.speed_value)
        sliders_section.addLayout(speed_layout)
        
        # Amplitude slider with value display
        amplitude_layout = QHBoxLayout()
        amplitude_layout.addWidget(QLabel("Amplitude:"))
        self.amplitude_slider = QSlider(Qt.Horizontal)
        self.amplitude_slider.setRange(1, 1000)  # 0.1x to 100.0x
        self.amplitude_slider.setValue(10)  # Default 1.0x
        self.amplitude_slider.setTickPosition(QSlider.TicksBelow)
        self.amplitude_slider.setTickInterval(100)
        self.amplitude_slider.valueChanged.connect(self.update_amplitude)
        amplitude_layout.addWidget(self.amplitude_slider)
        
        self.amplitude_value = QLabel("1.0x")
        self.amplitude_value.setMinimumWidth(40)
        amplitude_layout.addWidget(self.amplitude_value)
        sliders_section.addLayout(amplitude_layout)
        
        # Add direct scale input in the same section
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Custom:"))
        self.scale_input = QDoubleSpinBox()
        self.scale_input.setRange(0.1, 1000)
        self.scale_input.setValue(1.0)
        self.scale_input.setSingleStep(1.0)
        self.scale_input.valueChanged.connect(self.set_scale_from_input)
        scale_layout.addWidget(self.scale_input)
        sliders_section.addLayout(scale_layout)
        
        compact_controls.addLayout(sliders_section)
        
        controls_group_layout.addLayout(compact_controls)
        controls_layout.addWidget(controls_group)
        
        main_splitter.addWidget(controls_widget)
        
        # Set initial splitter sizes - make the visualization area larger
        main_splitter.setSizes([800, 150])
        layout.addWidget(main_splitter)
        
        # Create animation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation_frame)
        self.timer.setInterval(16)  # ~60 fps for smoother animation
        
        # Disable controls initially
        self.play_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.amplitude_slider.setEnabled(False)
        self.mode_combo.setEnabled(False)
        self.speed_slider.setEnabled(False)
        self.scale_input.setEnabled(False)
        
    def set_data(self, coords, mode_shapes, natural_frequencies=None):
        """Set the data for the animation"""
        print("ModeShapeAnimationWidget received new data")
        print(f"Coords shape: {coords.shape if hasattr(coords, 'shape') else 'list'}")
        print(f"Mode shapes shape: {mode_shapes.shape if hasattr(mode_shapes, 'shape') else 'unknown'}")
        print(f"Natural frequencies: {natural_frequencies[:5] if natural_frequencies is not None else None}")
        
        # Convert coords to numpy array if it's not already
        if not isinstance(coords, np.ndarray):
            coords = np.array(coords)
        
        # Store original data
        self.coords = coords
        self.original_mode_shapes = mode_shapes.copy() if hasattr(mode_shapes, 'copy') else np.array(mode_shapes)
        self.natural_frequencies = natural_frequencies
        
        # Check if mode shapes contain valid data
        if hasattr(mode_shapes, 'shape'):
            max_val = np.max(np.abs(mode_shapes))
            print(f"Maximum mode shape value: {max_val}")
            
            # If mode shapes have very small values, apply scaling
            if max_val < 1e-10:
                print("WARNING: Mode shapes have very small values, applying large scaling factor")
                # Apply a large scaling factor to make shapes visible
                mode_shapes = mode_shapes * 1e10
                self.original_mode_shapes = mode_shapes.copy()
        
        # Create a working copy of mode shapes
        self.mode_shapes = self.original_mode_shapes.copy()
        
        if self.coords is not None and self.mode_shapes is not None:
            print(f"Processing {self.mode_shapes.shape[1]} mode shapes")
            
            # Normalize mode shapes for better visualization
            beam_length = np.max(self.coords) - np.min(self.coords)
            auto_scale_factors = []
            
            for i in range(self.mode_shapes.shape[1]):
                max_abs = np.max(np.abs(self.mode_shapes[:, i]))
                print(f"Mode {i+1} max amplitude: {max_abs}")
                
                if max_abs > 1e-10:  # Non-zero mode shape
                    # Calculate auto-scale factor to make modes comparable
                    auto_scale = (beam_length * 0.2) / max_abs
                    auto_scale_factors.append(auto_scale)
                    # Apply auto-scaling
                    self.mode_shapes[:, i] = self.original_mode_shapes[:, i] * auto_scale * self.scale_factor
                else:
                    # If mode shape has zero amplitude, create a synthetic shape
                    print(f"Mode {i+1} has near-zero amplitude, creating synthetic shape")
                    auto_scale_factors.append(1.0)
                    # Apply a default sinusoidal shape with i+1 half-waves
                    x_normalized = (self.coords - np.min(self.coords)) / beam_length
                    synthetic_shape = np.sin((i+1) * np.pi * x_normalized) * beam_length * 0.1
                    self.mode_shapes[:, i] = synthetic_shape
                    # Also update original mode shapes to avoid division by zero later
                    self.original_mode_shapes[:, i] = synthetic_shape / self.scale_factor
                    
            # Store auto-scale factors for later use
            self.auto_scale_factors = np.array(auto_scale_factors)
            
            print(f"Auto-scale factors: {self.auto_scale_factors[:5]}")
            
            # Set axis limits with padding
            x_min, x_max = np.min(self.coords), np.max(self.coords)
            y_min = -np.max(np.abs(self.mode_shapes)) * 1.2
            y_max = np.max(np.abs(self.mode_shapes)) * 1.2
            
            # Add padding to axis limits
            x_padding = 0.05 * (x_max - x_min)
            
            self.ax.set_xlim(x_min - x_padding, x_max + x_padding)
            self.ax.set_ylim(y_min, y_max)
            
            # Plot undeformed beam
            self.undeformed_line.set_data(self.coords, np.zeros_like(self.coords))
            
            # Populate mode combo box
            self.mode_combo.clear()
            for i in range(self.mode_shapes.shape[1]):
                if self.natural_frequencies is not None and i < len(self.natural_frequencies):
                    self.mode_combo.addItem(f"Mode {i+1} - {self.natural_frequencies[i]:.2f} Hz")
                else:
                    self.mode_combo.addItem(f"Mode {i+1}")
            
            # Set current mode to first mode
            self.current_mode = 0
            self.mode_combo.setCurrentIndex(0)
            
            # Update frequency label
            if self.natural_frequencies is not None and len(self.natural_frequencies) > 0:
                self.freq_label.setText(f"Frequency: {self.natural_frequencies[0]:.2f} Hz")
            
            # Reset animation and start automatically
            self.current_frame = 0
            self.update_plot(1.0)  # Start with maximum amplitude for visibility
            
            # Enable controls
            self.play_button.setEnabled(True)
            self.reset_button.setEnabled(True)
            self.amplitude_slider.setEnabled(True)
            self.mode_combo.setEnabled(True)
            self.speed_slider.setEnabled(True)
            self.scale_input.setEnabled(True)
            
            # Refresh canvas
            self.canvas.draw()
            
            # Start animation automatically
            self.toggle_animation()
            
            print("Mode shape animation setup complete")
            
    def change_mode(self, index):
        """Change the current mode being animated"""
        if index < 0 or self.mode_shapes is None or index >= self.mode_shapes.shape[1]:
            return
        
        print(f"Changing to mode {index+1}")
        self.current_mode = index
        
        # Update frequency label
        if self.natural_frequencies is not None and index < len(self.natural_frequencies):
            self.freq_label.setText(f"Frequency: {self.natural_frequencies[index]:.2f} Hz")
        
        # Reset animation
        self.reset_animation()
        
        # Update y-axis limits for current mode
        self.update_y_limits()
        
    def toggle_animation(self):
        """Toggle animation play/pause state"""
        if self.animation_running:
            self.timer.stop()
            self.play_button.setText("▶ Play")
            self.play_button.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    font-weight: bold;
                    border-radius: 4px;
                    min-width: 80px;
                    padding: 5px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
            """)
        else:
            # Set a faster interval for smoother animation
            self.timer.setInterval(16)  # ~60 fps for smoother animation
            self.timer.start()
            self.play_button.setText("⏸ Pause")
            self.play_button.setStyleSheet("""
                QPushButton {
                    background-color: #FF9800;
                    color: white;
                    font-weight: bold;
                    border-radius: 4px;
                    min-width: 80px;
                    padding: 5px;
                }
                QPushButton:hover {
                    background-color: #e68a00;
                }
            """)
        
        self.animation_running = not self.animation_running
        
    def reset_animation(self):
        """Reset the animation to the starting position"""
        self.current_frame = 0
        self.update_plot(0)  # Reset to neutral position
        
        # Update play button text if animation is stopped
        if not self.animation_running:
            self.play_button.setText("▶ Play")
            self.play_button.setStyleSheet("""
                QPushButton {
                    background-color: #4CAF50;
                    color: white;
                    font-weight: bold;
                    border-radius: 4px;
                    min-width: 80px;
                    padding: 5px;
                }
                QPushButton:hover {
                    background-color: #45a049;
                }
            """)
    
    def update_speed(self, value):
        """Update animation speed"""
        self.animation_speed = value / 100.0  # Adjusted to handle the new range
        self.speed_value.setText(f"{self.animation_speed:.1f}x")
        
        # Adjust timer interval for smoother animation at higher speeds
        interval = max(5, int(16 / self.animation_speed))
        self.timer.setInterval(interval)
        
    def update_amplitude(self, value):
        """Update the animation amplitude"""
        self.scale_factor = value / 10.0
        self.amplitude_value.setText(f"{self.scale_factor:.1f}x")
        
        # Update spin box without triggering its valueChanged signal
        self.scale_input.blockSignals(True)
        self.scale_input.setValue(self.scale_factor)
        self.scale_input.blockSignals(False)
        
        # Apply new scaling to all modes
        if hasattr(self, 'original_mode_shapes') and self.original_mode_shapes is not None:
            for i in range(self.mode_shapes.shape[1]):
                self.mode_shapes[:, i] = self.original_mode_shapes[:, i] * self.auto_scale_factors[i] * self.scale_factor
        
        # Update y-axis limits
        self.update_y_limits()
        
        # Update the current frame to show the effect immediately
        if self.animation_running:
            self.update_plot()
        else:
            # If animation is paused, just update the current frame
            self.update_plot(0)
            
    def set_scale_from_input(self, value):
        """Set the scale factor directly from the input box"""
        # Update the slider without triggering its valueChanged signal
        self.amplitude_slider.blockSignals(True)
        self.amplitude_slider.setValue(int(value * 10))
        self.amplitude_slider.blockSignals(False)
        
        # Update the scale value label
        self.scale_factor = value
        self.amplitude_value.setText(f"{self.scale_factor:.1f}x")
        
        # Apply new scaling to all modes
        if hasattr(self, 'original_mode_shapes') and self.original_mode_shapes is not None:
            for i in range(self.mode_shapes.shape[1]):
                self.mode_shapes[:, i] = self.original_mode_shapes[:, i] * self.auto_scale_factors[i] * self.scale_factor
        
        # Update y-axis limits
        self.update_y_limits()
        
        # Update the current frame to show the effect immediately
        if self.animation_running:
            self.update_plot()
        else:
            # If animation is paused, just update the current frame
            self.update_plot(0)
    
    def update_y_limits(self):
        """Update y-axis limits based on current mode shape amplitude"""
        if self.mode_shapes is None or self.current_mode >= self.mode_shapes.shape[1]:
            return
            
        # Get the maximum amplitude for the current mode
        max_amp = np.max(np.abs(self.mode_shapes[:, self.current_mode]))
        
        # Add padding for better visualization
        y_min = -max_amp * 1.2
        y_max = max_amp * 1.2
        
        # Ensure minimum height for visibility
        if y_max - y_min < 0.01:
            y_mean = (y_max + y_min) / 2
            y_min = y_mean - 0.005
            y_max = y_mean + 0.005
        
        # Update axis limits
        self.ax.set_ylim(y_min, y_max)
        self.canvas.draw_idle()
    
    def update_animation_frame(self):
        """Update the animation frame"""
        if self.mode_shapes is None or self.current_mode >= self.mode_shapes.shape[1]:
            return
            
        # Increment frame counter
        self.current_frame = (self.current_frame + 1) % self.animation_frames
        
        # Calculate oscillation factor (sine wave for smooth animation)
        angle = 2 * np.pi * self.current_frame / self.animation_frames
        oscillation_factor = np.sin(angle)
        
        # Update the plot with new oscillation factor
        self.update_plot(oscillation_factor)
    
    def update_plot(self, oscillation_factor=None):
        """Update the mode shape plot with the given oscillation factor"""
        if self.coords is None or self.mode_shapes is None or self.current_mode >= self.mode_shapes.shape[1]:
            return
            
        # If no oscillation factor provided, use 0 (neutral position)
        if oscillation_factor is None:
            oscillation_factor = 0
            
        # Get the current mode shape
        mode_shape = self.mode_shapes[:, self.current_mode]
        
        # Apply oscillation factor
        current_shape = mode_shape * oscillation_factor
        
        # Update plot data
        self.mode_line.set_data(self.coords, current_shape)
        
        # Make sure the undeformed line is visible
        self.undeformed_line.set_data(self.coords, np.zeros_like(self.coords))
        
        # Ensure proper axis limits
        x_min, x_max = np.min(self.coords), np.max(self.coords)
        x_padding = 0.05 * (x_max - x_min)
        
        # Calculate y limits based on maximum possible amplitude
        max_amp = np.max(np.abs(self.mode_shapes[:, self.current_mode]))
        y_min = -max_amp * 1.2
        y_max = max_amp * 1.2
        
        # Ensure minimum height for visibility
        if y_max - y_min < 0.01:
            y_mean = (y_max + y_min) / 2
            y_min = y_mean - 0.005
            y_max = y_mean + 0.005
        
        # Set axis limits
        self.ax.set_xlim(x_min - x_padding, x_max + x_padding)
        self.ax.set_ylim(y_min, y_max)
        
        # Set title with mode number and frequency
        if self.natural_frequencies is not None and self.current_mode < len(self.natural_frequencies):
            self.ax.set_title(f"Mode {self.current_mode + 1} - {self.natural_frequencies[self.current_mode]:.2f} Hz")
        else:
            self.ax.set_title(f"Mode {self.current_mode + 1}")
        
        # Refresh canvas
        self.canvas.draw_idle()