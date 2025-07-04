import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QSlider, QGroupBox, QFormLayout, QSplitter, QDoubleSpinBox,
    QSpinBox, QComboBox, QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer


class BeamAnimationWidget(QWidget):
    """
    Widget for animating beam vibration
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.initUI()
        
        # Initialize data
        self.coords = None
        self.displacements = None
        self.original_displacements = None  # Store original displacements before scaling
        self.times = None
        
        # Animation state
        self.current_time_idx = 0
        self.speed_factor = 1.0
        self.frame_count = 0
        self.interpolation_frames = 10  # Number of frames to interpolate between time points
        self.scale_factor = 1.0  # User-adjustable scaling factor
        self.auto_scale_factor = 1.0  # Auto-calculated scaling factor
        
    def initUI(self):
        """Initialize the UI components"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(5)
        
        # Create main splitter to allow resizing animation vs controls
        main_splitter = QSplitter(Qt.Vertical)
        main_splitter.setChildrenCollapsible(False)
        
        # Top section - Visualization
        viz_widget = QWidget()
        viz_layout = QVBoxLayout(viz_widget)
        viz_layout.setContentsMargins(0, 0, 0, 0)
        viz_layout.setSpacing(5)
        
        # Add title
        title = QLabel("Beam Vibration Animation")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        viz_layout.addWidget(title)
        
        # Create canvas for animation
        self.figure = Figure(figsize=(8, 5), dpi=100)  # Increased height
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(350)  # Increased minimum height
        self.ax = self.figure.add_subplot(111)
        self.ax.set_xlabel('Position (m)')
        self.ax.set_ylabel('Displacement (m)')
        self.ax.set_title('Beam Vibration')
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        # Create initial empty plot
        self.beam_line, = self.ax.plot([], [], 'b-', lw=2, marker='o', markersize=4)
        self.undeformed_line, = self.ax.plot([], [], 'k--', lw=1)
        
        # Tight layout for better use of space
        self.figure.tight_layout(pad=2.0)
        
        viz_layout.addWidget(self.canvas, 1)  # Give canvas a stretch factor of 1
        main_splitter.addWidget(viz_widget)
        
        # Bottom section - Controls
        controls_widget = QWidget()
        controls_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        controls_layout = QVBoxLayout(controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(5)
        
        # Create controls group
        controls_group = QGroupBox("Animation Controls")
        controls_group_layout = QVBoxLayout(controls_group)
        controls_group_layout.setContentsMargins(10, 15, 10, 10)
        controls_group_layout.setSpacing(10)
        
        # Create a horizontal layout for all controls to make them more compact
        compact_controls = QHBoxLayout()
        
        # Time display section
        time_section = QVBoxLayout()
        
        # Time display with more information
        time_header = QHBoxLayout()
        time_header.addWidget(QLabel("Time:"))
        self.time_label = QLabel("0.00 s")
        self.time_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        time_header.addWidget(self.time_label)
        time_section.addLayout(time_header)
        
        # Add progress indicator
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("Progress:"))
        self.progress_label = QLabel("0 / 0")
        progress_layout.addWidget(self.progress_label)
        time_section.addLayout(progress_layout)
        
        compact_controls.addLayout(time_section)
        compact_controls.addSpacing(20)
        
        # Playback buttons section
        buttons_section = QVBoxLayout()
        buttons_layout = QHBoxLayout()
        
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
        buttons_layout.addWidget(self.play_button)
        
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
        buttons_layout.addWidget(self.reset_button)
        
        buttons_section.addLayout(buttons_layout)
        buttons_section.addStretch()
        
        compact_controls.addLayout(buttons_section)
        compact_controls.addSpacing(20)
        
        # Sliders section
        sliders_section = QVBoxLayout()
        
        # Speed slider with value display
        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Speed:"))
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setRange(10, 1000)  # Increased to 10x to 100x
        self.speed_slider.setValue(100)
        self.speed_slider.setTickPosition(QSlider.TicksBelow)
        self.speed_slider.setTickInterval(100)
        self.speed_slider.valueChanged.connect(self.update_speed)
        speed_layout.addWidget(self.speed_slider)
        
        self.speed_value = QLabel("1.0x")
        self.speed_value.setMinimumWidth(40)
        speed_layout.addWidget(self.speed_value)
        sliders_section.addLayout(speed_layout)
        
        # Add amplitude scaling slider
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Amplitude:"))
        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setRange(1, 1000)  # 0.1x to 100.0x
        self.scale_slider.setValue(10)  # Default 1.0x
        self.scale_slider.setTickPosition(QSlider.TicksBelow)
        self.scale_slider.setTickInterval(100)
        self.scale_slider.valueChanged.connect(self.update_scale)
        scale_layout.addWidget(self.scale_slider)
        
        self.scale_value = QLabel("1.0x")
        self.scale_value.setMinimumWidth(40)
        scale_layout.addWidget(self.scale_value)
        sliders_section.addLayout(scale_layout)
        
        # Add direct scale input
        custom_layout = QHBoxLayout()
        custom_layout.addWidget(QLabel("Custom:"))
        self.scale_input = QDoubleSpinBox()
        self.scale_input.setRange(0.1, 1000)
        self.scale_input.setValue(1.0)
        self.scale_input.setSingleStep(1.0)
        self.scale_input.valueChanged.connect(self.set_scale_from_input)
        custom_layout.addWidget(self.scale_input)
        sliders_section.addLayout(custom_layout)
        
        compact_controls.addLayout(sliders_section)
        
        controls_group_layout.addLayout(compact_controls)
        controls_layout.addWidget(controls_group)
        
        main_splitter.addWidget(controls_widget)
        
        # Set initial splitter sizes - make the visualization area larger
        main_splitter.setSizes([800, 150])
        
        layout.addWidget(main_splitter)
        
        # Create animation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.setInterval(16)  # ~60 fps for smoother animation
        
        # Disable controls initially
        self.play_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.speed_slider.setEnabled(False)
        self.scale_slider.setEnabled(False)
        self.scale_input.setEnabled(False)
        
    def update_animation(self, results=None):
        """Update the animation with new simulation results"""
        if results is not None:
            print("BeamAnimationWidget received new results")
            # Extract data from results
            self.coords = results.get('coords', None)
            
            # Check for both 'displacement' and 'displacements' keys
            if 'displacements' in results:
                print(f"Using 'displacements' from results, shape: {results['displacements'].shape}")
                self.displacements = results['displacements']
                self.original_displacements = self.displacements.copy()
            elif 'displacement' in results:
                # Get the displacement matrix from results
                displacement_matrix = results['displacement']
                print(f"Using 'displacement' from results, shape: {displacement_matrix.shape}")
                
                # Process the displacement data for animation
                try:
                    # Extract vertical displacements for each node
                    # In FEM, typically even indices (0, 2, 4...) are x-displacements and 
                    # odd indices (1, 3, 5...) are y-displacements
                    num_nodes = len(self.coords)
                    num_time_points = len(results.get('time', results.get('times', [])))
                    
                    print(f"Processing displacements for {num_nodes} nodes and {num_time_points} time points")
                    
                    # Create a matrix to store vertical displacements for each node at each time
                    y_displacements = []
                    
                    # For each time point, extract the vertical displacement at each node position
                    for t_idx in range(num_time_points):
                        node_displacements = []
                        for i in range(num_nodes):
                            # Get vertical displacement DOF for this node (odd indices)
                            node_dof = 2 * i + 1
                            if node_dof < displacement_matrix.shape[0]:
                                node_displacements.append(displacement_matrix[node_dof, t_idx])
                            else:
                                node_displacements.append(0.0)
                        y_displacements.append(node_displacements)
                    
                    # Convert to numpy array and transpose to get [nodes, time_points]
                    self.displacements = np.array(y_displacements).T
                    self.original_displacements = self.displacements.copy()  # Store original for scaling
                    
                    print(f"Processed displacements shape: {self.displacements.shape}")
                    
                    # Calculate auto-scaling factor based on beam length
                    max_disp = np.max(np.abs(self.displacements))
                    print(f"Maximum displacement: {max_disp}")
                    
                    if max_disp > 0:
                        # Calculate amplification factor based on beam length
                        beam_length = max(self.coords) - min(self.coords)
                        target_amplitude = beam_length * 0.1  # Target 10% of beam length
                        self.auto_scale_factor = target_amplitude / max_disp
                        
                        # Apply with a reasonable cap
                        self.auto_scale_factor = min(self.auto_scale_factor, 100.0)
                        
                        # Apply both auto and user scaling
                        total_scale = self.auto_scale_factor * self.scale_factor
                        self.displacements = self.original_displacements * total_scale
                        
                        # Update the scale input to match auto-scale
                        self.scale_input.setValue(self.scale_factor)
                        
                        print(f"Applied auto-scale factor of {self.auto_scale_factor:.2f}x with user scale of {self.scale_factor:.2f}x")
                    
                    print(f"Animation data prepared: {num_nodes} nodes, {num_time_points} time points")
                    print(f"Displacement matrix shape: {displacement_matrix.shape}")
                    print(f"Processed displacements shape: {self.displacements.shape}")
                except Exception as e:
                    print(f"Error processing displacement data: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    return
            else:
                print("No displacement data found in results")
                return
            
            # Get time points
            if 'times' in results:
                self.times = results['times']
                print(f"Using 'times' from results, length: {len(self.times)}")
            elif 'time' in results:
                self.times = results['time']
                print(f"Using 'time' from results, length: {len(self.times)}")
            else:
                print("No time data found in results")
                return
            
            if self.coords is None or self.displacements is None or self.times is None:
                print("Missing required data for animation")
                return
            
            # Clear the axes and reset
            self.ax.clear()
            
            # Set up the axes
            self.ax.set_xlabel('Position (m)')
            self.ax.set_ylabel('Displacement (m)')
            self.ax.set_title('Beam Vibration')
            self.ax.grid(True, linestyle='--', alpha=0.7)
            
            # Calculate displacement range for better visualization
            max_disp = np.max(np.abs(self.displacements))
            if max_disp < 1e-6:  # If displacements are very small
                max_disp = 0.01  # Set a minimum value
            
            # Set fixed axis limits for smoother animation
            x_min, x_max = min(self.coords), max(self.coords)
            y_min = -max_disp * 1.5  # Add 50% margin
            y_max = max_disp * 1.5   # Add 50% margin
            
            self.ax.set_xlim(x_min, x_max)
            self.ax.set_ylim(y_min, y_max)
            
            # Create the beam line and undeformed line
            self.beam_line, = self.ax.plot([], [], 'b-', lw=2, marker='o', markersize=4)
            self.undeformed_line, = self.ax.plot(self.coords, np.zeros_like(self.coords), 'k--', lw=1)
            
            # Apply tight layout for better use of space
            self.figure.tight_layout(pad=2.0)
            
            # Reset animation
            self.current_time_idx = 0
            self.update_frame()
            
            # Update progress label
            self.progress_label.setText(f"1 / {len(self.times)}")
            
            # Enable controls
            self.play_button.setEnabled(True)
            self.reset_button.setEnabled(True)
            self.speed_slider.setEnabled(True)
            self.scale_slider.setEnabled(True)
            self.scale_input.setEnabled(True)
            
            # Start the animation automatically
            self.toggle_animation()
            
            return
            
        # Advance animation frame
        self.update_frame()
    
    def update_frame(self):
        """Update the animation frame with smooth interpolation"""
        if self.coords is None or self.displacements is None or self.times is None:
            return
        
        # Calculate base time index and next time index
        base_idx = int(self.current_time_idx)
        next_idx = (base_idx + 1) % len(self.times)
        
        # Calculate interpolation factor (0.0 to 1.0)
        interp_factor = self.frame_count / self.interpolation_frames
        
        # Get displacements at base and next time points
        base_disp = self.displacements[:, base_idx]
        next_disp = self.displacements[:, next_idx]
        
        # Interpolate between the two time points for smoother animation
        current_disp = base_disp * (1 - interp_factor) + next_disp * interp_factor
        
        # Update beam position
        x_coords = self.coords
        y_coords = current_disp
        
        # Update plot data
        self.beam_line.set_data(x_coords, y_coords)
        
        # Update time label
        base_time = self.times[base_idx]
        next_time = self.times[next_idx]
        current_time = base_time * (1 - interp_factor) + next_time * interp_factor
        self.time_label.setText(f"{current_time:.4f} s")
        
        # Update progress label
        self.progress_label.setText(f"{base_idx + 1} / {len(self.times)}")
        
        # Increment frame counter
        self.frame_count += 1
        
        # If we've completed all interpolation frames, move to the next base time point
        if self.frame_count >= self.interpolation_frames:
            self.frame_count = 0
            self.current_time_idx = (self.current_time_idx + self.speed_factor) % len(self.times)
        
        # Refresh canvas
        self.canvas.draw_idle()
        
    def toggle_animation(self):
        """Toggle animation play/pause"""
        if self.timer.isActive():
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
    
    def reset_animation(self):
        """Reset the animation to the starting position"""
        self.current_time_idx = 0
        self.frame_count = 0
        self.update_frame()
        
        # Update play button text if animation is stopped
        if not self.timer.isActive():
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
        self.speed_factor = value / 100.0  # Adjusted to handle the new range
        self.speed_value.setText(f"{self.speed_factor:.1f}x")
        
        # Adjust timer interval for smoother animation at higher speeds
        interval = max(5, int(16 / self.speed_factor))
        self.timer.setInterval(interval)
    
    def update_scale(self, value):
        """Update the animation scale factor"""
        self.scale_factor = value / 10.0
        self.scale_value.setText(f"{self.scale_factor:.1f}x")
        
        # Update the scale input without triggering its valueChanged signal
        self.scale_input.blockSignals(True)
        self.scale_input.setValue(self.scale_factor)
        self.scale_input.blockSignals(False)
        
        # Apply new scaling if we have data
        if hasattr(self, 'original_displacements') and self.original_displacements is not None:
            # Apply both auto and user scaling
            total_scale = self.auto_scale_factor * self.scale_factor
            self.displacements = self.original_displacements * total_scale
            
            # Update the current frame to show the effect immediately
            if self.timer.isActive():
                self.update_frame()
            else:
                # If animation is paused, just update the current frame
                base_idx = int(self.current_time_idx)
                self.beam_line.set_data(self.coords, self.displacements[:, base_idx])
                self.canvas.draw()
                
            # Update y-axis limits
            max_disp = np.max(np.abs(self.displacements))
            if max_disp > 0:
                y_min = -max_disp * 1.5  # Add 50% margin
                y_max = max_disp * 1.5   # Add 50% margin
                self.ax.set_ylim(y_min, y_max)
                self.canvas.draw_idle()
    
    def set_scale_from_input(self, value):
        """Set the scale factor directly from the input box"""
        # Update the slider without triggering its valueChanged signal
        self.scale_slider.blockSignals(True)
        self.scale_slider.setValue(int(value * 10))
        self.scale_slider.blockSignals(False)
        
        # Update the scale value label
        self.scale_factor = value
        self.scale_value.setText(f"{self.scale_factor:.1f}x")
        
        # Apply new scaling if we have data
        if hasattr(self, 'original_displacements') and self.original_displacements is not None:
            # Apply both auto and user scaling
            total_scale = self.auto_scale_factor * self.scale_factor
            self.displacements = self.original_displacements * total_scale
            
            # Update the current frame to show the effect immediately
            if self.timer.isActive():
                self.update_frame()
            else:
                # If animation is paused, just update the current frame
                base_idx = int(self.current_time_idx)
                self.beam_line.set_data(self.coords, self.displacements[:, base_idx])
                self.canvas.draw()
                
            # Update y-axis limits
            max_disp = np.max(np.abs(self.displacements))
            if max_disp > 0:
                y_min = -max_disp * 1.5  # Add 50% margin
                y_max = max_disp * 1.5   # Add 50% margin
                self.ax.set_ylim(y_min, y_max)
                self.canvas.draw_idle()
    
    def set_data(self, coords, displacements, times):
        """Set the data for the animation directly"""
        self.coords = coords
        self.displacements = displacements
        self.original_displacements = displacements.copy()
        self.times = times
        
        # Calculate auto-scaling factor based on beam length
        max_disp = np.max(np.abs(self.displacements))
        if max_disp > 0:
            # Calculate amplification factor based on beam length
            beam_length = max(self.coords) - min(self.coords)
            target_amplitude = beam_length * 0.1  # Target 10% of beam length
            self.auto_scale_factor = target_amplitude / max_disp
            
            # Apply with a reasonable cap
            self.auto_scale_factor = min(self.auto_scale_factor, 100.0)
            
            # Apply both auto and user scaling
            total_scale = self.auto_scale_factor * self.scale_factor
            self.displacements = self.original_displacements * total_scale
            
        # Clear the axes and reset
        self.ax.clear()
        
        # Set up the axes
        self.ax.set_xlabel('Position (m)')
        self.ax.set_ylabel('Displacement (m)')
        self.ax.set_title('Beam Vibration')
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        # Calculate displacement range for better visualization
        max_disp = np.max(np.abs(self.displacements))
        if max_disp < 1e-6:  # If displacements are very small
            max_disp = 0.01  # Set a minimum value
        
        # Set fixed axis limits for smoother animation
        x_min, x_max = min(self.coords), max(self.coords)
        y_min = -max_disp * 1.5  # Add 50% margin
        y_max = max_disp * 1.5   # Add 50% margin
        
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        
        # Create the beam line and undeformed line
        self.beam_line, = self.ax.plot([], [], 'b-', lw=2, marker='o', markersize=4)
        self.undeformed_line, = self.ax.plot(self.coords, np.zeros_like(self.coords), 'k--', lw=1)
        
        # Apply tight layout for better use of space
        self.figure.tight_layout(pad=2.0)
        
        # Reset animation
        self.current_time_idx = 0
        self.update_frame()
        
        # Update progress label
        self.progress_label.setText(f"1 / {len(self.times)}")
        
        # Enable controls
        self.play_button.setEnabled(True)
        self.reset_button.setEnabled(True)
        self.speed_slider.setEnabled(True)
        self.scale_slider.setEnabled(True)
        self.scale_input.setEnabled(True)
        
        # Start the animation automatically
        self.toggle_animation() 